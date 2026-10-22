"""检索离线评测 + 消融表(M9 Step 3)。会调 embedding 和 rerank(0.6B、8B 各 ~85 次,几毛钱)。

    python -m eval.rag_offline

题目:eval/rag_golden.jsonl(生成方法和相关性规则见 eval/rag_golden.py)。
消融行(同一批题、同一批 24 篇文档):

    0   v1 基线:定长 500/100 切块,不拼 header,dense       (eval/rag_baseline_build.py 建的集合)
    1   v2 子块 + dense
    2   v2 子块 + sparse(BM25)
    3   v2 hybrid:dense + sparse → RRF
    4   hybrid + rerank(Qwen3-Reranker-0.6B)
    4b  hybrid + rerank(Qwen3-Reranker-8B)

指标(都看子块排名,去父块之前;深度 30):
    Hit@k   fact 题:前 k 名里有没有相关块。对 fact 题它就等于 Recall@k(每题只有一个"答案点")。
    MRR     第一个相关块名次的倒数,30 名内没有记 0。
    nDCG@5  相关块越靠前分越高;不同切块方式的相关块个数不同,跨第 0 行比较时要打折看。
    覆盖@5  compare 题:两个答案点各自被捞到的比例(只捞到一家 = 0.5)。

相关性:span ≥ 6 个字符 → 同一篇文档里包含 span 的块都算相关;span 太短("21.6")→ 用出题块里 span 前后各 8 个字符
组成的窗口去匹配,防止 "121.6"、"21.65" 这类误中。规则只看文本,所以第 0 行的定长块也能用同一份题。
"""

import json
import math
import re
import statistics
import time
from datetime import date
from pathlib import Path

from eval.rag_baseline_build import BASELINE_COLLECTION
from rag import db, hybrid
from rag.embedding import embed_query as _embed_query_raw
from rag.store import QdrantStore

GOLDEN = Path("eval/rag_golden.jsonl")
DEPTH = 30
KS = (1, 3, 5, 10, 20, 30)
ROWS = [
    ("0", "v1 基线:定长切块 + dense", None),
    ("1", "v2 子块 + dense", dict(use_dense=True, use_sparse=False, use_rerank=False)),
    ("2", "v2 子块 + sparse", dict(use_dense=False, use_sparse=True, use_rerank=False)),
    ("3", "hybrid(RRF)", dict(use_dense=True, use_sparse=True, use_rerank=False)),
    ("4", "hybrid + rerank 0.6B", dict(use_dense=True, use_sparse=True, use_rerank=True,
                                       rerank_model="Qwen/Qwen3-Reranker-0.6B")),
    ("4b", "hybrid + rerank 8B", dict(use_dense=True, use_sparse=True, use_rerank=True,
                                      rerank_model="Qwen/Qwen3-Reranker-8B")),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ---------------------------------------------------------------- embedding 缓存
# 6 行里有 4 行要同一个问题的向量;缓存起来既省钱,也让各行的检索结果不受 embedding 抖动影响。
# 时延按"第一次真实调用"的耗时记,保证各行时延可比。
_embed_cache: dict[str, tuple[list[float], float]] = {}


def embed_cached(q: str) -> list[float]:
    if q not in _embed_cache:
        t = time.perf_counter()
        v = _embed_query_raw(q)
        _embed_cache[q] = (v, (time.perf_counter() - t) * 1000)
    return _embed_cache[q][0]


hybrid.embed_query = embed_cached


# ---------------------------------------------------------------- 相关性
def _load_texts() -> tuple[dict, dict, dict]:
    """返回 (v2: source → [(id, 去空白文本)], 基线: 同上, 出题块 id → 去空白文本)。"""
    v2: dict[str, list] = {}
    gen: dict[str, str] = {}
    with db._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT c.id, d.source, c.text FROM fra_v2_chunks c JOIN fra_v2_documents d ON d.id = c.doc_id")
        for cid, source, text in cur.fetchall():
            n = _norm(text)
            v2.setdefault(source, []).append((cid, n))
            gen[cid] = n
    base: dict[str, list] = {}
    store = QdrantStore(collection=BASELINE_COLLECTION)
    offset = None
    while True:
        body = {"limit": 5000, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = store.http.post(f"/collections/{BASELINE_COLLECTION}/points/scroll", json=body).json()["result"]
        for p in r["points"]:
            base.setdefault(p["payload"]["source"], []).append((str(p["id"]), _norm(p["payload"]["text"])))
        offset = r.get("next_page_offset")
        if offset is None:
            return v2, base, gen


def _key(span: dict, gen: dict) -> str:
    s = _norm(span["span"])
    if len(s) >= 6:
        return s
    g = gen.get(span.get("chunk_id", ""), "")
    i = g.find(s)
    return g[max(0, i - 8): i + len(s) + 8] if i >= 0 else s


def groups(item: dict, texts: dict, gen: dict) -> list[set[str]]:
    """每个答案点一组相关块 id。fact 题 1 组,compare 题 2 组,no_answer 0 组。"""
    out = []
    for sp in item["spans"]:
        key = _key(sp, gen)
        out.append({cid for cid, t in texts.get(sp["source"], []) if key in t})
    return out


# ---------------------------------------------------------------- 跑一行
def run_row(cfg: dict | None, q: str, base_store: QdrantStore) -> tuple[list[str], float, list]:
    """返回 (按名次的子块 id, 总时延 ms, [(id, rerank 分)])。"""
    if cfg is None:
        t = time.perf_counter()
        pts = base_store.search(embed_cached(q), DEPTH)
        ms = (time.perf_counter() - t) * 1000 + _embed_cache[q][1]
        return [str(p["id"]) for p in pts], ms, []
    res = hybrid.search(q, top_k=DEPTH, expand=False, rerank_threshold=None, **cfg)
    ms = sum(res.timings_ms.values()) + (_embed_cache[q][1] if cfg.get("use_dense") else 0)
    scores = [(c.id, c.rerank_score) for c in res.ranked if c.rerank_score is not None]
    if res.degraded:
        print(f"    ⚠ 降级 {res.degraded}:{q[:40]}")
    return [c.id for c in res.ranked], ms, scores


def metrics(ranked: list[str], gs: list[set[str]]) -> dict:
    union = set().union(*gs) if gs else set()
    first = next((i for i, cid in enumerate(ranked, 1) if cid in union), None)
    m = {f"hit@{k}": float(first is not None and first <= k) for k in KS}
    m["mrr"] = 1 / first if first else 0.0
    dcg = sum(1 / math.log2(i + 1) for i, cid in enumerate(ranked[:5], 1) if cid in union)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(union), 5) + 1))
    m["ndcg@5"] = dcg / idcg if idcg else 0.0
    m["cover@5"] = sum(bool(g & set(ranked[:5])) for g in gs) / len(gs) if gs else 0.0
    m["first_rank"] = first
    return m


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def main() -> None:
    items = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    v2_texts, base_texts, gen = _load_texts()
    base_store = QdrantStore(collection=BASELINE_COLLECTION)
    answerable = [it for it in items if it["kind"] != "no_answer"]

    # 相关块个数:看两种切法下"答案点"各被几个块包含,以及有没有捞不到的题(跨块边界被切断)
    unreachable = {"v2": 0, "v1": 0}
    for it in answerable:
        it["_g"] = {"v2": groups(it, v2_texts, gen), "v1": groups(it, base_texts, gen)}
        for k in ("v2", "v1"):
            unreachable[k] += any(not g for g in it["_g"][k])
    print(f"{len(items)} 道题(可答 {len(answerable)},无答案 {len(items) - len(answerable)})")
    print(f"相关块个数中位数:v2 {statistics.median(len(g) for it in answerable for g in it['_g']['v2'])},"
          f" v1 {statistics.median(len(g) for it in answerable for g in it['_g']['v1'])};"
          f"答案点在该切法下找不到任何块(被块边界切断):v2 {unreachable['v2']} 题,v1 {unreachable['v1']} 题")

    results: dict[str, dict] = {}
    for row_id, name, cfg in ROWS:
        t0 = time.time()
        per_item, lat, rr_scores = {}, [], {}
        for it in items:
            ranked, ms, scores = run_row(cfg, it["question"], base_store)
            lat.append(ms)
            rr_scores[it["id"]] = scores
            if it["kind"] != "no_answer":
                per_item[it["id"]] = metrics(ranked, it["_g"]["v1" if cfg is None else "v2"])
        results[row_id] = {"name": name, "per_item": per_item, "lat": lat, "rerank": rr_scores}
        print(f"  行 {row_id:<3} {name:<26} 跑完 {time.time() - t0:5.0f}s")

    facts = [it for it in answerable if it["kind"] == "fact"]
    comps = [it for it in answerable if it["kind"] == "compare"]

    print(f"\n## 消融表(fact {len(facts)} 题;compare {len(comps)} 题只看覆盖@5)\n")
    print("| 行 | 配置 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@30 | MRR | nDCG@5 | compare 覆盖@5 | 时延 p50 / p95 ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row_id, name, _ in ROWS:
        r = results[row_id]
        pi = r["per_item"]
        fm = lambda key: _mean(pi[it["id"]][key] for it in facts)  # noqa: E731
        cov = _mean(pi[it["id"]]["cover@5"] for it in comps)
        print(f"| {row_id} | {name} | {fm('hit@1'):.3f} | {fm('hit@3'):.3f} | {fm('hit@5'):.3f} | {fm('hit@10'):.3f} | "
              f"{fm('hit@30'):.3f} | {fm('mrr'):.3f} | {fm('ndcg@5'):.3f} | {cov:.2f} | "
              f"{_pct(r['lat'], 0.5):.0f} / {_pct(r['lat'], 0.95):.0f} |")

    print("\n## 分组 Hit@5(fact)\n")
    buckets = {
        "中文问题": lambda it: it["lang"] == "zh",
        "英文问题": lambda it: it["lang"] == "en",
        "问英文财报(SEC)": lambda it: it["spans"][0]["source"].startswith("sec/"),
        "问中文年报(A 股)": lambda it: it["spans"][0]["source"].startswith("cninfo/"),
        "中文问英文财报": lambda it: it["lang"] == "zh" and it["spans"][0]["source"].startswith("sec/"),
        "数字题": lambda it: it.get("fact_kind") == "number",
        "叙述题": lambda it: it.get("fact_kind") == "narrative",
    }
    print("| 分组 | 题数 | " + " | ".join(f"行 {r}" for r, _, _ in ROWS) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in ROWS) + " |")
    for bname, f in buckets.items():
        sel = [it for it in facts if f(it)]
        cells = [f"{_mean(results[r]['per_item'][it['id']]['hit@5'] for it in sel):.2f}" for r, _, _ in ROWS]
        print(f"| {bname} | {len(sel)} | " + " | ".join(cells) + " |")

    # rerank 分数分布:相关 vs 不相关候选;可答题 top1 vs 无答案题 top1 → 阈值 / 拒答依据
    print("\n## rerank 分数分布\n")
    na = [it for it in items if it["kind"] == "no_answer"]
    print("| 模型 | 相关块分数 p10 / p50 | 不相关块分数 p50 / p90 | 可答题 top1 p10 / p50 | 无答案题 top1 p50 / max |")
    print("| --- | --- | --- | --- | --- |")
    for row_id in ("4", "4b"):
        r = results[row_id]
        rel, irr, top_ans, top_na = [], [], [], []
        for it in answerable:
            union = set().union(*it["_g"]["v2"])
            sc = r["rerank"][it["id"]]
            rel += [s for cid, s in sc if cid in union]
            irr += [s for cid, s in sc if cid not in union]
            if sc:
                top_ans.append(sc[0][1])
        for it in na:
            if r["rerank"][it["id"]]:
                top_na.append(r["rerank"][it["id"]][0][1])
        print(f"| {r['name']} | {_pct(rel, .1):.3f} / {_pct(rel, .5):.3f} | {_pct(irr, .5):.3f} / {_pct(irr, .9):.3f} | "
              f"{_pct(top_ans, .1):.3f} / {_pct(top_ans, .5):.3f} | {_pct(top_na, .5):.3f} / {max(top_na):.3f} |")
        sweep = []
        for thr in (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
            keep_rel = _mean(s >= thr for s in rel)
            cut_irr = _mean(s < thr for s in irr)
            na_pass = sum(s >= thr for s in top_na)
            sweep.append(f"{thr}: 留相关 {keep_rel:.0%} / 去不相关 {cut_irr:.0%} / 无答案题仍过线 {na_pass}/{len(top_na)}")
        print("  阈值扫描 " + r["name"] + ":\n    " + "\n    ".join(sweep))

    # 典型案例:rerank 救回来的 / 弄丢的
    print("\n## 名次变化最大的题(行 3 → 行 4b,fact)\n")
    deltas = []
    for it in facts:
        a = results["3"]["per_item"][it["id"]]["first_rank"]
        b = results["4b"]["per_item"][it["id"]]["first_rank"]
        deltas.append(((a or 31) - (b or 31), it, a, b))
    deltas.sort(key=lambda x: -x[0])
    for d, it, a, b in deltas[:5] + deltas[-3:]:
        print(f"  {it['id']} RRF 第 {a or '30+'} → 8B 第 {b or '30+'}  [{it['lang']}] {it['question'][:70]}")

    out = Path(f"eval/reports/rag_offline_{date.today():%Y%m%d}.json")
    out.write_text(json.dumps({
        "rows": {r: {"name": v["name"], "lat": v["lat"],
                     "per_item": {k: {m: x for m, x in mm.items()} for k, mm in v["per_item"].items()}}
                 for r, v in results.items()},
        "unreachable": unreachable,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n逐题结果 → {out}")


if __name__ == "__main__":
    main()
