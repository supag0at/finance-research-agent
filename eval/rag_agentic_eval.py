"""消融第 5~7 行 + 拒答统计(M9 Step 4)。调 LLM + embedding + rerank 8B,约 15~25 分钟、一两块钱。

    python -m eval.rag_agentic_eval

和 eval/rag_offline.py 不同:这里比的是**交给主 Agent 的最终证据列表**(不是深度 30 的排名),
因为 agentic 子图输出的就是一份去重、限额后的列表(单需求 5 条,多需求各 3 条)。

    4b*  Step 3 的终点:hybrid + rerank 8B + 阈值 0.5,取前 5            (对照组)
    5    + query 分析:拆需求、过白名单的过滤、英文子查询;不评估、不重检
    6    5 + HyDE(拿"假想答案"做 dense 查询)
    7    完整 agentic:5 + grade + 改写重检(≤2 轮);**联网关闭**,只看知识库本身

指标(fact 72 题 + compare 3 题,无答案 10 题单列):
    命中    最终列表里有没有相关块(fact)
    MRR     第一个相关块在列表里的名次倒数
    覆盖    compare 题两个答案点各自被捞到的比例
    拒答对  无答案题:最终没有交出任何知识库证据(行 7 = coverage 全 missing;其他行 = 列表为空)
    误拒    可答题:相关块明明在列表里,却被判 missing 扔掉了(只有行 7 会发生)
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from eval import rag_offline as ro  # 导入即启用 embedding 缓存(各行查询向量一致)
from rag import hybrid
from rag.agentic.build import run

ROWS = [
    ("4b*", "hybrid + rerank 8B + 阈值(Step 3 终点)", None),
    ("5", "+ query 分析(不评估)", {"grade": False, "max_rounds": 1, "web": False, "hyde": False}),
    ("6", "5 + HyDE", {"grade": False, "max_rounds": 1, "web": False, "hyde": True}),
    ("7", "完整 agentic(grade + 改写,联网关)", {"grade": True, "max_rounds": 2, "web": False, "hyde": False}),
]


def run_one(cfg, q: str) -> dict:
    t = time.perf_counter()
    if cfg is None:
        res = hybrid.search(q, top_k=5)
        return {"ids": [h["chunk_id"] for h in res.hits], "ms": (time.perf_counter() - t) * 1000,
                "tokens": 0, "rounds": 1, "coverage": {}, "events": [], "dropped_ids": []}
    st = run(q, opts=cfg, full_state=True)
    out = st["output"]
    sm = out[-1]
    kept = [e["chunk_id"] for e in out[:-1] if e.get("chunk_id")]
    # 被判 missing 而没交出去的块(用来算"误拒")
    dropped = [h["chunk_id"] for n in st["needs"] if sm["coverage"][n["id"]]["status"] == "missing" for h in n["hits"]]
    return {"ids": kept, "ms": sm["latency_ms"], "tokens": sm["llm_tokens"], "rounds": sm["rounds"],
            "coverage": {k: v["status"] for k, v in sm["coverage"].items()}, "events": st["events"],
            "dropped_ids": dropped, "degraded": sm["degraded"]}


def main() -> None:
    items = [json.loads(line) for line in ro.GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    v2_texts, _, gen = ro._load_texts()
    for it in items:
        it["_g"] = ro.groups(it, v2_texts, gen) if it["spans"] else []
    facts = [it for it in items if it["kind"] == "fact"]
    comps = [it for it in items if it["kind"] == "compare"]
    nas = [it for it in items if it["kind"] == "no_answer"]

    results = {}
    with ThreadPoolExecutor(4) as pool:
        for row_id, name, cfg in ROWS:
            t0 = time.time()
            outs = list(pool.map(lambda it: run_one(cfg, it["question"]), items))
            results[row_id] = dict(zip([it["id"] for it in items], outs))
            print(f"  行 {row_id:<4} {name:<36} 跑完 {time.time() - t0:5.0f}s", flush=True)

    def hit_rank(r, it):
        union = set().union(*it["_g"]) if it["_g"] else set()
        return next((i for i, cid in enumerate(r["ids"], 1) if cid in union), None)

    print(f"\n## 第 4b*~7 行(最终证据列表;fact {len(facts)} / compare {len(comps)} / 无答案 {len(nas)})\n")
    print("| 行 | 配置 | 命中 | MRR | compare 覆盖 | 平均证据条数 | 无答案拒答对 | 可答题误拒 | LLM token / 题 | 时延 p50 / p95 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row_id, name, cfg in ROWS:
        R = results[row_id]
        ranks = [hit_rank(R[it["id"]], it) for it in facts]
        hit = sum(r is not None for r in ranks) / len(facts)
        mrr = sum(1 / r for r in ranks if r) / len(facts)
        cov = statistics.mean(sum(bool(g & set(R[it["id"]]["ids"])) for g in it["_g"]) / len(it["_g"]) for it in comps)
        n_ev = statistics.mean(len(R[it["id"]]["ids"]) for it in items if it["kind"] != "no_answer")
        refuse = sum(not R[it["id"]]["ids"] for it in nas)
        # 误拒:相关块被检到了,但因为需求被判 missing 没交出去
        false_ref = 0
        for it in facts + comps:
            union = set().union(*it["_g"])
            r = R[it["id"]]
            if union & set(r["dropped_ids"]) and not union & set(r["ids"]):
                false_ref += 1
        tok = statistics.mean(R[it["id"]]["tokens"] for it in items)
        lat = sorted(R[it["id"]]["ms"] for it in items)
        print(f"| {row_id} | {name} | {hit:.3f} | {mrr:.3f} | {cov:.2f} | {n_ev:.1f} | {refuse}/{len(nas)} | "
              f"{false_ref}/{len(facts) + len(comps)} | {tok:.0f} | {lat[len(lat) // 2] / 1000:.1f}s / {lat[int(len(lat) * .95)] / 1000:.1f}s |")

    print("\n## 分组命中(fact)\n")
    buckets = {
        "中文问英文财报": lambda it: it["lang"] == "zh" and it["spans"][0]["source"].startswith("sec/"),
        "问中文年报": lambda it: it["spans"][0]["source"].startswith("cninfo/"),
        "英文问题": lambda it: it["lang"] == "en",
    }
    print("| 分组 | 题数 | " + " | ".join(f"行 {r}" for r, _, _ in ROWS) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in ROWS) + " |")
    for bname, f in buckets.items():
        sel = [it for it in facts if f(it)]
        cells = [f"{sum(hit_rank(results[r][it['id']], it) is not None for it in sel) / len(sel):.2f}" for r, _, _ in ROWS]
        print(f"| {bname} | {len(sel)} | " + " | ".join(cells) + " |")

    print("\n## 行 7 的路径分布\n")
    R7 = results["7"]
    paths = {"快速通道": 0, "grade 一次判够": 0, "改写后判够": 0, "两轮仍缺": 0, "analyze 降级": 0}
    for it in items:
        ev = R7[it["id"]]["events"]
        if "analyze" in (R7[it["id"]].get("degraded") or []):
            paths["analyze 降级"] += 1
        if any("快速通道" in e for e in ev):
            paths["快速通道"] += 1
        elif R7[it["id"]]["rounds"] == 1 and all(v == "found" for v in R7[it["id"]]["coverage"].values()):
            paths["grade 一次判够"] += 1
        elif all(v == "found" for v in R7[it["id"]]["coverage"].values()):
            paths["改写后判够"] += 1
        else:
            paths["两轮仍缺"] += 1
    print("  " + ",".join(f"{k} {v}" for k, v in paths.items()))

    print("\n## 无答案题在行 7 的处理\n")
    for it in nas:
        r = R7[it["id"]]
        print(f"  {it['question'][:40]:<42} coverage={r['coverage']} 交出 {len(r['ids'])} 条")

    print("\n## 行 7 被误拒的可答题(相关块检到了却被 grade 判 missing)\n")
    for it in facts + comps:
        union = set().union(*it["_g"])
        r = R7[it["id"]]
        if union & set(r["dropped_ids"]) and not union & set(r["ids"]):
            g = [e for e in r["events"] if e.startswith("grade")]
            print(f"  {it['id']} {it['question'][:60]}\n      {g[-1][:120] if g else ''}")

    print("\n## 行 5 比 4b* 多命中 / 少命中的题\n")
    for it in facts:
        a, b = hit_rank(results["4b*"][it["id"]], it), hit_rank(results["5"][it["id"]], it)
        if (a is None) != (b is None):
            ev = [e for e in results["5"][it["id"]]["events"] if e.startswith("analyze")]
            print(f"  {'+' if b else '-'} {it['id']} [{it['lang']}] {it['question'][:60]}\n      {ev[0][:150] if ev else ''}")

    out = ro.Path(f"eval/reports/rag_agentic_{ro.date.today():%Y%m%d}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n逐题结果 → {out}")


if __name__ == "__main__":
    main()
