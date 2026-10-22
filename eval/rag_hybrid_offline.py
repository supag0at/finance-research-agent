"""混合检索的离线测试(M9 Step 2)。零费用、确定性,不连 Qdrant / SiliconFlow / Postgres。

    python -m eval.rag_hybrid_offline

  ① RRF:用设计文档 §二 例子里那组名次手算核对;
  ② BM25:tf 项、idf 公式(与 Step 0 在 Qdrant 上实测的 0.105 / 1.204 对齐)、分词;
  ③ 降级:rerank 抛异常 → 退回 RRF 顺序;embedding 抛异常 → 只用 sparse;阈值过滤。
"""

from rag import hybrid
from rag.fusion import rrf
from rag.sparse import bm25_tf, doc_vector, idf, query_vector, tokenize

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def test_rrf() -> None:
    print("① RRF")
    dense = ["C5", "C1", "C3", "C7", "C2"]   # 术语表 C5 在 dense 里排第 1
    sparse = ["C1", "C2", "C7", "C3"]        # C5 在关键词那一路根本没出现
    out = dict(rrf([dense, sparse]))
    order = [cid for cid, _ in rrf([dense, sparse])]
    check("C1 = 1/62 + 1/61 = 0.0325", abs(out["C1"] - (1 / 62 + 1 / 61)) < 1e-12, f"{out['C1']:.4f}")
    check("C5 = 1/61 = 0.0164(只在一路出现)", abs(out["C5"] - 1 / 61) < 1e-12, f"{out['C5']:.4f}")
    check("C1 第 1,C5 掉到最后", order[0] == "C1" and order[-1] == "C5", str(order))
    check("只有一路时 = 原顺序", [c for c, _ in rrf([dense])] == dense)
    check("同样输入 → 同样输出(同分稳定)", rrf([dense, sparse]) == rrf([dense, sparse]))
    check("空输入", rrf([]) == [] and rrf([[], []]) == [])


def test_bm25() -> None:
    print("② BM25")
    check("idf(N=4, n=4) = 0.105(Step 0 Qdrant 实测)", round(idf(4, 4), 3) == 0.105)
    check("idf(N=4, n=1) = 1.204(Step 0 Qdrant 实测)", round(idf(4, 1), 3) == 1.204)
    short, long_ = bm25_tf(1, 50, 100), bm25_tf(1, 200, 100)
    check("同样出现 1 次,短文档权重更高(长度归一)", short > long_, f"{short:.3f} vs {long_:.3f}")
    sat = [bm25_tf(tf, 100, 100) for tf in (1, 2, 5, 50)]
    check("词频饱和:tf 1→50,权重上不封顶地涨吗?不,趋近 k1+1=2.2", sat[-1] < 2.2 and sat[0] < sat[1] < sat[2],
          " / ".join(f"{x:.2f}" for x in sat))
    toks = tokenize("营业收入 6,497,196,198.68 元,Data Center revenue $115.2 billion, H100")
    check("数字去千分位、英文小写、型号保留", {"6497196198.68", "115.2", "h100", "revenue"} <= set(toks), str(toks))
    check("中文走 jieba", "营业" in toks and "收入" in toks)
    qv = query_vector("revenue revenue growth")
    check("查询侧:重复词只算一次,值都是 1", len(qv["indices"]) == 2 and set(qv["values"]) == {1.0})
    dv = doc_vector("revenue revenue growth", avgdl=100)
    check("文档侧:values 是 tf 项而不是原始词频", all(0 < v < 2.2 for v in dv["values"]), str(dv["values"]))


class FakeStore:
    """dense 返回 [C5, C1, C3],sparse 返回 [C1, C2]。"""

    def query(self, query, using, limit, flt=None, with_payload=True):
        ids = ["C5", "C1", "C3"] if using == "dense" else ["C1", "C2"]
        return [{"id": i, "score": 1.0 - n / 10, "payload": {"text": f"text {i}", "parent_id": f"P{i}",
                                                            "source": "s", "doc_type": "10-K", "section": ""}}
                for n, i in enumerate(ids)]


def test_degrade() -> None:
    print("③ 降级与阈值")
    hybrid._store = FakeStore()
    hybrid.embed_query = lambda q: [0.0]

    def boom(*a, **kw):
        raise RuntimeError("模拟失败")

    hybrid.rerank = boom
    r = hybrid.search("q", expand=False)
    rrf_order = [c for c, _ in rrf([["C5", "C1", "C3"], ["C1", "C2"]])]
    check("rerank 失败 → degraded 记 rerank", r.degraded == ["rerank"], str(r.degraded))
    check("rerank 失败 → 结果 = RRF 顺序", [c.id for c in r.ranked] == rrf_order, str([c.id for c in r.ranked]))
    check("rerank 失败 → 仍返回结果", len(r.hits) == 4)

    hybrid.embed_query = boom
    hybrid.rerank = lambda q, docs, model=None: [0.5] * len(docs)
    r = hybrid.search("q", expand=False)
    check("embedding 失败 → degraded 记 dense,只剩 sparse 两条", r.degraded == ["dense"] and len(r.hits) == 2,
          f"{r.degraded} {[h['chunk_id'] for h in r.hits]}")

    hybrid.embed_query = lambda q: [0.0]
    hybrid.rerank = lambda q, docs, model=None: [{"text C1": 0.9, "text C2": 0.2}.get(d.split("\n")[-1], 0.01) for d in docs]
    r = hybrid.search("q", expand=False, rerank_threshold=0.1)
    check("阈值 0.1:只留 C1、C2,且 rerank 把顺序定为 C1 > C2", [h["chunk_id"] for h in r.hits] == ["C1", "C2"],
          str([h["chunk_id"] for h in r.hits]))
    r = hybrid.search("q", expand=False, use_rerank=False, use_sparse=False)
    check("消融开关:只开 dense = dense 原顺序", [h["chunk_id"] for h in r.hits] == ["C5", "C1", "C3"])


def main() -> None:
    test_rrf()
    test_bm25()
    test_degrade()
    print()
    print("全部通过" if not _failures else f"失败 {len(_failures)} 项:{_failures}")
    raise SystemExit(1 if _failures else 0)


if __name__ == "__main__":
    main()
