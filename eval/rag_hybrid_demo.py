"""混合检索真库演示(M9 Step 2):把每个候选在各阶段的名次摊开看。会调 embedding + rerank(几分钱)。

    python -m eval.rag_hybrid_demo                 # 内置的几条查询
    python -m eval.rag_hybrid_demo "你的问题"       # 单条

每条查询打印"最终前 5 + 任一阶段进过前 5 的块",列:dense 名次 / sparse 名次 / RRF 名次 / rerank 名次与分数。
这不是评测(没有标准答案),是让人直觉上看到每一路各自捞到了什么、融合和重排各自改变了什么。
正式的 Recall@k / MRR 在 Step 3。
"""

import sys

from rag.hybrid import search

QUERIES = [
    ("NVIDIA data center revenue growth fiscal 2026", None),          # Step 1 的封面页 badcase
    ("英伟达 FY2026 数据中心收入增长", None),                            # 中文问英文库
    ("寒武纪2025年营业收入同比增长多少", None),
    ("H20 export license China revenue impact", None),                # 型号类关键词
    ("What drove data center revenue growth?",
     {"must": [{"key": "ticker", "match": {"value": "AMD"}}, {"key": "fiscal_year", "match": {"value": 2025}}]}),
]


def _r(x) -> str:
    return "—" if x is None else str(x)


def show(query: str, flt: dict | None) -> None:
    res = search(query, flt=flt, top_k=5)
    print(f"\n{'=' * 110}\nQ: {query}" + (f"   filter={flt}" if flt else ""))
    t = res.timings_ms
    print("耗时 ms:" + "  ".join(f"{k} {v:.0f}" for k, v in t.items()) + (f"   降级:{res.degraded}" if res.degraded else ""))
    final = {c.id for c in res.ranked[:5]}
    show_ids = [c.id for c in res.ranked[:5]] + [
        cid for cid, c in res.trace.items()
        if cid not in final and any(r is not None and r <= 5 for r in (c.dense_rank, c.sparse_rank, c.rrf_rank))
    ]
    print(f"  {'dense':>5} {'sparse':>6} {'RRF':>4} {'rerank':>6} {'分数':>6}  {'文档':<20} {'页':>4}  内容")
    for cid in show_ids:
        c = res.trace[cid]
        pl = c.payload
        mark = "★" if cid in final else " "
        score = "—" if c.rerank_score is None else f"{c.rerank_score:.3f}"
        text = pl["text"].replace("\n", " ")[:58]
        print(f"{mark} {_r(c.dense_rank):>5} {_r(c.sparse_rank):>6} {_r(c.rrf_rank):>4} {_r(c.rerank_rank):>6} {score:>6}"
              f"  {pl.get('title', '')[:20]:<20} {_r(pl.get('page')):>4}  {text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]), None)
    else:
        for q, f in QUERIES:
            show(q, f)
