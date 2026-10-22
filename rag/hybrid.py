"""混合检索链路(M9 Step 2):dense + sparse → RRF → rerank → 小块换大块。

    query ──┬─ dense  top30(余弦,语义)  ──┐
            └─ sparse top30(BM25,关键词)──┴─ RRF → 前 30 → rerank(cross-encoder)→ 前 5 子块 → 换成父块

每个阶段都可以单独关掉(use_dense / use_sparse / use_rerank),Step 3 的消融表就是这么跑出来的。
search() 返回的不只是最终结果,还有每个候选在各阶段的名次和每阶段耗时(HybridResult.trace),
方便直接看"某个块在 dense 排第几、BM25 排第几、融合后第几、rerank 后第几"。

降级:rerank 调用失败 → 退回 RRF 顺序,degraded 里记 "rerank";dense 的 embedding 失败 → 只用 sparse,记 "dense"。
检索本身永远不因为某一路失败而抛异常(不变式 1),除非所有路都失败。
"""

import time
from dataclasses import dataclass, field

from rag import db
from rag.chunk import context_header
from rag.config import QDRANT_COLLECTION_V2, RAG_TOP_K, RECALL_K, RERANK_CANDIDATES, RERANK_THRESHOLD
from rag.embedding import embed_query
from rag.fusion import rrf
from rag.rerank import rerank
from rag.sparse import query_vector
from rag.store import QdrantStore

_store = QdrantStore(collection=QDRANT_COLLECTION_V2)
_RERANK_DOC_CHARS = 1500  # 子块最长 ~1200 字符,加 header 足够;防止个别超长块拖慢 rerank


@dataclass
class Candidate:
    id: str
    payload: dict
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None


@dataclass
class HybridResult:
    hits: list[dict]                      # 最终结果(父块文本 + 元数据),给 Agent 用
    ranked: list[Candidate]               # 最终排序的子块(去父块之前),给评测用
    trace: dict[str, Candidate]           # 所有进过任一阶段的候选
    timings_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


def _rerank_text(payload: dict) -> str:
    return f"{context_header(payload, payload.get('section') or '')}\n{payload['text']}"[:_RERANK_DOC_CHARS]


def search(
    query: str,
    flt: dict | None = None,
    top_k: int = RAG_TOP_K,
    use_dense: bool = True,
    use_sparse: bool = True,
    use_rerank: bool = True,
    expand: bool = True,
    rerank_threshold: float | None = RERANK_THRESHOLD,
) -> HybridResult:
    timings: dict[str, float] = {}
    degraded: list[str] = []
    trace: dict[str, Candidate] = {}
    lists: list[list[str]] = []

    def cand(p: dict) -> Candidate:
        return trace.setdefault(p["id"], Candidate(p["id"], p["payload"]))

    # ① 两路召回
    if use_dense:
        t = time.perf_counter()
        try:
            pts = _store.query(embed_query(query), "dense", RECALL_K, flt)
            for r, p in enumerate(pts, 1):
                c = cand(p)
                c.dense_rank, c.dense_score = r, p["score"]
            lists.append([p["id"] for p in pts])
        except Exception:  # noqa: BLE001 —— embedding 境外链路抖动,降级成只用关键词
            degraded.append("dense")
        timings["dense"] = (time.perf_counter() - t) * 1000
    if use_sparse:
        t = time.perf_counter()
        qv = query_vector(query)
        pts = _store.query(qv, "sparse", RECALL_K, flt) if qv["indices"] else []
        for r, p in enumerate(pts, 1):
            c = cand(p)
            c.sparse_rank, c.sparse_score = r, p["score"]
        lists.append([p["id"] for p in pts])
        timings["sparse"] = (time.perf_counter() - t) * 1000

    # ② 融合。只有一路时 RRF 等价于原排序。
    fused = rrf(lists)[:RERANK_CANDIDATES]
    for r, (cid, s) in enumerate(fused, 1):
        trace[cid].rrf_rank, trace[cid].rrf_score = r, s
    ranked = [trace[cid] for cid, _ in fused]

    # ③ 重排
    if use_rerank and ranked:
        t = time.perf_counter()
        try:
            scores = rerank(query, [_rerank_text(c.payload) for c in ranked])
            for c, s in zip(ranked, scores):
                c.rerank_score = s
            ranked.sort(key=lambda c: -c.rerank_score)
            for r, c in enumerate(ranked, 1):
                c.rerank_rank = r
            if rerank_threshold is not None:
                ranked = [c for c in ranked if c.rerank_score >= rerank_threshold]
        except Exception:  # noqa: BLE001 —— 不变式 1:rerank 挂了退回 RRF 顺序
            degraded.append("rerank")
        timings["rerank"] = (time.perf_counter() - t) * 1000

    # ④ 小块换大块:同一父块只留排名最高的子块,取够 top_k 个父块
    t = time.perf_counter()
    chosen, seen = [], set()
    for c in ranked:
        pid = c.payload["parent_id"]
        if pid in seen:
            continue
        seen.add(pid)
        chosen.append(c)
        if len(chosen) == top_k:
            break
    parents = db.get_parents_v2([c.payload["parent_id"] for c in chosen]) if expand else {}
    if expand:
        timings["expand"] = (time.perf_counter() - t) * 1000

    hits = []
    for c in chosen:
        pl = c.payload
        par = parents.get(pl["parent_id"])
        score = c.rerank_score if c.rerank_score is not None else c.rrf_score
        hits.append({
            "text": par["text"] if par else pl["text"],
            "child_text": pl["text"],
            "source": pl["source"],
            "title": pl.get("title", ""),
            "score": round(score or 0.0, 4),
            "chunk_id": c.id,
            "parent_id": pl["parent_id"],
            "section": pl.get("section", ""),
            "page": pl.get("page"),
            "ticker": pl.get("ticker"),
            "fiscal_year": pl.get("fiscal_year"),
            "trust_level": pl.get("trust_level"),
        })
    return HybridResult(hits, ranked, trace, timings, degraded)
