"""在线检索(online retrieval)——RAG 的"在线阶段"。

对 Agent 来说,RAG 就浓缩成这一个函数 retrieve():给一句话,返回最相关的
几段知识。内部怎么向量化、怎么在 Qdrant 里搜,Agent 全都不用管——这就是
我们反复强调的"接口稳定、实现可替换"。

这个函数会被注册成一个工具,让 Agent 在需要"背景知识/历史资料"时调用。
"""

from rag.embedding import embed_query
from rag.store import QdrantStore

_store = QdrantStore()


def retrieve(query: str, top_k: int = 4) -> list:
    """按语义检索知识库,返回最相关的若干段文本。

    参数:
        query: 检索问题,例如 "英伟达的主要业务板块"。
        top_k: 返回几段,默认 4。
    返回:
        列表,每项含 text / source / title / score(相似度分数)。
    """
    # TODO(RAG增强,见 rag/ENHANCEMENTS.md ②查询侧):检索前可先做
    #   查询改写 / Multi-Query / HyDE,把 query 处理得更适合检索。
    # 例如:query = rewrite_query(query)

    vector = embed_query(query)

    # TODO(RAG增强,见 ENHANCEMENTS.md ③粗排):这里可换成混合检索
    #   (dense + BM25 + RRF)与元数据过滤,而不仅是纯向量相似度。
    hits = _store.search(vector, top_k)

    # TODO(RAG增强,见 ENHANCEMENTS.md ④检索后):拿到 hits 后可加
    #   重排 Rerank(bge-reranker)/ MMR 去冗余 / 去重 / 上下文压缩。
    results = []
    for h in hits:
        payload = h.get("payload", {})
        results.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "title": payload.get("title", ""),
                "score": round(h.get("score", 0.0), 3),
            }
        )
    return results
