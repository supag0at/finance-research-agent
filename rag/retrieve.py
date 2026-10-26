"""在线检索(online retrieval)——RAG 的"在线阶段"。

对 Agent 来说,RAG 就浓缩成这一个函数 retrieve():给一句话,返回最相关的
几段知识。内部怎么向量化、怎么在 Qdrant 里搜,Agent 全都不用管——这就是
我们反复强调的"接口稳定、实现可替换"。

这个函数会被注册成一个工具,让 Agent 在需要"背景知识/历史资料"时调用。
"""

from rag.config import RAG_MODE
from rag.embedding import embed_query
from rag.store import QdrantStore

_store = QdrantStore()


_DESCRIPTIONS = {
    False: "从本地知识库按语义检索背景资料(公司背景、行业知识、术语解释等)。需要背景知识时调用。",
    True: (
        "检索本地财报知识库:NVDA/AMD/INTC/AVGO/MU 的 10-K、10-Q,寒武纪/中芯国际/海光信息的年报、半年报,"
        "以及术语表。问财报里的数字、业务、原因,或两家公司对比时调用,直接传用户的原问题即可(内部会拆解、改写)。"
        "返回带 ref 编号的证据,第一项是汇总(coverage 里 missing 的部分表示没找到),之后是证据。实时股价、今天的新闻不要用它。"
    ),
}
# 工具描述跟着 RAG_MODE 走:agentic 模式下要告诉模型库里有什么、直接传原问题、怎么读汇总。
# 放在这里而不是 graph/ 下:手写版 main.py 也要用,而手写版不依赖 LangChain。
RETRIEVE_DESC = _DESCRIPTIONS[RAG_MODE == "agentic"]


def retrieve(query: str, top_k: int = 4) -> list:
    """按 RAG_MODE 分派(M9):v1 = 下面的原链路;hybrid = rag/hybrid.py。

    返回形态向后兼容:每项都有 text / source / title / score;hybrid 额外带
    chunk_id / section / page / fiscal_year / trust_level 等字段(不变式 5)。
    """
    if RAG_MODE == "hybrid":
        from rag.hybrid import search

        return _for_agent(search(query, top_k=top_k).hits)
    if RAG_MODE == "agentic":
        # 子图自己决定每个需求取几条(单需求 5 条、多需求各 3 条),top_k 不再生效。
        # 返回的第一项是 type=summary 的汇总:覆盖情况、缺失、来源性质、降级。
        from rag.agentic.build import run

        return _for_agent(run(query))
    return _retrieve_v1(query, top_k)


def _for_agent(items: list[dict]) -> list[dict]:
    """交给主 Agent 前去掉 child_text:它是父块 text 的子串,原样再给一遍只会让工具结果翻倍变长
    (M9 §7.8:retrieve 返回 11~16k 字,其中一大块是这份重复)。内部的 grade 仍然用 child_text。"""
    return [{k: v for k, v in d.items() if k != "child_text"} for d in items]


def _retrieve_v1(query: str, top_k: int = 4) -> list:
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
