"""重排(rerank)——M9 Step 2。SiliconFlow 的 Qwen3-Reranker-0.6B(cross-encoder)。

双塔(dense)是"问题一个向量、文档一个向量,算夹角":快,可以预先算好 1 万个文档向量,
但问题和文档在编码时互相看不见。cross-encoder 把"问题 + 文档"拼在一起整体过一遍模型,
能看到"问的是 FY2026、这段写的是 FY2025"这种细节,所以准,但每一对都要现算,
只能用在召回之后的几十个候选上 —— 这就是"先召回再精排"两阶段的原因。

Step 0 实测:分数两极分化(相关 ≈ 0.99,不相关 ≈ 0.00x),阈值要在 Step 3 用真实分布标定。
调用失败时抛异常,由调用方降级(rag/hybrid.py:退回 RRF 顺序,标 degraded)。
"""

import httpx

from rag.config import EMBED_API_KEY, EMBED_BASE_URL, RERANK_MODEL

# 与 embedding 同一个 SiliconFlow 账号;trust_env=False 直连(同 rag/embedding.py)。
_http = httpx.Client(trust_env=False, timeout=30)


def rerank(query: str, documents: list[str]) -> list[float]:
    """返回与 documents 一一对应的相关度分数(0~1)。"""
    if not documents:
        return []
    resp = _http.post(
        f"{EMBED_BASE_URL}/rerank",
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": RERANK_MODEL, "query": query, "documents": documents, "return_documents": False},
    )
    resp.raise_for_status()
    scores = [0.0] * len(documents)
    for r in resp.json()["results"]:
        scores[r["index"]] = r["relevance_score"]
    return scores
