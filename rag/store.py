"""向量库封装(Qdrant)——RAG 三层里的第②层。

职责很单纯:存向量、按相似度检索。它不产生向量(那是 embedding 的事),
也不管原文的结构化存储(那是 Postgres 的事)。

我们用 httpx 直接调 Qdrant 的 REST 接口(而不是 qdrant-client 库),原因:
  1. 少一个依赖;
  2. 能用 trust_env=False 精确绕过 Clash 代理对 127.0.0.1 的拦截。
"""

import httpx

from rag.config import EMBED_DIM, QDRANT_COLLECTION, QDRANT_URL


class QdrantStore:
    def __init__(self, url: str = QDRANT_URL, collection: str = QDRANT_COLLECTION) -> None:
        self.collection = collection
        # trust_env=False:绕过 Clash 全局代理,直连本地 Qdrant。
        self.http = httpx.Client(base_url=url, timeout=30, trust_env=False)

    def ensure_collection(self, dim: int = EMBED_DIM) -> None:
        """确保集合存在;不存在就按指定维度 + 余弦距离创建。"""
        resp = self.http.get(f"/collections/{self.collection}")
        if resp.status_code == 200:
            return
        # 404 等 -> 创建
        create = self.http.put(
            f"/collections/{self.collection}",
            json={"vectors": {"size": dim, "distance": "Cosine"}},
        )
        create.raise_for_status()

    def upsert(self, points: list[dict]) -> None:
        """写入/更新一批点。每个点形如:
        {"id": <uuid或int>, "vector": [...], "payload": {...}}
        """
        if not points:
            return
        resp = self.http.put(
            f"/collections/{self.collection}/points?wait=true",
            json={"points": points},
        )
        resp.raise_for_status()

    def search(self, vector: list[float], top_k: int = 4) -> list[dict]:
        """给一个查询向量,返回最相近的 top_k 个点(含 payload 和相似度分数)。"""
        resp = self.http.post(
            f"/collections/{self.collection}/points/search",
            json={"vector": vector, "limit": top_k, "with_payload": True},
        )
        resp.raise_for_status()
        return resp.json()["result"]
