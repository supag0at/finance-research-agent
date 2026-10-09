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

    def search(
        self,
        vector: list[float],
        top_k: int = 4,
        flt: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """给一个查询向量,返回最相近的 top_k 个点(含 payload 和相似度分数)。

        flt:Qdrant 过滤条件,先过滤再按向量排序(M7:记忆按 user_id 隔离靠它)。
        score_threshold:低于该余弦相似度的点直接不返回(M7:拒答靠它)。
        两个新参数默认 None,原有调用方行为不变。
        """
        body: dict = {"vector": vector, "limit": top_k, "with_payload": True}
        if flt:
            body["filter"] = flt
        if score_threshold is not None:
            body["score_threshold"] = score_threshold
        resp = self.http.post(f"/collections/{self.collection}/points/search", json=body)
        resp.raise_for_status()
        return resp.json()["result"]

    # ---- M7 新增:记忆需要"不靠向量"的精确操作 ----

    def scroll(self, flt: dict, limit: int = 100) -> list[dict]:
        """按条件取点,不需要向量。用于"取全部有效画像""按 key 精确查"这类场景。"""
        resp = self.http.post(
            f"/collections/{self.collection}/points/scroll",
            json={"filter": flt, "limit": limit, "with_payload": True, "with_vector": False},
        )
        resp.raise_for_status()
        return resp.json()["result"]["points"]

    def set_payload(self, ids: list[str], payload: dict) -> None:
        """只改 payload 的指定字段,不动向量。用于把旧记忆标成 superseded。"""
        if not ids:
            return
        resp = self.http.post(
            f"/collections/{self.collection}/points/payload?wait=true",
            json={"payload": payload, "points": ids},
        )
        resp.raise_for_status()

    def ensure_index(self, field: str, schema: str = "keyword") -> None:
        """给 payload 字段建索引(重复建是幂等的)。要拿来过滤的字段都该建。"""
        resp = self.http.put(
            f"/collections/{self.collection}/index?wait=true",
            json={"field_name": field, "field_schema": schema},
        )
        resp.raise_for_status()

    def delete_by_filter(self, flt: dict) -> None:
        resp = self.http.post(
            f"/collections/{self.collection}/points/delete?wait=true",
            json={"filter": flt},
        )
        resp.raise_for_status()

    def drop_collection(self) -> None:
        """删除整个集合(评测/测试清理用)。"""
        self.http.delete(f"/collections/{self.collection}")
