"""记忆存储层(M7):封装 Qdrant 集合 fra_memory_v2。

## 隔离是在**接口层**保证的,不靠调用方自觉

本类所有读写方法的 `user_id` 都是**无默认值的关键字参数**(`*,` 之后),
调用时少写会直接 TypeError;内部每次查询都由 `_flt()` 强制带上 user_id 过滤,
没有任何一条不带 user_id 的查询路径。目的是让"忘了过滤、于是 A 看到了 B 的记忆"
这类错误在**写代码时**就暴露,而不是上线后才被发现。
(对比 legacy:`memory/long_term.py` 没有 user_id 概念,所有人共用一个集合。)

## 存储方式:不序列化、不拍平

Qdrant 的 payload 原生支持嵌套 JSON,所以 `MemoryRecord` 原样存进去;
只对 `text` 这一句自然语言做向量。要拿来过滤的字段放顶层并建 payload 索引,
`value` 这类嵌套字段只用来带回结构化内容,不参与过滤。

## 可注入的 embedder

默认用 rag.embedding(会调 SiliconFlow,联网计费)。构造时可传 `embed=` 换成
离线实现,这样隔离性、更新逻辑这些**与向量语义无关**的行为可以零成本、确定性地测试
(见 eval/memory_offline.py)。
"""

import os
import time
from typing import Callable, Optional

from memory.schema import MemoryRecord
from rag.config import EMBED_DIM
from rag.store import QdrantStore

MEM_COLLECTION = os.getenv("MEMORY_COLLECTION", "fra_memory_v2")

# 要拿来过滤的字段:必须在顶层且建索引。
_INDEXED = ("user_id", "type", "status", "tickers", "key", "subject")

Embedder = Callable[[list[str]], list[list[float]]]


def _default_embed(texts: list[str]) -> list[list[float]]:
    # 延迟导入:没配 embedding key 的环境(或离线测试)不该在 import 时就出问题。
    from rag.embedding import embed_texts

    return embed_texts(texts)


def _match(key: str, value) -> dict:
    return {"key": key, "match": {"value": value}}


def _flt(user_id: str, *, status: Optional[str] = "active", **eq) -> dict:
    """构造过滤条件。user_id 永远在内;status 默认只看有效记录。"""
    if not user_id or not user_id.strip():
        raise ValueError("user_id 不能为空")
    must = [_match("user_id", user_id)]
    if status is not None:
        must.append(_match("status", status))
    must += [_match(k, v) for k, v in eq.items() if v is not None]
    return {"must": must}


class MemoryStore:
    def __init__(
        self,
        collection: str = MEM_COLLECTION,
        embed: Optional[Embedder] = None,
        dim: int = EMBED_DIM,
    ) -> None:
        self._store = QdrantStore(collection=collection)
        self._embed = embed or _default_embed
        self._dim = dim
        self._ready = False

    @property
    def collection(self) -> str:
        return self._store.collection

    def ensure(self) -> None:
        """建集合 + 建过滤索引。只做一次。"""
        if self._ready:
            return
        self._store.ensure_collection(self._dim)
        for f in _INDEXED:
            self._store.ensure_index(f, "keyword")
        self._ready = True

    # ------------------------------------------------------------------ 写

    def add_many(self, *, user_id: str, records: list[MemoryRecord]) -> None:
        """批量写入。一次 embedding 调用覆盖所有记录(比逐条调便宜、快)。"""
        if not records:
            return
        for r in records:
            if r.user_id != user_id:
                # 防御:调用方传错了 user_id,宁可报错也不要把数据写进别人名下。
                raise ValueError(f"记录的 user_id({r.user_id!r}) 与参数({user_id!r}) 不一致")
        self.ensure()
        vectors = self._embed([r.text for r in records])
        self._store.upsert(
            [{"id": r.id, "vector": v, "payload": r.payload()} for r, v in zip(records, vectors)]
        )

    def supersede(
        self,
        *,
        user_id: str,
        old_id: str,
        new_id: Optional[str],
        status: str = "superseded",
    ) -> None:
        """把旧记忆标为失效,**不物理删除**。

        调用顺序要求:必须在新记录 add_many **成功之后**才调用。
        否则 embedding 一失败,就会留下"旧的已失效、新的没写进去"的状态 ——
        和 M6 ⑥ ingest 先盖章后做事是同一类 bug。

        user_id 用来确认这条旧记忆确实属于该用户,防止越权改别人的记忆。
        """
        self.ensure()
        pts = self._store.scroll({"must": [_match("user_id", user_id), {"has_id": [old_id]}]}, limit=1)
        if not pts:
            raise KeyError(f"用户 {user_id!r} 名下没有记忆 {old_id}")
        self._store.set_payload(
            [old_id],
            {"status": status, "valid_to": time.time(), "superseded_by": new_id},
        )

    def purge_user(self, *, user_id: str) -> None:
        """删除某用户的全部记忆(含已失效的)。评测清理用,也是"被遗忘权"的实现。"""
        self.ensure()
        self._store.delete_by_filter(_flt(user_id, status=None))

    # ------------------------------------------------------------------ 读

    def search(
        self,
        *,
        user_id: str,
        query: str,
        type: Optional[str] = None,
        ticker: Optional[str] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """向量检索:先按 user_id/status/type/ticker 过滤,再按相似度排序,低于阈值的丢弃。"""
        self.ensure()
        vec = self._embed([query])[0]
        flt = _flt(user_id, type=type, tickers=ticker.upper() if ticker else None)
        hits = self._store.search(vec, top_k, flt=flt, score_threshold=threshold)
        return [(MemoryRecord.from_point(h), float(h["score"])) for h in hits]

    def active_semantic(self, *, user_id: str, limit: int = 20) -> list[MemoryRecord]:
        """用户画像:取全部有效的语义记忆。不做向量检索,条数少、整份常驻上下文。"""
        self.ensure()
        pts = self._store.scroll(_flt(user_id, type="semantic"), limit=limit)
        return [MemoryRecord.from_point(p) for p in pts]

    def find_by_key(self, *, user_id: str, key: str, subject: str = "self") -> Optional[MemoryRecord]:
        """冲突检测的第一道:同用户、同 key、同主体的有效记录。精确匹配,不调 LLM。"""
        self.ensure()
        pts = self._store.scroll(_flt(user_id, key=key, subject=subject), limit=1)
        return MemoryRecord.from_point(pts[0]) if pts else None

    def history(self, *, user_id: str, key: str) -> list[MemoryRecord]:
        """某个 key 的全部版本(含已失效),按记录时间升序。回答"它怎么变的"用。"""
        self.ensure()
        pts = self._store.scroll(_flt(user_id, status=None, key=key), limit=100)
        return sorted((MemoryRecord.from_point(p) for p in pts), key=lambda r: r.created_at)

    def recent_episodic(self, *, user_id: str, days: int = 90) -> list[MemoryRecord]:
        """近 N 天的情景记忆,用来生成"调研目录"(NVDA×4 ...)。时间窗在代码里滤。"""
        self.ensure()
        pts = self._store.scroll(_flt(user_id, type="episodic"), limit=500)
        cutoff = time.time() - days * 86400
        return [r for r in (MemoryRecord.from_point(p) for p in pts) if r.created_at >= cutoff]

    def drop(self) -> None:
        """删除整个集合。仅测试/评测用。"""
        self._store.drop_collection()
        self._ready = False
