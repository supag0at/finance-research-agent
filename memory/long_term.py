"""长期记忆:跨会话的语义记忆。

本质就是"对一个记忆库做 RAG"——复用 RAG 那套 embedding + Qdrant 基建,只是
换一个集合 fra_memory,存的不是文档而是"记忆条目"。

两个操作:
    remember(text):把一条值得记的内容向量化后存进记忆库。
    recall(query): 按语义把相关记忆捞回来。

优雅降级:如果没配 EMBED_API_KEY,remember 变成空操作、recall 返回空列表,
且任何异常都不会让 Agent 崩溃(记忆是"锦上添花",不该阻断主流程)。
"""

import time
import uuid

from rag.config import EMBED_API_KEY, EMBED_DIM
from rag.embedding import embed_query, embed_texts
from rag.store import QdrantStore

MEM_COLLECTION = "fra_memory"

_store = QdrantStore(collection=MEM_COLLECTION)
_ready = False


def _enabled() -> bool:
    # 没配 embedding key 就整体禁用(优雅降级)。
    return bool(EMBED_API_KEY)


def _ensure_collection() -> None:
    global _ready
    if not _ready:
        _store.ensure_collection(EMBED_DIM)
        _ready = True


def remember(text: str, metadata: dict | None = None) -> None:
    """把一条内容写入长期记忆。失败时静默跳过(不影响主流程)。"""
    if not _enabled() or not text.strip():
        return
    try:
        _ensure_collection()
        vector = embed_texts([text])[0]
        payload = {"text": text, "ts": time.time()}
        if metadata:
            payload.update(metadata)
        _store.upsert([{"id": str(uuid.uuid4()), "vector": vector, "payload": payload}])
    except Exception:
        # 记忆写入失败不该让整个 Agent 挂掉。
        pass

    # TODO(记忆增强,见 memory/MEMORY.md):这里可加"写入筛选"(用 LLM 判断值不值得记)、
    #   记忆分型、更新/冲突消解、遗忘/衰减等。


def recall(query: str, top_k: int = 3) -> list[str]:
    """按语义检索相关记忆,返回文本列表。失败或未启用时返回空列表。"""
    if not _enabled() or not query.strip():
        return []
    try:
        _ensure_collection()
        vector = embed_query(query)
        hits = _store.search(vector, top_k)
    except Exception:
        return []

    # TODO(记忆增强,见 memory/MEMORY.md):检索评分可加"相关性 + 新近性 + 重要性"
    #   三因子加权(参考 Generative Agents),而不仅是向量相似度。
    return [h.get("payload", {}).get("text", "") for h in hits if h.get("payload")]
