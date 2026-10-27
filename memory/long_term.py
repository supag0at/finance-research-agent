"""长期记忆:跨会话的语义记忆。

⚠️ **LEGACY(M7 起)**:这是 M3.5 的旧实现,保留有两个原因 ——
  1. 手写版 `core/agent.py` 仍在使用;
  2. 作为 M7 消融评测的 `legacy` 对照档(`GraphAgent(memory_mode="legacy")`)。
**不要在它上面继续加功能。** 已知缺陷(M7 评测实测):没有用户隔离(隔离题 0/3 全泄漏)、
整段对话原文入库、无阈值注入 top3、无时间/取代关系。新实现见 `memory/{schema,store,write,recall}.py`,
设计与结果见 docs/design/M7-记忆强化.md。

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
