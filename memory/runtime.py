"""记忆的运行时单例:store 与 LLM 的获取点(M7)。

为什么单独一个模块:图节点(graph/nodes.py)、search_memory 工具(memory/tools.py)、
评测 runner 都要拿到**同一个** MemoryStore。集中在这里有两个好处:
  1. 评测可以用 `set_store()` 给每道题换一个临时集合(题与题之间互不污染),
     不需要改任何业务代码;
  2. 延迟创建 —— 没开 v2 的运行完全不会碰 Qdrant/embedding。
"""

from typing import Optional

from memory.store import MemoryStore

_store: Optional[MemoryStore] = None
_llm = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def set_store(store: Optional[MemoryStore]) -> None:
    """替换(或传 None 重置)全局 store。评测/测试用。"""
    global _store
    _store = store


def get_llm():
    """抽取器/冲突判定用的模型。复用 core.llm.LLM(自带 complete_json)。"""
    global _llm
    if _llm is None:
        from core.llm import LLM

        _llm = LLM()
    return _llm


def set_llm(llm) -> None:
    global _llm
    _llm = llm
