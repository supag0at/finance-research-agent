"""read_tool_result:让模型按需取回被截断或已归档的工具结果原文。

原文就在 state["messages"] 里(messages 只追加、随 checkpointer 落库),所以不需要
另外的存储 —— 回读只是"换一个窗口看同一份日志"。用 InjectedState 拿 state,
模型看到的 schema 里没有 state 这个参数,它也就没法伪造。

何时调用完全由模型决定,代码从不自动回读:预览够用时多一次回读就是多一次 LLM 调用。

不经 _reliable_tool 包装(理由同 search_memory):可靠性包装把调用丢进线程池,
注入的运行时参数传不过去;读本地内存也不需要重试和熔断。
"""

from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.sandbox import wrap_untrusted
from memory import events
from memory.context.config import ContextConfig
from memory.context.tokens import text_of


@tool("read_tool_result")
def read_tool_result(id: str, offset: int = 1500, state: Annotated[dict, InjectedState] = None) -> str:
    """回读某条工具结果的原文片段(每次最多 1500 字)。只在两种情况下调用:
    ① 工具结果末尾有"[结果已截断:id=…]"提示,而你需要的内容在后文;
    ② "此前调研归档"里某圈的要点不够用,需要看那圈的原文细节。
    预览或归档已经足够回答时不要调用。id 取自截断提示或归档条目里的 id,offset 为起始字符位置。"""
    chunk_chars = ContextConfig.from_env().read_chunk_chars
    messages = (state or {}).get("messages") or []
    target = None
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.tool_call_id == id:
            target = m
            break
    if target is None:
        events.emit("ctx_readback", found=False, offset=offset)
        return f"没有找到 id={id} 的工具结果,请检查截断提示或归档条目中的 id。"
    if target.name == "read_tool_result":
        # 回读结果只是原文的一个窗口,对它再回读没有意义;把模型指回原始结果。
        source = next((tc.get("args", {}).get("id") for m in messages if isinstance(m, AIMessage)
                       for tc in m.tool_calls or [] if tc.get("id") == id), None)
        events.emit("ctx_readback", found=False, offset=offset)
        return (f"id={id} 是一次回读的结果,不是原始工具结果。"
                + (f'请改用原始结果的 id="{source}"。' if source else "请改用截断提示里的原始 id。"))

    text = text_of(target.content)
    n = len(text)
    offset = max(int(offset or 0), 0)
    events.emit("ctx_readback", found=True, offset=offset)
    if offset >= n:
        return f"该结果已读到末尾(全长 {n} 字)。"
    end = min(offset + chunk_chars, n)
    nav = f"下一段 offset={end}" if end < n else "已到末尾"
    return (
        wrap_untrusted(text[offset:end], source=f"read_tool_result←{target.name or '?'}")
        + f"\n[第 {offset}–{end} 字 / 共 {n} 字;{nav}]"
    )
