"""视图的 token 估算。

不引入 tokenizer:以上一条 AIMessage 的 usage_metadata.input_tokens 为锚 —— 那是服务端
对上一版视图的精确计数;之后新增的内容才按字符估。这样误差只来自"最近一圈",
不会随历史变长而累积。估算误差会通过 ctx_est_error 事件暴露出来,用于校正系数。
"""

from typing import Any, Optional

from langchain_core.messages import AIMessage, AnyMessage


def text_of(content: Any) -> str:
    """消息内容统一成字符串。MCP 工具可能返回 content block 列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if content is None else str(content)


def message_chars(m: AnyMessage) -> int:
    n = len(text_of(m.content))
    for tc in getattr(m, "tool_calls", None) or []:
        n += len(str(tc.get("name", ""))) + len(str(tc.get("args", "")))
    return n


def find_anchor(messages: list[AnyMessage]) -> tuple[int, Optional[AIMessage]]:
    """最后一条带 usage 的 AIMessage 及其下标;没有则 (-1, None)。"""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and (m.usage_metadata or {}).get("input_tokens"):
            return i, m
    return -1, None


def estimate_tokens(
    view: list[AnyMessage],
    anchor: Optional[AIMessage],
    new_since_anchor: list[AnyMessage],
    pinned_chars: int,
    tokens_per_char: float,
    tools_schema_tokens: int,
) -> int:
    """估算视图的 token 数。

    anchor 为 None 时(本轮还没调过模型)整份视图按字符估。
    new_since_anchor 由调用方给出**渲染后**的版本(工具结果按 stub 后的长度算),
    因为模型看到的是视图,不是原始 messages。
    """
    if anchor is None:
        chars = sum(message_chars(m) for m in view)
        return int(chars * tokens_per_char) + tools_schema_tokens
    usage = anchor.usage_metadata or {}
    base = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    delta = sum(message_chars(m) for m in new_since_anchor) + pinned_chars
    return base + int(delta * tokens_per_char)


def usage_tokens(usage: Any) -> tuple[int, int]:
    """openai 的 CompletionUsage(或 dict)→ (输入, 输出)。拿不到就是 (0, 0)。"""
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    return int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)
