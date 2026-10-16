"""从 messages 里确定性地抽出本轮失败记录,供钉住区使用。

为什么扫 messages,而不是读 core/failure.current_log():
  - MCP 模式下 run_reliably 跑在服务端进程,本进程的 FailureLog 是空的;
  - messages 随 checkpointer 落库,时间旅行回到任何快照都能重新推出同样的结果;
    模块级单例做不到这一点。
失败文本由 core/failure.render 生成,首行格式固定为 "[工具失败:<工具>] <类型>"。
"""

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from core.failure import FAIL_PREFIX, target_of
from memory.context.tokens import text_of
from memory.context.types import FailureNote


def _kind_of(text: str) -> str:
    # 另一种形态是 reliability 的 "[工具失败,且降级也失败] <异常>",它的 ] 后面是异常原文
    if not text.startswith(FAIL_PREFIX + ":"):
        return "降级也失败"
    first = text.split("\n", 1)[0]
    return first.split("]", 1)[1].strip()[:20] if "]" in first else "失败"


def extract_failures(messages: list[AnyMessage], body: int, limit: int) -> list[FailureNote]:
    args_by_id: dict[str, dict] = {}
    for m in messages[body:]:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                args_by_id[tc.get("id", "")] = tc.get("args") or {}

    notes: dict[tuple[str, str], FailureNote] = {}
    order: list[tuple[str, str]] = []
    for m in messages[body:]:
        if not isinstance(m, ToolMessage):
            continue
        text = text_of(m.content)
        if not text.startswith(FAIL_PREFIX):
            continue
        tool = m.name or "?"
        key = (tool, target_of(args_by_id.get(m.tool_call_id, {})))
        if key in notes:
            n = notes[key]
            n["count"] += 1
            n["kind"] = _kind_of(text)
            n["source_id"] = m.tool_call_id
            order.remove(key)
        else:
            notes[key] = {
                "tool": key[0],
                "target": key[1],
                "kind": _kind_of(text),
                "count": 1,
                "source_id": m.tool_call_id,
            }
        order.append(key)  # 按最近一次出现排序
    return [notes[k] for k in order[-limit:]]
