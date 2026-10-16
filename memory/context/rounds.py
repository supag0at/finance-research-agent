"""把 messages 切成"轮"和"圈"。

  轮(turn)  一次用户提问。同一 thread 里可以有多轮,边界由 recall 写进 state["turns"]。
  圈(round) 本轮调研里的一次"模型决定调工具 + 这些工具的返回"。
            压缩以圈为最小单位 —— 拆开一圈就会出现没有发起者的 ToolMessage,API 直接报错。
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from memory.context.types import TurnBound


@dataclass
class Round:
    no: int  # 本轮第几圈,从 1 起;按 body 起算,所以不会因为归档前移而改号
    start: int  # messages 下标,含
    end: int  # 不含
    ai: Optional[AIMessage] = None
    tools: list[ToolMessage] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [t.tool_call_id for t in self.tools]


def split_rounds(messages: list[AnyMessage], body: int) -> list[Round]:
    """切出 messages[body:] 里的所有圈。

    带 tool_calls 的 AIMessage 开新圈,其后连续的 ToolMessage 归入该圈;
    其他消息(不带 tool_calls 的 AI 等)各自成一圈。
    """
    rounds: list[Round] = []
    cur: Optional[Round] = None
    for i in range(body, len(messages)):
        m = messages[i]
        if isinstance(m, ToolMessage) and cur is not None and cur.ai is not None and cur.ai.tool_calls:
            cur.tools.append(m)
            cur.end = i + 1
            continue
        cur = Round(no=len(rounds) + 1, start=i, end=i + 1)
        if isinstance(m, AIMessage):
            cur.ai = m
        elif isinstance(m, ToolMessage):
            cur.tools.append(m)  # 孤立的 tool(理论上不会出现),单独成圈,不和别人拼
        rounds.append(cur)
    return rounds


def resolve_turns(state: dict) -> list[TurnBound]:
    """取每轮边界。M8 之前创建的 thread 没有 turns,兜底成"整段历史是一轮"。

    兜底的 body 取最后一条 HumanMessage 之后:那条就是本轮问题。
    这种 thread 不做多轮折叠 —— 我们不知道旧轮次的边界,宁可不折叠也不猜错。
    """
    turns = state.get("turns") or []
    if turns:
        return list(turns)
    messages = state.get("messages") or []
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return [{"start": 0, "body": last_human + 1}]


def head_slice(messages: list[AnyMessage], turns: list[TurnBound]) -> tuple[int, int]:
    """本轮受保护头部(记忆块 + 问题原文)在 messages 里的 [head_start, body)。

    会话首轮 start=0,那条 SystemMessage 在视图里单独放在最前,头部要跳过它。
    """
    cur = turns[-1]
    start = cur["start"]
    if start == 0 and messages and isinstance(messages[0], SystemMessage):
        start = 1
    return start, cur["body"]
