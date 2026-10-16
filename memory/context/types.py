"""压缩状态的数据结构。

全部是 TypedDict(运行时就是普通 dict),不用 dataclass:它们要随 AgentState 被
Postgres checkpointer 序列化,普通 dict 最没有意外。
"""

from typing import TypedDict


class TurnBound(TypedDict):
    start: int  # 本轮 recall 追加之前的 len(messages)
    body: int  # 本轮问题之后的第一个下标 = 调研过程的起点


class ArchiveEntry(TypedDict):
    round: int  # 本轮第几圈,从 1 起
    ids: list[str]  # 该圈所有 tool_call_id,供 read_tool_result 回读
    calls: list[str]  # 如 "fetch_page(https://…)"
    gist: str  # LLM 写的要点;降级时为固定文案


class Fact(TypedDict):
    entity: str
    metric: str
    period: str
    value: str
    source_id: str


class FailureNote(TypedDict):
    tool: str
    target: str
    kind: str
    count: int
    source_id: str


class CtxState(TypedDict):
    turn: int  # 属于第几轮(= len(turns)-1);和当前轮不符就重置
    upto: int  # messages 下标:[body, upto) 已归档
    archive: list[ArchiveEntry]
    facts: list[Fact]
    failures: list[FailureNote]
    compactions: int
    last_est: int  # 上一圈的估算值,下一圈拿真实 input_tokens 对照(观测估算误差)


def empty_ctx(turn: int, body: int) -> CtxState:
    return {
        "turn": turn,
        "upto": body,
        "archive": [],
        "facts": [],
        "failures": [],
        "compactions": 0,
        "last_est": 0,
    }
