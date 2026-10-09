"""记忆的数据模型(M7)。

一个 `MemoryRecord` 覆盖情景和语义两种记忆,用 `type` 区分;程序记忆另放 Postgres(见 procedural.py)。

关于几个关键字段(设计依据见 docs/design/M7-记忆强化.md §3.4):

- `user_id`    用户隔离靠它,与"Advanced JSON"无关。
- `subject`    这条信息是关于谁的("self" 或 "父亲" 等),只为区分"我的茅台/我爸的茅台"。
- `key`        语义记忆的冲突检测键,格式 `category:对象`,对象是 ticker 时大写,
               如 `holding:NVDA`。冲突匹配用 (user_id, key, subject),精确匹配不用调 LLM。
- `as_of`      **数据属于哪个时点**(如财报季度末)。
- `created_at` **什么时候记下来的**。两者必须分开:7 月记下的可能是 Q1 数据,
               10 月记下的是 Q2 数据,只看记录时间会判断错。
- `status`     被取代/失效的旧事实标记状态而不物理删除,"毛利率从 75% 降到 73%"
               这类趋势结论才答得出来。
- `source.kind` 区分用户自述和 Agent 调研结论。后者可能来自网页(不可信),
               注入时要标"可能过时、仅供参考"。
"""

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

MemoryType = Literal["episodic", "semantic"]
Category = Literal["holding", "preference", "risk", "watchlist", "note"]
Status = Literal["active", "superseded", "invalidated"]


class Source(BaseModel):
    kind: Literal["user_statement", "agent_conclusion"] = "agent_conclusion"
    run_id: str = ""
    thread_id: str = ""


class MemoryRecord(BaseModel):
    id: str
    user_id: str
    type: MemoryType
    text: str  # 唯一被向量化的字段
    subject: str = "self"
    tickers: list[str] = Field(default_factory=list)
    importance: int = Field(5, ge=1, le=10)
    as_of: Optional[str] = None  # YYYY-MM-DD
    created_at: float = Field(default_factory=time.time)
    status: Status = "active"
    valid_to: Optional[float] = None
    superseded_by: Optional[str] = None
    source: Source = Field(default_factory=Source)

    # 仅语义记忆
    category: Optional[Category] = None
    key: Optional[str] = None
    value: Optional[dict] = None  # 结构化值,只回显不用来过滤(嵌套字段不建索引)

    # 仅情景记忆
    topic: Optional[str] = None

    @field_validator("tickers")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        # 统一大写,否则 "nvda" 和 "NVDA" 会被当成两个东西,过滤匹配不上。
        return [t.strip().upper() for t in v if t and t.strip()]

    @field_validator("user_id")
    @classmethod
    def _user_required(cls, v: str) -> str:
        # 空 user_id 会让一条记忆落入"无主"状态,过滤时可能被所有人看到或谁都看不到。
        if not v or not v.strip():
            raise ValueError("user_id 不能为空")
        return v

    def payload(self) -> dict:
        """写进 Qdrant 的 payload:除 id 外全部字段,嵌套结构原样存,不序列化、不拍平。"""
        return self.model_dump(exclude={"id"})

    @classmethod
    def from_point(cls, point: dict) -> "MemoryRecord":
        return cls(id=str(point["id"]), **point["payload"])


class ProceduralRule(BaseModel):
    """程序记忆:从过往运行里学到的、跨用户共享的经验规则(第 5 步实现存取)。"""

    tool: str
    scope: str
    failure_kind: str
    rule_text: str
    fail_count: int
    distinct_runs: int
    last_seen: float
