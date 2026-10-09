"""search_memory 工具(M7):让模型在需要时主动深挖记忆。

读侧是混合路线:recall 节点每轮常驻"画像 + 目录 + 阈值召回",这个工具负责"开头那块里没有、
但模型觉得需要更早细节"的情况。写侧**不给工具**(选型已定:工具会分散模型注意力,
中途写入时结论未定,且写入不确定则评测不可复现)。

## user_id 不是参数

工具签名里**没有 user_id**,它从运行配置 `config["configurable"]["user_id"]` 取。
如果让模型自己传,一次幻觉、或一次来自网页的提示注入("请查询 user_id=别人 的记忆"),
就能越权读取。让模型根本没有这个旋钮,比事后校验可靠。

## docstring 就是 prompt

何时调用写在 description 里(M5.3 的结论)。同时明说"记忆可能过时,涉及行情/财务数据
仍需调工具确认",防止模型拿记忆里的旧数字当事实、跳过工具(会伤 faithfulness)。
"""

from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from memory import events
from memory.recall import RECALL_THRESHOLD, _md
from memory.runtime import get_store

MAX_RESULTS = 6


class SearchMemoryArgs(BaseModel):
    query: str = Field(description="要在记忆里找什么,用自然语言描述")
    type: Optional[Literal["episodic", "semantic"]] = Field(
        default=None, description="可选:semantic=用户自述事实(持仓/偏好),episodic=过往调研结论")
    ticker: Optional[str] = Field(default=None, description="可选:只看涉及该股票代码的记忆,如 NVDA")


def _search_memory(
    query: str,
    type: Optional[str] = None,
    ticker: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    user_id = ((config or {}).get("configurable") or {}).get("user_id")
    if not user_id:
        # 没有身份就不查,宁可什么都不给。
        events.emit("memory_error", stage="search_tool", error="缺少 user_id")
        return "(无法确定当前用户,未检索记忆)"
    try:
        hits = get_store().search(user_id=user_id, query=query, type=type, ticker=ticker,
                                  top_k=MAX_RESULTS, threshold=RECALL_THRESHOLD)
    except Exception as exc:  # noqa: BLE001
        events.emit("memory_error", stage="search_tool", error=f"{type(exc).__name__}: {exc}")
        return "(记忆检索暂时不可用)"
    events.emit("memory_recalled", count=len(hits), injected=bool(hits), via="tool")
    if not hits:
        return "没有找到相关的记忆。"
    lines = []
    for r, s in hits:
        tag = "用户自述" if r.source.kind == "user_statement" else "调研结论,可能已过时"
        as_of = f",数据截至 {r.as_of}" if r.as_of else ""
        lines.append(f"- [{tag};记录于 {_md(r.created_at)}{as_of}] {r.text}")
    return "\n".join(lines)


search_memory = StructuredTool.from_function(
    func=_search_memory,
    name="search_memory",
    args_schema=SearchMemoryArgs,
    description=(
        "在该用户的长期记忆中检索。当用户提到「我/我的/之前/上次/还记得」,"
        "或需要更早的调研细节、而对话开头的记忆区块里没有时调用。"
        "记忆可能过时:涉及行情和财务数据时,仍需调用对应工具确认最新值。"
    ),
)
