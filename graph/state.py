"""图的状态定义 —— 对应手写版里那个手动传来传去的 messages 列表。

两个字段的分工很关键:

  messages   完整对话历史,**只增不改**(用 add_messages reducer 追加)。
             保持 append-only 是为了 checkpointer 的价值:每一步的快照都真实,
             才能做回放和时间旅行。

  llm_input  压缩后的"给模型看的视图",每次被覆盖(无 reducer = 后写覆盖前写)。
             短期记忆压缩写这里,而不是去改 messages。

为什么要分成两个?因为"完整历史"和"发给模型的内容"本来就是两回事:
历史要完整以便追溯,发给模型的要精简以省 token。手写版里两者混在一起,
这次借重构把它们分开了。
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 完整历史,追加
    llm_input: list[AnyMessage]  # 压缩后的视图,覆盖
    question: str  # 本轮用户问题(recall/remember 要用)
    # M6:预算为什么超限。由 should_continue 写入、finalize 读取。
    # 为什么要落在 state 里而不是只存在 RunBudget 对象上?因为它是一次运行的
    # **决策记录** —— checkpointer 存的快照里应该能看出"这一轮为什么被收尾了"。
    stop_reason: str
    # M8:上下文压缩 v2(memory/context/)。两者都无 reducer = 后写覆盖。
    # turns 是每轮边界 [{start, body}],recall 追加;ctx 是压缩状态(归档/事实/失败),
    # recall 重置、compress 更新。放进 state 而不是进程内缓存,是为了随 checkpointer 落库:
    # 中断恢复、时间旅行回到任何快照,压缩状态都和当时的 messages 对得上。
    turns: list[dict]
    ctx: dict
