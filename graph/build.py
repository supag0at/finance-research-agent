"""组图 —— 把节点和边装配成可执行的 Agent。

这个文件就是手写版 core/agent.py 那个 for 循环的"图版本"。
对照着看会很清楚:

    手写版                          图版本
    ------------------------------  ----------------------------------
    for step in range(max_steps)    act -> compress 这条**返回边**
    if not message.tool_calls       add_conditional_edges(条件边)
    for tc in message.tool_calls    ToolNode(还免费获得并行执行)
    max_steps=8                     recursion_limit
"""

from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graph.model import build_chat_model
from graph.nodes import (
    make_compress_node,
    make_plan_node,
    make_recall_node,
    make_remember_node,
    should_continue,
)
from graph.state import AgentState
from graph.tool_adapters import build_tools


def build_graph(
    use_short_term: bool = True,
    use_long_term: bool = True,
    checkpointer: Optional[object] = None,
):
    """构造并编译图。

    checkpointer 决定状态存哪:
      - InMemorySaver:进程内,重启即失(够用于演示与评测)
      - PostgresSaver:落库,可跨进程续跑(我们已有现成的 PG,后续可切)
    """
    tools = build_tools()
    # bind_tools 就是把工具 schema 告诉模型 —— 对应手写版的 get_schemas()
    model_with_tools = build_chat_model().bind_tools(tools)

    builder = StateGraph(AgentState)

    builder.add_node("recall", make_recall_node(use_long_term))
    builder.add_node("compress", make_compress_node(use_short_term))
    builder.add_node("plan", make_plan_node(model_with_tools))
    builder.add_node("act", ToolNode(tools))  # 预置节点,自动并行执行多个工具调用
    builder.add_node("remember", make_remember_node(use_long_term))

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "compress")
    builder.add_edge("compress", "plan")
    # 条件边:plan 之后是去调工具,还是去写记忆然后结束
    builder.add_conditional_edges(
        "plan",
        should_continue,
        {"act": "act", "remember": "remember"},
    )
    builder.add_edge("act", "compress")  # ← 这条边就是原来的 for 循环
    builder.add_edge("remember", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
