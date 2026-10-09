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

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from graph.checkpoint import default_checkpointer
from graph.model import build_chat_model
from graph.nodes import (
    make_compress_node,
    make_finalize_node,
    make_plan_node,
    make_recall_node,
    make_remember_node,
    should_continue,
)
from graph.mcp_tools import build_mcp_tools, mcp_enabled
from graph.state import AgentState
from graph.tool_adapters import build_tools


def build_graph(
    use_short_term: bool = True,
    use_long_term: bool = True,
    checkpointer: Optional[object] = None,
    memory_mode: Optional[str] = None,
):
    """构造并编译图。

    checkpointer 决定状态存哪。不传则由 graph/checkpoint.py 按环境变量决定
    (默认 Postgres 落库,失败自动退回进程内)—— 见那个文件里关于
    "持久化执行"的说明。
    """
    # M7 记忆档位:off=不用记忆;legacy=旧版(整段存、top3 召回);v2=新版(施工中)。
    mode = memory_mode or ("legacy" if use_long_term else "off")
    if mode not in ("off", "legacy", "v2"):
        raise ValueError(f"未知 memory_mode: {mode!r}(可选 off / legacy / v2)")

    # 工具来源二选一(M5.3)。开关式:MCP_TOOLS=1 才走 MCP,默认仍是进程内工具。
    # 两者产出的都是 StructuredTool,所以下面的装配代码完全不用区分。
    tools = []
    if mcp_enabled():
        tools = build_mcp_tools()  # 失败会返回空列表,不抛异常
    if not tools:
        tools = build_tools()
    if mode == "v2":
        # search_memory 不走 _reliable_tool:它需要 RunnableConfig(拿 user_id),
        # 而可靠性包装会把调用丢进线程池,config 传不过去。本地向量检索也不需要重试/熔断。
        from memory.tools import search_memory

        tools = [*tools, search_memory]

    base_model = build_chat_model()
    # bind_tools 就是把工具 schema 告诉模型 —— 对应手写版的 get_schemas()
    model_with_tools = base_model.bind_tools(tools)

    builder = StateGraph(AgentState)

    builder.add_node("recall", make_recall_node(mode))
    builder.add_node("compress", make_compress_node(use_short_term))
    builder.add_node("plan", make_plan_node(model_with_tools))
    builder.add_node("act", ToolNode(tools))  # 预置节点,自动并行执行多个工具调用
    # M6:强制收尾节点。刻意传**不带工具**的 base_model —— 从能力上拿掉"再调工具"
    # 这个选项,而不是靠 prompt 劝说一个已经在循环里的模型。
    builder.add_node("finalize", make_finalize_node(base_model))
    builder.add_node("remember", make_remember_node(mode))

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "compress")
    builder.add_edge("compress", "plan")
    # 条件边:plan 之后有三条路 —— 调工具 / 预算超限去强制收尾 / 直接写记忆结束。
    # M6 新增 finalize 这条:它必须在 recursion_limit 之前触发,否则 LangGraph
    # 直接抛 GraphRecursionError,我们来不及产出答案(见 docs/design/M6-可靠性强化.md §3.2)。
    builder.add_conditional_edges(
        "plan",
        should_continue,
        {"act": "act", "finalize": "finalize", "remember": "remember"},
    )
    builder.add_edge("act", "compress")  # ← 这条边就是原来的 for 循环
    builder.add_edge("finalize", "remember")
    builder.add_edge("remember", END)

    return builder.compile(checkpointer=checkpointer or default_checkpointer())
