"""图的四个节点:recall / compress / plan / remember。

对照手写版 core/agent.py:
  - recall   ← 主循环开头那段 "recall(question) 并注入 system 消息"
  - compress ← 每次调 LLM 前的 compress_messages()
  - plan     ← self.llm.chat(messages, schemas) 那一行
  - remember ← 给出最终答案时的 remember(...)
  - act      ← 用 LangGraph 预置的 ToolNode,不用自己写(见 build.py)

**记忆和压缩的业务逻辑一行没改**,仍然调用 memory/ 下的原函数。
这里只是把"函数调用"变成"图中的节点",好处是它们变得可见、可单独测试、
在 trace 里能看到各自耗时。
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import convert_to_messages, convert_to_openai_messages

from core.agent import SYSTEM_PROMPT  # 复用同一份系统提示,保证与手写版可比
from core.llm import LLM
from graph.state import AgentState
from memory.long_term import recall, remember
from memory.short_term import compress_messages

# 摘要用的模型:直接复用手写版的 LLM 封装(它有 complete 方法)。
# 这样 memory/short_term.py 完全不用改。
_summarizer = LLM()


def make_recall_node(use_long_term: bool = True):
    """开场节点:检索长期记忆,再把用户问题正式加入对话。

    注意顺序:记忆要放在用户问题**之前**,所以这两条消息在同一个节点里一起追加。
    """

    def recall_node(state: AgentState) -> dict:
        new_messages = []

        # 系统提示只在会话开头注入一次。用"历史是否为空"来判断,
        # 这样同一个 thread_id 二次提问时(状态已从 checkpointer 恢复)不会重复注入。
        if not state.get("messages"):
            new_messages.append(SystemMessage(content=SYSTEM_PROMPT))

        if use_long_term:
            memories = recall(state["question"])
            if memories:
                mem_text = "\n".join(f"- {m}" for m in memories)
                new_messages.append(
                    SystemMessage(content="【关于用户/过往的记忆,供参考】\n" + mem_text)
                )

        new_messages.append(HumanMessage(content=state["question"]))
        return {"messages": new_messages}

    return recall_node


def make_compress_node(use_short_term: bool = True):
    """短期记忆节点:把完整历史压成"给模型看的视图",写入 llm_input。

    关键:**不修改 messages**。压缩只影响发给模型的内容,不影响记录在案的历史。
    """

    def compress_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not use_short_term:
            return {"llm_input": messages}

        # memory/short_term.py 处理的是 OpenAI 字典格式,这里做一次来回转换,
        # 从而完全复用已有逻辑(包括那个"不能让窗口以孤立 tool 消息开头"的细节)。
        as_dicts = convert_to_openai_messages(messages)
        compressed = compress_messages(as_dicts, _summarizer)
        return {"llm_input": convert_to_messages(compressed)}

    return compress_node


def make_plan_node(model_with_tools):
    """决策节点:调 LLM,决定"再调工具"还是"给最终答案"。

    对应手写版的 self.llm.chat(messages, self.tools.get_schemas())。
    工具 schema 已经通过 bind_tools 绑好了,这里不用再传。
    """

    def plan_node(state: AgentState) -> dict:
        # 有压缩视图就用压缩视图,否则退回完整历史(第一次进来时)。
        messages = state.get("llm_input") or state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return plan_node


def make_remember_node(use_long_term: bool = True):
    """收尾节点:把本轮问答写入长期记忆。"""

    def remember_node(state: AgentState) -> dict:
        if use_long_term:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.content:
                remember(f"用户问题:{state['question']}\n调研结论:{last.content}")
        return {}

    return remember_node


def should_continue(state: AgentState) -> str:
    """条件边 —— 对应手写版的 `if not message.tool_calls: return answer`。

    这就是"控制流从代码里浮现成图上一条边"的那一刻。
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "act"
    return "remember"
