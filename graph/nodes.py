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

import time

from core.agent import SYSTEM_PROMPT  # 复用同一份系统提示,保证与手写版可比
from core.failure import target_of
from core.llm import LLM
from graph.state import AgentState
from memory import events as memory_events
from memory.context import compress_step, empty_ctx
from memory.context.tokens import usage_tokens
from memory.long_term import recall, remember
from memory.short_term import compress_messages


class _MeteredLLM:
    """给摘要器加计量,不改行为。

    摘要器走 core.LLM(裸 openai 客户端),LangChain 回调看不到它的 token 和耗时 ——
    M5.2 观测到 compress 占 29.8% 墙钟时,其实连它花了多少 token 都不知道。
    legacy 和 v2 共用这一层,A/B 时两边的压缩成本才是同一把尺子量的。
    """

    def __init__(self, llm: LLM, mode: str) -> None:
        self._llm = llm
        self.mode = mode

    @property
    def last_usage(self):
        return self._llm.last_usage

    def _metered(self, fn, messages):
        t0 = time.perf_counter()
        try:
            return fn(messages)
        finally:
            in_tok, out_tok = usage_tokens(self._llm.last_usage)
            memory_events.emit("ctx_summarizer_call", mode=self.mode,
                               ms=round((time.perf_counter() - t0) * 1000, 1),
                               in_tokens=in_tok, out_tokens=out_tok)

    def complete(self, messages: list[dict]) -> str:
        self._llm.last_usage = None
        return self._metered(self._llm.complete, messages)

    def complete_json(self, messages: list[dict]) -> dict:
        self._llm.last_usage = None
        return self._metered(self._llm.complete_json, messages)


# 摘要用的模型:直接复用手写版的 LLM 封装(它有 complete 方法)。
# 这样 memory/short_term.py 完全不用改。
_summarizer = LLM()

COMPRESS_MODES = ("off", "legacy", "v2")


def _mode_of(mode) -> str:
    """兼容旧调用:True/False → legacy/off。"""
    if isinstance(mode, bool):
        return "legacy" if mode else "off"
    return mode


def _user_id_of(config) -> str:
    uid = ((config or {}).get("configurable") or {}).get("user_id")
    if not uid:
        raise ValueError("v2 记忆需要 config.configurable.user_id(GraphAgent.run 会注入)")
    return uid


def make_recall_node(mode="legacy"):
    """开场节点:检索长期记忆,再把用户问题正式加入对话。

    注意顺序:记忆要放在用户问题**之前**,所以这两条消息在同一个节点里一起追加。

    mode:
      off     不读记忆;
      legacy  旧版(top3 向量检索,以 SystemMessage 注入。保留用于 ablation 对照);
      v2      M7 新版(画像 + 目录 + 阈值召回,以**用户级**消息注入,见 memory/recall.py)。

    ⚠️ v2 的 embedding 在本机实测约 1.3~2.6s(直接 POST SiliconFlow 约 1.74s),
    判断原因是跨境网络而非模型本身;**未在同地域环境验证**。这笔延迟每轮付一次(recall)
    加写入侧一次(remember),生产环境是否可忽略需要实测,不要拿本机数字下结论。
    """
    mode = _mode_of(mode)
    use_long_term = mode == "legacy"

    def recall_node(state: AgentState, config=None) -> dict:
        new_messages = []

        # 系统提示只在会话开头注入一次。用"历史是否为空"来判断,
        # 这样同一个 thread_id 二次提问时(状态已从 checkpointer 恢复)不会重复注入。
        if not state.get("messages"):
            prompt = SYSTEM_PROMPT
            if mode == "v2":
                # 程序记忆:代码模板生成的、可信的经验规则,所以放系统提示(与用户记忆块相反)。
                # 随系统提示只在会话开头注入一次,长会话里不会中途刷新。
                from memory.procedural import render_for_prompt

                prompt += render_for_prompt()
            new_messages.append(SystemMessage(content=prompt))

        if use_long_term:
            memories = recall(state["question"])
            if memories:
                mem_text = "\n".join(f"- {m}" for m in memories)
                new_messages.append(
                    SystemMessage(content="【关于用户/过往的记忆,供参考】\n" + mem_text)
                )

        if mode == "v2":
            from memory.recall import build_context
            from memory.runtime import get_store

            ctx = build_context(user_id=_user_id_of(config), question=state["question"],
                                store=get_store())
            if ctx.memory_block:
                # 用户级而非 system:记忆里可能含源自网页的内容,不能以 system 权限回流。
                new_messages.append(HumanMessage(content=ctx.memory_block))

        new_messages.append(HumanMessage(content=state["question"]))

        # M8:记下本轮边界,供压缩区分"受保护头部"和"可压缩的调研过程"。
        # 三种记忆档位都要写 —— 压缩与记忆档位无关。
        start = len(state.get("messages") or [])
        turns = [*(state.get("turns") or []), {"start": start, "body": start + len(new_messages)}]
        return {
            "messages": new_messages,
            "turns": turns,
            "ctx": empty_ctx(len(turns) - 1, start + len(new_messages)),
        }

    return recall_node


def make_compress_node(mode="v2"):
    """短期记忆节点:把完整历史压成"给模型看的视图",写入 llm_input。

    关键:**不修改 messages**。压缩只影响发给模型的内容,不影响记录在案的历史。

    mode(兼容旧调用:True → legacy,False → off):
      off     不压缩;
      legacy  M5 以来的 memory/short_term.py(按条数触发、每圈重算摘要)。保留用于消融对照;
      v2      M8 的 memory/context/(增量归档 + 钉住区 + 可回读)。
    """
    if isinstance(mode, bool):
        mode = "legacy" if mode else "off"
    if mode not in COMPRESS_MODES:
        raise ValueError(f"未知 compress_mode: {mode!r}(可选 {' / '.join(COMPRESS_MODES)})")
    summarizer = _MeteredLLM(_summarizer, mode)

    def compress_node(state: AgentState) -> dict:
        messages = state["messages"]
        if mode == "off":
            return {"llm_input": messages}
        if mode == "v2":
            return compress_step(state, summarizer)

        # memory/short_term.py 处理的是 OpenAI 字典格式,这里做一次来回转换,
        # 从而完全复用已有逻辑(包括那个"不能让窗口以孤立 tool 消息开头"的细节)。
        as_dicts = convert_to_openai_messages(messages)
        compressed = compress_messages(as_dicts, summarizer)
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


def make_remember_node(mode="legacy"):
    """收尾节点:把本轮问答写入长期记忆。

    v2 是**同步**写入:抽取(1 次 LLM)+ 可能的冲突判定 + embedding 都在用户等待的路径上。
    选同步的理由:评测可复现(异步写会让"刚写完马上读"的题随机失败),实现简单。
    代价:每轮多出若干秒延迟(⚠️ 未实测,见 RunRecord.nodes["remember"])。
    生产环境应改成后台异步写入(写完才对下一轮可见,需处理"紧接着的下一轮读不到"的窗口)。
    """
    mode = _mode_of(mode)
    use_long_term = mode == "legacy"

    def remember_node(state: AgentState, config=None) -> dict:
        if mode == "v2":
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.content:
                from memory.runtime import get_llm, get_store
                from memory.write import write_memories

                cfg = (config or {}).get("configurable") or {}
                budget = cfg.get("budget")
                # write_memories 自己吞异常并发 memory_error 事件,不会拖垮主流程。
                write_memories(
                    user_id=_user_id_of(config),
                    question=state["question"],
                    answer=str(last.content),
                    llm=get_llm(),
                    store=get_store(),
                    run_id=str(cfg.get("run_id", "")),
                    thread_id=str(cfg.get("thread_id", "")),
                    stop_reason=(budget.tripped_reason() if budget is not None else "")
                    or state.get("stop_reason", ""),
                )
            return {}
        if use_long_term:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.content:
                remember(f"用户问题:{state['question']}\n调研结论:{last.content}")
        return {}

    return remember_node


FINALIZE_PROMPT = """你的调研被提前终止了,原因:{reason}

请**立即**基于你目前已经获得的信息给出最终结论,不要再请求调用任何工具。
要求:
1. 把已经确认的数据和结论写清楚,并说明出处;
2. **明确列出哪些信息没能获取到**,以及这对结论的影响;
3. 不要编造任何未经工具确认的数字。

如果已获取的信息确实不足以回答,就直接说明不足在哪,并给出你能确定的部分。
"""


def make_finalize_node(model):
    """强制收尾节点(M6):预算超限 / 检测到循环时,逼模型基于现有信息产出答案。

    为什么需要它?改造前触达上限直接返回一句
    `"(已达到最大步数,未能得出最终结论)"`。实测 `nvda_perf_01` 明明已经调了
    29 次工具、拿到了部分数据,却交白卷,裁判三项全给 1 分。
    **我们在工具层精心做了降级,在 Agent 层却直接放弃** —— 降级链的最后一环是断的。

    关键细节:这里用的是**不带工具**的模型(`model` 而非 `model_with_tools`)。
    不是靠 prompt 说服它别调工具,而是从能力上拿掉这个选项 ——
    否则一个已经在循环里的模型大概率会继续要求调工具。
    这也是"bounds, not retries"那条原则在节点层面的体现:约束要是结构性的。
    """

    def finalize_node(state: AgentState, config=None) -> dict:
        # 原因从 budget 取(条件边没法写 state,所以在这里取并回写,
        # 让 checkpointer 的快照里能看出"这一轮为什么被收尾了")。
        budget = ((config or {}).get("configurable") or {}).get("budget")
        reason = (
            (budget.tripped_reason() if budget is not None else "")
            or state.get("stop_reason")
            or "已达到本轮调研的资源上限"
        )
        messages = list(state.get("llm_input") or state["messages"])
        messages.append(HumanMessage(content=FINALIZE_PROMPT.format(reason=reason)))
        try:
            response = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            # 连收尾都失败,才退回原来那句兜底。至少把原因带上。
            response = AIMessage(
                content=f"(调研提前终止:{reason};生成结论时也失败了:{type(exc).__name__})"
            )
        return {"messages": [response], "stop_reason": reason}

    return finalize_node


def should_continue(state: AgentState, config=None) -> str:
    """条件边 —— 对应手写版的 `if not message.tool_calls: return answer`。

    这就是"控制流从代码里浮现成图上一条边"的那一刻。

    M6 在这里多了一个分支:**先问预算,再看模型想干什么**。
    顺序是刻意的 —— 预算超了就不该再执行工具,哪怕模型要求了。
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    budget = ((config or {}).get("configurable") or {}).get("budget")
    if budget is not None:
        # 把模型这一轮的调用意图登记进去,用于循环检测。
        # 注意登记的是**意图**而非结果:工具每次都"成功"但返回无用内容时,
        # 反复调同一目标本身就是不收敛的信号。
        if tool_calls:
            budget.note_tool_calls(
                [(tc.get("name", "?"), target_of(tc.get("args") or {})) for tc in tool_calls]
            )
        reason = budget.exceeded()
        if reason:
            # 已经给出最终答案的话没必要再收尾一遍,直接去写记忆。
            return "remember" if not tool_calls else "finalize"

    if tool_calls:
        return "act"
    return "remember"
