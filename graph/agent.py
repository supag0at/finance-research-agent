"""GraphAgent —— 图版 Agent 的对外门面。

它刻意做成和手写版 core.agent.Agent **完全一样的接口**:

    answer, trajectory = agent.run(question)

好处是 M4 那套评测(eval/)一行都不用改就能跑图版,从而**直接验证"功能等价"**。
这正是设计文档里定的验收方式:重构前后跑同一套评测,分数不能掉。

额外提供了手写版没有的能力:传入 thread_id 即可续接上一轮对话(靠 checkpointer)。
"""

import asyncio
import uuid
from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from core.schema import ToolCall, TrajectoryStep
from graph.mcp_tools import mcp_enabled
from obs.callbacks import ObsHandler
from obs.metrics import RunRecord
from obs.tracing import flush_tracing, setup_tracing

# 与手写版 max_steps=8 大致等价:一圈是 compress+plan(+act),
# 加上首尾的 recall/remember,8 圈约需 25 个 superstep,这里留些余量。
DEFAULT_RECURSION_LIMIT = 30

MAX_STEPS_MESSAGE = "(已达到最大步数,未能得出最终结论)"


class GraphAgent:
    def __init__(
        self,
        use_short_term: bool = True,
        use_long_term: bool = True,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        checkpointer: Optional[object] = None,
        print_obs: bool = True,
    ) -> None:
        self.use_short_term = use_short_term
        self.use_long_term = use_long_term
        self.recursion_limit = recursion_limit
        self.print_obs = print_obs  # 跑完是否打印一行观测摘要
        self._checkpointer = checkpointer
        self._graph = None  # 延迟构建
        self.last_run: Optional[RunRecord] = None  # 最近一次运行的观测汇总

    @property
    def graph(self):
        """延迟构建图。

        为什么延迟?因为 eval/runner.py 是先 build 再改 `agent.use_long_term = False`。
        延迟到第一次 run() 才建图,这些开关才能生效。
        """
        if self._graph is None:
            from graph.build import build_graph

            self._graph = build_graph(
                use_short_term=self.use_short_term,
                use_long_term=self.use_long_term,
                checkpointer=self._checkpointer,
            )
        return self._graph

    def run(
        self, question: str, thread_id: Optional[str] = None
    ) -> tuple[str, list[TrajectoryStep]]:
        """跑一轮调研,返回(最终答案, 轨迹)。

        thread_id:
            不传则每次用一个新的随机 id(相互隔离,适合评测);
            传同一个 id 则会**接上该会话之前的状态**(checkpointer 的能力)。

        副产品:本次运行的观测汇总会挂在 self.last_run 上(供评测收集成本/耗时)。
        """
        # 开关式接线:没开 OBS_TRACING 或没装依赖时这行什么都不做。
        # 幂等,main_graph.py / eval 里重复调也安全。
        setup_tracing()

        tid = thread_id or str(uuid.uuid4())
        handler = ObsHandler(question=question, impl="langgraph", thread_id=tid)
        config = {
            "configurable": {"thread_id": tid},
            "recursion_limit": self.recursion_limit,
            # 观测就挂在这里。业务代码(nodes.py)完全不知道它的存在。
            "callbacks": [handler],
        }

        handler.start()
        hit_limit = False
        try:
            final_state = self._invoke({"question": question}, config)
            messages = final_state["messages"]
        except GraphRecursionError:
            # 撞到步数上限。取出已有状态,行为与手写版保持一致:给一句兜底说明。
            hit_limit = True
            snapshot = self.graph.get_state(config)
            messages = snapshot.values.get("messages", []) if snapshot else []
            answer = MAX_STEPS_MESSAGE
        else:
            answer = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break
        finally:
            self.last_run = handler.finish(hit_limit=hit_limit)
            if self.print_obs:
                print(self.last_run.summary_line(), flush=True)
            # BatchSpanProcessor 是异步攒批的,不 flush 的话刚跑完去 Phoenix 会看不到。
            flush_tracing()

        return answer, _to_trajectory(messages)

    def _invoke(self, inputs: dict, config: dict) -> dict:
        """执行图。同步还是异步,取决于工具来自哪里(M5.3)。

        MCP 的工具是**只支持异步调用**的(adapter 只实现了 _arun),所以开了
        MCP_TOOLS 之后整条图必须走 ainvoke —— 跨进程通信的异步性质会沿着调用链
        一路传染到这里。详见 graph/mcp_tools.py 的文件头。

        为什么不干脆统一走 ainvoke:默认路径(进程内工具)保持 M5.2 的同步执行,
        才能保证 M5.2.3 那份代价基线是可比的。否则一旦数字有变化,就分不清是
        MCP 的影响还是执行模型变了的影响。
        """
        if mcp_enabled():
            return asyncio.run(self.graph.ainvoke(inputs, config))
        return self.graph.invoke(inputs, config)


def _to_trajectory(messages: list) -> list[TrajectoryStep]:
    """把 LangChain 消息序列还原成我们自己的 Trajectory 结构。

    为什么还要还原?因为 eval/checks.py 是按 TrajectoryStep 写的。
    保持这层转换,评测代码就完全不用动 —— 这也是"接口不变、实现可换"的价值。
    """
    steps: list[TrajectoryStep] = []
    step_no = 0

    for msg in messages:
        if isinstance(msg, AIMessage):
            step_no += 1
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            reasoning = msg.additional_kwargs.get("reasoning_content", "") or ""

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append(
                        TrajectoryStep(
                            step=step_no,
                            reasoning=reasoning,
                            thought=content,
                            tool_call=ToolCall(
                                id=tc.get("id", ""),
                                name=tc.get("name", ""),
                                arguments=tc.get("args", {}) or {},
                            ),
                        )
                    )
            else:
                steps.append(
                    TrajectoryStep(
                        step=step_no,
                        reasoning=reasoning,
                        thought=content,
                        final_answer=content,
                    )
                )

        elif isinstance(msg, ToolMessage):
            # 把工具返回回填到对应的那一步(按 tool_call_id 配对)
            for s in steps:
                if s.tool_call and s.tool_call.id == msg.tool_call_id and s.observation is None:
                    s.observation = (
                        msg.content if isinstance(msg.content, str) else str(msg.content)
                    )
                    break

    return steps
