"""Agent 主循环 —— 整个系统的"大脑骨架"。

这就是我们反复讲的那个"带工具的循环":
  1. 把对话历史 + 工具清单发给 LLM。
  2. LLM 要么"要调工具"(Function Calling),要么"给最终答案"。
  3. 若要调工具:执行它 -> 拿到结果(observation)-> 塞回对话 -> 继续循环。
  4. 若给答案:结束,返回答案 + 整条轨迹(Trajectory)。

M1 不用任何 Agent 框架,全部手写。等你把这个循环看懂了,M5 再用 LangGraph
重做一遍,你就能亲身体会"框架到底帮我们省了什么、又多了什么复杂度"。
"""

import json

from core.llm import LLM
from core.schema import ToolCall, TrajectoryStep
from memory.long_term import recall, remember
from memory.short_term import compress_messages
from tools.registry import ToolRegistry

SYSTEM_PROMPT = """你是一个严谨的财经调研助手。
你可以调用工具来获取股票行情和相关新闻。
要求:
1. 只依据工具返回的真实数据作答,绝不编造数字。
2. 需要数据时,主动调用合适的工具,而不是凭空猜测。
3. 给出结论时,说明你的依据来自哪些数据/新闻。
"""


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        max_steps: int = 6,
        use_short_term: bool = True,
        use_long_term: bool = True,
    ) -> None:
        self.llm = llm
        self.tools = tools
        # max_steps:最多转多少圈。这是一个重要的可靠性细节——防止 LLM 陷入
        # "反复调工具却永远不给答案"的死循环。
        self.max_steps = max_steps
        self.use_short_term = use_short_term  # 短期记忆:上下文压缩
        self.use_long_term = use_long_term  # 长期记忆:跨会话语义记忆

    def run(self, question: str) -> tuple[str, list[TrajectoryStep]]:
        # messages 是发给模型的对话历史,采用 OpenAI 的消息格式。
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 【长期记忆·读】开工前先回忆:把相关的历史记忆注入上下文。
        if self.use_long_term:
            memories = recall(question)
            if memories:
                mem_text = "\n".join(f"- {m}" for m in memories)
                messages.append(
                    {"role": "system", "content": "【关于用户/过往的记忆,供参考】\n" + mem_text}
                )

        messages.append({"role": "user", "content": question})

        trajectory: list[TrajectoryStep] = []

        for step in range(1, self.max_steps + 1):
            # 【短期记忆】发给 LLM 前先压缩上下文(超过阈值才会真正压缩)。
            if self.use_short_term:
                messages = compress_messages(messages, self.llm)

            # (1) 规划 Planning:把历史 + 工具说明书发给 LLM,让它决定下一步。
            message = self.llm.chat(messages, self.tools.get_schemas())

            # 推理模型(如 deepseek-reasoner)会把"思考过程"放在 reasoning_content
            # 这个额外字段里;普通模型(deepseek-chat)没有该字段。用 getattr 安全读取:
            # 有就拿到,没有就是空字符串,不会报错。
            reasoning = getattr(message, "reasoning_content", "") or ""

            # 情况 A:LLM 没有要求调工具 -> 说明它已经给出了最终答案。
            if not message.tool_calls:
                answer = message.content or ""
                trajectory.append(
                    TrajectoryStep(
                        step=step,
                        reasoning=reasoning,
                        thought=answer,
                        final_answer=answer,
                    )
                )
                # 【长期记忆·写】把本次问答存进记忆,供以后会话回忆。
                # TODO(记忆增强,见 memory/MEMORY.md):当前是"整段都记";更好的做法是
                #   用 LLM 抽取关键要点、判断值不值得记(写入筛选),再存。
                if self.use_long_term:
                    remember(f"用户问题:{question}\n调研结论:{answer}")
                return answer, trajectory

            # 情况 B:LLM 要求调用工具(一圈里可能一次要调多个)。
            # 先把 LLM 的这条"要调工具"的消息也记进对话历史,后面工具结果要挂在它下面。
            assistant_msg = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            # 关键坑:DeepSeek 思考模式下,只要请求带了 tools,就必须把这一轮的
            # reasoning_content 原样回传,否则下一次请求会报 HTTP 400。所以这里
            # 有思考内容时一定要带上。
            if reasoning:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)

            for tc in message.tool_calls:
                # 模型给的参数是 JSON 字符串,解析成字典。
                args = json.loads(tc.function.arguments or "{}")

                # (2) 执行工具,得到观察结果 observation。
                observation = self.tools.call(tc.function.name, args)

                # (3) 记入轨迹:这一圈想了什么、调了什么、结果如何。
                trajectory.append(
                    TrajectoryStep(
                        step=step,
                        reasoning=reasoning,
                        thought=message.content or "",
                        tool_call=ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=args,
                        ),
                        observation=observation,
                    )
                )

                # (4) 把工具结果塞回对话,供下一圈使用。
                #     role="tool" + tool_call_id 是在告诉模型:这是你刚才那次调用的结果。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    }
                )

        # 转满 max_steps 还没给出答案 -> 兜底返回,避免死循环。
        return "(已达到最大步数,未能得出最终结论)", trajectory
