"""数据结构定义:消息、工具调用、轨迹(Trajectory)。

这些是整个 Agent 里"流动的数据"。把它们统一定义在一处,其他模块都引用
这里,保证格式一致。你之前问"Trajectory 到底体现在哪"——答案就是下面的
TrajectoryStep:每转一圈循环,就往轨迹里追加一条。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolCall:
    """LLM 决定"要调用某个工具"时产生的结构化数据。

    这就是 Function Calling 的产物:LLM 不直接回答,而是吐出"我要调哪个
    工具、传什么参数"。
    """

    id: str  # 本次调用的唯一 id(模型生成,用于把工具结果对应回去)
    name: str  # 工具名,例如 "get_stock_price"
    arguments: dict  # 工具参数,例如 {"symbol": "NVDA", "days": 5}


@dataclass
class TrajectoryStep:
    """轨迹中的一步:记录这一圈里 LLM 想了什么、调了什么工具、拿到什么结果。

    把每一步都存下来,跑完整个任务后就得到完整的 Trajectory(轨迹)。
    后面做 eval(评测)和可观测/回放,用的都是这份数据。
    """

    step: int  # 第几圈
    reasoning: str = ""  # 推理模型的"思考过程"(reasoning_content);普通模型为空
    thought: str = ""  # LLM 这一步在 content 里顺带说的话(不是严格的推理链)
    tool_call: Optional[ToolCall] = None  # 这一步调用的工具(纯思考/最终答案时为 None)
    observation: Optional[str] = None  # 工具返回的结果(观察)
    final_answer: Optional[str] = None  # 若这一步给出了最终答案,放这里
