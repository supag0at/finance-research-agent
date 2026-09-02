"""程序入口:把各模块拼装起来,跑一个例子,并把整条轨迹打印出来。

运行前:
1. 复制 .env.example 为 .env,填入你的 API key(DeepSeek / Kimi / OpenAI 三选一)。
2. pip install -r requirements.txt
3. python main.py
"""

import sys

from dotenv import load_dotenv

from core.agent import Agent
from core.llm import LLM
from core.schema import TrajectoryStep
from tools import finance
from tools.registry import ToolRegistry


def build_agent() -> Agent:
    """组装 Agent:创建工具注册表、登记工具、注入 LLM。"""
    tools = ToolRegistry()

    # 登记"查行情"工具。注意 parameters 就是给 LLM 看的 Tool Schema,
    # 描述写得越清楚,LLM 越知道该在什么时候、传什么参数来调它。
    tools.register(
        name="get_stock_price",
        description="查询某只股票最近若干天的涨跌幅、最新价和趋势。需要行情数据时调用。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码,例如 NVDA、AAPL"},
                "days": {"type": "integer", "description": "最近多少天,默认 5"},
            },
            "required": ["symbol"],
        },
        func=finance.get_stock_price,
    )

    # 登记"搜新闻"工具。
    tools.register(
        name="search_news",
        description="按关键词搜索与某公司或主题相关的新闻。需要了解事件/原因时调用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词,例如 '英伟达 上涨 原因'"},
            },
            "required": ["query"],
        },
        func=finance.search_news,
    )

    return Agent(llm=LLM(), tools=tools)


def print_trajectory(trajectory: list[TrajectoryStep]) -> None:
    """把轨迹打印成人类可读的形式——这就是最朴素的 trace replay(回放)。"""
    print("\n" + "=" * 60)
    print("轨迹 Trajectory(Agent 这一趟到底做了什么)")
    print("=" * 60)
    for s in trajectory:
        print(f"\n[第 {s.step} 圈]")
        if s.reasoning:
            print(f"  推理过程(reasoning): {s.reasoning}")
        if s.thought:
            print(f"  说明(content): {s.thought}")
        if s.tool_call:
            print(f"  调用工具: {s.tool_call.name}  参数: {s.tool_call.arguments}")
            print(f"  工具返回: {s.observation}")
        if s.final_answer:
            print("  -> 给出最终答案")


def main() -> None:
    # Windows 控制台默认可能是 GBK,强制 UTF-8 避免中文输出乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv()  # 从 .env 读取 API key 等配置
    agent = build_agent()

    question = "帮我分析英伟达(NVDA)最近的股价表现,以及可能的原因。"
    print(f"用户问题: {question}")

    answer, trajectory = agent.run(question)

    print_trajectory(trajectory)
    print("\n" + "=" * 60)
    print("最终答案")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
