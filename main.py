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
from rag.retrieve import retrieve
from tools import finance
from tools.registry import ToolRegistry


def _stock_fallback(symbol: str = "", days: int = 30) -> dict:
    """行情工具的降级兜底:真实接口失败时,至少告诉模型"数据暂不可用",
    而不是让流程崩掉。模型看到这个,会知道不要编造数字。"""
    return {"symbol": symbol, "error": "行情数据暂时不可用,请勿编造数字。"}


def build_agent() -> Agent:
    """组装 Agent:创建工具注册表、登记工具、注入 LLM。"""
    tools = ToolRegistry()

    # 登记"查行情"工具。parameters 就是给 LLM 看的 Tool Schema,描述写得越清楚,
    # LLM 越知道该在什么时候、传什么参数来调它。
    # 这里还给它配了可靠性策略:超时 20s、重试 2 次、失败降级。
    tools.register(
        name="get_stock_price",
        description="查询某只股票最近若干个交易日的涨跌幅、最新价、最高/最低价。需要行情数据时调用。",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码,例如 NVDA、AAPL"},
                "days": {"type": "integer", "description": "最近多少个交易日,默认 30"},
            },
            "required": ["symbol"],
        },
        func=finance.get_stock_price,
        timeout=20.0,
        retries=2,
        fallback=_stock_fallback,
    )

    # 登记"搜新闻"工具(真实网页搜索)。
    tools.register(
        name="search_news",
        description="按关键词做网页搜索,返回相关新闻的标题、链接和摘要。需要了解事件/原因时调用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词,例如 'NVDA stock surge reason'"},
            },
            "required": ["query"],
        },
        func=finance.search_news,
        timeout=20.0,
        retries=2,
    )

    # 登记"抓网页"工具:根据搜索到的链接,读取正文以获取细节。
    tools.register(
        name="fetch_page",
        description="抓取一个网页的正文文本。当搜索结果的摘要不够、需要看细节时调用。",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的网页地址(http/https)"},
            },
            "required": ["url"],
        },
        func=finance.fetch_page,
        timeout=15.0,
        retries=1,
    )

    # 登记"知识库检索"工具(RAG)。当需要背景知识、公司资料、术语解释时调用。
    # 对 Agent 来说,RAG 就是又一个平平无奇的工具——这正是解耦的好处。
    tools.register(
        name="retrieve",
        description="从本地知识库按语义检索背景资料(公司背景、行业知识、术语解释等)。需要背景知识时调用。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题,例如 '英伟达的主要业务板块'"},
                "top_k": {"type": "integer", "description": "返回几段,默认 4"},
            },
            "required": ["query"],
        },
        func=retrieve,
        timeout=20.0,
        retries=1,
    )

    # 工具变多了,把最大步数放宽一点,给它更充分的调研空间。
    return Agent(llm=LLM(), tools=tools, max_steps=8)


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
