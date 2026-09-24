"""图版入口 —— 对照 main.py(手写版)。

两者做的是同一件事,可以直接对比运行结果:
    python main.py         手写循环版
    python main_graph.py   LangGraph 版

本文件额外演示两个手写版没有的能力:
    1. 打印图结构(控制流可视化)
    2. 同一个 thread_id 续接对话(靠 checkpointer 持久化状态)
"""

import sys

from dotenv import load_dotenv

from graph.agent import GraphAgent
from main import print_trajectory
from obs.tracing import setup_tracing, shutdown_tracing


def draw_graph(graph) -> str:
    """画出控制流。

    draw_ascii() 需要额外装 grandalf;缺了就退回 Mermaid 文本(langchain 自带,无依赖)。
    画图只是演示功能,不该因为少一个可选依赖就让整个入口挂掉 —— 这属于"非核心功能
    必须优雅降级",和我们对记忆、工具的处理是同一个原则。
    """
    try:
        return graph.get_graph().draw_ascii()
    except ImportError:
        return (
            graph.get_graph().draw_mermaid()
            + "\n(提示:装 grandalf 可直接在终端画出 ASCII 图:pip install grandalf)"
        )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv()

    # 可观测第二层:设 OBS_TRACING=1 就把链路发给本地 Phoenix 看瀑布图。
    # 没开或没装依赖时这行什么都不做,所以放在这里是安全的。
    setup_tracing()

    agent = GraphAgent()

    # ---- 能力 1:控制流可视化。手写版的循环藏在代码里,图版可以直接画出来 ----
    print("=" * 60)
    print("图结构(控制流一目了然)")
    print("=" * 60)
    print(draw_graph(agent.graph))

    # ---- 正常跑一轮 ----
    question = "帮我分析英伟达(NVDA)最近的股价表现,以及可能的原因。"
    print(f"\n用户问题: {question}")

    thread = "demo-session"
    answer, trajectory = agent.run(question, thread_id=thread)

    print_trajectory(trajectory)
    print("\n" + "=" * 60)
    print("最终答案")
    print("=" * 60)
    print(answer)

    # ---- 能力 2:续接对话。同一个 thread_id,状态从 checkpointer 恢复 ----
    # 注意这里问的是"它"——只有记得上一轮聊的是英伟达才能答对。
    print("\n" + "=" * 60)
    print("续接对话(同一 thread_id,验证状态被持久化)")
    print("=" * 60)
    follow_up = "用一句话总结你刚才的结论。"
    print(f"追问: {follow_up}")

    answer2, _ = agent.run(follow_up, thread_id=thread)
    print(f"\n回答: {answer2}")

    # 把攒着的 span 推出去。BatchSpanProcessor 是异步攒批的,
    # 进程退出太快会丢掉最后几个 span(atexit 里也注册了,这里显式再调一次更稳)。
    shutdown_tracing()


if __name__ == "__main__":
    main()
