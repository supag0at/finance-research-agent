"""checkpointer 与时间旅行:回到某一步、改掉状态、再重跑。

运行:python -m graph.timetravel

先建立三个概念:

  thread(线程)      一场对话。由 config["configurable"]["thread_id"] 标识。
                     不同 thread 互不干扰,同一个 thread 才能续话。

  checkpoint(快照)  一个 superstep 跑完后的状态存档。每张快照带一个 checkpoint_id。
                     一个 thread 就是一串按时间排好的快照。

  checkpointer       负责存快照的东西。InMemorySaver 存进程内存,
                     PostgresSaver 存数据库(能跨进程、跨重启)。

由此得到三种用法,本脚本依次演示:

  用法1  续话      给同一个 thread_id,不指定快照 -> 从最新快照往后跑
  用法2  回放      指定某张旧快照的 checkpoint_id -> 从那一刻重跑(产生分叉)
  用法3  改写重跑  先 update_state 改掉状态,再从那里重跑 -> 人工介入的基础
"""

import sys
from pathlib import Path

# 让本脚本既能 `python -m graph.timetravel` 跑,也能 `python graph/timetravel.py` 直接跑。
# 直接跑时 sys.path[0] 是 graph/ 而不是项目根,会找不到 graph/core/tools 这些包;
# 把项目根插到最前面即可(顺带保证 `import tools` 命中根目录下的 tools/ 包)。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402  (必须在上面修好 sys.path 之后再导入)
from langchain_core.messages import HumanMessage

from graph.agent import GraphAgent


def _last_ai_text(snapshot_values: dict) -> str:
    for msg in reversed(snapshot_values.get("messages", [])):
        if type(msg).__name__ == "AIMessage" and msg.content:
            return str(msg.content).replace("\n", " ")
    return "(无)"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    load_dotenv()

    agent = GraphAgent(use_long_term=False)  # 关长期记忆,跑得快、也更好观察
    graph = agent.graph

    THREAD = "timetravel-demo"
    base_config = {
        "configurable": {"thread_id": THREAD},
        "recursion_limit": 30,
    }

    # ================================================================
    print("=" * 72)
    print("第一步:正常跑一轮,让 checkpointer 攒下一串快照")
    print("=" * 72)

    question = "什么是市盈率?一句话说清楚。"
    print(f"\n提问:{question}\n")
    state = graph.invoke({"question": question}, base_config)
    answer_a = _last_ai_text(state)
    print(f"原始答案 A:{answer_a[:150]}...\n")

    # ================================================================
    print("=" * 72)
    print("第二步:把这个 thread 的所有快照列出来")
    print("=" * 72)
    print("""
get_state_history() 返回的快照是**从新到旧**的。每张快照告诉你三件事:
  - values : 那一刻的完整状态
  - next   : 下一步准备执行哪个节点(空 = 已结束)
  - config : 含 checkpoint_id,**这就是时间旅行的坐标**
""")

    history = list(graph.get_state_history(base_config))
    for i, snap in enumerate(history):
        nxt = snap.next[0] if snap.next else "END"
        cid = snap.config["configurable"]["checkpoint_id"]
        print(
            f"  [{i}] 下一步={nxt:<10} messages={len(snap.values.get('messages', [])):<2} "
            f"checkpoint_id={cid[-12:]}"
        )

    # 找到"recall 已经跑完、下一步该 compress"那一刻。
    # 为什么挑这个点?见下面第四步的解释。
    target = next(s for s in history if s.next and s.next[0] == "compress")
    print("\n我们选中最近那张 [下一步=compress] 的快照作为回到过去的落点。")
    print("  那一刻状态里的消息是:")
    for m in target.values["messages"]:
        print(f"    - {type(m).__name__}")

    # ================================================================
    print("\n" + "=" * 72)
    print("第三步:用法2 —— 回放。从那张旧快照重跑一遍")
    print("=" * 72)
    print("""
关键写法:invoke 的第一个参数传 None(表示"没有新输入,接着已有状态跑"),
config 里带上那张旧快照的 checkpoint_id。
""")

    # ⚠️ 别手搓这个 config。快照自带的 snap.config 里除了 thread_id / checkpoint_id,
    # 还有一个 checkpoint_ns(命名空间,子图会用到)。手搓漏掉它,
    # 后面 update_state 会直接抛 KeyError: 'checkpoint_ns'。
    # 正确做法:拿快照自带的 config,只补上自己要加的运行参数。
    replay_config = {**target.config, "recursion_limit": 30}
    replayed = graph.invoke(None, replay_config)
    answer_b = _last_ai_text(replayed)
    print(f"重跑得到答案 B:{answer_b[:150]}...\n")
    print(f"  A 与 B 是否逐字相同:{answer_a == answer_b}")
    print("""  (LLM 有随机性,通常不同。重点不是答案变了,而是**我们真的回到了过去那个状态重跑**,
   而且原来的 A 那条历史线并没有被删掉——它成了一条分叉。)""")

    # ================================================================
    print("\n" + "=" * 72)
    print("第四步:用法3 —— 改写状态再重跑(人工介入的基础)")
    print("=" * 72)
    print("""
update_state() 可以在回到过去之后,先往状态里塞点东西,再往下跑。
这里我们注入一条"请全程用英文回答"的用户消息,看它会不会真的改变结果。

⚠️ 这里有个我们自己设计上的细节:plan 节点读的是 llm_input,不是 messages。
   所以落点必须选在 compress **之前**,让 compress 重新跑一次、
   把注入的消息带进 llm_input。如果落点选在 plan 之前,注入就白费了。
   —— 这就是"状态拆成两个字段"带来的代价,时间旅行时必须心里有数。
""")

    forked_config = graph.update_state(
        replay_config,
        {"messages": [HumanMessage(content="请忽略之前的语言要求,全程用英文回答。")]},
    )
    print(f"  update_state 返回了一个新的 checkpoint_id:"
          f"{forked_config['configurable']['checkpoint_id'][-12:]}")
    print("  (注意:它不是覆盖历史,而是**新建一张快照**,原来的线仍然在。)\n")

    forked = graph.invoke(None, {**forked_config, "recursion_limit": 30})
    answer_c = _last_ai_text(forked)
    print(f"改写后重跑得到答案 C:{answer_c[:200]}...\n")

    # ================================================================
    print("=" * 72)
    print("第五步:这三种用法分别解决什么现实问题")
    print("=" * 72)
    print("""
  续话      -> 多轮对话。用户说"它"、"刚才那个",Agent 能接上。
              main_graph.py 里那个追问演示就是这个。

  回放      -> 排查线上问题。日志只能告诉你"报错了",快照能让你**回到出错前
              那一刻,拿当时真实的状态反复重跑**,直到复现。
              这是 Agent 最难调试的点:它不确定,复现全靠运气,除非有快照。

  改写重跑  -> 人工介入(Human-in-the-loop)。Agent 打算调一个危险工具(下单、删库),
              先停住等人批准;或者人发现它跑偏了,手动改掉状态再让它继续。
              LangGraph 的 interrupt() 就建在这套机制上,这是 M5 之后可以做的。

手写版为什么一个都做不到?因为状态活在 Agent.run() 的函数栈里,
函数一返回就全没了。**不是写不出来,是没地方存。**
""")


if __name__ == "__main__":
    main()
