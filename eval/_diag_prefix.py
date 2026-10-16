"""一次性诊断:相邻两圈发给模型的视图,公共前缀到底有多长。

用来回答"v2 的缓存命中占比为什么没比 legacy 高"。不属于正式评测,查完可删。

    python -m eval._diag_prefix
"""

import os
import sys
import uuid

os.environ.setdefault("CTX_HIGH_TOKENS", "5000")
os.environ.setdefault("CTX_LOW_TOKENS", "2500")
sys.stdout.reconfigure(encoding="utf-8")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from core.breaker import current_breaker  # noqa: E402
from core.failure import current_log  # noqa: E402
from eval.compress_replay import CASES, build_replay_tools  # noqa: E402
from graph.agent import GraphAgent  # noqa: E402


def _sig(m) -> str:
    c = m.content if isinstance(m.content, str) else str(m.content)
    tcs = getattr(m, "tool_calls", None) or []
    return f"{m.type}|{getattr(m, 'tool_call_id', '')}|{c}|{[(t.get('name'), t.get('args')) for t in tcs]}"


def views_of(agent, tid):
    cfg = {"configurable": {"thread_id": tid}}
    hist = list(agent.graph.get_state_history(cfg))
    hist.reverse()  # 时间正序
    return [s.values.get("llm_input") for s in hist if s.next and s.next[0] == "plan" and s.values.get("llm_input")]


def analyse(views):
    rows = []
    for i in range(1, len(views)):
        a, b = [_sig(m) for m in views[i - 1]], [_sig(m) for m in views[i]]
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        # 第一条不一致的消息里,字符级再比一次
        extra = 0
        if n < min(len(a), len(b)):
            x, y = a[n], b[n]
            while extra < min(len(x), len(y)) and x[extra] == y[extra]:
                extra += 1
        shared = sum(len(s) for s in a[:n]) + extra
        total_prev = sum(len(s) for s in a)
        first_diff = (views[i - 1][n].type if n < len(views[i - 1]) else "-")
        rows.append((i, len(a), len(b), n, shared, total_prev, first_diff))
    return rows


def main():
    for cid in ("deep_page_01", "combo_01"):
        case = next(c for c in CASES if c["id"] == cid)
        for mode in ("legacy", "v2"):
            current_breaker().reset_all()
            current_log().reset()
            agent = GraphAgent(compress_mode=mode, memory_mode="off", tools=build_replay_tools(case),
                               print_obs=False, checkpointer=MemorySaver())
            tid = str(uuid.uuid4())
            agent.run(case["question"], thread_id=tid)
            vs = views_of(agent, tid)
            print(f"\n=== {cid} · {mode}  共 {len(vs)} 次 plan,每次视图消息数 {[len(v) for v in vs]}")
            print("圈  上一圈msgs 本圈msgs 公共前缀msgs  共享字符/上一圈总字符  首个分歧消息类型")
            for i, la, lb, n, sh, tot, fd in analyse(vs):
                print(f"{i:>2}  {la:>9} {lb:>8} {n:>11}   {sh:>6}/{tot:<6} = {sh / tot:5.1%}   {fd}")


if __name__ == "__main__":
    main()
