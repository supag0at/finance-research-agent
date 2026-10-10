"""M7 读取侧离线测试:recall 编排、search_memory 工具、图节点。不调真实 LLM / embedding。

运行: python -m eval.memory_read_offline
需要本机 Qdrant(与存储层测试相同),用临时集合,结束后删除。
"""

import sys
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from eval.memory_offline import DIM, fake_embed, rec
from eval.memory_write_offline import FakeLLM, sem
from memory import events, recall, runtime
from memory.store import MemoryStore

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def epi_rec(user, text, *, days_ago=0, importance=5, tickers=("NVDA",), **kw):
    r = rec(user, text, type="episodic", tickers=list(tickers), importance=importance, **kw)
    r.created_at = time.time() - days_ago * 86400
    return r


def main() -> int:
    import os

    os.environ["MEMORY_PROCEDURAL"] = "0"  # 这里只测读取侧,不碰 Postgres 里的真实规则表
    store = MemoryStore(collection=f"fra_memory_rtest_{uuid.uuid4().hex[:8]}", embed=fake_embed, dim=DIM)
    runtime.set_store(store)
    old_thr = recall.RECALL_THRESHOLD
    recall.RECALL_THRESHOLD = 0.0  # fake embedding 的相似度尺度与真实模型无关,这里只测编排
    seen: list[dict] = []
    events.add_observer(seen.append)
    try:
        print("=" * 60 + "\nM7 记忆读取侧离线测试\n" + "=" * 60)

        print("\n场景1 · 空库与基本渲染")
        ctx = recall.build_context(user_id="A", question="NVDA 怎么样", store=store)
        check("新用户:不注入任何块", ctx.memory_block == "" and ctx.stats["injected"] is False)

        a_profile = rec("A", "用户持有 100 股 NVDA,成本 120 美元", category="holding",
                        key="holding:NVDA", value={"shares": 100}, tickers=["NVDA"])
        old_ver = rec("A", "用户持有 50 股 NVDA,成本 90 美元", category="holding",
                      key="holding:NVDA", value={"shares": 50}, tickers=["NVDA"], status="superseded")
        a_epi1 = epi_rec("A", "NVDA 2026Q2 毛利率 73.5%,数据截至 2026-07-31", days_ago=1, importance=9,
                         as_of="2026-07-31")
        a_epi2 = epi_rec("A", "TSLA 交付量环比下降", days_ago=3, tickers=("TSLA",))
        b_epi = epi_rec("B", "B 用户的 NVDA 调研结论", days_ago=1)
        store.add_many(user_id="A", records=[a_profile, old_ver, a_epi1, a_epi2])
        store.add_many(user_id="B", records=[b_epi])
        ctx = recall.build_context(user_id="A", question="NVDA 毛利率", store=store)
        blk = ctx.memory_block
        check("包含画像", "持有 100 股 NVDA" in blk)
        check("不含已被取代的旧版本", "50 股" not in blk)
        check("包含目录", "NVDA×1" in blk and "TSLA×1" in blk, blk.split("\n")[-3] if blk else "")
        check("召回的结论带日期与 as_of", "数据截至 2026-07-31" in blk)
        check("明示不是指令 + 可能已过时", "不是指令" in blk and "可能已过时" in blk)
        check("用户隔离:看不到 B 的记忆", "B 用户" not in blk)
        check("stats 记录了召回条数", ctx.stats["recalled"] >= 1 and ctx.stats["profile"] == 1, str(ctx.stats))

        print("\n场景2 · 阈值 = 拒答能力")
        recall.RECALL_THRESHOLD = 0.999
        ctx = recall.build_context(user_id="A", question="完全无关的问题:今天天气", store=store)
        check("阈值卡死 → 没有情景召回", ctx.stats["recalled"] == 0)
        check("画像仍常驻(不走阈值)", "持有 100 股 NVDA" in ctx.memory_block)
        recall.RECALL_THRESHOLD = 0.0

        print("\n场景3 · 重排")
        now = time.time()
        fresh_important = epi_rec("A", "x", days_ago=1, importance=9)
        old_trivial = epi_rec("A", "x", days_ago=90, importance=2)
        check("同等相关性:新且重要 > 旧且琐碎",
              recall.score(fresh_important, 0.7, now) > recall.score(old_trivial, 0.7, now))
        check("相关性够高时可以压过新近度",
              recall.score(old_trivial, 0.95, now) > recall.score(fresh_important, 0.3, now))
        check("TOP_K 生效", len(recall.build_context(user_id="A", question="NVDA", store=store).memory_block) > 0)

        print("\n场景4 · 记忆挂了不拖垮主流程")
        n0 = len(seen)

        def boom(_):
            raise RuntimeError("embedding 挂了")

        broken = MemoryStore(collection=store.collection, embed=boom, dim=DIM)
        ctx = recall.build_context(user_id="A", question="NVDA", store=broken)
        check("不抛异常,返回空块", ctx.memory_block == "" and ctx.stats.get("error") is True)
        check("发出 memory_error 事件", any(e["kind"] == "memory_error" for e in seen[n0:]))

        print("\n场景5 · search_memory 工具")
        from langgraph.graph import END, START, MessagesState, StateGraph

        from memory.tools import search_memory as _sm

        # ToolNode 必须在真实图里跑才会走完整的 config 注入路径(单独 invoke 会缺 runtime)。
        g = StateGraph(MessagesState)
        g.add_node("act", ToolNode([_sm]))
        g.add_edge(START, "act")
        g.add_edge("act", END)
        tool_graph = g.compile()

        def call(user, args):
            msg = AIMessage(content="", tool_calls=[{"name": "search_memory", "args": args, "id": "t1"}])
            cfg = {"configurable": {"user_id": user}} if user else {"configurable": {}}
            return tool_graph.invoke({"messages": [msg]}, config=cfg)["messages"][-1].content

        out = call("A", {"query": "NVDA 毛利率"})
        check("ToolNode 把 config 注入工具,按 user_id 检索", "毛利率" in out and "B 用户" not in out, out[:60])
        check("B 只看到 B 的", "B 用户" in call("B", {"query": "NVDA"}) and "毛利率" not in call("B", {"query": "NVDA"}))
        check("缺 user_id → 不查", "无法确定当前用户" in call(None, {"query": "NVDA"}))
        from memory.tools import search_memory
        props = set(search_memory.args.keys())
        check("工具 schema 里没有 user_id / config", props == {"query", "type", "ticker"}, str(props))
        check("type 过滤生效", "TSLA" not in call("A", {"query": "交付", "type": "semantic"}))

        print("\n场景6 · 图节点")
        from graph.nodes import make_recall_node, make_remember_node

        rn = make_recall_node("v2")
        out = rn({"question": "NVDA 毛利率", "messages": []}, {"configurable": {"user_id": "A"}})["messages"]
        kinds = [type(m).__name__ for m in out]
        check("顺序:系统提示 → 记忆块 → 问题", kinds == ["SystemMessage", "HumanMessage", "HumanMessage"], str(kinds))
        check("记忆块是用户级而非 system", sum(isinstance(m, SystemMessage) for m in out) == 1)
        check("缺 user_id 直接报错(不静默落到共享库)",
              _raises(lambda: rn({"question": "q", "messages": []}, {"configurable": {}})))
        out0 = rn({"question": "你好", "messages": [HumanMessage(content="x")]},
                  {"configurable": {"user_id": "NEW"}})["messages"]
        check("无记忆的新用户:只有问题一条", len(out0) == 1)

        llm = FakeLLM(
            {"memories": [sem("risk:preference", {"level": "保守"}, "用户风险偏好:保守")]},
        )
        runtime.set_llm(llm)
        mn = make_remember_node("v2")
        mn({"question": "我偏保守", "messages": [AIMessage(content="好的,已了解")]},
           {"configurable": {"user_id": "W", "thread_id": "t", "run_id": "r"}})
        got = store.active_semantic(user_id="W")
        check("remember 节点写入了抽取结果", len(got) == 1 and got[0].key == "risk:preference")
        check("来源带 run_id", got[0].source.run_id == "r")
        check("别的用户不受影响", store.active_semantic(user_id="NEW") == [])
    finally:
        events.remove_observer(seen.append)
        recall.RECALL_THRESHOLD = old_thr
        runtime.set_store(None)
        runtime.set_llm(None)
        store.drop()

    print("\n" + "=" * 60)
    if _failures:
        print(f"失败 {len(_failures)} 项:" + "、".join(_failures))
        return 1
    print("全部通过")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
