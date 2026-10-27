"""记忆写入编排的离线测试(M7 第 3 步)。零费用、确定性。

    python -m eval.memory_write_offline

LLM 用脚本化的假实现(按调用顺序返回预设的 JSON),embedding 用假哈希向量。
所以这里测的是**编排逻辑**:筛选、校验、精确/模糊冲突、取代与撤销、写入顺序、
失败不抛异常 —— 而不是"抽取器 prompt 写得好不好"(那要用真模型做抽取探针)。
"""

import uuid

from eval.memory_offline import DIM, fake_embed
from memory import events
from memory.extract import MAX_CANDIDATES, extract
from memory.store import MemoryStore
from memory.write import write_memories

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


class FakeLLM:
    """按顺序返回预设结果。元素可以是 dict、Exception、或接收 messages 的函数。"""

    def __init__(self, *script) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def complete_json(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(messages) if callable(item) else item


def sem(key, value, text, **kw) -> dict:
    cat = key.split(":")[0]
    return {"type": "semantic", "category": cat, "key": key, "subject": "self",
            "value": value, "text": text, "tickers": kw.get("tickers", []), "importance": 7, **kw}


def epi(text, **kw) -> dict:
    return {"type": "episodic", "text": text, "tickers": ["NVDA"], "topic": "毛利率",
            "as_of": "2026-07-31", "importance": 9, **kw}


def write(store, user, llm, **kw) -> dict:
    return write_memories(user_id=user, question="q", answer="a", llm=llm, store=store, **kw)


def main() -> int:
    store = MemoryStore(collection=f"fra_memory_wtest_{uuid.uuid4().hex[:8]}", embed=fake_embed, dim=DIM)
    seen: list[str] = []
    events.add_observer(lambda e: seen.append(e["kind"]))
    try:
        print("=" * 60 + "\nM7 记忆写入编排离线测试\n" + "=" * 60)

        print("\n场景1 · 筛选:抽取器返回空 → 什么都不写")
        s = write(store, "U1", FakeLLM({"memories": []}))
        check("计数正确", s["extracted"] == 0 and s["skipped"] == 1 and s["added"] == 0, str(s))
        check("库里确实没有东西", store.active_semantic(user_id="U1") == [])

        print("\n场景2 · 新增 → 取代 → 重复")
        llm = FakeLLM(
            {"memories": [sem("holding:nvda", {"shares": 100}, "用户持有 100 股 NVDA", tickers=["nvda"])]},
            {"memories": [sem("holding:NVDA", {"shares": 150}, "用户持有 150 股 NVDA")]},
            {"memories": [sem("holding:NVDA", {"shares": 150}, "用户持有 150 股 NVDA")]},
        )
        s1, s2, s3 = (write(store, "U2", llm) for _ in range(3))
        check("首次:新增 1 条", s1["added"] == 1, str(s1))
        check("key 被规范化为大写", store.find_by_key(user_id="U2", key="holding:NVDA") is not None)
        check("二次:值不同 → 更新(取代)", s2["updated"] == 1 and s2["added"] == 0, str(s2))
        cur = store.find_by_key(user_id="U2", key="holding:NVDA")
        check("当前有效版本是 150", cur.value == {"shares": 150})
        hist = store.history(user_id="U2", key="holding:NVDA")
        check("旧版本仍可追溯且状态为 superseded",
              len(hist) == 2 and hist[0].status == "superseded" and hist[0].superseded_by == cur.id)
        check("三次:值相同 → NOOP,库里仍只有 2 个版本",
              s3["noop"] == 1 and len(store.history(user_id="U2", key="holding:NVDA")) == 2, str(s3))
        check("抽取器看到了已有画像(才能输出'又买 50 股'后的 150)",
              "holding:NVDA" in llm.calls[1][1]["content"])
        check("发出了 added/updated/noop 事件",
              {"memory_added", "memory_updated", "memory_noop"} <= set(seen))

        print("\n场景3 · 撤销(清仓)与重新买入")
        llm = FakeLLM(
            {"memories": [sem("holding:NVDA", {"shares": 100}, "用户持有 100 股 NVDA")]},
            {"memories": [{"type": "semantic", "category": "holding", "key": "holding:NVDA", "retract": True}]},
            {"memories": [{"type": "semantic", "category": "holding", "key": "holding:AMD", "retract": True}]},
            {"memories": [sem("holding:NVDA", {"shares": 30}, "用户持有 30 股 NVDA")]},
        )
        write(store, "U3", llm)
        s = write(store, "U3", llm)
        check("持仓撤销 → 取代为清仓记录(updated 1,不是单纯作废)",
              s["updated"] == 1 and s["invalidated"] == 0, str(s))
        cur = store.find_by_key(user_id="U3", key="holding:NVDA")
        check("当前有效记录是『已清仓』事实", cur is not None and cur.value == {"shares": 0, "closed": True}
              and "清仓" in cur.text, str(cur and cur.text))
        check("旧持仓记录被取代而非删除",
              any(h.status == "superseded" and h.value == {"shares": 100}
                  for h in store.history(user_id="U3", key="holding:NVDA")))
        s = write(store, "U3", llm)
        check("撤销不存在的东西 → 跳过而不是报错", s["skipped"] == 1 and s["errors"] == 0, str(s))
        s = write(store, "U3", llm)
        check("清仓后重新买入 → 取代清仓记录", s["updated"] == 1)
        check("当前有效值是 30", store.find_by_key(user_id="U3", key="holding:NVDA").value == {"shares": 30})

        print("\n场景3b · 非持仓类撤销(关注列表)仍然是作废")
        llm = FakeLLM(
            {"memories": [{"type": "semantic", "category": "watchlist", "key": "watchlist:AMD",
                           "value": {"on": True}, "text": "用户关注 AMD", "tickers": ["AMD"], "importance": 5}]},
            {"memories": [{"type": "semantic", "category": "watchlist", "key": "watchlist:AMD", "retract": True}]},
        )
        write(store, "U3b", llm)
        s = write(store, "U3b", llm)
        check("关注移除 → invalidated 1 且无新记录", s["invalidated"] == 1 and s["added"] == 0 and s["updated"] == 0, str(s))
        check("移除后没有有效记录", store.find_by_key(user_id="U3b", key="watchlist:AMD") is None)

        print("\n场景4 · 写入顺序:新记录没写成,旧记录不能失效")
        write(store, "U4", FakeLLM({"memories": [sem("holding:NVDA", {"shares": 100}, "用户持有 100 股 NVDA")]}))

        def boom(_texts):
            raise RuntimeError("embedding 挂了")

        flaky = MemoryStore(collection=store.collection, embed=boom, dim=DIM)
        s = write(flaky, "U4", FakeLLM({"memories": [sem("holding:NVDA", {"shares": 999}, "用户持有 999 股 NVDA")]}))
        check("记录了写入错误且没有抛异常", s["errors"] == 1 and s["updated"] == 0, str(s))
        still = store.find_by_key(user_id="U4", key="holding:NVDA")
        check("旧记录仍然有效(没有'旧的丢了、新的没写进去')",
              still is not None and still.value == {"shares": 100})

        print("\n场景5 · 候选校验:坏的丢弃,好的照写")
        s = write(store, "U5", FakeLLM({"memories": [
            {"type": "semantic", "text": "缺 category 和 key"},
            {"type": "bogus", "text": "类型不存在"},
            {"type": "semantic", "category": "weird", "key": "weird:x", "text": "category 不在枚举里"},
            sem("watchlist:AMD", None, "用户关注 AMD", tickers=["AMD"]),
        ]}))
        check("3 条被跳过、1 条写入", s["skipped"] == 3 and s["added"] == 1 and s["errors"] == 0, str(s))

        print("\n场景6 · 情景记忆:提前收尾降权;重复不重复存")
        s = write(store, "U6", FakeLLM({"memories": [epi("NVDA 毛利率 73%")]}), stop_reason="成本超限")
        e = store.recent_episodic(user_id="U6")[0]
        check("提前收尾:重要性被压到上限以下", e.importance <= 4, f"importance={e.importance}")
        check("文本里注明结论可能不完整", "提前收尾" in e.text)
        check("来源标为 agent_conclusion", e.source.kind == "agent_conclusion")
        write(store, "U7", FakeLLM({"memories": [epi("NVDA 毛利率 73%")]}))
        s = write(store, "U7", FakeLLM({"memories": [epi("NVDA 毛利率 73%")]}))
        check("同一结论再记一次 → NOOP,只有 1 条", s["noop"] == 1 and len(store.recent_episodic(user_id="U7")) == 1, str(s))

        print("\n场景7 · 失败不抛异常")
        s = write(store, "U8", FakeLLM(ValueError("模型返回的不是合法 JSON")))
        check("抽取失败 → 记 1 个错误,不抛", s["errors"] == 1 and s["added"] == 0, str(s))
        check("留下了 memory_error 事件", "memory_error" in seen)

        print("\n场景8 · 模糊匹配 → LLM 判定")
        old_text = "用户持有 100 股 NVDA 成本 120 美元"

        def seed(user):
            llm0 = FakeLLM({"memories": [sem("holding:NVDA", {"shares": 100}, old_text)]})
            write(store, user, llm0)
            return store.find_by_key(user_id=user, key="holding:NVDA")

        new = sem("holding:英伟达", {"shares": 120}, old_text)  # key 不同、文本相同 → 精确匹配不到,向量命中
        old = seed("U9a")
        s = write(store, "U9a", FakeLLM({"memories": [new]},
                  lambda m: {"action": "UPDATE", "target_id": old.id, "reason": "同一持仓"}))
        check("UPDATE → 新的取代旧的", s["updated"] == 1 and store.history(user_id="U9a", key="holding:NVDA")[0].status == "superseded", str(s))
        old = seed("U9b")
        s = write(store, "U9b", FakeLLM({"memories": [new]},
                  lambda m: {"action": "INVALIDATE", "target_id": old.id}))
        check("INVALIDATE → 旧的失效,不写新的", s["invalidated"] == 1 and s["added"] == 0, str(s))
        seed("U9c")
        s = write(store, "U9c", FakeLLM({"memories": [new]}, {"action": "NOOP", "target_id": None}))
        check("NOOP → 不写", s["noop"] == 1 and s["added"] == 0, str(s))
        seed("U9d")
        s = write(store, "U9d", FakeLLM({"memories": [new]}, {"action": "UPDATE", "target_id": "不存在的id"}))
        check("模型给了不存在的 id → 退化为 ADD(不信模型)", s["added"] == 1 and s["updated"] == 0, str(s))
        seed("U9e")
        s = write(store, "U9e", FakeLLM({"memories": [new]}, RuntimeError("judge 挂了")))
        check("判定器失败 → 当作新信息 ADD + 记错误(重复好过丢失)", s["added"] == 1 and s["errors"] == 1, str(s))

        print("\n场景9 · 同一批次内同 key 出现两次")
        s = write(store, "U10", FakeLLM({"memories": [
            sem("holding:MSFT", {"shares": 100}, "用户持有 100 股 MSFT"),
            sem("holding:MSFT", {"shares": 200}, "用户持有 200 股 MSFT")]}))
        check("只写一条,以后者为准",
              s["added"] == 1 and store.find_by_key(user_id="U10", key="holding:MSFT").value == {"shares": 200}, str(s))

        print("\n场景10 · 代码层硬约束(不依赖模型听话)")
        raw = [epi(f"结论{i}") for i in range(3)] + [sem(f"watchlist:T{i}", None, f"关注T{i}") for i in range(6)]
        out = extract(FakeLLM({"memories": raw}), question="q", answer="a")
        check(f"候选总数 ≤ {MAX_CANDIDATES}", len(out) <= MAX_CANDIDATES, f"{len(out)} 条")
        check("情景记忆最多 1 条", sum(c["type"] == "episodic" for c in out) == 1)
        try:
            extract(FakeLLM({"memories": "不是数组"}), question="q", answer="a")
            check("memories 不是数组 → 抛错(由写入层记录)", False)
        except ValueError:
            check("memories 不是数组 → 抛错(由写入层记录)", True)
    finally:
        events.remove_observer(events._observers[-1]) if events._observers else None
        store.drop()

    print("\n" + "=" * 60)
    if _failures:
        print(f"失败 {len(_failures)} 项:")
        for f in _failures:
            print("  -", f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
