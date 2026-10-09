"""记忆存储层的离线测试(M7 第 2 步)。零 LLM、零 embedding 费用,确定性。

    python -m eval.memory_offline

只需要本地 Qdrant。用"字符哈希词袋"当假 embedding —— 它不懂语义,但满足:
相同文本 → 相同向量(相似度 1.0),不同文本 → 较低相似度。
这足以测"隔离、过滤、取代、阈值"这些**与向量语义无关**的行为;
语义好不好(召回准不准)要靠带真 embedding 的 eval.memory_runner,这里不测。

为什么单独做这个:隔离是安全属性,必须有**每次改代码都能零成本重跑**的回归测试,
而不是只靠偶尔跑一次的 LLM 评测碰运气(评测里的隔离题要花钱、有噪声)。
"""

import hashlib
import math
import time
import uuid

from memory.schema import MemoryRecord, Source
from memory.store import MemoryStore

DIM = 64
_failures: list[str] = []


def fake_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        v = [0.0] * DIM
        for ch in t:
            v[int(hashlib.md5(ch.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def rec(user: str, text: str, **kw) -> MemoryRecord:
    kw.setdefault("type", "semantic")
    kw.setdefault("source", Source(kind="user_statement"))
    return MemoryRecord(id=str(uuid.uuid4()), user_id=user, text=text, **kw)


def raises(exc: type, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def main() -> int:
    store = MemoryStore(collection=f"fra_memory_test_{uuid.uuid4().hex[:8]}", embed=fake_embed, dim=DIM)
    try:
        print("=" * 60 + "\nM7 记忆存储层离线测试\n" + "=" * 60)

        print("\n场景1 · 用户隔离")
        a1 = rec("A", "用户持有 100 股 NVDA,成本 120 美元", category="holding",
                 key="holding:NVDA", value={"shares": 100}, tickers=["NVDA"])
        b1 = rec("B", "用户持有 200 股 NVDA", category="holding",
                 key="holding:NVDA", value={"shares": 200}, tickers=["NVDA"])
        store.add_many(user_id="A", records=[a1])
        store.add_many(user_id="B", records=[b1])
        only_a = store.search(user_id="A", query="NVDA 持仓")
        check("A 的检索只返回 A 的记录", [r.id for r, _ in only_a] == [a1.id])
        check("B 的检索只返回 B 的记录", [r.id for r, _ in store.search(user_id="B", query="NVDA 持仓")] == [b1.id])
        check("画像隔离", [r.id for r in store.active_semantic(user_id="B")] == [b1.id])
        check("同 key 精确查询各取各的",
              store.find_by_key(user_id="A", key="holding:NVDA").value == {"shares": 100}
              and store.find_by_key(user_id="B", key="holding:NVDA").value == {"shares": 200})
        check("没有数据的用户什么都查不到",
              store.search(user_id="C", query="NVDA 持仓") == [] and store.active_semantic(user_id="C") == [])

        print("\n场景2 · 接口层防呆")
        check("不传 user_id 直接 TypeError", raises(TypeError, lambda: store.search(query="x")))
        check("空 user_id 被拒绝", raises(ValueError, lambda: store.search(user_id="  ", query="x")))
        check("记录与参数 user_id 不一致被拒绝",
              raises(ValueError, lambda: store.add_many(user_id="A", records=[rec("B", "x")])))
        check("Schema 拒绝空 user_id", raises(ValueError, lambda: rec("", "x")))

        print("\n场景3 · 取代而不删除")
        new_a = rec("A", "用户持有 150 股 NVDA", category="holding",
                    key="holding:NVDA", value={"shares": 150}, tickers=["NVDA"])
        store.add_many(user_id="A", records=[new_a])           # 先写新的
        store.supersede(user_id="A", old_id=a1.id, new_id=new_a.id)  # 成功后再让旧的失效
        cur = store.find_by_key(user_id="A", key="holding:NVDA")
        check("当前有效版本是新值", cur is not None and cur.id == new_a.id and cur.value == {"shares": 150})
        hist = store.history(user_id="A", key="holding:NVDA")
        check("旧版本仍可追溯(未物理删除)", {h.id for h in hist} == {a1.id, new_a.id})
        old = next(h for h in hist if h.id == a1.id)
        check("旧版本状态与指向正确",
              old.status == "superseded" and old.superseded_by == new_a.id and old.valid_to is not None)
        check("失效记录不进检索", a1.id not in [r.id for r, _ in store.search(user_id="A", query="NVDA")])
        check("不能越权让别人的记忆失效",
              raises(KeyError, lambda: store.supersede(user_id="A", old_id=b1.id, new_id=None)))
        check("B 的数据没被碰", store.find_by_key(user_id="B", key="holding:NVDA").id == b1.id)

        print("\n场景4 · subject 消歧")
        mine = rec("D", "用户自己持有 200 股茅台", category="holding", key="holding:茅台", subject="self")
        dad = rec("D", "用户父亲持有 500 股茅台", category="holding", key="holding:茅台", subject="父亲")
        store.add_many(user_id="D", records=[mine, dad])
        check("同 key 不同 subject 互不覆盖",
              store.find_by_key(user_id="D", key="holding:茅台", subject="self").id == mine.id
              and store.find_by_key(user_id="D", key="holding:茅台", subject="父亲").id == dad.id)

        print("\n场景5 · 类型/ticker 过滤与阈值")
        ep = rec("E", "10-07 调研 NVDA 毛利率 73%", type="episodic", tickers=["nvda"], topic="毛利率")
        se = rec("E", "用户关注 AMD", category="watchlist", key="watchlist:AMD", tickers=["AMD"])
        store.add_many(user_id="E", records=[ep, se])
        check("ticker 入库时统一大写", ep.tickers == ["NVDA"])
        check("type 过滤", [r.id for r, _ in store.search(user_id="E", query="x", type="episodic")] == [ep.id])
        check("ticker 过滤(小写入参也能匹配)",
              [r.id for r, _ in store.search(user_id="E", query="x", ticker="amd")] == [se.id])
        hit = store.search(user_id="E", query="10-07 调研 NVDA 毛利率 73%", type="episodic")
        check("相同文本相似度≈1", bool(hit) and hit[0][1] > 0.99, f"{hit[0][1]:.3f}" if hit else "无")
        check("阈值过滤:高阈值下不相关内容不返回",
              store.search(user_id="E", query="zzzzzzzzzz qqqqqqqq", threshold=0.99) == [])

        print("\n场景6 · 数据完整性与清理")
        got = store.find_by_key(user_id="B", key="holding:NVDA")
        check("嵌套 value 原样往返(无序列化)", got.value == {"shares": 200})
        check("as_of 与 created_at 是两个独立字段",
              rec("F", "x", as_of="2026-07-31").as_of == "2026-07-31"
              and abs(rec("F", "x", as_of="2026-07-31").created_at - time.time()) < 5)
        store.purge_user(user_id="A")
        check("purge 清掉该用户(含已失效)",
              store.history(user_id="A", key="holding:NVDA") == [] and store.active_semantic(user_id="A") == [])
        check("purge 不影响其他用户", store.find_by_key(user_id="B", key="holding:NVDA") is not None)
    finally:
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
