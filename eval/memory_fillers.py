"""给记忆库"弄脏"用的填充数据(确定性,固定随机种子)。

为什么需要它:记忆库只有 1~3 条时,旧方案"top3 全取出"恰好把所有记忆都带回来,
模型自己读文本就能判断新旧、有无,于是看不出任何问题。真实使用里记忆库会越积越多,
**旧方案的弱点只有在库里有大量无关、相近、过时的条目时才暴露**(top3 被挤占、新旧混杂、
无关内容被硬塞)。这批填充数据就是用来复现那种状态的。

两类:
  general     别的股票的持仓/行情/偏好。**刻意不含 NVDA 与 TSLA**,
              因为评测题要问的正是这两只,混进去会让"正确答案"变得有歧义。
  stale_nvda  更早之前关于 NVDA 的持仓陈述(成本各不相同)。评测题随后写入最新值,
              所以正确答案永远是"最后写的那个"。模拟"同一件事被反复记过、版本混杂"。

填充写入时不经过 LLM,只做 embedding,所以便宜、快。
"""

import random
import re

_TICKERS = ["AMD", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "INTC", "QCOM", "AVGO", "比亚迪", "蔚来", "贵州茅台", "腾讯", "阿里巴巴"]


def general(n: int = 24, seed: int = 7) -> list[tuple[str, str]]:
    rnd = random.Random(seed)
    out: list[tuple[str, str]] = []
    for i in range(n):
        t = _TICKERS[i % len(_TICKERS)]
        kind = i % 4
        shares, cost, pct = rnd.randrange(50, 800, 50), rnd.randrange(30, 500, 5), rnd.randrange(1, 15)
        if kind == 0:
            out.append((f"我持有 {shares} 股 {t},成本 {cost} 美元,请记一下。",
                        f"好的,已记录:你持有 {shares} 股 {t},成本 {cost} 美元。"))
        elif kind == 1:
            out.append((f"{t} 最近的股价表现如何?",
                        f"{t} 近 30 天上涨 {pct}%,最新价约 {cost} 美元,成交量放大。"))
        elif kind == 2:
            out.append((f"帮我看看 {t} 的持仓成本和现价差多少。",
                        f"以你记录的成本 {cost} 美元计算,{t} 目前浮盈约 {pct}%。"))
        else:
            out.append((f"{t} 最近有什么新闻?",
                        f"{t} 近期发布了季度业绩,营收同比增长 {pct}%,管理层上调了全年指引。"))
    return out


def stale_nvda() -> list[tuple[str, str]]:
    costs = [95, 100, 105, 110, 115, 118]
    return [
        (f"我持有 100 股 NVDA,成本 {c} 美元,请记一下。", f"好的,已记录:你持有 100 股 NVDA,成本 {c} 美元。")
        for c in costs
    ]


_HOLD = re.compile(r"我持有 (\d+) 股 (\S+?),成本 (\d+) 美元")


def to_v2_records(pairs: list[tuple[str, str]], user_id: str, now: float | None = None):
    """把填充问答转成 v2 的结构化记录,等价于"写入链路已正常工作"之后库里的样子。

    ⚠️ 这是一个**假设**,不是实测:这里直接构造,而没有真的让抽取器跑 N 遍。
    - 持仓陈述 → 语义记忆 holding:T;同一 key 的多个版本按时间串成取代链,只有最后一个 active。
      这正是 v2 写入路径(精确 key 命中 → 取代旧版本)承诺做到的事,已由
      eval/memory_write_offline.py 离线验证过 —— 但"抽取器每次都抽对 key"没有在这里被检验。
    - 其余(行情/新闻/浮盈问答)→ 情景记忆,保留原文,tickers 取问句里出现的标的。
    因此 v2 在"脏库"题上的好成绩,**只证明读取侧在干净结构化数据上的表现**;
    写入侧的真实质量由 §5.3 的探针与端到端 --repeat 评估。
    """
    import time as _t
    import uuid

    from memory.schema import MemoryRecord, Source

    now = now or _t.time()
    n = len(pairs)
    recs: list[MemoryRecord] = []
    last_by_key: dict[str, MemoryRecord] = {}
    for i, (q, a) in enumerate(pairs):
        created = now - (n - i) * 60
        tk = next((t for t in _TICKERS + ["NVDA", "TSLA"] if t in q), None)
        tickers = [tk] if tk and tk.isascii() else []
        m = _HOLD.search(q)
        if m:
            shares, t, cost = int(m[1]), m[2], int(m[3])
            key = f"holding:{t}"
            r = MemoryRecord(id=str(uuid.uuid4()), user_id=user_id, type="semantic", category="holding",
                             key=key, value={"shares": shares, "cost": cost}, tickers=[t] if t.isascii() else [],
                             text=f"用户持有 {shares} 股 {t},成本 {cost} 美元", created_at=created,
                             source=Source(kind="user_statement"))
            prev = last_by_key.get(key)
            if prev is not None:
                prev.status, prev.superseded_by, prev.valid_to = "superseded", r.id, created
            last_by_key[key] = r
        else:
            r = MemoryRecord(id=str(uuid.uuid4()), user_id=user_id, type="episodic", tickers=tickers,
                             text=a, importance=4, created_at=created,
                             source=Source(kind="agent_conclusion"))
        recs.append(r)
    return recs


def build(kinds: list[str]) -> list[tuple[str, str]]:
    """按 kinds 拼出填充列表。stale 放在 general 之后,让它们比大多数填充更"新"但比评测写入更旧。"""
    out: list[tuple[str, str]] = []
    for k in kinds:
        if k == "general":
            out += general()
        elif k == "stale_nvda":
            out += stale_nvda()
        else:
            raise ValueError(f"未知填充类型: {k}")
    return out
