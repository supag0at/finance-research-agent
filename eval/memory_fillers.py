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
