"""召回阈值标定(M7 第 4 步):用真实 embedding 看"相关"与"无关"查询的相似度分布。

为什么要它:阈值是拒答能力的来源,而 embedding 模型的相似度尺度不是凭感觉能猜的
(M6 的教训:上限落进正常区间 = 随机终止器)。这里只花一次 embedding 批量调用(~40 条短文本)。

做法:每条查询取"对所有记忆的最高余弦相似度"。
  - 相关查询(有对应记忆):这个最高分是否足够高 → 决定漏召回率;
  - 无关查询(记忆里没有答案):这个最高分是否足够低 → 决定误召回率;
  - 近似查询(同一只股票但问的是别的方面):两可,单独列出,不参与阈值推荐。

运行: python -m eval.memory_calibrate
⚠️ 样本很小(约 10 对),只够选一个"在这批数据上合理"的初值,不能当成统计结论。
"""

import math
import sys

from memory.store import _default_embed

MEMORIES = [
    "2026-07 调研 NVDA 毛利率,结论 73%,数据截至 2026-07-31",
    "TSLA 2026Q3 交付量 46.2 万辆,同比下降 8%",
    "AMD 数据中心业务营收同比增长 38%,MI300 出货加速",
    "腾讯 2026Q2 游戏收入同比增长 12%",
    "比亚迪 9 月销量 41 万辆,海外占比提升",
    "NVDA 面临的主要风险:客户集中度高、出口管制、竞争加剧",
    "AAPL iPhone 17 需求偏弱,服务业务毛利率 74%",
    "台积电先进封装 CoWoS 产能扩张计划,2026 年底月产能 8 万片",
]

# (查询, 类别, 目标记忆下标)
QUERIES = [
    ("NVDA 毛利率最近怎么样", "相关", 0),
    ("我之前查过的 NVDA 利润率结论", "相关", 0),
    ("特斯拉上季度交付多少", "相关", 1),
    ("AMD 的数据中心增长情况", "相关", 2),
    ("腾讯游戏业务表现", "相关", 3),
    ("比亚迪海外销量", "相关", 4),
    ("英伟达有什么风险", "相关", 5),
    ("苹果服务业务毛利", "相关", 6),
    ("CoWoS 产能", "相关", 7),
    ("什么是市盈率?", "无关", None),
    ("今天天气怎么样", "无关", None),
    ("记住我的密码是 4417", "无关", None),
    ("用一句话总结你刚才的结论", "无关", None),
    ("帮我写一首诗", "无关", None),
    ("Python 里怎么读取 CSV", "无关", None),
    ("黄金价格最近的走势", "无关", None),
    ("美联储下次议息会议是什么时候", "无关", None),
    ("你好", "无关", None),
    ("NVDA 的自由现金流怎么样", "近似", 0),
    ("特斯拉的自动驾驶进展", "近似", 1),
]


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) or 1) * (math.sqrt(sum(y * y for y in b)) or 1))


def main() -> int:
    vecs = _default_embed(MEMORIES + [q for q, _, _ in QUERIES])
    mv, qv = vecs[: len(MEMORIES)], vecs[len(MEMORIES):]
    rows = []
    for (q, cat, tgt), v in zip(QUERIES, qv):
        sims = [cos(v, m) for m in mv]
        best = max(range(len(sims)), key=sims.__getitem__)
        score = sims[tgt] if (cat == "相关" and tgt is not None) else sims[best]
        rows.append((cat, q, score, best, tgt))
    for cat in ("相关", "近似", "无关"):
        print(f"\n== {cat} ==")
        for c, q, s, best, tgt in sorted((r for r in rows if r[0] == cat), key=lambda r: -r[2]):
            hit = "" if tgt is None else ("✓命中目标" if best == tgt else f"✗最近的是 #{best}")
            print(f"  {s:.3f}  {q}  {hit}")
    rel = [r[2] for r in rows if r[0] == "相关"]
    irr = [r[2] for r in rows if r[0] == "无关"]
    print(f"\n相关:min={min(rel):.3f} max={max(rel):.3f} | 无关:min={min(irr):.3f} max={max(irr):.3f}")
    if min(rel) > max(irr):
        print(f"可分:建议阈值落在 ({max(irr):.3f}, {min(rel):.3f}) 之间,中点 {(max(irr) + min(rel)) / 2:.3f}")
    else:
        print("⚠️ 不可分:相关与无关的分数区间有重叠,单一阈值无法同时做到不漏不误。")
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            print(f"  τ={t}: 漏召回 {sum(s < t for s in rel)}/{len(rel)},误召回 {sum(s >= t for s in irr)}/{len(irr)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
