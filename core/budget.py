"""运行预算与循环检测(M6)——Agent 可靠性里最吃重的一层。

## 为什么这是最重要的一层,而不是重试

调研时看到的最有价值的一句话:

    "bounds, not retries, are the load-bearing failure controls."
    Retries help with transient infra. Bounds (step limits, cost limits, tool scopes,
    approval gates) are what keep a **wrong model decision** small.

区别在于:重试修的是**基础设施抖动**,边界兜的是**模型判断错误**。
而 Agent 最贵的故障恰恰是后者 —— 模型"合理地"决定再查一次,循环十圈。
这类故障重试策略无法处理,只能靠硬边界止损。

另一句同样好用:

    A human operator who tries something three times and fails will stop and ask someone.
    An agent will keep going, and each attempt costs money and may leave partial state behind.

## 我们的实测依据

9-24 那轮评测:`nvda_perf_01` 转 10 圈、调 29 次工具、花 ¥0.0656、耗时 195.8 秒,
而同一轮里正常的题是 2 圈、¥0.005、5 秒。**贵 13 倍、慢 39 倍**。
当时唯一的刹车是步数上限,但步数不等于钱 —— 同样 10 圈可以差一个数量级。
另外还实测到过**单次 LLM 调用 942 秒**:`httpx` 的 `timeout=60` 是单次操作超时,
SDK 自己还会重试,叠起来远超 60 秒,而整个 run 根本没有墙钟上限。

## 四条互相独立的停止条件

刻意做成**互相独立**的,因为任何单一条件都有绕过的方式:

    成本上限    直接对应钱。步数少但每步很贵时只有它能拦住。
    墙钟上限    对应用户耐心。模型卡在一次超长调用上时只有它能拦住。
    步数上限    对应"绕圈子"。便宜又快但永不收敛时只有它能拦住。
    循环检测    对应"同一个动作重复做"。上面三个都没到上限时它能提前止损。

## 关键设计:超限之后是"升级",不是"静默降级"

    When a budget is exhausted, stop and **escalate** rather than degrade quietly.

所以超限不是直接返回一句"失败了",而是走 `finalize` 节点:不带工具地再问模型一次,
要求它基于已拿到的信息给出结论**并明确说明哪些数据缺失**。
这同时修复了改造前的一个缺陷 —— 原来撞上限直接返回
`"(已达到最大步数,未能得出最终结论)"`,`nvda_perf_01` 明明已经调了 29 次工具
拿到了部分数据,却交白卷,裁判三项全给 1 分。

## 为什么不自己算钱

成本在 `obs/metrics.py` 的 `RunRecord` 里算。但 `core/` 不能反向依赖 `obs/`
(`reliability.py` 的注释里立过这个规矩)。所以 `RunBudget` 只接收别人推过来的数字,
由 `ObsHandler` 在 `on_llm_end` 里调 `update()`。

**观测到控制的闭环就在这里闭上**:M5.2 采的数据第一次被用来做运行时决策,
而不只是事后看报告。
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


def _env_float(key: str, default: Optional[float]) -> Optional[float]:
    """读一个可选的数值配置。留空 = 不限制(而不是 0 = 一上来就超限)。"""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if v > 0 else None  # 显式填 0 表示关闭该项限制


def _env_int(key: str, default: Optional[int]) -> Optional[int]:
    v = _env_float(key, None)
    if v is None:
        return None if os.getenv(key, "").strip() else default
    return int(v)


# 默认值的取法:都参照 9-24 那轮实测。正常题是 2 圈 / ¥0.005 / 5 秒,
# 失控题是 10 圈 / ¥0.066 / 196 秒。所以默认值卡在两者之间,
# 既不会冤枉正常题,又能在失控早期就止损。
# ---------------------------------------------------------------------------
# 默认上限:**必须从实测分布里标定,不能凭经验估计**
# ---------------------------------------------------------------------------
#
# 这是实测交的学费。第一版我凭直觉填了 成本 0.03 元 / 耗时 90 秒,
# 跑完才发现健康基线(5 题全过那轮)的实测分布是:
#       成本 0.0074 ~ 0.0533 元,耗时 5.8 ~ 133.6 秒,LLM 2 ~ 6 次
# 也就是说,**我设的上限比最差的一次正常运行还低**(成本低 1.8 倍、耗时低 1.5 倍)。
#
# 后果不是"更安全",而是**更贵**:边界在正常工作中途就触发,
# 于是我们既付了已经做掉的那部分工作的钱,又多付了一次 finalize 收尾调用的钱。
# chip_news_01 实测从 ¥0.0533(不设边界,正常跑完)变成 ¥0.0717(边界触发 + 收尾),
# 多花 35%,而且答案质量并没有变好 —— 它本来就能过。
#
# 教训:止损线的作用是**区分"失控"和"正常"**。如果它落在正常区间里面,
# 它就不是止损线,而是一个随机的提前终止器 —— 既不省钱也不提质量。
# 正确做法:先测健康运行的分布,再把上限设在它的上方(这里取 ≈2 倍最差健康值),
# 让它只在真的跑飞时才生效。
#
# 当前标定依据(2026-10-04 基线,deepseek-flash,5 题):
#   成本上限 0.10 ≈ 最差健康值 0.0533 的 1.9 倍
#   耗时上限 240s ≈ 最差健康值 133.6s 的 1.8 倍
#   LLM 上限 8    ≈ 最差健康值 6 次 的 1.3 倍(它同时受框架 recursion_limit 约束)
# ⚠️ 换模型或换题目后必须重新标定,否则这几个数字就是过期的。
DEFAULT_MAX_COST = _env_float("RUN_MAX_COST_CNY", 0.10)  # 元
DEFAULT_MAX_WALL_S = _env_float("RUN_MAX_WALL_S", 240.0)  # 秒
DEFAULT_MAX_LLM_CALLS = _env_int("RUN_MAX_LLM_CALLS", 8)
DEFAULT_REPEAT_LIMIT = _env_int("RUN_REPEAT_LIMIT", 2)


@dataclass
class RunBudget:
    """一次运行的预算。线程安全(工具并行时回调会从多个线程推送用量)。

    各上限为 None 表示不限制。
    """

    max_cost_cny: Optional[float] = DEFAULT_MAX_COST
    max_wall_s: Optional[float] = DEFAULT_MAX_WALL_S
    max_llm_calls: Optional[int] = DEFAULT_MAX_LLM_CALLS
    repeat_limit: Optional[int] = DEFAULT_REPEAT_LIMIT

    cost_cny: Optional[float] = None  # None = 未配单价,算不出来
    llm_calls: int = 0
    _t0: float = field(default_factory=time.monotonic)
    _repeats: dict[tuple[str, str], int] = field(default_factory=dict)
    _tripped: str = ""  # 非空表示已超限,内容是人类可读的原因
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """把"显式填 0"归一成 None(= 关掉这条限制)。

        这个归一化必须在这里做一次,不能散在 exceeded() 的判断里。
        否则 0 会被当成一个**真实的上限**:"耗时不得超过 0 秒"在第一次检查时
        就必然成立,于是"我想关掉这条限制"变成了"每次运行立刻终止"。
        这是实测发现的 —— 配置项里最危险的值往往就是 0,因为它既是合法数字
        又是人类表达"不要"的习惯写法。
        """
        if not self.max_cost_cny:
            self.max_cost_cny = None
        if not self.max_wall_s:
            self.max_wall_s = None
        if not self.max_llm_calls:
            self.max_llm_calls = None
        if not self.repeat_limit:
            self.repeat_limit = None

    # ---------- 采集 ----------

    def update(self, cost_cny: Optional[float], llm_calls: int) -> None:
        """由 ObsHandler 在每次 LLM 调用结束后推送当前累计值。"""
        with self._lock:
            self.cost_cny = cost_cny
            self.llm_calls = llm_calls

    def note_tool_calls(self, calls: list[tuple[str, str]]) -> None:
        """登记模型这一轮想调的工具 [(工具名, 目标)],用于循环检测。

        注意记的是**模型的意图**而不是执行结果:即使工具每次都"成功"返回了
        无用内容,反复调同一个目标本身就是不收敛的信号。
        """
        with self._lock:
            for key in calls:
                self._repeats[key] = self._repeats.get(key, 0) + 1

    # ---------- 判定 ----------

    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    def exceeded(self) -> str:
        """返回超限原因;空字符串表示一切正常。

        一旦超限就**记住**(`_tripped`),避免同一轮里反复判定、也便于观测取用。
        """
        with self._lock:
            if self._tripped:
                return self._tripped
            reason = ""
            if self.max_cost_cny is not None and (self.cost_cny or 0) > self.max_cost_cny:
                reason = f"成本超限(已花 ¥{self.cost_cny:.4f} > 上限 ¥{self.max_cost_cny:.4f})"
            elif self.max_wall_s is not None and self.elapsed_s() > self.max_wall_s:
                reason = f"耗时超限(已用 {self.elapsed_s():.0f}s > 上限 {self.max_wall_s:.0f}s)"
            elif self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
                reason = f"模型调用次数超限({self.llm_calls} >= 上限 {self.max_llm_calls})"
            else:
                rep = self._worst_repeat_locked()
                if rep:
                    tool, target, n = rep
                    reason = (
                        f"检测到重复调用({tool} 对同一目标已请求 {n} 次,"
                        f"目标:{target or '无参数'})"
                    )
            if reason:
                self._tripped = reason
            return reason

    def _worst_repeat_locked(self) -> Optional[tuple[str, str, int]]:
        if self.repeat_limit is None:
            return None
        for (tool, target), n in self._repeats.items():
            if n > self.repeat_limit:
                return tool, target, n
        return None

    # ---------- 输出 ----------

    def tripped_reason(self) -> str:
        with self._lock:
            return self._tripped

    def to_json(self) -> dict:
        with self._lock:
            return {
                "tripped": self._tripped or None,
                "cost_cny": round(self.cost_cny, 6) if self.cost_cny is not None else None,
                "llm_calls": self.llm_calls,
                "wall_s": round(self.elapsed_s(), 1),
                "limits": {
                    "cost_cny": self.max_cost_cny,
                    "wall_s": self.max_wall_s,
                    "llm_calls": self.max_llm_calls,
                    "repeat": self.repeat_limit,
                },
            }
