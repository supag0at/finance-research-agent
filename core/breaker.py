"""熔断器(circuit breaker)——M6。

## 解决什么问题

重试解决的是"这一次抽了",熔断解决的是"这条路已经不通了,别再排队撞"。
两者是不同层面:9-24 那轮评测里 `fetch_page` 失败率 47%,每一次失败都老老实实
走完"重试 + 退避"的全套流程,而其实前面已经连续失败好几次了,
系统本该在第三次就得出"这个依赖现在不可用"的结论。

## 三态模型

    CLOSED(闭合)   正常放行。连续失败达到阈值 -> 打开。
    OPEN(断开)     直接拒绝,不碰外部服务。冷却期满 -> 半开。
    HALF_OPEN(半开) 放一个探针请求过去。成功 -> 闭合并清零;失败 -> 重新打开。

"半开放一个探针"是关键:否则冷却期一满,所有积压请求会同时涌向刚刚恢复的服务,
把它再打挂一次(惊群)。

## 刻意的取舍

**按"连续失败"计数,不按失败率。** 失败率需要滑动窗口和最小样本量,
在我们这种一轮几十次调用的量级下,滑动窗口统计意义不足,反而更难解释。
连续失败计数简单、直观、好测。

**状态是进程级的,不是按 run 隔离的。** 因为"Reuters 现在拒绝我们"这个事实
跨 run 依然成立,没理由每轮重新学一遍。代价是评测时前一道题的熔断会影响后一道题,
所以提供了 `reset_all()`,评测逐题跑之间会调用它,保证每题独立可比。
"""

import threading
import time
from dataclasses import dataclass, field

# 连续失败多少次就打开。3 是个折中:1 太敏感(一次抖动就断),
# 5 在我们这种单轮十几次调用的量级下基本等于不熔断。
DEFAULT_THRESHOLD = 3
# 冷却多久后允许探针。60 秒是因为反爬/限流这类封禁通常是分钟级的。
DEFAULT_COOLDOWN_S = 60.0


@dataclass
class _State:
    consecutive_failures: int = 0
    opened_at: float = 0.0
    half_open_in_flight: bool = False


@dataclass
class CircuitBreaker:
    threshold: int = DEFAULT_THRESHOLD
    cooldown_s: float = DEFAULT_COOLDOWN_S
    _states: dict[str, _State] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _state(self, name: str) -> _State:
        return self._states.setdefault(name, _State())

    def allow(self, name: str) -> tuple[bool, str]:
        """能不能放行。返回 (允许, 拒绝原因)。

        返回原因而不是只返回布尔,是为了让拒绝也能变成一条**可读、可塞回 state**
        的说明 —— 熔断对模型来说必须是"这条路关了",而不是一次无声的失败。
        """
        now = time.monotonic()
        with self._lock:
            st = self._state(name)
            if not st.opened_at:
                return True, ""

            elapsed = now - st.opened_at
            if elapsed < self.cooldown_s:
                left = self.cooldown_s - elapsed
                return False, (
                    f"该工具已连续失败 {st.consecutive_failures} 次,已熔断,"
                    f"还需 {left:.0f} 秒后才会重新尝试"
                )

            # 冷却期满:只放一个探针过去
            if st.half_open_in_flight:
                return False, "该工具处于半开状态,已有一个探针请求在试,请改用其他工具"
            st.half_open_in_flight = True
            return True, ""

    def on_success(self, name: str) -> None:
        with self._lock:
            self._states[name] = _State()  # 整体清零:计数、打开时间、半开标记

    def on_failure(self, name: str) -> bool:
        """登记一次最终失败(重试完仍失败)。返回是否因此打开了熔断。"""
        with self._lock:
            st = self._state(name)
            st.consecutive_failures += 1
            was_open = bool(st.opened_at)
            st.half_open_in_flight = False
            if st.consecutive_failures >= self.threshold:
                st.opened_at = time.monotonic()
                return not was_open  # 只在"刚打开"时返回 True,便于只报一次
            return False

    def snapshot(self) -> dict[str, dict]:
        """给观测用:当前各工具的熔断状态。"""
        now = time.monotonic()
        with self._lock:
            return {
                name: {
                    "consecutive_failures": st.consecutive_failures,
                    "open": bool(st.opened_at) and (now - st.opened_at) < self.cooldown_s,
                }
                for name, st in self._states.items()
                if st.consecutive_failures or st.opened_at
            }

    def reset_all(self) -> None:
        with self._lock:
            self._states.clear()


_breaker = CircuitBreaker()


def current_breaker() -> CircuitBreaker:
    return _breaker
