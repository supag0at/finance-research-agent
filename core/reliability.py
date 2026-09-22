"""可靠性层:给"执行一个工具"这件事包上重试、超时、降级。

为什么需要它?工具要么联网、要么调外部 API,现实中一定会出问题:
网络抖动、对方超时、返回垃圾、限流……如果不管,Agent 会直接崩掉。
生产级系统的要求(也是 JD 反复强调的)是:出错也别崩,能重试就重试,
实在不行就"降级"给个兜底结果,让整个流程继续走下去。

本文件实现一个核心函数 run_reliably(),把这些保护逻辑集中在一处。
"""

import concurrent.futures
import time
from typing import Any, Callable, Optional

# ---- 可观测钩子(M5.2)----
# 为什么必须在这里埋点?因为本文件的职责就是"把失败吞掉、悄悄重试、必要时降级",
# 对上层来说一切正常。于是 LangChain 的回调机制**永远收不到** on_tool_error,
# 内部重试也不走它的 on_retry —— 工具失败率和重试次数在外面是测不到的。
#
# 观测代码不该反向依赖 obs 模块(那样 core 就依赖上层了),所以这里只留一个
# 极简的观察者列表:谁想听,自己注册。默认没人听,零开销。
#
# 局限:观察者是模块级的,不是按调用上下文隔离的。我们的运行是串行的(一次只跑
# 一个 agent.run),所以够用;真要并发跑多个 Agent,得改成 contextvars。
_observers: list[Callable[[dict], None]] = []


def add_observer(fn: Callable[[dict], None]) -> None:
    _observers.append(fn)


def remove_observer(fn: Callable[[dict], None]) -> None:
    if fn in _observers:
        _observers.remove(fn)


def _emit(event: dict) -> None:
    for fn in list(_observers):
        try:
            fn(event)
        except Exception:  # noqa: BLE001 观测失败绝不能影响主流程
            pass


def run_reliably(
    func: Callable,
    kwargs: dict,
    timeout: float = 15.0,
    retries: int = 2,
    fallback: Optional[Callable] = None,
    name: Optional[str] = None,
) -> Any:
    """带"超时 + 重试 + 降级"地执行一个函数。

    参数:
        func: 真正要执行的工具函数。
        kwargs: 传给它的参数(字典)。
        timeout: 单次执行的超时时间(秒)。超过就算这次失败。
        retries: 失败后最多再重试几次(总尝试次数 = 1 + retries)。
        fallback: 全部失败后的兜底。可以是一个函数(会用同样的 kwargs 调用),
                  也可以为 None(那就返回一段错误说明)。
        name: 工具名,只用于观测事件。**调用方应显式传入注册时的工具名**:
              如果退回 func.__name__,当函数名和注册名不一致时(包装、别名、
              测试打桩),观测数据会分裂成两个条目——重试记在函数名下、
              调用次数记在工具名下,对不上。这是实测踩出来的。

    返回:
        成功时返回工具结果;失败且有 fallback 时返回兜底结果;
        失败且无 fallback 时返回一段可读的错误字符串(而不是抛异常、让 Agent 崩)。
    """
    last_error = "未知错误"
    name = name or getattr(func, "__name__", "unknown")

    for attempt in range(1, retries + 2):  # 第 1 次 + retries 次重试
        try:
            # 用一个单线程的执行器来实现"超时":如果 func 在 timeout 秒内没返回,
            # future.result 会抛 TimeoutError。
            #
            # 诚实的局限:线程方式的超时无法真正"杀死"仍在运行的 func,它会在后台
            # 继续跑到结束。所以最佳实践是"两道防线":
            #   (1) 工具内部自带超时(例如 httpx.get(..., timeout=10));
            #   (2) 这里再加一层外层兜底超时。
            # 对可以真正强杀的场景(如执行任意代码),才需要用子进程/容器。
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, **kwargs)
                return future.result(timeout=timeout)

        except concurrent.futures.TimeoutError:
            last_error = f"超时(超过 {timeout} 秒)"
        except Exception as exc:  # noqa: BLE001 教学项目统一兜底
            last_error = f"{type(exc).__name__}: {exc}"

        _emit({"kind": "attempt_failed", "tool": name, "attempt": attempt, "error": last_error})

        # 还有重试机会 -> 指数退避后再试(1s, 2s, 4s...,最多 5s)
        if attempt <= retries:
            backoff = min(2 ** (attempt - 1), 5)
            time.sleep(backoff)

    # 走到这里说明所有尝试都失败了 -> 降级
    if fallback is not None:
        _emit({"kind": "fallback_used", "tool": name, "error": last_error})
        try:
            return fallback(**kwargs)
        except Exception as exc:  # noqa: BLE001 连兜底都失败,只能给文字说明
            return f"[工具失败,且降级也失败] {type(exc).__name__}: {exc}"

    _emit({"kind": "failed", "tool": name, "error": last_error})
    return f"[工具失败,已重试 {retries} 次] 最后错误:{last_error}"
