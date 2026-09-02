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


def run_reliably(
    func: Callable,
    kwargs: dict,
    timeout: float = 15.0,
    retries: int = 2,
    fallback: Optional[Callable] = None,
) -> Any:
    """带"超时 + 重试 + 降级"地执行一个函数。

    参数:
        func: 真正要执行的工具函数。
        kwargs: 传给它的参数(字典)。
        timeout: 单次执行的超时时间(秒)。超过就算这次失败。
        retries: 失败后最多再重试几次(总尝试次数 = 1 + retries)。
        fallback: 全部失败后的兜底。可以是一个函数(会用同样的 kwargs 调用),
                  也可以为 None(那就返回一段错误说明)。

    返回:
        成功时返回工具结果;失败且有 fallback 时返回兜底结果;
        失败且无 fallback 时返回一段可读的错误字符串(而不是抛异常、让 Agent 崩)。
    """
    last_error = "未知错误"

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

        # 还有重试机会 -> 指数退避后再试(1s, 2s, 4s...,最多 5s)
        if attempt <= retries:
            backoff = min(2 ** (attempt - 1), 5)
            time.sleep(backoff)

    # 走到这里说明所有尝试都失败了 -> 降级
    if fallback is not None:
        try:
            return fallback(**kwargs)
        except Exception as exc:  # noqa: BLE001 连兜底都失败,只能给文字说明
            return f"[工具失败,且降级也失败] {type(exc).__name__}: {exc}"

    return f"[工具失败,已重试 {retries} 次] 最后错误:{last_error}"
