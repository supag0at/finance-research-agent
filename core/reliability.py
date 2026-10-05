"""可靠性层:给"执行一个工具"这件事包上超时、重试、熔断、降级。

为什么需要它?工具要么联网、要么调外部 API,现实中一定会出问题:
网络抖动、对方超时、返回垃圾、限流……如果不管,Agent 会直接崩掉。
生产级系统的要求是:出错也别崩,能重试就重试,实在不行就"降级"给个兜底结果,
让整个流程继续走下去。

## M6 改了什么,为什么

改造前这里是"`except Exception` 然后无脑重试 N 次"。9-24 那轮评测证明代价很实在:
`fetch_page` 17 次调用失败 8 次,其中 403/401/404 共 18 次尝试是**确定性失败** ——
重试一百次还是同样结果,每次都白付一个 HTTP 往返加一秒退避。
而失败塞回给模型的是一句 `[工具失败,已重试 2 次] 最后错误:RequestError(...)`,
读起来像"再试一次说不定行",于是模型换个关键词继续试,`nvda_perf_01` 转了 10 圈、
花了正常题 13 倍的钱,最后交白卷。

所以 M6 在这里加了三件事,每件都对应一个实测到的问题:

  1. **先分类再决定重试**(core/failure.py)—— 确定性失败直接不重试;
  2. **熔断**(core/breaker.py)—— 连续失败就暂时别碰了,不要排队撞;
  3. **三段式错误反馈** —— 塞回 state 的是"错误原文 + 历次尝试 + 代码给的恢复建议",
     而不是一个裸字符串。业界叫 Context-Injected Error Recovery,
     实测能把重试循环减少 25~40%。

但要记住一个优先级:**重试和熔断都只是配角**。它们修的是基础设施抖动;
Agent 最贵的故障是"模型判断错误导致的绕圈子",那个只有硬边界能止损
(见 core/budget.py)。详见 docs/design/M6-可靠性强化.md。
"""

import concurrent.futures
import time
from typing import Any, Callable, Optional

from core import failure
from core.breaker import current_breaker
from core.failure import FAIL_PREFIX, Kind, classify, current_log, render, target_of

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
    idempotent: bool = True,
) -> Any:
    """带"超时 + 分类重试 + 熔断 + 降级"地执行一个函数。

    参数:
        func: 真正要执行的工具函数。
        kwargs: 传给它的参数(字典)。
        timeout: 单次执行的超时时间(秒)。超过就算这次失败。
        retries: **最多**再重试几次。注意是上限 —— 确定性失败(403/404)会
                 立刻停止,不会把这个额度用完。
        fallback: 全部失败后的兜底。可以是一个函数(会用同样的 kwargs 调用),
                  也可以为 None(那就返回三段式错误说明)。
        name: 工具名,只用于观测事件。**调用方应显式传入注册时的工具名**:
              如果退回 func.__name__,当函数名和注册名不一致时(包装、别名、
              测试打桩),观测数据会分裂成两个条目——重试记在函数名下、
              调用次数记在工具名下,对不上。这是实测踩出来的。
        idempotent: 这个工具重复执行是否安全。我们现有四个工具都是只读的,
              所以默认 True。**留这个参数是为了不给未来挖坑**:一旦加入写操作
              工具(下单、发邮件),传 False 就会禁止自动重试,而不是默默重复执行。
              见 docs/design/M6-可靠性强化.md §3.1。

    返回:
        成功时返回工具结果;
        失败且有 fallback 时返回兜底结果;
        失败且无 fallback 时返回**三段式错误文本**(而不是抛异常、让 Agent 崩)。
    """
    name = name or getattr(func, "__name__", "unknown")
    breaker = current_breaker()
    log = current_log()
    target = target_of(kwargs)

    # ---- 熔断检查:这条路已经不通了,就别再付重试和退避的钱 ----
    allowed, deny_reason = breaker.allow(name)
    if not allowed:
        _emit({"kind": "circuit_open", "tool": name, "error": deny_reason})
        return render(
            tool=name,
            kwargs=kwargs,
            kind=Kind.TERMINAL,  # 对模型而言熔断就是"现在走不通",等同确定性失败
            error=deny_reason,
            attempts=0,
            log=log,
        )

    last_error = "未知错误"
    last_kind = Kind.RETRYABLE
    attempt = 0

    for attempt in range(1, retries + 2):  # 第 1 次 + 最多 retries 次重试
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
                result = future.result(timeout=timeout)
            breaker.on_success(name)
            return result

        except concurrent.futures.TimeoutError:
            # 超时是最微妙的一类:请求可能已经到达对方并生效了,我们只是没等到回答。
            # 对只读工具无所谓;对写操作就是 SIDE_EFFECT_UNKNOWN,绝不能重试。
            last_error = f"超时(超过 {timeout} 秒)"
            last_kind = Kind.RETRYABLE if idempotent else Kind.SIDE_EFFECT_UNKNOWN
        except Exception as exc:  # noqa: BLE001 教学项目统一兜底
            last_error = f"{type(exc).__name__}: {exc}"
            last_kind = classify(exc)

        log.record(name, target, last_kind, last_error)
        _emit(
            {
                "kind": "attempt_failed",
                "tool": name,
                "attempt": attempt,
                "error": last_error,
                "failure_kind": last_kind.value,
            }
        )

        # ---- 关键变化:先问"这类失败重试了有用吗" ----
        if not failure.should_retry(last_kind, idempotent=idempotent):
            _emit(
                {
                    "kind": "retry_skipped",
                    "tool": name,
                    "error": last_error,
                    "failure_kind": last_kind.value,
                }
            )
            break

        if attempt <= retries:
            time.sleep(failure.backoff_seconds(last_kind, attempt))

    # ---- 所有(值得做的)尝试都失败了 ----
    if breaker.on_failure(name):
        _emit({"kind": "circuit_opened", "tool": name, "error": last_error})

    if fallback is not None:
        _emit({"kind": "fallback_used", "tool": name, "error": last_error})
        try:
            return fallback(**kwargs)
        except Exception as exc:  # noqa: BLE001 连兜底都失败,只能给文字说明
            return f"{FAIL_PREFIX},且降级也失败] {type(exc).__name__}: {exc}"

    _emit({"kind": "failed", "tool": name, "error": last_error, "failure_kind": last_kind.value})
    return render(
        tool=name,
        kwargs=kwargs,
        kind=last_kind,
        error=last_error,
        attempts=attempt,
        log=log,
    )
