"""失败分类 + 恢复建议 + 三段式错误反馈(M6)。

## 为什么单独一个文件

改造前 `core/reliability.py` 只做一件事:`except Exception` 然后重试 N 次。
它把**所有失败一视同仁**,而 9-24 那轮评测证明这代价很实在:
`fetch_page` 17 次调用失败 8 次,拆开是 403 ×9、401 ×6、404 ×3、ConnectTimeout ×6 ——
**前 18 次是确定性失败**,重试一百次还是同样的结果,每一次都白付一个 HTTP 往返加一秒退避。

所以要先回答"这个失败重试了有用吗",才谈得上重试策略。本文件就负责这个判断,
以及判断之后"该告诉模型什么"。

## 四类失败(不是两类)

业界常见分法是 retryable / terminal 两类,但那漏了最危险的一种。
我们按 agent-arch 的四分法:

    TERMINAL        确定性失败(403/404/400)。重试无意义,应换路子。
    RETRYABLE       瞬时失败(超时/连接错误/5xx)。退避重试有意义。
    RATE_LIMITED    被限流(429)。要退避,但退避要更久。
    SIDE_EFFECT_UNKNOWN
                    **请求已发出但结果未知**(写操作超时)。最危险的一类:
                    盲目重试可能造成重复副作用,放弃又可能漏掉已生效的操作。
                    正确做法是"先确认再决定",而不是重试。

第四类对我们当前这套**全只读**的工具用不上(重试天然安全),但分类里保留它,
因为它决定了 `run_reliably` 该不该自动重试 —— 哪天加一个写操作工具,
这个位置已经预留好了,不会默默重复下单。详见 docs/design/M6-可靠性强化.md §3.1。

## 三段式错误反馈

失败时塞回给模型的不能是一个裸字符串。业界模式叫 Context-Injected Error Recovery,
实测能把重试循环减少 25~40%,要素是三段:

    1. 错误原文      —— 逐字保留,不加工
    2. 历次尝试      —— 本轮对同一 (工具, 目标) 的所有失败记录
    3. 定向恢复建议  —— 按失败类型给出,**由代码生成,不是让模型自己想**

第 3 点是关键。Google Cloud 的 self-healing 模式说得很直白:建议来自你的代码,
所以每次运行都一致、可以写进评测断言;让模型自己想恢复办法,
"LLM improvisation in error paths is one of the most common sources of
unpredictable agent behavior"。

改造前我们返回的是:

    [工具失败,已重试 2 次] 最后错误:DDGSException: RequestError(...)

"已重试 2 次"加一串 RequestError 读起来像"网络抽了,再试一次说不定行",
于是模型换个关键词又调一次,`nvda_perf_01` 就这样转了 10 圈、花了正常题 13 倍的钱。
"""

import threading
from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    """失败类型。值用中文,因为它会出现在给模型看的文本里。"""

    TERMINAL = "确定性失败"
    RETRYABLE = "瞬时失败"
    RATE_LIMITED = "被限流"
    SIDE_EFFECT_UNKNOWN = "结果未知"


# 失败标记前缀。obs/callbacks.py 靠它判断工具是否失败(回调机制拿不到 on_tool_error,
# 因为 run_reliably 把异常吞掉了),**改这里必须同步改那边**。
FAIL_PREFIX = "[工具失败"

# 按类型给的恢复建议。措辞是刻意设计的:
#   - TERMINAL 必须明说"重试无效",否则模型会不停换参数重试(实测如此);
#   - 每条都给出**可执行的下一步**,而不只是描述问题;
#   - 都提到"基于已有信息作答"这个出口,因为原来的 prompt 根本没给模型留退路。
_ADVICE: dict[Kind, str] = {
    Kind.TERMINAL: (
        "这是确定性失败,**重试和换措辞都不会有用**(目标站点拒绝访问或资源不存在)。"
        "请改用其他工具或其他信息源;若已有足够信息,请直接基于现有信息作答,"
        "并说明这部分数据未能获取。"
    ),
    Kind.RETRYABLE: (
        "这是瞬时故障,已自动重试过并仍然失败。**不要再用同样的参数调用同一工具**。"
        "可以换一个工具或换一个明显不同的查询方向;若信息已够,请基于现有信息作答。"
    ),
    Kind.RATE_LIMITED: (
        "被对方限流,已退避重试仍失败。短时间内继续调用只会继续被拒。"
        "请改用其他工具,或基于现有信息作答。"
    ),
    Kind.SIDE_EFFECT_UNKNOWN: (
        "请求可能已经发出但结果未知,**绝不要重试**(可能造成重复操作)。"
        "请先用查询类工具确认当前状态,再决定下一步。"
    ),
}

# HTTP 状态码 -> 类型。只列我们实际遇到过和明确有定论的。
_STATUS_KIND: dict[int, Kind] = {
    400: Kind.TERMINAL,  # 参数错误
    401: Kind.TERMINAL,  # 未授权(Reuters 实际返回这个)
    403: Kind.TERMINAL,  # 拒绝访问(反爬,最常见)
    404: Kind.TERMINAL,  # 资源不存在
    405: Kind.TERMINAL,
    410: Kind.TERMINAL,
    422: Kind.TERMINAL,
    429: Kind.RATE_LIMITED,
}

# 异常类名里出现这些字样 -> 瞬时失败。用类名而不是 isinstance,
# 是为了不让 core/ 依赖 httpx/ddgs 这些具体库。
_TRANSIENT_HINTS = (
    "timeout",
    "connect",
    "connection",
    "readerror",
    "remoteprotocol",
    "proxy",
    "ssl",
    "dns",
    "temporar",
)


def classify(exc: BaseException | None, text: str = "") -> Kind:
    """判断一次失败属于哪一类。

    exc 为 None 表示"超时但没有异常对象"(run_reliably 的线程超时就是这种),
    此时只能按 text 判断。

    诚实的局限:这是启发式。我们的工具会把底层异常包成 RuntimeError
    (例如 `get_stock_price` 两个主机都失败后抛 `RuntimeError("Yahoo chart 请求失败:...")`),
    状态码只剩在消息文本里,所以必须兼顾"看对象"和"看文本"两条路。
    """
    blob = f"{type(exc).__name__ if exc else ''} {exc or ''} {text}".lower()

    # ① 优先从异常对象上直接取状态码(httpx.HTTPStatusError 有 .response)
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int) and code in _STATUS_KIND:
        return _STATUS_KIND[code]
    if isinstance(code, int) and 500 <= code < 600:
        return Kind.RETRYABLE  # 服务端错误,重试有意义

    # ② 退回从文本里找状态码。只认我们认识的那几个,避免把正文里的数字误判成状态码。
    for status, kind in _STATUS_KIND.items():
        if f"'{status} " in blob or f" {status} " in blob or f"status {status}" in blob:
            return kind

    # ③ 限流的文字形式
    if "rate limit" in blob or "too many requests" in blob or "ratelimit" in blob:
        return Kind.RATE_LIMITED

    # ④ 瞬时故障的文字形式
    if any(h in blob for h in _TRANSIENT_HINTS):
        return Kind.RETRYABLE

    # ⑤ 兜底按瞬时处理(允许重试)。
    # 为什么默认偏"可重试"而不是"确定性失败"?因为误判成 TERMINAL 会让本来能成功的
    # 调用直接放弃,代价是功能损失;误判成 RETRYABLE 只是多花几秒。宁可多试。
    return Kind.RETRYABLE


def should_retry(kind: Kind, idempotent: bool = True) -> bool:
    """这类失败值不值得重试。

    idempotent=False(工具有副作用)时,除了明确的限流,一律不自动重试 ——
    重复执行的风险高于失败的代价。这是给未来的写操作工具预留的闸。
    """
    if kind in (Kind.TERMINAL, Kind.SIDE_EFFECT_UNKNOWN):
        return False
    if not idempotent:
        return kind is Kind.RATE_LIMITED
    return True


def backoff_seconds(kind: Kind, attempt: int) -> float:
    """退避时长。限流要等更久,其余指数退避且有上界。"""
    if kind is Kind.RATE_LIMITED:
        return min(5.0 * attempt, 15.0)
    return float(min(2 ** (attempt - 1), 5))


# --------------------------------------------------------------------------
# 本轮失败记录(三段式里的"历次尝试") 这些全是用来构造失败的。
# --------------------------------------------------------------------------


@dataclass
class _Entry:
    kind: Kind
    error: str
    count: int = 1


@dataclass
class FailureLog:
    """记录本轮运行里每个 (工具, 目标) 失败过几次、怎么失败的。

    "目标"是从参数里挑出来的那个**标识性字段**(url / query / symbol),
    而不是整个参数字典 —— 因为 `top_k` 这类无关参数变了不该算成"换了个目标"。
    """

    entries: dict[tuple[str, str], _Entry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, tool: str, target: str, kind: Kind, error: str) -> int:
        """登记一次失败,返回这个 (工具, 目标) 累计失败次数。"""
        key = (tool, target)
        with self._lock:
            e = self.entries.get(key)
            if e is None:
                self.entries[key] = _Entry(kind=kind, error=error)
                return 1
            e.count += 1
            e.kind = kind
            e.error = error
            return e.count

    def history_for(self, tool: str, exclude_target: str = "") -> list[str]:
        """本轮该工具**其他目标**上的失败历史,给"历次尝试"那一段用。

        为什么排除当前目标?当前这次失败的详情已经在"错误原文"里了,
        重复一遍只是浪费 token。这里要告诉模型的是"你还试过别的,也都不行"。
        """
        with self._lock:
            return [
                f"{target or '(无参数)'} → {e.kind.value}"
                + (f" ×{e.count}" if e.count > 1 else "")
                for (t, target), e in self.entries.items()
                if t == tool and target != exclude_target
            ]

    def reset(self) -> None:
        with self._lock:
            self.entries.clear()


# 模块级单例。
#
# 局限(和 reliability._observers 同一个):它不是按调用上下文隔离的。我们的运行是
# **串行**的(一次只跑一个 agent.run,评测也是逐题跑),所以够用。真要并发跑多个
# Agent,得改成 contextvars。这个局限写在这里,是因为它比"悄悄出错"好。
_log = FailureLog()


def current_log() -> FailureLog:
    return _log


# 从参数里挑"目标"字段的优先级。顺序即优先级。
_TARGET_KEYS = ("url", "query", "symbol", "q", "keyword")


def target_of(kwargs: dict) -> str:
    """从工具参数里提取标识性字段,作为"同一个目标"的判据。"""
    for k in _TARGET_KEYS:
        v = kwargs.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:200]
    # 没有已知的标识字段就退回整个参数的紧凑表示
    return ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))[:200] if kwargs else ""


def render(
    tool: str,
    kwargs: dict,
    kind: Kind,
    error: str,
    attempts: int,
    log: FailureLog | None = None,
) -> str:
    """拼出三段式错误文本,这就是最终塞回 state 给模型看的内容。

    格式刻意做成带小标题的分段:模型对结构化文本的遵循度明显好于一段流水话,
    而且**我们自己在 trace 里读也更快**。
    """
    log = log or current_log()
    target = target_of(kwargs)
    lines = [
        f"{FAIL_PREFIX}:{tool}] {kind.value}",
        f"【错误原文】{error}",
    ]

    same = log.entries.get((tool, target))
    if same and same.count > 1:
        lines.append(f"【同一目标】本轮已失败 {same.count} 次(目标:{target})")
    elif attempts > 1:
        lines.append(f"【本次尝试】已重试 {attempts - 1} 次")

    others = log.history_for(tool, exclude_target=target)
    if others:
        shown = others[:5]
        more = f" 等 {len(others)} 项" if len(others) > len(shown) else ""
        lines.append(f"【该工具本轮其他失败】{'; '.join(shown)}{more}")

    lines.append(f"【建议】{_ADVICE[kind]}")
    return "\n".join(lines)
