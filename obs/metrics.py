"""指标(Metrics):把一次运行里发生的事聚合成可比较的数值。

为什么要单独一层?因为日志是"一堆事件",没法直接回答"这次比上次贵了还是便宜了"。
指标就是把事件压成几个能横向比较的数,这样才能盯趋势、做回归对比。

成本这块有两个坑,都在 docs/design/M5.2-可观测性.md 的实测记录里:
  1. 我们请求的模型名是 deepseek-v4-flash,但响应里返回的是 deepseek-flash。
     所以"按模型名去查价格表"这条路不可靠 —— 单价我们自己配。
  2. prompt token 里有一大部分是**缓存命中**(实测 691 里有 512 命中),
     缓存命中的单价远低于未命中。不区分就会显著高估成本。
"""

import os
from dataclasses import dataclass, field

# 单价:元 / 百万 token。**必须从模型厂商官网抄**,不要猜。
# 未配置时按 0 计算,报告里会显式标成"未配置单价",而不是给一个假数字 ——
# 假的成本比没有成本更糟。
PRICE_IN = float(os.getenv("LLM_PRICE_IN", "0") or 0)  # 输入,未命中缓存
PRICE_CACHE_IN = float(os.getenv("LLM_PRICE_CACHE_IN", "0") or 0)  # 输入,命中缓存
PRICE_OUT = float(os.getenv("LLM_PRICE_OUT", "0") or 0)  # 输出


@dataclass
class NodeStat:
    """一个图节点的统计。count 会大于 1,因为图里有环,同一节点会跑多次。"""

    count: int = 0
    total_ms: float = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms


@dataclass
class ToolStat:
    """一个工具的统计。

    fail 统计的是"最终失败"(重试完还是不行);retry / fallback 来自
    core/reliability.py 的观察钩子 —— 这两个数**回调机制拿不到**,
    因为 run_reliably 把异常吞掉了,LangChain 根本不知道发生过失败。
    """

    count: int = 0
    fail: int = 0
    retry: int = 0
    fallback: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    # M6 新增的两个数,它们是"分类重试 + 熔断"是否真的在起作用的直接证据:
    #   retry_skipped 省掉的重试次数(确定性失败,重试无意义)
    #   circuit_open  被熔断直接拒绝的调用次数(连外部请求都没发)
    retry_skipped: int = 0
    circuit_open: int = 0

    def add(self, ms: float, ok: bool) -> None:
        self.count += 1
        if not ok:
            self.fail += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)


@dataclass
class RunRecord:
    """一次完整调研的汇总。"""

    run_id: str
    question: str
    impl: str = "langgraph"
    thread_id: str = ""

    total_ms: float = 0.0
    hit_limit: bool = False  # 是否撞到 recursion_limit(不收敛)
    # M6:预算快照与提前收尾的原因。
    # hit_limit 和 stop_reason 的区别值得记住:
    #   stop_reason 非空 = **我们自己的**边界生效了(走了 finalize,有答案);
    #   hit_limit=True  = 连我们的边界都没拦住,被框架硬停(没有答案)。
    # 改造后第二种应该几乎不再出现 —— 如果还出现,说明预算设得比 recursion_limit 还松。
    stop_reason: str = ""
    budget: dict = field(default_factory=dict)

    llm_calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0  # prompt 中命中缓存的部分(已含在 prompt_tokens 里)
    completion_tokens: int = 0
    reasoning_tokens: int = 0  # 思考模式产生的 token(已含在 completion 里)
    llm_ms: float = 0.0
    model_requested: str = ""  # 我们请求的模型名
    model_returned: str = ""  # 响应里实际返回的模型名(可能不同!)

    nodes: dict[str, NodeStat] = field(default_factory=dict)
    tools: dict[str, ToolStat] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    # ---------- 采集入口 ----------

    def add_node(self, name: str, ms: float) -> None:
        self.nodes.setdefault(name, NodeStat()).add(ms)

    def add_llm(
        self,
        ms: float,
        prompt: int,
        completion: int,
        cached: int = 0,
        reasoning: int = 0,
        model_returned: str = "",
    ) -> None:
        self.llm_calls += 1
        self.llm_ms += ms
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached
        self.reasoning_tokens += reasoning
        if model_returned:
            self.model_returned = model_returned

    def add_tool(self, name: str, ms: float, ok: bool) -> None:
        self.tools.setdefault(name, ToolStat()).add(ms, ok)

    def add_reliability(
        self,
        name: str,
        retry: int = 0,
        fallback: int = 0,
        retry_skipped: int = 0,
        circuit_open: int = 0,
    ) -> None:
        st = self.tools.setdefault(name, ToolStat())
        st.retry += retry
        st.fallback += fallback
        st.retry_skipped += retry_skipped
        st.circuit_open += circuit_open

    def add_error(self, where: str, message: str) -> None:
        self.errors.append({"where": where, "message": message[:500]})

    # ---------- 派生指标 ----------

    @property
    def tool_calls(self) -> int:
        return sum(s.count for s in self.tools.values())

    @property
    def tool_fails(self) -> int:
        return sum(s.fail for s in self.tools.values())

    @property
    def tool_retries(self) -> int:
        return sum(s.retry for s in self.tools.values())

    @property
    def retries_skipped(self) -> int:
        """因为"确定性失败,重试无意义"而省掉的重试次数(M6)。"""
        return sum(s.retry_skipped for s in self.tools.values())

    @property
    def circuit_opens(self) -> int:
        """被熔断拦住、连请求都没发出去的次数(M6)。"""
        return sum(s.circuit_open for s in self.tools.values())

    @property
    def tool_ms(self) -> float:
        """工具节点的墙钟耗时无法由此相加得到 —— 工具是**并行**执行的,
        相加会超过真实耗时。所以这里给的是"工具时间总和(串行等价)",
        真实墙钟耗时看 nodes['act'].total_ms。"""
        return sum(s.total_ms for s in self.tools.values())

    def cost(self) -> float | None:
        """估算成本(元)。未配置单价时返回 None,而不是 0 —— 区分"免费"和"不知道"。"""
        if not (PRICE_IN or PRICE_CACHE_IN or PRICE_OUT):
            return None
        uncached = max(self.prompt_tokens - self.cached_tokens, 0)
        return (
            uncached * PRICE_IN
            + self.cached_tokens * PRICE_CACHE_IN
            + self.completion_tokens * PRICE_OUT
        ) / 1_000_000

    # ---------- 输出 ----------

    def summary_line(self) -> str:
        """一行摘要,跑完直接打印。设计目标:一眼看出慢在哪、贵不贵、有没有异常。"""
        c = self.cost()
        cost_text = f"¥{c:.4f}" if c is not None else "¥未配单价"

        # 节点耗时按降序取前三,够定位瓶颈了
        top = sorted(self.nodes.items(), key=lambda kv: -kv[1].total_ms)[:3]
        node_text = " ".join(f"{n}={s.total_ms / 1000:.1f}s" for n, s in top)

        flags = []
        if self.hit_limit:
            flags.append("撞框架硬上限")
        if self.stop_reason:
            flags.append(f"提前收尾({self.stop_reason})")
        if self.tool_fails:
            flags.append(f"工具失败{self.tool_fails}")
        if self.tool_retries:
            flags.append(f"重试{self.tool_retries}")
        if self.circuit_opens:
            flags.append(f"熔断{self.circuit_opens}")
        if self.errors:
            flags.append(f"异常{len(self.errors)}")
        flag_text = f" ⚠️ {'/'.join(flags)}" if flags else ""

        return (
            f"[obs] run={self.run_id[:8]} {self.total_ms / 1000:.1f}s | "
            f"LLM {self.llm_calls}次 "
            f"(in {self.prompt_tokens}/缓存{self.cached_tokens}, out {self.completion_tokens}) "
            f"{cost_text} | 工具 {self.tool_calls}次 | {node_text}{flag_text}"
        )

    def to_json(self) -> dict:
        c = self.cost()
        return {
            "run_id": self.run_id,
            "impl": self.impl,
            "thread_id": self.thread_id,
            "question": self.question,
            "total_ms": round(self.total_ms, 1),
            "hit_limit": self.hit_limit,
            "stop_reason": self.stop_reason or None,
            "budget": self.budget or None,
            "llm": {
                "calls": self.llm_calls,
                "ms": round(self.llm_ms, 1),
                "prompt_tokens": self.prompt_tokens,
                "cached_tokens": self.cached_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "model_requested": self.model_requested,
                "model_returned": self.model_returned,
            },
            "cost_cny": round(c, 6) if c is not None else None,
            "nodes": {
                n: {"count": s.count, "total_ms": round(s.total_ms, 1)}
                for n, s in self.nodes.items()
            },
            "tools": {
                n: {
                    "count": s.count,
                    "fail": s.fail,
                    "retry": s.retry,
                    "fallback": s.fallback,
                    "retry_skipped": s.retry_skipped,
                    "circuit_open": s.circuit_open,
                    "total_ms": round(s.total_ms, 1),
                    "max_ms": round(s.max_ms, 1),
                }
                for n, s in self.tools.items()
            },
            "errors": self.errors,
        }
