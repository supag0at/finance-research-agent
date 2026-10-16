"""采集器:挂在 LangGraph 的 config["callbacks"] 上,自动收集执行过程。

为什么用回调而不是自己去包装节点?
  1. **不用改 nodes.py 一行代码** —— 观测和业务彻底解耦;
  2. 回调的父子关系由框架维护,天然正确。如果自己手写 OTel span 去包节点,
     会踩"span 孤儿":OTel 把当前 span 存在 contextvars 里,而图里有环,
     调度器第二圈回到同一节点时是新的调用栈,上下文不会自动带过去,
     那一圈的 span 就脱离父节点变成一棵新树。

两个实测出来的陷阱(见 docs/design/M5.2-可观测性.md):

  陷阱1  条件边函数 should_continue 也会触发 on_chain_start,而且 metadata 里的
         langgraph_node 同样标着 'plan'。只按 langgraph_node 计数会让 plan 翻倍。
         区分靠 tag:真正的节点执行带 'graph:step:N',条件边带 'seq:step:N'。

  陷阱2  工具失败**不会**触发 on_tool_error —— core/reliability.py 把异常吞掉、
         返回错误字符串了。所以失败要靠两条路识别:
           a) 注册 reliability 的观察钩子(拿到重试/降级/最终失败);
           b) 检查工具返回是不是我们的失败标记。
"""

import threading
import time
import uuid
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from core import reliability
from core.failure import FAIL_PREFIX as _FAIL_PREFIX  # 失败标记的唯一来源
from memory import events as memory_events
from obs import logger
from obs.metrics import RunRecord


def _text_of(value: Any) -> str:
    """工具返回可能是 ToolMessage 对象,也可能是字符串,统一取文本。"""
    content = getattr(value, "content", value)
    return content if isinstance(content, str) else str(content)


class ObsHandler(BaseCallbackHandler):
    def __init__(
        self,
        question: str,
        impl: str = "langgraph",
        thread_id: str = "",
        budget: Any = None,
    ) -> None:
        self.record = RunRecord(
            run_id=str(uuid.uuid4()),
            question=question,
            impl=impl,
            thread_id=thread_id,
        )
        # M6:观测 -> 控制的闭环。预算自己不算钱(core 不能依赖 obs),
        # 由这里在每次 LLM 调用结束后把累计用量推过去。
        self.budget = budget
        # run_id -> 开始时间 / 名字(节点名或工具名)。
        # 工具并行执行时这两个字典会被多个线程访问,所以统一加锁。
        self._starts: dict[UUID, float] = {}
        self._names: dict[UUID, str] = {}
        self._lock = threading.Lock()
        self._t0 = 0.0

    # ---------- 生命周期 ----------

    def start(self) -> "ObsHandler":
        self._t0 = time.perf_counter()
        reliability.add_observer(self._on_reliability)
        memory_events.add_observer(self._on_memory)
        return self

    def finish(self, hit_limit: bool = False, budget: Any = None) -> RunRecord:
        reliability.remove_observer(self._on_reliability)
        memory_events.remove_observer(self._on_memory)
        self.record.total_ms = (time.perf_counter() - self._t0) * 1000
        self.record.hit_limit = hit_limit
        b = budget if budget is not None else self.budget
        if b is not None:
            self.record.budget = b.to_json()
            self.record.stop_reason = b.tripped_reason()
        logger.log_run(self.record.to_json())
        return self.record

    def __enter__(self) -> "ObsHandler":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        # 注意:撞上限属于异常路径,由调用方通过 finish(hit_limit=True) 显式告知,
        # 这里只兜住"忘了 finish"的情况。
        if self._t0 and not self.record.total_ms:
            self.finish()

    # ---------- 可靠性层的观察钩子(陷阱2 的 a 路) ----------

    def _on_memory(self, event: dict) -> None:
        """记忆事件 → RunRecord.memory 计数(M7)。事件 kind 见 memory/events.py。"""
        kind = str(event.get("kind", ""))
        if kind.startswith("ctx_"):
            self._on_ctx(kind, event)
            logger.log_event(kind, self.record.run_id, **{k: v for k, v in event.items() if k != "kind"})
            return
        name = {"memory_error": "errors", "memory_recalled": "recalled"}.get(kind, kind.removeprefix("memory_"))
        with self._lock:
            m = self.record.memory
            m[name] = m.get(name, 0) + 1
            if kind == "memory_recalled":
                m["recalled_items"] = m.get("recalled_items", 0) + int(event.get("count", 0))
                if event.get("injected"):
                    m["injected"] = m.get("injected", 0) + 1
        logger.log_event(kind, self.record.run_id, **{k: v for k, v in event.items() if k != "kind"})

    def _on_ctx(self, kind: str, e: dict) -> None:
        """压缩事件 → RunRecord.ctx(M8)。事件 kind 见 memory/context/compactor.py。"""

        def inc(key: str, n: float = 1) -> None:
            c[key] = c.get(key, 0) + n

        with self._lock:
            c = self.record.ctx
            if kind == "ctx_view":
                inc("views")
                c["max_est_tokens"] = max(c.get("max_est_tokens", 0), int(e.get("est_tokens", 0)))
            elif kind == "ctx_est_error":
                ratios = c.setdefault("est_error_ratios", [])
                ratios.append(e.get("ratio"))
                del ratios[:-20]
            elif kind == "ctx_compacted":
                inc("compactions")
                inc("archived_rounds", int(e.get("rounds", 0)))
                inc("facts_kept", int(e.get("facts_kept", 0)))
                inc("facts_dropped", int(e.get("facts_dropped", 0)))
                if e.get("fallback"):
                    inc("fallbacks")
            elif kind == "ctx_compaction_skipped":
                inc("skipped")
            elif kind == "ctx_readback":
                inc("readbacks" if e.get("found") else "readback_misses")
            elif kind == "ctx_summarizer_call":
                inc("summarizer_calls")
                inc("summarizer_ms", float(e.get("ms", 0)))
                inc("summarizer_in_tokens", int(e.get("in_tokens", 0)))
                inc("summarizer_out_tokens", int(e.get("out_tokens", 0)))

    def _on_reliability(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "succeeded":
            return  # 只给程序记忆当反证用,逐次成功不进观测日志(太吵)
        tool = event.get("tool", "unknown")
        with self._lock:
            if kind == "attempt_failed":
                self.record.add_reliability(tool, retry=1)
            elif kind == "fallback_used":
                self.record.add_reliability(tool, fallback=1)
            # M6 的三个新事件。retry_skipped 和 circuit_open 是"分类重试 + 熔断"
            # 真的在省钱的直接证据,所以要单独计数,不能混进 retry 里。
            elif kind == "retry_skipped":
                self.record.add_reliability(tool, retry_skipped=1)
            elif kind == "circuit_open":
                self.record.add_reliability(tool, circuit_open=1)
        logger.log_event(
            f"reliability_{kind}",
            self.record.run_id,
            tool=tool,
            error=event.get("error", "")[:200],
        )

    # ---------- 图节点 ----------

    def on_chain_start(
        self,
        serialized: Optional[dict],
        inputs: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        if not node:
            return  # 框架内部的 chain,不是图节点
        # 陷阱1:只认带 graph:step: 的那次,过滤掉条件边(seq:step:)
        if not any(str(t).startswith("graph:step:") for t in (tags or [])):
            return
        with self._lock:
            self._starts[run_id] = time.perf_counter()
            # on_chain_end 不带 metadata,所以节点名必须在这里就记下来
            self._names[run_id] = node

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        ms = self._pop(run_id)
        if ms is None:
            return
        name = self._pop_name(run_id)
        if name:
            with self._lock:
                self.record.add_node(name, ms)
            logger.log_event("node_end", self.record.run_id, node=name, ms=round(ms, 1))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        ms = self._pop(run_id)
        name = self._pop_name(run_id)
        if name and ms is not None:
            with self._lock:
                self.record.add_node(name, ms)
                self.record.add_error(f"node:{name}", f"{type(error).__name__}: {error}")

    # ---------- LLM ----------

    def on_chat_model_start(
        self,
        serialized: Optional[dict],
        messages: Any,
        *,
        run_id: UUID,
        metadata: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._starts[run_id] = time.perf_counter()
            requested = (metadata or {}).get("ls_model_name")
            if requested and not self.record.model_requested:
                self.record.model_requested = requested

    # 非 chat 模型走这个入口(我们目前用不到,但补上更完整)
    on_llm_start = on_chat_model_start

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        ms = self._pop(run_id) or 0.0
        prompt = completion = cached = reasoning = 0
        model_returned = ""

        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            usage = getattr(msg, "usage_metadata", None) if msg is not None else None
            if usage:
                prompt = usage.get("input_tokens", 0) or 0
                completion = usage.get("output_tokens", 0) or 0
                cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
                reasoning = (usage.get("output_token_details") or {}).get("reasoning", 0) or 0
            if msg is not None:
                model_returned = (msg.response_metadata or {}).get("model_name", "") or ""
        except Exception:  # noqa: BLE001 取不到用量不该影响主流程
            pass

        with self._lock:
            self.record.add_llm(
                ms=ms,
                prompt=prompt,
                completion=completion,
                cached=cached,
                reasoning=reasoning,
                model_returned=model_returned,
            )
            # M6:把累计成本/调用次数推给预算。放在锁内,保证推过去的是一致的快照。
            if self.budget is not None:
                self.budget.update(self.record.cost(), self.record.llm_calls)
        logger.log_event(
            "llm_end",
            self.record.run_id,
            ms=round(ms, 1),
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=completion,
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._pop(run_id)
        with self._lock:
            self.record.add_error("llm", f"{type(error).__name__}: {error}")
        logger.log_event("llm_error", self.record.run_id, error=str(error)[:200])

    # ---------- 工具 ----------

    def on_tool_start(
        self,
        serialized: Optional[dict],
        input_str: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or "unknown"
        with self._lock:
            self._starts[run_id] = time.perf_counter()
            self._names[run_id] = name
        logger.log_event(
            "tool_start", self.record.run_id, tool=name, args=logger.redact(input_str, 200)
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        ms = self._pop(run_id) or 0.0
        name = self._pop_name(run_id) or "unknown"
        text = _text_of(output)
        ok = not text.startswith(_FAIL_PREFIX)  # 陷阱2 的 b 路
        with self._lock:
            self.record.add_tool(name, ms, ok)
        logger.log_event(
            "tool_end" if ok else "tool_failed",
            self.record.run_id,
            tool=name,
            ms=round(ms, 1),
            ok=ok,
            output=logger.redact(text, 300),
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """实际上很少触发(reliability 把异常吞了),但留着以防有工具绕过它。"""
        ms = self._pop(run_id) or 0.0
        name = self._pop_name(run_id) or "unknown"
        with self._lock:
            self.record.add_tool(name, ms, ok=False)
            self.record.add_error(f"tool:{name}", f"{type(error).__name__}: {error}")
        logger.log_event("tool_error", self.record.run_id, tool=name, error=str(error)[:200])

    # ---------- 工具方法 ----------

    def _pop(self, run_id: UUID) -> Optional[float]:
        with self._lock:
            t = self._starts.pop(run_id, None)
        return None if t is None else (time.perf_counter() - t) * 1000

    def _pop_name(self, run_id: UUID) -> Optional[str]:
        with self._lock:
            return self._names.pop(run_id, None)
