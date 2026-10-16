"""compress v2 的编排:每圈 plan 之前跑一次,产出 {ctx, llm_input}。

绝大多数圈走"估算 ≤ 高水位"这条路:只做确定性的失败抽取和视图渲染,0 次 LLM。
只有越过高水位的那一圈才归档,而且归档器只看**新溢出的圈** ——
单次压缩的成本只取决于溢出量,与总历史长度无关。

压缩用高低水位(越过高水位才压,一次压到低水位以下)而不是"每圈压到刚好够":
每压一次都会让归档块之后的前缀缓存失效一次,压缩频率本身就是成本。
"""

import time
from typing import Optional

from memory import events
from memory.context.archive import archive_rounds, fallback_entries, tool_sources
from memory.context.config import ContextConfig
from memory.context.facts import merge_facts, verify_facts
from memory.context.failures import extract_failures
from memory.context.rounds import resolve_turns, split_rounds
from memory.context.tokens import estimate_tokens, find_anchor, message_chars, text_of, usage_tokens
from memory.context.types import CtxState, empty_ctx
from memory.context.view import build_view, render_message, render_pinned


def _copy_ctx(ctx: dict) -> CtxState:
    """state 里的 ctx 不能就地改:checkpointer 的上一张快照还引用着它。"""
    return {
        **ctx,
        "archive": list(ctx.get("archive") or []),
        "facts": list(ctx.get("facts") or []),
        "failures": list(ctx.get("failures") or []),
    }  # type: ignore[return-value]


def _current_ctx(state: dict, turn: int, body: int) -> CtxState:
    ctx = state.get("ctx")
    if not ctx or ctx.get("turn") != turn or int(ctx.get("upto", -1)) < body:
        return empty_ctx(turn, body)
    return _copy_ctx(ctx)


def _pinned_chars(ctx: CtxState) -> int:
    p = render_pinned(ctx["facts"], ctx["failures"])
    return len(text_of(p.content)) if p is not None else 0


def compress_step(state: dict, llm, cfg: Optional[ContextConfig] = None) -> dict:
    cfg = cfg or ContextConfig.from_env()
    messages = state.get("messages") or []
    turns = resolve_turns(state)
    turn, body = len(turns) - 1, turns[-1]["body"]

    ctx = _current_ctx(state, turn, body)
    ctx["failures"] = extract_failures(messages, body, cfg.max_failures)
    view = build_view(messages, turns, ctx, cfg.preview_chars)

    # 锚点必须在本轮内:上一轮的 input_tokens 对应的是还没折叠的旧视图,拿来当基数会高估。
    anchor_idx, anchor = find_anchor(messages)
    if anchor_idx < body:
        anchor_idx, anchor = -1, None
    new_since = [render_message(m, cfg.preview_chars) for m in messages[anchor_idx + 1:]] if anchor else []
    est = estimate_tokens(view, anchor, new_since, _pinned_chars(ctx), cfg.tokens_per_char,
                          cfg.tools_schema_tokens)

    if anchor is not None and ctx["last_est"]:
        actual = int((anchor.usage_metadata or {}).get("input_tokens", 0))
        events.emit("ctx_est_error", est=ctx["last_est"], actual=actual,
                    ratio=round(actual / ctx["last_est"], 3))

    if est <= cfg.high_tokens:
        ctx["last_est"] = est
        events.emit("ctx_view", est_tokens=est, view_msgs=len(view), anchored=anchor is not None)
        return {"ctx": ctx, "llm_input": view}

    # ---- 越过高水位:从最旧的未归档圈开始吞,直到预计低于低水位 ----
    candidates = [r for r in split_rounds(messages, body) if r.start >= ctx["upto"]]
    spillable = candidates[: -cfg.keep_rounds] if len(candidates) > cfg.keep_rounds else []
    if not spillable:
        ctx["last_est"] = est
        events.emit("ctx_compaction_skipped", reason="没有可溢出的圈", est_tokens=est,
                    rounds=len(candidates))
        events.emit("ctx_view", est_tokens=est, view_msgs=len(view), anchored=anchor is not None)
        return {"ctx": ctx, "llm_input": view}

    chosen, est_after, chars = [], est, 0
    for r in spillable:
        c = sum(message_chars(render_message(m, cfg.preview_chars)) for m in messages[r.start:r.end])
        chars += c
        est_after = est_after - int(c * cfg.tokens_per_char) + cfg.entry_tokens
        chosen.append(r)
        if est_after <= cfg.low_tokens:
            break

    t0 = time.perf_counter()
    fallback, error = False, ""
    try:
        if llm is None:
            raise RuntimeError("未配置归档模型")
        entries, raw_facts = archive_rounds(llm, chosen, state.get("question", ""), cfg.archive_max_tool_chars)
    except Exception as exc:  # noqa: BLE001
        # 归档失败不能阻断调研:降级成只有调用记录和 id 的条目,照常前移 upto,
        # 否则视图会一直卡在高水位之上,每圈都重试一次失败的归档。
        entries, raw_facts = fallback_entries(chosen), []
        fallback, error = True, f"{type(exc).__name__}: {exc}"[:200]
    ms = (time.perf_counter() - t0) * 1000

    kept, dropped = verify_facts(raw_facts, tool_sources(chosen))
    for item, reason in dropped:
        events.emit("ctx_fact_dropped", reason=reason, entity=str(item.get("entity", ""))[:40],
                    metric=str(item.get("metric", ""))[:40], value=str(item.get("value", ""))[:40])

    ctx["archive"] = ctx["archive"] + entries
    ctx["facts"] = merge_facts(ctx["facts"], kept, cfg.max_facts)
    ctx["upto"] = chosen[-1].end
    ctx["compactions"] += 1
    ctx["last_est"] = est_after

    in_tok, out_tok = (0, 0) if fallback else usage_tokens(getattr(llm, "last_usage", None))
    events.emit(
        "ctx_compacted",
        rounds=len(chosen), chars=chars, ms=round(ms, 1), entries=len(entries),
        facts_kept=len(kept), facts_dropped=len(dropped), fallback=fallback, error=error,
        in_tokens=in_tok, out_tokens=out_tok, est_before=est, est_after=est_after,
    )
    if len(ctx["archive"]) > 30:
        events.emit("ctx_archive_long", entries=len(ctx["archive"]))

    view = build_view(messages, turns, ctx, cfg.preview_chars)
    events.emit("ctx_view", est_tokens=est_after, view_msgs=len(view), anchored=anchor is not None)
    return {"ctx": ctx, "llm_input": view}
