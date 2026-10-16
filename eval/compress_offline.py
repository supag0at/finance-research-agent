"""上下文压缩 v2 的离线回归(M8)—— 写法照 eval/faults.py。

不联网、不调模型、不花钱。归档器用脚本化的 FakeLLM 代替,水位直接通过 cfg 传低值。
场景编号与 docs/design/M8-上下文压缩.md §4.1 一致。

    python -m eval.compress_offline

全部通过则退出码 0。
"""

import re
import sys

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.failure import Kind, render
from memory import events
from memory.context import ContextConfig, compress_step
from memory.context.archive import FALLBACK_GIST
from memory.context.failures import extract_failures
from memory.context.readback import read_tool_result
from memory.context.rounds import head_slice, resolve_turns, split_rounds
from memory.context.types import empty_ctx
from memory.context.view import ARCHIVE_HEADER, PINNED_HEADER, PREVIOUS_TURNS_HEADER, build_view

PREVIEW = 1500

# 离线测试用"1 字 = 1 token、没有工具 schema 开销"的口径,水位直接按字数设,好算。
CFG = ContextConfig(high_tokens=2600, low_tokens=1300, keep_rounds=2, tokens_per_char=1.0,
                    tools_schema_tokens=0, entry_tokens=30)


class FakeLLM:
    """脚本化的归档器:为输入里出现的每一圈返回一条要点;facts 由调用方给定。"""

    def __init__(self, facts=None, fail: bool = False) -> None:
        self.facts = facts or []
        self.fail = fail
        self.calls: list[str] = []  # 每次调用的 user 内容
        self.last_usage = None

    def complete_json(self, messages: list[dict]) -> dict:
        user = messages[-1]["content"]
        self.calls.append(user)
        if self.fail:
            raise RuntimeError("模拟的归档失败")
        self.last_usage = {"prompt_tokens": len(user), "completion_tokens": 50}
        rounds = [int(n) for n in re.findall(r"^### R(\d+)$", user, flags=re.M)]
        facts = self.facts if len(self.calls) == 1 else []  # 事实只随第一次归档给出
        return {"entries": [{"round": n, "gist": f"要点R{n}"} for n in rounds], "facts": facts}

    def rounds_seen(self, k: int) -> list[int]:
        return [int(n) for n in re.findall(r"^### R(\d+)$", self.calls[k], flags=re.M)]


class _Events:
    """临时收集 memory/events 的事件。"""

    def __enter__(self):
        self.items: list[dict] = []
        events.add_observer(self.items.append)
        return self

    def __exit__(self, *exc):
        events.remove_observer(self.items.append)

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.items if e["kind"] == kind]

_PASS = "  [通过]"
_FAIL = "  [失败]"
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{_PASS if ok else _FAIL} {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


# --------------------------------------------------------------------------
# 构造消息的帮助函数
# --------------------------------------------------------------------------


def _round(no: int, tool: str, args: dict, content: str, n_tools: int = 1) -> list:
    """一圈:一条带 tool_calls 的 AIMessage + 它的 ToolMessage。id 形如 call_{no}_{k}。"""
    calls = [{"name": tool, "args": args, "id": f"call_{no}_{k}"} for k in range(n_tools)]
    out: list = [AIMessage(content="", tool_calls=calls)]
    for k in range(n_tools):
        out.append(ToolMessage(content=content, tool_call_id=f"call_{no}_{k}", name=tool))
    return out


def _fail(tool: str, args: dict, kind: Kind = Kind.TERMINAL, error: str = "403 Forbidden") -> str:
    return render(tool, args, kind, error, attempts=1)


def _session(question: str = "NVDA 本季毛利率?", rounds: list | None = None) -> dict:
    """单轮会话:system + 记忆块 + 问题 + 若干圈。返回带 turns 的 state。"""
    msgs: list = [
        SystemMessage(content="SYSTEM"),
        HumanMessage(content="【记忆】用户持有 100 股 NVDA"),
        HumanMessage(content=question),
    ]
    for r in rounds or []:
        msgs.extend(r)
    return {"messages": msgs, "question": question, "turns": [{"start": 0, "body": 3}]}


def _sig(m) -> tuple:
    """比较视图用的消息签名:类型 + 内容 + 配对 id。"""
    ids = tuple(tc["id"] for tc in getattr(m, "tool_calls", None) or [])
    return (type(m).__name__, str(m.content), getattr(m, "tool_call_id", ""), ids)


def _paired(view: list) -> bool:
    """不变式 3:每条 ToolMessage 之前都有携带其 id 的 AIMessage。"""
    seen: set[str] = set()
    for m in view:
        if isinstance(m, AIMessage):
            seen.update(tc["id"] for tc in m.tool_calls or [])
        elif isinstance(m, ToolMessage) and m.tool_call_id not in seen:
            return False
    return True


def _texts(view: list) -> str:
    return "\n".join(str(m.content) for m in view)


# --------------------------------------------------------------------------
# S3(Step 1 部分):切圈 —— 压缩的最小单位,永不拆圈
# --------------------------------------------------------------------------
def scenario_split_rounds() -> None:
    print("\nS3a · 切圈:一条 AI + 它的全部 Tool 是一圈")
    st = _session(
        rounds=[
            _round(1, "search_news", {"query": "nvda"}, "r1", n_tools=3),
            _round(2, "fetch_page", {"url": "http://a"}, "r2"),
            [AIMessage(content="最终答案")],
        ]
    )
    rs = split_rounds(st["messages"], 3)
    check("切出 3 圈", len(rs) == 3, f"实际 {len(rs)}")
    check("第 1 圈含 3 条 Tool", len(rs[0].tools) == 3 and rs[0].end - rs[0].start == 4)
    check("圈号从 1 起且连续", [r.no for r in rs] == [1, 2, 3])
    check("各圈首尾相接、覆盖 body 之后全部消息",
          rs[0].start == 3 and all(a.end == b.start for a, b in zip(rs, rs[1:])) and rs[-1].end == len(st["messages"]))
    check("最终答案单独成圈且没有 Tool", rs[2].ai is not None and not rs[2].tools)

    turns = resolve_turns({"messages": st["messages"]})
    check("没有 turns 时兜底:body = 最后一条 Human 之后", turns == [{"start": 0, "body": 3}], str(turns))
    check("首轮头部跳过 SystemMessage", head_slice(st["messages"], st["turns"]) == (1, 3))


# --------------------------------------------------------------------------
# S6(Step 1 部分):失败抽取是确定性的
# --------------------------------------------------------------------------
def scenario_extract_failures() -> None:
    print("\nS6a · 从 messages 确定性抽出失败记录")
    url = "https://reuters.com/x"
    st = _session(
        rounds=[
            _round(1, "fetch_page", {"url": url}, _fail("fetch_page", {"url": url})),
            _round(2, "search_news", {"query": "nvda"}, "正常结果"),
            _round(3, "fetch_page", {"url": url}, _fail("fetch_page", {"url": url})),
            _round(4, "search_news", {"query": "q2"},
                   _fail("search_news", {"query": "q2"}, Kind.RETRYABLE, "ConnectTimeout")),
        ]
    )
    fs = extract_failures(st["messages"], 3, limit=10)
    check("同一 (工具, 目标) 合并成 1 条", len(fs) == 2, str(fs))
    reuters = next((f for f in fs if f["tool"] == "fetch_page"), None)
    check("计数正确", reuters is not None and reuters["count"] == 2)
    check("目标取自配对 tool_call 的参数", reuters is not None and reuters["target"] == url)
    check("类型从首行解析", reuters is not None and reuters["kind"] == Kind.TERMINAL.value, str(reuters))
    check("source_id 指向最近一次失败", reuters is not None and reuters["source_id"] == "call_3_0")
    check("按最近出现排序", fs[-1]["tool"] == "search_news")
    check("正常结果不算失败", all(f["target"] != "nvda" for f in fs))
    check("limit 生效且保留最新", extract_failures(st["messages"], 3, limit=1)[0]["tool"] == "search_news")
    check("同一输入两次结果一致", fs == extract_failures(st["messages"], 3, limit=10))


# --------------------------------------------------------------------------
# S1:健康运行回归 —— 没有归档、没有失败时,视图就是今天的视图
# --------------------------------------------------------------------------
def scenario_healthy_view() -> None:
    print("\nS1 · 健康运行:视图 = system + 头部 + 原文(超长工具结果只多一行 stub)")
    long_text = "甲" * 4000
    st = _session(
        rounds=[
            _round(1, "search_news", {"query": "nvda"}, "短结果"),
            _round(2, "fetch_page", {"url": "http://a"}, long_text),
        ]
    )
    msgs = st["messages"]
    view = build_view(msgs, st["turns"], empty_ctx(0, 3), PREVIEW)
    check("消息条数不变", len(view) == len(msgs), f"{len(view)} vs {len(msgs)}")
    check("短工具结果原样(同一对象)", view[4] is msgs[4])
    long_view = str(view[6].content)
    check("长结果保留前 1500 字", long_view.startswith("甲" * PREVIEW) and "甲" * (PREVIEW + 1) not in long_view)
    check("长结果带 stub(含 id、全长、回读方式)",
          "id=call_2_0" in long_view and "全长 4000 字" in long_view and "read_tool_result" in long_view)
    check("messages 本身没有被改", str(msgs[6].content) == long_text)
    check("没有归档块和钉住区", ARCHIVE_HEADER not in _texts(view) and PINNED_HEADER not in _texts(view))
    check("同一输入两次渲染逐字一致",
          [_sig(m) for m in view] == [_sig(m) for m in build_view(msgs, st["turns"], empty_ctx(0, 3), PREVIEW)])


# --------------------------------------------------------------------------
# S2(手工构造 ctx):归档之后,头部仍逐字存在,已归档的圈不再出现
# --------------------------------------------------------------------------
def scenario_head_protected_view() -> None:
    print("\nS2a · 头部保护:归档后问题与记忆块原文仍在")
    q = "NVDA 本季毛利率和数据中心收入怎么样?对持仓有什么风险?"
    st = _session(question=q, rounds=[_round(i, "search_news", {"query": f"q{i}"}, f"结果{i}") for i in range(1, 5)])
    ctx = empty_ctx(0, 3)
    ctx["upto"] = 7  # 前两圈(下标 3..6)已归档
    ctx["archive"] = [
        {"round": 1, "ids": ["call_1_0"], "calls": ["search_news(q1)"], "gist": "要点1"},
        {"round": 2, "ids": ["call_2_0"], "calls": ["search_news(q2)"], "gist": "要点2"},
    ]
    view = build_view(st["messages"], st["turns"], ctx, PREVIEW)
    text = _texts(view)
    check("问题原文逐字存在", q in [str(m.content) for m in view])
    check("记忆块原文逐字存在", "【记忆】用户持有 100 股 NVDA" in [str(m.content) for m in view])
    check("头部紧跟 system", [type(m).__name__ for m in view[:3]] == ["SystemMessage", "HumanMessage", "HumanMessage"])
    check("归档块在头部之后", view[3].content.startswith(ARCHIVE_HEADER))
    check("归档条目带 id", "[id: call_1_0]" in text and "R2 search_news(q2) → 要点2" in text)
    check("已归档圈的原文不再出现", "结果1" not in text and "结果2" not in text)
    check("未归档圈原文保留", "结果3" in text and "结果4" in text)
    check("配对不变式成立", _paired(view))


# --------------------------------------------------------------------------
# S5(前半):两次压缩之间,上一圈视图是下一圈视图的前缀(钉住区除外)
# --------------------------------------------------------------------------
def scenario_prefix_stable_between_compactions() -> None:
    print("\nS5a · 前缀稳定:不压缩的圈之间只在末尾追加")
    url = "http://blocked"
    rounds = [
        _round(1, "fetch_page", {"url": url}, _fail("fetch_page", {"url": url})),
        _round(2, "search_news", {"query": "a"}, "乙" * 3000),
        _round(3, "search_news", {"query": "b"}, "丙"),
    ]
    ok = True
    prev = None
    for n in range(1, len(rounds) + 1):
        st = _session(rounds=rounds[:n])
        ctx = empty_ctx(0, 3)
        ctx["failures"] = extract_failures(st["messages"], 3, limit=10)
        view = [_sig(m) for m in build_view(st["messages"], st["turns"], ctx, PREVIEW)]
        if prev is not None:
            body = prev[:-1] if prev and PINNED_HEADER in prev[-1][1] else prev
            ok = ok and view[: len(body)] == body
        prev = view
    check("每圈的视图(去掉钉住区)都是下一圈视图的前缀", ok)
    check("有失败时钉住区在最末尾", prev is not None and PINNED_HEADER in prev[-1][1])


# --------------------------------------------------------------------------
# S10:多轮折叠 —— 旧轮次只留"问题 + 最终答案"
# --------------------------------------------------------------------------
def scenario_multi_turn_fold() -> None:
    print("\nS10 · 同一 thread 第二轮:第一轮过程被折叠")
    msgs: list = [
        SystemMessage(content="SYSTEM"),
        HumanMessage(content="【记忆】块1"),
        HumanMessage(content="第一问:AAPL 怎么样?"),
        *_round(1, "search_news", {"query": "aapl"}, "第一轮的工具原文"),
        AIMessage(content="第一轮结论:AAPL 稳健"),
        HumanMessage(content="【记忆】块2"),
        HumanMessage(content="第二问:那 NVDA 呢?"),
        *_round(2, "search_news", {"query": "nvda"}, "第二轮的工具原文"),
    ]
    turns = [{"start": 0, "body": 3}, {"start": 6, "body": 8}]
    view = build_view(msgs, turns, empty_ctx(1, 8), PREVIEW)
    text = _texts(view)
    check("第二条是折叠消息", str(view[1].content).startswith(PREVIOUS_TURNS_HEADER))
    check("折叠里有第一轮问题和结论", "第一问:AAPL 怎么样?" in str(view[1].content) and "第一轮结论:AAPL 稳健" in str(view[1].content))
    check("第一轮的工具原文不出现", "第一轮的工具原文" not in text)
    check("第一轮的记忆块不出现", "【记忆】块1" not in text)
    check("本轮头部与过程保留", "第二问:那 NVDA 呢?" in text and "第二轮的工具原文" in text and "【记忆】块2" in text)
    check("配对不变式成立", _paired(view))


# --------------------------------------------------------------------------
# 长会话:一圈一圈地跑 compress_step,模拟真实循环
# --------------------------------------------------------------------------


def _run_session(rounds: list, llm, cfg: ContextConfig = CFG, question: str = "NVDA 本季毛利率?"):
    """逐圈追加 rounds 并调用 compress_step;返回每圈的 (state, 输出)。ctx 在圈与圈之间传递。"""
    trace = []
    ctx = None
    for n in range(1, len(rounds) + 1):
        st = _session(question=question, rounds=rounds[:n])
        if ctx is not None:
            st["ctx"] = ctx
        out = compress_step(st, llm, cfg)
        ctx = out["ctx"]
        trace.append((st, out))
    return trace


def _archive_text(view: list) -> str:
    return next((str(m.content) for m in view if str(m.content).startswith(ARCHIVE_HEADER)), "")


# --------------------------------------------------------------------------
# S2 + S3 + S4 + S5b:长会话里的头部保护、配对、增量、归档块只追加
# --------------------------------------------------------------------------
def scenario_long_session() -> None:
    print("\nS2/S3/S4/S5b · 长会话:8 圈、每圈 1000 字,反复越过水位")
    q = "NVDA 本季毛利率和数据中心收入怎么样?"
    rounds = [_round(i, "search_news", {"query": f"q{i}"}, f"R{i}正文" + "丁" * 1000) for i in range(1, 9)]
    llm = FakeLLM()
    with _Events() as ev:
        trace = _run_session(rounds, llm, question=q)
    views = [out["llm_input"] for _, out in trace]
    final_ctx = trace[-1][1]["ctx"]

    check("发生了多次压缩", final_ctx["compactions"] >= 2, f"{final_ctx['compactions']} 次")
    check("S2 每圈视图里问题原文都在", all(q in [str(m.content) for m in v] for v in views))
    check("S2 每圈视图里记忆块都在", all("【记忆】用户持有 100 股 NVDA" in [str(m.content) for m in v] for v in views))
    check("S3 每圈视图都满足配对不变式", all(_paired(v) for v in views))
    check("S3 最近 2 圈始终是原文", all(f"R{n}正文" in _texts(v) for n, v in enumerate(views, 1))
          and all(f"R{n - 1}正文" in _texts(v) for n, v in enumerate(views, 1) if n > 1))

    seen = [llm.rounds_seen(k) for k in range(len(llm.calls))]
    flat = [r for s in seen for r in s]
    check("S4 归档器调用次数 = 压缩次数", len(llm.calls) == final_ctx["compactions"])
    check("S4 每次只收到新溢出的圈(圈号不重复)", len(flat) == len(set(flat)), str(seen))
    check("S4 归档顺序从旧到新", flat == sorted(flat))
    sizes = [len(c) for c in llm.calls]
    check("S4 单次归档输入不随轮数累积增长", max(sizes) <= 2 * min(sizes), f"各次输入字数 {sizes}")
    check("upto 前移到最后一个已归档圈之后",
          final_ctx["upto"] == split_rounds(trace[-1][0]["messages"], 3)[flat[-1] - 1].end)

    archives = [_archive_text(v) for v in views]
    grows = all(b.startswith(a) for a, b in zip(archives, archives[1:]) if a)
    check("S5b 归档块只追加:新文本以旧文本为前缀", grows)
    check("归档条目覆盖全部已归档的圈", all(f"R{r} " in archives[-1] for r in flat))
    check("每次压缩都发了 ctx_compacted", len(ev.of("ctx_compacted")) == final_ctx["compactions"])
    check("没有发生降级", not any(e["fallback"] for e in ev.of("ctx_compacted")))


# --------------------------------------------------------------------------
# S1 补充:低于水位时 0 次 LLM
# --------------------------------------------------------------------------
def scenario_no_llm_below_watermark() -> None:
    print("\nS1b · 低于水位:compress_step 不调 LLM,视图与 build_view 一致")
    rounds = [_round(i, "search_news", {"query": f"q{i}"}, "短") for i in range(1, 6)]
    llm = FakeLLM()
    trace = _run_session(rounds, llm)
    check("0 次 LLM", not llm.calls)
    st, out = trace[-1]
    expect = build_view(st["messages"], st["turns"], empty_ctx(0, 3), PREVIEW)
    check("视图与纯渲染一致", [_sig(m) for m in out["llm_input"]] == [_sig(m) for m in expect])


# --------------------------------------------------------------------------
# S6:失败钉住 —— 经过 3 次压缩,早期的失败仍在钉住区
# --------------------------------------------------------------------------
def scenario_failures_pinned() -> None:
    print("\nS6 · 早期的 3 个失败,经多次压缩后仍在钉住区")
    urls = [f"https://blocked{i}.com" for i in range(3)]
    rounds = [_round(i + 1, "fetch_page", {"url": u}, _fail("fetch_page", {"url": u})) for i, u in enumerate(urls)]
    rounds += [_round(i, "search_news", {"query": f"q{i}"}, "戊" * 1000) for i in range(4, 11)]
    trace = _run_session(rounds, FakeLLM())
    ctx = trace[-1][1]["ctx"]
    pinned = str(trace[-1][1]["llm_input"][-1].content)
    check("至少 3 次压缩", ctx["compactions"] >= 3, f"{ctx['compactions']} 次")
    check("失败圈的原文已被归档出视图", "【错误原文】" not in _texts(trace[-1][1]["llm_input"][:-1]))
    check("钉住区在最末尾", pinned.startswith(PINNED_HEADER))
    check("3 个失败目标都在钉住区", all(u in pinned for u in urls), pinned[:200])


# --------------------------------------------------------------------------
# S7:事实校验 —— 只放行数字能在来源原文里找到的
# --------------------------------------------------------------------------
def scenario_fact_verification() -> None:
    print("\nS7 · 事实校验:编造的数字、不存在的来源、残缺字段都被丢弃")
    page = "英伟达 FY26Q2 数据中心收入 $264 亿,同比 +56%,毛利率 72.7%。" + "己" * 1200
    rounds = [_round(1, "fetch_page", {"url": "http://nv"}, page)]
    rounds += [_round(i, "search_news", {"query": f"q{i}"}, "庚" * 1000) for i in range(2, 5)]
    good = {"entity": "nvda", "metric": "数据中心收入", "period": "FY26Q2", "value": "$264亿", "source_id": "call_1_0"}
    facts = [
        good,
        {**good, "metric": "毛利率", "value": "75.1%"},  # 原文是 72.7%
        {**good, "metric": "同比", "value": "+56%", "source_id": "call_x"},  # 来源不存在
        {"entity": "NVDA", "metric": "指引", "value": "偏强", "source_id": "call_1_0"},  # 缺 period
    ]
    with _Events() as ev:
        trace = _run_session(rounds, FakeLLM(facts=facts))
    ctx = trace[-1][1]["ctx"]
    reasons = sorted(e["reason"] for e in ev.of("ctx_fact_dropped"))
    check("只保留 1 条", len(ctx["facts"]) == 1, str(ctx["facts"]))
    check("保留的是正确那条", ctx["facts"] and ctx["facts"][0]["value"] == "$264亿")
    check("丢弃 3 条且原因各不相同", reasons == sorted(["数字不在原文", "来源不存在", "字段缺失"]), str(reasons))
    check("事实出现在钉住区", "数据中心收入 | FY26Q2 | $264亿" in str(trace[-1][1]["llm_input"][-1].content))


# --------------------------------------------------------------------------
# S9:归档 LLM 失败 → 确定性条目,照常前移
# --------------------------------------------------------------------------
def scenario_archive_fallback() -> None:
    print("\nS9 · 归档失败:降级成只有调用记录与 id 的条目,视图照样缩小")
    rounds = [_round(i, "search_news", {"query": f"q{i}"}, "辛" * 1000) for i in range(1, 5)]
    with _Events() as ev:
        trace = _run_session(rounds, FakeLLM(fail=True))
    ctx = trace[-1][1]["ctx"]
    archive = _archive_text(trace[-1][1]["llm_input"])
    compacted = ev.of("ctx_compacted")
    check("发生了压缩且标记为降级", compacted and all(e["fallback"] for e in compacted))
    check("降级条目带固定文案和 id", FALLBACK_GIST in archive and "[id: call_1_0]" in archive, archive[:160])
    check("upto 照常前移", ctx["upto"] > 3)
    check("没有可用事实", ctx["facts"] == [])
    check("llm=None 时同样降级而不是崩溃",
          compress_step({**_session(rounds=rounds), "ctx": None}, None, CFG)["ctx"]["compactions"] == 1)


# --------------------------------------------------------------------------
# S8:回读
# --------------------------------------------------------------------------
def scenario_readback() -> None:
    print("\nS8 · 回读:按 id 和 offset 取回原文片段")
    text = "".join(str(i % 10) for i in range(8000))
    st = _session(rounds=[_round(1, "fetch_page", {"url": "http://long"}, text)])
    props = read_tool_result.tool_call_schema.model_json_schema().get("properties", {})
    check("模型看到的 schema 里没有 state", "state" not in props and set(props) == {"id", "offset"}, str(list(props)))

    with _Events() as ev:
        out = read_tool_result.invoke({"id": "call_1_0", "offset": 1500, "state": st})
        end = read_tool_result.invoke({"id": "call_1_0", "offset": 9000, "state": st})
        miss = read_tool_result.invoke({"id": "nope", "state": st})
    check("拿到 [1500, 3000) 原文", text[1500:3000] in out and text[3000:3010] not in out.split("[第")[0][-20:])
    check("被不可信标记包住", "不可信" in out and "read_tool_result←fetch_page" in out)
    check("给出下一段 offset", "下一段 offset=3000" in out, out[-40:])
    check("越界提示已到末尾", "已读到末尾" in end)
    check("id 不存在返回友好提示", "没有找到 id=nope" in miss)
    check("发了 ctx_readback 事件", [e["found"] for e in ev.of("ctx_readback")] == [True, True, False])

    # A/B 实测踩到的坑:回读那一圈被归档后,模型拿"回读结果"的 id 去回读,读到的不是原文。
    rb_round = [
        AIMessage(content="", tool_calls=[{"name": "read_tool_result", "args": {"id": "call_1_0", "offset": 1500},
                                           "id": "call_rb"}]),
        ToolMessage(content=out, tool_call_id="call_rb", name="read_tool_result"),
    ]
    st2 = _session(rounds=[_round(1, "fetch_page", {"url": "http://long"}, text), rb_round])
    again = read_tool_result.invoke({"id": "call_rb", "state": st2})
    check("对回读结果再回读 → 指回原始 id", 'id="call_1_0"' in again and text[1500:1600] not in again, again)
    from memory.context.archive import fallback_entries
    entries = fallback_entries(split_rounds(st2["messages"], 3))
    check("归档条目不列出回读调用的 id", entries[1]["ids"] == [] and "call_1_0" in entries[1]["calls"][0],
          str(entries[1]))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("=" * 62)
    print("M8 上下文压缩离线回归(离线、不花钱)")
    print("=" * 62)
    for fn in (
        scenario_split_rounds,
        scenario_extract_failures,
        scenario_healthy_view,
        scenario_head_protected_view,
        scenario_prefix_stable_between_compactions,
        scenario_multi_turn_fold,
        scenario_no_llm_below_watermark,
        scenario_long_session,
        scenario_failures_pinned,
        scenario_fact_verification,
        scenario_archive_fallback,
        scenario_readback,
    ):
        fn()
    print("\n" + "=" * 62)
    if _failures:
        print(f"失败 {len(_failures)} 项:")
        for f in _failures:
            print("  -", f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
