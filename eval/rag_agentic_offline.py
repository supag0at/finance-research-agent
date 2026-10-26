"""Agentic RAG 子图的降级 / 路由离线测试(M9 Step 4)。零费用、确定性。

    python -m eval.rag_agentic_offline

LLM(analyze / grade / rewrite / hyde)、hybrid.search、知识库目录、search_news 全换成假实现,
测的是子图的**路由和降级**:LLM 坏 JSON / 超时、过滤值不在白名单、过滤后为空、rerank 失败、
grade 失败、改写两轮仍缺 → 联网、联网也失败、realtime 意图、快速通道、库里没有的公司不改写。
"""

from types import SimpleNamespace

from rag import hybrid
from rag.agentic import catalog, nodes
from rag.agentic.build import run

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


CATALOG = {
    "NVDA": {"company": "NVIDIA", "docs": [("10-K", 2025, "FY", "2025-01-26"), ("10-K", 2026, "FY", "2026-01-25")]},
    "688256": {"company": "寒武纪", "docs": [("年报", 2025, "FY", "2025-12-31")]},
}
catalog.load = lambda: CATALOG


def hit(cid: str, score: float, title: str = "NVIDIA 10-K FY2026") -> dict:
    return {"text": f"parent {cid}", "child_text": f"child {cid}", "source": "sec/x", "title": title,
            "score": score, "chunk_id": cid, "parent_id": f"P{cid}", "section": "Item 7", "page": 1,
            "ticker": "NVDA", "fiscal_year": 2026, "trust_level": "official"}


class Script:
    """按 prompt 类型分派的假 LLM。每类给一个回应列表(按调用顺序取),元素可以是 dict 或 Exception。"""

    def __init__(self, **by_kind) -> None:
        self.by_kind = {k: list(v) for k, v in by_kind.items()}
        self.calls: list[str] = []

    def __call__(self, prompt: str):
        kind = ("analyze" if "检索规划器" in prompt else "grade" if "审核员" in prompt
                else "rewrite" if "换一种查询" in prompt else "hyde")
        self.calls.append(kind)
        item = self.by_kind[kind].pop(0)
        if isinstance(item, Exception):
            raise item
        return item, 10


class FakeSearch:
    """fn(query, flt) → (hits, degraded)。记录每次调用的 query 和 filter。"""

    def __init__(self, fn) -> None:
        self.fn = fn
        self.calls: list[tuple[str, dict | None]] = []

    def __call__(self, query, flt=None, **kw):
        self.calls.append((query, flt))
        hits, deg = self.fn(query, flt)
        return SimpleNamespace(hits=hits, degraded=deg)


def setup(script: Script, search_fn, web=None) -> tuple[Script, FakeSearch]:
    nodes.json_call = script
    fs = FakeSearch(search_fn)
    hybrid.search = fs
    import tools.finance
    tools.finance.search_news = web or (lambda q, max_results=3: [{"title": "网页", "url": "https://x", "snippet": "s"}])
    return script, fs


def summary(out: list[dict]) -> dict:
    return out[0]


ANALYZE_NVDA = {"intent": "kb_fact", "needs": [{"desc": "NVDA FY2026 数据中心收入", "queries": ["NVIDIA FY2026 data center revenue"],
                                                 "filters": {"ticker": ["NVDA"], "fiscal_year": [2026]}}]}
GOOD = lambda q, f: ([hit("c1", 0.95), hit("c2", 0.7)], [])  # noqa: E731


def test_analyze_failures() -> None:
    print("① analyze 失败 → 原问题 + 无过滤")
    for label, err in (("坏 JSON", ValueError("模型返回的不是合法 JSON")), ("超时", TimeoutError("read timeout"))):
        s, fs = setup(Script(analyze=[err], grade=[{"needs": [{"id": "n1", "status": "sufficient"}]}]), GOOD)
        out = run("英伟达数据中心收入")
        check(f"{label}:degraded 记 analyze", "analyze" in summary(out)["degraded"], str(summary(out)["degraded"]))
        check(f"{label}:只用原问题检索、不带过滤", fs.calls == [("英伟达数据中心收入", None)], str(fs.calls))
        check(f"{label}:仍然返回证据", len(out) == 3)


def test_whitelist() -> None:
    print("② 过滤值过白名单")
    bad = {"intent": "kb_fact", "needs": [{"desc": "d", "queries": ["q"],
                                            "filters": {"ticker": ["NVIDA"], "fiscal_year": [2031]}}]}
    s, fs = setup(Script(analyze=[bad], grade=[{"needs": [{"id": "n1", "status": "sufficient"}]}]), GOOD)
    st = run("英伟达 2031 财年收入", full_state=True)
    check("拼错的 ticker 和不存在的年份都被丢掉,检索不带过滤", all(f is None for _, f in fs.calls), str(fs.calls))
    check("events 里记了丢弃原因", any("NVIDA" in e and "2031" in e for e in st["events"]))


def test_relax() -> None:
    print("③ 过滤后为空 → 按 fiscal_year → doc_type → ticker 放宽")
    def fn(q, f):
        keys = {c["key"] for c in (f or {}).get("must", [])}
        return ([hit("c9", 0.8)], []) if "fiscal_year" not in keys else ([], [])
    s, fs = setup(Script(analyze=[ANALYZE_NVDA], grade=[{"needs": [{"id": "n1", "status": "sufficient"}]}]), fn)
    st = run("q", full_state=True)
    n = st["needs"][0]
    check("放宽记录 = drop fiscal_year", n["relaxed"] == ["drop fiscal_year"], str(n["relaxed"]))
    check("放宽后拿到证据", [h["chunk_id"] for h in n["hits"]] == ["c9"])
    check("summary notes 提醒期间可能对不上", any("放宽" in x for x in summary(st["output"])["notes"]))


def test_rerank_degraded() -> None:
    print("④ rerank 失败 → 降级记录一路传到 summary")
    s, fs = setup(Script(analyze=[ANALYZE_NVDA], grade=[{"needs": [{"id": "n1", "status": "sufficient"}]}]),
                  lambda q, f: ([hit("c1", 0.02)], ["rerank"]))
    out = run("q")
    check("summary.degraded 含 rerank", "rerank" in summary(out)["degraded"])


def test_grade_failure() -> None:
    print("⑤ grade 失败 → 放行现有证据")
    s, fs = setup(Script(analyze=[ANALYZE_NVDA], grade=[ValueError("坏 JSON")]), lambda q, f: ([hit("c1", 0.6)], []))
    out = run("q")
    check("degraded 记 grade,证据照常返回", "grade" in summary(out)["degraded"] and out[1]["chunk_id"] == "c1")
    check("没有进入改写", "rewrite" not in s.calls)


def test_rewrite_then_web() -> None:
    print("⑥ 两轮都缺 → 联网兜底")
    missing = {"needs": [{"id": "n1", "status": "missing", "reason": "期间不对"}]}
    s, fs = setup(Script(analyze=[ANALYZE_NVDA], grade=[missing, missing],
                         rewrite=[{"queries": ["NVIDIA annual Data Center"], "drop_filters": ["fiscal_year"]}]),
                  lambda q, f: ([hit("c1", 0.6)], []))
    st = run("q", full_state=True)
    out = st["output"]
    check("调用顺序 analyze → grade → rewrite → grade", s.calls == ["analyze", "grade", "rewrite", "grade"], str(s.calls))
    second = fs.calls[-2:]
    check("第 2 轮 = q0 + 改写后的查询", [c[0] for c in second] == ["q", "NVIDIA annual Data Center"], str(second))
    check("第 2 轮按改写要求去掉了 fiscal_year,ticker 保留",
          all([c["key"] for c in f["must"]] == ["ticker"] for _, f in second), str(second))
    check("rounds = 2", summary(out)["rounds"] == 2)
    check("证据只剩网页,trust_level=web", [e["trust_level"] for e in out[1:]] == ["web"], str([e["trust_level"] for e in out[1:]]))
    check("被判 missing 的知识库证据不交给主 Agent", all(e["chunk_id"] is None for e in out[1:]))
    check("coverage = web_only", summary(out)["coverage"]["n1"]["status"] == "web_only")


def test_web_failure() -> None:
    print("⑦ 联网也失败 → 如实标缺失")
    missing = {"needs": [{"id": "n1", "status": "missing", "reason": "x"}]}

    def boom(q, max_results=3):
        raise ConnectionError("ddgs down")

    s, fs = setup(Script(analyze=[ANALYZE_NVDA], grade=[missing, missing], rewrite=[{"queries": ["q2"]}]),
                  lambda q, f: ([hit("c1", 0.6)], []), web=boom)
    out = run("q")
    sm = summary(out)
    check("coverage = missing,degraded 记 web", sm["coverage"]["n1"]["status"] == "missing" and "web" in sm["degraded"])
    check("没有任何证据交出去 + notes 说'不要编造'", len(out) == 1 and any("不要编造" in x for x in sm["notes"]))


def test_realtime() -> None:
    print("⑧ realtime 意图 → 不检索")
    s, fs = setup(Script(analyze=[{"intent": "realtime", "needs": []}]), GOOD)
    out = run("英伟达现在股价多少")
    check("没有调用检索", fs.calls == [])
    check("notes 引导去用行情工具", any("get_stock_price" in x for x in summary(out)["notes"]))


def test_fast_path() -> None:
    print("⑨ 快速通道 / 库里没有的公司")
    s, fs = setup(Script(analyze=[ANALYZE_NVDA]), lambda q, f: ([hit("c1", 0.97)], []))
    run("q")
    check("单需求 + 过滤有效 + top1 0.97 → 不调 grade", s.calls == ["analyze"], str(s.calls))

    apple = {"intent": "kb_fact", "needs": [{"desc": "苹果 FY2025 服务收入", "queries": ["Apple services revenue"],
                                              "filters": {"ticker": ["AAPL"], "fiscal_year": [2025]}}]}
    s, fs = setup(Script(analyze=[apple], grade=[{"needs": [{"id": "n1", "status": "missing", "reason": "是博通"}]}]),
                  lambda q, f: ([hit("c1", 0.91, "Broadcom 10-K FY2025")], []))
    st = run("苹果 2025 财年服务收入", full_state=True)
    check("ticker 不在库里:top1 0.91 也不走快速通道,交给 grade", "grade" in s.calls)
    check("库里没有的公司:grade 判缺后不改写,直接联网", "rewrite" not in s.calls and st["round"] == 1, str(s.calls))
    check("notes 说明库里没有 AAPL", any("AAPL" in x for x in summary(st["output"])["notes"]))


def test_pack_shape() -> None:
    print("⑩ pack 输出形态(不变式 4、5)")
    two = {"intent": "kb_fact", "needs": [
        {"desc": "A", "queries": ["qa"], "filters": {"ticker": ["NVDA"]}},
        {"desc": "B", "queries": ["qb"], "filters": {"ticker": ["688256"]}}]}
    s, fs = setup(Script(analyze=[two], grade=[{"needs": [{"id": "n1", "status": "sufficient"}, {"id": "n2", "status": "sufficient"}]}]),
                  lambda q, f: ([hit("same", 0.9), hit(q, 0.8)], []))
    out = run("A 和 B 对比")
    sm, ev = out[0], out[1:]
    check("ref 从 1 连续编号", [e["ref"] for e in ev] == list(range(1, len(ev) + 1)))
    check("两个需求共同命中的块只出现一次", sum(e["chunk_id"] == "same" for e in ev) == 1)
    check("每项都有 text / source / title / score(向后兼容)", all({"text", "source", "title", "score"} <= set(e) for e in out))
    check("每条证据都有 chunk_id 或是 web", all(e["chunk_id"] or e["trust_level"] == "web" for e in ev))
    check("第一项是 summary(M9 §7.8:放末尾会被截掉)", sm["type"] == "summary" and set(sm["coverage"]) == {"n1", "n2"})
    check("summary 列出证据 ref", sm["evidence_refs"] == [e["ref"] for e in ev])


def main() -> None:
    for t in (test_analyze_failures, test_whitelist, test_relax, test_rerank_degraded, test_grade_failure,
              test_rewrite_then_web, test_web_failure, test_realtime, test_fast_path, test_pack_shape):
        t()
    print()
    print("全部通过" if not _failures else f"失败 {len(_failures)} 项:{_failures}")
    raise SystemExit(1 if _failures else 0)


if __name__ == "__main__":
    main()
