"""一条问题的全链路追踪(M9 代码导读用)。会调 LLM / embedding / rerank,一题几分钱。

    python -m eval.rag_trace "英伟达和 AMD 最近一个完整财年的数据中心收入分别增长了多少?"
    python -m eval.rag_trace --no-agent "问题"     # 只跑 RAG 子图,不跑主 Agent
    python -m eval.rag_trace --render eval/reports/rag_trace_xxx.json   # 重新排版已保存的录像(不调 API)

做的事:让真实的主 Agent(LangGraph 版,RAG_MODE=agentic)回答这个问题,同时在旁边"录像":
  - 子图里每一次 LLM 调用(analyze / grade / rewrite)的原始 JSON;
  - 每一次 hybrid.search:每个候选在 dense / sparse / RRF / rerank 各排第几、分数多少、各阶段耗时;
  - 子图最后交给主 Agent 的证据(ref 编号)和 summary;
  - 主 Agent 调了哪些工具、最终回答、引用校验 / 数字溯源的结果。
不改任何业务代码,全部靠在运行时替换函数(录像完原样调用)。
"""

import os
import sys

os.environ.setdefault("RAG_MODE", "agentic")  # 必须在 import rag.config 之前

import json  # noqa: E402
import threading  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from rag import hybrid  # noqa: E402
from rag.agentic import build, nodes  # noqa: E402

_lock = threading.Lock()
LLM_CALLS: list[dict] = []
SEARCHES: list[dict] = []
STATES: list[dict] = []

_orig_json_call, _orig_search, _orig_run = nodes.json_call, hybrid.search, build.run


def _rec_json_call(prompt: str):
    kind = ("analyze" if "检索规划器" in prompt else "grade" if "审核员" in prompt
            else "rewrite" if "换一种查询" in prompt else "hyde")
    out, tok = _orig_json_call(prompt)
    with _lock:
        LLM_CALLS.append({"kind": kind, "tokens": tok, "output": out})
    return out, tok


def _rec_search(query, flt=None, **kw):
    res = _orig_search(query, flt=flt, **kw)
    rows = []
    for c in sorted(res.trace.values(), key=lambda c: (c.rerank_rank or 99, c.rrf_rank or 99)):
        pl = c.payload
        rows.append({"id": c.id[:8], "dense": c.dense_rank, "dense_score": c.dense_score, "sparse": c.sparse_rank,
                     "sparse_score": c.sparse_score, "rrf": c.rrf_rank, "rerank": c.rerank_rank,
                     "rerank_score": c.rerank_score, "title": pl.get("title"), "page": pl.get("page"),
                     "text": pl["text"][:90].replace("\n", " ")})
    with _lock:
        SEARCHES.append({"query": query, "filter": flt, "timings_ms": res.timings_ms, "degraded": res.degraded,
                         "n_candidates": len(res.trace), "n_after_threshold": len(res.ranked),
                         "final": [h["chunk_id"][:8] for h in res.hits], "rows": rows})
    return res


def _rec_run(question, opts=None, full_state=False):
    st = _orig_run(question, opts=opts, full_state=True)
    with _lock:
        STATES.append(st)
    return st if full_state else st["output"]


nodes.json_call, hybrid.search, build.run = _rec_json_call, _rec_search, _rec_run


def _p(*a) -> None:
    print(*a, flush=True)


def show_subgraph(st: dict, searches: list[dict], llm_calls: list[dict]) -> None:
    _p(f"\n## 子图:{st['question']}\n")
    for call in llm_calls:
        _p(f"### LLM 调用:{call['kind']}({call['tokens']} token)原始输出\n```json\n"
           f"{json.dumps(call['output'], ensure_ascii=False, indent=1)[:2500]}\n```")
    _p("\n### 过白名单后的需求")
    for n in st["needs"]:
        _p(f"- {n['id']} {n['desc']} | 过滤 {n['filters'] or '无'} | 查询 {n['queries']} | "
           f"状态 {n['status']} | 放宽 {n.get('relaxed') or '无'}")
    _p("\n### 过程事件")
    for e in st["events"]:
        _p(f"- {e}")
    for i, s in enumerate(searches, 1):
        _p(f"\n### hybrid.search #{i}:{s['query']!r}\n- 过滤:{json.dumps(s['filter'], ensure_ascii=False)}"
           f"\n- 候选 {s['n_candidates']} 个,rerank 过阈值 {s['n_after_threshold']} 个;耗时 "
           + " / ".join(f"{k} {v:.0f}ms" for k, v in s["timings_ms"].items()) + (f";降级 {s['degraded']}" if s["degraded"] else ""))
        _p("\n| dense | sparse | RRF | rerank | rerank 分 | 文档 | 页 | 内容 |\n| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in s["rows"][:8]:
            sc = "—" if r["rerank_score"] is None else f"{r['rerank_score']:.3f}"
            _p(f"| {r['dense'] or '—'} | {r['sparse'] or '—'} | {r['rrf'] or '—'} | {r['rerank'] or '—'} | {sc} "
               f"| {r['title']} | {r['page']} | {r['text'][:60]} |")
    _p("\n### 交给主 Agent 的证据")
    for e in st["output"][1:]:
        _p(f"- [{e['ref']}] {e['need']} {e['trust_level']} {e['score']} {e['title']} p.{e['page']} | "
           f"{e['section'][-50:]} | 父块 {len(e['text'])} 字符")
    sm = st["output"][0]
    _p(f"\n### summary\n```json\n{json.dumps({k: v for k, v in sm.items() if k not in ('text', 'source', 'title', 'score')}, ensure_ascii=False, indent=1)}\n```")


def _owner(states: list[dict], rec: dict, kind: str) -> int:
    """把一条录像归到它所属的那次子图。主 Agent 可能**并行**调多次 retrieve,录像按时间是交错的,
    不能按顺序切,要按内容认:查询文本属于哪个子图的需求、analyze 的输出和哪个子图的需求一致、
    grade 的理由出现在哪个子图的事件里。"""
    for i, st in enumerate(states):
        needs = st.get("needs") or []
        if kind == "search" and any(rec["query"] in n["queries"] for n in needs):
            return i
        if kind == "analyze" and [x.get("desc") for x in rec["output"].get("needs") or []] == [n["desc"] for n in needs]:
            return i
        if kind in ("grade", "rewrite"):
            reasons = [str(v.get("reason", ""))[:30] for v in rec["output"].get("needs") or []]
            if reasons and any(r and r in e for r in reasons for e in st["events"]):
                return i
            queries = [q for q in rec["output"].get("queries") or []]
            if queries and any(q in n["queries"] for q in queries for n in needs):
                return i
    return -1


def render(llm_calls: list[dict], searches: list[dict], states: list[dict]) -> None:
    for i, st in enumerate(states):
        mine_s = [r for r in searches if _owner(states, r, "search") == i]
        mine_l = [r for r in llm_calls if _owner(states, r, r["kind"]) == i]
        show_subgraph(st, mine_s, mine_l)


def main() -> None:
    if "--render" in sys.argv:  # 重新排版一份已保存的录像,不调任何 API
        d = json.loads(Path(sys.argv[sys.argv.index("--render") + 1]).read_text(encoding="utf-8"))
        render(d["llm_calls"], d["searches"], d["states"])
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    question = " ".join(args) or "英伟达和 AMD 最近一个完整财年的数据中心收入分别增长了多少?"
    if "--no-agent" in sys.argv:
        build.run(question)
    else:
        from eval.checks import run_checks
        from graph.agent import GraphAgent

        agent = GraphAgent()
        agent.use_long_term = False
        answer, traj = agent.run(question)
        _p("## 主 Agent 的工具调用\n")
        for s in traj:
            if s.tool_call:
                _p(f"- 第 {s.step} 步 `{s.tool_call.name}({json.dumps(s.tool_call.arguments, ensure_ascii=False)})` "
                   f"→ 返回 {len(str(s.observation or ''))} 字符")
        _p(f"\n## 最终回答\n\n{answer}\n")
        c = run_checks({"must_call_tools": ["retrieve"]}, answer, traj)
        cit = {k: v for k, v in c["citation"].items() if k != "rag_summaries"}
        _p(f"## 校验\n- 数字溯源(字面){c['grounded_ratio']},单位感知 {c['grounded_ratio_unit']},"
           f"字面找不到的:{c['ungrounded_numbers']}\n- 引用:{cit}")
        rec = getattr(agent, "last_run", None)
        if rec is not None:
            j = rec.to_json()
            _p(f"- 观测:总耗时 {j['total_ms'] / 1000:.1f}s,主 Agent LLM {j['llm']['calls']} 次 ¥{j['cost_cny']},"
               f"RunRecord.rag = {j['rag']}")
    render(LLM_CALLS, SEARCHES, STATES)
    out = Path(f"eval/reports/rag_trace_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.write_text(json.dumps({"llm_calls": LLM_CALLS, "searches": SEARCHES,
                               "states": [{k: v for k, v in s.items() if k != "t0"} for s in STATES]},
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _p(f"\n原始录像 → {out}")


if __name__ == "__main__":
    main()
