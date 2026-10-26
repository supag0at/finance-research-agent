"""RAG 子图的节点(M9 Step 4)。

    analyze ─► retrieve ─► grade ─┬─ 全部够 ──────────────────────────► pack
                  ▲               ├─ 有缺 且 轮数 < 上限 且 没超时 ─► rewrite ─┘(回 retrieve)
                  └───────────────┘
                                  └─ 有缺 且 不能再改写 ─► web_fallback ─► pack

每个"信息需求"(need)单独检索、单独评估:对比题"寒武纪 vs 海光"是两个需求,
各自带自己的 ticker 过滤,一边找到了另一边没找到时,只对缺的那边改写重检。

不变式(§2.2)在这里的落点:
  1. 任何 LLM 节点失败都降级:analyze 失败 → 原问题 + 无过滤;grade 失败 → 接受现有证据;
     rewrite 失败 → 不再重检;web 失败 → 如实标缺失。检索本身永不抛异常。
  2. 原问题 q0 永远是每个需求的第一条查询(改写只追加、不替换)。
  3. 过滤值全部过白名单(catalog.validate);过滤后为空按 fiscal_year → doc_type → ticker 的顺序放宽。
  4. pack 出来的每条证据要么有真实 chunk_id,要么是 trust_level=web 的 URL。
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor

from core.llm import LLM
from rag import hybrid
from rag.agentic import catalog
from rag.agentic.state import Need, RagState
from rag.config import (
    RAG_DEADLINE_S, RAG_EVIDENCE_CHARS, RAG_FAST_PATH, RAG_HYDE, RAG_MAX_ROUNDS, RAG_WEB_FALLBACK,
)
from rag.fusion import rrf

_llm = LLM()
# 两层并行用两个池:需求 → 查询。共用一个池会死锁(外层任务占满线程,等内层任务排不上队)。
_need_pool = ThreadPoolExecutor(4)
_query_pool = ThreadPoolExecutor(12)
RELAX_ORDER = ("fiscal_year", "doc_type", "ticker")


def _opt(state: RagState, key: str):
    defaults = {"grade": True, "web": RAG_WEB_FALLBACK, "hyde": RAG_HYDE, "max_rounds": RAG_MAX_ROUNDS}
    return (state.get("opts") or {}).get(key, defaults[key])


def json_call(prompt: str) -> tuple[dict, int]:
    """一次 JSON 模式调用,返回 (结果, token 数)。直接用 client 而不是 LLM.complete_json:
    后者把 usage 存在实例上,评测里多线程并发会互相覆盖。"""
    resp = _llm.client.chat.completions.create(
        model=_llm.model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        timeout=30,
    )
    text = (resp.choices[0].message.content or "").strip()
    data = json.loads(text)  # 不合法直接抛,由调用方降级
    if not isinstance(data, dict):
        raise ValueError("不是 JSON 对象")
    usage = getattr(resp, "usage", None)
    return data, (usage.total_tokens if usage else 0)


def _add(state: RagState, key: str, *items) -> list:
    return list(state.get(key) or []) + list(items)


# ---------------------------------------------------------------- analyze

_ANALYZE_PROMPT = """你是金融知识库的检索规划器。把用户问题拆成 1~3 个"信息需求",并给每个需求写检索查询和过滤条件。

【知识库里有什么】
{catalog}

注意财年口径:各公司财年和自然年不一致。NVDA 的 FY2026 是 2025 年 2 月到 2026 年 1 月;
Broadcom 截至 11 月初,Micron 截至 9 月初;AMD / Intel / A 股公司财年 = 自然年。
用户说"2026 财年"就用 FY2026;说"最近一个季度""最新"就选目录里最新那份 10-Q / 半年报。

【用户问题】
{question}

规则:
1. intent:问知识库里的财报 / 公司 / 行业知识 → "kb_fact";问实时股价、今天的新闻 → "realtime";和金融无关 → "out_of_scope"。
2. needs:每个需求一个对象。问一家公司就是 1 个需求;对比两家公司就是 2 个需求(各自一家)。最多 3 个。
3. queries:每个需求 1~2 条检索查询。**问的是美股公司(英文财报)时,至少写一条英文查询**(财报原文是英文);
   A 股公司用中文。查询要具体,写出公司名、期间和指标。
4. filters:只能用目录里出现的 ticker、fiscal_year(整数)、doc_type(10-K / 10-Q / 年报 / 半年报)。
   不确定就不写;库里没有的公司也照样写它的 ticker(程序会识别并处理)。

只输出 JSON:
{{"intent": "kb_fact", "needs": [{{"desc": "英伟达 FY2026 数据中心收入增速", "queries": ["NVIDIA fiscal 2026 Data Center revenue growth"],
  "filters": {{"ticker": ["NVDA"], "fiscal_year": [2026], "doc_type": ["10-K"]}}}}]}}"""


def _baseline_needs(q: str) -> list[Need]:
    return [Need(id="n1", desc=q, queries=[q], filters={}, filter_valid=False, status="pending", hits=[], relaxed=[])]


def analyze(state: RagState) -> RagState:
    q = state["question"]
    try:
        out, tok = json_call(_ANALYZE_PROMPT.format(catalog=catalog.prompt_text(), question=q))
        intent = out.get("intent", "kb_fact")
        raw_needs = out.get("needs") or []
        if intent not in ("kb_fact", "realtime", "out_of_scope") or (intent == "kb_fact" and not raw_needs):
            raise ValueError(f"字段不全:{str(out)[:80]}")
    except Exception as exc:  # noqa: BLE001 —— 不变式 1:分析坏了就退回基线
        return {"intent": "kb_fact", "needs": _baseline_needs(q), "round": 0,
                "degraded": _add(state, "degraded", "analyze"),
                "events": _add(state, "events", f"analyze 失败({type(exc).__name__}),退回原问题 + 无过滤")}

    needs: list[Need] = []
    events, notes = [], []
    for i, rn in enumerate(raw_needs[:3], 1):
        queries = [str(x) for x in (rn.get("queries") or []) if str(x).strip()][:2]
        flt, dropped = catalog.validate(rn.get("filters"))
        if dropped:
            events.append(f"n{i} 过滤值不在白名单,已丢弃:{'、'.join(dropped)}")
            if any(d.startswith("ticker=") for d in dropped):
                notes.append(f"知识库里没有 {'、'.join(d.split('=')[1].split('(')[0] for d in dropped if d.startswith('ticker='))} 的资料")
        # 要的公司全都不在库里(问苹果):检索一轮兜个底就够了,改写重检不可能变出苹果的财报
        kb_absent = any(d.startswith("ticker=") for d in dropped) and "ticker" not in flt
        needs.append(Need(
            id=f"n{i}", desc=str(rn.get("desc") or q), queries=[q] + [x for x in queries if x != q],
            filters=flt, filter_valid=bool(flt) and not dropped, status="pending", hits=[], relaxed=[],
            kb_absent=kb_absent,
        ))
    events.append(f"analyze:intent={intent},{len(needs)} 个需求:" +
                  ";".join(f"{n['id']} {n['filters'] or '无过滤'} {n['queries'][1:]}" for n in needs))
    return {"intent": intent, "needs": needs, "round": 0, "llm_tokens": state.get("llm_tokens", 0) + tok,
            "events": _add(state, "events", *events), "notes": _add(state, "notes", *notes)}


# ---------------------------------------------------------------- retrieve

_HYDE_PROMPT = """为下面的检索需求写一段"假想的财报原文"(80~150 字/词),用财报里会出现的措辞和数字格式,
数字可以编,但要像真的。{lang_rule}只输出 JSON:{{"passage": "..."}}

需求:{desc}"""


def _search_need(need: Need, hyde: bool) -> tuple[list[dict], list[str], list[str], int]:
    """一个需求的检索:每条查询跑一次 hybrid(并行),各自 rerank + 阈值,再用 RRF 合并。
    结果为空就按 RELAX_ORDER 逐级放宽过滤。返回 (hits, 放宽记录, 降级记录, token)。"""
    top_k = need.get("_top_k", 5)
    flt = dict(need["filters"])
    relaxed, degraded, tok = [], [], 0
    dense_q = None
    if hyde:
        en = any(t not in ("688256", "688981", "688041") for t in flt.get("ticker", [])) if flt.get("ticker") else False
        try:
            out, tok = json_call(_HYDE_PROMPT.format(desc=need["desc"], lang_rule="用英文写。" if en else ""))
            dense_q = str(out.get("passage") or "") or None
        except Exception:  # noqa: BLE001
            degraded.append("hyde")
    while True:
        qflt = catalog.to_qdrant(flt)
        results = list(_query_pool.map(
            lambda iq: hybrid.search(iq[1], flt=qflt, top_k=top_k, expand=True,
                                     dense_query=dense_q if iq[0] == 0 else None),
            enumerate(need["queries"]),
        ))
        for r in results:
            degraded += [d for d in r.degraded if d not in degraded]
        by_id = {}
        for r in results:
            for h in r.hits:
                if h["chunk_id"] not in by_id or h["score"] > by_id[h["chunk_id"]]["score"]:
                    by_id[h["chunk_id"]] = h
        fused = rrf([[h["chunk_id"] for h in r.hits] for r in results])
        hits = [by_id[cid] for cid, _ in fused][:top_k]
        if hits or not flt:
            return hits, relaxed, degraded, tok
        key = next(k for k in RELAX_ORDER if k in flt)
        flt.pop(key)
        relaxed.append(f"drop {key}")


def retrieve(state: RagState) -> RagState:
    needs = [dict(n) for n in state["needs"]]
    todo = [n for n in needs if n["status"] in ("pending", "missing")]
    top_k = 5 if len(needs) == 1 else 3
    for n in todo:
        n["_top_k"] = top_k
    outs = list(_need_pool.map(lambda n: _search_need(n, _opt(state, "hyde")), todo))
    events, degraded, tok = [], list(state.get("degraded") or []), 0
    for n, (hits, relaxed, deg, t) in zip(todo, outs):
        n["hits"], n["relaxed"] = hits, n.get("relaxed", []) + relaxed
        n.pop("_top_k", None)
        tok += t
        degraded += [d for d in deg if d not in degraded]
        top = f"{hits[0]['score']:.3f} {hits[0]['title']} p.{hits[0]['page']}" if hits else "无"
        events.append(f"retrieve 第 {state.get('round', 0) + 1} 轮 {n['id']}:{len(hits)} 条,top1 {top}"
                      + (f",放宽 {relaxed}" if relaxed else ""))
    return {"needs": needs, "round": state.get("round", 0) + 1, "degraded": degraded,
            "llm_tokens": state.get("llm_tokens", 0) + tok, "events": _add(state, "events", *events)}


# ---------------------------------------------------------------- grade

_GRADE_PROMPT = """你是检索证据的审核员。判断每个信息需求,现有证据**是否足以回答**。

【用户问题】
{question}

{blocks}

判定规则:
- sufficient:证据里明确写着回答这个需求所需的信息(数字 / 原因),而且**公司和期间都对得上**。
- missing:证据是别的公司、别的期间(比如问 2025 年报,证据是 2026 半年报),或者只沾边、没有需要的那个数字 / 结论。
- 不要求证据完美,有直接回答的那一句就够。

只输出 JSON:{{"needs": [{{"id": "n1", "status": "sufficient 或 missing", "reason": "一句话"}}]}}"""


def _evidence_block(n: Need) -> str:
    lines = [f"【需求 {n['id']}】{n['desc']}"]
    if not n["hits"]:
        lines.append("  (没有检索到任何证据)")
    for i, h in enumerate(n["hits"][:4], 1):
        snippet = h.get("child_text", h["text"])[:500].replace("\n", " ")
        lines.append(f"  证据 {i}:[{h['title']} | {h.get('section', '')} | 第 {h.get('page')} 页]{snippet}")
    return "\n".join(lines)


def grade(state: RagState) -> RagState:
    needs = [dict(n) for n in state["needs"]]
    todo = [n for n in needs if n["status"] in ("pending", "missing")]
    events = []
    # 快速通道:只有一个需求、过滤条件全部有效且没被放宽、top1 很高 → 不花一次 LLM。
    # 过滤有效这个前提很关键:问"苹果"时 ticker 不在白名单、没有过滤,博通的段落也能拿到 0.907(§6.4)。
    if (len(needs) == 1 and todo and todo[0]["filter_valid"] and not todo[0]["relaxed"]
            and todo[0]["hits"] and todo[0]["hits"][0]["score"] >= RAG_FAST_PATH):
        todo[0]["status"], todo[0]["reason"] = "sufficient", f"快速通道:top1 {todo[0]['hits'][0]['score']:.3f}"
        return {"needs": needs, "events": _add(state, "events", f"grade 快速通道:{todo[0]['id']} 直接判够")}
    if not _opt(state, "grade"):
        for n in todo:
            n["status"] = "sufficient" if n["hits"] else "missing"
        return {"needs": needs, "events": _add(state, "events", "grade 关闭(消融):有证据就算够")}
    # 一条证据都没有的需求不用问 LLM
    for n in todo:
        if not n["hits"]:
            n["status"], n["reason"] = "missing", "没有检索到证据"
    ask = [n for n in todo if n["hits"]]
    if not ask:
        return {"needs": needs, "events": _add(state, "events", "grade:所有待评需求都没有证据")}
    try:
        out, tok = json_call(_GRADE_PROMPT.format(question=state["question"],
                                                  blocks="\n\n".join(_evidence_block(n) for n in ask)))
        verdict = {v.get("id"): v for v in out.get("needs") or []}
    except Exception as exc:  # noqa: BLE001 —— grade 坏了就接受现有证据,别让检索卡死
        for n in ask:
            n["status"], n["reason"] = "sufficient", "grade 失败,按现有证据放行"
        return {"needs": needs, "degraded": _add(state, "degraded", "grade"),
                "events": _add(state, "events", f"grade 失败({type(exc).__name__}),放行现有证据")}
    for n in ask:
        v = verdict.get(n["id"], {})
        n["status"] = "sufficient" if v.get("status") == "sufficient" else "missing"
        n["reason"] = str(v.get("reason", "")) or ("评估结果里没有这个需求" if not v else "")
        events.append(f"grade {n['id']}:{n['status']}({n['reason'][:60]})")
    return {"needs": needs, "llm_tokens": state.get("llm_tokens", 0) + tok, "events": _add(state, "events", *events)}


def route_after_grade(state: RagState) -> str:
    missing = [n for n in state["needs"] if n["status"] == "missing"]
    if not missing:
        return "pack"
    timed_out = time.time() - state["t0"] > RAG_DEADLINE_S
    rewritable = [n for n in missing if not n.get("kb_absent")]
    if rewritable and state.get("round", 0) < _opt(state, "max_rounds") and not timed_out:
        return "rewrite"
    return "web_fallback" if _opt(state, "web") else "pack"


# ---------------------------------------------------------------- rewrite

_REWRITE_PROMPT = """上一轮检索没有找到足够的证据。请为这个信息需求换一种查询方式。

【用户问题】{question}
【需求】{desc}
【已经用过的查询】{queries}
【当前过滤条件】{filters}
【没找到的原因】{reason}
【知识库里有什么】
{catalog}

可以:换同义说法、换语言(美股公司财报是英文)、换成财报里的科目名称、把过于具体的问法放宽一点;
如果过滤条件可能错了(比如期间选错),可以在 drop_filters 里列出要去掉的字段(fiscal_year / doc_type / ticker)。

只输出 JSON:{{"queries": ["新查询 1", "新查询 2"], "drop_filters": []}}"""


def rewrite(state: RagState) -> RagState:
    needs = [dict(n) for n in state["needs"]]
    events, tok = [], 0
    for n in needs:
        if n["status"] != "missing":
            continue
        if n.get("kb_absent"):
            n["status"] = "missing_done"  # 库里没有这家公司:不改写,等其他需求重检完一起去联网
            continue
        try:
            out, t = json_call(_REWRITE_PROMPT.format(
                question=state["question"], desc=n["desc"], queries=n["queries"], filters=n["filters"] or "无",
                reason=n.get("reason", ""), catalog=catalog.prompt_text()))
            tok += t
        except Exception as exc:  # noqa: BLE001 —— 改写失败:这个需求不再重检,保持 missing
            n["status"] = "missing_final"
            events.append(f"rewrite {n['id']} 失败({type(exc).__name__}),不再重检")
            continue
        new_q = [str(x) for x in (out.get("queries") or []) if str(x).strip() and str(x) not in n["queries"]][:2]
        flt = dict(n["filters"])
        for k in out.get("drop_filters") or []:
            if k in flt:
                flt.pop(k)
                n["relaxed"] = n.get("relaxed", []) + [f"rewrite drop {k}"]
        n["queries"] = [n["queries"][0]] + new_q  # q0 永远保留;上一轮的改写换掉,避免越滚越多
        n["filters"] = flt
        events.append(f"rewrite {n['id']}:{new_q}" + (f",去掉过滤 {out.get('drop_filters')}" if out.get("drop_filters") else ""))
    for n in needs:  # 改写失败的需求不参与下一轮
        if n["status"] == "missing_final":
            n["status"] = "missing_done"
    return {"needs": needs, "llm_tokens": state.get("llm_tokens", 0) + tok, "events": _add(state, "events", *events)}


def route_after_rewrite(state: RagState) -> str:
    return "retrieve" if any(n["status"] == "missing" for n in state["needs"]) else "pack"


# ---------------------------------------------------------------- web_fallback

def web_fallback(state: RagState) -> RagState:
    from tools.finance import search_news  # 延迟导入:离线测试可以替换

    needs = [dict(n) for n in state["needs"]]
    events, degraded = [], list(state.get("degraded") or [])
    for n in needs:
        if n["status"] not in ("missing", "missing_done"):
            continue
        query = n["queries"][-1]
        try:
            results = [r for r in search_news(query, max_results=3) if r.get("url")]
        except Exception as exc:  # noqa: BLE001
            results = []
            if "web" not in degraded:
                degraded.append("web")
            events.append(f"web_fallback {n['id']} 失败({type(exc).__name__})")
        if results:
            n["hits"] = n["hits"] + [{
                "text": f"{r['title']}\n{r['snippet']}", "child_text": r["snippet"], "source": r["url"],
                "title": r["title"], "score": None, "chunk_id": None, "section": "", "page": None,
                "trust_level": "web",
            } for r in results]
            n["status"] = "web"
            events.append(f"web_fallback {n['id']}:{len(results)} 条网页结果(query={query!r})")
        else:
            n["status"] = "missing_done"
    return {"needs": needs, "degraded": degraded, "events": _add(state, "events", *events)}


# ---------------------------------------------------------------- pack

def pack(state: RagState) -> RagState:
    intent = state.get("intent", "kb_fact")
    notes = list(state.get("notes") or [])
    if intent == "realtime":
        notes.append("这是实时数据问题,知识库只有定期财报;请改用 get_stock_price / search_news")
    elif intent == "out_of_scope":
        notes.append("问题和金融知识库无关,未检索")

    out, seen, budget = [], set(), RAG_EVIDENCE_CHARS
    coverage = {}
    for n in state.get("needs") or []:
        status = {"sufficient": "found", "web": "web_only", "pending": "found"}.get(n["status"], "missing")
        coverage[n["id"]] = {"need": n["desc"], "status": status, "reason": n.get("reason", ""),
                             "relaxed": n.get("relaxed", [])}
        if status == "missing":
            notes.append(f"{n['desc']}:知识库里没有找到足够证据,不要编造")
        if status == "web_only":
            notes.append(f"{n['desc']}:只有网页来源(trust_level=web),未经知识库核验,引用时要注明")
        empty_relax = [r for r in n.get("relaxed", []) if r.startswith("drop")]
        if empty_relax and status != "missing":
            notes.append(f"{n['desc']}:原过滤条件下没有结果,已放宽({'、'.join(empty_relax)}),注意证据的期间 / 公司是否对得上")
        # 判了 missing 的需求,它的知识库证据就是"不相关"的,不交给主 Agent(否则模型会拿博通的数字回答苹果)
        hits = [h for h in n["hits"] if status != "missing" and (status != "web_only" or h["trust_level"] == "web")]
        for h in hits:
            key = h["chunk_id"] or h["source"]
            if key in seen:
                continue
            text = h["text"]
            if len(text) > budget:
                if budget < 300:
                    break
                text = text[:budget] + "…(截断)"
            budget -= len(text)
            seen.add(key)
            out.append({**h, "text": text, "ref": len(out) + 1, "need": n["id"]})
    summary = {
        "type": "summary", "text": "", "source": "", "title": "检索汇总", "score": None,
        "coverage": coverage, "notes": notes, "degraded": state.get("degraded") or [],
        "rounds": state.get("round", 0), "llm_tokens": state.get("llm_tokens", 0),
        "latency_ms": round((time.time() - state["t0"]) * 1000),
    }
    # summary 放在**最前面**(M9 §7.8):主 Agent 的上下文对长工具结果只显示开头一段,
    # 放末尾的话,coverage / notes 这些"哪些没找到、哪些不可靠"的信息恰好最先被截掉。
    summary["evidence_refs"] = [e["ref"] for e in out]
    return {"output": [summary] + out}
