"""生成检索评测的 golden set(M9 Step 3)。会调 LLM(DeepSeek,几毛钱)。

    python -m eval.rag_golden            # 生成 eval/rag_golden.jsonl + 抽检单 eval/reports/rag_golden_review.md

做法:从 v2 库里分层抽子块 → 让 LLM 针对这一块出一道题,并给出**原文里逐字出现的答案片段**(answer_span)
→ 程序校验片段确实在原文里、且题目没把答案泄露出来 → 落盘。

**相关性怎么判(关键设计)**:不是"出题用的那个块才算对",而是
    同一篇文档里、正文包含 answer_span 的块都算相关(空白全部去掉后比较)。
好处:① 同一个数字在年报里出现两次(摘要 + MD&A),两处都算对,不冤枉检索;
      ② 规则只依赖文本,所以**换一种切块方式(第 0 行的定长切块)也能用同一份题目算指标**。
局限:另一篇文档里出现同一数字(比如 10-Q 的上年同期列)不算相关——对所有配置一视同仁。

题型:
  fact      单块事实(数字 / 叙述),70~80 道。SEC 的块一半出中文题、一半英文题(用户多半用中文问英文财报)。
  compare   两家公司各一道 fact 合成一道对比题,相关集合 = 两道的并集,看能不能两边都捞到。
  no_answer 问库里没有的公司,相关集合为空;不进 Recall/MRR,用来看"什么都不相关时 rerank 最高分有多高"(拒答阈值)。

合成题的已知偏差:LLM 看着原文出题,措辞容易贴近原文,天然偏向 BM25。prompt 里要求改写,
英文块出中文题也能缓解,但不能消除——解读消融结果时要记得这一点(§6.4)。
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.llm import LLM
from rag import db

OUT = Path("eval/rag_golden.jsonl")
REVIEW = Path("eval/reports/rag_golden_review.md")
SEED = 20261022
PER_DOC = {"10-K": 3, "10-Q": 3, "年报": 5, "半年报": 3, "glossary": 1}
N_COMPARE = 10

_PROMPT = """你在为一个金融知识库构造检索评测题。下面是知识库里的一段原文,以及它的出处。

【出处】{title} | {section} | 第 {page} 页
【原文】
{text}

请针对这段原文出一道题,要求:
1. 题目用{lang}写,像真实用户提问;必须写明公司和期间(如"英伟达 2026 财年""寒武纪 2025 年"),
   否则脱离上下文无法回答。
2. 只靠这段原文就能回答;优先问具体数字,其次问原因/驱动因素。
3. 用自己的话改写,不要照抄原文句子;题目里不能出现答案本身(尤其是答案数字)。
4. answer_span 必须是原文里**逐字出现**的一个短片段(不超过 40 个字符),包含答案的关键信息(通常就是那个数字)。
5. 如果这段原文是封面、目录、法律套话、签名、纯表头这类没有实质信息的内容,usable 设为 false。{focus}

只输出 JSON:
{{"usable": true, "question": "...", "answer": "一句话答案", "answer_span": "原文逐字片段", "kind": "number 或 narrative"}}"""

_COMPARE_PROMPT = """下面是两道关于不同公司的题目。先判断:两道题问的是不是**同一类、拿来对比有意义**的指标
(比如都是收入增速、都是毛利率、都是研发投入、都是某业务板块收入)。

题目 A:{qa}
题目 B:{qb}

如果可比,把它们合成一道自然的中文对比题(写明两家公司和各自期间,回答需要同时用到两道题的答案,题目里不能出现答案);
如果不可比(比如一个问债务、一个问商誉减值测试时间),comparable 设为 false。

只输出 JSON:{{"comparable": true, "question": "..."}}"""

NO_ANSWER = [
    "苹果公司 2025 财年服务业务收入是多少?",
    "特斯拉 2025 年全年汽车交付量是多少?",
    "高通 2025 财年 QCT 部门收入同比增长多少?",
    "台积电 2025 年 3 纳米制程收入占比是多少?",
    "华为 2025 年研发投入占收入的比例是多少?",
    "What was Samsung Electronics' memory division operating profit in 2025?",
    "比亚迪 2025 年新能源汽车销量是多少?",
    "Arm Holdings fiscal 2026 royalty revenue growth?",
    "小米 2025 年智能手机出货量是多少?",
    "What was Meta's capital expenditure guidance for 2026?",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _sample_chunks() -> list[dict]:
    with db._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.text, c.page, p.section, d.source, d.title, d.doc_type, d.ticker, d.fiscal_year, d.lang "
            "FROM fra_v2_chunks c JOIN fra_v2_parents p ON p.id = c.parent_id JOIN fra_v2_documents d ON d.id = c.doc_id "
            "WHERE length(c.text) >= 200"
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    rng = random.Random(SEED)
    by_doc: dict[str, list[dict]] = {}
    for r in rows:
        by_doc.setdefault(r["source"], []).append(r)
    picked = []
    for source in sorted(by_doc):
        chunks = by_doc[source]
        n = PER_DOC.get(chunks[0]["doc_type"], 2)
        # 第一版纯随机抽,出来的多是"审计委员会哪天开会""养老金计息利率"这种没人会问的题。
        # 现在 70% 从分析师真正看的小节抽(MD&A / 业务 / 主要财务指标),30% 全库随机保留长尾。
        core = [c for c in chunks if _CORE_SECTION.search(c["section"] or "")]
        n_core = round(n * 0.7) if core else 0
        # 多抽一倍备用:LLM 判 usable=false 或校验失败的会被丢弃
        sel = rng.sample(core, min(len(core), n_core * 2))
        rest = [c for c in chunks if c not in sel]
        sel += rng.sample(rest, min(len(rest), (n - n_core) * 2))
        picked += sel
    return picked


_CORE_SECTION = re.compile(
    r"Item 7\.|Item 2\. Management|Management.s Discussion|Results of Operations|Item 1\. Business|"
    r"Fiscal Year 20\d\d Summary|Overview|Segment|Data Center|"
    r"第二节|第三节|主要会计数据|主要财务指标|经营情况|主营业务|核心竞争力|研发",
    re.I,
)


def _make_fact(llm: LLM, ch: dict, lang: str, focus: str = "") -> dict | None:
    """focus:对比题用,强制题目问某个指标(否则给一段含 gross margin 的原文,LLM 常去问运营费用)。"""
    focus_rule = f"\n6. 题目必须问原文中关于「{focus}」的具体数字;原文里没有这个指标的数字就把 usable 设为 false。" if focus else ""
    try:
        out = llm.complete_json([{"role": "user", "content": _PROMPT.format(
            title=ch["title"], section=ch["section"] or "(无)", page=ch["page"], text=ch["text"],
            lang="中文" if lang == "zh" else "英文", focus=focus_rule)}])
    except Exception:  # noqa: BLE001 —— 单题失败就跳过
        return None
    span = (out.get("answer_span") or "").strip()
    q = (out.get("question") or "").strip()
    if not out.get("usable") or not span or not q:
        return None
    if _norm(span) not in _norm(ch["text"]):        # 片段必须逐字在原文里
        return None
    if len(_norm(span)) >= 4 and _norm(span) in _norm(q):  # 题目泄露答案
        return None
    return {
        "kind": "fact", "fact_kind": out.get("kind", ""), "lang": lang, "question": q,
        "answer": out.get("answer", ""), "spans": [{"source": ch["source"], "span": span, "chunk_id": ch["id"]}],
        "gen_chunk_id": ch["id"], "gen_title": ch["title"], "gen_page": ch["page"], "ticker": ch["ticker"],
        "fiscal_year": ch["fiscal_year"],
    }


def main() -> None:
    llm = LLM()
    chunks = _sample_chunks()
    jobs = []
    for i, ch in enumerate(chunks):
        lang = "zh" if ch["lang"] == "zh" else ("zh" if i % 2 == 0 else "en")
        jobs.append((ch, lang))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda j: _make_fact(llm, *j), jobs))

    # 每篇文档只留 PER_DOC 道
    facts, per_source = [], {}
    for ch, r in zip(chunks, results):
        if r is None:
            continue
        quota = PER_DOC.get(ch["doc_type"], 2)
        if per_source.get(ch["source"], 0) >= quota:
            continue
        per_source[ch["source"]] = per_source.get(ch["source"], 0) + 1
        facts.append(r)
    print(f"抽了 {len(chunks)} 个块,出题成功 {sum(r is not None for r in results)},按配额留下 {len(facts)} 道 fact")

    compares, tries = _make_compares(llm)

    no_answer = [{"kind": "no_answer", "lang": "en" if q[0].isascii() else "zh", "question": q, "spans": []}
                 for q in NO_ANSWER]
    items = facts + compares + no_answer
    for i, it in enumerate(items):
        it["id"] = f"g{i:03d}"
    OUT.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n", encoding="utf-8")
    print(f"对比题:{len(COMPARE_TOPICS)} 个主题,尝试 {tries} 次,成功 {len(compares)} 道")
    print(f"共 {len(items)} 道:fact {len(facts)} / compare {len(compares)} / no_answer {len(no_answer)} → {OUT}")
    _write_review(facts, compares)


# 对比题按"公司对 × 主题"定向构造。第一版从已有 fact 里随机配对,300 次尝试 0 对可比
# (一道问存货、一道问股息,对比没有意义)。现在每个主题用关键词在 Postgres 里各找一块,
# 再各出一道 fact 题、合成对比题。用关键词选块会略偏向 BM25,解读时注意(§6.4)。
COMPARE_TOPICS = [
    ("NVDA", "AMD", ["Data Center", "revenue"]),
    ("NVDA", "AVGO", ["gross margin"]),
    ("AMD", "INTC", ["research and development"]),
    ("MU", "INTC", ["gross margin"]),
    ("AVGO", "MU", ["research and development"]),
    ("NVDA", "MU", ["revenue", "increase"]),
    ("688256", "688041", ["营业收入", "同比"]),
    ("688981", "688041", ["毛利率"]),
    ("688256", "688981", ["研发投入"]),
    ("AMD", "AVGO", ["net revenue", "increase"]),
]


def _topic_chunks(ticker: str, words: list[str], rng: random.Random) -> list[dict]:
    cond = " AND ".join(["c.text ILIKE %s"] * len(words))
    with db._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.text, c.page, p.section, d.source, d.title, d.doc_type, d.ticker, d.fiscal_year, d.lang "
            "FROM fra_v2_chunks c JOIN fra_v2_parents p ON p.id = c.parent_id JOIN fra_v2_documents d ON d.id = c.doc_id "
            f"WHERE d.ticker = %s AND length(c.text) >= 200 AND {cond}",
            (ticker, *[f"%{w}%" for w in words]),
        )
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    core = [r for r in rows if _CORE_SECTION.search(r["section"] or "")] or rows
    rng.shuffle(core)
    return core[:4]


def _make_compares(llm: LLM) -> tuple[list[dict], int]:
    rng = random.Random(SEED + 2)
    compares, tries = [], 0
    for ta, tb, words in COMPARE_TOPICS:
        sides = []
        for t in (ta, tb):
            fact = None
            for ch in _topic_chunks(t, words, rng):
                tries += 1
                fact = _make_fact(llm, ch, "zh", focus=" ".join(words))
                if fact and fact["fact_kind"] == "number":
                    break
                fact = None
            sides.append(fact)
        if None in sides:
            continue
        a, b = sides
        try:
            out = llm.complete_json([{"role": "user", "content": _COMPARE_PROMPT.format(qa=a["question"], qb=b["question"])}])
        except Exception:  # noqa: BLE001
            continue
        if not out.get("comparable") or not out.get("question"):
            continue
        compares.append({"kind": "compare", "lang": "zh", "question": out["question"],
                         "answer": f"{a['answer']} / {b['answer']}", "spans": a["spans"] + b["spans"],
                         "parts": [a["question"], b["question"]], "topic": words})
    return compares, tries


def _write_review(facts: list[dict], compares: list[dict]) -> None:
    """人工抽检单:fact 抽 20%(至少 15 道)+ 全部对比题。勾选后把通过率写回设计文档 §6.4。"""
    rng = random.Random(SEED + 1)
    sample = rng.sample(facts, max(15, len(facts) // 5))
    with db._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, text FROM fra_v2_chunks WHERE id = ANY(%s)", ([f["gen_chunk_id"] for f in sample],))
        text_of = dict(cur.fetchall())
    lines = ["# M9 golden set 人工抽检单", "",
             "逐题看三件事:① 题目只靠原文能答;② 题目写明了公司和期间;③ answer_span 确实是答案的关键信息。",
             "三条都满足就把 `[ ]` 改成 `[x]`;不满足在后面写原因。", ""]
    for f in sample:
        lines += [f"## {f['id']}  [ ] 通过", "",
                  f"- **题目**({f['lang']}):{f['question']}",
                  f"- **答案**:{f['answer']}",
                  f"- **answer_span**:`{f['spans'][0]['span']}`",
                  f"- **出处**:{f['gen_title']} 第 {f['gen_page']} 页", "",
                  "```", text_of.get(f["gen_chunk_id"], "")[:1200], "```", ""]
    lines += ["# 对比题(看合成得自然不自然、是否真的需要两边信息)", ""]
    for c in compares:
        lines += [f"- [ ] **{c['id']}** {c['question']}", f"  - 拆开:{c['parts'][0]} | {c['parts'][1]}"]
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"抽检单 {len(sample)} 道 fact + {len(compares)} 道对比 → {REVIEW}")


if __name__ == "__main__":
    main()
