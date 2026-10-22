"""知识库目录:过滤条件的白名单(M9 Step 4,不变式 3)。

query 分析让 LLM 输出过滤条件(ticker / fiscal_year / doc_type),但 LLM 给的值不可信:
可能拼错("NVIDA"),可能是库里没有的公司("AAPL"),可能是库里没有的年份。
一个错的硬过滤会让结果直接变成 0 条,所以**每个值都要过白名单**——白名单就是库里真实存在的取值。

同一份目录也写进 query 分析的 prompt,让 LLM 知道库里有什么、各家财年怎么对应自然年。
"""

from functools import lru_cache

from rag import db

# 中文 / 英文俗称 → ticker。库里 company 字段是英文名或 A 股简称,用户常用的是这些叫法。
ALIASES = {
    "NVDA": ["英伟达", "NVIDIA", "辉达"],
    "AMD": ["AMD", "超威", "超微半导体"],
    "INTC": ["英特尔", "Intel"],
    "AVGO": ["博通", "Broadcom"],
    "MU": ["美光", "Micron"],
    "688256": ["寒武纪"],
    "688981": ["中芯国际", "SMIC"],
    "688041": ["海光信息", "海光"],
}


@lru_cache(maxsize=1)
def load() -> dict:
    """{ticker: {"company": ..., "docs": [(doc_type, fiscal_year, fiscal_period, period_end), ...]}}。"""
    with db._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, company, doc_type, fiscal_year, fiscal_period, period_end FROM fra_v2_documents "
            "WHERE ticker IS NOT NULL ORDER BY ticker, period_end"
        )
        out: dict = {}
        for ticker, company, doc_type, fy, fp, pe in cur.fetchall():
            out.setdefault(ticker, {"company": company, "docs": []})["docs"].append((doc_type, fy, fp, str(pe)))
    return out


def prompt_text() -> str:
    lines = []
    for ticker, info in load().items():
        names = " / ".join(ALIASES.get(ticker, [info["company"]]))
        docs = ";".join(f"{dt} FY{fy}{'' if fp == 'FY' else ' ' + fp}(截至 {pe})" for dt, fy, fp, pe in info["docs"])
        lines.append(f"- {ticker}({names}):{docs}")
    lines.append("- 另有 3 篇术语 / 行业背景文档(无 ticker,不需要过滤条件)")
    return "\n".join(lines)


def validate(filters: dict | None) -> tuple[dict, list[str]]:
    """去掉白名单外的值。返回 (干净的过滤条件, 被丢掉的说明)。"""
    cat = load()
    filters = filters or {}
    dropped: list[str] = []
    tickers = [t for t in filters.get("ticker") or [] if isinstance(t, str)]
    good_t = [t for t in tickers if t in cat]
    dropped += [f"ticker={t}(库里没有)" for t in tickers if t not in cat]
    scope = good_t or list(cat)  # 年份 / 文档类型按"选中的公司"校验;没选公司就按全库
    valid_fy = {fy for t in scope for _, fy, _, _ in cat[t]["docs"]}
    valid_dt = {dt for t in scope for dt, _, _, _ in cat[t]["docs"]}
    years = [y for y in filters.get("fiscal_year") or [] if isinstance(y, int)]
    good_y = [y for y in years if y in valid_fy]
    dropped += [f"fiscal_year={y}(这些公司没有)" for y in years if y not in valid_fy]
    dts = [d for d in filters.get("doc_type") or [] if isinstance(d, str)]
    good_d = [d for d in dts if d in valid_dt]
    dropped += [f"doc_type={d}(没有)" for d in dts if d not in valid_dt]
    clean = {k: v for k, v in (("ticker", good_t), ("fiscal_year", good_y), ("doc_type", good_d)) if v}
    return clean, dropped


def to_qdrant(filters: dict) -> dict | None:
    must = [{"key": k, "match": {"any": v}} for k, v in filters.items() if v]
    return {"must": must} if must else None
