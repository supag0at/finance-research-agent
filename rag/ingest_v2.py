"""v2 离线建库管线(M9 Step 1)。

    python -m rag.ingest_v2

和 v1(rag/ingest.py)的区别:
  - 语料:SEC 10-K/10-Q(HTML)+ A 股年报/半年报(PDF)+ 原来的 3 篇术语 .md;
  - 解析:rag/parse.py 读出页码和标题层级,而不是把整篇当一串字符;
  - 切块:父子块(rag/chunk.build_parents / build_children);
  - 元数据:ticker / doc_type / fiscal_year / section / page / trust_level …,建 payload 索引;
  - 向量:named vector "dense";"sparse" 留到 Step 2 写入;
  - 存储:Qdrant finance_corpus_v2 + Postgres fra_v2_*。v1 的集合和表都不动(消融对照)。

幂等沿用 M6 ⑥ + M9 缺陷 10 的顺序:清哈希 → 解析/切块/向量化 → 删旧点 → 写新点 → 写 Postgres → 盖章。

contextual chunk header:子块拿去向量化时,前面拼一行"公司 文档类型 财年 | 小节"。
原因:子块正文里通常没有公司名和年份(10-K 里只写 "we"、"fiscal year 2026"),
不拼的话 "英伟达 FY2026 数据中心收入" 和 AMD 10-K 里同样措辞的段落在向量上分不开。
payload 里的 text 仍是原文,header 只影响向量。
"""

import hashlib
import json
import time
from pathlib import Path

from rag import db
from rag.chunk import build_children, build_parents
from rag.config import CORPUS_RAW_DIR, EMBED_BATCH, EMBED_DIM, QDRANT_COLLECTION_V2
from rag.embedding import embed_texts
from rag.parse import parse
from rag.store import QdrantStore

# 解析 / 切块逻辑一改,就把版本号加一。幂等哈希 = 原始文件字节 + 这个版本号:
# 只哈希原文的话,修完解析器重跑会"内容未变化"全部跳过,旧的错误切块永远留在库里。
PIPELINE_VERSION = "v2.1"  # v2.1:SEC 标题识别改为"加粗或大字号"(Intel 修复,§6.2)

COMPANY = {"NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel", "AVGO": "Broadcom", "MU": "Micron"}
GLOSSARY_DIR = Path("data/corpus")
PAYLOAD_INDEXES = {
    "source": "keyword", "ticker": "keyword", "doc_type": "keyword", "trust_level": "keyword",
    "lang": "keyword", "fiscal_year": "integer",
}


def discover() -> list[tuple[Path, str, dict]]:
    """列出要入库的文档:(路径, source 键, 来自 manifest 的元数据)。"""
    raw = Path(CORPUS_RAW_DIR)
    docs = []
    for sub in ("sec", "cninfo"):
        mpath = raw / sub / "manifest.json"
        if not mpath.exists():
            continue
        for fname, meta in json.loads(mpath.read_text(encoding="utf-8")).items():
            docs.append((raw / sub / fname, f"{sub}/{fname}", dict(meta)))
    for f in sorted(GLOSSARY_DIR.glob("*.md")):
        first = f.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        docs.append((f, f"glossary/{f.name}", {
            "doc_type": "glossary", "title": first, "trust_level": "internal", "lang": "zh",
        }))
    return docs


def _complete_meta(meta: dict, parsed_meta: dict, pages: int) -> dict:
    m = dict(meta)
    m["pages"] = pages
    if m.get("doc_type") in ("10-K", "10-Q"):
        m["company"] = COMPANY.get(m["ticker"], m["ticker"])
        fy = parsed_meta.get("fiscal_year")
        m["fiscal_year"] = int(fy) if fy and fy.isdigit() else int(m["period_end"][:4])
        m["fiscal_period"] = parsed_meta.get("fiscal_period") or ("FY" if m["doc_type"] == "10-K" else "")
        tail = "" if m["fiscal_period"] == "FY" else f" {m['fiscal_period']}"
        m["title"] = f"{m['company']} {m['doc_type']} FY{m['fiscal_year']}{tail}"
    elif m.get("doc_type") in ("年报", "半年报"):
        m["fiscal_period"] = "FY" if m["doc_type"] == "年报" else "H1"
        m["title"] = f"{m['company']} {m['fiscal_year']}年{m['doc_type']}"
    return m


def _header(meta: dict, section: str) -> str:
    if meta["doc_type"] == "glossary":
        return f"术语表 | {section}"
    who = meta.get("company", "")
    return f"{who} {meta['doc_type']} FY{meta.get('fiscal_year', '')} {meta.get('fiscal_period', '')} | {section}".strip()


def _embed_all(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        for attempt in range(3):
            try:
                out += embed_texts(batch)
                break
            except Exception:  # noqa: BLE001 —— 境外链路抖动常见,重试 3 次再放弃
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    return out


def ingest_one(store: QdrantStore, path: Path, source: str, meta: dict) -> tuple[int, int] | None:
    raw_hash = hashlib.sha256(path.read_bytes() + PIPELINE_VERSION.encode()).hexdigest()
    parsed = parse(path)
    meta = _complete_meta(meta, parsed.meta, parsed.pages)
    doc_id, changed = db.upsert_document_v2(source, raw_hash, meta)
    if not changed:
        return None

    lang = meta.get("lang", "en")
    parents = build_parents(parsed.blocks, source, lang)
    children = build_children(parents, source, lang)
    section_of = {p.id: p.section for p in parents}
    vectors = _embed_all([f"{_header(meta, section_of[c.parent_id])}\n{c.text}" for c in children])

    points = [
        {
            "id": c.id,
            "vector": {"dense": v},
            "payload": {
                "text": c.text, "doc_id": doc_id, "parent_id": c.parent_id, "chunk_index": c.index,
                "section": section_of[c.parent_id], "page": c.page, "source": source,
                **{k: meta.get(k) for k in ("ticker", "company", "doc_type", "fiscal_year", "fiscal_period",
                                            "period_end", "title", "source_url", "trust_level", "lang")},
            },
        }
        for c, v in zip(children, vectors)
    ]
    store.delete_by_filter({"must": [{"key": "source", "match": {"value": source}}]})  # 缺陷 10
    for i in range(0, len(points), 256):
        store.upsert(points[i : i + 256])
    db.insert_parents_children_v2(doc_id, parents, children)
    db.mark_ingested_v2(doc_id, raw_hash)  # 盖章必须最后(M6 ⑥)
    return len(parents), len(children)


def ingest_all() -> None:
    db.init_schema_v2()
    store = QdrantStore(collection=QDRANT_COLLECTION_V2)
    store.ensure_collection_v2(EMBED_DIM)
    for field, schema in PAYLOAD_INDEXES.items():
        store.ensure_index(field, schema)

    t0 = time.time()
    total_p = total_c = 0
    docs = discover()
    for path, source, meta in docs:
        t = time.time()
        res = ingest_one(store, path, source, meta)
        if res is None:
            print(f"跳过(内容未变化):{source}")
            continue
        total_p += res[0]
        total_c += res[1]
        print(f"入库:{source:<40} 父块 {res[0]:>4}  子块 {res[1]:>5}  {time.time() - t:5.1f}s")
    print(f"完成:{len(docs)} 篇,新增/更新 父块 {total_p} / 子块 {total_c},"
          f"集合现有 {store.count()} 个点,用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ingest_all()
