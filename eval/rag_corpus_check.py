"""v2 语料体检(M9 Step 1 验收)。只读,不调 embedding。

    python -m eval.rag_corpus_check

检查:
  1. 每篇文档:Qdrant 点数 == Postgres 子块数,且已盖章(content_hash 非空);
  2. 每个点都有 page / section / fiscal_year(术语表除外)/ trust_level;
  3. 抽一个子块,打印它的 payload 和所属父块,肉眼看对不对。
"""

import sys

from rag import db
from rag.config import QDRANT_COLLECTION_V2
from rag.store import QdrantStore

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def _scroll_all(store: QdrantStore) -> list[dict]:
    points, offset = [], None
    while True:
        body = {"limit": 2000, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = store.http.post(f"/collections/{store.collection}/points/scroll", json=body).json()["result"]
        points += r["points"]
        offset = r.get("next_page_offset")
        if offset is None:
            return points


def main() -> None:
    store = QdrantStore(collection=QDRANT_COLLECTION_V2)
    points = _scroll_all(store)
    with db._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT d.source, d.title, d.content_hash <> '', d.pages, "
            "(SELECT count(*) FROM fra_v2_parents p WHERE p.doc_id = d.id), "
            "(SELECT count(*) FROM fra_v2_chunks c WHERE c.doc_id = d.id) "
            "FROM fra_v2_documents d ORDER BY d.source"
        )
        docs = cur.fetchall()

    per_source: dict[str, int] = {}
    for p in points:
        per_source[p["payload"]["source"]] = per_source.get(p["payload"]["source"], 0) + 1

    print("① 每篇文档的点数对账")
    print(f"  {'source':<38} {'title':<24} {'页':>4} {'父块':>5} {'子块':>5} {'Qdrant':>6}")
    for source, title, stamped, pages, n_par, n_chunk in docs:
        q = per_source.get(source, 0)
        mark = "" if (stamped and q == n_chunk) else "  ← 不一致"
        print(f"  {source:<38} {title:<24} {pages:>4} {n_par:>5} {n_chunk:>5} {q:>6}{mark}")
    bad = [d[0] for d in docs if not d[2] or per_source.get(d[0], 0) != d[5]]
    check(f"{len(docs)} 篇全部盖章且 Qdrant == Postgres", not bad, f"不一致:{bad}")
    orphan = set(per_source) - {d[0] for d in docs}
    check("Qdrant 里没有 Postgres 不认识的 source", not orphan, str(orphan))
    check("文档数 ≥ 20", len(docs) >= 20, str(len(docs)))

    print("② payload 完整性")
    no_page = [p for p in points if not p["payload"].get("page")]
    no_section = [p for p in points if not p["payload"].get("section")]
    no_fy = [p for p in points if p["payload"]["doc_type"] != "glossary" and not p["payload"].get("fiscal_year")]
    no_trust = [p for p in points if not p["payload"].get("trust_level")]
    check(f"page 全有({len(points)} 个点)", not no_page, f"缺 {len(no_page)}")
    check("fiscal_year 全有(术语表除外)", not no_fy, f"缺 {len(no_fy)}")
    check("trust_level 全有", not no_trust, f"缺 {len(no_trust)}")
    share = len(no_section) / max(1, len(points))
    # 封面、目录前的说明性文字本来就不属于任何小节,允许少量为空
    check(f"section 为空的点 ≤ 3%(实际 {share:.1%})", share <= 0.03, f"{len(no_section)} 个")

    print("③ 抽样")
    key = sys.argv[1] if len(sys.argv) > 1 else "Data Center revenue for fiscal year 2026"
    hit = next((p for p in points if key in p["payload"]["text"]), None)
    if hit:
        pl = hit["payload"]
        print("  子块 payload:")
        for k in ("title", "ticker", "doc_type", "fiscal_year", "fiscal_period", "section", "page",
                  "trust_level", "parent_id", "source_url"):
            print(f"    {k:<13} {pl.get(k)}")
        print(f"    text({len(pl['text'])} 字符) {pl['text'][:200]!r}")
        par = db.get_parents_v2([pl["parent_id"]])[pl["parent_id"]]
        siblings = sum(1 for p in points if p["payload"]["parent_id"] == pl["parent_id"])
        print(f"  父块:页 {par['page_start']}-{par['page_end']},{len(par['text'])} 字符,含 {siblings} 个子块")

    print()
    print("全部通过" if not _failures else f"失败 {len(_failures)} 项:{_failures}")
    raise SystemExit(1 if _failures else 0)


if __name__ == "__main__":
    main()
