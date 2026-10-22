"""入库幂等 / 旧点清理的离线测试(M9 Step 0)。零费用、确定性,不需要 Qdrant / Postgres。

    python -m eval.rag_ingest_offline                 # 内存假 Qdrant
    python -m eval.rag_ingest_offline --real-qdrant   # 真 Qdrant 的临时集合(跑完删掉)

Qdrant、Postgres、embedding 全换成内存假实现,测的是 rag/ingest.py 的**编排顺序**:
  1. 文档变短(块数变少)后重入库,向量库里不能残留旧块(M9 缺陷 10);
  2. 内容没变时幂等跳过;
  3. embedding 中途失败后,下次重跑会重做(M6 ⑥ 的回归)。
"""

import sys
import tempfile
from pathlib import Path

from rag import ingest
from rag.config import EMBED_DIM
from rag.store import QdrantStore

REAL = "--real-qdrant" in sys.argv
TMP_COLLECTION = "m9_ingest_test_tmp"

_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'通过' if ok else '失败'}] {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


class FakeStore:
    """只实现 ingest 用到的接口;filter 只支持 must + match.value(够用)。"""

    points: dict[str, dict] = {}

    def __init__(self, *a, **kw) -> None:
        pass

    def ensure_collection(self, dim: int = 0) -> None:
        pass

    def ensure_index(self, field: str, schema: str = "keyword") -> None:
        pass

    def upsert(self, points: list[dict]) -> None:
        for p in points:
            FakeStore.points[p["id"]] = p

    def delete_by_filter(self, flt: dict) -> None:
        conds = flt["must"]
        FakeStore.points = {
            pid: p
            for pid, p in FakeStore.points.items()
            if not all(p["payload"].get(c["key"]) == c["match"]["value"] for c in conds)
        }


class FakeDB:
    """模拟 rag/db.py 的语义:upsert_document 先清哈希,mark_ingested 最后盖章。"""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.chunks: dict[str, tuple] = {}

    def init_schema(self) -> None:
        pass

    def upsert_document(self, source: str, title: str, content: str):
        h = str(hash(content))
        d = self.docs.get(source)
        if d is None:
            d = self.docs[source] = {"id": len(self.docs) + 1, "hash": ""}
            return d["id"], True
        if d["hash"] == h:
            return d["id"], False
        d["hash"] = ""
        self.chunks = {k: v for k, v in self.chunks.items() if v[0] != d["id"]}
        return d["id"], True

    def insert_chunks(self, rows) -> None:
        for cid, doc_id, idx, text in rows:
            self.chunks[cid] = (doc_id, idx, text)

    def mark_ingested(self, doc_id: int, content: str) -> None:
        for d in self.docs.values():
            if d["id"] == doc_id:
                d["hash"] = str(hash(content))


class RealTempStore(QdrantStore):
    """真 Qdrant,但写到临时集合,不碰 finance_corpus。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(collection=TMP_COLLECTION)


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(t))] + [1.0] * (EMBED_DIM - 1) for t in texts]


def all_points() -> dict[str, dict]:
    if not REAL:
        return dict(FakeStore.points)
    store = RealTempStore()
    resp = store.http.post(f"/collections/{TMP_COLLECTION}/points/scroll", json={"limit": 1000, "with_payload": True})
    resp.raise_for_status()
    return {p["id"]: p for p in resp.json()["result"]["points"]}


def _points_of(source: str) -> int:
    return sum(1 for p in all_points().values() if p["payload"]["source"] == source)


def main() -> None:
    fake_db = FakeDB()
    ingest.QdrantStore = RealTempStore if REAL else FakeStore
    if REAL:
        RealTempStore().drop_collection()
    print("模式:", f"真 Qdrant 临时集合 {TMP_COLLECTION}" if REAL else "内存假 Qdrant")
    ingest.db = fake_db
    ingest.embed_texts = fake_embed

    with tempfile.TemporaryDirectory() as d:
        doc = Path(d) / "a.md"
        other = Path(d) / "b.md"
        other.write_text("# B\n" + "y" * 900, encoding="utf-8")  # 旁边的文档,不应被误删

        print("① 文档变短后重入库")
        doc.write_text("# A\n" + "x" * 4000, encoding="utf-8")  # 定长 500/100 → 10 块
        ingest.ingest_corpus(d)
        before = _points_of("a.md")
        other_before = _points_of("b.md")
        doc.write_text("# A\n" + "z" * 3000, encoding="utf-8")  # → 8 块
        ingest.ingest_corpus(d)
        after = _points_of("a.md")
        expect = len(ingest.chunk_text(doc.read_text(encoding="utf-8")))
        check(f"a.md 向量点数 = 新块数({before} → {after},期望 {expect})", after == expect)
        check("旁边文档 b.md 的点没被误删", _points_of("b.md") == other_before)
        stale = [p for p in all_points().values() if p["payload"]["source"] == "a.md" and "x" in p["payload"]["text"]]
        check("没有旧版本文本残留", not stale, f"残留 {len(stale)} 个")

        print("② 内容没变 → 幂等跳过")
        snapshot = all_points()
        ingest.ingest_corpus(d)
        check("点集合完全不变", all_points() == snapshot)

        print("③ embedding 中途失败 → 下次重做")
        doc.write_text("# A\n" + "w" * 2000, encoding="utf-8")

        def boom(texts):
            raise RuntimeError("模拟 embedding 失败")

        ingest.embed_texts = boom
        try:
            ingest.ingest_corpus(d)
        except RuntimeError:
            pass
        ingest.embed_texts = fake_embed
        ingest.ingest_corpus(d)
        texts = [p["payload"]["text"] for p in all_points().values() if p["payload"]["source"] == "a.md"]
        expect = len(ingest.chunk_text(doc.read_text(encoding="utf-8")))
        old = [t for t in texts if "z" in t or "x" in t]
        check(f"重跑后 a.md 只有新内容({len(texts)} 块,期望 {expect})", len(texts) == expect and not old, f"旧内容 {len(old)} 块")

    if REAL:
        RealTempStore().drop_collection()
    print()
    print("全部通过" if not _failures else f"失败 {len(_failures)} 项:{_failures}")
    raise SystemExit(1 if _failures else 0)


if __name__ == "__main__":
    main()
