"""离线建库管线(offline ingestion)。

这就是我们讲过的 RAG "离线阶段":
  读文档 → 切块 → 向量化(embedding) → 存进 Qdrant(向量)+ Postgres(原文)。

它是一个独立、可重复运行的脚本。带幂等:内容没变的文档会自动跳过。
运行:  python -m rag.ingest
"""

import uuid
from pathlib import Path

from rag import db
from rag.chunk import chunk_text
from rag.config import EMBED_DIM
from rag.embedding import embed_texts
from rag.store import QdrantStore

# 固定命名空间,保证同一 (source, index) 每次生成相同的 chunk id(可重复入库)。
_NAMESPACE = uuid.UUID("8f14e45f-ceea-4e0a-9d2b-000000000001")


def _chunk_id(source: str, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{source}#{index}"))


def ingest_corpus(corpus_dir: str = "data/corpus") -> None:
    db.init_schema()
    store = QdrantStore()
    store.ensure_collection(EMBED_DIM)
    store.ensure_index("source")  # 按 source 删旧点要用(幂等)

    files = sorted(Path(corpus_dir).glob("*.md"))
    if not files:
        print(f"没有找到语料文件:{corpus_dir}/*.md")
        return

    total_chunks = 0
    for f in files:
        content = f.read_text(encoding="utf-8")
        source = f.name
        first_line = content.splitlines()[0] if content.strip() else source
        title = first_line.lstrip("# ").strip() or source

        doc_id, changed = db.upsert_document(source, title, content)
        if not changed:
            print(f"跳过(内容未变化):{source}")
            continue

        # 切块 -> 批量向量化
        # TODO(RAG增强,见 rag/ENHANCEMENTS.md ①建库侧):可换成语义切块 /
        #   结构化切块 / 父子块,并给 payload 附更多元数据(时间、标的)用于过滤。
        chunks = chunk_text(content)
        vectors = embed_texts(chunks)

        # 组装 Qdrant 的点 + Postgres 的行
        points, rows = [], []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            cid = _chunk_id(source, i)
            points.append(
                {
                    "id": cid,
                    "vector": vector,
                    "payload": {
                        "text": chunk,
                        "source": source,
                        "title": title,
                        "chunk_index": i,
                    },
                }
            )
            rows.append((cid, doc_id, i, chunk))

        # ⚠️ M9 缺陷 10:先按 source 删掉这篇文档在 Qdrant 里的旧点,再写新点。
        # chunk id = uuid5(source#index),upsert 只会覆盖"同号"的块;
        # 新版本块数变少时(10 块 → 8 块),第 8、9 号旧点没人覆盖,会一直被检索到。
        # Postgres 那边 upsert_document 已经删过旧 chunk,这里补上向量库这一侧。
        # 放在 embedding 成功之后:embedding 失败时旧点还在,至少不会"整篇检索不到"。
        store.delete_by_filter({"must": [{"key": "source", "match": {"value": source}}]})
        store.upsert(points)  # 向量进 Qdrant
        db.insert_chunks(rows)  # 原文进 Postgres
        # ⚠️ 顺序要紧:幂等标记必须在上面两步都成功之后才盖(M6 ⑥)。
        # 改造前是 upsert_document 一进来就把 content_hash 提交了,
        # 结果 embed_texts 一失败,这篇文档就被永久标记为"已入库"却没有任何 chunk。
        # 现在中途失败的后果是"下次重跑会重做",而不是"永久跳过且不报错"。
        db.mark_ingested(doc_id, content)
        total_chunks += len(chunks)
        print(f"入库:{source}  →  {len(chunks)} 块")

    print(f"完成:处理 {len(files)} 篇文档,新增/更新 {total_chunks} 个 chunk。")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    ingest_corpus()
