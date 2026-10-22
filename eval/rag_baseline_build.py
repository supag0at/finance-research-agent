"""建消融第 0 行的基线集合(M9 Step 3):**同一批 24 篇文档**,按 v1 的方式切块入库。

    python -m eval.rag_baseline_build      # 约 15k 块,~10 分钟 embedding

为什么不直接用旧集合 finance_corpus:它只有 3 篇 AI 生成的 .md,拿新 golden set 去测必然全军覆没,
那比较的是"语料",不是"方法"。第 0 行要回答的是:**同样的语料,M3 那套切法和检索能做到多少**。

v1 的做法:整篇正文(这里用 rag.parse 读出来的段落按顺序拼起来)→ 定长 500 / overlap 100 → 直接向量化
(**不拼 header**,v1 没有)→ 匿名单向量集合,余弦。
"""

import time

from rag.chunk import chunk_text
from rag.config import EMBED_DIM
from rag.ingest_v2 import _embed_all, discover
from rag.parse import parse
from rag.store import QdrantStore

BASELINE_COLLECTION = "finance_corpus_v1fixed"


def main() -> None:
    store = QdrantStore(collection=BASELINE_COLLECTION)
    store.drop_collection()
    store.ensure_collection(EMBED_DIM)
    t0 = time.time()
    total = 0
    for path, source, _ in discover():
        full = "\n".join(b.text for b in parse(path).blocks)
        chunks = chunk_text(full, size=500, overlap=100)
        vectors = _embed_all(chunks)
        points = [
            {"id": total + i, "vector": v, "payload": {"text": c, "source": source, "chunk_index": i}}
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        for i in range(0, len(points), 256):
            store.upsert(points[i : i + 256])
        total += len(chunks)
        print(f"{source:<40} {len(chunks):>5} 块  累计 {total}  {time.time() - t0:.0f}s", flush=True)
    print(f"完成:{total} 块 → {BASELINE_COLLECTION}")


if __name__ == "__main__":
    main()
