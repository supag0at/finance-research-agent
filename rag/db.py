"""关系库封装(Postgres)——RAG 三层里的第③层。

它存"原文的结构化信息":文档原文、来源、标题、时间,以及切出来的 chunk。
作用:
  1. 作为"事实来源(source of truth)":Qdrant 里存的是向量,原始内容以这里为准。
  2. 支持去重/幂等:同一篇文档内容没变就不重复入库。
  3. 未来可做结构化过滤(按来源、时间、股票代码筛选)。

为避免和你旧项目的表冲突,我们所有表都加 fra_ 前缀(finance-research-agent)。
"""

import hashlib

import psycopg

from rag.config import PG_DSN


def _connect() -> psycopg.Connection:
    # 本地直连,不走代理。
    return psycopg.connect(PG_DSN, connect_timeout=5)


def init_schema() -> None:
    """建表(若不存在)。fra_documents 存文档,fra_chunks 存切块。"""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fra_documents (
                id          SERIAL PRIMARY KEY,
                source      TEXT UNIQUE NOT NULL,   -- 来源(文件名/URL),用于去重
                title       TEXT,
                content     TEXT NOT NULL,
                content_hash TEXT NOT NULL,          -- 内容哈希,判断是否变化
                created_at  TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fra_chunks (
                id          TEXT PRIMARY KEY,        -- 与 Qdrant 点 id 一致(UUID)
                doc_id      INTEGER REFERENCES fra_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL
            )
            """
        )
        conn.commit()


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def upsert_document(source: str, title: str, content: str) -> tuple[int, bool]:
    """插入或更新一篇文档。

    返回 (doc_id, changed):
        changed=True 表示是新文档或内容有变化(需要重新切块/向量化);
        changed=False 表示内容没变(可跳过,幂等)。
    """
    h = content_hash(content)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, content_hash FROM fra_documents WHERE source = %s", (source,))
        row = cur.fetchone()

        if row is None:
            cur.execute(
                "INSERT INTO fra_documents (source, title, content, content_hash) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (source, title, content, h),
            )
            doc_id = cur.fetchone()[0]
            conn.commit()
            return doc_id, True

        doc_id, old_hash = row
        if old_hash == h:
            return doc_id, False  # 内容没变,幂等跳过

        # 内容变了:更新文档,并清掉旧 chunk(级联由调用方重建)
        cur.execute(
            "UPDATE fra_documents SET title=%s, content=%s, content_hash=%s WHERE id=%s",
            (title, content, h, doc_id),
        )
        cur.execute("DELETE FROM fra_chunks WHERE doc_id=%s", (doc_id,))
        conn.commit()
        return doc_id, True


def insert_chunks(rows: list[tuple[str, int, int, str]]) -> None:
    """批量插入 chunk。每行:(chunk_id, doc_id, chunk_index, text)。"""
    if not rows:
        return
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO fra_chunks (id, doc_id, chunk_index, text) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text",
            rows,
        )
        conn.commit()
