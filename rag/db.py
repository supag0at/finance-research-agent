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
    """插入或更新一篇文档,但**先不写 content_hash**。

    返回 (doc_id, changed):
        changed=True 表示是新文档或内容有变化(需要重新切块/向量化);
        changed=False 表示内容没变(可跳过,幂等)。

    ⚠️ M6 修的 bug:**幂等标记必须最后写**。

    原来这个函数在这里就把 content_hash 提交了,然后调用方才去做 embedding 和写向量。
    一旦 embedding 中途失败(实测常见:境外链路抖动),这篇文档已经被标记成"已入库",
    但 chunks 和 vectors 一个都没写。下次重跑会 `跳过(内容未变化)`,
    **这篇文档永久检索不到,而且不报任何错**。

    这是"先宣布成功,再去做事"的典型错误 —— 也正是业界说的
    "accepted-but-not-yet-applied" 那种最难查的状态:
    系统认为世界已经改变了,其实没有。

    改法:把 content_hash 置空(或旧值不动)落盘,等 chunk + 向量都写成功之后,
    由调用方调 `mark_ingested()` 再写哈希。这样中途失败的结果是"下次重做",
    而不是"永久跳过"。
    """
    h = content_hash(content)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, content_hash FROM fra_documents WHERE source = %s", (source,))
        row = cur.fetchone()

        if row is None:
            # content_hash 先写成空串 = "正文已存,但还没完成入库"
            cur.execute(
                "INSERT INTO fra_documents (source, title, content, content_hash) "
                "VALUES (%s, %s, %s, '') RETURNING id",
                (source, title, content),
            )
            doc_id = cur.fetchone()[0]
            conn.commit()
            return doc_id, True

        doc_id, old_hash = row
        if old_hash == h:
            return doc_id, False  # 内容没变且上次确实入库成功了,幂等跳过

        # 内容变了(或上次入库没跑完,old_hash 是空串):
        # 更新正文、清空哈希、清掉旧 chunk,等调用方重建完再盖章。
        cur.execute(
            "UPDATE fra_documents SET title=%s, content=%s, content_hash='' WHERE id=%s",
            (title, content, doc_id),
        )
        cur.execute("DELETE FROM fra_chunks WHERE doc_id=%s", (doc_id,))
        conn.commit()
        return doc_id, True


def mark_ingested(doc_id: int, content: str) -> None:
    """盖章:确认这篇文档的 chunk + 向量都写成功了,幂等标记才生效。

    必须在 `store.upsert()` 和 `insert_chunks()` **都成功之后**才调用。
    这就是"程序先确认结果,再宣布成功"。
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE fra_documents SET content_hash=%s WHERE id=%s",
            (content_hash(content), doc_id),
        )
        conn.commit()


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


# ---------------------------------------------------------------- M9:v2 表
#
# 为什么另起 fra_v2_* 而不是给 fra_documents 加列:幂等哈希按 source 记。
# glossary 的三篇 .md 两条管线都要入库,共用一张表会让 v2 看到"v1 已入库"而跳过。
# v1 表原样保留,作为消融对照(M9 §4.2 第 0 行)。

def init_schema_v2() -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fra_v2_documents (
                id           SERIAL PRIMARY KEY,
                source       TEXT UNIQUE NOT NULL,   -- 相对 raw 目录的路径
                ticker       TEXT,
                company      TEXT,
                doc_type     TEXT,                   -- 10-K / 10-Q / 年报 / 半年报 / glossary
                fiscal_year  INTEGER,
                fiscal_period TEXT,                  -- FY / Q2 / H1
                period_end   DATE,
                title        TEXT,
                source_url   TEXT,
                trust_level  TEXT,                   -- official / internal / web
                lang         TEXT,
                pages        INTEGER,
                content_hash TEXT NOT NULL,          -- 原始文件字节的哈希;'' = 未完成入库
                created_at   TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fra_v2_parents (
                id          TEXT PRIMARY KEY,
                doc_id      INTEGER REFERENCES fra_v2_documents(id) ON DELETE CASCADE,
                section     TEXT,
                page_start  INTEGER,
                page_end    INTEGER,
                text        TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fra_v2_chunks (
                id          TEXT PRIMARY KEY,        -- 与 Qdrant 点 id 一致
                doc_id      INTEGER REFERENCES fra_v2_documents(id) ON DELETE CASCADE,
                parent_id   TEXT REFERENCES fra_v2_parents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page        INTEGER,
                text        TEXT NOT NULL
            )
            """
        )
        conn.commit()


_V2_META = ("ticker", "company", "doc_type", "fiscal_year", "fiscal_period", "period_end",
            "title", "source_url", "trust_level", "lang", "pages")


def upsert_document_v2(source: str, raw_hash: str, meta: dict) -> tuple[int, bool]:
    """语义同 upsert_document(M6 ⑥):先清哈希、删旧父子块,成功后由 mark_ingested_v2 盖章。"""
    vals = [meta.get(k) for k in _V2_META]
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, content_hash FROM fra_v2_documents WHERE source = %s", (source,))
        row = cur.fetchone()
        if row is not None and row[1] == raw_hash:
            return row[0], False
        if row is None:
            cur.execute(
                f"INSERT INTO fra_v2_documents (source, {', '.join(_V2_META)}, content_hash) "
                f"VALUES (%s, {', '.join(['%s'] * len(_V2_META))}, '') RETURNING id",
                (source, *vals),
            )
            doc_id = cur.fetchone()[0]
        else:
            doc_id = row[0]
            cur.execute(
                f"UPDATE fra_v2_documents SET {', '.join(f'{k}=%s' for k in _V2_META)}, content_hash='' "
                "WHERE id=%s",
                (*vals, doc_id),
            )
            cur.execute("DELETE FROM fra_v2_parents WHERE doc_id=%s", (doc_id,))  # 级联删 chunks
        conn.commit()
        return doc_id, True


def insert_parents_children_v2(doc_id: int, parents: list, children: list) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO fra_v2_parents (id, doc_id, section, page_start, page_end, text) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(p.id, doc_id, p.section, p.page_start, p.page_end, p.text) for p in parents],
        )
        cur.executemany(
            "INSERT INTO fra_v2_chunks (id, doc_id, parent_id, chunk_index, page, text) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(c.id, doc_id, c.parent_id, c.index, c.page, c.text) for c in children],
        )
        conn.commit()


def mark_ingested_v2(doc_id: int, raw_hash: str) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE fra_v2_documents SET content_hash=%s WHERE id=%s", (raw_hash, doc_id))
        conn.commit()


def get_parents_v2(parent_ids: list[str]) -> dict[str, dict]:
    """small-to-big 用:子块命中后按 parent_id 取父块原文。"""
    if not parent_ids:
        return {}
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, section, page_start, page_end, text FROM fra_v2_parents WHERE id = ANY(%s)",
            (parent_ids,),
        )
        return {r[0]: {"section": r[1], "page_start": r[2], "page_end": r[3], "text": r[4]} for r in cur.fetchall()}
