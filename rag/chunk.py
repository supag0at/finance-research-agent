"""文本切块(chunking)。

为什么要切块?一篇文档可能几千上万字,直接整篇向量化会:
  1. 稀释语义——一个向量代表太多内容,检索不精准;
  2. 超出 embedding 模型的输入上限。
所以把长文切成一段段"语义相对完整"的小块,分别向量化。

overlap(重叠)是关键细节:相邻块之间留一点重叠,避免把一句话/一个论点
从中间劈开,导致检索时上下文断裂。
"""

import re
import uuid
from dataclasses import dataclass, field


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """把长文本按字符数切成带重叠的小块。

    参数:
        text: 原始文本。
        size: 每块大约多少字符。
        overlap: 相邻块之间重叠多少字符。
    返回:
        文本块列表。
    """
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        return [text]

    step = max(1, size - overlap)  # 每次前进的步长
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------- M9:父子块(small-to-big)
#
# 定长切块的问题:不管文档结构,一刀切在小节中间;块小了上下文不够,块大了检索不准。
# 父子块把这两个需求拆开:
#   父块 = 同一小节内连续段落拼到 ~PARENT 字符,永不跨小节 —— 交给 LLM 看的单位;
#   子块 = 父块内部再切到 ~CHILD 字符 —— 拿去向量化 / BM25 的单位。
# 检索命中子块,返回它的父块。
#
# 尺寸按语言分:英文 1 token ≈ 4.5 字符,中文 1 token ≈ 1.4 字符,
# 两种配置下子块都在 ~220~350 token,父块 ~700~1000 token。

SIZES = {
    "en": {"parent": 3000, "child": 1000},
    "zh": {"parent": 1500, "child": 500},
}

MIN_CHILD = 20  # "None." "否" "(Continued)" 这类碎块不值得进索引

_SENT_SPLIT = re.compile(r"(?<=[。！？；.!?;])\s*")


@dataclass
class Parent:
    id: str
    section: str          # "Item 7. MD&A > Data Center"
    page_start: int
    page_end: int
    text: str
    units: list = field(default_factory=list)  # [(text, page)],给子块用


@dataclass
class Child:
    id: str
    parent_id: str
    index: int            # 文档内全局序号
    page: int
    text: str


def _split_long(text: str, limit: int) -> list[str]:
    """超长段落 / 表格先按句子(表格按行)拆开,再按 limit 打包。"""
    pieces = text.split("\n") if "\n" in text else _SENT_SPLIT.split(text)
    out, buf = [], ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        while len(p) > limit:  # 单句超长(极少见):按长度强制切分
            out.append(p[:limit])
            p = p[limit:]
        if buf and len(buf) + len(p) + 1 > limit:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        out.append(buf)
    return out


_NAMESPACE = uuid.UUID("8f14e45f-ceea-4e0a-9d2b-000000000009")


def _stable_id(*parts) -> str:
    """同一 (source, 类型, 序号) 永远得到同一个 UUID(Qdrant 点 id 只收 UUID / 整数)。"""
    return str(uuid.uuid5(_NAMESPACE, "#".join(map(str, parts))))


def build_parents(blocks: list, source: str, lang: str) -> list[Parent]:
    """blocks 来自 rag.parse;同一 title_path 的连续段落拼成父块。"""
    limit = SIZES[lang]["parent"]
    parents: list[Parent] = []
    cur_path, units = None, []

    def flush() -> None:
        if not units:
            return
        text = "\n".join(t for t, _ in units)
        parents.append(
            Parent(
                id=_stable_id(source, "P", len(parents)),
                section=" > ".join(cur_path or ()),
                page_start=units[0][1],
                page_end=units[-1][1],
                text=text,
                units=list(units),
            )
        )
        units.clear()

    for b in blocks:
        if b.title_path != cur_path:
            flush()
            cur_path = b.title_path
        for piece in _split_long(b.text, limit) if len(b.text) > limit else [b.text]:
            if units and sum(len(t) for t, _ in units) + len(piece) > limit:
                flush()
            units.append((piece, b.page))
    flush()
    return parents


def build_children(parents: list[Parent], source: str, lang: str) -> list[Child]:
    """父块内部按段落打包到 ~CHILD 字符;相邻子块重叠上一子块的最后一段(若够短)。"""
    limit = SIZES[lang]["child"]
    children: list[Child] = []
    for p in parents:
        units = []
        for t, page in p.units:
            units += [(s, page) for s in (_split_long(t, limit) if len(t) > limit else [t])]
        buf: list = []
        for u in units:
            if buf and sum(len(t) for t, _ in buf) + len(u[0]) > limit:
                _append(children, _make_child(buf, p, source, len(children)))
                tail = buf[-1]
                buf = [tail] if len(tail[0]) <= limit * 0.2 else []  # overlap:短尾段带到下一块
            buf.append(u)
        if buf:
            _append(children, _make_child(buf, p, source, len(children)))
    return children


def _append(children: list, child: "Child | None") -> None:
    if child is not None:
        children.append(child)


def _make_child(buf: list, parent: Parent, source: str, index: int) -> Child | None:
    text = "\n".join(t for t, _ in buf)
    if len(text.strip()) < MIN_CHILD:
        return None
    return Child(
        id=_stable_id(source, "C", index),
        parent_id=parent.id,
        index=index,
        page=buf[0][1],
        text=text,
    )


def context_header(meta: dict, section: str) -> str:
    """contextual chunk header:"公司 文档类型 财年 期间 | 小节"。

    子块正文里通常只有 "we"、"fiscal year 2026",没有公司名;拼上这一行再向量化 / 建 BM25 /
    交给 rerank,才分得清 NVDA 和 AMD 里同样措辞的段落。meta 可以是入库时的元数据,也可以是 Qdrant payload。
    """
    if meta.get("doc_type") == "glossary":
        return f"术语表 | {section}"
    who = meta.get("company", "")
    return f"{who} {meta['doc_type']} FY{meta.get('fiscal_year', '')} {meta.get('fiscal_period', '')} | {section}".strip()
