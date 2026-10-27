"""文档解析:原始文件 → 有序的段落块列表(M9 Step 1)。

切块之前必须先"读懂"文档结构,否则只能定长切分(M3 的做法)。
三种格式统一输出 list[Block],每块带:
  - text:      段落 / 表格的纯文本
  - page:      所在页(PDF 是真实页码;SEC HTML 是按分页符数出来的页序)
  - title_path: 所属标题路径,例如 ("Item 7. Management's Discussion…", "Data Center")
  - kind:      "para" | "table"
后续 chunk.build_parents() 只按 title_path 和长度分组,不再关心原始格式。

各格式的结构线索(都是看真实文件定下来的):
  SEC HTML(iXBRL):段落是叶子 <div>;标题 = 全部 span 都是 font-weight:700 的短段落;
    分页 = <hr style="page-break-after:always">;页脚(纯页码、"Table of Contents")丢弃。
  A 股 PDF:PyMuPDF 的 text block ≈ 段落;每页前两块是页眉(公司名+报告名)和页脚("11 / 219");
    标题是"第三节 …" / "九、…" / "(三) …" 三级编号。
  Markdown:# 标题。
"""

import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

# SEC 的 iXBRL 是 XHTML,bs4 会提示"像 XML";按 HTML 解析结果是对的,静音即可。
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class Block:
    text: str
    page: int
    title_path: tuple[str, ...] = field(default_factory=tuple)
    kind: str = "para"


@dataclass
class ParsedDoc:
    blocks: list[Block]
    pages: int
    meta: dict = field(default_factory=dict)  # 从文档里读到的元数据(如 iXBRL 的财年)


# ---------------------------------------------------------------- SEC HTML

# "Item 7." / "Item 7. MD&A" / AMD 10-Q 的裸 "Item 2"(后面什么都没有)都要认
_ITEM_RE = re.compile(r"^Item\s+\d+[A-C]?(\.|\s|$)", re.I)
_PART_RE = re.compile(r"^PART\s+I{1,3}V?\b", re.I)
_NOISE_RE = re.compile(r"^(\d{1,3}|Table of Contents|NVIDIA Corporation and Subsidiaries)$", re.I)
_BLOCK_TAGS = {"div", "p", "table", "hr"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


_SIZE_RE = re.compile(r"font-size:\s*([\d.]+)pt")


def _size(style: str) -> float | None:
    m = _SIZE_RE.search(style or "")
    return float(m.group(1)) if m else None


def _heading_level(div: Tag, body_size: float) -> int:
    """0 = 正文;1 = 顶层标题;2 = 小标题。

    各家排版不一样:NVDA/AMD 的标题是加粗(font-weight:700)、字号同正文;
    Intel 的标题不加粗,而是 12pt 蓝字(正文 9pt)。所以判据是"≥80% 的字符
    加粗,或者比正文大 ≥2pt";比正文大 ≥5pt 的算顶层(Intel/Micron 的大节标题)。
    """
    total = styled = 0
    biggest = 0.0
    for s in div.find_all("span"):
        n = len(_norm(s.get_text()))
        if not n:
            continue
        style = s.get("style") or ""
        size = _size(style) or body_size
        total += n
        biggest = max(biggest, size)
        if "font-weight:700" in style or size >= body_size + 2:
            styled += n
    if not total or styled < 0.8 * total:
        return 0
    return 1 if biggest >= body_size + 5 else 2


def _table_text(table: Tag) -> str:
    """表格逐行展平成 "单元格 | 单元格"。财报的数字大多在表里,不能丢。"""
    rows = []
    for tr in table.find_all("tr"):
        cells = [_norm(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in ("$", "%", ")")]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_sec_html(path: Path) -> ParsedDoc:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    meta = {}
    for name, key in (("dei:DocumentFiscalYearFocus", "fiscal_year"), ("dei:DocumentFiscalPeriodFocus", "fiscal_period")):
        tag = soup.find(attrs={"name": name})
        if tag:
            meta[key] = _norm(tag.get_text())
    for hidden in soup.find_all(["ix:header"]):  # iXBRL 隐藏头,不是正文
        hidden.decompose()
    sizes = Counter(sz for s in soup.find_all("span") if (sz := _size(s.get("style"))))
    body_size = sizes.most_common(1)[0][0] if sizes else 10.0  # 出现最多的字号 = 正文

    blocks: list[Block] = []
    page = 1
    item, sub = "", ""

    def visit(node: Tag) -> None:
        nonlocal page, item, sub
        for child in node.children:
            if isinstance(child, NavigableString) or not isinstance(child, Tag):
                continue
            if child.name == "hr":
                if "page-break" in (child.get("style") or ""):
                    page += 1
                continue
            if child.name == "table":
                text = _table_text(child)
                if text and not _ITEM_RE.match(text):  # 目录表里全是 "Item 1. …",它不是正文
                    blocks.append(Block(text, page, _path(item, sub), "table"))
                continue
            if child.name in _BLOCK_TAGS and not child.find(list(_BLOCK_TAGS)):
                text = _norm(child.get_text(" "))
                if not text or _NOISE_RE.match(text) or _PART_RE.match(text):
                    continue
                level = _heading_level(child, body_size) if len(text) <= 150 else 0
                if level:
                    if _ITEM_RE.match(text) or level == 1:
                        item, sub = text, ""
                    else:
                        # Intel 的 10-K 正文不用 "Item X." 当标题(只在交叉索引表里出现),
                        # 所以没见过 Item 也要记小标题,否则整篇 section 全空(§6.2)。
                        sub = text
                    continue
                blocks.append(Block(text, page, _path(item, sub)))
                continue
            visit(child)

    visit(soup.body or soup)
    return ParsedDoc(blocks, page, meta)


def _path(*parts: str) -> tuple[str, ...]:
    return tuple(p for p in parts if p)


# ---------------------------------------------------------------- A 股 PDF

_ZH_L1 = re.compile(r"^第[一二三四五六七八九十]+节\s*\S")
_ZH_L2 = re.compile(r"^[一二三四五六七八九十]+、\S")
_ZH_L3 = re.compile(r"^[(（][一二三四五六七八九十]+[)）]\s*\S")
_ZH_FOOTER = re.compile(r"^\d+\s*/\s*\d+$")
_CJK = "一-鿿，。、；：“”（）《》"
_CJK_JOIN = re.compile(rf"(?<=[{_CJK}])[ \t]*\n[ \t]*(?=[{_CJK}])")
_ZH_NOISE = re.compile(r"^[□√]?适用[□√]?不适用$|^单位[::]")


def parse_cn_pdf(path: Path) -> ParsedDoc:
    import pymupdf

    doc = pymupdf.open(path)
    header = ""
    blocks: list[Block] = []
    l1 = l2 = l3 = ""
    for pno in range(doc.page_count):
        for b in doc[pno].get_text("blocks"):
            raw = b[4].strip()
            # 中文 PDF 的行内换行只是排版折行:两侧都是中文才无缝拼,否则补空格。
            # 表格块里一行一个数字,无缝拼会把 "6,497,196,198.68" 和 "1,174,464,377.35" 粘成一串。
            text = re.sub(r"\s*\n\s*", " ", _CJK_JOIN.sub("", raw))
            if not text:
                continue
            if pno == 0 and not header:
                header = text  # 第一块通常就是页眉"公司名 + 报告名"
            if text == header or _ZH_FOOTER.match(text) or _ZH_NOISE.match(text) or "....." in text:
                continue
            if len(text) <= 40 and not text.endswith("。"):
                if _ZH_L1.match(text):
                    l1, l2, l3 = re.sub(r"^(第\S+节)\s*", r"\1 ", text), "", ""
                    continue
                if _ZH_L2.match(text):
                    l2, l3 = text, ""
                    continue
                if _ZH_L3.match(text):
                    l3 = text
                    continue
            blocks.append(Block(text, pno + 1, _path(l1, l2, l3)))
    return ParsedDoc(blocks, doc.page_count)


# ---------------------------------------------------------------- Markdown

def parse_markdown(path: Path) -> ParsedDoc:
    blocks: list[Block] = []
    h1 = h2 = ""
    for para in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        para = para.strip()
        if not para:
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", para)
        if m and "\n" not in para:
            if len(m.group(1)) == 1:
                h1, h2 = m.group(2), ""
            else:
                h2 = m.group(2)
            continue
        blocks.append(Block(para, 1, _path(h1, h2)))
    return ParsedDoc(blocks, 1)


def parse(path: Path) -> ParsedDoc:
    suffix = path.suffix.lower()
    if suffix in (".htm", ".html"):
        return parse_sec_html(path)
    if suffix == ".pdf":
        return parse_cn_pdf(path)
    if suffix == ".md":
        return parse_markdown(path)
    raise ValueError(f"不支持的格式:{path}")
