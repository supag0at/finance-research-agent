"""文本切块(chunking)。

为什么要切块?一篇文档可能几千上万字,直接整篇向量化会:
  1. 稀释语义——一个向量代表太多内容,检索不精准;
  2. 超出 embedding 模型的输入上限。
所以把长文切成一段段"语义相对完整"的小块,分别向量化。

overlap(重叠)是关键细节:相邻块之间留一点重叠,避免把一句话/一个论点
从中间劈开,导致检索时上下文断裂。
"""


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
