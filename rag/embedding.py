"""Embedding:把文本变成向量(用 SiliconFlow 的 bge-m3)。

这是 RAG 三层里的第①层——"翻译器",负责把文字翻译成一串数字(向量),
之后 Qdrant 才能对这些向量做相似度检索。

SiliconFlow 提供 OpenAI 兼容的接口,所以我们直接复用 openai 客户端,只是
换个 base_url 和 model。用独立的 httpx client 并 trust_env=False,是为了让
这个域名的请求走直连(它是国内 API,没必要绕到国外代理)。
"""

import httpx
from openai import OpenAI

from rag.config import EMBED_API_KEY, EMBED_BASE_URL, EMBED_MODEL

# trust_env=False:忽略系统代理(Clash),直连国内 API,更快更稳。
_client = OpenAI(
    api_key=EMBED_API_KEY,
    base_url=EMBED_BASE_URL,
    http_client=httpx.Client(trust_env=False, timeout=30),
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量把文本转成向量。返回的顺序和输入一一对应。"""
    if not texts:
        return []
    resp = _client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def embed_query(text: str) -> list[float]:
    """把单条查询文本转成向量(检索时用)。"""
    return embed_texts([text])[0]
