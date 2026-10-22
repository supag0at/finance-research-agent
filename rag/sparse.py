"""稀疏向量(BM25)——M9 Step 2,混合检索的"关键词"那一路。

为什么要它:dense 向量擅长"意思相近",但对专名、型号、年份、具体数字不敏感。
Step 1 的冒烟检索里,封面页 "Washington, D.C. 20549" 靠 contextual header 和
"NVIDIA FY2026 数据中心收入" 的余弦相似度打平第 1 名——它根本不含 "data center revenue"。
BM25 只认词,这种块在关键词那一路排不上来。

BM25 打分 = Σ_词 idf(词) × tf 项(词):
    tf 项 = tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))      ← 本文件在客户端算好,写进 sparse values
    idf   = ln((N − n + 0.5)/(n + 0.5) + 1)                 ← Qdrant 集合开了 modifier=idf,服务端乘
查询侧每个词的值都是 1,于是 Qdrant 的点积正好等于 BM25 分数(Step 0 已实测 idf 公式对得上)。

分词:
  - 英文单词 / 型号 / 数字先用正则抽出来(数字去掉千分位逗号:"6,497,196,198.68" → "6497196198.68"),
    否则 jieba 会把 "205,922.85" 切成 "205" "," "922.85";
  - 剩下的中文片段交给 jieba 搜索模式(会给出重叠的子词,召回更好),公司名提前加进词典。
词 → 稀疏维度用 crc32 哈希(uint32),词表不用存;碰撞概率在万级词量下可忽略。
"""

import json
import logging
import math
import re
import zlib
from collections import Counter
from pathlib import Path

import jieba

jieba.setLogLevel(logging.WARNING)
for _w in ("英伟达", "寒武纪", "中芯国际", "海光信息", "博通", "美光", "英特尔", "超威半导体", "数据中心"):
    jieba.add_word(_w)

K1 = 1.2
B = 0.75
STATS_PATH = Path("data/rag_sparse_stats.json")

_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*"  # 单词 / 型号:blackwell、h100、gb200、10-k、u.s
    r"|\d[\d,]*(?:\.\d+)?"                       # 数字:2026、115.2、6,497,196,198.68
    r"|[一-鿿]+"                         # 中文片段,交给 jieba
)
_STOP = {
    # 英文:高频虚词,BM25 的 idf 本来就会压低它们,去掉只是为了少写点向量
    "the", "of", "and", "to", "in", "a", "an", "for", "on", "our", "we", "is", "was", "are", "were",
    "by", "with", "as", "or", "be", "that", "this", "it", "its", "from", "at", "which", "may", "not",
    # 中文
    "的", "了", "和", "与", "及", "是", "在", "对", "等", "或", "其", "为",
}


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if "一" <= tok[0] <= "鿿":
            out += [t for t in jieba.lcut_for_search(tok) if t not in _STOP]
        elif tok[0].isdigit():
            out.append(tok.replace(",", ""))
        else:
            low = tok.lower()
            if low not in _STOP:
                out.append(low)
    return out


def _index(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


def _to_sparse(weights: dict[int, float]) -> dict:
    idx = sorted(weights)
    return {"indices": idx, "values": [round(weights[i], 4) for i in idx]}


def doc_vector(text: str, avgdl: float) -> dict:
    """文档侧:每个词写 BM25 的 tf 项(含长度归一)。"""
    toks = tokenize(text)
    if not toks:
        return {"indices": [], "values": []}
    dl = len(toks)
    norm = K1 * (1 - B + B * dl / avgdl)
    weights: dict[int, float] = {}
    for tok, tf in Counter(toks).items():
        i = _index(tok)
        weights[i] = weights.get(i, 0.0) + tf * (K1 + 1) / (tf + norm)
    return _to_sparse(weights)


def query_vector(text: str) -> dict:
    """查询侧:每个不同的词值为 1(查询里重复的词不额外加权)。"""
    return _to_sparse({_index(t): 1.0 for t in set(tokenize(text))})


def bm25_tf(tf: int, dl: int, avgdl: float) -> float:
    """给测试 / 讲解用:单个词的 tf 项。"""
    return tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))


def idf(n_total: int, n_with_term: int) -> float:
    """给测试 / 讲解用:和 Qdrant modifier=idf 同一个公式。"""
    return math.log((n_total - n_with_term + 0.5) / (n_with_term + 0.5) + 1)


def save_stats(avgdl: float, n_docs: int) -> None:
    STATS_PATH.write_text(json.dumps({"avgdl": avgdl, "n_docs": n_docs}, indent=2), encoding="utf-8")


def load_avgdl(default: float = 200.0) -> float:
    """增量入库时用已存的 avgdl;没有统计文件就退回默认值(只影响长度归一的力度)。"""
    try:
        return float(json.loads(STATS_PATH.read_text(encoding="utf-8"))["avgdl"])
    except (OSError, ValueError, KeyError):
        return default
