"""RRF(Reciprocal Rank Fusion,倒数排名融合)——M9 Step 2。纯函数,无 IO。

    score(d) = Σ_每一路排名 1 / (k + rank_i(d))      rank 从 1 开始;某一路没出现就不加

为什么不直接把分数加起来:dense 是余弦(0~1),BM25 是无上界的正数(实测十几到几十),
量纲不同,直接加等于让 BM25 说了算。RRF 只看名次,不看分值,所以不用做归一化。

k 的作用:k 越大,第 1 名和第 10 名的差距越小(越"平均主义")。
k=60 时第 1 名 1/61≈0.0164,第 10 名 1/70≈0.0143——只差 13%,
所以"两路都排中上"的块,会胜过"一路第 1、另一路没出现"的块。
"""


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rank_lists:每一路是按相关度从高到低排好的 id 列表。返回 [(id, rrf 分)],从高到低。

    同分时按"第一次出现的位置"稳定排序(先出现在靠前那一路、靠前名次的优先),
    保证同样的输入永远得到同样的输出(评测可复现)。
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for li, ids in enumerate(rank_lists):
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(doc_id, (rank, li))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
