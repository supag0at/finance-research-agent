"""钉住区事实的校验与合并。

归档器(LLM)抽出的事实不能直接信:摘要器可能擅自补全数字。所以进钉住区之前
用来源工具结果的**原文**逐个核对数字 —— 判据与评测溯源(eval/checks.py)是同一套
(core/numbers.py),压缩放行的数字,评测也一定能找到出处。
"""

from core.numbers import meaningful_numbers, normalize
from memory.context.types import Fact

_FIELDS = ("entity", "metric", "period", "value", "source_id")
MAX_QUALITATIVE_CHARS = 60


def verify_facts(raw: list, sources: dict[str, str]) -> tuple[list[Fact], list[tuple[dict, str]]]:
    """返回 (通过的事实, [(被丢弃的候选, 原因)])。sources 是 {tool_call_id: 原文全文}。"""
    kept: list[Fact] = []
    dropped: list[tuple[dict, str]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            dropped.append(({"raw": str(item)[:80]}, "格式不对"))
            continue
        if not all(isinstance(item.get(k), str) and item[k].strip() for k in _FIELDS):
            dropped.append((item, "字段缺失"))
            continue
        fact: Fact = {k: item[k].strip() for k in _FIELDS}  # type: ignore[assignment]
        source = sources.get(fact["source_id"])
        if source is None:
            dropped.append((item, "来源不存在"))
            continue
        nums = meaningful_numbers(fact["value"])
        if nums:
            src = normalize(source)
            if not all(n in src for n in nums):
                dropped.append((item, "数字不在原文"))
                continue
        elif len(fact["value"]) > MAX_QUALITATIVE_CHARS:
            dropped.append((item, "定性描述过长"))
            continue
        kept.append(fact)
    return kept, dropped


def _key(f: Fact) -> tuple[str, str, str]:
    return f["entity"].upper(), f["metric"], f["period"]


def merge_facts(old: list[Fact], new: list[Fact], limit: int) -> list[Fact]:
    """同一 (主体, 指标, 时期) 新值覆盖旧值,并挪到末尾(视为最新);超过上限丢最旧。"""
    merged = {_key(f): f for f in old}
    for f in new:
        merged.pop(_key(f), None)
        merged[_key(f)] = f
    out = list(merged.values())
    return out[-limit:] if len(out) > limit else out
