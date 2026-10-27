"""归档:把溢出的若干圈交给 LLM,换回每圈一句要点 + 可核验的事实。

只追加:每次只为**新溢出的圈**生成条目,旧条目从不重写。
对比 legacy 的"旧摘要 + 新历史 → 新摘要":那种做法每压一次就再摘要一次,
信息一层层丢失(摘要的摘要);而且输入随历史线性增长,这就是 101 秒的来源。
"""

from langchain_core.messages import ToolMessage

from core.failure import target_of
from memory.context.rounds import Round
from memory.context.tokens import text_of
from memory.context.types import ArchiveEntry

FALLBACK_GIST = "(归档要点生成失败,仅保留调用记录)"
READBACK_TOOL = "read_tool_result"
MAX_GIST_CHARS = 120
MAX_LABEL_CHARS = 80

ARCHIVE_PROMPT = """你是调研归档器。下面是一次金融调研过程中较早的若干"圈"(每圈 = 一次工具调用决策 + 工具返回)。
你的输出会替代这些原文留在上下文里,所以要做两件事:

1. 为每一圈写一句不超过 80 字的要点:拿到了什么、对回答用户问题有什么用;失败的圈写清楚失败了什么。
2. 抽取与用户问题相关的、可核验的关键事实(数字、日期、明确的结论)。

硬性要求:
- 只能使用给定文本,不要补充任何你自己知道的信息;
- value 必须原样照抄原文中的数字与单位,不要换算、不要四舍五入;
- source_id 必须是给定的 id 之一(写在"[结果 id=…]"里);
- 没有可抽取的事实就返回空列表。

只输出一个 json 对象,格式:
{"entries": [{"round": 圈号, "gist": "要点"}],
 "facts": [{"entity": "主体,如 NVDA", "metric": "指标", "period": "时期,如 FY26Q2", "value": "原文中的值", "source_id": "id"}]}"""


def call_label(tool_call: dict) -> str:
    """'fetch_page(https://…)':目标用 target_of 取,与失败记录、循环检测同一判据。"""
    target = target_of(tool_call.get("args") or {})
    label = f"{tool_call.get('name', '?')}({target})"
    return label if len(label) <= MAX_LABEL_CHARS else label[: MAX_LABEL_CHARS - 1] + "…)"


def _calls(r: Round) -> list[str]:
    return [call_label(tc) for tc in (r.ai.tool_calls if r.ai is not None else None) or []]


def render_rounds(rounds: list[Round], max_tool_chars: int) -> str:
    parts = []
    for r in rounds:
        lines = [f"### R{r.no}"]
        if r.ai is not None:
            thought = text_of(r.ai.content).strip()
            if thought:
                lines.append(f"[模型的想法] {thought[:500]}")
            for tc in r.ai.tool_calls or []:
                lines.append(f"[调用 id={tc.get('id', '')}] {call_label(tc)}")
        for t in r.tools:
            text = text_of(t.content)
            more = f"\n…(后面还有 {len(text) - max_tool_chars} 字未展示)" if len(text) > max_tool_chars else ""
            lines.append(f"[结果 id={t.tool_call_id}]\n{text[:max_tool_chars]}{more}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _entry(r: Round, gist: str) -> ArchiveEntry:
    # 回读结果自己的 id 不列出来:它只是原文的一个窗口,标签里已经带着原始 id。
    # A/B 实测发现:列出来之后模型拿它去回读,读到的是一份 1678 字的"回读结果",以为原文到头了。
    ids = [t.tool_call_id for t in r.tools if t.name != READBACK_TOOL]
    return {"round": r.no, "ids": ids, "calls": _calls(r), "gist": gist}


def archive_rounds(llm, rounds: list[Round], question: str, max_tool_chars: int) -> tuple[list[ArchiveEntry], list]:
    """一次 LLM 调用。返回 (归档条目, 未校验的事实候选)。JSON 不合规时抛 ValueError,由调用方降级。"""
    prompt = [
        {"role": "system", "content": ARCHIVE_PROMPT},
        {
            "role": "user",
            "content": f"用户问题:{question}\n\n需要归档的圈:\n\n{render_rounds(rounds, max_tool_chars)}",
        },
    ]
    data = llm.complete_json(prompt)
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"归档结果缺少 entries 列表: {str(data)[:120]!r}")

    gists: dict[int, str] = {}
    for e in raw_entries:
        if isinstance(e, dict) and isinstance(e.get("gist"), str):
            try:
                gists[int(e.get("round"))] = e["gist"].strip()[:MAX_GIST_CHARS]
            except (TypeError, ValueError):
                continue
    facts = data.get("facts")
    return [_entry(r, gists.get(r.no, "")) for r in rounds], facts if isinstance(facts, list) else []


def fallback_entries(rounds: list[Round]) -> list[ArchiveEntry]:
    """归档 LLM 失败时的确定性条目:没有要点,但调用记录和 id 都在,仍可回读。"""
    return [_entry(r, FALLBACK_GIST) for r in rounds]


def tool_sources(rounds: list[Round]) -> dict[str, str]:
    return {t.tool_call_id: text_of(t.content) for r in rounds for t in r.tools if isinstance(t, ToolMessage)}
