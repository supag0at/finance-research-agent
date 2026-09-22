"""短期记忆:单次会话内的"上下文压缩"。

问题:Agent 每转一圈,messages 就变长(尤其工具返回的大段文本),迟早撑爆
上下文窗口、又贵又慢。
做法(auto-compact 思路):当 messages 超过阈值时,把较早的历史用 LLM 压成
一段"摘要",只保留 system + 摘要 + 最近若干条。

一个关键工程细节:OpenAI 消息格式里,每条 role="tool" 的消息必须紧跟在带
tool_calls 的 assistant 消息之后(配对关系)。压缩时不能让"最近窗口"以一条
孤立的 tool 消息开头,否则 API 会报错。下面 _safe_recent() 专门处理这个。
"""

from typing import Optional


def _truncate_tool_messages(messages: list[dict], max_tool_chars: int) -> list[dict]:
    """把过长的工具返回内容就地截断(单条工具结果的裁剪)。"""
    out = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "tool" and isinstance(content, str) and len(content) > max_tool_chars:
            m = {
                **m,
                "content": content[:max_tool_chars]
                + f"\n...[工具结果过长,已截断,原长 {len(content)} 字]",
            }
        out.append(m)
    return out


def _render_for_summary(messages: list[dict]) -> str:
    """把要压缩的历史渲染成纯文本,喂给摘要器。"""
    lines = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if m.get("tool_calls"):
            names = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
            lines.append(f"[assistant 决定调用工具: {names}] {content}")
        elif role == "tool":
            lines.append(f"[工具返回] {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def summarize_history(llm, messages: list[dict]) -> str:
    """用 LLM 把一段历史压成要点。"""
    prompt = [
        {
            "role": "system",
            "content": "你是对话压缩器。把给定的调研过程压成简洁要点,必须保留:"
            "已查到的关键数据与数字、已得出的结论、尚未完成的任务。不要编造任何内容。",
        },
        {"role": "user", "content": "请把以下调研历史压缩为要点:\n\n" + _render_for_summary(messages)},
    ]
    return llm.complete(prompt)


def compress_messages(
    messages: list[dict],
    llm=None,
    trigger_len: int = 12,
    keep_recent: int = 6,
    max_tool_chars: int = 1500,
) -> list[dict]:
    """压缩消息列表(供发给 LLM 用)。

    参数:
        messages: 完整对话消息。
        llm: 摘要器(有则用 LLM 生成摘要;为 None 则只做窗口+裁剪,不摘要)。
        trigger_len: 消息条数超过它才触发压缩。
        keep_recent: 保留最近多少条原始消息。
        max_tool_chars: 单条工具结果最多保留多少字符。
    返回:
        压缩后的消息列表。
    """
    messages = _truncate_tool_messages(messages, max_tool_chars)
    if len(messages) <= trigger_len:
        return messages  # 还不长,只做了工具结果裁剪

    # 分离最前面的 system 提示
    system_msg = None
    body = messages
    if messages and messages[0].get("role") == "system":
        system_msg = messages[0]
        body = messages[1:]

    recent = body[-keep_recent:]
    old = body[:-keep_recent]

    # 避免"最近窗口"以孤立的 tool 消息开头(破坏 assistant->tool 配对):
    # 把开头这些 tool 消息挪回 old(它们会被一起摘要掉)。
    while recent and recent[0].get("role") == "tool":
        old = old + [recent[0]]
        recent = recent[1:]

    if not old:
        return messages

    result = []
    if system_msg:
        result.append(system_msg)

    # 摘要要调一次 LLM,是会失败的(实测:工具持续失败导致历史暴涨后,
    # 摘要请求收到空响应)。摘要失败不该让整轮调研崩掉 —— 和长期记忆一样,
    # 记忆是锦上添花,不该阻断主流程。所以降级成"省略历史"继续跑。
    try:
        summary = summarize_history(llm, old) if llm is not None else "(历史较长已省略)"
    except Exception as exc:  # noqa: BLE001
        summary = f"(历史较长,且摘要生成失败:{type(exc).__name__},此处已省略)"
    result.append({"role": "system", "content": "【之前调研历史的摘要】\n" + summary})
    result.extend(recent)

    # TODO(记忆增强,见 memory/MEMORY.md):可加更精细的 token 预算(用 tiktoken 估算)、
    #   分型记忆、反思/巩固、把摘要缓存起来避免每圈重复摘要等。
    return result
