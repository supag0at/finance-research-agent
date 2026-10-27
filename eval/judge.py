"""LLM-as-Judge(模型裁判)——用一个 LLM 按评分标准给回答打分。

处理"规则写不出来"的主观维度:准确性、完整性、是否忠于事实。
关键工程技巧:
  1. 强制模型输出结构化 JSON(便于程序解析、汇总)。
  2. 把工具返回的真实数据当作"事实依据"喂给裁判,减少它自己凭空判断。
  3. 要求裁判给出打分理由(可解释、可复查),而不是只给一个数字。
"""

import json

from core.llm import LLM

# 每个维度都是 1-5 分。评分标准写清楚,裁判才稳定。
_JUDGE_PROMPT = """你是一个严格的评审。请根据【事实依据】评价【回答】对【问题】的质量。
只依据【事实依据】判断数字/事实是否正确,不要用你自己的先验知识去补充或否定。

【问题】
{question}

【期望覆盖的要点】
{expect_points}

【事实依据(Agent 调用工具获得的真实返回,可能被截断)】
{facts}

【回答】
{answer}

请从三个维度打分(1-5 的整数,5 最好),并给出简短理由:
- accuracy(准确性):回答中的事实/数字是否与【事实依据】一致。
- completeness(完整性):是否覆盖了【期望覆盖的要点】。
- faithfulness(忠实度/反编造):回答是否只基于【事实依据】,没有编造依据中不存在的数字或事实。

评分注意:
- 【事实依据】可能很长且被截断,请聚焦"回答里的关键结论/数字是否有依据支撑",
  不要因为某个细节在截断的依据里没找到就轻易判低分。
- 若是"概念解释类"问题,回答中用于举例说明的假设数字(如"假设股价100、每股收益10")
  属于合理的教学示例,不算编造。

只输出如下 JSON,不要有任何多余文字:
{{"accuracy": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "reason": "<50字以内理由>"}}
"""

# 最终安全阀:即便上游已做去重+限长(见 checks.build_judge_facts),这里再兜底截一次。
_MAX_FACTS_CHARS = 30000


def _parse_json(text: str) -> dict:
    """从模型输出里稳健地提取 JSON(容忍 ```json 代码块包裹或前后多余文本)。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 包裹
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # 取第一个 { 到最后一个 } 之间的内容
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def judge(
    llm: LLM,
    question: str,
    answer: str,
    expect_points: list[str],
    facts: str,
) -> dict:
    """让 LLM 裁判给一道题的回答打分。失败时返回 error 字段,不抛异常。"""
    points_text = "\n".join(f"- {p}" for p in expect_points) if expect_points else "(无特定要点)"
    facts_text = (facts or "(无工具返回)")[:_MAX_FACTS_CHARS]

    prompt = _JUDGE_PROMPT.format(
        question=question,
        expect_points=points_text,
        facts=facts_text,
        answer=answer,
    )

    try:
        raw = llm.complete([{"role": "user", "content": prompt}])
        data = _parse_json(raw)
        # 规整:确保三个分数都是 1-5 的 int。
        for k in ("accuracy", "completeness", "faithfulness"):
            data[k] = int(data.get(k, 0))
        data["reason"] = str(data.get("reason", ""))[:200]
        return data
    except Exception as e:  # 裁判本身也可能抽风,别让它拖垮整个评测
        return {
            "accuracy": 0,
            "completeness": 0,
            "faithfulness": 0,
            "reason": "",
            "error": f"裁判解析失败: {type(e).__name__}: {e}",
        }
