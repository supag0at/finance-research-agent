"""程序化规则检查(确定性评估)——不花钱、不调 LLM,纯代码判断客观事实。

为什么要有这一层?因为很多维度根本不需要"裁判"来判,用代码就能确定地回答:
  - 该调的工具到底调没调?(靠 Trajectory)
  - 回答里的数字,能不能在工具返回里找到出处?(反编造/幻觉的第一道防线)
  - 有没有绕圈子(步数/工具调用次数)?

这些检查便宜、稳定、可回归,所以能用规则的就绝不用 LLM。
"""

from core.numbers import meaningful_numbers as _meaningful_numbers
from core.numbers import normalize as _normalize
from core.schema import TrajectoryStep

# 数字规则已迁到 core/numbers.py(M8 的压缩事实校验也要用,且必须与这里同一个判据)。


def collect_observations(trajectory: list[TrajectoryStep]) -> str:
    """把整条轨迹里所有工具返回(observation)拼成一大段,作为"事实来源"。

    这是给"数字溯源"用的:需要完整、原样(仅去千分位逗号),不做删减。
    """
    parts = []
    for s in trajectory:
        if s.observation:
            parts.append(str(s.observation))
    return _normalize("\n".join(parts))


def build_judge_facts(
    trajectory: list[TrajectoryStep],
    max_per_obs: int = 4000,
    max_total: int = 30000,
) -> str:
    """给 LLM 裁判用的"精炼版事实依据"。

    与 collect_observations 不同,这里要控制体量,否则研究型问题(几十次工具调用、
    多篇网页正文)会撑爆裁判上下文,导致它"看不到证据"而误判 faithfulness 偏低:
      1. 去重:相同的工具返回只保留一次(如反复搜同关键词);
      2. 单条限长:一篇超长网页正文截断到 max_per_obs,避免一条挤占全部预算;
      3. 总量限长:所有证据合计不超过 max_total。
    """
    seen: set[str] = set()
    parts: list[str] = []
    total = 0
    for s in trajectory:
        if not s.observation:
            continue
        obs = str(s.observation)
        key = obs[:200]  # 用前 200 字做去重指纹,足够区分不同返回
        if key in seen:
            continue
        seen.add(key)
        if len(obs) > max_per_obs:
            obs = obs[:max_per_obs] + "…(截断)"
        if total + len(obs) > max_total:
            parts.append(obs[: max_total - total])
            break
        parts.append(obs)
        total += len(obs)
    return "\n---\n".join(parts)


def called_tools(trajectory: list[TrajectoryStep]) -> list[str]:
    """轨迹里实际调用过的工具名列表(按出现顺序,可能重复)。"""
    return [s.tool_call.name for s in trajectory if s.tool_call]


def run_checks(case: dict, answer: str, trajectory: list[TrajectoryStep]) -> dict:
    """对一道题的作答做全部程序化检查,返回确定性指标字典。"""
    tools_used = called_tools(trajectory)
    tools_used_set = set(tools_used)

    # (1) 该调的工具调了吗?
    must = case.get("must_call_tools", [])
    missing_tools = [t for t in must if t not in tools_used_set]
    tool_ok = len(missing_tools) == 0

    # (2) 反编造:回答里的"有意义数字"能否在工具返回里找到出处?
    obs_text = collect_observations(trajectory)
    ans_numbers = _meaningful_numbers(answer)
    if ans_numbers:
        grounded = [n for n in ans_numbers if n in obs_text]
        grounded_ratio = round(len(grounded) / len(ans_numbers), 3)
        ungrounded = [n for n in ans_numbers if n not in obs_text]
    else:
        # 回答里没有具体数字(如纯概念解释),不适用该指标,记为 1.0。
        grounded_ratio = 1.0
        ungrounded = []

    # (3) 效率:工具调用次数 & 最大步数。
    tool_call_count = len(tools_used)
    max_step = max((s.step for s in trajectory), default=0)

    return {
        "tool_ok": tool_ok,
        "missing_tools": missing_tools,
        "tools_used": tools_used,
        "grounded_ratio": grounded_ratio,
        "ungrounded_numbers": ungrounded,
        "tool_call_count": tool_call_count,
        "max_step": max_step,
    }
