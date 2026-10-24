"""程序化规则检查(确定性评估)——不花钱、不调 LLM,纯代码判断客观事实。

为什么要有这一层?因为很多维度根本不需要"裁判"来判,用代码就能确定地回答:
  - 该调的工具到底调没调?(靠 Trajectory)
  - 回答里的数字,能不能在工具返回里找到出处?(反编造/幻觉的第一道防线)
  - 有没有绕圈子(步数/工具调用次数)?

这些检查便宜、稳定、可回归,所以能用规则的就绝不用 LLM。
"""

import ast
import bisect
import re

from core.numbers import meaningful_numbers as _meaningful_numbers
from core.numbers import normalize as _normalize
from core.sandbox import UNTRUSTED_CLOSE, UNTRUSTED_OPEN
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

    # (1) 该调的工具调了吗?不该调的(M9 must_not_call)有没有调?
    must = case.get("must_call_tools", [])
    missing_tools = [t for t in must if t not in tools_used_set]
    route = route_checks(case, trajectory)
    tool_ok = len(missing_tools) == 0 and not route["must_not_violated"]

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
        **route,
        "citation": citation_checks(answer, trajectory),  # M9:只用于统计,不参与 PASS 判定
        # M9:单位感知的数字溯源。不替换 grounded_ratio(保持和 M4~M8 的报告可比),另算一个 PASS 口径。
        "grounded_ratio_unit": (gu := grounded_ratio_unit(answer, trajectory))[0],
        "ungrounded_numbers_unit": gu[1],
    }


# ---------------------------------------------------------------- M9:路由反例 + 引用校验

_REF_RE = re.compile(r"\[(\d+(?:\s*[,,、]\s*\d+)*)\]")
_SENT_RE = re.compile(r"[^。!?!?\n]+[。!?!?\n]?")


def retrieve_payloads(trajectory: list[TrajectoryStep]) -> list[list[dict]]:
    """把每次 retrieve 的返回还原成 list[dict]。工具层把结果 str() 后包进不可信区块,这里反着拆。"""
    out = []
    for s in trajectory:
        if not (s.tool_call and s.tool_call.name == "retrieve" and s.observation):
            continue
        text = str(s.observation)
        if UNTRUSTED_OPEN in text:
            text = text.split(">>>\n", 1)[-1].rsplit(UNTRUSTED_CLOSE, 1)[0]
        try:
            data = ast.literal_eval(text.strip())
        except (ValueError, SyntaxError):
            continue  # 失败说明(FAIL_PREFIX)或被 M8 截成了 stub
        if isinstance(data, list):
            out.append([d for d in data if isinstance(d, dict)])
    return out


def citation_checks(answer: str, trajectory: list[TrajectoryStep]) -> dict:
    """引用三问:有没有标 [n]?标的 n 存在吗?标了 [n] 的句子里的数字,在第 n 条证据里找得到吗?

    多次调用 retrieve 时 ref 每次都从 1 编号,这里把同号证据的文本合并(宽松口径)。
    只有 agentic 模式的证据带 ref;v1 / hybrid 下 has_refs=False,本函数返回 None 各项。
    """
    payloads = retrieve_payloads(trajectory)
    ref_text: dict[int, str] = {}
    summaries = []
    for p in payloads:
        for d in p:
            if d.get("type") == "summary":
                summaries.append(d)
            elif "ref" in d:
                ref_text[d["ref"]] = ref_text.get(d["ref"], "") + "\n" + _normalize(str(d.get("text", "")))
    if not ref_text:
        return {"has_refs": False, "cited": None, "ref_valid_ratio": None, "cite_support_ratio": None,
                "cite_support_ratio_unit": None,
                "rag_summaries": summaries}
    cited_refs, supported, supported_unit, checked = [], 0, 0, 0
    for sent in _SENT_RE.findall(answer):
        refs = [int(x) for m in _REF_RE.findall(sent) for x in re.split(r"\s*[,,、]\s*", m)]
        if not refs:
            continue
        cited_refs += refs
        pool = "".join(ref_text.get(r, "") for r in refs)
        pool_vals = _num_values(pool)
        for n in _meaningful_numbers(_REF_RE.sub("", sent)):
            checked += 1
            supported += n in pool
            supported_unit += _unit_found(float(n), pool_vals)
    valid = [r for r in cited_refs if r in ref_text]
    return {
        "has_refs": True,
        "cited": bool(cited_refs),
        "ref_valid_ratio": round(len(valid) / len(cited_refs), 3) if cited_refs else None,
        # 标了引用的句子里的数字,有多少能在**它引用的那条证据**里找到(比 grounded_ratio 严:要对得上号)
        "cite_support_ratio": round(supported / checked, 3) if checked else None,
        "cite_support_ratio_unit": round(supported_unit / checked, 3) if checked else None,  # 同上,单位感知
        "rag_summaries": summaries,
    }


# 中文回答常把财报的 "$27,029 million" 写成 "270.29 亿美元"、把 "649,719.62 万元" 写成 "64.97 亿元"。
# 原 grounded_ratio 按字面找,这些全算"无出处"——M9 Step 5 里有答案被裁判打 5/5/5 却因此 FAIL。
# 单位感知版:把回答里的数乘以常见换算倍数,和证据里的数在 0.5% 以内对上就算有出处(容忍四舍五入)。
_SCALES = (1, 10, 100, 1_000, 10_000, 100_000, 1e-1, 1e-2, 1e-3, 1e-4)
_NUM_ANY = re.compile(r"-?\d+(?:\.\d+)?")


def _num_values(text: str) -> list[float]:
    vals = []
    for tok in _NUM_ANY.findall(_normalize(text)):
        try:
            v = abs(float(tok))
        except ValueError:
            continue
        if v:
            vals.append(v)
    return sorted(vals)


def _unit_found(n: float, vals: list[float]) -> bool:
    """n 乘以某个换算倍数后,和 vals(已排序)里某个数相差 ≤ 0.5%。"""
    for s in _SCALES:
        target = abs(n) * s
        lo = bisect.bisect_left(vals, target * 0.995)
        if lo < len(vals) and vals[lo] <= target * 1.005:
            return True
    return False


def grounded_ratio_unit(answer: str, trajectory: list[TrajectoryStep]) -> tuple[float, list[str]]:
    vals = _num_values(collect_observations(trajectory))
    nums = _meaningful_numbers(answer)
    if not nums:
        return 1.0, []
    miss = [n for n in nums if not _unit_found(float(n), vals)]
    return round(1 - len(miss) / len(nums), 3), miss


def route_checks(case: dict, trajectory: list[TrajectoryStep]) -> dict:
    """路由反例:must_not_call 里的工具一次都不该调(如问实时股价却去翻财报库)。"""
    used = set(called_tools(trajectory))
    violated = [t for t in case.get("must_not_call", []) if t in used]
    return {"must_not_violated": violated}
