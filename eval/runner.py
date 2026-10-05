"""评测总指挥:批量跑测试集 -> 收集轨迹+答案 -> 规则检查 + LLM 裁判 -> 出报告。

运行:
    python -m eval.runner              # 跑全部用例
    python -m eval.runner concept_pe_01 tsla_price_01   # 只跑指定用例(快速迭代)

产物:
    eval/reports/<时间戳>.json  逐题明细(机器可读,便于回归对比)
    eval/reports/<时间戳>.md    人类可读的汇总报告

设计要点:
  - 评测时关闭长期记忆写入(use_long_term=False),避免把评测问答污染进记忆库,
    也让每次评测更接近"干净的冷启动",结果更可复现。
  - PASS 判定:该调的工具都调了 且 忠实度>=4 且 数字可溯源比例>=0.6。
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from core.breaker import current_breaker
from core.failure import current_log
from core.llm import LLM
from eval.checks import build_judge_facts, run_checks
from eval.judge import judge
from main import build_agent

# ---- PASS 阈值(可按需调整)----
MIN_FAITHFULNESS = 4  # 忠实度(反编造)至少 4/5
MIN_GROUNDED_RATIO = 0.6  # 回答里的数字至少 60% 能在工具返回里找到出处
MIN_ACCURACY = 3  # 准确性至少 3/5
MIN_COMPLETENESS = 3  # 完整性至少 3/5
# 为什么要卡 accuracy/completeness?
# M5 对照实验时发现:Agent 撞到步数上限、交出一个近乎空的答案(acc=1 comp=1),
# 竟然 PASS 了 —— 因为空答案什么都没编,faithfulness 反而满分 5。
# 这和我们在 docs/notes/2026-09-09-评测方法论调研.md 里批评 τ-bench "空回复算成功"
# 是同一个 bug,只不过出现在自己家里。教训:**只卡"不犯错"的维度,会奖励什么都不做。**
# 必须同时卡"有没有把事办成"的维度。

_CASES_PATH = Path(__file__).parent / "cases.jsonl"
_REPORTS_DIR = Path(__file__).parent / "reports"


def load_cases(path: Path = _CASES_PATH) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def is_pass(case: dict, checks: dict, judged: dict) -> bool:
    ok = (
        checks["tool_ok"]
        and judged.get("faithfulness", 0) >= MIN_FAITHFULNESS
        and judged.get("accuracy", 0) >= MIN_ACCURACY
        and judged.get("completeness", 0) >= MIN_COMPLETENESS
    )
    # 概念解释题里的举例数字不是"事实",不该用数字溯源当硬门槛。
    # 用例可显式关闭该门槛(check_number_grounding=false)。
    if case.get("check_number_grounding", True):
        ok = ok and checks["grounded_ratio"] >= MIN_GROUNDED_RATIO
    return ok


def evaluate_one(agent, judge_llm: LLM, case: dict) -> dict:
    """跑一道题的完整评测:作答 -> 规则检查 -> 裁判打分 -> 判定通过。

    开头的 reset 不是可选的卫生习惯,是**结果可比性的前提**:
    熔断器和失败台账都是进程级状态(见 core/breaker.py 的说明),不清的话,
    第 3 题会因为第 1 题把 `fetch_page` 打熔断了而直接拿不到工具 ——
    那道题的成绩就不是它自己的了,而且题目顺序一变分数就变。
    每题独立清零,才是"逐题可比"。
    """
    current_breaker().reset_all()
    current_log().reset()
    t0 = time.time()
    answer, trajectory = agent.run(case["question"])
    elapsed = round(time.time() - t0, 1)

    checks = run_checks(case, answer, trajectory)
    # 裁判用"精炼版"证据(去重+限长);数字溯源在 run_checks 内部仍用完整证据。
    facts = build_judge_facts(trajectory)
    judged = judge(
        judge_llm,
        question=case["question"],
        answer=answer,
        expect_points=case.get("expect_points", []),
        facts=facts,
    )

    # M5.2.3:把这一题的观测汇总一起收进来,报告里就能在"答得好不好"旁边
    # 写上"花了多少、慢在哪"。
    #
    # 只有图版有 last_run —— 手写版 core/agent.py 不走 LangChain 回调,拿不到。
    # 这个差异本身就是"框架给了什么"的一个具体答案,所以不强行补齐,老实留空。
    #
    # 注意这里统计的**只是被测 Agent 的开销**,不含 LLM 裁判。
    # 裁判是"考试的成本"不是"被测对象的成本",混在一起的话,
    # 改一次裁判 prompt 就会让 Agent 看起来变贵了。
    rec = getattr(agent, "last_run", None)

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "elapsed_s": elapsed,
        "checks": checks,
        "judge": judged,
        "passed": is_pass(case, checks, judged),
        "obs": rec.to_json() if rec is not None else None,
    }


def _avg(nums: list[float]) -> float:
    return round(sum(nums) / len(nums), 2) if nums else 0.0


def summarize_cost(results: list[dict]) -> dict | None:
    """把逐题的观测数据聚合成"这一轮评测花了多少代价"。

    没有任何一题带观测(比如跑的是手写版)时返回 None,报告里整段略过 ——
    宁可不显示,也不要显示一堆 0 让人误以为"这版不花钱"。
    """
    obs_list = [r["obs"] for r in results if r.get("obs")]
    if not obs_list:
        return None

    n = len(obs_list)
    llm = [o["llm"] for o in obs_list]

    # 成本:只要有一题算不出(没配单价),整轮就是"未配单价"。
    # 把能算的几题加起来当总数是错的,那个数没有意义。
    costs = [o.get("cost_cny") for o in obs_list]
    total_cost = None if any(c is None for c in costs) else round(sum(costs), 6)

    tool_calls = tool_fails = tool_retries = 0
    # M6 新增的两个计数:它们是"分类重试 + 熔断"真的在省钱的直接证据。
    # retry_skipped 高 = 很多确定性失败被挡在重试之前(每次省一个往返 + 一次退避);
    # circuit_open 高 = 整条链路不通时没有排队去撞。
    retry_skipped = circuit_open = 0
    for o in obs_list:
        for st in (o.get("tools") or {}).values():
            tool_calls += st["count"]
            tool_fails += st["fail"]
            tool_retries += st["retry"]
            retry_skipped += st.get("retry_skipped", 0)
            circuit_open += st.get("circuit_open", 0)

    # 各节点耗时占比:这一行最有用 —— 它直接指出下一步优化该砍哪。
    node_ms: dict[str, float] = defaultdict(float)
    for o in obs_list:
        for name, st in (o.get("nodes") or {}).items():
            node_ms[name] += st["total_ms"]
    total_node_ms = sum(node_ms.values())
    node_share = {
        name: round(ms / total_node_ms, 3)
        for name, ms in sorted(node_ms.items(), key=lambda kv: -kv[1])
    } if total_node_ms else {}

    return {
        "cases_with_obs": n,
        "avg_llm_calls": round(sum(x["calls"] for x in llm) / n, 1),
        "avg_prompt_tokens": round(sum(x["prompt_tokens"] for x in llm) / n),
        "avg_cached_tokens": round(sum(x["cached_tokens"] for x in llm) / n),
        "avg_completion_tokens": round(sum(x["completion_tokens"] for x in llm) / n),
        "avg_reasoning_tokens": round(sum(x["reasoning_tokens"] for x in llm) / n),
        "total_cost_cny": total_cost,
        "avg_cost_cny": None if total_cost is None else round(total_cost / n, 6),
        "tool_calls": tool_calls,
        "tool_fails": tool_fails,
        "tool_fail_rate": round(tool_fails / tool_calls, 3) if tool_calls else 0.0,
        "tool_retries": tool_retries,
        "retry_skipped": retry_skipped,
        "circuit_open": circuit_open,
        # 两个"被拦下来"的计数要分开看,它们含义完全不同:
        #   hit_limit  = 撞了框架的 recursion_limit,流程被硬切断,**没有答案**;
        #   stop_early = 我们自己的预算边界先生效,走了 finalize,**有答案**。
        # M6 的目标就是把前者变成后者。
        "hit_limit": sum(1 for o in obs_list if o.get("hit_limit")),
        "stop_early": sum(1 for o in obs_list if o.get("stop_reason")),
        "stop_reasons": sorted({
            o["stop_reason"] for o in obs_list if o.get("stop_reason")
        }),
        "node_time_share": node_share,
        # 记下模型名:回归对比时必须知道"这份成绩是哪个模型跑的"。
        # 请求名和返回名可能不一样(旧名已下线、由新模型代服务),两个都留。
        "model_requested": llm[0].get("model_requested", ""),
        "model_returned": llm[0].get("model_returned", ""),
    }


def summarize(results: list[dict]) -> dict:
    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "avg_accuracy": _avg([r["judge"].get("accuracy", 0) for r in results]),
        "avg_completeness": _avg([r["judge"].get("completeness", 0) for r in results]),
        "avg_faithfulness": _avg([r["judge"].get("faithfulness", 0) for r in results]),
        "avg_grounded_ratio": _avg([r["checks"]["grounded_ratio"] for r in results]),
        "avg_elapsed_s": _avg([r["elapsed_s"] for r in results]),
        "cost": summarize_cost(results),
    }


def _fmt_cost(value: float | None) -> str:
    """没配单价就老实说"未配单价",不拿 0 充数 —— 区分"免费"和"不知道"。"""
    return "未配单价" if value is None else f"¥{value:.4f}"


def _slowest_node(obs: dict) -> str:
    nodes = obs.get("nodes") or {}
    if not nodes:
        return "-"
    name, st = max(nodes.items(), key=lambda kv: kv[1]["total_ms"])
    return f"{name} {st['total_ms'] / 1000:.1f}s"


def _tool_brief(obs: dict) -> str:
    tools = obs.get("tools") or {}
    calls = sum(st["count"] for st in tools.values())
    fails = sum(st["fail"] for st in tools.values())
    retries = sum(st["retry"] for st in tools.values())
    skipped = sum(st.get("retry_skipped", 0) for st in tools.values())
    opened = sum(st.get("circuit_open", 0) for st in tools.values())
    brief = f"{calls} ({fails}/{retries})"
    # 只在真的发生时才加后缀,不然每行都挂两个 0,噪音大于信息。
    if skipped:
        brief += f" 免重试{skipped}"
    if opened:
        brief += f" 熔断{opened}"
    return brief


def render_cost_section(cost: dict | None) -> list[str]:
    """报告里的"代价"段。质量回答"答得好不好",这一段回答"花了多少、慢在哪"。"""
    if not cost:
        return [
            "",
            "## 代价",
            "",
            "- 本轮没有观测数据(手写版不走 LangChain 回调,拿不到 token/成本)。",
        ]

    share = " | ".join(
        f"{name} {pct * 100:.1f}%" for name, pct in cost["node_time_share"].items()
    )
    model = cost["model_requested"]
    if cost["model_returned"] and cost["model_returned"] != model:
        model += f" → 实际 {cost['model_returned']}"

    return [
        "",
        "## 代价(观测层数据,不含裁判开销)",
        "",
        f"- 模型:{model}",
        f"- 平均每题:LLM {cost['avg_llm_calls']} 次 | "
        f"输入 {cost['avg_prompt_tokens']}(其中缓存命中 {cost['avg_cached_tokens']})| "
        f"输出 {cost['avg_completion_tokens']}(其中思考 {cost['avg_reasoning_tokens']})",
        f"- 成本:总计 {_fmt_cost(cost['total_cost_cny'])},"
        f"平均每题 {_fmt_cost(cost['avg_cost_cny'])}",
        f"- 工具:共 {cost['tool_calls']} 次,失败 {cost['tool_fails']} 次"
        f"({cost['tool_fail_rate'] * 100:.1f}%),重试 {cost['tool_retries']} 次,"
        f"免重试 {cost.get('retry_skipped', 0)} 次,熔断拦截 {cost.get('circuit_open', 0)} 次",
        f"- 撞框架硬上限(无答案):{cost['hit_limit']}/{cost['cases_with_obs']} 题;"
        f"预算提前收尾(有答案):{cost.get('stop_early', 0)}/{cost['cases_with_obs']} 题"
        + (f" —— {'、'.join(cost['stop_reasons'])}" if cost.get("stop_reasons") else ""),
        f"- 耗时构成:{share or '-'}",
    ]


def render_markdown(summary: dict, results: list[dict], stamp: str) -> str:
    lines = [
        f"# 评测报告 {stamp}",
        "",
        f"- 被测实现:**{summary.get('impl', 'handwritten')}**",
        f"- 用例数:{summary['total']}  通过:{summary['passed']}  "
        f"**通过率:{summary['pass_rate'] * 100:.0f}%**",
        f"- 平均分:accuracy {summary['avg_accuracy']} | "
        f"completeness {summary['avg_completeness']} | "
        f"faithfulness {summary['avg_faithfulness']}",
        f"- 平均数字可溯源比例:{summary['avg_grounded_ratio']}  "
        f"平均耗时:{summary['avg_elapsed_s']}s",
        "",
        "## 逐题结果",
        "",
        "| 用例 | 通过 | acc | comp | faith | 溯源 | 用到的工具 | 步数 | 耗时 |",
        "| --- | :-: | :-: | :-: | :-: | :-: | --- | :-: | :-: |",
    ]
    for r in results:
        c, j = r["checks"], r["judge"]
        # 只列**去重后**的工具名。之前是把调用序列整个铺开,一道跑偏的题能列出
        # 38 个名字,把表格撑得没法看。具体次数看下面的代价表。
        used = list(dict.fromkeys(c["tools_used"]))
        lines.append(
            f"| {r['id']} | {'✅' if r['passed'] else '❌'} | "
            f"{j.get('accuracy', 0)} | {j.get('completeness', 0)} | "
            f"{j.get('faithfulness', 0)} | {c['grounded_ratio']} | "
            f"{','.join(used) or '-'} | {c['max_step']} | {r['elapsed_s']}s |"
        )

    lines += render_cost_section(summary.get("cost"))

    if summary.get("cost"):
        lines += [
            "",
            "## 逐题代价",
            "",
            "| 用例 | LLM | 输入(缓存) | 输出 | 成本 | 工具(失败/重试) | 最慢节点 | 收尾方式 |",
            "| --- | :-: | :-: | :-: | :-: | :-: | --- | --- |",
        ]
        for r in results:
            o = r.get("obs")
            if not o:
                lines.append(f"| {r['id']} | - | - | - | - | - | - | - |")
                continue
            m = o["llm"]
            # 最后一列换成"怎么结束的":正常 / 预算提前收尾(有答案)/ 撞框架硬上限(无答案)。
            if o.get("hit_limit"):
                ending = "⛔撞框架硬上限"
            elif o.get("stop_reason"):
                ending = f"⚠️{o['stop_reason']}"
            else:
                ending = "正常"
            lines.append(
                f"| {r['id']} | {m['calls']} | "
                f"{m['prompt_tokens']}({m['cached_tokens']}) | "
                f"{m['completion_tokens']} | {_fmt_cost(o.get('cost_cny'))} | "
                f"{_tool_brief(o)} | {_slowest_node(o)} | {ending} |"
            )

    # 失败/异常用例的详情,方便定位
    problems = [r for r in results if not r["passed"] or r["judge"].get("error")]
    if problems:
        lines += ["", "## 需要关注的用例", ""]
        for r in problems:
            c, j = r["checks"], r["judge"]
            lines.append(f"### {r['id']}")
            if c["missing_tools"]:
                lines.append(f"- 未调用应调工具:{c['missing_tools']}")
            if c["ungrounded_numbers"]:
                lines.append(f"- 疑似无出处的数字:{c['ungrounded_numbers']}")
            if j.get("error"):
                lines.append(f"- 裁判异常:{j['error']}")
            elif j.get("reason"):
                lines.append(f"- 裁判理由:{j['reason']}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    load_dotenv()

    args = set(sys.argv[1:])

    # --graph:评测 LangGraph 版(M5)而非手写版。两者接口相同,评测代码无需改动,
    # 因此可以用同一套用例做重构前后的回归对比。
    use_graph = "--graph" in args
    args.discard("--graph")

    if use_graph:
        from graph.agent import GraphAgent

        agent = GraphAgent()
        impl = "langgraph"
    else:
        agent = build_agent()
        impl = "handwritten"

    # 评测时不写长期记忆,保持干净、可复现。
    agent.use_long_term = False
    judge_llm = LLM()

    cases = load_cases()
    # 支持只跑指定用例(命令行传用例 id),方便快速迭代,不必每次全量跑。
    only = args
    if only:
        cases = [c for c in cases if c["id"] in only]
    print(f"载入 {len(cases)} 个用例,被测实现:{impl},开始评测...\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] 跑用例: {case['id']} ...", flush=True)
        r = evaluate_one(agent, judge_llm, case)
        j = r["judge"]
        print(
            f"    -> {'PASS' if r['passed'] else 'FAIL'} | "
            f"acc={j.get('accuracy', 0)} comp={j.get('completeness', 0)} "
            f"faith={j.get('faithfulness', 0)} | 溯源={r['checks']['grounded_ratio']} | "
            f"{r['elapsed_s']}s",
            flush=True,
        )
        results.append(r)

    summary = summarize(results)
    summary["impl"] = impl  # 记录被测实现,回归对比时必须知道跑的是哪一版

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (_REPORTS_DIR / f"{stamp}.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = render_markdown(summary, results, stamp)
    (_REPORTS_DIR / f"{stamp}.md").write_text(md, encoding="utf-8")

    print("\n" + "=" * 60)
    print(md)
    print("=" * 60)
    print(f"\n报告已保存: eval/reports/{stamp}.json 和 {stamp}.md")


if __name__ == "__main__":
    main()
