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
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from core.llm import LLM
from eval.checks import build_judge_facts, run_checks
from eval.judge import judge
from main import build_agent

# ---- PASS 阈值(可按需调整)----
MIN_FAITHFULNESS = 4  # 忠实度(反编造)至少 4/5
MIN_GROUNDED_RATIO = 0.6  # 回答里的数字至少 60% 能在工具返回里找到出处

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
    )
    # 概念解释题里的举例数字不是"事实",不该用数字溯源当硬门槛。
    # 用例可显式关闭该门槛(check_number_grounding=false)。
    if case.get("check_number_grounding", True):
        ok = ok and checks["grounded_ratio"] >= MIN_GROUNDED_RATIO
    return ok


def evaluate_one(agent, judge_llm: LLM, case: dict) -> dict:
    """跑一道题的完整评测:作答 -> 规则检查 -> 裁判打分 -> 判定通过。"""
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

    return {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "elapsed_s": elapsed,
        "checks": checks,
        "judge": judged,
        "passed": is_pass(case, checks, judged),
    }


def _avg(nums: list[float]) -> float:
    return round(sum(nums) / len(nums), 2) if nums else 0.0


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
    }


def render_markdown(summary: dict, results: list[dict], stamp: str) -> str:
    lines = [
        f"# 评测报告 {stamp}",
        "",
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
        "| 用例 | 通过 | acc | comp | faith | 溯源 | 工具 | 步数 | 耗时 |",
        "| --- | :-: | :-: | :-: | :-: | :-: | --- | :-: | :-: |",
    ]
    for r in results:
        c, j = r["checks"], r["judge"]
        lines.append(
            f"| {r['id']} | {'✅' if r['passed'] else '❌'} | "
            f"{j.get('accuracy', 0)} | {j.get('completeness', 0)} | "
            f"{j.get('faithfulness', 0)} | {c['grounded_ratio']} | "
            f"{','.join(c['tools_used']) or '-'} | {c['max_step']} | {r['elapsed_s']}s |"
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

    agent = build_agent()
    # 评测时不写长期记忆,保持干净、可复现。
    agent.use_long_term = False
    judge_llm = LLM()

    cases = load_cases()
    # 支持只跑指定用例(命令行传用例 id),方便快速迭代,不必每次全量跑。
    only = set(sys.argv[1:])
    if only:
        cases = [c for c in cases if c["id"] in only]
    print(f"载入 {len(cases)} 个用例,开始评测...\n")

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
