"""上下文压缩的在线 A/B(M8):同一组回放用例,legacy vs v2。

    python -m eval.compress_ab                       # 两种模式 × 全部用例 × 1 次
    python -m eval.compress_ab --repeat 2 --high 3000 --low 1500
    python -m eval.compress_ab --modes v2 --cases deep_page_01

真模型、假工具(eval/compress_replay.py):工具结果固定,变化的只有压缩策略和模型本身的随机性。
水位刻意调低,保证 v2 一定会归档;legacy 按自己的"消息超过 12 条"触发,不受水位影响 ——
两者的触发口径不同,报告里会写明,比较的是"同一组任务下谁更省、谁丢得更少"。

指标:
  事实保留率    golden_numbers 在最终答案里出现的比例(数字是虚构的,只能来自工具结果)
  重复失败调用  某 (工具, 目标) 已经失败过之后又被调用的次数
  compress 占比 compress 节点耗时 / 总耗时
  主模型输入    prompt_tokens 与缓存命中占比
  压缩自身成本  摘要/归档调用的 token(走 core.LLM,不在主模型的 cost 里)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 必须在导入 obs.metrics 之前:单价在 import 时读取

from core.breaker import current_breaker  # noqa: E402
from core.failure import FAIL_PREFIX, current_log, target_of  # noqa: E402
from core.numbers import normalize  # noqa: E402
from eval.compress_replay import CASES, build_replay_tools  # noqa: E402
from graph.agent import GraphAgent  # noqa: E402
from obs.metrics import PRICE_IN, PRICE_OUT  # noqa: E402

_REPORTS_DIR = Path(__file__).parent / "reports"


def repeat_failed_calls(trajectory) -> int:
    failed: set[tuple[str, str]] = set()
    repeats = 0
    for s in trajectory:
        if not s.tool_call:
            continue
        key = (s.tool_call.name, target_of(s.tool_call.arguments or {}))
        if key in failed:
            repeats += 1
        if str(s.observation or "").startswith(FAIL_PREFIX):
            failed.add(key)
    return repeats


def run_one(case: dict, mode: str) -> dict:
    current_breaker().reset_all()
    current_log().reset()
    agent = GraphAgent(compress_mode=mode, memory_mode="off", tools=build_replay_tools(case), print_obs=False)
    t0 = time.time()
    answer, trajectory = agent.run(case["question"])
    rec = agent.last_run.to_json() if agent.last_run else {}
    ans = normalize(answer)
    kept = [n for n in case["golden_numbers"] if n in ans]
    llm = rec.get("llm") or {}
    cx = rec.get("ctx") or {}
    nodes = rec.get("nodes") or {}
    total_ms = rec.get("total_ms") or 1
    s_in, s_out = cx.get("summarizer_in_tokens", 0), cx.get("summarizer_out_tokens", 0)
    return {
        "case": case["id"],
        "mode": mode,
        "elapsed_s": round(time.time() - t0, 1),
        "retention": round(len(kept) / len(case["golden_numbers"]), 3),
        "missing_numbers": [n for n in case["golden_numbers"] if n not in ans],
        "repeat_failed": repeat_failed_calls(trajectory),
        "tool_calls": sum(1 for s in trajectory if s.tool_call),
        "llm_calls": llm.get("calls", 0),
        "prompt_tokens": llm.get("prompt_tokens", 0),
        "cached_tokens": llm.get("cached_tokens", 0),
        "cost_cny": rec.get("cost_cny"),
        "compress_ms": (nodes.get("compress") or {}).get("total_ms", 0),
        "compress_share": round((nodes.get("compress") or {}).get("total_ms", 0) / total_ms, 3),
        "summarizer_calls": cx.get("summarizer_calls", 0),
        "summarizer_tokens": s_in + s_out,
        "summarizer_cost_cny": round((s_in * PRICE_IN + s_out * PRICE_OUT) / 1_000_000, 6),
        # legacy 没有"归档"概念,它的压缩次数就是摘要器被调用的次数
        "compactions": cx.get("compactions", 0) if mode == "v2" else cx.get("summarizer_calls", 0),
        "readbacks": cx.get("readbacks", 0),
        "facts_kept": cx.get("facts_kept", 0),
        "facts_dropped": cx.get("facts_dropped", 0),
        "fallbacks": cx.get("fallbacks", 0),
        "max_est_tokens": cx.get("max_est_tokens"),
        "stop_reason": rec.get("stop_reason"),
        "hit_limit": rec.get("hit_limit"),
        "answer": answer,
    }


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def summarize(rows: list[dict], modes: list[str]) -> dict:
    out = {}
    for m in modes:
        rs = [r for r in rows if r["mode"] == m]
        p, c = sum(r["prompt_tokens"] for r in rs), sum(r["cached_tokens"] for r in rs)
        out[m] = {
            "runs": len(rs),
            "retention": _avg(rs, "retention"),
            "repeat_failed": _avg(rs, "repeat_failed"),
            "llm_calls": _avg(rs, "llm_calls"),
            "prompt_tokens": _avg(rs, "prompt_tokens"),
            "cache_ratio": round(c / p, 3) if p else 0.0,
            "cost_cny": _avg(rs, "cost_cny"),
            "summarizer_tokens": _avg(rs, "summarizer_tokens"),
            "summarizer_cost_cny": _avg(rs, "summarizer_cost_cny"),
            "compress_ms": _avg(rs, "compress_ms"),
            "compress_share": _avg(rs, "compress_share"),
            "readbacks": _avg(rs, "readbacks"),
            "compactions": _avg(rs, "compactions"),
            "stopped_early": sum(1 for r in rs if r["stop_reason"]),
            "hit_limit": sum(1 for r in rs if r["hit_limit"]),
        }
    return out


def render(summary: dict, rows: list[dict], args, stamp: str) -> str:
    modes = list(summary)
    lines = [
        f"# 压缩 A/B {stamp}",
        "",
        f"- 回放用例 {len({r['case'] for r in rows})} 道 × 每种模式 {args.repeat} 次;"
        f"v2 水位 high={args.high} / low={args.low} token;legacy 按自己的\"超过 12 条消息\"触发",
        "- 关键数字均为虚构,只能来自工具结果;压缩自身成本单独列出(走 core.LLM,不在主模型 cost 里)",
        "",
        "## 汇总(每次运行的平均值)",
        "",
        "| 指标 | " + " | ".join(modes) + " |",
        "| --- | " + " | ".join(":-:" for _ in modes) + " |",
    ]
    labels = [
        ("事实保留率", "retention"), ("重复失败调用", "repeat_failed"), ("主模型 LLM 次数", "llm_calls"),
        ("主模型输入 token", "prompt_tokens"), ("缓存命中占比", "cache_ratio"), ("主模型成本 ¥", "cost_cny"),
        ("压缩自身 token", "summarizer_tokens"), ("压缩自身成本 ¥", "summarizer_cost_cny"),
        ("compress 耗时 ms", "compress_ms"), ("compress 耗时占比", "compress_share"),
        ("压缩次数", "compactions"), ("回读次数", "readbacks"),
        ("预算提前收尾(次)", "stopped_early"), ("撞框架硬上限(次)", "hit_limit"),
    ]
    for label, key in labels:
        lines.append(f"| {label} | " + " | ".join(str(summary[m][key]) for m in modes) + " |")
    lines += [
        "",
        "## 逐次结果",
        "",
        "| 用例 | 模式 | 保留率 | 缺失数字 | 重复失败 | LLM | 输入(缓存) | compress ms | 压缩/回读 | 事实 留/丢 | 收尾 |",
        "| --- | --- | :-: | --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |",
    ]
    for r in rows:
        ending = "⛔硬上限" if r["hit_limit"] else (f"⚠️{r['stop_reason']}" if r["stop_reason"] else "正常")
        lines.append(
            f"| {r['case']} | {r['mode']} | {r['retention']} | {','.join(r['missing_numbers']) or '-'} | "
            f"{r['repeat_failed']} | {r['llm_calls']} | {r['prompt_tokens']}({r['cached_tokens']}) | "
            f"{round(r['compress_ms'])} | {r['compactions']}/{r['readbacks']} | "
            f"{r['facts_kept']}/{r['facts_dropped']} | {ending} |"
        )
    return "\n".join(lines)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="legacy,v2")
    ap.add_argument("--cases", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--high", type=int, default=3000)
    ap.add_argument("--low", type=int, default=1500)
    args = ap.parse_args()

    os.environ["CTX_HIGH_TOKENS"] = str(args.high)
    os.environ["CTX_LOW_TOKENS"] = str(args.low)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cases = [c for c in CASES if not args.cases or c["id"] in args.cases.split(",")]

    rows = []
    for k in range(args.repeat):
        for case in cases:
            for m in modes:  # 同一用例的两种模式相邻跑,时间上的外部波动对两边影响相近
                print(f"[{k + 1}/{args.repeat}] {case['id']} · {m} ...", flush=True)
                r = run_one(case, m)
                print(f"    -> 保留率 {r['retention']} 缺失 {r['missing_numbers']} | 重复失败 {r['repeat_failed']} | "
                      f"LLM {r['llm_calls']} | compress {round(r['compress_ms'])}ms | "
                      f"压缩 {r['compactions']} 回读 {r['readbacks']} | {r['elapsed_s']}s", flush=True)
                rows.append(r)

    summary = summarize(rows, modes)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md = render(summary, rows, args, stamp)
    (_REPORTS_DIR / f"compress_ab_{stamp}.md").write_text(md, encoding="utf-8")
    (_REPORTS_DIR / f"compress_ab_{stamp}.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\n报告已保存: eval/reports/compress_ab_{stamp}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
