"""汇总 v1 / hybrid / agentic 三种检索模式的端到端评测(M9 Step 5)。只读报告,不调任何 API。

    RAG_MODE=v1      python -m eval.runner --graph --rag
    RAG_MODE=hybrid  python -m eval.runner --graph --rag
    RAG_MODE=agentic python -m eval.runner --graph --rag
    python -m eval.rag_e2e_compare

每种模式取 eval/reports/ 里最新的一份 *_rag-<mode>.json。另外把默认 5 题(eval/cases.jsonl)
在 v1 / agentic 下的最新报告也对一下,看 agentic 有没有把原来的题弄坏。
"""

import json
import statistics
from pathlib import Path

REPORTS = Path("eval/reports")
# 知识库本来就答不了 / 不该用知识库的题:不算进"全程没联网"
_NON_KB = {"rag_route_price_01", "rag_noanswer_apple_01", "rag_noanswer_period_01"}
MODES = ("v1", "hybrid", "agentic")


def _latest(pattern: str, pred=lambda d: True) -> dict | None:
    for f in sorted(REPORTS.glob(pattern), reverse=True):
        d = json.loads(f.read_text(encoding="utf-8"))
        if pred(d):
            d["_file"] = f.name
            return d
    return None


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _f(x, fmt="{:.2f}") -> str:
    return "—" if x is None else fmt.format(x)


def main() -> None:
    reports = {m: _latest(f"*_rag-{m}.json") for m in MODES}
    reports = {m: r for m, r in reports.items() if r}
    print("报告:" + ",".join(f"{m}={r['_file']}" for m, r in reports.items()))

    print("\n## 端到端总表(eval/rag_cases.jsonl)\n")
    print("| 模式 | 通过 | 通过(单位感知) | accuracy | completeness | faithfulness | 数字溯源 | 溯源(单位感知) | 路由正确 | 带引用的回答 | 引用号有效 | 引用支撑 | 引用支撑(单位感知) | 耗时 p50 | 主 Agent 成本 ¥/题 | 子图 LLM token/题 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m, r in reports.items():
        res = r["results"]
        n = len(res)
        cit = [x["checks"].get("citation") or {} for x in res]
        with_refs = [c for c in cit if c.get("has_refs")]
        rag_tok = [(x["obs"] or {}).get("rag", {}) or {} for x in res]
        print(f"| {m} | {sum(x['passed'] for x in res)}/{n} | {sum(x.get('passed_unit', x['passed']) for x in res)}/{n} "
              f"| {_mean(x['judge'].get('accuracy') for x in res):.2f} "
              f"| {_mean(x['judge'].get('completeness') for x in res):.2f} "
              f"| {_mean(x['judge'].get('faithfulness') for x in res):.2f} "
              f"| {_mean(x['checks']['grounded_ratio'] for x in res):.2f} "
              f"| {_f(_mean(x['checks'].get('grounded_ratio_unit') for x in res))} "
              f"| {sum(x['checks']['tool_ok'] for x in res)}/{n} "
              f"| {sum(bool(c.get('cited')) for c in with_refs)}/{len(with_refs) if with_refs else 0} "
              f"| {_f(_mean(c.get('ref_valid_ratio') for c in with_refs))} "
              f"| {_f(_mean(c.get('cite_support_ratio') for c in with_refs))} "
              f"| {_f(_mean(c.get('cite_support_ratio_unit') for c in with_refs))} "
              f"| {statistics.median(x['elapsed_s'] for x in res):.1f}s "
              f"| {_f(_mean((x['obs'] or {}).get('cost_cny') for x in res), '{:.4f}')} "
              f"| {_f(_mean(t.get('llm_tokens', 0) for t in rag_tok), '{:.0f}')} |")

    print("\n## 工具使用与真实成本\n")
    print("| 模式 | 工具调用/题 | 联网(search_news + fetch_page)/题 | 知识库题里全程没联网的 | 回读/题 | LLM 调用/题 | 主 Agent token/题 | 子图 token/题 | 估算总成本 ¥/题 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    web_tools = {"search_news", "fetch_page"}
    kb_ids = {x["id"] for x in next(iter(reports.values()))["results"]} - _NON_KB
    for m, r in reports.items():
        res = r["results"]
        obs = [x["obs"] or {} for x in res]
        tok = [o["llm"]["prompt_tokens"] + o["llm"]["completion_tokens"] for o in obs]
        price = sum(o.get("cost_cny") or 0 for o in obs) / max(1, sum(tok))  # 用主 Agent 的实际单价估子图成本
        rag_tok = [(o.get("rag") or {}).get("llm_tokens", 0) for o in obs]
        no_web = sum(1 for x in res if x["id"] in kb_ids and not web_tools & set(x["checks"]["tools_used"]))
        print(f"| {m} | {_mean(len(x['checks']['tools_used']) for x in res):.1f} "
              f"| {_mean(sum(t in web_tools for t in x['checks']['tools_used']) for x in res):.1f} "
              f"| {no_web}/{len(kb_ids)} | {_mean(((o.get('ctx') or {}).get('readbacks', 0)) for o in obs):.1f} "
              f"| {_mean(o['llm']['calls'] for o in obs):.1f} | {_mean(tok):.0f} "
              f"| {_mean(rag_tok):.0f} | {_mean((o.get('cost_cny') or 0) + t * price for o, t in zip(obs, rag_tok)):.4f} |")

    print("\n## 逐题(PASS / accuracy / faithfulness)\n")
    ids = [x["id"] for x in next(iter(reports.values()))["results"]]
    print("| 用例 | " + " | ".join(reports) + " |")
    print("| --- | " + " | ".join("---" for _ in reports) + " |")
    for cid in ids:
        cells = []
        for m, r in reports.items():
            x = next((y for y in r["results"] if y["id"] == cid), None)
            if x is None:
                cells.append("—")
                continue
            j = x["judge"]
            mark = "✅" if x["passed"] else ("☑️" if x.get("passed_unit") else "❌")  # ☑️ = 只在单位感知口径下通过
            extra = "" if x["checks"]["tool_ok"] else " 路由✗"
            cells.append(f"{mark} {j.get('accuracy', 0)}/{j.get('faithfulness', 0)}{extra}")
        print(f"| {cid} | " + " | ".join(cells) + " |")

    print("\n## 关键题的回答摘录\n")
    for cid in ("rag_noanswer_apple_01", "rag_noanswer_period_01", "rag_compare_dc_01", "rag_route_price_01"):
        print(f"### {cid}\n")
        for m, r in reports.items():
            x = next((y for y in r["results"] if y["id"] == cid), None)
            if x is None:
                continue
            ans = x["answer"].replace("\n", " ")[:260]
            print(f"- **{m}**({'PASS' if x['passed'] else 'FAIL'},工具 {x['checks']['tools_used']}):{ans}")
            print(f"  - 裁判:{x['judge'].get('reason', '')[:160]}")
        print()

    print("## 默认 5 题回归(eval/cases.jsonl)\n")
    for m in ("v1", "agentic"):
        d = _latest("2*.json", lambda d, m=m: d["summary"].get("cases") == "default" and d["summary"].get("rag_mode") == m)
        if d:
            res = d["results"]
            print(f"- {m}:{sum(x['passed'] for x in res)}/{len(res)} 通过({d['_file']});"
                  + ",".join(f"{x['id']} {'✅' if x['passed'] else '❌'}" for x in res))


if __name__ == "__main__":
    main()
