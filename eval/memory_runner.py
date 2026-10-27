"""记忆评测:按"基础回忆 / 多会话更新 / 拒答 / 隔离 / 主动服务"分层,对 off/legacy/v2 做消融。

用法:
    python -m eval.memory_runner --mode off
    python -m eval.memory_runner --mode legacy
    python -m eval.memory_runner --mode v2          # 第 2~4 步完成后才可用
    python -m eval.memory_runner --mode legacy --only L2_update

为什么单独一个 runner,而不是并入 eval/runner.py:
  那套评测问的是"单次调研答得好不好",每题独立;记忆评测要**按顺序跑多轮对话**,
  最后一个问题只能靠记忆回答,而且需要控制 user_id。两者的用例结构和判分标准都不同。

两种用例模式:
  session  真的把 sessions 里每句话各跑一轮(每句用**新 thread_id**,模拟跨会话),
           最后再提问。测完整的"写入 → 读取"链路,会花钱。
  seed     直接往记忆库灌记录再提问,只测读取、结果确定。需要 v2 的结构化存储,
           所以 off/legacy 档标为"不适用"。

隔离性(⚠️ 评测自己的污染问题):
  - 每题使用独立 user_id;
  - legacy 档没有 user_id 概念、全部写进同一个集合,所以这里给**每道题**换一个临时集合
    (跑完删除),否则第 N 题会被第 N-1 题写进去的记忆污染 —— 就是 M6 里
    "熔断器不清,第 3 题分数不是它自己的"同一类问题。
    但同一题内部不换,这样 isolation 题才能暴露"旧方案没有隔离"这件事。
  - 程序记忆(M7 第 5 步)在评测里强制关闭:MEMORY_PROCEDURAL=0。
"""

import argparse
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["MEMORY_PROCEDURAL"] = "0"  # 必须在导入业务模块之前设置

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from core.llm import LLM  # noqa: E402
from eval.judge import _parse_json  # noqa: E402
from graph.agent import GraphAgent  # noqa: E402

_CASES = Path(__file__).parent / "memory_cases.jsonl"
_REPORTS = Path(__file__).parent / "reports"

_JUDGE_PROMPT = """你是严格的评审。判断【回答】是否满足要求。

【问题】{question}
【回答必须体现的要点】
{expect}
【回答不得出现的情况】
{must_not}

只依据上面的要求判断,不要用自己的先验知识。每个要点都体现、且没有任何禁止情况,才算通过。
只输出 JSON:{{"passed": true或false, "reason": "50字以内理由"}}
"""


def load_cases() -> list[dict]:
    out = []
    for line in _CASES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _judge(llm: LLM, case: dict, answer: str) -> dict:
    prompt = _JUDGE_PROMPT.format(
        question=case["question"],
        expect="\n".join(f"- {p}" for p in case["expect_points"]) or "(无)",
        must_not="\n".join(f"- {p}" for p in case["must_not"]) or "(无)",
    ) + f"\n【回答】\n{answer}"
    try:
        data = _parse_json(llm.complete([{"role": "user", "content": prompt}]))
        return {"passed": bool(data.get("passed")), "reason": str(data.get("reason", ""))[:200]}
    except Exception as exc:  # noqa: BLE001 裁判抽风不能拖垮整轮
        return {"passed": False, "reason": "", "error": f"{type(exc).__name__}: {exc}"}


# ---- legacy 档的临时集合 ----

def _swap_legacy_collection() -> str:
    """把旧版长期记忆指向一个临时集合,返回集合名。"""
    from memory import long_term
    from rag.store import QdrantStore

    name = f"fra_memory_eval_{uuid.uuid4().hex[:8]}"
    long_term._store = QdrantStore(collection=name)
    long_term._ready = False
    return name


def _drop_legacy_collection(name: str) -> None:
    from memory import long_term

    try:
        long_term._store.http.delete(f"/collections/{name}")
    except Exception:  # noqa: BLE001 清理失败不影响结果
        pass


def _pollute(mode: str, kinds: list[str], user_id: str) -> None:
    """跑题之前先把记忆库弄脏(见 eval/memory_fillers.py 的说明)。

    不经过 agent/LLM,直接批量 embedding 后写库:便宜,而且保证填充内容是确定的。
    legacy 的写入格式与 long_term.remember 完全一致,所以和真实累积出来的库没有区别
    (除了没有走抽取 —— 旧方案本来就没有抽取)。
    """
    from eval.memory_fillers import build

    pairs = build(kinds)
    if mode == "off":
        return  # 不用记忆,脏不脏无所谓
    if mode == "legacy":
        from memory import long_term
        from rag.embedding import embed_texts

        texts = [f"用户问题:{q}\n调研结论:{a}" for q, a in pairs]
        long_term._ensure_collection()
        now = time.time()
        vectors = embed_texts(texts)
        # 时间戳依次递增:填充都比评测里的写入"更旧"。
        long_term._store.upsert([
            {"id": str(uuid.uuid4()), "vector": v,
             "payload": {"text": t, "ts": now - (len(texts) - i) * 60}}
            for i, (t, v) in enumerate(zip(texts, vectors))
        ])
        return
    if mode == "v2":
        from eval.memory_fillers import to_v2_records
        from memory.runtime import get_store

        store = get_store()
        store.add_many(user_id=user_id, records=to_v2_records(pairs, user_id))
        return
    raise ValueError(mode)


# ---- v2 的临时集合与 seed ----

def _swap_v2_store():
    """给 v2 每道题一个临时集合(题间互不污染),返回该 store。"""
    from memory import runtime
    from memory.store import MemoryStore

    store = MemoryStore(collection=f"fra_memory_eval_v2_{uuid.uuid4().hex[:8]}")
    runtime.set_store(store)
    return store


def _drop_v2_store(store) -> None:
    from memory import runtime

    try:
        store.drop()
    except Exception:  # noqa: BLE001
        pass
    runtime.set_store(None)


def _seed_v2(store, seed: list[dict], user_id: str) -> None:
    """把 seed 条目直接写成结构化记录(绕过抽取),用来构造时间/安全/主动服务等场景。"""
    from memory.schema import MemoryRecord, Source

    now = time.time()
    recs = []
    for s in seed:
        typ = s["type"]
        kind = s.get("source_kind") or ("user_statement" if typ == "semantic" else "agent_conclusion")
        recs.append(MemoryRecord(
            id=str(uuid.uuid4()), user_id=user_id, type=typ, text=s["text"],
            tickers=s.get("tickers", []), importance=s.get("importance", 6),
            as_of=s.get("as_of"), created_at=now - s.get("created_at_days_ago", 0) * 86400,
            category=s.get("category"), key=s.get("key"), value=s.get("value"),
            source=Source(kind=kind),
        ))
    store.add_many(user_id=user_id, records=recs)


# ---- 单题 ----

def run_case(agent: GraphAgent, judge_llm: LLM, case: dict, mode: str) -> dict:
    base = {"id": case["id"], "level": case["level"]}

    if case["mode"] == "seed" and mode != "v2":
        return {**base, "status": "n/a", "reason": "seed 模式需要 v2 的结构化存储"}

    uid_of = lambda tag: f"eval-{case['id']}-{tag}-{uuid.uuid4().hex[:6]}"  # noqa: E731
    users: dict[str, str] = {}

    def uid(tag: str) -> str:
        return users.setdefault(tag, uid_of(tag))

    coll = _swap_legacy_collection() if mode == "legacy" else ""
    v2_store = _swap_v2_store() if mode == "v2" else None
    cost = 0.0
    t0 = time.time()
    mem_stats: dict = {}
    try:
        if case.get("pollute"):
            _pollute(mode, case["pollute"], uid(case.get("pollute_user", "default")))
        if case["mode"] == "seed":
            _seed_v2(v2_store, case["seed"], uid(case.get("seed_user", "default")))
        for s in case.get("sessions", []):
            text, who = (s["text"], s["user"]) if isinstance(s, dict) else (s, "default")
            agent.run(text, thread_id=f"mem-{uuid.uuid4().hex[:8]}", user_id=uid(who))
            cost += (agent.last_run.cost() or 0.0) if agent.last_run else 0.0

        who = case.get("question_user", "default")
        answer, _ = agent.run(
            case["question"], thread_id=f"mem-{uuid.uuid4().hex[:8]}", user_id=uid(who)
        )
        cost += (agent.last_run.cost() or 0.0) if agent.last_run else 0.0
        mem_stats = dict(agent.last_run.memory) if agent.last_run else {}  # 提问那一轮的记忆计数
    finally:
        if coll:
            _drop_legacy_collection(coll)
        if v2_store is not None:
            _drop_v2_store(v2_store)

    verdict = _judge(judge_llm, case, answer)
    return {
        **base,
        "status": "pass" if verdict["passed"] else "fail",
        "answer": answer,
        "reason": verdict["reason"],
        "judge_error": verdict.get("error"),
        "memory": mem_stats,
        "cost_cny": round(cost, 6),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ---- 报告 ----

def run_repeated(agent, judge_llm, case: dict, mode: str, repeat: int) -> dict:
    """同一题跑 repeat 次。LLM 既当被测又当裁判,单次结果有噪声。

    全过=pass,全挂=fail,其余=flaky(不稳定)。flaky 本身是信息:
    说明这题的结果取决于模型当次发挥,不能拿来证明某个机制有效。
    """
    runs = [run_case(agent, judge_llm, case, mode) for _ in range(repeat if case["mode"] == "session" else 1)]
    if runs[0]["status"] == "n/a":
        return runs[0]
    passes = sum(r["status"] == "pass" for r in runs)
    n = len(runs)
    status = "pass" if passes == n else "fail" if passes == 0 else "flaky"
    last = runs[-1]
    return {**last, "status": status, "passes": passes, "n": n,
            "cost_cny": round(sum(r["cost_cny"] for r in runs), 6)}


def summarize(results: list[dict]) -> dict:
    by_level: dict[str, dict] = defaultdict(
        lambda: {"pass": 0, "fail": 0, "flaky": 0, "n/a": 0})
    for r in results:
        by_level[r["level"]][r["status"]] += 1
    return dict(by_level)


def render(mode: str, results: list[dict]) -> str:
    lines = [f"# 记忆评测 · {mode}", "", "## 分层结果", "",
             "| 层级 | 稳定通过 | 不稳定 | 稳定失败 | 不适用 |", "| --- | :-: | :-: | :-: | :-: |"]
    for lv, c in summarize(results).items():
        lines.append(f"| {lv} | {c['pass']} | {c['flaky']} | {c['fail']} | {c['n/a']} |")
    lines += ["", "## 逐题", "", "| 用例 | 层级 | 结果 | 成本 | 耗时 | 理由(最后一次) |",
              "| --- | --- | :-: | :-: | :-: | --- |"]
    icon = {"pass": "✅", "fail": "❌", "flaky": "🟡", "n/a": "—"}
    for r in results:
        frac = f" {r['passes']}/{r['n']}" if "passes" in r else ""
        lines.append(
            f"| {r['id']} | {r['level']} | {icon[r['status']]}{frac} | "
            f"{('¥%.4f' % r['cost_cny']) if 'cost_cny' in r else '-'} | "
            f"{r.get('elapsed_s', '-')}s | {r.get('reason', '')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["off", "legacy", "v2"], required=True)
    ap.add_argument("--only", help="只跑某一题(按 id)")
    ap.add_argument("--repeat", type=int, default=1, help="每题重复次数,抗 LLM 噪声(建议 3)")
    ap.add_argument("--level", help="只跑某一层(L1/L2/abstain/noise/isolation/safety/L3)")
    ap.add_argument("--group", choices=["clean", "dirty", "all"], default="clean",
                    help="clean=常规题(默认);dirty=先弄脏记忆库的题,单独跑;all=全部")
    args = ap.parse_args()

    cases = [c for c in load_cases()
             if args.group == "all" or c.get("group", "clean") == args.group]
    cases = [c for c in cases if (not args.only or c["id"] == args.only)
             and (not args.level or c["level"] == args.level)]
    agent = GraphAgent(memory_mode=args.mode, print_obs=False)
    judge_llm = LLM()

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ({case['level']}) ...", flush=True)
        r = run_repeated(agent, judge_llm, case, args.mode, args.repeat)
        print(f"    -> {r['status']}  {r.get('reason', '')}", flush=True)
        results.append(r)

    text = render(args.mode, results)
    print("\n" + text)
    _REPORTS.mkdir(exist_ok=True)
    tag = f"memory_{args.mode}_{datetime.now():%Y-%m-%d_%H%M%S}"
    (_REPORTS / f"{tag}.json").write_text(
        json.dumps({"mode": args.mode, "summary": summarize(results), "results": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (_REPORTS / f"{tag}.md").write_text(text, encoding="utf-8")
    print(f"报告已保存:eval/reports/{tag}.(json|md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
