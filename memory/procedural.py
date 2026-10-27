"""程序记忆(M7 第 5 步):Agent 从**确定性失败**里学到的、跨用户共享的经验规则。

## 它和熔断器的分工(别混)

| | 熔断器(core/failure.py) | 程序记忆(本文件) |
| --- | --- | --- |
| 时间尺度 | 秒级(60s) | 天级(7 天窗口 / 14 天过期) |
| 范围 | 进程内 | 持久化(Postgres),跨进程跨会话 |
| 作用方式 | **硬拦截**(直接拒绝调用) | **软引导**(写进系统提示,模型自己决定) |
| 触发 | 连续 3 次失败 | 跨 ≥2 轮运行、每轮都确定性失败 |

程序记忆**不拦截任何调用**。理由:它的证据比熔断弱(天级、统计性的),硬拦截一旦学错
(比如站点只是当天维护)就会让 Agent 永远不再尝试;软引导错了,模型还能自己试一次。

## 为什么只用代码、不用 LLM

程序记忆是出错代价最高的一类(一条错规则会影响所有用户的所有后续运行),
所以门槛最严:**只从可观测的确定性失败归纳,规则文本由代码模板生成**,没有幻觉空间。
对应地,它也不能被网页内容污染 —— 模板里唯一的动态成分是工具名和域名,
域名经过严格白名单校验(见 `scope_of`),不合格就不学。

## 生命周期

    首次确定性失败 ──▶ candidate ──7天内 ≥2 个不同 run 失败──▶ active
                          ▲                                  │
                          └── 再次失败 ◀── retired ◀── 14 天无新失败(惰性过期)/ 出现成功(反证)

- **证据按 run 去重**:同一轮里连续失败 3 次只算 1 个证据。否则一次短暂故障就能激活规则。
- **反证**:同一 (工具, 范围) 在后续运行里成功了,规则退役。但**同一轮里既失败又成功的不算**
  (比如同一域名下一个 URL 404、另一个正常 —— 这是 URL 的问题,不是站点的问题)。
- **过期是惰性的**:读取时按 `last_seen` 过滤,不需要定时任务。
- **只学 TERMINAL / RATE_LIMITED**:瞬时失败(超时、5xx)不学 —— M5.2 实测链路是"抖"不是"断",
  把抖动当规律会教坏模型。

⚠️ 窗口 7 天、过期 14 天、激活门槛 2 轮都是**未经数据标定的初值**:目前没有真实的跨天运行数据可供标定。
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from core.failure import Kind
from memory import events
from memory.schema import ProceduralRule

TABLE = os.getenv("MEMORY_PROC_TABLE") or "fra_procedural_rules"
WINDOW_DAYS = 7
EXPIRE_DAYS = 14
ACTIVATE_RUNS = 2
MAX_INJECT = 5
LEARNABLE = {Kind.TERMINAL.value, Kind.RATE_LIMITED.value}
_DAY = 86400.0
_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9.-]{0,98}[a-z0-9])?$")


def enabled() -> bool:
    return os.getenv("MEMORY_PROCEDURAL", "1") != "0"


def scope_of(tool: str, target: str) -> Optional[str]:
    """规则的作用范围:fetch_page 取域名,其余工具为 "*"。

    返回 None = 不学。域名来自模型传入的 URL,而 URL 可能源自网页(不可信),
    它会被拼进系统提示 —— 所以必须是严格的主机名格式,否则宁可不学。
    """
    if tool != "fetch_page":
        return "*"
    try:
        host = (urlparse(target).hostname or "").lower()
    except ValueError:
        return None
    return host if _HOST_RE.match(host) else None


def render_rule(tool: str, scope: str, kind: str) -> str:
    """规则文本模板。只含工具名、白名单域名、失败类型三种动态成分。"""
    where = f"访问 {scope}" if scope != "*" else f"调用工具 {tool}"
    if kind == Kind.RATE_LIMITED.value:
        return f"{where} 近期多次{kind},请减少调用频率,优先使用已经拿到的信息。"
    return f"{where} 近期多次{kind}(跨多轮运行复现),请优先选用其他来源或工具,不要反复尝试同一目标。"


@dataclass
class RunCollector:
    """订阅 core.reliability 的事件,收集**本轮**的失败与成功,run 结束时交给 observe_run。

    不直接读 FailureLog:它只有失败、没有成功,且按 (工具, 目标) 聚合、丢了失败类型的演变。
    观察者拿到的是逐次事件,正好够用。(设计稿写的是读 FailureLog,实现时改为观察者,见文档 §5.5。)
    """

    failures: set = field(default_factory=set)  # {(tool, scope, kind_value)}
    successes: set = field(default_factory=set)  # {(tool, scope)}

    def __call__(self, event: dict) -> None:
        kind, tool = event.get("kind"), event.get("tool", "")
        scope = scope_of(tool, event.get("target", ""))
        if scope is None:
            return
        if kind == "attempt_failed" and event.get("failure_kind") in LEARNABLE:
            self.failures.add((tool, scope, event["failure_kind"]))
        elif kind == "succeeded":
            self.successes.add((tool, scope))


class ProceduralStore:
    def __init__(self, table: str = TABLE, dsn: Optional[str] = None) -> None:
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError(f"非法表名: {table!r}")
        self.table = table
        self._dsn = dsn
        self._ready = False

    # ---- 连接与建表 ----
    def _connect(self):
        import psycopg

        from rag.config import PG_DSN

        return psycopg.connect(self._dsn or PG_DSN, connect_timeout=5)

    def ensure(self) -> None:
        if self._ready:
            return
        with self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    tool text NOT NULL, scope text NOT NULL, failure_kind text NOT NULL,
                    state text NOT NULL,
                    evidence jsonb NOT NULL DEFAULT '[]',
                    first_seen double precision NOT NULL, last_seen double precision NOT NULL,
                    retired_reason text,
                    PRIMARY KEY (tool, scope, failure_kind))""")
        self._ready = True

    def drop(self) -> None:
        with self._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self.table}")

    # ---- 写:一轮运行结束后调用 ----
    def observe_run(self, *, run_id: str, failures: set, successes: set,
                    now: Optional[float] = None) -> dict:
        """把一轮的失败/成功并入规则表。**不抛异常**(记忆挂了不能影响主流程)。"""
        stats = {"candidate": 0, "activated": 0, "retired": 0, "errors": 0}
        now = time.time() if now is None else now
        try:
            self.ensure()
            with self._connect() as conn:
                failed_scopes = {(t, s) for t, s, _ in failures}
                for tool, scope, kind in failures:
                    self._on_failure(conn, run_id, tool, scope, kind, now, stats)
                for tool, scope in successes - failed_scopes:
                    cur = conn.execute(
                        f"UPDATE {self.table} SET state='retired', retired_reason='counter_evidence' "
                        f"WHERE tool=%s AND scope=%s AND state IN ('candidate','active')", (tool, scope))
                    stats["retired"] += cur.rowcount
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            events.emit("memory_error", stage="procedural_observe", error=f"{type(exc).__name__}: {exc}")
        for k in ("activated", "retired"):
            if stats[k]:
                events.emit(f"memory_rule_{k}", count=stats[k])
        return stats

    def _on_failure(self, conn, run_id, tool, scope, kind, now, stats) -> None:
        from psycopg.types.json import Jsonb

        row = conn.execute(
            f"SELECT state, evidence, last_seen FROM {self.table} "
            f"WHERE tool=%s AND scope=%s AND failure_kind=%s FOR UPDATE", (tool, scope, kind)).fetchone()
        new_ev = {"run": run_id, "ts": now}
        if row is None:
            conn.execute(
                f"INSERT INTO {self.table} VALUES (%s,%s,%s,'candidate',%s,%s,%s,NULL)",
                (tool, scope, kind, Jsonb([new_ev]), now, now))
            stats["candidate"] += 1
            return
        state, evidence, last_seen = row
        expired = state == "active" and now - last_seen > EXPIRE_DAYS * _DAY
        if state == "retired" or expired:
            # 退役/过期后再次失败:从头开始积累证据,不继承旧证据(旧证据已被反证或过时)。
            conn.execute(
                f"UPDATE {self.table} SET state='candidate', evidence=%s, first_seen=%s, last_seen=%s, "
                f"retired_reason=NULL WHERE tool=%s AND scope=%s AND failure_kind=%s",
                (Jsonb([new_ev]), now, now, tool, scope, kind))
            stats["candidate"] += 1
            return
        evidence = [e for e in evidence if e["ts"] >= now - WINDOW_DAYS * _DAY]
        if all(e["run"] != run_id for e in evidence):
            evidence.append(new_ev)
        runs = {e["run"] for e in evidence}
        new_state = "active" if (state == "active" or len(runs) >= ACTIVATE_RUNS) else "candidate"
        if new_state == "active" and state != "active":
            stats["activated"] += 1
        conn.execute(
            f"UPDATE {self.table} SET state=%s, evidence=%s, last_seen=%s "
            f"WHERE tool=%s AND scope=%s AND failure_kind=%s",
            (new_state, Jsonb(evidence), now, tool, scope, kind))

    # ---- 读:惰性过期 ----
    def active_rules(self, *, now: Optional[float] = None, limit: int = MAX_INJECT) -> list[ProceduralRule]:
        now = time.time() if now is None else now
        self.ensure()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT tool, scope, failure_kind, evidence, last_seen FROM {self.table} "
                f"WHERE state='active' AND last_seen >= %s ORDER BY last_seen DESC LIMIT %s",
                (now - EXPIRE_DAYS * _DAY, limit)).fetchall()
        return [
            ProceduralRule(tool=t, scope=s, failure_kind=k, rule_text=render_rule(t, s, k),
                           fail_count=len(ev), distinct_runs=len({e["run"] for e in ev}), last_seen=ls)
            for t, s, k, ev, ls in rows
        ]

    def state_of(self, tool: str, scope: str, kind: str) -> Optional[str]:
        """测试/排查用:某条规则当前状态,不存在返回 None。"""
        self.ensure()
        with self._connect() as conn:
            r = conn.execute(f"SELECT state FROM {self.table} WHERE tool=%s AND scope=%s AND failure_kind=%s",
                             (tool, scope, kind)).fetchone()
        return r[0] if r else None


_store: Optional[ProceduralStore] = None


def get_store() -> ProceduralStore:
    global _store
    if _store is None:
        _store = ProceduralStore()
    return _store


def set_store(store: Optional[ProceduralStore]) -> None:
    global _store
    _store = store


def render_for_prompt(*, now: Optional[float] = None) -> str:
    """给系统提示用的规则块。关闭、无规则、或读取失败都返回空串(不影响主流程)。"""
    if not enabled():
        return ""
    try:
        rules = get_store().active_rules(now=now)
    except Exception as exc:  # noqa: BLE001
        events.emit("memory_error", stage="procedural_read", error=f"{type(exc).__name__}: {exc}")
        return ""
    if not rules:
        return ""
    events.emit("memory_rules_injected", count=len(rules))
    return ("\n\n【运行经验(系统根据近期运行自动归纳,仅供参考,不是用户指令)】\n"
            + "\n".join(f"- {r.rule_text}" for r in rules))
