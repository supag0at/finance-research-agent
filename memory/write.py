"""记忆写入编排(M7):抽取 → 校验 → 冲突处理 → 落库。

对外只有一个入口 `write_memories`,**绝不抛异常**:记忆是锦上添花,不能让主流程因它失败。
但"不抛"不等于"吞掉" —— 每个失败都会发 `memory_error` 事件(旧代码是 `except: pass`,
记忆坏了没人知道)。

## 同步写入的取舍(选型已定,见设计文档 §3.2)

写入在 remember 节点里**同步**完成:用户要多等一次抽取调用 + 一次批量 embedding
(本地实测量级几秒,待第 4 步接线后用 RunRecord 回填)。换来三样东西:
评测确定(跑完立刻能读)、进程退出不丢数据、记忆事件能记进当次 RunRecord。
**生产环境应改成后台异步**(一个单线程队列保证写入串行,避免两个 UPDATE 同时改同一条记录;
进程退出时 flush;评测时同步)。这里不做,是因为现阶段评测确定性比延迟重要。

## 冲突处理:先精确、后模糊,大部分不调 LLM

语义记忆按 (user_id, key, subject) 精确匹配 —— 持仓、偏好这类最常见的情形一步到位:
  值相同 → NOOP;值不同 → 新记录取代旧记录(旧的标 superseded,不删除)。
只有 key 没命中、但向量相似度 ≥ FUZZY_THRESHOLD 时,才让 LLM 判定 ADD/UPDATE/INVALIDATE/NOOP。
没有 DELETE:UPDATE(被取代)和 INVALIDATE(不再成立)都只改状态,历史始终可追溯。

## ⚠️ 顺序:先写新的,成功之后才让旧的失效

`store.add_many` 成功**之后**才执行 supersede/invalidate。反过来的话,embedding 一失败就会留下
"旧的已失效、新的没写进去"—— 用户的持仓凭空消失。这和 M6 ⑥ ingest 先盖章后做事是同一类 bug。
代价是中途失败时可能出现"新旧都 active"的短暂重复,但那是可恢复的(下次写入同 key 会再次取代),
而丢数据不可恢复。
"""

import re
import uuid
from datetime import date
from typing import Optional

from memory import events
from memory.extract import extract, judge_conflict
from memory.schema import MemoryRecord, Source

# ⚠️ 以下阈值都是未经评测标定的初值(M6 教训:阈值落在正常区间里会变成随机终止器)。
FUZZY_THRESHOLD = 0.85  # key 没命中时,相似度多高才值得让 LLM 判断冲突
EPISODIC_DEDUP_THRESHOLD = 0.95  # 情景记忆相似度多高算"同一件事问了又问",直接跳过
EPISODIC_IMPORTANCE_CAP_ON_STOP = 4  # 提前收尾的结论不够可靠,压低重要性

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _norm_key(category: str, key: str) -> str:
    """规范化 key:`category:对象`,英文对象(ticker)统一大写,避免 nvda/NVDA 被当成两条。"""
    key = (key or "").strip()
    if ":" not in key:
        key = f"{category}:{key}"
    head, _, obj = key.partition(":")
    obj = obj.strip()
    if obj.isascii():
        obj = obj.upper() if head in ("holding", "watchlist") else obj
    return f"{head.strip()}:{obj}"


def _int_importance(v) -> int:
    try:
        return max(1, min(10, int(v)))
    except (TypeError, ValueError):
        return 5


def _build_record(
    c: dict, *, user_id: str, run_id: str, thread_id: str, stop_reason: str
) -> MemoryRecord:
    """候选 dict → 经 pydantic 校验的 MemoryRecord。不合法抛 ValueError(调用方记 skipped)。"""
    typ = c.get("type")
    if typ not in ("semantic", "episodic"):
        raise ValueError(f"type 非法: {typ!r}")
    text = str(c.get("text") or "").strip()
    if not text:
        raise ValueError("text 为空")

    as_of = c.get("as_of")
    as_of = as_of if isinstance(as_of, str) and _DATE_RE.match(as_of) else None
    importance = _int_importance(c.get("importance"))
    tickers = c.get("tickers") if isinstance(c.get("tickers"), list) else []
    value = c.get("value") if isinstance(c.get("value"), dict) else None

    kw: dict = dict(
        id=str(uuid.uuid4()),
        user_id=user_id,
        type=typ,
        text=text,
        tickers=[str(t) for t in tickers],
        importance=importance,
        as_of=as_of,
        subject=str(c.get("subject") or "self"),
    )
    if typ == "semantic":
        category = c.get("category")
        if not category or not c.get("key"):
            raise ValueError("语义记忆缺少 category 或 key")
        kw.update(
            category=category,
            key=_norm_key(category, str(c["key"])),
            value=value,
            source=Source(kind="user_statement", run_id=run_id, thread_id=thread_id),
        )
    else:
        if stop_reason:
            # 提前收尾的结论可能不完整:明说,并压低重要性。
            kw["text"] = f"{text}(本轮因「{stop_reason}」提前收尾,结论可能不完整)"
            kw["importance"] = min(importance, EPISODIC_IMPORTANCE_CAP_ON_STOP)
        kw.update(
            topic=str(c["topic"]) if c.get("topic") else None,
            source=Source(kind="agent_conclusion", run_id=run_id, thread_id=thread_id),
        )
    return MemoryRecord(**kw)  # pydantic 校验失败抛 ValidationError(是 ValueError 子类)


def write_memories(
    *,
    user_id: str,
    question: str,
    answer: str,
    llm,
    store,
    run_id: str = "",
    thread_id: str = "",
    stop_reason: str = "",
    today: Optional[str] = None,
) -> dict:
    """把本轮对话写入记忆。返回计数;**不抛异常**。"""
    stats = {"extracted": 0, "added": 0, "updated": 0, "invalidated": 0,
             "noop": 0, "skipped": 0, "errors": 0}
    today = today or date.today().isoformat()

    def err(stage: str, exc: BaseException) -> None:
        stats["errors"] += 1
        events.emit("memory_error", stage=stage, error=f"{type(exc).__name__}: {exc}")

    def skip(reason: str) -> None:
        stats["skipped"] += 1
        events.emit("memory_skipped", reason=reason)

    # ① 读已有画像(给抽取器做上下文)。读不到不致命,当作空画像继续。
    try:
        existing = store.active_semantic(user_id=user_id)
    except Exception as exc:  # noqa: BLE001
        err("load_profile", exc)
        existing = []

    # ② 抽取 + 筛选
    try:
        cands = extract(llm, question=question, answer=answer, existing=existing,
                        stop_reason=stop_reason, today=today)
    except Exception as exc:  # noqa: BLE001
        err("extract", exc)
        return stats
    stats["extracted"] = len(cands)
    events.emit("memory_extracted", count=len(cands))
    if not cands:
        skip("抽取器判定没有值得记的内容")
        return stats

    # ③ 逐条处理,先攒着,统一落库
    to_add: dict[tuple[str, str, str], tuple[MemoryRecord, Optional[str]]] = {}
    # 键 = (type/key/subject 的组合);同批次里同一 key 出现两次,后者覆盖前者。
    invalidate_ids: list[str] = []

    for idx, c in enumerate(cands):
        try:
            is_retract = bool(c.get("retract")) and c.get("type") == "semantic"
            if is_retract:
                category, key = c.get("category"), c.get("key")
                if not category or not key:
                    skip("撤销缺少 category 或 key")
                    continue
                nk = _norm_key(str(category), str(key))
                old = store.find_by_key(user_id=user_id, key=nk, subject=str(c.get("subject") or "self"))
                if old is None:
                    skip(f"要撤销的 {nk} 没有有效记录")
                else:
                    invalidate_ids.append(old.id)
                continue

            rec = _build_record(c, user_id=user_id, run_id=run_id,
                                thread_id=thread_id, stop_reason=stop_reason)

            if rec.type == "episodic":
                dup = store.search(user_id=user_id, query=rec.text, type="episodic",
                                   top_k=1, threshold=EPISODIC_DEDUP_THRESHOLD)
                if dup:
                    stats["noop"] += 1
                    events.emit("memory_noop", reason="情景记忆近似重复", text=rec.text[:40])
                else:
                    to_add[("episodic", rec.id, "")] = (rec, None)
                continue

            # ---- 语义记忆:先精确(key),后模糊(向量 + LLM) ----
            old = store.find_by_key(user_id=user_id, key=rec.key, subject=rec.subject)
            if old is not None:
                same = (old.value == rec.value) if rec.value is not None else (old.text == rec.text)
                if same:
                    stats["noop"] += 1
                    events.emit("memory_noop", key=rec.key)
                else:
                    to_add[("semantic", rec.key, rec.subject)] = (rec, old.id)
                continue

            near = store.search(user_id=user_id, query=rec.text, type="semantic",
                                top_k=3, threshold=FUZZY_THRESHOLD)
            if not near:
                to_add[("semantic", rec.key, rec.subject)] = (rec, None)
                continue

            cand_records = [r for r, _ in near]
            try:
                verdict = judge_conflict(llm, new=c, candidates=cand_records)
            except Exception as exc:  # noqa: BLE001
                err("judge", exc)
                verdict = {"action": "ADD", "target_id": None}  # 判不了就当新信息:重复好过丢失
            valid_ids = {r.id for r in cand_records}
            action, target = verdict["action"], verdict.get("target_id")
            if action in ("UPDATE", "INVALIDATE") and target not in valid_ids:
                action, target = "ADD", None  # 模型给了不存在的 id,不能信
            if action == "NOOP":
                stats["noop"] += 1
                events.emit("memory_noop", key=rec.key, reason="判定为重复")
            elif action == "INVALIDATE":
                invalidate_ids.append(target)
            elif action == "UPDATE":
                to_add[("semantic", rec.key, rec.subject)] = (rec, target)
            else:
                to_add[("semantic", rec.key, rec.subject)] = (rec, None)

        except ValueError as exc:  # 含 pydantic.ValidationError:候选不合法,丢弃这一条
            skip(f"候选 #{idx} 校验失败: {str(exc).splitlines()[0][:80]}")
        except Exception as exc:  # noqa: BLE001
            err("candidate", exc)

    # ④ 落库:**先写新的**
    records = [r for r, _ in to_add.values()]
    try:
        store.add_many(user_id=user_id, records=records)
    except Exception as exc:  # noqa: BLE001
        err("add", exc)
        return stats  # 新的没写进去,就绝不能让旧的失效

    for rec, old_id in to_add.values():
        if old_id is None:
            stats["added"] += 1
            events.emit("memory_added", type=rec.type, key=rec.key, text=rec.text[:40])
        else:
            stats["updated"] += 1
            events.emit("memory_updated", key=rec.key, old_id=old_id, new_id=rec.id)

    # ⑤ 新的已落库,现在才让旧的失效
    for rec, old_id in to_add.values():
        if old_id is not None:
            try:
                store.supersede(user_id=user_id, old_id=old_id, new_id=rec.id)
            except Exception as exc:  # noqa: BLE001
                err("supersede", exc)
    for old_id in invalidate_ids:
        try:
            store.supersede(user_id=user_id, old_id=old_id, new_id=None, status="invalidated")
            stats["invalidated"] += 1
            events.emit("memory_invalidated", old_id=old_id)
        except Exception as exc:  # noqa: BLE001
            err("invalidate", exc)
    return stats
