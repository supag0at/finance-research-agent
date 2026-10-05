"""Checkpointer 选择:持久化执行(durable execution)——M6 ④。

## 为什么这件事比看起来重要

改造前 `build_graph` 默认用 `InMemorySaver`。这不是"够用的简化",而是**零持久性**:
进程一死,整轮调研的状态全部蒸发。

Temporal 在讲 Agent 可靠性时直接点了这个:

    LangGraph's interrupt() provides durable execution **when paired with a persistent
    checkpointer**. That qualifier matters. This demo compiles the graph with an
    in-memory checkpointer, so the suspended graph lives in process memory and
    wouldn't survive a crash.

而面试里被看重的那个认知是:

    an agent run is a stateful long-lived process that needs
    **the same durability guarantees as a payment workflow**.

一轮财经调研要跑几十秒到几分钟、花真金白银、调十几次外部接口。
中间崩一次就全部重来 —— 既浪费已经花掉的钱,也可能重复触发副作用。
换成落库的 checkpointer,崩溃后可以从最后一个成功的 superstep 续跑。

## 为什么是小改动大收益

接缝 M5 时就留好了(`build_graph(checkpointer=...)`),而且 Postgres 是现成的
(RAG 第三层用的 `newsrag` 库)。所以这一项的代码量很小,但它补上的是一个
**量级上的能力差距**:从"不可恢复"到"可恢复"。

## 刻意的取舍

**默认开启,但失败自动退回内存。** 可靠性改造本身不该成为新的故障点 ——
Postgres 没起来的时候,退回 `InMemorySaver` 继续跑(并在 stderr 说明),
比直接报错更符合这个项目其他地方的一贯做法(长期记忆、摘要、tracing 都是这个路子)。

**表名由 langgraph 自己管。** 它会建 `checkpoints` / `checkpoint_writes` /
`checkpoint_blobs` 等表。我们自己的 RAG 表都有 `fra_` 前缀,这几张没有 ——
因为它们属于框架,不是我们的业务数据,混在一个库里但归属清楚。
"""

import os
import sys
from typing import Optional

_pg_saver = None  # 进程内复用:连接池不该每次建图都新建一个
# ⚠️ 这个全局变量不是多余的,它是**必须的**:见 default_checkpointer 里的说明。
# 只保住 saver 而不保住上下文管理器,连接会被 GC 关掉。
_pg_cm = None


def durable_enabled() -> bool:
    """CHECKPOINT_BACKEND=memory 可强制关掉落库(评测想要绝对干净的冷启动时用)。"""
    return os.getenv("CHECKPOINT_BACKEND", "postgres").strip().lower() != "memory"


def default_checkpointer() -> object:
    """返回一个 checkpointer。优先 Postgres,不可用则退回内存。"""
    global _pg_saver, _pg_cm

    from langgraph.checkpoint.memory import InMemorySaver

    if not durable_enabled():
        return InMemorySaver()

    if _pg_saver is not None:
        return _pg_saver

    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        from rag.config import PG_DSN

        # from_conn_string 返回的是上下文管理器。我们要的是一个**进程生命周期内
        # 长期存活**的 saver,所以手动 __enter__ 而不是用 with —— 否则出了作用域
        # 连接就关了。进程退出时由 OS 回收;这不优雅,但比把连接池的生命周期
        # 搅进建图逻辑要简单,而且我们是单进程 CLI/评测场景。
        #
        # ⚠️ 实测踩坑:**必须同时把 cm 也挂到模块全局上**。
        # 第一版只存了 saver,`cm` 是个局部变量,函数一返回就被 GC;
        # 而 from_conn_string 是 @contextmanager 生成器,GC 时会执行 __exit__,
        # 于是连接被关掉。表能建成功(setup() 在连接还活着时跑完了),
        # 之后第一次真正读检查点就炸 `OperationalError: the connection is closed`。
        # 教训:手动 __enter__ 时,"谁持有 cm"就是"连接活多久"。
        cm = PostgresSaver.from_conn_string(PG_DSN)
        saver = cm.__enter__()
        saver.setup()  # 幂等建表
        _pg_cm, _pg_saver = cm, saver
        return saver
    except Exception as exc:  # noqa: BLE001
        # 退回内存。可靠性改造本身不该成为新的故障点。
        print(
            f"[checkpoint] Postgres 不可用,退回进程内存储(重启即失状态):"
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return InMemorySaver()


def describe() -> str:
    """给日志/文档用的一句话说明当前用的是哪种。"""
    if not durable_enabled():
        return "memory(CHECKPOINT_BACKEND=memory)"
    return "postgres" if _pg_saver is not None else "postgres(尚未初始化)"
