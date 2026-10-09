"""记忆事件的观察者(M7)。

和 core/reliability.py 的 `_observers` 是同一个模式:记忆代码只管 emit,
谁想听谁注册(ObsHandler 会注册,把事件计入 RunRecord.memory)。
这样 memory/ 不用依赖 obs/,默认没人听时零开销。

为什么需要它:旧代码里写入失败是 `except Exception: pass`,记忆坏了没人知道。
M5.2 定下的原则是"观测要放在知道真相的那一层"——写入/冲突处理发生在这里,
所以事件也必须从这里发出来。

事件 kind:
  memory_extracted    抽取器产出了 N 条候选
  memory_added / memory_updated / memory_invalidated / memory_noop
  memory_skipped      没有值得记的,或某条候选校验不过被丢弃(带 reason)
  memory_error        某个阶段失败(带 stage、error);写入失败**不影响**主流程

局限:模块级列表,不按调用上下文隔离。我们串行运行,够用(同 reliability 的说明)。
"""

from typing import Callable

_observers: list[Callable[[dict], None]] = []


def add_observer(fn: Callable[[dict], None]) -> None:
    _observers.append(fn)


def remove_observer(fn: Callable[[dict], None]) -> None:
    if fn in _observers:
        _observers.remove(fn)


def emit(kind: str, **fields) -> None:
    event = {"kind": kind, **fields}
    for fn in list(_observers):
        try:
            fn(event)
        except Exception:  # noqa: BLE001 观测失败绝不能影响记忆写入
            pass
