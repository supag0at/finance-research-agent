"""日志(Logs):把发生过的事按结构化格式落盘。

两个文件,对应两个支柱,刻意分开:

    obs/runs/events-<日期>.jsonl   逐事件(Logs)   —— 回答"当时发生了什么"
    obs/runs/runs-<日期>.jsonl     每次一行(Metrics)—— 回答"整体健康度/趋势"

为什么坚持 JSONL 而不是人类可读的日志行?因为要**聚合**。
"这周 search_news 失败率多少"这种问题,结构化数据一行 jq/pandas 就能答,
自由文本日志得写正则。日志的读者主要是程序,不是人。

⚠️ 脱敏:实测发现自动埋点默认会把 prompt 全文、工具返回全文都记下来。
这里反过来:**默认只记元数据**
(长度、耗时、token),要记全文必须显式设 OBS_CONTENT=1。
生产环境里把用户输入原文写进日志是事故,不是功能。
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_RUNS_DIR = Path(__file__).parent / "runs"


def enabled() -> bool:
    """总开关。设 OBS=0 可以完全关掉落盘(评测大批量跑时可能想关)。"""
    return os.getenv("OBS", "1").lower() not in ("0", "false", "no")


def keep_content() -> bool:
    """要不要记录内容原文。默认不记。"""
    return os.getenv("OBS_CONTENT", "").lower() in ("1", "true", "yes")


def redact(value: Any, limit: int = 200) -> Any:
    """对内容做脱敏:默认只留长度,开了开关才留截断后的原文。"""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if not keep_content():
        return {"len": len(text)}
    return text[:limit] + (f"...(共{len(text)}字)" if len(text) > limit else "")


def _write(filename: str, payload: dict) -> None:
    if not enabled():
        return
    try:
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False)
        with (_RUNS_DIR / filename).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # 观测失败绝不能影响主流程。宁可丢日志,不能挂 Agent。
        pass


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def log_event(event: str, run_id: str, **fields) -> None:
    """记录一个事件(Logs)。"""
    _write(
        f"events-{_today()}.jsonl",
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": run_id,
            "event": event,
            **fields,
        },
    )


def log_run(record_json: dict) -> None:
    """记录一次运行的汇总(Metrics)。"""
    _write(f"runs-{_today()}.jsonl", {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record_json})
