"""压缩参数。每次 compress 都现读环境变量(不在 import 时固化),评测改环境变量即生效。

## 水位怎么定的(方法同 core/budget.py:先测健康分布,再把线放在它上方)

止损线落进正常区间里,就只是一个随机的提前打断器:正常题被打断一次,
既多付一次归档调用,又让归档块之后的缓存失效一次。所以高水位要只拦失控的运行。

当前标定依据(2026-10-12,deepseek-flash,eval/cases 5 题,v2 不触发归档跑一轮):
  视图峰值 3093 ~ 9872 token(最差:chip_news_01)
  高水位 20000 ≈ 最差健康值的 2 倍;低水位取一半
  估算误差(真实 input_tokens / 估算值,14 个样本):中位数 0.923,最低 0.756,最高 0.988
    —— 估算整体偏高 8%,大段英文网页时偏高可达三成(英文每字符的 token 更少)。
    系数从 0.6 按中位数校正为 0.55;偏高是安全方向(宁可早压一点),所以不再往下调。
  无锚点时(本轮第一次调模型前)误差约 0.97,tools_schema_tokens=800 不用改。
⚠️ 换模型或换题目后必须重新标定。
"""

import os
from dataclasses import dataclass


def _env_num(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return v if v > 0 else default


# 按工具覆盖预览长度(M9 端到端追踪发现)。
# 1500 是从 M3 的截断沿用来的,对 fetch_page 这种"整页网页、有用的只占一小段"合适;
# 但 retrieve 的输出已经被 rerank 筛过、按 9000 字限额打包过,再截到 1500,主 Agent 只看得到第一条证据,
# 末尾的 summary 也看不到。所以已经自带长度上限的工具单独放宽。
# 格式:CTX_PREVIEW_BY_TOOL="retrieve=12000,fetch_page=1500";没列的工具用 preview_chars。
_DEFAULT_PREVIEW_BY_TOOL = "retrieve=12000"


def tool_preview_chars(tool_name: str | None, default: int) -> int:
    """某个工具的预览长度。现读环境变量(和 from_env 一样,评测改了就生效)。"""
    raw = os.getenv("CTX_PREVIEW_BY_TOOL", _DEFAULT_PREVIEW_BY_TOOL)
    for item in raw.split(","):
        name, _, value = item.partition("=")
        if name.strip() == tool_name and value.strip().isdigit():
            return int(value)
    return default


@dataclass(frozen=True)
class ContextConfig:
    high_tokens: int = 20000  # 越过它才归档
    low_tokens: int = 10000  # 一次归档压到它以下
    keep_rounds: int = 2  # 压缩时至少保留最近几圈原文
    preview_chars: int = 1500  # 工具结果超过它就只显示前这么多字 + stub(与 legacy 截断一致)
    read_chunk_chars: int = 1500  # read_tool_result 每次返回多少字
    archive_max_tool_chars: int = 4000  # 喂给归档器的单条工具结果上限
    tokens_per_char: float = 0.55  # 字符→token 的估算系数
    tools_schema_tokens: int = 800  # 无锚点时补上工具 schema 的开销
    entry_tokens: int = 60  # 每条归档条目的预估 token
    max_facts: int = 20
    max_failures: int = 10

    @classmethod
    def from_env(cls) -> "ContextConfig":
        d = cls()
        return cls(
            high_tokens=int(_env_num("CTX_HIGH_TOKENS", d.high_tokens)),
            low_tokens=int(_env_num("CTX_LOW_TOKENS", d.low_tokens)),
            keep_rounds=int(_env_num("CTX_KEEP_ROUNDS", d.keep_rounds)),
            preview_chars=int(_env_num("CTX_PREVIEW_CHARS", d.preview_chars)),
            tokens_per_char=_env_num("CTX_TOKENS_PER_CHAR", d.tokens_per_char),
        )
