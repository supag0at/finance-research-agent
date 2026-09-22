"""obs —— 自建可观测层(Logs / Metrics / Traces 三大支柱的第一层)。

设计见 docs/design/M5.2-可观测性.md,模块讲解见 obs/OBS.md。

对外只需要两样东西:
    ObsHandler   挂到 LangGraph 的 config["callbacks"] 上,自动采集
    RunRecord    一次运行的汇总(耗时/token/成本/节点/工具)
"""

from obs.callbacks import ObsHandler
from obs.metrics import RunRecord

__all__ = ["ObsHandler", "RunRecord"]
