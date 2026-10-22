"""RAG 子图的状态(M9 Step 4)。和主图 AgentState 完全隔离:主 Agent 只看到 pack 出来的证据。"""

from typing import TypedDict


class Need(TypedDict, total=False):
    id: str              # "n1"、"n2"
    desc: str            # 这个信息需求是什么(给 grade 看)
    queries: list[str]   # 检索用的查询;q0(原问题)总在第一个
    filters: dict        # 过白名单之后的过滤条件 {"ticker": [...], "fiscal_year": [...], "doc_type": [...]}
    filter_valid: bool   # 分析器给了过滤条件且全部通过白名单(快速通道的前提)
    status: str          # "pending" | "sufficient" | "missing" | "web"
    reason: str          # grade 给的理由
    hits: list[dict]     # 本需求当前的证据(hybrid.search 的 hits,带 rerank 分)
    relaxed: list[str]   # 过滤放宽记录:"drop fiscal_year"(结果为空自动放宽)/ "rewrite drop ..."(改写主动去掉)
    kb_absent: bool      # 要的公司不在库里:不改写重检,直接联网


class RagState(TypedDict, total=False):
    question: str
    intent: str          # "kb_fact" | "realtime" | "out_of_scope"
    needs: list[Need]
    round: int           # 已完成的检索轮数
    degraded: list[str]  # 降级记录:"analyze" / "grade" / "rerank" / "dense" / "rewrite" / "web"
    notes: list[str]     # 给主 Agent 看的说明(财年口径、来源性质、缺失)
    events: list[str]    # 给人看的过程记录(观测 / 评测用)
    llm_tokens: int
    t0: float
    output: list[dict]   # pack 的结果:证据列表 + 末尾一条 summary
    opts: dict           # 开关(消融用):grade / web / hyde / max_rounds,缺省取 rag.config
