"""组装 RAG 子图(M9 Step 4),并提供 run():主 Agent 的 retrieve 工具在 RAG_MODE=agentic 时调它。

子图和主图是两张独立的图:主 Agent 把它当成一个工具,看不到里面的分析、评估、改写过程,
只拿到 pack 的证据列表——中间过程不进主 Agent 的上下文(和 M8 压缩同一个目的)。
"""

import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from memory import events
from rag.agentic import nodes
from rag.agentic.state import RagState


@lru_cache(maxsize=1)
def build():
    g = StateGraph(RagState)
    g.add_node("analyze", nodes.analyze)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("rewrite", nodes.rewrite)
    g.add_node("web_fallback", nodes.web_fallback)
    g.add_node("pack", nodes.pack)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", lambda s: "retrieve" if s.get("intent") == "kb_fact" else "pack",
                            {"retrieve": "retrieve", "pack": "pack"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", nodes.route_after_grade,
                            {"pack": "pack", "rewrite": "rewrite", "web_fallback": "web_fallback"})
    g.add_conditional_edges("rewrite", nodes.route_after_rewrite, {"retrieve": "retrieve", "pack": "pack"})
    g.add_edge("web_fallback", "pack")
    g.add_edge("pack", END)
    return g.compile()


def run(question: str, opts: dict | None = None, full_state: bool = False):
    """返回证据列表(第一条是 summary,之后是证据);full_state=True 时返回整个子图状态(评测 / 演示用)。"""
    state = build().invoke({"question": question, "t0": time.time(), "opts": opts or {},
                            "degraded": [], "notes": [], "events": [], "llm_tokens": 0},
                           config={"recursion_limit": 20})
    sm = state["output"][0]  # summary 在最前(pack)
    status = [v["status"] for v in sm["coverage"].values()]
    # 观测:子图的 LLM 调用不走 LangChain,ObsHandler 看不到,这里主动报给 RunRecord.rag(M5.2 的原则:
    # 观测放在知道真相的那一层)。memory.events 只是一个通用的 emit/observer,没人听时零开销。
    events.emit("rag_run", rounds=sm["rounds"], llm_tokens=sm["llm_tokens"], latency_ms=sm["latency_ms"],
                needs=len(status), missing=status.count("missing"), web=status.count("web_only"),
                fast_path=any("快速通道" in e for e in state.get("events") or []), degraded=sm["degraded"])
    return state if full_state else state["output"]
