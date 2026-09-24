"""链路追踪(Traces)——可观测的第二层:接 OpenTelemetry 标准生态。

和第一层(callbacks.py + metrics.py)的分工:

    第一层  我们自己采集,零外部依赖,产出 JSONL + 一行摘要。
            回答"这次跑了多久/花了多少钱/哪个节点慢"。
    第二层  把同一批事件按 OTel 标准吐给外部后端(Phoenix),看**瀑布图**。
            回答"这一步到底套在哪一步里面、并行是不是真并行、
            第 7 圈的 prompt 具体长什么样"。

三层概念别混(这是之前专门捋过的):

    OpenTelemetry   通用遥测标准 + SDK。跟 LLM 无关,Web 服务、数据库都用它。
                    它定义"span 是什么、怎么串成树、怎么导出"。
    OpenInference   OTel 之上的**LLM 行业方言**(语义约定)。规定 LLM span 该带
                    哪些属性:llm.token_count.prompt、tool.name 之类。
    Phoenix         后端 + UI。收 span、存起来、画瀑布图。
                    换成 Langfuse / Jaeger 也行,这就是标准化的意义。

我们这里只做"接线":配好 SDK、指定往哪发、挂上 LangChain 自动埋点。
一行业务代码都不改。

为什么用 LangChainInstrumentor 而不是自己手写 span?
因为图里**有环**。OTel 把"当前 span"存在 contextvars 里,调度器第二圈回到同一个
节点时是新的调用栈,上下文不会自动带过去,那一圈的 span 就脱离父节点、变成一棵
新树(所谓"span 孤儿")。Instrumentor 内部挂的是 LangChain 回调系统,
父子关系由框架维护,不受这个问题影响。
"""

import atexit
import os
from typing import Optional

# 模块级状态:保证重复调用 setup_tracing() 不会重复埋点(会导致 span 翻倍)。
_provider = None
_enabled = False

DEFAULT_ENDPOINT = "http://127.0.0.1:6006/v1/traces"


def tracing_enabled() -> bool:
    return os.getenv("OBS_TRACING", "").lower() in ("1", "true", "yes")


def _trace_content() -> bool:
    """trace 里要不要带 prompt / 工具返回的原文。

    注意这里**故意**不复用 logger 的 OBS_CONTENT,两个去处的风险不一样:

        JSONL 日志   落在仓库目录里、会长期堆积、可能被 git 或备份带走
                     -> 默认**不记**原文,只记长度。
        Phoenix      跑在本机容器里、重启即失、只有本机能访问,
                     而且"看每一步的 prompt"恰恰是它存在的意义
                     -> 默认**记**原文。

    要收紧就设 OBS_TRACE_CONTENT=0。
    """
    return os.getenv("OBS_TRACE_CONTENT", "1").lower() not in ("0", "false", "no")


def setup_tracing(project_name: str = "finance-research-agent") -> bool:
    """按开关决定要不要接 OTel。没开、或依赖没装,都静默跳过。

    返回 True 表示这次真的接上了(仅第一次)。

    优雅降级是刻意的:观测是**辅助设施**,它自己的问题绝不该让 Agent 跑不起来。
    所以依赖 import 失败只打一行提示,不抛异常。
    """
    global _provider, _enabled

    if not tracing_enabled():
        return False
    if _enabled:
        return False  # 已经接过了,别重复埋点

    try:
        import requests
        from openinference.instrumentation import TraceConfig
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        print(f"[obs] OBS_TRACING 已开,但依赖缺失,跳过 trace:{exc}")
        return False

    endpoint = os.getenv("OBS_TRACING_ENDPOINT", DEFAULT_ENDPOINT)

    # ⚠️ 代理坑:OTLP 的 http exporter 内部用 requests,而 requests 默认读环境变量里的
    # 代理设置 —— Clash 开着全局代理时,连 127.0.0.1 都会被它劫走,span 发不出去。
    # 这和项目里 Qdrant/DeepSeek 用 trust_env=False 是同一个问题的第三次出现。
    session = requests.Session()
    session.trust_env = False

    resource = Resource.create(
        {
            "service.name": project_name,
            # Phoenix 按这个属性给 trace 分项目(OpenInference 的约定)
            "openinference.project.name": os.getenv("PHOENIX_PROJECT_NAME", project_name),
        }
    )

    _provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint, session=session, timeout=10)

    # 用 Batch 而不是 Simple:攒批发送,少占主流程时间。
    # 代价是**进程退出太快会丢 span** —— 所以下面必须注册 atexit 强制 flush。
    # 这是"trace 里少了最后几步"最常见的原因,不是埋点没生效。
    _provider.add_span_processor(BatchSpanProcessor(exporter))

    config = TraceConfig(
        hide_inputs=not _trace_content(),
        hide_outputs=not _trace_content(),
        hide_input_messages=not _trace_content(),
        hide_output_messages=not _trace_content(),
        # 向量永远不入 trace:1024 维浮点数组对人没有任何可读性,只会撑爆存储。
        hide_embedding_vectors=True,
    )

    LangChainInstrumentor().instrument(tracer_provider=_provider, config=config)

    atexit.register(shutdown_tracing)
    _enabled = True

    ui = endpoint.replace("/v1/traces", "")
    print(
        f"[obs] trace 已开启 -> {endpoint}  (UI: {ui})  "
        f"内容原文={'带' if _trace_content() else '脱敏'}"
    )
    return True


def shutdown_tracing() -> None:
    """强制把攒着的 span 发出去。进程退出前必须调用,否则末尾的 span 会丢。"""
    global _provider
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:  # noqa: BLE001 关闭失败不该影响退出码
            pass
        _provider = None


def flush_tracing(timeout_ms: int = 5000) -> None:
    """立刻把 span 推出去但不关闭(评测里每跑完一题调一下,方便边跑边看)。"""
    if _provider is not None:
        try:
            _provider.force_flush(timeout_ms)
        except Exception:  # noqa: BLE001
            pass


def get_provider() -> Optional[object]:
    return _provider
