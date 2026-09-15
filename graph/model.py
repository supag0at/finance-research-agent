"""LangGraph 版的模型接入层,对应手写版的 core/llm.py。

区别:手写版直接用 openai 客户端;这里用 LangChain 的 ChatOpenAI,
因为只有 LangChain 的消息/工具格式才能和 LangGraph 的 ToolNode 无缝配合。

⚠️ 关键:必须继承手写版那个 trust_env=False 的坑。DeepSeek 是国内可直连服务,
一旦走了 Clash 系统代理就会报 "Authentication Fails (governor)"。
详见 docs/troubleshooting/01-环境与网络.md。
"""

import os

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def build_chat_model(**kwargs) -> ChatOpenAI:
    """构造聊天模型。参数来自 .env,与手写版共用同一套配置。"""
    extra: dict = {}
    if os.environ.get("LLM_THINKING", "").lower() in ("1", "true", "yes"):
        # DeepSeek 思考模式:非标准字段,通过 extra_body 透传。
        extra = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}

    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY", ""),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        # 绕过系统代理,直连 DeepSeek(和 core/llm.py 同样的处理)
        http_client=httpx.Client(trust_env=False, timeout=60),
        # langchain-openai 默认会注入自己的 httpx transport(为了 TCP keepalive),
        # 检测到系统代理时会刷警告。我们的代理策略完全由上面这个 http_client 决定,
        # 置空即可让它别插手。
        http_socket_options=(),
        extra_body=extra or None,
        **kwargs,
    )
