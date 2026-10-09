"""模型封装层:把"调用大模型"这件事包起来。

为什么要单独封装?这样 Agent 主循环就不用关心底层用的是 DeepSeek、Kimi
还是 OpenAI。想换模型,只改这个文件或改环境变量即可,主循环一行都不用动。
这正是招聘 JD 里反复强调的:"把框架封装成可替换能力,不要把业务绑死在
某个具体工具上。"

这三家的接口都兼容 OpenAI 格式,所以用同一个 openai 客户端,只是
base_url / model 不同。
"""

import json
import os

import httpx
from dotenv import load_dotenv
from openai import OpenAI

# 保证无论从哪个入口导入本模块,.env 都已被加载进 os.environ。
# 否则 os.environ.get("LLM_API_KEY") 会拿到空字符串,导致 DeepSeek 报
# "Authentication Fails (governor)"(看起来像 key 失效,实则 key 根本没读到)。
load_dotenv()


class LLM:
    def __init__(self) -> None:
        # 关键:trust_env=False 让底层 httpx 忽略系统代理(Windows 上 Clash 全局模式会
        # 在注册表里设系统代理)。否则请求会被代理转发到境外节点,DeepSeek 判成鉴权
        # 异常(报 "Authentication Fails (governor)"),即使 key 完全有效也会失败。
        # 这和 rag/embedding、rag/store 绕代理是同一个原因。
        self.client = OpenAI(
            api_key=os.environ.get("LLM_API_KEY", ""),
            # 默认用 DeepSeek(便宜、工具调用支持好);想换就改环境变量
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            http_client=httpx.Client(trust_env=False, timeout=60),
        )
        # 注意:deepseek-chat / deepseek-reasoner 这两个旧名字已于 2026/07/24 废弃,
        # 现在应使用 V4 显式名:deepseek-v4-flash(便宜)或 deepseek-v4-pro(旗舰)。
        self.model = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

        # 思考模式开关。打开后,模型会先"思考"再回答,思考过程放在
        # reasoning_content 字段里。设环境变量 LLM_THINKING=1 即可开启。
        self.thinking = os.environ.get("LLM_THINKING", "").lower() in ("1", "true", "yes")

    @staticmethod
    def _message_of(response, where: str):
        """从响应里取出消息,并对"空响应"给出可诊断的报错。

        为什么需要这个?M5.2 加上可观测之后,故意让工具持续失败跑了一轮,
        `complete()` 收到了 None 响应,直接 `response.choices` 抛
        AttributeError: 'NoneType' object has no attribute 'choices' ——
        报错信息完全看不出发生了什么。这里把上下文补齐,便于下次定位。
        """
        if response is None or not getattr(response, "choices", None):
            raise RuntimeError(
                f"{where}:模型返回空响应(response={response!r})。"
                "常见原因:输入超出上下文上限、被内容过滤拦截、"
                "或上游返回了非标准错误体。"
            )
        return response.choices[0].message

    def chat(self, messages: list[dict], tools: list[dict]):
        """发一轮对话给模型,返回模型的回复消息。

        返回的消息里可能带 tool_calls(模型想调工具),也可能只有 content
        (模型给出了最终答案)。到底是哪种,由主循环去判断。
        """
        # 思考模式相关的参数不是 OpenAI 标准字段,通过 extra_body 透传给 DeepSeek。
        extra_body = {}
        if self.thinking:
            extra_body = {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 让模型自己决定:要不要调工具、调哪个
            extra_body=extra_body,
        )
        return self._message_of(response, "LLM.chat")

    def complete(self, messages: list[dict]) -> str:
        """纯文本补全:不带工具,只要模型返回一段文字。

        用于"摘要压缩"这类内部任务(短期记忆里把长历史压成要点)。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return self._message_of(response, "LLM.complete").content or ""

    def complete_json(self, messages: list[dict]) -> dict:
        """要求模型返回一个 JSON 对象并解析(M7:记忆抽取、冲突判定用)。

        用 JSON 模式而不是 strict schema:DeepSeek 对 strict 的支持我们没验证过。
        JSON 模式只保证"能解析",**字段对不对由调用方校验**(记忆写入走 pydantic),
        校验不过就跳过并记事件,而不是写入残缺数据。
        注意:DeepSeek 的 JSON 模式要求 prompt 里出现 "json" 字样。
        解析失败直接抛 ValueError,由调用方决定怎么降级。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = (self._message_of(response, "LLM.complete_json").content or "").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"模型返回的不是合法 JSON: {text[:120]!r}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"期望 JSON 对象,实际是 {type(data).__name__}")
        return data
