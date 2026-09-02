"""模型封装层:把"调用大模型"这件事包起来。

为什么要单独封装?这样 Agent 主循环就不用关心底层用的是 DeepSeek、Kimi
还是 OpenAI。想换模型,只改这个文件或改环境变量即可,主循环一行都不用动。
这正是招聘 JD 里反复强调的:"把框架封装成可替换能力,不要把业务绑死在
某个具体工具上。"

这三家的接口都兼容 OpenAI 格式,所以用同一个 openai 客户端,只是
base_url / model 不同。
"""

import os

from openai import OpenAI


class LLM:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("LLM_API_KEY", ""),
            # 默认用 DeepSeek(便宜、工具调用支持好);想换就改环境变量
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        )
        # 注意:deepseek-chat / deepseek-reasoner 这两个旧名字已于 2026/07/24 废弃,
        # 现在应使用 V4 显式名:deepseek-v4-flash(便宜)或 deepseek-v4-pro(旗舰)。
        self.model = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

        # 思考模式开关。打开后,模型会先"思考"再回答,思考过程放在
        # reasoning_content 字段里。设环境变量 LLM_THINKING=1 即可开启。
        self.thinking = os.environ.get("LLM_THINKING", "").lower() in ("1", "true", "yes")

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
        return response.choices[0].message
