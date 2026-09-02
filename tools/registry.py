"""工具注册表:管理 Agent 能用的所有工具。

它负责两件事:
1. get_schemas():收集所有工具的"说明书"(Tool Schema),发给 LLM 看。
   LLM 正是看了这份说明书,才知道有哪些工具、每个要传什么参数。
2. call():根据 LLM 返回的工具名,找到对应函数并执行,返回结果(观察)。

你之前问"Tool Schema 体现在哪、Function Calling 怎么落地"——就在这个文件:
说明书在 get_schemas() 里生成,执行在 call() 里发生。
"""

from typing import Callable, Optional

from core.reliability import run_reliably


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable,
        timeout: float = 15.0,
        retries: int = 2,
        fallback: Optional[Callable] = None,
    ) -> None:
        """登记一个工具。

        除了名字/说明/参数/函数,还可以为每个工具单独配置可靠性策略:
            timeout:  单次执行超时(秒)
            retries:  失败后最多重试几次
            fallback: 全部失败后的兜底函数(可选)
        """
        self._tools[name] = {
            "description": description,
            "parameters": parameters,
            "func": func,
            "timeout": timeout,
            "retries": retries,
            "fallback": fallback,
        }

    def get_schemas(self) -> list[dict]:
        """生成 Tool Schema(OpenAI 格式):告诉 LLM 有哪些工具、怎么用。"""
        schemas = []
        for name, tool in self._tools.items():
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
            )
        return schemas

    def call(self, name: str, arguments: dict) -> str:
        """执行指定工具,返回字符串形式的结果(observation 观察)。

        M2 起,执行不再是"裸调用",而是交给 run_reliably 统一包上
        超时 + 重试 + 降级。这样任何工具出问题,Agent 都不会崩,最坏也只是
        拿到一段兜底说明,继续往下走。
        """
        if name not in self._tools:
            return f"错误:不存在名为 {name} 的工具。"

        tool = self._tools[name]
        result = run_reliably(
            func=tool["func"],
            kwargs=arguments,
            timeout=tool["timeout"],
            retries=tool["retries"],
            fallback=tool["fallback"],
        )
        return str(result)
