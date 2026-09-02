"""工具注册表:管理 Agent 能用的所有工具。

它负责两件事:
1. get_schemas():收集所有工具的"说明书"(Tool Schema),发给 LLM 看。
   LLM 正是看了这份说明书,才知道有哪些工具、每个要传什么参数。
2. call():根据 LLM 返回的工具名,找到对应函数并执行,返回结果(观察)。

你之前问"Tool Schema 体现在哪、Function Calling 怎么落地"——就在这个文件:
说明书在 get_schemas() 里生成,执行在 call() 里发生。
"""

from typing import Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        func: Callable,
    ) -> None:
        """登记一个工具:名字、说明、参数结构、真正执行的函数。"""
        self._tools[name] = {
            "description": description,
            "parameters": parameters,
            "func": func,
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

        这里用 try/except 兜底:工具出错时不让整个 Agent 崩掉,而是把错误
        信息作为观察返回给 LLM——这是"可靠性"最朴素的一步,M2 会继续加
        重试、超时、降级等。
        """
        if name not in self._tools:
            return f"错误:不存在名为 {name} 的工具。"
        try:
            result = self._tools[name]["func"](**arguments)
            return str(result)
        except Exception as exc:  # noqa: BLE001 教学项目,先统一兜底
            return f"工具 {name} 执行出错:{exc}"
