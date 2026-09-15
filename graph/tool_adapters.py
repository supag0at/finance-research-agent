"""把现有工具包装成 LangChain 工具,供 LangGraph 的 ToolNode 使用。

设计要点:**业务函数一行都不改**。tools/finance.py 和 rag/retrieve.py 保持原样,
这里只做"格式适配 + 可靠性包装":

  1. 格式适配:LangChain 需要 pydantic 的参数 schema(对应手写版 registry 里
     手写的那份 JSON Schema);
  2. 可靠性:LangGraph 的节点级 RetryPolicy 只能整节点重试,粒度太粗;
     我们仍然复用自研的 run_reliably,保留"每个工具独立配置超时/重试/降级"的能力。

这正好印证了设计文档里的判断:被框架替代的是骨架,业务逻辑原样保留。

⚠️ 关于文件名:本文件原名 graph/tools.py,与项目根目录下的 tools/ 包**同名**。
直接运行 graph 下的脚本时(python graph/xxx.py),Python 会把 graph/ 放到
sys.path 最前面,于是 `import tools` 会命中这个文件、把真正的 tools/ 包遮蔽掉,
报出 "cannot import name 'finance' from 'tools'" 这种莫名其妙的错。
因此改名为 tool_adapters.py。**顶层包名不要被子目录里的模块名占用。**
"""

from typing import Callable, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from core.reliability import run_reliably
from rag.retrieve import retrieve
from tools import finance


# ---------- 参数 schema(对应手写版 registry.register 里的 parameters) ----------


class StockPriceArgs(BaseModel):
    symbol: str = Field(description="股票代码,例如 NVDA、AAPL")
    days: int = Field(default=30, description="最近多少个交易日,默认 30")


class SearchNewsArgs(BaseModel):
    query: str = Field(description="搜索关键词,例如 'NVDA stock surge reason'")


class FetchPageArgs(BaseModel):
    url: str = Field(description="要抓取的网页地址(http/https)")


class RetrieveArgs(BaseModel):
    query: str = Field(description="检索问题,例如 '英伟达的主要业务板块'")
    top_k: int = Field(default=4, description="返回几段,默认 4")


# ---------- 可靠性包装 ----------


def _stock_fallback(symbol: str = "", days: int = 30) -> dict:
    """行情工具的降级兜底(与 main.py 保持一致)。"""
    return {"symbol": symbol, "error": "行情数据暂时不可用,请勿编造数字。"}


def _reliable_tool(
    func: Callable,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    timeout: float = 15.0,
    retries: int = 2,
    fallback: Optional[Callable] = None,
) -> StructuredTool:
    """把一个普通函数包成"带超时/重试/降级"的 LangChain 工具。"""

    def wrapped(**kwargs) -> str:
        result = run_reliably(
            func=func,
            kwargs=kwargs,
            timeout=timeout,
            retries=retries,
            fallback=fallback,
        )
        # 统一转成字符串,和手写版 registry.call() 的行为一致。
        return str(result)

    return StructuredTool.from_function(
        func=wrapped,
        name=name,
        description=description,
        args_schema=args_schema,
    )


def build_tools() -> list[StructuredTool]:
    """构造全部工具。描述与可靠性参数与手写版 main.py 保持一致,保证可比性。"""
    return [
        _reliable_tool(
            finance.get_stock_price,
            name="get_stock_price",
            description="查询某只股票最近若干个交易日的涨跌幅、最新价、最高/最低价。需要行情数据时调用。",
            args_schema=StockPriceArgs,
            timeout=20.0,
            retries=2,
            fallback=_stock_fallback,
        ),
        _reliable_tool(
            finance.search_news,
            name="search_news",
            description="按关键词做网页搜索,返回相关新闻的标题、链接和摘要。需要了解事件/原因时调用。",
            args_schema=SearchNewsArgs,
            timeout=20.0,
            retries=2,
        ),
        _reliable_tool(
            finance.fetch_page,
            name="fetch_page",
            description="抓取一个网页的正文文本。当搜索结果的摘要不够、需要看细节时调用。",
            args_schema=FetchPageArgs,
            timeout=15.0,
            retries=1,
        ),
        _reliable_tool(
            retrieve,
            name="retrieve",
            description="从本地知识库按语义检索背景资料(公司背景、行业知识、术语解释等)。需要背景知识时调用。",
            args_schema=RetrieveArgs,
            timeout=20.0,
            retries=1,
        ),
    ]
