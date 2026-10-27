"""MCP Server —— 把本项目的四个工具暴露成标准 MCP 服务(M5.3)。

## 这个文件解决什么问题

在它之前,同一批工具在项目里被定义了**两遍**:
  - `tools/registry.py`      手写版:手写 JSON Schema + 手写 description
  - `graph/tool_adapters.py` 图版:  pydantic BaseModel + 手写 description
同一个 `finance.get_stock_price`,参数写两遍、说明文案抄两遍。这就是 MCP 要
解决的 M×N 问题在单个项目里的缩影(M=2 个框架,N=4 个工具)。

FastMCP 把这件事收敛成一份:**函数签名自动变成参数 schema,docstring 自动
变成工具说明**。下面每个 @mcp.tool() 函数就是唯一的定义源。

## 三个关键设计决策(详见 docs/design/M5.3-MCP.md)

1. **可靠性层放在服务端**:超时/重试/降级都在这里,而不是留给调用方。
   因为失败语义和兜底文案是工具自己的业务知识(见 tools/fallbacks.py)。
   好处是任何客户端(我们的 Agent、Claude Desktop、Cursor)连上来都自带韧性。

2. **SSRF 护栏必须跟着工具走**:`fetch_page` 内部就会先过 `core/sandbox.py`
   的护栏,所以它天然留在服务端。这不是偏好而是硬要求 ——
   **把内部函数变成服务,攻击面就变了**:以前只有自己的 Agent 能调它,
   现在任何连上来的客户端都能。护栏要是留在客户端侧,等于没有。
   (已实测:通过 MCP 调 `fetch_page("http://169.254.169.254/...")` 被拦下。)

3. **观测在进程边界会缺一块**:见下面 _log_reliability 处的说明。

## stdio 传输的铁律

本 server 默认走 stdio:客户端把它当子进程拉起,用**标准输入输出**收发
JSON-RPC。因此 —— **绝对不能往 stdout 打印任何东西**。一句 print() 就会
插进协议流里,破坏 JSON 解析,而且报错信息通常完全看不出是这个原因。
所有日志必须走 stderr(下面统一用 _log)。

## ⚠️ 已知未解决:retrieve 在 MCP 子进程里极慢

`retrieve` 走 MCP 时,embedding 那一步要 ~52 秒(直接在进程内调只要 ~2 秒),
于是被 run_reliably 的 20 秒超时判成失败。已排查到的事实:

  - `sleep(1)` 精确 1.00s，本地 Qdrant HTTP 0.23s  → 工作线程没有被饿死
  - DNS 解析 0.05s                                  → 不是域名解析问题
  - **裸 httpx POST 同一个 embedding 接口:1.74s,200 OK**
  - **openai SDK 调同一个接口(已关重试):52.46s**
  - 把 MCP 过滤后的环境变量原样喂给一个普通子进程:2.84s,正常
    → 所以也不是 `get_default_environment()` 剥掉变量导致的

结论:范围锁定在 **openai SDK 内部**,且只在"被当作 MCP 子进程拉起"时出现。
下一步该量 SDK 拼 `X-Stainless-*` 请求头时的 `platform.*` 探测调用。
**尚未定位,因此 MCP_TOOLS 默认关闭。** 另外三个工具(get_stock_price /
search_news / fetch_page)不经过 openai SDK,走 MCP 正常。
"""

import sys

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from core import reliability
from core.reliability import run_reliably
from rag.retrieve import retrieve as _retrieve
from tools import finance
from tools.fallbacks import stock_fallback

mcp = FastMCP("finance-research")


# --------------------------------------------------------------------------
# 日志:只能走 stderr(见文件头"stdio 传输的铁律")
# --------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[mcp-server] {msg}", file=sys.stderr, flush=True)


def _log_reliability(event: dict) -> None:
    """把可靠性层的事件打到 stderr。

    ⚠️ **这是 MCP 化带来的观测缺口,也是这一步最该记住的教训。**

    M5.2.1 的观测层是靠 `reliability.add_observer()` 拿到"重试了几次、
    有没有走兜底"的。但 `run_reliably` 现在跑在**这个进程**里,而 ObsHandler
    注册的 observer 在 Agent 进程里 —— 进程边界把事件挡住了,于是评测报告里
    `retry` / `fallback` 两列会变成 0。

    工具调用次数和耗时不受影响(客户端侧的 StructuredTool 照样触发回调),
    丢的恰好是本项目诊断工具失败时最有用的那两个数。

    这是 M5.2.1 那条教训的第二次应验:**观测必须做在知道真相的那一层。**
    当时是 run_reliably 吞掉异常导致框架回调看不见,现在是进程边界挡住事件;
    形式不同,道理一样。

    这里先用 stderr 打出来保底(手工跑 server 时看得见)。要真正补上,得让本
    进程也接 OTel 上报到同一个 Phoenix,或者把计数放进 MCP 响应的 _meta 带回去。
    """
    kind = event.get("kind", "?")
    tool = event.get("tool", "?")
    _log(f"{kind}: {tool} {event.get('error', '')}".rstrip())


reliability.add_observer(_log_reliability)


# --------------------------------------------------------------------------
# 工具定义
#
# ⚠️ docstring 不是写给人看的注释,**它会原样进 prompt 给模型看**。
#    所以下面每个 docstring 都和 graph/tool_adapters.py 里原来手写的
#    description 逐字一致 —— 这样 M5.3 前后评测结果才可比,不会因为
#    换了说明文案而分不清"是 MCP 的影响还是 prompt 的影响"。
#    想写实现细节的注释,请写在函数体里。
# --------------------------------------------------------------------------


@mcp.tool()
def get_stock_price(
    symbol: Annotated[str, Field(description="股票代码,例如 NVDA、AAPL")],
    days: Annotated[int, Field(description="最近多少个交易日,默认 30")] = 30,
) -> str:
    """查询某只股票最近若干个交易日的涨跌幅、最新价、最高/最低价。需要行情数据时调用。"""
    return str(
        run_reliably(
            func=finance.get_stock_price,
            kwargs={"symbol": symbol, "days": days},
            timeout=20.0,
            retries=2,
            fallback=stock_fallback,
            name="get_stock_price",
        )
    )


@mcp.tool()
def search_news(
    query: Annotated[str, Field(description="搜索关键词,例如 'NVDA stock surge reason'")],
) -> str:
    """按关键词做网页搜索,返回相关新闻的标题、链接和摘要。需要了解事件/原因时调用。"""
    return str(
        run_reliably(
            func=finance.search_news,
            kwargs={"query": query},
            timeout=20.0,
            retries=2,
            name="search_news",
        )
    )


@mcp.tool()
def fetch_page(
    url: Annotated[str, Field(description="要抓取的网页地址(http/https)")],
) -> str:
    """抓取一个网页的正文文本。当搜索结果的摘要不够、需要看细节时调用。"""
    # SSRF 护栏在 finance.fetch_page 内部,跟着工具留在服务端(见文件头决策 2)。
    return str(
        run_reliably(
            func=finance.fetch_page,
            kwargs={"url": url},
            timeout=15.0,
            retries=1,
            name="fetch_page",
        )
    )


@mcp.tool()
def retrieve(
    query: Annotated[str, Field(description="检索问题,例如 '英伟达的主要业务板块'")],
    top_k: Annotated[int, Field(description="返回几段,默认 4")] = 4,
) -> str:
    """从本地知识库按语义检索背景资料(公司背景、行业知识、术语解释等)。需要背景知识时调用。"""
    # ⚠️ 这个工具目前走 MCP 会超时,原因见文件头"已知未解决"。
    return str(
        run_reliably(
            func=_retrieve,
            kwargs={"query": query, "top_k": top_k},
            timeout=20.0,
            retries=1,
            name="retrieve",
        )
    )


def main() -> None:
    # .env 得自己加载:server 是被客户端当子进程拉起的,而 MCP SDK 默认只传一小份
    # 环境变量白名单(get_default_environment(),Windows 上只有 PATH/APPDATA/TEMP 等
    # 十来个),**我们的 API key 一个都传不过来**。所以必须在这里读 .env。
    from dotenv import load_dotenv

    load_dotenv()
    _log("启动,传输=stdio,工具=get_stock_price/search_news/fetch_page/retrieve")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
