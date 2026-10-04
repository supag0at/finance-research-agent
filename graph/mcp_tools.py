"""MCP 客户端 —— 把远程 MCP server 的工具接回 LangGraph(M5.3)。

## 它和 graph/tool_adapters.py 的关系

两者产出**同一种东西**:一批 `StructuredTool`,`ToolNode` 直接能用。

    tool_adapters.build_tools()  →  进程内函数,手写 pydantic schema
    mcp_tools.build_mcp_tools()  →  跨进程 RPC,schema 由 server 自动给出

因为产出类型一致,`graph/nodes.py`、`ToolNode`、`ObsHandler` 一行都不用改。
这和当初从手写版迁到 LangGraph 时是同一个套路:**接口不变,实现可换**。

## ⚠️ async 是这一步唯一真正的硬摩擦

MCP 客户端 SDK 是 async 的,而我们整条图执行原本是同步的(`graph.invoke`)。

我一开始以为可以"只在建图时 `asyncio.run()` 把工具列表取回来,之后保持同步"。
**这是错的**,第一次跑就撞墙:

    NotImplementedError: StructuredTool does not support sync invocation.

adapter 产出的工具**只实现了 `_arun`**,连 schema 带执行全是异步的。也就是说
async 不止污染"取工具"这一步,而是污染**每一次工具执行**,进而污染整条图的执行方式。

所以只剩两条路:

  1. `GraphAgent.run()` 里改用 `asyncio.run(graph.ainvoke(...))` ← 选这条
  2. 给每个异步工具再包一层同步壳,壳里 `asyncio.run()` —— 每次工具调用新建一个
     事件循环,而且会**毁掉 ToolNode 的并行**(M5.2.1 专门量化证明过那个并行是
     真实存在的收益),不可接受

选 1 的代价很小:LangGraph 在异步执行下照样能跑同步节点,ObsHandler 的同步回调
也会被框架自动放到执行器里跑。实测 token / 耗时 / 节点拆解全部照常采到。

**这就是"协议标准化"没能消除的那类成本**:MCP 统一了工具的描述和调用格式,
但它统一不了宿主程序的并发模型。跨进程通信天然是异步的,这个性质会顺着调用链
一路往上渗透到你的入口函数。这种"底层技术选型沿调用链向上传染"的现象值得记住,
面试里比背协议三原语有意思得多。
"""

import asyncio
import os
import sys


def mcp_enabled() -> bool:
    """MCP_TOOLS=1 才走 MCP。默认关 —— 默认路径保持 M5.2 的行为不变。"""
    return os.getenv("MCP_TOOLS", "").strip() not in ("", "0", "false", "False")


def build_mcp_tools() -> list:
    """连上本地 MCP server,把它的工具取回来包成 StructuredTool。

    失败时**不抛异常**,返回空列表并在 stderr 上说明,由调用方决定退回进程内工具。
    理由和观测层一致:接线层出问题不该让主流程崩。
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        print(f"[mcp] MCP_TOOLS 已开,但依赖缺失,退回进程内工具:{exc}", file=sys.stderr)
        return []

    client = MultiServerMCPClient(
        {
            "finance": {
                # 用 sys.executable 而不是写死 "python":保证子进程用的是
                # 当前这个虚拟环境的解释器,否则 server 会因为找不到 httpx/qdrant 而起不来。
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                # 子进程的工作目录要是项目根,否则 `-m mcp_server.server` 找不到包。
                "cwd": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            }
        }
    )

    try:
        tools = asyncio.run(client.get_tools())
    except Exception as exc:
        print(f"[mcp] 连接 MCP server 失败,退回进程内工具:{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return []

    print(f"[mcp] 已从 MCP server 载入 {len(tools)} 个工具:"
          f"{', '.join(t.name for t in tools)}", file=sys.stderr)
    return tools
