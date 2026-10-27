"""上下文压缩 v2(M8):增量归档 + 钉住区 + 可回读。

和 legacy(memory/short_term.py)的根本区别:

  legacy  每圈从完整历史现算一份摘要,摘要不存;越过阈值后每圈都要调一次摘要器,
          且摘要文本每圈都变,system 之后的前缀缓存全部失效。
  v2      压缩状态存在 AgentState.ctx 里(随 checkpointer 落库),只为**新溢出的圈**
          调一次 LLM;视图由纯函数从 (messages, turns, ctx) 渲染,两次压缩之间前缀不变。

三条不变式(测试以它们为准,见 eval/compress_offline.py):
  1. messages 永不修改 —— stub、折叠、归档、钉住区只存在于视图;
  2. build_view 是纯函数 —— 同一输入永远渲染出逐字相同的视图;
  3. 视图里每条 ToolMessage 之前都有携带其 tool_call_id 的 AIMessage,压缩永不拆圈。

本包不依赖 graph/ 和 obs/:观测走 memory/events.emit,节点接线在 graph/nodes.py。
"""

from memory.context.compactor import compress_step
from memory.context.config import ContextConfig
from memory.context.types import empty_ctx

__all__ = ["ContextConfig", "compress_step", "empty_ctx"]
