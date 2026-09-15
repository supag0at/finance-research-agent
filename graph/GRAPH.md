# graph/ 模块说明:LangGraph 版 Agent 是怎么搭起来的

> 配套阅读:`docs/design/M5-LangGraph重构.md`(为什么要做这次重构)
> 对照实现:`core/agent.py`(手写循环版,**刻意保留**)

这个目录把原来手写的 Agent 主循环换成了 LangGraph 的"图"。
业务逻辑(工具、记忆、RAG、可靠性)**一行都没改**,换掉的只是骨架。

---

## 一、一句话理解这次重构

手写版里,Agent 的控制流是**藏在 Python 代码里的**:

```python
for step in range(self.max_steps):        # 循环
    message = self.llm.chat(messages, schemas)
    if not message.tool_calls:            # 分支
        return message.content
    for tc in message.tool_calls:         # 串行执行工具
        obs = self.tools.call(tc)
```

图版里,同样的控制流被**显式声明成一张图**:

```
        +-----------+
        | __start__ |
        +-----------+
              *
         +--------+
         | recall |          ← 长期记忆:捞回相关记忆 + 落用户问题
         +--------+
              *
        +----------+
        | compress |  ←──┐   ← 短期记忆:把历史压成"给模型看的视图"
        +----------+     │
          *       **     │
  +------+         +-----+
  | plan |         | act |   ← ToolNode:并行执行所有 tool_calls
  +------+         +-----+
      .                │
      .                └────┘  这条返回边 == 原来的 for 循环
+----------+
| remember |                  ← 长期记忆:把本轮结论写回记忆库
+----------+
      *
+---------+
| __end__ |
+---------+
```

**换来了什么?** 控制流从"读代码才能看懂"变成"看图就懂",而且状态每步自动存档
(checkpointer),于是白拿了续话、回放、并行执行这些能力。

---

## 二、文件分工

| 文件 | 职责 | 对应手写版的哪一块 |
| --- | --- | --- |
| `state.py` | 定义图在跑的过程中传递什么状态 | 那个手动传来传去的 `messages` 列表 |
| `model.py` | 接 DeepSeek(LangChain 的 `ChatOpenAI`) | `core/llm.py` |
| `tool_adapters.py` | 把现有工具包成 LangChain 工具 | `tools/registry.py` |
| `nodes.py` | recall / compress / plan / remember 四个节点 | 主循环里的各个片段 |
| `build.py` | 把节点和边装配成图 | `Agent.run()` 的 `for` 循环本身 |
| `agent.py` | `GraphAgent` 门面,对外接口与手写版一致 | `Agent` 类 |

---

## 三、几个值得讲清楚的设计决定

### 3.1 为什么 state 里有两个消息字段?

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 完整历史,只增不改
    llm_input: list[AnyMessage]                          # 压缩后的视图,每次覆盖
```

手写版里,压缩是**就地改掉** messages 再发给模型的——历史和"发给模型的内容"是同一份。
上了 checkpointer 之后这样做就亏了:每步存档存的是被压过的历史,回放时看到的不是真实发生过的事。

所以这次把两者拆开:`messages` 保持 append-only(存档才有价值),
压缩结果单独写到 `llm_input`(`plan` 节点只读它)。

**代价**:多一个字段要维护。**收益**:历史可追溯,而模型仍然只看精简版。

> `messages` 用了 `add_messages` reducer 所以是追加;`llm_input` 没有 reducer,
> 默认就是后写覆盖前写。reducer 决定"新值怎么并入旧值",这是 LangGraph 里最容易忽略的一个概念。

### 3.2 为什么保留自研的 run_reliably,不用 LangGraph 的 RetryPolicy?

LangGraph 支持给节点配 `RetryPolicy`,但粒度是**整个节点**。
我们的 `act` 节点一次可能并行跑 3 个工具,其中一个超时就整节点重试,
另外两个已经成功的也会被重跑——既浪费钱又可能重复副作用。

而 `core/reliability.py` 的粒度是**单个工具**,还能按工具配不同的超时/重试/降级
(比如 `get_stock_price` 失败时降级返回"数据不可用,请勿编造数字")。

所以 `tool_adapters.py` 里的做法是:先用 `run_reliably` 包住业务函数,再包成 LangChain 工具。
**框架给的默认能力不一定比自己的好,该保留就保留。**

### 3.3 系统提示为什么在 recall 节点里注入?

```python
if not state.get("messages"):
    new_messages.append(SystemMessage(content=SYSTEM_PROMPT))
```

因为有了 checkpointer,同一个 `thread_id` 第二次提问时状态是**从存档恢复**的,
历史里已经有系统提示了。用"历史是否为空"判断,可以让这个节点对新会话和续话都正确
(幂等)。如果在入口处无脑塞一条系统提示,续话时就会越堆越多。

### 3.4 GraphAgent 为什么要延迟建图?

因为 `eval/runner.py` 是**先创建 agent、再改开关**:

```python
agent = GraphAgent()
agent.use_long_term = False   # 评测不写记忆
```

如果构造函数里就把图编译好,这个开关就晚了。所以 `graph` 做成 property,
第一次 `run()` 时才真正建图。

### 3.5 为什么要把 LangChain 消息转回我们自己的 Trajectory?

`eval/checks.py` 是按 `TrajectoryStep` 写的。`agent.py` 里的 `_to_trajectory()`
把 LangChain 的消息序列还原成我们的结构,于是**整套评测一行都不用改**就能跑图版。

这层转换看着像多余的胶水,实际上是这次重构能被验证的关键:
接口不变,实现可换,才能用同一把尺子量重构前后。

---

## 四、怎么跑

```bash
# 手写版(对照)
python main.py

# 图版:会先打印图结构,再跑一轮,最后演示续话
python main_graph.py

# 评测:同一套用例,两个实现
python -m eval.runner            # 手写版
python -m eval.runner --graph    # 图版
```

---

## 五、图版比手写版**多**出来的能力

| 能力 | 怎么来的 | 手写版为什么没有 |
| --- | --- | --- |
| 续接对话 | checkpointer 自动存档,传同一个 `thread_id` 即可 | 状态在函数栈里,跑完就没了 |
| 工具并行执行 | `ToolNode` 内置 | 手写的是 `for tc in tool_calls` 串行 |
| 控制流可视化 | `graph.get_graph().draw_ascii()` | 循环画不出来 |
| 每步状态可回放 | `graph.get_state_history(config)` | 没有存档 |
| 流式输出 | `graph.stream(...)` | 要自己攒 |

---

## 六、已知局限 / 下一步

- **checkpointer 目前是内存版**(`InMemorySaver`),进程退出即失。
  项目里已经跑着 Postgres,后续换 `PostgresSaver` 即可跨进程续跑。
- **长期记忆还没用 LangGraph 的 Store 抽象**,仍走我们自己的 `memory/long_term.py`
  (Qdrant)。等做记忆增强时再评估要不要迁。
- **`recursion_limit` 是按 superstep 计数的**,和手写版的 `max_steps`(按 LLM 调用次数)
  不是一个单位。目前取 30 ≈ 8 轮,是估算值,不是精确等价。
- 可观测性(trace/指标)留给 M5.2。
