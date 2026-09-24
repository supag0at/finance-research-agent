# obs/ 模块说明:自建可观测层是怎么做的

> 设计方案:`docs/design/M5.2-可观测性.md`(为什么做、选型对比)
> 本文档:代码怎么组织、指标怎么读、坑在哪

M5.2.1 做的是**第一层:零外部依赖的自建观测**。
M5.2.2 做的是**第二层:OTel + Phoenix 瀑布图**,用环境变量开关,不装/不开都能跑。

---

## 一、一句话理解

在 `GraphAgent.run()` 里**只加了一行** `config["callbacks"] = [handler]`,
从此每次调研都会:

1. 控制台打一行摘要;
2. 往 `obs/runs/events-<日期>.jsonl` 写逐条事件(Logs);
3. 往 `obs/runs/runs-<日期>.jsonl` 写一行汇总(Metrics);
4. 把汇总挂在 `agent.last_run` 上,评测报告(M5.2.3)直接取用,不重复采集。

`nodes.py`、`tools/`、`rag/`、`memory/` **一行没改**。观测和业务完全解耦。

摘要行长这样:

```
[obs] run=42216156 19.1s | LLM 4次 (in 4694/缓存3200, out 546) ¥未配单价 |
      工具 6次 | plan=9.3s act=7.2s compress=0.0s ⚠️ 撞上限/工具失败4/重试12
```

---

## 二、文件分工

| 文件 | 职责 | 对应哪根支柱 |
| --- | --- | --- |
| `callbacks.py` | `ObsHandler`:挂在 callbacks 上采集事件 | 采集层(三根都靠它) |
| `metrics.py` | `RunRecord`:聚合成数值 + 成本估算 | **Metrics** |
| `logger.py` | 结构化落盘 + 脱敏 | **Logs** |
| `tracing.py` | OTel + Phoenix 开关式接入 | **Traces** |
| `runs/` | 产物目录(git 忽略) | — |

---

## 三、为什么用回调机制

LangGraph 执行节点、调 LLM、跑工具时都会触发 LangChain 的回调事件。
我们实现一个 `BaseCallbackHandler` 传进 `config` 就能全收到。

**好处一:业务代码零改动。** 观测是横切关注点,不该污染业务。

**好处二:避开"span 孤儿"。** 如果自己手写 OpenTelemetry span 去包节点,
会踩一个跟"图有环"直接相关的坑:OTel 把当前 span 存在 `contextvars` 里,
而调度器第二圈回到同一个节点时是新的调用栈,上下文不会自动带过去,
那一圈的 span 就脱离父节点、变成一棵新的树。走回调机制则由框架维护父子关系。

---

## 四、四个实测踩出来的坑(都已处理)

这几个都不是想出来的,是先写探针跑真实数据、看载荷才发现的。

### 4.1 `plan` 会被重复计数

条件边函数 `should_continue` 也会触发 `on_chain_start`,而且 `metadata` 里的
`langgraph_node` **同样标着 `'plan'`**(因为它挂在 plan 之后)。
只按 `langgraph_node` 计数,`plan` 的次数会翻倍。

区分靠 tag:

| 谁 | tags |
| --- | --- |
| 真正的节点执行 | `['graph:step:3']` |
| 条件边函数 | `['seq:step:3']` |

所以过滤条件是:

```python
if not any(str(t).startswith("graph:step:") for t in (tags or [])):
    return
```

**自检方法**:`plan` 节点的执行次数应该恰好等于 LLM 调用次数。
不等就说明这个过滤失效了。

### 4.2 工具失败**不会**触发 `on_tool_error`

这是最反直觉的一个。`core/reliability.py` 的职责就是"把失败吞掉、悄悄重试、
必要时降级",对上层来说一切正常。于是:

- `on_tool_error` **永远收不到**(异常被吞了,工具"成功"返回了错误字符串);
- `on_retry` 也收不到(重试是我们自己实现的,不走 LangChain 的重试机制)。

结论:**工具失败率和重试次数,靠回调是测不到的。**

处理办法是两条路并用:

1. **在知道真相的那一层埋点**:`core/reliability.py` 里加了一个极简的观察者列表
   (`add_observer` / `remove_observer`),失败/重试/降级时 `_emit` 一个事件。
   `core` 不反向依赖 `obs`——谁想听自己注册,默认没人听、零开销。
2. **检查返回值**:工具返回以 `[工具失败` 开头就判失败。

> 一般化的教训:**可观测性必须埋在"知道真相"的那一层。**
> 一个把错误优雅吞掉的可靠性层,天然是观测的盲区——它的设计目标就是"让上层看不见问题"。

### 4.3 工具身份在两层之间会分裂

可靠性层原来用 `func.__name__` 标识工具,回调层用注册的工具名。
真实工具里两者恰好相同,所以平时看不出来;测试时把 `search_news` 打桩成一个
叫 `broken` 的函数,数据立刻分裂:

```
broken       count=0 fail=0 retry=12    ← 幽灵条目
search_news  count=4 fail=4 retry=0     ← 重试丢了
```

修法:`run_reliably(..., name=name)` 由调用方显式传注册名。
`tools/registry.py` 和 `graph/tool_adapters.py` 都已传。

### 4.4 请求的模型名 ≠ 返回的模型名

```
metadata.ls_model_name            = 'deepseek-v4-flash'   ← 我们请求的
response_metadata.model_name      = 'deepseek-flash'      ← 实际返回的
```

两个都记(`model_requested` / `model_returned`)。这也是**成本不能靠模型名查价格表**的
原因之一——名字都对不上。

---

## 五、成本怎么算(以及为什么不敢猜)

token 用量里有分项:

```python
usage_metadata = {
  'input_tokens': 691,  'input_token_details':  {'cache_read': 512},
  'output_tokens': 125, 'output_token_details': {'reasoning': 67},
}
```

**691 个输入 token 里有 512 个是缓存命中**,缓存命中的单价远低于未命中。
不区分就会显著高估。所以成本公式是三段:

```
成本 = (输入 - 缓存命中) × 未命中单价
     + 缓存命中        × 缓存单价
     + 输出            × 输出单价
```

单价从 `.env` 读,**代码里不硬编码**:

```
LLM_PRICE_IN=        # 元/百万 token,输入未命中缓存
LLM_PRICE_CACHE_IN=  # 元/百万 token,输入命中缓存
LLM_PRICE_OUT=       # 元/百万 token,输出
```

没配置时 `cost()` 返回 **`None` 而不是 0**,摘要里显示"未配单价"。
这是有意的:**区分"免费"和"不知道"**。一个凭空捏造的成本数字比没有成本更糟,
因为它会被当真。单价请从模型厂商官网抄。

---

## 六、指标怎么读(几个容易误读的地方)

### `nodes['act'].total_ms` 才是工具的真实墙钟耗时

`RunRecord.tool_ms`(各工具耗时之和)会**大于**真实耗时,因为工具是并行的。
实测:

```
act 墙钟 = 2837ms      各工具耗时之和 = 4789ms
```

这个差值本身就是**并行生效的证据**。要看真实耗时用 `act`,
要看"如果串行会花多久"用 `tool_ms`。

### `count > 1` 是正常的

图里有环,`plan` / `compress` / `act` 一次调研会跑多次。
`count` 恰好反映了"绕了几圈",是收敛性指标——**突然变高就是开始跑偏了**。

### 单次数据不要过度解读

同一个问题两次跑的耗时能差几倍(LLM 有随机性)。
看趋势要看 `runs-*.jsonl` 的分布,不要拿一次采样下结论。
这条教训来自 `docs/notes/2026-09-09-评测方法论调研.md`。

---

## 七、开关

| 环境变量 | 作用 | 默认 |
| --- | --- | --- |
| `OBS` | 总开关,`0` 关闭落盘 | 开 |
| `OBS_CONTENT` | 是否记录内容原文 | **关**(只记长度) |
| `OBS_TRACING` | 是否把链路发给 Phoenix | **关** |
| `OBS_TRACING_ENDPOINT` | OTLP HTTP 入口 | `http://127.0.0.1:6006/v1/traces` |
| `OBS_TRACE_CONTENT` | trace 里要不要带 prompt 原文 | **开**(和 JSONL 相反,见下) |
| `LLM_PRICE_IN` / `LLM_PRICE_CACHE_IN` / `LLM_PRICE_OUT` | 单价 | 空(不算成本) |

`print_obs=False` 可以构造一个不打印摘要的 agent:`GraphAgent(print_obs=False)`。

**关于脱敏**:默认只记长度而不是原文,是因为实测发现自动埋点默认会把 prompt 全文、
工具返回全文都记下来(见设计文档发现3)。生产环境把用户输入原文写进日志是事故,
不是功能。要看原文时临时开 `OBS_CONTENT=1`。

---

## 八、意外收获:观测上线当天就抓到两个真问题

M5.2.1 刚跑通就做了个压力实验(故意让 `search_news` 每次都抛异常),
立刻暴露两件此前完全不知道的事:

1. **`compress` 会变成瓶颈。** 正常情况它只花 0.5ms;工具持续失败导致历史暴涨后,
   它触发 LLM 摘要,一次运行里累计花了 **153 秒**——比所有 LLM 决策加起来还多。
2. **`core/llm.py` 有个崩溃 bug。** `complete()` 收到 `None` 响应时直接
   `response.choices`,抛出 `AttributeError: 'NoneType' object has no attribute 'choices'`,
   报错完全看不出发生了什么。**手写版也有这个 bug**,只是从没被触发过。
   已修:空响应给出可诊断的报错;`memory/short_term.py` 里摘要失败降级为"省略历史"
   而不是让整轮调研崩掉。

> 这就是可观测性的价值最直白的体现:**它不是让你更懂已知的问题,
> 而是让你看见此前不知道自己不知道的问题。**

---

## 九、局限 / 下一步

- **观察者是模块级的,不是按上下文隔离的。** 我们的运行是串行的(一次只跑一个
  `agent.run`),够用;真要并发跑多个 Agent,`core/reliability.py` 那个观察者列表
  得改成 `contextvars`。
- **手写版 `core/agent.py` 没有观测。** 它不走 LangChain 回调,保持对照实现的纯粹。
  想给它加就得手写埋点——这本身就是"框架带来了什么"的一个具体答案。
- **P95 之类的分位数还没算。** 目前只有 count / total / max,分位数要跨运行聚合,
  等 `runs-*.jsonl` 攒够了再做。
- **单价不分峰谷。** DeepSeek 空闲时段是高峰的半价,我们只配一套单价(填的高峰价),
  算出来是**成本上限**。要精确得按调用时间戳选价。

---

## 十、M5.2.2:怎么看瀑布图

### 怎么开

```bash
# 1. 起 Phoenix(一次即可,和 Qdrant/Postgres 一样走 Docker)
docker run -d --name fra-phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

# 2. .env 里设
OBS_TRACING=1

# 3. 跑一轮
python main_graph.py
# 或任何走 GraphAgent 的入口(eval --graph 也行)

# 4. 浏览器打开
http://127.0.0.1:6006
```

`GraphAgent.run()` 会自动调 `setup_tracing()`(幂等)。没开开关或没装依赖时静默跳过。

### 实测一条短问题长什么样(2026-09-23)

问"什么是市盈率?用两句话解释。",6.3 秒,Phoenix 里是一棵 13 个 span 的树:

```
LangGraph                          3.55s
├─ recall
├─ compress
├─ plan
│  ├─ ChatOpenAI                   (prompt 1579 / 缓存 768 / out 102)
│  └─ should_continue              ← 条件边,也会出 span,这是正常的
├─ act
│  └─ retrieve
├─ compress
├─ plan
│  ├─ ChatOpenAI
│  └─ should_continue
└─ remember
```

父子关系完整,没有"span 孤儿"(第二圈的 `plan` 仍然挂在同一棵 `LangGraph` 下)。
这就是走 `LangChainInstrumentor`(回调)而不是手写 OTel span 的收益。

### 读瀑布图时三个容易误会的点

**1. `should_continue` 会出现在树上。**
我们自己的 `ObsHandler` 用 `graph:step:` 这个 tag 把它滤掉了,所以摘要行里 `plan` 次数 = LLM 次数。
Phoenix 不过滤——条件边在框架内部也是一个 Runnable,它就是有自己的 span。
**别拿 Phoenix 的 `plan` 次数去对摘要行**,要对的是 `ChatOpenAI` 次数。

**2. Phoenix 里成本是 `$0`。**
它按模型名查自己的价格表,DeepSeek 不在表里。成本以我们自己的 `RunRecord.cost()` 为准
(单价从 `.env` 读、区分缓存命中)。两边数字对不上不是 bug。

**3. JSONL 默认不记原文,Phoenix 默认记。**
这是有意的。JSONL 落在仓库目录里会长期堆积;Phoenix 跑在本机容器里、重启即失,
而且"点开某一步看当时的 prompt"恰恰是瀑布图存在的意义。要收紧设 `OBS_TRACE_CONTENT=0`。

### 接线时踩过的坑(第三次代理问题)

OTLP HTTP exporter 内部用的是 `requests`,而 `requests` 默认读系统代理。
Clash 开着全局代理时,连 `127.0.0.1:6006` 都会被劫走,span 发出去了 Phoenix 却收不到。
这和 Qdrant / DeepSeek 是同一个问题的第三次出现。
`obs/tracing.py` 里已经用 `session.trust_env = False` 绕过。

另一个容易踩的:BatchSpanProcessor **是异步攒批的**。进程退出太快,末尾几个 span 会丢,
瀑布图缺最后几步,看起来像埋点没生效。`GraphAgent.run()` 结束时会 `flush_tracing()`,
`main_graph.py` 退出前还会 `shutdown_tracing()`。
