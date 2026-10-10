# finance-research-agent

一个从零手写的财经调研 Agent(学习项目)。目标是不依赖框架,亲手把
"Agent = LLM 决策 + 工具执行 + 循环"这件事做出来,把每个核心概念都落到代码里。

## 工程复盘 & 高级特性手册

见 [`docs/`](docs/README.md):
- `docs/troubleshooting/` —— 真实踩过的坑的复盘(现象→排查→根因→解决→面试话术)。
- `docs/playbook/` —— "问题→高级特性"对照手册(RAG/记忆/评估的进阶方向)。

## 里程碑规划

- **M1(已完成):手写最小 Agent** —— tool registry + 主循环 + Function Calling + Trajectory。
- **M2(已完成):真实工具 + 可靠性 + Sandbox** —— 行情/搜索/抓网页;重试/超时/降级;SSRF 护栏。
- **M3(已完成):RAG** —— 三层架构:embedding + Qdrant 向量库 + Postgres 关系库。
- **M3.5(已完成):记忆** —— 短期(上下文压缩)+ 长期(语义记忆,复用 RAG 基建)。
- **M4(已完成):Eval 评测体系** —— 测试集 + 程序化规则检查 + LLM-as-Judge + 报告/回归,见 `eval/`。
- **M5.1(已完成):LangGraph 重构** —— 手写主循环换成显式的图,白拿续话/并行/可视化,见 `graph/`。
- **M5.2.1(已完成):自建观测层** —— 每次运行的耗时/token/成本/节点/工具落 JSONL,见 `obs/`。
- **M5.2.2(已完成):OTel + Phoenix** —— 开关式接线,本地看瀑布图。
- **M5.2.3(已完成):质量 + 代价双维评测** —— 报告里同时给出通过率和 token/成本/耗时构成。
- **M5.3(已完成):工具 MCP 化** —— `mcp_server/` 用 FastMCP 把四个工具暴露成标准 MCP 服务,
  `MCP_TOOLS=1` 开关切换。工具定义从两份收敛成一份(函数签名即 schema、docstring 即说明)。
  留了两个已知问题:`retrieve` 走 MCP 会超时、每次调用新拉子进程导致 `act` 耗时涨约 10 倍,
  详见 [`docs/design/M5.3-MCP.md`](docs/design/M5.3-MCP.md)。
- **M6(已完成):可靠性强化** —— 六项:预算四边界 + `finalize` 收口、三段式结构化错误、
  失败四分类 + 熔断、`PostgresSaver` 持久化执行、外部内容不可信隔离、ingest 幂等顺序修正。
  验收靠**故障注入**(`python -m eval.faults`,29 项、离线零成本)而不是看通过率 ——
  因为可靠性改善的是最坏情况,而真实评测的网络噪声比效果大一个量级。
  期间踩的一个坑很值得看:**边界设得比正常运行还低,反而更贵**(多花 35%)。
  详见 [`docs/design/M6-可靠性强化.md`](docs/design/M6-可靠性强化.md)。
- **M7(已完成):记忆强化** —— 记忆分型(情景/语义/程序)、写入筛选 + 结构化抽取、
  按 key 精确取代与向量模糊冲突判定、`user_id` 接口级隔离、阈值召回 + 三因子重排、
  `search_memory` 工具、程序记忆(从确定性失败归纳)。用 off / legacy / v2 三档消融评测。
  旧方案在小库下近乎满分,**真正的差距在隔离(0/3→3/3)、链式更新、版本混杂的脏库**;
  评测还抓出一个设计缺陷(清仓被当成遗忘)并修复。
  详见 [`docs/design/M7-记忆强化.md`](docs/design/M7-记忆强化.md)。

## RAG 三层架构(M3)

```
① Embedding(rag/embedding.py)  文字 → 向量   用 SiliconFlow bge-m3(维度 1024)
② 向量库 Qdrant(rag/store.py)  存向量+相似检索  本地 docker,集合 finance_corpus
③ 关系库 Postgres(rag/db.py)   存原文+去重     本地 docker newsrag 库,表 fra_*
```

离线建库:`python -m rag.ingest`(读 data/corpus/*.md → 切块 → 向量化 → 入库)
在线检索:`rag/retrieve.py` 的 `retrieve()`,已注册为 Agent 工具。

依赖本地 docker 的 Qdrant(6333)与 Postgres(5432);Clash 全局代理需放行
127.0.0.1(代码已用 trust_env=False 绕过)。

## 记忆(M3.5)

- 短期记忆 `memory/short_term.py`:上下文压缩(工具结果裁剪 + 滑动窗口 + LLM 摘要),
  在 `core/agent.py` 每次调 LLM 前触发,超过阈值才压缩,且不破坏 assistant→tool 配对。
- 长期记忆 `memory/long_term.py`:`remember()` / `recall()`,复用 embedding + Qdrant
  (集合 `fra_memory`)。未配 `EMBED_API_KEY` 时自动降级(不报错)。
- ⚠️ 以上长期记忆是 **legacy**(手写版 `core/agent.py` 仍在用)。LangGraph 版可用
  `GraphAgent(memory_mode="v2")` 切到 M7 新记忆(`memory/{schema,store,extract,write,recall,tools,procedural}.py`),
  `off / legacy / v2` 三档用于消融评测:`python -m eval.memory_runner --mode v2`。
- 高阶技巧与状态见 `memory/MEMORY.md`。

## 评测(M4)

`eval/`:给 Agent 建自动化考卷,量化"改动前后变好还是变坏"。
- `eval/cases.jsonl`:测试集(问题 + 必调工具 + 期望要点,而非标准答案)。
- `eval/checks.py`:程序化规则检查(工具是否调用、数字是否可溯源、步数),确定不花钱。
- `eval/judge.py`:LLM-as-Judge,按 rubric 给 accuracy/completeness/faithfulness 打分。
- `eval/runner.py`:批量跑 → 检查 + 裁判 → 出报告到 `eval/reports/`。

运行:`python -m eval.runner`(或 `python -m eval.runner <用例id>` 只跑指定题;
加 `--graph` 则评测 M5 的图版实现)。

M5.2.3 起,图版的报告里除了质量,还会给出**代价**:每题的 LLM 次数、token(含缓存命中)、
成本、工具失败/重试、最慢节点,以及全局的耗时构成(`plan 73.6% | act 26.4% | ...`)。
数据直接来自观测层,不额外花钱。成本需先在 `.env` 配三档单价,否则显示"未配单价"。

- 实现讲解 / 怎么加题 / 怎么读报告:`eval/EVAL.md`
- 设计理念与首跑发现:`docs/playbook/eval.md`

## LangGraph 重构(M5.1)

`graph/`:把手写的 Agent 主循环换成 LangGraph 的显式图。**业务逻辑一行没改**,
换掉的只是骨架;`core/agent.py` 手写版**刻意保留**作为对照实现。

```
START → recall → compress → plan ─┬─(有工具调用)→ act ─┐
                     ↑            │                    │
                     └────────────┴────────────────────┘
                                  └─(无工具调用)→ remember → END
```

- `graph/state.py` 状态定义 / `graph/nodes.py` 四个节点 / `graph/build.py` 组图
- `graph/agent.py` 的 `GraphAgent` 与手写版**接口完全一致**,所以整套评测不用改就能跑图版

运行:`python main_graph.py`(会先打印图结构,跑一轮,再演示同 `thread_id` 续话)。

比手写版多出来的能力:续接对话(checkpointer)、工具并行执行(ToolNode)、
控制流可视化、每步状态可回放。

- 模块讲解与设计取舍:`graph/GRAPH.md`
- 为什么要做这次重构:`docs/design/M5-LangGraph重构.md`

## 可观测(M5.2.1)

`obs/`:挂在 LangGraph 的 `config["callbacks"]` 上自动采集,**业务代码零改动**。
每次调研会打一行摘要,并把逐条事件(Logs)和运行汇总(Metrics)写进 `obs/runs/`。

```
[obs] run=42216156 19.1s | LLM 4次 (in 4694/缓存3200, out 546) ¥未配单价 |
      工具 6次 | plan=9.3s act=7.2s compress=0.0s ⚠️ 撞上限/工具失败4/重试12
```

- `obs/callbacks.py` 采集 / `obs/metrics.py` 聚合+成本 / `obs/logger.py` 落盘+脱敏
- 成本按**缓存命中分档**计算(实测一次请求 691 个输入 token 里 512 个是缓存命中);
  单价从 `.env` 读,没配就显示"未配单价"而不是假装是 0
- 内容原文**默认不记**,要记得显式开 `OBS_CONTENT=1`
- 瀑布图(可选):`.env` 设 `OBS_TRACING=1`,先起
  `docker run -d --name fra-phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest`,
  然后打开 http://127.0.0.1:6006

- 模块讲解、指标怎么读、坑在哪:`obs/OBS.md`
- 设计与选型:`docs/design/M5.2-可观测性.md`
- 踩坑复盘:`docs/troubleshooting/03-可观测性.md`

## M1 概念对照表(名词都落在哪)

| 概念 | 位置 |
| --- | --- |
| Tool Schema(工具说明书) | `tools/registry.py` 的 `get_schemas()` |
| Function Calling(工具调用) | LLM 的返回,在 `core/agent.py` 里被解析 |
| Planning(规划) | `core/agent.py` 主循环里每次调用 LLM 的那一刻 |
| Trajectory(轨迹) | `core/schema.py` 定义,`core/agent.py` 里逐圈填充 |
| 模型可替换 | `core/llm.py`(改环境变量即可切换) |
| 防死循环(可靠性雏形) | `core/agent.py` 的 `max_steps` |
| 重试 / 超时 / 降级(M2) | `core/reliability.py`,注册工具时按需配置 |
| Sandbox / SSRF 护栏(M2) | `core/sandbox.py`,`fetch_page` 抓取前先校验 |
| 真实工具(M2) | `tools/finance.py`(yfinance / DuckDuckGo / httpx+bs4) |

## 运行方式

```bash
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# 编辑 .env,填入 DeepSeek / Kimi / OpenAI 三者之一的 API key
python main.py
```

M1 阶段的行情、新闻均为写死的假数据(见 `tools/finance.py`),目的是先跑通
流程;M2 会替换为真实数据源,而工具的对外接口保持不变。
