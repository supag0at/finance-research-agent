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
- **M3(进行中):RAG** —— 三层架构:bge-m3 embedding + Qdrant 向量库 + Postgres 关系库。
- **M3.5(已完成):记忆** —— 短期(上下文压缩)+ 长期(语义记忆,复用 RAG 基建)。
- M4:Eval 评测体系(golden 数据集 + LLM-as-Judge + 回归)。
- M5:用 LangGraph 重构 + 可观测(日志/指标/trace);把工具改造成 MCP server。

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
- 高阶技巧留白见 `memory/MEMORY.md`。

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
