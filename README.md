# finance-research-agent

一个从零手写的财经调研 Agent(学习项目)。目标是不依赖框架,亲手把
"Agent = LLM 决策 + 工具执行 + 循环"这件事做出来,把每个核心概念都落到代码里。

## 里程碑规划

- **M1(当前):手写最小 Agent** —— tool registry + 主循环 + Function Calling + Trajectory。
- M2:接真实工具(行情/搜索/抓网页)+ 可靠性(重试/超时/降级)+ Sandbox。
- M3:RAG(向量库检索历史研报)+ 记忆。
- M4:Eval 评测体系(golden 数据集 + LLM-as-Judge + 回归)。
- M5:用 LangGraph 重构 + 可观测(日志/指标/trace)。

## M1 概念对照表(名词都落在哪)

| 概念 | 位置 |
| --- | --- |
| Tool Schema(工具说明书) | `tools/registry.py` 的 `get_schemas()` |
| Function Calling(工具调用) | LLM 的返回,在 `core/agent.py` 里被解析 |
| Planning(规划) | `core/agent.py` 主循环里每次调用 LLM 的那一刻 |
| Trajectory(轨迹) | `core/schema.py` 定义,`core/agent.py` 里逐圈填充 |
| 模型可替换 | `core/llm.py`(改环境变量即可切换) |
| 防死循环(可靠性雏形) | `core/agent.py` 的 `max_steps` |

## 运行方式

```bash
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env
# 编辑 .env,填入 DeepSeek / Kimi / OpenAI 三者之一的 API key
python main.py
```

M1 阶段的行情、新闻均为写死的假数据(见 `tools/finance.py`),目的是先跑通
流程;M2 会替换为真实数据源,而工具的对外接口保持不变。
