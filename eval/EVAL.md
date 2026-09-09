# eval 模块说明:这套评测到底怎么运转

> 这份文档讲**实现与用法**:数据怎么流、每个文件干什么、怎么加题、怎么读报告、怎么扩展。
>
> 想看**为什么这么设计**、以及**首次评测发现了什么问题**,见
> [`docs/playbook/eval.md`](../docs/playbook/eval.md);
> 评测过程中踩的坑的复盘,见 [`docs/troubleshooting/02-评测校准.md`](../docs/troubleshooting/02-评测校准.md)。

---

## 一、一道题的完整生命周期(数据流)

这是理解整个模块最重要的一张图。**每道题都会走一遍下面的流程**:

```
cases.jsonl 里的一条用例
   │  {"id": "...", "question": "...", "must_call_tools": [...], "expect_points": [...]}
   ▼
agent.run(question)                          ← runner.evaluate_one()
   │  产出两样东西:
   │    answer      最终回答(字符串)
   │    trajectory  轨迹(list[TrajectoryStep],记录每一步调了什么工具、返回什么)
   ▼
┌────────────────────────┬─────────────────────────────┐
│ ① 规则检查(不花钱)    │ ② LLM 裁判(花钱)          │
│ checks.run_checks()    │ judge.judge()               │
│                        │                             │
│ 输入: case + answer     │ 输入: question + answer     │
│      + trajectory      │      + expect_points        │
│                        │      + facts(精炼版证据)   │
│ 输出: tool_ok          │ 输出: accuracy   1-5        │
│      grounded_ratio    │      completeness 1-5       │
│      ungrounded_numbers│      faithfulness 1-5       │
│      tool_call_count   │      reason(打分理由)      │
│      max_step          │                             │
└────────────────────────┴─────────────────────────────┘
   │
   ▼
is_pass(case, checks, judged)   ← 综合两边结果,判定 PASS / FAIL
   │
   ▼
汇总所有题 → summarize() → render_markdown()
   │
   ▼
eval/reports/<时间戳>.json  (逐题明细,机器可读,用于回归对比)
eval/reports/<时间戳>.md    (汇总报告,人类可读)
```

**关键设计:双轨评估**。左边是纯代码的确定性检查,右边是 LLM 主观打分。
两者互相校验——**当它们结论矛盾时,往往说明评测器自己出了问题**(这正是首次评测踩的坑)。

---

## 二、四个文件分别干什么

### 1. `cases.jsonl` —— 考卷(每行一道题)

用 JSONL(每行一个 JSON)是为了方便增删单题、也方便 diff。

```json
{"id": "nvda_perf_01", "question": "帮我分析英伟达(NVDA)最近的股价表现,以及可能的原因。", "must_call_tools": ["get_stock_price"], "expect_points": ["最近一段时间的涨跌幅或价格水平", "至少一条可溯源的原因"], "forbid": ["编造工具未返回的具体数字"]}
```

字段含义:

| 字段 | 必填 | 作用 | 被谁消费 |
| --- | :-: | --- | --- |
| `id` | ✅ | 用例唯一标识,也用于命令行只跑指定题 | `runner` |
| `question` | ✅ | 丢给 Agent 的问题 | `runner` |
| `must_call_tools` | ✅ | **必须**调用的工具名;没调就 `tool_ok=false` | `checks` |
| `expect_points` | ✅ | 期望覆盖的要点,交给裁判判"完整性" | `judge` |
| `forbid` | ❌ | 禁止项。**目前是预留字段,尚未参与判定**(见文末"待扩展") | 暂无 |
| `check_number_grounding` | ❌ | 默认 `true`;设 `false` 则**跳过数字溯源硬门槛**(概念解释题用) | `runner.is_pass` |

> ⚠️ **注意:存的不是"标准答案",而是"评分要点"**。因为股价天天变、措辞千变万化,
> 对标准答案是行不通的。

### 2. `checks.py` —— 程序化规则检查(确定性,不调 LLM)

能用代码确定回答的问题,就绝不花钱请裁判。核心函数:

| 函数 | 输入 → 输出 | 说明 |
| --- | --- | --- |
| `run_checks(case, answer, trajectory)` | → 指标字典 | 对外总入口,跑下面全部检查 |
| `collect_observations(trajectory)` | → 完整证据字符串 | **给数字溯源用**,原样不删减 |
| `build_judge_facts(trajectory)` | → 精炼证据字符串 | **给裁判用**,去重+限长 |
| `called_tools(trajectory)` | → 工具名列表 | 从轨迹里提取实际调用序列 |

`run_checks` 返回的指标:

```python
{
  "tool_ok": True,               # must_call_tools 是否都调了
  "missing_tools": [],           # 缺了哪些
  "tools_used": ["get_stock_price", "search_news", ...],  # 实际调用序列(可重复)
  "grounded_ratio": 0.923,       # 回答里的数字有多少能在工具返回里找到出处
  "ungrounded_numbers": ["129"], # 找不到出处的数字(疑似编造)
  "tool_call_count": 18,         # 总调用次数(看有没有绕圈子)
  "max_step": 8,                 # 最大步数(撞到 max_steps 说明没收敛)
}
```

#### 两个值得细看的实现细节

**(a) 数字溯源:怎么判"编造"**

思路很朴素:把回答里的数字抠出来,逐个去工具返回里做子串匹配,找不到的就是疑似编造。

```python
# _meaningful_numbers():只保留"有意义的财经数字"
if "." in tok or val >= 100:   # 带小数点的,或 >=100 的整数
    nums.append(tok)
```

为什么加这个过滤?因为 `3 个板块`、年份 `2026`、月份 `8` 这类小整数太常见,
拿它们当"是否编造"的判据会**大量误伤**。所以只盯 `224.41`、`7.5`、`962`、`1080` 这种。

同时 `_normalize()` 会去掉千分位逗号,让 `1,080` 和 `1080` 能对上。

> 这是**启发式**,不是精确算法:可能漏判(如"80%市占率",80<100 被跳过),
> 也可能误判。所以它只作参考信号,配合裁判一起看。

**(b) 为什么给裁判的证据要单独构造(`build_judge_facts`)**

因为研究型问题可能调几十次工具、抓好几篇网页正文,证据长达数万字。直接全塞给裁判会撑爆上下文;
但截得太狠,裁判"没看到出处"就会把好答案误判成编造(**首次评测就栽在这**)。所以做三件事:

```python
key = obs[:200]           # 1. 去重:用前200字当指纹,重复的工具返回只留一份
obs[:max_per_obs]         # 2. 单条限长(4000字):避免一篇长网页挤占全部预算
total + len(obs) > max_total  # 3. 总量限长(30000字)
```

**注意二者的区别**:数字溯源用 `collect_observations()`(完整证据,越全越准);
裁判用 `build_judge_facts()`(精炼证据,受上下文限制)。

### 3. `judge.py` —— LLM-as-Judge(模型裁判)

处理规则写不出来的主观维度。三个工程技巧:

**技巧 1:强制结构化输出**,便于程序解析汇总:
```
只输出如下 JSON,不要有任何多余文字:
{"accuracy": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "reason": "<50字以内理由>"}
```

**技巧 2:把真实证据喂给裁判**,让它"有据可依"地判,而不是靠自己的先验知识瞎猜。
prompt 里明确要求:*"只依据【事实依据】判断,不要用你自己的先验知识去补充或否定。"*

**技巧 3:要求给出理由**(`reason`)。只给数字不给理由的评分是不可复查的;
有了理由,你才能判断"这个低分到底是 Agent 的问题,还是裁判的问题"。

解析要容错——模型有时会用 ```json 包裹或前后加废话:
```python
# _parse_json():剥掉代码块围栏,再取第一个 { 到最后一个 } 之间的内容
start, end = text.find("{"), text.rfind("}")
```

裁判自己也可能抽风,所以整个调用包在 `try/except` 里,失败时返回带 `error` 字段的结果,
**不会中断整场评测**。

三个维度的分工:

| 维度 | 判什么 |
| --- | --- |
| `accuracy` 准确性 | 回答里的事实/数字是否与证据一致 |
| `completeness` 完整性 | 是否覆盖了 `expect_points` |
| `faithfulness` 忠实度 | 是否只基于证据,有没有编造证据里没有的内容 |

prompt 里还写了两条**校准说明**(都是踩坑后加的):
- 证据可能被截断,别因个别细节没找到就判低分;
- 概念题里的举例数字(如"假设股价100")属合理教学示例,不算编造。

### 4. `runner.py` —— 总指挥

串起全流程。几个要点:

**PASS 判定**(阈值在文件顶部,可调):
```python
MIN_FAITHFULNESS = 4      # 忠实度至少 4/5
MIN_GROUNDED_RATIO = 0.6  # 数字至少 60% 有出处

def is_pass(case, checks, judged):
    ok = checks["tool_ok"] and judged["faithfulness"] >= MIN_FAITHFULNESS
    if case.get("check_number_grounding", True):   # 概念题可关掉这道门槛
        ok = ok and checks["grounded_ratio"] >= MIN_GROUNDED_RATIO
    return ok
```

**评测时关闭长期记忆写入**:
```python
agent.use_long_term = False
```
两个原因:① 不把评测问答污染进记忆库;② 每次评测更接近干净冷启动,结果更可复现。

**支持只跑指定题**(命令行传 id),这样调一个参数不用等全量跑完:
```python
only = set(sys.argv[1:])
if only:
    cases = [c for c in cases if c["id"] in only]
```

---

## 三、怎么用

### 跑评测

```bash
python -m eval.runner                    # 全部用例
python -m eval.runner concept_pe_01      # 只跑一道题(快速迭代)
python -m eval.runner nvda_perf_01 tsla_price_01   # 跑指定几道
```

> 全量跑一次可能要 5~10 分钟(Agent 每题都要真实联网调工具),调参时建议先单题跑。

### 加一道新题

往 `cases.jsonl` 追加一行即可:

```json
{"id": "aapl_price_01", "question": "苹果(AAPL)最近的股价表现如何?", "must_call_tools": ["get_stock_price"], "expect_points": ["最新价或涨跌幅", "走势描述"], "forbid": []}
```

写用例的几个经验:
- `must_call_tools` 只填**真正必须**的,填多了会制造假失败;
- `expect_points` 写得具体、可判断,别写"回答得好"这种没法判的;
- **概念解释题记得加 `"check_number_grounding": false`**,否则举例数字会把它误杀。

### 读报告

`eval/reports/` 下每次跑会生成一对文件:

- **`.md`**:人类看的。顶部是通过率和各维度均分,中间是逐题表格,底部"需要关注的用例"列出
  失败原因(缺哪个工具、哪些数字没出处、裁判理由)。
- **`.json`**:机器读的。含每题**完整答案**和全部指标,用于回归对比或事后深挖。

仓库里保留了两份作为**回归对比样例**:
`2026-09-03_162903`(校准前,通过率 20%)与 `2026-09-03_163938`(校准后,60%)。

### 做回归对比

这是这套东西最大的价值——**改动前后各跑一次,用数字说话**:

1. 改动前跑一次,记下通过率和三个均分;
2. 改 prompt / 换模型 / 调工具;
3. 再跑一次,对比。分数掉了就是退步,别自我感觉良好。

---

## 四、常见坑(都是踩过的)

| 现象 | 多半是什么原因 |
| --- | --- |
| `faithfulness` 普遍很低,但 `grounded_ratio` 很高 | **裁判证据不足**。两个指标打架时优先信规则层,去检查 `build_judge_facts` 的限额 |
| 概念解释题总是 FAIL,溯源=0 | 忘了加 `"check_number_grounding": false` |
| 某题分数忽高忽低 | Agent 本身方差大(不收敛/过度调工具),不是评测器的问题 |
| 裁判返回 `error` | 模型没按要求输出 JSON。看 `_parse_json` 是否需要加容错 |
| 全量跑太慢 | 用 `python -m eval.runner <id>` 只跑关心的题 |

---

## 五、待扩展(留白)

- **`forbid` 字段还没接上**:目前只是写在用例里当说明,没有代码消费它。
  可以在 `checks.py` 加一条规则(如关键词命中检测),或在裁判 prompt 里注入禁止项。
- **多次取均值**:裁判打分有波动,关键用例可以跑 N 次取平均,降低随机性。
- **更多维度**:引用可点击性、时效性(是否用了最新新闻)、拒答恰当性。
- **成本统计**:记录每题消耗的 token 和费用,把"效果"和"成本"一起看。
- **接入 CI**:每次提交自动跑一轮小规模评测,分数掉了就报警。
