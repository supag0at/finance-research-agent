"""压缩 A/B 用的回放工具与用例(M8)。

为什么不用真实网络:M6 的结论 —— 境外链路的失败率在不同日子是 5.6% / 9.7% / 47%,
噪声比要测的效果大一个量级,拿两轮真实评测做 A/B 等于在比较天气。
所以这里把四个业务工具换成**同名、同参数 schema** 的确定性版本:
返回固定的长文本,在固定的目标上固定地失败。

几个刻意的设计:
  - 关键数字是**虚构**的(如数据中心收入 387.6 亿美元)。模型不可能凭参数知识答对,
    答案里出现它们,只能说明它真的从工具结果里拿到并保留住了;
  - 关键句放在网页正文的固定位置:有的在 1500 字预览之内,有的在之后(只能靠回读);
  - 工具仍经 graph/tool_adapters._reliable_tool 包装:失败文本走真实的三段式渲染、
    分类重试和熔断路径,和线上看到的一模一样。
"""

import httpx

from graph.tool_adapters import (
    FetchPageArgs,
    RetrieveArgs,
    SearchNewsArgs,
    StockPriceArgs,
    _reliable_tool,
)

PAGE_CHARS = 8000  # 与 core/sandbox.MAX_TEXT_CHARS 一致:真实 fetch_page 的上限

# 不含任何"有意义数字"(带小数或 >=100 的整数)的套话,避免干扰数字溯源。
_FILLER = [
    "分析人士指出,行业景气度仍取决于下游需求的恢复节奏。",
    "多家机构在近期报告中维持了对板块的中性偏积极判断。",
    "市场普遍关注供应链各环节的库存水平与资本开支计划。",
    "从历史经验看,半导体周期的拐点往往领先于宏观数据。",
    "部分投资者担忧估值已经充分反映了乐观预期。",
    "公司管理层在电话会上回避了对长期利润率的具体承诺。",
    "海外监管政策的变化仍是影响产业格局的重要变量。",
    "渠道调研显示,终端客户的采购节奏较上季度有所放缓。",
    "产业链人士认为,先进封装产能仍是制约出货的关键环节。",
    "该报道发布后,相关概念股在盘中出现明显分化。",
]


def make_page(title: str, inserts: list[tuple[int, str]], length: int = PAGE_CHARS) -> str:
    """生成一篇确定性的长网页正文,并把关键句插在指定字符位置附近(句子边界)。"""
    filler, i = [], 0
    while sum(len(s) for s in filler) < length:
        filler.append(_FILLER[i % len(_FILLER)])
        i += 1
    text = f"{title}\n" + "".join(filler)
    for pos, sentence in sorted(inserts, reverse=True):
        cut = text.rfind("。", 0, pos) + 1 or pos
        text = text[:cut] + sentence + text[cut:]
    return text[: length + sum(len(s) for _, s in inserts)]


class _Resp:
    def __init__(self, code: int) -> None:
        self.status_code = code


def _raise(spec) -> None:
    if spec == "timeout":
        raise httpx.ConnectTimeout("connection timed out")
    raise httpx.HTTPStatusError(f"{spec} Client Error", request=None, response=_Resp(int(spec)))


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------

NV_URL = "https://news.example-fin.com/nvda-fy26q2-earnings"
REUTERS_NV = "https://www.reuters.com/technology/nvidia-supply-cowos"
REUTERS_MU = "https://www.reuters.com/technology/micron-hbm-capacity"
BBG_MU = "https://www.bloomberg.com/news/micron-hbm"
SEMI_MU = "https://www.example-semi.com/micron-hbm-expansion"

_NV_PAGE = make_page(
    "英伟达公布最新季度财报:数据中心业务继续扩张",
    [
        (3200, "财报显示,本季度数据中心业务收入为 387.6 亿美元,继续是公司最大的收入来源。"),
        (5200, "公司整体毛利率为 71.8%,管理层称下季度仍面临封装产能约束。"),
    ],
)


def _generic(title: str) -> str:
    return make_page(title, [])


CASES: list[dict] = [
    {
        "id": "deep_page_01",
        "question": "英伟达最新一季财报里,数据中心业务收入是多少?整体毛利率是多少?请以新闻正文为准并注明出处。",
        "golden_numbers": ["387.6", "71.8"],
        "stocks": {"NVDA": (181.52, 6.37, 186.9, 168.2)},
        "search": [{"title": "英伟达公布最新季度财报", "url": NV_URL, "snippet": "英伟达公布最新季度财报,数据中心业务继续增长。"},
                   {"title": "英伟达财报前瞻", "url": "https://news.example-fin.com/nvda-preview", "snippet": "市场关注数据中心与毛利率。"},
                   {"title": "AI 芯片需求观察", "url": "https://news.example-fin.com/ai-chip-demand", "snippet": "云厂商资本开支维持高位。"}],
        "pages": {NV_URL: _NV_PAGE},
    },
    {
        "id": "early_fact_01",
        "question": "比较 AMD、英特尔(INTC)、台积电(TSM)最近 30 个交易日的股价表现(最新价和涨跌幅),"
                    "再分别抓一篇新闻正文解释各自涨跌的原因。",
        "golden_numbers": ["163.27", "4.81", "21.94", "7.36", "188.45", "2.19"],
        "stocks": {"AMD": (163.27, -4.81, 176.4, 158.9), "INTC": (21.94, 7.36, 22.8, 19.7),
                   "TSM": (188.45, 2.19, 193.1, 180.6)},
        "search_by_keyword": {
            ("AMD",): [{"title": "AMD 指引低于预期", "url": "https://news.example-fin.com/amd-guidance", "snippet": "AMD 股价承压。"}],
            ("英特尔", "INTC", "Intel"): [{"title": "英特尔获新代工订单", "url": "https://news.example-fin.com/intel-foundry", "snippet": "英特尔股价走强。"}],
            ("台积电", "TSM", "TSMC"): [{"title": "台积电月度营收", "url": "https://news.example-fin.com/tsmc-revenue", "snippet": "台积电营收稳健。"}],
        },
        "pages": {
            "https://news.example-fin.com/amd-guidance": make_page("AMD 指引低于预期", [(900, "AMD 下调了数据中心 GPU 的出货指引,这是股价近期回落的主因。")]),
            "https://news.example-fin.com/intel-foundry": make_page("英特尔获新代工订单", [(900, "英特尔宣布获得一家大型云厂商的先进制程代工订单,股价随之走强。")]),
            "https://news.example-fin.com/tsmc-revenue": make_page("台积电月度营收", [(900, "台积电公布的月度营收保持增长,先进制程需求稳定。")]),
        },
    },
    {
        "id": "fail_mix_01",
        "question": "路透社关于美光(MU)HBM 产能扩张的最新报道说了什么?美光最近 30 个交易日股价表现如何?",
        "golden_numbers": ["6.2", "102.36", "12.04"],
        "stocks": {"MU": (102.36, 12.04, 104.9, 90.1)},
        "search": [{"title": "Reuters: Micron to expand HBM capacity", "url": REUTERS_MU, "snippet": "美光计划扩大 HBM 产能。"},
                   {"title": "Bloomberg: Micron HBM", "url": BBG_MU, "snippet": "美光 HBM 扩产细节。"},
                   {"title": "美光 HBM 扩产(转载路透社报道)", "url": SEMI_MU, "snippet": "转载路透社关于美光 HBM 产能的报道。"}],
        "pages": {
            REUTERS_MU: {"fail": 403},
            BBG_MU: {"fail": "timeout"},
            SEMI_MU: make_page("美光 HBM 扩产(转载路透社报道)", [(1100, "据路透社报道,美光计划将 HBM 月产能提升至 6.2 万片晶圆,新产线将于明年投产。")]),
        },
    },
    {
        "id": "combo_01",
        "question": "汇总英伟达最新季度的数据中心收入、整体毛利率和最近 30 个交易日股价表现,"
                    "并说明路透社关于其封装产能有什么报道。",
        "golden_numbers": ["387.6", "71.8", "181.52"],
        "stocks": {"NVDA": (181.52, 6.37, 186.9, 168.2)},
        "search": [{"title": "英伟达公布最新季度财报", "url": NV_URL, "snippet": "英伟达公布最新季度财报。"},
                   {"title": "Reuters: Nvidia CoWoS supply", "url": REUTERS_NV, "snippet": "路透社:英伟达封装产能。"},
                   {"title": "英伟达封装产能观察", "url": "https://news.example-fin.com/nvda-cowos", "snippet": "先进封装仍是瓶颈。"}],
        "pages": {NV_URL: _NV_PAGE, REUTERS_NV: {"fail": 403},
                  "https://news.example-fin.com/nvda-cowos": make_page("英伟达封装产能观察", [(1200, "业内人士称英伟达的先进封装产能仍偏紧,路透社此前也报道了相关瓶颈。")])},
    },
]


# --------------------------------------------------------------------------
# 回放工具
# --------------------------------------------------------------------------


def build_replay_tools(case: dict) -> list:
    """与 graph/tool_adapters.build_tools() 同名、同 schema、同可靠性参数的确定性工具。"""

    def get_stock_price(symbol: str, days: int = 30) -> dict:
        row = case.get("stocks", {}).get(symbol.upper().strip())
        if row is None:
            return {"symbol": symbol, "error": "未取到行情(代码可能不存在):None"}
        last, chg, hi, lo = row
        return {"symbol": symbol.upper(), "days": days, "last_close": last, "change_pct": chg,
                "high": hi, "low": lo, "source": "yahoo-chart"}

    def search_news(query: str) -> list:
        for keys, results in (case.get("search_by_keyword") or {}).items():
            if any(k.lower() in query.lower() for k in keys):
                return results
        if case.get("search"):
            return case["search"]
        return [r for rs in (case.get("search_by_keyword") or {}).values() for r in rs]

    def fetch_page(url: str) -> str:
        spec = case.get("pages", {}).get(url.strip())
        if spec is None:
            _raise(404)
        if isinstance(spec, dict):
            _raise(spec["fail"])
        return spec

    def retrieve(query: str, top_k: int = 4) -> str:
        return _generic(f"知识库:{query}")[:1200]

    return [
        _reliable_tool(get_stock_price, name="get_stock_price",
                       description="查询某只股票最近若干个交易日的涨跌幅、最新价、最高/最低价。需要行情数据时调用。",
                       args_schema=StockPriceArgs, timeout=20.0, retries=2),
        _reliable_tool(search_news, name="search_news",
                       description="按关键词做网页搜索,返回相关新闻的标题、链接和摘要。需要了解事件/原因时调用。",
                       args_schema=SearchNewsArgs, timeout=20.0, retries=2),
        _reliable_tool(fetch_page, name="fetch_page",
                       description="抓取一个网页的正文文本。当搜索结果的摘要不够、需要看细节时调用。",
                       args_schema=FetchPageArgs, timeout=15.0, retries=1),
        _reliable_tool(retrieve, name="retrieve",
                       description="从本地知识库按语义检索背景资料(公司背景、行业知识、术语解释等)。需要背景知识时调用。",
                       args_schema=RetrieveArgs, timeout=20.0, retries=1),
    ]


def case_by_id(case_id: str) -> dict:
    return next(c for c in CASES if c["id"] == case_id)
