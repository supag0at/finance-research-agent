"""M1 阶段的工具:先用"假数据"跑通整个流程。

重要提示:这里的行情、新闻全是写死的假数据(mock),目的是让你先看清
Agent 的循环是怎么转起来的——不用先去申请一堆外部 API。
到了 M2,我们会把这两个函数的内部实现换成真实的行情接口和搜索接口,
而它们的"对外样子"(函数签名)保持不变,所以主循环和工具注册都不用改。
这就是"接口稳定、实现可替换"的好处。
"""

# 一点假数据:几只常见美股 + 一个兜底
_FAKE_PRICES = {
    "NVDA": {"change_pct": 18.2, "last": 178.4, "trend": "近一个月大幅上涨"},
    "AAPL": {"change_pct": 3.1, "last": 231.5, "trend": "小幅震荡上行"},
    "TSLA": {"change_pct": -6.4, "last": 242.0, "trend": "回调"},
}

_FAKE_NEWS = {
    "NVDA": [
        {"title": "英伟达 Q3 数据中心营收超预期,同比大增", "source": "财经网"},
        {"title": "多家投行上调英伟达目标价", "source": "华尔街日报"},
        {"title": "AI 芯片需求持续旺盛,供应仍偏紧", "source": "路透"},
    ],
}


def get_stock_price(symbol: str, days: int = 5) -> dict:
    """查询某只股票最近若干天的表现(M1 为假数据)。

    参数:
        symbol: 股票代码,如 "NVDA"。
        days: 最近多少天,默认 5。
    返回:
        一个含涨跌幅、最新价、趋势描述的字典。
    """
    symbol = symbol.upper()
    data = _FAKE_PRICES.get(
        symbol,
        {"change_pct": 0.0, "last": 100.0, "trend": "无该股票的假数据"},
    )
    return {"symbol": symbol, "days": days, **data}


def search_news(query: str) -> list:
    """按关键词搜索相关新闻(M1 为假数据)。

    参数:
        query: 搜索关键词,如 "英伟达 上涨 原因"。
    返回:
        新闻列表,每条含标题和来源。
    """
    for key, news in _FAKE_NEWS.items():
        if key in query.upper() or key.lower() in query.lower():
            return news
    return [{"title": f"未找到与「{query}」高度相关的新闻(假数据)", "source": "-"}]
