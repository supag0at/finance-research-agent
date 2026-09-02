"""M2 阶段的工具:真实数据源。

和 M1 相比,函数的"对外样子"(名字、参数、返回类型)几乎没变,变的只是
内部实现——从写死的假数据换成了真实接口。这正是我们一直强调的
"接口稳定、实现可替换":主循环和注册逻辑基本不用动。

三个工具:
  get_stock_price:用 yfinance 拿真实美股行情(免费、无需 key)。
  search_news:    用 DuckDuckGo(ddgs)做真实网页搜索(免费、无需 key)。
  fetch_page:     抓取网页正文,抓之前先过 sandbox 的安全护栏。
"""

import httpx
from bs4 import BeautifulSoup

from core.sandbox import (
    HTTP_TIMEOUT,
    MAX_DOWNLOAD_BYTES,
    MAX_TEXT_CHARS,
    is_safe_url,
)


def get_stock_price(symbol: str, days: int = 30) -> dict:
    """查询某只股票最近若干个交易日的表现(真实数据,来自 Yahoo chart API)。

    实现说明:我们不再用 yfinance 库(它内部有个 endpoint 容易被 Yahoo 限流,
    尤其在共享代理 IP 下)。改成直接用 httpx 调 Yahoo 的原始 chart 接口,
    更稳、更轻,而且拿到的就是我们要的历史序列。

    参数:
        symbol: 股票代码,如 "NVDA"、"AAPL"。
        days: 最近多少个交易日,默认 30。
    返回:
        含最新价、区间涨跌幅、最高/最低价的字典。
    """
    symbol = symbol.upper().strip()
    params = {"range": "3mo", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0 (finance-research-agent)"}

    # query1 偶尔会限流,自动切到 query2 再试一次(多主机降级)。
    last_error = "未知错误"
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{symbol}"
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, headers=headers) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001 换下一个主机再试
            last_error = f"{type(exc).__name__}: {exc}"
            continue

        result = (data.get("chart") or {}).get("result")
        if not result:
            err = (data.get("chart") or {}).get("error")
            return {"symbol": symbol, "error": f"未取到行情(代码可能不存在):{err}"}

        quote = result[0]["indicators"]["quote"][0]
        # 行情序列里可能有 None(停牌等),先过滤掉。
        closes = [c for c in (quote.get("close") or []) if c is not None]
        highs = [h for h in (quote.get("high") or []) if h is not None]
        lows = [x for x in (quote.get("low") or []) if x is not None]
        if not closes:
            return {"symbol": symbol, "error": "行情序列为空"}

        closes, highs, lows = closes[-days:], highs[-days:], lows[-days:]
        first_close, last_close = closes[0], closes[-1]
        change_pct = (last_close - first_close) / first_close * 100 if first_close else 0.0

        return {
            "symbol": symbol,
            "days": len(closes),
            "last_close": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "high": round(max(highs), 2) if highs else None,
            "low": round(min(lows), 2) if lows else None,
            "source": "yahoo-chart",
        }

    # 两个主机都失败,抛出去让上层的可靠性层去重试/降级。
    raise RuntimeError(f"Yahoo chart 请求失败:{last_error}")


def search_news(query: str, max_results: int = 5) -> list:
    """按关键词做真实网页搜索,返回若干条结果(标题 + 链接 + 摘要)。

    参数:
        query: 搜索关键词,如 "NVDA stock surge reason"。
        max_results: 返回条数,默认 5。
    返回:
        结果列表,每条含 title / url / snippet。
    """
    # ddgs 是 DuckDuckGo 的非官方客户端,免费、无需 API key。
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
    if not results:
        return [{"title": "无搜索结果", "url": "", "snippet": query}]
    return results


def fetch_page(url: str) -> str:
    """抓取网页并提取正文文本(抓取前先过安全护栏)。

    参数:
        url: 要抓取的网页地址(http/https)。
    返回:
        网页正文纯文本(截断到上限长度);若 URL 不安全或抓取失败则返回说明。
    """
    # (1) 先过 Sandbox 安全护栏:挡掉内网/本机地址,防 SSRF。
    if not is_safe_url(url):
        return f"[已拦截] 不允许访问该地址(可能是内网/本机/非法协议):{url}"

    # (2) 抓取,带超时;follow_redirects 防止被重定向绕过检查时也控制大小。
    headers = {"User-Agent": "Mozilla/5.0 (finance-research-agent)"}
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        # (3) 大小上限:只取前 MAX_DOWNLOAD_BYTES 字节,避免超大页面撑爆内存。
        raw = resp.content[:MAX_DOWNLOAD_BYTES]

    # (4) 用 BeautifulSoup 提取可见正文,去掉脚本/样式。
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)

    # (5) 文本上限:避免把太长的内容塞进上下文。
    return text[:MAX_TEXT_CHARS]
