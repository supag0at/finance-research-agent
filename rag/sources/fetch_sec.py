"""从 SEC EDGAR 拉 10-K / 10-Q 原文(M9 Step 1)。

    python -m rag.sources.fetch_sec

每个 ticker 取最近 N_10K 份 10-K + 最近 1 份 10-Q,存到 data/corpus/raw/sec/,
元数据写进同目录的 manifest.json(入库时用,不再回头查 EDGAR)。

EDGAR 的两个要求:
  1. 必须带 User-Agent(写明用途 + 联系方式),否则 403;
  2. 限速 10 次/秒。我们一共十几个请求,串行 + 小睡就够了。
直连比走 Clash 快一个数量级(实测 0.35s vs 4.6s),所以 trust_env=False。
"""

import json
import time
from pathlib import Path

import httpx

TICKERS = {  # ticker -> CIK(10 位补零)
    "NVDA": "0001045810",
    "AMD": "0000002488",
    "INTC": "0000050863",
    "AVGO": "0001730168",
    "MU": "0000723125",
}
N_10K = 2
OUT_DIR = Path("data/corpus/raw/sec")
UA = {"User-Agent": "finance-research-agent study project admin@example.com"}


def _pick(recent: dict) -> list[dict]:
    rows = [
        {k: recent[k][i] for k in ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument")}
        for i in range(len(recent["form"]))
    ]
    tenk = [r for r in rows if r["form"] == "10-K"][:N_10K]
    tenq = [r for r in rows if r["form"] == "10-Q"][:1]
    return tenk + tenq


def fetch_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    http = httpx.Client(trust_env=False, timeout=60, headers=UA, follow_redirects=True)

    for ticker, cik in TICKERS.items():
        sub = http.get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
        for r in _pick(sub["filings"]["recent"]):
            fname = f"{ticker.lower()}-{r['form'].lower()}-{r['reportDate']}.htm"
            path = OUT_DIR / fname
            acc = r["accessionNumber"].replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{r['primaryDocument']}"
            if not path.exists():
                resp = http.get(url)
                resp.raise_for_status()
                path.write_bytes(resp.content)
                time.sleep(0.3)
            manifest[fname] = {
                "ticker": ticker,
                "doc_type": r["form"],
                "period_end": r["reportDate"],
                "filed": r["filingDate"],
                "source_url": url,
                "trust_level": "official",
                "lang": "en",
            }
            print(f"{fname}  {path.stat().st_size / 1e6:.1f} MB")
        time.sleep(0.3)

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共 {len(manifest)} 份,manifest → {manifest_path}")


if __name__ == "__main__":
    fetch_all()
