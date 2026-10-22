"""从巨潮资讯(cninfo,证监会指定披露平台)拉 A 股年报 / 半年报 PDF(M9 Step 1)。

    python -m rag.sources.fetch_cninfo

每家公司取最新一份年报 + 最新一份半年报(跳过"摘要"和"港股公告"),
存到 data/corpus/raw/cninfo/,元数据写进同目录 manifest.json。

两个接口(都是网页自己在用的,无需登录):
  1. topSearch/query:股票代码 → orgId
  2. hisAnnouncement/query:按 category 查公告列表(ndbg=年报,bndbg=半年报)
国内站点,直连(trust_env=False)。
"""

import json
import time
from pathlib import Path

import httpx

COMPANIES = {"688256": "寒武纪", "688981": "中芯国际", "688041": "海光信息"}
CATEGORIES = {"category_ndbg_szsh": "年报", "category_bndbg_szsh": "半年报"}
OUT_DIR = Path("data/corpus/raw/cninfo")
BASE = "http://www.cninfo.com.cn/new"
STATIC = "http://static.cninfo.com.cn/"


def _latest_full_report(anns: list[dict]) -> dict | None:
    """公告列表按时间倒序;取第一份不是摘要、不是港股公告的。"""
    for a in anns:
        title = a["announcementTitle"]
        if "摘要" in title or "港股" in title or "英文" in title:
            continue
        return a
    return None


def fetch_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    http = httpx.Client(trust_env=False, timeout=60, headers={"User-Agent": "Mozilla/5.0"})

    for code, name in COMPANIES.items():
        org = http.post(f"{BASE}/information/topSearch/query", data={"keyWord": code, "maxNum": 5}).json()[0]
        for cat, doc_type in CATEGORIES.items():
            resp = http.post(
                f"{BASE}/hisAnnouncement/query",
                data={
                    "stock": f"{code},{org['orgId']}", "tabName": "fulltext", "pageSize": 30, "pageNum": 1,
                    "column": "sse", "category": cat, "plate": "sh", "isHLtitle": "true",
                },
            )
            a = _latest_full_report(resp.json().get("announcements") or [])
            if a is None:
                print(f"{code} {name} 没有找到{doc_type}")
                continue
            title = a["announcementTitle"].replace("<em>", "").replace("</em>", "")
            year = int(title[title.index("20") : title.index("20") + 4])  # "2025年年度报告" → 2025
            period_end = f"{year}-12-31" if doc_type == "年报" else f"{year}-06-30"
            fname = f"{code}-{doc_type}-{year}.pdf"
            path = OUT_DIR / fname
            url = STATIC + a["adjunctUrl"]
            if not path.exists():
                pdf = http.get(url)
                pdf.raise_for_status()
                path.write_bytes(pdf.content)
                time.sleep(0.5)
            manifest[fname] = {
                "ticker": code,
                "company": name,
                "doc_type": doc_type,
                "fiscal_year": year,  # A 股财年 = 自然年
                "period_end": period_end,
                "title": title,
                "source_url": url,
                "trust_level": "official",
                "lang": "zh",
            }
            print(f"{fname}  {title}  {path.stat().st_size / 1e6:.1f} MB")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共 {len(manifest)} 份,manifest → {manifest_path}")


if __name__ == "__main__":
    fetch_all()
