"""财经数字的抽取与归一化 —— 评测溯源和压缩时的事实校验共用同一套规则。

原本写在 eval/checks.py 里。M8 的事实校验(memory/context/facts.py)也要用,
而 memory/ 不能反向依赖 eval/,所以迁到 core/。规则本身一字未改:
"压缩时认为这个数字在原文里"和"评测时认为这个数字有出处"必须是同一个判据,
否则会出现压缩放行、评测判编造(或反过来)的情况。
"""

import re

# 匹配"数字"的正则:整数或小数(逗号已在预处理里去掉)。
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """去掉千分位逗号,方便 '1,080' 和 '1080' 能对上。"""
    return text.replace(",", "")


def meaningful_numbers(text: str) -> list[str]:
    """从文本里抽取"有意义的财经数字"。

    为降低误报,只保留:带小数点的数(如 224.41、7.5)或 >=100 的整数
    (如 962、1080)。像 '3 个板块'、年份 '2026'、月份 '8' 这种通用小整数
    不作为"是否编造"的判据(它们太常见,极易误伤)。
    """
    nums = []
    for m in NUM_RE.finditer(normalize(text)):
        tok = m.group()
        try:
            val = float(tok)
        except ValueError:
            continue
        if "." in tok or val >= 100:
            nums.append(tok)
    return nums
