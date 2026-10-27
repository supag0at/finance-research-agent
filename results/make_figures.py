"""Regenerate the README figures from the files in results/.

    python results/make_figures.py

Writes light and dark PNGs to assets/figures/. Retrieval numbers are recomputed
from results/retrieval/rag_offline_20261022.json; the other figures use the
aggregates printed in the cited report files (noted next to each constant).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "dim": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "dim": "#4a4a46"},
}
FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def retrieval_rows():
    """Fact-question Hit@5 / MRR per ablation row (72 questions)."""
    data = json.loads((ROOT / "results/retrieval/rag_offline_20261022.json").read_text(encoding="utf-8"))
    golden = [json.loads(l) for l in (ROOT / "eval/rag_golden.jsonl").read_text(encoding="utf-8").splitlines() if l]
    kind = {f"g{i:03d}": g["kind"] for i, g in enumerate(golden)}
    rows = []
    for key in ["0", "1", "2", "3", "4", "4b"]:
        items = [v for q, v in data["rows"][key]["per_item"].items() if kind.get(q) == "fact"]
        rows.append((sum(x["hit@5"] for x in items) / len(items), sum(x["mrr"] for x in items) / len(items)))
    return rows


def _axes(t, figsize=(8, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _title(fig, t, title, subtitle):
    fig.text(0.02, 0.965, title, color=t["ink"], fontsize=12.5, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.905, subtitle, color=t["ink2"], fontsize=9, ha="left", va="top")


def _legend(ax, t):
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0, 1.02))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


def _label(ax, t, x, y, text, bold=False):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=8.5, color=t["ink"] if bold else t["ink2"],
            fontweight="bold" if bold else "normal")


def fig_retrieval(t):
    rows = retrieval_rows()
    names = ["Fixed chunks\n+ dense (v1)", "Parent-child\n+ dense", "Parent-child\n+ BM25",
             "Hybrid\n(RRF)", "+ rerank\n0.6B", "+ rerank\n8B"]
    fig, ax = _axes(t)
    w = 0.36
    xs = range(len(names))
    ax.bar([x - w / 2 - 0.01 for x in xs], [r[0] for r in rows], w, color=t["s1"], label="Hit@5")
    ax.bar([x + w / 2 + 0.01 for x in xs], [r[1] for r in rows], w, color=t["s2"], label="MRR")
    for x, (hit, _) in zip(xs, rows):
        _label(ax, t, x - w / 2 - 0.01, hit + 0.015, f"{hit:.2f}", bold=True)
    ax.set_xticks(list(xs), names, color=t["ink2"], fontsize=8.5)
    ax.set_ylim(0, 1.08)
    _legend(ax, t)
    _title(fig, t, "Retrieval ablation: each layer's contribution",
           "Same 24 filings, same 72 fact questions. Hit@5 0.31 → 0.93; rerank cannot exceed the Hit@30 ceiling of 0.93.")
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.06, right=0.98)
    return fig


def fig_refusal(t):
    # results/retrieval/rag_agentic_20261022.json, rows 4b* / 5 / 6 / 7
    names = ["Hybrid + rerank\n(no agent)", "+ query analysis", "+ HyDE", "Full agentic\n(+ evidence grading)"]
    refused = [6, 3, 5, 10]
    fig, ax = _axes(t, figsize=(8, 3.8))
    colors = [t["dim"]] * 3 + [t["s1"]]
    ax.bar(range(4), refused, 0.5, color=colors)
    for x, v in enumerate(refused):
        _label(ax, t, x, v + 0.15, f"{v}/10", bold=(x == 3))
    ax.set_xticks(range(4), names, color=t["ink2"], fontsize=8.5)
    ax.set_ylim(0, 11.5)
    ax.set_yticks([0, 5, 10])
    _title(fig, t, "Correct refusals on 10 unanswerable questions",
           "Multi-query recall alone makes refusal worse (6 → 3); evidence grading restores it (10/10) with 0/75 false refusals.")
    fig.subplots_adjust(top=0.76, bottom=0.18, left=0.06, right=0.98)
    return fig


def fig_e2e(t):
    # results/e2e/: v1 = 2026-10-22_084500, hybrid = 2026-10-24_082937, agentic = 2026-10-24_082818 (14 cases each)
    names = ["v1 (dense, toy corpus)", "Hybrid", "Agentic"]
    tools = [9.2, 4.2, 2.4]
    web = [6.1, 1.6, 0.8]
    fig, ax = _axes(t, figsize=(8, 3.8))
    w = 0.32
    xs = range(3)
    ax.bar([x - w / 2 - 0.01 for x in xs], tools, w, color=t["s1"], label="Tool calls / question")
    ax.bar([x + w / 2 + 0.01 for x in xs], web, w, color=t["s2"], label="Web calls / question")
    for x, a, b in zip(xs, tools, web):
        _label(ax, t, x - w / 2 - 0.01, a + 0.15, f"{a:.1f}", bold=True)
        _label(ax, t, x + w / 2 + 0.01, b + 0.15, f"{b:.1f}")
    ax.set_xticks(list(xs), names, color=t["ink2"], fontsize=9)
    ax.set_ylim(0, 11)
    _legend(ax, t)
    _title(fig, t, "End-to-end: how hard the agent works to answer",
           "14 end-to-end questions. Pass rates are within run-to-run noise; process metrics are stable across runs.")
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.06, right=0.98)
    return fig


def fig_budget(t):
    # M6 calibration runs on the 5-case regression set (total cost of one run)
    names = ["No run budget\n(baseline)", "Budget set below\nhealthy runs", "Budget calibrated\n(≈2× worst healthy run)"]
    cost = [0.0990, 0.1389, 0.1179]
    fig, ax = _axes(t, figsize=(8, 3.6))
    colors = [t["dim"], t["s2"], t["s1"]]
    ax.bar(range(3), cost, 0.5, color=colors)
    for x, v in enumerate(cost):
        _label(ax, t, x, v + 0.002, f"¥{v:.4f}", bold=(x == 1))
    ax.set_xticks(range(3), names, color=t["ink2"], fontsize=8.5)
    ax.set_ylim(0, 0.16)
    _title(fig, t, "A guardrail set inside the normal range costs more, not less",
           "Total cost of the 5-case regression run; pass rate 5/5 in all three. The too-tight budget cut healthy runs mid-way (+40%).")
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.08, right=0.98)
    return fig


def main():
    plt.rcParams["font.family"] = FONT
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("retrieval-ablation", fig_retrieval), ("agentic-refusal", fig_refusal),
                     ("e2e-process", fig_e2e), ("budget-lesson", fig_budget)]:
        for mode, t in THEMES.items():
            fig = fn(t)
            fig.savefig(OUT / f"{name}-{mode}.png", facecolor=t["surface"])
            plt.close(fig)
            print("wrote", OUT / f"{name}-{mode}.png")


if __name__ == "__main__":
    main()
