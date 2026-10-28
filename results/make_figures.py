"""Regenerate the README figures from the files in results/.

    python results/make_figures.py

Writes light and dark PNGs to assets/figures/ (backgrounds match GitHub's themes).
Retrieval numbers are recomputed from results/retrieval/rag_offline_20261022.json;
the other figures use the aggregates in the cited report files (noted next to
each constant). Figures carry no titles: captions live in the README.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "figures"

THEMES = {
    "light": {"bg": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "muted": "#818b98",
              "grid": "#eaecef", "s1": "#2e6bb8", "s2": "#d9713c", "dim": "#c8ccd1", "empty": "#eff1f3"},
    "dark": {"bg": "#0d1117", "ink": "#e6edf3", "ink2": "#9198a1", "muted": "#7d8590",
             "grid": "#21262d", "s1": "#5689cf", "s2": "#cc7843", "dim": "#3d444d", "empty": "#21262d"},
}
FONT = ["Bahnschrift", "Segoe UI", "Arial", "DejaVu Sans"]
W = 7.2  # inches; ~1440 px at 200 dpi, shown at GitHub's ~880 px column


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


def _fig(t, h):
    fig, ax = plt.subplots(figsize=(W, h), dpi=200)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=t["muted"], length=0, labelsize=8.5)
    ax.set_axisbelow(True)
    return fig, ax


def _ylabels(ax, t, labels, bold_last=False):
    ax.set_yticks(range(len(labels)), labels)
    for i, lab in enumerate(ax.get_yticklabels()):
        lab.set_color(t["ink"] if (bold_last and i == len(labels) - 1) else t["ink2"])
        lab.set_fontsize(9)
        if bold_last and i == len(labels) - 1:
            lab.set_fontweight("bold")
    ax.invert_yaxis()


def _legend(ax, t, handles, loc="lower left", anchor=(0, 1.0)):
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, frameon=False, fontsize=8.5,
                    handletextpad=0.4, borderaxespad=0.1, ncol=len(handles))
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])


# Warm "stage" palette for the retrieval ablation: one color per stage of the pipeline.
STAGES = {
    "light": {"base": "#C6C8BC", "mid": "#E9DCBC", "final": "#D49F7C",
              "axis": "#2b2a27", "divider": "#9a988f"},
    "dark": {"base": "#8E9186", "mid": "#B9A983", "final": "#C98B66",
             "axis": "#9198a1", "divider": "#5b5a55"},
}


def _rounded_bar(ax, x, h, w, color, radius_px=7):
    """Bar with rounded top corners; radius given in pixels so it survives the axes aspect."""
    from matplotlib.patches import FancyBboxPatch
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rx = radius_px * (x1 - x0) / bbox.width
    ry = radius_px * (y1 - y0) / bbox.height
    r = min(rx, w / 2)
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=ry / rx, facecolor=color, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((x - w / 2, 0), w, min(h, ry * 1.2), facecolor=color, edgecolor="none", zorder=2))


def _bars(t, *, xs, values, colors, labels, ticks, ylim, yticks, ylabel, ysub,
          groups=(), dividers=(), bold=(), h=3.7, width=0.78):
    """Shared style for every bar chart: warm stage colors, rounded tops, values on top,
    dashed dividers between groups, bold metric label on the left."""
    p = {**t, **STAGES["dark" if t["bg"] == "#0d1117" else "light"]}
    fig, ax = plt.subplots(figsize=(W, h), dpi=200)
    fig.patch.set_facecolor(p["bg"])
    ax.set_facecolor(p["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(p["axis"])
        ax.spines[side].set_linewidth(1.4)
    ax.yaxis.grid(True, color=p["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(0, ylim)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.24 if groups else 0.17)
    fig.canvas.draw()
    for i, (x, v, c, lab) in enumerate(zip(xs, values, colors, labels)):
        _rounded_bar(ax, x, v, width, p[c])
        ax.text(x, v + ylim * 0.025, lab, ha="center", va="bottom", fontsize=10.5 if i in bold else 9.5,
                color=p["ink"], fontweight="bold" if i in bold else "normal")
    for xd in dividers:
        ax.axvline(xd, color=p["divider"], linestyle=(0, (3, 3)), linewidth=1, zorder=1)
    ax.set_xticks(xs, ticks)
    ax.tick_params(axis="x", colors=p["ink2"], length=0, labelsize=8.5, pad=6)
    ax.tick_params(axis="y", colors=p["ink2"], length=0, labelsize=8.5)
    ax.set_yticks([v for v, _ in yticks], [l for _, l in yticks])
    for x, lab in groups:
        ax.text(x, -0.2, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=11, color=p["ink"])
    mid = 0.61 if groups else 0.56
    fig.text(0.018, mid, ylabel, rotation=90, ha="center", va="center", fontsize=12,
             fontweight="bold", color=p["ink"])
    fig.text(0.045, mid, ysub, rotation=90, ha="center", va="center", fontsize=9, color=p["ink2"])
    return fig


def fig_retrieval(t):
    fail = [(1 - hit) * 100 for hit, _ in retrieval_rows()]
    return _bars(
        t, xs=[0, 1, 2.4, 3.4, 4.8, 5.8], values=fail,
        colors=["base", "base", "mid", "mid", "final", "final"],
        labels=[f"{v:.1f}%" for v in fail], bold={5},
        ticks=["Fixed chunks\n+ dense", "Parent-child\n+ dense", "Parent-child\n+ BM25", "Hybrid\n(RRF)",
               "+ rerank\n0.6B", "+ rerank\n8B"],
        ylim=80, yticks=[(0, ""), (20, "20%"), (40, "40%"), (60, "60%"), (80, "80%")],
        ylabel="FAILED %", ysub="Missed in top 5",
        groups=[(0.5, "Chunking"), (2.9, "Hybrid retrieval"), (5.3, "Reranking")], dividers=(1.7, 4.1))


def fig_refusal(t):
    # results/retrieval/rag_agentic_20261022.json, rows 4b* / 5 / 6 / 7
    refused = [6, 3, 5, 10]
    return _bars(
        t, xs=[0, 1.4, 2.4, 3.8], values=refused, colors=["base", "mid", "mid", "final"],
        labels=[f"{v} / 10" for v in refused], bold={3},
        ticks=["Hybrid + rerank", "+ query\nanalysis", "+ query analysis\n+ HyDE", "+ evidence\ngrading"],
        ylim=11, yticks=[(0, ""), (5, "5"), (10, "10")],
        ylabel="REFUSALS", ysub="Correct, out of 10",
        groups=[(0, "No agent"), (1.9, "Multi-query recall"), (3.8, "Full agentic")], dividers=(0.7, 3.1))


def fig_e2e(t):
    # results/e2e/: v1 = 2026-10-22_084500, hybrid = 2026-10-24_082937, agentic = 2026-10-24_082818
    vals = [9.2, 4.2, 2.4, 6.1, 1.6, 0.8]
    return _bars(
        t, xs=[0, 1, 2, 3.4, 4.4, 5.4], values=vals, colors=["base", "mid", "final"] * 2,
        labels=[f"{v:.1f}" for v in vals], bold={2, 5},
        ticks=["v1\n(toy corpus)", "Hybrid", "Agentic"] * 2,
        ylim=10.5, yticks=[(0, ""), (2, "2"), (4, "4"), (6, "6"), (8, "8"), (10, "10")],
        ylabel="CALLS", ysub="Per question, 14 questions",
        groups=[(1, "Tool calls"), (4.4, "Web calls")], dividers=(2.7,))


def fig_budget(t):
    # M6 calibration runs on the 5-case regression set (total cost of one run; pass 5/5 in all three)
    cost = [0.0990, 0.1389, 0.1179]
    return _bars(
        t, xs=[0, 1, 2], values=cost, colors=["base", "final", "mid"],
        labels=["¥0.099", "¥0.139  (+40%)", "¥0.118  (+19%)"], bold={1},
        ticks=["No run budget", "Budget below\nhealthy runs", "Budget calibrated\n(≈2× worst healthy)"],
        ylim=0.16, yticks=[(0, ""), (0.05, "¥0.05"), (0.10, "¥0.10"), (0.15, "¥0.15")],
        ylabel="COST", ysub="5-case regression run", h=3.3, width=0.6)


def main():
    plt.rcParams["font.family"] = FONT
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("retrieval-ablation", fig_retrieval), ("agentic-refusal", fig_refusal),
                     ("e2e-process", fig_e2e), ("budget-lesson", fig_budget)]:
        for mode, t in THEMES.items():
            fig = fn(t)
            fig.savefig(OUT / f"{name}-{mode}.png", facecolor=t["bg"])
            plt.close(fig)
    print("wrote", len(list(OUT.glob("*.png"))), "files to", OUT)


if __name__ == "__main__":
    main()
