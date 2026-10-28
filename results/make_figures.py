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
    "light": {"chunk": "#C6C8BC", "hybrid": "#E9DCBC", "rerank": "#D49F7C", "axis": "#2b2a27", "divider": "#9a988f"},
    "dark": {"chunk": "#8E9186", "hybrid": "#B9A983", "rerank": "#C98B66", "axis": "#9198a1", "divider": "#5b5a55"},
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


def fig_retrieval(t):
    rows = retrieval_rows()
    fail = [(1 - hit) * 100 for hit, _ in rows]
    p = {**t, **STAGES["dark" if t["bg"] == "#0d1117" else "light"]}
    names = ["Fixed chunks\n+ dense", "Parent-child\n+ dense", "Parent-child\n+ BM25", "Hybrid\n(RRF)",
             "+ rerank\n0.6B", "+ rerank\n8B"]
    stage = ["chunk", "chunk", "hybrid", "hybrid", "rerank", "rerank"]
    xs = [0, 1, 2.4, 3.4, 4.8, 5.8]
    fig, ax = plt.subplots(figsize=(W, 3.7), dpi=200)
    fig.patch.set_facecolor(p["bg"])
    ax.set_facecolor(p["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(p["axis"])
    ax.spines["bottom"].set_color(p["axis"])
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.yaxis.grid(True, color=p["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, 6.4)
    ax.set_ylim(0, 80)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.24)
    fig.canvas.draw()
    for x, v, st in zip(xs, fail, stage):
        _rounded_bar(ax, x, v, 0.78, p[st])
        last = x == xs[-1]
        ax.text(x, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5 if last else 9.5,
                color=p["ink"], fontweight="bold" if last else "normal")
    for xd in (1.7, 4.1):
        ax.axvline(xd, color=p["divider"], linestyle=(0, (3, 3)), linewidth=1, zorder=1)
    ax.set_xticks(xs, names)
    ax.tick_params(axis="x", colors=p["ink2"], length=0, labelsize=8.5, pad=6)
    ax.tick_params(axis="y", colors=p["ink2"], length=0, labelsize=8.5)
    ax.set_yticks([0, 20, 40, 60, 80], ["", "20%", "40%", "60%", "80%"])
    for x, lab in [(0.5, "Chunking"), (2.9, "Hybrid retrieval"), (5.3, "Reranking")]:
        ax.text(x, -0.2, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=11, color=p["ink"])
    fig.text(0.018, 0.61, "FAILED %", rotation=90, ha="center", va="center", fontsize=12,
             fontweight="bold", color=p["ink"])
    fig.text(0.045, 0.61, "Missed in top 5", rotation=90, ha="center", va="center", fontsize=9,
             color=p["ink2"])
    return fig


def fig_refusal(t):
    # results/retrieval/rag_agentic_20261022.json, rows 4b* / 5 / 6 / 7
    names = ["Hybrid + rerank (no agent)", "+ query analysis", "+ query analysis + HyDE", "Full agentic (+ grading)"]
    refused = [6, 3, 5, 10]
    fig, ax = _fig(t, 2.1)
    size, gap = 0.62, 0.38
    for y, n in enumerate(refused):
        color = t["s1"] if y == len(refused) - 1 else t["ink2"]
        for k in range(10):
            ax.add_patch(Rectangle((k * (size + gap) * 0.1, y - size / 2), size * 0.1, size,
                                   color=color if k < n else t["empty"], linewidth=0))
        ax.text(1.03, y, f"{n} / 10", va="center", fontsize=9, color=t["ink"],
                fontweight="bold" if y == len(refused) - 1 else "normal")
    _ylabels(ax, t, names, bold_last=True)
    ax.set_xlim(-0.01, 1.14)
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.set_xticks([])
    fig.subplots_adjust(left=0.27, right=0.98, top=0.97, bottom=0.04)
    return fig


def fig_e2e(t):
    # results/e2e/: v1 = 2026-10-22_084500, hybrid = 2026-10-24_082937, agentic = 2026-10-24_082818
    modes = ["v1 (toy corpus)", "Hybrid", "Agentic"]
    series = [("Tool calls / question", [9.2, 4.2, 2.4], "s1"), ("Web calls / question", [6.1, 1.6, 0.8], "s2")]
    fig, ax = _fig(t, 2.9)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    handles = []
    for label, vals, key in series:
        (h,) = ax.plot(range(3), vals, color=t[key], linewidth=2, marker="o", markersize=6.5,
                       markeredgecolor=t["bg"], markeredgewidth=1.5, label=label, zorder=3)
        handles.append(h)
        for x, v in enumerate(vals):
            ax.text(x, v + 0.45, f"{v:.1f}", ha="center", fontsize=8.5, color=t["ink"],
                    fontweight="bold" if x == 2 else "normal")
    ax.set_xticks(range(3), modes)
    for lab in ax.get_xticklabels():
        lab.set_color(t["ink2"])
        lab.set_fontsize(9)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0, 10.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    _legend(ax, t, handles)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.12)
    return fig


def fig_budget(t):
    # M6 calibration runs on the 5-case regression set (total cost of one run; pass 5/5 in all three)
    names = ["No run budget (baseline)", "Budget below healthy runs", "Budget calibrated (≈2× worst healthy)"]
    cost = [0.0990, 0.1389, 0.1179]
    colors = [t["dim"], t["s2"], t["s1"]]
    fig, ax = _fig(t, 1.9)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.barh(range(3), cost, height=0.42, color=colors, zorder=2)
    for y, v in enumerate(cost):
        delta = "" if y == 0 else f"  (+{(v / cost[0] - 1) * 100:.0f}%)"
        ax.text(v + 0.002, y, f"¥{v:.4f}{delta}", va="center", fontsize=8.5, color=t["ink"],
                fontweight="bold" if y == 1 else "normal")
    _ylabels(ax, t, names)
    ax.set_xlim(0, 0.175)
    ax.set_xticks([0, 0.05, 0.10, 0.15], ["¥0", "¥0.05", "¥0.10", "¥0.15"])
    fig.subplots_adjust(left=0.31, right=0.98, top=0.97, bottom=0.14)
    return fig


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
