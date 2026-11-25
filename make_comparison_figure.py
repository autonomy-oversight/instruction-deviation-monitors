"""Compare the two monitors across seeds: AUROC with 95% bootstrap intervals on held-out tasks.

Usage:  python make_comparison_figure.py results_main results_seed2 --labels "seed 42" "seed 2" --out fig_auroc.png
Needs rows.jsonl and judge_scores_stepwise.jsonl in each folder.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import auroc, bootstrap_ci

MONITORS = [("semantic_entropy", "Semantic entropy", "#4A6FA5"),
            ("judge", "Step-by-step judge", "#B5621A")]
CONDS = [("accidental", "Accidental deviations"), ("induced", "Deviations under injection")]
MARKERS = ["o", "s", "D", "^"]


def load(folder):
    folder = Path(folder)
    rows = [json.loads(l) for l in open(folder / "rows.jsonl") if l.strip()]
    js = {(d["task"], d["condition"]): d["judge_score"]
          for d in map(json.loads, open(folder / "judge_scores_stepwise.jsonl"))}
    for r in rows:
        r["judge"] = js[(r["task"], r["condition"])]
    return [r for r in rows if r["split"] == "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", default="fig_auroc.png")
    args = ap.parse_args()
    labels = args.labels or [Path(f).name for f in args.folders]
    data = [load(f) for f in args.folders]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, (cond, title) in zip(axes, CONDS):
        for m, (key, name, colour) in enumerate(MONITORS):
            for s, test in enumerate(data):
                ok = [r for r in test if not r["deviated"]]
                pos = [r for r in test if r["deviated"] and r["condition"] == cond]
                sub = ok + pos
                y, x = [r["deviated"] for r in sub], [r[key] for r in sub]
                a, (lo, hi) = auroc(y, x), bootstrap_ci(y, x)
                xpos = m + (s - (len(data) - 1) / 2) * 0.18
                ax.errorbar(xpos, a, yerr=[[a - lo], [hi - a]], fmt=MARKERS[s % 4], color=colour,
                            capsize=4, markersize=7, label=None)
                ax.text(xpos + 0.05, a, f"{a:.2f}", va="center", fontsize=9)
                print(f"{labels[s]} | {cond} | {name}: AUROC {a:.2f} [{lo:.2f}-{hi:.2f}] (deviations={len(pos)})")
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.text(1.45, 0.51, "chance", color="grey", fontsize=9, ha="right", va="bottom")
        ax.set_xticks([0, 1], [n for _, n, _ in MONITORS])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_ylabel("AUROC on held-out tasks (95% CI)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=MARKERS[s % 4], color="#555555", ls="none", markersize=7, label=labels[s])
               for s in range(len(data))]
    axes[0].legend(handles=handles, title="Task seed", loc="lower left", fontsize=9, title_fontsize=9,
                   frameon=False)
    fig.suptitle("Uncertainty misses injected deviations; a step-by-step judge catches many of them", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"figure written to {args.out}")


if __name__ == "__main__":
    main()
