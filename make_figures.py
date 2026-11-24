"""Make the two write-up figures from a results folder.

Usage:  python make_figures.py results_main
Writes  results_main/fig_detection.png  and  results_main/fig_confidence.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part2_toy_semantic_entropy import calibrate, wilson

GREY, ORANGE, RED = "#8c8c8c", "#e08a1e", "#c0392b"


def main(folder, max_fpr=0.10):
    folder = Path(folder)
    rows = [json.loads(l) for l in open(folder / "rows.jsonl") if l.strip()]
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    thr = calibrate([r["semantic_entropy"] for r in dev if not r["deviated"]], max_fpr)

    # Figure 1: false alarms vs detection rates on held-out tasks, with 95% Wilson intervals
    groups = [("False alarms\n(compliant steps)", [r for r in test if not r["deviated"]], GREY),
              ("Accidental\ndeviations", [r for r in test if r["deviated"] and r["condition"] == "accidental"], ORANGE),
              ("Deviations under\ninjected instruction",
               [r for r in test if r["deviated"] and r["condition"] == "induced"], RED)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (label, sub, colour) in enumerate(groups):
        k, n = sum(r["semantic_entropy"] >= thr for r in sub), len(sub)
        p = k / n if n else 0
        lo, hi = wilson(k, n)
        ax.bar(i, p, color=colour, width=0.6)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="black", capsize=5, lw=1)
        ax.text(i, hi + 0.03, f"{p:.0%}\n({k}/{n})", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("share of steps flagged")
    ax.set_title("Semantic entropy flags accidental deviations, not injected ones\n"
                 f"(held-out tasks, threshold set for {max_fpr:.0%} false alarms on dev tasks)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / "fig_detection.png", dpi=200)

    # Figure 2: how often the worker is fully confident (all samples agree, entropy = 0)
    cats = [("clean", 0), ("clean", 1), ("accidental", 0), ("accidental", 1), ("induced", 0), ("induced", 1)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (cond, d) in enumerate(cats):
        sub = [r for r in rows if r["condition"] == cond and r["deviated"] == d]
        share = np.mean([r["semantic_entropy"] == 0 for r in sub]) if sub else 0
        colour = GREY if d == 0 else (RED if cond == "induced" else ORANGE)
        ax.bar(i, share, color=colour, width=0.6, alpha=0.55 if d == 0 else 1)
        ax.text(i, share + 0.02, f"{share:.0%}\nn={len(sub)}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(6), [f"{c}\n{'deviated' if d else 'compliant'}" for c, d in cats], fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("share of steps with zero entropy\n(all 10 samples agree)")
    ax.set_title("Deviations caused by an injected instruction are made confidently", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / "fig_confidence.png", dpi=200)
    print(f"threshold: entropy >= {thr:.3f}; figures written to {folder}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_part2")
