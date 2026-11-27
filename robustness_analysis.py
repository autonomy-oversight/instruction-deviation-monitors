"""Robustness checks: task-clustered intervals and condition-specific AUROC (not preregistered).

Two concerns with the main analysis:
1. Each task contributes three steps (clean, accidental, induced) built on the same hotel list, so
   steps are not independent. Here every interval comes from a bootstrap that resamples whole tasks
   (clusters) instead of single steps; the step-level interval is printed alongside for comparison.
2. The main AUROC uses compliant steps from ALL conditions as negatives. If compliant steps of one
   condition (e.g. accidental, with ambiguous instructions) are unusually uncertain, deviations of
   another condition can look confident by comparison. Here deviations are also compared with
   compliant steps of the clean condition only, and of their own condition only.

Usage:
  python robustness_analysis.py $D/results_main $D/results_seed2 $D/results_seed3 [--entropy full]
  python robustness_analysis.py $D/results_llama_seed42 $D/results_llama_seed2 $D/results_llama_seed3
Pools the held-out steps of all folders (--split all to use every step). Writes robustness_<first>.txt
into the parent directory of the first folder.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOT = 2000


def load(folder, entropy):
    folder = Path(folder)
    rows = [json.loads(l) for l in open(folder / "rows.jsonl") if l.strip()]
    js = {(d["task"], d["condition"]): d["judge_score"]
          for d in map(json.loads, (l for l in open(folder / "judge_scores_stepwise.jsonl") if l.strip()))}
    full = {}
    if entropy == "full":
        full = {(d["task"], d["condition"]): d["semantic_entropy_full"]
                for d in map(json.loads, (l for l in open(folder / "entropy_full.jsonl") if l.strip()))}
    for r in rows:
        r["judge"] = js[(r["task"], r["condition"])]
        r["entropy"] = full[(r["task"], r["condition"])] if entropy == "full" else r["semantic_entropy"]
        r["cluster"] = f"{folder.name}/{r['task']}"
        r["kind"] = r["condition"]
        if r["condition"] == "accidental":
            r["kind"] = "blurred" if "not too expensive" in r["manager_message"] else "buried"
    return rows


def auc(y, s):
    return roc_auc_score(y, s) if 0 < sum(y) < len(y) else np.nan


def contrast(pos, neg, keys, rng):
    """AUROC of each score for pos vs neg, with step-level and task-clustered 95% bootstrap intervals,
    plus the clustered interval of the difference judge - entropy."""
    sub = [(r, 1) for r in pos] + [(r, 0) for r in neg]
    y = np.array([t for _, t in sub])
    S = {k: np.array([r[k] for r, _ in sub]) for k in keys}
    point = {k: auc(y, S[k]) for k in keys}
    # step-level bootstrap
    step = {k: [] for k in keys}
    idx = np.arange(len(sub))
    for _ in range(N_BOOT):
        b = rng.choice(idx, len(idx))
        for k in keys:
            step[k].append(auc(y[b], S[k][b]))
    # task-clustered bootstrap
    by_cluster = defaultdict(list)
    for i, (r, _) in enumerate(sub):
        by_cluster[r["cluster"]].append(i)
    clusters = list(by_cluster)
    clus = {k: [] for k in keys}
    diff = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(clusters), len(clusters))
        b = np.concatenate([by_cluster[clusters[c]] for c in pick])
        vals = {k: auc(y[b], S[k][b]) for k in keys}
        for k in keys:
            clus[k].append(vals[k])
        diff.append(vals["judge"] - vals["entropy"])
    ci = lambda v: np.nanpercentile(v, [2.5, 97.5])
    return point, {k: ci(step[k]) for k in keys}, {k: ci(clus[k]) for k in keys}, ci(diff), len(pos), len(neg)


def describe(rows, label):
    e = np.array([r["entropy"] for r in rows])
    return (f"  {label:<34} n={len(rows):>4} | mean entropy {e.mean():.3f} | zero entropy {np.mean(e == 0):.0%}"
            if rows else f"  {label:<34} n=   0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--entropy", choices=["default", "full"], default="default")
    ap.add_argument("--split", choices=["test", "all"], default="test")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = [r for f in args.folders for r in load(f, args.entropy)]
    if args.split == "test":
        rows = [r for r in rows if r["split"] == "test"]
    ok = lambda c=None: [r for r in rows if not r["deviated"] and (c is None or r["condition"] == c)]
    dev = lambda k: [r for r in rows if r["deviated"] and r["kind"] == k]
    acc_dev = [r for r in rows if r["deviated"] and r["condition"] == "accidental"]

    L = [f"Robustness analysis (NOT preregistered): {', '.join(Path(f).name for f in args.folders)}",
         f"Steps: {args.split}; entropy sampling: {args.entropy}; {N_BOOT} bootstrap resamples; "
         f"{len({r['cluster'] for r in rows})} tasks (clusters)", "",
         "Entropy by group (descriptive):"]
    for label, grp in [("compliant, clean", ok("clean")), ("compliant, accidental", ok("accidental")),
                       ("compliant, induced", ok("induced")), ("deviated, clean", dev("clean")),
                       ("deviated, accidental: buried", dev("buried")),
                       ("deviated, accidental: blurred", dev("blurred")), ("deviated, induced", dev("induced"))]:
        L.append(describe(grp, label))

    contrasts = [
        ("Under injection vs all compliant (main analysis)", dev("induced"), ok()),
        ("Under injection vs clean compliant", dev("induced"), ok("clean")),
        ("Under injection vs compliant under injection", dev("induced"), ok("induced")),
        ("Accidental vs all compliant (main analysis)", acc_dev, ok()),
        ("Accidental vs clean compliant", acc_dev, ok("clean")),
        ("Accidental vs compliant accidental", acc_dev, ok("accidental")),
        ("Buried requirements vs clean compliant", dev("buried"), ok("clean")),
        ("Blurred price vs clean compliant", dev("blurred"), ok("clean")),
    ]
    for title, pos, neg in contrasts:
        L += ["", f"{title}:"]
        if not pos or not neg:
            L.append("  not enough steps")
            continue
        point, step, clus, d, npos, nneg = contrast(pos, neg, ["entropy", "judge"], rng)
        L.append(f"  deviations={npos}, negatives={nneg}" + ("  (few negatives: indicative only)" if nneg < 30 else ""))
        for k, name in [("entropy", "semantic entropy"), ("judge", "step-by-step judge")]:
            L.append(f"  {name:<20} AUROC {point[k]:.2f} | task-clustered 95% CI {clus[k][0]:.2f}-{clus[k][1]:.2f}"
                     f" | step-level {step[k][0]:.2f}-{step[k][1]:.2f}")
        L.append(f"  judge - entropy      {point['judge'] - point['entropy']:+.2f} | task-clustered 95% CI "
                 f"{d[0]:+.2f} to {d[1]:+.2f}")

    text = "\n".join(L)
    suffix = "" if args.entropy == "default" else "_full"
    split = "" if args.split == "test" else "_all"
    out = Path(args.folders[0]).parent / f"robustness_{Path(args.folders[0]).name}{suffix}{split}.txt"
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
