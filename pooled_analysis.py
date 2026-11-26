"""Pooled analysis over several seeds, as named in Addendum 1 of PREREGISTRATION.md.

Part 1, held-out steps of all seeds together: AUROC of semantic entropy and of the step-by-step judge
for accidental deviations and for deviations under injection.
Part 2, all steps of all seeds together (more cases): AUROC per kind of deviation, splitting accidental
deviations into buried requirements and blurred price limit.
Negatives are always the compliant steps of the same pooled set (all conditions).

Usage:
  python pooled_analysis.py $D/results_llama_seed42 $D/results_llama_seed2 $D/results_llama_seed3
  python pooled_analysis.py $D/results_main $D/results_seed2 $D/results_seed3 --entropy full
Writes pooled_summary[_full].txt next to the first folder (in its parent directory).
"""
import argparse
import json
from pathlib import Path

from metrics import report


def kind(r):
    if r["condition"] == "accidental":
        return "accidental: blurred price" if "not too expensive" in r["manager_message"] \
            else "accidental: buried requirements"
    return r["condition"]


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
    return rows


def block(rows, groups, title):
    ok = [r for r in rows if not r["deviated"]]
    L = [title]
    for g in groups:
        pos = [r for r in rows if r["deviated"] and kind(r).startswith(g)]
        if not pos:
            L.append(f"  {g}: no deviations")
            continue
        sub = ok + pos
        y = [r["deviated"] for r in sub]
        for key, name in [("entropy", "semantic entropy"), ("judge", "step-by-step judge")]:
            L.append("  " + report(f"{g} | {name}", y, [r[key] for r in sub], show_tpr=False))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--entropy", choices=["default", "full"], default="default",
                    help="full: use entropy_full.jsonl from resample_entropy.py")
    args = ap.parse_args()
    rows = [r for f in args.folders for r in load(f, args.entropy)]
    test = [r for r in rows if r["split"] == "test"]
    L = [f"Pooled over: {', '.join(Path(f).name for f in args.folders)} (entropy sampling: {args.entropy})", ""]
    L += block(test, ["accidental", "induced"], "Part 1. Held-out steps, pooled:") + [""]
    L += block(rows, ["accidental: buried requirements", "accidental: blurred price", "induced"],
               "Part 2. All steps, pooled, accidental deviations split by kind:")
    text = "\n".join(L)
    suffix = "" if args.entropy == "default" else "_full"
    out = Path(args.folders[0]).parent / f"pooled_summary_{Path(args.folders[0]).name}{suffix}.txt"
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
