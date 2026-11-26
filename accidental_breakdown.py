"""Exploratory: split accidental deviations by how the instruction was made unclear.

Half the accidental tasks blur the price limit ("not too expensive"); the other half bury clear
requirements in a long, irrelevant message. The two may behave differently: a blurred limit invites
a confident but different reading, while buried requirements may be missed with more hesitation.

Usage:  python accidental_breakdown.py $D/results_main $D/results_seed2 $D/results_seed3
Uses all steps (dev and test) to have enough cases; labeled exploratory.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import auroc


def variant(r):
    return "blurred price" if "not too expensive" in r["manager_message"] else "buried requirements"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    args = ap.parse_args()
    pooled = []
    for f in args.folders:
        f = Path(f)
        rows = [json.loads(l) for l in open(f / "rows.jsonl") if l.strip()]
        js = {}
        p = f / "judge_scores_stepwise.jsonl"
        if p.exists():
            js = {(d["task"], d["condition"]): d["judge_score"] for d in map(json.loads, open(p)) if d}
        for r in rows:
            r["judge"] = js.get((r["task"], r["condition"]))
            r["seed_folder"] = f.name
        pooled += rows
    for name, rows in [(Path(f).name, [r for r in pooled if r["seed_folder"] == Path(f).name])
                       for f in args.folders] + [("pooled", pooled)]:
        ok = [r for r in rows if not r["deviated"]]
        print(f"== {name} (all steps, EXPLORATORY)")
        for v in ["blurred price", "buried requirements"]:
            acc = [r for r in rows if r["condition"] == "accidental" and variant(r) == v]
            dev = [r for r in acc if r["deviated"]]
            line = (f"  {v:<20} steps {len(acc):>3} | deviated {len(dev):>3} ({np.mean([r['deviated'] for r in acc]):.0%})"
                    f" | deviated with zero entropy: {np.mean([r['semantic_entropy'] == 0 for r in dev]):.0%}"
                    if dev else f"  {v:<20} steps {len(acc):>3} | no deviations")
            if dev:
                sub = ok + dev
                y = [r["deviated"] for r in sub]
                line += f" | AUROC entropy {auroc(y, [r['semantic_entropy'] for r in sub]):.2f}"
                if all(r["judge"] is not None for r in sub):
                    line += f", judge {auroc(y, [r['judge'] for r in sub]):.2f}"
            print(line)


if __name__ == "__main__":
    main()
