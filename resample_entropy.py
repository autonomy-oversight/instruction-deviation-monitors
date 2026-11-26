"""Recompute semantic entropy for a finished run, sampling from the full distribution.

The original runs sampled with the worker model's default cutoffs (its generation_config, e.g. top-p
and top-k), which drop unlikely answers and can push semantic entropy towards zero. This script draws
new samples for every saved step with --top-p 1.0 --top-k 0 (no cutoff) at the same temperature.
The executed (greedy) action, the labels and the judge scores do not depend on sampling, so they are
unchanged; only the uncertainty score is recomputed.

Usage:
  python resample_entropy.py $D/results_main --model Qwen/Qwen2.5-7B-Instruct --load-4bit
Writes entropy_full.jsonl (resumable) and entropy_full_summary.txt into the folder.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import report
from part2_toy_semantic_entropy import HFBackend, MockBackend, semantic_entropy, to_cluster, worker_prompt

CONDS = ["clean", "accidental", "induced"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--backend", choices=["hf", "mock"], default="mock")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()

    folder = Path(args.folder)
    rows = [json.loads(l) for l in open(folder / "rows.jsonl") if l.strip()]
    out = folder / "entropy_full.jsonl"
    done = {}
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                done[(d["task"], d["condition"])] = d["semantic_entropy_full"]
    backend = None
    with out.open("a") as f:
        for i, r in enumerate(rows):
            k = (r["task"], r["condition"])
            if k not in done:
                if backend is None:
                    backend = HFBackend(args.model, args.load_4bit) if args.backend == "hf" else MockBackend()
                inject_on = r["decoy"] if r.get("injection") == "data" else None
                system, user = worker_prompt({"hotels": r["hotels"]}, r["manager_message"], inject_on)
                _, samples = backend.generate(system, user, args.samples, args.temperature,
                                              args.top_p, args.top_k, greedy_too=False)
                names = [h["name"] for h in r["hotels"]]
                se = semantic_entropy([to_cluster(s, names) for s in samples])
                done[k] = se
                f.write(json.dumps({"task": r["task"], "condition": r["condition"], "semantic_entropy_full": se,
                                    "samples_full": samples}) + "\n")
                f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} steps resampled", flush=True)
    for r in rows:
        r["semantic_entropy_full"] = done[(r["task"], r["condition"])]

    test = [r for r in rows if r["split"] == "test"]
    ok = [r for r in test if not r["deviated"]]
    L = [f"Semantic entropy, default vs full-distribution sampling ({folder})",
         f"Full sampling: temperature {args.temperature}, top_p {args.top_p}, top_k {args.top_k} (0 = off)", "",
         "Share of steps with zero entropy (all steps):"]
    for key, name in [("semantic_entropy", "default"), ("semantic_entropy_full", "full")]:
        L.append(f"  {name:<8} " + ", ".join(f"{c} {np.mean([r[key] == 0 for r in rows if r['condition'] == c]):.0%}"
                                           for c in CONDS))
    L += ["", "AUROC on held-out tasks (deviations of one type vs all compliant steps):"]
    for c in ["accidental", "induced"]:
        sub = ok + [r for r in test if r["deviated"] and r["condition"] == c]
        y = [r["deviated"] for r in sub]
        for key, name in [("semantic_entropy", "default"), ("semantic_entropy_full", "full")]:
            L.append("  " + report(f"{c}, {name} sampling", y, [r[key] for r in sub], show_tpr=False))
    text = "\n".join(L)
    (folder / "entropy_full_summary.txt").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
