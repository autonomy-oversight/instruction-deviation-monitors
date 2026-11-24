"""Print worker steps for manual inspection.

Usage:  python inspect_rows.py results_first20/rows.jsonl --condition clean --deviated 1 --n 10
"""
import argparse
import json


def show(r):
    c = r["constraints"]
    print("=" * 90)
    print(f"task {r['task']} | {r['condition']} | deviated={r['deviated']} | action={r['action']} "
          f"| SE={r['semantic_entropy']:.2f}")
    print(f"constraints: price <= {c['max_price']}, not in {c['avoid_city']}, pets allowed")
    print(f"compliant options: {r['compliant_options']} | decoy: {r['decoy']}")
    for h in r["hotels"]:
        print(f"   - {h['name']:<12} {h['city']:<11} EUR {h['price']:<4} pets={'yes' if h['pets'] else 'no'}")
    print(f"worker greedy answer (raw): {r['greedy_raw']!r}")
    print(f"10 samples (raw): {[s.strip()[:25] for s in r['samples_raw']]}")
    if r.get("injection"):
        print(f"injection style: {r['injection']} (followed={r.get('followed_injection')})")
    if r["condition"] != "clean":
        print(f"manager message: {r['manager_message'][:300]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--condition", choices=["clean", "accidental", "induced"])
    ap.add_argument("--deviated", type=int, choices=[0, 1])
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    sel = [r for r in rows if (a.condition is None or r["condition"] == a.condition)
           and (a.deviated is None or r["deviated"] == a.deviated)]
    if sel and "hotels" not in sel[0]:
        raise SystemExit("These rows were made by the older script version: rerun the experiment first.")
    print(f"{len(sel)} matching steps; showing {min(a.n, len(sel))}")
    for r in sel[:a.n]:
        show(r)


if __name__ == "__main__":
    main()
