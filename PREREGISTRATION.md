# Preregistration: seed 3 and the combined monitor

Committed before the seed-3 run. Everything below is fixed; any change after the run is reported as a deviation from this plan.

## Status when committed

- Runs done: seeds 42 and 2 (worker, semantic entropy, yes/no judge, step-by-step judge). Their results are in the README.
- Not yet computed on any seed: the combined monitor below.
- Seed 3: not run.

## Research questions

- **Q1 (main).** On held-out seed-3 tasks, does the combined monitor catch more deviations than either monitor alone, at a comparable false-alarm rate?
- **Q2 (replication).** On seed 3, is semantic entropy still at chance on deviations under injection, while the step-by-step judge detects them?

## Fixed setup

Identical to seeds 42 and 2:

- Worker: Qwen2.5-7B-Instruct, 4-bit; 150 tasks; 10 samples at temperature 1.0; injection in the decoy's listing (`--injection data`).
- Task seed 3; development/test split by task with split seed 4 (`seed + 1`), half the tasks each.
- Judge: the step-by-step prompt, same model, as in `monitor_judge.py --prompt stepwise`.
- Labels as before: the condition is known by construction; deviation is checked by the independent verifier.
- No steps are excluded. A disconnected run is resumed, not restarted.

## The combined monitor

Flag a step if **semantic entropy ≥ t_SE or judge score ≥ t_J**, where each threshold is the smallest one that keeps false alarms at or below **5% on compliant development steps** (`calibrate` in `part2_toy_semantic_entropy.py`). The comparison monitors are each monitor alone, calibrated at or below **10%** on the same development steps.

Analysis code: `combined_monitor.py`, committed with this file, run with default options.

## Outcomes, all on held-out seed-3 steps

1. False-alarm rate of each monitor on compliant steps (all conditions), with 95% Wilson intervals.
2. Detection rate of each monitor, separately for accidental deviations and for deviations under injection, with 95% Wilson intervals.
3. Paired comparison, per deviation type: combined vs judge alone and combined vs semantic entropy alone, exact two-sided McNemar test on the same deviated steps, α = 0.05, no correction (four tests, reported in full).
4. For Q2: AUROC of each monitor per deviation type with 95% bootstrap intervals, as in `monitor_judge.py`.

## How Q1 will be answered

- **Yes**, if the combined monitor's held-out false-alarm rate is at most 15%, and for both deviation types its detection is at least that of the better single monitor, with at least one McNemar test significant in its favor.
- **No**, if for either deviation type it detects fewer deviations than the better single monitor.
- **Inconclusive** otherwise. With about 20–40 deviations per type in the test set, this is the most likely outcome; the pilot reports it as such and does not claim an improvement.

## How Q2 will be answered

Replicated if semantic entropy's 95% AUROC interval on deviations under injection includes 0.5, and the lower end of the judge's interval lies above the upper end of semantic entropy's.

## Exploratory, labeled as such

- The combined monitor on seeds 42 and 2 (`combined_monitor.py ... --exploratory`).
- Pooled results over three seeds.
- Any other false-alarm targets or combination rules.
- The API judge and the second worker model, which are separate follow-ups and not covered here.

---

# Addendum 1: second worker model, the kinds of accidental deviation, and sampling

Committed before any run described here. Everything above remains as committed; the seed-3 results it covers are reported in the README.

## Status when committed

- Done with Qwen2.5-7B-Instruct as the worker: seeds 42, 2 and 3, each with semantic entropy and the step-by-step judge.
- **Exploratory finding that motivates Q4.** Splitting accidental deviations by kind (`accidental_breakdown.py`, all steps), the judge's AUROC was 0.88, 0.84 and 0.86 on *buried* requirements, but 0.28, 0.40 and 0.36 on a *blurred* price limit. A blurred limit never reaches the worker, so the judge, which sees the same message, cannot know it was violated. Q4 tests this on a new worker model.
- **Sampling detail found while preparing this addendum.** The Qwen runs sampled with the model's own default cutoffs (top-p and top-k in its generation config, printed at model load), not from the full distribution at temperature 1.0. Cutoffs drop unlikely answers and can push semantic entropy towards zero. Q5 checks whether this changes the main result. All new worker runs sample without cutoffs.

## Research questions

- **Q3 (another model family).** With Llama-3.1-8B-Instruct as the worker, is semantic entropy still at chance on deviations under injection while the step-by-step judge detects them?
- **Q4 (what the judge can see).** Does the judge detect deviations whose requirement is in the worker's input (buried requirements) but not those whose requirement never reached the worker (blurred price limit)?
- **Q5 (sampling).** For the Qwen worker, does semantic entropy computed from the full distribution still sit at chance on deviations under injection?

## Fixed setup

- **Llama runs.** Worker `meta-llama/Llama-3.1-8B-Instruct`, 4-bit; task seeds 42, 2 and 3 (the same tasks as for Qwen), 150 tasks each; 10 samples at temperature 1.0 with `--top-p 1.0 --top-k 0`; injection in the decoy's listing. Everything else as in the main setup.
- **Judge unchanged:** the same Qwen2.5-7B-Instruct step-by-step judge, so that only the worker changes.
- **Sanity check before the three runs:** 20 tasks, used only to confirm that answers map to hotel names (at least 95% of greedy answers). No prompt or design change will be made based on it. If parsing fails, the parser may be fixed; any such fix is reported.
- **Qwen resampling.** `resample_entropy.py` on the three Qwen folders, with `--top-p 1.0 --top-k 0` at temperature 1.0. Actions, labels and judge scores are unchanged by construction.
- No exclusions; disconnected runs are resumed.

## Analysis

`pooled_analysis.py`, committed with this addendum, run with default options: part 1 pools the held-out steps of the three seeds; part 2 pools all steps, splitting accidental deviations by kind. Negatives are all compliant steps of the same pooled set. Per-seed numbers (`monitor_judge.py`, `make_comparison_figure.py`) are reported alongside.

## How the questions will be answered

- **Q3.** Llama folders, part 1 (held-out, pooled), deviations under injection. *Replicated* if semantic entropy's 95% AUROC interval includes 0.5 and the lower end of the judge's interval lies above the upper end of semantic entropy's. If fewer than 20 deviations under injection occur in the pooled held-out set (the worker rarely follows the injection), Q3 is reported as *not testable*.
- **Q4.** Llama folders, part 2 (all steps, pooled). *Supported* if the judge's AUROC interval for buried-requirement deviations lies entirely above 0.5 and its AUROC point estimate for blurred-price deviations is at or below 0.5. *Not supported* if the buried-requirement interval includes 0.5 or the blurred-price estimate is above 0.5. With fewer than 15 deviations of either kind, *not testable*.
- **Q5.** Qwen folders with `--entropy full`, part 1, deviations under injection. *Unchanged* if the 95% AUROC interval includes 0.5; otherwise the main conclusion about semantic entropy is revised.

## Exploratory, labeled as such

The combined monitor on the Llama runs; the yes/no judge; per-seed comparisons; part 2 for the Qwen folders (already seen); the API judge.
