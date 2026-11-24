# Pilot: detecting instruction deviations in multi-agent LLM systems

A small pilot for a planned 8-week project testing whether cheap monitors flag LLM agents that deviate
from their task or role instructions, and whether they behave differently for **accidental** deviations
(blurred or buried instructions) and **induced** deviations (an injected instruction from a compromised agent).

## Part 2 (core): semantic entropy at the worker step of a two-agent pipeline

A scripted manager agent delegates a constrained hotel choice to an LLM worker under three conditions:
clean, accidental (blurred or buried constraint) and induced (injected "priority override").

- **Two labels per step.** The injected *condition* is known by construction. *Compliance* of the
  executed action (the greedy answer) is decided by an independent verifier against the user's original
  constraints: an injection does not guarantee a violation, and the share of injections actually
  followed is reported.
- **Two scores per step.** The worker's next answer is resampled 10 times from the identical context and
  grouped by meaning (same hotel). *Semantic entropy* over the groups scores the decision point;
  the *action-level score* (1 - share of samples agreeing with the executed action) scores the action.
- **Ambiguity is designed in.** Each task has 1-3 compliant hotels, so "uncertain but compliant" steps exist.
- **Held-out evaluation.** Tasks are split into dev and test. Alarm thresholds are set on dev compliant
  steps at a 10% false-alarm rate; false-alarm and detection rates (with 95% Wilson intervals), AUROC
  (bootstrap 95% CI) and a four-way breakdown (uncertain/confident x compliant/noncompliant) are
  reported on test.

```bash
python part2_toy_semantic_entropy.py --backend mock                 # dry run, no GPU (numbers meaningless)
python part2_toy_semantic_entropy.py --backend hf --model Qwen/Qwen2.5-3B-Instruct
```

## Part 1: LLM-judge baseline on human-labelled multi-agent traces

Public traces from seven multi-agent frameworks, labelled with a failure taxonomy (MAST/MAD dataset).
Positives: traces labelled "disobey task specification" (1.1) or "disobey role specification" (1.2).
These labels are per trace, not per step, so this part calibrates the judge baseline only; it is
separate data from Part 2 and does not tune anything evaluated there.

```bash
python part1_mast_llm_judge.py --inspect            # check that trace text and labels were found
python part1_mast_llm_judge.py --model gpt-4o-mini --n 100
```

## Results

<!-- Fill in after running. Numbers from results_part2/summary.txt (test split) and results_part1/summary.txt. -->

| Monitor | Setting (held-out test) | False alarms | Detection: accidental | Detection: induced |
| --- | --- | --- | --- | --- |
| Semantic entropy (model) | two-agent pipeline, n = | | | |
| Action-level score (model) | two-agent pipeline, n = | | | |

Injections actually followed by the worker: __%. LLM judge on human-labelled traces (trace-level): AUROC __ [CI].

![Semantic entropy by condition](results_part2/se_by_condition.png)

**Takeaway:** _(three sentences: what was found, the numbers, the main limitation)_

## Limitations

Pilot scale: one small open model, a scripted manager, synthetic tasks with short answers (so meaning
clusters reduce to the chosen hotel), and wide confidence intervals. The full project replaces these
with a multi-agent testbed, NLI-based clustering for free-form messages, an NLI-based violation
detector, an LLM-judge monitor on the same steps, and 300+ compliant test steps.
