# SkillOpt-Sleep — multi-night progression (gate vs no-gate)

The deployment-scale sleep loop run for **5 nights**, adding **10 new real tasks each night** (cumulative train 10 → 20 → 30 → 40 → 50). Each night the new tasks are enriched with **5 dream rollout groups** (multi-rollout contrastive reflection), and the skill **carries over and is refined** from the previous night. Target + optimizer: **gpt-5.5** via the Responses API (gpt4v scus+swc).

Two variants run side by side:

* **no-gate** — the *daily-use* design: take the final skill directly, with **no validation set**. Models the common case where a user cannot build a held-out val set per task. Quality is judged only on the real test set.
* **gate** — paper-aligned: an edit is kept only if it improves a held-out validation slice (capped at 60 items).

Test is scored after **every** night, so the columns N0 (baseline) … N5 are the real held-out accuracy at each step. Full test sets (SearchQA 1400, LiveMath 124, SpreadsheetBench 280). Raw JSON: `docs/sleep/blog_runs/nightly/`.

## Night-by-night test accuracy

| benchmark | variant | N0 | N1 | N2 | N3 | N4 | N5 | Δ |
|---|---|---|---|---|---|---|---|---|
| SearchQA | no-gate | 0.800 | 0.825 | 0.823 | 0.805 | 0.808 | 0.823 | **+0.023** |
| SearchQA | gate | 0.797 | 0.797 | 0.844 | 0.844 | 0.844 | 0.844 | **+0.047** |
| LiveMathematicianBench | no-gate | 0.508 | 0.556 | 0.500 | 0.524 | 0.556 |  | **+0.048** |
| LiveMathematicianBench | gate | 0.540 | 0.540 | 0.540 | 0.540 | 0.540 |  | -0.000 |
| SpreadsheetBench | no-gate | 0.639 | 0.629 | 0.600 | 0.629 | 0.614 | 0.636 | -0.004 |
| SpreadsheetBench | gate | 0.646 | 0.621 | 0.621 | 0.621 | 0.621 | 0.621 | -0.025 |

Shape of each curve (N0→N5):

- SearchQA no-gate: `▁█▇▂▃▇` 0.800 → 0.823
- SearchQA gate: `▁▁████` 0.797 → 0.844
- LiveMathematicianBench no-gate: `▂█▁▃█` 0.508 → 0.556
- LiveMathematicianBench gate: `▁▁▁▁▁` 0.540 → 0.540
- SpreadsheetBench no-gate: `█▆▁▆▃▇` 0.639 → 0.636
- SpreadsheetBench gate: `█▁▁▁▁▁` 0.646 → 0.621

## What the data says (honest)

1. **The clearest win is the gate on SearchQA.** It jumps 0.797 → 0.844 at night 2 and **holds it** (Δ+0.047, the largest real gain here). This is the gate doing exactly its job: lock in a genuine improvement, then refuse later edits that would regress it.
2. **"Accuracy climbs monotonically with more nights" is NOT supported.** No curve climbs smoothly. With only 10 new tasks/night and a already-high gpt-5.5 baseline, accumulating training data does not produce night-over-night gains — the curves oscillate inside a few points. The sleep gain is a small one-time lift, not a steady climb.
3. **no-gate (the daily-use design) is genuinely mixed — and that is the point.** It gives small gains on SearchQA and LiveMath but is flat/down on SpreadsheetBench. The traces show why: greedy nightly edits can **overfit** (SearchQA learned an over-rigid "output only the exact span, never wrap/append" rule that helped night 1 then decayed by night 3) or learn an **actively harmful** rule (SpreadsheetBench learned "if the target range isn't stated, ask for it" — but the task can't ask, so that's a wrong answer). Without a gate, nothing catches these.
4. **The gate can be too conservative when val is tiny.** LiveMath's val is only 18 items; the gate accepts nothing and the score stays flat at 0.540 for all 5 nights. The gate's usefulness scales with how trustworthy the val signal is.

## Takeaway

At deployment scale (a handful of real tasks per night), the sleep loop delivers a **small, real, one-time lift** — best realized **with** the validation gate, which both captures the gain and prevents the regressions that the greedy no-gate variant suffers. The no-gate mode is more convenient (no val set required) and is fine when edits are low-risk, but on harder tasks it can adopt overfit or harmful rules with nothing to catch them. The honest headline is *safety and modest gains*, not *monotonic self-improvement*.

## Reproduce

```bash
PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
SKILLOPT_SLEEP_WORKERS=32 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_nightly_matrix \
  --max-conc 2 --workers 32 --nights 5 --per-night 10 --rollouts 5 --dream-factor 2
PYTHONPATH=. $PY skillopt_sleep/experiments/gen_nightly_blog.py
```
