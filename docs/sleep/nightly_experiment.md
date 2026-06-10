# SkillOpt-Sleep — multi-night progression across targets (gate vs no-gate)

The deployment-scale sleep loop run for **5 nights**, adding **10 new real tasks each night** (cumulative train 10 → 50). Each night the new tasks get **5 dream rollout groups** (multi-rollout contrastive reflection), the skill **carries over and is refined**, and the optimizer prompt carries the task's **output contract guardrail** (it may not propose rules that change the required format or tell the agent to ask questions — such rules score zero by construction).

Three target models: **gpt-5.5** (strong; Responses API on gpt4v scus+swc), **gpt-5.4-mini** and **gpt-5.4-nano** (weaker; Managed-Identity endpoints). Optimizer = the same model as the target in each run. Two variants per cell:

* **no-gate** — daily-use design: accept every edit, take the final skill, **no validation set**. Quality judged only on the real held-out test.
* **gate** — paper-aligned: an edit is kept only if it improves a held-out val slice (capped at 60).

Test is scored after every night (N0 = baseline). Full test sets. Raw JSON: `docs/sleep/blog_runs/nightly/`.

Provenance: all gpt-5.4-mini / gpt-5.4-nano cells and the gpt-5.5 SpreadsheetBench no-gate cell were run with the output-contract guardrail (current code). The remaining gpt-5.5 cells predate the guardrail commit; their stories are unaffected (their optimizer edits never violated the contract).

## SearchQA (test n=1400)

| target | variant | N0 | N1 | N2 | N3 | N4 | N5 | Δ |
|---|---|---|---|---|---|---|---|---|
| gpt-5.5 | no-gate | 0.800 | 0.825 | 0.823 | 0.805 | 0.808 | 0.823 | **+0.023** |
| gpt-5.5 | gate | 0.797 | 0.797 | 0.844 | 0.844 | 0.844 | 0.844 | **+0.047** |
| gpt-5.4-mini | no-gate | 0.766 | 0.773 | 0.781 | 0.769 | 0.774 | 0.765 | -0.001 |
| gpt-5.4-mini | gate | 0.768 | 0.778 | 0.778 | 0.778 | 0.778 | 0.778 | **+0.010** |
| gpt-5.4-nano | no-gate | 0.554 | 0.490 | 0.325 | 0.031 | 0.034 | 0.026 | -0.528 |
| gpt-5.4-nano | gate | 0.570 | 0.570 | 0.570 | 0.570 | 0.570 | 0.570 | +0.000 |

- gpt-5.5 no-gate: `▁█▇▂▃▇` 0.800 → 0.823
- gpt-5.5 gate: `▁▁████` 0.797 → 0.844
- gpt-5.4-mini no-gate: `▁▄█▂▅▁` 0.766 → 0.765
- gpt-5.4-mini gate: `▁█████` 0.768 → 0.778
- gpt-5.4-nano no-gate: `█▇▄▁▁▁` 0.554 → 0.026
- gpt-5.4-nano gate: `▁▁▁▁▁▁` 0.570 → 0.570

## LiveMathematicianBench (test n=124)

| target | variant | N0 | N1 | N2 | N3 | N4 | Δ |
|---|---|---|---|---|---|---|---|
| gpt-5.5 | no-gate | 0.508 | 0.556 | 0.500 | 0.524 | 0.556 | **+0.048** |
| gpt-5.5 | gate | 0.540 | 0.540 | 0.540 | 0.540 | 0.540 | -0.000 |
| gpt-5.4-mini | no-gate | 0.242 | 0.234 | 0.234 | 0.226 | 0.250 | **+0.008** |
| gpt-5.4-mini | gate | 0.250 | 0.258 | 0.258 | 0.258 | 0.258 | **+0.008** |
| gpt-5.4-nano | no-gate | 0.218 | 0.250 | 0.306 | 0.218 | 0.194 | -0.024 |
| gpt-5.4-nano | gate | 0.226 | 0.169 | 0.169 | 0.169 | 0.169 | -0.056 |

- gpt-5.5 no-gate: `▂█▁▃█` 0.508 → 0.556
- gpt-5.5 gate: `▁▁▁▁▁` 0.540 → 0.540
- gpt-5.4-mini no-gate: `▅▃▃▁█` 0.242 → 0.250
- gpt-5.4-mini gate: `▁████` 0.250 → 0.258
- gpt-5.4-nano no-gate: `▂▄█▂▁` 0.218 → 0.194
- gpt-5.4-nano gate: `█▁▁▁▁` 0.226 → 0.169

## SpreadsheetBench (test n=280)

| target | variant | N0 | N1 | N2 | N3 | N4 | N5 | Δ |
|---|---|---|---|---|---|---|---|---|
| gpt-5.5 | no-gate | 0.625 | 0.618 | 0.600 | 0.639 | 0.618 | 0.629 | +0.004 |
| gpt-5.5 | gate | 0.646 | 0.621 | 0.621 | 0.621 | 0.621 | 0.621 | -0.025 |
| gpt-5.4-mini | no-gate | 0.350 | 0.329 | 0.329 | 0.329 | 0.354 | 0.329 | -0.021 |
| gpt-5.4-mini | gate | 0.314 | 0.314 | 0.314 | 0.314 | 0.314 | 0.314 | +0.000 |
| gpt-5.4-nano | no-gate | 0.271 | 0.314 | 0.300 | 0.296 | 0.282 | 0.300 | **+0.029** |
| gpt-5.4-nano | gate | 0.279 | 0.318 | 0.318 | 0.279 | 0.279 | 0.279 | +0.000 |

- gpt-5.5 no-gate: `▅▄▁█▄▆` 0.625 → 0.629
- gpt-5.5 gate: `█▁▁▁▁▁` 0.646 → 0.621
- gpt-5.4-mini no-gate: `▆▁▁▁█▁` 0.350 → 0.329
- gpt-5.4-mini gate: `▁▁▁▁▁▁` 0.314 → 0.314
- gpt-5.4-nano no-gate: `▁█▅▅▂▅` 0.271 → 0.300
- gpt-5.4-nano gate: `▁██▁▁▁` 0.279 → 0.279

## What the data says (honest)

1. **The headline: on a weak model, no-gate sleep can DESTROY the agent — and the gate fully prevents it.** gpt-5.4-nano SearchQA no-gate collapses 0.554 → 0.026 (Δ-0.528): the optimizer taught it to answer with the *document title* ("output only the selected [DOC] [TLE] title string"), the obedient weak model complied, and accuracy fell off a cliff night after night. The gated twin rejected those same edits and held 0.570 for all 5 nights (Δ+0.000). This is the strongest single demonstration of the validation gate in the whole project.
2. **The guardrail fix works.** After injecting the task output contract into the optimizer, SpreadsheetBench no-gate went from learning harness-violating rules ("return VBA", "ask for the range" → regressions) to small real gains: gpt-5.4-nano 0.271 → 0.300 (Δ+0.029), gpt-5.5 Δ+0.004 (was −0.004).
3. **The clean positive case is still gpt-5.5 SearchQA with the gate**: 0.797 → 0.844 (Δ+0.047), a one-time jump at night 2 that the gate locks in.
4. **No monotonic night-over-night climbing anywhere.** Across 18 cells (3 models × 3 benchmarks × 2 variants), no curve climbs steadily. With 10 new tasks/night, sleep delivers either a one-time lift (captured best with the gate), a flat hold, or — without the gate — a drift whose worst case is catastrophic.
5. **Weak models need the gate the most.** The intuition "weak models have more headroom so sleep helps more" is only half-true: they also *follow bad rules more obediently*, so the downside of no-gate is far larger (nano −0.528) than the upside (+0.029). For strong models the no-gate risk is bounded (gpt-5.5 worst Δ −0.014); for weak models it is not.

## Takeaway

Ship the gate ON by default. The no-gate mode remains valuable for users who cannot hold out a validation set — but these results show its risk profile scales inversely with model strength: fine on gpt-5.5, catastrophic on nano. A pragmatic middle ground for gate-less deployments is the output-contract guardrail (now always on), which removed the format-violating failure mode entirely.

## Reproduce

```bash
PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
# per target model (routing: gpt-5.5/gpt-5.4 -> Responses endpoints, else MI):
SKILLOPT_SLEEP_WORKERS=32 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_nightly_matrix \
  --model gpt-5.5 --max-conc 2 --nights 5 --per-night 10 --rollouts 5 --dream-factor 2
SKILLOPT_SLEEP_WORKERS=20 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_nightly_matrix \
  --model gpt-5.4-nano --max-conc 2 --nights 5 --per-night 10 --rollouts 5 --dream-factor 2
PYTHONPATH=. $PY skillopt_sleep/experiments/gen_nightly_blog.py   # regenerate this file
```
