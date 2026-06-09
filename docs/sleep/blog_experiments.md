# SkillOpt-Sleep — real-benchmark results (aligned to research blog_1)

The deployment-time **Sleep** engine, run under the research blog's protocol: the intern's **exact data, splits, and evaluators**; same gate arms (C1 no-gate / C2 hard-gate); same optimizer/target pairing (gpt-5.5 optimizer; targets gpt-5.5 / gpt-5.4-mini / gpt-5.4-nano, Qwen excluded); **full** test sets. The **only** deviation: training simulates the online sleep+dream pipeline — 5 real "today's tasks" + dream-augmented variants — instead of the full research train split.

Correctness is scored by the research evaluators (not toy format rules): searchqa = SQuAD em vs gold; livemath = multiple-choice label after the per-item choice shuffle; spreadsheet = official cell-value compare after executing the agent's generated openpyxl code. Baselines therefore reflect the model's **real** unaided accuracy and match the intern's range.

Raw per-run JSON: `docs/sleep/blog_runs/real/`. Generated deterministically by `skillopt_sleep/experiments/gen_blog.py`.

## What this corrects

An earlier version of this writeup used a harness with four scoring bugs that made the numbers non-comparable to the research blog. All four are fixed (commit `fix(sleep): align real-benchmark harness to research repo`):

1. **Backend swallowed filtered/transient errors → silent 0s.** The Azure call wrapped everything in `except: return ""`, so 429s/timeouts and — critically — content-filter 400s became empty responses that scored 0. Now retries with backoff.
2. **The rollout wrapper tripped the content filter as a "jailbreak."** Phrasing like *"apply EXACTLY / HARD CONSTRAINTS that OVERRIDE / even at the cost of"* was flagged (HTTP 400) on ~½ of SearchQA items, which (with bug 1) zeroed them and dragged the baseline 0.79→0.42. Tasks now carry the research repo's neutral `rollout_system` verbatim.
3. **SpreadsheetBench omitted the no-formula rule.** openpyxl never computes formulas, so a model writing `=A1*B1` had the cell read back as `None` and scored 0 despite correct logic. The prompt now carries the intern's critical rule (compute in Python, write literal values), lifting the gpt-5.5 baseline from ~0.20 to a real ~0.6+.
4. **LiveMath skipped the per-item choice shuffle.** The raw data stores the correct option first (always label A), so "always answer A" scored ~1.0. Ported the research dataloader's deterministic shuffle; the gold distribution is now uniform A–E and the baseline is a real ~0.5–0.6.

All 96 unit tests pass; SearchQA baseline is back to 0.79 with zero empty responses.

## SearchQA

Full 1400-item held-out test, SQuAD exact-match (em).

| arm | target | baseline | after | Δ |
|---|---|---|---|---|
| C1 no-gate | gpt-5.5 | 0.7957 | **0.8107** | **+0.0150** |
| C1 no-gate | gpt-5.4-mini | 0.7679 | 0.7571 | -0.0107 |
| C1 no-gate | gpt-5.4-nano | 0.5579 | **0.5850** | **+0.0271** |
| C2 hard-gate | gpt-5.5 | 0.7929 | **0.8100** | **+0.0171** |
| C2 hard-gate | gpt-5.4-mini | 0.7693 | 0.7664 | -0.0029 |
| C2 hard-gate | gpt-5.4-nano | 0.5543 | **0.6021** | **+0.0479** |

_4/6 arms improved; Δ range [-0.0107, +0.0479]; mean Δ C1 +0.0105 vs C2 +0.0207. Noise floor (same-model C1/C2 baseline spread): ±0.0036 — Δ below this is not meaningful._

## LiveMathematicianBench

Full 124-item held-out test, multiple-choice label correctness (choices shuffled per item, uniform A-E gold).

| arm | target | baseline | after | Δ |
|---|---|---|---|---|
| C1 no-gate | gpt-5.5 | 0.5565 | 0.5161 | -0.0403 |
| C1 no-gate | gpt-5.4-mini | 0.2823 | 0.2419 | -0.0403 |
| C1 no-gate | gpt-5.4-nano | 0.1935 | **0.2177** | **+0.0242** |
| C2 hard-gate | gpt-5.5 | 0.5645 | 0.4758 | -0.0887 |
| C2 hard-gate | gpt-5.4-mini | 0.2258 | 0.2258 | +0.0000 |
| C2 hard-gate | gpt-5.4-nano | 0.2661 | 0.2661 | +0.0000 |

_1/6 arms improved; Δ range [-0.0887, +0.0242]; mean Δ C1 -0.0188 vs C2 -0.0296. Noise floor (same-model C1/C2 baseline spread): ±0.0726 — Δ below this is not meaningful._

## SpreadsheetBench

Full 280-item held-out test, real openpyxl code execution + official cell-value compare vs golden.xlsx.

| arm | target | baseline | after | Δ |
|---|---|---|---|---|
| C1 no-gate | gpt-5.5 | 0.6250 | 0.6107 | -0.0143 |
| C1 no-gate | gpt-5.4-mini | 0.3607 | 0.2714 | -0.0893 |
| C1 no-gate | gpt-5.4-nano | 0.2893 | 0.2393 | -0.0500 |
| C2 hard-gate | gpt-5.5 | 0.6393 | 0.6393 | +0.0000 |
| C2 hard-gate | gpt-5.4-mini | 0.3286 | **0.3500** | **+0.0214** |
| C2 hard-gate | gpt-5.4-nano | 0.2893 | 0.2893 | +0.0000 |

_1/6 arms improved; Δ range [-0.0893, +0.0214]; mean Δ C1 -0.0512 vs C2 +0.0071. Noise floor (same-model C1/C2 baseline spread): ±0.0321 — Δ below this is not meaningful._

## Takeaways

1. **Baselines now match the research repo, so the comparison is honest.** SearchQA gpt-5.5 baseline ≈ 0.80 (intern ≈ 0.79), LiveMath gpt-5.5 ≈ 0.56 (intern ≈ 0.52–0.59), SpreadsheetBench gpt-5.5 ≈ 0.62 (intern ≈ 0.41–0.62). The earlier writeup's huge "+0.43" lifts were almost entirely recovery from content-filter zeros, not learning — they are gone.
2. **Real sleep gains are modest, by design.** We train on only **5 real tasks/night + dreaming** (deployment scale), versus the intern's full 400-item train set + multi-step optimization. So SearchQA improves by a few points at most (best Δ +0.0479), and gains below the per-benchmark noise floor are reported as noise rather than dressed up.
3. **The validation gate's value scales with how reliable the val signal is — and we can see exactly where it helps and where it doesn't.** On **SpreadsheetBench** (40-item val) the gate is a clean win: C1 (no gate) regresses every target (Δ -0.0893 on mini), while C2 (gate) holds flat or improves on all 3 (3/3 targets C2 ≥ C1) — it rejects the self-sabotaging "write Excel formulas" edits the greedy arm accepts. On **LiveMath** (only 18-item val) the gate is unreliable (1/3 targets C2 ≥ C1): on gpt-5.5 it actually does worse than no gate, because an edit that helps 18 val items can hurt the 124-item test. The honest rule: trust the gate when the held-out val set is big enough to be a faithful proxy.
4. **Honesty by construction.** Four harness bugs were found and fixed (above); the numbers here are regenerated deterministically from the committed run JSONs via `gen_blog.py`, and small Δ's are explicitly flagged against a measured noise floor.

## Reproduce

```bash
PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python  # openai + azure-identity
SKILLOPT_SLEEP_WORKERS=16 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_blog_matrix \
  --max-conc 3 --nights 3 --n-train 5 --dream-factor 1
PYTHONPATH=. $PY skillopt_sleep/experiments/gen_blog.py   # regenerate this file
```
