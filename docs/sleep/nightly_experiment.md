# SkillOpt-Sleep — deployment grid, before and after the dream fix

Protocol (identical for every cell): 5 nights, +10 new real tasks/night, dream rollout groups (k=5) + synthetic variants (x2), skill carries over and is refined; full held-out tests (SearchQA 1400 / LiveMath 124 / SpreadsheetBench 280), execution-grade evaluators; gpt-5.5 optimizer; seed 42.

Three grid versions:

* **v1 (broken dream)** — run before we found the attempt-cache bug that collapsed all K dream rollouts into one cached sample (contrastive reflection never fired). Kept as the honest reference.
* **v2 fixed dream** — rollouts are independent samples (`sample_id` in the cache key); no history replay (each night sees only its 10 new tasks).
* **v2 fixed dream + recall** — the framework design: each night the 10 most lexically similar historical tasks join the dream (associative recall, k=10).

## SearchQA (test n=1400)

| target | mode | v1 Δ | v2 fixed Δ | v2 +recall Δ | v2 +recall: baseline → final |
|---|---|---|---|---|---|
| gpt-5.5 | gate-free | +2.3 | +3.9 | **+5.1** | 0.799 → 0.850 |
| gpt-5.5 | gated | +4.7 | +2.0 | **+4.4** | 0.797 → 0.841 |
| gpt-5.4-mini | gate-free | -0.1 | -0.3 | **-1.4** | 0.776 → 0.762 |
| gpt-5.4-mini | gated | +1.0 | +0.5 | **+1.4** | 0.776 → 0.790 |
| gpt-5.4-nano | gate-free | -52.8 | +2.7 | **+0.6** | 0.557 → 0.563 |
| gpt-5.4-nano | gated | +0.0 | +5.6 | **-1.9** | 0.554 → 0.535 |

v2 +recall night-by-night (N0..N5):

- gpt-5.5 gate-free: `▂▅▁▇█▇` 0.799 → 0.850
- gpt-5.5 gated: `▁▇████` 0.797 → 0.841
- gpt-5.4-mini gate-free: `▄█▅▃▄▁` 0.776 → 0.762
- gpt-5.4-mini gated: `▁▁█▆▆▆` 0.776 → 0.790
- gpt-5.4-nano gate-free: `▁█▁▁▂▁` 0.557 → 0.563
- gpt-5.4-nano gated: `██▁▁▁▁` 0.554 → 0.535

## LiveMathematicianBench (test n=124)

| target | mode | v1 Δ | v2 fixed Δ | v2 +recall Δ | v2 +recall: baseline → final |
|---|---|---|---|---|---|
| gpt-5.5 | gate-free | +4.8 | -1.6 | **+0.0** | 0.508 → 0.508 |
| gpt-5.5 | gated | -0.0 | +0.0 | **-0.8** | 0.548 → 0.540 |
| gpt-5.4-mini | gate-free | +0.8 | -2.4 | **-2.4** | 0.266 → 0.242 |
| gpt-5.4-mini | gated | +0.8 | +0.8 | **-1.6** | 0.234 → 0.218 |
| gpt-5.4-nano | gate-free | -2.4 | -4.0 | **+3.2** | 0.161 → 0.194 |
| gpt-5.4-nano | gated | -5.6 | +0.0 | **-0.0** | 0.202 → 0.202 |

v2 +recall night-by-night (N0..N5):

- gpt-5.5 gate-free: `▁▄█▂▁` 0.508 → 0.508
- gpt-5.5 gated: `████▁` 0.548 → 0.540
- gpt-5.4-mini gate-free: `█▆▁▆▄` 0.266 → 0.242
- gpt-5.4-mini gated: `██▁▁▁` 0.234 → 0.218
- gpt-5.4-nano gate-free: `▁█▆▆▄` 0.161 → 0.194
- gpt-5.4-nano gated: `▁▁▁▁▁` 0.202 → 0.202

## SpreadsheetBench (test n=280)

| target | mode | v1 Δ | v2 fixed Δ | v2 +recall Δ | v2 +recall: baseline → final |
|---|---|---|---|---|---|
| gpt-5.5 | gate-free | +0.4 | -1.1 | **-1.1** | 0.650 → 0.639 |
| gpt-5.5 | gated | -2.5 | +0.0 | **-1.8** | 0.636 → 0.618 |
| gpt-5.4-mini | gate-free | -2.1 | -2.1 | **+0.4** | 0.339 → 0.343 |
| gpt-5.4-mini | gated | +0.0 | -1.4 | **+0.0** | 0.339 → 0.339 |
| gpt-5.4-nano | gate-free | +2.9 | -1.1 | **+4.6** | 0.293 → 0.339 |
| gpt-5.4-nano | gated | +0.0 | +2.9 | **+0.7** | 0.318 → 0.325 |

v2 +recall night-by-night (N0..N5):

- gpt-5.5 gate-free: `█▇▇▁▇▇` 0.650 → 0.639
- gpt-5.5 gated: `███▁▁▁` 0.636 → 0.618
- gpt-5.4-mini gate-free: `▆▅▃█▁▇` 0.339 → 0.343
- gpt-5.4-mini gated: `▁▁▁▁▁▁` 0.339 → 0.339
- gpt-5.4-nano gate-free: `▁▂▁▁▁█` 0.293 → 0.339
- gpt-5.4-nano gated: `▁▁████` 0.318 → 0.325

## Replay-mode ablation (SearchQA, gpt-5.5, fixed dream)

| replay mode | gate-free Δ | gated Δ |
|---|---|---|
| none (new tasks only) | +3.9 | +2.0 |
| retrieval (recall k=10) | +5.1 | +4.4 |
| cumulative (full history) | +4.8 | +6.0 |

cumulative+gated climbs night over night and accepts new edits as late as night 5: `▁▂▇▇▇█` 0.798 → 0.858.

## Summary (computed from the grids)

| grid | cells >+0.5 | cells <−0.5 | mean Δ | worst Δ |
|---|---|---|---|---|
| v1 (broken dream) | 7/18 | 5/18 | -2.7 | -52.8 |
| v2 fixed dream | 6/18 | 7/18 | +0.2 | -4.0 |
| v2 fixed dream + recall | 7/18 | 7/18 | +0.5 | -2.4 |

Raw per-run JSON under `docs/sleep/blog_runs/{nightly,nightly_v2,replay}/`; this file is regenerated deterministically by `gen_nightly_blog.py`.
