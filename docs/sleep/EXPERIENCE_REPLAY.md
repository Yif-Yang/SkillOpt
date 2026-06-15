# SkillOpt-Sleep — experience replay & dream rollouts (opt-in)

Two opt-in mechanisms that strengthen the nightly consolidation when your tasks
have a clean correctness signal. Both default **off**, so enabling them is the
only way they change behavior.

## What they do

| Config knob | Default | Effect |
|---|---|---|
| `dream_rollouts` | `1` | Run each task **K** times and learn from the *contrast* between the good and bad attempts (contrastive reflection) instead of a single failure. |
| `recall_k` | `0` | **Associative recall** — each night, pull the `K` past tasks most similar to tonight's new ones (from a persisted task archive) into the dream, so related experience is revisited without replaying the whole history. |
| `dream_factor` | `0` | Add `N` lightweight synthetic variants of each task to the training pool. |

The validation gate still governs what ships, so these only ever *enlarge the
signal the optimizer reflects on* — the held-out gate decides what is kept.

## How to enable

```jsonc
// ~/.skillopt-sleep/config.json (or pass via the plugin's config)
{
  "dream_rollouts": 5,   // contrastive dreaming
  "recall_k": 20,        // recall ~20 similar past tasks each night
  "gate_mode": "on"      // keep the gate on (recommended)
}
```

`recall_k` draws from a capped `task_archive` that the cycle persists in
`state.json`, so recall becomes useful from the second night onward (once there
is history to recall from).

## Measured effect

Deployment protocol (5 nights × 10 new real tasks/night, full held-out test
sets, GPT-5.5 optimizer), run through the **same engine the plugin executes**
(`skillopt_sleep.dream.dream_consolidate`):

**SearchQA (GPT-5.5, full 1,400-item test, gated) — the gain scales with recall depth:**

| Config | Δ vs baseline |
|---|---|
| `recall_k=10, dream_rollouts=5` | +3.1 |
| `dream_rollouts=8` | +3.7 |
| **`recall_k=20, dream_rollouts=5`** | **+4.5** |
| full-history replay (reference) | +5.6 |

**Second-benchmark confirmation** (SpreadsheetBench, GPT-5.4-nano, gate-free,
shipped path): 0.279 → **0.314 (+3.6)**.

## When it helps — and when it doesn't

- **Helps** when tasks recur and have a checkable correctness signal (the
  optimizer has something real to learn and the gate can verify it).
- **Roughly flat** on saturated or noisy tasks (e.g. a strong model already near
  ceiling) — within run-to-run noise (±1–2 points, single seed).
- The validation gate keeps the downside bounded; keep it on by default.

Trade-off: `dream_rollouts > 1` multiplies the per-night rollout cost (K×), and
`recall_k > 0` adds the recalled tasks to each night's replay. Since the cycle
runs offline on idle quota this is usually acceptable, but budget accordingly
(`budget_tokens` / `budget_seconds`).

Raw per-run results for the table above: `docs/sleep/blog_runs/v2_port/`.
