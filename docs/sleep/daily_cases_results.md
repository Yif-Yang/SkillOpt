# SkillOpt-Sleep — academic daily-case benchmark results

This runs the academic "daily case" suite — three task families modeled on the
SkillOpt paper's evaluation (math / spreadsheet / search-QA), each with a
deliberately deficient skill and a programmatic rule judge — using the paper's
**4:1:5 train/val/test** split with a **dream-augmented training pool**.

The protocol mirrors the paper: train supplies experience (here, real tasks +
synthetic "dreamed" variants), val gates updates, and the held-out **test** set
is the reported score. Dream tasks never enter val/test.

## Reproduce

```bash
# deterministic plumbing check (no API):
python -m skillopt_sleep.experiments.run_daily --backend mock --dream-factor 1

# real (strong optimizer -> cheap target), all three families:
python -m skillopt_sleep.experiments.run_daily \
  --optimizer-backend claude --optimizer-model sonnet \
  --target-backend claude --target-model haiku \
  --families math,spreadsheet,searchqa --dream-factor 1 --nights 2
```

The ablation knobs from the paper are exposed as flags:
`--train-size` (training-set-size), `--rollouts-k` (reflection breadth),
`--edit-budget` (textual learning rate), `--gate on|off`, `--dream-factor`.

## Results

Real run, **gpt-5.5 optimizer → gpt-5.4-mini target** (Azure managed identity,
the same models as the research blog), 4:1:5 split, 3 nights. Full C1-vs-C2
ablation and analysis: [`blog_experiments.md`](blog_experiments.md).

| Family | House rule the optimizer must learn | Arm | Test before → after |
|---|---|---|---|
| math | answers wrapped in `\boxed{...}` | C1 / C2 | **0.00 → 1.00** / **0.00 → 1.00** |
| spreadsheet | formula starts with `=` and references a cell range | C1 / C2 | **0.55 → 0.91** / 0.55 → 0.55 |
| searchqa | answer cites a source as `[DOC n]` | C1 / C2 | **0.00 → 1.00** / **0.00 → 1.00** |

C1 (no gate) improved **3/3** families; C2 (hard gate) **2/3**. The one
difference — spreadsheet under the hard gate — is an honest small-validation-set
effect documented in `blog_experiments.md` (and the reason the plugin defaults to
the `mixed` gate metric). A reference run on Claude (Sonnet→Haiku) showed the same
direction (spreadsheet 0.18→0.91, searchqa 0.00→1.00).
