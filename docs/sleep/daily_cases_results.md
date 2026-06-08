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

_Real-model results are recorded here once the run completes; see the committed
JSON alongside this file. The mock run confirms the plumbing and the
anti-overfitting invariant (dream tasks only in train)._

| Family | Held rule the optimizer must learn | Backend | Test before → after |
|---|---|---|---|
| math | answers wrapped in `\boxed{...}` | Sonnet→Haiku | _pending_ |
| spreadsheet | formula starts with `=` and references a cell range | Sonnet→Haiku | _pending_ |
| searchqa | answer cites a source as `[DOC n]` | Sonnet→Haiku | _pending_ |
