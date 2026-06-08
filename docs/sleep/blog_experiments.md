# SkillOpt-Sleep — blog-aligned experiments

These plugin experiments mirror the SkillOpt research blog (`outputs/blog_1`) so
the two read as one story. Same **gate-ablation arms**, same **optimizer/target
pairing** (gpt-5.5 optimizer → gpt-5.4-mini target, Azure managed identity),
same **4:1:5 train/val/test split**, same **metric columns**.

The difference: this runs the deployment-time **Sleep** engine
(`skillopt_sleep`) on "daily case" task families, not the research `scripts/train.py`.
So it answers: *does the same validation-gated self-evolution work when the
artifact is a user's nightly plugin?*

## Arms (aligned to blog_1)

| Arm | Setting | Meaning |
|---|---|---|
| **C1** | `--gate off` | no gate — every proposed edit accepted (greedy). Ablation control. |
| **C2** | `--gate on --gate-metric hard` | hard-gate: keep an edit only if it raises the held-out **hard** score. |
| (C3 / C4) | `--gate-metric soft` / `mixed` | soft / mixed gate metrics (same as blog_1). |

Fixed: optimizer = **gpt-5.5**; target = **gpt-5.4-mini**; 4:1:5 split; seed 42;
3 nights. Three families:

| Family | House rule the optimizer must learn (judge) |
|---|---|
| math | final answer wrapped in `\boxed{...}` |
| spreadsheet | formula starts with `=` and references a cell/range |
| searchqa | answer cites a source as `[DOC n]` |

## Results (test = held-out, scored by the rule judge)

From `docs/sleep/blog_runs/{c1,c2}_mini.json`. Optimizer **gpt-5.5** → target
**gpt-5.4-mini**, 4:1:5 split, seed 42, 3 nights.

| arm | family | split (tr/val/te) | baseline(test) | after(test) | Δ |
|---|---|---|---|---|---|
| **C1** no-gate | math | 12/1/7 | 0.000 | **1.000** | **+1.000** |
| **C1** no-gate | spreadsheet | 6/3/11 | 0.545 | **0.909** | **+0.364** |
| **C1** no-gate | searchqa | 3/3/14 | 0.000 | **1.000** | **+1.000** |
| **C2** hard-gate | math | 12/1/7 | 0.000 | **1.000** | **+1.000** |
| **C2** hard-gate | spreadsheet | 6/3/11 | 0.545 | 0.545 | +0.000 |
| **C2** hard-gate | searchqa | 3/3/14 | 0.000 | **1.000** | **+1.000** |

**C1 improved 3/3 families; C2 improved 2/3.** The gpt-5.5 optimizer learned the
right house rule for each family (e.g. for searchqa: *"include at least one
document label matching the exact required pattern `[DOC n]`"*) and gpt-5.4-mini
applied it to the unseen test tasks.

### The honest finding (aligned with the blog's own caveats)

The one C1-vs-C2 difference is **spreadsheet**: greedy C1 captured **+0.364**, but
the **hard gate (C2) rejected the same edit (+0.000)**. Why? The val split here is
tiny (3 items), and on those 3 the edit didn't register a strict hard-score gain,
so the gate dropped it — even though it *did* help the larger test set. This is
exactly the small-selection-set effect the research blog documents and is why it
also runs **soft / mixed** gate metrics (C3/C4): a smoother gate signal accepts
such edits. The takeaway matches the paper: the hard gate is the safe default but
can be conservative on tiny validation sets; soft/mixed help there.

This mirrors the deployment reality: a user's nightly val set is often small, so
the plugin defaults to `mixed` (not pure `hard`) for exactly this reason.

## Reproduce

```bash
PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python  # has openai + azure-identity
# C1 (no gate)
$PY -m skillopt_sleep.experiments.run_daily \
  --optimizer-backend azure --optimizer-model gpt-5.5 \
  --target-backend azure --target-model gpt-5.4-mini \
  --families math,spreadsheet,searchqa --gate off \
  --nights 3 --val-fraction 0.1 --test-fraction 0.5 --seed 42 --json
# C2 (hard gate): add  --gate on --gate-metric hard
```

The Azure backend uses managed identity (client `8cafa2b1-…`) and the endpoints
from the blog's `avail_api.md`, so the models are identical to the research runs.
