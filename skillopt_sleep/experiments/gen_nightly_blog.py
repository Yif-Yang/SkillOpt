#!/usr/bin/env python3
"""Generate docs/sleep/nightly_experiment.md from the 6 nightly run JSONs.

Reads docs/sleep/blog_runs/nightly/{bench}_{off,on}.json (enhanced multi-night
experiment) and writes the Night-by-Night progression report comparing the
no-gate (daily-use) and gate (paper-aligned) variants. Deterministic: no API.

  PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
  PYTHONPATH=. $PY skillopt_sleep/experiments/gen_nightly_blog.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(ROOT, "docs", "sleep", "blog_runs", "nightly")
OUT = os.path.join(ROOT, "docs", "sleep", "nightly_experiment.md")

BENCHES = [
    ("searchqa", "SearchQA", 1400),
    ("livemath", "LiveMathematicianBench", 124),
    ("spreadsheet", "SpreadsheetBench", 280),
]


def load(bench, gate):
    p = os.path.join(RUNS, f"{bench}_{gate}.json")
    if not os.path.exists(p) or os.path.getsize(p) < 10:
        return None
    try:
        return json.load(open(p))["results"][0]
    except Exception:
        return None


def all_cells():
    cells, missing = {}, []
    for bench, _, _ in BENCHES:
        for gate in ("off", "on"):
            r = load(bench, gate)
            if r is None:
                missing.append(f"{bench}_{gate}")
            else:
                cells[(bench, gate)] = r
    return cells, missing


def spark(prog):
    """A tiny text sparkline for the progression."""
    if not prog:
        return ""
    lo, hi = min(prog), max(prog)
    blocks = "▁▂▃▄▅▆▇█"
    if hi - lo < 1e-9:
        return blocks[0] * len(prog)
    return "".join(blocks[min(7, int((x - lo) / (hi - lo) * 7))] for x in prog)


def prog_table(cells):
    """One row per (bench,gate): Night 0..5 test scores."""
    maxn = max(len(cells[k]["progression"]) for k in cells)
    head = "| benchmark | variant | " + " | ".join(f"N{i}" for i in range(maxn)) + " | Δ |"
    sep = "|" + "---|" * (maxn + 3)
    rows = [head, sep]
    for bench, title, _ in BENCHES:
        for gate, gname in (("off", "no-gate"), ("on", "gate")):
            r = cells.get((bench, gate))
            if not r:
                continue
            prog = r["progression"]
            # pad missing night columns with blanks so Δ stays in its own column
            cols = [f"{x:.3f}" for x in prog] + [""] * (maxn - len(prog))
            dstr = f"**{r['delta']:+.3f}**" if r["delta"] > 0.005 else f"{r['delta']:+.3f}"
            rows.append(f"| {title} | {gname} | " + " | ".join(cols) + f" | {dstr} |")
    return "\n".join(rows)


def build(cells):
    P = []
    P.append("# SkillOpt-Sleep — multi-night progression (gate vs no-gate)")
    P.append("")
    P.append("The deployment-scale sleep loop run for **5 nights**, adding **10 new real "
             "tasks each night** (cumulative train 10 → 20 → 30 → 40 → 50). Each night the "
             "new tasks are enriched with **5 dream rollout groups** (multi-rollout contrastive "
             "reflection), and the skill **carries over and is refined** from the previous "
             "night. Target + optimizer: **gpt-5.5** via the Responses API (gpt4v scus+swc).")
    P.append("")
    P.append("Two variants run side by side:")
    P.append("")
    P.append("* **no-gate** — the *daily-use* design: take the final skill directly, with **no "
             "validation set**. Models the common case where a user cannot build a held-out val "
             "set per task. Quality is judged only on the real test set.")
    P.append("* **gate** — paper-aligned: an edit is kept only if it improves a held-out "
             "validation slice (capped at 60 items).")
    P.append("")
    P.append("Test is scored after **every** night, so the columns N0 (baseline) … N5 are the "
             "real held-out accuracy at each step. Full test sets (SearchQA 1400, LiveMath 124, "
             "SpreadsheetBench 280). Raw JSON: `docs/sleep/blog_runs/nightly/`.")
    P.append("")
    P.append("## Night-by-night test accuracy")
    P.append("")
    P.append(prog_table(cells))
    P.append("")
    # sparklines
    P.append("Shape of each curve (N0→N5):")
    P.append("")
    for bench, title, _ in BENCHES:
        for gate, gname in (("off", "no-gate"), ("on", "gate")):
            r = cells.get((bench, gate))
            if r:
                P.append(f"- {title} {gname}: `{spark(r['progression'])}` "
                         f"{r['progression'][0]:.3f} → {r['progression'][-1]:.3f}")
    P.append("")

    # the clean win
    sg = cells.get(("searchqa", "on"))
    P.append("## What the data says (honest)")
    P.append("")
    P.append("1. **The clearest win is the gate on SearchQA.** It jumps "
             f"{sg['progression'][0]:.3f} → {sg['progression'][2]:.3f} at night 2 and **holds "
             f"it** (Δ{sg['delta']:+.3f}, the largest real gain here). This is the gate doing "
             "exactly its job: lock in a genuine improvement, then refuse later edits that would "
             "regress it.")
    P.append("2. **\"Accuracy climbs monotonically with more nights\" is NOT supported.** No "
             "curve climbs smoothly. With only 10 new tasks/night and a already-high gpt-5.5 "
             "baseline, accumulating training data does not produce night-over-night gains — the "
             "curves oscillate inside a few points. The sleep gain is a small one-time lift, not "
             "a steady climb.")
    P.append("3. **no-gate (the daily-use design) is genuinely mixed — and that is the point.** "
             "It gives small gains on SearchQA and LiveMath but is flat/down on SpreadsheetBench. "
             "The traces show why: greedy nightly edits can **overfit** (SearchQA learned an "
             "over-rigid \"output only the exact span, never wrap/append\" rule that helped night "
             "1 then decayed by night 3) or learn an **actively harmful** rule (SpreadsheetBench "
             "learned \"if the target range isn't stated, ask for it\" — but the task can't ask, "
             "so that's a wrong answer). Without a gate, nothing catches these.")
    P.append("4. **The gate can be too conservative when val is tiny.** LiveMath's val is only "
             "18 items; the gate accepts nothing and the score stays flat at "
             f"{cells[('livemath','on')]['progression'][0]:.3f} for all 5 nights. The gate's "
             "usefulness scales with how trustworthy the val signal is.")
    P.append("")
    P.append("## Takeaway")
    P.append("")
    P.append("At deployment scale (a handful of real tasks per night), the sleep loop delivers a "
             "**small, real, one-time lift** — best realized **with** the validation gate, which "
             "both captures the gain and prevents the regressions that the greedy no-gate variant "
             "suffers. The no-gate mode is more convenient (no val set required) and is fine when "
             "edits are low-risk, but on harder tasks it can adopt overfit or harmful rules with "
             "nothing to catch them. The honest headline is *safety and modest gains*, not "
             "*monotonic self-improvement*.")
    P.append("")
    P.append("## Reproduce")
    P.append("")
    P.append("```bash")
    P.append("PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python")
    P.append("SKILLOPT_SLEEP_WORKERS=32 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_nightly_matrix \\")
    P.append("  --max-conc 2 --workers 32 --nights 5 --per-night 10 --rollouts 5 --dream-factor 2")
    P.append("PYTHONPATH=. $PY skillopt_sleep/experiments/gen_nightly_blog.py")
    P.append("```")
    P.append("")
    return "\n".join(P)


def main(argv=None):
    cells, missing = all_cells()
    if missing:
        print(f"REFUSING: missing {len(missing)} cells: {', '.join(missing)}", file=sys.stderr)
        return 1
    with open(OUT, "w") as f:
        f.write(build(cells))
    print(f"wrote {OUT} from {len(cells)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
