#!/usr/bin/env python3
"""Generate docs/sleep/nightly_experiment.md from the nightly run JSONs.

Multi-model edition: reads the enhanced multi-night experiment for every target
model — gpt-5.5 in docs/sleep/blog_runs/nightly/*.json (legacy flat layout) and
gpt-5.4-mini / gpt-5.4-nano in nightly/<model>/ subdirs — and writes the
Night-by-Night progression report comparing the no-gate (daily-use) and gate
(paper-aligned) variants across all targets. Deterministic: no API calls.

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
# (model label, subdir under nightly/; "" = legacy flat layout)
MODELS = [
    ("gpt-5.5", ""),
    ("gpt-5.4-mini", "gpt-5_4-mini"),
    ("gpt-5.4-nano", "gpt-5_4-nano"),
]


def load(subdir, bench, gate):
    p = os.path.join(RUNS, subdir, f"{bench}_{gate}.json") if subdir else \
        os.path.join(RUNS, f"{bench}_{gate}.json")
    if not os.path.exists(p) or os.path.getsize(p) < 10:
        return None
    try:
        return json.load(open(p))["results"][0]
    except Exception:
        return None


def all_cells():
    cells, missing = {}, []
    for model, sub in MODELS:
        for bench, _, _ in BENCHES:
            for gate in ("off", "on"):
                r = load(sub, bench, gate)
                if r is None:
                    missing.append(f"{model}/{bench}_{gate}")
                else:
                    cells[(model, bench, gate)] = r
    return cells, missing


def spark(prog):
    if not prog:
        return ""
    lo, hi = min(prog), max(prog)
    blocks = "▁▂▃▄▅▆▇█"
    if hi - lo < 1e-9:
        return blocks[0] * len(prog)
    return "".join(blocks[min(7, int((x - lo) / (hi - lo) * 7))] for x in prog)


def bench_table(cells, bench):
    maxn = max(len(r["progression"]) for k, r in cells.items() if k[1] == bench)
    head = "| target | variant | " + " | ".join(f"N{i}" for i in range(maxn)) + " | Δ |"
    sep = "|" + "---|" * (maxn + 3)
    rows = [head, sep]
    for model, _ in MODELS:
        for gate, gname in (("off", "no-gate"), ("on", "gate")):
            r = cells.get((model, bench, gate))
            if not r:
                continue
            prog = r["progression"]
            cols = [f"{x:.3f}" for x in prog] + [""] * (maxn - len(prog))
            d = r["delta"]
            dstr = f"**{d:+.3f}**" if d > 0.005 else f"{d:+.3f}"
            rows.append(f"| {model} | {gname} | " + " | ".join(cols) + f" | {dstr} |")
    return "\n".join(rows)


def build(cells):
    P = []
    P.append("# SkillOpt-Sleep — multi-night progression across targets (gate vs no-gate)")
    P.append("")
    P.append("The deployment-scale sleep loop run for **5 nights**, adding **10 new real "
             "tasks each night** (cumulative train 10 → 50). Each night the new tasks get "
             "**5 dream rollout groups** (multi-rollout contrastive reflection), the skill "
             "**carries over and is refined**, and the optimizer prompt carries the task's "
             "**output contract guardrail** (it may not propose rules that change the required "
             "format or tell the agent to ask questions — such rules score zero by construction).")
    P.append("")
    P.append("Three target models: **gpt-5.5** (strong; Responses API on gpt4v scus+swc), "
             "**gpt-5.4-mini** and **gpt-5.4-nano** (weaker; Managed-Identity endpoints). "
             "Optimizer = the same model as the target in each run. Two variants per cell:")
    P.append("")
    P.append("* **no-gate** — daily-use design: accept every edit, take the final skill, **no "
             "validation set**. Quality judged only on the real held-out test.")
    P.append("* **gate** — paper-aligned: an edit is kept only if it improves a held-out val "
             "slice (capped at 60).")
    P.append("")
    P.append("Test is scored after every night (N0 = baseline). Full test sets. "
             "Raw JSON: `docs/sleep/blog_runs/nightly/`.")
    P.append("")
    P.append("Provenance: all gpt-5.4-mini / gpt-5.4-nano cells and the gpt-5.5 "
             "SpreadsheetBench no-gate cell were run with the output-contract guardrail "
             "(current code). The remaining gpt-5.5 cells predate the guardrail commit; "
             "their stories are unaffected (their optimizer edits never violated the contract).")
    P.append("")
    for bench, title, n in BENCHES:
        P.append(f"## {title} (test n={n})")
        P.append("")
        P.append(bench_table(cells, bench))
        P.append("")
        for model, _ in MODELS:
            for gate, gname in (("off", "no-gate"), ("on", "gate")):
                r = cells.get((model, bench, gate))
                if r:
                    P.append(f"- {model} {gname}: `{spark(r['progression'])}` "
                             f"{r['progression'][0]:.3f} → {r['progression'][-1]:.3f}")
        P.append("")

    nso = cells[("gpt-5.4-nano", "searchqa", "off")]
    nsg = cells[("gpt-5.4-nano", "searchqa", "on")]
    g5g = cells[("gpt-5.5", "searchqa", "on")]
    nspo = cells[("gpt-5.4-nano", "spreadsheet", "off")]
    P.append("## What the data says (honest)")
    P.append("")
    P.append("1. **The headline: on a weak model, no-gate sleep can DESTROY the agent — and the "
             "gate fully prevents it.** gpt-5.4-nano SearchQA no-gate collapses "
             f"{nso['progression'][0]:.3f} → {nso['progression'][-1]:.3f} "
             f"(Δ{nso['delta']:+.3f}): the optimizer taught it to answer with the *document "
             "title* (\"output only the selected [DOC] [TLE] title string\"), the obedient weak "
             "model complied, and accuracy fell off a cliff night after night. The gated twin "
             f"rejected those same edits and held {nsg['progression'][0]:.3f} for all 5 nights "
             f"(Δ{nsg['delta']:+.3f}). This is the strongest single demonstration of the "
             "validation gate in the whole project.")
    P.append("2. **The guardrail fix works.** After injecting the task output contract into the "
             "optimizer, SpreadsheetBench no-gate went from learning harness-violating rules "
             "(\"return VBA\", \"ask for the range\" → regressions) to small real gains: "
             f"gpt-5.4-nano {nspo['progression'][0]:.3f} → {nspo['progression'][-1]:.3f} "
             f"(Δ{nspo['delta']:+.3f}), gpt-5.5 Δ+0.004 (was −0.004).")
    P.append("3. **The clean positive case is still gpt-5.5 SearchQA with the gate**: "
             f"{g5g['progression'][0]:.3f} → {g5g['progression'][-1]:.3f} "
             f"(Δ{g5g['delta']:+.3f}), a one-time jump at night 2 that the gate locks in.")
    P.append("4. **No monotonic night-over-night climbing anywhere.** Across 18 cells "
             "(3 models × 3 benchmarks × 2 variants), no curve climbs steadily. With 10 new "
             "tasks/night, sleep delivers either a one-time lift (captured best with the gate), "
             "a flat hold, or — without the gate — a drift whose worst case is catastrophic.")
    P.append("5. **Weak models need the gate the most.** The intuition \"weak models have more "
             "headroom so sleep helps more\" is only half-true: they also *follow bad rules more "
             "obediently*, so the downside of no-gate is far larger (nano −0.528) than the upside "
             "(+0.029). For strong models the no-gate risk is bounded (gpt-5.5 worst Δ −0.014); "
             "for weak models it is not.")
    P.append("")
    P.append("## Takeaway")
    P.append("")
    P.append("Ship the gate ON by default. The no-gate mode remains valuable for users who "
             "cannot hold out a validation set — but these results show its risk profile scales "
             "inversely with model strength: fine on gpt-5.5, catastrophic on nano. A pragmatic "
             "middle ground for gate-less deployments is the output-contract guardrail (now "
             "always on), which removed the format-violating failure mode entirely.")
    P.append("")
    P.append("## Reproduce")
    P.append("")
    P.append("```bash")
    P.append("PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python")
    P.append("# per target model (routing: gpt-5.5/gpt-5.4 -> Responses endpoints, else MI):")
    P.append("SKILLOPT_SLEEP_WORKERS=32 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_nightly_matrix \\")
    P.append("  --model gpt-5.5 --max-conc 2 --nights 5 --per-night 10 --rollouts 5 --dream-factor 2")
    P.append("SKILLOPT_SLEEP_WORKERS=20 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_nightly_matrix \\")
    P.append("  --model gpt-5.4-nano --max-conc 2 --nights 5 --per-night 10 --rollouts 5 --dream-factor 2")
    P.append("PYTHONPATH=. $PY skillopt_sleep/experiments/gen_nightly_blog.py   # regenerate this file")
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
