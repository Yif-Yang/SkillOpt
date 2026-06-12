#!/usr/bin/env python3
"""Generate docs/sleep/nightly_experiment.md — three-grid edition.

Compares three versions of the full 18-cell deployment grid (3 benchmarks x
3 targets x {gate-free, gated}):

  v1            nightly/            broken-dream reference (the attempt-cache
                                    bug collapsed all K rollouts into one
                                    sample, so contrastive dreaming never fired)
  v2-none       nightly_v2/none/    FIXED dream, no history replay
  v2-retrieval  nightly_v2/         FIXED dream + associative recall (top-K
                                    similar historical tasks join each dream)

plus the searchqa replay-mode ablation (blog_runs/replay/). Deterministic: no
API calls, no timestamps.

  PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
  PYTHONPATH=. $PY skillopt_sleep/experiments/gen_nightly_blog.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(ROOT, "docs", "sleep", "blog_runs")
OUT = os.path.join(ROOT, "docs", "sleep", "nightly_experiment.md")

BENCHES = [
    ("searchqa", "SearchQA", 1400),
    ("livemath", "LiveMathematicianBench", 124),
    ("spreadsheet", "SpreadsheetBench", 280),
]
MODELS = [("gpt-5.5", ""), ("gpt-5.4-mini", "gpt-5_4-mini"), ("gpt-5.4-nano", "gpt-5_4-nano")]
GRIDS = [  # (key, label, base dir under blog_runs)
    ("v1", "v1 (broken dream)", "nightly"),
    ("v2n", "v2 fixed dream", os.path.join("nightly_v2", "none")),
    ("v2r", "v2 fixed dream + recall", "nightly_v2"),
]


def load(grid_dir, model_sub, bench, gate):
    p = os.path.join(RUNS, grid_dir, model_sub, f"{bench}_{gate}.json") if model_sub \
        else os.path.join(RUNS, grid_dir, f"{bench}_{gate}.json")
    if not os.path.exists(p) or os.path.getsize(p) < 10:
        return None
    try:
        return json.load(open(p))["results"][0]
    except Exception:
        return None


def all_cells():
    cells, missing = {}, []
    for gkey, _, gdir in GRIDS:
        for model, sub in MODELS:
            for bench, _, _ in BENCHES:
                for gate in ("off", "on"):
                    r = load(gdir, sub, bench, gate)
                    if r is None:
                        missing.append(f"{gkey}/{model}/{bench}_{gate}")
                    else:
                        cells[(gkey, model, bench, gate)] = r
    return cells, missing


def load_replay():
    out = {}
    for mode in ("none", "retrieval", "cumulative"):
        for gate in ("off", "on"):
            p = os.path.join(RUNS, "replay", f"searchqa_{mode}_{gate}.json")
            if os.path.exists(p) and os.path.getsize(p) > 10:
                try:
                    out[(mode, gate)] = json.load(open(p))["results"][0]
                except Exception:
                    pass
    return out


def spark(prog):
    if not prog:
        return ""
    lo, hi = min(prog), max(prog)
    blocks = "▁▂▃▄▅▆▇█"
    if hi - lo < 1e-9:
        return blocks[0] * len(prog)
    return "".join(blocks[min(7, int((x - lo) / (hi - lo) * 7))] for x in prog)


def d(r):
    return f"{100 * r['delta']:+.1f}" if r else "—"


def build(cells, replay):
    P = []
    P.append("# SkillOpt-Sleep — deployment grid, before and after the dream fix")
    P.append("")
    P.append("Protocol (identical for every cell): 5 nights, +10 new real tasks/night, dream "
             "rollout groups (k=5) + synthetic variants (x2), skill carries over and is refined; "
             "full held-out tests (SearchQA 1400 / LiveMath 124 / SpreadsheetBench 280), "
             "execution-grade evaluators; gpt-5.5 optimizer; seed 42.")
    P.append("")
    P.append("Three grid versions:")
    P.append("")
    P.append("* **v1 (broken dream)** — run before we found the attempt-cache bug that collapsed "
             "all K dream rollouts into one cached sample (contrastive reflection never fired). "
             "Kept as the honest reference.")
    P.append("* **v2 fixed dream** — rollouts are independent samples (`sample_id` in the cache "
             "key); no history replay (each night sees only its 10 new tasks).")
    P.append("* **v2 fixed dream + recall** — the framework design: each night the 10 most "
             "lexically similar historical tasks join the dream (associative recall, k=10).")
    P.append("")

    # per-benchmark tables
    for bench, title, n in BENCHES:
        P.append(f"## {title} (test n={n})")
        P.append("")
        P.append("| target | mode | v1 Δ | v2 fixed Δ | v2 +recall Δ | v2 +recall: baseline → final |")
        P.append("|---|---|---|---|---|---|")
        for model, _ in MODELS:
            for gate, gname in (("off", "gate-free"), ("on", "gated")):
                r1 = cells.get(("v1", model, bench, gate))
                r2 = cells.get(("v2n", model, bench, gate))
                r3 = cells.get(("v2r", model, bench, gate))
                tail = f"{r3['test_baseline']:.3f} → {r3['test_final']:.3f}" if r3 else "—"
                P.append(f"| {model} | {gname} | {d(r1)} | {d(r2)} | **{d(r3)}** | {tail} |")
        P.append("")
        # sparklines for the recall grid
        P.append("v2 +recall night-by-night (N0..N5):")
        P.append("")
        for model, _ in MODELS:
            for gate, gname in (("off", "gate-free"), ("on", "gated")):
                r3 = cells.get(("v2r", model, bench, gate))
                if r3:
                    P.append(f"- {model} {gname}: `{spark(r3['progression'])}` "
                             f"{r3['progression'][0]:.3f} → {r3['progression'][-1]:.3f}")
        P.append("")

    # replay ablation
    if replay:
        P.append("## Replay-mode ablation (SearchQA, gpt-5.5, fixed dream)")
        P.append("")
        P.append("| replay mode | gate-free Δ | gated Δ |")
        P.append("|---|---|---|")
        for mode, label in (("none", "none (new tasks only)"),
                            ("retrieval", "retrieval (recall k=10)"),
                            ("cumulative", "cumulative (full history)")):
            ro, rn = replay.get((mode, "off")), replay.get((mode, "on"))
            P.append(f"| {label} | {d(ro)} | {d(rn)} |")
        P.append("")
        rc = replay.get(("cumulative", "on"))
        if rc:
            P.append(f"cumulative+gated climbs night over night and accepts new edits as late as "
                     f"night 5: `{spark(rc['progression'])}` "
                     f"{rc['progression'][0]:.3f} → {rc['progression'][-1]:.3f}.")
        P.append("")

    # data-driven summary
    def stats(gkey):
        ds = [100 * r["delta"] for (g, m, b, gt), r in cells.items() if g == gkey]
        pos = sum(1 for x in ds if x > 0.5)
        neg = sum(1 for x in ds if x < -0.5)
        return pos, neg, (sum(ds) / len(ds) if ds else 0.0), (min(ds) if ds else 0.0)

    P.append("## Summary (computed from the grids)")
    P.append("")
    P.append("| grid | cells >+0.5 | cells <−0.5 | mean Δ | worst Δ |")
    P.append("|---|---|---|---|---|")
    for gkey, label, _ in GRIDS:
        pos, neg, mean, worst = stats(gkey)
        P.append(f"| {label} | {pos}/18 | {neg}/18 | {mean:+.1f} | {worst:+.1f} |")
    P.append("")
    P.append("Raw per-run JSON under `docs/sleep/blog_runs/{nightly,nightly_v2,replay}/`; this "
             "file is regenerated deterministically by `gen_nightly_blog.py`.")
    P.append("")
    return "\n".join(P)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    cells, missing = all_cells()
    if args.check:
        per = {}
        for m in missing:
            per[m.split("/")[0]] = per.get(m.split("/")[0], 0) + 1
        done = {g: 18 - per.get(g, 0) for g, _, _ in GRIDS}
        print("present:", {g: f"{n}/18" for g, n in done.items()})
        if missing:
            print("missing:", ", ".join(missing[:8]), "..." if len(missing) > 8 else "")
        return 0 if not missing else 1
    if missing:
        print(f"REFUSING: {len(missing)} cells missing: {', '.join(missing[:6])} ...", file=sys.stderr)
        return 1
    md = build(cells, load_replay())
    with open(OUT, "w") as f:
        f.write(md)
    print(f"wrote {OUT} from {len(cells)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
