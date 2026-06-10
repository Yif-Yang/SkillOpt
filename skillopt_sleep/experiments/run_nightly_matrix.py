#!/usr/bin/env python3
"""Launch the enhanced multi-night experiment: 3 benchmarks x {gate off, on}.

Each (bench, gate) cell runs as its own process writing one JSON under
docs/sleep/blog_runs/nightly/, so the orchestrator is resumable and bounded.
LiveMath is the slow one (gpt-5.5 ~110s/math attempt) so it is launched FIRST
to grab max wall-clock. Uses the Responses-API backend on gpt4v scus+swc.

  PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
  SKILLOPT_SLEEP_WORKERS=40 PYTHONPATH=. $PY skillopt_sleep/experiments/run_nightly_matrix.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_BASE = os.path.join(ROOT, "docs", "sleep", "blog_runs", "nightly")

# (bench, gate). LiveMath first (slowest). gate variants of the same bench can
# share the train pool but run independently.
CELLS = [
    ("livemath", "off"), ("livemath", "on"),
    ("spreadsheet", "off"), ("spreadsheet", "on"),
    ("searchqa", "off"), ("searchqa", "on"),
]


def is_done(path):
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        return False
    try:
        json.load(open(path))
        return True
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-conc", type=int, default=3)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--nights", type=int, default=5)
    ap.add_argument("--per-night", type=int, default=10)
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--dream-factor", type=int, default=2)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--out-subdir", default="", help="output subdir under nightly/ (default: derived from model)")
    ap.add_argument("--only", default="", help="comma filter e.g. searchqa_off")
    args = ap.parse_args(argv)

    # model-specific output dir so different targets don't clobber each other.
    # gpt-5.5 keeps the legacy flat path; others go to nightly/<model>/.
    subdir = args.out_subdir or ("" if args.model == "gpt-5.5" else args.model.replace(".", "_"))
    OUT = os.path.join(OUT_BASE, subdir) if subdir else OUT_BASE
    os.makedirs(OUT, exist_ok=True)
    cells = CELLS
    if args.only:
        keep = set(args.only.split(","))
        cells = [c for c in cells if f"{c[0]}_{c[1]}" in keep]

    env = dict(os.environ, SKILLOPT_SLEEP_WORKERS=str(args.workers))
    running = {}
    todo = list(cells)
    print(f"[nightly] {len(cells)} cells, max_conc={args.max_conc}, workers={args.workers}, "
          f"nights={args.nights}, per_night={args.per_night}, rollouts={args.rollouts}", flush=True)

    while todo or running:
        for name in list(running):
            proc, fh, eh = running[name]
            if proc.poll() is not None:
                fh.close(); eh.close()
                print(f"[nightly] done: {name} (rc={proc.returncode})", flush=True)
                del running[name]
        while todo and len(running) < args.max_conc:
            bench, gate = todo.pop(0)
            name = f"{bench}_{gate}"
            out = os.path.join(OUT, name + ".json")
            if is_done(out):
                print(f"[nightly] skip (done): {name}", flush=True)
                continue
            # Route by model: the Responses endpoints (gpt4v scus/swc) serve ONLY
            # gpt-5.5 / gpt-5.4. nano + mini are 404 there, so they must use the
            # Managed-Identity "azure" backend (oaidr9, chat completions). Using
            # the wrong backend silently 404s every call -> all-zero scores.
            backend = "azure-responses" if args.model in ("gpt-5.5", "gpt-5.4") else "azure"
            cmd = [
                sys.executable, "-m", "skillopt_sleep.experiments.run_nightly",
                "--backend", backend, "--model", args.model,
                "--benchmarks", bench, "--gate", gate,
                "--nights", str(args.nights), "--per-night", str(args.per_night),
                "--rollouts", str(args.rollouts), "--dream-factor", str(args.dream_factor),
                "--seed", "42", "--out", out, "--json",
            ]
            fh = open(out + ".log", "w"); eh = open(out + ".err", "w")
            print(f"[nightly] launch: {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=eh)
            running[name] = (proc, fh, eh)
        time.sleep(5)
    print("[nightly] all done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
