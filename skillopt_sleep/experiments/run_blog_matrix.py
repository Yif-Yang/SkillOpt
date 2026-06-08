#!/usr/bin/env python3
"""Orchestrate the blog real-benchmark runs (aligned to the intern's blog_1).

Matrix: benchmarks x {C1 no-gate, C2 hard-gate} x targets {gpt-5.5, gpt-5.4-mini,
gpt-5.4-nano}; optimizer always gpt-5.5; FULL research test sets. Each run writes
its own JSON under docs/sleep/blog_runs/real/, so the orchestrator is resumable
(skip finished) and bounded-concurrency. Mirrors _tools/launch_mini.py.

Run with the reflact env (has openai + azure-identity):
  PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
  $PY -m skillopt_sleep.experiments.run_blog_matrix --max-conc 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "docs", "sleep", "blog_runs", "real")

# the intern's models minus Qwen
TARGETS = {"g55": "gpt-5.5", "g54m": "gpt-5.4-mini", "g54n": "gpt-5.4-nano"}
ARMS = {"c1": ("off", "hard"), "c2": ("on", "hard")}  # C1 no-gate, C2 hard-gate
BENCHES = ["searchqa", "livemath", "spreadsheet"]


def matrix():
    runs = []
    for bench in BENCHES:
        for arm, (gate, metric) in ARMS.items():
            for tkey, target in TARGETS.items():
                runs.append({
                    "name": f"{arm}_{bench}_{tkey}",
                    "bench": bench, "arm": arm, "gate": gate, "metric": metric,
                    "target": target,
                })
    return runs


def cmd_for(run, args):
    out = os.path.join(OUT, run["name"] + ".json")
    py = sys.executable
    c = [
        py, "-m", "skillopt_sleep.experiments.run_realbench",
        "--optimizer-backend", "azure", "--optimizer-model", "gpt-5.5",
        "--target-backend", "azure", "--target-model", run["target"],
        "--benchmarks", run["bench"],
        "--gate", run["gate"], "--gate-metric", run["metric"],
        "--n-train", str(args.n_train), "--dream-factor", str(args.dream_factor),
        "--nights", str(args.nights), "--seed", "42", "--json",
    ]
    if args.test_limit:
        c += ["--test-limit", str(args.test_limit)]
    return c, out


def is_done(out):
    if not os.path.exists(out) or os.path.getsize(out) < 10:
        return False
    try:
        json.load(open(out))
        return True
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-conc", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16, help="parallel replay per run")
    ap.add_argument("--n-train", type=int, default=5)
    ap.add_argument("--dream-factor", type=int, default=1)
    ap.add_argument("--nights", type=int, default=3)
    ap.add_argument("--test-limit", type=int, default=0, help="0 = FULL test")
    ap.add_argument("--only", default="", help="comma name filter")
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    runs = matrix()
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r["name"] in keep]

    env = dict(os.environ, SKILLOPT_SLEEP_WORKERS=str(args.workers))
    running = {}  # name -> (proc, fh)
    todo = list(runs)
    print(f"[matrix] {len(runs)} runs, max_conc={args.max_conc}, workers={args.workers}, "
          f"test={'FULL' if not args.test_limit else args.test_limit}")
    while todo or running:
        # reap finished
        for name in list(running):
            proc, fh = running[name]
            if proc.poll() is not None:
                fh.close()
                print(f"[matrix] done: {name} (rc={proc.returncode})")
                del running[name]
        # launch up to capacity
        while todo and len(running) < args.max_conc:
            run = todo.pop(0)
            c, out = cmd_for(run, args)
            if is_done(out):
                print(f"[matrix] skip (done): {run['name']}")
                continue
            fh = open(out, "w")
            errh = open(out + ".err", "w")
            print(f"[matrix] launch: {run['name']} -> {os.path.basename(out)}")
            proc = subprocess.Popen(c, cwd=ROOT, env=env, stdout=fh, stderr=errh)
            running[run["name"]] = (proc, fh)
        time.sleep(5)
    print("[matrix] all done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
