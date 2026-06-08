"""SkillOpt-Sleep — run the sleep pipeline on the intern's REAL benchmarks.

Aligns with research blog_1: same data, same splits, same evaluators, same
gate-ablation arms, same optimizer/target pairing. The ONLY deviation is the
TRAIN side — it simulates the online sleep+dream pipeline (few real "today's
tasks" + dream-augmented variants) instead of the full research train split.

Scores correctness with the research evaluators (searchqa SQuAD-em, livemath
multiple-choice, spreadsheet cell-value), so the baseline is the model's REAL
unaided accuracy and the lift is genuine (not a toy 0->1.0).

Usage (reflact env has openai + azure-identity):
  PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
  $PY -m skillopt_sleep.experiments.run_realbench \
     --optimizer-backend azure --optimizer-model gpt-5.5 \
     --target-backend azure --target-model gpt-5.4-mini \
     --benchmarks searchqa,livemath --gate on --gate-metric hard \
     --n-train 5 --dream-factor 1 --nights 3 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from skillopt_sleep.backend import build_backend
from skillopt_sleep.consolidate import consolidate, select_gate_score
from skillopt_sleep.experiments.daily_cases import dream_augment
from skillopt_sleep.experiments.real_bench import (
    BENCHMARKS, available, load_benchmark,
)
from skillopt_sleep.replay import aggregate_scores, replay_batch


def _score(backend, tasks, skill, split):
    sub = [t for t in tasks if t.split == split]
    if not sub:
        return 0.0
    h, _ = aggregate_scores(replay_batch(backend, sub, skill, ""))
    return h


def run_bench(backend, bench, *, n_train, dream_factor, nights, edit_budget,
              gate_mode, gate_metric, test_limit, seed, data_root) -> dict:
    tasks = load_benchmark(bench, data_root=data_root, n_train=n_train,
                           test_limit=test_limit, seed=seed)
    # dream-augment the few real train tasks (the sleep/dream pipeline)
    real_train = [t for t in tasks if t.split == "train"]
    if dream_factor > 0:
        tasks = tasks + dream_augment(real_train, factor=dream_factor)

    n = {s: sum(1 for t in tasks if t.split == s) for s in ("train", "val", "test")}
    n_dream = sum(1 for t in tasks if t.origin == "dream")

    before = _score(backend, tasks, "", "test")  # baseline = empty skill
    trace = [{"night": 0, "test_hard": round(before, 4), "action": "baseline"}]
    cur = ""
    for night in range(1, nights + 1):
        res = consolidate(backend, tasks, cur, "", edit_budget=edit_budget,
                          gate_metric=gate_metric, gate_mode=gate_mode,
                          evolve_skill=True, evolve_memory=False, night=night)
        if res.accepted:
            cur = res.new_skill
        # the full TEST set is the FINAL measure — we do NOT re-score it every
        # night (that is the dominant, wasteful cost). We track the VAL score per
        # night (already computed inside consolidate) and score TEST once at the
        # end. This changes nothing about the reported baseline/after numbers.
        trace.append({"night": night, "val_hard": round(res.holdout_candidate, 4),
                      "action": res.gate_action,
                      "edits": [e.content[:90] for e in res.applied_edits]})
    after = _score(backend, tasks, cur, "test")
    return {
        "benchmark": bench, "split": n, "dream_train": n_dream,
        "test_before": round(before, 4), "test_after": round(after, 4),
        "delta": round(after - before, 4), "improved": after > before,
        "nights": nights, "trace": trace, "tokens": backend.tokens_used(),
        "final_skill_tail": cur[-400:],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SkillOpt-Sleep on the real research benchmarks")
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model", default="")
    ap.add_argument("--optimizer-backend", default="")
    ap.add_argument("--optimizer-model", default="")
    ap.add_argument("--target-backend", default="")
    ap.add_argument("--target-model", default="")
    ap.add_argument("--azure-endpoint", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--benchmarks", default="", help="comma list; default = all available")
    ap.add_argument("--n-train", type=int, default=5, help="few real 'today' tasks (sleep scale)")
    ap.add_argument("--dream-factor", type=int, default=1, help="dream variants per real train task")
    ap.add_argument("--nights", type=int, default=3)
    ap.add_argument("--edit-budget", type=int, default=4)
    ap.add_argument("--gate", default="on", choices=["on", "off"])
    ap.add_argument("--gate-metric", default="hard", choices=["hard", "soft", "mixed"])
    ap.add_argument("--test-limit", type=int, default=0, help="0 = FULL research test set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from skillopt_sleep.experiments.real_bench import DATA_ROOT
    data_root = args.data_root or DATA_ROOT
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()] or available(data_root)
    backend = build_backend(
        backend=args.backend, model=args.model,
        optimizer_backend=args.optimizer_backend, optimizer_model=args.optimizer_model,
        target_backend=args.target_backend, target_model=args.target_model,
        azure_endpoint=args.azure_endpoint,
    )

    results = []
    for b in benches:
        r = run_bench(backend, b, n_train=args.n_train, dream_factor=args.dream_factor,
                      nights=args.nights, edit_budget=args.edit_budget,
                      gate_mode=("off" if args.gate == "off" else "on"),
                      gate_metric=args.gate_metric, test_limit=args.test_limit,
                      seed=args.seed, data_root=data_root)
        results.append(r)
        if not args.json:
            sp = r["split"]
            print(f"  {b:<12} train={sp['train']}(+{r['dream_train']}d) val={sp['val']} test={sp['test']}"
                  f"  TEST {r['test_before']:.4f} -> {r['test_after']:.4f}  (Δ {r['delta']:+.4f})")

    summary = {"benchmark": "skillopt-sleep/realbench", "backend": backend.name,
               "gate": args.gate, "gate_metric": args.gate_metric,
               "n_improved": sum(1 for r in results if r["improved"]),
               "results": results}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
