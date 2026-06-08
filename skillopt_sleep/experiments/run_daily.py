"""SkillOpt-Sleep — run the academic 'daily case' benchmark.

Reproduces the SkillOpt paper's ablation *protocol* on simulated daily cases:
  * paper's 4:1:5 train/selection(val)/test split
  * optional dream-augmented TRAIN pool (synthetic variants; never in val/test)
  * the ablation knobs are exposed as flags (training-set-size, rollouts-k,
    edit-budget/learning-rate, gate on/off)

For each task family (math / spreadsheet / searchqa) it:
  1. scores the held-out TEST set with the deficient skill        -> before
  2. runs N nights (reflect -> bounded gated edit) on TRAIN/VAL    -> evolve
  3. scores the held-out TEST set with the evolved skill          -> after

Usage:
  python -m skillopt_sleep.experiments.run_daily --backend mock
  python -m skillopt_sleep.experiments.run_daily --optimizer-backend claude \
     --optimizer-model sonnet --target-backend claude --target-model haiku \
     --families math,spreadsheet --dream-factor 1 --nights 2
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from skillopt_sleep.backend import build_backend
from skillopt_sleep.consolidate import consolidate, select_gate_score
from skillopt_sleep.experiments.daily_cases import (
    DEFICIENT_SKILLS, FAMILIES, dream_augment, make_tasks,
)
from skillopt_sleep.memory import ensure_skill_scaffold
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.replay import aggregate_scores, replay_batch


def _score(backend, tasks, skill, split="test"):
    sub = [t for t in tasks if t.split == split]
    if not sub:
        sub = [t for t in tasks if t.split == "val"] or tasks
    h, _ = aggregate_scores(replay_batch(backend, sub, skill, ""))
    return h


def run_family(backend, family: str, *, nights: int, edit_budget: int,
               rollouts_k: int, gate_mode: str, gate_metric: str, dream_factor: int,
               train_size: int, val_fraction: float, test_fraction: float,
               seed: int) -> dict:
    real = make_tasks(family)
    # paper-style 4:1:5 (defaults) split on the REAL tasks
    real = assign_splits(real, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed)
    # optionally shrink the train pool (paper ablation (a): training set size)
    train = [t for t in real if t.split == "train"]
    if train_size and train_size < len(train):
        for t in train[train_size:]:
            t.split = "unused"
    train = [t for t in real if t.split == "train"]
    # dream-augment the TRAIN pool (synthetic experience; never in val/test)
    dream = dream_augment(train, factor=dream_factor) if dream_factor > 0 else []
    tasks = real + dream

    skill = ensure_skill_scaffold(DEFICIENT_SKILLS[family],
                                  name=f"{family}-helper", description=family)
    before = _score(backend, tasks, skill, "test")
    trace = [{"night": 0, "test_hard": round(before, 3), "action": "baseline"}]
    cur = skill
    for night in range(1, nights + 1):
        res = consolidate(backend, tasks, cur, "", edit_budget=edit_budget,
                          gate_metric=gate_metric, gate_mode=gate_mode,
                          rollouts_k=rollouts_k, evolve_skill=True,
                          evolve_memory=False, night=night)
        if res.accepted:
            cur = res.new_skill
        th = _score(backend, tasks, cur, "test")
        trace.append({"night": night, "val_hard": round(res.holdout_candidate, 3),
                      "test_hard": round(th, 3), "action": res.gate_action,
                      "edits": [e.content[:80] for e in res.applied_edits]})
        if th >= 0.999:
            break
    after = _score(backend, tasks, cur, "test")
    n = {s: sum(1 for t in tasks if t.split == s) for s in ("train", "val", "test")}
    return {
        "family": family, "split": n, "dream_train": len(dream),
        "test_before": round(before, 3), "test_after": round(after, 3),
        "improved": after > before, "nights": len(trace) - 1, "trace": trace,
        "tokens": backend.tokens_used(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SkillOpt-Sleep academic daily-case benchmark")
    ap.add_argument("--backend", default="mock", choices=["mock", "claude", "codex", "azure"])
    ap.add_argument("--model", default="")
    ap.add_argument("--optimizer-backend", default="")
    ap.add_argument("--optimizer-model", default="")
    ap.add_argument("--target-backend", default="")
    ap.add_argument("--target-model", default="")
    ap.add_argument("--codex-path", default="")
    ap.add_argument("--azure-endpoint", default="", help="override Azure endpoint (else auto by deployment)")
    ap.add_argument("--families", default="", help="comma list; default = all")
    ap.add_argument("--nights", type=int, default=2)
    ap.add_argument("--edit-budget", type=int, default=4, help="textual learning rate")
    ap.add_argument("--rollouts-k", type=int, default=1)
    ap.add_argument("--gate", default="on", choices=["on", "off"],
                    help="off = arm C1 (no gate); on = arms C2/C3/C4 by --gate-metric")
    ap.add_argument("--gate-metric", default="hard", choices=["hard", "soft", "mixed"],
                    help="C2=hard, C3=soft, C4=mixed (used when --gate on)")
    ap.add_argument("--dream-factor", type=int, default=0, help="synthetic train variants per real task")
    ap.add_argument("--train-size", type=int, default=0, help="cap train pool (paper ablation a); 0 = all")
    ap.add_argument("--val-fraction", type=float, default=0.1, help="paper 4:1:5 -> val=0.1")
    ap.add_argument("--test-fraction", type=float, default=0.5, help="paper 4:1:5 -> test=0.5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    families = [f.strip() for f in args.families.split(",") if f.strip()] or FAMILIES
    backend = build_backend(
        backend=args.backend, model=args.model,
        optimizer_backend=args.optimizer_backend, optimizer_model=args.optimizer_model,
        target_backend=args.target_backend, target_model=args.target_model,
        codex_path=args.codex_path, azure_endpoint=args.azure_endpoint,
    )

    results = []
    for fam in families:
        r = run_family(backend, fam, nights=args.nights, edit_budget=args.edit_budget,
                       rollouts_k=args.rollouts_k, gate_mode=("off" if args.gate == "off" else "on"),
                       gate_metric=args.gate_metric,
                       dream_factor=args.dream_factor, train_size=args.train_size,
                       val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                       seed=args.seed)
        results.append(r)
        if not args.json:
            print(f"  {fam:<12} split(train/val/test)={r['split']['train']}/{r['split']['val']}/{r['split']['test']}"
                  f" dream+{r['dream_train']}  TEST {r['test_before']:.2f} -> {r['test_after']:.2f}"
                  f"  ({'IMPROVED' if r['improved'] else 'no change'})")

    summary = {"benchmark": "skillopt-sleep/daily-cases", "backend": backend.name,
               "n_improved": sum(1 for r in results if r["improved"]),
               "n_families": len(results), "results": results}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"\n=== {summary['n_improved']}/{summary['n_families']} families improved on held-out TEST "
              f"(backend={backend.name}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
