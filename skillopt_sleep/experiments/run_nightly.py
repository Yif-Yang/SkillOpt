"""SkillOpt-Sleep — enhanced multi-night experiment (cumulative train + dream).

Per the enhanced-experiment spec:

  * nights = 5; each night adds 10 NEW real "today's tasks" so the cumulative
    train set grows 10 -> 20 -> 30 -> 40 -> 50.
  * each night, the 10 new tasks are enriched with many dream rollout GROUPS
    (multi-rollout contrastive reflection — ``rollouts_k`` large) since 10 real
    items is too few to learn a stable skill from.
  * the skill carries over: night N initializes from night N-1's skill and
    REFINES it on the new data.
  * two variants run side by side:
      - ``no-gate`` (the core daily-use design): take the final skill directly,
        no validation set required — this matches a user who can't build a
        per-task val set. Quality is judged ONLY on the real test set.
      - ``gate`` (paper-aligned): validation-gated; an edit must improve the
        held-out val slice to be kept.
  * we record the TEST score after every night for BOTH variants, so the report
    answers: does test accuracy climb Night 1 -> 5, with vs without the gate?

Run on the high-throughput Responses endpoints (gpt4v-scus + gpt4v-swc) via the
logged-in Azure CLI identity:

  PY=/home/azureuser/workspace-gzy/miniconda3/envs/reflact/bin/python
  SKILLOPT_SLEEP_WORKERS=24 PYTHONPATH=. $PY -m skillopt_sleep.experiments.run_nightly \
    --benchmarks searchqa,livemath,spreadsheet --model gpt-5.5 \
    --nights 5 --per-night 10 --rollouts 5 --json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

from skillopt_sleep.backend import build_backend
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.experiments.daily_cases import dream_augment
from skillopt_sleep.experiments.real_bench import _BENCH, DATA_ROOT, _load
from skillopt_sleep.replay import aggregate_scores, replay_batch
from skillopt_sleep.types import TaskRecord


def _score_test(backend, test_tasks, skill, memory="") -> float:
    if not test_tasks:
        return 0.0
    h, _ = aggregate_scores(replay_batch(backend, test_tasks, skill, memory))
    return h


def _make_tasks(bench: str, items: List[dict], split: str, data_root: str, seed: int) -> List[TaskRecord]:
    spec = _BENCH[bench]
    mk = spec["mk"]
    extra = {}
    if spec.get("data_root_sub"):
        extra["data_root"] = os.path.join(data_root, spec["data_root_sub"])
    out = []
    for it in items:
        if bench == "livemath":
            out.append(mk(it, split, seed=seed))
        elif extra:
            out.append(mk(it, split, **extra))
        else:
            out.append(mk(it, split))
    return out


def run_nightly(backend, bench, *, nights, per_night, rollouts, edit_budget,
                gate_mode, dream_factor, test_limit, seed, data_root,
                gate_val_cap=60) -> dict:
    """One variant (gate on or off) of the multi-night cumulative experiment."""
    data_root = data_root or DATA_ROOT
    spec = _BENCH[bench]
    base = os.path.join(data_root, spec["sub"])
    tr_f, val_f, te_f = spec["files"]

    # full pools
    rng = random.Random(seed)
    train_pool = _load(os.path.join(base, tr_f))
    rng.shuffle(train_pool)
    val_items = _load(os.path.join(base, val_f))
    # the gate only needs a reliable signal, not the entire val split. Cap it so
    # the gated variant stays affordable (full searchqa val is 200 -> ~40s/pass,
    # scored several times per night). The no-gate variant ignores val entirely.
    if gate_val_cap and gate_val_cap < len(val_items):
        rv = random.Random(seed + 7)
        rv.shuffle(val_items)
        val_items = val_items[:gate_val_cap]
    test_items = _load(os.path.join(base, te_f))
    if test_limit and test_limit < len(test_items):
        r2 = random.Random(seed + 1)
        r2.shuffle(test_items)
        test_items = test_items[:test_limit]

    val_tasks = _make_tasks(bench, val_items, "val", data_root, seed)
    test_tasks = _make_tasks(bench, test_items, "test", data_root, seed)

    # baseline (empty skill) on TEST
    base_test = _score_test(backend, test_tasks, "")
    nights_log = [{"night": 0, "n_train": 0, "test_hard": round(base_test, 4),
                   "action": "baseline", "accepted": False}]

    skill = ""
    used = 0
    for night in range(1, nights + 1):
        # this night's 10 NEW real tasks (cumulative pool grows as we consume)
        new_items = train_pool[used: used + per_night]
        used += per_night
        if not new_items:
            break
        new_real = _make_tasks(bench, new_items, "train", data_root, seed)
        # dream: many rollout groups happen inside consolidate via rollouts_k;
        # we ALSO add light synthetic variants so the train pool is richer.
        train_tasks = list(new_real)
        if dream_factor > 0:
            train_tasks = train_tasks + dream_augment(new_real, factor=dream_factor)

        # consolidate: reflect over this night's (real+dream) tasks, refine the
        # carried-over skill; gate on the val slice (or greedily if gate off).
        night_tasks = train_tasks + val_tasks  # consolidate splits by .split
        res = consolidate(
            backend, night_tasks, skill, "",
            edit_budget=edit_budget, gate_metric="hard", gate_mode=gate_mode,
            rollouts_k=rollouts, evolve_skill=True, evolve_memory=False, night=night,
        )
        if res.accepted:
            skill = res.new_skill
        # measure TEST after this night (the real progression signal)
        test_after = _score_test(backend, test_tasks, skill)
        nights_log.append({
            "night": night, "n_train": used,
            "n_dream": sum(1 for t in train_tasks if t.origin == "dream"),
            "val_hard": round(res.holdout_candidate, 4),
            "test_hard": round(test_after, 4),
            "action": res.gate_action, "accepted": res.accepted,
            "n_edits": len(res.applied_edits),
        })

    final_test = nights_log[-1]["test_hard"]
    return {
        "benchmark": bench, "gate": gate_mode,
        "nights": nights, "per_night": per_night, "rollouts": rollouts,
        "n_val": len(val_tasks), "n_test": len(test_tasks),
        "test_baseline": round(base_test, 4), "test_final": round(final_test, 4),
        "delta": round(final_test - base_test, 4),
        "progression": [n["test_hard"] for n in nights_log],
        "nights_log": nights_log,
        "tokens": backend.tokens_used(),
        "final_skill_tail": skill[-500:],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SkillOpt-Sleep enhanced multi-night experiment")
    ap.add_argument("--backend", default="azure-responses")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--azure-endpoint", default="", help="comma-separated; default = gpt4v scus+swc")
    ap.add_argument("--benchmarks", default="searchqa,livemath,spreadsheet")
    ap.add_argument("--nights", type=int, default=5)
    ap.add_argument("--per-night", type=int, default=10)
    ap.add_argument("--rollouts", type=int, default=5, help="dream rollout groups per task")
    ap.add_argument("--dream-factor", type=int, default=2, help="synthetic variants per new task")
    ap.add_argument("--edit-budget", type=int, default=4)
    ap.add_argument("--gate", default="both", choices=["on", "off", "both"])
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--gate-val-cap", type=int, default=60, help="cap val size for the gated variant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--out", default="", help="write JSON here")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    data_root = args.data_root or DATA_ROOT
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    gates = ["off", "on"] if args.gate == "both" else [args.gate]

    results = []
    for bench in benches:
        for gate_mode in gates:
            backend = build_backend(backend=args.backend, model=args.model,
                                    azure_endpoint=args.azure_endpoint)
            r = run_nightly(backend, bench, nights=args.nights, per_night=args.per_night,
                            rollouts=args.rollouts, edit_budget=args.edit_budget,
                            gate_mode=gate_mode, dream_factor=args.dream_factor,
                            test_limit=args.test_limit, seed=args.seed, data_root=data_root,
                            gate_val_cap=args.gate_val_cap)
            results.append(r)
            if not args.json:
                prog = " -> ".join(f"{x:.3f}" for x in r["progression"])
                print(f"  {bench:<12} gate={gate_mode:<3} test {r['test_baseline']:.3f} "
                      f"=> {r['test_final']:.3f} (Δ{r['delta']:+.3f})  [{prog}]")

    summary = {"experiment": "skillopt-sleep/nightly", "model": args.model,
               "results": results}
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(payload)
    if args.json:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
