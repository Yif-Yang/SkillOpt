"""SkillOpt-Sleep — real benchmark loader (the intern's data + splits + scoring).

Loads the EXACT data the research blog_1 used, keeps its TEST sets full, and
reuses its evaluators (via real_eval). The ONLY deviation, per the experiment
design, is the TRAIN side: instead of the full research train split, we simulate
the online sleep+dream pipeline — sample a few real "today's tasks" and
dream-augment them. val + test stay the research splits (real, untouched).

Benchmarks (research splits): searchqa 400/200/1400, livemath 35/18/124,
spreadsheet 80/40/280. Data root defaults to the intern's path.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

from skillopt_sleep.types import TaskRecord

DATA_ROOT = "/home/azureuser/workspace-gzy/SkillReflection/data"

# answer-format instruction appended to each task so the model replies in <answer>
_ANSWER_FMT = (
    "\n\nThink step by step, then give your final answer inside <answer>...</answer> tags."
)
_MC_FMT = (
    "\n\nThink step by step, then give ONLY the letter of the correct choice inside "
    "<answer>...</answer> tags (e.g. <answer>A</answer>)."
)


def _load(path: str) -> List[dict]:
    with open(path) as f:
        d = json.load(f)
    return d if isinstance(d, list) else [d]


def _truncate(ctx: str, n: int = 6000) -> str:
    return ctx if len(ctx) <= n else ctx[:n] + "\n...[truncated]"


# ── per-benchmark item -> TaskRecord ──────────────────────────────────────────

def _searchqa_task(it: dict, split: str) -> TaskRecord:
    q = str(it.get("question", ""))
    ctx = _truncate(str(it.get("context", "")))
    intent = f"## Context\n{ctx}\n\n## Question\n{q}{_ANSWER_FMT}"
    return TaskRecord(
        id=f"searchqa:{it.get('id','')}", project="/bench/searchqa", intent=intent,
        reference_kind="answer",
        judge={"kind": "answer", "bench": "searchqa", "gold": list(it.get("answers", []))},
        tags=["bench:searchqa"], split=split, origin="real",
    )


def _livemath_task(it: dict, split: str) -> TaskRecord:
    q = str(it.get("question", ""))
    choices = it.get("choices", [])
    ctext = "\n".join(f"{c.get('label')}. {c.get('text')}" for c in choices)
    intent = f"## Question\n{q}\n\n## Choices\n{ctext}{_MC_FMT}"
    return TaskRecord(
        id=f"livemath:{it.get('id','')}", project="/bench/livemath", intent=intent,
        reference_kind="answer",
        judge={"kind": "answer", "bench": "livemath",
               "correct_choice": it.get("correct_choice", {}), "choices": choices},
        tags=["bench:livemath"], split=split, origin="real",
    )


def _spreadsheet_task(it: dict, split: str, data_root: str) -> TaskRecord:
    instr = str(it.get("instruction", it.get("question", "")))
    intent = (f"## Spreadsheet task\n{instr}{_ANSWER_FMT}")
    return TaskRecord(
        id=f"spreadsheet:{it.get('id','')}", project="/bench/spreadsheet", intent=intent,
        reference_kind="answer",
        judge={"kind": "answer", "bench": "spreadsheet", "item": it, "data_root": data_root},
        tags=["bench:spreadsheet"], split=split, origin="real",
    )


_BENCH = {
    "searchqa": {
        "sub": "searchqa_split", "files": ("train/train.json", "val/sel.json", "test/test.json"),
        "mk": _searchqa_task,
    },
    "livemath": {
        "sub": "ablation_splits/livemathematicianbench/2-1-7_seed42",
        "files": ("train/items.json", "val/items.json", "test/items.json"),
        "mk": _livemath_task,
    },
    "spreadsheet": {
        "sub": "spreadsheetbench_split", "files": ("train/train.json", "val/sel.json", "test/test.json"),
        "mk": _spreadsheet_task, "data_root_sub": "spreadsheetbench_verified_400",
    },
}

BENCHMARKS = list(_BENCH.keys())


def load_benchmark(
    bench: str, *, data_root: str = DATA_ROOT, n_train: int = 5,
    test_limit: int = 0, seed: int = 42,
) -> List[TaskRecord]:
    """Load a benchmark with the sleep protocol.

    train = ``n_train`` real items sampled from the research train split (the
    'today's tasks'); val + test = the research splits (full, unless test_limit).
    Dream augmentation is added by the runner, not here.
    """
    spec = _BENCH[bench]
    base = os.path.join(data_root, spec["sub"])
    tr_f, val_f, te_f = spec["files"]
    mk = spec["mk"]
    extra = {}
    if spec.get("data_root_sub"):
        extra["data_root"] = os.path.join(data_root, spec["data_root_sub"])

    def _mk(it, split):
        return mk(it, split, **extra) if extra else mk(it, split)

    rng = random.Random(seed)
    train_items = _load(os.path.join(base, tr_f))
    rng.shuffle(train_items)
    train = [_mk(it, "train") for it in train_items[:max(1, n_train)]]
    val = [_mk(it, "val") for it in _load(os.path.join(base, val_f))]
    test_items = _load(os.path.join(base, te_f))
    if test_limit and test_limit < len(test_items):
        # deterministic subsample of the FULL test split
        rng2 = random.Random(seed + 1)
        rng2.shuffle(test_items)
        test_items = test_items[:test_limit]
    test = [_mk(it, "test") for it in test_items]
    return train + val + test


def available(data_root: str = DATA_ROOT) -> List[str]:
    return [b for b, s in _BENCH.items() if os.path.isdir(os.path.join(data_root, s["sub"]))]
