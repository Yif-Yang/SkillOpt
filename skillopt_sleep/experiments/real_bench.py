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

import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from skillopt_sleep.types import TaskRecord

DATA_ROOT = "/home/azureuser/workspace-gzy/SkillReflection/data"

# ── intern's exact rollout system prompts (skillopt/envs/*/prompts) ───────────
# Carried on each TaskRecord so the backend frames the rollout EXACTLY as the
# research repo does. {skill_section} is filled by the backend with the current
# skill. These neutral prompts also avoid the content-filter 'jailbreak' trip
# that the plugin's generic OVERRIDE-style wrapper caused.
_SYS_SEARCHQA = (
    "You are an expert question answering agent.\n\n"
    "{skill_section}## Task Format\n"
    "You will receive a CONTEXT containing document passages and a QUESTION. "
    "Read the context carefully and answer the question based on the information provided.\n\n"
    "## Answer Format\n"
    "Think step by step, then provide your final answer inside <answer>...</answer> tags. "
    "Keep your answer concise -- typically a few words or a short phrase. Do not repeat the "
    "question. Do not include unnecessary explanation in the answer tags.\n\n"
    "Example:\n<answer>Abraham Lincoln</answer>\n\n"
)
_SYS_LIVEMATH = (
    "You are an expert mathematical reasoning agent solving multiple-choice questions.\n\n"
    "{skill_section}## Task Format\n"
    "You will receive one mathematics multiple-choice question and its answer choices. "
    "Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.\n\n"
    "## Answer Format\n"
    "Think step by step, then provide your final answer inside <answer>...</answer> tags. "
    "Inside the tags, output only the single choice label, such as A or C.\n\n"
    "Example:\n<answer>B</answer>\n\n"
)
# Spreadsheet: the research codegen system + its critical rules (the no-formula
# rule is essential — openpyxl never computes formulas, so a formula string is
# read back as None and scored 0 despite correct logic).
_SS_CRITICAL = (
    "## Critical Rules (MUST follow)\n"
    "1. NEVER write Excel formulas to cells that will be graded on their displayed value. "
    "openpyxl does NOT compute formulas -- the evaluator will see None. Instead, compute "
    "results in Python and write literal values (numbers/strings).\n"
    "2. Iterate over actual rows; do not hardcode cell values.\n\n"
)
_SYS_SPREADSHEET = (
    "You are an expert Python programmer specializing in spreadsheet manipulation.\n\n"
    + _SS_CRITICAL +
    "{skill_section}You will be given a user instruction and the target answer cells. "
    "Write a single self-contained Python script that reads the input workbook at the "
    "variable INPUT_PATH, performs the requested manipulation, and saves the result to "
    "OUTPUT_PATH. INPUT_PATH and OUTPUT_PATH are already defined -- do NOT reassign them. "
    "Use only the standard library, openpyxl, and pandas. Do not print anything. Do not use "
    "input(). Return ONLY the Python code inside a single ```python ... ``` fenced block.\n\n"
)

# answer-format instruction appended to each task so the model replies in <answer>
_ANSWER_FMT = (
    "\n\nThink step by step, then give your final answer inside <answer>...</answer> tags."
)
_MC_FMT = (
    "\n\nThink step by step, then give ONLY the letter of the correct choice inside "
    "<answer>...</answer> tags (e.g. <answer>A</answer>)."
)

_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _norm_label(t: str) -> str:
    return str(t).strip().upper().rstrip(".):")


def _shuffle_choices(item: dict, seed: int) -> dict:
    """Port of the intern's dataloader._shuffle_item_choices.

    The raw LiveMath data stores the correct option FIRST (always label 'A'); the
    research loader shuffles choices per item (deterministic sha256(seed:id)) and
    re-labels the correct choice to its new position, giving a uniform A/B/C/D/E
    label distribution. Skipping this lets 'always answer A' score ~1.0 — which is
    exactly the degenerate artifact we hit before. This restores comparability.
    """
    choices = item.get("choices", [])
    if not choices:
        return item
    digest = hashlib.sha256(f"{seed}:{item.get('id','')}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = [dict(c) for c in choices]
    rng.shuffle(shuffled)
    orig_correct = _norm_label((item.get("correct_choice") or {}).get("label", ""))
    new_choices: List[dict] = []
    new_correct = dict(item.get("correct_choice") or {})
    for idx, c in enumerate(shuffled):
        new_label = _CHOICE_LABELS[idx]
        new_choices.append({"label": new_label, "text": c.get("text", "")})
        if _norm_label(c.get("label", "")) == orig_correct:
            new_correct = {"label": new_label, "text": c.get("text", "")}
    out = dict(item)
    out["choices"] = new_choices
    out["correct_choice"] = new_correct
    return out


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
        system=_SYS_SEARCHQA, reference_kind="answer",
        judge={"kind": "answer", "bench": "searchqa", "gold": list(it.get("answers", []))},
        tags=["bench:searchqa"], split=split, origin="real",
    )


def _livemath_task(it: dict, split: str, seed: int = 42) -> TaskRecord:
    it = _shuffle_choices(it, seed)  # align to intern: per-item choice shuffle
    q = str(it.get("question", ""))
    choices = it.get("choices", [])
    ctext = "\n".join(f"{c.get('label')}. {c.get('text')}" for c in choices)
    intent = f"## Question\n{q}\n\n## Choices\n{ctext}{_MC_FMT}"
    return TaskRecord(
        id=f"livemath:{it.get('id','')}", project="/bench/livemath", intent=intent,
        system=_SYS_LIVEMATH, reference_kind="answer",
        judge={"kind": "answer", "bench": "livemath",
               "correct_choice": it.get("correct_choice", {}), "choices": choices},
        tags=["bench:livemath"], split=split, origin="real",
    )


def _spreadsheet_task(it: dict, split: str, data_root: str) -> TaskRecord:
    instr = str(it.get("instruction", it.get("question", "")))
    pos = it.get("answer_position", "")
    intent = (
        "# Instruction\n"
        f"{instr}\n\n"
        f"# Target answer cells\n{pos}\n\n"
        "Write Python (openpyxl/pandas) that reads INPUT_PATH, performs the "
        "manipulation, and saves to OUTPUT_PATH. Put ONLY the code in a single "
        "```python ... ``` block."
    )
    return TaskRecord(
        id=f"spreadsheet:{it.get('id','')}", project="/bench/spreadsheet", intent=intent,
        system=_SYS_SPREADSHEET, reference_kind="answer",
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
        if bench == "livemath":
            # livemath maker needs the seed for the deterministic choice shuffle
            return mk(it, split, seed=seed)
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
