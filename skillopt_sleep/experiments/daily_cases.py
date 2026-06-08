"""SkillOpt-Sleep — academic "daily case" benchmark suite.

Borrows the *settings* of the SkillOpt paper's ablation study (4:1:5
train/selection/test split, the same task families: math, spreadsheet, search-QA)
and turns them into self-contained "daily cases" the sleep cycle can train on.
The paper's actual QA/spreadsheet content is not shipped in the repo (only
id-manifests), so these cases are simulated but realistic, with **programmatic
rule judges** so scoring needs no judge model.

Each family models a deliberately deficient initial skill plus tasks whose judge
encodes the house rule the optimizer must learn:

  * math      — answers must be wrapped in \\boxed{...} (paper's verifier-facing
                format requirement); deficient skill answers in prose.
  * spreadsheet — formulas must start with '=' and reference a cell range;
                deficient skill returns plain arithmetic.
  * searchqa  — answers must cite a source as [DOC n]; deficient skill answers
                from memory with no citation.

These map onto the paper's "the agent consistently ... writes an answer in the
wrong format, or fails to verify a tool result" failure modes (Section 3.4).

This mirrors the user's intended design: the TRAIN split can be dream-augmented
(synthetic variants), while VAL and TEST stay real/held-out.
"""
from __future__ import annotations

from typing import Dict, List

from skillopt_sleep.types import TaskRecord


# ── deficient initial skills (one per family) ─────────────────────────────────

DEFICIENT_SKILLS: Dict[str, str] = {
    "math": (
        "# Math Solver\n\nSolve the problem and give the final numeric answer. "
        "Show your reasoning briefly.\n"
    ),
    "spreadsheet": (
        "# Spreadsheet Helper\n\nAnswer the spreadsheet question with the value "
        "or computation the user asks for.\n"
    ),
    "searchqa": (
        "# Search QA\n\nAnswer the question concisely and directly.\n"
    ),
}

# the rule each family's judge enforces (the "house rule" to be learned)
_JUDGE: Dict[str, dict] = {
    "math": {"kind": "rule", "checks": [{"op": "regex", "arg": r"\\boxed\{"}]},
    "spreadsheet": {"kind": "rule", "checks": [
        {"op": "regex", "arg": r"^\s*="},                 # formula starts with =
        {"op": "regex", "arg": r"[A-Z]+\d+(:[A-Z]+\d+)?"},  # references a cell/range
    ]},
    "searchqa": {"kind": "rule", "checks": [{"op": "regex", "arg": r"\[DOC\s*\d+\]"}]},
}

# task prompts per family (kept distinct so train/val/test don't overlap content)
_PROMPTS: Dict[str, List[str]] = {
    "math": [
        "A train travels 120 km in 2 hours. What is its average speed in km/h?",
        "If 3x + 7 = 22, what is x?",
        "What is 15% of 240?",
        "A rectangle is 8 by 5. What is its area?",
        "The sum of two consecutive integers is 27. What is the larger one?",
        "A shirt costs $40 after a 20% discount. What was the original price?",
        "How many seconds are in 3.5 hours?",
        "What is the greatest common divisor of 48 and 36?",
        "A circle has radius 7. What is its circumference (use 3.14)?",
        "If a car uses 6 liters per 100 km, how many liters for 250 km?",
        "What is the median of 3, 9, 4, 1, 7?",
        "Factor: what are the prime factors of 84?",
        "A recipe for 4 serves needs 300g flour. How much for 10 serves?",
        "What is 2 to the power of 10?",
        "A 25% tip on a $48 bill is how many dollars?",
        "If you save $35 a week, how much in a year (52 weeks)?",
        "What is the area of a triangle with base 10 and height 6?",
        "Convert 5 kilometers to meters.",
        "The angles of a triangle are 40 and 75 degrees. What is the third?",
        "What is the average of 12, 18, and 30?",
    ],
    "spreadsheet": [
        "Sum the values in cells B2 through B10.",
        "Average the values in column C, rows 1 to 50.",
        "Count the non-empty cells in A1:A100.",
        "Find the maximum value in D2:D20.",
        "Look up the price for SKU in cell E5 from the table A1:B200.",
        "Compute the running total of B2 down to the current row.",
        "Return the value in C7 if A7 is greater than 100, else 0.",
        "Concatenate the first and last name in A2 and B2.",
        "Round the value in F3 to two decimals.",
        "Count how many cells in G1:G500 are greater than the average.",
        "Get the minimum order date in H2:H300.",
        "Sum B2:B100 only where the category in A2:A100 equals 'Books'.",
        "Return the number of unique values in column J.",
        "Compute year-over-year growth between cells K10 and K22.",
        "Pull the 3rd column from a VLOOKUP of L4 in M1:Q400.",
        "Compute the standard deviation of N2:N40.",
        "Flag rows where the amount in O2:O200 exceeds 1000.",
        "Compute the weighted average of P2:P10 weighted by Q2:Q10.",
        "Return today's date minus the date in R5 in days.",
        "Sum the top 5 values in S2:S100.",
    ],
    "searchqa": [
        "Who wrote the novel that introduced the character Sherlock Holmes?",
        "What year did the first manned moon landing occur?",
        "Which element has the chemical symbol Fe?",
        "What is the capital of Australia?",
        "Who painted the ceiling of the Sistine Chapel?",
        "What is the longest river in South America?",
        "Which company developed the first commercial graphical web browser?",
        "What is the speed of light in vacuum, approximately?",
        "Who proposed the theory of general relativity?",
        "What is the largest planet in our solar system?",
        "Which treaty ended the First World War?",
        "What programming language was created by Guido van Rossum?",
        "What is the smallest prime number greater than 50?",
        "Who is credited with inventing the telephone?",
        "What is the currency used in Japan?",
        "Which ocean is the deepest?",
        "What gas do plants primarily absorb during photosynthesis?",
        "Who wrote the play 'Romeo and Juliet'?",
        "What is the boiling point of water at sea level in Celsius?",
        "Which planet is known as the Red Planet?",
    ],
}

FAMILIES = list(_PROMPTS.keys())


def make_tasks(family: str) -> List[TaskRecord]:
    """Return all real tasks for a family (origin='real'); caller splits them."""
    judge = _JUDGE[family]
    out: List[TaskRecord] = []
    for i, q in enumerate(_PROMPTS[family]):
        out.append(TaskRecord(
            id=f"{family}_{i:02d}", project=f"/daily/{family}", intent=q,
            reference_kind="rule", judge=judge, tags=[f"family:{family}"],
            origin="real",
        ))
    return out


def dream_augment(real_tasks: List[TaskRecord], *, factor: int = 1) -> List[TaskRecord]:
    """Create synthetic TRAIN variants from real tasks (origin='dream').

    A light, deterministic augmentation: rephrase the prompt with a neutral
    wrapper. These never enter val/test (guaranteed by assign_splits). This is
    the "dream more on the training set" idea — synthetic experience to learn the
    format rule from, without overfitting the real held-out cases.
    """
    out: List[TaskRecord] = []
    wrappers = [
        "(quick one) {q}",
        "Please handle this request: {q}",
        "For the daily report: {q}",
    ]
    for t in real_tasks:
        for k in range(max(0, factor)):
            w = wrappers[k % len(wrappers)]
            out.append(TaskRecord(
                id=f"{t.id}_dream{k}", project=t.project,
                intent=w.format(q=t.intent),
                reference_kind=t.reference_kind, judge=dict(t.judge),
                tags=t.tags + ["dream"], origin="dream", derived_from=t.id,
            ))
    return out
