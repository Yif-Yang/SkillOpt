"""SkillOpt-Sleep — real-benchmark scoring, ported from the research repo.

These are faithful ports of the intern's per-benchmark evaluators so the plugin
scores correctness EXACTLY as the research blog does (not toy format rules):

  * searchqa  — SQuAD exact-match (em) + token-F1, gold answers
  * livemath  — multiple-choice: predicted label == correct label
  * spreadsheet — official SpreadsheetBench cell-value compare (openpyxl), lazy

The intern's `hard` score is `em` (searchqa), label-correct (livemath), or the
cell-value pass (spreadsheet). We expose score_* returning (hard, soft) where
soft = F1 for searchqa (else = hard).

Source: /home/azureuser/skillopt-main-wt/skillopt/envs/*/evaluator.py
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import List, Tuple


# ── shared answer extraction ──────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return (text or "").strip()


# ── searchqa: SQuAD EM + F1 ───────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split()).strip()


def _em(pred: str, golds: List[str]) -> float:
    np = _normalize_answer(pred)
    return 1.0 if any(_normalize_answer(g) == np for g in golds) else 0.0


def _f1(pred: str, golds: List[str]) -> float:
    np = _normalize_answer(pred)
    pt = np.split()
    best = 0.0
    for g in golds:
        gt = _normalize_answer(g).split()
        if not pt or not gt:
            best = max(best, 1.0 if pt == gt else 0.0)
            continue
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0:
            continue
        prec = ns / len(pt)
        rec = ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def score_searchqa(response: str, gold_answers: List[str]) -> Tuple[float, float, str]:
    ans = extract_answer(response)
    em = _em(ans, gold_answers)
    f1 = _f1(ans, gold_answers)
    return em, max(em, f1), f"em={em:.0f} f1={f1:.2f} pred={ans[:40]!r}"


# ── livemath: multiple-choice label ───────────────────────────────────────────

def _norm_label(t: str) -> str:
    return str(t).strip().upper().rstrip(".):")


def score_livemath(response: str, correct_choice: dict, choices: List[dict]) -> Tuple[float, float, str]:
    ans = extract_answer(response)
    label = _norm_label(ans)
    valid = {_norm_label(c.get("label", "")) for c in choices}
    if label not in valid:
        low = ans.lower()
        for c in choices:
            if str(c.get("text", "")).strip().lower() == low:
                label = _norm_label(c.get("label", ""))
                break
        else:
            toks = ans.split()
            if toks and _norm_label(toks[0]) in valid:
                label = _norm_label(toks[0])
    correct = _norm_label(correct_choice.get("label", ""))
    hard = 1.0 if label == correct else 0.0
    return hard, hard, f"pred={label} gold={correct}"


# ── spreadsheet: official cell-value compare (lazy openpyxl) ───────────────────

def score_spreadsheet(response: str, item: dict, data_root: str) -> Tuple[float, float, str]:
    """Compare the model's produced workbook cell value(s) to gold.

    The official evaluator reads the answer xlsx and compares the target cell.
    This requires openpyxl + the workbook files; imported lazily. If anything is
    unavailable, returns (0,0,'spreadsheet-eval-unavailable') so the run still
    proceeds (and the caller can skip spreadsheet).
    """
    try:
        from skillopt_sleep.experiments._spreadsheet_eval import score_case
        return score_case(response, item, data_root)
    except Exception as e:  # noqa: BLE001
        return 0.0, 0.0, f"spreadsheet-eval-unavailable: {type(e).__name__}"


def score_answer_judge(judge: dict, response: str) -> Tuple[float, float, str]:
    """Dispatch a TaskRecord.judge of kind 'answer' to the right real evaluator.

    judge = {"kind": "answer", "bench": "searchqa"|"livemath"|"spreadsheet", ...}
    """
    bench = (judge or {}).get("bench")
    if bench == "searchqa":
        return score_searchqa(response, judge.get("gold", []))
    if bench == "livemath":
        return score_livemath(response, judge.get("correct_choice", {}), judge.get("choices", []))
    if bench == "spreadsheet":
        return score_spreadsheet(response, judge.get("item", {}), judge.get("data_root", ""))
    return 0.0, 0.0, f"unknown-answer-bench={bench}"
