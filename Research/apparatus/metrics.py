"""JSON validity and Jaccard span metrics (CUAD-style)."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import clauses_from_output, validate_clause

KAPPA = 0.5


def tokenize(text: str) -> set[str]:
    return {tok for tok in text.lower().split() if tok}


def jaccard(a: str, b: str) -> float:
    sa, sb = tokenize(a), tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def json_valid(raw: Any) -> bool:
    try:
        for clause in clauses_from_output(raw):
            validate_clause(clause)
        return True
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def match_quotes(gold: list[str], pred: list[str], kappa: float = KAPPA) -> tuple[int, int, int]:
    """Greedy 1-1 matches. Returns tp, fp, fn."""
    used: set[int] = set()
    tp = 0
    for g in gold:
        best_i, best = -1, kappa
        for i, p in enumerate(pred):
            if i in used:
                continue
            score = jaccard(g, p)
            if score >= best:
                best, best_i = score, i
        if best_i >= 0:
            used.add(best_i)
            tp += 1
    fp = len(pred) - len(used)
    fn = len(gold) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score_rows(
    gold_rows: Iterable[dict[str, Any]],
    pred_by_id: dict[str, Any],
    kappa: float = KAPPA,
) -> dict[str, Any]:
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    json_ok = 0
    n = 0
    missing = 0
    for gold in gold_rows:
        n += 1
        row_id = gold["id"]
        gold_clauses = clauses_from_output(gold["output"])
        raw_pred = pred_by_id.get(row_id)
        if raw_pred is None:
            missing += 1
            for clause in gold_clauses:
                per_type[clause["clause_type"]][2] += 1
            continue
        payload = raw_pred["output"] if isinstance(raw_pred, dict) and "output" in raw_pred else raw_pred
        if json_valid(payload):
            json_ok += 1
            pred_clauses = clauses_from_output(payload)
        else:
            pred_clauses = []
        gold_by: dict[str, list[str]] = defaultdict(list)
        pred_by: dict[str, list[str]] = defaultdict(list)
        for clause in gold_clauses:
            gold_by[clause["clause_type"]].append(clause["exact_quote"])
        for clause in pred_clauses:
            pred_by[str(clause.get("clause_type", ""))].append(str(clause.get("exact_quote", "")))
        types = set(gold_by) | set(pred_by)
        for name in types:
            tp, fp, fn = match_quotes(gold_by[name], pred_by[name], kappa=kappa)
            bucket = per_type[name]
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn
    micro_tp = sum(v[0] for v in per_type.values())
    micro_fp = sum(v[1] for v in per_type.values())
    micro_fn = sum(v[2] for v in per_type.values())
    type_f1 = []
    by_type = {}
    for name, (tp, fp, fn) in sorted(per_type.items()):
        stats = prf(tp, fp, fn)
        by_type[name] = stats
        type_f1.append(stats["f1"])
    macro_f1 = sum(type_f1) / len(type_f1) if type_f1 else 0.0
    return {
        "n": n,
        "missing_preds": missing,
        "json_validity": json_ok / n if n else 0.0,
        "micro": prf(micro_tp, micro_fp, micro_fn),
        "macro_f1": macro_f1,
        "kappa": kappa,
        "by_type": by_type,
    }
