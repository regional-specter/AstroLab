"""Shared instruction-row shape for JSON/XML extraction papers."""

from __future__ import annotations

import json
from typing import Any

CLAUSE_KEYS = ("clause_type", "exact_quote", "confidence", "statute_code")
ROW_KEYS = ("id", "instruction", "input", "output")


def parse_output(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty output")
    return json.loads(raw)


def clauses_from_output(raw: Any) -> list[dict[str, Any]]:
    payload = parse_output(raw)
    clauses = payload.get("clauses")
    if not isinstance(clauses, list):
        raise ValueError("output JSON must contain a clauses list")
    return clauses


def validate_clause(clause: dict[str, Any]) -> None:
    missing = [key for key in CLAUSE_KEYS if key not in clause]
    if missing:
        raise ValueError(f"clause missing keys: {missing}")
    if not isinstance(clause["clause_type"], str) or not clause["clause_type"]:
        raise ValueError("clause_type must be a non-empty string")
    if not isinstance(clause["exact_quote"], str):
        raise ValueError("exact_quote must be a string")
    conf = clause["confidence"]
    if not isinstance(conf, (int, float)) or not 0 <= float(conf) <= 1:
        raise ValueError("confidence must be in [0, 1]")


def validate_row(row: dict[str, Any], require_verbatim: bool = True) -> list[str]:
    errors: list[str] = []
    for key in ROW_KEYS:
        if key not in row:
            errors.append(f"missing {key}")
    if errors:
        return errors
    try:
        clauses = clauses_from_output(row["output"])
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return [f"invalid output JSON: {exc}"]
    excerpt = row.get("input") or ""
    for i, clause in enumerate(clauses):
        try:
            validate_clause(clause)
        except ValueError as exc:
            errors.append(f"clauses[{i}]: {exc}")
            continue
        quote = clause["exact_quote"]
        if require_verbatim and quote and quote not in excerpt:
            collapsed_q = " ".join(quote.split())
            collapsed_e = " ".join(excerpt.split())
            if collapsed_q not in collapsed_e:
                errors.append(f"clauses[{i}]: exact_quote not verbatim in input")
    return errors
