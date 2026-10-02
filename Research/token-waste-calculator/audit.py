#!/usr/bin/env python3
"""Trace audit stub. No model download. Safe to run on the M3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify(prompt: str) -> str:
    text = prompt.lower()
    if any(word in text for word in ("json", "extract", "schema", "xml", "invoice", "clause")):
        return "deterministic_extraction"
    return "needs_frontier"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate token waste on simple extraction traces")
    parser.add_argument("path", type=Path, help="JSONL with a prompt or messages field")
    args = parser.parse_args()
    n = 0
    waste = 0
    with args.path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("input") or ""
            if not prompt and isinstance(row.get("messages"), list):
                prompt = " ".join(m.get("content", "") for m in row["messages"])
            n += 1
            if classify(prompt) == "deterministic_extraction":
                waste += 1
    pct = 100.0 * waste / n if n else 0.0
    print(json.dumps({"n": n, "deterministic_extraction": waste, "pct": round(pct, 1)}, indent=2))


if __name__ == "__main__":
    main()
