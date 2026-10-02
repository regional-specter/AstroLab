#!/usr/bin/env python3
"""Validate extraction rows. Default: committed fixtures (safe on the M3).

  python3 scripts/validate_rows.py
  python3 scripts/validate_rows.py --path /tmp/cuad_qa_valid.jsonl.gz

Do not point this at a Hugging Face cache on the Mac.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPARATUS = ROOT.parent / "apparatus"
sys.path.insert(0, str(APPARATUS))

from schema import validate_row  # noqa: E402
from streaming import iter_jsonl  # noqa: E402

DEFAULT = APPARATUS / "fixtures" / "tiny_gold.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()
    bad = 0
    n = 0
    for row in iter_jsonl(args.path):
        n += 1
        errors = validate_row(row)
        if errors:
            bad += 1
            print(json.dumps({"id": row.get("id"), "errors": errors}))
        if args.limit and n >= args.limit:
            break
    print(json.dumps({"n": n, "invalid": bad, "ok": n - bad}))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
