#!/usr/bin/env python3
"""Score clause-extraction predictions against gold JSONL (CUAD Jaccard).

Mac smoke (committed fixtures, no Hub download):

  python3 scripts/eval_cuad.py

Colab / cloud (after a baseline or Unsloth run):

  python3 scripts/eval_cuad.py \\
    --gold cuad_qa_test_gold.jsonl.gz \\
    --pred preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPARATUS = ROOT.parent / "apparatus"
sys.path.insert(0, str(APPARATUS))

from metrics import KAPPA, score_rows  # noqa: E402
from streaming import iter_jsonl  # noqa: E402

DEFAULT_GOLD = APPARATUS / "fixtures" / "tiny_gold.jsonl"
DEFAULT_PRED = APPARATUS / "fixtures" / "tiny_pred.jsonl"


def load_preds(path: Path) -> dict:
    out = {}
    for row in iter_jsonl(path):
        out[row["id"]] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--kappa", type=float, default=KAPPA)
    args = parser.parse_args()
    gold = list(iter_jsonl(args.gold))
    pred = load_preds(args.pred)
    report = score_rows(gold, pred, kappa=args.kappa)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
