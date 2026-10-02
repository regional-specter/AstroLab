"""O(1) readers for jsonl / jsonl.gz. Do not slurp corpora into lists unless scoring a small eval set."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
