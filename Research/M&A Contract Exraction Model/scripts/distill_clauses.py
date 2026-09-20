#!/usr/bin/env python3
"""Teacher-extract 41 CUAD clause types from Hub chunks; keep verbatim quotes only."""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuad_labels import CUAD_CLAUSE_TYPES

HF_REPO = "Aby-ss/ma-extraction-3B-research"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
MODEL = os.environ.get("TEACHER_MODEL", "claude-sonnet-4-5-20250929")
BATCH_COMMIT = 50

SYSTEM = """You are senior M&A counsel reviewing a contract excerpt.
Extract every clause that matches the 41 CUAD types listed in the user message.
Rules:
- exact_quote MUST be a verbatim substring of the excerpt (copy-paste, no paraphrase).
- If a type is absent, omit it.
- confidence is in [0, 1].
- statute_code is an empty string unless a statute or code is explicitly named.
Return JSON only: {"clauses": [...]}
"""


class Clause(BaseModel):
    clause_type: str
    exact_quote: str
    confidence: float = Field(ge=0.0, le=1.0)
    statute_code: str = ""

    @field_validator("clause_type")
    @classmethod
    def known_type(cls, value: str) -> str:
        if value not in CUAD_CLAUSE_TYPES:
            raise ValueError(f"unknown clause_type: {value}")
        return value


class TeacherOut(BaseModel):
    clauses: list[Clause]


def normalize(text: str) -> str:
    return " ".join(text.split())


def verbatim(quote: str, chunk: str) -> bool:
    if not quote or not quote.strip():
        return False
    if quote in chunk:
        return True
    return normalize(quote) in normalize(chunk)


def alpaca_row(chunk: dict, clauses: list[dict]) -> dict:
    instruction = (
        "Extract CUAD clause spans from this contract excerpt. "
        "Return JSON {\"clauses\": [{\"clause_type\", \"exact_quote\", \"confidence\", \"statute_code\"}]}. "
        "exact_quote must be copied verbatim. Allowed clause_type values:\n"
        + "\n".join(f"- {name}" for name in CUAD_CLAUSE_TYPES)
    )
    return {
        "id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "split": chunk["split"],
        "slice": chunk["slice"],
        "instruction": instruction,
        "input": chunk["text"],
        "output": json.dumps({"clauses": clauses}, ensure_ascii=False),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": instruction + "\n\nExcerpt:\n" + chunk["text"]},
            {"role": "assistant", "content": json.dumps({"clauses": clauses}, ensure_ascii=False)},
        ],
    }


def load_chunks(split: str) -> list[dict]:
    path = hf_hub_download(
        HF_REPO,
        f"chunks/{split}.jsonl.gz",
        repo_type="dataset",
        cache_dir=str(CACHE / "hf"),
    )
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def already_done() -> set[str]:
    done: set[str] = set()
    api = HfApi()
    try:
        files = [
            s.path
            for s in api.list_repo_tree(HF_REPO, "distilled", repo_type="dataset", recursive=True)
            if s.path.endswith(".jsonl.gz")
        ]
    except Exception:
        return done
    CACHE.mkdir(parents=True, exist_ok=True)
    for repo_path in files:
        local = hf_hub_download(HF_REPO, repo_path, repo_type="dataset", cache_dir=str(CACHE / "hf"))
        with gzip.open(local, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    return done


def teacher_extract(client, text: str) -> list[dict]:  # noqa: ANN001
    types = "\n".join(f"- {name}" for name in CUAD_CLAUSE_TYPES)
    user = f"Allowed clause types:\n{types}\n\nExcerpt:\n{text}"
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(block.text for block in resp.content if getattr(block, "text", None))
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    parsed = TeacherOut.model_validate_json(raw)
    keep: list[dict] = []
    for clause in parsed.clauses:
        if not verbatim(clause.exact_quote, text):
            continue
        keep.append(clause.model_dump())
    return keep


def distill_split(split: str, client, limit: int | None) -> int:  # noqa: ANN001
    chunks = load_chunks(split)
    done = already_done()
    pending = [c for c in chunks if c["chunk_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    print(f"{split}: {len(chunks)} chunks, {len(done)} done, {len(pending)} to label", flush=True)
    api = HfApi()
    buf: list[dict] = []
    kept = 0
    shard_idx = 0
    out_dir = CACHE / "distilled"
    out_dir.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        nonlocal shard_idx, buf, kept
        if not buf:
            return
        local = out_dir / f"{split}-{shard_idx:04d}.jsonl.gz"
        with gzip.open(local, "wt", encoding="utf-8") as f:
            for row in buf:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        dest = f"distilled/{split}/{local.name}"
        print(f"Uploading {dest} ({len(buf)} rows)…", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=dest,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Distill {split} shard {shard_idx} ({len(buf)} examples)",
        )
        local.unlink(missing_ok=True)
        kept += len(buf)
        shard_idx += 1
        buf = []

    for i, chunk in enumerate(pending, start=1):
        try:
            clauses = teacher_extract(client, chunk["text"])
        except Exception as exc:  # noqa: BLE001
            print(f"  fail {chunk['chunk_id']}: {exc}", flush=True)
            time.sleep(2)
            continue
        buf.append(alpaca_row(chunk, clauses))
        if i % 10 == 0:
            print(f"  {split} {i}/{len(pending)}", flush=True)
        if len(buf) >= BATCH_COMMIT:
            flush()
    flush()
    return kept


def concat_split(split: str) -> None:
    """Merge distilled shards into splits/{split}.jsonl.gz on the Hub."""
    api = HfApi()
    files = [
        s.path
        for s in api.list_repo_tree(HF_REPO, f"distilled/{split}", repo_type="dataset", recursive=True)
        if s.path.endswith(".jsonl.gz")
    ]
    files.sort()
    CACHE.mkdir(parents=True, exist_ok=True)
    merged = CACHE / f"{split}.jsonl.gz"
    n = 0
    with gzip.open(merged, "wt", encoding="utf-8") as out:
        for repo_path in files:
            local = hf_hub_download(HF_REPO, repo_path, repo_type="dataset", cache_dir=str(CACHE / "hf"))
            with gzip.open(local, "rt", encoding="utf-8") as inp:
                for line in inp:
                    if line.strip():
                        out.write(line)
                        n += 1
    api.upload_file(
        path_or_fileobj=str(merged),
        path_in_repo=f"splits/{split}.jsonl.gz",
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=f"Freeze {split} split ({n} verified examples)",
    )
    merged.unlink(missing_ok=True)
    print(f"froze splits/{split}.jsonl.gz ({n} rows)", flush=True)


def main() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it, then rerun:\n"
            "  python3 scripts/distill_clauses.py --split train\n"
            "Do not store the key in the repo."
        )
    import anthropic

    client = anthropic.Anthropic(api_key=key)
    args = sys.argv[1:]
    split = "train"
    limit = None
    freeze_only = "--freeze-only" in args
    if "--split" in args:
        split = args[args.index("--split") + 1]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    CACHE.mkdir(parents=True, exist_ok=True)
    if not freeze_only:
        distill_split(split, client, limit)
    concat_split(split)
    import shutil

    hf_dir = CACHE / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    print("Local chunk/distill cache removed.", flush=True)


if __name__ == "__main__":
    main()
