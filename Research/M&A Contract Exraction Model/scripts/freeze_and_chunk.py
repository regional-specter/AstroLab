#!/usr/bin/env python3
"""Freeze document-level splits and chunk Hub contracts. Upload, then delete local copies."""

from __future__ import annotations

import csv
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))

HF_REPO = "Aby-ss/ma-extraction-3B-research"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
SEED = 42
MAX_TOKENS = 2048
OVERLAP = 256
STRIDE = MAX_TOKENS - OVERLAP
VALID_PER_SLICE = {
    "merger": 25,
    "commercial": 25,
    "nda": 18,
    "lease": 8,
    "other": 4,
}


def load_docs(path: Path) -> list[dict]:
    docs: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def assign_splits(docs: list[dict]) -> dict[str, str]:
    rng = random.Random(SEED)
    by_slice: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        by_slice[doc["slice"]].append(doc["id"])
    split_of: dict[str, str] = {}
    for slice_name, ids in by_slice.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_valid = VALID_PER_SLICE[slice_name]
        if n_valid > len(ids):
            raise SystemExit(f"{slice_name}: need {n_valid} valid docs, have {len(ids)}")
        for i, doc_id in enumerate(ids):
            split_of[doc_id] = "valid" if i < n_valid else "train"
    return split_of


def iter_chunks(text: str, enc: tiktoken.Encoding) -> list[str]:
    ids = enc.encode(text)
    if not ids:
        return []
    if len(ids) <= MAX_TOKENS:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(ids):
        window = ids[start : start + MAX_TOKENS]
        out.append(enc.decode(window))
        if start + MAX_TOKENS >= len(ids):
            break
        start += STRIDE
    return out


def write_splits_csv(path: Path, docs: list[dict], split_of: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "split", "slice", "source", "license", "n_chars", "n_tokens_est", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for doc in sorted(docs, key=lambda d: d["id"]):
            row = {k: doc.get(k, "") for k in fields}
            row["split"] = split_of[doc["id"]]
            w.writerow(row)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    src = Path(
        hf_hub_download(
            HF_REPO,
            "parsed/documents.jsonl.gz",
            repo_type="dataset",
            cache_dir=str(CACHE / "hf"),
        )
    )
    docs = load_docs(src)
    split_of = assign_splits(docs)
    counts = Counter(split_of[d["id"]] for d in docs)
    slice_split = Counter((d["slice"], split_of[d["id"]]) for d in docs)
    print("docs", len(docs), "splits", dict(counts), flush=True)
    for key in sorted(slice_split):
        print(f"  {key[0]} {key[1]}: {slice_split[key]}", flush=True)

    enc = tiktoken.get_encoding("cl100k_base")
    writers = {
        "train": gzip.open(CACHE / "chunks_train.jsonl.gz", "wt", encoding="utf-8"),
        "valid": gzip.open(CACHE / "chunks_valid.jsonl.gz", "wt", encoding="utf-8"),
    }
    n_chunks = {"train": 0, "valid": 0}
    try:
        for doc in docs:
            split = split_of[doc["id"]]
            pieces = iter_chunks(doc.get("text") or "", enc)
            for idx, piece in enumerate(pieces):
                rec = {
                    "chunk_id": f"{doc['id']}::{idx}",
                    "doc_id": doc["id"],
                    "split": split,
                    "slice": doc["slice"],
                    "source": doc.get("source"),
                    "chunk_index": idx,
                    "n_chunks": len(pieces),
                    "n_tokens_est": len(enc.encode(piece)),
                    "text": piece,
                }
                writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks[split] += 1
    finally:
        for w in writers.values():
            w.close()

    print("chunks", n_chunks, "total", sum(n_chunks.values()), flush=True)
    splits_csv = CACHE / "splits.csv"
    write_splits_csv(splits_csv, docs, split_of)

    api = HfApi()
    uploads = [
        (splits_csv, "manifests/splits.csv"),
        (CACHE / "chunks_train.jsonl.gz", "chunks/train.jsonl.gz"),
        (CACHE / "chunks_valid.jsonl.gz", "chunks/valid.jsonl.gz"),
    ]
    for local, dest in uploads:
        print(f"Uploading {dest} ({local.stat().st_size} bytes)…", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=dest,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Add {dest} (seed={SEED}, {MAX_TOKENS}/{OVERLAP} chunks)",
        )

    # Keep only the tiny split manifest locally; drop contract text.
    for p in (CACHE / "chunks_train.jsonl.gz", CACHE / "chunks_valid.jsonl.gz"):
        p.unlink(missing_ok=True)
    hf_dir = CACHE / "hf"
    if hf_dir.exists():
        import shutil

        shutil.rmtree(hf_dir)
    print("Done. Splits frozen and chunks are on the Hub.", flush=True)


if __name__ == "__main__":
    main()
