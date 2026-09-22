#!/usr/bin/env python3
"""Teacher-extract 41 CUAD clause types from Hub chunks; keep verbatim quotes only.

Default teacher: Gemini 3.5 Flash-Lite (free-tier headroom: 15 RPM / 500 RPD).
"""

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
MODEL = os.environ.get("TEACHER_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = "gemini-3.1-flash-lite"
BATCH_COMMIT = 25
# Stay under free-tier 15 RPM / 500 RPD for Flash-Lite.
MIN_INTERVAL_SEC = 4.2
DEFAULT_DAILY_CAP = 450

SYSTEM = """You are senior M&A counsel reviewing a contract excerpt.
Extract every clause that matches the 41 CUAD types listed in the user message.
Rules:
- exact_quote MUST be a verbatim substring of the excerpt (copy-paste, no paraphrase).
- If a type is absent, omit it.
- confidence is in [0, 1].
- statute_code is an empty string unless a statute or code is explicitly named.
Return JSON only: {"clauses": [...]}
"""

CLAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause_type": {"type": "string", "enum": list(CUAD_CLAUSE_TYPES)},
                    "exact_quote": {"type": "string"},
                    "confidence": {"type": "number"},
                    "statute_code": {"type": "string"},
                },
                "required": ["clause_type", "exact_quote", "confidence"],
            },
        }
    },
    "required": ["clauses"],
}


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
        "teacher_model": MODEL,
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


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def teacher_extract(client, text: str) -> list[dict]:  # noqa: ANN001
    from google.genai import types

    types_blob = "\n".join(f"- {name}" for name in CUAD_CLAUSE_TYPES)
    user = f"Allowed clause types:\n{types_blob}\n\nExcerpt:\n{text}"
    last_err: Exception | None = None
    for model in (MODEL, FALLBACK_MODEL):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_json_schema=CLAUSE_SCHEMA,
                ),
            )
            raw = _strip_fences(resp.text or "")
            parsed = TeacherOut.model_validate_json(raw)
            keep: list[dict] = []
            for clause in parsed.clauses:
                if not verbatim(clause.exact_quote, text):
                    continue
                keep.append(clause.model_dump())
            return keep
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            msg = str(exc).lower()
            if "429" in msg or "resource exhausted" in msg or "quota" in msg:
                raise
            continue
    raise last_err or RuntimeError("teacher_extract failed")


def distill_split(split: str, client, limit: int | None, daily_cap: int) -> int:  # noqa: ANN001
    chunks = load_chunks(split)
    done = already_done()
    pending = [c for c in chunks if c["chunk_id"] not in done]
    if limit is not None:
        pending = pending[:limit]
    pending = pending[:daily_cap]
    print(
        f"{split}: {len(chunks)} chunks, {len(done)} done, "
        f"{len(pending)} this run (cap {daily_cap}), model={MODEL}",
        flush=True,
    )
    api = HfApi()
    buf: list[dict] = []
    kept = 0
    shard_idx = int(time.time()) % 100000
    out_dir = CACHE / "distilled"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_call = 0.0

    def flush() -> None:
        nonlocal shard_idx, buf, kept
        if not buf:
            return
        local = out_dir / f"{split}-{shard_idx:05d}.jsonl.gz"
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
            commit_message=f"Distill {split} shard {shard_idx} ({len(buf)} examples, {MODEL})",
        )
        local.unlink(missing_ok=True)
        kept += len(buf)
        shard_idx += 1
        buf = []

    for i, chunk in enumerate(pending, start=1):
        wait = MIN_INTERVAL_SEC - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            last_call = time.time()
            clauses = teacher_extract(client, chunk["text"])
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            print(f"  fail {chunk['chunk_id']}: {exc}", flush=True)
            if "429" in msg or "resource exhausted" in msg or "quota" in msg:
                print("Daily/minute quota hit. Flushing and stopping so you can resume tomorrow.", flush=True)
                flush()
                return kept
            time.sleep(min(30, 2 + i * 0.1))
            continue
        buf.append(alpaca_row(chunk, clauses))
        if i % 10 == 0:
            print(f"  {split} {i}/{len(pending)}", flush=True)
        if len(buf) >= BATCH_COMMIT:
            flush()
    flush()
    return kept


def concat_split(split: str) -> None:
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


def resolve_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit(
        "No Gemini key in the environment. Export GEMINI_API_KEY (do not put it in the repo), then:\n"
        "  python3 scripts/distill_clauses.py --split train\n"
        "Free tier Flash-Lite is ~500 requests/day; rerun daily until 9,698 chunks are labeled."
    )


def main() -> None:
    from google import genai

    args = sys.argv[1:]
    split = "train"
    limit = None
    freeze_only = "--freeze-only" in args
    daily_cap = DEFAULT_DAILY_CAP
    if "--split" in args:
        split = args[args.index("--split") + 1]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--daily-cap" in args:
        daily_cap = int(args[args.index("--daily-cap") + 1])
    CACHE.mkdir(parents=True, exist_ok=True)
    if not freeze_only:
        client = genai.Client(api_key=resolve_key())
        distill_split(split, client, limit, daily_cap)
    if freeze_only:
        concat_split(split)
    import shutil

    hf_dir = CACHE / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    print("Local chunk/distill cache removed.", flush=True)


if __name__ == "__main__":
    main()
