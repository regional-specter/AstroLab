#!/usr/bin/env python3
"""Convert official CUAD-QA labels into the same JSON extraction rows as distillation.

No teacher API. Train/valid come from CUAD train contracts only. Official CUAD test stays eval-only.

Run convert/upload on Colab or any machine you will wipe — not as a long-lived cache on the Air.

  python3 scripts/convert_cuad_qa.py              # train/valid + upload
  python3 scripts/convert_cuad_qa.py --upload-only
  python3 scripts/convert_cuad_qa.py --eval-test  # official 102-contract gold, eval-only
"""

from __future__ import annotations

import gzip
import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import tiktoken
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuad_labels import CUAD_CLAUSE_TYPES

HF_REPO = "Aby-ss/ma-extraction-3B-research"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
SEED = 42
MAX_TOKENS = 2048
OVERLAP = 256
STRIDE = MAX_TOKENS - OVERLAP
VALID_FRACTION = 0.10

SYSTEM = """You are senior M&A counsel reviewing a contract excerpt.
Extract every clause that matches the 41 CUAD types listed in the user message.
Rules:
- exact_quote MUST be a verbatim substring of the excerpt (copy-paste, no paraphrase).
- If a type is absent, omit it.
- confidence is in [0, 1].
- statute_code is an empty string unless a statute or code is explicitly named.
Return JSON only: {"clauses": [...]}
"""

ALIASES = {
    "rofrroforofn": "Right of First Refusal, Offer or Negotiation (ROFR/ROFO/ROFN)",
    "noticeperiodtoterminaterenewal": "Notice to Terminate Renewal",
    "noticetoterminaterenewal": "Notice to Terminate Renewal",
    "pricerestrictions": "Price Restriction",
    "pricerestriction": "Price Restriction",
    "affiliatelicenselicensor": "Affiliate IP License-Licensor",
    "affiliateiplicenselicensor": "Affiliate IP License-Licensor",
    "affiliatelicenselicensee": "Affiliate IP License-Licensee",
    "affiliateiplicenselicensee": "Affiliate IP License-Licensee",
    "unlimitedallyoucaneatlicense": "Unlimited/All-You-Can-Eat License",
    "ipownershipassignment": "IP Ownership Assignment",
    "caponliability": "Cap on Liability",
}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def type_lookup() -> dict[str, str]:
    mapping = {norm(name): name for name in CUAD_CLAUSE_TYPES}
    mapping.update(ALIASES)
    return mapping


def clause_type_from_question(question: str, lookup: dict[str, str]) -> str | None:
    match = re.search(r'related to "([^"]+)"', question)
    raw = match.group(1) if match else question
    return lookup.get(norm(raw))


def iter_chunks(text: str, enc: tiktoken.Encoding) -> list[str]:
    ids = enc.encode(text)
    if not ids:
        return []
    if len(ids) <= MAX_TOKENS:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(ids):
        out.append(enc.decode(ids[start : start + MAX_TOKENS]))
        if start + MAX_TOKENS >= len(ids):
            break
        start += STRIDE
    return out


def alpaca_row(chunk_id: str, doc_id: str, split: str, text: str, clauses: list[dict]) -> dict:
    instruction = (
        "Extract CUAD clause spans from this contract excerpt. "
        "Return JSON {\"clauses\": [{\"clause_type\", \"exact_quote\", \"confidence\", \"statute_code\"}]}. "
        "exact_quote must be copied verbatim. Allowed clause_type values:\n"
        + "\n".join(f"- {name}" for name in CUAD_CLAUSE_TYPES)
    )
    payload = json.dumps({"clauses": clauses}, ensure_ascii=False)
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "split": split,
        "slice": "cuad",
        "source": "cuad-qa",
        "teacher_model": "cuad-expert-labels",
        "instruction": instruction,
        "input": text,
        "output": payload,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": instruction + "\n\nExcerpt:\n" + text},
            {"role": "assistant", "content": payload},
        ],
    }


def load_contracts(zip_path: Path, json_name: str) -> list[dict]:
    lookup = type_lookup()
    with zipfile.ZipFile(zip_path) as zf:
        payload = json.loads(zf.read(json_name))
    docs: list[dict] = []
    unknown: set[str] = set()
    for item in payload["data"]:
        title = item["title"]
        spans: list[tuple[str, str]] = []
        context = item["paragraphs"][0]["context"]
        for qa in item["paragraphs"][0]["qas"]:
            clause_type = clause_type_from_question(qa["question"], lookup)
            if clause_type is None:
                unknown.add(qa["question"][:80])
                continue
            for answer in qa.get("answers") or []:
                quote = (answer.get("text") or "").strip()
                if quote:
                    spans.append((clause_type, quote))
        docs.append({"id": title, "text": context, "spans": spans})
    if unknown:
        raise SystemExit(f"Unmapped CUAD questions: {sorted(unknown)[:10]}")
    return docs


def assign_splits(doc_ids: list[str]) -> dict[str, str]:
    rng = random.Random(SEED)
    ids = sorted(doc_ids)
    rng.shuffle(ids)
    n_valid = max(1, round(len(ids) * VALID_FRACTION))
    return {doc_id: ("valid" if i < n_valid else "train") for i, doc_id in enumerate(ids)}


def convert(docs: list[dict], split_of: dict[str, str], enc: tiktoken.Encoding) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for doc in docs:
        split = split_of[doc["id"]]
        pieces = iter_chunks(doc["text"], enc)
        for idx, piece in enumerate(pieces):
            seen: set[tuple[str, str]] = set()
            clauses: list[dict] = []
            for clause_type, quote in doc["spans"]:
                key = (clause_type, quote)
                if key in seen:
                    continue
                if quote not in piece and " ".join(quote.split()) not in " ".join(piece.split()):
                    continue
                seen.add(key)
                clauses.append(
                    {
                        "clause_type": clause_type,
                        "exact_quote": quote,
                        "confidence": 1.0,
                        "statute_code": "",
                    }
                )
            rows[split].append(
                alpaca_row(
                    f"cuad:{doc['id']}::{idx}",
                    doc["id"],
                    split,
                    piece,
                    clauses,
                )
            )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_files(files: dict[str, Path], message: str) -> None:
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing local splits: {missing}")
    api = HfApi()
    for dest, local in files.items():
        print(f"Uploading {dest} ({local.stat().st_size} bytes)…", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=dest,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=message,
        )


def upload_splits() -> None:
    files = {
        "splits/cuad_qa_train.jsonl.gz": CACHE / "cuad_qa_train.jsonl.gz",
        "splits/cuad_qa_valid.jsonl.gz": CACHE / "cuad_qa_valid.jsonl.gz",
    }
    upload_files(files, "Add CUAD-QA expert-labeled train/valid JSON extraction splits")
    print("Uploaded CUAD-QA splits.", flush=True)


def convert_test(docs: list[dict], enc: tiktoken.Encoding) -> list[dict]:
    split_of = {doc["id"]: "test" for doc in docs}
    return convert(docs, split_of, enc)["test"]


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if "--upload-only" in sys.argv:
        upload_splits()
        return
    if "--eval-test" in sys.argv:
        zip_path = CACHE / "cuad_data.zip"
        if not zip_path.exists():
            print("Downloading CUAD data.zip…", flush=True)
            urlretrieve(CUAD_ZIP_URL, zip_path)
        docs = load_contracts(zip_path, "test_separate_questions.json")
        enc = tiktoken.get_encoding("cl100k_base")
        rows = convert_test(docs, enc)
        dest_local = CACHE / "cuad_qa_test_gold.jsonl.gz"
        write_jsonl(dest_local, rows)
        print(f"CUAD test contracts={len(docs)} gold_chunks={len(rows)}", flush=True)
        upload_files(
            {"splits/cuad_qa_test_gold.jsonl.gz": dest_local},
            "Add CUAD official test gold JSON extraction split (eval-only)",
        )
        dest_local.unlink(missing_ok=True)
        zip_path.unlink(missing_ok=True)
        print("Done. Test gold is on the Hub. Do not train on it.", flush=True)
        return
    zip_path = CACHE / "cuad_data.zip"
    if not zip_path.exists():
        print("Downloading CUAD data.zip…", flush=True)
        urlretrieve(CUAD_ZIP_URL, zip_path)
    docs = load_contracts(zip_path, "train_separate_questions.json")
    split_of = assign_splits([d["id"] for d in docs])
    enc = tiktoken.get_encoding("cl100k_base")
    rows = convert(docs, split_of, enc)
    print(
        f"CUAD train contracts={len(docs)} "
        f"train_chunks={len(rows['train'])} valid_chunks={len(rows['valid'])} "
        f"train_with_clauses={sum(1 for r in rows['train'] if '[]' not in r['output'][:20])} "
        f"valid_with_clauses={sum(1 for r in rows['valid'] if json.loads(r['output'])['clauses'])}",
        flush=True,
    )
    files = {
        "splits/cuad_qa_train.jsonl.gz": CACHE / "cuad_qa_train.jsonl.gz",
        "splits/cuad_qa_valid.jsonl.gz": CACHE / "cuad_qa_valid.jsonl.gz",
    }
    write_jsonl(files["splits/cuad_qa_train.jsonl.gz"], rows["train"])
    write_jsonl(files["splits/cuad_qa_valid.jsonl.gz"], rows["valid"])
    api = HfApi()
    for dest, local in files.items():
        print(f"Uploading {dest} ({local.stat().st_size} bytes)…", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=dest,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message="Add CUAD-QA expert-labeled train/valid JSON extraction splits",
        )
        local.unlink(missing_ok=True)
    zip_path.unlink(missing_ok=True)
    print("Done. CUAD-QA splits are on the Hub. Official CUAD test was not used for training.", flush=True)


if __name__ == "__main__":
    main()
