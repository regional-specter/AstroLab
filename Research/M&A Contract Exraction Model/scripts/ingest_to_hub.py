#!/usr/bin/env python3
"""Stage source contracts in a small cache, push to Hugging Face, then delete local copies."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import random
import re
import sys
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "Aby-ss/ma-extraction-3B-research"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
SEED = 42
USER_AGENT = "AstroLab-MA-research huggingface.co/Aby-ss"

QUOTAS = {
    "merger": 250,  # 152 MAUD + 98 EDGAR
    "nda": 180,
    "commercial": 250,
    "lease": 80,
    "other": 40,
}

EDGAR_QUERIES = {
    "merger": [
        ('"agreement and plan of merger"', "8-K"),
        ('"plan of merger"', "8-K"),
        ("EX-2.1", "8-K"),
    ],
    "commercial": [
        ('"supply agreement"', "8-K"),
        ('"license agreement"', "8-K"),
        ('"distribution agreement"', "8-K"),
        ('"master services agreement"', "8-K"),
        ('"service agreement"', "10-K"),
    ],
    "lease": [
        ('"lease agreement"', "8-K"),
        ('"office lease"', "10-K"),
        ('"real property lease"', "10-K"),
    ],
    "other": [
        ('"joint venture agreement"', "8-K"),
        ('"asset purchase agreement"', "8-K"),
        ('"employment agreement"', "8-K"),
    ],
}

MIN_CHARS = {
    "merger": 8000,
    "nda": 2500,
    "commercial": 4000,
    "lease": 4000,
    "other": 3000,
}


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def html_to_text(raw: str) -> str:
    parser = _HTMLText()
    try:
        parser.feed(raw)
        parser.close()
        text = " ".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def http_get(url: str, timeout: int = 25) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.write_bytes(http_get(url))
    return dest


def record(
    doc_id: str,
    slice_name: str,
    source: str,
    license_id: str,
    text: str,
    extra: dict | None = None,
) -> dict:
    return {
        "id": doc_id,
        "slice": slice_name,
        "source": source,
        "license": license_id,
        "n_chars": len(text),
        "n_tokens_est": len(text.split()),
        "sha256": sha256_text(text),
        "text": text,
        **(extra or {}),
    }


def load_maud(seen: set[str]) -> list[dict]:
    """Prefer full agreement files in the Zenodo zip; fall back to concatenated deal-point text."""
    zpath = download(
        "https://zenodo.org/records/7500064/files/maud_v1.zip?download=1",
        CACHE / "maud_v1.zip",
    )
    docs: list[dict] = []
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        text_like = [
            n
            for n in names
            if n.lower().endswith((".txt", ".html", ".htm")) and not n.endswith("/")
        ]
        if text_like:
            for name in text_like:
                raw = zf.read(name)
                try:
                    body = raw.decode("utf-8")
                except UnicodeDecodeError:
                    body = raw.decode("latin-1")
                text = html_to_text(body) if name.lower().endswith((".html", ".htm")) else body
                if len(text) < MIN_CHARS["merger"]:
                    continue
                rec = record(
                    f"maud:{Path(name).stem}",
                    "merger",
                    "maud",
                    "cc-by-4.0",
                    text,
                    {"path": name},
                )
                if rec["sha256"] in seen:
                    continue
                seen.add(rec["sha256"])
                docs.append(rec)
        else:
            # Annotation-only zip: stitch unique deal-point spans per agreement id.
            buckets: dict[str, list[str]] = {}
            for name in names:
                if not name.lower().endswith((".json", ".jsonl", ".csv")):
                    continue
                raw = zf.read(name)
                if name.endswith(".jsonl"):
                    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
                elif name.endswith(".json"):
                    payload = json.loads(raw.decode("utf-8"))
                    rows = payload if isinstance(payload, list) else [payload]
                else:
                    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    key = str(
                        row.get("agreement")
                        or row.get("contract_id")
                        or row.get("filename")
                        or row.get("id")
                        or name
                    )
                    span = (
                        row.get("text")
                        or row.get("deal_point_text")
                        or row.get("context")
                        or row.get("passage")
                        or ""
                    )
                    if isinstance(span, str) and len(span) > 80:
                        buckets.setdefault(key, []).append(span)
            for key, spans in buckets.items():
                # Unique-ish concatenation; not a substitute for full EDGAR text.
                uniq: list[str] = []
                seen_span: set[str] = set()
                for span in spans:
                    h = sha256_text(span[:500])
                    if h in seen_span:
                        continue
                    seen_span.add(h)
                    uniq.append(span)
                text = "\n\n".join(uniq)
                if len(text) < MIN_CHARS["merger"]:
                    continue
                rec = record(f"maud:{key}", "merger", "maud", "cc-by-4.0", text)
                if rec["sha256"] in seen:
                    continue
                seen.add(rec["sha256"])
                docs.append(rec)
    docs.sort(key=lambda r: r["id"])
    return docs[:152]


def load_contractnli(seen: set[str]) -> list[dict]:
    zpath = download(
        "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip",
        CACHE / "contract-nli.zip",
    )
    unique: dict[str, dict] = {}
    with zipfile.ZipFile(zpath) as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        for name in json_names:
            payload = json.loads(zf.read(name).decode("utf-8"))
            documents = payload.get("documents", payload if isinstance(payload, list) else [])
            if isinstance(payload, dict) and "data" in payload:
                documents = payload["data"]
            for doc in documents:
                if not isinstance(doc, dict):
                    continue
                text = doc.get("text") or doc.get("document") or ""
                if not text or len(text) < MIN_CHARS["nda"]:
                    continue
                rec = record(
                    f"contractnli:{doc.get('id', sha256_text(text)[:12])}",
                    "nda",
                    "contractnli",
                    "cc-by-4.0",
                    text,
                    {"url": doc.get("url"), "file_name": doc.get("file_name")},
                )
                unique[rec["sha256"]] = rec
    rng = random.Random(SEED)
    pool = [r for r in unique.values() if r["sha256"] not in seen]
    rng.shuffle(pool)
    chosen = pool[: QUOTAS["nda"]]
    for rec in chosen:
        seen.add(rec["sha256"])
    return chosen


def edgar_search(query: str, form: str, start: int = 0) -> list[dict]:
    params = {
        "q": query,
        "dateRange": "custom",
        "startdt": "2018-01-01",
        "enddt": "2025-12-31",
        "forms": form,
        "from": str(start),
    }
    url = "https://efts.sec.gov/LATEST/search-index?" + urlencode(params, quote_via=quote)
    time.sleep(0.4)
    payload = json.loads(http_get(url).decode("utf-8"))
    return payload.get("hits", {}).get("hits", [])


def edgar_url(hit: dict) -> str | None:
    src = hit.get("_source", {})
    hit_id = str(hit.get("_id", ""))
    if ":" not in hit_id:
        return None
    adsh, filename = hit_id.split(":", 1)
    ciks = src.get("ciks") or []
    if not ciks:
        return None
    cik = str(ciks[0]).lstrip("0") or "0"
    adsh_nodash = adsh.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_nodash}/{filename}"


def looks_like_pdf(url: str, raw: bytes) -> bool:
    return url.lower().endswith(".pdf") or raw[:4] == b"%PDF"


def collect_edgar(slice_name: str, need: int, seen: set[str]) -> list[dict]:
    docs: list[dict] = []
    if need <= 0:
        return docs
    for query, form in EDGAR_QUERIES[slice_name]:
        start = 0
        empty_pages = 0
        attempts = 0
        while len(docs) < need and empty_pages < 3:
            try:
                hits = edgar_search(query, form, start)
            except Exception as exc:  # noqa: BLE001
                print(f"  search failed {query!r} {form}: {exc}", file=sys.stderr)
                break
            if not hits:
                empty_pages += 1
                start += 100
                continue
            empty_pages = 0
            print(f"  search {query!r} {form} page {start} hits={len(hits)}", flush=True)
            for hit in hits:
                if len(docs) >= need:
                    break
                src = hit.get("_source", {})
                url = edgar_url(hit)
                if not url or url.lower().endswith(".pdf"):
                    continue
                desc = f"{src.get('file_description', '')} {src.get('file_type', '')}".lower()
                if slice_name == "lease" and "lease" not in desc and "lease" not in query:
                    continue
                if slice_name == "merger" and not any(
                    k in desc for k in ("merger", "ex-2", "plan of merger", "amalgamation")
                ):
                    if not str(src.get("file_type", "")).upper().startswith("EX-2"):
                        continue
                attempts += 1
                try:
                    time.sleep(0.35)
                    raw = http_get(url, timeout=25)
                except Exception as exc:  # noqa: BLE001
                    if attempts % 10 == 0:
                        print(f"  skip download ({attempts}): {exc}", flush=True)
                    continue
                if looks_like_pdf(url, raw):
                    continue
                try:
                    body = raw.decode("utf-8")
                except UnicodeDecodeError:
                    body = raw.decode("latin-1", errors="ignore")
                text = html_to_text(body)
                if len(text) < MIN_CHARS[slice_name]:
                    continue
                rec = record(
                    f"edgar:{src.get('adsh')}:{Path(url).name}",
                    slice_name,
                    "edgar",
                    "sec-public",
                    text,
                    {
                        "url": url,
                        "file_date": src.get("file_date"),
                        "file_type": src.get("file_type"),
                        "file_description": src.get("file_description"),
                        "company": (src.get("display_names") or [None])[0],
                    },
                )
                if rec["sha256"] in seen:
                    continue
                seen.add(rec["sha256"])
                docs.append(rec)
                print(f"  {slice_name} {len(docs)}/{need} {rec['id']}", flush=True)
            start += 100
        if len(docs) >= need:
            break
    return docs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "slice",
        "source",
        "license",
        "n_chars",
        "n_tokens_est",
        "sha256",
        "url",
        "file_date",
        "file_type",
        "file_description",
        "company",
        "file_name",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def dataset_card(counts: dict[str, int], n_docs: int) -> str:
    return f"""---
language:
- en
license: cc-by-4.0
size_categories:
- 10K<n<100K
task_categories:
- token-classification
pretty_name: M&A Contract Extraction 3B Research
tags:
- legal
- contracts
- mergers-and-acquisitions
- clause-extraction
---

# M&A Contract Extraction (3B research corpus)

Private source corpus for distilling 41 CUAD-style clause types into a 3B extractor.
**Size tag:** `10K<n<100K` is the *planned training-example count after distillation*
(about 8,000–12,000 JSON rows). It is **not** 1M–10M rows and it is **not** disk size.
This revision contains **{n_docs} source contracts** (document count is `n<1K` until
chunking/distillation is pushed).

## Current document mix

| Slice | Docs | Source |
|---|---:|---|
| Merger / stock-purchase | {counts.get("merger", 0)} | MAUD + EDGAR |
| Commercial / vendor | {counts.get("commercial", 0)} | EDGAR EX-10 |
| NDAs | {counts.get("nda", 0)} | ContractNLI sample |
| Leases | {counts.get("lease", 0)} | EDGAR |
| Other M&A-adjacent | {counts.get("other", 0)} | EDGAR |

CUAD is **eval-only** and is not copied here. Load `theatticusproject/cuad` at eval time.

## License

- MAUD and ContractNLI: CC BY 4.0 (credit the original authors).
- EDGAR filings: public SEC access; reuse rights can still vary by document.
- This repo card uses CC BY 4.0 to match the research datasets. Do not treat the
  Hugging Face UI "MIT" tag as authoritative if it was set at dataset creation.

## Files

- `parsed/documents.jsonl.gz` — one contract per line (`id`, `slice`, `source`, `text`, …)
- `manifests/sources.csv` — same rows without full text
"""


def load_hub_documents() -> list[dict]:
    path = hf_hub_download(
        HF_REPO,
        "parsed/documents.jsonl.gz",
        repo_type="dataset",
    )
    docs: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def push_corpus(docs: list[dict]) -> None:
    counts = {k: sum(1 for d in docs if d["slice"] == k) for k in QUOTAS}
    print("Counts:", counts, "total", len(docs), flush=True)
    parsed = CACHE / "parsed" / "documents.jsonl.gz"
    manifest = CACHE / "manifests" / "sources.csv"
    card = CACHE / "README.md"
    write_jsonl(parsed, docs)
    write_manifest(manifest, docs)
    card.write_text(dataset_card(counts, len(docs)), encoding="utf-8")
    api = HfApi()
    for local, dest in (
        (card, "README.md"),
        (manifest, "manifests/sources.csv"),
        (parsed, "parsed/documents.jsonl.gz"),
    ):
        print(f"Uploading {dest}…", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=dest,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Update {dest} ({len(docs)} source contracts)",
        )
    parsed.unlink(missing_ok=True)
    for name in ("maud_v1.zip", "contract-nli.zip"):
        (CACHE / name).unlink(missing_ok=True)
    print("Done. Local bulky files removed. Canonical copy is on the Hub.", flush=True)


SLICE_KEYWORDS = {
    "merger": ("merger", "amalgamation", "plan of merger", "stock purchase", "share purchase"),
    "commercial": (
        "supply",
        "license",
        "licence",
        "distribution",
        "distributor",
        "master service",
        "services agreement",
        "vendor",
        "outsourcing",
        "manufacturing",
    ),
    "lease": ("lease", "sublease", "tenancy"),
    "other": ("joint venture", "employment", "asset purchase", "non-compete"),
}


def classify_desc(desc: str) -> str | None:
    blob = desc.lower()
    for slice_name, keys in SLICE_KEYWORDS.items():
        if any(k in blob for k in keys):
            return slice_name
    return None


def fill_edgar(docs: list[dict], seen: set[str]) -> list[dict]:
    need_merger = max(0, QUOTAS["merger"] - sum(1 for d in docs if d["slice"] == "merger"))
    print(f"EDGAR merger top-up: {need_merger}", flush=True)
    docs.extend(collect_edgar("merger", need_merger, seen))
    for slice_name in ("commercial", "lease", "other"):
        have = sum(1 for d in docs if d["slice"] == slice_name)
        need = max(0, QUOTAS[slice_name] - have)
        print(f"EDGAR {slice_name}: {need}", flush=True)
        docs.extend(collect_edgar(slice_name, need, seen))
    return docs


def fill_from_ex10_shards(docs: list[dict], seen: set[str]) -> list[dict]:
    """Sample Exhibit 10 text from public HF shards instead of hitting sec.gov."""
    need = {k: max(0, QUOTAS[k] - sum(1 for d in docs if d["slice"] == k)) for k in QUOTAS}
    print("EX-10 remaining:", need, flush=True)
    if sum(need.values()) == 0:
        return docs
    api = HfApi()
    shards = [
        s.path
        for s in api.list_repo_tree("chenghao/sec-material-contracts", "data", repo_type="dataset")
        if s.path.endswith(".parquet")
    ]
    shards.sort()
    for shard in shards:
        if sum(need.values()) == 0:
            break
        print(f"  shard {shard}", flush=True)
        local = hf_hub_download(
            "chenghao/sec-material-contracts",
            shard,
            repo_type="dataset",
        )
        import pyarrow.parquet as pq

        table = pq.read_table(
            local,
            columns=["desc", "doc_type", "extension", "file_content", "file_url", "name", "filing_date", "cik"],
        )
        for row in table.to_pylist():
            if sum(need.values()) == 0:
                break
            ext = str(row.get("extension") or "").lower()
            if ext in {"pdf"}:
                continue
            content = row.get("file_content") or ""
            if not isinstance(content, str) or len(content) < 2000:
                continue
            if content[:4] == "%PDF":
                continue
            slice_name = classify_desc(str(row.get("desc") or "") + " " + str(row.get("doc_type") or ""))
            if not slice_name or need[slice_name] <= 0:
                continue
            text = html_to_text(content)
            if len(text) < MIN_CHARS[slice_name]:
                continue
            rec = record(
                f"ex10:{row.get('cik')}:{sha256_text(str(row.get('file_url')))[:12]}",
                slice_name,
                "edgar-ex10-hf",
                "sec-public",
                text,
                {
                    "url": row.get("file_url"),
                    "file_date": str(row.get("filing_date") or ""),
                    "file_type": row.get("doc_type"),
                    "file_description": row.get("desc"),
                    "company": row.get("name"),
                },
            )
            if rec["sha256"] in seen:
                continue
            seen.add(rec["sha256"])
            docs.append(rec)
            need[slice_name] -= 1
            print(f"  {slice_name} kept, remaining {need}", flush=True)
        Path(local).unlink(missing_ok=True)
    return docs
    need_merger = max(0, QUOTAS["merger"] - sum(1 for d in docs if d["slice"] == "merger"))
    print(f"EDGAR merger top-up: {need_merger}", flush=True)
    docs.extend(collect_edgar("merger", need_merger, seen))
    for slice_name in ("commercial", "lease", "other"):
        have = sum(1 for d in docs if d["slice"] == slice_name)
        need = max(0, QUOTAS[slice_name] - have)
        print(f"EDGAR {slice_name}: {need}", flush=True)
        docs.extend(collect_edgar(slice_name, need, seen))
    return docs


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    topup_edgar = "--topup-edgar" in sys.argv
    topup_ex10 = "--topup-ex10" in sys.argv
    seen: set[str] = set()
    docs: list[dict] = []

    if topup_edgar or topup_ex10:
        print("Resuming from Hub documents.jsonl.gz…", flush=True)
        docs = load_hub_documents()
        seen = {d["sha256"] for d in docs if d.get("sha256")}
        print(f"  already have {len(docs)} documents", flush=True)
    else:
        print("Loading MAUD…", flush=True)
        maud = load_maud(seen)
        print(f"  MAUD documents: {len(maud)}", flush=True)
        docs.extend(maud)

        print("Loading ContractNLI NDAs…", flush=True)
        ndas = load_contractnli(seen)
        print(f"  NDA documents: {len(ndas)}", flush=True)
        docs.extend(ndas)

    if topup_edgar:
        fill_edgar(docs, seen)
    if topup_ex10 or not topup_edgar:
        fill_from_ex10_shards(docs, seen)
    push_corpus(docs)


if __name__ == "__main__":
    main()
