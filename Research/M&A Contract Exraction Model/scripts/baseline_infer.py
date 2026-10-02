#!/usr/bin/env python3
"""Untuned 3B JSON extraction on a gold split. Run on Colab T4 / Modal GPU, not the M3.

Writes preds.jsonl for scripts/eval_cuad.py.

  python3 scripts/baseline_infer.py --split valid --limit 32
  python3 scripts/baseline_infer.py --split test --model unsloth/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

HF_REPO = "Aby-ss/ma-extraction-3B-research"
DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"


def load_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_split(split: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    name = {
        "train": "splits/cuad_qa_train.jsonl.gz",
        "valid": "splits/cuad_qa_valid.jsonl.gz",
        "test": "splits/cuad_qa_test_gold.jsonl.gz",
    }[split]
    path = Path(hf_hub_download(HF_REPO, name, repo_type="dataset"))
    return load_jsonl_gz(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="preds.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    with Path(args.out).open("w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            messages = [m for m in row["messages"] if m["role"] != "assistant"]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            handle.write(json.dumps({"id": row["id"], "output": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(rows)}", flush=True)
    print(f"Wrote {args.out} n={len(rows)} model={args.model}")


if __name__ == "__main__":
    main()
