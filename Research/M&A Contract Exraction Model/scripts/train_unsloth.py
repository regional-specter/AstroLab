#!/usr/bin/env python3
"""Unsloth QLoRA for CUAD JSON clause extraction. Run this in Google Colab (T4).

Colab:
  1. Runtime → GPU (T4).
  2. Upload .cache/cuad_qa_train.jsonl.gz and cuad_qa_valid.jsonl.gz
     OR set HF_TOKEN and pull from Aby-ss/ma-extraction-3B-research.
  3. !pip install unsloth
  4. !python train_unsloth.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from datasets import Dataset

HF_REPO = "Aby-ss/ma-extraction-3B-research"
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 4096
LOCAL_TRAIN = Path("cuad_qa_train.jsonl.gz")
LOCAL_VALID = Path("cuad_qa_valid.jsonl.gz")


def load_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_split(split: str) -> Dataset:
    local = LOCAL_TRAIN if split == "train" else LOCAL_VALID
    if local.exists():
        rows = load_jsonl_gz(local)
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            HF_REPO,
            f"splits/cuad_qa_{split}.jsonl.gz",
            repo_type="dataset",
        )
        rows = load_jsonl_gz(Path(path))
    return Dataset.from_list(rows)


def main() -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    train_ds = load_split("train")
    valid_ds = load_split("valid")
    print(f"train={len(train_ds)} valid={len(valid_ds)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    train_ds = train_ds.map(formatting_prompts_func, batched=True)
    valid_ds = valid_ds.map(formatting_prompts_func, batched=True)

    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            output_dir="outputs_cuad_qa",
            seed=42,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    trainer.train()
    model.save_pretrained("lora_cuad_qa")
    tokenizer.save_pretrained("lora_cuad_qa")
    print("Saved LoRA adapter to ./lora_cuad_qa")


if __name__ == "__main__":
    main()
