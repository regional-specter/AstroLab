"""Modal job template. Copy into a paper folder; do not run PDF parsing on a GPU.

  modal run modal_jobs.py --convert    # CPU
  modal run modal_jobs.py --train      # NVIDIA Unsloth

First Unsloth run for each paper should still be Google Colab T4.
Graduate that notebook into these functions only after it works.
"""

from __future__ import annotations

import modal

app = modal.App("astrolab-research")

cpu_image = modal.Image.debian_slim().pip_install(
    "huggingface_hub",
    "tiktoken",
)

# Unsloth needs a CUDA image. Pin versions in the paper folder, not here.
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("unsloth", "datasets", "trl", "huggingface_hub")
)


@app.function(image=cpu_image, timeout=60 * 60)
def convert():
    raise NotImplementedError("Paper-specific convert: stream Hub/benchmark → jsonl → upload. CPU only.")


@app.function(
    image=gpu_image,
    gpu="T4",
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def train():
    raise NotImplementedError("Paper-specific Unsloth. Pull jsonl from Hub, save adapter, exit.")


@app.local_entrypoint()
def main(convert_only: bool = False, train_only: bool = False):
    if convert_only:
        convert.remote()
        return
    if train_only:
        train.remote()
        return
    print("Pass --convert-only or --train-only. Default is a no-op so you do not spend GPU by mistake.")
