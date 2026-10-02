"""Modal Unsloth for CUAD-QA. Use after the Colab T4 notebook works.

  modal setup
  modal secret create huggingface HF_TOKEN=...
  modal run modal_train.py

Pulls splits from Aby-ss/ma-extraction-3B-research. GPU only for training.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_DIR = Path(__file__).resolve().parent

app = modal.App("astrolab-cuad-qlora")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("unsloth", "datasets", "trl", "huggingface_hub")
    .add_local_file(str(APP_DIR / "scripts" / "train_unsloth.py"), remote_path="/app/train_unsloth.py")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
)
def train():
    import runpy
    import sys

    sys.path.insert(0, "/app")
    sys.argv = ["train_unsloth.py"]
    runpy.run_path("/app/train_unsloth.py", run_name="__main__")


@app.local_entrypoint()
def main():
    train.remote()
