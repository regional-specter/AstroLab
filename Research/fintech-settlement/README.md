# Paper 2 — Fintech / trade settlement (not started)

**Order:** second, after legal CUAD has a published F1.

| | |
|---|---|
| **Model** | 3B + guided decoding (Outlines / vLLM) |
| **Data** | Stream **FinRED** and **FinTagging**; convert to the same JSON instruction rows as CUAD |
| **Hub (planned)** | `Aby-ss/fintech-settlement-3B-research` |
| **First GPU** | Colab T4 |
| **New difficulty** | Grammar constraints, not more PDFs |

Do not invent a SWIFT dump. Copy `../M&A Contract Exraction Model/scripts/convert_cuad_qa.py` as the convert template.

Mac: schema + 50-row smoke only. Unsloth on NVIDIA.
