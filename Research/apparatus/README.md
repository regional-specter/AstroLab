# Tuning apparatus (shared)

Every Phase 1 paper uses this loop. Copy the schema and metrics; do not fork a new data warehouse.

```
M3: schema + 50-row smoke
    → Hugging Face Hub (canonical jsonl)
    → Colab T4 Unsloth (first GPU)
    → Modal CPU convert / GPU train (when the job repeats)
    → RunPod 4090 (8B / OOM only, then destroy)
```

| Module | Role |
|---|---|
| `schema.py` | Instruction-row shape and JSON clause payload checks |
| `metrics.py` | JSON validity, Jaccard span overlap, P/R/F1 |
| `streaming.py` | Read jsonl/jsonl.gz without loading the file into RAM |
| `modal_jobs.py` | Template: CPU convert vs GPU Unsloth. Do not put PDF parse on a GPU. |
| `fixtures/` | Tiny committed rows for Mac smoke tests (not a corpus) |

**Rules:** MLX on the Air only. Unsloth on NVIDIA only. Never set `HF_HOME` / `HF_DATASETS_CACHE` to a folder on the Mac.
