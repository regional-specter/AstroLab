# Paper 4 — Healthcare / clinical eligibility (not started)

**Order:** last. DUA / PhysioNet paperwork and 8B VRAM.

| | |
|---|---|
| **Model** | 8B QLoRA |
| **Data** | n2c2 / TREC Clinical Trials after access; prefer **synthetic FHIR notes** so PHI never sits on the Mac |
| **Hub (planned)** | private `Aby-ss/clinical-eligibility-8B-research` |
| **GPU** | Wiped RunPod 4090 or Modal GPU. Colab T4 is usually too tight. |

The Air must never cache notes. Teacher labeling is after the legal/fintech loop is proven.
