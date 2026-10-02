# Paper 3 — Logistics / noisy document extraction (not started)

**Order:** third.

| | |
|---|---|
| **Model** | 1.5B |
| **Data** | **FUNSD** + **CORD** as OCR-noise *proxies*, then synthetic error injection into a frozen ISO / HS / UN/LOCODE schema |
| **Hub (planned)** | `Aby-ss/logistics-ocr-1.5B-research` |
| **Convert** | CPU (M3 prototype, Modal CPU at scale). Do not attach a GPU to PDF/OCR convert. |
| **First GPU** | Colab T4 / Modal L4 |

The paper must say CORD/FUNSD are receipts/forms, not bills of lading, until BoL labels exist.
