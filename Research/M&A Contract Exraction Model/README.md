### Phase 1: Synthetic Dataset & Distillation Pipeline

1. **Source Legal Corpus Setup:** Gather unannotated M&A agreements, NDAs, and vendor contracts. Segment text using sliding-window chunking (~2,048 tokens with 256-token overlap) to mirror CUAD context lengths.
2. **Teacher Model Distillation (Claude 3.5 Sonnet / o1):**

- Prompt teacher models to act as senior M&A counsel, extracting 41 predefined clause types (e.g., *Governing Law*, *Termination for Convenience*, *Anti-Assignment*).
- Enforce response output format strictly matching a Pydantic/JSON Schema (e.g., `{clause_type: str, exact_quote: str, confidence: float, statute_code: str}`).

1. **Data Verification & Cleaning:**

- Reject instances where `exact_quote` is not verbatim inside the chunk context to eliminate hallucinations.
- Format final dataset into standard ChatML or alpaca instruction-tuning schema.

---



### Phase 2: Baseline Benchmarking (Pre-Fine-Tuning)

1. **CUAD Standardized Evaluation Setup:** Prepare CUAD test split across its 41 legal clause categories.
2. **Base SLM Evaluation:** Run un-tuned `Llama-3.2-3B` or `Qwen2.5-3B` using strict JSON sampling (e.g., `outlines` or `vLLM` structured decoding).
3. **Record Metrics:** Log Precision, Recall, Macro/Micro F1, Precision @ 80% Recall, and inference latency.

---



### Phase 3: Fine-Tuning with Unsloth

1. **Adapter & Precision Configuration:**

- Load base 3B model in 4-bit (`load_in_4bit = True`) using Unsloth.
- Apply PEFT (LoRA/QLoRA) targeting **all projection modules** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) to capture both attention routing and token representation mapping.

1. **Loss Masking Strategy:**

- Enforce **Target-Only Loss Masking** (`train_on_inputs = False`). The model must calculate cross-entropy loss exclusively on output response tokens (the target JSON structure), not on lengthy contract inputs.

1. **Optimization & Execution:**

- Use `paged_adamw_8bit` optimizer with cosine learning rate schedule ($2 \times 10^{-4}$ peak rate) and warmup ratio $0.05$–$0.10$.
- Train until cross-entropy loss plateaus around $\approx 0.2$–$0.3$ without divergence.

---



### Phase 4: Comparative Post-Fine-Tuning Evaluation

1. **Benchmark Execution:** Evaluate fine-tuned model checkpoint on CUAD benchmark test split using exact Jaccard token overlap rules ($J(A, B) \ge \kappa_0$).
2. **Metric Target Audit:** Measure against target goals: F1 score $> 0.88$ across 41 clause types.
3. **Latency Benchmarking:** Deploy fine-tuned LoRA merged weights using TensorRT-LLM or vLLM on local GPUs to confirm sub-80ms end-to-end extraction latency.

---



## Critical Foundational Principles with Unsloth

To prevent structural failure, loss spikes, or model degradation during fine-tuning, strictly adhere to these principles:

### 1. Target Loss Masking (Critical for Long Contexts)

- **Principle:** Legal documents have input prompts with thousands of tokens, while the target output (JSON clause metadata) is small.
- **Risk:** If you train on both input prompt and output tokens, 95% of your gradient updates are spent predicting contract text rather than structured extraction.
- **Rule:** Use `DataCollatorForCompletionOnlyLM` or Unsloth’s native prompt masking so that loss is computed **only** on response tokens.



### 2. Low-Rank Matrix Target Allocation

- **Principle:** Extraction tasks require updating both context retrieval mechanics (Attention) and reasoning/formatting mechanics (MLPs).
- **Rule:** Do not restrict LoRA adapters to `q_proj` and `v_proj`. Set target modules to all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Hyperparameters:** Keep rank $r = 16$ or $r = 32$ and set $\alpha = r$ or $\alpha = 2r$. Set LoRA dropout to `0.0` (required for Unsloth 4-bit optimizations).



### 3. Native Quantization & Logit Bias Constraints

- **Principle:** Forced JSON Schema enforcement during post-training fine-tuning can cause models to fail if structural syntax tokens (like `{`, `"`, `]`) are missing in training targets.
- **Rule:** Ensure all synthetic training targets pass strict JSON parsing before passing them to Unsloth. During inference, combine the fine-tuned model with guided decoding engines (e.g., `outlines` or `vLLM` logit processors) to force schema compliance without breaking output distributions.



### 4. Sequence Length & VRAM Buffer Management

- **Principle:** CUAD and legal contracts push `max_seq_length` to 4,096 or 8,192 tokens. Memory allocation scales quadratically with sequence length unless properly managed.
- **Rule:** Call `FastLanguageModel.for_inference(model)` or `FastLanguageModel.for_training(model, max_seq_length=4096)` explicitly in Unsloth. Use RoPE scaling (e.g., dynamic YARN or linear scaling) if context window expands beyond base model limits.



### 5. Weight Merging for Sub-80ms Latency

- **Principle:** Running LoRA adapters in production adds execution latency during inference forward passes.
- **Rule:** Save fine-tuned adapters, then merge adapters directly back into base 16-bit float weights (`model.save_pretrained_merged(..., save_method="merged_16bit")`). Export to AWQ/EXL2 or GGUF for local GPU inference engines to achieve sub-80ms response target.

