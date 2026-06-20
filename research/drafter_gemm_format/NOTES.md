# PR #786 — MTP drafter GEMM format: is the q4_0 drafter BW-optimal on bi0?

Baseline: bi0 = `int4_mtp_bi0_surgattn`, official TPS 218.02, PPL 2.0058, 128/128,
W&B `s63tb03x`. Drafter = `google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant`.

## Step 1 — drafter serving format (code-inspection, no GPU)

**Finding: the drafter is served dense bf16, by design and by checkpoint.**

1. `serve.py` builds `--speculative-config` as
   `{"model": drafter_model, "num_speculative_tokens": 6}` with **no `quantization`
   key** (serve.py:122-129). vLLM therefore infers the drafter format from the
   checkpoint config.
2. Drafter `config.json` (`Gemma4AssistantForCausalLM`, model_type
   `gemma4_assistant`) has **`dtype: bfloat16`** and **no `quantization_config`**.
   → vLLM loads it as dense bf16 (2 bytes/param). The "q4_0" in the repo name
   refers to the QAT checkpoint it was *matched to*; the assistant weights
   themselves are stored **unquantized** ("…-unquantized-assistant").
3. serve.py docstring (lines 16-19) already documents the deliberate decision:
   "The draft head is left in its native bf16/centroid path (never
   force-quantized): the assistant's masked-embedding centroid logits have no
   packed-weight branch, so quantizing it would force the ~11x-slower dense path."

### Drafter weight footprint (why bf16 is plausibly already fine)

From `config.json.text_config`:
- `num_hidden_layers: 4`, `hidden_size: 256`, `intermediate_size: 2048`,
  `num_attention_heads: 4`, `head_dim: 256`, `num_kv_shared_layers: 4`
  (Q-only, shares the target KV cache → no own K/V projection cost).
- Per-layer Linear params ≈ q_proj(256×1024) + o_proj(1024×256) + mlp
  gate/up/down(3×256×2048) ≈ 2.1M. × 4 layers ≈ **8.4M params ≈ 16.8 MB bf16**.
- Output: centroid head (`num_centroids: 2048`, `centroid_intermediate_top_k: 32`,
  `vocab_size: 262144`, `tie_word_embeddings: true`). NOT a dense 256→262144
  matmul — this is the "masked-embedding centroid" path. The tied embedding
  (262144×256 ≈ 67M params, 134 MB bf16) is read by **gather** (active rows
  only) on input, not a per-pass GEMM.

So the per-draft-pass *dense GEMM* weight read is only ~16.8 MB bf16. The target
int4 W4A16 verifier read per step is multi-GB (profiler: ~92% GEMM). Quantizing
the 16.8 MB drafter body to int4 (→ ~4.2 MB) saves a few MB per pass — a small
fraction of total step BW — and at M=1 (decode) the tiny drafter GEMMs are
latency-bound, not weight-BW-bound, where int4 Marlin dequant overhead can
exceed the BW saving. **Prior: likely null/negative. Measuring to confirm.**

## Step 2 — drafter vs verifier pass cost (GPU) — PENDING
## Step 3 — `--speculative-quantization marlin` acceptance test (GPU) — PENDING
