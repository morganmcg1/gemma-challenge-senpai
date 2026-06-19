STUDENT land:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["rwk498ve"],"primary_metric":{"name":"byte_exact_fixable","value":1},"test_metric":{"name":"bi0_e2e_argmax_flip_rate","value":0.01042}}

## Results — strict-clean spec divergence locus PINNED + classified

**Locus = layer 0, the attention context op (`attn_out`, the flash split-KV reduction). Classification = deterministic reduction-ORDER artifact → BYTE-EXACT FIXABLE.** The whole M=1-decode-vs-M=K-verify divergence collapses to byte-exact when the M=1 decode attention is forced to `num_splits=1` (same reduction order as the verify path). The strict-clean (byte-exact, G1-immune) spec path is therefore **viable** — but the fix is the batch-invariant / fixed-split attention kernel, which is **not free** (see follow-ups).

### 1. Locus (layer + op) — walked the residual stream, first non-zero bitdiff
Layer-0 op chain, M=1 decode (3D split-KV) vs M=K=6 verify (2D one-shot), at matched absolute positions, bf16→fp32 lossless `torch.equal`:

| layer-0 op | frac_bitdiff | max_abs | n_bitdiff/n |
|---|---|---|---|
| `qkv_proj` (Marlin int4 GEMM) | 0.0000 | 0.0 | 0/186 |
| `attn_q` (post-norm+RoPE) | 0.0000 | 0.0 | 0/186 |
| `attn_k` (post-norm+RoPE) | 0.0000 | 0.0 | 0/186 |
| `attn_v` | 0.0000 | 0.0 | 0/186 |
| **`attn_out` (flash context output)** | **1.0000** | **0.0625** | **186/186** ← FIRST DIVERGENCE |
| `o_proj` out | 1.0000 | 0.0625 | 186/186 (inherits) |

The qkv GEMM and the q/k/v inputs to attention are **byte-identical** across batch width — this **reconstructs wirbel #736's int4-GEMV M-invariance in-scope on my own served stack** (the PR gave it as a measured input). The divergence is created **inside the attention kernel**: the served TRITON_ATTN kernel runs a 3D segmented-LSE split-KV reduction for `max_seqlen_q==1` (M=1 decode) but a 2D one-shot reduction for `max_seqlen_q>1` (M=K verify) — a different reduction order over the same KV.

### 2. Magnitude (max bitdiff / max logits delta) + argmax-flip count — deployed path (bi0, VLLM_BATCH_INVARIANT=0)
- Max bitdiff at the locus (layer-0 `attn_out`): **0.0625** (one bf16 quantum at that scale).
- Accumulated to pre-`lm_head` hidden: **max_abs = 1.875**, frac_bitdiff = **1.000** (100% of 186 positions differ in bits).
- e2e logprob bitdiff rate (M=K verify logprob vs M=1 decode logprob): **94.3%** of positions.
- **Per-position argmax-flip count: 2 / 192 = 1.04%.** Pervasive in bits, tiny in argmax effect — consistent with the int4 near-tie "don't-care" residual (land #654 / #680).
- **Control — AR-vs-AR M=1 token identity: 1.000000 (96/96).** Same-shape determinism is byte-perfect, so the measured delta is purely the M=1-vs-M=K **shape** difference, not run-to-run noise.

### 3. Classification (instr 4) — reduction-order vs genuine numeric, via the num_splits=1 lever
Re-ran identically with `VLLM_BATCH_INVARIANT=1`, which forces the M=1 decode attention to `num_splits=1` (2D one-shot) — the **same** reduction order the M=K verify path already uses (the verify path is unchanged by the flag, since `max_seqlen_q>1` is already 2D). Only the decode arm moves onto the verify arm's reduction order:

| metric | bi0 (deployed) | bi1 (num_splits=1) | verdict |
|---|---|---|---|
| layer-0 `attn_out` frac_bitdiff | 1.0 (186/186) | **0.0 (0/186)** | collapses |
| pre-`lm_head` max_abs delta | 1.875 | **0.0** | collapses |
| e2e logprob bitdiff rate | 94.3% | **0.0%** | collapses |
| e2e argmax-flips | 2 / 192 | **0 / 192** | collapses |
| first divergent locus | layer-0 `attn_out` | **None (all 42 layers byte-exact)** | collapses |
| AR-vs-AR control | 1.000000 | 1.000000 | unchanged |

**Verdict: BYTE-EXACT FIXABLE — deterministic reduction-order artifact.** The divergence is entirely the 3D-split-KV-vs-2D-one-shot reduction order in the flash-attention kernel; pinning the order (num_splits=1) makes the full forward byte-identical between the decode and verify shapes. It is **not** a genuine numeric difference (no residual bitdiff survives at any layer under bi1).

### What this means
A **byte-exact spec-verify path is achievable** by serving the spec stack with a batch-invariant / fixed-split attention kernel (so the M=1 draft-decode forwards and the M=K verify forward share one reduction order). That path carries **no public→private TPS-reproduction drift risk** (G1-immune), unlike the K=6 #730 fire. This card closes the attention leg the PR set out to pin: GEMV leg (wirbel #736) ✓ + attention leg (this card) ✓ = the strict-#319 divergence is **fully attributed and fixable**.

### The catch (honest)
"Fixable" ≠ "free." The fix is exactly the fixed-split / batch-invariant attention kernel. This card measured **fixability only, not TPS** (per the instructions: locus + magnitude + flip-count + verdict). Prior in-scope land measurements priced the batch-invariant-attention decode tax at ~5% (#484/#623 paired A/B), with a targeted fixed-split (#363) being the route to make it ~free. So the byte-exact spec path is viable **and priced** — whether it clears 126.378 depends on that attention tax vs the spec acceptance gain, which is the natural next card.

### Methodology / fidelity caveat
- Stack: vLLM **0.22.0** offline `LLM`, `quantization=compressed-tensors` (int4 g128), `enforce_eager=True`, `max_num_seqs=1`, in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so PyTorch forward hooks fire in my process. Backend `TRITON_ATTN` (forced; Gemma4 heterogeneous head_dim).
- Per-op capture via forward hooks on each layer's `qkv_proj`/`self_attn`(q,k,v,out)/`o_proj` + the final `norm` (pre-`lm_head`). M=1 decode forwards (3D) are stored during a real `generate`; the M=K verify shape is re-forwarded via chunked-prefill `prompt_logprobs` (2D, query-row-independent — a faithful verify-shape proxy). Only genuine decode positions are compared; the M=1 prefill (2D) is excluded.
- Fidelity caveat: measured on the **loadable full-vocab QAT ckpt** (`google/gemma-4-E4B-it-qat-w4a16-ct`) — the deployed pruned-16k-head ckpt won't load in vanilla vLLM. The attention M-dependence is a **kernel-occupancy / reduction-order** property, independent of weight-quant and head-vocab, so the locus + fixability attribution is faithful to the deployed stack.

### Commands
```bash
# GPU measurement (vLLM-0.22.0 venv), bi0 = deployed:
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/senpai-venvs/20f658587e8a6643/bin/python \
  research/validity/strict_clean_attn_locus_743/locus_pin.py \
  --n-prompts 8 --n-new 24 --ctx-cap 256 --det-prompts 4 --verify-width 6 \
  --out research/validity/strict_clean_attn_locus_743/runs/locus_bi0.json
# bi1 = classification (force num_splits=1): add  --batch-invariant 1  --out .../locus_bi1.json

# 0-GPU wandb log (land-mb-venv):
/tmp/land-mb-venv/bin/python research/validity/strict_clean_attn_locus_743/wandb_log.py \
  --bi0 .../runs/locus_bi0.json --bi1 .../runs/locus_bi1.json --group strict-clean-locus-pin
```

### Run facts
- **W&B run:** `rwk498ve` (group `strict-clean-locus-pin`, project `wandb-applied-ai-team/gemma-challenge-senpai`).
- **Peak GPU memory:** ~18.94 GiB (A10G 24 GB), `enforce_eager`, single-seq.
- 8 prompts × 24 new tokens = 192 e2e decode positions; 7812 per-op decode captures; verify width K=6. bi0 61 s + bi1 ~119 s of inference (excl. model load).

### Public evidence used (in-scope reconstruction)
- **wirbel #736** (W&B `624ypc14`): "int4 Marlin GEMV is bit-exactly M-invariant" — given by the PR as a measured input. I **reconstructed it on my own in-scope served stack** (layer-0 `qkv_proj` frac_bitdiff = 0.0 across M=1 vs M=6); I did not read wirbel's branch for any number.
- **land #680** (W&B `5iy1mhe4`, my own prior): predicted the verify break is **attention, not the GEMM**, with a 90%-bitdiff / ~0.16%-flip signature. This card confirms (94.3% bitdiff / 1.04% flip) and now **pins the exact op (layer-0 `attn_out`) and proves byte-exact fixability** via the num_splits=1 collapse.

### Suggested follow-ups
1. **Price the fixed-split attention tax** on the served spec stack: A/B `num_splits=1` (batch-invariant attention) vs deployed split-KV at the served decode shapes, then net it against the spec acceptance gain — does byte-exact spec clear 126.378? (This is the deciding measurement; #363 targeted fixed-split is the ~free route.)
2. **Build the byte-exact spec submission** end-to-end (draft decode + verify both `num_splits=1`) and check strict-#319 identity holds at served scale (128 prompts), confirming 0 confident misses, not just 0 flips on this 192-position probe.
