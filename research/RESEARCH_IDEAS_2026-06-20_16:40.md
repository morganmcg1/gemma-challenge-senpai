# Research Ideas — 2026-06-20 16:40

## Context

Model: google/gemma-4-E4B-it, 1x A10G (23 GB VRAM, sm_86 Ampere)
Stack: vLLM 0.22.0, concurrency=1, output_len=512, 128 prompts
Current best: ~255 TPS, int4 W4A16 g32 Marlin body + MTP K=6 + int4 g32 lm_head + VLLM_BATCH_INVARIANT=0
Quality gate: <=5% degradation on AIME/MMLU-Pro/GPQA-Diamond, PPL<=2.42, 128/128 completions

Decode-cycle profile:
- Body verify-GEMM: 5.92 ms (47.6%) — dominant; MLP is 68% of body, near bandwidth-bound
- Non-GEMM residual: 3.28 ms (26.4%) — ~250 RMSNorm/RoPE/KV-write launches
- MTP drafter forward: 2.48 ms (20.0%)
- lm_head GEMV: 0.75 ms (6.0%) — already optimized

Ranked by: expected TPS gain x quality-gate probability x implementation cheapness on vLLM 0.22.

---

## Hypothesis 1: W4A8 Activation Quantization via SERQ + RMSNorm Fusion (HIGHEST PRIORITY)

**Rank: 1 of 7**

### Mechanism
Move from W4A16 (int4 weights, fp16 activations) to W4A8 (int4 weights, int8 activations) to engage the INT8 GEMM compute path. SERQ (ICLR 2026, "Saliency-Aware Low-Rank Error Reconstruction") uses a single low-rank compensation matrix plus offline weight permutation to recover accuracy after activation quantization, avoiding the per-block rotation overhead of QuaRot/SpinQuant.

Simultaneously, fuse the per-token int8 activation scaling into the preceding RMSNorm and post-GEMM SwiGLU gate kernels, eliminating standalone quantization kernel launches. TensorRT-LLM v0.16.0 already ships this fusion for the FP8 path; the same approach applies to W4A8.

### Why it helps
The body GEMM accounts for 47.6% of total decode time. W4A8 replaces the current Marlin W4A16 kernel (which reads int4 weights and fp16 activations) with a path where both operands are narrower. Ampere sm_86 has INT8 tensor cores; at M=1 (GEMV regime), moving from fp16 activation reads to int8 cuts activation memory traffic by 2x, which directly reduces HBM bandwidth pressure (currently 74-79% of peak). Fusion eliminates a subset of the ~250 small kernel launches that make up the 26.4% non-GEMM residual.

Combined attack on two bottlenecks simultaneously is the strongest lever available.

### Implementation sketch
1. Requantize the model offline with SERQ: apply saliency-aware error reconstruction to Gemma-4 linear layers, targeting W4A8 with group_size=32 for the body (keep lm_head at W4A16 as-is).
   - SERQ code: openreview.net/forum?id=nFjj8NEBqv (ICLR 2026); check for released code under the authors' GitHub by the time this is run.
   - Alternative starting point: AutoAWQ W4A8 mode as a quality floor-check before SERQ overhead is warranted.
2. In vLLM's model runner, add an INT8 GEMM execution path for the repacked weights. The Marlin kernel swap is the main engineering work; vLLM 0.22 has a plugin hook for custom quantization kernels.
3. Fuse per-token activation quantization scale computation into the RMSNorm CUDA kernel (one kernel reads hidden state, writes normalized fp32, writes int8 + scale instead of fp16).
4. Validate PPL and quality benchmarks before any serve change.

### Main risk
Activation outliers in Gemma-4's MoE routing or MLP SwiGLU gate can cause large W4A8 accuracy degradation — this is the primary failure mode. SERQ is validated on Llama/Mistral families; Gemma-4 MoE architecture is untested. PPL probe first (cheap). If SERQ accuracy is insufficient, fall back to W4A8 with per-channel smoothing (SmoothQuant-style) before the GEMM.

**Source:** SERQ — openreview.net/forum?id=nFjj8NEBqv; FP8 fusion reference — arxiv 2502.01070v2

---

## Hypothesis 2: Kernel Megafusion — RMSNorm + RoPE + KV-Write (Non-GEMM Bottleneck)

**Rank: 2 of 7**

### Mechanism
The 26.4% non-GEMM residual is dominated by ~250 individually launched small CUDA kernels: per-layer RMSNorm, RoPE embedding application, KV-cache write, and per-token sampling ops. Each launch has ~5-10 us overhead on A10G; 250 launches contribute ~1.25-2.5 ms of pure scheduling overhead independent of compute.

Write a single fused Triton or CUDA kernel per transformer block that does: RMSNorm -> (optional W4A8 quant scale, if Hypothesis 1 is live) -> attention QKV projection -> RoPE -> KV-write in one pass over the hidden state vector.

### Why it helps
At M=1 (single-stream decode), each of these operations touches the same 4096-d (or 8192-d) hidden state vector. Fusing them keeps the data in L2 cache across the chain instead of writing it back to HBM between kernel launches. The effective "activation traffic" per block drops proportionally. This directly targets the non-GEMM 26.4% share without touching quantization.

### Implementation sketch
1. Profile with CUDA nsys to confirm which of the 250 launches dominate latency vs. which are trivial — this is a 30-minute diagnostic, do it first.
2. Write a Triton kernel for the decode path (M=1 path only): `rmsnorm_rope_kvcopy_fused(hidden, weight, cos, sin, k_cache, v_cache, position)`. The vLLM custom op registry accepts Triton ops.
3. Test byte-exact parity against the unfused path on a single forward pass before any quality eval.
4. vLLM 0.22 already has CUDA graph capture for the decode step; verify the fused kernel is graph-capturable (no host-device sync inside).

### Main risk
Triton kernel correctness for RoPE on complex-valued rotations is fiddly; a subtle indexing bug can cause silent quality regression that only surfaces on AIME. Always check byte-exact parity first. If Triton doesn't capture cleanly in CUDA graph, launch overhead savings are partially negated.

**Source:** FP8 fusion analysis — arxiv 2502.01070v2 (TRT-LLM v0.16.0 RMSNorm+SwiGLU fusion); CUDA kernel launch overhead profiling documented in multiple vLLM/SGLang blog posts.

---

## Hypothesis 3: 2:4 Structured Sparsity on MLP Weight Matrices

**Rank: 3 of 7**

### Mechanism
Apply 2:4 unstructured (semi-structured) sparsity to the body MLP weight matrices using SparseGPT or Wanda pruning. Ampere sm_86 has NVIDIA 2:4 sparse tensor cores that deliver 2x the effective throughput for weight-sparse GEMMs with zero additional weight reads (the hardware handles the sparsity pattern natively via a compressed metadata format).

MLP is 68% of the body verify-GEMM time. A 2x throughput multiplier on the MLP kernels would reduce the 5.92 ms body time by ~30-35%, cutting total decode latency by ~14-16%.

### Why it helps
The body GEMM is already bandwidth-bound (74-79% HBM peak). 2:4 sparsity reduces the effective weight data volume by 2x, which directly cuts bandwidth pressure. Unlike activation quantization, this requires no change to the vLLM serving path — the sparse weights are loaded offline and the standard cuSPARSELt / Ampere sparse kernel handles the rest.

### Implementation sketch
1. Use SparseGPT (github.com/IST-DASLab/sparsegpt) or Wanda (github.com/locuslab/wanda) to prune the body MLP weight tensors (gate_proj, up_proj, down_proj) to 2:4 sparsity. Do NOT prune the lm_head or attention projections first.
2. Requantize to W4 after pruning (prune-then-quantize order matters; SparseGPT supports this jointly).
3. Use cuSPARSELt or the `torch.nn.utils.parametrize` sparse path to pack weights into compressed 2:4 format.
4. Load in vLLM by overriding the weight loading hook to unpack the sparse metadata.
5. PPL probe before quality suite.

### Main risk
Stacking 2:4 sparsity on top of existing W4 quantization (W4+2:4) has known additive accuracy loss. Several papers report this stack degrades significantly past the 5% quality gate on reasoning tasks. MoE architecture with sparse activations further complicates this. Treat as medium-risk; PPL gate must be <2.42. If combined W4+2:4 fails, try 2:4 sparsity alone at W8 as a deconfound.

**Source:** arxiv 2410.04466v4 (LLM inference hardware survey — SpInfer, SoLA, R-Sparse sections); SparseGPT arxiv 2301.00774; Wanda arxiv 2306.11695.

---

## Hypothesis 4: Q-Palette Per-Layer Mixed-Precision (W4A16 MLP + W8A16 Attention)

**Rank: 4 of 7**

### Mechanism
Q-Palette (NeurIPS 2025, github.com/snu-mllab/Q-Palette) assigns optimal fractional-bit quantization schemes per layer using a data-free sensitivity oracle. The key insight: not all layers contribute equally to inference latency or quality loss. Attention projection layers are small (O(d^2) with small d in grouped-query attention) and can tolerate lower precision; MLP layers are large and dominate both latency and sensitivity.

Strategy: keep MLP layers at W4A16 (current), push attention QKV/output projections to W8A16 or even W4 with larger group size. This is the opposite of the typical sensitivity order — but in MoE models with small active experts, the attention layers see more relative use and their accuracy matters more.

Alternatively: keep the first 4 and last 4 transformer layers at W8A16 (sensitive boundary layers), push middle 40+ layers to W4A8 (cheap for TPS).

### Why it helps
At decode M=1, memory traffic dominates. Fewer bits for the layers that are not on the critical quality path means less HBM bandwidth pressure with minimal accuracy impact. Q-Palette's data-free oracle avoids the calibration data requirement of GPTQ/AWQ sensitivity analysis.

### Implementation sketch
1. Run Q-Palette sensitivity analysis on the Gemma-4 body to identify which layers can safely absorb lower precision.
2. Requantize the identified layers with AutoAWQ or llm-compressor.
3. Load in vLLM using the existing mixed-precision weight loading path (vLLM 0.22 supports per-layer quant config via JSON).
4. PPL gate check, then quality suite.

### Main risk
Q-Palette's data-free sensitivity oracle may not accurately predict quality loss on reasoning benchmarks (AIME especially). The paper's eval covers perplexity but reasoning-task accuracy can decouple from PPL. Test PPL first, but do not skip the reasoning eval.

**Source:** Q-Palette — openreview.net/forum?id=l4F50jpiVH (NeurIPS 2025); Mixed-precision survey — arxiv 2510.16805v1.

---

## Hypothesis 5: CTC-Based or Lightweight Recurrent Drafter Head

**Rank: 5 of 7**

### Mechanism
Replace the current MTP K=6 drafter (which runs a full forward pass through the model's shared body) with a CTC-based drafter that uses a single shallow projection head to propose multiple tokens in parallel without autoregressive conditioning. CTC drafting (from "Speculative Decoding and Beyond", arxiv 2502.19732) produces K draft tokens in one forward pass at O(1) cost, vs. the current MTP which runs K sequential micro-forward passes.

A recurrent drafter alternative (SSM-style, e.g., a 1-layer Mamba head) produces a fixed-cost sequence of K draft tokens with a constant state update.

### Why it helps
MTP drafter forward is 20% of decode time (2.48 ms). If the drafter can be made 2x cheaper while preserving acceptance rate ~3.28, the drafter overhead drops from 20% to ~10%, contributing ~2% total TPS gain. Not the highest leverage but adds to compounding gains.

### Implementation sketch
1. Train a 2-layer MLP projection head that maps the last hidden state to K=6 draft logit vectors simultaneously (independence assumption, similar to Medusa but without the autoregressive tree).
2. Keep acceptance sampling identical to current MTP; only the proposal mechanism changes.
3. Integrate via vLLM's existing speculative decoding plugin interface.
4. Acceptance rate probe before full quality eval.

### Main risk
Independence assumption between draft tokens reduces acceptance rate. If acceptance rate drops from ~3.28 to <2.5, the TPS gain from cheaper drafting is fully offset by more verify passes. Must measure acceptance rate, not just drafting latency.

**Source:** arxiv 2502.19732v4 (speculative decoding survey, CTC drafting and Medusa sections).

---

## Hypothesis 6: CUDA Graph Widening — Capture Sampling + Top-K Inside Graph

**Rank: 6 of 7**

### Mechanism
vLLM 0.22's CUDA graph capture typically covers the forward pass (attention + MLP + lm_head) but may exclude token sampling (top-k/top-p, temperature scaling, multinomial draw). These operations are individually fast but launch multiple small kernels outside the captured graph, adding host-device synchronization.

Widen the CUDA graph capture scope to include the full sampling pipeline: logit scaling + top-k filter + multinomial sample + token buffer update. This converts synchronous host-driven sampling into a fully asynchronous graph replay.

### Why it helps
At concurrency=1, the benchmark loop is: forward pass -> sample -> schedule next. If sampling is outside the graph, there is a host-sync after every token that prevents kernel pipelining. Moving sampling inside the graph eliminates this sync and allows the next forward's input token preparation to overlap with any residual GPU work.

### Implementation sketch
1. Inspect vLLM 0.22 CUDA graph capture scope in `vllm/worker/worker.py` and `vllm/model_executor/models/`.
2. Identify which sampling ops are currently outside the captured region.
3. Patch `CUDAGraphRunner` to extend capture scope to cover the sampling path for the greedy/temperature=0 case first (deterministic, easiest to capture).
4. Measure decode latency before/after with `--enforce-eager False` baseline.

### Main risk
CUDA graphs require static tensor shapes and no Python-side branching during replay. Sampling with dynamic top-k thresholds or complex logit processors breaks graph capture silently. Scope carefully to temperature=0 greedy path first, then extend.

**Source:** vLLM CUDA graph documentation; general CUDA graph widening analysis in Flashinfer and SGLang blog posts.

---

## Hypothesis 7: PLE Dequantization Fusion for Scale Broadcast (Body MLP)

**Rank: 7 of 7**

### Mechanism
The current W4A16 g32 Marlin GEMM requires dequantizing the int4 weight blocks to fp16 before the actual GEMM computation. At group_size=32, this means one dequant scale broadcast per 32 weight elements. The dequantization and scale broadcast can be fused directly into the Marlin GEMM prologue, avoiding a separate memory pass over the scale tensor.

This is the PLE-dequant optimization referenced in the program's prior results (+5.3% lever from PR #798 already merged for lm_head). The same fusion opportunity exists in the body MLP kernels, which are larger and may benefit more.

### Why it helps
If PLE-dequant gave +5.3% on lm_head (6% of total time), applying the same fusion to the body MLP (68% of the 47.6% body = ~32% of total time) could yield a proportionally larger gain: roughly 5-10% total TPS if the scale-broadcast overhead scales with weight volume.

### Implementation sketch
1. Identify whether vLLM 0.22's Marlin kernel for body MLP already uses PLE-dequant or still uses the standard scale-broadcast path.
2. If not already applied: port the PLE-dequant Marlin kernel variant (from the lm_head work) to the body GEMM path.
3. Ablation: run body-MLP PLE-dequant in isolation with lm_head PLE-dequant held fixed to isolate the body contribution.
4. Quick: this is a config/kernel-swap change, no requantization needed.

### Main risk
The body MLP Marlin path may already have PLE-dequant enabled from the lm_head work (if the same kernel template is shared). In that case this is a no-op. Confirm with a profiler trace first (5 min diagnostic).

**Source:** PR #798 (PLE-dequant +5.3% lever, lm_head); post-int4head re-profile from program state.

---

## Summary Table

| # | Hypothesis | Target bottleneck | Expected TPS gain | Quality risk | Implementation cost |
|---|-----------|------------------|------------------|--------------|---------------------|
| 1 | W4A8 + SERQ + RMSNorm fusion | Body GEMM (47.6%) + Non-GEMM (26.4%) | 15-25% | Medium (outliers) | High |
| 2 | RMSNorm+RoPE+KV-write megafusion | Non-GEMM (26.4%) | 8-12% | Low (correctness bug) | Medium |
| 3 | 2:4 structured sparsity on MLP | Body GEMM (47.6%) | 10-16% | High (W4+2:4 stack) | Medium |
| 4 | Q-Palette per-layer mixed precision | Body GEMM (47.6%) | 5-10% | Medium | Low |
| 5 | CTC/recurrent drafter replacement | Drafter (20%) | 2-4% | Low (acceptance rate) | High |
| 6 | CUDA graph widening (sampling) | Non-GEMM overhead | 2-5% | Low | Medium |
| 7 | PLE-dequant body MLP | Body GEMM (47.6%) | 5-10% | Very low | Low |

Hypotheses 1 and 2 can be pursued in parallel. Hypothesis 7 is a cheap diagnostic worth running first (30 min profiler check). If W4A8 (H1) clears quality, it opens the door to stacking H3 (sparsity) and H4 (mixed precision) on the activation-quantized substrate.
