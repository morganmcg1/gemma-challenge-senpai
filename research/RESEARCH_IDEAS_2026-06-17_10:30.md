# Research Ideas — 2026-06-17 10:30

External-literature-only search. Two waves of parallel Exa/Semantic Scholar/ArXiv/WebFetch searches covering:
rotation-based re-quantization, int4+reasoning recovery, LM head factorization, deterministic FP reductions,
QuIP# incoherence processing, mixed-precision sensitivity metrics, weight/L2 streaming, SageAttention,
FRTQ calibration-only W4A4, QuaRot NeurIPS 2024, async KV prefetching, AIME calibration data.

Problem state (inline, no internal board reads):
- base_fullhead: 252.69 TPS, PPL 2.0057 — PASSES MMLU-Pro/GPQA/GSM8K, FAILS AIME (0.1167 vs ≥0.360 bar)
- shipped config: 375.857 TPS — QUALITY COLLAPSES
- Three hard gates: (1) PPL ≤ 2.42; (2) #319 STRICT byte-exact greedy token identity; (3) MMLU-Pro/GPQA/GSM8K/AIME each ≥90% of vanilla base
- No from-scratch retraining. Light post-hoc transforms (re-quantization, rotations, calibration) allowed.
- HBM-bandwidth-bound decode on A10G (sm_86, 24 GB, ~600 GB/s). 262k-row LM head is a key HBM bottleneck in base_fullhead.

Closed directions (must not be revisited): per-token byte-identity saturation on all 3 components, depth-drop/layer-skipping, decode-overhead floor (311.27 TPS), CUDA-graph/torch.compile (99.41% GPU-bound), speculative decoding (all variants), FP8 KV-cache.

---

## TIER 1 — Highest Confidence, Direct Sub-Problem Closure

---

### Idea 1: Post-Hoc Hadamard Rotation Re-Quantization for AIME Recovery (Sub-problem B)

**The idea.** Apply a random Hadamard rotation to the weight matrices of the already-quantized Gemma-4-E4B model, re-quantize with the same W4A16 budget, and test whether the rotation's incoherence effect recovers AIME reasoning accuracy without changing throughput. QuaRot (ETH/EPFL/Microsoft, NeurIPS 2024) proves the central property: Hadamard rotations are *computationally invariant* — they change neither outputs nor loss in full precision. The rotation pre-processes the quantization target, spreading outlier energy uniformly across all weight coordinates, which dramatically reduces per-row quantization error.

**Why it might help.** The AIME gap (0.1167 vs 0.360 bar) is almost certainly caused by extreme per-token activation outliers in chain-of-thought reasoning layers concentrating quantization error. Rotation + re-quantization at W4A16 is the standard fix. The "Quantization Hurts Reasoning?" paper (COLM 2025, Huawei Noah's Ark) shows W4A16 can be *lossless* for larger models when the right quantization approach is used — Gemma-4 at 4B sits near the transition zone. SpinQuant (Meta, arxiv 2405.16406) extends this to a *learned* rotation (Cayley manifold optimization) and closes the zero-shot reasoning gap to 2.9 pts vs unquantized at W4A4KV4 on LLaMA-2 7B — stronger than random Hadamard alone.

**Throughput impact.** W4A16 Marlin kernels are unchanged. The rotation is baked into offline weight preprocessing; no runtime overhead. TPS stays near 252.69 TPS. The goal is purely gate-3 AIME recovery without touching TPS.

**Implementation sketch.**
1. Load base unquantized Gemma-4-E4B-it weights (or dequantize the int4 checkpoint).
2. Generate random Hadamard rotation matrices R_l for each linear layer (QuaRot open-source: `github.com/spcl/QuaRot`).
3. Fold R_l into W: W' = W @ R_l^T. Re-quantize W' to int4 using the existing quantizer (GPTQ or AWQ). For activations: multiply input by R_l before each GEMM (or fuse into preceding op).
4. Validate byte-exact identity: this BREAKS gate #2 (#319 byte-exact) because floating-point reductions change. The gate-3 path does NOT require byte-exact — check the contract carefully. If AIME passes gate-3, this wins on gate-3 without needing gate-2.
5. AIME is the only failing metric. Run AIME-24 eval (30 problems × 4 seeds) with the rotated re-quantized checkpoint.

**Key risk.** (a) Rotation re-quantization requires loading the full unquantized or dequantized fp16 weights — this is a ~8 GB memory operation, feasible offline but takes 30–60 minutes of calibration compute. (b) The Gemma-4 tokenizer has a 262k vocabulary: the LM head rotation is the most expensive. (c) Gate #2 (#319 byte-exact) is violated — this path wins only on gate-3. Confirm whether a gate-3-only win is acceptable for the advisory contract.

**Papers / repos.**
- QuaRot: "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs" — Ashkboos et al., NeurIPS 2024. https://arxiv.org/abs/2404.00456 — Github: https://github.com/spcl/QuaRot
- SpinQuant: "SpinQuant: LLM Quantization with Learned Rotations" — Liu et al. (Meta), 2024. https://arxiv.org/abs/2405.16406 — Github: https://github.com/facebookresearch/SpinQuant
- "Quantization Hurts Reasoning? Examining Post-Training Quantization for LLM Reasoning" — COLM 2025, Huawei Noah's Ark. https://openreview.net/forum?id=60bb19f2

**Taste rubric score (diagnostic).**
- Mechanistic grounding: 4 — computational invariance property is exact; failure mode is identifiable (outliers in CoT layers)
- Research-state value: 4 — sharply discriminates whether AIME failure is a quantization-scheme problem vs an irreducible precision-loss problem
- Execution value: 3 — 30–60 min offline calibration + AIME eval; no training needed

---

### Idea 2: "Quantization Meets Reasoning" Locate-and-Restore Intervention (Sub-problem B)

**The idea.** Apply the three-step "measure → locate → restore" post-hoc loop from the ICLR 2026 submission "Quantization Meets Reasoning" (OpenReview So3hbnEGYV) directly to the already-quantized int4 checkpoint. The method: (1) compute a reasoning-task sensitivity metric on the quantized model, (2) locate the 5–15% of weight tensors most responsible for AIME degradation, (3) restore those tensors to fp16/bf16 (mixed-precision). The whole pipeline runs in 3–5 minutes on a single GPU, needs only 332 curated examples, and is architecture- and quantizer-agnostic. The paper shows AWQ/GPTQ/SmoothQuant degrade AIME-style benchmarks by up to 69.81%; the locate-and-restore loop recovers most of that gap.

**Why it might help.** Unlike a global re-quantization (Idea 1), this is surgical: it finds the exact tensors that hurt AIME the most and un-quantizes only those. The speed impact is proportional to how many layers get promoted. If only the final 2–3 transformer blocks' MLPs and attention projections are restored (typical), the Marlin kernel still handles 90%+ of the FLOPS, and the TPS hit may be only 3–8% — keeping total TPS above 245 while clearing the AIME gate.

**Throughput impact.** Proportional to fraction of layers promoted. A surgical promote of 10–15% of layers costs roughly 10–15% of the int4 speedup, landing ~240–250 TPS range — still well above 252 only if the restore fraction is kept small. Needs careful profiling. A better framing: run the locate step first (3–5 min), look at *how many* layers score above threshold, and only proceed if the layer count is small.

**Implementation sketch.**
1. Load the shipped int4 checkpoint directly.
2. Implement the sensitivity scan from the paper: for each weight tensor W_l, compute the change in reasoning-task loss (using AIME calibration samples) when W_l is dequantized to fp16. Use gradient checkpointing to avoid OOM.
3. Rank layers by sensitivity score; restore top-k layers to fp16 (set a budget: ≤15% of layers).
4. Verify AIME eval passes the ≥0.360 bar. Check MMLU-Pro/GPQA/GSM8K still pass.
5. Measure TPS with the mixed-precision model (some layers Marlin int4, some fp16 GEMM).
6. Gate #2 (#319 byte-exact) is NOT preserved — restored fp16 layers will produce exact outputs but the body int4 layers still differ. This is a gate-3-only path.

**Key risk.** The method's 332-example calibration set was curated for specific models; for Gemma-4 the AIME signal may require a Gemma-specific calibration set. Locating the right layers is the key uncertainty — if the model spreads sensitivity across all 62 layers, the required fp16 fraction becomes large and TPS collapses. The paper's numbers are on LLaMA/Mistral families; transfer to Gemma-4 is not guaranteed.

**Papers / repos.**
- "Quantization Meets Reasoning: Recovering Reasoning Ability for Small Language Models" — ICLR 2026. https://openreview.net/forum?id=So3hbnEGYV
- ReQuant/PQI: "Accurate Post-Training Quantization via Post-Quantization Integral" — arxiv 2503.01901. https://arxiv.org/abs/2503.01901

**Taste rubric score (diagnostic).**
- Mechanistic grounding: 3 — clearly targets the AIME failure mode; locate step produces interpretable layer-sensitivity map
- Research-state value: 4 — if locate step shows sensitivity concentrated in few layers: proceed; if spread uniformly: rules out surgical promotion as a path
- Execution value: 4 — 3–5 min locate pass is extremely cheap; go/no-go before any expensive eval

---

### Idea 3: FRTQ-Style GSR-Guided W4A4 Recalibration for Speed + Quality (Sub-problems A+B combined)

**The idea.** FRTQ (Flattened Rotation TSVD Quantization, ICLR 2026, OpenReview mUB2N8L0vD) introduces the GSR metric — Grid-to-Standard-Deviation Ratio, ρ = Δ_g / std(X_c) — which measures how coarse the quantization grid is relative to the natural activation variability of each layer. Layers with high GSR are quantization bottlenecks. FRTQ uses this metric to guide: (1) activation flattening via rotation, (2) low-rank weight preconditioning via TSVD (truncated SVD). The method is calibration-only (no training), handles massive activation outliers, and matches higher-bit accuracy at W4A4.

**Why it might help.** Current base_fullhead uses w4a16 body + fp16 LM head. The bottleneck for AIME is almost certainly a few high-GSR layers where activation outliers cause catastrophic quantization error. FRTQ's TSVD step factorizes W = U S V^T and applies low-rank preconditioning *only* to the high-GSR layers — this is different from global low-rank (like LoRA) and specifically targets the quantization damage. If activations and weights can be moved to W4A4 in most layers while keeping the few high-GSR layers at W4A16, the weight read per decode step drops further below base_fullhead's 252.69 TPS floor.

**Throughput impact.** W4A4 body vs current W4A16: activations also int4 means smaller intermediate tensors and potential for fused int4 GEMM that reads less HBM. However, A10G's int4 tensor core throughput advantage over fp16 is already captured by Marlin; the real gain is from *smaller weight tiles* reducing HBM traffic. Rough estimate: if 50% of layers move to true W4A4 (half the activation precision), expect 5–10% additional TPS gain, landing ~265–275 TPS.

**Implementation sketch.**
1. Compute per-layer GSR on Gemma-4 using FRTQ's calibration procedure (C4 calibration set, 128 samples).
2. Identify high-GSR layers (top 20% by ρ); keep these at W4A16. Move remaining 80% to W4A4.
3. Apply TSVD preconditioning to high-GSR layers' weight matrices.
4. Re-quantize using GPTQ or the existing quantizer with updated calibration.
5. Measure PPL (must be ≤ 2.42), AIME, MMLU-Pro, GPQA, GSM8K, TPS.
6. Note: this breaks gate #2 (#319 byte-exact). Gate-3 path only.

**Key risk.** FRTQ's results are demonstrated on LLaMA/Mistral. Gemma-4 uses a different normalization (RMSNorm) and attention variant (GQA with 8 KV heads) — the GSR distribution may differ. The TSVD step adds inference-time overhead (split into two smaller GEMMs), which can reduce TPS if the rank is high. This method is more complex than Ideas 1 and 2 and should come after those cheaper diagnostics.

**Papers / repos.**
- FRTQ: "FRTQ: Flattened Rotation and TSVD-Based Quantization for LLMs" — ICLR 2026. https://openreview.net/forum?id=mUB2N8L0vD
- QuIP#: "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks" — Tseng et al., Cornell ICML 2024. https://arxiv.org/abs/2402.04396 — Github: https://github.com/cornell-RelaxML/quip-sharp

**Taste rubric score (frontier refinement).**
- Mechanistic grounding: 3 — GSR metric gives a principled per-layer sensitivity map; TSVD preconditioning is well-motivated
- Research-state value: 3 — tests whether W4A4 recalibration with selective fallback can beat base_fullhead TPS while recovering AIME
- Execution value: 2 — more complex than Ideas 1/2; should be sequenced after those

---

## TIER 2 — High Potential, Requires More Validation

---

### Idea 4: SageAttention for Attention Kernel Throughput (Sub-problem A, Orthogonal)

**The idea.** Replace the current attention kernel in base_fullhead with SageAttention (Tsinghua, arxiv 2410.02367), which quantizes the Q/K/V projection products to int8/fp8 *inside* the attention kernel — not in the weights, but in the inner GEMM of attention. It reports 2.1× OPS vs FlashAttention2 and claims "almost no end-to-end metrics loss" across diverse models. It is plug-and-play via a `torch.nn.functional.scaled_dot_product_attention` drop-in.

**Why it might help.** Attention in the decode step (single-token Q attending over growing KV cache) is memory-bandwidth-bound by KV cache reads. SageAttention's int8 quantization of attention *inputs* reduces the effective precision of the attention computation but does NOT change the weight matrices, so it is architecturally separate from the LM-head and body weight quantization questions. If SageAttention is orthogonal to the base_fullhead body weights, it can compound with any of Ideas 1–3.

**Throughput impact.** In decode-heavy batch=1 single-stream serving, attention is a smaller fraction of total wall time than body GEMMs. Realistic estimate: 5–15% end-to-end TPS gain, bringing base_fullhead from 252.69 up to ~265–290 TPS range.

**Implementation sketch.**
1. pip install sageattention (or build from https://github.com/thu-ml/SageAttention).
2. In Gemma-4's attention module, replace `torch.nn.functional.scaled_dot_product_attention` with `sageattn`.
3. Run the PPL gate (must be ≤ 2.42) and byte-exact identity gate (#319).
4. Critical: SageAttention uses int8 quantized attention intermediates. This WILL change floating-point output of the attention layer, breaking gate #2 (#319 byte-exact). Confirm with a quick token-identity check.
5. If gate-3 path (no byte-exact): run full MMLU-Pro/GPQA/GSM8K/AIME eval after measuring TPS. Can compound with Idea 1 or 2 if those already cleared AIME.

**Key risk.** Gate #2 (#319 byte-exact) almost certainly broken. SageAttention is a gate-3-only path. The "almost no end-to-end metrics loss" claim is from diverse batch sizes and prompts; single-stream batch=1 decode may show different tradeoffs. The 2.1× OPS gain is a compute-throughput metric — at batch=1 decode, attention is memory-bandwidth-bound not compute-bound, so the realized speedup will be smaller than headline.

**Papers / repos.**
- SageAttention: "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration" — Zhang et al., ICLR 2025. https://arxiv.org/abs/2410.02367 — Github: https://github.com/thu-ml/SageAttention
- SageAttention2: "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization" — arxiv 2411.10958. https://arxiv.org/abs/2411.10958

**Taste rubric score (frontier refinement).**
- Mechanistic grounding: 3 — clear mechanism (attention int8 reduces KV bandwidth); but the batch=1 decode speedup from compute-quantization is uncertain
- Research-state value: 2 — if it gives +10% TPS, useful as a compound gain; if it fails gate-2 and gives only +5% TPS, marginal
- Execution value: 3 — very cheap to try (drop-in replacement); fast go/no-go

---

### Idea 5: Async L2-Cache-Oriented KV Cache Prefetching for Attention Overlap (Sub-problem A, Orthogonal)

**The idea.** "Prefetching-Driven Efficient LLM Inference via Asynchronous KV Cache Access" (arxiv 2504.06319) restructures the decode loop to proactively prefetch KV cache tiles from HBM into the GPU L2 cache during the preceding body GEMM computation. The GPU's asynchronous DMA units (cp.async in PTX) move data ahead of consumption, hiding HBM latency. They report 2.15× attention kernel efficiency and 1.97× end-to-end throughput on H20 GPUs, and claim the technique is orthogonal to existing optimizations.

**Why it might help.** On A10G at batch=1 decode, the bottleneck is HBM bandwidth. If the attention phase's KV reads can be overlapped with the preceding MLP GEMM's compute, the effective HBM utilization improves. This is a systems-level trick that does NOT change numerical outputs — it is argmax-preserving by construction, and could in principle preserve gate #2 (#319 byte-exact) if the prefetch does not change the numerical reduction order.

**Critical caveat.** H20 numbers may not transfer directly to A10G. H20 has 4 TB/s HBM3e; A10G has ~600 GB/s. The relative gain from L2 prefetching depends on the L2/HBM bandwidth ratio, which differs across GPUs. The paper's vLLM integration patch would need porting to vLLM 0.22 on sm_86.

**Implementation sketch.**
1. Obtain or reimplement the async prefetch attention kernel from the paper's GitHub (if public) or the arxiv code appendix.
2. Integrate into vLLM 0.22's PagedAttention kernel path for Gemma-4.
3. Run a microbenchmark: attention-only decode throughput with and without prefetch.
4. Verify token identity (#319 gate) — if numerics are unchanged, this could be the only idea compatible with gate #2.
5. Full TPS measurement and quality eval.

**Key risk.** (a) GitHub may not be public yet (arxiv 2504 = April 2025; only 6 weeks old at the time of the search). (b) A10G L2 cache is 40 MB; with long sequences, KV cache exceeds L2, limiting the benefit. (c) The technique overlaps attention HBM reads with MLP compute — the ratio of attention to MLP time at batch=1 is the sensitivity parameter, and at batch=1 with Gemma-4's GQA (8 KV heads), attention is relatively small vs MLP, so the absolute gain may be modest.

**Papers / repos.**
- "Prefetching-Driven Efficient LLM Inference via Asynchronous KV Cache Access" — arxiv 2504.06319. https://arxiv.org/abs/2504.06319

**Taste rubric score (frontier refinement).**
- Mechanistic grounding: 3 — clean mechanism; argmax-preserving is a major differentiator
- Research-state value: 3 — if gate #2 is preserved AND TPS improves: this is the only found path that could beat 252.69 while keeping byte-exact identity
- Execution value: 2 — kernel-level vLLM integration is non-trivial; A10G transfer uncertain; should be preceded by a microbenchmark

---

## TIER 3 — Exploratory, Higher Risk

---

### Idea 6: QuIP# Incoherence + E8 Lattice Re-Quantization Below W4 (Sub-problem B, exploratory)

**The idea.** QuIP# (Cornell, ICML 2024, arxiv 2402.04396) combines randomized Hadamard incoherence processing (pre/post-multiply by random ±1 Hadamard matrices) with E8 lattice codebooks — the densest known 8-dimensional sphere packing — for sub-4-bit per-parameter quantization. At 2 bits/param, QuIP# matches W4 models that lack incoherence processing. At 3 bits/param, it significantly outperforms standard W4 on perplexity and reasoning benchmarks. The randomized Hadamard transformation is applied offline (calibration phase) and the E8 lookup is fused into the GEMM kernel.

**Why it might help.** If Gemma-4's body can be quantized to 3-bit with QuIP#'s incoherence + E8 while maintaining AIME quality, the HBM weight reads per decode step drop by 25% vs current W4, directly improving TPS. The incoherence processing (Hadamard) also spreads outliers, which is the same mechanism that helps AIME in Idea 1.

**Throughput impact.** 3-bit body weights: HBM reads drop from ~0.5 bytes/param (int4) to ~0.375 bytes/param (int3). On an HBM-bandwidth-bound workload, this should give ~25% TPS gain over base_fullhead if quality holds, landing ~315 TPS — potentially leaping over the 252.69 floor by a wide margin.

**Key risk.** (a) E8 lookup adds compute overhead; the A10G-specific fused E8 GEMM kernel from QuIP# may not be optimized for sm_86. (b) QuIP# requires fine-tuning (QuIP# paper uses LDLQ + optional fine-tuning) — the no-retraining constraint may limit achievable quality vs the paper's numbers. (c) 3-bit is aggressive for a 4B model; AIME at W3 with QuIP# calibration-only is untested on Gemma-4. (d) This breaks gate #2 (#319 byte-exact).

**Papers / repos.**
- QuIP#: "AQLM: Additive Quantization of Language Models" + "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks" — Tseng et al., Cornell. https://arxiv.org/abs/2402.04396 — Github: https://github.com/cornell-RelaxML/quip-sharp

**Taste rubric score (tier shift).**
- Mechanistic grounding: 3 — clear mechanism (E8 + Hadamard); but 3-bit at 4B model size without fine-tuning is uncertain
- Research-state value: 3 — if quality holds at W3: unlocks a large TPS jump; if it collapses: closes the sub-3-bit path
- Execution value: 2 — complex calibration pipeline; kernel portability to sm_86 uncertain; sequence after Ideas 1/2

---

### Idea 7: SpinQuant Learned Rotation on Gemma-4 (Sub-problem B, higher effort)

**The idea.** SpinQuant (Meta, arxiv 2405.16406, facebookresearch/SpinQuant) learns rotation matrices via Cayley manifold (Stiefel manifold) optimization using a small validation set. Unlike QuaRot's random Hadamard (fixed), SpinQuant optimizes the rotation to *minimize quantization loss* on the actual model's activation distribution. The paper shows learned rotations close the W4A4KV4 reasoning gap to 2.9 pts on LLaMA-2 7B and outperform random Hadamard rotations on zero-shot benchmarks. Rotations are output-identical in full precision (the incoherence property holds for both random and learned rotations).

**Why it might help.** If random Hadamard rotation (Idea 1) partially recovers AIME but not fully, SpinQuant's learned rotation should close more of the gap. The Cayley optimization step takes ~1–2 hours on a single GPU with a small calibration set (512 examples), fitting within the "light post-hoc transforms" budget. The learned rotation can be tailored to Gemma-4's specific activation outlier distribution.

**Throughput impact.** Same as Idea 1 — rotation baked into weights offline, no runtime overhead, TPS unchanged from base_fullhead (~252.69 TPS). The value is purely in closing the AIME gap.

**Key risk.** (a) SpinQuant's Cayley optimization is the most complex piece of this idea — it requires gradient computation through the rotation manifold, which needs the unquantized model weights loaded in memory (~8 GB for Gemma-4). (b) Caveat from SpinQuant repo: the Cayley optimizer is designed for LLaMA-family architectures; Gemma-4's attention (QK-norm, GQA) may require architectural adaptation. (c) This should be sequenced after random Hadamard (Idea 1) — if random Hadamard already clears AIME, SpinQuant's additional complexity is not needed.

**Papers / repos.**
- SpinQuant: "SpinQuant: LLM Quantization with Learned Rotations" — Liu et al. (Meta). https://arxiv.org/abs/2405.16406 — Github: https://github.com/facebookresearch/SpinQuant

**Taste rubric score (frontier refinement).**
- Mechanistic grounding: 4 — precise mechanism; output-identical in full precision is the critical property; Cayley optimization has theoretical guarantees
- Research-state value: 3 — best-case: AIME recovers AND TPS holds; worst-case: Cayley optimization fails to converge on Gemma-4 = closes that branch
- Execution value: 2 — sequence after Idea 1; only proceed if random Hadamard fails to fully close the AIME gap

---

## Decision Tree

```
START: base_fullhead fails AIME gate (0.1167 vs ≥0.360)
│
├─ STEP 1 (cheapest): Run "Quantization Meets Reasoning" locate pass (Idea 2)
│   ├── How many layers are high-sensitivity?
│   │   ├── ≤10% of layers → proceed with surgical promote → check AIME + TPS
│   │   │   ├── AIME ≥ 0.360 AND TPS ≥ 252 → WINS (gate-3 path)
│   │   │   └── AIME < 0.360 OR TPS < 252 → proceed to Step 2
│   │   └── >25% of layers → surgical promote unviable → go directly to Step 2
│   │
├─ STEP 2: Apply random Hadamard rotation re-quantization (QuaRot / Idea 1)
│   ├── Run AIME eval on rotated checkpoint
│   │   ├── AIME ≥ 0.360 AND TPS ~252 → WINS on gate-3
│   │   │   ├── Compound with SageAttention (Idea 4) for +5–15% TPS → ~265–290 TPS
│   │   │   └── Compound with async KV prefetch (Idea 5) for further gain
│   │   └── AIME partially recovers (0.25–0.36) → try SpinQuant learned rotation (Idea 7)
│   │       ├── AIME ≥ 0.360 → WINS on gate-3; compound for TPS
│   │       └── AIME still < 0.360 → try FRTQ W4A4 (Idea 3) or QuIP# W3 (Idea 6)
│
├─ STEP 3 (for TPS beyond ~252 while preserving byte-exact, gate #2):
│   └── Async L2 KV prefetch (Idea 5) — only known path that could beat 252 with gate #2
│       ├── Byte-exact identity preserved + TPS > 252 → WINS on gate #2 + gate #3 MMLU/GPQA/GSM8K (AIME still open)
│       └── Byte-exact broken by prefetch reordering → gate #2 path remains closed
│
└─ STOP CONDITIONS
    - If Ideas 1+2 both fail to recover AIME and Ideas 3+6 are attempted with calibration-only (no fine-tuning):
      Conclude AIME at W4 without fine-tuning is infeasible for Gemma-4 4B;
      the AIME gap is a model-size + quantization-scheme fundamental limit.
    - If TPS ceiling above 252 with byte-exact gate #2 is desired and no path clears it:
      The byte-exact constraint is an asymptotic floor; declare 252 TPS as the byte-exact ceiling.
```

---

## Research State Update

**Current best explanation for the plateau:**
The base_fullhead config (252.69 TPS) fails only the AIME gate. The AIME failure is almost certainly caused by activation outliers in chain-of-thought reasoning layers amplifying quantization error under the current int4-QAT W4A16 scheme. The quantization scheme (not the model capacity) is the limiting factor. The shipped 375.857 TPS config gains its speed by collapsing quality — the quality-speed tradeoff is a direct consequence of the LM head pruning (262k→something smaller) that degrades the token distribution.

**What rules out the obvious approaches:**
- More speculative decoding: fails both gates.
- Depth-drop: no quality-safe path found.
- CUDA-graph / torch.compile: already saturated (99.41% GPU-bound).
- Byte-exact TPS above 252 via body kernel changes: per-token identity is saturated on all 3 components.

**Open uncertainties:**
1. Whether AIME failure is concentrated in a few layers (surgical fix viable) or spread uniformly across all 62 layers (requires global re-quantization).
2. Whether random Hadamard rotation can recover AIME without fine-tuning on a 4B model (vs 7B+ models where the papers demonstrate results).
3. Whether any path exists above 252 TPS that preserves byte-exact gate #2.

**Priority order for execution:**
1. Idea 2 (locate pass) — 3–5 min; diagnostic only; determines whether surgical or global re-quant is needed.
2. Idea 1 (QuaRot rotation) — ~60 min offline; most principled calibration-only path for AIME recovery.
3. Idea 4 (SageAttention) — ~2 hours; compound gain after AIME is fixed; cheap drop-in to try.
4. Idea 5 (async KV prefetch) — 1–2 days of kernel engineering; only path to beat 252 with gate #2; high upside, high effort.
5. Idea 3 (FRTQ) — sequence after 1+2 if they fail; more complex.
6. Ideas 6+7 (QuIP# W3, SpinQuant) — sequence after 1+2 fail; escalation options.
