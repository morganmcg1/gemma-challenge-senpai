# Speed Levers: Bold Fresh Ideas for Gemma-4-E4B-it on A10G
# 2026-06-14 — External literature scouting, single-stream 128/128 greedy decode

## Problem Summary

- **Model**: google/gemma-4-E4B-it (dense ~4B edge decoder)
- **Hardware**: Single NVIDIA A10G, 24GB VRAM, sm_86 (Ampere)
- **Task**: single-stream, 128 prompt → 128 generated tokens, greedy decode
- **Constraint**: greedy-IDENTICAL output (token-exact), PPL ≤ 2.42 (1.77% headroom), 128/128 completion
- **Baseline**: 481.53 TPS
- **Target**: >500 TPS
- **Current frontier**: M=32 depth-9 draft tree, deployed M=8 verify width, E[T]=5.066; int4 Marlin verify body; ρ-optimal max-branch-3 topology E[T]=5.219; public λ=1 ceiling 520.95 TPS

**Do NOT re-propose**: M=32 depth-9 tree, int4 Marlin body, deployed M=8 verify width (all already running).

---

## ANGLE 1: Topology — Closing the 0.153 E[T] Gap (Highest-Value Unlock)

The gap between deployed E[T]=5.066 and ρ-optimal max-branch-3 E[T]=5.219 is the single largest lever. Every 0.1 E[T] translates directly to TPS at fixed verify cost.

### Idea T-1: OPT-Tree — Per-Step Adaptive Optimal Tree (Highest priority)

**Mechanism**: Instead of a fixed tree topology chosen offline, OPT-Tree computes the acceptance-length-maximizing topology at each decode step using the draft model's current output distribution and a dynamic programming algorithm over the branching budget. This removes the gap between the deployed static tree and the per-step ρ-optimal tree.

**Concrete change**: Replace the static M=8 verify-width tree with OPT-Tree's per-step DP tree construction. The DP runs on CPU/GPU at the start of each step before draft sampling; latency is O(depth × branches) arithmetic operations, typically <0.1ms. The verify step remains identical — the tree shape changes but the verify kernel does not.

**Expected TPS/E[T] effect**: The ρ-optimal max-branch-3 static topology yields E[T]=5.219. OPT-Tree closes this gap by adapting to the current distribution; on similar setups authors report E[acceptance length] gains of 8–15% over fixed topologies. Conservative estimate: E[T] → 5.25–5.35, translating to +3–6% TPS beyond the deployed ceiling (→ 496–511 TPS range from baseline mechanics).

**Greedy-identity / PPL risk**: Zero — the verify step is unchanged; acceptance criterion is identical. Any token that would be accepted by the standard verify kernel is accepted here. PPL impact: none.

**Local-profilable**: Yes. The DP construction is a pure Python/NumPy routine at tree-planning time; can be unit-tested against the static tree on CPU in minutes with synthetic draft distributions.

**arXiv / GitHub**: arXiv:2406.17276 (OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure). No official repo but the DP algorithm is described fully in §3.2 and is reproducible in ~50 lines of Python.

---

### Idea T-2: CAST — Inference-Cost-Aware Dynamic Tree (ICLR 2026)

**Mechanism**: CAST profiles device-specific costs (memory bandwidth, compute throughput) and batch size jointly, then constructs a tree that maximizes the expected speedup ratio rather than just acceptance length. On A10G specifically, bandwidth-bound single-stream decode means that the cost model will favor wider shallow trees (more bandwidth pressure per step) differently than Sequoia's compute-optimal assumptions.

**Concrete change**: Run CAST's calibration script (offline, 10 min) to derive the A10G-specific optimal tree, then deploy that tree in place of the current fixed topology. The deploy path is the same as today — just a different tree JSON passed to the speculative sampler.

**Expected TPS/E[T] effect**: CAST reports 8–19% speedup over static baseline trees on A100/H100; on A10G the gap may be larger because the bandwidth/compute ratio differs more from training hardware. Expect E[T] improvement comparable to T-1 but potentially higher due to device-specific calibration.

**Greedy-identity / PPL risk**: Zero — verify criterion unchanged.

**Local-profilable**: Yes. Calibration runs as a standalone profiling job; the tree is then frozen and deployed. No training required.

**arXiv / GitHub**: "CAST: Inference-Cost-Aware Speculative Decoding" (ICLR 2026). Search by title; paper and supplementary code available on OpenReview.

---

### Idea T-3: C2T — Lightweight Classifier for Tree Pre-Pruning (7–17% on top of EAGLE-2/3)

**Mechanism**: A 241-parameter linear classifier (trained on draft-head activations) predicts which draft candidates will be rejected before they enter the verify step. Rejected candidates are pruned from the draft tree before the verify forward pass, reducing verify-side FLOPs and memory traffic without touching the acceptance criterion.

**Concrete change**: Train the C2T classifier (241 parameters, 30 min on the same A10G) on rollout data from the current draft model. Integrate as a pre-verify filter: for each leaf in the draft tree, run classifier; prune nodes with predicted rejection probability above threshold (0.7 works in the paper). The verify kernel then runs on a sparser tree.

**Expected TPS/E[T] effect**: Paper reports 25% reduction in draft candidates reaching verify, translating to 7–17% walltime speedup on top of EAGLE-2/3. On the current M=8 verify tree, pruning 2 of 8 leaves reduces verify FLOPs by ~25% while holding E[T] nearly constant (pruned candidates were likely to be rejected anyway).

**Greedy-identity / PPL risk**: Low but non-zero. The classifier introduces a probabilistic pruning step that can occasionally prune an accepted candidate. The acceptance criterion for surviving candidates is unchanged; the risk is that a false-positive prune removes a token that would have been accepted. The paper shows <0.5% E[T] degradation at threshold 0.7.

**Local-profilable**: Yes. Classifier training is local; correctness can be verified by comparing per-step accepted token sequences against the reference.

**arXiv / GitHub**: "C2T: Classifier-to-Tree Pre-Pruning for Speculative Decoding" (ICLR 2026 submitted). Search OpenReview 2026 for "C2T speculative decoding classifier".

---

### Idea T-4: Sequoia DP — Offline Optimal Tree with Accurate Cost Model

**Mechanism**: Sequoia uses dynamic programming to find the tree topology that maximizes expected tokens accepted per step under a given draft budget, using an accurate model of the acceptance probability at each tree node. Unlike OPT-Tree (per-step online), Sequoia is an offline calibration that produces a single optimal static tree — but the DP accounts for the joint structure of the tree (correlations between sibling/cousin nodes) rather than treating branches independently.

**Concrete change**: Run Sequoia's DP calibration with the actual draft and verify model pair, using A10G-measured step costs. Replace the current depth-9 M=32 draft / M=8 verify tree with the Sequoia-optimal tree under the same draft budget and verify width budget.

**Expected TPS/E[T] effect**: Sequoia reports up to 4.04x speedup (aggregate, favorable settings), but in the single-stream near-optimal setting the marginal gain is specifically the correction from naively-branched to DP-optimally-branched at the same budget. Expect E[T] → 5.15–5.25 (modest, since the current tree may already be close to DP-optimal by hand-tuning).

**Greedy-identity / PPL risk**: Zero — verify criterion unchanged.

**Local-profilable**: Yes. DP runs offline; GitHub: `FasterDecoding/Sequoia` (NeurIPS 2024 Spotlight).

**arXiv / GitHub**: arXiv:2406.07754; GitHub: https://github.com/FasterDecoding/Sequoia

---

## ANGLE 2: Draft Architecture — Improving E[T] Beyond Tree Shape

Even with optimal topology, E[T] is bounded by draft head acceptance rate. Architecture improvements to the draft head directly raise the acceptance probability ceiling.

### Idea D-1: Mixture of Attentions for EAGLE-2 Draft Head (ICLR 2025, +25% acceptance length)

**Mechanism**: The standard EAGLE-2 draft head uses a shallow auto-regressive transformer that conditions on verify model features via a single cross-attention layer. The Mixture of Attentions (MoA) variant routes draft-head attention across multiple attention patterns — local, global, and cross-attention to verify features — using a learned gating mechanism. This gives the draft head richer conditioning on the verify model's hidden state, improving acceptance rate.

**Concrete change**: Replace the single-attention EAGLE-2 draft head with the MoA variant. Requires re-training or fine-tuning the draft head (the verify model is unchanged). Training time on A10G: ~2–4 hours for the draft head alone.

**Expected TPS/E[T] effect**: +25% acceptance length reported on LLaMA-3 / Qwen-2 benchmarks; speedup +9.5% beyond EAGLE-2. On the current E[T]=5.066, +25% acceptance would push E[T] → ~6.3, which is beyond the ρ-optimal ceiling — suggesting real-world gain will be smaller (tree budget-constrained), but even +10–15% E[T] is high value.

**Greedy-identity / PPL risk**: Zero on verify. Draft head changes do not affect the verify model or acceptance criterion. PPL: unchanged (verify model unchanged).

**Local-profilable**: Yes. Draft head trains in isolation; acceptance rate can be measured cheaply on 256 prompts before a full benchmark run.

**arXiv / GitHub**: "Mixture of Attentions for Speculative Decoding" (ICLR 2025). Search iclr.cc/2025 for title. Related to EAGLE-2: https://github.com/SafeAILab/EAGLE

---

### Idea D-2: Hydra++ Sequential Draft Heads (COLM 2024, 1.31x–2x throughput)

**Mechanism**: Hydra replaces independent parallel Medusa heads with sequentially-dependent heads: head k+1 conditions on head k's output, modeling the joint token distribution across draft positions instead of assuming independence. Hydra++ adds a compact recurrent state to each head, further improving multi-token acceptance without increasing the number of verify-side checks.

**Concrete change**: Replace the current draft head with a Hydra++ head stack. The key hyperparameter is the number of sequential heads (typically 3–5). Verify integration is compatible with tree-verify (Hydra generates a tree of joint hypotheses, verified in one shot).

**Expected TPS/E[T] effect**: 1.31x–2x throughput gain reported over Medusa baselines; against EAGLE-2, the gain is smaller (EAGLE-2 already captures some sequential structure), but the paper shows +15–25% over EAGLE-2 on average. On the current baseline, conservative +10% E[T] gain is plausible.

**Greedy-identity / PPL risk**: Zero on verify. Verify model and acceptance criterion unchanged.

**Local-profilable**: Yes. Heads train in isolation from verify model.

**arXiv / GitHub**: "Hydra: Sequentially-Dependent Draft Heads for Speculative Decoding" (COLM 2024). Search ACL Anthology / arXiv for "Hydra speculative decoding sequential heads".

---

### Idea D-3: FAFO — Fumble Decoding with Lossy KV n-gram Draft (ICLR 2026, 1.20–2.71x)

**Mechanism**: Instead of a trained draft model, FAFO uses a lossy (heavily quantized or compressed) KV cache of the verify model itself as an approximate n-gram generator. The lossy cache is cheap to evaluate (low-precision matmul), and the full-precision verify step remains lossless. For short-context single-stream decode, the lossy cache can be maintained in HBM alongside the full cache, eliminating draft model inference entirely.

**Concrete change**: Replace the current trained draft model with a 4-bit or 2-bit compressed mirror of the verify model's KV cache. At each step, the compressed cache produces draft continuations via approximate attention; the full cache verifies. Implementation fits in <100 lines of Python atop FlashInfer.

**Expected TPS/E[T] effect**: 1.20–2.71x speedup reported, varying by task. For single-stream 128/128 where the KV cache is small (128 positions), the overhead of maintaining dual caches is low. Expected gain: moderate (1.2–1.5x), since n-gram match rate for general language may be lower than task-specific settings.

**Greedy-identity / PPL risk**: Low-to-zero. Verify step uses full-precision KV; lossy draft cannot affect accepted tokens. PPL: unchanged.

**Local-profilable**: Yes. No training required; KV compression runs at startup.

**arXiv / GitHub**: "FAFO: Fumble Decoding via Lossy KV Cache Draft" (ICLR 2026). Search OpenReview 2026.

---

## ANGLE 3: Verify-Side Quantization — PPL Budget and Greedy-Identity Risk

Current: int4 Marlin body, bf16 lm_head/attention (0.73% residual divergence). PPL headroom: 1.77% (2.3772 vs 2.42 ceiling). Greedy-identity is the binding constraint, not PPL.

### Idea Q-1: QSpec — Complementary Quantization for Draft/Verify (1.78x–1.80x)

**Mechanism**: QSpec uses a higher-precision verify model (bf16/fp16 or int8) paired with an aggressively quantized draft model (int4/int3), exploiting the fact that draft errors are corrected by the verify step. This is the reverse of the typical tradeoff: instead of quantizing verify, QSpec quantizes draft more aggressively. The key insight for the current setup: if the draft model is already int4 and the verify model is int4+bf16 hybrid, QSpec suggests moving the draft to int3 or int2 to save draft inference time, while keeping verify at current precision.

**Concrete change**: Quantize the draft head to int3 (GPTQ or AWQ) while keeping the verify model at current int4+bf16. Measure draft inference latency reduction vs. E[T] degradation.

**Expected TPS/E[T] effect**: QSpec reports 1.78–1.80x throughput. In the single-stream setting, the draft model runs once per tree step (not once per token); draft latency may be a small fraction of total step time. Estimate: 5–10% step time reduction if draft is bandwidth-bound.

**Greedy-identity / PPL risk**: Draft quantization does not affect verify or acceptance; greedy-identity is unchanged. PPL: unchanged (verify model unchanged).

**Local-profilable**: Yes. Draft model can be quantized and E[T] profiled offline in 30 min.

**arXiv / GitHub**: "QSpec: Speculative Decoding with Complementary Quantization Schemes" arXiv:2411.11514.

---

### Idea Q-2: W8A8-FP8 Verify Body on Ampere sm_86 — Lossless Precision Upgrade Path

**Mechanism**: Recent comprehensive study ("Give Me BF16 or Give Me Death?", arXiv:2407.xxxxx) shows W8A8-FP8 is effectively lossless for Llama-3.1 class models, with <0.05 PPL regression. On Ampere (sm_86), FP8 GEMM is emulated via int8 path but can still yield ~1.3–1.5x throughput on GEMM-heavy layers. For the verify model's body GEMMs, replacing int4+bf16 with W8A8-FP8 may improve throughput while closing the residual 0.73% greedy divergence.

**Concrete change**: Quantize the verify model body to W8A8-FP8 using the `llm-compressor` or `AutoFP8` library. Keep lm_head in bf16. Measure PPL vs. current int4+bf16 hybrid; if PPL improves (<2.3772) with greedy-identity passing, this is a win.

**Expected TPS/E[T] effect**: On Ampere where FP8 hardware is not native, GEMM throughput gain is 1.0–1.3x (emulated). Primary value: closes the 0.73% divergence gap without further PPL cost.

**Greedy-identity / PPL risk**: FP8 on Ampere should be lossless per the cited study; PPL regression <0.05 is well within the 1.77% headroom. Greedy-identity: requires measurement. Risk: medium.

**Local-profilable**: Yes. PPL and greedy-identity can be evaluated in <1 hour on the local A10G.

**arXiv / GitHub**: "Give Me BF16 or Give Me Death? Evaluating FP8 Quantization for Large Language Models" (2024). AutoFP8: https://github.com/neuralmagic/AutoFP8

---

### Idea Q-3: KVQuant — 3-bit KV Cache with <0.1 PPL Regression (~1.7x decode speedup)

**Mechanism**: KVQuant applies per-channel / pre-RoPE / non-uniform quantization to the KV cache, exploiting the observation that pre-RoPE key vectors have stable channel distributions across tokens. At 3-bit, PPL regression is <0.1 on LLaMA benchmarks. Custom CUDA kernels handle dequant-fused attention. For 128-position KV cache on A10G, total KV memory is small, but dequant during attention still saves memory bandwidth.

**Concrete change**: Replace the fp16/bf16 KV cache with KVQuant 3-bit. Integrate with FlashInfer's custom attention dispatch. The custom CUDA kernel compiles for sm_86.

**Expected TPS/E[T] effect**: ~1.7x decode-phase speedup reported on long-context benchmarks. For 128-position context, KV cache is already tiny and bandwidth savings are modest; expect 5–15% step time reduction rather than 1.7x.

**Greedy-identity / PPL risk**: PPL: <0.1 regression at 3-bit — well within 1.77% headroom, but must be verified. Greedy-identity: 3-bit KV quantization changes attention outputs, making token-exact greedy match unlikely. This is the BINDING RISK for this idea — it likely fails the greedy-identity gate.

**Local-profilable**: Yes, and profilability is the correct approach here — measure greedy-identity failure rate before committing to a full run.

**arXiv / GitHub**: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization" (NeurIPS 2024). arXiv:2401.18079. GitHub: https://github.com/squeezeailab/KVQuant

---

### Idea Q-4: Progressive Mixed-Precision Decoding (PMPD) — Gradually Lower Precision (1.4–12.2x matmul speedup)

**Mechanism**: PMPD observes that tokens generated later in the sequence can tolerate lower precision because early high-confidence tokens anchor the context. Precision degrades from FP16 at position 0 to INT4 by position 128. This directly maps to the 128-token generation window in the current task.

**Concrete change**: Apply PMPD's precision schedule to the verify model's GEMMs during speculative decode: positions 0–32 at int8 (already near current), positions 33–96 at int4, positions 97–128 at int3. The draft model is unchanged.

**Expected TPS/E[T] effect**: 1.4–12.2x matmul speedup on matrix-vector (decode-phase) operations. In the 128-token regime, average precision reduction is ~25%; expect 15–25% GEMM speedup in latter half of generation.

**Greedy-identity / PPL risk**: HIGH RISK for greedy-identity. Changing verify model precision mid-sequence changes accept/reject decisions. This likely fails the token-exact greedy-identity gate. Recommend PPL measurement first as a quick kill/keep signal.

**Local-profilable**: Yes — PPL and greedy-identity measurement first, then decide.

**arXiv / GitHub**: "PMPD: Progressive Mixed-Precision Decoding for Efficient LLM Inference" arXiv:2410.13461.

---

## ANGLE 4: Kernel / Systems — A10G Single-Stream Launch and Bandwidth Optimization

### Idea K-1: FlashInfer Customizable JIT Attention with CUDAGraph (Highest systems leverage)

**Mechanism**: FlashInfer provides a JIT-compiled, customizable attention engine that is natively CUDAGraph-compatible. The key: at decode time (128-position KV, single stream), attention is memory-bandwidth-bound. FlashInfer's decode kernel is tuned for small-KV / single-stream workloads and outperforms stock FlashAttention-2 in this regime. Integration with vLLM is already supported via a config flag.

**Concrete change**: Enable FlashInfer's decode-phase attention in vLLM (`--attention-backend flashinfer`). Wrap the full step (draft expand, verify, sample) in a CUDAGraph replay after 1 warmup step. This eliminates per-step Python/CUDA launch overhead (~0.5–2ms per step on A10G).

**Expected TPS/E[T] effect**: CUDAGraph elimination of launch overhead: +2–5% TPS in bandwidth-bound decode. FlashInfer decode kernel throughput advantage over FlashAttention-2 at small KV size: +5–15%. Combined: potentially +7–18% TPS, significant.

**Greedy-identity / PPL risk**: Zero. FlashInfer implements the same mathematical attention operation; outputs are bit-identical to reference for fp16/bf16 (the kernel is a fused implementation of exact matmul+softmax+matmul). CUDAGraph replay is deterministic.

**Local-profilable**: Yes. Enable via vLLM flag; profile TPS immediately. If CUDAGraph shape conflicts arise, fall back to eager mode.

**arXiv / GitHub**: "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving" arXiv:2501.01005. GitHub: https://github.com/flashinfer-ai/flashinfer. vLLM FlashInfer integration: https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/flashinfer.py

---

### Idea K-2: Persistent Speculative Decode Kernel — Eliminate Step-Boundary Overhead

**Mechanism**: Standard speculative decode launches separate CUDA kernels for (1) draft expansion, (2) verify forward, (3) acceptance sampling. Each kernel launch incurs ~10–30 μs of CPU-side overhead plus CUDA stream synchronization. A persistent kernel fuses all three into a single kernel that loops internally, eliminating inter-step launch overhead for the full 128-step generation.

**Concrete change**: Profile step-boundary overhead using `nsys profile` on the current vLLM serving loop. If per-step launch overhead is >0.5ms, implement a persistent kernel wrapper that replays the draft-verify-accept triple without returning to CPU between steps. Alternatively, extend the CUDAGraph from K-1 to capture the entire 128-step loop.

**Expected TPS/E[T] effect**: Per-step launch overhead on A10G is typically 0.3–1ms. At 128 steps, eliminating this saves 38–128ms total for a 128-token generation. At baseline ~10ms/step (481.53 TPS / 128 tokens per query ≈ 3.76 queries/s → 267ms per query), this is 15–48% latency reduction — but only if launch overhead is the bottleneck. Must be measured first.

**Greedy-identity / PPL risk**: Zero if the fused kernel implements the same operations. Launch overhead elimination does not change arithmetic.

**Local-profilable**: Yes — `nsys profile` on A10G in <30 min gives exact overhead numbers before any implementation.

**arXiv / GitHub**: Persistent kernel technique is standard CUDA; no specific paper. See PyTorch 2.x `torch.compile` with `mode="reduce-overhead"` as a lightweight alternative before custom kernel work.

---

### Idea K-3: Fused Speculative Sampler — Eliminate Accept/Reject CPU Roundtrip

**Mechanism**: In standard speculative decode, the accept/reject step (comparing draft token probability to verify token probability) runs on CPU, requiring GPU→CPU transfer of logits and CPU→GPU transfer of the accept mask. For M=8 verify width, this is a small transfer but the synchronization barrier costs ~0.2–0.5ms per step.

**Concrete change**: Implement the accept/reject criterion as a CUDA kernel that runs on GPU immediately after the verify forward pass, returning only the final accepted token sequence (not the full logit tensor). This eliminates one CPU-GPU round-trip per step.

**Expected TPS/E[T] effect**: 0.2–0.5ms per step elimination → +2–5% TPS on the A10G at current step times.

**Greedy-identity / PPL risk**: Zero if the GPU kernel implements the identical criterion. The greedy accept/reject criterion is deterministic arithmetic.

**Local-profilable**: Yes — the criterion is <20 lines of CUDA; unit-testable against the CPU reference.

**arXiv / GitHub**: Technique described in vLLM speculative decoding implementation. See also SpecInfer (arXiv:2305.09781) for the original GPU-side accept kernel design.

---

### Idea K-4: Max-Speedup Speculative Sampling — Optimal Acceptance Criterion (ICLR 2026)

**Mechanism**: "Max-Speedup Speculative Sampling" (ICLR 2026) proves that the standard independent-token acceptance criterion is suboptimal for throughput. The paper derives the acceptance criterion that maximizes expected speedup (rather than maximizing the probability of each individual token being accepted), and shows that the optimal criterion accepts more tokens on average by exploiting correlations between the draft and verify distributions.

**Concrete change**: Replace the current acceptance criterion (compare p_draft / p_verify per token) with the max-speedup criterion derived in the paper. This is a change to the sampler logic only, not to the model or tree topology.

**Expected TPS/E[T] effect**: Paper reports 30% block efficiency gain, 15% walltime reduction. If applicable to the current tree-verify setting (the paper focuses on block verification), this is a large gain. Verify: the greedy-identity constraint may conflict with the optimal criterion if it allows accepting tokens the greedy reference would not accept.

**Greedy-identity / PPL risk**: CRITICAL RISK. The max-speedup criterion is designed for distribution-matching (not greedy), so it may accept tokens that differ from greedy reference tokens. Must verify greedy-identity compatibility before any deployment. This is a potential blocker.

**Local-profilable**: Yes — acceptance criterion can be swapped and greedy-identity checked on 128 prompts in <15 min.

**arXiv / GitHub**: "Greedy Multi-Path Block Verification for Speculative Decoding" (ICLR 2026). Search iclr.cc/2026 for title. Also related: "Max-Speedup Speculative Sampling" ICLR 2026.

---

## ANGLE 5: Alternative Draft Sources — Beyond Trained Draft Models

### Idea N-1: REST / n-gram Datastore Draft (Training-Free, Low-Overhead)

**Mechanism**: REST (Retrieval-based Speculative Decoding) and the simpler n-gram draft use a datastore of token n-grams (built from the prompt corpus or a reference corpus) to generate draft continuations without a trained draft model. For repetitive or templated prompts (e.g., code, structured text), n-gram match rates can be very high.

**Concrete change**: Build an n-gram datastore from the 128 prompts in the benchmark. At each decode step, look up the last 3–5 tokens as a key and retrieve the most likely continuation. Feed these as draft tokens into the existing verify tree.

**Expected TPS/E[T] effect**: Highly dependent on prompt distribution. If prompts are diverse natural language, n-gram match rates may be low (E[T] < 2). If prompts contain structured patterns, match rates can be high (E[T] > 4). Must profile on the actual benchmark prompts before committing.

**Greedy-identity / PPL risk**: Zero — verify model and criterion unchanged.

**Local-profilable**: Yes — n-gram lookup is trivially implementable in <50 lines of Python. Profile match rate on benchmark prompts before any further work.

**arXiv / GitHub**: "REST: Retrieval-Based Speculative Decoding" arXiv:2311.08252. GitHub: https://github.com/fasterdecoding/REST

---

## Ranked Priority Table

| Rank | Idea | Mechanism Level | Expected TPS Gain | Greedy-ID Risk | Local-Profilable | Implementation Effort |
|------|------|----------------|-------------------|----------------|------------------|-----------------------|
| 1 | **T-1: OPT-Tree** | Topology (online DP) | +3–10% | Zero | Yes | Low (50-line DP) |
| 2 | **K-1: FlashInfer + CUDAGraph** | Kernel/systems | +7–18% | Zero | Yes | Low (flag + warmup) |
| 3 | **D-1: Mixture of Attentions (MoA)** | Draft architecture | +10–25% E[T] | Zero | Yes | Medium (draft retrain 4h) |
| 4 | **T-2: CAST** | Topology (device-aware) | +8–19% | Zero | Yes | Low (calibration script) |
| 5 | **T-3: C2T** | Tree pre-pruning | +7–17% on top | Low (0.5% E[T] degradation) | Yes | Medium (classifier train 30m) |
| 6 | **D-2: Hydra++ heads** | Draft architecture | +10–25% over EAGLE-2 | Zero | Yes | Medium (draft retrain) |
| 7 | **K-2: Persistent kernel** | Systems | +2–15% (if launch-bound) | Zero | Yes (nsys first) | Medium-High |
| 8 | **Q-1: QSpec draft int3** | Quantization (draft) | +5–10% | Zero | Yes | Low |
| 9 | **K-3: Fused sampler** | Systems | +2–5% | Zero | Yes | Low-Medium |
| 10 | **T-4: Sequoia DP** | Topology (offline DP) | +3–8% | Zero | Yes | Low |
| 11 | **D-3: FAFO** | Draft source | +5–20% (task-dependent) | Zero | Yes | Low-Medium |
| 12 | **K-4: Max-speedup criterion** | Acceptance criterion | +15–30% | HIGH (greedy-identity) | Yes (quick check) | Low |
| 13 | **N-1: REST / n-gram** | Draft source | Task-dependent | Zero | Yes (quick profiling) | Very Low |
| 14 | **Q-2: W8A8-FP8 verify** | Quantization (verify) | +5–10% + fixes divergence | Medium | Yes | Low |
| 15 | **Q-3: KVQuant 3-bit** | KV quantization | +5–15% in this context | HIGH (greedy-ID) | Yes (quick kill check) | Medium |
| 16 | **Q-4: PMPD** | Mixed-precision decode | +15–25% matmul | HIGH (greedy-ID) | Yes | Medium |

---

## Recommended First Experiments (Sequential Cheapest Diagnostics)

**Step 1 (30 min, zero risk)**: Enable FlashInfer decode backend in vLLM and wrap in CUDAGraph. Measure TPS. If ≥+3% → merge immediately (zero-risk systems win).

**Step 2 (1 hour, zero risk)**: Implement OPT-Tree DP topology construction. Measure E[T] on 128 prompts vs. current static tree. If E[T] improves → deploy and benchmark.

**Step 3 (15 min, greedy-ID kill check)**: Swap in max-speedup acceptance criterion (K-4). Run greedy-identity check on 128 prompts. If passes → full benchmark. If fails → rule out immediately.

**Step 4 (2 hours, zero risk)**: Re-train draft head with Mixture of Attentions. Measure E[T] on 256 prompts. If E[T] → 5.5+ → full benchmark.

**Step 5 (30 min)**: Run `nsys profile` to measure exact per-step launch overhead. If >0.5ms → pursue K-2 persistent kernel. If <0.2ms → deprioritize.

---

## Stop Conditions

- **K-1 FlashInfer + CUDAGraph**: Stop if TPS gain <1% — then launch overhead is not the bottleneck.
- **T-1 OPT-Tree**: Stop if E[T] gain vs. static tree <0.05 — DP overhead may not be worth it.
- **K-4 Max-speedup criterion**: Stop immediately if greedy-identity check fails on >1% of prompts.
- **Q-3 KVQuant, Q-4 PMPD**: Stop at quick greedy-identity kill check — if >0.1% failure rate, these violate the benchmark contract.
- **D-1 MoA draft retrain**: Stop if E[T] after 4h training run is <5.2 — overhead of draft retrain is not worth <0.2 E[T] gain.

---

## Research State Update

**Current best explanation**: The primary TPS bottleneck is the gap between deployed E[T]=5.066 and the ρ-optimal ceiling E[T]=5.219 (topology) plus per-step launch overhead (systems). The residual 0.73% greedy-identity divergence in bf16 attention/lm_head is a secondary risk that limits verify-side quantization options.

**Highest-leverage unblocked experiments**: (1) FlashInfer + CUDAGraph (zero risk, low effort, potentially +7–18%), (2) OPT-Tree per-step adaptive topology (zero risk, low effort, closes E[T] gap), (3) Mixture of Attentions draft head retrain (zero risk on verify, medium effort, potentially +10–25% E[T]).

**Ruled out without new evidence**: Any verify-side quantization that changes attention outputs (KVQuant at 3-bit, PMPD) — greedy-identity constraint is binding. Any large-batch amortization argument — single-stream constraint is hard.

**Open uncertainties**: (1) Exact fraction of per-step time spent in launch overhead vs. compute vs. bandwidth — `nsys` profile resolves this. (2) Whether the draft model's E[T] ceiling is architectural (draft head capacity) or topological (tree shape) — OPT-Tree vs. MoA comparison resolves this.

**Next discriminating experiment**: FlashInfer + CUDAGraph (K-1) — zero risk, 30-min implementation, directly measures the systems overhead contribution to the TPS gap.
