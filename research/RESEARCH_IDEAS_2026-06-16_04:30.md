# Research Ideas: Pushing Past the 467 TPS Equivalence Frontier
**Date:** 2026-06-16 04:30  
**Context:** Single-stream (batch-size-1) decode throughput for `google/gemma-4-E4B-it` on single NVIDIA A10G (sm_86, 24GB, ~600 GB/s peak DRAM BW). Hard constraint: byte-exact greedy-decode token-identity with reference model (every token matches argmax bit-for-bit). Benchmark: 128 prompts × 128 new tokens.

**Current state:**
- Deployed incumbent: 481.53 TPS (NON-equivalent, flips ~3 tokens per 128, OUTSIDE feasible set)
- Strict equivalence frontier: 467.14 TPS (K=7/M=8, ONEGRAPH=1, int4)
- Hard ceiling: 520.95 TPS (verify-bandwidth λ=1 wall)
- Equivalence tax: ~14.4 TPS gap between deployed and equivalent frontier
- Live lead: verify-attention joint-autotune models +15.86 TPS (→483.0) — unproven end-to-end (evaporated ≤0 twice before)

**Stack context:**
- vLLM 0.22.1rc1, MTP speculative decoding, K=7 drafter + M=8 verify = ONEGRAPH
- Marlin int4 GEMM: ~21% of verify wall, BW-saturated at ~92% peak DRAM BW
- Non-GEMM: ~79% of verify wall = per-layer DISPATCH + elementwise (RMSNorm/GeGLU/residual/RoPE) + memory contention
- Attention: ~7–9% of verify, M-flat (does not scale with M=8 batch)
- Acceptance: a₁≈0.727, E[accept-length]≈3.82 of 7, sharp cliff at position-1
- Drafter forward: ~1.433 ms (~18% of cycle); Verify forward: ~6.445 ms (~82% of cycle)

**Closed levers (do not repeat without new evidence):**
- CUDA-graph capture of K=7/M=8 loop (ONEGRAPH=1, already deployed)
- Marlin GEMM config tuning (BW-saturated at 92%, minimal headroom)
- Async KV-cache L2 prefetch in Triton verify-attention (null result, host overhead 0.46% < 0.5% gate)
- Standard attention autotune variants (null results multiple times)
- Any M-value change (ONEGRAPH restructuring cost too high, M=8 already near optimal E[accepted])

---

## Direction 1: Lookahead / Jacobi Exact Parallel Decoding — No Drafter Weights

### Hypothesis
Replace or supplement the MTP drafter with training-free Jacobi-iteration lookahead decoding that generates multiple token candidates in parallel via n-gram consistency windows. Because the verify step remains byte-exact argmax, token identity is guaranteed.

### Mechanism + Why This Stack
Lookahead Decoding (Fu et al., 2024, ICML) exploits the observation that Jacobi iterations converge locally: given a window W of "guess" tokens, re-run the model forward and accept any consistent n-gram (length N) that has stabilized. The acceptance condition is deterministic argmax agreement — there is no probabilistic sampling, no rejection sampling, no temperature adjustment. This is structurally byte-exact because:
1. Every accepted token is the argmax of the full model run over the actual preceding context.
2. No auxiliary model is involved — the main model IS the verifier.
3. The lookahead and verification windows can be sized to fit the A10G's L2 cache (6.29 MB) without exceeding DRAM pressure.

**Why it could beat 467 on this stack:** The MTP drafter currently consumes ~1.433 ms per step plus cross-attention overhead, with E[accepted]=3.82 of 7 (54.6% acceptance). Lookahead generates candidates via a second forward pass over the same weights (no separate drafter weights) — at batch-size W+N-1 instead of 1. On A10G DRAM-bound inference, doubling the "batch" of a 4B model from 1 to ~10 tokens may yield more accepted tokens per second than the current MTP approach, particularly if the lookahead window can be sized for L2 reuse.

### Byte-Exact Identity Preservation
The original paper explicitly proves this: Theorem 1 in Fu et al. (2024) shows that lookahead decoding produces the identical distribution as autoregressive decoding. Concretely, a token is accepted if and only if it equals argmax(model(full_context)). The verify step never accepts a token that the reference model would not have produced.

### Cheapest Validation Experiment (~90 min, single A10G)
1. Install the reference `lookahead-decoding` library from `hao-ai-lab/LookaheadDecoding` or use the FlashInfer lookahead integration (FlashInfer 0.2+).
2. Configure: lookahead window W=5, guess length N=7, no drafter.
3. Run the 128×128 benchmark with strict greedy-identity gate against the reference model.
4. Compare: TPS vs 467.14, token-identity rate (must be 1.0 to pass the gate).
5. Sweep W ∈ {3,5,7} in the remaining time budget.

**Expected observable:** If lookahead E[accepted] ≥ 3.82 with lower per-step overhead than the MTP drafter (no cross-attention), TPS should exceed 467. If E[accepted] < 3.0, close this direction.

### Citation
- Fu, Y. et al. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." ICML 2024. arXiv:2402.02057. GitHub: `hao-ai-lab/LookaheadDecoding`.
- Shyam, P. et al. "Towards Optimal Multi-Draft Speculative Decoding." ICLR 2025. arXiv:2502.18779. (theoretical grounding for multi-candidate acceptance)

### Risk of Collapse
- Lookahead on a 4B model may have lower acceptance rate than on the 70B models in the original evaluation; the n-gram window is more likely to diverge on compact models.
- vLLM integration may require non-trivial kernel changes to support the W×N rectangular batched forward pass within the ONEGRAPH capture.
- If the additional memory traffic from the wider batch exceeds savings from parallel acceptance, TPS regresses.

---

## Direction 2: N-Gram Prompt-Lookup Drafting — Zero Auxiliary Model Cost

### Hypothesis
Replace the MTP drafter entirely with a training-free n-gram retrieval draft from the prompt (prompt-lookup decoding / PLD). For prompts with repetition or domain patterns, the draft cost drops from ~1.433 ms (MTP drafter forward) to sub-millisecond hash-lookup. Since the verify step is unchanged, token identity is preserved exactly.

### Mechanism + Why This Stack
Prompt-Lookup Decoding (Saxena, 2023) works by: given the last k tokens as a query, search the prompt+context for matching n-gram continuations. The top-K matches are used as draft tokens. This costs ~O(context_length) with simple string matching — roughly 10–100× cheaper than any neural drafter forward pass. The crucial property: the verify step runs identically and remains the sole token arbiter, so byte-exact identity is structurally guaranteed.

**Why it could beat 467 on this stack:**
- MTP drafter costs ~1.433 ms = ~18% of cycle time. Even halving drafter cost to ~0.7 ms would reclaim ~35+ TPS toward the ceiling.
- For the 128-prompt benchmark, many prompts likely contain repeated phrases or domain n-grams that match well.
- "The N-Grammys" (Bhatt & Cohen, 2024, arXiv:2411.03786) demonstrates this on production workloads: lossless, zero training, 1.2–1.8× throughput gain.
- Combined with the existing ONEGRAPH verify, the draft phase becomes a fast CPU/CUDA lookup rather than a neural forward pass.

### Byte-Exact Identity Preservation
The verify pass is unchanged and remains byte-exact argmax. Draft tokens are speculative only — none are emitted without passing through the verify filter. The n-gram lookup is purely a "guess generator"; correctness is independent of guess quality.

### Cheapest Validation Experiment (~90 min, single A10G)
1. Implement a CUDA/Python n-gram index over the running context (KV cache indices already exist).
2. On each draft step, query: last 3-token n-gram → 7-token continuation from context.
3. Fall back to MTP drafter when no match is found (hybrid mode).
4. Benchmark: measure E[accepted] with n-gram drafts vs MTP drafts; measure TPS.
5. Binary gate: if TPS > 467 AND token-identity = 1.0, log as winner.

**Key tunable:** n (query length) ∈ {2,3,4}; K (number of candidates). Start with n=3, K=1.

### Citation
- Saxena, A. "Prompt Lookup Decoding." GitHub 2023. `apoorvumang/prompt-lookup-decoding`.
- Bhatt, J. & Cohen, W. "The N-Grammys: Improving Lossless Speculative Decoding with Batched n-gram Lookups." arXiv:2411.03786, 2024.
- "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies." arXiv:2502.05202, 2025. (2.8× speedup, no training, no shared vocab constraint)

### Risk of Collapse
- Gemma-4-E4B-it is a chat model; instruction-following prompts may have low n-gram repeat rate, collapsing acceptance to near zero.
- If E[accepted] with n-gram drafts < 2.5, the cheaper draft cost doesn't compensate for lower acceptance, and hybrid mode falls back to MTP with added overhead.
- The 128-prompt benchmark may be artificially low on n-gram matches if prompts are diverse. Pre-analyze prompt repetition rates before investing GPU time.

---

## Direction 3: CLaSp / Self-Speculative Layer-Skip — Zero Extra Weights, Plug-and-Play

### Hypothesis
Apply Context-Aware Layer Skipping with Speculation (CLaSp, ACL 2025) to the verify forward pass: dynamically skip a subset of transformer layers during the speculative draft phase, using the full model only for verification. This reduces drafter cost without any additional weights, fine-tuning, or distribution change.

### Mechanism + Why This Stack
CLaSp (Zhao et al., ACL 2025, arXiv:2505.24196) uses dynamic programming to find the optimal set of layers to skip during drafting, exploiting the observation that lower layers are more critical for getting the direction right, while middle layers are often redundant for speculation purposes. Crucially:
- No training or fine-tuning required — plug-and-play.
- Token identity is preserved: the FULL model (no skips) verifies all draft tokens. The verify step remains byte-exact.
- DP finds the optimal skip pattern per-context, trading accuracy for speed.

**Why it could beat 467 on this stack:**
- Gemma-4-E4B is ~26 transformer layers. If CLaSp can skip 4–6 middle layers during drafting, MTP drafter forward drops from ~1.433 ms to ~1.1 ms — freeing ~15–20 TPS toward the ceiling.
- This is orthogonal to the GEMM and attention optimizations already tried. It attacks drafter cost, not verify cost.
- The paper reports 1.3×–1.7× speedup on LLaMA3-8B without distribution change, which maps structurally to this setting.

### Byte-Exact Identity Preservation
The verify step runs ALL layers of the full model with no skipping. A token is accepted only if it matches the full-model argmax exactly. Layer skipping affects only the speculative draft tokens — which are never emitted without verify confirmation.

### Cheapest Validation Experiment (~90 min, single A10G)
1. Profile which 4–6 layers contribute least to draft acceptance via a brief ablation: run MTP forward with layer i masked as identity (skip), measure E[accepted] drop.
2. Fix the DP-optimal skip set (static for the benchmark).
3. Measure TPS and token-identity on 128×128 benchmark.
4. Target: TPS > 467, token-identity = 1.0.

**Optional:** Use CLaSp's DP formulation to find the skip set that maximizes E[accepted]/compute trade-off.

### Citation
- Zhao, X. et al. "CLaSp: Efficient Inference with Context-Aware Layer Skipping and Speculation." ACL 2025 (long paper). arXiv:2505.24196.
- DEL: "Dynamic Exit Layer for Efficient Decoding." COLM 2025. (context-aware dynamic exit, also lossless)
- CAS-Spec: "Cascade Adaptive Self-Speculative Decoding." NeurIPS 2025. (on-the-fly, lossless, no training)

### Risk of Collapse
- Gemma's MTP drafter is already a separate cross-attention module (not simple early-exit from the backbone). CLaSp as described assumes the main model IS the drafter. This requires adapting the skip logic to the MTP architecture — non-trivial.
- If Gemma-4-E4B has "thin" residual paths (attention+MLP each contribute strongly), skipping layers may collapse E[accepted] below the break-even point.
- CLaSp's DP calibration requires a representative calibration set; if prompt distribution in benchmark differs from calibration, the skip set may be suboptimal.

---

## Direction 4: FlashInfer JIT-Compiled Fused Verify Attention Kernels

### Hypothesis
Replace the current Triton verify-attention kernel with FlashInfer's JIT-compiled attention engine, which generates fused CUDA kernels tuned to the exact (M=8, head_dim, seq_len) configuration of this benchmark, recovering the verify-attention autotune instability observed in prior runs.

### Mechanism + Why This Stack
FlashInfer (Ye et al., 2025, arXiv:2501.01005) provides a customizable attention library with:
- JIT compilation of attention kernels per (batch_size, seq_len, head_dim, dtype) — no generic code paths.
- Paged KV-cache support natively, matching vLLM's memory layout.
- Cascaded/speculative decoding attention variants with built-in M-batch verify support.
- Persistent kernel options that eliminate per-kernel dispatch overhead.

**Why it could beat 467 on this stack:**
- Verify-attention is ~7–9% of the verify wall. The current Triton kernel's autotune was shown to model +15.86 TPS (→483) but evaporated end-to-end twice. FlashInfer's JIT approach eliminates the autotune instability: the kernel is compiled ONCE for the exact static shapes of the 128×128 benchmark.
- FlashInfer's fused kernel avoids the round-trip between PyTorch dispatcher → Triton → CUDA for each attention head, reducing per-layer dispatch overhead.
- For M=8 verify (batch 8), FlashInfer's split-KV / cascaded attention is specifically optimized.

### Byte-Exact Identity Preservation
FlashInfer is a drop-in replacement for the attention computation. It computes the same mathematical operation (softmax(QKᵀ/√d)V) in IEEE-754 float16 arithmetic. At identical precision, the output is numerically equivalent to the reference. The greedy-identity gate will confirm equivalence.

### Cheapest Validation Experiment (~90 min, single A10G)
1. Install FlashInfer 0.2.x: `pip install flashinfer-python --index-url https://flashinfer.ai/whl/cu124/torch2.4/`.
2. Swap the verify-attention backend: set `VLLM_ATTENTION_BACKEND=FLASHINFER`.
3. Run greedy-identity gate (single prompt, 128 tokens) to confirm equivalence.
4. Run 128×128 TPS benchmark.
5. Compare: TPS vs 467.14. If > 467 and identity = 1.0, log as winner.

**Expected observable:** If FlashInfer's persistent kernel reduces per-layer attention dispatch, TPS should improve vs current Triton kernel. If TPS doesn't change, the bottleneck is not attention dispatch.

### Citation
- Ye, Z. et al. "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." arXiv:2501.01005, 2025. GitHub: `flashinfer-ai/flashinfer`.
- vLLM FlashInfer integration: `vllm/attention/backends/flashinfer.py` (already in vLLM codebase — may be a config flag change only).

### Risk of Collapse
- vLLM 0.22.1rc1 may have partial or broken FlashInfer integration for the MTP speculative decoding path; the cascaded attention API differs from standard attention.
- FlashInfer's numerics for float16 may differ from the reference implementation at the 1e-4 level, causing greedy-identity failures on borderline logit differences.
- The ONEGRAPH CUDA graph may not be compatible with FlashInfer's JIT-compiled kernels without re-capture.

---

## Direction 5: Fused RMSNorm + GeGLU + Residual Megakernel (Persistent Kernel, Triton)

### Hypothesis
Fuse the per-layer elementwise sequence (RMSNorm → linear projection → GeGLU activation → residual add) into a single persistent Triton kernel that reads input once, writes output once, and eliminates intermediate DRAM round-trips for the ~79% non-GEMM verify overhead.

### Mechanism + Why This Stack
The verify forward pass's ~79% non-GEMM time is dominated by per-layer DISPATCH + elementwise operations. Each layer executes:
1. RMSNorm: read X, compute sum(x²), normalize, write X_norm.
2. QKV/gate+up projections: Marlin int4 GEMM (already optimized).
3. GeGLU: read gate, up; compute gelu(gate)*up; write.
4. Residual add: read X_prev + X_out; write X_new.

Each of these is a separate kernel launch with its own DRAM read/write round-trip. For a 4B model with ~26 layers × M=8 verify tokens, this is ~26 × 4 = 104 separate kernel dispatches per step, each touching ~4-16 MB of DRAM.

A fused persistent Triton kernel (ClusterFusion style, NeurIPS 2025) would:
- Keep X in SRAM/registers across norm → activation → residual.
- Reduce DRAM reads by ~3× per layer for the elementwise sequence.
- Eliminate ~78 of 104 dispatch overheads per step.

**Why it could beat 467:** If non-GEMM elementwise accounts for ~30% of verify wall time (conservative estimate within the 79% non-GEMM budget), and fusion reduces DRAM traffic by ~3×, this could reclaim 8–15 TPS. The ceiling for this direction is bounded by the elementwise share of verify wall.

**FFN Fusion angle (NeurIPS 2025 spotlight):** "FFN Fusion: Rethinking Sequential FFN Layers in Transformers" proposes fusing consecutive FFN blocks to eliminate inter-block DRAM round-trips. For Gemma's architecture with alternating attention+FFN, consecutive FFN blocks may be fusible.

### Byte-Exact Identity Preservation
Mathematical equivalence: fusing elementwise operations in Triton preserves IEEE-754 semantics if accumulation order is respected. The fusion changes kernel launch patterns, not arithmetic operations or precision. The greedy-identity gate will confirm.

**Caveat:** If the fusion changes accumulation order (e.g., vectorizing across token positions), there may be float16 rounding differences. This must be caught by the identity gate. Use `torch.allclose(rtol=0, atol=0)` check on a single-token forward pass before benchmarking.

### Cheapest Validation Experiment (~90 min, single A10G)
1. Write a Triton kernel that fuses RMSNorm + residual for a single layer (not full FFN — just the pre-GEMM norm and post-GEMM residual).
2. Measure kernel launch time: current (2 launches) vs fused (1 launch) for the A10G.
3. If savings are > 0.1 ms/step, extend to full layer fusion.
4. Run greedy-identity gate to confirm numerical equivalence.
5. Run TPS benchmark.

**Key question this experiment answers:** Is per-kernel dispatch overhead (not DRAM BW) the binding constraint on the non-GEMM 79%? If launch overhead is < 5 μs/kernel and DRAM BW is the real limit, fusion of small elementwise kernels won't help.

### Citation
- "ClusterFusion: Accelerating Grouped Computation for Transformers on the GPU." NeurIPS 2025. (operator fusion via cluster-level collective primitives)
- "FFN Fusion: Rethinking Sequential FFN Layers in Transformers." NeurIPS 2025 (spotlight). (fusing consecutive FFN blocks)
- Triton persistent kernel tutorial: `triton-lang/triton`, `tutorials/06-fused-attention.py`. (reference for persistent kernel patterns)
- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. (vLLM architecture baseline)

### Risk of Collapse
- Writing a correct, high-performance fused Triton kernel for Gemma's architecture (SwiGLU variant) requires ~3–5 days of careful engineering; in a 90-minute experiment only a partial prototype is feasible.
- The fused kernel must match vLLM's tensor layouts (PagedAttention KV format) exactly; layout mismatches are a common failure mode.
- If per-kernel dispatch overhead is already sub-microsecond on A10G (CUDA graph already captures the launch sequence), the savings may be < 0.5 TPS.

---

## Direction 6: Multi-Candidate Speculative Decoding with Tree Verification

### Hypothesis
Replace the current single-sequence MTP draft with a multi-candidate tree-structured draft, verifying multiple speculative branches in a single batched forward pass. This increases expected accepted tokens per verify cycle without increasing verify wall-time proportionally.

### Mechanism + Why This Stack
Multi-candidate speculative decoding (Miao et al., 2024; DySpec, ICLR 2025) generates a token tree where each node branches into B candidate continuations. The verify pass scores all branches simultaneously in a batched forward pass. The acceptance condition remains byte-exact: a branch is accepted only if every token on the path matches argmax of the full model.

**Why it could beat 467 on this stack:**
- Current: E[accepted] = 3.82 tokens per 7 draft tokens. Tree decoding with branching factor B=2 at positions where a₁ < 0.8 could increase E[accepted] to ~5.0+, even with the same verify window.
- The "sharp acceptance cliff at position-1" (a₁≈0.727) is exactly the signal for tree expansion: position 1 has a 27.3% rejection rate, meaning 27.3% of cycles currently waste 6 draft tokens. A tree that branches at position 1 recovers most of those wasted cycles.
- DySpec (ICLR 2025) dynamically adjusts tree structure based on token-level acceptance probabilities, exactly matching this calibration data.
- Traversal Verification for Speculative Tree Decoding (NeurIPS 2025) proves that tree-level verification is lossless even when tokens at intermediate nodes are uncertain.

**The math:** If branching at position 1 increases E[accepted] from 3.82 to 4.8 with only 10% more verify FLOPs (wider batch for position-1 forward), TPS gain = (4.8/3.82 - 1) × 467 × (1 - 0.1) ≈ +53 TPS in the optimistic case. Even at 50% of that estimate, +26 TPS would cross the 481 barrier cleanly.

### Byte-Exact Identity Preservation
Tree verification is structurally byte-exact: a token at position i is accepted if and only if it equals argmax(model(context + accepted_prefix[:i])). The verify pass computes this for all tree nodes simultaneously; no token is emitted without full-model confirmation.

### Cheapest Validation Experiment (~90 min, single A10G)
1. Configure: tree branching only at position 1 (B=2), all other positions linear (no branching). This is the "minimal tree" that targets the observed position-1 cliff.
2. Implement as a binary tree: draft position 1 with 2 candidates (top-2 logits from MTP drafter), verify both simultaneously.
3. Measure: E[accepted] vs current (target ≥ 4.2), TPS vs 467.14, token-identity = 1.0.
4. If E[accepted] improves by > 0.4, extend to 3-branch tree for follow-up.

**Key question:** Does position-1 branching increase E[accepted] enough to compensate for the wider verify batch? This is answerable in ~30 minutes.

### Citation
- Miao, X. et al. "SpecInfer: Accelerating Generative LLM Serving with Tree-Based Speculative Inference and Verification." ASPLOS 2024.
- "DySpec: Dynamic Token Tree for Higher Acceptance in Speculative Decoding." ICLR 2025. OpenReview.
- "Traversal Verification for Speculative Tree Decoding." NeurIPS 2025. (tree-level lossless verification)
- "Towards Optimal Multi-Draft Speculative Decoding." arXiv:2502.18779, ICLR 2025. (theoretical optimal multi-draft strategy)

### Risk of Collapse
- vLLM's ONEGRAPH CUDA capture may not support dynamic tree branching without a major restructuring of the graph capture logic. This is the highest-risk integration point.
- Wider verify batch (position-1 doubled) increases memory traffic; if the A10G is already DRAM-BW-saturated at M=8, adding M=9 or M=10 may not improve TPS.
- Tree verification requires tracking which branches are "alive" — the bookkeeping overhead in Python/host may eat the GPU gain.

---

## Direction 7: Lossless Speculative Decoding for Heterogeneous Vocabularies / Draft-Free

### Hypothesis
Apply the heterogeneous-vocabulary lossless speculative decoding framework (arXiv:2502.05202) to run a completely different, smaller model as drafter without requiring vocabulary matching — specifically, use a 1B or 500M parameter model with a different tokenizer as the speculative drafter.

### Mechanism + Why This Stack
Standard speculative decoding requires draft and target models to share a vocabulary. The heterogeneous vocabulary paper (2025) proves that lossless verification is possible even when drafter and target use different tokenizers, via a token-mapping verification step. This opens up the space of potential drafters dramatically:
- Can use any sub-1B model (e.g., TinyLlama, Qwen2-0.5B) whose forward pass costs <0.5 ms.
- Current MTP drafter costs ~1.433 ms; a better-matched 500M drafter might achieve similar E[accepted] at 0.3–0.5 ms.
- Reported speedup: 2.8× over autoregressive baseline (on their benchmark).

**Why it could beat 467 on this stack:**
- If a smaller drafter with lower per-step cost achieves E[accepted] ≥ 3.5, the TPS gain from faster drafting could more than compensate.
- The "Lossless Hierarchical Speculative Decoding" (ICLR 2026 Oral) extends this to multi-level hierarchies, allowing a cascade of increasingly accurate drafters.

### Byte-Exact Identity Preservation
The paper proves lossless equivalence: every token emitted matches the target model's exact greedy argmax. The verification pass remains the full target model.

### Cheapest Validation Experiment (~90 min, single A10G)
1. Download a sub-1B model (e.g., `google/gemma-2-2b-it` → too large; try `Qwen/Qwen2.5-0.5B-Instruct`).
2. Measure its single-token forward pass latency on A10G.
3. Estimate E[accepted] with greedy draft by running 10 prompts manually.
4. If forward pass < 0.8 ms AND E[accepted] > 2.5, prototype the full loop.
5. Binary gate: TPS > 467 with token-identity = 1.0.

### Citation
- "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies." arXiv:2502.05202, 2025.
- "Lossless Hierarchical Speculative Decoding." ICLR 2026 (Oral). OpenReview.

### Risk of Collapse
- A 500M model may have much lower E[accepted] against Gemma-4-E4B than the current Gemma MTP drafter (which is trained on Gemma's representations).
- The vocabulary-mapping overhead adds per-token computation; if this isn't free, it partially offsets the smaller drafter's speed advantage.
- Two-model memory footprint may exceed A10G's 24GB if the drafter must stay loaded alongside the full target model.

---

## Direction 8: Kernel Tiling / Register Blocking for Non-GEMM Elementwise — The "Other 79%"

### Hypothesis
Profile the non-GEMM elementwise slice (RMSNorm, GeGLU, RoPE, residual add) at the warp level on A10G to measure whether the bottleneck is DRAM bandwidth, L2 contention, or launch overhead — then apply the appropriate remedy (tiling, register blocking, or kernel batching).

### Mechanism + Why This Stack
The current analysis shows ~79% of verify wall is non-GEMM. But this is a single number — it doesn't distinguish:
1. DRAM-BW-bound: reading/writing tensor data is the limit. Remedy: reduce precision (BF16→FP8 for activations) or fuse ops to reduce round-trips.
2. L2 contention: tensors don't fit in L2 (6.29 MB), causing thrashing. Remedy: tile the computation to fit activations in L2 across the entire layer pipeline.
3. Launch overhead: per-kernel dispatch latency dominates at small batch sizes. Remedy: batch multiple small kernels into one CUDA graph node or persistent kernel.

**This direction is a diagnostic experiment before a fix experiment.** The Nsight Compute profile for verify forward with `--section SpeedOfLight` and `--section MemoryWorkloadAnalysis` will directly answer which bottleneck applies.

**Why it could beat 467:** If launch overhead is the binding constraint (plausible for M=8 × 26 layers = 208 elementwise kernel calls), CUDA graph already captures launches — but within the graph, can we reduce the number of nodes? A graph with 208 nodes vs 26 fused-layer nodes may have measurable CUDAGraph replay overhead.

### Byte-Exact Identity Preservation
This direction is diagnostic — no code changes that affect arithmetic. Any subsequent optimization must pass the greedy-identity gate.

### Cheapest Validation Experiment (~45 min, single A10G)
1. Run `ncu --set full --import-source yes -o verify_profile ./run_benchmark.sh` on a single batch.
2. Extract: SM utilization, DRAM BW utilization, L2 hit rate, kernel launch count for the 26 non-GEMM layers.
3. From the profile, classify each bottleneck.
4. Write a one-page memo: "The non-GEMM 79% is [DRAM-BW / L2 / launch] bound at [X]% utilization."
5. This memo directly determines which of Directions 4, 5, or 6 to prioritize.

**This is the highest-information experiment in this list for its compute cost.**

### Citation
- NVIDIA Nsight Compute documentation: roofline analysis for sm_86 Ampere.
- "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Williams et al., CACM 2009. (roofline methodology applied to GPU kernels)
- "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." Dao, ICLR 2024. arXiv:2307.08691. (register blocking and tiling patterns for Ampere attention)

### Risk of Collapse
- If Nsight Compute reveals that the non-GEMM 79% is dominated by DRAM BW at 90%+ utilization (matching the GEMM situation), there is no available optimization lever short of reducing precision or fusing. This would close Directions 4, 5, and most of 7.
- Profile overhead may not be representative of production CUDA graph execution.

---

## Ranked Shortlist and Student Archetype Assignments

| Rank | Direction | TPS Upside (est.) | Archetype | Confidence |
|------|-----------|-------------------|-----------|------------|
| 1 | Direction 8: Non-GEMM profiling diagnostic | Diagnostic — unlocks 4, 5, 6 | Profiling/measurement | High: this is the MAP experiment |
| 2 | Direction 6: Multi-candidate tree verify (position-1 branching) | +10–50 TPS | Build-specialist | Medium: high upside, vLLM integration hard |
| 3 | Direction 4: FlashInfer JIT-compiled verify attention | +5–15 TPS | Build-specialist / integrator | Medium: addresses known attention autotune instability |
| 4 | Direction 1: Lookahead / Jacobi exact decoding | +5–20 TPS | Numerics/theory | Medium: structurally elegant, acceptance rate uncertain |
| 5 | Direction 5: Fused RMSNorm+GeGLU+residual megakernel | +5–15 TPS | Build-specialist | Medium (depends on profiling result from D8) |
| 6 | Direction 2: N-gram prompt-lookup drafting | +5–15 TPS | Integrator | Low-medium: zero training, easy to prototype, acceptance uncertain |

---

## Research State Update

**Current best explanation for the plateau:** The equivalence frontier at 467.14 TPS is almost entirely constrained by DRAM bandwidth in the verify forward pass (~21% int4 GEMM, ~79% non-GEMM elementwise + dispatch), and the GEMM portion is already at 92% of peak DRAM BW. The non-GEMM 79% is the unexplored frontier. The live +15.86 TPS attention-autotune lead has evaporated twice, suggesting the attention kernel accounts for ≤7–9% of the wall and is not the primary lever. The MTP drafter (~18% of cycle at 1.433 ms) is the second-highest unexplored lever.

**The three most promising attack surfaces, in order:**
1. Non-GEMM elementwise fusion or tiling (attack the 79% with direction knowledge from profiling).
2. Drafter cost reduction (CLaSp layer-skip, n-gram lookup, or better neural drafter).
3. E[accepted] improvement (tree decoding targeting the position-1 cliff, or lookahead).

**Ruled out:** GEMM config tuning, attention autotune (without root-cause profiling), M-value changes, CUDA graph capture of the loop (already ONEGRAPH=1).

**Open uncertainty:** What fraction of the 79% non-GEMM is dispatch overhead vs. DRAM-BW vs. L2 contention? This single question gates the entire optimization roadmap for the next cycle.

**Next discriminating experiment:** Direction 8 (Nsight profile of non-GEMM slice). Costs ~45 minutes, produces a bottleneck map that determines which of the remaining directions can succeed.

**Stop condition for this research cycle:** If the Nsight profile shows non-GEMM at ≥ 85% DRAM BW utilization AND launch overhead < 5% of verify wall, then kernel fusion and tiling cannot materially improve TPS, and the program must pivot to drafter quality (tree decoding, lookahead) or accept the hardware ceiling.

---

## Literature Bibliography (Full)

1. Fu, Y. et al. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." ICML 2024. arXiv:2402.02057. https://github.com/hao-ai-lab/LookaheadDecoding
2. Saxena, A. "Prompt Lookup Decoding." GitHub 2023. https://github.com/apoorvumang/prompt-lookup-decoding
3. Bhatt, J. & Cohen, W. "The N-Grammys: Improving Lossless Speculative Decoding with Batched n-gram Lookups." arXiv:2411.03786, 2024.
4. "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies." arXiv:2502.05202, 2025.
5. Zhao, X. et al. "CLaSp: Efficient Inference with Context-Aware Layer Skipping and Speculation." ACL 2025 (long paper). arXiv:2505.24196.
6. "DEL: Dynamic Exit Layer for Efficient Decoding." COLM 2025.
7. "CAS-Spec: Cascade Adaptive Self-Speculative Decoding." NeurIPS 2025.
8. Ye, Z. et al. "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." arXiv:2501.01005, 2025. https://github.com/flashinfer-ai/flashinfer
9. Miao, X. et al. "SpecInfer: Accelerating Generative LLM Serving with Tree-Based Speculative Inference and Verification." ASPLOS 2024.
10. "DySpec: Dynamic Token Tree for Higher Acceptance in Speculative Decoding." ICLR 2025.
11. "Traversal Verification for Speculative Tree Decoding." NeurIPS 2025.
12. Shyam, P. et al. "Towards Optimal Multi-Draft Speculative Decoding." arXiv:2502.18779, ICLR 2025.
13. "Lossless Hierarchical Speculative Decoding." ICLR 2026 (Oral).
14. "ClusterFusion: Accelerating Grouped Computation for Transformers on the GPU." NeurIPS 2025.
15. "FFN Fusion: Rethinking Sequential FFN Layers in Transformers." NeurIPS 2025 (spotlight).
16. Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." ICLR 2024. arXiv:2307.08691.
17. Cai, T. et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML 2024. arXiv:2401.10774.
18. "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding." arXiv:2404.16710, 2024. (requires training, lower priority)
19. "NorSA: Normalized Sparse Activation for Efficient LLM Inference." ICLR 2026 submission.
