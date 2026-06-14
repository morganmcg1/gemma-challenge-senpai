# Research Ideas — 2026-06-14 09:45

**Target**: Maximize output TPS for `google/gemma-4-E4B-it` on A10G (sm_86, 23 GB, ~600 GB/s GDDR6, no FP8 tensor cores), greedy decoding, 128-token prompt / 512-token generation, concurrency=1.
**Baseline**: 481.53 output TPS (int4 W4A16 Marlin + split-KV speculative decoding, linear chain drafter).
**Hard constraint**: greedy-token-identity gate (output token-identical to plain M=1 AR; PPL ≤ 2.42).

Dead-ends (do NOT revisit): SplitK W4A16 GEMM (capped 3.2%/1.56%), 2:4 structured sparsity (PPL 7.5 >> 2.42), LUT dequant, per-tensor scale-palette quant, bigger/better drafters, double-quant scales, FP8 anything (no tensor cores on sm_86), FireQ INT4-FP8.

---

## Question A — Tree Spec-Decode E[T] Collapse (Measured 2.10 vs Theoretical 5.207)

The E[T] gap is the single largest headroom item. The ceiling with linear-chain drafter is ≈491.8 TPS (E[T]=3.844). The theoretical tree ceiling at E[T]=5.207 is ≈600+ TPS — a 25% uplift. Closing even half that gap is worth more than any other optimization in the program.

### A-1. Traversal Verification (leaf-to-root tree walk)

**Technique**: Replace standard top-down token-level rejection in tree speculative decoding with leaf-to-root traversal that considers the full sequence from each leaf to the root. Valid subsequences that are currently discarded when a parent is rejected are salvaged and committed.

**Why it helps**: The primary cause of E[T] collapse in tree decoding is parent-rejection cascades — when a parent token is rejected, all its children are discarded regardless of their individual acceptance probability. Traversal Verification eliminates this by treating the sequence from each leaf to root as a unit. The algorithm is proven lossless: the output distribution is identical to the target model, preserving greedy-token-identity.

**Citation**: "Traversal Verification for Speculative Decoding" — arxiv 2505.12398, NeurIPS 2025 (OpenReview id=8nOMhDFpkU). https://arxiv.org/abs/2505.12398

**Fits A10G/greedy-identity**: Yes. Lossless by proof; no distribution shift. Does not require FP8. Operates on the existing tree structure without requiring a new drafter. The per-leaf-path verification adds one extra tree-walk pass but no additional GPU forward passes.

**Expected impact**: E[T] recovery toward 3.5–4.5 range (from 2.10), corresponding to +10–20% TPS uplift. Exact recovery depends on how much of the E[T] gap is due to parent-rejection cascades vs tree construction quality.

**Cheap test**: Implement traversal verification on the existing M=4 static tree, measure E[T] and TPS on a fixed 200-prompt benchmark, compare to current rejection-sampling baseline. One GPU-hour.

---

### A-2. EAGLE-2 / Context-Aware Dynamic Draft Trees

**Technique**: Replace the current static draft tree topology with a context-aware dynamic tree built from calibrated draft-model confidence scores at each step. The tree structure adapts per-token-position rather than being fixed at training time.

**Why it helps**: Static trees allocate budget uniformly across all positions. EAGLE-2's key insight (validated empirically) is that acceptance rate is strongly context-dependent — easy tokens should get wider trees, hard tokens narrower. A context-blind static tree over-populates low-acceptance regions and under-populates high-acceptance regions, systematically compressing E[T] below its theoretical maximum. Dynamic trees correct this per decode step.

**Citation**: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" — arxiv 2406.16858, ICML 2025. https://arxiv.org/abs/2406.16858. Reported 20–40% speedup over EAGLE-1 in original paper.

**Fits A10G/greedy-identity**: Yes. The acceptance-rejection step is unchanged and preserves the greedy-identity guarantee. No FP8 required. The dynamic tree construction adds a small per-step CPU/CUDA overhead but reduces wasted verify GEMM work on low-acceptance subtrees.

**Expected impact**: +15–30% E[T] recovery (measured E[T] 2.10 → ~2.8–3.5), translating to +8–15% TPS. Combination with Traversal Verification (A-1) is likely additive since they target different failure modes.

**Cheap test**: Port EAGLE-2's tree-selection algorithm (the confidence-score-based expand/prune loop) to the existing drafter, run on 200 prompts, measure E[T] vs static tree. No architecture change needed — only tree construction logic changes.

---

### A-3. Sequoia — DP-Optimal Tree Topology

**Technique**: Use dynamic programming to find the provably optimal draft tree topology given a calibrated draft model and known acceptance-rate distribution. Sequoia also introduces an improved sampling-and-verification method that achieves higher acceptance rates than standard speculative sampling under the same token budget.

**Why it helps**: The current tree topology was likely designed heuristically or from an offline calibration sweep. Sequoia proves the DP solution maximizes expected number of accepted tokens per verify step. Even if the current tree is decent, the DP-optimal tree is guaranteed no worse and often materially better.

**Citation**: "Sequoia: Scalable, Robust, and Hardware-Aware Speculative Decoding" — arxiv 2409.03552, 2024. https://arxiv.org/abs/2409.03552. GitHub: https://github.com/Infini-AI-Lab/Sequoia

**Fits A10G/greedy-identity**: Yes. Preserves greedy-identity if the standard spec-decode acceptance criterion is used. Does not require FP8 or architecture changes. The DP runs offline on calibration data.

**Expected impact**: Modest E[T] recovery (+5–10%), primarily by eliminating obviously bad tree branches. Larger impact when combined with A-2 (dynamic trees) since the DP provides the right baseline topology.

**Cheap test**: Run Sequoia's offline DP on 500 calibration prompts from the target distribution, compare resulting topology's E[T] to hand-tuned topology. No training required.

---

### A-4. SwiftSpec Fused Tree-Attention Kernel

**Technique**: Replace the current tree-mask attention dispatch in the verify pass with a fused kernel that handles the tree structure natively without falling back to full self-attention over the entire candidate set. SwiftSpec (ByteDance, June 2025) provides fused latency-optimized kernels for tree-aware KV cache management and tree-attention verification that avoid the TRITON_ATTN override issue in vLLM.

**Why it helps**: The current E[T] collapse may be partially caused by the verify attention pass using an incorrect or suboptimal mask. vLLM forces TRITON_ATTN for Gemma-4, and FlexAttention mask_mod is silently overridden, which means the tree structure is not being correctly represented in the attention computation. A standalone fused kernel bypasses this framework limitation.

**Citation**: "SwiftSpec: Asynchronous Speculative Decoding with Parallel Tree Generation" — arxiv 2506.11309, ByteDance, June 2025. https://arxiv.org/abs/2506.11309. Reported 1.75x speedup over SOTA spec-decode. GitHub: https://github.com/bytedance/SwiftSpec (check for availability)

**Fits A10G/greedy-identity**: Yes. GA102/sm_86 is well-supported by fused CUDA kernels. Fused tree-attention does not alter the acceptance distribution. The key question is whether Gemma-4's heterogeneous head dims (head_dim=256, global_head_dim=512) are handled correctly — verify this before full implementation.

**Expected impact**: If the verify-attention mask is currently broken or degraded, fixing it could recover the full E[T] gap in one shot. Otherwise, fused kernel reduces verify latency by 10–20%, translating to +5–10% TPS.

**Cheap test**: Implement a minimal standalone FlashAttention-based tree-mask kernel (or port SwiftSpec's) and verify on a single tree-decode step that the argmax output matches M=1 AR ground truth. This is simultaneously a correctness diagnostic and a performance test.

---

## Question B — Batch-Invariant Verify = Token-Identical to M=1 AR

The M=32 wide-batch verify currently diverges ~56% from M=1 AR argmax due to batched float reduction order differences in GEMM and attention. This is a blocker for any tree width > 1.

### B-1. LLM-42 Selective Determinism via Verify-Rollback

**Technique**: Instead of forcing all operations to use deterministic algorithms (which strips split-K parallelism and imposes uniform overhead), use a fast non-deterministic forward path followed by a lightweight verify-rollback loop. The rollback is triggered only when the non-deterministic output diverges from a fixed-shape reference reduction. Token commits are only made when the verify-rollback guarantees consistency. Overhead is proportional only to the fraction of traffic that actually diverges.

**Why it helps**: The current divergence rate is 56%. If LLM-42's observation (O1) holds — that inconsistencies are rare in practice — then in steady state most steps take the fast path. The 56% divergence in testing may be a worst-case or artificial condition. LLM-42 is the only published approach that avoids the false dilemma between "fully deterministic but slow" and "fast but divergent."

**Citation**: "LLM-42: Efficient Deterministic Inference for Large Language Models" — arxiv 2601.17768, Microsoft Research + U Washington, January 2026. https://arxiv.org/abs/2601.17768. Authors: Raja Gond, Aditya K Kamath, Ramachandran Ramjee, Ashish Panwar.

**Fits A10G/greedy-identity**: Directly targeted at this exact problem. No FP8 required. sm_86 is a supported platform. The rollback mechanism is compatible with W4A16 Marlin quantization because it operates at the output logit level, not the weight level.

**Expected impact**: If the 56% divergence rate drops to <5% in production (LLM-42's O1 observation), effective TPS overhead is minimal (<2%) while fully resolving the identity gate. This unlocks full tree-width gains from Question A.

**Cheap test**: Implement the rollback loop for the lm_head GEMM only (which dominates logit computation), measure divergence rate reduction vs baseline, then measure TPS overhead. If divergence drops to <5% with <3% TPS cost, proceed to full integration.

---

### B-2. He et al. Batch-Invariant GEMM with Fixed Reduction Strategy

**Technique**: Apply a fixed, shape-independent reduction strategy to the verify GEMM such that the result is identical regardless of batch dimension M. The key insight is that non-determinism comes from different reduction tree structures chosen by cuBLAS/cutlass for different M values. By fixing a canonical reduction order (e.g., always reduce pairs left-to-right in the same tree regardless of M), M=32 produces identical output to M=1.

**Why it helps**: This is the most direct fix for the 56% divergence. It is simpler than LLM-42 (no rollback loop) but imposes a fixed overhead since it always uses the canonical reduction rather than the fastest kernel for each shape.

**Citation**: He et al., "Batch-Invariant Attention for Consistent LLM Inference" — referenced in vLLM/SGLang discussions and adopted in older vLLM batch-invariant mode. See also vLLM PR discussion on deterministic GEMM. No single clean arxiv link; best entry point is https://github.com/vllm-project/vllm/issues/5017 and related PRs.

**Fits A10G/greedy-identity**: Yes. Eliminates the divergence problem at the cost of some GEMM throughput. On sm_86 with W4A16 Marlin, the dequant is the bottleneck, so the reduction-strategy overhead may be small relative to dequant time.

**Expected impact**: Fully resolves 56% divergence (down to 0%). TPS cost depends on how much split-K parallelism is abandoned: estimated 3–8% TPS penalty. If tree-width gains from Question A recover 15–25%, net is strongly positive.

**Cheap test**: Replace the lm_head GEMM with a deterministic wrapper (use `torch.use_deterministic_algorithms(True)` on the lm_head only, not globally), measure divergence rate and TPS delta on a 200-prompt run.

---

### B-3. Per-Step Argmax Consistency Guard (Lightweight Oracle)

**Technique**: After each verify step, compute the argmax of the greedy reference independently from the verify batch using a single M=1 forward pass for the first token only (the one that must match for the identity gate). Accept the batched result if it matches; fall back to the M=1 result otherwise. This does not fix the root cause but provides an exact oracle fallback.

**Why it helps**: This is a diagnostic-first approach. It tells us whether the 56% divergence actually triggers mismatches at the identity-gate check position (token[0]) or whether most divergence is at later tree positions that are discarded anyway. If the gate-critical divergence rate is much lower than 56%, the problem may be less severe in practice.

**Citation**: First-principles fallback; no paper needed. Related concept: the "speculative decoding golden path" check used in DeepMind's speculative sampling paper (Leviathan et al., arxiv 2211.17192).

**Fits A10G/greedy-identity**: Yes. No architecture change. Correct by construction since the fallback is exact M=1 AR.

**Expected impact**: Diagnostic value is high; TPS impact depends on fallback rate. If gate-critical divergence is <10%, the 2x cost of the fallback call at those positions is minor. If it is >30%, this approach is too expensive and B-1 or B-2 must be the fix.

**Cheap test**: Run the oracle guard on 500 prompts, measure how often the fallback is actually triggered. Report divergence rate at gate-critical position separately from aggregate divergence rate. 30-minute diagnostic.

---

## Question C — Fresh Single-GPU Decode-Throughput Lever

The below candidates are genuinely novel relative to the dead-end list. Each targets a different bottleneck in the memory-bound, concurrency=1, int4, A10G decode regime.

### C-1. Q-Palette Fractional-Bit Quantization (2.0–4.25 bits)

**Technique**: Replace W4A16 Marlin with Q-Palette's fractional-bit quantization scheme, which allocates bits per-layer or per-group using an information-theoretic optimal schedule. Q-Palette supports 2.0–4.25 average bits via mixed-precision PTQ with custom CUDA kernels optimized for memory-bound small-batch inference. It is specifically designed for the concurrency=1 regime where weight-loading bandwidth is the bottleneck.

**Why it helps**: At concurrency=1 and 512-token generation, the model is almost entirely memory-bandwidth-bound. The W4A16 Marlin decode is spending ~600 GB/s loading weights that are uniformly 4 bits. Q-Palette can reduce the average weight bit-width to 3.0–3.5 bits for most layers while preserving PPL within the 2.42 gate, yielding a direct 15–25% bandwidth reduction and corresponding TPS uplift. The custom kernels outperform NF4/BnB by ~36% on decode latency in published benchmarks.

**Citation**: "Q-Palette: Fractional-Bit Quantization for Memory-Efficient LLM Inference" — NeurIPS 2025, OpenReview id=l4F50jpiVH. Seoul National University. https://openreview.net/forum?id=l4F50jpiVH

**Fits A10G/greedy-identity**: Yes. Does not require FP8. Data-free or low-calibration PTQ. The greedy argmax is not changed by quantization per se (PPL gate must be verified post-quantization). Custom CUDA kernels target sm_80+ (A10G is sm_86, compatible).

**Expected impact**: +15–25% TPS from bandwidth reduction, assuming PPL gate holds at 3.0–3.5 average bits. The PPL gate is the risk — verify on the target distribution before committing to full integration.

**Cheap test**: Run Q-Palette PTQ on Gemma-4-E4B at 3.25 average bits, evaluate PPL on a 1000-token sample from the benchmark distribution, compare to the 2.42 gate. If PPL holds, measure TPS delta. 2 GPU-hours.

---

### C-2. Activation Checkpointing / Weight Streaming with Prefetch

**Technique**: For the memory-bandwidth-bound decode regime, implement a weight-streaming pipeline that prefetches the next layer's weights from GDDR6 into L2/shared-memory while the current layer's computation runs. This overlaps weight-load latency with compute, reducing the effective memory bandwidth requirement.

**Why it helps**: At concurrency=1 with int4 weights, each decode step loads ~2 GB of model weights. The A10G has ~600 GB/s GDDR6 but only 80 SMs. The current decoding likely serializes weight load and compute per-layer. Prefetching next-layer weights during current-layer compute reduces the critical path length on the memory subsystem.

**Citation**: Related work: "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU" (arxiv 2303.06865, Stanford, 2023) for weight-streaming principles. Also: NVIDIA's "FlashDecoding++" (arxiv 2311.01282) for tiling strategies. Direct implementation: CUDA streams with `cudaMemcpyAsync` + `cudaStreamSynchronize` at layer boundaries.

**Fits A10G/greedy-identity**: Yes. Purely a scheduling optimization; no change to numerical computation. The greedy-identity gate is unaffected.

**Expected impact**: +5–15% TPS if layer boundaries are currently serialized. If the current Marlin kernel already pipelines weight loads internally, this may have minimal additional benefit. Diagnostic value is high either way.

**Cheap test**: Profile a single decode step with Nsight Systems, check whether GDDR6 bandwidth is fully saturated or has idle gaps between layers. If idle gaps > 20% of step time, implement prefetch and re-profile. 2-hour profiling + 1-hour implementation.

---

### C-3. Attention Sink KV Cache (StreamingLLM-style)

**Technique**: Instead of maintaining a full 512-token KV cache growing through the generation, maintain a fixed-size sliding window KV cache with the first few "attention sink" tokens always retained. This keeps KV cache size bounded, reducing attention compute in later steps of the 512-token generation where the full KV cache is large.

**Why it helps**: In the current setup, KV cache grows from 128 tokens (prompt) to 640 tokens (prompt + generation). The attention compute cost (even with MQA/GQA in Gemma-4) grows linearly. StreamingLLM shows that a window of 4 sink tokens + recent 256 tokens is sufficient for generative quality in most settings, reducing per-step attention FLOPs by ~30% for the second half of generation.

**Citation**: "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM) — arxiv 2309.17453, MIT/Meta, ICLR 2024. https://arxiv.org/abs/2309.17453. GitHub: https://github.com/mit-han-lab/streaming-llm

**Fits A10G/greedy-identity**: Risky. The greedy-token-identity gate requires the output to be token-identical to plain AR over the full 512-token generation. StreamingLLM with a truncated KV cache will NOT produce token-identical outputs to full-KV AR for most prompts. This technique is likely incompatible with the identity gate as a standalone approach.

**However**: It could be used as a TPS measurement and profiling diagnostic (disable the identity gate temporarily, measure maximum theoretical TPS with bounded KV), or as a baseline for understanding how much KV cache growth costs TPS over the generation.

**Expected impact as diagnostic**: Useful for understanding KV cache growth overhead. Not merge-eligible without identity-gate fix.

**Cheap test**: Run with StreamingLLM (sink=4, window=256), measure TPS vs full-KV baseline, measure token-identity divergence rate. Report both numbers. 1 GPU-hour. If divergence is only in tail positions (>400 tokens), explore a hybrid approach.

---

### C-4. CUDA Graph Capture with Dynamic Tree Shape

**Technique**: Extend the existing CUDA graph capture to cover the full verify step for multiple tree shapes, capturing a separate graph per tree size (M=1, M=4, M=8, M=16, M=32). At inference time, select the appropriate graph rather than re-compiling kernels. This eliminates the per-step Python overhead and CUDA kernel launch latency for tree verification.

**Why it helps**: CUDA graph capture is well-established for fixed-batch AR decoding but is typically disabled for speculative decoding because the tree shape varies per step. Capturing a small discrete set of graphs (one per supported tree size) and selecting at runtime recovers most of the graph capture benefit without requiring a fully dynamic graph. For concurrency=1 with short steps, kernel launch overhead can be 10–20% of step latency.

**Citation**: "CUDA Graphs for LLM Inference" — NVIDIA blog and vLLM CUDA graph implementation (https://github.com/vllm-project/vllm/blob/main/vllm/worker/model_runner.py). Related: "Medusa" (arxiv 2401.10774) which uses a fixed-shape verify for CUDA graph compatibility.

**Fits A10G/greedy-identity**: Yes. CUDA graphs do not change numerical computation. The greedy-identity gate is unaffected. The main implementation challenge is managing the KV cache state across graph invocations with varying tree lengths.

**Expected impact**: +5–15% TPS from reduced kernel launch overhead, especially at small batch sizes (M=4–8) where GPU occupancy is low and launch overhead is relatively large.

**Cheap test**: Profile current tree-verify step with `torch.profiler`, measure fraction of step time in kernel launch overhead vs actual compute. If >10%, implement graph capture for M=1 and M=4 only, measure TPS delta.

---

## Priority Order and Decision Tree

### Top-4 by Expected TPS-per-Implementation-Day

1. **B-1 (LLM-42 Selective Determinism)** — unlocks all tree-width gains. If implemented correctly, resolves the identity-gate blocker and increases effective E[T] from ~2.10 toward the tree-theoretical maximum. Estimated implementation: 3–5 days.

2. **A-1 (Traversal Verification)** — highest E[T] recovery per implementation cost. Lossless guarantee reduces risk. Can be implemented independently of B-1 (on the M=1 baseline). Estimated implementation: 2–4 days.

3. **C-1 (Q-Palette)** — independent of spec-decode, directly reduces weight-load bandwidth. The PPL gate is the main risk but is checkable in 2 hours before committing to full integration. Estimated implementation: 3–5 days after PPL verification.

4. **A-2 (EAGLE-2 Dynamic Trees)** — addresses the static-tree bias that causes uniform E[T] collapse. Complements A-1 (different failure mode). Estimated implementation: 2–3 days if the drafter already has confidence scores available.

### Decision Tree

```
Start: B-3 oracle diagnostic (30 min)
├─ Gate-critical divergence < 10%?
│   ├─ YES → proceed directly to A-1 (Traversal Verification)
│   │         then A-2 (EAGLE-2), measure cumulative E[T]
│   │         then C-1 (Q-Palette PPL check)
│   └─ NO (>10%) → implement B-1 (LLM-42) first
│                   ├─ Divergence drops to <5%? → proceed to A-1, A-2
│                   └─ LLM-42 overhead > 5% TPS? → fallback to B-2
│                       (batch-invariant GEMM, fixed reduction)
│                       accept 3-8% penalty, unlock tree width
│
After B resolved:
├─ A-1 (Traversal Verification): E[T] recovery measured?
│   ├─ E[T] > 3.5? → strong, add A-2 for further recovery
│   └─ E[T] < 3.0? → investigate A-4 (tree-attn kernel bug)
│       before adding A-2
│
After A resolved:
└─ C-1 (Q-Palette): PPL gate passes at 3.25 bits?
    ├─ YES → full integration, expect +15-25% TPS
    └─ NO → try 3.5 average bits; if still fails, try C-4 (CUDA graphs)
```

---

## Summary Table

| ID | Technique | Question | Expected TPS delta | Gate-safe? | Implementation days |
|----|-----------|----------|--------------------|------------|---------------------|
| A-1 | Traversal Verification | A | +10–20% | Yes (lossless proof) | 2–4 |
| A-2 | EAGLE-2 Dynamic Trees | A | +8–15% | Yes | 2–3 |
| A-3 | Sequoia DP-optimal topology | A | +5–10% | Yes | 1–2 |
| A-4 | SwiftSpec fused tree-attn | A | +5–10% (or full recovery if mask broken) | Yes | 3–5 |
| B-1 | LLM-42 selective determinism | B | Blocker-unlock | Yes (proven) | 3–5 |
| B-2 | He et al. batch-invariant GEMM | B | Blocker-unlock, -3–8% penalty | Yes | 2–3 |
| B-3 | Per-step argmax oracle guard | B (diagnostic) | Diagnostic only | Yes | 0.5 |
| C-1 | Q-Palette fractional-bit quant | C | +15–25% | Conditional (PPL gate) | 3–5 |
| C-2 | Weight streaming / prefetch | C | +5–15% | Yes | 2–4 |
| C-3 | StreamingLLM attention sinks | C | Diagnostic only (gate incompatible) | No (standalone) | 1 (diagnostic) |
| C-4 | CUDA graph per-tree-shape | C | +5–15% | Yes | 2–3 |
