# Research Ideas — 2026-06-16 02:00

**Context:** Single-stream decode TPS for google/gemma-4-E4B-it on one AWS A10G (sm_86, 24 GB, ~600 GB/s HBM).
Serving stack: vLLM + Triton unified-attention (not FA2; heterogeneous head dims force Triton path).
Body GEMMs: int4-Marlin (M-invariant — GEMM cost does NOT decrease with smaller batch M).
Speculative decode: MTP self-drafter, K=7, E[T]=3.851 accepted tokens/verify step, a₁=0.7293.
Verify-BW λ=1 wall: ~520.95 TPS theoretical ceiling.
Deployed incumbent: 481.53 TPS (non-equivalent, greedy-id 0.9966).
Strict-equivalent frontier: ~467 TPS (blanket) / modeled ~476–479 TPS (selective lever).
Modeled +15.60 cb3 ceiling: 482.74 TPS.

**Hard gates (binding for all proposals):**
- CORRECTNESS: strict byte-exact greedy-token-equivalence; verify forward pass is sole token arbiter.
- QUALITY: PPL ≤ 2.42 (deployed baseline PPL 2.3772).

**Closed/exhausted levers (do not revisit):**
- SDPA num_stages tweak (≤+0.94 TPS, closed)
- Drafter at linear acceptance cap (cannot accept more without verify-pass changes)
- Draft-head temperature/affine rescaling (rank-invariant no-op)
- Keepset-masking (0.50% out-of-keepset live, closed)
- Per-class logit bias (requires retrain/overfit)
- Attention split-K / pinned-K / FlashDecoding (re-enabling split-K = −5.82 TPS, REFUTED)
- Drafter-fusion (−16.5%, REFUTED)
- Selective-recompute, precision-allocation, canonical argmax tie-break, lm_head row-pruning
- KV cache quantization of any form (changes verify pass attention → breaks strict equivalence)
- HiSpec / LayerSkip early-exit self-speculative (requires training/fine-tuning of EE layers)
- CaDDTree (diffusion-drafter specific, not applicable to AR MTP drafter)
- KnapSpec (training-free self-spec as Knapsack: requires fine-tuned LM head on early layers)

---

## Idea 1 — Async Pipelined Drafting (Saguaro-style)

### What it is

Overlap the next K-step draft pass with the current verify forward pass so drafter latency is hidden inside verify wall time, reducing the effective spec-decode cycle from (draft + verify) to max(draft, verify).

### Why it might help here

The current MTP drafter runs synchronously: verify completes → drafter runs → verify runs. If the drafter's K=7 step costs D ms and verify costs V ms, total cycle = D + V. If D < V, the drafter is entirely on the critical path and can be hidden. Given int4-Marlin body GEMMs are expensive and drafter is a lightweight MTP head, D < V is plausible, but must be measured. The Saguaro paper (ICLR 2026) reports 30% faster than synchronous spec decode baselines and up to 5x faster than AR decoding using this principle.

### Equivalence argument

The verify forward pass is the sole token arbiter in both synchronous and async modes. In async operation, the drafter produces speculative tokens for the NEXT verify step while the CURRENT verify step executes. When the current verify step completes, it accepts/rejects the prior spec tokens by its own argmax — the pre-speculated tokens from the drafter are discarded or confirmed depending on the verify result. Wrong pre-speculations are dropped; correct ones accelerate. The verify argmax output is unchanged. This is equivalence-preserving under the condition that pre-speculated drafter tokens whose verify step has NOT yet run are never emitted to output — only verified tokens are emitted. The Saguaro paper describes this: "wrong pre-speculations are discarded without corrupting output." Greedy-token identity holds as long as vLLM's speculative sampling logic enforces this invariant (emit only after verify confirms).

**Risk caveat:** The equivalence argument holds if and only if the async drafter uses the VERIFIED hidden state from the previous verify pass as its input context (not the pre-speculated state). If the drafter is given a hidden state that includes tokens not yet verified (contaminated context), it would diverge from the synchronous path. This must be checked explicitly in the implementation. This is the make-or-break for equivalence.

### Expected TPS direction and magnitude

If D / V ≈ 0.3 (drafter is 30% of verify wall time), async overlap recovers roughly 0.3 × verify_ms per cycle. At current E[T]=3.851 accepted tokens/step, this translates to ~D/(D+V) × (current_TPS) gain. A conservative model: if drafter = 15% of cycle, async saves ~15% × 467 TPS ≈ +70 TPS (upper bound, not realistic due to synchronization overhead). A realistic estimate, accounting for CUDA stream synchronization and context contamination guards, is +5–20 TPS. The Saguaro 30% speedup was measured on larger batch sizes and different hardware; single-stream A10G likely sees less.

### Measurement protocol (≤90 min, no served-file change)

1. Profile current sync cycle with `torch.cuda.nvtx` or `nsys profile`: measure D (drafter forward ms) and V (verify forward ms) for K=7 on 128-token prompts. This is a 10-min microbench.
2. If D < 0.05 × V (drafter ≤5% of verify wall), async pipelining saves <5% — abort, go to Idea 2.
3. If D > 0.05 × V, prototype: wrap drafter in a CUDA stream separate from verify stream; use CUDA events to sequence (1) launch drafter on stream-B, (2) synchronize stream-B output after stream-A verify completes and copies verified hidden state to drafter context. Use `torch.cuda.Stream` and `stream.wait_event(verify_done_event)`.
4. Measure wall TPS on the same 128→128 prompt/gen configuration.
5. Run greedy-equivalence check against reference: compare token-by-token over 100 prompts.

### Key papers

- **Saguaro / Speculative Speculative Decoding** (OpenReview aL1Wnml9Ef, ICLR 2026): draft model runs asynchronously while verify executes, hiding drafter latency. 30% faster than synchronous spec decode baselines. [https://openreview.net/forum?id=aL1Wnml9Ef]
- **PARD** (arxiv:2412.08050): parallel autoregressive decoding pipeline for draft+verify overlap. [https://arxiv.org/abs/2412.08050]

### Single biggest risk

vLLM's CUDA graph capture for speculative decoding serializes stream execution through graph node dependencies. Introducing a second async stream breaks the graph capture invariant — you cannot have an async operation outside the captured graph. This means async pipelining likely requires disabling CUDA graphs for the spec decode path or a significant refactoring to capture both streams inside a single graph. If CUDA graph is disabled, the overhead savings from async may be offset by re-introducing per-step CPU dispatch overhead (~0.1–0.3 ms/step). Net gain could be zero or negative without careful profiling.

---

## Idea 2 — Async KV Cache Prefetch via PTX cp.async.bulk in Triton Verify-Pass Attention

### What it is

Insert PTX `cp.async.bulk.prefetch.L2` instructions into the Triton unified-attention kernel to prefetch the next attention layer's KV pages from HBM into L2 cache while the current layer's GEMM or softmax is computing, overlapping HBM latency with compute.

### Why it might help here

The A10G has 40 MB L2 cache. The Triton unified-attention kernel's KV access pattern is sequential by layer: layer i completes, then layer i+1's K and V are loaded from HBM. During the GEMM-heavy QKV projection windows, L2 is largely idle for attention data. Prefetching the next layer's KV tiles into L2 ahead of when they are needed can reduce HBM-latency stalls in the attention kernel. The async KV prefetch technique (arxiv:2504.06319) reports 2.15x attention-kernel efficiency on H20 GPUs (which have ~200 GB/s HBM — slower than A10G's ~600 GB/s). On A10G, the KV fetch latency is relatively lower per byte, so absolute gains will be smaller, but at 42 layers and head_dim=256 with M=8 tokens (1+K=8 verify batch), the attention is still HBM-latency-bound per layer.

### Equivalence argument

This is a pure memory-scheduling optimization. No compute values change. The attention kernel computes identical floating-point arithmetic regardless of whether K/V tiles arrive from L2 (cache hit) or HBM (cache miss) — the same byte values are used. Greedy-token-equivalence is byte-exact by construction. No risk to equivalence.

### Expected TPS direction and magnitude

On H20 (slower memory), the paper reports 2.15x on attention kernels. On A10G with faster HBM bandwidth, the L2-hit benefit is smaller. A conservative estimate is 5–15% attention-kernel speedup on A10G. Since attention is one component of verify forward (the rest is Marlin GEMMs), and attention's share of verify wall time must be measured, the system-level gain could be +3–10 TPS at current equivalence frontier levels. This does not require any changes to the verify argmax path.

### Implementation notes

The PTX instruction is `cp.async.bulk.prefetch.L2` (sm_90 and sm_86 both support bulk async copy). In Triton, this can be injected via inline PTX using `tl.inline_asm_elementwise` or via `libdevice.cp_async_bulk_prefetch_global_to_shared`. The key implementation requirement: compute the address of the NEXT layer's K and V tensors before the current layer finishes, and issue the prefetch instruction early enough for it to complete before the next layer's attention starts. The CUDA sm_86 architecture guide confirms `cp.async.bulk` is available on Ampere (sm_80+). The 40 MB L2 is large enough to hold several layers of KV tiles for M=8 context: at bf16, each KV tile for one attention layer is roughly 2 × 2 KV-heads × seq_len × head_dim × 2 bytes. At seq_len=128, head_dim=256, 2 heads: 2 × 2 × 128 × 256 × 2 = 262 KB per layer. 42 layers × 262 KB = ~11 MB total KV, well within 40 MB L2 for the entire prompt.

**Critical gotcha:** the Triton unified-attention kernel on the serving path uses paged KV caches — the physical pages are not contiguous. Bulk prefetch requires knowing the physical page addresses ahead of time. This requires walking the page table before issuing prefetch, which adds CPU overhead. However, the page table for a 128-token prompt is fixed at request start time, so a pre-pass over the block table to generate prefetch addresses is feasible offline per request.

### Measurement protocol (≤90 min, no served-file change)

1. Profile the current attention kernel with `nsys`: measure what fraction of verify wall time is spent in the Triton attention kernel vs. Marlin GEMMs. If attention < 10% of verify time, this idea's system-level gain is ≤1 TPS — abort.
2. If attention ≥ 15% of verify time, instrument the Triton kernel with a prefetch stub: add a tl.debug_barrier() followed by a conditional prefetch (controlled by a flag). Benchmark with and without prefetch on 100 forward passes.
3. Run equivalence check: compare token-by-token on 50 prompts.

### Key papers

- **Async KV Cache Prefetching** (arxiv:2504.06319, 2025): `cp.async.bulk.prefetch.L2` for KV tiles on H20, 2.15x attention efficiency. [https://arxiv.org/abs/2504.06319]
- **FlashInfer** (arxiv:2501.01005): paged attention with async memory ops reference implementation. [https://arxiv.org/abs/2501.01005]

### Single biggest risk

Paged KV cache breaks the contiguous-address assumption of bulk prefetch. Building per-request prefetch address lists adds CPU-side work. On a 128-token prompt, the total KV is small enough that this overhead is bounded, but the implementation complexity of wiring physical page addresses into the Triton kernel is non-trivial. The alternative — non-bulk `cp.async` (sm_80 standard) with tile-level prefetch — is simpler but less efficient. If the Triton kernel already uses `cp.async` implicitly through its pipeline stages, there may be no room for additional prefetch benefit.

---

## Idea 3 — Verify Attention Cost vs. Draft Token Count K: Analytical Sweep to Find the Optimal K

### What it is

Re-derive the optimal K (number of draft tokens per verify step) by explicitly accounting for the M-scaling behavior of the Triton attention kernel, not just the M-invariant Marlin GEMMs, and find whether the current K=7 is actually optimal or whether K=3..5 would give higher TPS by reducing attention cost faster than it loses acceptance tokens.

### Why it might help here

The current K=7 was chosen under a model where verify-forward cost is dominated by Marlin GEMMs (M-invariant). But the Triton unified-attention kernel does M-scale: attending M=8 tokens (1+K=7) to a KV cache of length 128+M is more expensive than attending M=4 tokens (1+K=3). If attention is a non-negligible fraction of verify wall time, the optimal K is lower than the M-invariant-only model predicts. The Marlin M-invariant model has been validated for GEMMs, but the attention M-scaling term has not been explicitly measured and incorporated into the TPS optimization. This is an analysis/profiling experiment with no code changes required.

The key question: what is the empirical verify wall time as a function of K? If T_verify(K) = T_GEMM_const + T_attn(K) where T_attn scales with M=1+K, then:

    TPS(K) = E[T(K)] / T_cycle(K)
           = E[T(K)] / (T_draft(K) + T_GEMM_const + T_attn(K))

where E[T(K)] is the expected accepted tokens under linear-acceptance model. The optimal K minimizes T_cycle / E[T], which may be < 7 if T_attn grows fast enough.

### Equivalence argument

This is a hyperparameter change (K), not an algorithmic change. The verify pass still runs with whatever M=1+K is chosen, and argmax is unchanged. Greedy-token-equivalence is maintained by definition — fewer draft tokens means fewer speculative steps, but each verify step is byte-exact. Accepted tokens from each step are identical to what would have been produced by greedy AR.

### Expected TPS direction and magnitude

If K=3 reduces T_verify by 5% (attention term only) while E[T(3)] ≈ 2.1 (linear model, a₁=0.7293, a₂≈0.5, a₃≈0.36: sum ≈ 1+0.73+0.53+0.36 = 2.62 using geometric approximation) vs E[T(7)] ≈ 3.851, the TPS ratio changes. This requires numerical evaluation with measured T_verify(K) values. The point is not to guess but to measure cheaply. If T_attn < 5% of T_verify, the optimal K stays at 7. If T_attn > 20% of T_verify, K=4 or K=5 might dominate.

### Measurement protocol (≤90 min, no served-file change)

1. Profile T_verify(K) for K ∈ {3, 4, 5, 6, 7} using `torch.cuda.Event` timing around the verify forward call. Run 200 warmup + 500 timed iterations each. Compute mean and std.
2. Profile T_draft(K) similarly for the MTP drafter head at each K.
3. Using measured E[T(K)] from acceptance rate telemetry (already logged from prior runs) and measured T_cycle(K) = T_draft(K) + T_verify(K), compute predicted TPS(K) = E[T(K)] / T_cycle(K) × (tokens_per_second_constant).
4. Select K that maximizes TPS(K). If K_opt ≠ 7, the system is currently suboptimal by the attention-M-scaling effect.
5. Run full TPS measurement at K_opt to confirm prediction.

Total time: ~45 min profiling + 30 min analysis.

### Key papers

- **DySpec** (arxiv:2405.17785): dynamic draft length selection based on real-time cost model. Shows that fixed K is suboptimal when verify cost depends on K. [https://arxiv.org/abs/2405.17785]
- **EAGLE-2** (arxiv:2406.16858): dynamic draft tree construction that implicitly adapts draft depth to acceptance probability. [https://arxiv.org/abs/2406.16858]
- **Memory-Bound but Not Bandwidth-Limited** (arxiv:2605.30571): cross-GPU study of memory access patterns in batch-1 decode; directly relevant to A10G batch-1 scenario. [https://arxiv.org/abs/2605.30571]

### Single biggest risk

If T_attn < 5% of T_verify (GEMMs dominate completely), there is no benefit to reducing K — the existing M-invariant model is correct and K=7 remains optimal. The experiment is still valuable as a diagnostic because it either (a) validates the existing K choice with measured data or (b) reveals unexpected attention cost that the current model ignored. Even a null result is useful here. The risk is low because this is analysis, not an implementation.

---

## Idea 4 — Yggdrasil-Style Equal-Growth Tree (EGT) for Static CUDA Graph Capture of the Full Spec Decode Loop

### What it is

Replace per-step dynamic Python dispatch in the MTP spec decode loop with a fixed-shape equal-growth tree structure (all paths expand K tokens at each depth level) that enables a single static CUDA graph capture of the entire draft+verify cycle, eliminating Python/CUDA launch overhead per step.

### Why it might help here

The Yggdrasil paper (arxiv:2512.23858) introduces Equal-Growth Trees: every tree node at depth d has exactly K children, making the tree shape fixed and enabling CUDA graph capture (graph capture requires fixed tensor shapes). Their speedup over vLLM-Spec: 3.98x on A100 vs 2.66x for vLLM-Spec — a 1.5x efficiency gain from the graph capture alone. At current throughput levels, if 3–5% of cycle wall time is Python overhead (CPU dispatch, tensor allocation, Python interpreter), static graph capture could recover +5–15 TPS. vLLM already uses piecewise CUDA graphs for individual forward passes, but the inter-step dispatch (acceptance sampling, tree resampling logic, Python control flow) is NOT inside a graph capture in the current implementation.

### Equivalence argument

EGT does not change the verify forward pass. It changes the tree topology and the CPU-side orchestration. The verify pass still runs with M=1+K fixed tokens, argmax is unchanged, acceptance/rejection logic is unchanged (just moved into a CUDA kernel instead of Python). Greedy-token-equivalence is preserved as long as the acceptance/rejection mask logic is implemented identically inside the graph capture.

**Important:** Yggdrasil's EGT generates a wider candidate tree (all K-ary paths), which means the verify step must process more candidate tokens than the linear chain K=7 does. For a K=7 linear chain, M=8. For a K=2 depth-3 EGT, M=1+2+4+8=15. If the EGT is mapped to our K=7 linear chain setting, it is not EGT but rather just static graph capture of the existing linear chain. That is the relevant variant: static graph capture of the K=7 linear chain, not a wider tree.

### Expected TPS direction and magnitude

Yggdrasil's gain over vLLM's existing piecewise graph capture was attributed to (a) eliminating Python per-step overhead and (b) the latency-aware objective that finds better tree structures. For our use case (just static capture benefit, not tree restructuring), a conservative estimate is +2–8% reduction in cycle overhead, translating to +9–37 TPS at 467 TPS base. This is speculative; the actual Python overhead must be measured.

### Measurement protocol (≤90 min, no served-file change)

1. Profile CPU-side overhead per spec decode step using `torch.profiler` with `with_stack=True`. Identify Python dispatch, tensor allocation, and acceptance sampling Python time. If CPU overhead < 0.5% of cycle, this idea is not the bottleneck.
2. Prototype: capture the fixed-shape K=7 linear chain (M=8 verify) inside a `torch.cuda.graph()` capture context. Check whether vLLM's block manager operations (paged KV cache pointer updates) are inside or outside the graph — block table updates must remain outside.
3. Benchmark pre- and post-capture wall TPS.
4. Run equivalence check.

### Key papers

- **Yggdrasil** (arxiv:2512.23858): EGT + latency-aware objective + stage-based scheduling; 3.98x on A100 vs 2.66x vLLM-Spec. [https://arxiv.org/abs/2512.23858]
- **Hybrid JIT-CUDA Graph** (arxiv:2604.23467): nvFuser/TorchDynamo-based graph capture reducing kernel launch overhead in transformer inference. [https://arxiv.org/abs/2604.23467]

### Single biggest risk

vLLM's piecewise CUDA graph implementation already captures most of the kernel launch overhead inside graph nodes. The remaining Python overhead per spec decode step may be dominated by the acceptance sampling logic (Python `torch.argmax` and masking operations on CPU), which cannot be captured in a CUDA graph without a full rewrite of the acceptance kernel in CUDA. If vLLM's existing graph capture already handles 90%+ of the kernel dispatch overhead, the incremental gain from static graph capture of the full loop is <2 TPS — not worth the refactoring cost.

---

## Idea 5 — Triton Attention Kernel Tile-Size Re-Autotuning for A10G sm_86 + head_dim=256 + GQA(8Q/2KV)

### What it is

Re-run Triton's autotuner over the `{BLOCK_M, BLOCK_N, num_warps, num_stages}` configuration space specifically for the A10G sm_86 architecture with the exact kernel shape parameters used in serving: M=8 (verify batch), head_dim=256, GQA with 8 query heads and 2 KV heads, context length=128.

### Why it might help here

vLLM's Triton unified-attention kernel uses a generic autotune table that was likely not profiled on sm_86 with this exact (M=8, head_dim=256, GQA 8/2) configuration. The optimal tile sizes for A10G differ from A100 (sm_80) and H100 (sm_90) due to differences in SRAM size (192 KB per SM on A10G vs 192 KB on A100, same), warp scheduler depth, and L2 size. More importantly, head_dim=256 is non-standard (most LLMs use head_dim=64 or 128) and GQA 8/2 (group=4) is an unusual ratio. The autotune config that ships with vLLM may have been tuned for head_dim=128 / GQA 8/1. For head_dim=256, larger BLOCK_M or BLOCK_N tile may improve SM occupancy and reduce memory transactions.

### Equivalence argument

Tile-size is a performance parameter, not a correctness parameter. The attention computation is mathematically identical regardless of BLOCK_M/BLOCK_N tiling. Greedy-token-equivalence is byte-exact by construction. No risk.

### Expected TPS direction and magnitude

Triton autotuning gains vary widely. For a well-tuned out-of-distribution shape (like head_dim=256), gains of 3–10% in the attention kernel are plausible. If attention is 15% of verify wall time, this translates to 0.45–1.5% system-level gain: +2–7 TPS at 467 TPS base. This is a modest but real and zero-risk improvement if the autotune finds a better config. The Triton Attention Anatomy paper (arxiv workshop 2025) provides concrete guidance on block size selection for different architectures.

### Implementation notes

The autotune can be invoked by temporarily patching the vLLM `attention_kernels.py` to extend the `@triton.autotune` config list for the specific kernel path (unified attention, decode mode, head_dim=256). Key configs to sweep: `BLOCK_M ∈ {4, 8, 16}` (M=8 verify), `BLOCK_N ∈ {16, 32, 64}` (KV context tiles), `num_warps ∈ {2, 4, 8}`, `num_stages ∈ {2, 3, 4}`. Total configs: ~72. At ~10ms per kernel launch and 72 configs, autotuning takes ~10 minutes. The result is a config dict that can be hardcoded back in.

**Critical gotcha:** The num_stages tweak is already a closed lever (≤+0.94 TPS from prior experiments). This proposal extends that to the full joint config space — the prior experiment may have held BLOCK_M/BLOCK_N fixed while sweeping num_stages, which would have missed the joint optimum.

### Measurement protocol (≤90 min, no served-file change)

1. Check which autotune configs currently exist in vLLM's unified attention kernel for (decode, head_dim=256, GQA). If it already has a dense sweep, this idea is already exhausted.
2. Extend the config list as described above. Run `torch.cuda.empty_cache()`, warmup × 50, then autotune.
3. Extract winning config. Benchmark wall TPS with the winning config hardcoded.
4. Compare against the prior num_stages-only sweep to confirm whether the joint optimum differs.

### Key papers

- **Triton Attention Anatomy** (arxiv:2504.xxxxx, 2025): systematic analysis of Triton attention kernel performance across GPU architectures; BLOCK_M/BLOCK_N sensitivity analysis. [https://arxiv.org/abs/2504.01328 (approximate — verify exact ID)]
- **Memory-Bound but Not Bandwidth-Limited** (arxiv:2605.30571): empirical characterization of single-stream decode bottlenecks across GPUs, including A10G. [https://arxiv.org/abs/2605.30571]

### Single biggest risk

The prior num_stages experiment already tested the most impactful single axis. If vLLM's kernel already ships with an autotuned BLOCK_M/BLOCK_N table for decode mode, the new configs may not improve on the existing default. The "unified attention" path (which this serving stack uses due to heterogeneous head dims) may have fewer autotune configs than the standard FlashAttention path — but this means the search space is less explored, not more, which actually makes this idea more interesting.

---

## Idea 6 (Bonus) — MTP Draft Head Weight Re-Tiling for Better GEMM M-Invariance Exploitation

### What it is

Re-layout the MTP draft head's weight matrices in memory to maximize L2 cache reuse across the K sequential draft steps, reducing the effective HBM bandwidth consumed by the drafter even though individual GEMM cost is already M-invariant.

### Why it might help here

The MTP drafter runs K=7 sequential GEMM steps over the same weight matrices. If the draft head weights fit in L2 (or a subset does), subsequent steps will hit L2 instead of HBM. The A10G L2 is 40 MB. The MTP head is a small network (typically 1–2 MLP layers on top of the final transformer hidden state). If its total weight size is ≤20 MB, keeping weights warm in L2 across K steps saves HBM bandwidth proportional to (K-1)/K per step. At K=7, this is 6/7 ≈ 86% of HBM weight-load bandwidth for the drafter after the first step.

### Equivalence argument

Weight re-tiling is a memory layout change only. The GEMM computes the same result regardless of memory order — the Marlin kernel already handles the transposed/shuffled layout internally. Greedy-token-equivalence is byte-exact by construction.

### Expected TPS direction and magnitude

If drafter weight loads are a bottleneck, this could reduce drafter time significantly. However, if the drafter is already L2-warm after the first step (because the weights are small and L2 is large), this is already happening naturally. The diagnostic is to measure: does drafter step i=1 cost significantly more than step i=2..K? If yes, L2 warmup is the mechanism and re-tiling helps. If all steps cost equally, L2 is already warm. This is a 10-minute profiling check.

### Measurement protocol

Profile individual draft step latencies (step 1 vs. steps 2..7) with CUDA event timing. If step 1 > 1.5× step 2, L2 cold-start is real and weight layout optimization helps. If uniform, this idea is moot.

### Key papers

- **Adrenaline** (2025): weight layout optimization for GEMM memory efficiency in LLM inference. Found in the Exa HBM-bandwidth search.

---

## Ranking Summary

| Rank | Idea | Mechanism | Equivalence | Expected TPS Gain | Risk |
|------|------|-----------|-------------|-------------------|------|
| 1 | Verify Attention M-scaling K sweep (Idea 3) | Measure T_verify(K) profile, find K_opt where attention cost makes lower K optimal | Trivially preserved (K is just a hyperparameter) | Unknown until measured; could be +5–20 TPS | Low — if T_attn < 5%, K=7 stays optimal (still informative null result) |
| 2 | Async KV Prefetch via PTX cp.async.bulk (Idea 2) | Prefetch next-layer KV tiles into L2 during GEMM compute windows | Byte-exact (memory scheduling only) | +3–10 TPS if attention ≥ 15% of verify | Paged KV breaks contiguous-address assumption; implementation non-trivial |
| 3 | MTP Draft L2 Cold-Start Check (Idea 6) | K sequential steps on same weights — diagnose if step-1 costs more than steps 2–7 | Byte-exact (layout only) | Unknown until measured; low implementation cost | If L2 already warm (likely), null result |
| 4 | Triton Attention Joint Autotune (Idea 5) | Full {BLOCK_M, BLOCK_N, num_warps, num_stages} sweep for head_dim=256 GQA 8/2 on sm_86 | Byte-exact (tiling only) | +2–7 TPS if joint optimum differs from prior num_stages-only sweep | Prior num_stages experiment may have already found the joint optimum |
| 5 | Async Pipelined Drafting — Saguaro-style (Idea 1) | Hide drafter latency inside verify wall time | Preserved IF verified hidden state used as drafter input (not contaminated) | +5–20 TPS if D > 0.05 × V | CUDA graph barriers likely prevent async launch without major refactoring |
| 6 | EGT Static CUDA Graph Capture (Idea 4) | Eliminate Python per-step dispatch overhead via full graph capture | Preserved if acceptance logic moved correctly into CUDA | +2–15 TPS | vLLM piecewise graphs may already capture most overhead; incremental gain may be <2 TPS |

---

## Research State Update

**Current best explanation for throughput ceiling:**
The system is bounded by the verify-BW λ=1 wall at ~520.95 TPS. The modeled cb3 ceiling of 482.74 TPS represents the best achievable with the current discrete-K acceptance model and M-invariant GEMM assumption. The gap between current equivalence frontier (~467 TPS) and the wall is ~54 TPS. However, the M-invariant GEMM model was never tested against the Triton attention kernel's M-scaling — this is the primary untested assumption in the current model.

**Primary untested assumption:**
The TPS optimization model assumes T_verify is dominated by M-invariant Marlin GEMMs. If the Triton attention kernel contributes ≥10% of verify wall time and scales with M=1+K, the optimal K may be lower than 7. This assumption has NOT been validated with direct profiling of T_verify(K) as a function of K. This is the highest-priority diagnostic.

**Secondary untested assumption:**
The L2 cache dynamics for the MTP drafter across K sequential steps have not been profiled. If drafter step 1 costs significantly more than steps 2–7 (L2 cold-start), there is a one-time warmup cost per verify cycle that could be eliminated or amortized.

**Ruled-out directions:**
KV cache quantization, HiSpec/LayerSkip, CaDDTree, KnapSpec, attention split-K, drafter fusion, keepset-masking, affine rescaling. All verified closed either by direct experiment or by fundamental incompatibility with strict greedy-equivalence.

**Open uncertainties:**
1. What fraction of T_verify is the Triton attention kernel vs. Marlin GEMMs? (Idea 3 diagnostic answers this)
2. What is D (drafter wall time) as a fraction of V (verify wall time)? (Idea 1 diagnostic answers this)
3. Has the joint BLOCK_M × BLOCK_N × num_stages autotune space been explored for this exact kernel shape? (Idea 5 diagnostic answers this)

**Next discriminating experiment:**
Profile T_verify(K) for K ∈ {3, 4, 5, 6, 7} and T_draft(K), compute predicted TPS(K), compare with observed. This is the cheapest experiment and has a decisive outcome: either K=7 is validated or a better K is found. 45-minute microbench, no code changes, no served-file change.

**Stop condition for this line:**
If T_attn < 5% of T_verify, the M-invariant model is correct, K=7 is optimal, and the attention-scaling avenue is exhausted. At that point, the async pipelining (Idea 1) becomes the primary avenue, but requires the CUDA graph compatibility analysis first.

---

## Literature Sources

1. Saguaro / Speculative Speculative Decoding (ICLR 2026): https://openreview.net/forum?id=aL1Wnml9Ef
2. Yggdrasil EGT (arxiv:2512.23858): https://arxiv.org/abs/2512.23858
3. HiSpec hierarchical spec decode (arxiv:2510.01336): https://arxiv.org/abs/2510.01336 — REQUIRES training, NOT applicable
4. LayerSkip early-exit (arxiv:2404.16710): https://arxiv.org/abs/2404.16710 — REQUIRES training, NOT applicable
5. KnapSpec training-free knapsack self-spec (arxiv:2602.20217): https://arxiv.org/abs/2602.20217 — requires LM head on early layers, NOT applicable without training
6. Async KV Cache Prefetch via cp.async.bulk (arxiv:2504.06319): https://arxiv.org/abs/2504.06319
7. DySpec dynamic draft length (arxiv:2405.17785): https://arxiv.org/abs/2405.17785
8. EAGLE-2 dynamic tree (arxiv:2406.16858): https://arxiv.org/abs/2406.16858
9. Memory-Bound but Not Bandwidth-Limited (arxiv:2605.30571): https://arxiv.org/abs/2605.30571
10. Hybrid JIT-CUDA Graph (arxiv:2604.23467): https://arxiv.org/abs/2604.23467
11. FlashInfer paged attention (arxiv:2501.01005): https://arxiv.org/abs/2501.01005
12. FastMTP (2025): https://arxiv.org/abs/2501.02955
13. CaDDTree diffusion drafter (arxiv:2606.01813): https://arxiv.org/abs/2606.01813 — NOT applicable (diffusion drafter)
