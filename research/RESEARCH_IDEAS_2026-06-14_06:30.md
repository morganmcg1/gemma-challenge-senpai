# Research Ideas — 2026-06-14 06:30

> **⚠️ ADVISOR CROSS-CHECK (06:38Z) — read before assigning. The researcher-agent ran on inline context and did NOT have our "Confirmed dead ends" list, so it re-derived 4 already-closed lanes. De-conflicted against `CURRENT_RESEARCH_STATE.md` dead-ends:**
>
> **GENUINELY FRESH (queue these):**
> - **B1 (MaskLLM 2:4 structured sparsity + argmax-margin gate) — THE REAL TOP PICK.** Not in our dead-ends. Distinct mechanism from the int4 weight-byte floor: Ampere Sparse Tensor Cores do 2:4 sparse matmul at up to 2× dense throughput — a *hardware* lever, not lower bit-width. Highest conditional upside (+25–35% if >60% of layers pass the safety gate). The offline argmax-margin safety scanner is a **zero-GPU build-or-kill** that reuses kanna #87's 65,536-position margin map + wirbel #93/#98 numerics discipline. **Assign the GATE first** (free); sparse training only if it passes. Fits a numerics-strong seat (kanna/wirbel).
> - **B2 (mixed W4/W3 per-layer argmax-gated) — FRESH, conditional.** Per-layer int3 sensitivity scan (offline, no retrain) → int3 Marlin on layers where int3-error < argmax-margin for >99.9% of positions. +8–12% if ~50% of layers qualify. Cheap offline first-experiment. Composes with B1.
>
> **RISKY (drafter likely MTP-parity-bound — frame as a direct challenge):**
> - **A2 (Hydra sequential MTP heads) — not explicitly closed, but our drafter lane is heavily mined.** #80 found EAGLE/HASS reached only MTP parity; #77 found no addressable drafter non-GEMM headroom; #75 int4-drafter ~+5% ceiling. Hydra's sequential head-conditioning is a *different* axis (each head sees the prior draft token), so it's not strictly refuted — but the prior is that the linear MTP head is already near this model's drafter-quality ceiling. Only assign if B1/B2 are taken, and only as a cheap offline per-head-acceptance probe (h_1…h_7) build-or-kill BEFORE any retrain.
>
> **ALREADY CLOSED — DO NOT RE-ASSIGN (researcher re-derivation = independent confirmation, not a new lever):**
> - **A1 (EAGLE-3 multi-layer fusion drafter) — CLOSED, #80.** Dead-ends: "EAGLE-3 drafter training: CLOSED — MTP parity (HASS +57% offline ≪ MTP, #80)." We already tested EAGLE-style drafting; even HASS-harmonized it could not beat the existing linear MTP head. The multi-layer-fusion variant is unlikely to clear the same parity wall; not worth a seat over B1/B2.
> - **A3 (n-gram / prompt-lookup hybrid) — CLOSED, #89.** Dead-ends: "Prompt-lookup augment: +1.67% gross, structural corr ceiling +2.38% (#89)." denken #89 measured it and dropped it (structural redundancy with the MTP first-reject). Only a REST-style *external datastore* (beyond the prompt) is arguably untested — low priority.
> - **C1 (async drafter-verify overlap) — CLOSED, #94.** Dead-ends + denken #94 (AMBER): measured `bandwidth_limited_overlap_ceiling_pct = 4.22%` wall → +1.2–2.9% realized. The single A10G HBM bus serializes two memory-bound streams (`bus_contention_factor=0.506`). Re-opens only on a 2nd GPU or a compute-bound regime. The researcher's +3–7% is the naive number our measurement already collapsed.
> - **C2 (Triton RMSNorm+residual+dequant fusion) — CLOSED, #67.** Dead-ends: "Decode norm/elementwise fusion: ceiling <0.5% (#67)." Already measured sub-bar. (Distinct from the megakernel lane #97, but the byte-saving is too small.)
>
> **De-conflicted assignment order for the next freed seats:** (1) **B1 MaskLLM 2:4 sparsity** — assign the zero-GPU argmax-margin SAFETY GATE first → (2) **B2 mixed W4/W3** per-layer sensitivity scan → (3) **A2 Hydra** per-head-acceptance probe (only if it can challenge the #80 MTP-parity wall). **SKIP A1, A3, C1, C2** (closed). Board is currently saturated (8/8 WIP) — these queue for the next freed seat (likely ubel #84 deadline repoint or a CPU gate finishing).
>
> **Process note for next researcher-agent run:** include the "Confirmed dead ends (comprehensive)" list from `CURRENT_RESEARCH_STATE.md` inline so it doesn't re-derive closed lanes (4 of 7 here were closed). The B-bucket (weight-byte reduction beyond int4: structured sparsity, mixed-precision, palettization) is the least-mined fresh territory — point future runs there.

## Context snapshot

- **Primary metric:** output TPS (tokens/sec), greedy, 128-token prompt / 512-token generation, concurrency=1
- **Current frontier:** 481.53 TPS
- **Target:** ~500 TPS (+3.8%) and beyond
- **HARD CONSTRAINT:** greedy-token-identity — served output must be bit-identical token-for-token to plain greedy autoregressive decode of the same checkpoint. ANY emitted-token change is DISQUALIFYING.

### Current stack

- int4 W4A16 Marlin weights
- Linear MTP drafter K=7
- 3D split-KV paged attention
- Sparse lm-head verify (12k vocab subset)
- Fused sliding-window attention
- CUDA-graph one-graph capture
- Weight precache

### Step profiling breakdown

- ~53% verify-GEMM (dominant: streaming int4 weight bytes)
- ~7% drafter
- ~6-8% attention
- ~30% small-kernel-dispatch tail (genuine GPU-busy small-kernel compute, NOT idle)
- ~2% intra-graph idle

### Ruled out (do NOT re-propose)

- Wide speculative TREE verify
- Persistent-kernel / megakernel (REFUTED — 30% is genuine GPU compute)
- Double-quantization of int4 group-scales (already assigned)
- LK-loss / direct acceptance-rate drafter fine-tuning (already queued)

---

## BUCKET A — More accepted tokens per weight read, without a wider tree

These attacks increase E[accepted tokens per verify step], reducing how many times the full 53% verify-GEMM weight-read has to run per output token. Greedy-identity is trivially preserved because these ideas only affect the drafter; verification remains exact argmax on the full model.

---

### A1. EAGLE-3-style multi-layer hidden-state fusion drafter (HIGHEST PRIORITY)

**Mechanism:** Replace the current linear MTP head's single-layer feature prediction with direct next-token prediction conditioned on a weighted mixture of hidden states from multiple verifier layers (e.g., layers N, N/2, N/4). EAGLE-3 calls this "training-time test" — the drafter is trained to predict tokens rather than features, using multi-layer context that gives it a richer signal than a single-layer projection. Each draft head operates in parallel on the fused state.

**Why greedy-token-identity is preserved:** The drafter is a proposal generator only. Verification is still exact: the full verifier model runs a forward pass on the draft sequence and applies the standard argmax acceptance rule. Any draft that the verifier would reject under greedy is rejected. No output token ever comes from the drafter alone.

**TPS-upside estimate:** EAGLE-3 reports 6.5× over autoregressive on MT-Bench across 7B-70B models. Current K=7 MTP is already speculative, so the gain is the delta between current drafter acceptance rate and an EAGLE-3-equivalent rate. Conservative estimate: if acceptance rate lifts from ~0.55 to ~0.68, the expected tokens per verify step rises from ~3.84 to ~4.7, a ~22% reduction in verify-GEMM calls. On Ampere with W4A16 Marlin, that translates to roughly +15–25% TPS. The 7% drafter step grows slightly (deeper drafter), but is dominated by the verify savings.

**Step bucket attacked:** Primarily verify-GEMM (53%). Secondarily drafter (7%) — net reduction in verify calls outweighs increased drafter cost at the reported acceptance ratios.

**Cheap first experiment:** Offline calibration trace: run the current model on 500 MT-Bench prompts and log the hidden states at layers N, N/2, N/4 and the final logits. Train a simple linear probe from each set of hidden states to predict the greedy argmax token. Measure prediction accuracy vs. the current single-layer MTP head. If the multi-layer fusion probe has >5 pp higher top-1 accuracy, the EAGLE-3 mechanism is alive and worth a full drafter retrain. This is a CPU job on saved activations — no GPU training needed to validate the premise.

**References:**
- EAGLE-3: Exploiting More Redundancy in Large Language Model Serving (2025). arxiv:2503.01840. Multi-layer feature fusion, direct token prediction, 6.5× speedup on MT-Bench.
- EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (2024). arxiv:2406.16858. Dynamic draft tree construction, 3.05× over standard EAGLE.

---

### A2. Hydra heads — sequentially-dependent MTP draft heads

**Mechanism:** Current linear MTP heads generate K=7 draft tokens independently in parallel (Medusa-style). Hydra replaces this with sequentially-conditioned heads: head k+1 receives the output embedding of head k as additional context before predicting token k+1. This makes each head's prediction conditional on the preceding draft token, dramatically improving acceptance probability on any token where local context matters (which is most tokens in coherent text).

**Why greedy-token-identity is preserved:** Identical argument to A1. Hydra only changes the drafter proposal distribution. The verify step uses standard greedy acceptance: the full model forward pass on the draft sequence and argmax matching. Draft tokens that the verifier would reject are rejected. Identity is exact.

**TPS-upside estimate:** Hydra reports 2.70× over autoregressive and 1.31× over Medusa (which has independent heads). The current stack is already at K=7 speculative with linear heads, so the incremental gain is the Hydra-over-Medusa delta applied to the acceptance-rate improvement. If acceptance rate improves by ~15 pp (from Hydra's sequential conditioning), verify call reduction is roughly 10–15%, targeting +8–12% TPS on the current frontier. Lower bound than A1 but architecturally simpler to retrofit onto the existing MTP framework.

**Step bucket attacked:** Verify-GEMM (53%) via fewer verify calls per output token.

**Cheap first experiment:** On a frozen current MTP K=7 model: measure per-head acceptance rates h_1 through h_7. If the acceptance rate drops sharply after h_3 (a common pattern in independent-head drafters), the sequential conditioning hypothesis is alive. A 1-day fine-tune of the MTP heads with sequential conditioning (each head receives the embedding of the previous accepted/draft token) costs roughly the same compute as the original MTP head training.

**References:**
- Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding (2024). arxiv:2402.05109. 2.70× over autoregressive, 1.31× over Medusa.
- Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (2024). arxiv:2401.10774. Original independent-head baseline that Hydra improves upon.

---

### A3. N-gram / prompt-lookup hybrid drafter — zero-parameter acceptance boost on structured outputs

**Mechanism:** Augment the neural MTP drafter with a zero-cost n-gram lookup table built from the input prompt and recent context window at inference time. For each position where the neural drafter proposes token t_k, also propose a "retrieval candidate" by doing an exact-match lookup of the preceding 3-gram in the prompt/context. If the retrieval candidate matches or outscores the neural candidate, substitute it as the draft proposal. This is the "prompt lookup decoding" mechanism from Saxena et al. (2023), extended to fuse with the neural drafter.

**Why greedy-token-identity is preserved:** All retrieved draft tokens are subject to the same exact verify step. A retrieved n-gram is never emitted unless the full verifier model assigns it the highest logit at that position. The retrieval only changes the *proposal distribution*, not the acceptance rule.

**TPS-upside estimate:** Prompt lookup alone achieves 2.5–4× speedup on tasks with high context repetition (code completion, structured documents, chain-of-thought reasoning with repeated patterns). On MT-Bench-style queries (low repetition), the gain is near zero — it's purely additive. Since this is zero-parameter (just a hashmap at inference time), the floor is +0% and the ceiling is significant on structured workloads. Targets verify-GEMM (53%) by reducing verify calls on repetitive subsequences.

**Step bucket attacked:** Verify-GEMM (53%) on structured/repetitive outputs.

**Cheap first experiment:** Measure the fraction of ground-truth greedy tokens on the MT-Bench calibration set that appear verbatim in the preceding 512-token context window (as a 3-gram completion). If >15% of tokens are context-recoverable, the retrieval path adds real acceptance rate. This is a pure Python script on saved token sequences — no model needed.

**References:**
- Prompt Lookup Decoding (Saxena, 2023). github.com/apoorvumang/prompt-lookup-decoding. Zero-parameter retrieval speculation from the context window.
- REST: Retrieval-Based Speculative Decoding (2023). arxiv:2311.08252. Generalizes retrieval speculation to a datastore beyond just the prompt.

---

## BUCKET B — Fewer weight bytes per token while staying bit-exact greedy

These attacks reduce the number of bytes the GPU must stream from HBM per verify call, directly targeting the 53% verify-GEMM bottleneck. The greedy-identity constraint is the hardest filter here: any change to model weights must be provably argmax-safe or the idea is disqualifying.

---

### B1. Learnable 2:4 structured sparsity (MaskLLM) with offline argmax-margin safety gate

**Mechanism:** Apply 2:4 structured sparsity to the W4A16 Marlin weight matrices of the verifier. On Ampere, 2:4 sparse matrix multiplication is hardware-accelerated via Sparse Tensor Cores at up to 2× the throughput of dense matmul. MaskLLM (NeurIPS 2024, NVIDIA) introduces learnable mask selection via Gumbel Softmax sampling, treating the sparsity pattern as a discrete latent variable. This achieves 6.72 PPL on LLaMA-2 (vs 10+ for magnitude pruning) — far better accuracy than magnitude pruning. The argmax-margin safety gate: after training the sparse mask, run an offline calibration pass over 1000 prompts and, for every output position, check whether the logit gap between rank-1 and rank-2 tokens exceeds a conservative bound on the per-layer reconstruction error from the sparse approximation. Layers/positions where the gap is too small are left at the original int4 dense weights (or flagged for mixed treatment).

**Why greedy-token-identity is preserved (conditional):** This is the only idea in bucket B that might survive the greedy-identity gate, but it requires careful per-layer argmax safety analysis. The sparse model is not bit-identical to the dense model in general. Argmax safety holds if, for every position in the calibration distribution, the rank-1 logit under the sparse model equals the rank-1 logit under the dense model. Concretely: offline analysis must show that the logit perturbation from sparsity (bounded by the masked weight norm times activation norm) never exceeds the logit margin at any position. If even one layer fails the gate, that layer stays dense. The resulting model is a mixed sparse/dense model — but if 60-70% of layers pass the safety gate, the GEMM bandwidth reduction is substantial.

**TPS-upside estimate:** If 2:4 sparsity passes the safety gate for 70% of verifier layers, the effective weight-byte throughput for those layers doubles. On the 53% verify-GEMM bottleneck, a 70% layer coverage at 2× throughput implies roughly 37% reduction in verify-GEMM wall time. Net TPS gain depends on whether verify-GEMM is truly bandwidth-bound: on A10G at 600 GB/s HBM, W4A16 GEMM at batch=1 is heavily bandwidth-bound, so the speedup should be close to 2× on covered layers. Estimated total TPS gain: +25–35% if coverage is good.

**Step bucket attacked:** Verify-GEMM (53%) directly — reduces weight bytes streamed per token.

**Cheap first experiment (critical — do this before any training):** On the frozen current int4 model, compute per-layer argmax margins on 1000 calibration prompts: for each output token position, record the logit gap margin[i] = logit[rank1] - logit[rank2]. Also compute an upper bound on the perturbation that 2:4 sparsity would introduce (||W - W_sparse||_F * ||activation||_2 per layer). If margin[i] > perturbation_bound for >95% of positions and >80% of layers, the idea is viable. This is a Python script on saved activations — no sparse training needed to validate the safety premise.

**References:**
- MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models (NeurIPS 2024, NVIDIA). arxiv:2409.17481. github.com/NVlabs/MaskLLM. 6.72 PPL on LLaMA-2, Gumbel Softmax mask learning.
- 2:4 Structured Sparsity: Exploiting NVIDIA Ampere Sparse Tensor Cores (NVIDIA docs, 2021). hardware.accelerated 2:4 on sm_86, up to 2× dense matmul throughput.

---

### B2. Mixed W4/W3 quantization with per-layer argmax-margin gating

**Mechanism:** Not all layers are equally sensitive to quantization. On a frozen int4 model, run a per-layer sensitivity analysis: for each transformer layer, compute the expected logit perturbation from int3 quantization (via analytical bound or empirical activation-weighted error), and compare against the per-position argmax margin on a calibration corpus. Layers where int3 error < margin for >99% of calibration positions are "safe" to quantize to int3. Safe layers use int3 Marlin kernels (which exist and are slightly faster due to reduced weight bytes). Unsafe layers stay at int4. No model retraining needed if the analysis is purely applied to the frozen int4 model.

**Why greedy-token-identity is preserved (conditional):** Same mechanism as B1 — per-layer, per-position argmax margin check. The key is that int3 on a "safe" layer cannot flip the final logit ranking at the output head. This requires a careful bound on how int3 error in layer L propagates to the output logits — either via worst-case sensitivity analysis or empirical measurement on the calibration set.

**TPS-upside estimate:** int3 vs int4 reduces weight bytes by 25% (3/4 × 4-bit). If 50% of verifier layers are safe for int3, the expected verify-GEMM reduction is ~12.5%, yielding roughly +8–12% TPS. Modest but achievable without any training.

**Step bucket attacked:** Verify-GEMM (53%).

**Cheap first experiment:** Implement a Python sensitivity scanner that, for the frozen int4 model, takes each layer's int4 weights, requantizes them to int3, computes the forward-pass logit difference on 500 calibration prompts, and reports the fraction of positions where the top-1 argmax is unchanged. Layers with >99.9% argmax stability are "safe" candidates. This is a CPU/single-GPU job, no training needed.

**References:**
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (2024). arxiv:2402.04396. Analyzes per-layer quantization sensitivity and argmax stability bounds.
- SqueezeLLM: Dense-and-Sparse Quantization (2023). arxiv:2306.07629. Mixed-precision LLM quantization framework with sensitivity-guided layer selection.

---

## BUCKET C — Cut the ~30% small-kernel-dispatch tail

These attacks target the 30% genuine GPU-busy small-kernel compute (NOT idle), which was confirmed as real work by the persistent-kernel overhead gate experiment. The goal is to reduce the number of distinct kernel launches per step and fuse short-running kernels into fewer, better-utilized kernels.

---

### C1. Async drafter-verify compute overlap (drafter hidden inside verify latency)

**Mechanism:** In the current sequential pipeline, the full verify forward pass runs, then the drafter forward pass runs. Since the drafter is only ~7% of wall time but occupies a serial slot, launching the drafter asynchronously on a second CUDA stream while the verify matmuls are still executing would hide most of the drafter cost inside the verify cost. On A10G with separate SM partitions, the drafter (a small linear network) can run concurrently with the early layers of verify without resource contention. The key: the drafter for step t+1 can begin computation as soon as the K accepted tokens from step t are known — before verify for step t+1 has started. This is a CUDA stream overlap, not a kernel change.

**Why greedy-token-identity is preserved:** This is a scheduling optimization only. The verify forward pass and acceptance rule are identical. No weights change. No output token ever comes from the drafter alone.

**TPS-upside estimate:** If the 7% drafter cost is fully hidden inside the 53% verify cost, the drafter contributes 0% to latency instead of 7%, a net gain of ~7% TPS on top of whatever the current frontier is. On the current 481.53 TPS frontier, that is roughly +34 TPS, landing at ~515 TPS — above the 500 TPS target. In practice, stream contention on a single A10G may reduce this to 3–5% effective gain, but even the lower bound is worth the implementation cost.

**Step bucket attacked:** Drafter (7%) — effectively eliminates its contribution to serial latency.

**Cheap first experiment:** CUDA event timing: add `cudaEvent_t` markers around the drafter kernel and the first N verify-layer GEMM. Confirm on the A10G that the drafter's SM utilization (visible via Nsight) does not saturate the GPU when the verify GEMM is running. If SM utilization during verify is <85%, there is headroom for concurrent drafter execution. This is a profiling run (1 inference pass with Nsight), not a training experiment.

**References:**
- Orca: A Distributed Serving System for Transformer-Based Generative Models (2022). OSDI 2022. Discusses stream-level overlap between prefill and decode for serving throughput.
- FlexFlow Serve: Low-Latency, High-Performance LLM Serving (2023). arxiv:2302.12307. Multi-stream speculative decoding overlap strategies.

---

### C2. Triton kernel fusion: RMSNorm + residual add + dequant epilogue

**Mechanism:** The 30% small-kernel tail is composed of many individually short kernels: RMSNorm, residual add, gating (SwiGLU), dequant scale-apply, and potentially attention output projection bias. Each of these is a separate CUDA launch with its own dispatch overhead and HBM round-trip for intermediate tensors. Fusing them into a single Triton kernel (e.g., dequant output epilogue → residual add → RMSNorm → output) eliminates both launch overhead and the intermediate tensor HBM write/read round-trips. This is a well-established pattern in modern inference kernels (FlashAttention-2 fuses attention with softmax; Flash-Decoding fuses partial softmax reductions).

**Why greedy-token-identity is preserved:** Fusing elementwise operations (residual add, RMSNorm, gating) produces bit-identical results to running them separately (within floating-point associativity; RMSNorm implemented in fp32 accumulation preserves identity). The only risk is if the fusion changes the accumulation order in a way that shifts a logit. This is mitigated by: (a) keeping fp32 accumulation in RMSNorm, and (b) running a diff check on 100 prompts between the fused and unfused paths before deploying.

**TPS-upside estimate:** Each eliminated kernel launch on A10G costs roughly 5–10 µs of dispatch + ~2 µs HBM round-trip for a small tensor. If the 30% tail is composed of ~40 small kernels per verify step at 128-token context, fusing 50% of them into 20 kernels could reduce the tail by 10–15%, yielding roughly +3–5% total TPS. Modest but compound with other gains and free of correctness risk.

**Step bucket attacked:** Small-kernel-dispatch tail (30%).

**Cheap first experiment:** Nsight Systems trace: count the number of distinct CUDA kernel launches per verify step that are shorter than 10 µs. If >20 such kernels exist (expected: yes), the fusion opportunity is confirmed. Then prototype a single fused Triton kernel for dequant-epilogue + residual-add + RMSNorm and verify bit-level identity against the unfused path on 100 prompts. This is a 2-day implementation job.

**References:**
- Flash-Decoding for Long-Context Inference (2023). arxiv:2311.01100 / Tri Dao blog. Kernel fusion pattern for decode-time attention with split-KV.
- FlashAttention-2 (2023). arxiv:2307.08691. Demonstrates kernel fusion of tiled attention + softmax, eliminating intermediate tensor round-trips.

---

## Ranked priority order

| Rank | ID | Idea | TPS-upside (est.) | Greedy-gate risk | Implementation cost |
|------|-----|------|------------------|-------------------|---------------------|
| 1 | A1 | EAGLE-3 multi-layer feature fusion drafter | +15–25% | None (drafter only) | Medium (drafter retrain) |
| 2 | B1 | MaskLLM 2:4 sparsity + argmax-margin gate | +25–35% (conditional on safety) | High (requires offline safety proof) | High (sparse training) |
| 3 | C1 | Async drafter-verify overlap | +3–7% | None (scheduling only) | Low (CUDA stream) |
| 4 | A2 | Hydra sequential MTP heads | +8–12% | None (drafter only) | Low-medium (head retrain) |
| 5 | A3 | N-gram prompt-lookup hybrid drafter | +0–15% (task-dependent) | None (retrieval only) | Very low (hashmap) |
| 6 | C2 | Triton RMSNorm+residual+dequant fusion | +3–5% | Very low (fp32 accumulation) | Medium (Triton kernel) |
| 7 | B2 | Mixed W4/W3 per-layer argmax-gated | +8–12% (conditional) | Medium (propagation analysis) | Low (no training) |

**Top pick for immediate student assignment:** A1 (EAGLE-3 drafter) — highest uncapped upside, no greedy-identity risk, directly addresses the verify-GEMM dominance by increasing accepted tokens per step. The offline activation probe is a cheap go/no-go gate before committing to drafter retraining.

**Second pick:** C1 (async drafter overlap) — almost free TPS gain with no correctness risk. Profiling confirmation is a half-day task; implementation is 1–2 days. Should be assigned in parallel with A1.

**Conditional high-upside bet:** B1 (MaskLLM 2:4 sparsity) — if the offline argmax-margin safety test passes for >60% of layers, the TPS upside is the largest of any idea in this list. Assign the safety diagnostic as a standalone sub-task before committing to sparse training.

---

## Stop conditions

- **A1 stop:** If the offline activation probe shows multi-layer fusion has <2 pp accuracy advantage over the single-layer MTP head, retrain is not justified; close.
- **B1 stop:** If the argmax-margin safety analysis finds fewer than 40% of layers are safe for 2:4 sparsity, the bandwidth reduction is too small to justify sparse training; close.
- **C1 stop:** If Nsight profiling shows SM utilization during verify >90% (no headroom for concurrent drafter), async overlap will be throttled; deprioritize.
- **A3 stop:** If the calibration trace shows <5% of output tokens are context-recoverable via 3-gram lookup, prompt-lookup decoding adds negligible acceptance rate; skip.
