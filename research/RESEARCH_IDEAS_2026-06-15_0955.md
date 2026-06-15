# Research Ideas: Fast Gemma Challenge — Strict-Compliant >500 TPS
**Date:** 2026-06-15 09:55
**Context:** Single-stream vLLM inference on A10G 24GB (sm_86 Ampere). Target: >500 TPS greedy-identity-exact. Deployed baseline: 481.53 TPS (PPL 2.3772). Hard constraint #192: token-for-token match with reference greedy AR decode.

**Banked analysis:**
- Non-speculative strict-compliant ceiling: ~469.68 TPS (provably below 500 on A10G)
- EAGLE-3: NO-GO (top-4 root coverage ~0.8903; worst-case private-α floors compliant throughput at ~492.87 TPS)
- Speculative decoding (accepted-token amortization) is the only demonstrated path to >500 strict-compliant

---

## Priority Ranking (descending expected probability of cracking strict-compliant >500)

| Rank | Idea | Focus Area | Lossless/Exact | Expected TPS Mechanism | Confidence |
|------|------|-----------|----------------|----------------------|-----------|
| 1 | DySpec dynamic token tree | A | Yes (greedy temp=0) | 9.1× on Llama2-70B greedy; dynamic tree avoids wasted speculative capacity | High |
| 2 | Medusa-1 + Traversal Verification | A | Yes (Medusa-1 variant) | Tree-attention multi-head drafting; traversal verification recovers discarded valid subsequences | High |
| 3 | REST retrieval-based SD | C | Yes (greedy acceptance) | No draft-model training; retrieval fills token tree from datastore; 1.62–2.36× | High |
| 4 | Hydra sequential draft heads | A | Yes | 1.31× over Medusa++; sequential conditioning reduces wasted draft capacity | Medium-High |
| 5 | Goodput-optimized γ selection | C | Yes | Per-step optimal draft length; avoids overspeculation tax on low-acceptance positions | Medium-High |
| 6 | Lossless Vocab Reduction | C | Yes (provably lossless) | Reduces lm_head matrix cost by removing never-sampled tokens; orthogonal to SD | Medium |
| 7 | BI FlashInfer kernel on sm_86 | B | Yes (BI by construction) | Replaces non-BI SDPA; closes ~3–5 TPS gap from bf16 reduction variability | Medium |
| 8 | SAM (Synergistic Augmentation for SD) | C | Yes | Beats REST/Lookahead/PIA; up to 3.0× on reasoning models | Medium |
| 9 | LogitSpec next-next retrieval | C | Yes | 2.61×; next-next-token speculation via logit-based retrieval; vLLM-native integration path | Medium |
| 10 | CAS-Spec cascade adaptive self-spec | A | Yes (on-the-fly lossless) | No draft model; adaptive exit layer; orthogonal to vocab reduction | Medium |
| 11 | PPD hardware-aware sparse tree | C | Yes | 0.0002% trainable params; 28% higher acceptance on long-range; 16h A100-40GB train | Medium |
| 12 | Hierarchical SD (ICLR 2026 Oral) | A | Yes (provably lossless) | Overcomes joint intractability; boosts expected accepted tokens across multi-level draft | Medium |
| 13 | LayerSkip self-speculative | C | Yes (lossless) | No draft model; early layers draft, remaining verify; requires special training recipe | Low-Medium |
| 14 | DASH deterministic attention scheduling | B | Yes (deterministic) | Deterministic attention scheduling for reproducible high-throughput; may unlock BI-compatible optimizations | Low-Medium |
| 15 | NanoFlow intra-device parallelism | C | Neutral | Intra-device compute/memory/network overlap; orthogonal optimization | Low |

---

## Focus Area A: Speculative Decoding with High Acceptance / Low Private-α Tax

### A1. DySpec — Dynamic Token Tree Speculative Decoding
**What it is:** Dynamically constructs and adapts the speculative token tree structure at inference time, rather than using a fixed tree topology. Allocates speculative capacity to the highest-probability branches.

**Why it might help here:** The private-α tax on EAGLE-3 comes from static tree structure wasting capacity on low-probability branches. DySpec's dynamic allocation should significantly improve worst-case acceptance rates, reducing the private-α floor. At greedy (temperature=0), the paper reports 9.1× throughput on Llama2-70B — the greedy condition is exactly our #192 requirement.

**Key papers:**
- "DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure," OpenReview orr5uPZY28, submitted ICLR 2025. [https://openreview.net/forum?id=orr5uPZY28] — Dynamic tree construction; 9.1× TPS at greedy on Llama2-70B; ablation shows static trees leave significant capacity on the table.

**Implementation notes:**
- Dynamic tree construction requires per-step tree topology decisions — adds CPU-side planning overhead. On A10G single-stream this may be manageable since there is no batch contention.
- Must verify that dynamic branching decisions are deterministic given the same input. If tree structure selection uses any stochastic element, add a seed-fixed path.
- The vLLM fork's CUDA graph capture may conflict with variable tree topologies. Check whether dynamic trees require disabling CUDA graph capture or whether a shape-bucketing approach is feasible.
- Acceptance verification must remain greedy-exact. DySpec does not change the verification protocol — accepted tokens are still those that pass standard speculative acceptance under greedy, so #192 compliance is structural.

**Suggested experiment:**
1. Integrate DySpec tree construction into the vLLM fork's speculative decoding path (replacing the static EAGLE-3 tree).
2. Run greedy-identity check on a fixed 100-sample prompt set against reference AR output — verify zero token mismatches before any TPS measurement.
3. Measure single-stream TPS at 128/128 prompt/completion. If >500, run full PPL evaluation to confirm PPL ≤ 2.42.
4. Ablation: compare fixed-depth tree (depth=4) vs. dynamic tree to quantify acceptance gain.

**Taste rubric:**
- Mode: Tier shift (new SD mechanism replacing EAGLE-3)
- Mechanistic grounding: 4 — dynamic tree directly targets the static-tree capacity waste that drives private-α tax; greedy-temperature=0 result directly matches our constraint
- Research-state value: 4 — either cracks >500 compliant or sharply constrains the acceptance-rate hypothesis
- Execution value: 3 — moderate integration effort; cheap greedy-identity check gates the full run

---

### A2. Medusa-1 + Traversal Verification
**What it is:** Medusa adds multiple draft heads to a frozen backbone that each predict tokens k steps ahead in parallel. Medusa-1 is the provably lossless variant. Traversal Verification (NeurIPS 2025) replaces the standard top-down tree verification with a leaf-to-root traversal that recovers valid subsequences that top-down verification discards.

**Why it might help here:** Medusa-1 is already used in several vLLM deployments. The Traversal Verification upgrade is a drop-in replacement for the verification pass — it consistently improves acceptance length without changing the draft model or backbone. Given EAGLE-3's root-coverage bottleneck, any improvement in per-position acceptance rate compounds. The combination also has well-understood failure modes.

**Key papers:**
- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads," arXiv 2401.10774, ICML 2024. [https://arxiv.org/abs/2401.10774] — Medusa-1 lossless variant; tree-based attention; widely reproduced.
- "Traversal Verification: Unlocking the Full Potential of Tree-based Speculative Decoding," OpenReview 8nOMhDFpkU, NeurIPS 2025. [https://openreview.net/forum?id=8nOMhDFpkU] — Leaf-to-root verification; provably lossless; consistently improves acceptance length over top-down verification.
- "Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding," arXiv 2402.05109, OpenReview FbhjirzvJG, COLM 2024. [https://arxiv.org/abs/2402.05109] — Sequential conditioning of draft heads; 1.31× over Medusa++; improves accuracy by conditioning each head on predecessors.

**Implementation notes:**
- Medusa requires fine-tuning the draft heads on the target model (google/gemma-4-E4B-it). The Medusa paper recommends 1–3 heads at minimum; 5 heads is a common configuration.
- Traversal Verification is a pure inference-time change — no retraining needed. The reference implementation is available at the NeurIPS 2025 OpenReview page.
- Hydra adds sequential dependency between heads, increasing draft quality at the cost of a slight increase in draft head compute. Given the A10G's memory bandwidth constraint, the tradeoff may favor 3 Hydra heads over 5 standard Medusa heads.
- Critical: Medusa-1 (lossless) requires the typical acceptance criterion from the original speculative decoding paper. Medusa-2 (lossy) must NOT be used — it violates #192.
- The CUDA graph capture in vLLM works well with Medusa since the tree structure is fixed. This is an advantage over DySpec.

**Suggested experiment:**
1. Fine-tune 3–5 Medusa-1 draft heads on Gemma-4-E4B-it using the Medusa training recipe (standard 1–2 epoch fine-tune on domain-representative data).
2. Integrate Traversal Verification as a drop-in replacement for the standard top-down verification pass.
3. Greedy-identity gate: verify zero token mismatches on 100-sample test set.
4. Measure TPS. If baseline Medusa-1 is near but below 500, apply Hydra sequential conditioning as a second step.

**Taste rubric:**
- Mode: Tier shift (adding draft heads to frozen backbone)
- Mechanistic grounding: 4 — Medusa-1 is lossless by construction; Traversal Verification has formal proof; Hydra's sequential conditioning directly improves acceptance
- Research-state value: 3 — would confirm or refute whether acceptance-rate improvements alone cross 500 TPS
- Execution value: 3 — staged: traversal verification is free (no retraining); full Medusa requires fine-tune

---

### A3. Hierarchical Speculative Decoding (ICLR 2026 Oral)
**What it is:** A provably lossless framework that introduces a hierarchy of draft models, overcoming the joint intractability problem that limits single-draft speculative decoding.

**Why it might help here:** EAGLE-3's acceptance ceiling is partly a fundamental single-draft limitation. Hierarchical SD breaks that ceiling by allowing multiple draft levels, each conditioning on the previous level's accepted output. The ICLR 2026 Oral designation indicates strong reviewer consensus on correctness.

**Key papers:**
- "Hierarchical Speculative Decoding," OpenReview LaVrNaBNwM, ICLR 2026 Oral. [https://openreview.net/forum?id=LaVrNaBNwM] — Provably lossless; overcomes joint intractability; boosts expected accepted tokens.

**Implementation notes:**
- Requires at least two draft models: a small "level-1" draft and a medium "level-2" verifier/draft. Given the 24GB VRAM constraint on A10G, model size budget is tight. Check whether a 1B + 4B hierarchy fits within 24GB alongside the target model.
- The "joint intractability" fix applies when the draft models are independent; if EAGLE-3's draft heads are reused as level-1, the hierarchy may simplify.
- Lossless proof relies on rejection sampling at each level; verify the implementation does not introduce batch-size-dependent acceptance decisions.

**Suggested experiment:**
1. Profile VRAM usage with Gemma-4-E4B-it alone, then estimate headroom for a 1B-parameter level-1 draft model.
2. If VRAM allows: implement 2-level hierarchy with a small distilled draft as level-1 and Gemma-4-E4B-it as the verifier.
3. Greedy-identity gate first. Then measure TPS.

**Taste rubric:**
- Mode: Tier shift
- Mechanistic grounding: 3 — formal lossless proof; directly overcomes EAGLE-3's single-draft ceiling; but VRAM constraint is real
- Research-state value: 3 — either confirms multi-level draft breaks 500 or shows VRAM is the bottleneck
- Execution value: 2 — higher implementation complexity; VRAM constraint may disqualify without careful sizing

---

### A4. Ouroboros Phrase Pool Speculative Decoding
**What it is:** Builds a phrase candidate pool from the LLM's own verification process during inference, providing high-quality candidates to the draft model without separate training. 2.8× over standard speculative decoding, 1.9× over lookahead.

**Why it might help here:** The phrase pool improves draft quality by reusing already-verified token sequences. This directly addresses the private-α tax — previously rejected phrases are discarded; Ouroboros recycles verified phrases, improving expected acceptance length.

**Key papers:**
- "Ouroboros: Speculative Decoding with Large Model Enhanced Drafting," arXiv 2402.13720, ICML. [https://arxiv.org/abs/2402.13720] — Phrase candidate pool; 2.8× over SD; code at github.com/thunlp/Ouroboros.

**Implementation notes:**
- The phrase pool grows over the course of inference. For short sequences (128-token completion), the pool may not be large enough to provide meaningful benefit on the first few steps. The gain may be larger for longer sessions or repeated prompts.
- Code is available at github.com/thunlp/Ouroboros — examine whether the draft model integration is clean enough to swap in a Gemma draft.
- Verify that phrase pool lookups are deterministic (hash-based, not sampling-based) to maintain #192 compliance.

**Suggested experiment:**
1. Clone Ouroboros repo; adapt the phrase pool integration to the vLLM fork.
2. Measure TPS on the 128/128 benchmark with and without phrase pool enabled.
3. If the pool is too sparse at 128 tokens to show benefit, test at 512-token completions to confirm the mechanism is alive before investing further.

**Taste rubric:**
- Mode: Tier shift
- Mechanistic grounding: 3 — phrase recycling targets acceptance-rate directly; but 128-token completion may be too short to build a useful pool
- Research-state value: 3 — negative result at 128 tokens constrains the pool-warmup hypothesis usefully
- Execution value: 2 — code exists; 128-token limitation may cap benefit in this specific benchmark

---

## Focus Area B: Batch-Invariant / Deterministic Attention Kernels on Ampere (sm_86)

### B1. FlashInfer BI Kernel for sm_86
**What it is:** FlashInfer is a customizable attention engine for LLM inference serving. Unlike standard SDPA (which uses parallel reductions whose order varies with batch size), FlashInfer supports explicit segment-level accumulation strategies that can be made batch-invariant.

**Why it might help here:** The bf16 SDPA reduction is identified as the dominant bandwidth-bound cost (~34.9% of per-step time). Non-BI reduction means the current code either uses a slow BI fallback or takes on non-determinism. FlashInfer's customizable accumulation can provide BI semantics without the performance penalty — closing a potential 3–5 TPS gap.

**Key papers:**
- "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving," arXiv 2501.01005. [https://arxiv.org/abs/2501.01005] — Segmented accumulation; BI-compatible; benchmarked on Ampere and Hopper.
- "Bit-Exact AI Inference," arXiv 2606.00279. [https://arxiv.org/abs/2606.00279] — Framework for deterministic-but-non-invariant vs. batch-invariant inference; provides BI-without-penalty recipes.
- "DASH: Deterministic Attention Scheduling for High-throughput Reproducible LLM Training," OpenReview bMi5ssfPoM, ICLR 2026. [https://openreview.net/forum?id=bMi5ssfPoM] — Deterministic attention scheduling; may provide compatible BI kernel patterns.

**Implementation notes:**
- FlashInfer's `FlashInferAttention` kernel uses segmented softmax with a fixed accumulation order. The key parameter is the "segment layout" — for single-stream decode, this collapses to a single segment per request, which is inherently BI.
- On sm_86 (A10G), FlashInfer supports bf16 with full performance. The `batch_prefill_with_ragged_kvcache_return_lse` path is most relevant for decode-phase attention.
- The critical check: confirm that the vLLM fork's current attention kernel is using `torch.nn.functional.scaled_dot_product_attention` with `enable_math=True` fallback (which is not BI on multi-GPU or with certain batch sizes). If so, replacing with FlashInfer's segmented path gives BI + potential speedup.
- Compatibility with CUDA graph capture: FlashInfer supports CUDA graphs. Verify the segment descriptor tensors are captured correctly.
- The `nanomaoli/llm_reproducibility` GitHub monkey-patch approach provides a lighter-weight alternative if full FlashInfer integration is too invasive: it patches the SDPA call to force deterministic accumulation order.

**Suggested experiment:**
1. Profile the current vLLM fork: identify what percentage of per-step wall time is spent in SDPA vs. linear projections vs. other ops.
2. Swap in FlashInfer's segmented bf16 decode kernel for the attention pass.
3. Run greedy-identity check (should pass by construction for BI kernel with identical precision).
4. Measure TPS delta. Expected: 3–8 TPS gain (closes part of the 469→500 gap; not sufficient alone, but orthogonal to SD).

**Taste rubric:**
- Mode: Diagnostic + frontier refinement
- Mechanistic grounding: 3 — targets identified 34.9% bandwidth cost; FlashInfer BI semantics are well-characterized on sm_86
- Research-state value: 3 — quantifies the BI attention tax precisely; orthogonal to SD experiments
- Execution value: 4 — cheap swap; directly measures the BI attention overhead without confounds; stacks with any SD experiment

---

### B2. Triton FA-2 Deterministic Kernel (Ampere-native)
**What it is:** A Triton-based reimplementation of FlashAttention-2 for Ampere and later, with explicit control over reduction order, enabling batch-invariant bf16 attention without falling back to slower deterministic CUDA paths.

**Why it might help here:** The `flash-attention-triton` (GitHub egaoharu-kensei) project provides an Ampere-native Triton kernel. Triton's tile-level control means the reduction order can be made lexicographic (hence BI) without the overhead of torch's `use_deterministic_algorithms` global flag.

**Key papers:**
- flash-attention-triton, GitHub egaoharu-kensei. [https://github.com/egaoharu-kensei/flash-attention-triton] — Triton FA-2 for Ampere+; configurable reduction order.
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022. [https://arxiv.org/abs/2205.14135] — Foundation.
- "Flashlight: Compiler-Level Acceleration of Attention Variants," MLSys 2026, OpenReview lboOMA8XWr. [https://openreview.net/forum?id=lboOMA8XWr] — PyTorch compiler extensions to accelerate attention variants; relevant for custom kernel integration paths.

**Implementation notes:**
- The Triton kernel's tile size (BLOCK_M, BLOCK_N) is the main tunable. On A10G (sm_86, 32 SMs), a 64×64 or 128×64 tile typically saturates memory bandwidth for decode-phase attention.
- Compile with `triton.compile()` and cache the compiled kernel to avoid JIT overhead inside CUDA graphs.
- Reduction order fix: change the inner accumulation loop from `for j in range(0, N, BLOCK_N)` to iterate tiles in a fixed row-major order and ensure the final `lse` (log-sum-exp) accumulation uses the same order regardless of thread block scheduling. On Triton this is straightforward since tile launch order is explicit.
- Test: run the kernel with batch_size=1 and batch_size=2 on identical single sequences; assert output tensors are bit-exact.

**Suggested experiment:**
1. Benchmark current attention kernel TFLOPS/s and memory bandwidth utilization on A10G.
2. Integrate Triton FA-2 with fixed-order reduction.
3. Verify bit-exact output for batch_size ∈ {1, 2, 4}.
4. Measure TPS delta. Stack with SD experiment to verify orthogonal gains.

**Taste rubric:**
- Mode: Frontier refinement (BI kernel upgrade)
- Mechanistic grounding: 3 — Triton tile control provides BI by design; Ampere-native avoids Hopper-specific paths in FA-3
- Research-state value: 3 — directly tests whether BI attention overhead is the residual 30 TPS gap source
- Execution value: 3 — moderate integration; cheap profile first step before full integration

---

## Focus Area C: Lossless Throughput Levers (Closing the 469→500 Gap)

### C1. REST — Retrieval-Based Speculative Decoding
**What it is:** Retrieval-based SD that uses a datastore of token sequences (built from the inference corpus) to propose draft tokens without any draft model training. Greedy acceptance strategy makes it lossless by construction.

**Why it might help here:** REST requires zero draft-model fine-tuning. The datastore is built from domain-representative text. For a fixed benchmark prompt distribution (128-token prompts), the datastore can be pre-populated with high-coverage n-gram continuations, providing consistently high acceptance rates on in-distribution prompts.

**Key papers:**
- "REST: Retrieval-Based Speculative Decoding," arXiv 2311.08252, OpenReview jpZRI3dj2xn. [https://arxiv.org/abs/2311.08252] — No draft-model training; 1.62–2.36× speedup; greedy acceptance strategy lossless.

**Implementation notes:**
- Build the datastore from the benchmark prompt distribution or close proxies (common English text, code, etc.).
- The retrieval lookup must be fast enough not to add more latency than it saves. On single-stream A10G, the GPU is idle during the CPU-side retrieval lookup — this is a CPU-GPU synchronization cost. Benchmark retrieval latency vs. per-token generation time.
- For 128-token prompts, the retrieval context window should be tuned to match typical continuation lengths.
- Greedy acceptance: accepted tokens are those where the retrieved continuation matches what the target would have generated. This is lossless by the standard speculative decoding argument.
- REST is already integrated into some vLLM versions — check whether the vLLM fork has a REST plugin or skeleton.

**Suggested experiment:**
1. Build a 1M-token datastore from a Wikipedia + code sample matching the benchmark prompt style.
2. Enable REST in the vLLM fork. Measure average draft acceptance length on 100 benchmark prompts.
3. Greedy-identity gate. Measure TPS.
4. Ablation: datastore size (100K vs. 1M vs. 10M tokens) to find the point of diminishing returns.

**Taste rubric:**
- Mode: Tier shift (no training required)
- Mechanistic grounding: 3 — retrieval-based draft is lossless; acceptance rate depends on datastore coverage of the benchmark prompt distribution
- Research-state value: 4 — zero training cost means this can be run in hours; directly tests whether retrieval acceptance rates can close the TPS gap
- Execution value: 4 — highest information-per-compute ratio in this list; failure would falsify the "coverage is high enough" assumption

---

### C2. Goodput-Optimized γ Selection
**What it is:** Per-step selection of the optimal draft length γ (number of speculative tokens to propose) based on the current estimated acceptance rate, maximizing tokens accepted per unit time rather than using a fixed γ.

**Why it might help here:** Fixed γ is a common suboptimality in SD systems. When acceptance rate is high (common at the start of a completion or after deterministic prefixes), larger γ improves throughput. When acceptance rate is low (after rare tokens), smaller γ avoids wasted verification cost. A per-step adaptive γ, even a simple rule-based one, can improve effective throughput by 10–20% over fixed γ.

**Key papers:**
- "Optimal Speculative Decoding Draft Length via Goodput Maximization," arXiv 2406.14066. [https://arxiv.org/abs/2406.14066] — Per-step optimal γ; goodput maximization; closed-form solution given estimated acceptance rate.

**Implementation notes:**
- The closed-form optimal γ requires an estimate of the current acceptance rate α. A simple exponential moving average over the last k accepted/rejected decisions provides a good online estimate.
- The γ decision must be made before the draft forward pass; the acceptance estimate must not depend on the target model's output (that would be circular).
- Implementation is lightweight: add an α tracker and a γ = f(α̂) lookup table alongside the existing SD path. No model changes.
- This is orthogonal to all other SD experiments — can be combined with DySpec, Medusa, or REST.

**Suggested experiment:**
1. Implement an EMA-based α tracker over a window of the last 16 speculative steps.
2. Implement the goodput-optimal γ schedule from arXiv 2406.14066 (closed-form given α̂).
3. Measure TPS with fixed γ=4 vs. adaptive γ. Expect 5–15% TPS improvement on benchmark.
4. Combine with whichever SD variant provides the highest baseline TPS (most natural stacking point).

**Taste rubric:**
- Mode: Frontier refinement
- Mechanistic grounding: 4 — closed-form optimal γ given α; directly addresses fixed-γ suboptimality; orthogonal to draft model choice
- Research-state value: 3 — quantifies the fixed-γ tax; improves any SD system; stacks well
- Execution value: 4 — minimal code change; no retraining; directly stacks with any other SD experiment

---

### C3. SAM — Synergistic Augmentation for Speculative Decoding
**What it is:** SAM augments the draft model's proposals by combining retrieval-based candidates with the draft model output, creating a synergistic multi-source draft. Beats REST, Lookahead, and PIA; up to 3.0× speedup on reasoning models.

**Why it might help here:** If the draft model (e.g., Medusa heads) has moderate acceptance rates, SAM's retrieval augmentation can boost accepted-token length without additional compute on the critical path. The "synergistic" combination avoids double-counting candidates by choosing the highest-probability draft from either source.

**Key papers:**
- "SAM: Synergistic Augmentation for Speculative Decoding with a Retrieval Mechanism," arXiv 2509.04474. [https://arxiv.org/abs/2509.04474] — Beats REST/Lookahead/PIA; up to 3.0×; August 2025 publication.

**Implementation notes:**
- SAM requires both a draft model and a retrieval datastore. The combination step selects the better candidate per position. If the draft model acceptance rate is already >0.9 for the top-1 token, SAM may add little. The gain is largest when α_draft ≈ 0.5–0.7 and the retrieval datastore has high coverage.
- Lossless proof: SAM uses the same acceptance criterion as standard SD, applied to the best candidate from either source. This is lossless by the standard argument.
- Implementation complexity is moderate: requires both the draft model and REST datastore to be active simultaneously. VRAM cost is the sum of both.

**Suggested experiment:**
1. Establish a baseline with Medusa-1 alone on the benchmark.
2. Add REST datastore (from C1 experiment) as the retrieval source for SAM.
3. Implement candidate selection: for each speculative position, pick max(p_medusa[token], p_rest[token]) as the proposed token.
4. Greedy-identity gate. Measure TPS delta vs. Medusa-alone baseline.

**Taste rubric:**
- Mode: Frontier refinement (combining existing approaches)
- Mechanistic grounding: 3 — synergistic combination targets moderate-acceptance regime; lossless by construction
- Research-state value: 3 — tests whether multi-source drafting provides additive gain over either source alone
- Execution value: 2 — requires both draft model and datastore active simultaneously; higher VRAM/complexity cost than isolated experiments

---

### C4. Lossless Vocabulary Reduction
**What it is:** Identifies tokens that are never sampled by the target model (have effectively zero output probability for any input) and removes them from the lm_head matrix. The remaining computation is identical to the original model, so greedy output is unchanged.

**Why it might help here:** On Gemma-4-E4B-it, the vocabulary size is ~256K tokens (Gemma 4 uses a large sentencepiece vocabulary). The lm_head matrix-vector product is a significant per-step cost. Even a 10% vocabulary reduction would proportionally reduce lm_head cost. Given the model's specialized fine-tuning, many vocabulary entries may have near-zero probability mass.

**Key papers:**
- "Lossless Vocabulary Reduction for Autoregressive Language Models," OpenReview xAvqHtLVgz, ICLR 2026. [https://openreview.net/forum?id=xAvqHtLVgz] — Provably lossless; removes never-sampled tokens; reduces lm_head cost.

**Implementation notes:**
- Run a calibration pass over a representative corpus (10K–100K tokens) to estimate per-vocabulary-entry output probability. Identify tokens with max probability < threshold ε (e.g., ε = 1e-6).
- Prune the lm_head weight matrix rows and the corresponding embedding rows. Re-verify greedy output matches on calibration set and held-out set.
- The pruned model is smaller in memory — freeing VRAM for a larger KV cache or for a draft model.
- Critical: verify that special tokens (BOS, EOS, padding, chat template tokens) are not pruned regardless of their probability distribution on the calibration corpus.
- This is orthogonal to all attention and SD experiments — provides a constant per-step speedup multiplier.

**Suggested experiment:**
1. Profile lm_head cost as a fraction of per-step time on A10G.
2. Run calibration pass; identify vocabulary coverage at ε = 1e-6, 1e-5, 1e-4.
3. Prune at the most aggressive ε that maintains greedy-identity on held-out 1K prompts.
4. Measure TPS delta. Combine with whichever SD experiment is leading.

**Taste rubric:**
- Mode: Frontier refinement (lm_head cost reduction)
- Mechanistic grounding: 4 — lossless proof is straightforward; lm_head cost is measurable; Gemma's large vocab makes the target clear
- Research-state value: 3 — directly quantifies lm_head overhead; orthogonal to all other experiments
- Execution value: 3 — cheap calibration pass; no retraining; free to stack

---

### C5. LogitSpec — Next-Next Token Retrieval Speculation
**What it is:** Extends retrieval-based SD to propose not just the next token but the next-next token using logit predictions from the current step. 2.61× speedup; designed for vLLM-native integration.

**Why it might help here:** LogitSpec's "next-next token" speculation directly extends the speculative length beyond what pure retrieval can achieve, without a separate draft model. The vLLM-native integration path reduces implementation friction.

**Key papers:**
- "LogitSpec: Accelerating Speculative Decoding via Next-Next Token Speculation," arXiv 2507.01449, OpenReview 8TAIXl6GDM. [https://arxiv.org/abs/2507.01449] — 2.61×; next-next-token retrieval via logit prediction; vLLM-native integration.

**Implementation notes:**
- The "next-next" prediction uses the logit vector from the current step to predict what the most probable next-next token will be. This is then used to pre-populate the draft tree for the next step.
- Since this is logit-based (not sampling-based), it is deterministic given deterministic logit computation. BI compliance requires the same logit determinism as the main decode pass.
- Integration into vLLM is described as "easy" in the paper — check the OpenReview supplementary for code pointers.

**Suggested experiment:**
1. Integrate LogitSpec into the vLLM fork following the paper's vLLM integration notes.
2. Greedy-identity gate: verify the logit-based next-next speculation produces greedy-identical output.
3. Measure TPS. Compare vs. REST alone (C1) to determine whether next-next speculation adds meaningful gain over pure retrieval.

**Taste rubric:**
- Mode: Frontier refinement
- Mechanistic grounding: 3 — next-next logit prediction extends speculative length; vLLM integration path reduces friction
- Research-state value: 3 — tests whether logit-based next-next extends REST-style gains into longer speculation horizons
- Execution value: 3 — described as easy to integrate; relatively cheap to validate

---

## Cross-Cutting Notes

### CUDA Graph Compatibility
All speculative decoding experiments must verify CUDA graph capture compatibility. Variable-length speculative trees (DySpec) may require shape-bucketed CUDA graphs or disabling CUDA graph capture for the speculative path. Fixed-topology methods (Medusa, REST, LogitSpec) are more likely to be CUDA-graph-compatible.

### #192 Greedy-Identity Verification Protocol
Every experiment must pass a zero-tolerance greedy-identity check before any TPS measurement is reported:
1. Generate completions for 100 fixed prompts using the new method.
2. Generate reference completions with standard greedy AR decode.
3. Assert zero token mismatches across all positions in all 100 completions.
If any mismatch is found, the method is #192-non-compliant and must not be reported as a TPS result.

### VRAM Budget on A10G (24 GB)
Gemma-4-E4B-it in bf16 requires approximately 8 GB. Remaining ~16 GB is available for KV cache and draft model. This allows a 1–3B parameter draft model in bf16 (2–6 GB), leaving 10–14 GB for KV cache. Hierarchical SD with a second model tier may be tight — profile VRAM carefully before committing to a 2-model architecture.

### Stacking Priority
The most efficient path to >500 TPS is expected to be:
1. Pick the best single SD method (DySpec or Medusa+Traversal Verification) to cross ~500 TPS if acceptance rates are sufficient.
2. Add goodput-optimized γ selection as a free multiplier on top.
3. Add lossless vocab reduction as a constant speedup multiplier.
4. Add FlashInfer BI kernel to close the residual gap.

These four together are orthogonal and stackable — their gains should multiply rather than add.

---

## Experiment Decision Tree

```
START: Need strict-compliant >500 TPS

├── A. Can SD-based drafting cross 500?
│   ├── Try DySpec (A1) — dynamic tree, greedy temp=0 result
│   │   ├── DySpec > 500 TPS AND greedy-exact: DONE (stack goodput-γ + vocab-reduction for margin)
│   │   └── DySpec < 500 TPS:
│   │       ├── Try Medusa-1 + Traversal Verification (A2)
│   │       │   ├── Medusa > 500: DONE
│   │       │   └── Medusa < 500: Try Hydra sequential heads (A2 variant)
│   │       │       ├── Hydra > 500: DONE
│   │       │       └── Hydra < 500: Move to REST (C1) — lower training cost
│   │       └── (Parallel path) Try REST (C1) — no training
│   │           ├── REST > 500: DONE
│   │           └── REST < 500: Try SAM = REST + Medusa (C3)
│
├── B. Can BI attention kernel close remaining gap?
│   ├── Try FlashInfer BI kernel (B1) — measure TPS delta
│   │   ├── Delta > 30 TPS alone: DONE
│   │   └── Delta < 30 TPS: Stack with best SD result above
│   └── Try Triton FA-2 BI (B2) — if FlashInfer integration too invasive
│
├── C. Are lossless multipliers worth stacking?
│   ├── Goodput-optimized γ (C2) — free to stack with any SD method
│   ├── Lossless vocab reduction (C4) — constant per-step speedup
│   └── LogitSpec (C5) — if REST datastore is already built
│
└── ESCALATION (if no single method or 2-way stack crosses 500):
    ├── Hierarchical SD (A3) — requires careful VRAM sizing
    ├── Ouroboros phrase pool (A4) — if inference sessions are long enough to warm the pool
    └── SAM full combination (C3) — highest-complexity path
```

---

## References Summary

| # | Paper | arXiv/OpenReview | Year | Key Result |
|---|-------|-----------------|------|-----------|
| 1 | DySpec | OpenReview orr5uPZY28 | 2025 | 9.1× TPS greedy Llama2-70B |
| 2 | Medusa | arXiv 2401.10774 | 2024 | Lossless tree-attention SD |
| 3 | Traversal Verification | OpenReview 8nOMhDFpkU | 2025 | Leaf-to-root; consistently improves acceptance length |
| 4 | Hydra | arXiv 2402.05109 | 2024 | 1.31× over Medusa++; sequential draft heads |
| 5 | Hierarchical SD | OpenReview LaVrNaBNwM | 2026 | ICLR Oral; overcomes joint intractability |
| 6 | Ouroboros | arXiv 2402.13720 | 2024 | 2.8× over SD; phrase pool; github.com/thunlp/Ouroboros |
| 7 | REST | arXiv 2311.08252 | 2024 | No training; 1.62–2.36×; greedy-exact |
| 8 | Goodput-optimal γ | arXiv 2406.14066 | 2024 | Closed-form optimal draft length |
| 9 | SAM | arXiv 2509.04474 | 2025 | 3.0× on reasoning models |
| 10 | LogitSpec | arXiv 2507.01449 | 2025 | 2.61×; vLLM-native |
| 11 | Lossless Vocab Reduction | OpenReview xAvqHtLVgz | 2026 | ICLR; provably lossless lm_head reduction |
| 12 | FlashInfer | arXiv 2501.01005 | 2025 | Customizable BI-compatible attention engine |
| 13 | Bit-Exact AI Inference | arXiv 2606.00279 | 2026 | BI without performance penalty recipes |
| 14 | DASH | OpenReview bMi5ssfPoM | 2026 | ICLR; deterministic attention scheduling |
| 15 | PPD | arXiv 2405.18628 | 2024 | Hardware-aware sparse tree; 28% better acceptance |
| 16 | Flashlight | OpenReview lboOMA8XWr | 2026 | MLSys; compiler extensions for attention variants |
| 17 | NanoFlow | arXiv 2408.12757 | 2024 | Intra-device parallelism for LLM serving |
| 18 | CAS-Spec | OpenReview m0bR0sxhfL | 2025 | NeurIPS; on-the-fly cascade adaptive self-spec |
| 19 | LayerSkip | arXiv 2404.16710 | 2024 | Self-speculative; early-exit lossless; requires training |
| 20 | Global Resolution Optimal Multi-Draft | OpenReview gpsczXOsHn | 2026 | ICLR Oral; optimal multi-draft via convex optimization |
