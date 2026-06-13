# Research Ideas — 2026-06-13 20:15
## Context
Hardware: single NVIDIA A10G, 23 GB VRAM, Ampere sm_86
Runtime: vLLM 0.22.0, transformers 5.9.0
Model: google/gemma-4-E4B-it (~4B multimodal)
Regime: memory-bandwidth-bound at conc=1 (~92% weight-GEMM, ~2.6% attn, ~0.2% sampling)

Current public frontier: ~459 VALID TPS
Stack: int4 W4A16 Marlin + untied lm_head + g128 + lmhead12k prune + fa2sw + MTP/EAGLE spec decode + one-CUDA-graph + KV-precache warmup + acceptance-history dynamic-K

Dead ends (must NOT re-propose):
- Sub-4-bit weights (AWQ/GPTQ/AQLM/QuIP#/2:4-sparse-Marlin/NVFP4)
- fp8 KV cache
- n-gram/prompt-lookup spec decode
- attn-backend swaps
- body channel-wise quant
- tree-causal attn mask for sparse-tree verify
- verify-rollback for greedy-valid spec
- VLLM_BATCH_INVARIANT=1

Currently assigned (do not duplicate):
- denken: accepthist dynamic-K cost-model
- wirbel: fa2sw 3D split-KV dispatch guard patch
- land + fern: serve-faithful step-0 fix + private-stable drafter

---

## Category A — Fresh angle for stark (int4-numerics / verify-path specialist)

### A1 [RANK 1] — QSpec-Style Draft-Precision Switching (W8A8 draft → W4A16 verify)

**Mechanism**: QSpec (ICLR 2025 submission, OpenReview NnExMNiTHw) shows that running draft steps with joint activation-weight low-precision (W8A8) and verifying with weight-only W4A16 achieves up to 1.80× speedup. The key insight is that draft steps dominate latency when acceptance rate is high, so making draft GEMMs faster (W8A8 CUTLASS has higher theoretical FLOPS/s on Ampere than W4A16 Marlin due to tensor-core alignment) cuts per-draft-token cost even if acceptance rate is unchanged. The stack already uses W4A16 Marlin for verify; the only change is adding a W8A8 kernel path for draft forward passes on the same weights (quantize activations online at draft-time using per-token dynamic scaling, keep the W4 weight but dequantize to bf16 then requantize to int8 activation — or keep a parallel W8 weight copy). Because draft and verify use the same underlying model, the result is greedy-exact: the verify tokens are still produced by the full W4A16 path.

**TPS lever**: Greedy-exact. Cuts per-draft-token latency. If the current pipeline spends ~30% of step time on draft forward passes (conservative for a K=4 drafter with ~3 accepted tokens), a 1.4× draft speedup yields ~12% end-to-end TPS gain.

**Cheapest de-risk test**: Profile one forward pass with torch.compile + W8A8 path vs. W4A16 Marlin draft pass on a single gemma-4B layer on the A10G. Measure wall time. If W8A8 is not faster than W4A16 Marlin on Ampere (Marlin is highly hand-tuned), this approach has no ceiling and can be ruled out in <15 min.

**Key risk**: Marlin's W4A16 kernel is extremely hand-tuned for Ampere bandwidth. W8A8 CUTLASS may actually be slower per token on sm_86 for the matrix shapes in gemma-4B because the arithmetic gain is eaten by activation quantization overhead. This is the single most important unknown.

**References**:
- QSpec: "Speculative Decoding with Low-precision Draft Model" (ICLR 2025 submission), OpenReview https://openreview.net/forum?id=NnExMNiTHw
- Marlin: "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models" (MLSys 2024), https://arxiv.org/abs/2408.11743

---

### A2 [RANK 2] — Token Recycling (Training-Free Adjacency-Matrix Drafting)

**Mechanism**: Token Recycling (arXiv 2408.08696) builds a lightweight adjacency matrix from the model's own output token sequences at inference time (no training, <2 MB storage). At each draft step, it performs BFS over the adjacency matrix to propose a branching token tree, then uses standard tree-verification. The adjacency matrix is updated online from each step's accepted tokens, so it naturally adapts to the 128-prompt distribution after warmup. The key property: it is a drop-in replacement for the draft model's token proposal step, independent of how verification is done. It achieves ~88.65% of EAGLE-2's speedup with zero training.

**TPS lever**: Greedy-exact (tree verification with standard lm_head). Expected TPS gain over no-spec baseline: 1.6–1.8×. Relative to the current MTP/EAGLE stack, it replaces or augments the draft proposal. The gain over the existing stack depends on whether MTP acceptance rate is already near its ceiling; if accepted tokens per step is stuck at ~2.8, Token Recycling's BFS tree could increase it to ~3.2 at low overhead.

**Cheapest de-risk test**: Build the adjacency matrix on 16 of the 128 prompts (offline), then measure accepted-tokens/step with Token Recycling tree vs. current MTP single-sequence draft. This is a pure Python probe (<1 hour). If accepted tokens/step does not improve, stop.

**Key risk**: The 128-prompt set may be too small and too homogeneous for the adjacency matrix to be meaningfully populated during the evaluation run itself. If the distribution is narrow, accepted tokens from EAGLE/MTP may already be near the acceptance ceiling for this distribution, leaving nothing for Token Recycling to add.

**References**:
- Token Recycling: "Token Recycling: Towards Simultaneous Decoding and Drafting in Speculative Decoding for Efficient LLM Inference" (arXiv 2408.08696), https://arxiv.org/abs/2408.08696
- EAGLE-2: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (EMNLP 2024), https://arxiv.org/abs/2406.16858

---

## Category B — Highest-ceiling idea to push past ~459 VALID TPS

### B1 [RANK 1] — HASS Top-K Harmonized Distillation on Existing MTP Drafter

**Mechanism**: HASS (Harmonized Speculative Sampling, arXiv 2408.15766) adds two objectives on top of EAGLE-style auto-regressive feature prediction: (1) a Top-K token distribution alignment loss between draft and target model's output distributions, and (2) context representation alignment between draft hidden states and target hidden states at the same positions. The result is 8–20% higher acceptance rate than EAGLE-2. The key new finding: the Top-K distribution alignment loss alone (without full context alignment) is responsible for the majority of the improvement, and is significantly cheaper to implement — it only requires storing the target model's top-K logits during distillation training, not full hidden-state trajectories.

**Why this is the highest ceiling**: The current public frontier stack's bottleneck, after the serve-faithful step-0 fix is applied by land+fern, will be raw acceptance rate. Each additional accepted token per step directly multiplies TPS. An 8–20% acceptance rate improvement on top of the fixed serve-faithful drafter translates to ~8–20% TPS gain — the single largest remaining lever in the pipeline short of a fundamentally different architecture.

**TPS lever**: Quality-risked (it's a fine-tune of the drafter, so PPL must be re-checked). Expected TPS gain: 8–20% over the land+fern serve-faithful result, if Top-K alignment loss is applied during drafter training. In absolute terms: if land+fern reaches ~480 TPS, HASS Top-K pushes to ~520–576 TPS.

**Cheapest de-risk test**: Run the Top-K distribution alignment loss as a frozen-target distillation on 1k samples from the 128-prompt distribution for 1 epoch. Measure accepted tokens/step before and after. If accepted tokens/step does not improve by ≥3%, stop. This probe is ~30 min on one A10G.

**Key risk**: HASS was developed with a separate smaller draft model (not MTP heads). Applying Top-K loss to MTP heads requires the target model's top-K logits at each position, which means one target-model forward pass per training sample. This doubles training cost but does not block correctness. The risk is that MTP's shared-weight structure is not well-suited to Top-K alignment in the same way a fully-independent draft model is.

**References**:
- HASS: "HASS: Harmonized Speculative Sampling" (arXiv 2408.15766), https://arxiv.org/abs/2408.15766
- FastMTP: "FastMTP: Efficient Language Model Inference through Position-Shared MTP Head" (arXiv 2509.18362), https://arxiv.org/abs/2509.18362 — for contrast on lightweight MTP training tricks

---

### B2 [RANK 2] — FastMTP Self-Distillation on Existing MTP Head

**Mechanism**: FastMTP (arXiv 2509.18362) trains a single MTP head with position-shared weights using self-distillation from the target model's own predictions. Unlike standard MTP training which uses next-token cross-entropy, FastMTP uses the target model's output distribution as soft labels — essentially a form of knowledge distillation at the MTP head level. It achieves 2.03× speedup (vs. EAGLE/HASS's ~3–4×) but requires only one small head, and the training is fast (~few hours on one GPU for a 7B model).

**Why rank below B1**: The speedup ceiling is lower, and the existing pipeline already has MTP heads. FastMTP's gain comes from better acceptance rate on the existing head structure rather than a new architecture. It is more of a training recipe improvement than a ceiling-shift.

**TPS lever**: Quality-risked (drafter fine-tune). Expected TPS gain: 5–12% over the current MTP drafter if self-distillation improves acceptance rate. Compatible with the serve-faithful fix from land+fern.

**Cheapest de-risk test**: Fine-tune the existing MTP head for 500 steps with FastMTP's self-distillation loss on 2k samples from the prompt distribution. Measure accepted tokens/step before and after.

**Key risk**: If the current MTP head was already trained with a form of distribution matching, FastMTP may offer minimal improvement. Also: 2.03× is measured on a baseline without the full frontier stack (no CUDA graph, no dynamic-K), so the incremental gain above the current 459 TPS stack may be smaller.

**References**:
- FastMTP: arXiv 2509.18362, https://arxiv.org/abs/2509.18362
- DeepSeek-V3: "DeepSeek-V3 Technical Report" (arXiv 2412.19437) — MTP head training context, https://arxiv.org/abs/2412.19437

---

## Category C — Contrarian / "Old idea, new application" for bandwidth-bound int4 decode

### C1 [RANK 1] — Jacobi / Lookahead Parallel Decoding as Drafter Fallback

**Mechanism**: Lookahead Decoding (Fu et al., ICML 2024, arXiv 2402.08559) uses Jacobi iteration to generate candidate tokens in parallel without a draft model. The core idea: run L steps of Jacobi iteration (each step a full forward pass with a guess for future positions), then verify the longest consistent n-gram suffix. This is training-free, lossless (greedy-exact), and composes with any verify path. On bandwidth-bound hardware at batch=1, it achieves up to 1.8× TPS. The "old idea" connection: Jacobi iteration for fixed-point problems in numerical methods dates to 1845 — applied here to the autoregressive fixed point.

**Why contrarian**: The current pipeline invests heavily in a learned drafter (MTP/EAGLE). Lookahead needs no drafter at all. In the specific regime where the existing drafter has poor acceptance rate on some prompt suffixes (e.g., code completions, long factual sequences), Lookahead provides a graceful floor that doesn't rely on drafter quality. It also composes with the CUDA-graph infrastructure already in place.

**TPS lever**: Greedy-exact. Expected gain: 1.3–1.8× over base, which is below the existing spec-decode stack (~3–4×). The value is as a fallback on low-acceptance prompts or as a verification of the theoretical ceiling under a no-drafter assumption.

**Cheapest de-risk test**: Run the open-source Lookahead implementation (https://github.com/hao-ai-lab/LookaheadDecoding) on the 128-prompt set with the quantized gemma-4B, measure TPS and accepted n-gram rate. This is a pure inference probe, no training. If TPS is below 300, it confirms the drafter path is the right investment.

**Key risk**: Lookahead's 1.8× is measured at typical text generation lengths and distributions. On short-output-length prompts (output_len=512 but with high acceptance in the existing stack), Lookahead may underperform because it wastes Jacobi iterations on already-well-predicted positions.

**References**:
- Lookahead Decoding: "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (ICML 2024), arXiv 2402.08559, https://arxiv.org/abs/2402.08559
- Implementation: https://github.com/hao-ai-lab/LookaheadDecoding

---

### C2 [RANK 2] — Online Distribution Adaptation via Token-Frequency Biasing (Prompt-Distribution-Aware Drafter)

**Mechanism**: The 128-prompt set is fixed and public. The token distribution of the drafter's proposals can be biased toward the specific vocabulary sub-distribution of those prompts by (a) collecting token unigram frequencies from all 128 prompts offline and (b) adding a small learned bias vector to the drafter's logits before sampling. This is a version of OmniDraft's (NeurIPS 2025) "online adaptation" idea reduced to its simplest form: a single bias vector updated by gradient descent for 100 steps on the 128-prompt outputs. The bias vector is <1 KB and does not change the verify path at all.

**Why contrarian**: The standard spec-decode literature focuses on improving acceptance rate across a broad distribution. Here, the evaluation distribution is narrow and known in advance. A prompt-specific drafter bias exploits this fixed evaluation contract in a way that would be considered "overfitting" in a general setting but is entirely legitimate for a fixed benchmark.

**TPS lever**: Greedy-exact if the bias vector is small enough that it doesn't shift the accepted token distribution away from the target model's greedy choices. The bias shifts draft proposals toward high-frequency tokens in the 128 prompts, increasing draft-target agreement without changing target output.

**Cheapest de-risk test**: Compute the token unigram distribution of all 128-prompt reference outputs. Add this as a fixed (non-learned) logit bias to the drafter. Measure acceptance rate change in 5 minutes. If acceptance rate increases by ≥2 pp, the gradient-descent version is worth training.

**Key risk**: If the 128 prompts are sufficiently diverse, the unigram bias will be near-uniform and add nothing. Also: the benchmark may detect and penalize distribution-specific tuning if PPL is measured on a held-out set.

**References**:
- OmniDraft: "OmniDraft: Adaptive Online Draft Generation for Efficient Speculative Decoding" (NeurIPS 2025 workshop)
- SpecDec++: "SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths" (OpenReview NnExMNiTHw), https://openreview.net/forum?id=NnExMNiTHw — optimal K-selection framework applicable to bias-tuned drafter

---

### C3 [RANK 3] — Sequoia DP-Optimal Tree Structure for Existing MTP Drafter

**Mechanism**: Sequoia (arXiv 2402.12374) formulates spec-decode tree construction as a dynamic programming problem, finding the tree topology that maximizes expected accepted tokens per verification step under hardware-specific cost models. The key insight is that the optimal tree shape is hardware-dependent: on bandwidth-bound A10G, shallow-wide trees (more candidates at depth 1) may be better than deep-narrow trees, and the optimal shape changes with acceptance rate. Sequoia achieves 4.04× speedup on A100, and its DP solver is a ~200-line Python function that can be run offline to generate a static optimal tree for the 128-prompt acceptance statistics.

**Why contrarian**: The "old idea" is dynamic programming on trees, dating to Bellman 1957. Applied to speculative decoding, it replaces heuristic tree designs with a principled optimum. The dynamic-K cost-model that denken is working on is a 1D version of this; Sequoia is the full 2D (depth × width) generalization.

**TPS lever**: Greedy-exact. If the current tree topology is suboptimal for the A10G's specific memory bandwidth vs. compute trade-off, the DP-optimal tree could yield 5–15% more accepted tokens per step at identical verify cost.

**Cheapest de-risk test**: Run Sequoia's offline DP solver (code at https://github.com/Infini-AI-Lab/Sequoia) on the acceptance rate statistics logged from the current MTP stack (denken's accepthist data). Generate the optimal tree shape. Compare expected accepted tokens/step analytically. If the theoretical gain is <3%, skip implementation.

**Key risk**: Sequoia's DP solver requires per-depth acceptance rate statistics from the specific model + hardware combination. These may not be available from the current logging. Also: the current dynamic-K work (denken) may already be covering most of the gain that Sequoia would find.

**References**:
- Sequoia: "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding" (arXiv 2402.12374), https://arxiv.org/abs/2402.12374
- SpecInfer: "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification" (ASPLOS 2024), https://arxiv.org/abs/2305.09781

---

## Summary Ranking

| Rank | ID | Category | Mechanism | TPS lever | Greedy-exact? | Expected gain | Cheapest test |
|------|-----|----------|-----------|-----------|---------------|---------------|---------------|
| 1 | B1 | Highest-ceiling | HASS Top-K distillation on MTP drafter | Acceptance rate +8–20% | No (drafter fine-tune) | +8–20% TPS over land+fern result | 500-step Top-K distillation probe |
| 2 | A1 | stark fresh | W8A8 draft + W4A16 Marlin verify | Faster draft GEMM | Yes | +5–15% TPS | Profile one layer W8A8 vs W4A16 |
| 3 | A2 | stark fresh | Token Recycling adjacency BFS tree | Higher accepted tokens/step | Yes | +5–10% TPS over current tree | Build adj matrix on 16 prompts |
| 4 | B2 | Highest-ceiling | FastMTP self-distillation on MTP head | Better acceptance rate | No (drafter fine-tune) | +5–12% TPS | 500-step self-distillation probe |
| 5 | C3 | Contrarian | Sequoia DP-optimal tree for A10G | Optimal tree topology | Yes | +3–15% TPS | Run DP solver on accepthist data |
| 6 | C2 | Contrarian | Token-frequency bias on drafter logits | Distribution-specific acceptance | Yes | +2–8% TPS | Add static unigram bias, measure |
| 7 | C1 | Contrarian | Lookahead / Jacobi parallel decode | Training-free fallback | Yes | 1.3–1.8× over base only | Run LookaheadDecoding on 128 prompts |

---

## Assignment Recommendation for stark

**Primary assignment**: A1 (W8A8 draft-precision switching) — directly in stark's int4-numerics/verify-path domain, training-free, greedy-exact, cheap to de-risk. The 15-minute profiling probe should gate the full implementation.

**Secondary assignment if A1 fails the profiling gate**: A2 (Token Recycling) — also training-free, greedy-exact, no model changes, purely algorithmic. The adjacency matrix approach is the simplest possible "structural change to the draft proposal" with a concrete prior result (88.65% of EAGLE-2).

**For the next post-land+fern round**: B1 (HASS Top-K distillation) — assign a student who has the serve-faithful fixed drafter checkpoint as a starting point, since HASS needs a working base drafter to improve acceptance rate.
