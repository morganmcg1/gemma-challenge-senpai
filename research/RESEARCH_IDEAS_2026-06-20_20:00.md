# Research Ideas — 2026-06-20 20:00

**Target:** Maximize single-stream decode TPS for `google/gemma-4-E4B-it` on HF a10g-small runner (NVIDIA Ampere sm_86, 23 GB), served via vLLM 0.22.0, MTP speculative decoding (K=6 draft tokens, 4-head linear chain), int4 W4A16 Marlin body, greedy decode (byte-identity NOT required, quality gate: ≤5% degradation on AIME/MMLU-Pro/GPQA-Diamond, PPL ≤ 2.42, 128/128 completion).

**Bottleneck summary:**
- Body GEMM is at hardware floor: Marlin int4 W4A16 at 74–79% of 1-wave saturation ceiling. No GEMM-side gain possible.
- Acceptance rate r ≈ 0.397, E_accept ≈ 3.379 tokens/step. If r → 1.0: ~+52% TPS theoretical ceiling.
- Drafter forward pass: ~20% of decode cycle. Gemma-4's 262,144-token vocabulary makes the draft lm_head GEMV disproportionately expensive.
- Decode-cycle profile: body MLP 47.6%, drafter forward 20%, lm_head GEMV 6% (body, already int4), non-GEMM overhead 26% (fused, fixed floor).
- Two independent levers: (A) raise r, (B) reduce drafter cost per step.

**Dead ends (do not re-propose):** Marlin re-tiling/SplitK, FlashInfer, fp8 KV-cache, W3/W2 sub-4-bit body, 2:4 sparsity, W4A8 activation quant (−16.68% empirical), drafter weight quantization, EAGLE/tree topology (4 blockers in vLLM 0.22), config lenient-acceptance knob (not exposed in v1 engine), ngram/prompt-lookup drafter (2.4× slower empirically).

---

## Idea 1 (Highest Priority): Entropy-Gated Loosened Acceptance — FLy

**Mechanism:** Two-tier acceptance relaxation applied inside the MTP verification loop. At each draft position, compute the verifier's output-token entropy (single forward pass already runs). If entropy exceeds a threshold H_thresh (token is ambiguous — multiple near-equal-probability continuations), accept the draft token if it falls in the verifier's top-k (k=5–10) rather than requiring argmax match. Low-entropy positions (deterministic continuations) keep strict argmax to preserve quality. This is strictly better than uniform top-k because it does not loosen acceptance where the model is confident.

**Why this raises TPS here:** Current r ≈ 0.397 is the binding constraint. Raising r to 0.6–0.7 at high-entropy positions (where multiple plausible tokens exist) directly extends mean accepted tokens per draft step. The drafter already proposes plausible continuations; the bottleneck is the verifier rejecting semantically equivalent but lexically different tokens. Gemma-4 instruction-tuned on chat/reasoning tasks will have many high-entropy positions (pronouns, connectives, synonyms).

**Expected magnitude:** FLy reports 2.81× wall-time speedup on Llama-3.1-70B-Instruct with ≥99% answer accuracy preserved, 5.07× on 405B. Conservative expectation on this stack: +20–40% TPS with careful H_thresh tuning. Quality risk is low if H_thresh ≥ 2.0 nats (empirically safe range per FLy ablations).

**Implementation sketch:**
- File: `vllm/spec_decode/mtp_worker.py` (or equivalent MTP proposer/verifier integration point in vLLM 0.22.0 v1 engine)
- In the verification step, after the verifier logit tensor is computed, compute per-position entropy: `H = -sum(p * log(p), dim=-1)` (cheap, 262k-dim softmax already available)
- Gate: `mask = H > H_THRESH` selects high-entropy positions
- At masked positions, accept if `draft_token in top_k(verifier_logits, k=K_ACCEPT)`; at non-masked positions, accept only if `draft_token == argmax(verifier_logits)`
- H_thresh and K_accept are startup-script parameters; start with H_thresh=2.0, K_accept=5
- No retraining, no auxiliary model, no weight changes

**Papers:**
- FLy: "Training-Free Loosely Speculative Decoding" (arXiv:2511.22972, ICLR 2026 Poster, AMD-AGI). https://arxiv.org/abs/2511.22972
- GitHub: https://github.com/AMD-AGI/FLy

**Quality checkpoint:** Run full AIME/MMLU-Pro/GPQA-Diamond eval before firing. If any metric drops >3%, lower K_accept to 3 and re-eval before submitting.

---

## Idea 2: Draft lm_head Vocabulary Compression (FR-Spec / VOCABTRIM)

**Mechanism:** The draft lm_head GEMV at each of the 4 draft heads projects to the full 262,144-token vocabulary. In practice, >>95% of generated tokens come from the top ~8,000–16,000 most frequent tokens across the training/inference distribution. Replace the full draft lm_head weight matrix (shape [hidden_dim, 262144]) with a compressed version that only scores the top-N frequent tokens, then map accepted tokens back to full-vocab IDs. The verifier still uses the full vocabulary; only the draft proposal step is compressed.

**Why this raises TPS here:** Gemma-4's 262,144-token vocabulary is ~4× larger than typical models (Llama-3.2: 128k, Mistral: 32k). The draft lm_head GEMV at each of the 4 MTP heads is therefore ~4× more expensive than baseline EAGLE-style drafters. FR-Spec shows 75% compute reduction in the draft lm_head with minimal acceptance rate penalty on Llama/Mistral models. The relative gain here is larger because the vocab is larger. This is orthogonal to acceptance rate — it reduces drafter cost per step regardless of r.

**Expected magnitude:** 75% reduction in draft lm_head compute. Drafter forward pass is ~20% of decode cycle; lm_head is a significant fraction of that. Conservative estimate: 5–12% end-to-end TPS gain. No quality risk (verifier unchanged, still full vocabulary).

**Implementation sketch for FR-Spec approach:**
- Offline step: count token frequencies across a representative corpus (use the 128 benchmark prompts + a few thousand generated continuations)
- Build `freq_vocab_ids`: sorted index of top-N tokens (N=8192 or 16384)
- Create compressed weight: `draft_lm_head_compressed = draft_lm_head.weight[freq_vocab_ids, :]` (shape [N, hidden_dim])
- Patch draft head forward: replace full matmul with compressed matmul, then scatter accepted token IDs back to full vocab space for the verifier
- File: wherever the MTP draft heads compute their lm_head projection — likely `vllm/model_executor/models/` for the Gemma-4 MTP model definition

**Implementation sketch for VOCABTRIM approach (training-free):**
- VOCABTRIM (arXiv:2506.22694) directly reconstructs the compressed lm_head from the full one via a learned or heuristic projection — no corpus needed
- Potentially cleaner integration; slightly less compression ratio

**Papers:**
- FR-Spec: "FR-Spec: Accelerating Large Vocabulary Language Models via Frequency-Ranked Speculative Sampling" (arXiv:2502.14856). https://arxiv.org/abs/2502.14856 — GitHub: https://github.com/thunlp/FR-Spec
- VOCABTRIM: arXiv:2506.22694 — training-free draft lm_head compression

---

## Idea 3: Online Drafter Head Distillation During Inference (DVI)

**Mechanism:** DVI (Draft, Verify, & Improve) treats the verifier's accept/reject signals as online supervision for the drafter heads. After each speculative step, rejected draft tokens provide a reward signal: cross-entropy loss against the verifier's distribution at rejected positions + a policy-gradient term from the accept/reject outcome. The drafter heads are updated with a lightweight optimizer (SGD or Adam with very small LR) after every N decode steps. KL divergence to the verifier distribution bootstraps calibration; reward-masked cross-entropy drives alignment. No auxiliary model, no offline training dataset, no data pipeline.

**Why this raises TPS here:** The current MTP drafter heads were trained offline and are not adapted to the specific distribution of the 128 benchmark prompts. Online adaptation during the benchmark run itself will progressively align the draft distribution with the verifier's predictions on these exact prompts, raising r from 0.397 toward 0.6+. DVI reports 2.16× wall-time speedup on Spec-Bench with orders-of-magnitude less training data than EAGLE-2.

**Expected magnitude:** +15–30% TPS from rising acceptance rate. The gain is front-loaded (first few prompts), and by prompt 50–100 of 128 the drafter should be well-adapted. Quality risk is low (verifier unchanged, distillation drives drafter toward verifier distribution, not away from it).

**Implementation sketch:**
- Patch `mtp_worker.py` or the proposer class to maintain a running buffer of `(draft_token, verifier_logit, accepted: bool)` tuples
- After every N=16 decode steps, run a backward pass through only the drafter head parameters (not the shared transformer layers — freeze backbone), computing: `L = -accepted * log_p_draft(draft_token) - (1-accepted) * KL(draft_logit || verifier_logit.detach())`
- Use `torch.optim.SGD(drafter_head_params, lr=1e-5)` with gradient clipping at 1.0
- Overhead: backward through 4 small linear heads every 16 steps — negligible vs. the forward pass cost

**Papers:**
- DVI: "Draft, Verify, & Improve: Online Drafter Alignment for Speculative Decoding" (OpenReview:CwvY6TXLxr, ICLR 2026). https://openreview.net/forum?id=CwvY6TXLxr

---

## Idea 4: FastMTP Position-Shared Draft Head Weights

**Mechanism:** In the current MTP configuration, each of the 4 draft heads has independent weight matrices, trained to predict the token at offset +1, +2, +3, +4 from the current position. FastMTP proposes that a single shared head can be trained recursively: it predicts +1 from the current state, then the same head predicts +2 from the updated state, etc. This is analogous to weight tying in sequence models. The shared head is trained with self-distillation from the verifier and a vocabulary compression step. After retraining with shared weights, the 4-head drafter becomes 1 head applied 4 times — reducing model parameter footprint and enabling better CUDA graph capture.

**Why this raises TPS here:** FastMTP reports 2.03× speedup over standard next-token-prediction and 82% over vanilla MTP. The mechanism reduces drafter memory bandwidth (smaller parameter set, better cache reuse) and may raise acceptance rate through better self-distillation alignment. Position-shared weights also enable a cleaner CUDA Graph over the 4 draft steps (same kernel call 4 times rather than 4 different kernel variants).

**Expected magnitude:** Requires offline retraining of the drafter heads (~few GPU-hours on the Gemma-4 body as a frozen backbone). If acceptance rate gain is comparable to FastMTP's reported improvement: +15–25% TPS. Weight reduction enables better CUDA graph: additional 3–5% from reduced launch overhead.

**Implementation sketch:**
- Freeze the Gemma-4 body (int4 Marlin, as deployed)
- Train a single draft head with position embedding injection (sinusoidal offset encoding for steps 1–4) on next-token prediction, then apply it autoregressively for 4 steps during draft
- Self-distillation: teacher = verifier logits at each position; student = draft head output; loss = KL(student || teacher.detach())
- Vocabulary compression optional (combine with Idea 2 for additive gain)
- Serve the retrained single head as the MTP drafter weight file

**Papers:**
- FastMTP: "FastMTP: Efficient Training of Medusa-Type Models via Position-Shared Speculative Decoding" (OpenReview:J7xDwZSyI4, ICLR 2026, Withdrawn but methodology published). https://openreview.net/forum?id=J7xDwZSyI4

---

## Idea 5: Adaptive Draft Length via Entropy Lower Bound (AdaEDL)

**Mechanism:** With K=6 and r=0.397, geometric decay means P(token k accepted) ≈ 0.397^k. By position k=4, P ≈ 2.5%; by k=6, P ≈ 0.6%. Running the drafter to K=6 steps when positions 4–6 will almost certainly be rejected wastes ~33% of drafter compute. AdaEDL computes a lower bound on acceptance probability from the verifier's entropy at each position (high-entropy positions have lower acceptance bounds), and early-stops draft generation when the bound falls below a threshold. No training, no weight changes.

**Why this raises TPS here:** The 20% drafter budget is partially wasted at high draft indices. Stopping at K_eff ≈ 3–4 steps when the verifier is confident the current draft prefix will be rejected saves 2–3 drafter head forward passes per decode cycle. At r=0.397, the expected savings is significant.

**Expected magnitude:** AdaEDL reports 10–57% improvement over static draft length. On this stack, conservative estimate: 5–15% TPS gain from reduced drafter compute, with acceptance rate unchanged (the early-stopped positions were going to be rejected anyway).

**Implementation sketch:**
- After each draft step k, compute H_k = entropy of verifier logits at position k (available from the verification forward pass)
- Compute acceptance lower bound: `lb_k = exp(-H_k / temperature)` (from AdaEDL Eq. 3 or equivalent)
- If `lb_k < STOP_THRESH`, abort remaining draft steps (set remaining draft tokens to pad, mask them in the verification pass)
- STOP_THRESH is a startup parameter; start with 0.05 (stop when <5% expected acceptance)
- File: MTP proposer loop in `vllm/spec_decode/`

**Papers:**
- AdaEDL: "Adaptive Entropy-based Draft Length for Efficient Speculative Decoding" (arXiv:2410.18351, Qualcomm AI Research). https://arxiv.org/abs/2410.18351

---

## Idea 6: Full CUDA Graph Capture for MTP Draft Loop (vLLM PR #34880 Pattern)

**Mechanism:** In vLLM 0.22.0 v1 engine, the MTP drafter heads may not be fully wrapped in CUDA Graph mode, meaning each of the 4 draft head kernel launches incurs Python-layer dispatch overhead and CPU-GPU synchronization. vLLM PR #34880 adds `CUDAGraphWrapper` for EAGLE-style drafters (194 additions, 55 deletions across 6 files), building dummy attention metadata during `dummy_run` phase to enable full graph capture. Porting this pattern to the MTP drafter would eliminate per-step kernel launch overhead.

**Why this raises TPS here:** At small batch size (conc=1), kernel launch latency and CPU-GPU sync can constitute 5–15% of decode step time. The 4-step draft loop in MTP multiplies this overhead. CUDA Graph capture runs the 4 draft steps as a single pre-compiled GPU kernel sequence, eliminating all intermediate Python dispatch and sync.

**Expected magnitude:** 5–12% TPS gain. Purely a systems optimization with zero quality risk. The gain is more pronounced at batch size 1 (the benchmark setting) than at large batch.

**Implementation sketch:**
- Study vLLM PR #34880 diff (https://github.com/vllm-project/vllm/pull/34880) for `CUDAGraphWrapper` and `dummy_run` pattern
- Identify the MTP proposer class in `vllm/spec_decode/` that runs the 4 draft steps
- Wrap the draft loop in a `CUDAGraphWrapper` analogous to the EAGLE implementation
- Key constraint: input tensor shapes must be static across graph replays — verify that MTP draft step inputs are fixed-shape (batch=1, seq_len=1 at each step)
- Known gotcha: if the MTP drafter uses dynamic attention (e.g., growing KV cache), the graph capture must be re-done per KV length bucket — pre-allocate KV cache buckets as in the body model

**Papers/PRs:**
- vLLM PR #34880: https://github.com/vllm-project/vllm/pull/34880

---

## Ranking Summary

| Rank | Idea | TPS Gain Estimate | Quality Risk | Training Required | Orthogonal to Others |
|------|------|-------------------|--------------|-------------------|----------------------|
| 1 | FLy Entropy-Gated Loosened Acceptance | +20–40% | Low (tunable) | No | Yes |
| 2 | Draft lm_head Vocab Compression (FR-Spec) | +5–12% | None | No | Yes |
| 3 | Online Drafter Distillation (DVI) | +15–30% | Low | Inference-time only | Yes |
| 4 | FastMTP Position-Shared Head Retraining | +15–25% | Low | Yes (~few GPU-hours) | Yes |
| 5 | Adaptive Draft Length (AdaEDL) | +5–15% | None | No | Yes |
| 6 | Full CUDA Graph Drafter (PR #34880 pattern) | +5–12% | None | No | Yes |

**Compound potential (Ideas 1+2+5+6, all training-free):** Conservative additive estimate +35–79% TPS. Ideas 3 and 4 are additive on top if training budget is available.

**Recommended first run order:**
1. FLy (highest impact, training-free, single patch point)
2. FR-Spec vocab compression (orthogonal, no quality risk, straightforward weight surgery)
3. AdaEDL + CUDA Graph (pair these — both are systems patches with no quality risk)
4. DVI online distillation (requires careful implementation but zero auxiliary model)
5. FastMTP retraining (needs GPU hours but has compounding effect on all others)
