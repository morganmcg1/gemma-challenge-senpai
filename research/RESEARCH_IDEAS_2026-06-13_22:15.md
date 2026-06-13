# Research Ideas — 2026-06-13 22:15

## Context: Post-PR #43 Split-KV Frontier

Current official best: **481.53 TPS** (`fa2sw_precache_kenyan` submission).
Target: **>500 TPS** with **<5% private-gap** (currently predicted ~12.4% gap — 2.5× over the gate).

Post-split-KV decode composition (verified by PR #43 + #30 profile):
- Verify-body int4 GEMM: **53.2%** (Marlin W4A16, at bandwidth floor — essentially untouchable)
- fa2sw / Triton attention: **19.6%** (partially resolved by PR #43 split-KV; residual ~5%)
- Drafter forward pass: **15.5%** (EAGLE-3 4B; primary addressable lever)
- lm_head: **1.0%** (lmhead12k fully collapsed — dead lever)
- Other overhead: ~10.7%

Primary bottleneck is no longer latency math but **drafter acceptance quality**:
- Current tf_acc: 0.7314 on reasoning held-out (PR #25)
- tf_acc needed for >500 TPS: ≥0.85
- Private acceptance collapse: E_accept 4.06→3.57, accept-rate 43.7%→36.7% on chat/reasoning prompts
- The 11.33% of the 12.4% predicted private gap is drafter acceptance collapse

---

## Category A: Private-Stability & Programme Risk (DO FIRST)

### A1 — Private-Gap Re-Calibration on Post-#43 Split-KV Frontier
**Status:** Builds on PR #55 concept (never ran). URGENT: submit before next HF launch.

**What:** Re-run `private_gap_probe.py` against the current `fa2sw_precache_kenyan` stack (post-#43). The previous 12.4% gap estimate was computed on the pre-#43 frontier. The 3D split-KV patch changed the decode composition non-trivially — verify that the gap model still holds. More importantly: measure whether the reasoning-distribution acceptance collapse changes at the new M=45 operating point.

**Mechanism:** The private-gap estimate is a linear model: `gap = distribution_gap + precache_gap`. Distribution gap (11.33%) is driven by drafter acceptance on reasoning prompts, which is independent of the split-KV patch. Precache gap (1.24%) may shift slightly because the verify-step budget changed. The updated estimate feeds directly into go/no-go for the next HF submission.

**Key deliverable:** Updated gap model with confidence interval. If actual gap is <5%, green-light next HF submit immediately. If gap is 8–12%, quantify which acceptance arm (CoT vs. math vs. chat) drives it.

**Experiment:** 
1. Re-run `scripts/private_gap_probe.py` with `--submission fa2sw_precache_kenyan`
2. Sweep `--distribution_mix reasoning:0.6,chat:0.4` to match private benchmark profile
3. Compare E_accept and accept-rate with pre-#43 baseline from PR #44
4. Update gap model coefficients in `research/EXPERIMENTS_LOG.md`

**Estimated cost:** 15 minutes, 0 GPU-hours beyond normal serving.

---

### A2 — Distribution-Matched Reasoning Prefix Prewarming (Novel, Not in Any Prior File)
**Status:** No prior PR. Novel idea synthesized from private-stability root cause analysis.

**What:** The private benchmark is reasoning/CoT heavy. The drafter acceptance collapses because the KV cache starts "cold" on reasoning-style prompts — the drafter has not warmed up to the distribution before the prompt begins. Prepopulate the KV cache with a small set of reasoning-heavy prefix templates (e.g., 5–10 synthetic reasoning preambles of 32–64 tokens) and serve them as a fixed "prewarming" prefix that is prepended to all requests before the actual prompt, then masked out of the output. This shifts the KV cache into a distribution closer to the private benchmark's reasoning prompts.

**Mechanism:** EAGLE-3 acceptance rate is a function of the current hidden-state distribution. Reasoning prompts trigger a different activation regime than chat prompts. By forcing the model through a few reasoning-style tokens at the start of each request, the hidden states at prompt-start are closer to the distribution the drafter was trained on. This is a training-free intervention that does not change any weights and preserves greedy-token identity (the prewarming prefix is masked out before generation starts).

**Risk:** The prewarming prefix adds latency on the prefill path. Needs to be short enough that the prefill cost is dominated by the actual prompt. Needs to be greedy-identity-safe (mask properly). May not survive the PPL gate if the prefix bleeds into the logits.

**Experiment:**
1. Implement prefix masking in `submissions/fa2sw_precache_kenyan/serve.py` — prepend 32-token reasoning template and mask first 32 positions from output generation
2. Measure `private_gap_probe.py` gap with and without prewarming prefix
3. Measure TPS impact (prefill cost) vs. gap reduction tradeoff
4. Run greedy-identity PPL gate to confirm ≤2.42

---

## Category B: Drafter Quality — Acceptance Rate Improvement

### B1 — HASS Top-K Harmonized Distillation Loss (arXiv 2408.15766)
**Status:** No prior PR. From 20:15 ideas file (B1) — highest-priority unexecuted drafter quality idea.

**What:** Replace the current cross-entropy teacher-forcing loss in EAGLE-3 training with HASS (Harmonized Sampling Score), which uses a Top-K alignment loss that penalizes the drafter when its top-K predictions diverge from the teacher's top-K distribution. The key insight is that the speculative decoding acceptance criterion only needs the drafter to produce tokens in the teacher's high-probability region — not exact probability matching. HASS trains for this directly.

**Mechanism (from arXiv 2408.15766):** Standard EAGLE-3 training uses cross-entropy loss that is dominated by the target token, giving no gradient signal for tokens the drafter correctly ranks #2–#5 in the teacher's distribution. HASS adds a sampling-consistency term: the drafter's token distribution must "harmonize" with the teacher's when sampling — i.e., the probability mass in the acceptance region (teacher top-K) must be high. This directly optimizes for the speculative decoding acceptance criterion. Reported: 8–20% acceptance rate improvement on CodeLLaMA 7B and LLaMA-2 13B.

**Implementation:** Modify `research/eagle3_drafter/train_eagle3.py`:
- Add `hass_loss` alongside the existing `cross_entropy_loss`
- `hass_loss = -log P_drafter(token in teacher_topK(t))` for K=10
- Mix: `total_loss = 0.8 * ce_loss + 0.2 * hass_loss` (start with these weights, sweep if needed)
- Use `model_best.pt` (step 3500) as initialization

**Key hyperparameters (from paper):** K=10 for the harmonization set; lambda=0.2 for HASS weight; batch size ≥512 for stable gradient estimates of the Top-K boundary.

**Expected gain:** If HASS reproduces its reported 8–20% acceptance rate improvement on our setting, E_accept goes from 4.06 to 4.39–4.87, which pushes TPS well above 500 on the private benchmark.

**Risk:** The paper reports results on models trained from scratch with HASS; our drafter is a fine-tuned head. The gain from fine-tuning an already-trained drafter with HASS may be smaller. Start with 5k steps fine-tuning from `model_best.pt`.

---

### B2 — Reasoning-Corpus EAGLE-3 Retraining (PR #34 Concept — Highest TPS Ceiling)
**Status:** PR #34 never ran (was in never-ran list). Reproposing with updated recipe.

**What:** The EAGLE-3 drafter plateau at 0.73 tf_acc is a data bottleneck: the training corpus is not reasoning-heavy enough to match the private benchmark distribution. Retrain from scratch (or from `model_best.pt`) with a corpus that has ≥60% reasoning/CoT content (AIME 2024, GPQA Diamond, ARC-Challenge, OpenMathReasoning, GSM8K-CoT, Hendrycks MATH-CoT) vs. the current ~30% reasoning mix.

**Mechanism:** PR #34 diagnosed: the data is the constraint, not the number of training steps. At step 3500 (current best), tf_acc on reasoning held-out is 0.7314 and on chat is ~0.81. The gap (0.08) is purely distributional. The private benchmark is heavily reasoning-weighted. Closing the reasoning tf_acc gap from 0.73 to ≥0.85 is worth ~15% TPS gain on the private benchmark and ~10% on the public benchmark.

**Recipe (updated for post-#43 stack):**
1. Build corpus: `data/eagle3_reasoning_v2/` — 60% AIME/GPQA/MATH-CoT, 20% code, 20% chat
2. Use hidden states from layers (2, 21, 39) as auxiliary inputs (same as current EAGLE-3)
3. Train for 20k steps with cosine LR decay from 3e-4 to 3e-5, batch size 512
4. Evaluate tf_acc on the full held-out set AND the reasoning-only held-out set
5. Stop condition: reasoning tf_acc ≥ 0.85 or 20k steps (whichever first)

**Estimated TPS ceiling (from cost model in PR #18/#26):** At E_accept=5.1 (tf_acc≥0.85), TPS ceiling = ~530 TPS local, ~560 TPS official (given split-KV gains).

---

### B3 — FastMTP Position-Shared Self-Distillation (arXiv 2509.18362)
**Status:** No prior PR. From 20:15 ideas file (B2) — slightly lower priority than HASS.

**What:** FastMTP replaces EAGLE-3's separate draft-head architecture with a self-distillation scheme where the model uses position-shared token prediction: the same MTP head is trained to predict token t+1 using only the hidden state at position t, with a consistency loss that forces the MTP predictions to agree with the base model's greedy output. This removes the "hidden-state transport" overhead of EAGLE-3 (which requires routing hiddens from layers 2, 21, 39 to the draft head).

**Mechanism:** EAGLE-3's two forward passes (base + draft) are sequential. FastMTP trains the MTP head to use fewer auxiliary inputs by learning to distill the multi-layer information into a single-layer representation. The paper reports 1.4–1.9× speedup on the draft-head forward pass with similar or better acceptance rates than EAGLE-3 on reasoning tasks.

**Implementation fit:** The 15.5% drafter cost is the primary addressable latency lever. If FastMTP reduces the drafter forward pass cost by 30–40%, that translates to 4–6% TPS gain, roughly additive with W8A8 quantization.

---

## Category C: Kernel / Systems Optimizations

### C1 — W8A8 Drafter INT8 Quantization (PR #47 — Never Ran)
**Status:** PR #47 exists but never ran. Bringing back with updated context.

**What:** Quantize the EAGLE-3 drafter (4B Gemma MTP head) to W8A8 (INT8 weight + INT8 activation) using `bitsandbytes` or `torch.ao.quantization.quantize_dynamic`. The drafter is 15.5% of decode time. INT8 tensor-core throughput on A10G is 2× vs FP16 for large matrices. Expected: ~7–8% TPS improvement (15.5% × 0.5 kernel speedup).

**Mechanism:** The drafter runs FP16 currently. INT8 quantization on A10G uses `imma` (INT8 matrix multiply accumulate) instructions rather than `hmma` (FP16). For the drafter's linear layers (QKV projections, FFN), this yields 2× arithmetic throughput. Activation quantization adds overhead for the per-token scale computation, but this is amortized over the large batch size at M=45.

**Greedy-identity safety:** The drafter does not participate in the greedy-identity gate directly — only the verifier's logits matter. INT8 drafter changes acceptance probabilities but not the final accepted token (which is still drawn from the verifier's FP16 logits). PPL gate should be unaffected.

**Key hyperparameters from similar deployments:**
- Use `group_size=128` for W8 (matches current Marlin g128 scheme)
- A8 with per-token dynamic scaling (not static)
- Avoid INT8 for `lm_norm` and attention output projections (accuracy-sensitive)

**Expected TPS:** 481.53 + (0.155 × 0.4 × 481.53) ≈ **511 TPS** (if 40% kernel speedup achieved).

---

### C2 — SAM-Decoding Triton In-Graph Suffix Automaton (arXiv: SAM-Decoding, + PR #10 GO signal)
**Status:** PR #10/#13 established the 8.93% causal budget (GO signal). No implementation PR yet.

**What:** Implement SAM-Decoding as a real Triton kernel that runs inside the vLLM decode graph. The suffix automaton (built once from the batch prompt tokens) provides deterministic draft proposals for positions where the prompt contains a matching suffix. At K>8, 8.93% of decode steps have a SAM match. These steps cost zero drafter forward passes.

**Mechanism:** In SAM-Decoding, a deterministic causal suffix automaton scans the current KV-cache history for the longest matching suffix of length ≤K. When a match exists, the next K tokens are proposed deterministically from the cached suffix — no drafter network needed. For the 8.93% of steps with a match, the drafter is completely bypassed. For the remaining 91.07%, normal EAGLE-3 drafting occurs.

**Implementation path (from PR #13 template):**
1. Build suffix automaton in Python once per request from the prompt tokens (`scripts/analyze_suffix_budget.py` already exists)
2. Implement Triton kernel `sam_lookup` that takes current decode position and returns next-K tokens if a SAM match exists
3. Route: if `sam_lookup` returns a match, skip drafter forward pass; else fall through to EAGLE-3
4. Verify greedy identity: SAM proposals are exact deterministic copies from the prompt, so they are inherently greedy-valid

**Expected TPS:** 8.93% of steps skip a 15.5%-cost drafter. Net gain: 8.93% × 15.5% ≈ **~1.4% TPS gain**. Small but free (training-free) and combinable with W8A8.

**Risk:** SAM construction is O(n²) in prompt length. For long prompts (>2048 tokens), construction latency may exceed the drafter savings. Needs profiling.

---

### C3 — AdaEDL Entropy-Adaptive Dynamic-K (PR #54 — Never Ran)
**Status:** PR #54 exists but never ran. Corrected accepthist lever.

**What:** Replace the acceptance-history-based dynamic-K from PR #51 with an entropy-adaptive scheme: at each decode step, measure the drafter's output entropy (H = -sum p log p over the vocabulary) and set K dynamically. High-entropy steps (drafter is uncertain) use K=1 to avoid bad drafts; low-entropy steps (drafter is confident) use K=16 for maximum parallelism.

**Mechanism:** PR #51's accepthist dynamic-K gained ~20 TPS by adapting K to recent accept history. However, PR #54's hypothesis is that the drafter's own entropy is a better real-time signal than the lagged accept history. The accept history is a batch-level average; entropy is per-step, per-request, zero-lag. When the drafter is in a high-entropy region (reasoning context switch, long CoT), K should be reduced aggressively to avoid wasting verify cycles.

**Implementation:**
1. After each drafter forward pass, compute `H = -sum(softmax(logits) * log_softmax(logits))`
2. K = `max(1, min(16, round(baseline_K * (1 - H / H_max))))`
3. Calibrate `H_max` on the public benchmark prompts
4. Fallback: if `H` is unavailable (e.g., batched speculative step), use accepthist K from PR #51

**Key uncertainty:** Does entropy predict accept quality better than accept history at the M=45 operating point? PR #51 gained 20 TPS suggesting accept history has signal. Entropy adds sub-ms overhead per step.

---

## Category D: Distribution Robustness (Private-Gap Direct Attacks)

### D1 — Acceptance-Calibrated Verifier Temperature Scaling
**Status:** Novel, not in any prior ideas file. No prior PR.

**What:** The private benchmark's reasoning prompts cause the verifier's logits to be sharper (lower temperature / more peaky) than the drafter's predictions. This creates a systematic acceptance mismatch: the drafter proposes token t+1 with moderate probability, but the verifier assigns it very low probability (sharp distribution peaked on a different token). Apply post-hoc temperature scaling to the verifier's logits (T=1.0→0.9) on a per-request basis, calibrated to match the drafter's distribution width.

**Mechanism:** Speculative decoding acceptance criterion: `accept token_k if u < min(1, p_verifier(token_k) / p_drafter(token_k))`. If p_verifier is too sharp (reasoning prompts), this ratio is frequently <1 and rejection happens more often. Softening the verifier's distribution (temperature scaling) raises p_verifier for the drafter's likely tokens, increasing the acceptance rate. The key constraint: temperature scaling must not change the greedy token (argmax is invariant to monotone scaling) — so greedy identity is preserved.

**Implementation:** In `submissions/fa2sw_precache_kenyan/serve.py`, wrap the logits before the acceptance check:
```python
if temperature_scale != 1.0:
    verify_logits = verify_logits / temperature_scale
```
Sweep T in {0.85, 0.90, 0.95} on the public benchmark; measure TPS and PPL.

**Risk:** Temperature scaling on the verifier can change the accepted token distribution and therefore the output quality (PPL). Needs careful PPL gate check.

---

### D2 — Lookahead / Jacobi Parallel Decode as Fallback (arXiv 2402.08559)
**Status:** No prior PR (C1 from 20:15 ideas file). LOWER priority but training-free.

**What:** Lookahead decoding generates multiple token candidates in parallel using Jacobi iteration on the transformer. Unlike speculative decoding, it requires no drafter — only the base model. It provides a 1.5–2× speedup on reasoning tasks where the drafter acceptance collapses (exactly the private-gap failure mode). Implement as a fallback path: when drafter accept rate falls below threshold (e.g., <0.35), switch to Lookahead decoding for that request.

**Mechanism:** Lookahead decoding solves the fixed-point equations `x_k = f(x_{k-1})` for the autoregressive transformer in parallel using a "window" of parallel iterations. On reasoning tasks with long CoT chains, the output has low entropy (the next token is predictable from context) making Lookahead converge quickly. The fallback trigger (accept_rate < 0.35) precisely targets the private distribution collapse case.

**Implementation complexity:** High. Requires a separate execution path in vLLM's decode loop. More of a medium-term investment.

---

## Priority Ranking

| Rank | Idea | Category | Mechanism | Est. TPS Gain | Private-Gap Impact | Cost |
|------|------|----------|-----------|--------------|-------------------|------|
| 1 | A1: Private-gap re-calibration | Infra | Measure gap on current stack | 0 (diagnostic) | Direct | 15 min |
| 2 | C1: W8A8 drafter INT8 | Systems | 15.5% × 40% kernel speedup | +7–8% (~+33 TPS) | Neutral | 4h |
| 3 | B2: Reasoning-corpus retraining | Drafter quality | 0.73→0.85 tf_acc | +15–20% (~+70 TPS) | Major fix | 48h |
| 4 | B1: HASS Top-K distillation | Drafter quality | Top-K alignment loss | +8–20% accept rate | Major fix | 24h |
| 5 | C3: AdaEDL entropy dynamic-K | Systems | Per-step entropy K selection | +10–20 TPS | Partial fix | 6h |
| 6 | A2: Reasoning prefix prewarming | Private-stability | KV-cache distribution shift | 0 (gap diagnostic) | Direct | 2h |
| 7 | D1: Verifier temperature scaling | Distribution | Accept-calibrated logit scaling | +5–10 TPS | Partial fix | 2h |
| 8 | C2: SAM-Decoding Triton | Systems | 8.93% steps skip drafter | +1–2% (~+7 TPS) | Neutral | 12h |
| 9 | B3: FastMTP self-distillation | Architecture | Smaller drafter forward pass | +4–6% | Depends on quality | 48h |
| 10 | D2: Lookahead fallback | Fallback | Training-free CoT speedup | +? on private | Partial fix | 2 weeks |

## Assignment Recommendation

**Immediate (assign now):**
- **stark** → C1 (W8A8 drafter INT8, PR #47 never ran) — quickest high-value lever on the current stack; measures real drafter cost reduction
- **wirbel** → A1 (private-gap re-calibration on post-#43 stack) — essential diagnostic before any HF submission

**Next round (if above succeed):**
- **stark** → B1 (HASS Top-K distillation) if W8A8 shows drafter is the bottleneck
- **wirbel** → B2 (reasoning-corpus retraining) — data-driven drafter quality fix; highest ceiling

**If drafter quality is improving:**
- Test A2 (prewarming) and D1 (temperature scaling) as cheap private-gap probes

**If drafter quality plateau persists past B2:**
- Move to B3 (FastMTP), C2 (SAM-Decoding), then D2 (Lookahead) as escalation path
