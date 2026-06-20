# Research Ideas — 2026-06-20 18:00

## Target Stack

- Model: `google/gemma-4-E4B-it` with `gemma4_mtp` MTP drafter (K=6, single-chain)
- Runtime: vLLM 0.22.0, transformers 5.9.0, compressed-tensors 0.15.0.1, torch 2.11.0
- Hardware: 1× NVIDIA A10G (sm_86 Ampere, 23 GB HBM)
- Quantization: W4A16 int4 g32 Marlin body + int4 g32 lm_head
- Metric: `summary.json:tps` at concurrency=1, output_len=512, 128 fixed prompts
- Current best ("int4head"): ~255 TPS (+17% vs 218 TPS baseline)

## Decode-Cycle Profile (int4head, 12.42 ms/step)

| Component | Time (ms) | % of step |
|-----------|-----------|-----------|
| Body verify-GEMM (W4A16 Marlin) | 5.92 | 47.6% |
| Non-GEMM residual (attn, norms, routing) | 3.28 | 26.4% |
| MTP drafter forward | 2.48 | 20.0% |
| lm_head GEVV | 0.75 | 6.0% |

E_accept: ~3.38 bulk, ~2.67 chat (collapse on longer contexts / conversational distribution).

## Quality Gate

≤5% degradation on AIME/MMLU-Pro/GPQA-Diamond vs vanilla base, PPL ≤ 2.42, 128/128 completions, all 4 modalities load. Non-byte-identical variants explicitly allowed.

## Dead List (DO NOT REPEAT)

- fp8 KV cache: hardware wall on sm_86
- Cutlass/Machete kernels: sm_90-only
- Sparse-Marlin 2:4: stripped from pinned vLLM binary
- vLLM MTP tree/EAGLE topology: 4 code-level blockers
- Null sampling sync / async schedule overlap: confirmed no-op (lawine #809)
- PLE-input-gate de-quant: already landed (+5.3%)
- int4 lm_head: already landed (int4head)
- VLLM_BATCH_INVARIANT=0 + force-2D attention: already landed
- Base_fullhead: collapses at 252 TPS, body-collapse confirmed (stark #536)

---

## Ranked Hypotheses

### RANK 1 — `rejection_sample_method=probabilistic` (zero-code-change, native vLLM)

**What it is.** Switch the speculative-decode rejection sampler from `strict` (default) to `probabilistic` in the existing `--speculative-config` JSON. No code change. No recompilation. No model rebuild.

**Mechanism.** vLLM 0.22.0 ships three rejection-sampling modes: `strict` (exact target-distribution equality required), `probabilistic` (accepts draft tokens with probability proportional to target/draft likelihood ratio — the standard speculative decoding formulation), and `synthetic` (constant acceptance rate oracle). Strict mode over-rejects low-risk drafts whose target probability is merely lower than draft probability rather than zero. Probabilistic mode allows partial acceptance of drafts that are "close enough" under the original Leung-Chen-Singer formulation. This directly addresses the E_accept gap: bulk 3.38 vs chat 2.67 — on chat distribution the model produces more uncertain continuations that strict rejection rejects disproportionately.

**Why it helps here.** Every 0.1 increase in E_accept ≈ +2–3 TPS at current profile. The 47.6% body-GEMM bottleneck is multiplied by 1/(E_accept+1) — fewer verification cycles per output token are needed when acceptance is higher. The strict→probabilistic switch recovers exactly the "over-rejection on near-correct drafts" gap. This is the same mechanism as ARC-Decode (Rank 2) but requires ZERO implementation effort.

**vLLM 0.22.0 / sm_86 viability.** Confirmed available. Both `rejection_sample_method` and `synthetic_acceptance_rate` keys are parsed in `vllm/model_executor/layers/spec_decode/util.py` and respected by the MTP spec-decode path. No binary recompile needed.

**Quality risk.** Low but non-zero. Probabilistic mode is the theoretically correct formulation — output distribution converges to target asymptotically. In practice, with int4 quantization, the draft and target distributions already have some gap; probabilistic mode will occasionally accept a lower-quality token. PPL should stay near current levels. Requires measuring against the 5% quality band on AIME/MMLU-Pro/GPQA-Diamond.

**First probe.** Add `"rejection_sample_method": "probabilistic"` to the existing int4head `--speculative-config` JSON. Run the 128-prompt benchmark. Compare TPS and then run the quality gate suite. If TPS improves ≥2%, proceed; if quality drops >5%, dial back or combine with a tighter JSD threshold (see Rank 2).

**Expected gain.** +3–8% TPS (conservative: E_accept goes from 2.67→2.85 on chat; aggressive: reaches bulk parity at 3.38). Exact gain depends on current distribution mismatch between int4 body and drafter.

**Papers.** Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (arXiv:2302.01318, 2023) — the original probabilistic rejection sampling formulation. Leung et al. extensions on exact-correctness guarantees. vLLM blog post 2026-05-26 on speculative config options.

---

### RANK 2 — ARC-Decode: JSD-Bounded Relaxed Acceptance Patch

**What it is.** Training-free acceptance criterion relaxation using Jensen-Shannon Divergence upper bounds on next-step distribution shift. Patches into vLLM's `spec_decode` rejection sampler — no extra forward passes.

**Mechanism.** ARC-Decode (ICLR 2026, OpenReview ID: `jhJjW2DFKD`) computes a JSD upper bound from the difference in draft and target logits at each speculation depth. If the bound is below a configurable threshold δ, the draft token is accepted even if strict rejection would reject it. The JSD bound is computed cheaply from the already-available logit vectors — no additional model calls needed. The "confidence-based pre-verification filtering" further prunes high-entropy positions where the draft is likely to diverge, concentrating acceptance on positions where relaxation is safe.

**Why it helps here.** ARC-Decode is explicitly designed for the scenario where E_accept collapses on harder/OOD inputs — exactly the chat→bulk gap (2.67→3.38) we observe. Up to 1.6× speedup over EAGLE-3 reported in the paper. On our profile: if E_accept rises by 0.5 on chat distribution, that is roughly +6–10 TPS from reduced body verify cycles. Quality risk is bounded by the JSD threshold δ (configurable; the paper recommends δ=0.05–0.15 for <1% quality degradation).

**vLLM 0.22.0 / sm_86 viability.** Requires patching `vllm/model_executor/layers/spec_decode/` — specifically the token verification logic in `rejection_sampler.py` or equivalent. The patch adds ~20–30 lines: compute per-position JSD bound from draft/target logit tensors, apply threshold gate. No CUDA kernel change needed — pure Python/PyTorch on top of existing tensors. Compatible with Marlin int4 body since it only touches post-GEMM logits.

**Quality risk.** Bounded by δ. At δ=0.10, the paper reports <0.5% degradation on MMLU/GSM8K equivalents. At δ=0.15, up to 1.2% degradation. Tune δ to stay inside the 5% quality band. Risk is also modulated by the pre-verification entropy filter — positions where the model is uncertain get strict treatment.

**First probe.** Implement the JSD bound computation in Python using draft/target logits already materialized in the spec-decode verification path. Start with δ=0.05 (conservative). Run 128-prompt TPS benchmark. If quality gate passes and TPS improves, sweep δ to 0.10 and 0.15. If Rank 1 (`probabilistic` mode) already captures much of this gain, ARC-Decode adds an additional ~0.3–0.5 E_accept on top via the entropy filter.

**Expected gain.** +5–12% TPS beyond Rank 1 if both stack. Paper reports 1.6× over EAGLE-3 — conservative estimate for our setting (smaller drafter, shallower K=6) is +8–15% over strict mode.

**Key paper.** ARC-Decode, ICLR 2026. OpenReview: https://openreview.net/forum?id=jhJjW2DFKD

---

### RANK 3 — FLy (Loosely Speculative Decoding): Entropy Gate + Deferred Window

**What it is.** Two-tier verification loosening: (1) entropy-level gate that identifies near-deterministic positions (high confidence → accept freely), (2) token-level deferred window that distinguishes genuine drafting errors from differently-worded but semantically equivalent continuations.

**Mechanism.** FLy (arXiv:2511.22972, ICLR 2026 Poster, AMD-AGI/FLy) adds two stages to the speculative decode acceptance check. First, compute entropy of the target distribution at each position. If entropy < τ_low, the token is near-deterministic — accept without strict rejection (the draft almost certainly matches the top token). If entropy > τ_high, standard strict rejection applies (uncertain positions are risky). In between, apply the "deferred window": maintain a sliding buffer of recent draft tokens and compare semantic similarity rather than exact token identity. This allows paraphrases to be accepted as valid speculation continuations. Result: ≥99% accuracy (relative to strict baseline quality), 2.81× on Llama-3.1-70B-Instruct, 5.07× on 405B, beats EAGLE-3 by 1.62× on OOD data.

**Why it helps here.** The entropy gate directly targets our bottleneck: on chat distribution, many positions are actually near-deterministic (common phrases, punctuation, repeated structure) but the strict sampler still runs full rejection. FLy's entropy gate bypasses this overhead. The deferred window addresses the OOD gap — chat prompts produce more paraphrase divergence from the drafter's training distribution. The 5.07× on 405B (a model with similar MoE structure to Gemma-4-E4B's mixture architecture) is especially encouraging.

**vLLM 0.22.0 / sm_86 viability.** Requires patching the spec-decode verification loop. Entropy computation adds ~0.1ms per step (softmax over vocab, already partially computed for rejection sampling). The deferred window requires a small embedding similarity buffer. Code is open-source at https://github.com/AMD-AGI/FLy — can adapt to vLLM 0.22.0. The main risk is the deferred window implementation complexity; the entropy gate alone is a simpler first probe.

**Quality risk.** ≥99% accuracy reported in paper. The entropy gate is conservative — only bypasses rejection at positions where the target model is already highly confident. Semantic similarity in the deferred window introduces the only real quality risk; recommend implementing the entropy gate first (no semantic similarity needed) as a cleaner first probe.

**First probe.** Implement entropy gate only (simpler). Add: `if target_entropy[i] < τ_low: accept_token[i] = True` before running the full rejection sampler. Tune τ_low ∈ {0.1, 0.3, 0.5 nats}. If entropy gate alone gives +3% TPS with quality gate pass, proceed to deferred window implementation.

**Expected gain.** Entropy gate alone: +4–8% TPS. Full FLy (entropy gate + deferred window): +10–20% TPS based on paper results scaled down for our smaller K=6 drafter and shallower model. On OOD/chat data specifically, the gain is larger than on standard benchmarks.

**Key paper.** FLy: Training-Free Loosely Speculative Decoding, arXiv:2511.22972, ICLR 2026 Poster. Code: https://github.com/AMD-AGI/FLy

---

### RANK 4 — Drafter Attention-Drift Normalization (EAGLE 3.1 Insight)

**What it is.** Add FC normalization after each target hidden state fed into the MTP drafter, preventing magnitude growth across speculation depth that causes E_accept collapse at depth 4–6.

**Mechanism.** vLLM blog (2026-05-26) documented "attention drift" in EAGLE 3.1: as speculation depth increases beyond 3–4 tokens, the drafter gradually shifts attention away from sink tokens (BOS, early context) toward its own generated speculation tokens. This creates a feedback loop where deeper speculation tokens have increasingly misaligned KV attention, causing the drafter's predictions to diverge from the target model's expectations. E_accept decays exponentially with depth. EAGLE 3.1 fixes this by normalizing the FC projection of target hidden states before concatenating them into the drafter's input — a single LayerNorm or RMSNorm wrapper with ~0 parameter overhead. Our observed E_accept drop from 3.38 (bulk, shorter sequences) to 2.67 (chat, longer context) is consistent with this attention-drift mechanism: chat prompts are longer, providing more "drift fuel."

**Why it helps here.** At K=6 speculation depth, positions 4–6 (0-indexed) are most affected by attention drift. If we recover even 0.3 E_accept on chat distribution by stabilizing depth-4+ attention, that is +4–6 TPS. The normalization wrapper is added in `gemma4_mtp` model code — no binary recompile, no weight rebuild. The target hidden states are already available in the MTP forward pass.

**vLLM 0.22.0 / sm_86 viability.** High. The fix is pure Python/PyTorch in the `gemma4_mtp` model file. No CUDA change. No quantization interaction (operates on the bf16 drafter, not the int4 body). Can be added as a 3-line patch: `hidden = F.rms_norm(hidden, hidden.shape[-1:], eps=1e-6)` before each drafter input concatenation.

**Quality risk.** Very low. The normalization stabilizes magnitudes without changing the learned weights' expressivity. The worst case is a slight reduction in drafter's ability to use magnitude cues across depths — unlikely to affect quality, may slightly decrease TPS if the drafter was previously using magnitude differences as depth signals.

**First probe.** Add RMSNorm after each target-to-drafter hidden state projection in `gemma4_mtp`. Run a 128-prompt benchmark comparing E_accept distribution per depth position (depth 1 vs 2 vs ... vs 6). If E_accept at depth 5–6 improves, the mechanism is confirmed. Full quality gate run after confirming the mechanism.

**Expected gain.** +3–7% TPS from recovering chat E_accept. Composable with Ranks 1–3 (orthogonal mechanism).

**Key reference.** EAGLE 3.1 blog post, vLLM 2026-05-26, describing FC normalization for attention-drift mitigation. Builds on: Li et al., EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (ICML 2024).

---

### RANK 5 — `synthetic_acceptance_rate` Oracle Probe (TPS Ceiling Measurement)

**What it is.** A diagnostic oracle, not a production technique. Set `"rejection_sample_method": "synthetic", "synthetic_acceptance_rate": 0.85` (or sweep 0.70, 0.80, 0.90) to measure TPS under a hypothetical acceptance rate. Output is semantically garbage but TPS is real.

**Mechanism.** The `synthetic` mode in vLLM 0.22.0 replaces the rejection sampler with a Bernoulli gate: accept each draft token with probability `synthetic_acceptance_rate`, regardless of quality. This tells us the TPS ceiling if we could achieve that E_accept level. The benchmark at concurrency=1, fixed prompts, measures wall-clock time, so the synthetic TPS directly answers: "how much TPS is left to unlock from acceptance-rate improvement alone?"

**Why it helps here.** Before implementing ARC-Decode or FLy patches (which require days of dev time), this 10-minute probe answers: "is there ≥10% TPS headroom from E_accept improvement?" If synthetic at 0.85 gives only +5% TPS, the acceptance-rate subspace is nearly exhausted. If it gives +20%, the subspace is large and Ranks 1–4 are high-priority. This oracle grounds all other estimates.

**vLLM 0.22.0 / sm_86 viability.** Confirmed. Zero code change. Single JSON key addition.

**Quality risk.** N/A — this is a TPS measurement only, not a submission candidate. Run with --ignore-quality-gate or equivalent.

**First probe.** Add two lines to the speculative config: `"rejection_sample_method": "synthetic", "synthetic_acceptance_rate": 0.85`. Run 128-prompt benchmark. Note TPS. Sweep to 0.70, 0.90, 1.00. Plot TPS vs acceptance rate to get the ceiling curve.

**Expected output.** A TPS-vs-E_accept curve that calibrates all other experiments. If TPS ceiling at 1.00 acceptance is, say, 310 TPS, we know there is 22% headroom from the current 255 TPS — a significant research target.

---

### RANK 6 — AdaSPEC Selective KD Draft Alignment (Training-Based)

**What it is.** Fine-tune the MTP drafter using selective knowledge distillation: filter tokens where the drafter's capacity is insufficient, focus KD on tokens where improvement is achievable (NeurIPS 2025 Spotlight).

**Mechanism.** AdaSPEC (NeurIPS 2025 Spotlight, "Adaptive Speculative Decoding via Selective Knowledge Distillation") uses the reference (target) model to identify tokens where the draft model's acceptance rate can be meaningfully improved by additional training. Tokens that are too hard (reference confidence is low, i.e., the target itself is uncertain) are excluded from KD — these are "irreducibly difficult" positions where no amount of distillation will help. Tokens that are too easy (draft already matches target) are also excluded — no marginal gain. The KD focuses on the "improvable middle" — positions where the draft is wrong but the target is confident. This produces a higher E_accept without compromising quality.

**Why it helps here.** The bulk→chat E_accept gap (3.38→2.67) suggests the drafter was trained primarily on bulk/summarization distribution and underperforms on instruction-following/conversational data — exactly the AdaSPEC "improvable middle" scenario. The pinned transformers 5.9.0 supports fine-tuning `gemma4_mtp`. Training time is modest (drafter is a small linear-chain MTP head, not a full LLM). Data: use the same 128-prompt chat benchmark distribution + augmented conversational data.

**vLLM 0.22.0 / sm_86 viability.** Training uses the host training stack (transformers 5.9.0 + compressed-tensors 0.15.0.1), not vLLM. The fine-tuned drafter weights are loaded by vLLM at serve time — no binary change needed.

**Quality risk.** Low for selective KD (by design, it avoids distilling uncertain tokens). The main risk is distribution shift if training data diverges too far from evaluation distribution. Recommend fine-tuning on a mix of 80% original drafter training distribution + 20% chat distribution to preserve bulk performance.

**First probe.** Profile which token positions in the 128-prompt benchmark have both (a) low draft acceptance and (b) high target confidence. This is the "improvable middle" size estimate. If the improvable middle is >30% of tokens, AdaSPEC fine-tuning is justified. If it is <10%, the drafter is already near-optimal for the achievable set.

**Expected gain.** +5–15% TPS from E_accept improvement on chat distribution. Training time: ~4–8 GPU-hours on A10G for the small MTP head. Orthogonal to Ranks 1–4 and composable after they land.

**Key paper.** AdaSPEC: Adaptive Speculative Decoding via Selective Knowledge Distillation, NeurIPS 2025 Spotlight. Related: SpecDec++, ICLR 2025 (adaptive candidate length K via trained acceptance prediction head, arXiv:2405.05500).

---

### RANK 7 — Dynamic Token Halting via QuickSilver (Exploratory / High Risk)

**What it is.** Token-level computation halting for body-GEMM layers where the hidden state has converged before the final layer — targeting the 47.6% body-GEMM bottleneck by skipping later layers for "easy" tokens.

**Mechanism.** QuickSilver (arXiv:2506.22396, Jun 2025) introduces four modular training-free techniques: Dynamic Token Halting (hidden state cosine similarity threshold triggers early exit from transformer layers), KV Cache Skipping (skip attention for halted tokens), Contextual Token Fusion (merge near-duplicate token representations), and Adaptive Matryoshka Quantization (variable precision per token). Together: 39.6% FLOP reduction, ≤0.2 PPL degradation on GPT-2/Llama-2.

**Why it might help here.** The 47.6% body-GEMM bottleneck is the largest single target. If even 20–30% of tokens can exit the body early (at layer 16 instead of layer 26 for Gemma-4-E4B-it), the GEMM time per step drops proportionally. Dynamic Token Halting requires no weight modification — just a cosine similarity check of hidden states at intermediate layers.

**vLLM 0.22.0 / sm_86 viability.** HIGH RISK. Marlin's W4A16 block-quantized GEMM kernel operates on fixed layer-width inputs — per-token halting would require either (a) a custom Marlin kernel variant that handles variable-length token sets, or (b) masking halted tokens before the Marlin call and padding the result, which may not reduce actual GEMM time (Marlin pads to block boundaries). The paper was tested on dense (non-quantized) GPT-2 and Llama-2 — int4-Marlin compatibility is UNVERIFIED.

**Quality risk.** ≤0.2 PPL is the paper's claim on dense models. With int4 quantization, the hidden-state cosine similarity threshold may need recalibration (int4 introduces additional rounding noise that could make convergence signals noisier). PPL ≤ 2.42 gate is tighter than the paper's regime.

**First probe.** Instrument the body forward pass to measure per-layer hidden-state cosine similarity distribution on the 128-prompt benchmark. If >20% of tokens converge by layer 20 (of 26), the halting mechanism has a viable target. Only proceed to implementation if this diagnostic shows the convergence signal exists. Do NOT attempt Marlin integration without first verifying convergence signal.

**Expected gain.** If viable: +10–20% TPS. If Marlin integration fails: 0% TPS (masked tokens still incur GEMM cost). Recommend starting with diagnostic only.

**Key paper.** QuickSilver: Efficient Large Language Model Serving with Adaptive Token-Level Computation Halting, arXiv:2506.22396, Jun 2025.

---

## Summary Table

| Rank | Idea | Expected TPS gain | Quality risk | Implementation effort | First probe cost |
|------|------|-------------------|--------------|----------------------|------------------|
| 1 | `rejection_sample_method=probabilistic` | +3–8% | Low | Zero | 10 min |
| 2 | ARC-Decode JSD patch | +5–12% (additive) | Bounded by δ | Medium (~100 LOC) | 1–2 days |
| 3 | FLy entropy gate | +4–8% (entropy gate alone) | Very low | Medium (~80 LOC) | 1–2 days |
| 4 | Drafter attention-drift normalization | +3–7% | Very low | Low (~10 LOC) | 2 hours |
| 5 | `synthetic_acceptance_rate` oracle | N/A (diagnostic) | N/A | Zero | 10 min |
| 6 | AdaSPEC selective KD | +5–15% | Low | High (requires training) | 1 day profiling |
| 7 | QuickSilver dynamic halting | +10–20% IF viable | Medium (PPL risk) | Very high | 2 hours (diagnostic) |

## Recommended Sequencing

**Immediate (zero-code, run today):**
- Rank 5 (synthetic oracle): measure TPS ceiling curve — grounds all other estimates
- Rank 1 (probabilistic mode): highest expected gain/effort ratio

**Short-term (2–4 days, code patch):**
- Rank 4 (attention-drift normalization): low effort, orthogonal to Rank 1
- Rank 3 (FLy entropy gate only, first): simpler than full FLy, composable with Ranks 1+4

**Medium-term (1–2 weeks):**
- Rank 2 (ARC-Decode JSD patch): if Ranks 1+3 are insufficient to reach next quality-safe ceiling
- Rank 7 diagnostic: profile convergence signal before committing to implementation

**Long-term (training required):**
- Rank 6 (AdaSPEC): only if acceptance-rate ceiling from training-free methods is confirmed insufficient

## Composability Notes

- Ranks 1, 2, 3 all target the same mechanism (acceptance criterion relaxation) — they interact. Start with Rank 1 (probabilistic), add Rank 4 (orthogonal), then add Rank 3 entropy gate, then add Rank 2 JSD bound on top of probabilistic.
- Rank 4 (attention-drift) is fully orthogonal to all acceptance criterion changes — compose freely.
- Rank 7 (QuickSilver halting) targets the body-GEMM directly and is orthogonal to all acceptance changes — but viability is uncertain.
- If Ranks 1+4 together give ≥10% TPS improvement with quality gate pass, that is already a strong result to ship while the deeper patches (Ranks 2, 3) are being developed.
