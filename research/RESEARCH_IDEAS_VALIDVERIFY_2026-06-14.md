# Valid-Verify-500: Research Findings on 4th/5th Structural Paths
## Date: 2026-06-14
## Context: Greedy-Identity Blocker #192

---

## Strategic Framing

**The specific problem:** int4-Marlin GEMM in the verify pass uses split-K reduction, whose
reduction order is a function of batch width M. AR decode runs at M=1; speculative verify
runs at M=K+1 (K=7 → M=8). Different M → different split-K tile geometry → different FP
accumulation order → 56% argmax flip rate. Greedy-identity fails.

**Three paths already mapped (do NOT re-propose):**
1. fp16/bf16 verify — bit-stable but ~380-430 TPS (too slow, 14-29% below 500)
2. No speculation — ~165 TPS (disqualified)
3. Custom batch-invariant int4 kernel — ceiling ~537 TPS but +51.78% overhead vs budget
   of 7.33%; budget cleared at NO physical λ

**Target:** valid >500 TPS with PPL ≤ 2.42, 128/128 completions, greedy token identity.
**Overhead budget:** ~7.33% over current 481.53 TPS baseline to stay valid and >500.

---

## Direction 1: MarginGate — Argmax-Robust Verify (SKIP for high-margin positions)

### What it is
Compute the logit margin (top-1 minus top-2 logit) at each speculative position. When
the margin exceeds a threshold τ derived from the maximum possible FP perturbation
from any valid split-K reduction order, the argmax is provably stable regardless of
which GEMM path ran. For high-margin positions, skip the verify step entirely — the
AR argmax and the speculative argmax are guaranteed identical. For low-margin positions
(margin < τ), fall back to a deterministic bf16 verify GEMM.

### Mechanism: why it targets the bottleneck
Split-K non-determinism creates a bounded perturbation in the logit vector. The
magnitude ε of this perturbation is bounded by the number of split points times the
rounding error per multiply-accumulate. If logit_top1 - logit_top2 > 2ε, no valid FP
reduction order can flip the argmax. The margin is computable cheaply on the CPU side
(or a tiny GPU kernel) from the draft's existing logit output.

From the MarginGate paper (arxiv 2605.30218):
- 81-85% of speculative positions have margin > τ (empirical across diverse tasks)
- Those 81-85% contribute zero verify GEMM cost (skip entirely)
- Remaining 15-19% fall to a safe deterministic path
- Reported 2.23x / 1.99x latency reduction end-to-end on A100/A10G-class hardware

### Batch-invariance reasoning
High-margin positions: batch-invariant by construction — the argmax cannot flip for
any valid FP perturbation. No GEMM is needed for the verify pass at these positions.
Low-margin positions: a bf16 matmul at M=1 (run as a separate single-row GEMM, not
as a batched verify) is fully deterministic. There is no split-K tile ambiguity at M=1.

**The key insight:** we never need batch-invariance across M=1 vs M=K+1. Instead, we
skip the verify GEMM for positions where batch-variance cannot change the answer, and
force M=1 for positions where it could.

### Quantitative TPS estimate
- Verify GEMM eliminated for 81-85% of positions → roughly 6-8% wall-clock reduction
  on the verify pass
- bf16 M=1 verify for remaining 15-19% → modest overhead vs current int4 verify
- Net expected system TPS: ~490-510 (speculative; depends on bf16 M=1 overhead
  vs int4 M=8 baseline on this architecture)

### Requirements / implementation complexity
- Needs: logit margin computation per speculative position (trivial — top-2 logit diff)
- Needs: τ calibration — empirically or analytically bound ε_max for int4-Marlin
  on A10G (depends on group size, num_splits, and VRAM bandwidth). Can be done
  offline on a representative input set.
- Needs: bf16 fallback single-row GEMM for low-margin positions
- No custom CUDA kernel required; bf16 M=1 via torch.matmul is deterministic
- Integration point: verify loop in the MTP K=7 speculation pipeline

### Main risk
**τ calibration drift:** if ε_max is underestimated, some genuinely unstable positions
are incorrectly skipped → residual token flips. The calibration must be done against
the specific Marlin kernel and group size in production.
**PPL impact:** rarely-verified positions effectively trust the draft at high confidence.
If the margin criterion is too loose, PPL could degrade. Monitor PPL through the margin
gate carefully.

### P(reaches valid >500 TPS)
0.45 — strong mechanism, proven skip rate, but τ calibration for int4-Marlin (not fp16)
is untested; the bf16 M=1 fallback overhead and its interaction with the speculation
pipeline could erode or eliminate the net gain.

---

## Direction 2: Batch-Invariant-by-Construction Verify GEMM (verify-only fixed-tile)

### What it is
Replace the int4-Marlin GEMM in the verify pass only with a fixed-tile GEMM whose
reduction tree does not depend on M. Keep int4-Marlin for all AR decode passes (M=1
is already batch-invariant by coincidence — single row, no split-K ambiguity). This is
a narrower and cheaper version of Path 3 (custom batch-invariant int4 kernel) that
avoids rewriting the full kernel.

### Mechanism: why it targets the bottleneck
The non-determinism comes from split-K: Marlin tiles the K dimension into splits that
change count as M changes. A fixed-tile kernel sets tile_M, tile_K at compile time and
refuses to adapt them dynamically. With a compile-time-fixed tile, the same split-K
geometry runs for both M=1 AR and M=K+1 verify → same FP accumulation order → same
argmax.

### Key evidence
- Thinking Machines blog (2024): fixed-tile Triton kernel achieves batch-invariance
  at ~20% overhead vs cuBLAS best-case; significantly less than the +51.78% overhead
  of the custom int4 kernel approach (Path 3).
- llm_reproducibility project (vLLM): TP-invariant row-split linear layer replacements
  show the verify-GEMM-only substitution approach is architecturally sound.
- Critical: only the VERIFY GEMM needs to be fixed-tile. The AR decode at M=1 is
  already deterministic (one row → one tile → trivially consistent).
- FP4 NVFP4 kernel on sm_86: 85-129 TFLOPS, 1.4-2.4x faster than bf16; potentially
  batch-invariant if a fixed-schedule variant exists.

### Quantitative TPS estimate
- ~20% verify-GEMM overhead replaces the int4-Marlin verify GEMM overhead
- Current verify GEMM fraction of total wall clock (MTP K=7): ~25-30%
- Net system overhead from switching only verify GEMM: ~5-6%
- Expected system TPS: ~455-480 (below 500 unless verify GEMM fraction is smaller
  than estimated, or FP4 route recovers more speed)
- FP4 route (batch-invariant fixed-tile FP4 kernel, if buildable): ~500-520 TPS

### Requirements / implementation complexity
- Moderate: write or adapt a fixed-tile Triton int4 verify GEMM (tile_M=1 or fixed
  small power of 2, fixed tile_K matching group_size)
- FP4 path: requires int4 → FP4 weight conversion and NV FP4 kernel availability;
  sm_86 FP4 support is unconfirmed (FP4 mma requires sm_89+ in official docs)
- Integration: replace only the linear layers in the verify forward pass

### Main risk
**sm_86 FP4:** NVIDIA's NVFP4 MXFP4 requires sm_89+ (Hopper/Ada). On A10G (sm_86)
this path likely requires emulation and will not be faster than int8. Do not rely on FP4.
**20% overhead budget:** even if fixed-tile works, system-level TPS with 20% verify
overhead may land at ~455-480, still below 500. Marginal gain only if verify is a
smaller wall-clock fraction than estimated.

### P(reaches valid >500 TPS)
0.30 — batch-invariant fixed-tile GEMM is technically sound, but the overhead math
at system level does not clearly clear 500 TPS. More promising as a building block
combined with Direction 1 (margin gate reduces verify GEMM invocations).

---

## Direction 3 (NEW — Primary 4th Structural Path): LLM-42 DVR — Decode-Verify-Rollback

### What it is
Decode-Verify-Rollback (DVR): generate tokens at full speed with unmodified non-
deterministic kernels, then replay a sliding window of candidate tokens under a
fixed-shape verification schedule (padding to constant batch M_verify), commit tokens
where the fixed-shape verify agrees with the decode, and roll back + recompute tokens
where it disagrees. arxiv 2601.17768, github microsoft/llm-42.

### Mechanism: why it targets the bottleneck
DVR attacks the root cause differently from the other paths: instead of making the
decode numerically stable, it accepts that the decode is non-deterministic and ensures
the committed output is identical to what a fixed-shape verification pass would have
produced. The key property is that the verify pass always runs at a FIXED M (M_verify),
so the split-K tile geometry is constant → the reduction tree is constant → the argmax
is reproducible across runs.

**The specific greedy-identity implication:** if M_verify is set to 1 (single-row verify,
identical to AR decode geometry), then the verify pass IS the M=1 AR decode. Tokens
committed are tokens that would have been produced by plain AR decode. Greedy identity
is satisfied by construction.

### Quantitative evidence from paper (fp16, Llama-3.1-8B-Instruct)
- Rollback/recompute rates: 0.32% tokens (ShareGPT) to 10.97% tokens (ArXiv)
- Median: "fewer than one rollback per request on average"
- Overhead vs unmodified SGLang: ~6% throughput penalty at 100% deterministic traffic
- vs SGLang-Deterministic (their baseline fp16 deterministic system): 33% faster
- FlashAttention setting: num_splits=1 in the verify pass (eliminates SDPA
  non-determinism independently)
- Known limitation: verification introduces global pauses, no speculative decoding
  integration, no prefix cache sharing

### Applying DVR to int4-Marlin: the critical analysis

**Why DVR is structurally more robust than margin-gate for int4:**
DVR does not rely on the 56% flip rate being "survivable." It tolerates ANY flip rate
because it rolls back and recomputes every flip. The question is not whether the
argmax is stable — it is purely a throughput question: how much does 56% rollback rate
(at the speculative verify window boundary) cost?

**The math on rollback overhead with int4-Marlin:**
- At K=7 speculation, each speculation window has 7 candidate tokens
- If 56% of tokens flip at verify: expected ~3.9 of 7 candidates roll back
- Each rollback recomputes that token via M_verify=1 verify GEMM (deterministic)
- At 56% flip rate, effective accept length per speculation step ≈ 1 token
  (flip happens at first token most of the time → mean accepted = 1/flip_rate ~1.8)
- This degrades speculation benefit from K=7 mean ~3.5 to effective ~1.8 accepted tokens
- Wall-clock impact: speculation at K=7 with 56% rollback is roughly equivalent to
  speculation at K=3 with no rollback → ~15-25% TPS degradation vs current 481.53
- Estimated TPS with DVR + int4-Marlin: ~380-430 TPS — below the 500 target

**The 56% flip rate is the DVR killer for int4-Marlin at K=7 speculation.** The fp16
paper's 0.32-10.97% recompute rate works because fp16 single-GPU GEMM is much more
stable; int4-Marlin's split-K non-determinism at M=1 vs M=8 is a qualitatively
different regime.

**Mitigation path A: DVR + MarginGate hybrid**
- Apply MarginGate first: skip verify for 81-85% of positions (high-margin)
- Apply DVR for remaining 15-19% of positions (low-margin, potentially unstable)
- At 56% flip rate on the 15-19% low-margin subset: rollback applies to only
  56% × 15-19% = 8.4-10.6% of total tokens → ~same as the ArXiv worst-case (10.97%)
- Net overhead: manageable; net TPS: ~460-490

**Mitigation path B: DVR at reduced K**
- Reduce speculation from K=7 to K=3 before applying DVR
- Rollback rate at K=3 with 56% flip: expected ~1.7 rollbacks per step
- Mean accepted per step ≈ 1.3-1.5 tokens; system TPS ~350-380 — not viable

**Mitigation path C: DVR + quantization-stable drafter**
- Use the non-speculative but int4-stable AR decode for the "decode" step
- Use DVR's fixed-shape verify at M=1 as the commit gate
- Equivalent to single-step AR with DVR as a consistency checker
- Provides 100% greedy identity; no speculation benefit; ~165-200 TPS — not viable

### Implementation buildability assessment
- **Framework:** LLM-42 is built on SGLang v0.5.3. The challenge currently uses vLLM.
  Porting DVR to vLLM is a substantial engineering effort (custom scheduler, rollback
  KV cache management, fixed-shape forward pass injection).
- **KV cache rollback:** requires storing the KV cache state at each verify boundary
  and restoring on rollback. vLLM's paged attention KV cache management does not
  natively support fine-grained rollback. Significant adaptation needed.
- **Integration with MTP speculative decoding:** LLM-42's limitation list explicitly
  notes "no integration with speculative decoding frameworks." Combining DVR with
  the existing MTP K=7 pipeline would require a non-trivial protocol redesign.
- **Timeline risk:** high. This is a multi-week engineering project, not a few-day
  experiment. The research question (will 56% flip rate kill throughput?) is answerable
  cheaply but the full implementation is not.

### Verdict on DVR as the 4th structural path
DVR is theoretically the cleanest solution: it decouples "fast decode" from "deterministic
commit" and satisfies greedy identity by construction at M_verify=1. The fundamental
problem is that it was designed for fp16 single-GPU systems where the non-determinism
is small (sub-1% rollback rate typical) and the cost of rollbacks is low.

At int4-Marlin's 56% argmax flip rate, DVR becomes a rollback engine rather than an
optimistic fast-path. The throughput math does not clear 500 TPS unless paired with
MarginGate to pre-filter high-stability positions (Mitigation A), which is effectively
just Direction 1 (MarginGate) with DVR as the fallback for the last 15-19%.

**P(standalone DVR reaches valid >500 TPS with int4-Marlin):** 0.10
**P(DVR-as-fallback in MarginGate-DVR hybrid reaches valid >500 TPS):** 0.40
(see Direction 1 analysis — this is effectively the same as the MarginGate path)

---

## Direction 4: Mixed-Precision FP8 Verify (ELIMINATED)

### Status: DEFINITIVELY ELIMINATED for A10G (sm_86)

### Why eliminated
A10G has compute capability sm_86. Native FP8 tensor core computation requires sm_89+
(Ada Lovelace / Hopper). On sm_86:
- vLLM routes all FP8 requests through Marlin W8A16 (weight-only int8 emulation)
- The Marlin W8A16 path has IDENTICAL split-K non-determinism to int4-Marlin
- There is no fast FP8 batch-invariant path available on sm_86
- vLLM issue #40127: "fix: add SM>=89 guard for Triton block FP8 and Marlin fallback
  on Ampere" — confirmed in vLLM codebase
- vLLM GPU support matrix: "NVIDIA Ampere A100, A10G, A30, RTX 3090, A40 | sm_80/sm_86 |
  INT4/INT8 Only | Unoptimized FP8 Fallback — vLLM routes FP8 files through Marlin
  Kernels (running as W8A16 weight-only execution)"

**Do not pursue any FP8-based verify path on A10G.**

---

## Direction 5: Argmax Stability as Proxy for GEMM Bit-Identity (Theoretical Framework)

### What it is
A theoretical framework underpinning Direction 1: quantify the maximum possible
perturbation ε_max that int4-Marlin split-K non-determinism can introduce into any
given logit, then use (logit_top1 - logit_top2) > 2 * ε_max as a provable argmax
stability certificate.

### Mechanism
- int4-Marlin split-K: K dimension is tiled into S splits; reduction of partial sums
  has S! possible orderings (in practice a small number based on hardware mapping)
- Each int4-to-fp16 multiply-accumulate has bounded rounding error: δ = 2^-10 * |w * x|
  (BF16 mantissa precision)
- Over N weights in one row of the GEMM: ε_max ≤ N * max(|w|) * max(|x|) * 2^-10
  (loose upper bound; tighter bounds possible with group-wise scale tracking)
- In practice: empirically measured by running the same forward pass at M=1 and M=K+1
  and recording max|logit_diff| across a calibration corpus
- Empirical measurement from the existing verify_flip_probe experiments: 56% token flip
  rate implies median margin < 2 * ε_max at flipped positions, but also implies 44%
  of positions have sufficient margin even with current int4-Marlin noise

### Practical use
This framework enables:
1. **Calibrated τ for MarginGate** (Direction 1): set τ = 2 * ε_max_empirical; this
   gives a falsifiable, calibration-grounded threshold rather than a heuristic
2. **Identify the 44% already-stable positions**: even without any code change, 44%
   of speculative positions pass argmax identity despite split-K non-determinism. The
   current 56% flip rate is for the marginal (unstable) subset. MarginGate routes
   these to the deterministic fallback; the 44% stable positions need no special treatment
3. **Monitor threshold drift**: if model weights change or group size changes, ε_max
   changes; the calibration catches this drift before it manifests as a validity failure

### P(standalone Direction 5 reaches valid >500 TPS)
Direction 5 is a framework, not an independent TPS path. It enables Direction 1 and
quantifies the risk model for Direction 3 (DVR). It has no standalone TPS impact.

---

## Synthesis: Priority Rankings

### Rank 1 — MarginGate (Direction 1)
**One-line crux:** skip the verify GEMM for 81-85% of positions where the logit
margin provably exceeds split-K FP noise, and run bf16 M=1 verify for the rest.
**P(valid >500):** 0.45
**Build time:** 1-2 weeks (τ calibration + bf16 M=1 fallback integration)
**Key risk:** τ calibration for int4-Marlin not yet demonstrated; bf16 fallback
overhead may erode net gain below 500 TPS

### Rank 2 — MarginGate + DVR Hybrid (Directions 1 + 3 combined)
**One-line crux:** MarginGate filters the 81-85% stable positions; DVR with M_verify=1
handles the remaining 15-19% with guaranteed greedy identity and manageable rollback
rate (~8-11%).
**P(valid >500):** 0.40
**Build time:** 3-5 weeks (DVR port to vLLM + MarginGate integration)
**Key risk:** vLLM KV cache rollback is a substantial engineering lift; DVR integration
with MTP K=7 is explicitly unsupported in LLM-42

### Rank 3 — Fixed-Tile Verify GEMM (Direction 2) combined with MarginGate
**One-line crux:** replace only the verify-pass GEMM with a fixed-tile Triton kernel
(~20% overhead) to achieve batch-invariance for the low-margin positions, avoiding the
DVR KV-rollback engineering entirely.
**P(valid >500):** 0.35
**Build time:** 2-3 weeks (Triton kernel + integration)
**Key risk:** 20% verify GEMM overhead at 25-30% wall-clock fraction → ~5-6% system
overhead; may land at 455-480 TPS without MarginGate, or 490-510 with MarginGate

### Rank 4 — DVR Standalone (Direction 3)
**One-line crux:** pad verify to fixed M=1, accept any flip rate, roll back and
recompute mismatches; greedy identity by construction.
**P(valid >500):** 0.10
**Build time:** 3-5 weeks
**Key risk:** 56% int4-Marlin flip rate converts DVR from a fast optimistic path into
a heavy rollback engine; throughput math projects ~380-430 TPS, below target

### Eliminated — FP8 verify (Direction 4)
A10G sm_86 has no native FP8 tensor cores; falls back to Marlin W8A16 (same problem).
Do not pursue.

---

## Recommended Experiment Sequence

### Step 0 (Cheap diagnostic, 1-2 days): Measure ε_max empirically
Run the existing verify_flip_probe setup and record max|logit_diff| across 1000 inputs.
Also record the margin distribution (top1 - top2) at positions that DID flip vs DID NOT.
This data determines τ for MarginGate and confirms the 44% already-stable estimate.

Expected output: histogram of margins at flipped positions, value of ε_max, fraction
of positions with margin > 2*ε_max (should be ~44% from the 56% flip rate).

### Step 1 (2-3 days): MarginGate with bf16 M=1 fallback
Implement: for each speculative position, if margin > τ, accept draft argmax without
running verify GEMM; if margin ≤ τ, run a single-row bf16 GEMM (deterministic).
Measure: token flip rate should drop to 0% for properly calibrated τ; TPS measured
under the full speculation pipeline.

### Step 2 (1 week, parallel with Step 1): Fixed-tile Triton verify GEMM
Write a fixed-tile (tile_M=8, tile_K=group_size=128, fixed) Triton kernel for the
verify GEMM. Verify: batch-invariant argmax across M=1 and M=8. Measure overhead vs
cuBLAS on A10G.

If Step 1 alone clears 500 TPS with valid greedy identity: DONE.
If Step 1 lands at 490-499: combine with Step 2 (fixed-tile handles the fallback GEMM).
If Step 1 lands at <490: the bf16 fallback overhead is too large; consider reducing K
or using the fixed-tile Triton kernel instead of bf16 for the fallback.

### Stop condition
If MarginGate + best available deterministic fallback (bf16 M=1 or fixed-tile int4)
produces valid TPS < 480 after full implementation, the numerical non-determinism in
int4-Marlin is too large to overcome within the 7.33% overhead budget via any skip/gate
mechanism. At that point, the only remaining path is either a full custom batch-invariant
kernel (Path 3, +51.78% overhead, ruled out by budget math) or accepting the fp16 verify
penalty (~380-430 TPS, below 500).

---

## Confidence Assessment

**Direction 1 (MarginGate):** Strong external evidence (MarginGate paper, 81-85%
skip rate), plausible mechanism, unvalidated for int4-Marlin specifically. The 44%
already-stable fraction from our 56% flip rate data supports the framework. Overall:
"promising theory with partial empirical grounding from similar settings."

**Direction 2 (Fixed-tile GEMM):** Technically sound, verified at ~20% overhead in
similar Triton work. System-level TPS math is uncertain. "Reasonable but overhead
budget is tight."

**Direction 3 (DVR standalone):** Theoretically clean, definitively fails the overhead
math at 56% flip rate. LLM-42 paper confirms the mechanism works for fp16; int4 is an
extrapolation into a hostile rollback regime. "Strong external evidence of mechanism,
strong internal evidence it will not close the budget at 56% flip rate."

**Direction 4 (FP8):** Definitively eliminated by hardware capability.

**Direction 5 (Stability framework):** Theoretical; enables calibration of Direction 1.
No standalone TPS impact.
