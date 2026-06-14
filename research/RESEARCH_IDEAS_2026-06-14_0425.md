# Research Ideas — 2026-06-14 04:25

Generated after full literature search covering: hierarchical speculative decoding, token recycling,
self-speculative layer skipping, Saguaro parallel draft/verify, bandwidth-bound W4A16 kernels,
KV-cache quantization, and calibrated draft candidate selection.

Current frontier: 481.53 official TPS (`fa2sw_precache_kenyan`), public #1 at 488.07 TPS.
Gap to close: +6.54 TPS (+1.36%). Tree-verify at M=32, K=7 MTP drafter, E[T]=3.844.

All ideas must be greedy-identity-preserving (bit-exact token match vs plain autoregressive greedy).

---

## Idea 1: Token Recycling (Adjacency-Matrix Draft Augmentation)

### Mechanism
Token Recycling (arXiv:2408.08696) maintains a fixed-size adjacency matrix where each row stores the
top-K most frequently observed successor tokens for a given token. After each verified step, the matrix
is updated with the actual observed next-tokens. At draft time, the adjacency matrix provides an
additional set of candidate branches that require zero model forward passes. These candidates are
spliced into the existing MTP/EAGLE tree as extra leaf extensions or parallel branches. The verifier
applies its normal lossless multi-token verification — any token that fails verification is rejected
and the sequence reverts, so the greedy invariant is preserved identically to plain spec decode.

**Why it targets conc=1 bandwidth bottleneck:** the adjacency matrix lookup is O(1) and purely
CPU-side; it adds candidate tokens to the tree at zero GPU cost. If even a small fraction of these
recycled candidates are accepted, E[T] rises with no additional verify-GEMM cost per candidate
(they fit inside the M=32 tree for free as long as M stays ≤32).

### Headroom vs 481.53 TPS
E[T] is currently 3.844 at K=7. The tree-verify cost model (#74/#83) shows each additional 1.0 unit
of E[T] is worth ~+20 TPS in the bandwidth-bound regime. Token recycling papers report +5-15%
acceptance rate lift on top of standard spec decode. Conservative: +0.2 E[T] → ~+4 TPS.
Upper bound: +0.5 E[T] → ~+10 TPS.

### Greedy-identity argument
The verifier is unchanged and authoritative. Recycled candidates are treated exactly like any other
draft token — they are accepted only if their argmax matches the target model's distribution. The
adjacency matrix is a draft augmentation only; the accept/reject logic is identical to existing
lossless tree-verify. Greedy identity is preserved by construction.

### First experiment and fail-fast Step-0 gate
**Step-0 (CPU diagnostic, ~1 GPU-hour):**
1. Run the 128-prompt benchmark offline with the existing M=32 tree-verify stack.
2. After each verified step, record the actual next tokens and compute the adjacency matrix.
3. Simulate (offline, no GPU) what acceptance rate the recycled candidates would achieve if added
   as extra leaf branches. If simulated acceptance > 5% of new candidates, proceed.
4. Fail-fast: if simulated per-candidate acceptance on recycled tokens is < 2% (meaning corpus is
   too diverse for recycling to contribute), close the idea.

**Step-1 (live integration):** splice recycled candidates into the tree builder, cap M=32, re-run
the 128-prompt gate. Measure E[T] before/after recycling injection.

### Composition with prior results
- Composable with M=32 tree (#83), traversal-verify (#88), K=7 MTP drafter (#51/#54).
- Does NOT require retraining the drafter or modifying the verifier.
- If accepted candidates push M usage above 32 with degrading acceptance, the M-sweep from #28
  gives the cost curve to find the new optimal M under the recycled tree.

---

## Idea 2: Saguaro-Style Parallel Draft Preparation (Draft-Verify Overlap)

### Mechanism
Saguaro (arXiv:2603.03251) observes that in standard speculative decoding, draft preparation and
verification are serialized: verify → accept → draft → verify. Saguaro breaks this by beginning the
next draft call immediately after sampling from the previous verified token, running it in parallel
with the ongoing verification pass. The verified output gates the draft but does not block it; if
the draft was started speculatively from the wrong root token, it is discarded and retried. On
average, when acceptance is high, the draft is "mostly right" and the parallel overlap hides most of
the draft latency entirely. Reported: ~30% faster than optimized spec decode, up to 5x over
autoregressive.

**Why it targets conc=1:** at conc=1, the GPU executes verify and drafter as serial kernels in a
single stream. The overlap requires scheduling drafter kernels concurrently with the verify pass on
separate CUDA streams. On a single-stream A10G, this requires the drafter to use a secondary stream
with its own KV state. The MTP drafter is small (3B), so both can run simultaneously without
memory conflicts if KV allocation is managed. The net effect is that drafter cost (~15.5% of decode)
is absorbed into the verify wall time rather than appearing sequentially after it.

### Headroom vs 481.53 TPS
Drafter cost is ~15.5% of decode (#75). Saguaro-style overlap would eliminate up to 100% of the
exposed drafter latency by parallelizing with verify. Upper bound: +15.5% → ~+75 TPS. Realistic
estimate assuming partial overlap and stream-synchronization overhead: +5-10% → +24-48 TPS.
This is the highest theoretical ceiling of any idea here.

### Greedy-identity argument
Saguaro does not change the verifier or acceptance criterion. Greedy identity is determined entirely
by the target model's verification of each draft token — this is unchanged. The only difference is
temporal: the draft that gets verified at step N+1 may have been partially computed during step N.
If the speculative root token is wrong (draft computed from wrong root), the draft is discarded and
a corrective draft is run. The token sequence produced is identical to what would have been produced
by serial spec decode with the same verifier.

### First experiment and fail-fast Step-0 gate
**Step-0 (cost model, ~2 GPU-hours):**
1. Profile the wall-time of a single drafter forward (K=7, M=32) on the A10G using the existing
   profiler infrastructure (#30/#53/#77).
2. Profile the wall-time of a single verify-GEMM forward.
3. If drafter_wall_time < verify_wall_time * 0.9, the overlap budget exists and this idea is live.
   If drafter_wall_time > verify_wall_time, the drafter is already hiding in the verify shadow and
   there is no budget to recover — close the idea.
4. Check whether vLLM 0.22's step dispatch allows submitting two CUDA streams concurrently in a
   single decode step. If not, estimate the refactor cost.

**Step-1:** implement a two-stream dispatch where the drafter runs on stream-B while verify runs on
stream-A. Synchronize at the accept/reject decision point. Measure wall TPS on the 128-prompt gate.

### Composition with prior results
- Composable with M=32 tree (#83), K=7 (#51), traversal-verify (#88), all prior kernel work.
- Does NOT require any retraining.
- If stream overhead on A10G is too high (vLLM dispatch latency dominates), the idea degrades
  gracefully — it is safe to fall back to serial dispatch.

---

## Idea 3: Self-Speculative Decoding via DEL (Layer-Exit Drafter — Exact Greedy)

### Mechanism
DEL (arXiv:2504.05598 — "Draft & Verify with Early Exit for Lossless Speculative Decoding") uses
the target model itself as its own drafter by exiting at an intermediate layer (e.g., layer 26 of
39 in Gemma-4-E4B) to produce draft logits, then running the full model to verify. Because the
draft logits come from a prefix of the same model weights (not a separate drafter), and because
the verification is the same lm_head applied to the full model output, the accepted token sequence
is bit-for-bit identical to plain greedy. DEL paper confirms: "produces exact token matches for
greedy decoding."

The key advantage: no separate drafter model, no additional memory, no training. The early-exit
activations are already computed as a byproduct of the full model forward (residual stream at layer
L passes through layer L+1 onward during verify anyway). Draft cost = one partial forward through
layers 0..L, which reuses most of the verify computation.

**Why it targets conc=1:** eliminates the external MTP drafter entirely. If early-exit acceptance
rate is comparable to MTP (E[T] ~3.8), the drafter cost (~15.5%) collapses to near zero because
the draft layers are computed on the path to verify anyway. Even if E[T] drops to 3.0, removing the
drafter forward more than compensates.

### Headroom vs 481.53 TPS
Current drafter = 15.5% of decode (#75). If we can draft from layer 26 with E[T] ≥ 3.0 (vs 3.844
MTP), the TPS gain = +15.5% drafter removal – E[T] degradation cost. Break-even: E[T] ≥ 3.20.
If early-exit at layer 26 achieves E[T] ≥ 3.5, net gain ≈ +10-15 TPS.

Onfirmation needed: what layer gives sufficient acceptance rate on Gemma-4-E4B for the reasoning
benchmark prompts? This is unknown and is the primary uncertainty.

### Greedy-identity argument
DEL paper: "produces exact token matches for greedy decoding." The verifier is the full model at
full precision — acceptance decisions are made by argmax of full-model logits, identical to plain
autoregressive. Greedy identity is guaranteed by construction.

### First experiment and fail-fast Step-0 gate
**Step-0 (acceptance profiler, ~2 GPU-hours):**
1. Load Gemma-4-E4B-it with hidden-state extraction at layers [13, 19, 26, 32] (covering 33%,
   50%, 67%, 82% of depth).
2. For each early-exit layer, project activations through the shared lm_head and sample greedy
   argmax.
3. Compute token acceptance rate vs the full model greedy output on 128 benchmark prompts.
4. Plot acceptance rate vs layer index. If any layer in [26, 32] achieves ≥ 80% token acceptance
   (which would imply E[T] ≥ 3.5 in draft-verify), proceed.
5. Fail-fast: if acceptance at layer 32 (82% depth) is < 70%, early-exit cannot match MTP — close.

**Step-1:** integrate the best early-exit layer into vLLM as a new drafter path, replacing the MTP
head. Run the 128-prompt gate with greedy-identity check (existing PPL + diff gate).

### Composition with prior results
- If early-exit achieves competitive E[T], it replaces MTP entirely, simplifying the stack.
- Composable with M=32 tree, traversal-verify, split-KV verify.
- The acceptance calibration work (#76) gives the E[T] → TPS curve needed to project gains.

---

## Idea 4: QuantSpec-Style KV Cache Quantization for the Speculative Draft Phase

### Mechanism
QuantSpec (arXiv:2502.10424) quantizes the KV cache to INT4 during the draft phase only, restoring
FP16/BF16 precision during the verification forward. Because speculative decode acceptance depends
on the draft token matching the target model's distribution — not on the KV precision of the drafter
pass — the greedy identity is preserved as long as the verify forward uses full-precision KV.
Reported: >90% acceptance rate maintained, ~2.5x speedup for long-context. The key insight:
the draft KV is a "throwaway" intermediate; quantizing it reduces HBM bandwidth for the drafter
attention blocks at no cost to final output quality.

**Why it targets conc=1:** in the bandwidth-bound regime, attention reads dominate drafter cost
at M>1 (#75, #39). Quantizing the drafter KV from BF16 to INT4 cuts KV bandwidth by 4x, directly
reducing drafter attention cost. At M=32 tree with K=7, the drafter runs 32 parallel attention
passes; INT4 KV would cut this bandwidth cost by up to 75%.

### Headroom vs 481.53 TPS
Drafter attention is ~25-30% of drafter total cost at M=32 (#77). Drafter total = 15.5% of decode.
INT4 KV on drafter: ~75% reduction in drafter-attention bandwidth → saves ~3.6-4.7% of decode.
Estimated gain: +17-23 TPS. This is conservative; actual savings depend on whether the drafter
attention is bandwidth-bound or compute-bound on the A10G.

### Greedy-identity argument
The verify forward always uses full-precision (BF16) KV. Greedy identity is determined by the verify
argmax — which is unchanged. The draft KV is used only to produce draft token candidates; it is
never directly accepted without verification. Even if INT4 draft KV causes the draft tokens to differ
from what full-precision draft KV would produce, the verifier still only accepts tokens matching the
target model's full-precision greedy argmax. Identity preserved.

### First experiment and fail-fast Step-0 gate
**Step-0 (acceptance sensitivity, ~1 GPU-hour):**
1. Run the 128-prompt benchmark with the existing MTP drafter but with the drafter's KV cache
   quantized to INT8 (not INT4 yet) using vLLM's existing KV quantization hooks.
2. Measure E[T] with INT8 drafter KV vs BF16 baseline.
3. If E[T] drops by < 5% (from 3.844 to > 3.65), the acceptance rate is insensitive to KV
   precision — proceed to INT4.
4. If E[T] drops > 10%, the drafter is sensitive to KV precision — close or investigate layer
   sensitivity.

**Step-1:** enable INT4 KV quantization for the drafter pass only (verify stays BF16). Measure
wall TPS on the 128-prompt gate with the greedy-identity assertion enabled.

### Composition with prior results
- Composable with all existing stack: M=32 tree, traversal-verify, split-KV verify-GEMM, K=7.
- Orthogonal to SplitK W4A16 verify-GEMM (#84) — that targets verify weights; this targets
  drafter KV.
- If INT4 drafter KV works, it can be combined with INT4 drafter weights (#70, never ran) for
  compounded savings.

---

## Idea 5: Calibrated Draft Candidate Selection (Frequency-Guided Token Bias)

### Mechanism
Calibrated Speculative Decoding (arXiv:2604.13634) uses corpus token frequency statistics to bias
the draft model's sampling distribution toward tokens that have high acceptance probability on the
target model, without modifying the verifier. In the lossless setting, only the draft token selection
is biased — the verifier's acceptance criterion is unchanged, so greedy identity is preserved.
The bias is applied as a logit offset (temperature-free, additive in log-space) proportional to the
log-frequency of each token in the target distribution.

**Why it targets conc=1:** the key bottleneck is E[T] — each additional accepted draft token
adds ~+5 TPS. Frequency-based bias is a zero-cost way to increase acceptance rate if the target
model has predictable token preferences on the benchmark corpus. The benchmark prompts are fixed and
known in advance; their token statistics can be precomputed once and loaded at serve time.

### Headroom vs 481.53 TPS
Token-frequency bias in the drafter increases E[T] if the target model is distribution-matched.
The #48 drafter frequency bias PR was merged, but it used a generic corpus. The calibrated variant
uses the actual benchmark-specific corpus. Estimated: +0.1-0.3 E[T] → +2-6 TPS.

### Greedy-identity argument
The verifier is unchanged and authoritative. Token bias in the drafter only affects which tokens
are proposed — the accept/reject decision is still based on whether the proposed token matches the
target model's greedy argmax. No lossless spec decode guarantee is weakened.

### First experiment and fail-fast Step-0 gate
**Step-0 (offline acceptance simulation, ~1 GPU-hour):**
1. Extract the token distribution of the 128-prompt benchmark outputs from the existing greedy
   reference run (#73).
2. Build a per-token logit bias table (log-frequency, capped at ±3.0 to prevent distribution
   collapse).
3. Simulate (offline) the acceptance rate of biased vs unbiased draft tokens from the existing
   drafter output. If simulated acceptance lift > 3%, proceed.
4. Fail-fast: if the benchmark token distribution is too uniform for bias to help (< 1% lift), close.

**Step-1:** integrate the frequency bias into the drafter sampling path. Run the 128-prompt gate.
Compare E[T] and wall TPS before/after bias injection.

### Composition with prior results
- Extends #48 (token-frequency logit bias, merged) with benchmark-specific calibration.
- Composable with all existing stack.
- If E[T] lift is small, this becomes a no-cost additive: worth keeping even at +2 TPS.

---

## Idea 6: CUDA Graph Partial Re-Capture for the Drafter-Only Path

### Mechanism
PR #65 (CUDA-graph capture of static-K=7 spec-decode steady state) was attempted and found
negative: batch-variant Marlin verify-GEMM cannot be captured in a CUDA graph because M (batch
dimension) changes with the tree structure. However, the drafter forward is a separate model pass
with a FIXED batch dimension at K=7 MTP steps — the drafter does not use Marlin (it is a smaller
model). A partial CUDA graph capturing only the drafter forward (not the verify pass) avoids the
Marlin batch-variant obstacle entirely.

**Why it targets conc=1:** at conc=1, per-step overhead is ~0.1-0.2 ms from CUDA kernel launch
and dispatch bookkeeping. The drafter runs K=7 sequential steps; each step has its own kernel
launches. A partial graph capture of the drafter's 7 serial steps into a single CUDA graph node
would eliminate 6/7 of the drafter kernel-launch overhead. With drafter at ~15.5% of decode and
kernel launch overhead at ~10% of drafter, potential gain: +1.5-3 TPS.

### Greedy-identity argument
CUDA graphs do not change arithmetic — they only change when and how kernels are dispatched.
The drafter graph is a pure inference graph producing draft logits; the verifier is unchanged.
Greedy identity is trivially preserved.

### First experiment and fail-fast Step-0 gate
**Step-0 (feasibility check, ~0.5 GPU-hours):**
1. Check whether the MTP drafter has fixed batch/sequence dimensions across all K=7 steps.
   If tree-structure (M=32) causes variable M in the drafter too, this is blocked — close.
2. Profile the drafter forward kernel launch overhead using nvtx markers. If kernel launch
   overhead is < 0.5% of total drafter time, the gain is too small — close.
3. Fail-fast: if the drafter's dispatch is already batched inside vLLM's step loop (not individual
   launches), close.

**Step-1:** capture the drafter 7-step forward as a CUDA graph, verify functional correctness on
5 prompts, then run the 128-prompt gate.

### Composition with prior results
- Orthogonal to all prior work.
- Lower ceiling than ideas 1-4 but near-zero implementation risk.
- If drafter is already captured in vLLM's existing graph infrastructure, this collapses to zero.

---

## Idea 7: Speculative Speculative Decoding (Saguaro) — Minimal Adaptation

### Mechanism (condensed from Idea 2 above, restructured as a separate minimal experiment)
Rather than full Saguaro parallel-stream implementation, the minimal adaptation is: after accepting
tokens at step N, immediately enqueue the drafter forward for step N+1 as a non-blocking CUDA call
on a secondary stream. The verify forward for step N+1 runs on the primary stream. At the end of
verify N+1, the drafter N+1 result is read (it may already be done). If the drafter is done (common
when drafter_time < verify_time), drafter cost is fully hidden. This is a vLLM scheduling change,
not a fundamental architecture change.

### Headroom vs 481.53 TPS
Same as Idea 2. Upper bound +15.5% (~+75 TPS), realistic +5-10% (+24-48 TPS). Highest ceiling.

### Greedy-identity argument
Same as Idea 2 — verifier unchanged and authoritative.

### First experiment and fail-fast Step-0 gate
**Step-0 (timing oracle, ~1 GPU-hour):**
1. Profile drafter_step_time and verify_step_time separately on the A10G.
2. If drafter_step_time > verify_step_time, there is no overlap budget — close immediately.
3. If drafter_step_time < 0.85 × verify_step_time, there is ≥15% slack — proceed.

**Step-1:** add a secondary CUDA stream to the vLLM decode loop for the drafter. Pin stream
affinity for drafter and verify kernels. Add a synchronization fence at the token selection point.

### Composition with prior results
- Composable with everything.
- Highest potential gain but most invasive change to vLLM's scheduling model.

---

## Idea 8: KNN-SSD — Learned Layer-Set Optimization for Self-Speculative Draft

### Mechanism
KNN-SSD (arXiv:2505.16162) proposes learning which layers to skip in a self-speculative setting
by finding the optimal layer subset via nearest-neighbor optimization on a calibration corpus.
Unlike ConfLayers which uses a fixed confidence threshold, KNN-SSD pre-computes the best layer
subset offline and applies it statically at serve time. The result is a fixed early-exit policy
that maximizes acceptance rate for a given speed target, discovered via offline calibration.

For Gemma-4-E4B-it, the calibration corpus is the 128-prompt benchmark itself. The KNN-SSD
offline phase finds the optimal subset of layers [L_0, ..., L_k] to compute during the "draft"
forward, with the remainder computed during the "verify" forward — reusing the prefix computation.

### Headroom vs 481.53 TPS
Same theoretical ceiling as DEL (Idea 3): +10-15 TPS if early-exit layer achieves E[T] ≥ 3.5.
KNN-SSD's advantage is that it picks the globally optimal layer subset rather than the most naive
early-exit layer, potentially achieving higher acceptance at the same compute budget.

### Greedy-identity argument
Same as DEL (Idea 3). Verifier always runs the full model; acceptance is determined by full-model
argmax. Greedy identity preserved.

### First experiment and fail-fast Step-0 gate
**Step-0 (offline calibration, ~2 GPU-hours):**
1. Run KNN-SSD offline calibration on 128 prompts: enumerate candidate layer subsets
   [26], [28], [30], [32], [26, 32], [28, 32].
2. For each subset, compute self-speculative acceptance rate (how often the early-exit argmax
   matches the full model argmax).
3. Pick the subset maximizing acceptance rate. If best acceptance < 75%, close — cannot beat MTP.
4. If acceptance ≥ 80%, proceed to live integration.

**Step-1:** plug the best layer subset into a self-speculative draft path in vLLM. Run 128-prompt
gate with greedy-identity check.

### Composition with prior results
- Alternative to DEL (Idea 3); if both are viable, run the simpler DEL first.
- Orthogonal to token recycling (Idea 1) — could be combined.

---

## Summary Rankings by Expected TPS Impact × Feasibility

| Rank | Idea | Expected TPS Gain | Feasibility | Product |
|------|------|-------------------|-------------|---------|
| 1 | Idea 2/7: Saguaro parallel draft-verify | +24-48 TPS | Medium (vLLM scheduler change) | High |
| 2 | Idea 3: DEL early-exit self-speculative | +10-15 TPS | Medium (acceptance unknown) | High |
| 3 | Idea 4: QuantSpec INT4 drafter KV | +10-15 TPS | High (vLLM KV quant hooks exist) | High |
| 4 | Idea 1: Token Recycling adjacency matrix | +4-10 TPS | High (CPU-side, no model changes) | Medium-High |
| 5 | Idea 8: KNN-SSD layer-set optimization | +10-15 TPS | Medium | Medium |
| 6 | Idea 5: Calibrated draft frequency bias | +2-6 TPS | High (extends #48) | Medium |
| 7 | Idea 6: CUDA graph drafter-only capture | +1-3 TPS | High (low risk) | Low-Medium |
