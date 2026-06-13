# FINDING — Post-#43 decode re-profile: split-KV collapsed attention, drafter-forward is the new #2

_Submission_: `submissions/fa2sw_precache_kenyan` (int4-pck04 + MTP drafter K=7 + PLE-fold + fa2sw + onegraph + precache + **#43 split-KV verify, ACTIVE**)
_Workload_: conc=1, 128 official sharegpt prompts, output_len 512, **CUDA graphs ON**. Isolation variants `spec_off`/`lmhead_off` light 32×256.
_W&B_: run **`r0ahjs45`**, group `decode-reprofile-postsplitkv`. Diff baseline: #30 run `07kg6bn7` (pre-split-KV, `research/profiling/frontier_decode/`).

> **Local A10G exploratory probe — NOT the official a10g-small TPS.** Absolute tok/s are a single-GPU in-container probe; composition fractions and relative shifts are the trustworthy output. No-precache locally (same as #30 and the #43 A/B), so absolutes are ~9% depressed vs official; the **relative** shift vs #30 is the apples-to-apples result.

## Split-KV engagement confirmed (not a silent 2D no-op)

Server log: `[splitkv-verify] wrapped unified_attention (redirect 1<M<=64 verify batches to 3D split-KV)` then `verify batch M=8 q_rows=8 -> 3D split-KV (n=1..5)` (n capped by `SPLITKV_VERIFY_LOG=5`; redirect continues every verify step). **No** `redirect skipped` / `patch error` / 2D-fallback lines. `gpu_busy_share_of_wall_raw=99.4%`. This is the same patch-engaged check the #43 launch gate uses.

## Headline: the composition shifted exactly as hypothesised

Split-KV cut the **verify forward GPU −17.5%** (7.906→6.519 ms) — entirely from the M=8 verify-attention op, leaving body GEMM, drafter, lm_head untouched. Decode stays **99.4% GPU-bound**.

| quantity (p50 steady) | #30 pre-splitKV | **post-#43** | Δ |
|---|--:|--:|--:|
| drafter forward GPU (ms) | 1.446 | **1.445** | −0.001 |
| verify forward GPU (ms) | 7.906 | **6.519** | −1.387 (−17.5%) |
| GPU-busy / cycle (ms) | 9.352 | **7.964** | −1.388 (−14.8%) |
| inter-op gap / host (ms) | 0.064 | **0.047** | −0.017 |
| cycle wall (ms) | 9.416 | **8.011** | −1.405 (−14.9%) |
| E_accept (tok/cycle) | 3.817 | **3.847** | +0.030 |
| local steady gen TPS | 391.3 | **455.3** | +64.0 (+16.4%) |
| wall-clock token-wtd TPS | 391.2 | **454.4** | +63.2 (+16.1%) |

### De-duped decode GPU-time composition (% of GPU-busy + µs/step)

| block | #30 % | **post-#43 %** | Δ pp | #30 µs | **post-#43 µs** |
|---|--:|--:|--:|--:|--:|
| #1 verify body int4-Marlin GEMM | 53.2% | **60.6%** | +7.4 | 4971 | 4826 |
| **#2 drafter forward (K=7 MTP)** | 15.5% | **18.1%** | +2.7 | 1446 | **1445** |
| #3 verify attention (fa2sw) | **19.6%** | **7.6%** | **−12.0** | 1836 | **605** |
| #4 verify norm / elementwise | 6.7% | 7.5% | +0.8 | 623 | 595 |
| #5 sampling | 2.6% | 2.9% | +0.3 | 245 | 233 |
| #6 verify lm_head12k GEMV | 1.0% | 0.3% | −0.6 | 90 | 26 |
| inter-op gap / CUDA-graph launch | — | — | — | +64 | **+47** (host fully overlapped, ~0% of cycle) |

**Attention collapse: CONFIRMED.** 19.6% → **7.6%** of GPU-busy (1836 → 605 µs, a **3.03×** absolute drop). The predicted ~5–6% was slightly overshot to 7.6% because the in-serving blend is less than the op-level 4.38× sliding microbench: only the Triton-path verify layers (global head-512, KV-shared) get the full M=8→3D speedup; FA2 sliding-window layers and the already-3D M=1 decode attention don't move. Direction and magnitude hold.

**New #2 block: drafter-forward (K=7 sequential MTP passes), 18.1% of GPU-busy, 1445 µs/step.** It was #3 pre-split-KV; with attention gone it is now the largest *addressable non-GEMM* block. Body GEMM stays #1 — its **share** rose +7.4 pp purely because the GPU-busy denominator shrank (its absolute µs is flat, 4971→4826, within trace noise).

### Isolation: split-KV flattened verify cost(M) ~6×

| verify GPU (ms) | M=1 (spec_off) | M=8 (frontier) | slope M=1→8 |
|---|--:|--:|--:|
| #30 pre-splitKV | 6.326 | 7.906 | **+25.0%** |
| **post-#43** | 6.252 | 6.519 | **+4.3%** |

The +25% pre-split-KV slope **was** the M=8 attention under-occupancy (2D grid starving the A10G's 80 SMs). Split-KV removed it: verify is now **near-perfectly flat in M** (+4.3% for 8× the query rows). This is the central enabler for the K-raising levers below.

## Next lever — ranked, quantified (local steady cost model, cycle ≈ GPU-busy)

| lever | block hit | projected ΔTPS (local 455 →) | tractability |
|---|---|--:|---|
| **acceptance E_accept 3.85→4.5** (land #9 / fern #34) | none (multiplier) | **+17.0% → 533** | drafter training (cluster) |
| body GEMM −15% (sub-int4 / faster Marlin) | #1 (60.6%) | +9.9% → 501 | **hard** (new weight format) |
| **drafter W8A8 INT8 −30%** (stark #47) | **#2 (18.1%)** | **+5.7% → 481** | **tractable kernel lever** |
| drafter W8A8 INT8 −40% (stark #47) | #2 | +7.8% → 491 | tractable |
| attention −100% (ceiling, unreachable) | #3 (7.6%) | +8.2% → 492 | **de-prioritised** |
| denken #51 (Task 1) split-KV ctx-gate (M=8) | #3 attn | **~0% on served (cost-model penalty doesn't transfer)** | **NOT supported by served data** |

### In-flight-lever cross-check (does post-#43 composition SUPPORT or DE-PRIORITISE?)

- **stark #47 (W8A8 INT8 draft-forward) — SUPPORTED ↑↑.** The re-profile *promotes* this: drafter-forward is now the **#2 block (18.1%, 1445 µs)**, up from #3. The drafter is currently **BF16** (49 BF16 tensors in `/tmp/qat-assistant`), and its K=7 passes are bandwidth-bound GEMVs — exactly what W8A8 halves the weight stream on. At a conservative −30% it returns **+5.7% TPS**; this is the single highest-leverage *kernel* lever after the (hard) body GEMM.
- **denken #51 — two facets, split verdict.**
  - *(a) Dynamic-K consequence — SUPPORTED ↑ (premise), bounded by drafter.* Split-KV made verify cost(M) nearly flat (**+4.3%** M=1→8 vs +25% before), so *raising K is now almost free on the verify side*. BUT the cost that grows with K is now the **drafter** (#2 block, ~linear in K). So the K-raising headroom is real but capped by drafter cost — **complementary with stark #47**: a cheaper drafter directly buys deeper-K headroom.
  - *(b) Task-1 split-KV context-gate — NOT supported on the served stack.* denken's M=8 cost-model curve says split-KV is net-negative at every served ctx (+15.5%@256 → +5.1%@1024), but my served re-profile + the #43 served A/B measure **−17.5%** (a win) over exactly that ctx distribution. Quantified cross-check in its own section below; the cost-model M=8 penalty does not transfer to the CUDA-graph-captured served path.
- **land #9 / fern #34 (drafter acceptance) — SUPPORTED ↑ (still dominant).** TPS = E_accept / cycle; acceptance is a linear multiplier at **zero per-step cost**, so it remains the single highest-leverage lever overall (E_accept 3.85→4.5 = +17%). Split-KV is orthogonal (it cut per-step cost); the two **multiply**.

**Recommendation.** The dominant TPS lever is unchanged — **acceptance (land #9 / fern #34)**. The re-profile's new contribution is the **#2 per-step block: drafter-forward**, which makes **stark #47 (W8A8 INT8 drafter, +5.7–7.8%)** the top tractable kernel lever, and — because split-KV flattened verify cost(M) — makes **stark #47 + denken #51 dynamic-K complementary** (cheaper drafter ⇒ cheaper deep-K). **De-prioritise further attention work**: split-KV already captured it (7.6%, ceiling +8.2%). **The advisor-requested denken #51 (Task 1) split-KV context-gate is NOT supported by the served data** (next section): the cost-model M=8 short-ctx penalty predicts +9.7% slower over this run's contexts, but the served stack measured −17.5% faster — a gate would forfeit a measured win.

## denken #51 (Task 1) split-KV context-gate — evaluated on the served stack: NOT supported

The advisor (PR #53 comment, 2026-06-13) asked me to fold in denken #51's cost-model finding that the #43 split-KV redirect is **net-negative at small-M / short-ctx, net-positive at long ctx**, and to evaluate a **`seqlen_kv ≥ threshold` context-gate** (explicitly *not* an `M≥33` gate — the deployed linear MTP K=7 stack is always M=8) that would recover the predicted early-decode penalty. I reproduced denken's curve (`scripts/profiler/compare_splitkv_curves.py --operating-M 8` on the merged `research/spec_cost_model/results_pr51_*` JSONs) and cross-checked it against this run's **real** per-cycle context distribution.

**denken's M=8 cost-model verify Δ% (split-KV ON vs OFF), reproduced:**

| M=8 verify Δ% | ctx256 | ctx512 | ctx1024 |
|---|--:|--:|--:|
| split-KV vs baseline | **+15.5%** | **+8.0%** | **+5.1%** |

At **M=8** the redirect is *slower at every measured ctx* — on only 8 query rows the `reduce_segments` merge + extra CTA launches exceed the small attention saving; the crossover to net-positive never occurs in the served range. (M=45 cross-check reproduces denken's headline exactly: −2.6 / −6.3 / −7.1% @ctx256/512/1024 → the win is an **M≥33 / tree-path** effect, not a linear-M=8 effect.)

**This run's served ctx distribution** (128 prompts, prompt-len median 230 / mean 272, output_len 512 → ~133 cycles/prompt, **17,024 verify cycles**):

| seqlen_kv regime | % of served verify cycles | denken M=8 says |
|---|--:|---|
| ctx < 256 | 8.2% | +15.5% slower |
| ctx < 512 | **52.0%** | +8.0–15.5% slower |
| ctx ≥ 1024 | 2.6% | +5.1% slower |

The majority of decode (**52% of cycles below ctx512**) sits in denken's predicted M=8 penalty band. Cost-weighting denken's M=8 Δ%(ctx) by this exact distribution:

> **cost-model PREDICTS aggregate verify ≈ +9.7% SLOWER; SERVED MEASURED = −17.5% FASTER. Gap = −27.2 pp.**

The served stack (this re-profile + the #43 served A/B, both p50 over all steps) shows a **clean −17.5% verify win over exactly the ctx range the cost model says should be +9.7% slower**. A cost-model-honest `seqlen_kv` gate would fire **~never at M=8** (the M=8 curve is positive at all served ctx) — equivalent to split-KV OFF — and would **forfeit the measured win, not recover a penalty**.

**Why the gap (most likely):** the cost model is a synthetic verify-only microbench — its M=8 step is ~1.6× the served verify in absolute terms (11.7–12.8 vs 6.5–7.9 ms) and its M=8 attention fraction (17.6–30.2%) is far above the served verify-attention (7.6% of GPU-busy), so it over-weights both the attention the redirect targets and the segment-reduction overhead. The served path is CUDA-graph-captured (`ONEGRAPH`), 99.4% GPU-bound with host fully overlapped, so the extra-CTA-launch / `reduce_segments` overhead denken flags is captured in-graph and overlapped rather than charged to wall time.

**Verdict: do NOT add the context-gate on current evidence.** The #53 served re-profile directly measures the deployed M=8 stack across denken's flagged ctx range and finds a win. **Definitive test (follow-up):** a ctx-resolved served A/B — log per-step `(seqlen_kv, verify_ms)` in both `SPLITKV_VERIFY=0/1` arms and bin Δ%(ctx); if that *served* curve shows an early-ctx penalty, a gate is warranted and the served crossover can be located. Until then, leave split-KV unconditionally ON. Reproduce: `ctx_gate_analysis.py`.

## Validity / provenance

- **Read-only profile. Zero submission-file changes** (`git status`: only `research/profiling/frontier_decode_postsplitkv/` added). PPL / greedy / serving definitionally untouched. #43 already validated PPL **2.3767 ≤ 2.42** and greedy-equivalence-to-baseline for this exact stack.
- Peak GPU mem ~19.4 GiB observed (model load 8.85 GiB, KV 9.46 GiB; budget 0.90×23 = 20.7 GiB; #43 peak 21.0 GiB).
- Public anchor (digest, 2026-06-13): our base `kenyan-duma` osoi5 stack = leaderboard rank 7 (~421 TPS official, pre-split-KV); #1 VALID rock-ai 459.72 (accepthist+cap); the top-4 all use accepthist+cap, corroborating the land #9/fern #34 + denken #51 lanes.
- Projections are LOCAL-steady estimates on the team cost-model scale (drafter≈1.4 ms, verify≈6.5 ms post-split-KV). Official a10g-small absolutes differ; relative gains are the robust quantity.

## Reproduce

```bash
cd target/
python -m scripts.local_validation.profile_decode \
    --submission submissions/fa2sw_precache_kenyan \
    --num-prompts 128 --output-len 512 \
    --variants frontier,lmhead_off,spec_off \
    --wandb-name wirbel/decode-reprofile-postsplitkv-128 \
    --wandb-group decode-reprofile-postsplitkv \
    --out-dir research/profiling/frontier_decode_postsplitkv/
python research/profiling/frontier_decode_postsplitkv/analyze_diff.py   # diff vs #30

# denken #51 (Task 1) context-gate cross-check (reads denken's merged curve JSONs)
python scripts/profiler/compare_splitkv_curves.py \
    --baseline research/spec_cost_model/results_pr51_baseline.json research/spec_cost_model/results_pr51_baseline_longctx.json \
    --splitkv  research/spec_cost_model/results_pr51_splitkv.json  research/spec_cost_model/results_pr51_splitkv_longctx.json \
    --operating-M 8
python research/profiling/frontier_decode_postsplitkv/ctx_gate_analysis.py
```
