# PR #642 — De-project the #636 recompute acceptor (stark)

## Goal
Turn #636's **projected** `rescued_wall_tps_projected=139.20` (+10.1% over the 126.378
locked AR rung) into a **real served local wall-TPS** for the gap-flagged M=1-recompute
acceptor, and decide:
- `DEPROJECT_HOLDS` — real local ratio supports >126.378 official → open HF-Job approval issue.
- `DEPROJECT_ERODED` — overhead shrinks the margin to marginal.
- `DEPROJECT_DEAD` — acceptor not faster than the AR rung locally.

## Why the #636 number is a projection (the optimism to de-project)
#636 projected `rescued_wall_tps = 1/(1/152.291 + ftr/126.378)`:
1. **Composition-additive**: assumes a recompute adds cleanly to the amortized spec loop —
   ignores the gap-flag branch, the width-1 recompute that can't batch with the M=8 verify,
   and the CUDA-graph break / serialization (spec speed is loopgraph amortization).
2. **Wrong recompute checkpoint cost**: used `1/126.378` (the **int4_g128_lmhead** forward) as the
   per-recompute cost. The real recompute is a **w4a16-ct** target width-1 forward (g32 + bf16
   lm_head) which is **slower** than int4_g128_lmhead → the projection under-counts the recompute.
3. **Teacher-forced, drafter-not-loaded, offline** trajectory.

## Stack
- Option-B = `submissions/int4_mtp_batchinv`: target `google/gemma-4-E4B-it-qat-w4a16-ct` (cached) +
  Gemma4-MTP drafter `gemma-4-E4B-it-qat-q4_0-unquantized-assistant`, BI=1.
  NOTE: manifest ships `NUM_SPECULATIVE_TOKENS=6` (K=6/M=7); PR says K=7/M=8 — **flag to advisor**;
  default to K=6 (the config the 152.291 anchor was measured at) and note it.
- AR rung (b) = `submissions/int4_g128_lmhead` (126.378 official). Weights present at
  `/workspace/gemma_build/int4_g128_lmhead`.
- Benchmark harness: `scripts/profiler/paired_tps_ab.py` (validated #623 wall_tps tool: median-of-N
  fresh serves, conc=1, output_len 512, 128 prompts, `wall_tps = completion_tokens/decode_s`).

## Measurements (same local A10G, same GPU, same warmup)
- **(c) un-rescued Option-B** — int4_mtp_batchinv, BI=1, acceptor OFF. Reproduce 152.291 local ceiling.
- **(b) AR rung** — int4_g128_lmhead local wall_tps. Pairs with official 126.378.
- **(d) w4a16-ct M=1 AR** — int4_mtp_batchinv + SENPAI_REFERENCE_MODE=1 (spec off). The **true**
  per-recompute forward cost (corrects #636's wrong checkpoint).
- **(a) recompute-acceptor** — int4_mtp_batchinv + env-gated acceptor firing **real** width-1 target
  recompute forwards at the per-output-token gap rate. The de-projected headline.
- **slope sweep** — inject recomputes at rate r∈{0,0.05,0.10,0.20}; fit real per-recompute in-loop
  marginal cost C; compare to #636's assumed 1/126.378 and to arm (d).

### Decisive ratios
- `acceptor_over_ar_ratio = a/b` → `× 126.378` = projected official TPS of the acceptor.
- `acceptor_over_unrescued_ratio = a/c` → the acceptor's pay-down from the spec ceiling.

## τ optimization
Pick τ maximizing wall-TPS while holding `rescued_break_rate=0`; report real `flag_trigger_rate`.
(τ↑ → more flags → slower but safer; τ↓ → fewer flags but risk a missed seed. #636: min τ for 0
breaks = 0.3 fine / 0.5 coarse; ftr 5.5% / 7.8%.)

## Identity de-teacher-force (instruction 3)
#636 built its AR trajectory **offline** (`vllm.LLM`). De-teacher-force:
- Serve the **M=1 AR reference** via the spec-off API (SENPAI_REFERENCE_MODE=1) → real served R_served.
- Serve the **un-rescued Option-B** → S_served; measure the **real served cascade** break_rate(S_served vs R_served).
- Run the validated #381/#622/#636 chunk-read M=8 verify scan **along R_served** → `rescued_break_rate(τ)`,
  `flag_trigger_rate(τ)` on the **real served** trajectory. rescued_break_rate=0 ⇒ acceptor holds on
  real generation. (M=8 spec-verify numerics are unavailable from the API, so the verify stays an
  offline reconstruction over the served trajectory — this is the de-teacher-forceable part.)

## Deliverable
SENPAI-RESULT: `rescued_wall_tps_real_local`, `acceptor_over_ar_ratio`, `acceptor_over_unrescued_ratio`,
`break_rate_served`, `optimal_tau`, `flag_trigger_rate_real`. Verdict ∈
{DEPROJECT_HOLDS, DEPROJECT_ERODED, DEPROJECT_DEAD}.

## Scope
Local A10G, `analysis_only=true`, `official_tps=0`, **NO HF Job / NO submission / NO /v1/jobs:run**.
W&B group `optionb-rescue-deproject-stark`. CUDA_VISIBLE_DEVICES=0 (env ships =1 → torch sees 0 GPUs).
