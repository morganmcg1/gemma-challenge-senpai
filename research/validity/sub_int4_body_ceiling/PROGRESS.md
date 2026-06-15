# PR #372 mixed-precision sub-int4 — run state

## Session resume 2026-06-15 16:58Z (lawine)
- Harness reviewed end-to-end: measure_ppl is DETERMINISTIC (teacher-forced,
  no sampling); per-module dNLL spread (+-300 on 41,308 base) >> any CUDA-noise
  floor (~0.25), so deltas are REAL but mixed-sign -> verdict correctly rests on
  MEASURED joint configs (deciles+binary-search), not the additive sum. cb3=3.125
  / int4=4.125 bpw both at deployed g128 (scale-honest; no finer-grouping erosion,
  which is the clean answer to denken #356's grouping-vise caveat). All 4 PR
  deliverables + GO criterion implemented; wandb_logging fns present.
- Sweep 250/258 @16:58Z; chunk PID 813243 (start 16:27, --max-seconds 4200) has
  ~2100s budget left -> will finish sweep (~4min) + run FINALIZE (~15-20min).
  Driver PID 798174 will relaunch a resumable chunk if needed (finalize-reserve
  guard defers finalize if <1000s left). Expected results ~17:20Z.
- TODO after results: compare achievable_avg_bpw vs denken #373
  residual_avg_bpw_for_mixedprec + ~3.7bpw analytic mark; write SENPAI-RESULT
  (primary=mixed_precision_ceiling_lift_go, test=achievable_avg_bpw) + submit.


## Driver-managed (hands-off)
- `drive_mixed.sh <pid>` waits for an in-flight chunk, then relaunches resumable
  chunks (`--max-seconds 4200`, < the 90-min run cap) until finalize writes
  `measure_mixed_precision_results.json`. Driver PID file: `drive_mixed.pid`.
  Driver log: `drive_mixed.log`. Per-chunk stdout: `mixed_chunk_<i>.out`.
- Sweep checkpoints after EVERY module to `mixed_precision_checkpoint.json`
  (skips cached modules+anchors; the chunk that completes 258/258 also runs
  FINALIZE unless <1000s budget left, in which case a fresh chunk finalizes).
- Manual resume (if driver dies): re-run the same chunk command; it resumes.

## Anchors (cached in checkpoint)
- bf16 PPL 1.9526 | int4 1.951205 (gate denom) | uniform_cb3 2.0712 gate 2.5234 (+6.15%)
- Gate: gate_PPL = 2.3772*(PPL/1.951205) <= 2.42  <=>  local PPL <= 1.9863
- NLL headroom over int4 ~ +1098 ; uniform cb3 costs +3687 NLL (super-additive).

## Key in-flight finding (141/258 swept @ 16:07Z, ~28.5s/module)
- Per-module cb3 dNLL is NOISE-dominated: at 141 swept, range -334.8..+369.3 on
  a 41,308 NLL base (mean +6.1, median -1.68, 52% "improve"). Single-module
  3-bit perturbations are dominated by the noise floor; ~half show negative
  deltas. So the ascending-sensitivity ranking is near-random and the additive
  prediction is unusable — the verdict rests on the MEASURED allocation curve
  (deciles + binary-search) in finalize, exactly as the PR demands.
  additivity_gap is expected to be large/positive (super-additive).
- sensitivity_is_concentrated likely FALSE (top-decile share < 0.40), but the
  lever does NOT need concentration: the private-500 residual is tiny. Spec
  ceiling 520.953 * 0.957 (private gap) = 498.5, only 1.5 short of 500 -> ANY
  passing cb3 fraction (avg_bpw < ~4.107) tips spec_clears_private_500 TRUE.
  So GO is near-certain; the substantive number is the MEASURED achievable
  shrink (k_star/achievable_avg_bpw) vs denken #373 residual_avg_bpw_for_mixedprec.
- Public evidence (digest 16:07Z): leaderboard top frantic-penguin 489.63 TPS
  VALID (PPL 2.3774), all top-10 are spec-decode split-KV verify stacks on the
  int4 body; NO sub-int4 body-shrink deployed. Body-shrink is the orthogonal
  lever to close the private-500 gap. No human issues open for lawine.
- Grouping HELD at deployed g128 for both options (int4 4.125 / cb3 3.125 eff
  bpw, scale bytes included). We do NOT invoke finer g32/g64 grouping, which per
  denken #356's caveat erodes the BW saving (~3.5 eff bpw at g32). So
  achievable_avg_bpw is a clean, scale-honest number.

## Harness verified (no correctness bugs)
- cb3 = #367 RHT-incoherence + Gaussian-VQ (g128, vq_dim2, K64); int4 = deployed
  RTN asym g128. PPL = deterministic teacher-forced token-weighted NLL over the
  official 128 pre-tokenized records / 61,797 tok. Matches #355/#367.
