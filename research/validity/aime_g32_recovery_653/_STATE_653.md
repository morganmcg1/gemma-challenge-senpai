# PR #653 state (lawine) — AIME g32 recovery, 3-arm self-consistent eval

## Harness (reuse, do NOT rebuild)
- Instrument: research/downstream_quality_aime/aime_eval.py (ubel #638, tracked)
- Per-arm wrapper: research/validity/aime_g32_recovery_653/run_arm.sh <arm>
- Protocol: years 2024,2025-I,2025-II (n=60), k=1, greedy temp0 top_p1.0 top_k-1,
  max_tokens 6144, min_tokens 8, --no-thinking, seed 1234, client_concurrency 1.
  Server: vLLM 0.22.0, --max-num-seqs 1, VLLM_BATCH_INVARIANT=1,
  VLLM_USE_FLASHINFER_SAMPLER=0, mml 8192. CONFIRMED identical across all 3 arms
  (only the served checkpoint changes): seqs=1/BI=1/mml=8192 verified in each server log.

## Arms + serve scripts
1. shipped_g128 = /workspace/gemma_build/int4_g128_lmhead (minmax, untied int4 head)
   serve: optionb_denom_0p22_gb6144/serve_int4ar_0p22.sh
2. official_g32 = google/gemma-4-E4B-it-qat-w4a16-ct (tied bf16 head). HF snapshot
   ef0a4c4... present (11G). serve: int4_body_quality_upside_639/serve_official_g32.sh (MAX_NUM_SEQS=1)
3. ours_g32 = /workspace/gemma_build/int4_g32_lmhead (untied int4 head, #639 Arm-2) HEADLINE
   serve: int4_body_quality_upside_639/serve_ours_g32.sh

## RESULTS so far (2026-06-18)
- Arm1 shipped_g128: DONE = 0.3833 (23/60), Wilson [0.2709,0.5098], 82.1% bf16,
  trunc 8/60 (13.3%), censored 0.4423 (94.8% bf16), extract_fail 0.0. wall 42.5min.
- Arm3 ours_g32: DONE = 0.3667 (22/60), Wilson [0.2562,0.4932], 78.6% bf16,
  trunc 8/60, censored 0.4231 (90.7% bf16), extract_fail 0.0. wall 45.7min.
- HEADLINE group-size delta (ours_g32 - shipped_g128) = -0.0167 (-1 problem).
  McNemar b=3 c=4 p=1.0000 (NOT sig). Newcombe95 [-0.1015,+0.0686] (includes 0).
  => NO RECOVERY. GPQA contrast: GPQA moved +0.0283; AIME moved -0.0167 (-0.59x).
- Arm2 official_g32: RUNNING. started 09:55:05Z, eta ~10:38Z. server pid 2028658,
  eval pid 2030035, out=_official_g32_aime.out, result=results/official_g32_aime_gb6144.json.
- Reconciliation: my shipped_g128 0.3833 sits between ubel 0.350 and denken 0.400;
  vs denken AR completions: 5/60 correctness-flips by serve-stack alone (chaotic int4 body).

## BUGFIX applied (my own new aggregate.py, pre-report)
- newcombe_paired_ci phi denom was a SUM of two marginal products under one sqrt ->
  phi blew to 15.4, clamped 1.0, falsely narrowed CI to exclude 0 (contradicting
  McNemar p=1.0). FIXED to the standard phi coeff: product of all four marginals
  (e+f)(g+h)(e+g)(f+h) under sqrt -> phi=0.7517, CI now includes 0. Note in results.

## VERDICT (pending Arm2): AIME_DEFICIT_DEEPER_THAN_GROUP_SIZE
g128->g32 does NOT recover AIME (0.3667 vs 0.3833, indistinguishable, McNemar p=1.0),
opposite of GPQA where g32 recovered +0.0283. AIME int4 deficit is intrinsic, not the
group-size knob. Hardens Reading-B; recipe-tuning prong exhausted on AIME.
Arm2 official_g32 (Google's full g32 recipe + bf16 head) is the cross-check: if it ALSO
fails to recover -> even stronger; if it recovers -> bf16-head/calibration not group size.

## Next after Arm2 done
1. verify results/official_g32_aime_gb6144.json + DONE line in _official_g32_aime.out
2. kill official_g32 server (int4_body_quality_upside_639/_server_official_g32.pid 2028658), drain GPU
3. /tmp/eval-serve-venv/bin/python aggregate.py  -> panel_summary.json (3 arms)
4. ./.venv/bin/python log_wandb.py (group aime-g32-recovery-lawine, analysis_only=true, official_tps=0)
5. write terminal SENPAI-RESULT comment on PR #653; submit via senpai:submit-experiment-results 653 target/
## Budget: SENPAI_TIMEOUT_MINUTES per run (each arm ~42-46min, fits). NO HF jobs. analysis_only.
