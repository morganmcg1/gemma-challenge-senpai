# PR #220 — FP16-verify valid-path TPS ceiling (stark)

**CPU-only analytic. NOT a launch.** Maps the fp16/bf16-verify VALID path's TPS
ceiling over the banked launch composition + a swept fp16 step-multiplier, to
answer the Issue #211 question: *can any draft (however strong, Blackwell-trained
or not) clear 500 via fp16-verify, or is its ceiling capped below 500?*

## Construction (all imported; nothing re-derived)

- Composition `official = K_cal·(E[T]/step)·τ`: `K_cal=125.26795`, `step_int4=1.2182`,
  `τ∈{1.0, 0.9924}` (#148/#169, #168, #181).
- int4-spec λ=1 ceiling **520.9527** (#204 `launch_sigma_unit_rebase`,
  `imported_legs_201.lambda1_ceiling`).
- E[T](λ) reach-DP shape `g(λ)=E[T](λ)/E[T](1)` from the #175/#184 reach-DP forward
  map (the SAME machinery #199/#213 used), on the banked both-bugs floor/ceiling
  spines (`compliant_spec_et_results.json`; reach-DP needs no shard).
- Achievable-λ band: floor `λ̂=0.34186` (#193), spine `λ≈0.997` (land #71), 1.0 sat.

fp16 shares the int4 draft tree + E[T](λ); only the step cost changes:
`step_fp16 = step_int4·M_step`, so
`fp16verify_tps(λ) = K_cal·(E[T](λ)/step_fp16)·τ = (520.9527/M_step)·g(λ)`, and the
draft-independent λ=1 cap is `fp16verify_ceiling_at_lambda1 = 520.9527/M_step`.

`M_step` is **swept** ∈ {1.3, 1.5, 1.7, 2.0, 2.3}; **lawine PR #221**
(`fp16-verify-valid-cost`) MEASURES which column is real and confirms the fp16
token-identity (validity) premise. (Advisor cross-ref fix: empirical leg is #221,
not #220.)

## Headline (pre-registered from the smoke pass)

Crossover `M_step* = 520.9527/500 = 1.0419`. Every swept M_step ≥ 1.3 is **above**
it, so the λ=1 ceiling is below 500 at every M_step (400.7 → 226.5). Expected
verdict: `FP16VERIFY_CEILING_BELOW_500_AT_ALL_MSTEP` → the fp16-valid path is dead
at every λ; the only valid-500 route is wirbel #216's batch-invariant int4 kernel.

- PRIMARY `fp16_verify_ceiling_self_test_passes` (self-test).
- TEST `fp16verify_ceiling_at_lambda1` at central M_step=1.7 (≈306.4, well below 500).

Output: `fp16_verify_ceiling.py` + `results.json` here. `--wandb_group fp16-verify-valid-ceiling`.
