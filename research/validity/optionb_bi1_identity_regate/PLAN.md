# PR #622 — Option-B #319 re-gate: BI=1 both-sides spec-vs-AR greedy identity (stark)

## Claim to confirm
Re-run the #607 greedy-identity probe with `VLLM_BATCH_INVARIANT=1` pinned on
**both** the reference AR (M=1) side and the served spec-verify (M=8) side. The
47% break (#607, 31048/65536, BI=0 reference on the 3D split-KV path) should
collapse to the ~0.43% int4-Marlin grid-tie residual that wirbel #616
characterized (`raw_structural_flip_rate_m8_vs_m1 = 0.004318`, CI
[0.3738%, 0.4944%]), all flips < 0.5 nat, τ=0.3 → 0.

## Mechanism (from stark #621 op-level proof)
- Under BI=1 the M=1 AR decode and the M=8 verify chunk both take the **2D**
  TRITON_ATTN path → attention op is bit-exact (`spec_verify_vs_ar_maxdiff_under_BI = 0.0`).
- The #607 47% was driven by the reference AR running the **3D split-KV** path
  (BI=0) vs the always-2D M=8 verify; a ~1-ULP near-tie flip then cascades.
- Residual under BI=1 is the int4-Marlin M-variance at the verify width
  (knife-edge near-ties < 0.5 nat), orthogonal to attention.

## Harness
Real-model teacher-forced per-step argmax (no cascade), built on the validated
stark #381 decode-width geometry (`decodewidth_e2e_identity.py`): the M=8 verify
argmax is read from a size_m=8 prompt_logprobs chunk against a prefix-cache hit,
compared to the M=1 AR greedy token at the same position/context. Slid across a
long AR trajectory to reach the #607 token budget (target 128×512 = 65,536).

Two arms (isolated subprocesses, process-wide BI env):
- `pinned`    — `VLLM_BATCH_INVARIANT=1` on BOTH sides (the decisive measurement)
- `heuristic` — `VLLM_BATCH_INVARIANT=0` (BI=0 contrast)

Model: `google/gemma-4-E4B-it-qat-w4a16-ct` (int4 W4A16 body, the #607/#621
checkpoint). The MTP drafter is NOT loaded — under greedy temp=0 the drafter only
changes acceptance/speed, never the verify argmax (#621), so it is irrelevant to
the kernel-variance measurement.

## Deliverables (terminal SENPAI-RESULT)
`break_rate_bi1_both_sides`, `attention_path_break_count` (flips ≥ 0.5 nat,
expect ~0), `int4_tie_residual_rate`, `residual_frac_under_0p5nat`,
`residual_after_tau_0p3nat`, τ-ladder {0.0, 0.2, 0.3}, and VERDICT
(`ATTENTION_RECOVERED_RESIDUAL_IS_INT4_TIES` vs `ATTENTION_BREAK_PERSISTS`).

## Scope
Local A10G, analysis_only=true, official_tps=0. NO HF Job, NO submission, NO
served-file change. vLLM 0.22.0 (the #607/#616/#621 chain version; the PR body's
"dev307" is flagged to the advisor as inconsistent with "exactly the #607/#621
config" — 0.22.0 is required to corroborate #616's 0.22.0 number).
W&B group `optionb-bi1-identity-regate`.
