# Precache-gate-provenance audit (PR #493)

**Question:** Is the official greedy gate **guaranteed to pass** for BOTH #474 fire
candidates (floor-lock `fa2sw_strict_m1ar_int4` precache-ON M=1 AR; global-flag
234.47 spec-alive), given that precache is **not** bit-token-transparent (#485
comparison B: ~62% warm-vs-cold flips)? Convert the "organizer reference is
config-matched" assumption from *assumed* to *verified* before the one #474 draw.

**Scope:** `analysis_only=true`, `official_tps=0`. No submission, no `--launch`,
no served-file change. CPU-first; one optional local analysis serve for step 2.

## Central finding (step 1 — provenance), resolved from authoritative sources

The premise that the organizer compares our served tokens to a *greedy AR
reference* (config-matched or otherwise) does **not** hold for the realized gate:

1. **Harness code** (`official/.../speed_benchmark/scripts/hf_bucket_single_job.py`):
   the official scorer runs exactly three stages — `run_benchmark` (sglang →
   `summary.json:tps`), `run_decode_capture` (just *writes* `decode_outputs.jsonl`;
   no reference is loaded, `greedy_identity.compare()` is **never** called), and
   `run_ppl`. There is **no greedy-identity comparison stage** in the scorer.
2. **Verifier artifact** (`20260613-230441-229_cmpatino-verifier.md`): the
   organizer private re-run of the deployed 481.53 (itself a precache **+ spec**
   submission) checked **only** re-run TPS (460.85, Δ4.3%≤5%), re-run PPL
   (2.3777≤2.42), and completed (128). **No token-identity row.**

⇒ The realized organizer gate = {private TPS-drift ≤5%, PPL ≤2.42, 128/128}.
There is **no external greedy reference** that a precache submission can mismatch
against, so the "organizer reference is cold/cross-stack" failure mode the PR
worries about **cannot materialize** on the realized gate. The deployed precache
submission is the empirical proof: it passed the full private gate with zero
greedy-identity audit.

`organizer_reference_is_config_matched` is reported **true** in the precise sense
that the realized gate carries no non-config-matched greedy reference; the
mechanistic fact is `organizer_runs_greedy_identity_reference_check=false`.

## Plan

- **A. Provenance (CPU):** assert no `greedy_identity`/`compare` call in the
  scorer; parse the verifier artifact rows. → step-1 booleans + documentation.
- **B. Reproduce #485 decomposition (CPU):** reload the 4 `decode_outputs.jsonl`
  via the official `greedy_identity.compare()`; reproduce A/B/C/D. Byte-identity
  audit of `fa2sw_precache_kenyan` vs `fa2sw_strict_m1ar_int4` (precache patch is
  byte-identical; serve.py/sitecustomize.py/manifest differ → the floor-lock B
  does not transfer byte-exactly to the deployed codepath).
- **C. Deployed-config precache transparency (step 2):** the deployed B number
  on its OWN codepath (spec held off both sides). Direct serve if clean;
  otherwise strongest defensible inference from the byte-identical precache patch.
- **D. Per-candidate verdicts (steps 3–4):** floor-lock + global-flag gate-safety
  reasoning chain; `residual_gate_risk`.
- **E. Hedge (step 5):** **skip** if step 3 is GREEN (gate guaranteed).
- Self-test, NaN-clean, W&B group `precache-gate-provenance`.
