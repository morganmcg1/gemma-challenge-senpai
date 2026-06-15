# Static-K wall-clock A/B (PR #273)

**Question.** Does static draft-depth **K=4/5 actually beat the deployed K=7** in
MEASURED local wall-clock TPS, or is the +4.28%/+4.00% from the #256/#266
*composition* bookkeeping? The composition prices a draft-pass saving against
`E[T]/model-forward-step`, but the served wall step is dominated by a large FIXED
serving overhead (CPU/Python/scheduler/sampler/detokenize) that does **not** shrink
when draft passes are dropped. The standing evidence that it over-credits: the
deployed path is K=7.

**Method.** Vary ONLY `num_speculative_tokens` (the static MTP draft length) in the
serve-time `SPECULATIVE_CONFIG` of the real served submission
`submissions/fa2sw_precache_kenyan` (the 481.53 path). Sweep K ∈ {3,4,5,6,7},
K=7 = deployed baseline. Everything else byte-identical: same model, KV cache,
greedy sampler, 128×512 prompts, seed. Measure median `wall_tps` over ≥3 fresh
runs per K via `scripts/profiler/paired_tps_ab.py` (PR #82/#90 paired runner).
**LOCAL only — no HF Job, no submission, no served-file edit, NOT a launch.**

**Greedy-safe by construction.** Changing K is proposal-only; the verify step is
greedy-exact, so emitted token-ids are identical across all K. Certified
empirically (128/128 token-id identity vs K=7) in `greedy_identity_check.py`.

**Realization ratio.** `measured-delta% / composition-delta%` for K=4 and K=5.
Composition deltas (#266 run cpjafa3h): K=4 +4.277%, K=5 +3.999%. A ratio well
below 1 (or negative) quantifies the over-credit.

## Files
- `run_sweep.sh` — orchestrates the K sweep (paired_tps_ab.py per K, reuse K=7 baseline).
- `greedy_identity_check.py` — per-prompt token-id identity certificate vs K=7.
- `analyze.py` — per-K table, realization ratios, booleans, self-test → `report.json`.

## Headlines
- PRIMARY (self-test boolean) `static_k_wallclock_ab_self_test_passes`.
- TEST `measured_local_wall_tps_gain_k4_vs_k7_pct` + per-K table + K4/K5 realization ratios.
- Booleans `static_k4_beats_k7_measured`, `composition_over_credits` (ratio < 0.5).
