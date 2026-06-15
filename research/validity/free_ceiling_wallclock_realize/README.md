# Free-ceiling wall-clock realization (PR #298, stark)

**Question.** The whole programme treats **487.7 TPS** as the closed "free step
ceiling" (wirbel #285 lossless envelope 487.729; denken #291 kernel-event floor
487.7289; kanna #286 basis-honest 493.64). But that number is an
**analytically-composed / kernel-µs** quantity. The launch gate is a **MEASURED
≥500 host-to-host wall TPS**, and nobody has measured whether the 487.7 free
ceiling actually **realizes on a direct local wall-clock A/B** — the exact thing
my #273 method exists to check, and the exact thing static-K *failed*
(realization ratio **−2.02**: a composed +4.28% measured as −8.63%).

**CRUX.** Apply the banked greedy-safe lossless lever — the verify SDPA
`num_stages` 3→2 tune that wirbel #285/#279 isolated and denken #291 priced — on
the deployed stack locally, re-run the **#273 wall-clock A/B harness**
(`scripts/profiler/paired_tps_ab.py`, 128×512 single-stream greedy, ≥2 seeds,
p50), and compute the **realization ratio = measured Δ%_wall / composed Δ%**.
Does the composed **+1.287%** (481.53→487.729) realize on the host-to-host wall
(ratio≈1, 487.7 is wall-honest), or under-realize / over-credit like static-K
(the measured denominator the build stacks on is even tighter)?

**0 TPS.** This wall-audits the banked free step ceiling. It does NOT produce a
≥500 build and does NOT change the served checkpoint. The launch gate stays
land #245's MEASURED ≥500 at λ̂≥0.9780 AND PPL≤2.42, human-approval-gated.

## The lever and how it is toggled (local, reversible)

wirbel #285 (`lossless_micro_lever_envelope.json`): the SDPA `num_stages`=3→2 tune
is the **ONLY incremental free lever** (lm_head fused-epilogue + RMSNorm-fold are
`already_captured` in the deployed baseline → 0 incremental). It is a
**bit-identical cp.async pipeline-depth change** (NOT MMA/K-reduction order →
maxdiff 0.0, greedy-safe), saving a standalone 15.48µs/step (7 global head-512 ×
2.0µs + 14 sliding head-256 × 13.5µs) → composed 487.729 (an **upper bound**: no
in-graph overlap in a standalone replay).

The deployed served `unified_attention` launches `kernel_unified_attention`
(`vllm.v1.attention.ops.triton_unified_attention`) as a **bare `@triton.jit`**
with the triton-default `num_stages=3` (confirmed by #279 self-test lynchpin (i):
the real wrapper matches the forced-(w4,s3) baseline). It is therefore **NOT a
serve-time env toggle** — to wall-measure it I inject `num_stages=2` at the
kernel launch via a **temporary, env-gated** patch that reuses the submission's
own `_TargetFinder`/`_PatchingLoader` meta-path mechanism (`sitecustomize.py`):

- `SDPA_NUM_STAGES_AB` unset → patch is a **no-op** (baseline arm = byte-identical
  deployed K=7, the #273 reproduction).
- `SDPA_NUM_STAGES_AB=2` → `kernel_unified_attention.run` injects `num_stages=2`
  on every launch (counted + logged), compiled into the ONEGRAPH capture during
  warmup → the served verify/draft attention runs s2.
- **toggle → measure → revert, nothing submitted.** The PR diff carries only
  `research/**`; the `sitecustomize.py` toggle is reverted and never committed.

## Method (reuse #273)

1. **Baseline (s3)** = deployed K=7, ≥2 seeds, p50 median `wall_tps`
   (`paired_tps_ab.py`), reproducing my #273 deployed-K7 within noise.
2. **Candidate (s2)** = same stack + `SDPA_NUM_STAGES_AB=2`, ≥2 seeds, p50.
   Server logs asserted to show forced-num_stages count > 0 (proves it applied —
   guards against a silent no-op that would fake an over-credit verdict).
3. `realized_delta_pct_wall = 100·(s2_med − s3_med)/s3_med`.
   `realization_ratio_487 = realized_delta_pct_wall / +1.287%`.
   classify: `realizes` (ratio≥0.8) / `partial` (0<ratio<0.8) /
   `over_credits` (ratio≤0, the static-K class).
4. `measured_free_ceiling_tps = s3_med_official_projection × (1+realized_Δ%)`;
   `free_ceiling_below_composed = measured_free_ceiling < 487.729`.

## Imported EXACT (not re-derived)
- 481.53 frontier / 487.729 wirbel-#285 envelope / 487.7289 denken-#291 floor /
  493.64 kanna-#286 basis-honest / 1218.2µs step / 1202.717µs new_step /
  0.2147 denken-#278 bridge / 125.268 K_cal / 3.844 E[T].
- composed_delta_pct = (487.729−481.53)/481.53 = **+1.287%**.
- #273 `51bdsbpw`: K4-vs-K7 = −8.629%, ratio −2.018 (the static-K precedent).

## Files
- `sdpa_num_stages_ab.py` — env-gated `num_stages` injector (meta-path patch).
- `free_ceiling_wallclock_realize.py` — orchestrator: temp sitecustomize toggle →
  paired #273 wall A/B (s3 vs s2, ≥2 seeds) → realization ratio + self-test →
  W&B (`--wandb_group free-ceiling-wallclock-realize`).

## Headlines
- PRIMARY (self-test bool) `free_ceiling_wallclock_realize_self_test_passes`.
- TEST `free_ceiling_realizes_on_wall` (bool) + `measured_free_ceiling_tps` +
  `realization_ratio_487`.

**NOT a launch. NOT a build. NOT open2. No served-file change (toggle reverted).**
