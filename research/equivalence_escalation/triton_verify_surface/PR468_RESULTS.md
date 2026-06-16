STUDENT wirbel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["91wb8mln"],"primary_metric":{"name":"numstages_realized_tps_delta","value":1.1050119580263376},"test_metric":{"name":"ppl","value":2.3772}}

## Results

**Verdict — it did NOT invert.** The modeled `+1.1982` byte-exact `num_stages 3→2` ceiling is the **first of the six modeled leads to survive end-to-end with the right sign**: realized point estimate **+1.1050 TPS = 92.2% of modeled** (4 of 5 seeds positive). But +1.20 is a sub‑σ_hw signal, so at N=5 the served-wall noise swamps it — the realized delta is statistically **within-noise** (CI95 crosses zero), neither a clean realize nor the 6th isolation-trap. **Identity is byte-exact** (frac 1.0, 0 flips added/removed). The config-level attention change **SURVIVES ONEGRAPH capture** (the #466 cross-check). **Not a reopener** (+1.105 ≪ +2 materiality bar; even the optimistic CI upper +3.345 < σ_hw 4.8). Self-test 9/9, served file unchanged, no submission/HF job.

This is analysis-only LOCAL on my pod A10G (served int4 stack), through the same env-gated, auto-reverted injector A/B harness as #442/#459 — `official_tps=0`, `no_served_file_change=true`.

### 1. End-to-end served A/B — deliverable #1 (5 seeds, paired)

| Arm | config | pooled p50 wall TPS | projected-official median |
|---|---|---|---|
| **A baseline** `bm16_s3` | `num_stages=3` (deployed default) | 454.357 | 481.703 |
| **B candidate** `bm16_s2` | `num_stages=2`, BLOCK_M=16 held | 455.933 | 483.374 |

- **per-seed-paired Δ = +0.2295%** (CI95 ±0.4653%, 5 seeds, Student-t df=4)
- pooled cross-check Δ = +0.3467% (CI95 ±0.3366%)
- **`numstages_realized_tps_delta` = +1.1050 TPS** (= +0.2295% × deployed 481.53), **CI95 [−1.1354 .. +3.3455]**

Per seed (A→B, Δ%): seed1 454.357→452.360 **−0.4395%** (sole negative) · seed2 454.275→455.933 +0.3648% · seed3 453.913→455.780 +0.4114% · seed4 454.592→456.540 +0.4284% · seed5 454.607→456.344 +0.3822%. The single seed-1 dip is what widens the paired CI across zero; the other 4 seeds are a tight +0.36…+0.43% cluster sitting right on the model.

Baseline anchored to the **deployed served config** (481.53 TPS / PR #52, the config B is measured against), per the PR's instruction.

### 2. Does the modeled +1.20 realize? — deliverable #2

`realization_ratio = 0.9222` → **classification = `within_noise`**, `numstages_realizes = False`.

- It is **NOT** the 6th isolation-trap. The five prior modeled leads all flipped sign end-to-end (pinned-K +13.998→−5.82 · cb3 +15.60→0.0 · static-K +13.2%→−8.63% · autotune-isolated +15.86→−5.65 · relax-prize +17→−0.94). `num_stages` is the **only one that kept its sign** — point estimate +1.105 lands within rounding of the modeled +1.198. As predicted, because `num_stages` changes cp.async pipeline depth (occupancy), **not** the grid/CTA count, it carries none of bm4's CTA‑tripling penalty.
- It does **NOT** cleanly realize either: a +1.10 TPS signal is ~0.23σ_hw (σ_hw ≈ 4.8), and the N=5 paired half-width (±2.24 TPS) is larger than the signal. The model is corroborated at the point-estimate level but the served wall cannot resolve it from zero at this seed count. **No inversion to decompose** — the candidate is faster than baseline in the central estimate, not slower.

### 3. Identity byte-exact — deliverable #3

`numstages_identity_fraction = 1.0` · `frac_token_prefix_match = 1.0` · **`byte_exact = True`** (64 prompts / 32,768 tokens, both heads 256+512 overridden, served verify proven 3D).

`num_stages=2` produces a **bit-identical token stream** to `num_stages=3` → it adds/removes **zero** flips. The deployed path's identity **0.9966 / 3 flips {11,18,118}** is therefore preserved **exactly** (B ≡ A byte-for-byte, so B inherits A's flip set unchanged). The "byte-exact" claim holds — `num_stages` is a **legitimate strict lever**, unlike bm4. PPL anchor **2.3772** (≤ 2.42 gate; cap = reference PPL + 5%).

### 4. ONEGRAPH-survival — deliverable #4 (the stark #466 cross-check)

**`numstages_survives_onegraph = True`.**

Mechanism: the `num_stages=2` cubin is compiled during warmup and **BAKED into the captured ONEGRAPH whole-step graph**; the Python wrapper fires only during capture warmup (`max_forced_milestone = 100`, bounded — **not** the ~1e5+ a per-step eager fallback would log), then the graph replays the baked kernel. Candidate wall_tps (455.93) did **not** collapse toward the serial/BI=1 161.70 → the config-level attention change **survived capture with no recapture and no BI=1 fallback**. Both head geometries (256, 512) overridden, `served_verify_is_3d = True`, `candidate_not_collapsed = True`.

→ **Datapoint for #466:** a config-level Triton-attention change applied through the meta-path finder *does* survive ONEGRAPH capture (bakes in → realized speedup), it does **not** force serialization toward 161.70. This is the direct end-to-end confirmation #466 needed, landing on the "survives capture" side.

### 5. Self-test + metric surface — deliverable #5

`numstages_self_test_passes = True` (**9/9**): constants_exact · candidate_actually_applied_numstages · both_heads_overridden_256_and_512 · served_verify_is_3d_census · realized_delta_classified · all_tps_finite_positive · onegraph_survival_resolved · at_least_5_seeds · **identity_census_byte_exact**. `toggle_reverted_clean = True`.

Constants reproduce exactly: surface = 7×0.6656 + 30×0.549547 = **21.1456 µs**; ceiling = 21.1456 × 0.056663 = **+1.19817 TPS**; geometry 37 = 30+7.

`attention_surface_closed_three_ways = **False**` — honest nuance: the strict criterion requires the **realized CI95 upper bound < +2** materiality, and it is +3.345 > 2.0, so the *realized measurement alone* cannot certify immateriality at 95% (the wall-noise band is wide at N=5). **But the surface is plainly not reopened**: both the point estimate (+1.105) and the model (+1.198) sit far below +2, and even the optimistic 95% upper bound (+3.345) is below σ_hw (4.8). So the surface is closed by **point-estimate + model** (third corroboration), just not certifiable as "closed three ways" under the strict realized-CI test. Logged honestly as `False`.

Logged metrics (W&B `91wb8mln`): `numstages_realized_tps_delta=1.1050`, `numstages_modeled_ceiling=1.1982`, `numstages_realizes=0`, `numstages_identity_fraction=1.0`, `numstages_identity_byte_exact=1`, `numstages_survives_onegraph=1`, `attention_surface_closed_three_ways=0`, `numstages_self_test_passes=1`, `census_run=1`, `analysis_only=1`, `no_served_file_change=1`, `official_tps=0`, `ppl=2.3772`, classification=`within_noise`.

### Command

```bash
# Finalize: reuse the 5 on-disk paired seeds, ingest the identity census, run self-test, log W&B
python research/equivalence_escalation/triton_verify_surface/numstages_realize_wall_ab.py \
  --seeds 1,2,3,4,5 \
  --census-json research/equivalence_escalation/triton_verify_surface/numstages_census_out/results.json \
  --self-test
```

Underlying per-seed A/B (reused, run earlier this session): candidate arm `WIRBEL_BM4_AB=1 WIRBEL_BM4_BLOCK_M=16 WIRBEL_BM4_NUM_STAGES=2` (num_stages-only isolation — BLOCK_M held at the deployed 16 so the grid recompute is a no-op) vs deployed default baseline, 128 prompts × 512 output_len, paired per seed, through the reverted sitecustomize injector. Identity census: `served_bm4_census` at BLOCK_M=16 / num_stages=2, 64 prompts × 512.

### Peak memory

~**20.7 GiB** of the 23,028 MiB A10G (`gpu_memory_utilization=0.90` cap), **identical in both arms** — `num_stages 3→2` frees per-CTA shared memory (improves occupancy headroom) but does not change the vLLM allocation, weights, or KV footprint. Served KV cache 9.46 GiB / 376,880 tokens / 92.01× max concurrency at 4,096 tokens/req.

### What happened

The +1.20 modeled ceiling is the **rare byte-exact lever**, and it behaved exactly as the byte-exact physics predicts: it did **not** invert (point estimate +1.105 ≈ 92% of modeled, sign preserved, identity bit-exact), distinguishing it cleanly from the five grid/scheduling-changing leads that all flipped sign. The catch is purely statistical: +1.20 TPS is sub-σ_hw, so N=5 paired seeds cannot resolve it from zero (CI95 crosses zero, one seed dips negative). So the disciplined verdict is **corroborated-but-within-noise**, not "realizes". It is **not a frontier reopener** by any reading — the lever is real but immaterial (< +2 by both model and point estimate). Net: the Triton verify-attention surface is now closed a **third** way at the point-estimate+model level (greedy-unsafe tile #442 + byte-exact-immaterial-modeled #459 + byte-exact-realized-here), and #466 gets its cross-check that a config-level attention change survives ONEGRAPH capture.

Housekeeping (transparency, not a code bug): the prior interrupted session (it stopped mid-seed-9) had left the env-gated injector toggle applied in `submissions/fa2sw_precache_kenyan/sitecustomize.py`. The toggle is behaviorally inert when `WIRBEL_BM4_AB` is unset, but I reverted it via `git checkout` before finalizing so `no_served_file_change` strictly holds; the orchestrator re-confirmed `toggle_reverted_clean=True` and the served file is byte-identical to the committed version. The harness's own `ensure_clean_toggle()` is idempotent and would have stripped it on the next run regardless.

### Suggested follow-ups

- **Powering the realize, if it ever matters:** to resolve a +1.1 TPS signal from zero at 95% you need the paired half-width below ~1.1 TPS → roughly N≈20–25 seeds (or median-of-n>1 per arm). Not worth a card here — the lever is immaterial either way — but that is the seed budget that would convert `within_noise` → a clean `realizes`. I would **not** spend it.
- **The surface is now exhausted.** This was the last un-realized modeled byte-exact lever in the supply sweep; it corroborates rather than reopens. I would treat the Triton verify-attention surface as closed and not re-propose `num_stages`, tile, autotune, pinned/static-K, or relax variants on it.
- **The realize-or-collapse ledger now has its one non-inverting datapoint.** Worth recording that the discriminator is *grid/CTA-count change* (all five inverters changed it; `num_stages` did not) — a cheap pre-screen for whether a future modeled microbench lead is likely to survive the wall.
