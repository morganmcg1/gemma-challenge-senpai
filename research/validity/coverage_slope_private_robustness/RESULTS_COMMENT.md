STUDENT ubel:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["bn0v5rqr"],"primary_metric":{"name":"slope_robustness_self_test_passes","value":1},"test_metric":{"name":"slope_flattening_ratio","value":0.8928},"headline":{"slope_public":489.76,"slope_private_oob":437.27,"slope_flattening_ratio":0.8928,"coverage_target_for_3p2_private":0.9024,"coverage_target_for_3p2_private_conservative":0.9109,"target_inflation":0.0013,"budget_margin_private":0.0189,"budget_margin_private_conservative":0.0104,"flattening_breakeven":0.3472,"flattening_margin":0.5456,"slope_is_private_robust":true,"demand_route_survives_private_oob":true,"verdict_band":"GREEN_slope_private_robust"}}

## Results

**GREEN — the coverage→gap slope IS private-robust.** Re-deriving the slope under the **directly-measured #263 private per-draft-position profile** gives `slope_flattening_ratio = 0.893` (slope 489.8 → **437.3 TPS/unit**), moving `coverage_target_for_3p2` only 0.9011 → **0.9024**. Even the deliberately-pessimistic **conservative stress** (−34.5% uniform downstream collapse) lands at **0.9109** — still **inside #336's +0.031 budget** with +0.0104 margin. The slope only breaks the route at flattening **< 0.347** (a downstream collapse of **0.66 ≈ 1.9× the #263-measured rank-2+ collapse**), far below every plausible model. **fern #357 can bank the slope** — I recommend the conservative private-anchored target **≈0.911**, not the bare public 0.9011.

### The decisive mechanism — why a₁ is irrelevant to the slope (and only the deep tail matters)

Writing the K=7 draft chain's acceptance-length with per-position **conditional** acceptances `a_j = P(token j accepted | 1..j-1 accepted)`:

```
E[T] = 1 + Σ_{k=1..K} Π_{j=1..k} a_j  =  1 + a₁·T2          (EXACT identity)
T2   = 1 + a₂ + a₂a₃ + … + a₂…a_K      =  dE[T]/da₁          (the DOWNSTREAM tail)
```

The slope is `dTPS/dcov = T2·K_cal`. Because E[T] is **linear in a₁**, `T2 = dE[T]/da₁` **does not depend on a₁**. **Consequence: the slope flattening depends ONLY on how the deep (k≥2) conditional acceptances degrade on private — the first-token collapse, however severe, is irrelevant to the slope.** (Self-test `c_t2_independent_of_a1` verifies T2 is invariant under an a₁ perturbation.) This is the structural pivot of the whole card: private OOD hits a₁ hardest, but a₁ is exactly the part the slope doesn't see.

### The private per-position profile is DIRECTLY MEASURED — not modeled (#263)

My #263 (`he7glotf`) rank-probe recorded the **per-draft-position conditional acceptance** on a private-proxy slice (`conditional_rank1_acceptance_q`) alongside a matched public repro (`cross_check.conditional76`, == #289 to <1e-3):

| position k | a_k public (#263) | a_k private (#263) | private − public |
|---:|---:|---:|---:|
| 1 | 0.7287 | **0.5975** | **−0.1312** ← first-token cliff (severe, but ⊥ slope) |
| 2 | 0.7590 | 0.6914 | −0.0676 |
| 3 | 0.7925 | 0.7470 | −0.0455 |
| 4 | 0.8217 | 0.7688 | −0.0529 |
| 5 | 0.8343 | **0.8446** | **+0.0103** ← deep tail HOLDS |
| 6 | 0.8353 | **0.8583** | **+0.0230** |
| 7 | 0.8473 | **0.8917** | **+0.0444** |

The collapse is concentrated at the **first two** positions; the **deep tail holds up or improves** (a₅–a₇ are *higher* on private). This is a **survivor effect**: on the harder private slice only well-tracked prompts reach depth, so conditional-on-reaching-depth acceptance stays high. Since T2 is the deep tail, the measured private leverage barely flattens: **T2_pub = 3.903 → T2_priv = 3.485 (ratio 0.893)** even on this **adversarial** slice (the #258/#263 decode-proxy over-reads the benchmark gap 3–5×). I did **not** need the optional GPU leg — the profile is *already measured*.

### Five models — every one clears the budget

| Model | what it assumes | flattening ratio | slope (TPS/unit) | cov_target_private | budget margin | within +0.031? |
|---|---|---:|---:|---:|---:|:---:|
| **central (measured #263)** | the directly-measured private profile | **0.893** | **437.3** | **0.9024** | +0.0189 | ✅ |
| benchmark (realistic) | true-dist net E[T] loss, all charged to tail | 0.942 | 461.3 | 0.9017 | +0.0196 | ✅ |
| adversarial floor | whole downstream pinned at #289 low-tail α=0.694 | 0.772 | 378.3 | 0.9042 | +0.0171 | ✅ |
| mild (survival-sum) | downstream survival SUM ×(1−0.345) | 0.743 | 364.1 | 0.9048 | +0.0165 | ✅ |
| **conservative (stress)** | every downstream a_k ×(1−0.345) | **0.521** | **255.3** | **0.9109** | +0.0104 | ✅ |
| — breakeven — | route exits +0.031 budget | **0.347** | **170.0** | 0.9213 (=bar) | 0.0000 | — |

The realistic benchmark degradation (E[T]_priv = ρ·E[T]_pub = 3.679, only ~4.4% below public) is **milder** than the adversarial #263 measurement — so the measured 0.893 is already a conservative-leaning central. The **conservative stress** (0.521) deliberately over-reads: it applies the −34.5% *rank-2+ marginal-mass* collapse as a uniform *per-position conditional* drop (a different, larger quantity) and **ignores** the survivor-boosted deep tail — exactly the worst case for the "marginal prompts have worse tails" objection — and it still clears with +0.0104 margin.

### Breakeven & budget

```
tps_shrink_to_3p2 = 481.53·(1−0.032) − 460.85 = 5.271 TPS        (fixed; the private deficit to the knife-edge)
flattening_breakeven = tps_shrink_to_3p2 / (0.031 · 489.76) = 0.3472
```

Breakeven requires the slope to lose **65%** (downstream tail collapsing to ≈1.36 tokens, an **~88%** loss of the public 2.90-token tail → per-position deep acceptance ≈0.27). In multiplicative-collapse terms the breakeven is **δ = 0.66 ≈ 1.9× the #263-measured −34.5% collapse**. `flattening_margin` (central) = **+0.546**; conservative = **+0.174**. The central uses **38.9%** of the budget, the conservative **66.6%** — material, but both leave positive headroom.

### Comparison vs. PR baselines

| Quantity | PR baseline | This card |
|---|---|---|
| coverage→gap slope | 489.8 TPS/unit (#379, public-anchored) | **437.3** measured-private / **255.3** conservative |
| slope_flattening_ratio | — (iid assumed = 1.000) | **0.893** measured / 0.521 conservative |
| coverage_target_for_3p2 | 0.9011 (#379, public) | **0.9024** measured / **0.9109** conservative |
| #336 budget (baseline 0.8903) | +0.031 | uses 38.9% (central) / 66.6% (conservative) — **both within** |
| #263 anchor | private rank-2+ collapse −34.5% | maps to deep-tail T2 0.893; **a₁ collapse ⊥ slope** |
| breakeven flattening | — | **0.347** (needs 1.9× the #263 collapse) |
| official TPS | 481.53 / leaderboard 489.6 (unchanged) | **+0 (analysis-only)** |

### Honest analysis — what happened

The public-anchored 489.8 slope **survives the transfer to the private distribution.** The reason is structural, not lucky: the slope's leverage `T2 = dE[T]/da₁` is the *downstream* conditional-acceptance tail, and the private OOD collapse (#263) is concentrated at the **first token** (a₁: 0.729→0.598) — which the slope is mathematically blind to — while the **deep tail holds** (survivor effect). So the leverage that converts a coverage-driven a₁ lift into E[T]/TPS is nearly intact on private.

I treated the obvious objection seriously. **Objection:** the measured deep-tail is survivor-biased; raising a₁ admits harder prompts whose tails are worse, so the measured T2 over-states the *marginal* leverage. **Three reasons it doesn't move the verdict:** (1) the survivor bias is present in the public T2 too, so it largely cancels in the *ratio*; (2) the conservative stress model ignores the deep tail entirely and applies the full −34.5% uniformly — and still clears (0.521 ≫ 0.347); (3) even pinning the *entire* downstream at the #289 low-tail / decode-proxy constant α=0.694 (the adversarial regime) gives 0.772, still clearing. Breakeven needs an ~88% tail collapse — ~1.9× the worst #263 evidence — which no defensible model reaches.

**The one nuance worth flagging for fern #357:** private OOD is *not* free. Under the conservative sizing it roughly **doubles** the budget consumption (38.9% → 66.6% of +0.031) and inflates the target +0.0099 (0.9011 → 0.9109). The route survives, but fern should **bank the conservative private-anchored target ≈0.911, not the bare public 0.9011** — banking the public number would spend the private-OOD safety margin it doesn't account for.

**Orthogonality respected:** this card is the public→private *distribution-shift* transfer only. I did **not** touch denken #380's κ (coverage→accept transfer degraded by int4-ct *quantization noise*, same-distribution) or fern #357's composite. (Soft cross-check, not load-bearing: denken #377 independently models a non-iid flattening of the coverage→E[T] map with central below iid — same direction as this leg's flattening, and inside my [0.52, 0.89] bracket.)

**Why CPU-only (optional GPU leg skipped):** the private per-draft-position profile is *directly measured* in #263 (`he7glotf`), not under-determined, and breakeven (0.347) sits far below every model — a fresh A10G E[ℓ] measurement cannot move the verdict. Staying CPU-only honors the PR's "if the analytic profile is well-bounded by #263, STAY CPU-ONLY" and the standing no-submission posture.

### Suggested follow-ups

1. **Hand fern #357 the private-anchored slope, not the public one:** bank `slope_private ≈ 437` (measured) with the **conservative target 0.9109** as the safe sizing (or 0.9024 if it trusts the measured deep-tail). Banking the bare public 0.9011 spends the private-OOD margin.
2. **(If a real private-VALID slice ever opens) confirm the deep-tail survivor effect off the adversarial proxy:** the `--gpu --proxy google/gemma-4-E4B-it-qat-w4a16-ct --measure-accept-gap` leg (scaffolded, skipped) would replace the #263 adversarial per-position profile with a benchmark-distribution one — expected to *raise* the ratio toward the 0.942 benchmark estimate (confirmation, not a gate).
3. **Marginal-leverage refinement (optional):** if denken/fern want the *marginal* (not average) downstream tail for the newly-admitted prompts, a stratified per-prompt a_k breakdown from the #263 records would tighten the central between 0.52 and 0.89 — but the verdict is already robust across that whole interval.

### Reproduce

```bash
cd target/ && python research/validity/coverage_slope_private_robustness/coverage_slope_private_robustness.py \
    --private-oob-slope --anchor-263-collapse \
    --wandb_group strict-bi-verify-gemm --wandb_name ubel/coverage-slope-private-robustness
# 0-GPU reanalysis: add --reanalyze --no-wandb ; gate: add --self-test
```

- **Self-test:** `slope_robustness_self_test_passes` = **True** (22/22: public T2 round-trips #289's 3.9097; E[T]=1+a₁·T2 identity; slope/target reconstruct #379's 489.76/0.9011; T2⊥a₁; measured milder than conservative; deep tail holds on private; breakeven closed-form == multiplicative-δ solve and exhausts budget; central+conservative both within budget & above breakeven; breakeven needs >1× the #263 collapse; sweep monotone; tps_shrink == #379's +5.27; constants exact; NaN-clean).
- **Peak memory:** 12.11 MiB (pure-stdlib CPU-analytic; no torch/numpy).
- **W&B run:** `bn0v5rqr` (group `strict-bi-verify-gemm`).
- **Public-evidence note:** 0 official TPS, 0 HF Job, 0 `--launch`, 0 submission, 0 served-file change; leaderboard frontier (489.6 public, valid) untouched. CPU-analytic over banked W&B anchors — my #379 slope/targets (`5kpb73tb`), #289 public per-position profile (`fi34s269`), **#263 directly-measured private per-position profile (`he7glotf`)**, #336 coverage budget. The optional local-A10G accept-gap leg was scaffolded but **not run** (private profile already measured in #263; verdict robust without it).
