STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["t68af2yw"],"primary_metric":{"name":"max_private_strict_demand_only_floor","value":483.28},"test_metric":{"name":"supply_lift_required_first_tps","value":17.22},"headline":{"demand_route_reaches_500_on_deployable_floor":false,"demand_route_reaches_500_with_attn_rebuild":false,"max_private_strict_demand_only_floor":483.28,"max_private_strict_demand_only_floor_worst":470.33,"rho_to_1_cap_floor_no_et_credit":469.68,"required_coverage_delta_floor":0.0572,"required_coverage_delta_attn_rebuild":0.0446,"supply_lift_required_first_tps":17.22,"pilot_on_critical_path":false,"accept_headroom_sufficient_for_required_delta":"False","reproduces_377_under_revival":true,"verdict_band":"RED_demand_alone_insufficient_on_honest_base"}}

## Results

**RED — demand-alone does NOT reach PRIVATE-strict-500 on wirbel #378's honest deployable-strict base, at ANY coverage within the #336 +0.031 envelope.** This is a clean verdict FLIP from my own #377 (`+5.44 TPS, in budget, c≥0.9010`), and the flip is driven **entirely by the base move 518.92 → ≤480.8** — the harness is unchanged (it reproduces #377 exactly under the old 518.92 base). The demand route is **necessary-but-not-sufficient**: a supply-side **lm_head-BI** lift must raise the public-strict base **first**.

### The re-pricing (transfer at current coverage 0.8903; req Δcov to private-500; max-private spending the FULL #336 budget)

| public-strict base | source | priv@cov0 (reg / #318-haircut) | residual→500 | **req Δcov** (×#336 budget) | **max-priv @ full #336 budget** (central / worst) | reaches 500? |
|---|---|---:|---:|---:|---:|:---:|
| off-the-shelf 357.32 | #378 VBI=1 off-the-shelf | 348.9 / 342.0 | +151.1 | +0.228 (7.3×) | 374.1 / 364.3 | ❌ |
| **floor 469.68** | **#378 VBI=1 floor** (= my #327 bf16 ceiling) | 450.2 / 449.5 | **+49.8** | **+0.0572 (1.84×)** | **483.3 / 470.3** | ❌ |
| +attn-rebuild 480.8 | floor + #378 `eta_attn=0.0215` (~11 TPS) | 460.2 / 460.2 | +39.8 | +0.0446 (1.44×) | 494.1 / 480.9 | ❌ |
| eta-revival 518.92 | #366/#370 (#377's premise) | 494.6 / 496.6 | +5.44 | +0.0056 (0.18×) | 531.1 / 516.8 | ✅ *(this is #377)* |

Two transfer models agree (instruction 2): the **#373 regression-to-the-mean projection** `project(B;ρ=0.9421)` and the **#318 ρ-haircut** `0.9571·B` land within ~1 TPS on every honest base. The **#379 additive gap identity** (85.25% acceptance / 14.75% ctxlen, irreducible floor 0.633%) is the cross-check ceiling: even driving ρ→1 by closing the *whole* acceptance bucket caps at ρ_max = 0.9937.

### Why the verdict flips (and why it is NOT a harness artifact)

- **#377's affordability was 100% downstream of the 518.92 base.** At 518.92 the public ceiling already sits +19 TPS above the 500 bar, so the regression transfer lands private at 494.6 and the demand closer only has to supply **+5.44 TPS** (Δcov +0.0056 / +0.0107 robust). wirbel #378 shows 518.92 is **not a served knob** — it needs a kernel rebuild that buys only ~11 TPS (`eta_attn=0.0215`, *not* #326's whole-step 0.3141); the dominant strict overhead is **lm_head-BI**, untouched by the attention un-pack.
- **On the honest base the residual is ~7–9× larger.** Floor 469.68 → private 450.2 → residual **+49.8 TPS**; the demand closer would need Δcov **+0.0572** (1.84× the entire #336 budget). The #379 a₁-only slope cross-check is even harsher: **+0.1157**. Both ≫ +0.031.
- **Round-trip is exact (instruction 6):** feeding 518.92 back through *this* harness reproduces private **494.56**, residual **+5.44**, Δcov_central **0.00565**, Δcov_robust **0.01071**, target **0.9010** → `reproduces_377_under_revival = True`. So the flip is the base move, not a re-modeling.

### The load-bearing decision fields for fern #357

- **`demand_route_reaches_500_on_deployable_floor` = False.** Even spending the *entire* #336 budget, the floor base tops out at private **483.3 (central) / 470.3 (worst)** < 500.
- **`demand_route_reaches_500_with_attn_rebuild` = False.** The ~11-TPS attention rebuild (→480.8) lifts the full-budget ceiling only to **494.1 (central) / 480.9 (worst)** — still short in *both* corners.
- **`max_private_strict_demand_only_floor` = 483.28** (central) / 470.33 (worst). The pure gap-closure cap (ρ→1, *no* E[T] credit — private ≤ public) is the base itself, **469.68**. Every reading is < 500.
- **`supply_lift_required_first_tps` = 17.2** (floor, joint E[T]+gap) / **23.8** (floor, E[T]-only/robust) / **6.1** (from the attn-rebuild base). The attention rebuild *alone* (~11 TPS) does **not** close the floor supply gap — `attn_rebuild_alone_closes_supply_gap = False`. The remainder must come from the **lm_head-BI** lever (wirbel's next card).
- **`pilot_on_critical_path` = False.** The ~25 A10G-GPU-hr robust coverage pilot (#352) is **not** the next gating step: it would de-risk *coverage deliverability* (#380's 0.811 robust), but coverage deliverability is moot when demand-alone cannot reach 500 at any coverage on this base. The critical path is the **supply-side lm_head-BI lift first**; the pilot only becomes relevant once the public-strict base clears ~487–493.
- **`accept_headroom_sufficient_for_required_delta` = False** (instruction 7). The required Δcov (+0.057 floor / +0.045 attn) exceeds the **trainable head's own coverage headroom** (the #336 budget *is* `identity_bar − prior` = +0.031). a₁=0.7293 has +0.071 of room vs the #308 published 0.80 envelope, but depths 4–7 are already ≥0.80 (near-ceiling) and the realized coverage→accept transfer is only κ≈0.67 — the head cannot deliver a budget-busting delta.

### Comparison vs PR baselines

| Quantity | PR baseline | This card |
|---|---|---|
| public-strict base demand transfers from | 518.92 (#377 premise) | **≤469.68 floor / ~480.8 attn-rebuild** (#378 `gghmgtk9`) |
| residual to private-500 (central) | +5.44 TPS (#373/#377) | **+49.8 (floor) / +39.8 (attn-rebuild)** |
| required Δcoverage | +0.0056 / +0.0107 robust (#377) | **+0.0572 (floor) / +0.0446 (attn)** — 1.4–1.8× the +0.031 budget |
| demand-alone reaches private-500 | implied yes (#377) | **No — caps at 483.3 (floor) / 494.1 (attn), full budget** |
| official TPS | 481.53 (unchanged) | **+0 (analysis-only, 0 GPU, 0 HF Job)** |

### Honest analysis — what happened

The whole `+5.44 TPS / 35%-of-budget` affordability of the demand route was a **leverage artifact of a high base**: a regression transfer of a base that is +19 above the 500 bar only needs a sliver of demand lift to finish. Strip the un-deployable revival and the demand closer is asked to manufacture a **+40–50 TPS** public lift through acceptance alone — and because the strict serving runs at a *lower* steps/sec (`K_cal_strict` ≈ 122 on the floor vs 125.3 deployed), each unit of coverage buys *less* TPS than #377's high-base arithmetic assumed. The two channels demand can pull — (a) E[T] lift, (b) gap-closure to the #379 irreducible 0.633% floor — are **both** budget-limited: at +0.031 coverage the gap only partially closes (0.0308, not the 0.0063 floor), and the joint ceiling still lands < 500 on both honest bases. The verdict is robust to the central↔worst slope, to regression↔haircut transfer, and to the program-secant↔#379-a₁ inversion. I did **not** stretch a coverage target to "fit" — the honest-band requirement is met: **the deployable base genuinely caps demand-alone below 500 at ρ=1.**

**Bottom line for fern #357:** the demand-side route is **not a standalone GO** on today's served base. It re-centers the composite on the **supply-side lm_head-BI lever** as the critical path: lift the public-strict base by ~17–24 TPS (to ~487–493), *then* the demand closer (now back to a #377-sized sliver) finishes to private-500. Demand + the attention rebuild together still leave a ~6–13 TPS lm_head-BI residual.

### Suggested follow-ups

1. **Hand wirbel the supply target:** the public-strict base must rise to **~487 (joint) / ~493 (E[T]-only robust)** before demand-side coverage retrain can finish within the #336 budget. That is the sizing bar for the lm_head-BI determinism-cost card.
2. **Joint supply+demand isocline:** once wirbel sizes the achievable lm_head-BI lift `ΔB_supply`, this harness inverts directly for the *residual* demand Δcov at base `469.68 + ΔB_supply` — a one-line re-call (`price_base`) gives fern the costed composite.
3. **Re-confirm the floor base** if #378's `[357.32, 469.68]` bracket tightens: every field here is a closed-form function of the base, so a base refinement re-prices in seconds (no GPU).

### Reproduce

```bash
cd target/ && .venv/bin/python research/validity/demand_residual_honest_base/demand_residual_honest_base.py \
    --honest-base --reconcile-377 \
    --wandb_group strict-bi-verify-gemm --wandb_name denken/demand-residual-honest-base
# self-test only (0-GPU, no W&B):
cd target/ && .venv/bin/python research/validity/demand_residual_honest_base/demand_residual_honest_base.py --self-test
```

- **Self-test:** `demand_residual_honest_base_self_test_passes` = **True** (37/37: transfer identities, the #377 round-trip under 518.92, sub-500 honest bases ordered, the Δcov-exceeds-budget flip on every honest base, demand no-clear at full budget, #379 gap-decomp consistency, accept-headroom insufficiency, NaN-clean).
- **Peak memory:** 13.59 MiB (pure-stdlib CPU-analytic; no torch/numpy/GPU).
- **W&B run:** `t68af2yw` (group `strict-bi-verify-gemm`).
- **Public-evidence note:** 0 official TPS, 0 HF Job, 0 `--launch`, 0 submission, 0 served-file change. CPU-analytic over banked merged-branch W&B anchors: wirbel **#378** `gghmgtk9` (honest deployable-strict bracket + `eta_attn`), my **#377** `030uc5mk` (non-iid secants), ubel **#379** `5kpb73tb` (gap decomposition + irreducible floor), my **#373** `oqs8lddd` (regression projection), **#380** `00oijpwg` (deliverability), and the #289 per-position accept profile. No GPU leg required — the verdict rides on closed-form transfer arithmetic, robust across both transfer models and both slope corners.
