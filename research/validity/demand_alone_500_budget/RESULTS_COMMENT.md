STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["yc5ji486"],"no_hf_job":true,"official_tps":0,"analysis_only":true,"primary_metric":{"name":"demand_alone_500_self_test_passes","value":1},"test_metric":{"name":"required_dcov_demand_alone_500","value":0.02946}}

## Results

**Verdict: KNIFE-EDGE. In the bare deployed-basis frame (#390's 471.42 vs literal-500), demand alone closes the full 28.58 TPS gap with `required_dcov = +0.02946` — INSIDE the +0.031 budget, but using 94.9% of it (headroom +0.00158). At the full +0.031 budget demand-alone buys 501.53 TPS (+1.53). BUT it is NOT robust: even the MINIMUM #389-measured private attn-identity floor (central 0.5764%, which itself clears the 3.2% identity margin) charges the base enough to push `required_dcov` to 0.03244 — just OUTSIDE +0.031. Demand-alone reaches deployed-basis-500 with no robustness margin; the public→private charge busts it. It needs the small cb3 supply assist (#392 residual +0.0117) for a safe private-500 path.**

Pure-CPU analytic card (0 GPU, 0 official TPS, 0 HF Job, no served-file change, no submission). Baseline **481.53 TPS / PPL 2.3772 UNCHANGED**. W&B `yc5ji486` (group `demand-alone-500-budget`). Self-test **35/35** (≥20 required). New file: `research/validity/demand_alone_500_budget/`.

### The demand-alone composition (deliverable 1)

The served stack is MTP K=7 spec-decode: `TPS = (1 + E[accepted]) / T_step`. With **only the demand side moving**, `T_step` (incl. `T_verify`) is **held FIXED** — raising drafter coverage → acceptance does not change the per-step average context length, so the verify-forward attention cost is unchanged (the #389 slope enters as a base stress, not here). Hence **TPS ∝ E[T]**, and the inversion is exact:

- `E[accepted]` from the #289 ladder = **2.851** → `E[T] = 3.851`, matching deployed `E_T_REALIZED = 3.844` to **0.19%** (instruction-4 sanity ✅, `test/b_et_ladder_matches_realized`).
- coverage→E[T] central secant (#383/#387/#340, base-independent) `S = 7.9126 E[T]/cov`.
- corrected strict base **471.4163** (#390 `5y64zbjz`, `realized_shippable_strict_tps_decode`); `gap_to_500 = 28.5837`; λ=1 ceiling 520.953.

`required_dcov_demand_alone_500 = E_T_REALIZED·(500/471.42 − 1) / S = `**`+0.02946`** → `demand_alone_500_inside_budget = True` (94.9% of the +0.031 budget).

### Full-budget demand-alone lift (deliverable 3)

At the entire **+0.031 d-cov** budget ceiling (T_step fixed):

| | value |
|---|---|
| ΔE[T] at full budget | +0.2456 → E[T] 4.0896 |
| `tps_at_full_dcov_budget_demand_alone` | **501.53** |
| `demand_alone_reaches_500` | **True** |
| `demand_alone_margin_tps` | **+1.53** |

So the **entire** #336 budget, spent on demand alone, buys ~30 TPS (471.42 → 501.53), clearing the literal-500 bar by +1.53 — but the bar sits at 94.9% of budget, so the headroom is razor-thin.

### Robustness under the #389 attention slope (deliverable 2)

#389 (`fqt33bj3`) measured the per-L attention-penalty slope at **0.001071** = **0.353×** the #386/#375 pessimistic interpolation (0.003037) — **refuting** the #386 breach (all corners clear the 3.2% identity margin; `pessimistic_breaches_3p2 = False`). I stress the realized base by the irreducible attn-identity floor (the public→private charge) under both the measured and the refuted-interpolated slopes, then re-invert:

| attn-identity charge | floor % | charged base | `required_dcov` | % of +0.031 | inside? |
|---|---|---|---|---|---|
| **bare deployed-basis** (#390 frame) | 0.0000 | 471.42 | **+0.02946** | 94.9% | **✅** |
| #389-measured **central** (L=578, modeled private) | 0.5764 | 468.70 | +0.03244 | 104.5% | ❌ |
| #389-measured **pessimistic** (L=658, worst clearing) | 1.2723 | 465.42 | +0.03610 | 116.3% | ❌ |
| #386-interp pessimistic (**refuted breach**) | 3.5235 | 454.81 | +0.04827 | 155.5% | ❌ |
| full public→private gap (4.29%) | 4.2946 | 451.17 | +0.05258 | 169.4% | ❌ |

Two distinct readings, both reported:
- **`base_holds_under_389_slope = True`** (A): the base 471.42 is a *legitimate* base — the #389 slope is milder than the interpolation (ratio 0.353 < 1), the measured floor sits below the interpolated floor, all corners clear 3.2%, so the base does **not** degrade to the refuted #386-pessimistic value.
- **`robust_under_389_slope = False`** (B, HEADLINE): the demand-alone-500 *conclusion* does **not** survive the private attn-identity charge the slope implies. Even the **minimum** honest private charge — #389-measured central 0.5764%, mild and itself clearing 3.2% — pushes `required_dcov` to 0.03244 > 0.031. `knife_edge_no_margin = True`.

The #389 measurement is decision-useful precisely here: under the **refuted** #386-pessimistic slope, demand-alone would bust the budget by **55%** (0.0483); the measured slope **softens** that to a **mild** bust (+4.5% over budget at the central corner). It turns a hard "no" into a knife-edge "barely no."

### Where this lands vs the program

| route | base | target | demand d-cov | vs +0.031 budget |
|---|---|---|---|---|
| #383 demand-alone @469.68 | 469.68 | **private-500** (pstar 524.95) | +0.0572 | busts (184%) |
| **#396 demand-alone @471.42 (this card)** | 471.42 | deployed-basis-500 | **+0.02946** | **fits bare (95%), busts on any private charge** |
| #392 combined (supply+demand) | 469.68→512.60 | private-500 | residual **+0.0117** | fits (38%), with margin |

Demand-alone is feasible **only** if the public→private gap is ignored (bare deployed-basis frame). The **supply leg is what provides the robustness margin** — the #392 combined route reaches private-500 with a 38%-of-budget demand sliver, whereas demand-alone consumes 95% just to clear the *deployed-basis* bar and then busts on the irreducible private attention floor.

### Deliverables (W&B `summary/`)

`gap_to_500_corrected = 28.5837`; `required_dcov_demand_alone_500 = +0.02946`; `dcov_budget_336 = 0.031035`; `demand_alone_500_inside_budget = True`; `tps_at_full_dcov_budget_demand_alone = 501.53`; `demand_alone_reaches_500 = True`; `demand_alone_margin_tps = +1.53`; `robust_under_389_slope = False`; `base_holds_under_389_slope = True`; `knife_edge_no_margin = True`; `demand_alone_500_self_test_passes = True` (**PRIMARY**, 35/35).

### Command

```
cd target/ && .venv/bin/python research/validity/demand_alone_500_budget/demand_alone_500_budget.py --self-test
cd target/ && .venv/bin/python research/validity/demand_alone_500_budget/demand_alone_500_budget.py \
  --wandb_group demand-alone-500-budget --wandb_name denken/demand-alone-500-budget
```

Peak memory: negligible (stdlib `math`/`json`, no torch/GPU). `analysis_only=True`, `no_hf_job=True`, `official_tps=0`. W&B run `yc5ji486`.

### What happened

The hypothesis asked whether the demand route ALONE — drafter coverage → acceptance → E[T] → TPS, no served-kernel change — closes the corrected 28.58 TPS gap to 500 inside the #336 budget. The honest answer is a **knife-edge**: yes in the bare deployed-basis frame (0.02946 ≤ 0.031, 94.9% of budget, +1.53 TPS at full budget), but with essentially **no robustness margin**. The composition is exact because in demand-alone mode `T_step` is *physically* fixed (raising acceptance doesn't change the per-step context length), so TPS scales linearly in E[T]; the #389 slope therefore cannot enter as a step-time inflation — it enters only as the base stress, and that is where the result is fragile. Charging the **minimum** unavoidable private attention-identity floor (#389-measured central 0.5764%) already exhausts the +0.00158 d-cov headroom. The result holds **only** if we never charge the public→private gap — which a real private-500 win must.

This refines, not contradicts, the program: #383 showed demand-alone to *private*-500 busts (+0.0572); #392 showed *supply+demand* reaches private-500 with a +0.0117 residual; #396 pins the in-between — demand-alone clears the *deployed-basis* bar within budget but cannot absorb even the mildest measured private attention charge. The supply leg (cb3 body shrink, #392) is the load-bearing lever for robustness; demand-alone is a thin necessary-not-sufficient condition.

### Suggested follow-ups

- **Bank the combined route as the private-500 plan, not demand-alone.** #392's M=1 combined route (supply 469.68→512.60, residual demand +0.0117, 38% of budget) is the robust path; this card shows demand-alone has no margin. Recommend fern #357's composite use the combined residual, not the demand-alone d-cov.
- **Sharpen the private attn-identity charge.** I used the #389-measured central floor (0.5764%) as the minimum private charge. The true operative charge depends on the realized private-eval length distribution (L_priv ∈ [528, 658]); pinning the modeled private L would replace the 0.5764–1.272% band with a point and tighten the bust margin (currently +4.5% to +16.3% over budget).
- **Re-test demand-alone if the strict base is lifted.** The knife-edge is set by the corrected base 471.42. Any *supply-side* lift to the strict base (e.g. the #390 Arm-A ceiling 509.78 if the attention split-pin is realized in-config, or a partial cb3 leg) drops `required_dcov` below 0.0294 and restores headroom. Demand-alone only becomes robust once the base clears ~474 TPS.

### Public evidence used

0-submission internal demand-route inversion card; **no public leaderboard method was reproduced**. Checked the shared digest (`?as=senpai`, 2026-06-15): the valid public a10g frontier is **frantic-penguin 489.63 TPS / PPL 2.3774** (split-KV verify: osoi5+e1+lmhead12k+fa2sw+precache+skv64), with hayai-agent 489.61, openevolve 489.00, need-for-speed 488.07 (valid) close behind — all PPL ≈ 2.377 ≤ cap, none ≥ 500 valid, confirming **private-500 is still the open bar**. The card is grounded in the internal frontier (PR #52, 481.53 / `2x9fm2zx`), the #390 corrected strict base (`5y64zbjz`), the #389 attention slope (`fqt33bj3`), and the #336 demand budget (0.031035), per the PR baseline table.
