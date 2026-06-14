# Deep-tail build-bar budget (PR #215)

**Question.** Given the certified both-bugs private go-bar **0.977978** (#208, `wi4gxxx8`) and
land #71's POSTED interim shallow-mid spine `lambda_spine_min_q2_q7 = 0.997`, what is the
MINIMUM reach-weighted deep-tail acceptance over q[8..9] that still keeps the depth-aggregate
λ̂ = Σ_d ŵ_d·λ_d ≥ 0.977978? Convert the scalar #208 bar into a per-depth build target for
land #71. **CPU-only bank-the-analysis. NOT a launch. Adds 0 TPS. BASELINE stays 481.53.**

## Model (imports — NOT re-derived)

- Reach-weights `w_d_at_bar` (depths 1..7, draft horizon) imported from #208/#203
  (`[0.41283, 0.33589, 0.27098, 0.23950, 0.17173, 0.10332, 0.08953]`).
- Deep tail q[8..9] is OUTSIDE the horizon-7 forward map, so the deep-tail reach mass is
  EXTENDED via #193's β-staleness mechanism (β = 0.765124): `w_8 = w_7·β`, `w_9 = w_8·β`.
  Justified: #203's reach-weights already decay at geomean ratio ≈0.775/rung ≈ β, so the
  β-extension is mechanism-consistent (this is exactly the #193 λ(depth) curve the PR invokes).
- Aggregate = **normalized** reach-weighted mean over the q[2..9] ladder (Σŵ = 1, forced by
  self-test (a): a uniform-0.997 build must aggregate to exactly 0.997). Depth-1 head is the
  liveprobe anchor (λ̂_1, #193) and is excluded from the q[2..9] aggregate; reported as a
  sensitivity arm (depth-1-inclusive).
- Bands: shallow-mid spine q[2..7] (held at λ_spine) | deep tail q[8..9] (unknown).
- Certified bar = 0.9779783323491393 (#208 worst-case-blend, β-band [0.977978, 0.978015]).

## Deliverables

1. `w_mass_shallow_q2q7`, `w_mass_deeptail_q8q9`, `lambda_hat_shallow_only` (deep tail → 0).
2. `min_deeptail_lambda_q8q9_clears_bar` (TEST) + `d_lambdahat_d_deeptail` (= W_deep);
   `deeptail_lambda_mechanism_proj` (#193 import for d∈{8,9}) + `mechanism_clears_bar` + margin.
3. `budget_vs_spine` over λ_spine ∈ {0.990, 0.995, 0.997, 0.999, 1.000};
   `spine_value_where_deeptail_budget_hits_zero` (= bar / w_mass_shallow).
4. **PRIMARY** `deeptail_bar_budget_self_test_passes` (a–e).
5. Hand-off line to land #71 + fern #185.

## Closed form

λ̂ = W_shallow·λ_spine + W_deep·λ_deeptail (W_shallow + W_deep = 1 over q[2..9]).
budget = λ_spine + (bar − λ_spine)/W_deep ; slope vs spine = −W_shallow/W_deep.

Run: `python research/validity/deeptail_bar_budget/deeptail_bar_budget.py --wandb_group deeptail-bar-budget --wandb_name stark/deeptail-bar-budget`
