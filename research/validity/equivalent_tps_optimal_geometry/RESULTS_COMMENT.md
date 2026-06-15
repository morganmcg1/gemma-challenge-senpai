STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["se8mf9ax"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"k_star":7,"m_star":8,"equiv_tps_at_kstar":478.93,"equiv_tps_gain_vs_deployed7":0.0,"kstar_below_7":false,"equiv_tax_at_m8_used":2.6,"neartie_frac_model":"rows_linear","kstar_robust_across_neartie_models":true,"kstar_ever_below_7_in_band":true,"max_equiv_gain_anywhere_in_band":0.948,"equivalent_tps_optimal_geometry_self_test_passes":true,"primary_metric":{"name":"equiv_tps_at_kstar","value":478.93},"test_metric":{"name":"equivalent_tps_optimal_geometry_self_test_passes","value":1.0}}

## Results

**Headline — the hypothesis is LARGELY REFUTED. K\* = 7 = deployed. The deployed draft is NOT too long for the equivalence objective.** `equiv_tps_gain_vs_deployed7 = +0.000 TPS` at the nominal operating point, and K\* dips to 6 only in a thin sliver at the expensive-drafter band edge, where the gain is still **< 1 TPS** (max +0.948 anywhere in the deployment-consistent band). `equiv_tps(K\*=7) = 481.53 − 2.6 = 478.93 TPS`, which reconciles the #397 selective-recompute band [476, 479] and sits above the #393 blanket-strict floor 467.48.

### Headline fields (PR deliverable)

| field | value |
|---|---|
| `equivalent_tps_optimal_geometry_self_test_passes` (**PRIMARY**) | **True** (50/50 checks, ≥20 required) |
| `k_star` | **7** (== deployed) |
| `m_star` | **8** |
| `equiv_tps_at_kstar` | **478.93 TPS** |
| `equiv_tps_gain_vs_deployed7` | **+0.000 TPS** |
| `kstar_below_7` | **False** |
| `equiv_tax_at_m8_used` | 2.6 (#397; one-line calibratable, #412 supersedes) |
| `neartie_frac_model` | rows_linear (nominal) |
| `kstar_robust_across_neartie_models` | **True** (all 4 models give K\*=7 at center) |
| `kstar_ever_below_7_in_band` | True (only the `hi`/expensive-drafter edge) |
| `max_equiv_gain_anywhere_in_band` | +0.948 TPS (**< 1**) |
| scope | `analysis_only`=True, `no_hf_job`=True, `no_served_file_change`=True, `official_tps`=0 |

### The K-sweep (nominal t_d = 0.0425, rows-linear near-tie model)

| K | M=K+1 | E[T] | N_nr(M) | τ_step | fast_tps | equiv_tax | **equiv_tps** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | 6 | 3.3855 | 6 | −0.0850 | 462.61 | 1.857 | 460.749 |
| 6 | 7 | 3.6377 | 6 | −0.0425 | 475.02 | 2.229 | 472.787 |
| **7** | **8** | **3.8512** | **6** | **0.0000** | **481.53** | **2.600** | **478.930 ← K\*** |
| 8 | 9 | 4.0319 | 8 | +0.0742 | 469.32 | 2.971 | 466.346 |
| 9 | 10 | 4.1848 | 8 | +0.1166 | 468.59 | 3.343 | 465.250 |

### Why K\* = 7 (two mechanisms kill the "shorter K is cheaper" intuition)

1. **The verify-M roofline is FLAT across M ∈ {5,6,7,8}.** `N_nr(M) = (⌈M/4⌉+1)·2` = **6** for all four widths (BLOCK_Q=4 query-block quantization → one shared tile tier; imported byte-exact from #402). So trimming K=7→6 (M=8→7) recovers **exactly ZERO** step-time roofline tax. To drop a query-block tier you must reach **M≤4 (K≤3)**, which collapses E[T] from 3.851 to 2.722 — **losing 1.129 tokens/cycle**, far too much acceptance.
2. **The marginal identity tax of the LAST draft step is small.** The 2.6-TPS figure is the *absolute* M=8 batched-verify tax, but the *marginal* cost of the 7th draft step (M=7→8) is only **+0.371 TPS** (rows-linear), **+0.690** (quadratic), **+0.963** (cubic), or **0** (reductions_nr / query-block model). Cutting K=7→6 buys back only that small marginal tax while it **loses ~6.5 TPS** of raw acceptance speed (475.02 → 481.53 fast_tps) for any drafter cost in the band interior. The premise conflated the absolute M=8 tax with the marginal cost of the 7th draft step.

### Robustness (the one soft model input, handled like β in #409)

- **K\* = 7 across all 4 near-tie growth models at the nominal operating point** (rows_linear, reductions_nr, rows_quadratic, rows_cubic) → `kstar_robust_across_neartie_models = True`.
- **reductions_nr (query-block model): K\* = 7 across the ENTIRE band** [lo, center, hi] (flat N_nr ⇒ zero marginal tax ⇒ never trims).
- K\* dips to 6 ONLY at the `hi` (expensive-drafter) band edge for rows_linear/quadratic/cubic, where K=7 is **already a raw near-tie** with K=6. K\*-flip crossover t_d: rows_linear 0.05475, quadratic 0.05410, cubic 0.05355, reductions_nr never — all in the top ~3% of the band [0.0296, 0.0554].
- **Base-invariant**: K\* at the high edge is identical anchored at MU_P=481.53 (fast) vs BASE_467 (floor). The K\* sign does not depend on which equivalence base you pick.

### The flagged approximation (PR Instruction 3: "report the assumption")

The PR said to cost the draft forward "from the same roofline the verify uses." I checked the deployed manifest + `sitecustomize.py`: the drafter is a **SEPARATE small model** (`/tmp/qat-assistant`, `SPECULATIVE_CONFIG method=mtp num_speculative_tokens=7`) drafting **autoregressively width-1** for K iterations — **NOT** an on-target MTP head sharing the target body. Costing it from the target verify roofline would over-cost it and would make K=7 NOT raw-optimal — **refuted by deployment**, which settled on K=7. So I treated the drafter per-forward cost `t_d` as the genuinely-uncertain parameter and **pinned its plausible band by deployment-consistency**: `t_d ∈ [0.0296, 0.0554]` is exactly the range for which the deployed K=7 is the raw-TPS argmax (revealed preference, center 0.0425). **The headline (K\*=7) is robust to this choice**: mechanism (1) — the flat roofline — is independent of `t_d` entirely, and the K\* flip never happens before the top 3% expensive-drafter edge.

### Greedy identity (exact by construction, PPL unchanged)

The linear-chain spec verify emits the target's argmax token at the first mismatch (drafter only PROPOSES), so the emitted token is the target greedy token at **every K** → PPL **unchanged 2.3772 ≤ 2.42** for all K. The equiv_tax is purely the cost of making the M=K+1 *batched* verify byte-identical to the M=1 sequential reference (removing the #381/#405 reduction-order flips); at M=1 (K=0) there is no batch → 0 tax → trivially identical. The tree dimension (M>K+1) is already closed negligible by my #409 (+1.33 TPS, β-fragile); this card is the linear chain only.

### Reproduce (0-GPU, stdlib-only)

```bash
cd target/ && .venv/bin/python -m research.validity.equivalent_tps_optimal_geometry.equivalent_tps_optimal_geometry --self-test
cd target/ && .venv/bin/python -m research.validity.equivalent_tps_optimal_geometry.equivalent_tps_optimal_geometry \
  --wandb_group equivalent-tps-optimal-geometry --wandb_name denken/equivalent-tps-optimal-geometry
```

- **Peak memory:** 12.1 MiB (pure-CPU, no GPU, no HF Job)
- **W&B run:** `se8mf9ax` (entity wandb-applied-ai-team, project gemma-challenge-senpai; 136 summary keys, `summary/` prefix)
- **Self-test:** 50/50 checks pass (provenance, E[T] ladder, roofline flat-tier, cost model + deployment-consistency, equiv_tax calibration, the decision, robustness, base-invariance, PPL, numeric hygiene)

### What happened — honest analysis

The hypothesis (deployed draft too long for equivalence; shorter K nets higher strictly-equivalent TPS) **does not hold**. The advisor's premise rested on the equivalence tax growing fast enough with M that a shorter chain would pay it back. It doesn't, for two independent reasons that both trace to **banked, non-tunable facts**: (a) the #402 verify roofline is BLOCK_Q=4-quantized, so N_nr is flat across the entire M=5–8 window — a K=7→6 trim buys exactly zero step-time tax; and (b) the 2.6-TPS M=8 tax is an absolute batched-verify level, but its *marginal* last-step component (what a trim actually recovers) is sub-TPS under every growth model. Meanwhile K=7 sits right at the E[T]/cost sweet spot (the next tier-jump is M=9, K=8, which both over-costs the verify roofline AND adds tax). So under strict equivalence the deployed geometry is **already optimal** — the equivalence constraint does not move the optimum off 7. The decision-useful corollary: **the path to higher strictly-equivalent TPS is lowering the absolute M=8 tax (the EQUIV_TAX_AT_M8 anchor), not re-shaping the geometry.** When stark #412 lands its measured M=8 tax, swap it into `EQUIV_TAX_AT_M8` (one line) and `equiv_tps(7)` updates directly; K\* will not move (the marginal/flat-roofline argument is tax-level-invariant in sign).

### Suggested follow-ups

1. **Swap in stark #412's measured M=8 tax** (one-line `EQUIV_TAX_AT_M8` change) and re-stamp `equiv_tps(7)` — that's the live lever for strictly-equivalent TPS, not K. If #412 measures the tax materially below 2.6, equiv_tps(7) rises toward the 481.53 fast frontier with K unchanged.
2. **Per-position M-asymmetric verify** is the only geometry lever left that could matter: if the *last* drafted position could be verified at a strictly lower-flip precision (it carries the smallest marginal acceptance), you might shave the absolute tax without losing E[T]. This is an equiv_tax-engineering question, not a draft-length question — out of scope here, flagged per boundaries.
3. **The BLOCK_Q=4 flat tier (M5–8) means M can be pushed to 8 "for free" on the roofline** but K (acceptance) saturates first; the binding constraint is E[T] saturation, not the tile. If a future drafter lifts the #289 ladder's late rungs (a_6, a_7 ≈ 0.836/0.846), K\* could move to 8 (M=9 jumps a tier but E[T] gain might pay for it) — re-run this card against any new ladder.

### Public evidence used

Human re-scope **#407** (maximize fastest strictly-equivalent TPS, forget 500). Banked byte-exact: #402 roofline `N_nr` + bases (run `8pcyhe2r`), #393 strict base 467.48 (`0q7ynumg`), #289 acceptance ladder + E[T]=3.851 (`fi34s269`), #399/#387 secant (`ec7i3z5t`/`z8osvif8`), #378/#393 served attn step-fraction 9.5%, #332 BLOCK_Q tiling (`y5cl0ena`). Identity tax anchor #397 (selective recompute ~2.6 TPS @ M=8, stark **#412** measuring → supersedes). Flip mechanism #381/#405 (3/882 @ M=8). Tree dimension closed by my #409 (+1.33 TPS, β-fragile). Nothing re-derived; the only new modelling is E[T](K), T_cycle(K) with deployment-pinned t_d, and equiv_tax(M) under 4 growth models.
