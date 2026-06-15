STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["8pcyhe2r"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"primary_metric":{"name":"tree_verify_net_tps_self_test_passes","value":1.0},"test_metric":{"name":"g_star_threshold_to_close_500","value":0.2478564028010594}}

## Results

**Verdict: the locked +0.1286 does NOT survive its verify-M tax in the honest single-forward shape — `tree_closes_500_alone = False`.** The verify-M tax scales with the *same width* that buys coverage, so there is no free lunch: at the first tree width that buys any coverage uplift (K=8, full top-K fan-out M=1+7K=57), the step-time tax `τ = 0.412` eats `−136.4 TPS`, and the net at the full ceiling band (g=0.1097) is **−61.6 TPS** (net-negative). The break-even/clear-500 threshold `g* = 0.2479` sits **≫** the gross ceiling band g_max=0.1097 **and** above the locked prize 0.1286 → **a single-forward tree cannot net-supply the +32.53 needed to clear 500 alone.** `tree_verify_net_positive = True` only weakly, via the optimistic depth-1-branch floor shape (branch top-K at one position only). `tree_plus_cb3_required = True`; `reconciles_396_corrected_base = True`.

Pure-CPU analytic card (roofline × demand-secant composition, parameterized over the ceiling band g ∈ [0, 0.1097]). **0 GPU, 0 official TPS, 0 HF Job, no served-file change, no kernel build.** Deployed **481.53 TPS / PPL 2.3772 / 128÷128** (PR #52, `2x9fm2zx`) UNCHANGED; corrected strict base **467.48 (#393 `0q7ynumg`)** UNCHANGED. W&B `8pcyhe2r` (group `tree-verify-net-tps-go-nogo`). Self-test **46/46** (≥20 required). New file: `research/validity/tree_verify_net_tps_go_nogo/`.

### Step 1 — gain leg: coverage → E[accepted] → TPS (deliverable 1)

A depth-1 width-K tree raises per-position accepted-coverage from top-1 (0.7617) toward top-K. I parameterize the realized coverage uplift `g ∈ [0, 0.1097]` over the locked-gap band and map g → TPS along the demand secant `S = 7.9126` (E[T]/cov, #387/#383) with the #289 ladder (E[T]=3.8512, E[accepted]=2.8512, `fi34s269`):

`gross_tps_gain(g) = base · S · g / E[T]` ⇒ **962.27 TPS per unit Δcov** on the corrected 467.48 base (`gross_tps_gain_per_unit_cov`). The old-base cross-check reproduces ubel #399's anchor exactly: **968.57 TPS/unit-cov** on 471.42 (`gross_tps_gain_per_unit_cov_oldbase_399`, matches the PR's ≈968.57). The corrected-base 962.27 is the load-bearing slope for the net.

A **K=4 tree gives g=0** (it just reproduces the deployed top-4 anchor 0.890), so any g>0 requires K≥8 — and that is exactly where the tax bites.

### Step 2 — tax leg: verify-M step-time roofline (deliverable 2)

I state the tree shape explicitly and price two shapes as a band:

- **HEADLINE `full_fanout`** (honest single-forward): top-K candidates at each of the 7 MTP draft slots + 1 bonus row, all verified in one forward → **M(K) = 1 + 7K** (K=4→29, K=8→57, K=16→113).
- **FLOOR `depth1_branch`** (optimistic): branch top-K at ONE position only → **M(K) = 8 + (K−1)** (K=8→15).

The deployed split-KV attention (LOCKED 16-way split, NO kernel change) re-reads KV per query-block, so attention time scales with `N_nonreduction(M) = (ceil(M/BLOCK_Q)+NUM_SEQS)·NUM_KV_HEADS`. #332's geometry: M=8 deployed → N_nr=6 → **96 CTAs > 80 SMs**, occupancy-saturated, BW-floored at 34.9%; the attention lane is **9.51%** of the step. Widening M re-enters that regime linearly in N_nonreduction.

| shape/K | M | N_nr | attn× | τ = tstep_tax_frac | tps_loss (467.48) | net@full_gap | g* |
|---|---|---|---|---|---|---|---|
| full_fanout K=4 | 29 | 18 | 3.00 | **0.1901** | **74.68** | +14.04 | 0.1326 |
| **full_fanout K=8** | 57 | 32 | 5.33 | **0.4120** | **136.39** | **−61.61** | **0.2479** |
| full_fanout K=16 | 113 | 60 | 10.00 | 0.8560 | 215.55 | −158.64 | 0.4784 |
| depth1_branch K=4 | 11 | 8 | 1.33 | 0.0317 | 14.36 | +87.99 | 0.0503 |
| depth1_branch K=8 | 15 | 10 | 1.67 | 0.0634 | 27.86 | +71.44 | 0.0667 |
| depth1_branch K=16 | 23 | 14 | 2.33 | 0.1268 | 52.59 | +41.13 | 0.0997 |

`tps_loss = base · τ/(1+τ)`. **#390 CUDA-graph rebuild:** a tree of a fixed verify width is **one** new static shape → **`n_distinct_graph_rebuilds_for_tree = 1`** (one-time, amortized over the run the way #390 counts rebuilds — negligible per-step). The real hazard is *dynamic* tree shapes (variable accepted-prefix → variable M per step) which fall to the eager path: `dynamic_shape_eager_hazard_x ≈ 2.0×` (the #371 capture-vs-eager gap). The card prices the static-capture case (best case for the tree); even there the roofline tax dominates.

### Step 3 — net + go/no-go (deliverables 3 + 7–10)

`net_tps(g,K) = base · [ (1 + S·g/E[T]) / (1+τ(K)) − 1 ]` (exact multiplicative composition; the linear `gross − loss` form agrees to <1 TPS).

- **`tree_verify_net_positive = True`** — but *weakly*: only the optimistic **depth-1-branch** shape is net-positive at the measured band. The **honest full_fanout @ K≥8 is net-NEGATIVE** (−61.6 at K=8, −158.6 at K=16). The headline `net_tps_at_full_gap = −61.61` (full_fanout K=8); the optimistic floor `net_tps_at_full_gap_optimistic_depth1_k8 = +71.44`.
- **`g_star_threshold_to_close_500 = 0.2479`** (full_fanout K=8, solves net≥32.53). This is the go/no-go boundary: it is **2.26× the gross ceiling band g_max=0.1097** and **1.93× the locked prize 0.1286** → unreachable. Optimistic floor `g_star_optimistic_depth1_k8 = 0.0667` (inside the band — but that shape harvests only one position's top-K, far less gross coverage than the full prize).
- **`tree_closes_500_alone = False`** — at every width that actually buys g>0 in the honest shape, g* ≥ g_max.

### Step 4 — reconcile the bases (deliverable 4)

On the corrected 467.48 (#393), the **tax-free** required d-cov to clear 500 is `required_dcov_tax_free_corrected = 0.03380` (= base/(S·base) · (500/base − 1) along the secant). This reproduces **denken #396's corrected-base finding exactly** (required_dcov ~0.0338 > +0.031 #336 budget → demand-alone busts even *before* any tax) ⇒ **`reconciles_396_corrected_base = True`**. Cross-check vs ubel #399's old-base number: +0.0295 on 471.42 (95% of the +0.031 budget) — the base correction (471.42→467.48) is what pushes the requirement from inside-budget to over-budget. Adding the verify-M tax only makes the tree's *net* requirement strictly worse than this tax-free floor.

⇒ **`tree_plus_cb3_required = True`**: the tree cannot close 500 alone in its honest shape; the corrected-base gap is fundable only by combining a (small, in-band) tree coverage lift with kanna #207's conservative-k cb3 supply lift — the combined route is the only robust plan, consistent with #396.

### PPL (the gate)

A greedy tree verify keeps the longest top-1-matching prefix and emits the **target model's** greedy token exactly → greedy identity preserved → **PPL unchanged at 2.3772 ≤ 2.42 (passes)**. The tree's binding constraint is the **step-time roofline tax**, never PPL — widening the verify changes *speed*, not *output*.

### Deliverables (W&B `summary/`)

`gross_tps_gain_per_unit_cov = 962.27` (oldbase #399 = 968.57); `tree_shape` (full_fanout headline / depth1_branch floor, stated above); `tstep_tax_frac_k4 = 0.1901`, `tstep_tax_frac_k8 = 0.4120`; `tps_loss_k4 = 74.68`, `tps_loss_k8 = 136.39`; `net_tps_at_full_gap = −61.61`; `tree_verify_net_positive = True`; `g_star_threshold_to_close_500 = 0.2479`; `tree_closes_500_alone = False`; `tree_plus_cb3_required = True`; `reconciles_396_corrected_base = True`; PRIMARY **`tree_verify_net_tps_self_test_passes = True` (46/46)**. Flags `analysis_only / no_hf_job / no_served_file_change = True`, `official_tps = 0`.

### Command

```
cd target/ && .venv/bin/python -m research.validity.tree_verify_net_tps_go_nogo.tree_verify_net_tps_go_nogo --self-test
cd target/ && .venv/bin/python -m research.validity.tree_verify_net_tps_go_nogo.tree_verify_net_tps_go_nogo \
  --wandb_group tree-verify-net-tps-go-nogo --wandb_name denken/tree-verify-net-tps-go-nogo
```

Peak memory: **~12 MiB** (stdlib-only analytic card; no numpy/torch/GPU). W&B run `8pcyhe2r`.

### What happened

ubel #399 explicitly held the tree go/no-go out of scope ("converts the locked coverage into a *net* TPS number — the actual tree go/no-go"); this card answers it. The result is structural and decisive: **the verify-M tax is paid in the same width that buys the coverage.** The gain leg pays 962.27 TPS per unit Δcov *only at a fixed T_step* — but a tree that actually moves coverage must widen M, and on the deployed LOCKED split-KV roofline (already occupancy-saturated at M=8 / 96 CTAs > 80 SM, #332) that width inflates attention linearly in N_nonreduction. The two legs are tied to the same K, so the honest single-forward `full_fanout` shape is **net-negative from the first width that buys g>0** (K=8: gain ≤ +105 at the full band, tax −136 → net −62). The clear-500 threshold g*=0.2479 is ~2× the entire locked prize — unreachable by any amount of realized coverage. Only a degenerate `depth1_branch` (branch one position) stays net-positive, and it harvests only a sliver of the +0.1286. Reconciled on the corrected 467.48, the tax-free required d-cov (0.0338) already busts the #336 budget (#396), so the tree's *net* requirement is strictly worse: **the tree does not close 500 alone; it must combine with kanna #207's cb3 supply lift.** The +0.1286 is a real coverage prize, but it is not a *net-TPS* prize at deployed roofline — the demand-route plan should cost the tree against its verify-M tax, not its gross coverage.

### Suggested follow-ups

- **Evaluate at the measured top-8/16 point once ubel #401 lands.** This card emits net as a function of g over [0, 0.1097]; the moment #401 pins the true realized top-8/top-16 coverage, plug that g in for the single measured net number. If the measured g is small (likely, since most of the +0.1286 is top-2/3), the honest-shape net is *more* negative than the full-band figure, hardening the no-go.
- **Price a width-tapered tree.** The full_fanout M=1+7K is the worst case (top-K at *every* draft slot). A tree that widens only the *early* (high-acceptance) positions and stays top-1 on the tail would buy most of the coverage at a fraction of the M-tax — the analytic next step is a per-position width vector w_j minimizing Σ tax(w_j) s.t. realized g ≥ target, which could move depth1_branch's +71 toward a deployable middle ground.
- **Combined tree + cb3 ledger (with kanna #207).** Since `tree_plus_cb3_required = True`, the decision-critical composition is net_tree(g_small) + cb3_supply ≥ 32.53 on 467.48 — a joint card that adds this tree's net curve to #207's cb3 lift would test whether the *combined* route actually clears 500 with both taxes counted, which is the real frontier plan #396 points to.

### Public evidence used

0-submission internal analytic card; **no public leaderboard method reproduced** (roofline + demand-secant composition only). Grounded entirely in the PR #402 baseline anchors named by the advisor: the deployed frontier (PR #52, 481.53 / `2x9fm2zx`), the corrected strict base (#393 `0q7ynumg`, 467.48), the #289 acceptance ladder (`fi34s269`), #387 coverage (`z8osvif8`), the demand secant S=7.9126 (#399 `ec7i3z5t`), the #332 M=8 verify roofline (96 CTAs > 80 SM, 34.9% BW), the #390 CUDA-graph rebuild count, and denken #396's corrected-base required_dcov ~0.0338. No AWS-only numbers reported as challenge results; deployed 481.53 / PPL 2.3772 / 128÷128 unchanged.
