STUDENT denken:
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["3zr7i8ad"],"no_hf_job":true,"official_tps":0.0,"analysis_only":true,"no_served_file_change":true,"primary_metric":{"name":"tree_per_position_width_dp_self_test_passes","value":1.0},"test_metric":{"name":"max_net_tps_combined_vs_500","value":6.821740701509555}}

## Results

**Verdict: the net-TPS-optimal interior tree shape is `w* = (3,2,2,1,1,1,1)` (M=12), and its honest net is `max_w net_tps(w) = +1.33 TPS` — net-POSITIVE but NEGLIGIBLE.** The DP sweeps the full per-position-width interior #402 never priced, and confirms #402's headline robustly: the verify-M tax eats the head-ceiling coverage prize at every width that buys non-trivial `g`. The combined plan `tree(w*) + cb3` does clear 500 (506.82), **but cb3-optimistic alone (+38.02) already clears it (505.50)** — the tree contributes a +1.33 TPS cushion, not the lever. `tree_net_positive = True` (at the calibrated β; **fragile** — see β-sweep). 

Pure-CPU analytic card (exact bounded-knapsack DP × the banked #402 roofline-tax). **0 GPU, 0 official TPS, 0 HF Job, no served-file change, no kernel build, `analysis_only=True`.** Deployed **481.53 TPS / PPL 2.3772 / 128÷128** (`2x9fm2zx`) UNCHANGED; corrected strict base **467.48 (#393 `0q7ynumg`)** UNCHANGED. W&B **`3zr7i8ad`** (group `tree-per-position-width-dp`). Self-test **42/42**. New file: `research/validity/tree_per_position_width_dp/`.

**Public evidence used:** extends my merged **#402** (`8pcyhe2r`, the two-corner tree-verify go/no-go — imported as a Python module so the tax/secant/base are byte-identical) and its banked anchors **#401** (`i2qsjyp6`, g_max=0.1097, deployed MTP top-1=0.7293), **#387/#340** (top-1=0.7617, top-4=0.890 coverage), **#289** (`fi34s269`, per-position acceptance ladder), **#332** (`y5cl0ena`, M=8 verify roofline), **#388** (kanna cb3 realized +38.02). No new HF artifact consumed.

### Unifying the two #402 corners as a per-position-width vector (deliverable 5 self-test)

#402 priced only two corner shapes. Both are the same row-count model `M(w) = 1 + Σ w_p`:
- `full_fanout` = top-K at every slot = `(K,K,K,K,K,K,K)` → M=1+7K (K=8→**57**).
- `depth1_branch` = top-K at one slot = `(K,1,1,1,1,1,1)` → M=8+(K−1) (K=8→**15**).

Reusing #402's `attn_scale(M)` byte-exactly, the DP reproduces both corners at the ceiling coverage g=g_max to **<1e-9**: `net(full_fanout,8) = −61.61`, `net(depth1,8) = +71.44` ✓. In that same optimistic ceiling frame the DP dominates both hand-picked corners (cheapest tree M=9 → **+87.99** ≥ max(corners)) — the self-test consistency check. Perturbation sanity `g(all-ones)=0` ✓.

### Step 1 — honest saturating per-position coverage g(w) (deliverables 1–2)

Each position p has miss-room `u_p = 1 − a_p` (from the #289 ladder a=[0.7293,…,0.8465]; early positions have the **largest** room → steepest slope, exactly the taper the hypothesis predicts). Widening to width `w_p` captures a geometric top-k tail fraction:

`captured_p(w_p) = u_p · (1 − β^(w_p−1))`,  `g(w) = g_max · Σ_p captured_p(w_p) / Σ_p u_p`

with **β = ((1−top4)/(1−top1))^(1/3) = 0.7722** calibrated from the banked program anchors: a program top-k chain `cov(W)=top1+(1−top1)(1−β^(W−1))` hits the top-4 anchor 0.890 at W=4 **iff** that β, and then the residual head-ceiling band `(1−top1)·β³ = 1−top4 = g_max` reproduces 0.1097 **exactly**. So `g(all-ones)=0`, `g(all→∞)→g_max`, saturating and concave per position.

### Step 2 — the tax is a STEP function of M (the binding structure)

`tps_loss(M) = base·τ(M)/(1+τ(M))`, `τ(M)=F_attn·(N_nr(M)/N_nr(8)−1)`, `N_nr(M)=(⌈M/4⌉+1)·2` (#402, LOCKED 16-way split-KV, no kernel change). Because `⌈M/4⌉` only increments every BLOCK_Q=4 rows, **tps_loss is flat within each 4-row tier** and jumps between tiers:

| M tier | slots over baseline | tps_loss | net@g(w*) for that tier's best shape |
|---|---|---|---|
| 8 | 0 (all-ones, no tree) | 0.00 | 0.00 |
| 9–12 | 1–4 | **14.36** | **+1.33** (slots=4, M=12) ← optimum |
| 13–16 | 5–8 | 27.86 | −1.28 (slots=8, M=16) |
| 17–20 | 9–12 | 40.61 | −4.92 (slots=12, M=20) |

The DP packs the steepest-slope early positions up to the top of the **first** tier (M=12) and stops — entering tier 2 costs +13.5 TPS of tax to buy <+3 TPS of coverage.

### Step 3 — the DP optimum (deliverables 3, 5, 6, exact)

`net_tps(w) = base·[(1 + S·g(w)/E[T])/(1+τ(M(w))) − 1]` (exact multiplicative; the PR's named linear form `g·962.27 − tps_loss` gives +1.82, overstating by the cross-term). Exact bounded-knapsack DP over per-position widths (cap 16 = #401 top-k support), independently cross-checked by a marginal-greedy optimizer — **greedy==DP** ✓:

- **w\* = (3,2,2,1,1,1,1)**, M=12, slots=4, **g(w\*) = 0.0168** (15% of g_max).
- **max_w net_tps(w) = +1.327 TPS** — net-positive (deliverable b: **yes, some tree beats M=8**, barely).
- Far below the optimistic depth1 corner (+71.44): that corner credited a single position with the **full** g_max; the honest saturating model gives it ~0.013, confirming #402's flag that the depth1 corner is unrealizable.

### Step 4 — combine with cb3, does it clear 500? (deliverable 6 / fern #357 input)

| plan | TPS | clears 500? |
|---|---|---|
| corrected base (#393) | 467.48 | — |
| **cb3 alone** (#388 +38.02 optimistic) | **505.50** | **✓** |
| tree(w*) alone (+1.33) | 468.80 | ✗ |
| **tree(w\*) + cb3** | **506.82** | **✓ (+6.82)** |

**The combination clears 500, but cb3-optimistic alone already clears the 32.53 gap (+38.02 > 32.53).** The tree's honest net contribution is **+1.33 TPS** — a cushion, not a co-requirement. For fern #357's GO/NO-GO this is the decisive input: the tree cannot substitute for a cb3 shortfall larger than ~1.3 TPS, so if kanna #403's conservative-k cb3 re-cost lands **below** the 32.53 gap, the per-position tree does **not** rescue it.

### Robustness — the net-positivity is fragile to β (deliverable analysis)

| β (tail decay) | w* | max_net | net-positive? |
|---|---|---|---|
| 0.70 (faster coverage accrual) | (3,3,2,2,2,2,1) | **+6.34** | yes |
| **0.772 (calibrated)** | (3,2,2,1,1,1,1) | **+1.33** | yes |
| 0.85 (slower accrual) | (1,1,1,1,1,1,1) | **+0.00** | **no — no tree beats M=8** |

At β=0.85 the optimum is all-ones (the tax outruns coverage at every width). So `tree_net_positive` is **true only in a band around the calibrated β**, and is at most a few TPS even at the optimistic end. This sharpens #402's `tree_closes_500_alone = False` to: *the tree is not a materially net-positive lever at all under conservative coverage accrual.*

### PPL / greedy identity (deliverable 7)

The shape search changes only **which candidate rows are verified**, never which token is emitted: a greedy tree verify keeps the longest target-argmax-matching path → emitted token = target greedy token → **greedy identity preserved → PPL unchanged 2.3772 ≤ 2.42** for every w. Confirmed across the search.

### Command

```bash
cd target/ && .venv/bin/python -m research.validity.tree_per_position_width_dp.tree_per_position_width_dp \
  --per-position-width-dp --wandb_group tree-per-position-width-dp --wandb_name denken/tree-per-position-width-dp
# self-test: .venv/bin/python -m research.validity.tree_per_position_width_dp.tree_per_position_width_dp --self-test
```

(Note: the PR-body path `tree_verify_net_tps/…` resolves to my merged #402 module `research/validity/tree_verify_net_tps_go_nogo/`; this card's new `--per-position-width-dp` driver lives at `research/validity/tree_per_position_width_dp/`. The documented `--corrected-base/--tau/--secant` flags parse and pin to the banked 467.48 / #402-calibrated τ / 962.27.) Peak mem **12.1 MiB**, 0 GPU.

### What happened

The hypothesis was right about the **shape** (the optimum taper-allocates width to early, steep-slope positions: `(3,2,2,1,1,1,1)`) but the **magnitude** is the story: the verify-M tax is a 4-row step function, and the head-ceiling coverage prize (g_max=0.1097 → only ~105 TPS at full saturation M=57) is too small to survive the tax beyond the first tier. The net-optimal tree harvests ~15% of the ceiling for **+1.33 TPS**. This is a clean, honest interior result that **confirms and sharpens #402**: no tree shape — corner or interior — is a meaningful standalone 500-clearer, and `tree_plus_cb3_required` stands, with the new nuance that the tree's share of that burden is negligible (cb3 must carry essentially all of the 32.53 gap).

### Suggested follow-ups

1. **Resolve the cb3 number, not the tree.** Since tree(w*) ≈ +1.3 TPS, fern #357's GO/NO-GO collapses to "does kanna #403 conservative-k cb3 ≥ 32.53 alone?" Prioritize landing #403's conservative re-cost; the tree adds no decision-relevant headroom.
2. **β is the one soft input.** β=0.772 is calibrated to top1/top4/ceiling, but the per-position application is a model choice. A direct top-8/16 per-position coverage read (the #401 measurement the GPU read was blocked on) would replace the β-sweep with measured marginals and settle whether the optimum is +1.3 or 0.
3. **Tax-tier targeting is the only tree knob with leverage.** The optimum sits exactly at M=12 (top of tier 1). If a cheaper verify-M attention (e.g. a width that stays inside N_nr=6, i.e. M≤8) could be found, the tax floor would move — but #332 already shows M=8 is occupancy-saturated, so this door looks closed without a kernel change (out of scope here).
