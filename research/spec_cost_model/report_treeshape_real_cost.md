<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TPS-optimal draft-tree shape under denken #68's M≤32 real verify-cost curve

**PR:** #74 · **Author:** wirbel · **Date:** 2026-06-14 · **Extends:** #49
(Sequoia DP tree + acceptance model + V(M)) · **Consumes:** #68 (measured
verify-GEMM cost curve, merged) · **Hands off to:** land #71 (tree-verify build)
**Script:** `scripts/profiler/treeshape_real_cost.py` ·
**Results:** `research/spec_cost_model/treeshape_real_cost_results.json` ·
**W&B:** group `tree-shape-cost-model`

**Question.** My #49 proved a Sequoia DP-optimal draft tree gives +34% E[T] / +16%
TPS over the deployed linear MTP K=7 chain — but it priced verify with the *modeled*
tile-corrected curve V(M) (#28/#33). denken #68 (merged) **measured** the real int4
Marlin verify-GEMM cost per verify-width M: bandwidth-bound at M=8 (so widening is
affordable), **non-uniform** (a Marlin 16-row tile staircase), hard **M=33 cliff
(+53%)**. **Does re-optimizing the tree against the real curve shift the optimal
(shape, M) — and what 1–2 topologies should land #71 build first?**

## Verdict / headline

> **The TPS-optimal tree is the M=32 Sequoia DP tree → ~514 local TPS, +20.1% over
> the deployed linear K=7 chain (428.37 local steady). The optimal (shape, M) does
> NOT shift from #49 — it is the same deep-spine DP tree at M=32, now confirmed
> affordable with #68's *measured* cost (the projection even ticks up: +20.1% real
> vs +19.0% under #49's modeled V(M), both vs the deployed M=8 chain). A second,
> cheaper build target emerges at the Marlin tile-1 top — the M=16 DP tree →
> ~485 TPS, +13.1% — the simplest first build. Build to a tile-TOP (M=16 or M=32);
> never stop mid-tile (M=12/20/24), which pays a tile-entry step without collecting
> the near-free interior rows. Hard ceiling M≤32: M=33 craters to ~442 TPS.**

| operating point (real #68 cost, g=0.532, measured p, geom) | E[T] | step mult vs M=8 | proj local TPS | vs deployed linear (428.37) |
|---|--:|--:|--:|--:|
| deployed **linear K=7 / M=8** (anchor) | 2.976 | 1.000 | **428.37** | — |
| linear own-optimum (M=16, saturated) | 3.111 | 1.034 | 433.1 | +1.1% |
| **DP tree M=16** (tile-1 top, *smaller first build*) | **3.481** | **1.034** | **484.7** | **+13.1%** |
| DP tree M=24 (mid-tile — *avoid*) | 3.740 | 1.090 | 493.9 | +15.3% |
| **DP tree M=32** (tile-2 top, **TPS-optimal**) | **3.924** | **1.098** | **514.3** | **+20.1%** |
| DP tree M=33 (past the cliff) | 3.945 | 1.284 | 442.3 | +3.3% (crater) |

**Primary metric** `treeshape_opt_proj_tps_gain_real_costcurve = +0.201` (M=32).
Applying the same gain to the official 481.53 → **~578 official**, if the
local↔official ratio holds (a kernel-contingent projection, not a measured result).

---

## 1 · Why shape-optimization and budget-optimization separate cleanly

A key correctness point that makes this tractable: **the verify cost depends only on
the node budget M, not on the tree topology.** #68 caveat 4 — a width-W tree changes
only the *attention mask*, not the weight GEMM, which processes all M rows
regardless. So for any fixed M, the E[T]-maximizing tree (the #49 Sequoia DP) is
*also* the TPS-maximizing tree, and the global TPS optimum is

```
max_M  [ F_DP(M) / cost(M) ]   with  F_DP(M) = Sequoia-DP E[T] at budget M.
```

No cost-aware DP is needed: I run the #49 DP (unmodified, brute-force-verified) at
each M, then take the outer argmax against #68's real cost(M). This is exactly the
joint solve the PR asks for — (a) my measured acceptance model × (b) #68's real
per-M cost — and it is provably the TPS optimum because cost is shape-independent.

## 2 · The cost model — #68's real curve, anchored to the deployed chain

The full speculative step `S(M) = drafter + verify(M)`. Only the int4 weight-GEMM
block scales materially with M, per #68. I price it two independent ways and they
agree to ≈1pp:

- **Model B (primary, #68 real GEMM curve):**
  `S(M) = S(8)·[(1−g) + g·GEMM₆₈(M)/GEMM₆₈(8)]`, where `GEMM₆₈(M)` is #68's measured
  aggregate `total_gemm_us` and `g` = GEMM share of the decode step (#30/#68: 0.532).
- **Model A (cross-check, #28/#33 modeled full-step V(M)):**
  `S(M) = drafter + V_old(M)` — what #49 used.

`S(8)` cancels in every TPS *ratio*, so the projected **gain** is independent of the
absolute step-time / E[T] anchor; absolute TPS is reported by pinning the deployed
linear M=8 chain to its measured **428.37**. #68's real GEMM ratios (cheap-marginal
sweet spots **bold**):

| M | 8 | 12 | **16** | 24 | **32** | 33 |
|--|--:|--:|--:|--:|--:|--:|
| GEMM₆₈(M)/GEMM₆₈(8) | 1.000 | 1.056 | **1.064** | 1.169 | **1.184** | 1.533 (cliff) |
| step mult cB (g=.532) | 1.000 | 1.030 | **1.034** | 1.090 | **1.098** | 1.284 |

The staircase: rows 13–16 and 25–32 are ~9 µs/row (near-free tile interiors); rows
9–12 and 17–24 carry the tile-entry steps (~64–68 µs/row); M=33 opens a new Marlin
tile (+53%). Model A's full-step curve shows the same structure (tile steps at M=17
and M=33), confirming the micro-structure is real, not a #68 artifact.

## 3 · The optimum did NOT shift — and why that is the right answer

| | #49 (modeled V(M), model A) | **#74 (real #68 curve, model B)** |
|---|--:|--:|
| TPS-optimal M | 32 (pinned by M=33 cliff) | **32 (pinned by M=33 cliff)** |
| optimal shape | deep-spine DP tree | **same deep-spine DP tree** |
| proj TPS vs deployed linear M=8 | +19.0% (509.6) | **+20.1% (514.3)** |

The real curve *could* have demoted M=32 (had the measured M=32 GEMM been
expensive); instead #68 confirms M=32 sits in the cheap tile-2 interior, so the
optimum holds and the projection ticks **up** ~1pp (the real M=32 cost, mult 1.098,
is slightly below the modeled 1.108). **The headline result is reassurance, not a
pivot:** #49's M=32 recommendation survives the cost-model refinement intact. What
the real curve *adds* is the M=16 secondary target and the "avoid mid-tile" rule
(§4) — structure the smooth modeled V(M) could not resolve.

Note the +20.1% (vs deployed M=8) exceeds #49's headline +16%: that figure compared
DP-M=32 vs *linear at its own optimum* (M=16). The deployed chain is M=8, and a
longer linear chain barely helps (linear saturates at the geometric ceiling
1/(1−0.6792)=3.117 → M=16 linear gives only +1.1%). So ~+19pp of the +20.1% is
genuine tree structure breaking the saturation ceiling, ~+1pp is "M=8 isn't even
linear's best budget."

## 4 · Hand-off to land #71 — build these, avoid those

**Build first (primary): the M=32 DP tree — +20.1% (514 TPS).** 32 nodes, depth 9,
9 rank-2+ branch points, max branch 4; a bushy crown (widths 4/5/7/5/6 at depths
1–5) feeding a deep rank-1 spine (depths 6–9). Parent array:
```
[-1,0,0,0,0,1,1,1,2,3,5,5,5,6,7,8,9,10,10,11,13,15,17,17,18,19,20,21,22,28,29,30]
```

**Smaller first build (secondary): the M=16 DP tree — +13.1% (485 TPS).** 16 nodes,
depth 8, 4 rank-2+ branches, max branch 3; crown widths 3/3/4 at depths 1–3 then a
rank-1 spine. The **cheapest verify (tile-1 top)** and simplest topology — a good
Step-0/Step-1 milestone before scaling to M=32. Parent array:
```
[-1,0,0,0,1,1,2,4,4,5,6,7,11,12,13,14]
```

**Avoid mid-tile widths (M=12, M=20, M=24).** They pay a Marlin tile-entry step
without reaching the near-free tile-top: M=24 (493.9 TPS) is strictly dominated by
M=32 (514.3) at nearly the same verify cost (mult 1.090 vs 1.098) but lower E[T];
M=12 is dominated by M=16. **Size the tree to land exactly on M=16 or M=32.**

## 5 · Validation & robustness

- **DP optimality:** brute-force over all labelled rooted trees for n≤7 == DP
  (`selfcheck` PASS); the recommended trees come from that exact validated DP.
- **E[T] correctness:** Monte-Carlo greedy tree-accept (400k trials) matches the
  path-product E[T] for both recommended trees — M=16: 3.4815 vs 3.4813; M=32:
  3.9282 vs 3.9239 (**max rel-err 0.11%**).
- **Decay-robust** (rank-2..4 split): M=32 gain +20.1% (geom) / +16.5% (uniform) /
  +17.2% (sqrt); M=16 +13.1/+9.4/+10.1% — conclusion independent of the split model.
- **GEMM-share robust:** M=32 gain +21.7% (g=.45) … +16.8% (g=.70), central +20.1%
  (g=.532); M=16 +13.7%…+12.0%.
- **Base-acceptance robust (dominant axis):** #68 notes the deployed chain emits
  ~3.8 tok/step; under a geometric linear chain that implies top-1 ≈ 0.775 > the
  measured 0.6792. Even at top-1=0.78 (linear M=8 → 3.92 tok, matching the ~3.8
  observation) the M=32 gain holds at **+18.6%** (top-1=0.74 → +18.0%). The fixed
  M=8 baseline saturates at its geometric ceiling regardless, so the tree's
  ceiling-breaking advantage persists — unlike #49's own-optima framing where high
  acceptance let linear extend its budget and narrow the gap.

**Net:** across cost-pricing method, GEMM share, rank-decay, and base acceptance,
the M=32 DP tree projects **+16.5% to +21.7%** over the deployed linear chain,
central **+20.1%**.

## 6 · Scope & deployability

This is the *build target*, not the build. Realizing it needs a tree drafter +
tree-attention verifier (land #71) — vLLM 0.22's MTP/EAGLE proposer emits a linear
chain and has no tree-verify path (#49 §5). Contract-safe: the recommended tree
stays verifier-authoritative (greedy identity preserved by construction; the tree
only proposes, the target verifies). The projected official ~578 is a cost-model
extrapolation contingent on (a) the kernel existing, (b) the measured acceptance
holding on the served distribution, and (c) the GEMM share. The cost curve consumed
here is #68 (merged); it is not re-derived.

## 7 · Reproduce

```bash
cd target
.venv/bin/python scripts/profiler/treeshape_real_cost.py \
  --wandb-group tree-shape-cost-model --wandb-name wirbel/tree-shape-real-cost
# selfcheck: DP==bruteforce (n<=7) + linear==geometric + MC E[T]==path-product F
# inputs: research/spec_cost_model/verify_gemm_roofline.json (#68),
#         research/spec_cost_model/results_msweep.json (#28/#33 cross-check curve)
# writes research/spec_cost_model/treeshape_real_cost_results.json
```

Peak memory: <0.2 GB (CPU, numpy only). Runtime: ~20 s. Local only, no GPU, no HF Job.
