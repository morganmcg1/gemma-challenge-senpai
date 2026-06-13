<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Sequoia DP-optimal draft tree vs balanced-W4 vs linear — cost-model closure

**PR:** #49 · **Author:** wirbel · **Date:** 2026-06-13 · **Extends:** PR #26 (`tree_acceptance_model.py`), PR #28/#33 (`results_msweep.json`, `report_tree_mask.md`)
**Script:** `scripts/profiler/sequoia_dp_tree.py` · **Results:** `research/spec_cost_model/sequoia_dp_results.json` · **W&B:** `bvbg81v4` (group `sequoia-dp-tree`)

**Question (PR #49):** Sequoia (arXiv 2402.12374, Chen et al. 2024) builds the draft tree that *maximises* E[accepted tokens/step], E[T], by a dynamic program over a fixed node budget, instead of a fixed balanced topology. The PR claims +3–15% TPS over a "fixed width-4 tree". Two sub-questions our prior tree work (#26/#28/#33/#37) never answered:
1. On **our** measured acceptance, is the **balanced** width-4 tree (the locked K=11/M=45 serving guidance) actually optimal, or does a DP-optimal tree beat it — and by how much?
2. Does the answer survive our **measured tile-corrected** verify-latency curve V(M)?

**LOCAL, CPU-ONLY analytic study.** No GPU, no vLLM, no HF Job, no submission, no drafter/greedy/PPL surface touched. It reuses the measured EAGLE-3 acceptance scalars (top-1 = 0.6792, top-4 = 0.8605; PR #16/#26) and the measured `graph|ctx256` verify curve V(M) (PR #28/#33).

> **Premise correction (grounded before running — see PR comment).** There is **no width-4 tree in the served path**. The deployed `fa2sw_precache_kenyan` runs **linear MTP K=7** (`SPECULATIVE_CONFIG={"method":"mtp","num_speculative_tokens":7}`, M=8 verify). vLLM 0.22 has **no tree-attention verify path**, and tree-causal masking is a merged dead-end (PR #33: 0 ms saved on the production dense-SDPA path). So this study is the analytic verdict the PR Notes authorise — it **cannot be deployed** as-is.

## Verdict / headline

| quantity (canonical = **graph, ctx256**, measured p, geom decay, flat drafter 1.446 ms) | linear (deployed family) | balanced-W4 (prior model) | **Sequoia DP-optimal** |
|---|--:|--:|--:|
| E[T] at the deployed budget **M=8** | 2.976 | 2.430 | **3.019** (+1.5% vs linear) |
| E[T] at the locked budget **M=45** | 3.117 | 3.178 | **4.132** |
| **max E[T] gain** vs this column (across budgets) | — | DP/bal = **1.4333** | — |
| **max E[T] gain** vs this column (across budgets) | DP/lin = **1.3409** | — | — |
| TPS-optimal budget n\* | **16** | 31 | **32** (pinned by the M=33 Marlin cliff) |
| TPS at n\* (cost-model scale) | 235.7 | 216.7 | **275.2** |
| **DP TPS gain at each topology's own n\*** | **×1.168** (flat) / ×1.156 (depth-scaled) | ×1.269 | — |
| decay-robustness (geom/uniform/sqrt) of DP/linear TPS gain | — | — | **1.134–1.168** |

**One-line result.** On our measured, rank-1-dominant acceptance, the **balanced width-4 tree is NOT optimal**: a Sequoia DP-optimal tree beats it by **+43% E[T]** and beats the deployed **linear** chain by **+34% E[T]** at matched budget → **+16% TPS** at each topology's own optimum (decay-robust 13–17%). The gain exceeds the paper's +3–15% band *because our acceptance is steeper than the paper's* (rank-1 = 0.68 ≫ rank-2 = 0.11), which makes balanced-W4 especially wasteful, and *because our V(M) is flat to M=32* (the splitkv result, PR #43), making a 32-node tree nearly as cheap to verify as M=8. **But the deployable gain is exactly 0**: vLLM 0.22 has no tree-verify path and the tree-causal mask saves 0 ms (#33), so realising this needs a large gated drafter+verifier build. The result **closes the "is balanced-W4 optimal?" question (no)** and quantifies the ceiling a future gated tree build could chase.

**Secondary find (a tightening of the prior tree ceiling — flagged for advisor review):** the prior model `tree_acceptance_model.py` scores the width-4 tree as a **salvage spine** — geometric at the top-W rate (q = 0.86) with M = K·W+1 — which at M=45 gives **E = 5.99**. That is an **upper bound not achievable by any single-pass drafted tree** of 45 nodes; the honest achievable optimum (path-product DP, Monte-Carlo-verified) is **E = 4.13** (a +45% over-count). The locked **K=11/M=45 → 440 TPS** therefore overstates the achievable tree ceiling; the honest figure is **~248 TPS** at M=45 (cost-model scale), strictly **below the deployed linear frontier** — which makes the team's "trees don't reach 500 / ship linear" conclusion *firmer*, not weaker.

---

## 1 · The acceptance model (path-product) and why it is the right E[T]

A drafted tree is verified greedily in one forward pass with a tree-attention mask. Define `p[k]` = probability the rank-k sibling token is accepted, with `p[1] ≥ p[2] ≥ …` and `Σ_k p[k] = top-W ≤ 1` (the ranks at a node are **mutually exclusive** matches of the target's single argmax). The Sequoia objective is

```
F(T) = Σ_{v ∈ T} path_product(v),   path_product(v) = Π_{u on root→v} p[rank(u)]   (root = 1).
```

**F(T) = E[committed tokens].** At each accepted node the probability of extending the path by one more token = Σ of the present children's `p[rank]`; by linearity of expectation over depths, `Σ_v path_product(v) = E[accepted path length] + 1`. We verify this two ways in `selfcheck()`:
- **Exhaustive:** the DP optimum equals brute-force over *all* labelled rooted trees for n ≤ 7 (4 p-vectors × depth/branch caps).
- **Monte-Carlo:** a greedy tree-accept simulation (200k trials) matches F to <0.02 for linear, balanced, and DP trees at n=16/21.

We derive `p` from the measured cumulative acceptance `C[1]=top-1=0.6792`, `C[4]=top-4=0.8605`, so `p = [0.6792, 0.1097, 0.0494, 0.0222]` (geom decay; uniform/sqrt swept for robustness). A linear chain (W=1) reproduces the team's geometric model exactly (`F = Σ 0.6792^i`).

## 2 · DP-optimal vs balanced-W4 vs linear at matched node budget

`graph|ctx256`, measured p, geom decay (`research/spec_cost_model/sequoia_dp_results.json`, MC-validated):

| M (nodes / verify positions) | linear F | balanced-W4 F | **DP F** | DP/lin | DP/bal | DP depth | in V(M) range |
|--:|--:|--:|--:|--:|--:|--:|:--:|
| 4  | 2.454 | 1.838 | **2.454** | 1.000 | 1.335 | 3 | ✓ |
| 8  | 2.976 | 2.430 | **3.019** | 1.015 | 1.243 | 6 | ✓ |
| 16 | 3.111 | 2.581 | **3.481** | 1.119 | 1.349 | 8 | ✓ |
| 32 | 3.117 | 3.090 | **3.924** | 1.259 | 1.270 | 9 | ✓ |
| 45 | 3.117 | 3.178 | **4.132** | 1.326 | 1.300 | 10 | ✓ |
| 49 | 3.117 | 3.183 | **4.180** | 1.341 | 1.313 | 11 | ✓ |

Reads:
- **At n=4 the DP *is* the linear chain** (F_dp = F_linear = 2.454): when rank-1 (0.68) dominates a weak rank-2 (0.11), the optimal tiny tree does not branch. The DP earns its gains only once the budget is large enough that linear's tail goes to waste.
- **Linear saturates at 3.117** (the geometric limit 1/(1−0.6792)) by n≈24 — every node past ~16 is wasted on a vanishing tail. This is *why* a tree wins under a flat V(M): the wasted budget can be spent on shallow rank-2/3 children instead.
- **Balanced-W4 is the worst topology here** (F=2.43 at M=8): full 4-ary branching spends 3 of every 4 nodes on rank-2/3/4 slots whose marginal acceptance (0.11/0.05/0.02) is tiny. It only overtakes linear past M≈40, where linear has run out of useful depth.
- The DP tree is **deep, not bushy** (depth 9 at M=32): with rank-1 ≫ rank-2 it prefers chains, adding rank-2 children only at shallow, high-value nodes (M=8 DP topology: a near-linear spine with a single rank-2 branch at the root).

## 3 · TPS under the measured tile-corrected V(M)

`V(M)` from PR #28/#33 (`graph|ctx256`, Marlin 16-row tiles): flat to M=32, then cliffs.

| M | 8 | 16 | 32 | **33** | 45 | **49** |
|--|--:|--:|--:|--:|--:|--:|
| V(M) ms | 11.42 | 11.75 | 12.81 | **14.99** | 15.24 | **18.13** |

TPS = E[T] / ((drafter_ms + V(M))/1000), drafter_ms = 1.446 (PR #43, flat) or depth-scaled 1.446/7 per expansion:

- **DP TPS peaks at M=32 (275.2) and craters at M=33 (240.0)** — the DP maximises E[T] (F keeps rising to M=49) but the *TPS* optimum is pinned exactly at the M=33 Marlin tile cliff. The flat-then-cliff V(M) from the splitkv work is what makes a 32-node tree affordable.
- **Each topology at its own TPS optimum:** DP (M=32) = 275.2, linear (M=16) = 235.7, balanced (M=31) = 216.7. DP/linear = **×1.168** (flat) / **×1.156** (depth-scaled drafter); DP/balanced = ×1.269.
- **Decay-robust:** DP/linear TPS gain is 1.168 (geom) / 1.133 (uniform) / 1.140 (sqrt) — the conclusion does not depend on the rank-2..4 split model, matching Sequoia's decay-stability claim.
- **Higher base acceptance shrinks the gain** (DP/linear TPS 1.168 → 1.060 → 1.028 as top-1 0.68 → 0.78 → 0.85): trees help most where the rescue mass is largest, i.e. exactly at our measured operating point.

## 4 · Reconciliation with the prior tree ceiling (salvage spine vs achievable)

`tree_acceptance_model.py` (#26) and `report_tree_mask.md` (#33) score the width-W tree as `E_iid(q, K)` with `q = q_rescue(p,W) ≈ top-W = 0.86` and `M = K·W+1` — a **depth-K spine that succeeds at the top-W rate at every level**. That requires W children at *every* spine node, but a single-pass drafted tree must pre-commit all branches and cannot know the spine in advance; compounding at 0.86 to depth K needs **full W-ary branching (≈ W^K nodes)**, not K·W+1. So the salvage spine is an **optimistic upper bound** (and is *pessimistic* at small K, where it wastes width on a too-short spine):

| M | K | salvage-spine E (q=0.86) | **achievable DP E (MC-verified)** | over-count |
|--:|--:|--:|--:|--:|
| 8  | 1 | 1.861 | **3.019** | spine **under**-counts (−38%) |
| 16 | 3 | 3.238 | **3.481** | −7% |
| 32 | 7 | 5.014 | **3.924** | **+28%** |
| 45 | 11 | 5.987 | **4.132** | **+45%** |

So the locked **K=11/M=45 → 440 TPS** rests on E=5.99, +45% above the achievable 4.13. The honest DP at M=45 is **~248 TPS** (cost-model scale), and the honest tree optimum overall is **275 TPS at M=32** — both **below the deployed linear served frontier**. This *strengthens* PR #33/#37's conclusion (ship linear; trees do not reach 500). I have **not** edited the prior model — flagging it for advisor review as a possible cost-model tightening, since it may have been intended as a deliberate optimistic ceiling.

## 5 · Deployability and scope

Zero deployable gain today: (a) vLLM 0.22's MTP/EAGLE proposer emits a **linear chain** — there is no tree drafter or tree-attention verifier in the served path; (b) PR #33 proved the tree-causal mask saves **0 ms** on the production dense-SDPA path (attention is ~2.6% of the int4 step, GEMM-bound). Realising the +16% ceiling needs a custom proposer+verifier build, gated behind the spec-decode linchpin (Issue #46) and out of scope for a local-only PR. This study is the analytic gate-check that says **the build is worth ~+16% TPS at best, on our distribution** — not nothing, but far from the PR's "fixed-tree replacement" framing (no fixed tree is served) and contingent on a tree-verify kernel that does not exist.

## 6 · Reproduce

```bash
cd target
python scripts/profiler/sequoia_dp_tree.py \
  --wandb-name wirbel/sequoia-dp-tree --wandb-group sequoia-dp-tree
# selfcheck: DP==bruteforce (n<=7) + linear==geometric + MC E[T]==path-product F
# writes research/spec_cost_model/sequoia_dp_results.json
```

Peak memory: <0.2 GB (CPU, numpy only). Runtime: ~30 s.
