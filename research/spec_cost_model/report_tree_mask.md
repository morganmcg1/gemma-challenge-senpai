<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Tree-causal mask verify cost + Marlin tile boundary — cost-model closure

**PR:** #33 · **Author:** denken · **Date:** 2026-06-13 · **Extends:** PR #26 (`tree_acceptance_model.py`), PR #28 (`report_msweep.md`, `results_msweep.json`)
**Question (two open follow-ups from PR #28 §6):**
1. **(#1 tree mask)** The PR #28 verify curve was measured with a **dense-causal** mask. A real tree-verify only needs each draft token to attend its **ancestors**, not all earlier draft tokens. Does swapping the dense-causal among-token block for a **sparse tree-causal** mask buy back measurable verify latency at the tree shapes we actually use (K=6/8/12, W=4 → M=25/33/49)?
2. **(#3 Marlin tiles)** The PR #28 sweep is coarse around M=20 and M=40. Marlin W4A16 tiles the GEMM in **16-row** blocks (`thread_m_blocks = ceil(M/16)`), so the verify step should jump at **M=17, 33, 49**. `LatencyCurve` interpolates those boundaries **linearly**, which *under-states* the cost of a tree whose M lands just past a tile cliff. Measure the cliffs directly and fold them in.

**LOCAL profiling only** — same harness (`scripts/profiler/spec_cost_model.py`), same int4 `google/gemma-4-E4B-it-qat-w4a16-ct` base, same `graph|ctx256` config. No HF Job, no submission, no drafter, no greedy/PPL surface touched.

## Verdict / headline

| quantity (canonical = **graph, ctx256**, p=0.78, W=4) | PR #28 dense baseline | **this PR (tree-masked + tile-corrected)** |
|---|--:|--:|
| K=6 tree TPS @ p=0.6792 / 0.78 / 0.85 (w/ drafter) | 331.2 / 375.1 / 409.7 | **333.0 / 377.1 / 411.9** (tree mask: **+0.5%**) |
| tree-mask saving at M=25 / 33 / 49 (production SDPA path) | — | **0.000 / 0.000 / 0.000 ms** |
| tree-mask saving at M=25 / 33 / 49 (FLOP-ideal ceiling) | — | **0.076 / 0.104 / 0.175 ms** (≤1.1% of step) |
| Marlin cliff M=17 / 33 / 49 (measured Δ vs prior M) | (interpolated, hidden) | **+0.772 / +2.176 / +2.869 ms** |
| M=49 verify step (direct) | 15.45 ms (msweep interp) | **18.13 ms** (interp under-stated by **2.68 ms**) |
| **tree K\* @ p=0.78 (w/ drafter / verify-only)** | K\*=12 (M=49): 452.4 / 493.4 | **K\*=11 (M=45): 440.4 / 480.8** |
| K=12 (M=49) tree TPS @ p=0.78 (w/ drafter) | 452.4 (artifact) | **393.9** (the honest M=49 number) |
| **>500 TPS @ p=0.78 at full scale?** | NO (max 452/493) | **NO — max 440/481; even more firmly refuted** |

**One-line result:** the tree-causal mask buys **essentially nothing** (≤0.18 ms, and **exactly 0** on the production dense-attention path), because core attention is only ~2.6% of the int4 verify step. The real find is the **Marlin tile staircase**: the verify step jumps **+2.18 ms at M=33** and **+2.87 ms at M=49**, so PR #28's linear interpolation **under-stated M=49 by 2.68 ms**. Correcting it moves the tree optimum from **K\*=12 (M=49) to K\*=11 (M=45) = 440 TPS** — a **12%-cheaper verify for ~the same accepted length, for free** — and kills the borderline-500 reading the un-corrected curve produced.

---

## 1 · Why the tree mask cannot help (mechanism)

The int4 verify step decomposes as
`t_step(M) = GEMM(M)  +  core-attn softmax(QKᵀ+mask)V  +  RoPE/KV-write  +  lm_head`.
Only the **core-attn** term sees the mask. Everything else (the W4A16 Marlin GEMMs, RoPE, the paged KV write, and the `lm_head` over M positions) is **mask-independent**. The tree mask shrinks only the **among-token** sub-block of core attention — from `M(M+1)/2` dense-causal pairs to the sum of root→node path lengths — while the much larger `M·ctx` cross-context block and all GEMM are untouched.

At our shapes the among-token block is tiny next to the cross-context block:

| M | K | dense among-M pairs | tree among-M pairs | ctx pairs (M·256) | FLOP ratio tree/dense | ideal core-attn saving |
|--:|--:|--:|--:|--:|--:|--:|
| 25 | 6 | 325 | 73 | 6 400 | 0.963 | 3.75% |
| 33 | 8 | 561 | 105 | 8 448 | 0.949 | 5.06% |
| 49 | 12 | 1 225 | 169 | 12 544 | 0.923 | 7.67% |

Core attention is itself only **~13–16% of the step** (and most of *that* is the un-maskable ctx block), so a ≤7.7% FLOP cut on it is **≤1.1% of the step**.

### Measured three ways (`results_tree_mask.json`)

A microbench builds the real `[L=42, Hq=8, M, D=256]` bf16 Q and `[L, Hkv=2, ctx+M, D]` K/V (GQA 4:1 replicated), then times the among+ctx attention under dense vs tree masks:

| M | **SDPA** dense→tree (production path) | **FlexAttention** dense→tree (block-sparse) | **FLOP-ideal** saving (unrealizable ceiling) |
|--:|--:|--:|--:|
| 25 | 0.311 → 0.311 ms (**Δ 0.000**) | 0.353 → 0.441 ms (**Δ −0.088**) | 0.076 ms |
| 33 | 0.351 → 0.351 ms (**Δ 0.000**) | — (**Δ −0.083**) | 0.104 ms |
| 49 | 0.394 → 0.394 ms (**Δ 0.000**) | — (**Δ −0.147**) | 0.175 ms |

- **SDPA** (`F.scaled_dot_product_attention(attn_mask=bool[M,ctx+M])`) is the path **every production tree-verify uses** (SpecInfer Eq 4, EAGLE, Medusa, vLLM): dense `O(M·(ctx+M))` attention + a topology mask. Passing the tree mask instead of the causal mask changes **which** scores are masked, not **how many** are computed → **saving = 0 by construction.**
- **FlexAttention** only skips fully-masked **128×128** blocks; at M≤49 the entire tree fits inside **one** 128-block, so it skips nothing and pays partial-block overhead → **negative** saving. (`BLOCK_SIZE<128` is a hard compiler error — pytorch #133562 — so sub-block tree sparsity is unreachable on this stack.)
- **FLOP-ideal** assumes a kernel that runs *only* the unmasked pairs at zero overhead — physically unrealizable below the warp/tile granularity, but it is the **most optimistic** number and it is still ≤0.18 ms.

**Conclusion (#1):** the tree-causal mask is **not** a lever for this model on this hardware. We carry the FLOP-ideal saving as the headline curve (most-optimistic), and it moves the K=6 tree TPS by **+0.5%** and the verdict by nothing.

---

## 2 · The Marlin tile staircase (`results_tile_boundary.json`)

Re-running the **dense** profiler at unit-M resolution around the two coarse PR #28 steps confirms the `ceil(M/16)` tile theory **exactly**:

| transition | M before → after | step latency | **Δ (cliff)** | `thread_m_blocks` |
|---|---|--:|--:|:--:|
| tile 1→2 | 16 → **17** | 11.75 → 12.53 ms | **+0.772 ms** | 1→2 |
| (plateau) | 17 → 32 | 12.53 → 12.81 ms | +0.29 ms / 15M | 2 |
| tile 2→3 | 32 → **33** | 12.81 → 14.99 ms | **+2.176 ms** | 2→3 |
| (plateau) | 33 → 48 | 14.99 → 15.27 ms | +0.28 ms / 15M | 3 |
| tile 3→4 | 48 → **49** | 15.27 → 18.13 ms | **+2.869 ms** | 3→4 |
| (plateau) | 49 → 52 | 18.13 → 18.19 ms | +0.05 ms / 3M | 4 |

The cliffs grow (+0.77, +2.18, +2.87 ms) because each new 16-row tile adds a full Marlin GEMM pass over the **whole** weight matrix while the tile is mostly empty. **Within** a tile the step is ~flat (≤0.02 ms/token). Continuity is clean: the directly-measured M=16/32/48 match the PR #28 msweep within ±0.06 ms (run-to-run/thermal).

**The interpolation error this exposes:** PR #28's `LatencyCurve` linearly interpolates M=48→64, giving **M=49 ≈ 15.45 ms**. The direct measurement is **18.13 ms** — the interpolation **under-states the most-used deep-tree shape by 2.68 ms (17%)**. Same story at M=33: interp 13.10 ms vs measured 14.99 ms.

### Safe ("plateau") tree shapes

A tree should sit at the **top of a tile plateau**, never just past a cliff:

| tile | plateau M | best W=4 tree on it | verdict |
|--:|--:|---|---|
| tmb=2 | M=17…32 | **K=7 → M=29** | cheap, shallow |
| tmb=3 | M=33…48 | **K=11 → M=45** | **the sweet spot** |
| tmb=4 | M=49…64 | K=15 → M=61 | only if accept-rate is very high |

**K=11 (M=45) verifies for 15.24 ms; K=12 (M=49) costs 18.13 ms (+19%)** for one extra draft slot — a bad trade unless p is very high. **Prefer M that ends a tile (15, 31, 45), avoid M = 17, 33, 49.**

---

## 3 · Tree model on the corrected curve (`tree_acceptance_model.py --cost-model-json`)

Merged curve = PR #28 msweep **+ directly-measured tile-boundary points** (`merge_tree_mask_curve.py`) **+ tree-mask FLOP-ideal saving** at M=25/33/49 (`merged_treemask_flopideal.json`). K-range capped at **[1,15]** (M≤61 — inside the directly-measured range; PR #28's [1,20] extrapolated past unmeasured cliffs at M=65/81, an artifact).

**K=6 headline tree (tree-masked), w/ drafter / verify-only:**

| p | tps (drafter) | tps (verify-only) | vs PR #28 dense |
|--:|--:|--:|--:|
| 0.6792 | **333.0** | 370.0 | +1.8 |
| 0.78 | **377.1** | 419.0 | +2.0 |
| 0.85 | **411.9** | 457.7 | +2.2 |

**K\* (optimal tree depth) @ p=0.78, W=4:** **K\*=11 (M=45) = 440.4 TPS (drafter) / 480.8 (verify-only).** The optimum is the **top of the tmb=3 plateau**, not the cliff at M=49.

**Sensitivity across cost-model variants (K=12 = M=49, and K\*, @ p=0.78 drafter / verify-only):**

| cost model | K=12 (M=49) | K\* | why |
|---|--:|---|--:|
| **A. tile-corrected dense** (control) | 390.4 / 420.5 | K=11 (M=45): 440.4 / 480.8 | M=49 measured 18.13 ms |
| **B. tile-corrected + tree-mask** (headline) | **393.9 / 424.6** | K=11 (M=45): 440.4 / 480.8 | + 0.175 ms mask saving at M=49 only |
| **C. msweep + tree-mask, NO tile fix** (literal Step-3 reading) | 457.2 / **499.1** | K=12 (M=49): 457.2 / 499.1 | M=49 **interpolated** 15.28 ms (artifact) |

The headline **primary metric `K12_tree_tps_p078_tree_masked = 393.9`** is variant **B** — the honest number that uses the **directly-measured** M=49 = 18.13 ms. Variant **C** (the literal "msweep + tree mask" reading) gives 457.2/499.1 **only because** it inherits the 2.68 ms interpolation error Step 2 was built to catch; its verify-only 499.1 is the borderline-500 reading, and the direct M=49 measurement **kills it**. The tree mask itself moves K=12 by only **+3.5 TPS** (390.4 → 393.9).

### Final verdict

**`verdict_exceeds_500_at_p078_tree_masked = FALSE`** (test metric = 0). On the honest tile-corrected curve the tree ceiling @ p=0.78 is **440 TPS (drafter) / 481 (verify-only)** — well under 500, across all three saving lenses and both the control and tree-masked curves. The only reading that even approached 500 (variant C, 499.1) was a tile-interpolation artifact. The PR #28 conclusion (**>500 @ p=0.78 does not survive measurement**) stands, now with the M=49 cliff measured directly rather than interpolated.

> At the *optimistic* p=0.85, K=11 (M=45) does reach 511 / 558 — but p=0.85 is above the measured top-4 accept rate; at the realistic p=0.6792 / 0.78 the tree stays ≤440 / 481.

---

## 4 · What's actionable

1. **Drop the tree-causal-mask idea for this model/hardware.** Core attention is ~2.6% of the int4 step; the production SDPA path saves exactly 0. Not worth kernel work.
2. **Pick tree shapes at tile-plateau tops: M ∈ {29 (K=7), 45 (K=11)}. Avoid M = 17, 33, 49.** K=11 (M=45) is the verify-cost-optimal tree: **12% more TPS than K=12 (M=49) for free.**
3. **The cost model now has measured tile cliffs.** `tree_acceptance_model.py --cost-model-json merged_treemask_flopideal.json` no longer interpolates across Marlin boundaries; future drafter-ladder estimates inherit the correction.

## Public evidence used

- **Marlin W4A16 16-row tiling** (`thread_m_blocks = ceil(M/16)`): Marlin paper arXiv:2408.11743; IST-DASLab/marlin kernel; vLLM #7317. Predicts cliffs at M=17/33/49 — **confirmed exactly**.
- **Production tree-verify uses dense attention + topology mask** (so tree sparsity saves 0 wall-time): SpecInfer (arXiv:2305.09781, Eq 4), EAGLE, Medusa, vLLM tree-attn. Matches the SDPA Δ=0 measurement.
- **FlexAttention 128-block granularity / `BLOCK_SIZE<128` compiler error:** PyTorch `torch.nn.attention.flex_attention`, pytorch/pytorch#133562. Explains the negative Flex saving at M≤49.
- **PR #28** (`report_msweep.md`, `results_msweep.json`): the dense verify curve this PR corrects and extends.

## Repro

```bash
# Step 1 — tree-causal mask microbench (3 lenses)
uv run python scripts/profiler/spec_cost_model.py --tree-mask \
  --tree-M 25,33,49 --tree-W 4 --tree-ctx 256 \
  --dense-curve-json research/spec_cost_model/results_msweep.json \
  --tree-output research/spec_cost_model/results_tree_mask.json \
  --wandb_name denken/tree-causal-mask --wandb_group spec-verify-tree-mask

# Step 2 — Marlin tile-boundary fine sweep (dense, no mask)
uv run python scripts/profiler/spec_cost_model.py --modes graph --ctx-sweep 256 \
  --m-sweep 16,17,18,19,20,21,22,32,33,34,35,36,37,38,39,40,41,46,48,49,50,52 \
  --output research/spec_cost_model/results_tile_boundary.json \
  --wandb_name denken/marlin-tile-boundary --wandb_group spec-verify-tree-mask

# Step 3 — merge + re-run the tree acceptance model on the corrected curve
uv run python scripts/profiler/merge_tree_mask_curve.py \
  --msweep research/spec_cost_model/results_msweep.json \
  --tile-boundary research/spec_cost_model/results_tile_boundary.json \
  --tree-mask research/spec_cost_model/results_tree_mask.json \
  --saving flopideal --out research/spec_cost_model/merged_treemask_flopideal.json
uv run python scripts/profiler/tree_acceptance_model.py \
  --cost-model-json research/spec_cost_model/merged_treemask_flopideal.json \
  --cost-key 'graph|ctx256' --top1-acc 0.6792 --top4-acc 0.8605 \
  --drafter-ms 1.4 --verify-base-ms 7.0 --K-range 1 15 --widths 1 4 \
  --p-list 0.6792 0.78 0.85 --sim-K 6 \
  --output research/spec_cost_model/tree_results_tree_masked.json \
  --wandb_name denken/tree-model-tree-masked --wandb_group spec-verify-tree-mask
```

**W&B (project `wandb-applied-ai-team/gemma-challenge-senpai`):** mask `k56d6cxe` · tile `36hkaj14` · tree-model `aid45far`.
**Peak GPU:** A10G 24 GiB; vLLM profiler reserves `gpu_memory_utilization=0.9` ≈ 21.6 GiB (model load 9.81 GiB + KV 8.32 GiB + 0.18 GiB CUDA graphs); the Step-1 mask microbench is a lightweight tensor bench (<1 GiB).
