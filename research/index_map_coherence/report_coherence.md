<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Shared index-map coherence — does ONE corrected `target_logits_indices` fix
# BUG-1 spine AND BUG-2 descent? (PR #165 · wirbel)

**Gate: GREEN** — SHARED -- build ONE unified `target_logits_indices` correction (slot-0 own-row rank-1 + descent-ordered node layout) -- one contract, one validation, lower risk.

- **PRIMARY** `index_map_coherence_self_test_passes` = **True**
- `shared_index_map_fixes_both_bugs` = **True**
- `composed_fix_E_T` = **5.2070** (single corrected map, == both-fixed ceiling)
- `composed_fix_greedy_identity_safe` = **True**

## Honest scope
Pure-analytic CPU-only **build-coherence decision leg**, not a TPS lever. BASELINE stays 481.53; 0 TPS. Synthesis of wirbel #160 (spine spec) + wirbel #135 (BUG-2 salvage descent) + denken #158 (greedy-exact harness) + denken #133 (the shared-index-map hypothesis). Committed leg outputs are IMPORTED, not re-derived.

## 1. Index-path trace (the evidence, not a guess)
Kernel: `submissions/fa2sw_precache_kenyan/sitecustomize.py:921`

| path | indexing expression |
|---|---|
| BUG-1 spine root | `target_argmax_ptr + start_idx + pos   (pos == 0)` |
| BUG-2 descent walk | `target_argmax_ptr + start_idx + pos   (pos in range(num_draft_tokens))` |

BUG-1 (spine root) and BUG-2 (descent walk) both dereference the SAME pointer `target_argmax_ptr` through the SAME index base `start_idx + pos` (sitecustomize.py:945). The spine root is slot 0; the descent nodes are slots 1..N. Both are entries in the ONE upstream `target_logits_indices` gather that fills the single flat `target_argmax` array. The kernel holds no second index map and its `draft == target_argmax` greedy test is already correct -> both bugs live in WHAT fills that one array, i.e. one map.

=> `reads_same_logical_index_map` = **True** (same pointer `target_argmax_ptr`, same index base `start_idx + pos`, one upstream `target_logits_indices` gather, no second map).

## 2. Coherence model — SHARED, composed without double-counting
The single corrected map lands `composed_fix_E_T = ET_tree(q_true) = 5.2070` computed **once** — it simultaneously points slot-0 at the spine-root's own rank-1 row (BUG-1: f→0) and lays the descent nodes out so the linear kernel walk descends (BUG-2).

**No double-counting.** The composed value is NOT `descent-only + a separate spine delta`. A FALSE-independent additive model under-counts:

| term | E[T] |
|---|---|
| base (neither fixed, linear) | 2.6210 |
| + BUG-1 delta (LINEAR spine fix, no descent) | +0.1254 |
| + BUG-2 delta (descent) | +2.4354 |
| = false-independent additive | **5.1818** |
| true both-fixed (single map) | **5.2070** |
| super-additive interaction (missed by independence) | **+0.0252** |

The +interaction is the coupling a higher spine feeds into the descending branches — captured only by the joint single-map composition. The legitimate within-tree cross-check uses the TREE residual spine delta 0.1505: `descent-only 5.0564 + 0.1505 = 5.2070` (resid 0.0e+00).

## 3. Greedy-safety of the composed fix
Under SHARED the composed fix changes ONLY the upstream `target_logits_indices` gather; the kernel arithmetic (`_dixie_fused_accept_prep_kernel`) is UNCHANGED. denken #158 already certified that exact kernel **GREEDY_EXACT** (rate 1.0, sha match True; the harness catches the injected BUG-2 over-accept as VIOLATION; PPL 2.3767 ≤ 2.42). The kernel stores `target_argmax_id` at every committed position, so feeding the corrected (rank-1, descent-ordered) argmax stream keeps `committed[p]==argmax[p]` by construction ⇒ the GREEDY_EXACT verdict transfers. The harness `--audit-kernel-symbol` is armed for the instant land #71 assembles an actual new descent kernel.

=> `composed_fix_greedy_identity_safe` = **True**.

## 4. Bracketing-anchor self-test (PRIMARY)
| anchor | target | modelled |
|---|---|---|
| (a) neither-fixed (oracle) | 2.621 | 2.6210 |
| (b) BUG-2-only (descent) | 5.0564 | 5.0564 |
| (c) both-fixed | 5.207 | 5.2070 |
| (d) **single-map reproduces both** (DECISIVE) | 5.207 | **5.2070** |

=> `index_map_coherence_self_test_passes` = **True** (all 13/13 checks GREEN, NaN-clean).

## 5. Hand-off to land #71
**build ONE unified `target_logits_indices` correction (slot-0 own-row rank-1 + descent-ordered node layout) -- one contract, one validation, lower risk.**

Binding build-risk: BUG-2 (descent) carries the binding build-risk: it is the dominant E[T] lever (~19x BUG-1) and the structural change (linear break -> descending walk) where a build error (over-acceptance) is the only path that could break greedy identity. BUG-1's slot-0 re-point is a trivial single-index rider on the same map.

## Public / banked evidence used
- denken #133 (MERGED): rank-2 contamination root-cause + the shared-index-map hypothesis (`target_logits_indices`).
- wirbel #160 spine spec: the BUG-1 depth-1 fix contract + both-bugs E[T]=5.2070 / descent-only 5.0564 (imported).
- wirbel #135 BUG-2 salvage descent: the realized neither-fixed E[T]=2.621 + the fern hand-off columns (imported).
- denken #158 greedy-exact harness: the live linear kernel GREEDY_EXACT rate-1.0 certificate (imported) + the armed `--audit-kernel-symbol` instrument.

Official projection (context only; 0 TPS): composed @ measured step 535.4, @ roofline step 537.9 (τ=1).
