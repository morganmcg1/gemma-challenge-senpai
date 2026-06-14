<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #164 — Descent-only vs both-bugs: NATIVE private-drop decision

LOCAL single-A10G profiling/analysis only. No training, no HF Job, no submission,
no served-file change. Greedy/PPL untouched, BASELINE unchanged (481.53).
`--wandb_group descent-vs-bothbugs-private-decision`.

## The problem #164 fixes

#156 pinned the tree's private acceptance drop at **1.80% (descent-only) / 1.86%
(both-bugs)** at the organizer GT-4.3% LINEAR anchor → descent-only **510.6 TPS**,
both-bugs **525.5 TPS**. But that 1.80% came from **shape-transfer + interpolation**:
the ONE deliberately-hard chat proxy (sglang-scored ~10.7% linear drop) was scaled
toward public by a single fraction `frac=0.40` (`build_calibrated_ladder`) until its
*aggregate* linear-E[T] drop hit 4.3%, and the tree drop was read off that one
interpolated per-position shape. That bakes in the assumption that the ladder **shape**
scales linearly between the hard tail and public.

The launch-topology decision turns on this number:
- if the tree's REAL private drop ≈ the 4.3%-faithful 1.80%, **descent-only** (the
  simpler build, no spine) is private-safe → launch it;
- if it sits materially higher, the **both-bugs** spine (wirbel #160 + lawine #161)
  becomes a HARD launch dependency.

## Native fix (remove the single-shape interpolation)

Propagate the tree drop under **≥2 independent organizer-faithful proxies**, each
NATIVELY ~4.3% on the LINEAR stack (not one hard tail scaled by frac). Vary the
construction axis (prompt-length mix, domain mix, chat-template hardness) so they are
genuinely independent realizations. For each native proxy:
1. take its per-position acceptance ladder under the sglang `vllm-chat` **scored**
   protocol (the organizer-matching protocol pinned in #156) — measured directly on a
   real 128-prompt set, or **count-pooled** from real measured component pools;
2. `relative_transfer` it onto the banked decode-frame public reference (the accepted
   #156 harness-path bridge — re-bases the protocol, does NOT synthesize the 4.3% shape);
3. feed the native ladder **directly** into the banked descent-walk E[T] DP
   (`tree_private_acceptance_gap.project_one`) — NO `frac` — for descent-only AND both-bugs.
Report the **CI across proxies** on `tree_private_drop_pct` and projected TPS.

### Why pooling is native and interpolation is not

The deployed drafter's per-draft accept events pool linearly at the COUNT level:
`C_mix[k] = Σ_pool C_pool[k]·drafts_pool / Σ drafts` — the exact cumulative ladder the
drafter produces on the combined real prompt set. #156's `build_calibrated_ladder`
instead blends the CONDITIONALS linearly (`q = q_pub − frac·(q_pub−q_proxy)`), which is
**not** realizable as any single prompt distribution. Count-pooling real measured pools
removes exactly that assumption while hitting the 4.3% calibration anchor exactly
(continuous pool weight) and varying the SHAPE across genuinely different component pools.

## Plan

1. [liveness] CPU self-test of the banked descent-walk DP + machinery-faithfulness
   cross-check (4.3% spine → descent 510.6 / both 525.5, reproduces #156) + W&B run in
   this group. **(this commit — DONE: xcheck PASS, run `ls5v1b04`)**
2. [GPU] Construct ≥2 (target 3) native proxies and measure each per-position acceptance
   ladder under the **sglang scored** protocol (`private_gap_probe.py`, public_cold vs
   private_rerun) on the deployed linear `fa2sw_precache_kenyan` stack. Each must
   reproduce GT-4.3% LINEAR to ≤0.5pp by construction (the calibration gate).
3. [propagate] Feed each native ladder DIRECTLY through the DP (no frac).
   Report `tree_private_drop_pct` + projected TPS (descent-only + both-bugs) per proxy and
   the **CI** across proxies.
4. [decide] `descent_only_private_safe_native` (bool — CI keeps descent ≥500 with margin)
   and `both_bugs_required_private` (bool — descent dips sub-500, spine is a hard dep).
5. [self-validate] `native_proxies_reproduce_flagship_4p3` (PRIMARY, AND across proxies),
   `tree_private_drop_pct_native_ci` (TEST, CI-midpoint drop). NaN-clean.
6. Reconciliation JSON under this dir + PR report.

## Hand-off

Converts #156's *transferred* 1.80% into a **directly-measured** drop with a CI.
Informs (does NOT authorize) the launch go/no-go between descent-only and both-bugs.
