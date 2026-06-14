<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# PR #176 — Adverse domain-skew private stress: descent-only survival at 500

LOCAL single-A10G inference profiling / CPU analysis only. No HF Job, no
submission, no served-file change, no kernel deploy. Greedy/PPL untouched,
BASELINE unchanged (481.53). `--wandb_group descent-private-adverse-skew`.
Output under `research/validity/private_adverse_skew/`.

## The residual #164 flagged (the one this hardens)

My #164 (`5hz3dfrq`, MERGED) proved descent-only is private-safe across 3 native
proxies: native drop CI mid 2.04% [1.87, 2.21], descent-only central band
[508.5, 510.2] mid 509.3, worst τ-low 504.6 (+4.6 margin), `both_bugs_required_private=False`.
But the 3-proxy band [1.87, 2.21] is a **construction-variance** band over 3 axes,
NOT a sampling CI over the real private distribution. The binding residual is the
**adverse tail**: if the organizer's real private set is skewed toward the hardest
domains, could descent-only's τ-low corner be pushed < 500 at the same calibrated
aggregate-4.3%? #164 already proved the tree drop is **shape-sensitive** (same
aggregate-4.3%, two ladders → drops 0.34pp apart). An adverse domain-skew pulls
exactly that lever.

## What #176 produces

Convert the construction-variance band into an **adverse-skew stress certificate**:

1. **Widen to 5–6 calibrated axes.** Keep #164's three (code / casual / sharegpt,
   imported byte-identically from `descent_vs_bothbugs_private/proxies_native.json`)
   + 2–3 NEW genuinely-distinct hard tails from {multilingual / non-Latin script,
   math / reasoning chain, long-context tail}. Each NEW axis built by the SAME
   organizer-faithful construction as #164: a distinct hard component measured on
   the deployed `fa2sw_precache_kenyan` sglang `vllm-chat` scored stack, then
   count-pooled with the shared public reference at the continuous weight that lands
   the DECODE-frame linear drop on GT-4.3% (≤0.5pp calibration gate; pooling
   cumulatives = realizable mixture, NO `frac`, NO conditional interpolation).

2. **Adverse vertex over a diversity-capped domain simplex.** The eval set is a
   count-pool over {public, domain_1..N}. Overall count weights g (g≥0, Σg=1), each
   HARD domain capped `g_i ≤ cap` (cap = 0.5: no single domain is >half the eval
   set — a realistic diversity floor; a private set that is >50% one narrow domain
   like pure-code is an implausible organizer choice). At fixed aggregate-4.3%
   decode drop, **maximize the tree private-drop** over the capped polytope.
   Tree drop and decode drop are ~linear in the cumulative ladder C_mix = Σ g_d C_d,
   which is linear in g → optimum at a polytope vertex (LP / vertex enumeration +
   exact-evaluation refinement; the single axes are feasible points so the optimum
   is ≥ the worst admissible single axis by construction).

3. **Descent-only survival at the adverse corner.** Propagate the adverse-vertex
   ladder through `official = K_cal·(E[T]/step)·τ` (K_cal=125.268, step 1.2182) at
   central τ=1.0 AND the conservative tree-class τ-low=0.9924. Report
   `descent_only_taulow_tps_adverse_corner` (TEST) and `descent_only_clears_500_adverse`
   (both τ corners). Does descent-only survive the worst realistic private skew, or
   does the adverse corner flip #164's `both_bugs_required_private=False`?

4. **Tighten the shape-sensitivity envelope.** Widened `tree_private_drop_ci`
   (min/mid/max across 5–6 axes) + the adverse-vertex drop as the certified
   worst-case ceiling. Does #164's central-band conclusion (mid 509.3) hold and has
   the worst-case margin to 500 tightened or held?

5. **Self-validate (PRIMARY).** (a) reproduce #164 three axes within tol
   (code 2.21 / casual 1.98 / sharegpt 1.87 → descent 508.5 / 509.6 / 510.2);
   (b) conservative ordering `adverse_tree_drop ≥ max(admissible per-axis drops)`;
   (c) explicit descent-only adverse-corner clear-500 verdict (central AND τ-low);
   (d) NaN-clean full scan. Report `adverse_skew_stress_self_test_passes` (PRIMARY)
   + `descent_only_taulow_tps_adverse_corner` (TEST).

6. **Hand-off.** Hardens fern #167's launch-packet private-drop input from a central
   point (#164 CI mid 2.04%) to a stressed adverse worst-case. Honest scope: the one
   truly unmeasurable input remains an organizer re-run of the *tree* stack on the
   real private set; the adverse vertex is the worst REALISTIC skew, not a sampling
   CI over the true private distribution. Does NOT authorize a launch.

## Reused tools (imported, NOT re-derived)

- `scripts/profiler/tree_private_acceptance_gap.py` — banked #151/#156 descent-walk
  E[T] DP (`project_one`, `linear_et_from_q`, `official_tps_map`, τ band, K_cal/step).
- `scripts/validity/descent_vs_bothbugs_native.py` — #164 pooled-mode calibration +
  `propagate_native` + `relative_transfer` decode bridge + machinery xcheck.
- `scripts/validity/build_native_proxies.py` — #164 axis builder (extend with new axes).
- `scripts/validity/private_gap_probe.py` — sglang vllm-chat scored per-position ladder.
- #164 manifest `descent_vs_bothbugs_private/proxies_native.json` (import 3 axes).

## Constants (banked, confirmed from #164 results.json)

K_cal=125.26795, step=1.2182, bug1_mult=0.93175, gt_drop=4.2946%, τ∈[0.9924, 1.0],
clear-500 bar E[T]≥4.862. Public decode ladder linear E[T]=3.84445. #164 pool
weights code 0.4075 / casual 0.31425 / sharegpt 0.402 (all ≤ cap 0.5 → admissible).

## Plan / status

1. [liveness] CPU xcheck of the banked DP + W&B run in this group. **(this commit)**
2. [build] 2–3 new hard-component sets (CPU).
3. [GPU] measure each new component ladder (single-A10G sglang scored).
4. [analysis] 5–6 axis table + adverse-vertex LP + descent-only survival + self-tests.
5. [report] reconciliation JSON + PR SENPAI-RESULT.
