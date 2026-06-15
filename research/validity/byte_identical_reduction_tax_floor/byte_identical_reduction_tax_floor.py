#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Byte-identical reduction tax floor: decompose the 14.4-TPS cost (PR #423, denken).

THE QUESTION (human re-scope #407: maximize FASTEST STRICTLY-EQUIVALENT TPS). My #418 closed the
  precision-allocation lever and named the only remaining tax-reduction lever: "making the high-precision
  reduction itself cheaper -- a faster byte-identical verify path, shrinking the 14.4-TPS blanket-strict
  attention tax rather than skipping work." This card opens exactly that, ANALYTICALLY, no build.
  The realizable strict stack is blanket-strict @ 467.14 (full ~14.4-TPS attention tax; #412 measured) +
  cb3 supply +15.60 (#403) = 482.74 (knife-edge +1.21 over the NON-strict deployed 481.53, but WITH the
  byte-identity guarantee the deployed config lacks). The lever that WIDENS that +1.21 margin without
  touching the drafter is shrinking the 14.4 tax. We DECOMPOSE the tax into (a) fp32-accum / (b) single-
  segment serialization / (c) lost batch-parallelism, CLASSIFY each removable-vs-floor, and BOUND the
  theoretical floor of a byte-identical-to-M1 reduction -- telling us how much of #393's gap-to-floor is
  still on the table.

THE ANSWER (decision-critical, honest -- the lever is CLOSED under the strict contract):
  byte_identical_tax_floor_tps = 14.39 (the FULL tax). #393's fa2_unpack_ns1 pin is ALREADY AT the floor
  (current_pin_gap_to_floor = 0). removable_tax_tps = 0 under byte-identity to the EXISTING M=1 reference.
  The decomposition is single-component and the knife edge decides the floor:

  * WHERE the tax lives (a correction to the card's "M=8 verify body" framing): the 14.4 TPS is the
    un-pack penalty on the M=1 DRAFT-LANE attention, not the M=8 verify. The M=8 verify attention is
    num_splits-FREE (#393 verify_penalty_free=True). The M=1 single-row decode attention is OCCUPANCY-
    STARVED at num_splits=1: 8 CTAs / 80 SM = 10.0% (#400) vs the heuristic's data-dependent 9-way split
    (72 CTAs / 88%). The whole tax is that occupancy collapse.

  * (a) fp32-accumulate overhead = 0.0 TPS. FlashAttention accumulates the online-softmax P*V in fp32 on
    BOTH the deployed heuristic AND the byte-exact un-pack path (intrinsic on sm_8x). The byte-break
    (#393 maxdiff 0.0039 at M=8) is reduction-ORDER reassociation from the data-dependent split COUNT,
    NOT an accumulator-precision downgrade. The byte-identical path adds zero fp32 cost over deployed.
    (Distinct from the lm_head logits reduction -- out of scope for this attention tax.)

  * (b) single-segment serialization (lost split-K) = 14.39 TPS -- the WHOLE tax. num_splits=1 forbids
    the data-dependent split-K that filled the SMs; the M=1 GEMV drops to 10% occupancy (#400/#408
    pinnedk_attn_removable_us = 36.5us -> +13.998 TPS realistic).

  * (c) lost batch-parallelism (flagged 24.6% serialized) = 0.0 TPS. The realizable strict path is
    BLANKET (one kernel, no flagging). The SELECTIVE alternative that peels the 24.6% flagged steps onto
    a separate high-precision path is net-NEGATIVE: #412 measured 384.11 < blanket 467.14 (-83 TPS) AND
    degrades identity (0.9853 < 0.9989). So (c) is not a saving -- attempting it is a -83 TPS mistake.

  * THE FLOOR (the decider, my #418 knife edge applied to split-K reassociation): is (b) removable?
    A parallelism-recovering reduction (a fixed deterministic split-K, #400's pinned 64-CTA) reassociates
    the sum. ANY reassociation carries max|dlogit| = REDUCTION_ORDER_PERTURB_MAX = 0.125 = 1 bf16-ULP
    (#87/#381), which EQUALS the near-tie margin eps* = 0.125 (#405): a KNIFE EDGE with ZERO proof margin.
    The 40 near-ties at gap<=eps* BLANKET all readable positions (#405; my #418). So a split-K flips
    near-ties vs the frozen M=1 reference -> NOT byte-identical. #400's own new_reference_probe MEASURED
    this independently: multisplit_eq_serial_bytes = False. => single-segment serialization is FLOOR, not
    removable. The cheapest byte-identical-to-M1 reduction IS the serial un-pack (#393, the ONLY byte-
    exact config, n=1). byte_identical_tax_floor_tps = 14.39; the pin is at the floor.

  * THE ONLY SUB-FLOOR PATH is a REFERENCE RE-CAPTURE, not a byte-identical reduction. #400's pinned-K is
    M-invariant byte-exact-feasible (M=1==M=8 under the NEW kernel; Marlin-grounded) and recovers ~14
    TPS, but it produces a NEW reference whose bytes differ from today's served output at the same eps*
    near-tie population (~O(3/882) flips, PPL-neutral). It is greedy/PPL-VALID (a legal strictly-
    equivalent submission in the #407 M-invariance sense) but it is NOT frozen-byte-identical, requires a
    FA2 kernel rebuild (num_splits>1 is NotImplementedError on shipped FA2) + a full greedy/PPL re-
    validation, and is a flagged served-file change. That +14 (stack -> ~496.7; lawine #411 ceiling
    497.44) is a CONTRACT decision (frozen-byte vs M-invariant) for humans, NOT a byte-identical win.

  Net: the byte-identity tax floor is a CONTRACT floor, not a physical floor. Under frozen-byte-identity
  (the strict line's contract) the 14.4 is irreducible and this lever is DEAD -- the knife-edge margin
  stays +1.21 (482.74). The +14 exists only behind a reference re-capture. This CLOSES the last tax-
  reduction lever I named in #418, on the byte-identical-preserving side.

WHAT THIS IS / IS NOT:
  Pure-CPU analytic card (stdlib math). 0 GPU, 0 official TPS, 0 HF Job, NO served-file change, NO
  submission, NO kernel build, analysis_only=True. Imports my merged #413 (MU_P/BASE_467/ladder/PPL) and
  my merged #418 (eps*, perturbation ceiling, per-position near-tie census -- the knife edge) byte-
  exactly. Pins the #393 attention-pin, #400 pinned-K-headroom, #408 M=1 budget, #412 selective-recompute,
  and #403 cb3-supply scalars and CROSS-CHECKS them against the merged result JSONs at runtime (provenance
  self-test). NOTHING is re-derived; the new modelling is the 3-way tax attribution and the byte-identity
  floor classification (the knife-edge gate, reused from #418, applied to split-K reassociation).

REPRODUCE (0-GPU):
    cd target/ && .venv/bin/python -m research.validity.byte_identical_reduction_tax_floor.\
byte_identical_reduction_tax_floor --self-test
    cd target/ && .venv/bin/python -m research.validity.byte_identical_reduction_tax_floor.\
byte_identical_reduction_tax_floor \
      --wandb_group cb3-tax-floor --wandb_name denken/byte-identical-reduction-tax-floor
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

# ---- import my merged #413 (speed/ladder anchors) and #418 (the knife edge) byte-exactly --------------
from research.validity.equivalent_tps_optimal_geometry import equivalent_tps_optimal_geometry as g413
from research.validity.position_asymmetric_verify_tax import position_asymmetric_verify_tax as g418

HERE = Path(__file__).resolve().parent
VAL = HERE.parent  # research/validity

# ===========================================================================
# Section 0 -- banked anchors re-exported byte-exactly from #413 and #418
# ===========================================================================
MU_P: float = g413.MU_P                  # 481.53 deployed FAST (non-equivalent) frontier (#52, 2x9fm2zx)
BASE_467_MODELED_393: float = g413.BASE_467   # 467.475218449957 -- #393 modeled decode-eta blanket strict
PPL_DEPLOYED: float = g413.PPL_DEPLOYED  # 2.3772
PPL_GATE: float = g413.PPL_GATE          # 2.42
M_DEPLOYED: int = g413.M_DEPLOYED        # 8 verify rows = K_spec(7)+1 (linear chain)
K_DEPLOYED: int = 7                       # deployed draft length (num_speculative_tokens=7, manifest)
TARGET: float = 500.0

# ---- the KNIFE EDGE, re-exported from my #418 (which banks #405/#381/#87), pinned not re-derived -------
EPS_STAR: float = g418.EPS_STAR                                # 0.125 near-tie margin (16 bf16-ULP) (#405)
REDUCTION_ORDER_PERTURB_MAX: float = g418.REDUCTION_ORDER_PERTURB_MAX  # 0.125 reassociation |dlogit| (#87/#381)
THINNEST_GAP_GLOBAL: float = g418.THINNEST_GAP_GLOBAL         # 0.03125 (#87) sub-perturbation near-ties exist
NEARTIE_BY_J: dict[int, int] = dict(g418.NEARTIE_BY_J)        # {1:6,2:7,3:7,4:1,5:7,6:5,7:7} -- blankets all
N_NEARTIE_AT_EPS: int = g418.N_NEARTIE_AT_EPS                 # 40 near-ties at gap<=eps* (#405)
N_SERVED_FLIPS: int = g418.N_SERVED_FLIPS                     # 3/882 served flips (#381/#405)
N_SERVED_POSITIONS: int = g418.N_SERVED_POSITIONS            # 882 readable chain positions (#405)
N_CHAIN: int = K_DEPLOYED

# ===========================================================================
# Section 1 -- the measured blanket-strict tax (#412 measured base; #393 modeled base) ------------------
# ===========================================================================
# #412 (selective_recompute_equivalent_tps) MEASURED the realizable blanket-strict TPS on the local A10G.
BASE_467_MEASURED_412: float = 467.1400155438763   # blanket_strict_measured_tps (#412), +/- 0.16105 std
BASE_467_MEASURED_STD_412: float = 0.16105003370123783
TAX_MEASURED: float = MU_P - BASE_467_MEASURED_412     # 481.53 - 467.14 = 14.39 (the PRIMARY tax to decomp)
TAX_MODELED_393: float = MU_P - BASE_467_MODELED_393   # 481.53 - 467.475 = 14.055 (the #393 modeled tax)

# ===========================================================================
# Section 2 -- the attention-pin structure (#393) and M=1 occupancy (#400/#408) -------------------------
# ===========================================================================
# #393 attention_strict_pin_cost (run 0q7ynumg): the cheapest byte-EXACT attention reduction is the
# un-packed single-split (num_splits=1, FA_SLIDING=0 deterministic). It is the ONLY byte-exact config.
CHEAPEST_STRICT_ATTN_BACKEND: str = "fa2_unpack_ns1"
N_BYTE_EXACT_ATTN_CONFIGS_393: int = 1
ATTN_ETA_REDUCIBLE_393: bool = False             # not reducible among REACHABLE configs (no rebuild)
FA_SLIDING0_IS_STRICT_FLOOR_393: bool = True     # un-pack ns=1 IS the no-rebuild strict floor
VERIFY_PENALTY_FREE_393: bool = True             # M=8 verify attention: un-pack ~ heuristic (penalty~1.0)

# #400 attn_pinnedk_headroom (run o7yhpkej) + #408 m1_decode_latency_budget (run qc9bz8sv): the tax lives
# on the M=1 draft-lane attention; the un-pack penalty is an OCCUPANCY collapse, and a deterministic
# 64-CTA (num_splits=8) split-reduce would recover it -- BUT produces a NEW reference.
M1_ATTN_OCCUPANCY_FRAC_400: float = 0.1          # un-pack M=1: 8 CTAs / 80 SM
M1_HEURISTIC_OCCUPANCY_FRAC_400: float = 0.88    # heuristic M=1: 70.4 CTAs / 80 SM (data-dependent split)
M1_ATTN_BW_FRAC_400: float = 0.020557947294027247   # 2.06% of 600 GB/s -> far below BW floor (occupancy-bound)
ATTN_LEVER_GAIN_REALISTIC_400: float = 14.100746977089045   # #400 attn_lever_realistic_tps_gain_deployed
ATTN_LEVER_GAIN_REALISTIC_408: float = 13.998600706082982   # #408 attn_lever_gain_realistic_tps
PINNEDK_RECOVERY_FRAC_ROOFLINE_400: float = 1.0
PINNEDK_RECOVERY_FRAC_REALISTIC_400: float = 0.9871794871794871
PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400: bool = True     # M=1==M=8 under the NEW kernel (Marlin-grounded)
PINNEDK_PRODUCES_NEW_REFERENCE_400: bool = True              # bytes differ from the EXISTING M=1 reference
MULTISPLIT_EQ_SERIAL_BYTES_400: bool = False                 # new_reference_probe: split != serial, MEASURED
ATTN_REBUILD_IS_FLAGGED_SERVED_CHANGE_400: bool = True       # FA2 num_splits>1 = NotImplementedError -> rebuild
T_ATTN_FRAC_408: float = 0.09506718019009251                 # attention share of the M=1 step (#408 budget)

# ===========================================================================
# Section 3 -- selective-recompute closure (#412) and cb3 supply (#403) for the stack -------------------
# ===========================================================================
SELECTIVE_TPS_412: float = 384.11448835129306        # selective recompute MEASURED -- WORSE than blanket
SELECTIVE_BEATS_BLANKET_412: bool = False
FLAGGED_STEP_FRACTION_412: float = 0.24603174603174602   # ~24.6% of steps flagged (the card's "~23.6%")
SELECTIVE_IDENTITY_412: float = 0.9852607709750567       # selective DEGRADES identity ...
BLANKET_IDENTITY_412: float = 0.9988662131519275         # ... vs blanket 0.9989 (1 residual tie @ prompt 90)
FASTEST_REALIZABLE_STRICT_CONFIG_412: str = "blanket_strict"

CB3_SUPPLY_403: float = 15.60            # cb3 conservative deployable supply lift, k*=229 (#403 iv9i2wks)
CB3_KSTAR_403: int = 229

# the realizable FROZEN-byte-identical strict stack and its knife-edge margin over the deployed frontier.
STACK_FROZEN: float = BASE_467_MEASURED_412 + CB3_SUPPLY_403          # 482.74 (blanket-strict + cb3)
KNIFE_EDGE_MARGIN_FROZEN: float = STACK_FROZEN - MU_P                 # +1.21 over deployed 481.53
# the RE-CAPTURE stack (pinned-K rebuild + cb3) -- NEW reference, NOT frozen-byte-identical.
STACK_RECAPTURE: float = BASE_467_MEASURED_412 + ATTN_LEVER_GAIN_REALISTIC_408 + CB3_SUPPLY_403  # ~496.7
STACK_RECAPTURE_CEILING_411: float = 497.44          # lawine #411 max_stack_tps_under_current_floor

TOL: float = 1e-6

# source artifacts for the provenance cross-check (pinned constants must match these merged JSONs).
ART_393 = VAL / "attention_strict_pin_cost" / "attention_strict_pin_cost_results.json"
ART_400 = VAL / "attention_strict_pin_cost" / "attn_pinnedk_headroom_results.json"
ART_408 = VAL / "attention_strict_pin_cost" / "m1_decode_latency_budget_results.json"
ART_412 = VAL / "selective_recompute_equivalent_tps" / "selective_recompute_equivalent_tps_results.json"
ART_403 = VAL / "cb3_conservative_k_deployable_lift" / "cb3_conservative_k_deployable_lift_results.json"

SRC_393_RUN = "0q7ynumg"   # wirbel attention_strict_pin_cost
SRC_400_RUN = "o7yhpkej"   # wirbel attn_pinnedk_headroom
SRC_408_RUN = "qc9bz8sv"   # wirbel m1_decode_latency_budget
SRC_412_RUN = "stark/selective-recompute"  # selective_recompute_equivalent_tps (#412)
SRC_403_RUN = "iv9i2wks"   # kanna cb3_conservative_k_deployable_lift
SRC_418_RUN = "uc7jg6vs"   # denken position_asymmetric_verify_tax (the knife edge)


# ===========================================================================
# Section 4 -- the 3-way tax decomposition (instruction 2) ----------------------------------------------
# ===========================================================================

def decompose_tax(tax: float) -> dict:
    """Attribute the blanket-strict attention tax to (a) fp32-accum / (b) serialization / (c) lost-batch.
    Grounded in #393 (verify_penalty_free), #400/#408 (M=1 occupancy collapse), #412 (selective net-neg)."""
    # (a) fp32-accumulate overhead: FA accumulates P*V in fp32 on BOTH heuristic and un-pack (intrinsic on
    #     sm_8x). The byte-break is reduction-ORDER (data-dependent split count), not accumulator precision
    #     -> the byte-identical path adds ZERO incremental fp32 cost on attention.
    a_fp32_accum = 0.0
    # (b) single-segment serialization (lost split-K): the WHOLE tax. num_splits=1 -> M=1 GEMV at 10%
    #     occupancy; the heuristic's data-dependent 9-way split filled 88%. The tax IS that occupancy gap.
    b_serialization = tax
    # (c) lost batch-parallelism (flagged 24.6% serialized): 0 for the BLANKET path (no flagging). The
    #     SELECTIVE peel is net-NEGATIVE (#412: 384.11 < 467.14), so (c) is not a recoverable saving.
    c_lost_batch_parallel = 0.0
    # 4th component check (launch/sync of a flagged segment): un-pack is a SINGLE kernel (fewer launches
    #     than split-K, which needs a combine kernel) -> no separable launch/sync component falls out.
    d_launch_sync = 0.0
    residual = tax - (a_fp32_accum + b_serialization + c_lost_batch_parallel + d_launch_sync)
    return {
        "tax_total": tax,
        "tax_decomp_fp32_accum_tps": a_fp32_accum,
        "tax_decomp_serialization_tps": b_serialization,
        "tax_decomp_lost_batch_parallel_tps": c_lost_batch_parallel,
        "tax_decomp_launch_sync_tps": d_launch_sync,
        "decomp_residual_tps": residual,
        "dominant_component": "single_segment_serialization",
        "where_tax_lives": "M=1 draft-lane attention (occupancy-starved); M=8 verify is num_splits-free",
        "a_zero_reason": "FA fp32-accumulates on both deployed and byte-exact; byte-break is ORDER not precision",
        "c_zero_reason": "blanket has no flagging; selective peel net-NEGATIVE -83 TPS (#412)",
    }


# ===========================================================================
# Section 5 -- the byte-identity FLOOR gate (instruction 3/4; my #418 knife edge on split-K) ------------
# ===========================================================================
# Is component (b) removable? A parallelism-recovering reduction (deterministic split-K, #400's pinned
# 64-CTA) reassociates -> carries max|dlogit| = REDUCTION_ORDER_PERTURB_MAX = 0.125 = eps*. A reduction is
# byte-identical to the FROZEN M=1 reference only if it provably flips NO near-tie: perturb < eps* (proof
# margin) AND no near-tie at any affected position. Both fail (same wall as #418): perturb == eps* (knife
# edge, zero margin) and the 40 near-ties blanket all positions. #400's new_reference_probe MEASURED the
# same: multisplit_eq_serial_bytes = False. => serialization is FLOOR, not removable.

def has_proof_margin() -> bool:
    """perturb_max < eps* would prove a reassociation cannot cross a near-tie. Here 0.125 == 0.125 -> no."""
    return REDUCTION_ORDER_PERTURB_MAX < EPS_STAR


def neartie_blankets_all_positions() -> bool:
    return all(NEARTIE_BY_J.get(j, 0) > 0 for j in range(1, N_CHAIN + 1))


def split_k_is_byte_identical_to_frozen_ref() -> bool:
    """A split-K (parallel) reduction is byte-identical to the frozen M=1 reference iff it provably flips
    no near-tie. Requires proof margin AND a near-tie-free target -- BOTH fail; #400 measured it False."""
    safe = has_proof_margin() and (not neartie_blankets_all_positions())
    # corroborated by #400's direct measurement that a multi-split reduction != the serial reference bytes.
    return bool(safe and MULTISPLIT_EQ_SERIAL_BYTES_400)


def classify_floor(tax: float, decomp: dict) -> dict:
    """Classify removable-vs-floor under TWO reference contracts and bound the byte-identical tax floor."""
    serialization_removable_frozen = split_k_is_byte_identical_to_frozen_ref()   # False (knife edge)

    # FROZEN-byte-identity contract (the strict line's contract; the card's "last-bit-identical to the
    # M=1 AR reference" criterion): nothing is removable; the un-pack pin IS the floor.
    removable_frozen = (decomp["tax_decomp_serialization_tps"] if serialization_removable_frozen else 0.0)
    floor_frozen = tax - removable_frozen
    current_pin_gap_to_floor = TAX_MEASURED - floor_frozen   # #393 un-pack pin sits at floor -> 0

    # M-INVARIANT re-capture contract (#407 "strictly-equivalent" = M=1==M=8 under the kernel + PPL/greedy
    # valid, on a NEW reference): the pinned-K rebuild recovers ~14 TPS; the residual floor is the small
    # realistic non-recovery. This is NOT frozen-byte-identical (it flips ~O(3/882) near-ties vs today).
    recap_recovered = TAX_MEASURED * PINNEDK_RECOVERY_FRAC_REALISTIC_400
    floor_recapture = TAX_MEASURED * (1.0 - PINNEDK_RECOVERY_FRAC_REALISTIC_400)

    return {
        # the knife-edge verdict
        "perturb_max": REDUCTION_ORDER_PERTURB_MAX,
        "eps_star": EPS_STAR,
        "perturb_ge_eps_knife_edge": REDUCTION_ORDER_PERTURB_MAX >= EPS_STAR,
        "has_proof_margin": has_proof_margin(),
        "neartie_blankets_all_positions": neartie_blankets_all_positions(),
        "thinnest_gap_below_perturb": THINNEST_GAP_GLOBAL < REDUCTION_ORDER_PERTURB_MAX,
        "multisplit_eq_serial_bytes_400": MULTISPLIT_EQ_SERIAL_BYTES_400,
        "split_k_byte_identical_to_frozen_ref": serialization_removable_frozen,
        # FROZEN contract (PRIMARY)
        "removable_tax_tps": removable_frozen,                       # 0.0
        "floor_tax_tps": floor_frozen,                              # 14.39
        "byte_identical_tax_floor_tps": floor_frozen,              # PRIMARY == 14.39
        "current_pin_gap_to_floor_tps": current_pin_gap_to_floor,  # 0.0 (un-pack pin AT floor)
        "pin_is_at_floor": abs(current_pin_gap_to_floor) < 1e-6,
        # M-INVARIANT re-capture contract (secondary, contract-gated)
        "recapture_recovered_tps": recap_recovered,                # ~13.99
        "recapture_residual_floor_tps": floor_recapture,           # ~0.18
        "recapture_is_new_reference": PINNEDK_PRODUCES_NEW_REFERENCE_400,
        "recapture_is_byte_identical_to_frozen": False,
        "recapture_is_flagged_served_change": ATTN_REBUILD_IS_FLAGGED_SERVED_CHANGE_400,
    }


# ===========================================================================
# Section 6 -- the scoped served-kernel change (instruction 5; DO NOT build) ----------------------------
# ===========================================================================

def scoped_served_change() -> str:
    return (
        "NONE preserves byte-identity to the current served reference; this lever is CLOSED on the "
        "byte-identical side. The ONLY sub-floor path is wirbel #400's deterministic 64-CTA "
        "(num_splits=8) pinned-K split-reduce on the FA2 decode-attention kernel: it is M-invariant "
        "byte-exact-feasible (M=1==M=8 under the new kernel; Marlin atomic_add=False / fp32_reduce / "
        "fixed-order grounded) and recovers ~13.998 TPS realistic (occupancy 10%->~88%). BUT it produces "
        "a NEW reference: #400's new_reference_probe MEASURED multisplit_eq_serial_bytes=False, and my "
        "#418 knife edge (reassociation perturbation 0.125 = eps* 0.125, zero proof margin, 40 near-ties "
        "blanketing all 7 positions) proves it flips ~O(3/882) near-ties vs today's served bytes "
        "(PPL-neutral). Kernel surface: FA2 varlen-paged decode attention (num_splits>1 is "
        "NotImplementedError on shipped FA2 -> a kernel REBUILD). Identity-verify cost: a full greedy + "
        "PPL re-capture / re-validation on the new reference (PPL must re-clear <=2.42). Blast radius: a "
        "flagged served-file change touching the hottest decode kernel; realistic vs floor recovery 98.7% "
        "vs 100% roofline. VERDICT: greedy/PPL-VALID (a legal strictly-equivalent submission in the #407 "
        "M-invariance sense, stack -> ~496.7; lawine #411 ceiling 497.44) but NOT frozen-byte-identical "
        "-- this is a CONTRACT decision (frozen-byte vs M-invariant reference) for HUMAN approval, not a "
        "byte-identical tax reduction. Recommend: surface to the team as a re-capture proposal; do NOT "
        "auto-build."
    )


# ===========================================================================
# Section 7 -- self-tests (>= 20 checks; PRIMARY gate) --------------------------------------------------
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def _load(art: Path) -> dict | None:
    if not art.exists():
        return None
    try:
        return json.loads(art.read_text())
    except Exception:  # noqa: BLE001
        return None


def run_self_tests(decomp: dict, floor: dict) -> dict:
    c: dict[str, bool] = {}

    # a) provenance: anchors imported byte-exactly from merged #413 / #418.
    c["a_mu_p_is_481p53"] = abs(MU_P - 481.53) < TOL
    c["a_base_modeled_393"] = abs(BASE_467_MODELED_393 - 467.475218449957) < TOL
    c["a_base_measured_412"] = abs(BASE_467_MEASURED_412 - 467.1400155438763) < TOL
    c["a_eps_star_is_0p125"] = abs(EPS_STAR - 0.125) < 1e-12
    c["a_perturb_is_0p125"] = abs(REDUCTION_ORDER_PERTURB_MAX - 0.125) < 1e-12
    c["a_neartie_sums_40"] = sum(NEARTIE_BY_J.values()) == N_NEARTIE_AT_EPS == 40
    c["a_m_deployed_8"] = M_DEPLOYED == 8 and K_DEPLOYED == 7
    c["a_eps_eq_perturb_from_418"] = abs(EPS_STAR - REDUCTION_ORDER_PERTURB_MAX) < 1e-12

    # b) the tax: measured 14.39, modeled band ~14.05.
    c["b_tax_measured_is_14p39"] = abs(TAX_MEASURED - 14.389984456123698) < 1e-6
    c["b_tax_modeled_is_14p05"] = abs(TAX_MODELED_393 - 14.054781550043) < 1e-6
    c["b_tax_measured_gt_modeled"] = TAX_MEASURED > TAX_MODELED_393   # measured base is below modeled base
    c["b_tax_about_3pct"] = 0.028 < TAX_MEASURED / MU_P < 0.031

    # c) the decomposition: single-component (b); (a)=(c)=0; closure exact.
    c["c_fp32_accum_zero"] = decomp["tax_decomp_fp32_accum_tps"] == 0.0
    c["c_serialization_is_whole_tax"] = abs(decomp["tax_decomp_serialization_tps"] - TAX_MEASURED) < 1e-9
    c["c_lost_batch_parallel_zero"] = decomp["tax_decomp_lost_batch_parallel_tps"] == 0.0
    c["c_launch_sync_zero"] = decomp["tax_decomp_launch_sync_tps"] == 0.0
    c["c_decomp_closes_exact"] = abs(decomp["decomp_residual_tps"]) < 1e-9
    c["c_dominant_is_serialization"] = decomp["dominant_component"] == "single_segment_serialization"

    # d) #393/#400/#408 structure: M=8 verify free, M=1 occupancy-starved, only 1 byte-exact config.
    c["d_verify_penalty_free"] = VERIFY_PENALTY_FREE_393 is True
    c["d_one_byte_exact_config"] = N_BYTE_EXACT_ATTN_CONFIGS_393 == 1
    c["d_m1_occupancy_10pct"] = abs(M1_ATTN_OCCUPANCY_FRAC_400 - 0.1) < 1e-9
    c["d_m1_far_below_bw_floor"] = M1_ATTN_BW_FRAC_400 < 0.05
    c["d_unpack_is_strict_floor_norebuild"] = FA_SLIDING0_IS_STRICT_FLOOR_393 is True

    # e) THE KNIFE EDGE -> serialization is FLOOR under the frozen reference.
    c["e_knife_edge_perturb_ge_eps"] = floor["perturb_ge_eps_knife_edge"] is True
    c["e_no_proof_margin"] = floor["has_proof_margin"] is False
    c["e_neartie_blankets_all"] = floor["neartie_blankets_all_positions"] is True
    c["e_multisplit_not_serial_400"] = floor["multisplit_eq_serial_bytes_400"] is False
    c["e_split_k_not_byte_identical"] = floor["split_k_byte_identical_to_frozen_ref"] is False

    # f) the FLOOR verdict (PRIMARY): removable=0, floor=14.39, pin at floor.
    c["f_removable_is_zero"] = floor["removable_tax_tps"] == 0.0
    c["f_floor_is_full_tax"] = abs(floor["floor_tax_tps"] - TAX_MEASURED) < 1e-9
    c["f_byte_identical_floor_eq_tax"] = abs(floor["byte_identical_tax_floor_tps"] - TAX_MEASURED) < 1e-9
    c["f_pin_gap_to_floor_zero"] = abs(floor["current_pin_gap_to_floor_tps"]) < 1e-6
    c["f_pin_is_at_floor"] = floor["pin_is_at_floor"] is True

    # g) the re-capture path is NEW-reference / flagged, NOT frozen-byte-identical.
    c["g_recapture_new_reference"] = floor["recapture_is_new_reference"] is True
    c["g_recapture_not_frozen_identical"] = floor["recapture_is_byte_identical_to_frozen"] is False
    c["g_recapture_flagged_change"] = floor["recapture_is_flagged_served_change"] is True
    c["g_recapture_recovers_most"] = floor["recapture_recovered_tps"] > 13.0
    c["g_recapture_residual_small"] = 0.0 < floor["recapture_residual_floor_tps"] < 0.5

    # h) component (c): selective is net-NEGATIVE -> blanket is the realizable strict path.
    c["h_selective_worse_than_blanket"] = SELECTIVE_TPS_412 < BASE_467_MEASURED_412
    c["h_selective_not_beats_blanket"] = SELECTIVE_BEATS_BLANKET_412 is False
    c["h_fastest_realizable_is_blanket"] = FASTEST_REALIZABLE_STRICT_CONFIG_412 == "blanket_strict"

    # i) the stack + knife-edge margin (instruction: ~482.74, +1.21 over deployed).
    c["i_stack_frozen_is_482p74"] = abs(STACK_FROZEN - 482.74001554387627) < 1e-6
    c["i_margin_frozen_is_1p21"] = abs(KNIFE_EDGE_MARGIN_FROZEN - 1.2100155438762) < 1e-6
    c["i_margin_positive_with_identity"] = KNIFE_EDGE_MARGIN_FROZEN > 0.0
    c["i_recapture_stack_higher"] = STACK_RECAPTURE > STACK_FROZEN

    # j) PPL/greedy preserved; numeric hygiene.
    c["j_ppl_within_gate"] = PPL_DEPLOYED <= PPL_GATE
    flat = [TAX_MEASURED, floor["byte_identical_tax_floor_tps"], floor["removable_tax_tps"],
            floor["recapture_recovered_tps"], STACK_FROZEN, KNIFE_EDGE_MARGIN_FROZEN]
    c["j_no_nan_inf"] = all(_finite(v) for v in flat)

    # k) artifact provenance cross-check (pinned constants == merged JSONs, when present).
    d393, d400, d408, d412, d403 = (_load(a) for a in (ART_393, ART_400, ART_408, ART_412, ART_403))
    if d393 is not None:
        c["k_393_base_matches"] = abs(d393.get("deployed_tps_decode_eta", 0) - BASE_467_MODELED_393) < 1e-6
        c["k_393_one_byte_exact"] = d393.get("n_byte_exact_attn_configs") == N_BYTE_EXACT_ATTN_CONFIGS_393
        c["k_393_cheapest_backend"] = d393.get("cheapest_strict_attn_backend") == CHEAPEST_STRICT_ATTN_BACKEND
    if d400 is not None:
        rp = d400.get("compose", {}).get("new_reference_probe", {})
        c["k_400_multisplit_changes_bytes"] = rp.get("multisplit_changes_bytes_vs_serial") is True
        c["k_400_pinnedk_new_reference"] = d400.get("pinnedk_produces_new_reference") is True
        c["k_400_m1_occupancy"] = abs(d400.get("m1_attn_occupancy_frac", 0) - M1_ATTN_OCCUPANCY_FRAC_400) < 1e-9
    if d408 is not None:
        c["k_408_attn_frac"] = abs(d408.get("t_attn_frac", 0) - T_ATTN_FRAC_408) < 1e-9
        c["k_408_attn_gain_realistic"] = abs(
            d408.get("compose", {}).get("attn_lever_gain_realistic_tps", 0) - ATTN_LEVER_GAIN_REALISTIC_408) < 1e-6
    if d412 is not None:
        c["k_412_blanket_measured"] = abs(d412.get("blanket_strict_measured_tps", 0) - BASE_467_MEASURED_412) < 1e-9
        c["k_412_selective_worse"] = d412.get("selective_beats_blanket") is False
        c["k_412_flagged_frac"] = abs(d412.get("flagged_step_fraction", 0) - FLAGGED_STEP_FRACTION_412) < 1e-9
    if d403 is not None:
        # #403 nests deliverables under result.recost_at_kstar (vs top-level on the others).
        recost_403 = d403.get("result", {}).get("recost_at_kstar", {})
        c["k_403_cb3_lift"] = abs(recost_403.get("m8_lift_at_kstar", 0) - 15.604) < 0.02

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# ===========================================================================
# Section 8 -- report assembly + W&B + CLI --------------------------------------------------------------
# ===========================================================================

def build_report() -> dict:
    decomp = decompose_tax(TAX_MEASURED)
    floor = classify_floor(TAX_MEASURED, decomp)
    served_change = scoped_served_change()
    selftest = run_self_tests(decomp, floor)

    headline = (
        "Byte-identical reduction tax floor. The 14.39-TPS blanket-strict ATTENTION tax (481.53->467.14, "
        "#412 measured) is single-component: (a) fp32-accum=0 (FA already fp32-accumulates; byte-break is "
        "reduction ORDER not precision), (b) single-segment serialization=14.39 (M=1 draft-lane occupancy "
        "collapse 10%->88%, NOT the M=8 verify which is num_splits-free #393), (c) lost-batch-parallel=0 "
        "(blanket has no flagging; selective peel net-NEGATIVE -83 TPS #412). FLOOR: (b) is IRREDUCIBLE "
        "under byte-identity to the frozen M=1 reference -- my #418 knife edge (reassociation perturb "
        "0.125 = eps* 0.125, zero margin, 40 near-ties blanket all positions; #400 measured "
        "multisplit!=serial) forbids the split-K parallelism recovery. byte_identical_tax_floor_tps=14.39; "
        "#393's un-pack pin is AT the floor (gap=0); removable=0. The only sub-floor path (#400 pinned-K, "
        "+14 TPS) is a NEW-reference re-capture (greedy/PPL-valid, M-invariant, but flips ~O(3/882) vs "
        "today's bytes) -- a CONTRACT decision for humans, not a byte-identical win. Knife-edge margin "
        "stays +1.21 (482.74). CLOSES the last tax-reduction lever from #418 on the byte-identical side."
    )

    inputs = {
        "mu_p_fast_52": MU_P, "base_467_modeled_393": BASE_467_MODELED_393,
        "base_467_measured_412": BASE_467_MEASURED_412, "base_467_measured_std_412": BASE_467_MEASURED_STD_412,
        "tax_measured": TAX_MEASURED, "tax_modeled_393": TAX_MODELED_393,
        "eps_star_405": EPS_STAR, "reduction_order_perturb_max_87_381": REDUCTION_ORDER_PERTURB_MAX,
        "thinnest_gap_global_87": THINNEST_GAP_GLOBAL, "neartie_by_j_405": NEARTIE_BY_J,
        "n_neartie_at_eps_405": N_NEARTIE_AT_EPS, "n_served_flips_405": N_SERVED_FLIPS,
        "n_served_positions_405": N_SERVED_POSITIONS,
        "cheapest_strict_attn_backend_393": CHEAPEST_STRICT_ATTN_BACKEND,
        "n_byte_exact_attn_configs_393": N_BYTE_EXACT_ATTN_CONFIGS_393,
        "verify_penalty_free_393": VERIFY_PENALTY_FREE_393,
        "fa_sliding0_is_strict_floor_393": FA_SLIDING0_IS_STRICT_FLOOR_393,
        "m1_attn_occupancy_frac_400": M1_ATTN_OCCUPANCY_FRAC_400,
        "m1_heuristic_occupancy_frac_400": M1_HEURISTIC_OCCUPANCY_FRAC_400,
        "m1_attn_bw_frac_400": M1_ATTN_BW_FRAC_400,
        "attn_lever_gain_realistic_400": ATTN_LEVER_GAIN_REALISTIC_400,
        "attn_lever_gain_realistic_408": ATTN_LEVER_GAIN_REALISTIC_408,
        "pinnedk_recovery_frac_realistic_400": PINNEDK_RECOVERY_FRAC_REALISTIC_400,
        "pinnedk_m_invariant_byte_exact_feasible_400": PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400,
        "pinnedk_produces_new_reference_400": PINNEDK_PRODUCES_NEW_REFERENCE_400,
        "multisplit_eq_serial_bytes_400": MULTISPLIT_EQ_SERIAL_BYTES_400,
        "attn_rebuild_is_flagged_served_change_400": ATTN_REBUILD_IS_FLAGGED_SERVED_CHANGE_400,
        "t_attn_frac_408": T_ATTN_FRAC_408,
        "selective_tps_412": SELECTIVE_TPS_412, "selective_beats_blanket_412": SELECTIVE_BEATS_BLANKET_412,
        "flagged_step_fraction_412": FLAGGED_STEP_FRACTION_412,
        "fastest_realizable_strict_config_412": FASTEST_REALIZABLE_STRICT_CONFIG_412,
        "cb3_supply_403": CB3_SUPPLY_403, "cb3_kstar_403": CB3_KSTAR_403,
        "stack_frozen": STACK_FROZEN, "knife_edge_margin_frozen": KNIFE_EDGE_MARGIN_FROZEN,
        "stack_recapture": STACK_RECAPTURE, "stack_recapture_ceiling_411": STACK_RECAPTURE_CEILING_411,
        "m_deployed": M_DEPLOYED, "k_deployed": K_DEPLOYED, "target": TARGET,
        "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
        "src_393_run": SRC_393_RUN, "src_400_run": SRC_400_RUN, "src_408_run": SRC_408_RUN,
        "src_412_run": SRC_412_RUN, "src_403_run": SRC_403_RUN, "src_418_run": SRC_418_RUN,
        "src_407_ref": "human re-scope: maximize fastest strictly-equivalent TPS",
    }

    return {
        "pr": 423, "agent": "denken", "kind": "byte-identical-reduction-tax-floor",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps": 0,
        "baseline_fast_frontier_tps": MU_P, "baseline_fast_frontier_ppl": PPL_DEPLOYED,
        "blanket_strict_base_tps_measured": BASE_467_MEASURED_412,
        "headline": headline,
        "inputs": inputs,
        "tax_decomposition": decomp,
        "floor_classification": floor,
        # ---- HEADLINE deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "byte_identical_tax_floor_tps": floor["byte_identical_tax_floor_tps"],   # PRIMARY
        "tax_decomp_fp32_accum_tps": decomp["tax_decomp_fp32_accum_tps"],
        "tax_decomp_serialization_tps": decomp["tax_decomp_serialization_tps"],
        "tax_decomp_lost_batch_parallel_tps": decomp["tax_decomp_lost_batch_parallel_tps"],
        "removable_tax_tps": floor["removable_tax_tps"],
        "floor_tax_tps": floor["floor_tax_tps"],
        "current_pin_gap_to_floor_tps": floor["current_pin_gap_to_floor_tps"],
        "served_change_to_approach_floor": served_change,
        "knife_edge_margin_if_floor_reached_tps": KNIFE_EDGE_MARGIN_FROZEN,
        "byte_identical_floor_self_test_passes": selftest["passes"],
        # secondary (re-capture path, contract-gated)
        "recapture_recovered_tps": floor["recapture_recovered_tps"],
        "recapture_residual_floor_tps": floor["recapture_residual_floor_tps"],
        "recapture_stack_tps": STACK_RECAPTURE,
        "self_test": selftest,
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        f = report["floor_classification"]
        d = report["tax_decomposition"]
        wandb.summary.update({
            "headline": report["headline"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "byte_identical_tax_floor_tps": report["byte_identical_tax_floor_tps"],
            "tax_decomp_fp32_accum_tps": report["tax_decomp_fp32_accum_tps"],
            "tax_decomp_serialization_tps": report["tax_decomp_serialization_tps"],
            "tax_decomp_lost_batch_parallel_tps": report["tax_decomp_lost_batch_parallel_tps"],
            "removable_tax_tps": report["removable_tax_tps"],
            "floor_tax_tps": report["floor_tax_tps"],
            "current_pin_gap_to_floor_tps": report["current_pin_gap_to_floor_tps"],
            "served_change_to_approach_floor": report["served_change_to_approach_floor"],
            "knife_edge_margin_if_floor_reached_tps": report["knife_edge_margin_if_floor_reached_tps"],
            "byte_identical_floor_self_test_passes": report["byte_identical_floor_self_test_passes"],
        })
        wandb.log({
            "summary/byte_identical_tax_floor_tps": report["byte_identical_tax_floor_tps"],
            "summary/tax_total_measured": TAX_MEASURED,
            "summary/tax_total_modeled": TAX_MODELED_393,
            "summary/tax_decomp_fp32_accum_tps": report["tax_decomp_fp32_accum_tps"],
            "summary/tax_decomp_serialization_tps": report["tax_decomp_serialization_tps"],
            "summary/tax_decomp_lost_batch_parallel_tps": report["tax_decomp_lost_batch_parallel_tps"],
            "summary/removable_tax_tps": report["removable_tax_tps"],
            "summary/floor_tax_tps": report["floor_tax_tps"],
            "summary/current_pin_gap_to_floor_tps": report["current_pin_gap_to_floor_tps"],
            "summary/knife_edge_margin_if_floor_reached_tps": report["knife_edge_margin_if_floor_reached_tps"],
            "summary/recapture_recovered_tps": report["recapture_recovered_tps"],
            "summary/recapture_residual_floor_tps": report["recapture_residual_floor_tps"],
            "summary/recapture_stack_tps": report["recapture_stack_tps"],
            "summary/stack_frozen_tps": STACK_FROZEN,
            "summary/perturb_max": f["perturb_max"], "summary/eps_star": f["eps_star"],
            "summary/has_proof_margin": float(f["has_proof_margin"]),
            "summary/split_k_byte_identical_to_frozen": float(f["split_k_byte_identical_to_frozen_ref"]),
            "summary/multisplit_eq_serial_bytes_400": float(f["multisplit_eq_serial_bytes_400"]),
            "summary/pin_is_at_floor": float(f["pin_is_at_floor"]),
            "summary/m1_occupancy_frac": M1_ATTN_OCCUPANCY_FRAC_400,
            "summary/m1_heuristic_occupancy_frac": M1_HEURISTIC_OCCUPANCY_FRAC_400,
            "summary/cb3_supply_403": CB3_SUPPLY_403,
            "summary/ppl_deployed": PPL_DEPLOYED,
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # decomposition bar (TPS by component) as a small table.
        tbl = wandb.Table(columns=["component", "tps", "classification"])
        tbl.add_data("fp32_accum_(a)", d["tax_decomp_fp32_accum_tps"], "floor=0 (already paid)")
        tbl.add_data("serialization_(b)", d["tax_decomp_serialization_tps"], "FLOOR (knife edge)")
        tbl.add_data("lost_batch_parallel_(c)", d["tax_decomp_lost_batch_parallel_tps"], "0 (selective net-neg)")
        wandb.log({"tax_decomp_table": tbl})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    d = r["tax_decomposition"]
    f = r["floor_classification"]
    print("\n=== Byte-identical reduction tax floor (PR #423, denken) ===")
    print(f"deployed NON-strict (#52) = {MU_P:.2f}   blanket-strict measured (#412) = {BASE_467_MEASURED_412:.4f}"
          f" +/- {BASE_467_MEASURED_STD_412:.3f}   tax = {TAX_MEASURED:.3f} (modeled {TAX_MODELED_393:.3f})")
    print("\n-- tax decomposition (instruction 2) --")
    print(f"  (a) fp32-accumulate overhead    = {d['tax_decomp_fp32_accum_tps']:>7.3f} TPS  "
          f"[{d['a_zero_reason']}]")
    print(f"  (b) single-segment serialization= {d['tax_decomp_serialization_tps']:>7.3f} TPS  "
          f"[M=1 draft-lane occupancy 10%->88%; the WHOLE tax]")
    print(f"  (c) lost batch-parallelism      = {d['tax_decomp_lost_batch_parallel_tps']:>7.3f} TPS  "
          f"[{d['c_zero_reason']}]")
    print(f"  (d) launch/sync of flagged seg  = {d['tax_decomp_launch_sync_tps']:>7.3f} TPS  "
          f"[no separable component; un-pack is single-kernel]")
    print(f"  closure residual = {d['decomp_residual_tps']:.2e}   where: {d['where_tax_lives']}")
    print("\n-- byte-identity FLOOR gate (instruction 3/4; my #418 knife edge on split-K) --")
    print(f"  reassociation perturb = {f['perturb_max']}   near-tie margin eps* = {f['eps_star']}   "
          f"perturb >= eps* (knife edge): {f['perturb_ge_eps_knife_edge']}  proof margin: {f['has_proof_margin']}")
    print(f"  near-ties blanket all positions: {f['neartie_blankets_all_positions']}   "
          f"#400 multisplit==serial bytes: {f['multisplit_eq_serial_bytes_400']}")
    print(f"  => split-K byte-identical to FROZEN M=1 ref: {f['split_k_byte_identical_to_frozen_ref']}  "
          f"(serialization is FLOOR)")
    print("\n-- FLOOR verdict (PRIMARY; FROZEN-byte-identity contract) --")
    print(f"  byte_identical_tax_floor_tps = {f['byte_identical_tax_floor_tps']:.3f}   "
          f"removable_tax_tps = {f['removable_tax_tps']:.3f}   floor_tax_tps = {f['floor_tax_tps']:.3f}")
    print(f"  current_pin_gap_to_floor_tps = {f['current_pin_gap_to_floor_tps']:.3f}  "
          f"(#393 fa2_unpack_ns1 IS the floor: {f['pin_is_at_floor']})")
    print("\n-- the ONLY sub-floor path: M-invariant RE-CAPTURE (NOT frozen-byte-identical; contract-gated) --")
    print(f"  pinned-K (#400) recovers {f['recapture_recovered_tps']:.3f} TPS -> residual floor "
          f"{f['recapture_residual_floor_tps']:.3f}; new reference: {f['recapture_is_new_reference']}; "
          f"flagged change: {f['recapture_is_flagged_served_change']}")
    print(f"  frozen stack {STACK_FROZEN:.2f} (+{KNIFE_EDGE_MARGIN_FROZEN:.2f} w/ identity)  vs  "
          f"re-capture stack ~{STACK_RECAPTURE:.2f} (#411 ceiling {STACK_RECAPTURE_CEILING_411:.2f}, NEW ref)")
    print(f"\nPPL unchanged {PPL_DEPLOYED} <= {PPL_GATE}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"byte_identical_floor_self_test_passes = {r['byte_identical_floor_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Byte-identical reduction tax floor (PR #423).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #423 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="cb3-tax-floor")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/byte-identical-reduction-tax-floor")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/byte_identical_reduction_tax_floor/byte_identical_reduction_tax_floor_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = Path("research/validity/byte_identical_reduction_tax_floor/byte_identical_reduction_tax_floor_selftest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\nbyte_identical_floor_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "byte_identical_tax_floor_tps": float(report["byte_identical_tax_floor_tps"]),
        "tax_decomp_fp32_accum_tps": float(report["tax_decomp_fp32_accum_tps"]),
        "tax_decomp_serialization_tps": float(report["tax_decomp_serialization_tps"]),
        "tax_decomp_lost_batch_parallel_tps": float(report["tax_decomp_lost_batch_parallel_tps"]),
        "removable_tax_tps": float(report["removable_tax_tps"]),
        "floor_tax_tps": float(report["floor_tax_tps"]),
        "current_pin_gap_to_floor_tps": float(report["current_pin_gap_to_floor_tps"]),
        "knife_edge_margin_if_floor_reached_tps": float(report["knife_edge_margin_if_floor_reached_tps"]),
        "byte_identical_floor_self_test_passes": bool(report["byte_identical_floor_self_test_passes"]),
        "primary_metric": {"name": "byte_identical_tax_floor_tps", "value": float(report["byte_identical_tax_floor_tps"])},
        "test_metric": {"name": "byte_identical_floor_self_test_passes",
                        "value": float(report["byte_identical_floor_self_test_passes"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
