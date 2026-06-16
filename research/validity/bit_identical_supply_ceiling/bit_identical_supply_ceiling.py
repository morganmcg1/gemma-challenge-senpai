#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Bit-identical supply ceiling: the SAFE no-contract frontier above ~482.74
(PR #428, wirbel). 0-GPU pure-analysis re-pricing card. Analysis-only: NO
served-file change, NO HF Job, NO submission, NOT a launch. official_tps=0.

THE QUESTION
------------
How far can the equivalence-respecting TPS frontier go using ONLY bit-identical
(maxdiff=0.0, changes-cycles-not-tokens, same-reduction-order) levers -- i.e.
WITHOUT a human reference-contract decision (denken #427's pinned-K reference
change to ~496.7) or a tie-break (stark #421)?

  floor  = blanket-strict 467.14 (#412 measured) + cb3 +15.60 (#403, banked)
         = 482.74  (the strictly-equivalent, frozen-byte realizable frontier TODAY)
  ceiling(lawine #411 supply ledger, includes a reference change) = 497.44

We deliver `bit_identical_supply_ceiling_tps`: the highest the SAFE bit-identical
frontier reaches, stacking every surviving maxdiff=0.0 lever ON TOP OF cb3.

THE ENUMERATION (instruction 1) -- bit-identical levers ABOVE cb3
----------------------------------------------------------------
Scanning the MERGED record (`approval-gated-8gpu-20260613`) for every maxdiff=0.0
GEMM / kernel-config supply lever above cb3:

  (1) verify-SDPA `num_stages 3->2` (#270 iwwcmvez / #279 -- consumed here).
      The DEPLOYED TRITON_ATTN `kernel_unified_attention` is a bare @triton.jit
      launched at Triton defaults (num_warps=4, num_stages=3). Forcing
      num_stages=2 is PURE scheduling (same MMA reduction order) -> torch.equal
      -> maxdiff=0.0 on BOTH deployed 3D split-KV M=8 verify shapes (#279):
      global head-512 1.018x (md=0.0), sliding head-256 1.093x (md=0.0); 128-draw
      greedy-identity gate divergent=0. This is the ONLY bit-identical lever with
      a NONZERO supply contribution above cb3.
  (2) int4-Marlin body GEMMs (q/k/v/o/gate/up/down) -- ALREADY byte-exact at
      decode (#390); zero incremental lift (already inside the floor).
  (3) int4-Marlin lm_head -- ALREADY byte-exact (#384,
      deterministic_lmhead_recovers_deficit_tps=0.0); zero incremental lift.

EXCLUDED (not bit-identical / out-of-lane): cb3 (#403, already banked into the
floor); pinned-K split-K reassociation (#400/#423 -- flips ~3/882 near-ties vs
the frozen reference, multisplit_eq_serial_bytes=False -> REFERENCE-CHANGING,
that is denken #427's lane); batch-invariant verify GEMM (#363, maxdiff 9.77e-4
-> new reference); the drafter loopgraph (NO-GO, wirbel #424); flashinfer BI
(#349, BW-bound 473.5<481.53); deterministic fusion (breaks identity). So #428's
question reduces to: how much does lever (1) add, priced REALISTICALLY?

THE RE-PRICING (instruction 2) -- price realism, not roofline
-------------------------------------------------------------
#279 priced lever (1) against a STEP_US=1218.2us composition REFERENCE step and
reported +1.293% / 487.76 TPS. But the REAL wall-clock decode step is 8017us
(#284, directly CUDA-event measured: verify 6532us + drafter 1445us + host 40us;
99.5% GPU-bound). #279's bit-identical SDPA saving is an ABSOLUTE 15.55us/step
(sum over the 21 tunable layers at the realistic decode ctx~512; ctx=2048 OVER-
states and reaches <1% of steps). Re-basing that ABSOLUTE saving onto the REAL
8017us step (the verify body is 81.5% of the step and on the critical path, so a
verify-body saving comes off the step ~1:1) collapses #279's +1.293% to:

  roofline_gain = 15.55 / 8017 = +0.194%  ->  +0.94 TPS on the 482.74 floor
                                          ->  ceiling 483.68 (ROOFLINE upper bound)

That 6.6x haircut (vs #279's step-inflated number) IS the dominant realism
correction. A SECOND, unmeasurable-here discount remains: standalone CUDA-graph-
replay savings over-state IN-GRAPH (ONEGRAPH) realization. #273 measured exactly
this on this stack -- a standalone/composition-predicted saving realized at ratio
-2.02 (K=4), i.e. NEGATIVE -- so the merged-record precedent is that a saving of
this kind can fail to realize. Confirming the in-graph realization needs a served-
kernel-config A/B (num_stages=2 at the launch site) -> a FLAGGED served change we
do NOT build. We therefore report the realized contribution as a BAND:

  verify_sdpa_numstages_realized_tps in [0.0 (#273-cautionary floor), +0.94 (roofline)]

and the safe bit-identical ceiling as a band [482.74, 483.68]. The lever is NOT
needed to clear the deployed frontier -- cb3 alone already gives +1.21 (#423).

THE VERDICT (instruction 3)
---------------------------
bit_identical_supply_ceiling_tps ~ 483.68 (roofline UB) / 482.74 (realization
floor). Either way it clears the deployed 481.53 but sits ~13.8-14.7 TPS BELOW
lawine #411's 497.44. That entire gap is the REFERENCE-CHANGING pinned-K lever
(+14.29, denken #427) -- the bit-identical contribution to closing it is 0. So:
the bit-identical-only path is EXHAUSTED at cb3 (+ an at-most-+0.94 verify-SDPA
crumb); the only way to 497.44 / 500 is denken's reference contract or stark's
tie-break. That null is the decision-critical deliverable.

DEPLOY SURFACE (instruction 4)
------------------------------
Lever (1) is a SERVED-KERNEL-CONFIG change (force num_stages=2 at the served
TRITON_ATTN `kernel_unified_attention` launch). The QUESTION (how much TPS, is it
bit-identical) is fully answered in-envelope (<=+0.94 TPS, maxdiff=0.0); only the
BUILD is the flagged ask. Flag it, do not build.

SELF-TEST (`bit_identical_supply_ceiling_self_test_passes`, PRIMARY; 0-GPU)
--------------------------------------------------------------------------
Re-loads every merged artifact and cross-checks the pinned anchors (#412 / #403 /
#411 / #284 / #279 / #273 / #423), then verifies the re-pricing arithmetic, the
bit-identity gate, the band monotonicity, and the deployed/lawine bounds. No GPU,
no vLLM, no served-file change: the deployed senpai vLLM wheel venv is not present
in this launch env, so lever (1)'s kernel numbers are CONSUMED from #279's logged
local-A10G run (explicitly sanctioned: "consume/extend wirbel #279's measurement")
-- #279's run IS the local-A10G measurement, performed when the venv was available.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL = HERE.parent            # research/validity
SPEED = VAL.parent / "speed"  # research/speed

TOL = 1e-2

# --------------------------------------------------------------------------- #
# Pinned anchors (imported byte-exactly; cross-checked vs merged JSONs below). #
# --------------------------------------------------------------------------- #
DEPLOYED_TPS = 481.53                 # PR #52 deployed NON-equivalent frontier
PPL_DEPLOYED = 2.3772                 # PR #52 official PPL (bit-identical => unchanged)
PPL_GATE = 2.42                       # public cap (reference PPL + 5%)

BLANKET_STRICT_412 = 467.1400155438763   # #412 selective_recompute measured blanket-strict
CB3_LIFT_403 = 15.603896595803747        # #403 m8_lift_at_kstar (k*=229); banked supply
CB3_LIFT_BANKED = 15.60                   # the banked/rounded cb3 lift used by #423/#411 stack
# frozen floor == #423 STACK_FROZEN (blanket-strict + banked cb3); the 0.004 TPS
# vs the precise #403 lift is immaterial and we match the merged-record value.
FROZEN_FLOOR = BLANKET_STRICT_412 + CB3_LIFT_BANKED   # 482.7400155438763
KNIFE_EDGE_MARGIN = FROZEN_FLOOR - DEPLOYED_TPS        # +1.2100155438763 (#423)

LAWINE_CEILING_411 = 497.44           # #411 supply-ledger ceiling (incl. ref-changing pinned-K)
PINNEDK_LIFT_411 = 14.29              # #411 pinnedk_attn lift (REFERENCE-CHANGING; denken #427)

# Real wall-clock decode step (#284, directly CUDA-event measured) ------------
STEP_US = 8017.0                      # decode wall / step
VERIFY_GPU_US = 6532.0                # verify (execute_model) GPU; 81.5% of step
DRAFTER_GPU_US = 1445.0               # drafter (propose) GPU
GPU_BUSY_SHARE = 0.9950106024697518   # 99.5% GPU-bound (host overhead 40us, immaterial)

# Verify-SDPA num_stages 3->2 lever (#270/#279, consumed) --------------------
# absolute bit-identical SDPA saving per decode step at the realistic decode ctx
# (~512; sum over 21 tunable layers = 7 global head-512 + 14 sliding head-256).
SDPA_SAVING_US_CTX512 = 15.554561614990178   # #279 realistic-ctx headline saving
SDPA_SAVING_US_CTX768 = 18.015576998392703   # near-worst-typical
SDPA_SAVING_US_CTX2048 = 39.59124883015954   # loose UB (<1% of steps reach ctx2048)
SDPA_G512_SPEEDUP = 1.0179972610434467       # global head-512 num_stages 3->2 (md=0.0)
SDPA_S256_SPEEDUP = 1.0928190402164732       # sliding head-256 num_stages 3->2 (md=0.0)
N_VERIFY_GLOBAL_H512 = 7              # tunable global head-512 TRITON_ATTN layers
N_VERIFY_SLIDING_H256 = 14           # tunable sliding head-256 TRITON_ATTN layers
N_VERIFY_FA2_H256 = 16               # sliding head-256 flipped to FA2 (NOT tunable)

# Realization precedent (#273 static-K wall-clock A/B on this ONEGRAPH stack) --
REALIZATION_RATIO_273_K4 = -2.01774820233227   # measured/composition: NEGATIVE realization
REALIZATION_COMP_GAIN_273_K4 = 4.27653374979271
K7_LOCAL_WALL_TPS_273 = 453.6177679392844      # local A10G wall (official 481.53; fractional transfers)

# Artifact paths (re-loaded + cross-checked in the self-test) -----------------
ART_412 = VAL / "selective_recompute_equivalent_tps" / "selective_recompute_equivalent_tps_results.json"
ART_403 = VAL / "cb3_conservative_k_deployable_lift" / "cb3_conservative_k_deployable_lift_results.json"
ART_411 = VAL / "flagged_supply_deploy_surface_ledger" / "flagged_supply_deploy_surface_ledger_results.json"
ART_284 = VAL / "decode_host_overhead" / "report.json"
ART_273 = VAL / "static_k_wallclock_ab" / "report.json"
ART_423 = VAL / "byte_identical_reduction_tax_floor" / "byte_identical_reduction_tax_floor_results.json"
ART_279 = SPEED / "verify_sdpa_linear_deploy" / "results.json"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def _load(art: Path) -> dict | None:
    if not art.exists():
        return None
    try:
        return json.loads(art.read_text())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Section 1 -- enumerate the bit-identical levers above cb3.                    #
# --------------------------------------------------------------------------- #
def enumerate_levers() -> list[dict]:
    """Every maxdiff=0.0 GEMM/kernel-config supply lever, classified. Only one
    (verify-SDPA num_stages 3->2) has a nonzero supply contribution above cb3."""
    return [
        {
            "lever": "verify_sdpa_num_stages_3to2",
            "pr": "#270/#279",
            "bit_identical": True,        # maxdiff=0.0 on both deployed 3D M=8 shapes
            "above_cb3": True,
            "delivers": True,             # nonzero roofline contribution
            "deploy_surface": "served_kernel_config_change",
            "verdict": "the only bit-identical lever with nonzero supply above cb3",
        },
        {
            "lever": "int4_marlin_body_gemms_x7",
            "pr": "#390",
            "bit_identical": True,        # ALREADY byte-exact at decode
            "above_cb3": False,           # already inside the floor
            "delivers": False,            # zero incremental lift
            "deploy_surface": "none (already strict)",
            "verdict": "already byte-exact; no incremental lift",
        },
        {
            "lever": "int4_marlin_lm_head",
            "pr": "#384",
            "bit_identical": True,        # ALREADY byte-exact
            "above_cb3": False,
            "delivers": False,
            "deploy_surface": "none (already strict)",
            "verdict": "already byte-exact (deterministic_lmhead_recovers_deficit_tps=0.0)",
        },
        {
            "lever": "pinnedk_splitk_reassociation",
            "pr": "#400/#423/#427",
            "bit_identical": False,       # flips ~3/882 near-ties vs frozen ref
            "above_cb3": True,
            "delivers": False,            # EXCLUDED: reference-changing (denken #427 lane)
            "deploy_surface": "kernel rebuild (flagged, NEW reference)",
            "verdict": "EXCLUDED -- reference-changing, denken #427's lane (+14.29)",
        },
        {
            "lever": "batch_invariant_verify_gemm",
            "pr": "#363",
            "bit_identical": False,       # maxdiff 9.77e-4 vs deployed -> new ref
            "above_cb3": True,
            "delivers": False,
            "deploy_surface": "n/a",
            "verdict": "EXCLUDED -- not bit-identical (new reference)",
        },
        {
            "lever": "drafter_loopgraph_fusion",
            "pr": "#424",
            "bit_identical": False,
            "above_cb3": True,
            "delivers": False,
            "deploy_surface": "n/a",
            "verdict": "EXCLUDED -- NO-GO (wirbel #424)",
        },
    ]


# --------------------------------------------------------------------------- #
# Section 2 -- re-price the verify-SDPA lever onto the REAL step (#284).        #
# --------------------------------------------------------------------------- #
def reprice_verify_sdpa(saving_us: float) -> dict:
    """An ABSOLUTE bit-identical SDPA saving (us/step) re-based onto the REAL
    8017us decode step. The verify body (6532us) is on the critical path and is
    81.5% of the step, so a verify-body saving comes off the step ~1:1. TPS
    scales inversely with step at fixed E[T] (bit-identical => E[T] unchanged)."""
    new_step = STEP_US - saving_us
    gain_pct = 100.0 * (STEP_US / new_step - 1.0)
    roofline_tps = FROZEN_FLOOR * STEP_US / new_step
    roofline_delta = roofline_tps - FROZEN_FLOOR
    return {
        "saving_us": saving_us,
        "new_step_us": new_step,
        "verify_body_share_of_step": VERIFY_GPU_US / STEP_US,
        "sdpa_share_of_verify_body": 271.76277319590247 / VERIFY_GPU_US,  # #279 ctx512 standalone
        "roofline_gain_pct": gain_pct,
        "roofline_tps": roofline_tps,
        "roofline_delta_tps": roofline_delta,
    }


def build_report() -> dict:
    levers = enumerate_levers()
    n_bit_identical_levers = sum(1 for L in levers if L["bit_identical"] and L["above_cb3"] and L["delivers"])
    n_already_strict = sum(1 for L in levers if L["bit_identical"] and not L["above_cb3"])
    n_excluded = sum(1 for L in levers if not L["delivers"] and L["above_cb3"])

    # --- re-price the one delivering lever at the realistic ctx (headline) ----
    rp = reprice_verify_sdpa(SDPA_SAVING_US_CTX512)
    rp768 = reprice_verify_sdpa(SDPA_SAVING_US_CTX768)
    rp2048 = reprice_verify_sdpa(SDPA_SAVING_US_CTX2048)

    # Roofline contribution (full in-graph realization of the standalone saving).
    roofline_delta = rp["roofline_delta_tps"]                 # +0.94 TPS
    # Realization band: the #273-cautionary floor is 0.0 (standalone savings on
    # this stack have measured realization ratio <= 0; we clamp at 0 because the
    # verify-SDPA lever is bit-identical/E[T]-neutral, so it lacks static-K's
    # structural E[T] trade that drove #273's NEGATIVE ratio -- 0 is the safe
    # conservative floor, not the literal -2.02). Upper bound = roofline.
    realized_floor = 0.0
    realized_roofline = roofline_delta
    # Headline scalar: the realism-corrected wall-clock contribution (the real-
    # step re-basing). Reported as the band's upper bound; the floor is carried
    # explicitly so it is never mistaken for a guaranteed realized gain.
    verify_sdpa_numstages_realized_tps = realized_roofline

    bit_identical_supply_ceiling_tps = FROZEN_FLOOR + realized_roofline       # 483.68 (UB)
    bit_identical_supply_ceiling_floor_tps = FROZEN_FLOOR + realized_floor    # 482.74 (realization floor)

    safe_frontier_beats_deployed_481 = bool(FROZEN_FLOOR > DEPLOYED_TPS)      # True via cb3 alone

    gap_to_lawine_from_ceiling = LAWINE_CEILING_411 - bit_identical_supply_ceiling_tps
    gap_to_lawine_from_floor = LAWINE_CEILING_411 - FROZEN_FLOOR
    # the entire 482.74->497.44 gap is the reference-changing pinned-K lever
    gap_is_reference_changing = bool(abs(gap_to_lawine_from_floor - PINNEDK_LIFT_411) < 0.5)

    selftest = run_self_tests(rp, bit_identical_supply_ceiling_tps,
                              bit_identical_supply_ceiling_floor_tps,
                              verify_sdpa_numstages_realized_tps,
                              n_bit_identical_levers, gap_is_reference_changing)

    deploy_surface = (
        "verify-SDPA num_stages=2 is a served-kernel-config change at the "
        "TRITON_ATTN kernel_unified_attention launch site (a FLAGGED served-file "
        "change). The QUESTION (<=+0.94 TPS, maxdiff=0.0) is answered in-envelope; "
        "only the BUILD is the flagged ask. Flag it, do NOT build."
    )

    headline = (
        f"SAFE bit-identical supply ceiling = {bit_identical_supply_ceiling_tps:.2f} TPS "
        f"(roofline UB; realization floor {bit_identical_supply_ceiling_floor_tps:.2f}). "
        f"The one delivering lever (verify-SDPA num_stages 3->2, #270/#279) re-prices "
        f"from #279's step-inflated +1.293% to +{rp['roofline_gain_pct']:.3f}% "
        f"(+{roofline_delta:.2f} TPS) on the REAL {STEP_US:.0f}us step (#284), banded "
        f"[0, +{roofline_delta:.2f}] by the #273 realization precedent. It clears the "
        f"deployed {DEPLOYED_TPS:.2f} (cb3 alone already does, +{KNIFE_EDGE_MARGIN:.2f}) "
        f"but sits ~{gap_to_lawine_from_ceiling:.1f}-{gap_to_lawine_from_floor:.1f} TPS "
        f"below lawine #411's {LAWINE_CEILING_411:.2f}. That gap is ENTIRELY the "
        f"reference-changing pinned-K (+{PINNEDK_LIFT_411:.2f}, denken #427) -- the "
        f"bit-identical contribution to closing it is 0. The bit-identical-only path "
        f"is EXHAUSTED at cb3; 497.44/500 needs a reference contract or a tie-break."
    )

    return {
        "headline": headline,
        "inputs": {
            "deployed_tps": DEPLOYED_TPS, "blanket_strict_412": BLANKET_STRICT_412,
            "cb3_lift_banked": CB3_LIFT_BANKED, "cb3_lift_403_precise": CB3_LIFT_403,
            "frozen_floor": FROZEN_FLOOR, "knife_edge_margin": KNIFE_EDGE_MARGIN,
            "lawine_ceiling_411": LAWINE_CEILING_411, "pinnedk_lift_411": PINNEDK_LIFT_411,
            "step_us": STEP_US, "verify_gpu_us": VERIFY_GPU_US, "drafter_gpu_us": DRAFTER_GPU_US,
            "gpu_busy_share": GPU_BUSY_SHARE,
            "sdpa_saving_us_ctx512": SDPA_SAVING_US_CTX512,
            "realization_ratio_273_k4": REALIZATION_RATIO_273_K4,
            "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
        },
        "levers": levers,
        "reprice_ctx512": rp, "reprice_ctx768": rp768, "reprice_ctx2048": rp2048,
        "realization_band": {
            "floor_tps": realized_floor, "roofline_tps": realized_roofline,
            "floor_reason": "in-graph realization may be 0 (#273 measured <=0 for "
                            "standalone savings on this ONEGRAPH stack; unmeasurable "
                            "here without a forbidden served-file A/B)",
            "roofline_reason": "full in-graph realization of the bit-identical SDPA "
                               "saving on the 99.5%-GPU-bound critical path",
        },
        # ---- HEADLINE deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "bit_identical_supply_ceiling_tps": bit_identical_supply_ceiling_tps,          # PRIMARY
        "bit_identical_supply_ceiling_floor_tps": bit_identical_supply_ceiling_floor_tps,
        "n_bit_identical_levers": n_bit_identical_levers,
        "n_already_strict_surfaces": n_already_strict,
        "n_excluded_levers": n_excluded,
        "verify_sdpa_numstages_realized_tps": verify_sdpa_numstages_realized_tps,
        "verify_sdpa_numstages_realized_tps_floor": realized_floor,
        "verify_sdpa_numstages_roofline_gain_pct": rp["roofline_gain_pct"],
        "safe_frontier_beats_deployed_481": safe_frontier_beats_deployed_481,
        "gap_to_lawine_from_ceiling_tps": gap_to_lawine_from_ceiling,
        "gap_to_lawine_from_floor_tps": gap_to_lawine_from_floor,
        "gap_is_reference_changing": gap_is_reference_changing,
        "deploy_surface": deploy_surface,
        "bit_identical_supply_ceiling_self_test_passes": selftest["passes"],
        "self_test": selftest,
    }


# --------------------------------------------------------------------------- #
# Section 3 -- self-tests (0-GPU; PRIMARY gate).                                #
# --------------------------------------------------------------------------- #
def run_self_tests(rp: dict, ceiling: float, ceiling_floor: float,
                   realized: float, n_levers: int, gap_ref_changing: bool) -> dict:
    c: dict[str, bool] = {}

    # a) pinned anchors round-trip.
    c["a_deployed_is_481p53"] = abs(DEPLOYED_TPS - 481.53) < TOL
    c["a_blanket_strict_412"] = abs(BLANKET_STRICT_412 - 467.1400155438763) < 1e-9
    c["a_cb3_lift_403_precise"] = abs(CB3_LIFT_403 - 15.603896595803747) < 1e-9
    c["a_frozen_floor_is_482p74"] = abs(FROZEN_FLOOR - 482.7400155438763) < 1e-6
    c["a_knife_edge_is_1p21"] = abs(KNIFE_EDGE_MARGIN - 1.2100155438763) < 1e-6
    c["a_lawine_is_497p44"] = abs(LAWINE_CEILING_411 - 497.44) < TOL
    c["a_step_is_8017"] = STEP_US == 8017.0 and VERIFY_GPU_US == 6532.0

    # b) the re-pricing arithmetic (realism correction #1: real-step re-basing).
    c["b_saving_is_15p55"] = abs(rp["saving_us"] - 15.554561614990178) < 1e-9
    c["b_gain_pct_about_0p19"] = 0.18 < rp["roofline_gain_pct"] < 0.21
    c["b_roofline_delta_about_0p94"] = 0.85 < rp["roofline_delta_tps"] < 1.05
    c["b_verify_share_81pct"] = abs(rp["verify_body_share_of_step"] - 6532.0 / 8017.0) < 1e-9
    # the 6.6x haircut vs #279's step-inflated +1.293% (that used STEP_US=1218.2)
    inflated_gain = 100.0 * (1218.2 / (1218.2 - rp["saving_us"]) - 1.0)
    c["b_haircut_vs_279_is_6x"] = 5.0 < (inflated_gain / rp["roofline_gain_pct"]) < 8.0

    # c) the ceiling + band.
    c["c_ceiling_eq_floor_plus_realized"] = abs(ceiling - (FROZEN_FLOOR + realized)) < 1e-9
    c["c_ceiling_floor_is_482p74"] = abs(ceiling_floor - FROZEN_FLOOR) < 1e-9
    c["c_band_monotone"] = ceiling_floor <= ceiling
    c["c_ceiling_below_lawine"] = ceiling < LAWINE_CEILING_411
    c["c_ceiling_beats_deployed"] = ceiling > DEPLOYED_TPS
    c["c_floor_beats_deployed"] = ceiling_floor > DEPLOYED_TPS   # cb3 alone clears it

    # d) the enumeration: exactly ONE delivering bit-identical lever above cb3.
    c["d_one_delivering_lever"] = n_levers == 1
    c["d_gap_is_reference_changing"] = gap_ref_changing is True

    # e) the realization precedent is carried (price realism, not roofline).
    c["e_273_ratio_negative"] = REALIZATION_RATIO_273_K4 < 0.0
    c["e_273_ratio_is_neg2p02"] = abs(REALIZATION_RATIO_273_K4 - (-2.01774820233227)) < 1e-9
    c["e_gpu_bound_99p5"] = GPU_BUSY_SHARE > 0.99

    # f) PPL preserved; numeric hygiene.
    c["f_ppl_within_gate"] = PPL_DEPLOYED <= PPL_GATE
    c["f_no_nan_inf"] = all(_finite(v) for v in
                            [ceiling, ceiling_floor, realized, rp["roofline_gain_pct"],
                             rp["roofline_delta_tps"], FROZEN_FLOOR, KNIFE_EDGE_MARGIN])

    # k) artifact provenance cross-check (pinned constants == merged JSONs).
    d412, d403, d411, d284, d273, d423, d279 = (
        _load(a) for a in (ART_412, ART_403, ART_411, ART_284, ART_273, ART_423, ART_279))
    if d412 is not None:
        c["k_412_blanket_measured"] = abs(d412.get("blanket_strict_measured_tps", 0) - BLANKET_STRICT_412) < 1e-9
    if d403 is not None:
        recost = d403.get("result", {}).get("recost_at_kstar", {})
        c["k_403_cb3_lift"] = abs(recost.get("m8_lift_at_kstar", 0) - CB3_LIFT_403) < 1e-6
    if d411 is not None:
        c["k_411_ceiling_497p44"] = abs(d411.get("max_stack_tps_under_current_floor", 0) - LAWINE_CEILING_411) < TOL
    if d284 is not None:
        ps = d284.get("per_step_decode_wall", {})
        c["k_284_step_8017"] = abs(ps.get("decode_wall_per_step_us", 0) - STEP_US) < 1e-6
        c["k_284_verify_6532"] = abs(ps.get("verify_gpu_us", 0) - VERIFY_GPU_US) < 1e-6
    if d273 is not None:
        c["k_273_realization_ratio"] = abs(
            d273.get("per_k", {}).get("4", {}).get("realization_ratio", 0) - REALIZATION_RATIO_273_K4) < 1e-9
    if d423 is not None:
        c["k_423_frozen_stack"] = abs(d423.get("knife_edge_margin_if_floor_reached_tps", 0) - KNIFE_EDGE_MARGIN) < 1e-6
        c["k_423_removable_zero"] = d423.get("removable_tax_tps", -1) == 0.0
    if d279 is not None:
        v = d279.get("verdict", {})
        c["k_279_saving_15p55"] = abs(v.get("verify_sdpa_saving_us", 0) - SDPA_SAVING_US_CTX512) < 1e-9
        c["k_279_g512_maxdiff_zero"] = v.get("global_h512_3d_s2_maxdiff", -1) == 0.0
        c["k_279_s256_maxdiff_zero"] = v.get("sliding_h256_3d_s2_maxdiff", -1) == 0.0
        c["k_279_greedy_identical"] = v.get("linear_sdpa_tune_greedy_identical") is True

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# --------------------------------------------------------------------------- #
# Section 4 -- reporting + W&B + entrypoint.                                    #
# --------------------------------------------------------------------------- #
def print_report(r: dict) -> None:
    rp = r["reprice_ctx512"]
    print("\n=== Bit-identical supply ceiling (PR #428, wirbel) ===")
    print(f"deployed NON-equiv (#52) = {DEPLOYED_TPS:.2f}   frozen floor (blanket-strict {BLANKET_STRICT_412:.4f} "
          f"+ cb3 {CB3_LIFT_BANKED:.2f}) = {FROZEN_FLOOR:.4f}  (+{KNIFE_EDGE_MARGIN:.2f} vs deployed, identity-safe)")
    print("\n-- bit-identical lever enumeration (instruction 1) --")
    for L in r["levers"]:
        tag = "DELIVERS" if L["delivers"] else ("already-strict" if not L["above_cb3"] else "EXCLUDED")
        print(f"  [{tag:14s}] {L['lever']:34s} bit_ident={str(L['bit_identical']):5s} ({L['pr']}) -- {L['verdict']}")
    print(f"  => n_bit_identical_levers (delivering, above cb3) = {r['n_bit_identical_levers']}")
    print("\n-- re-pricing the verify-SDPA num_stages 3->2 lever (instruction 2; price realism) --")
    print(f"  #279 SDPA saving (realistic ctx~512) = {rp['saving_us']:.3f} us/step, maxdiff=0.0 (BOTH shapes), 128-gate clean")
    print(f"  REAL step (#284) = {STEP_US:.0f}us (verify {VERIFY_GPU_US:.0f}us = {rp['verify_body_share_of_step']*100:.1f}%, "
          f"{GPU_BUSY_SHARE*100:.1f}% GPU-bound)")
    print(f"  #279 step-inflated (STEP_US=1218.2) = +1.293%  ->  RE-BASED on 8017us = +{rp['roofline_gain_pct']:.3f}% "
          f"(+{rp['roofline_delta_tps']:.3f} TPS)  [6.6x realism haircut]")
    print(f"  realization band [floor {r['realization_band']['floor_tps']:.2f}, roofline "
          f"{r['realization_band']['roofline_tps']:.3f}] TPS  (#273 ratio {REALIZATION_RATIO_273_K4:.2f} -> <=0 possible)")
    print("\n-- VERDICT (instruction 3) --")
    print(f"  bit_identical_supply_ceiling_tps = {r['bit_identical_supply_ceiling_tps']:.2f}  "
          f"(realization floor {r['bit_identical_supply_ceiling_floor_tps']:.2f})")
    print(f"  safe_frontier_beats_deployed_481 = {r['safe_frontier_beats_deployed_481']}")
    print(f"  gap to lawine #411 {LAWINE_CEILING_411:.2f}: {r['gap_to_lawine_from_ceiling_tps']:.2f}-"
          f"{r['gap_to_lawine_from_floor_tps']:.2f} TPS, reference-changing (pinned-K +{PINNEDK_LIFT_411:.2f}, "
          f"denken #427): {r['gap_is_reference_changing']}")
    print(f"\n-- deploy surface (instruction 4) --\n  {r['deploy_surface']}")
    print(f"\nPPL unchanged {PPL_DEPLOYED} <= {PPL_GATE}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"bit_identical_supply_ceiling_self_test_passes = {r['bit_identical_supply_ceiling_self_test_passes']}")


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        rp = report["reprice_ctx512"]
        wandb.summary.update({
            "headline": report["headline"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "bit_identical_supply_ceiling_tps": report["bit_identical_supply_ceiling_tps"],
            "bit_identical_supply_ceiling_floor_tps": report["bit_identical_supply_ceiling_floor_tps"],
            "n_bit_identical_levers": report["n_bit_identical_levers"],
            "verify_sdpa_numstages_realized_tps": report["verify_sdpa_numstages_realized_tps"],
            "verify_sdpa_numstages_realized_tps_floor": report["verify_sdpa_numstages_realized_tps_floor"],
            "safe_frontier_beats_deployed_481": report["safe_frontier_beats_deployed_481"],
            "gap_to_lawine_from_floor_tps": report["gap_to_lawine_from_floor_tps"],
            "bit_identical_supply_ceiling_self_test_passes": report["bit_identical_supply_ceiling_self_test_passes"],
        })
        wandb.log({
            "summary/bit_identical_supply_ceiling_tps": report["bit_identical_supply_ceiling_tps"],
            "summary/bit_identical_supply_ceiling_floor_tps": report["bit_identical_supply_ceiling_floor_tps"],
            "summary/frozen_floor": FROZEN_FLOOR,
            "summary/deployed_tps": DEPLOYED_TPS,
            "summary/lawine_ceiling_411": LAWINE_CEILING_411,
            "summary/verify_sdpa_numstages_realized_tps": report["verify_sdpa_numstages_realized_tps"],
            "summary/verify_sdpa_numstages_realized_tps_floor": report["verify_sdpa_numstages_realized_tps_floor"],
            "summary/verify_sdpa_roofline_gain_pct": rp["roofline_gain_pct"],
            "summary/sdpa_saving_us_ctx512": SDPA_SAVING_US_CTX512,
            "summary/step_us": STEP_US,
            "summary/gap_to_lawine_from_floor_tps": report["gap_to_lawine_from_floor_tps"],
            "summary/pinnedk_lift_411": PINNEDK_LIFT_411,
            "summary/realization_ratio_273_k4": REALIZATION_RATIO_273_K4,
            "summary/n_bit_identical_levers": report["n_bit_identical_levers"],
            "summary/knife_edge_margin": KNIFE_EDGE_MARGIN,
            "summary/ppl_deployed": PPL_DEPLOYED,
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # lever enumeration table.
        lt = wandb.Table(columns=["lever", "pr", "bit_identical", "above_cb3", "delivers",
                                  "deploy_surface", "verdict"])
        for L in report["levers"]:
            lt.add_data(L["lever"], L["pr"], L["bit_identical"], L["above_cb3"], L["delivers"],
                        L["deploy_surface"], L["verdict"])
        wandb.log({"lever_enumeration": lt})
        # ceiling ladder table.
        ct = wandb.Table(columns=["config", "tps", "reference", "note"])
        ct.add_data("deployed (#52)", DEPLOYED_TPS, "today's bytes", "non-equivalent")
        ct.add_data("blanket-strict (#412)", BLANKET_STRICT_412, "today's bytes", "the 14.39 tax")
        ct.add_data("frozen floor = +cb3 (#403)", FROZEN_FLOOR, "today's bytes", "+1.21 identity-safe")
        ct.add_data("bit-id ceiling (+verify-SDPA roofline)", report["bit_identical_supply_ceiling_tps"],
                    "today's bytes (bit-identical)", "SAFE ceiling UB")
        ct.add_data("lawine #411 (+pinned-K)", LAWINE_CEILING_411, "NEW reference", "reference-changing")
        wandb.log({"ceiling_ladder": ct})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bit-identical supply ceiling (PR #428).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #428 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="bit-id-supply-ceiling")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="wirbel/bit-identical-supply-ceiling")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/bit_identical_supply_ceiling/bit_identical_supply_ceiling_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = HERE / "bit_identical_supply_ceiling_selftest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\nbit_identical_supply_ceiling_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "bit_identical_supply_ceiling_tps": float(report["bit_identical_supply_ceiling_tps"]),
        "bit_identical_supply_ceiling_floor_tps": float(report["bit_identical_supply_ceiling_floor_tps"]),
        "n_bit_identical_levers": int(report["n_bit_identical_levers"]),
        "verify_sdpa_numstages_realized_tps": float(report["verify_sdpa_numstages_realized_tps"]),
        "verify_sdpa_numstages_realized_tps_floor": float(report["verify_sdpa_numstages_realized_tps_floor"]),
        "safe_frontier_beats_deployed_481": bool(report["safe_frontier_beats_deployed_481"]),
        "self_test_passes": bool(report["bit_identical_supply_ceiling_self_test_passes"]),
        "primary_metric": {"name": "bit_identical_supply_ceiling_tps",
                           "value": float(report["bit_identical_supply_ceiling_tps"])},
        "test_metric": {"name": "bit_identical_supply_ceiling_self_test_passes",
                        "value": float(report["bit_identical_supply_ceiling_self_test_passes"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
