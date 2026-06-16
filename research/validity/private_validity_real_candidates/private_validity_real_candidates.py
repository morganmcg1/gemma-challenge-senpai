#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Private re-run validity for the REAL #474 fire candidates (PR #486, denken).

CPU-ONLY ANALYSIS. NO kernel re-measure, NO served-file change, NO HF Job, NO
submission, NO --launch. analysis_only=true, official_tps=0, no_served_file_change=true.

THE PREMISE CORRECTION
----------------------
My #480 (`3oudivg1`) priced private-validity for the strict submission on the
COMPOSED 457-TPS config. That config does NOT realize: land's full-serve pre-check
came back 222.32, not ~457.5 -- the ~457 family was confirmed composed/locus, never
served. The TWO configs that actually fire are now:
  * FLOOR-LOCK 161.70 -- the M=1 AR (autoregressive, NON-speculative) floor. The only
    config-reachable literal-1.0 byte-exact config (denken #476 `m1_floor_tps`). NO
    drafter -> NO acceptance shift.
  * GLOBAL-FLAG 234.47 -- the blanket VLLM_BATCH_INVARIANT=1 pin (ubel #470 `ugqnytji`).
    Operative byte-exact, but STILL SPECULATIVE: spec stays alive, E_accept~3.87 on the
    SAME MTP-K7 drafter as the deployed stack.

THE LOAD-BEARING PHYSICS (why the two candidates land on opposite verdicts)
---------------------------------------------------------------------------
The organizer-measured deployed gap is 481.53 public -> 460.85 private = 4.295%.
ubel #379 decomposes it: ~85% ACCEPTANCE (3.661%, the drafter accepting fewer tokens
on the private prompt distribution) + ~15% CTXLEN (0.633%, global-layer KV growth).
The acceptance bucket is a DRAFTER property -> it transfers to ANY config that shares
the drafter, and vanishes for a config that has no drafter.

  * FLOOR-LOCK is non-speculative -> the 3.661% acceptance bucket is GONE. Its only
    systematic public->private gap is the (tiny) M=1-AR ctxlen term. So its expected
    delta ~= 0 and the live risk is purely sigma_hw / measurement.
  * GLOBAL-FLAG is speculative on the same drafter -> it RE-INHERITS the full 3.661%
    acceptance bucket (+ its own ctxlen). Byte-exact OUTPUT does NOT buy private SPEED
    safety: its expected delta ~= the deployed 4.295%, a 0.7pp hair off the 5% gate.

So the PR's framing ("a byte-exact config has ~0% quality-driven private delta") is
EXACT for floor-lock (truly non-spec) but OPTIMISTIC for global-flag (still spec). We
report BOTH frames and headline the spec-honest one.

THE sigma_hw-FRACTION EFFECT (instruction 2)
--------------------------------------------
The single official private re-run draws TPS ~ N(mu_priv, sigma). The banked between-
session sigma_hw is 4.864 TPS (lawine #467 reconciliation; sigma_within 0.349 same-session,
sigma_oneshot 4.876 = sqrt(between^2+within^2)). The QUESTION the PR poses: is that 4.864
an ABSOLUTE TPS noise (same 4.864 for every config -> its FRACTION inflates for slow
configs: 1.01% of 481.53 but 3.01% of 161.70) or a FRACTIONAL noise (~1.01% of whatever
TPS the config runs at)? Hardware timing noise (clock/thermal/contention) is MULTIPLICATIVE
on wall-time -> FRACTIONAL on TPS is the physically-grounded model; the banked sigmas are
all ~1.0% relative (the "convention" is literally 1%). We report breach prob under BOTH:
fractional (physical, headline) and absolute (the PR's conservative worst-case). The
fraction table isolates the effect at zero systematic gap.

Reproduce: cd target/ && .venv/bin/python \
  research/validity/private_validity_real_candidates/private_validity_real_candidates.py \
  --wandb_group private-validity-real-candidates --wandb_name denken/private-validity-real-candidates
"""
from __future__ import annotations

import argparse
import json
import math
import os

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))

# ---- banked inputs (all read-only; this card measures nothing on the GPU) -----------
GAP_JSON = os.path.join(
    _root, "research/validity/public_private_gap_decomposition/public_private_gap_decomposition_results.json")
SIGMA_RECON = os.path.join(_root, "research/empirical_sigma_hw/fresh_n10/reconciliation.json")
SIGMA_JSON = os.path.join(_root, "research/empirical_sigma_hw/fresh_n10/sigma_hw.json")
LITERAL_JSON = os.path.join(
    _root, "research/validity/literal_1p0_config_reachable/literal_1p0_config_reachable_results.json")
CROSSCHECK_JSON = os.path.join(
    _root, "research/validity/strict_frontier_realize_crosscheck/strict_frontier_realize_crosscheck_report.json")

# Validity gates (BASELINE.md / program.md).
DELTA_GATE = 0.05            # public<->private TPS reproduction gate
PPL_GATE = 2.42
PPL_ANCHOR = 2.3772         # deployed public PPL; byte-exact greedy configs reproduce the base PPL

# Deployed ground-truth public/private pair (organizer cmpatino-verifier, BASELINE.md).
DEPLOYED_PUBLIC_TPS = 481.53
DEPLOYED_PRIVATE_TPS = 460.85


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_here, "private_validity_real_candidates_results.json"))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="private-validity-real-candidates")
    ap.add_argument("--wandb_name", default="denken/private-validity-real-candidates")
    ap.add_argument("--job_type", default="analysis")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    gap = json.load(open(GAP_JSON))
    recon = json.load(open(SIGMA_RECON))["reconciliation"]
    sigma_an = json.load(open(SIGMA_JSON))["analysis"]
    literal = json.load(open(LITERAL_JSON))
    cross = json.load(open(CROSSCHECK_JSON))

    # ---- deployed public->private gap decomposition (ubel #379) ----
    deployed_gap = float(gap["decomposition_central"]["total_gap_frac"])          # 0.042946
    accept_bucket = float(gap["decomposition_central"]["bucket_acceptance_abs_pct"]) / 100.0  # 0.03661 (DRAFTER)
    ctxlen_bucket = float(gap["decomposition_central"]["bucket_ctxlen_abs_pct"]) / 100.0      # 0.00633 (global KV)
    deployed_gap_recon = (DEPLOYED_PUBLIC_TPS - DEPLOYED_PRIVATE_TPS) / DEPLOYED_PUBLIC_TPS
    # attn-growth anchors for the M=1-AR ctxlen roofline (same #379 corners).
    attn_us_m8 = float(gap["imported"]["attn_us_257"])                             # 557.9 (M=8 verify attn)
    attn_pub = float(gap["corners"]["central"]["step_loss"]["attn_us_pub"])        # 557.9
    attn_priv_central = float(gap["corners"]["central"]["step_loss"]["attn_us_priv"])    # 610.73 (+9.47%)
    attn_priv_pess = float(gap["corners"]["pessimistic"]["step_loss"]["attn_us_priv"])   # 695.26 (+24.6%)
    rel_attn_growth_central = (attn_priv_central - attn_pub) / attn_pub            # 0.0947
    rel_attn_growth_pess = (attn_priv_pess - attn_pub) / attn_pub                  # 0.2461

    # ---- sigma_hw anchors (lawine #467 reconciliation) ----
    sigma_within = float(recon["sigma_within_measured_tps_at_481"])               # 0.349 (same-session)
    sigma_between = float(recon["sigma_between_cited_tps"])                        # 4.864 (between-session)  <- PR
    sigma_oneshot = float(recon["sigma_oneshot_reconstructed_tps"])               # 4.8765 (single official draw)
    sigma_convention = float(sigma_an["convention_sigma_hw"])                     # 4.8153 (1% convention)
    # fractional interpretation: the relative sigma the noise carries at ANY TPS.
    sigma_between_frac = sigma_between / DEPLOYED_PUBLIC_TPS                       # 1.010%
    sigma_oneshot_frac = sigma_oneshot / DEPLOYED_PUBLIC_TPS                      # 1.013%

    # ---- candidate public TPS (the REAL fire configs) ----
    floorlock_tps = float(literal["m1_floor_tps"])                               # 161.70 (M=1 AR floor)
    globalflag_tps = float(cross["realized_strict_tps_bi_pin"])                  # 234.4667 (blanket BI)
    globalflag_eaccept = float(cross["e_accept_under_bi"])                       # 3.8695 (spec ALIVE)
    globalflag_lo = float(cross["realized_strict_tps_bi_pin_official_lo"])       # 229.87
    globalflag_hi = float(cross["realized_strict_tps_bi_pin_official_hi"])       # 239.06
    floorlock_ppl = float(literal["ppl"])                                        # 2.3772 (base greedy)
    globalflag_ppl = float(cross["ppl"])                                         # 2.3770 (byte-exact spec)

    # ---- (1) FLOOR-LOCK 161.70 (M=1 AR, NON-spec) systematic gap = ctxlen ONLY ----
    # M=1 AR per-token step from the calibrated floor; attn for 1 query token ~ M=8 attn / 8
    # (decode attention ~ O(M*L)); roofline-estimate the ctxlen fraction, bound it above by the
    # deployed (M=8) ctxlen bucket -- M=1's smaller attn fraction CANNOT exceed it.
    step_m1_us = 1.0e6 / floorlock_tps                                            # ~6184 us/token
    attn_m1_us = attn_us_m8 / 8.0                                                 # ~69.7 us (1 query token)
    attn_frac_m1 = attn_m1_us / step_m1_us                                        # ~1.13%
    floorlock_ctxlen_roofline = attn_frac_m1 * rel_attn_growth_central            # ~0.107% (M=1 estimate)
    floorlock_ctxlen_conservative = ctxlen_bucket                                 # 0.633% (deployed UPPER bound)
    floorlock_accept_gap = 0.0                                                    # NO drafter -> NO acceptance

    # ---- (2) GLOBAL-FLAG 234.47 (blanket BI, SPEC ALIVE) inherits the acceptance bucket ----
    # spec-honest: same MTP-K7 drafter -> re-inherits 3.661% acceptance + comparable ctxlen.
    # (BI runs full single-segment LOCAL attn -> ctxlen could be LARGER, so deployed_gap is a
    #  conservative-LOW central for global-flag, not an upper bound.)
    globalflag_accept_gap = accept_bucket                                         # 0.03661 (INHERITED)
    globalflag_ctxlen_gap = ctxlen_bucket                                         # 0.00633 (>= this; BI full-local)
    globalflag_gap_spechonest = deployed_gap                                      # 4.295% central
    # byte-exact frame (PR premise): pretend no acceptance, ctxlen only.
    globalflag_gap_byteexact = ctxlen_bucket

    # ---- candidate table ----
    def predict(tps, gap_frac):
        return tps * (1.0 - gap_frac)

    def threshold(tps):
        return (1.0 - DELTA_GATE) * tps

    def breach(tps, gap_frac, sigma_abs):
        """P(single private draw < 0.95*public) under N(mu_priv, sigma_abs)."""
        mu = predict(tps, gap_frac)
        return _phi((threshold(tps) - mu) / sigma_abs)

    def breach_fractional(tps, gap_frac, sigma_frac):
        mu = predict(tps, gap_frac)
        return _phi((threshold(tps) - mu) / (sigma_frac * tps))

    candidates = {}
    # FLOOR-LOCK: systematic = ctxlen-only. Headline central = conservative ctxlen (upper bound).
    fl_gap = floorlock_ctxlen_conservative + floorlock_accept_gap
    candidates["floor_lock_161p70"] = {
        "label": "floor-lock 161.70 (M=1 AR, non-spec, literal-1.0)",
        "public_tps": floorlock_tps,
        "is_speculative": False,
        "acceptance_gap_pct": 100.0 * floorlock_accept_gap,
        "ctxlen_gap_pct_roofline": 100.0 * floorlock_ctxlen_roofline,
        "ctxlen_gap_pct_conservative": 100.0 * floorlock_ctxlen_conservative,
        "systematic_gap_pct": 100.0 * fl_gap,
        "predicted_private_tps": predict(floorlock_tps, fl_gap),
        "strict_private_delta_pct": 100.0 * fl_gap,
        "headroom_pp": 100.0 * (DELTA_GATE - fl_gap),
        "sigma_hw_frac_of_tps_pct": 100.0 * sigma_between / floorlock_tps,
        "breach_prob_between_abs": breach(floorlock_tps, fl_gap, sigma_between),
        "breach_prob_oneshot_abs": breach(floorlock_tps, fl_gap, sigma_oneshot),
        "breach_prob_fractional": breach_fractional(floorlock_tps, fl_gap, sigma_oneshot_frac),
        "breach_prob_zero_systematic_abs": breach(floorlock_tps, 0.0, sigma_between),
        "ppl": floorlock_ppl,
    }
    # GLOBAL-FLAG: spec-honest (inherits acceptance) is the headline; byte-exact frame reported too.
    gf_gap = globalflag_gap_spechonest
    candidates["global_flag_234p47"] = {
        "label": "global-flag 234.47 (blanket BI, SPEC alive E_accept~3.87, byte-exact output)",
        "public_tps": globalflag_tps,
        "public_tps_official_band": [globalflag_lo, globalflag_hi],
        "is_speculative": True,
        "e_accept": globalflag_eaccept,
        "acceptance_gap_pct": 100.0 * globalflag_accept_gap,
        "ctxlen_gap_pct": 100.0 * globalflag_ctxlen_gap,
        "systematic_gap_pct_spechonest": 100.0 * globalflag_gap_spechonest,
        "systematic_gap_pct_byteexact_premise": 100.0 * globalflag_gap_byteexact,
        "predicted_private_tps": predict(globalflag_tps, gf_gap),
        "predicted_private_tps_byteexact": predict(globalflag_tps, globalflag_gap_byteexact),
        "strict_private_delta_pct": 100.0 * gf_gap,
        "strict_private_delta_pct_byteexact": 100.0 * globalflag_gap_byteexact,
        "headroom_pp": 100.0 * (DELTA_GATE - gf_gap),
        "headroom_pp_byteexact": 100.0 * (DELTA_GATE - globalflag_gap_byteexact),
        "sigma_hw_frac_of_tps_pct": 100.0 * sigma_between / globalflag_tps,
        "breach_prob_between_abs": breach(globalflag_tps, gf_gap, sigma_between),
        "breach_prob_oneshot_abs": breach(globalflag_tps, gf_gap, sigma_oneshot),
        "breach_prob_fractional": breach_fractional(globalflag_tps, gf_gap, sigma_oneshot_frac),
        "breach_prob_byteexact_abs": breach(globalflag_tps, globalflag_gap_byteexact, sigma_between),
        "breach_prob_zero_systematic_abs": breach(globalflag_tps, 0.0, sigma_between),
        "ppl": globalflag_ppl,
    }

    # ---- (2) sigma_hw-FRACTION table: the effect at ZERO systematic gap (isolate the noise) ----
    # how much breach prob a byte-exact (zero-gap) config carries PURELY from the sigma fraction.
    fraction_table = {}
    for tag, tps in [("deployed_481p53", DEPLOYED_PUBLIC_TPS),
                     ("global_flag_234p47", globalflag_tps),
                     ("floor_lock_161p70", floorlock_tps)]:
        fraction_table[tag] = {
            "public_tps": tps,
            "sigma_between_abs_tps": sigma_between,
            "sigma_between_frac_of_tps_pct": 100.0 * sigma_between / tps,
            "breach_byteexact_abs_sigma": breach(tps, 0.0, sigma_between),       # absolute -> inflates for slow
            "breach_byteexact_frac_sigma": breach_fractional(tps, 0.0, sigma_oneshot_frac),  # fractional -> flat
            "delta_to_gate_in_abs_sigma": (threshold(tps) - tps) / (-sigma_between),  # +N sigma of headroom
        }

    # ---- (3) verdict ----
    # FLOOR-LOCK: SAFE -- expected delta <= ctxlen (no acceptance, ~5pp headroom); breach ~0 under
    #   physical fractional sigma; only the conservative ABSOLUTE-sigma worst-case lifts breach to
    #   ~5-7% (a pure measurement-unluck tail, NOT a systematic problem).
    fl = candidates["floor_lock_161p70"]
    floorlock_safe = bool(fl["strict_private_delta_pct"] < 100.0 * DELTA_GATE
                          and fl["breach_prob_fractional"] < 0.05)
    floorlock_abs_sigma_flag = bool(fl["breach_prob_between_abs"] >= 0.05)
    # GLOBAL-FLAG: NOT safe under the spec-honest model -- it re-inherits the ~4.3% acceptance gap,
    #   leaving ~0.7pp headroom and 24-37% breach. SAFE only under the (false) byte-exact premise.
    gf = candidates["global_flag_234p47"]
    globalflag_safe_spechonest = bool(gf["strict_private_delta_pct"] < 100.0 * DELTA_GATE
                                      and gf["breach_prob_fractional"] < 0.05)
    globalflag_safe_byteexact_premise = bool(gf["strict_private_delta_pct_byteexact"] < 100.0 * DELTA_GATE
                                             and gf["breach_prob_byteexact_abs"] < 0.05)

    safest_ship = "floor_lock_161p70"
    ppl_clears_both = bool(floorlock_ppl <= PPL_GATE and globalflag_ppl <= PPL_GATE)

    # ---- self-tests ----
    st = {}
    st["deployed_gap_reconstructs_4p295"] = bool(abs(deployed_gap - deployed_gap_recon) < 1e-4)
    st["accept_plus_ctxlen_equals_deployed_gap"] = bool(abs((accept_bucket + ctxlen_bucket) - deployed_gap) < 1e-4)
    st["acceptance_is_majority_of_gap"] = bool(accept_bucket > ctxlen_bucket)
    st["floorlock_nonspec_zero_acceptance"] = bool(floorlock_accept_gap == 0.0)
    st["floorlock_ctxlen_roofline_below_conservative"] = bool(
        floorlock_ctxlen_roofline < floorlock_ctxlen_conservative)
    st["floorlock_delta_well_below_gate"] = bool(fl["strict_private_delta_pct"] < 100.0 * DELTA_GATE - 3.0)
    st["floorlock_safe_under_fractional_sigma"] = bool(fl["breach_prob_fractional"] < 0.01)
    st["floorlock_sigma_fraction_is_largest"] = bool(
        fl["sigma_hw_frac_of_tps_pct"] > gf["sigma_hw_frac_of_tps_pct"]
        > 100.0 * sigma_between / DEPLOYED_PUBLIC_TPS)
    st["globalflag_is_speculative"] = bool(globalflag_eaccept > 1.0)
    st["globalflag_inherits_acceptance"] = bool(abs(globalflag_accept_gap - accept_bucket) < 1e-9)
    st["globalflag_delta_near_gate"] = bool(4.0 < gf["strict_private_delta_pct"] < 5.0)
    st["globalflag_thin_headroom"] = bool(gf["headroom_pp"] < 1.0)
    st["globalflag_breach_material"] = bool(gf["breach_prob_oneshot_abs"] > 0.10)
    st["globalflag_byteexact_premise_would_be_safe"] = bool(globalflag_safe_byteexact_premise)
    st["globalflag_spechonest_not_safe"] = bool(not globalflag_safe_spechonest)
    st["sigma_fraction_inflates_for_slow_configs"] = bool(
        fraction_table["floor_lock_161p70"]["breach_byteexact_abs_sigma"]
        > fraction_table["global_flag_234p47"]["breach_byteexact_abs_sigma"]
        > fraction_table["deployed_481p53"]["breach_byteexact_abs_sigma"])
    st["fractional_sigma_is_config_invariant"] = bool(
        abs(fraction_table["floor_lock_161p70"]["breach_byteexact_frac_sigma"]
            - fraction_table["deployed_481p53"]["breach_byteexact_frac_sigma"]) < 1e-6)
    st["sigma_oneshot_reconstructs_between_within"] = bool(
        abs(sigma_oneshot - math.sqrt(sigma_between**2 + sigma_within**2)) < 0.05)
    st["floorlock_safest_systematic"] = bool(fl["strict_private_delta_pct"] < gf["strict_private_delta_pct"])
    st["ppl_clears_both_candidates"] = ppl_clears_both
    st["threshold_is_95pct_of_public"] = bool(abs(threshold(floorlock_tps) - 0.95 * floorlock_tps) < 1e-9)
    st["candidates_below_deployed_flagship"] = bool(globalflag_tps < DEPLOYED_PUBLIC_TPS
                                                    and floorlock_tps < DEPLOYED_PUBLIC_TPS)
    finite = [fl["strict_private_delta_pct"], fl["predicted_private_tps"], fl["breach_prob_between_abs"],
              fl["breach_prob_fractional"], gf["strict_private_delta_pct"], gf["predicted_private_tps"],
              gf["breach_prob_oneshot_abs"], sigma_between, sigma_oneshot]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    self_test_passes = all(st.values())

    verdict = {
        # ---- PR-required per-candidate metrics ----
        "floorlock_predicted_private_tps": fl["predicted_private_tps"],
        "floorlock_strict_private_delta_pct": fl["strict_private_delta_pct"],
        "floorlock_private_validity_safe": floorlock_safe,
        "floorlock_breach_prob_empirical_between": fl["breach_prob_between_abs"],
        "floorlock_breach_prob_fractional": fl["breach_prob_fractional"],
        "floorlock_headroom_pp": fl["headroom_pp"],
        "floorlock_sigma_hw_frac_pct": fl["sigma_hw_frac_of_tps_pct"],
        "floorlock_abs_sigma_flag": floorlock_abs_sigma_flag,
        "globalflag_predicted_private_tps": gf["predicted_private_tps"],
        "globalflag_strict_private_delta_pct": gf["strict_private_delta_pct"],   # spec-honest (PRIMARY)
        "globalflag_strict_private_delta_pct_byteexact": gf["strict_private_delta_pct_byteexact"],
        "globalflag_private_validity_safe": globalflag_safe_spechonest,
        "globalflag_breach_prob_empirical_between": gf["breach_prob_between_abs"],
        "globalflag_breach_prob_oneshot": gf["breach_prob_oneshot_abs"],
        "globalflag_breach_prob_fractional": gf["breach_prob_fractional"],
        "globalflag_headroom_pp": gf["headroom_pp"],
        "globalflag_sigma_hw_frac_pct": gf["sigma_hw_frac_of_tps_pct"],
        # ---- shared / decision ----
        "safest_ship": safest_ship,
        "ppl_clears_both": ppl_clears_both,
        "ppl_gate": PPL_GATE,
        "delta_gate": DELTA_GATE,
        "deployed_gap_pct": 100.0 * deployed_gap,
        "deployed_acceptance_bucket_pct": 100.0 * accept_bucket,
        "deployed_ctxlen_bucket_pct": 100.0 * ctxlen_bucket,
        "sigma_within_tps": sigma_within,
        "sigma_between_tps": sigma_between,
        "sigma_oneshot_tps": sigma_oneshot,
        "sigma_between_frac_pct": 100.0 * sigma_between_frac,
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
        "no_kernel_rebuild": True, "no_hf_job": True, "no_launch": True, "no_submission": True,
        "self_test_passes": self_test_passes,
    }

    reconcile = (
        f"Real #474 fire candidates, private re-run validity. The composed-457 config is DEFUNCT "
        f"(land full-serve pre-check 222.32); the two configs that fire are FLOOR-LOCK 161.70 "
        f"(M=1 AR, non-spec) and GLOBAL-FLAG 234.47 (blanket BI, spec alive E_accept={globalflag_eaccept:.2f}). "
        f"Deployed gap 4.295% = {100*accept_bucket:.3f}% ACCEPTANCE (drafter) + {100*ctxlen_bucket:.3f}% ctxlen. "
        f"FLOOR-LOCK has NO drafter -> acceptance bucket GONE -> systematic delta "
        f"{fl['strict_private_delta_pct']:.3f}% (ctxlen-only), headroom {fl['headroom_pp']:.2f}pp; breach "
        f"{100*fl['breach_prob_fractional']:.4f}% (fractional sigma, physical) .. "
        f"{100*fl['breach_prob_between_abs']:.2f}% (absolute sigma_hw {sigma_between:.3f}, conservative). "
        f"GLOBAL-FLAG is STILL SPECULATIVE on the same drafter -> RE-INHERITS the {100*accept_bucket:.3f}% "
        f"acceptance gap -> systematic delta {gf['strict_private_delta_pct']:.3f}%, headroom only "
        f"{gf['headroom_pp']:.2f}pp; breach {100*gf['breach_prob_oneshot_abs']:.1f}% (one-shot sigma). "
        f"sigma_hw-FRACTION effect: 4.864 TPS is {fraction_table['deployed_481p53']['sigma_between_frac_of_tps_pct']:.2f}% "
        f"of 481.53 but {fl['sigma_hw_frac_of_tps_pct']:.2f}% of 161.70 -> a zero-gap byte-exact config's "
        f"breach inflates 0% -> {100*fraction_table['floor_lock_161p70']['breach_byteexact_abs_sigma']:.2f}% "
        f"purely from the slower TPS (absolute-sigma); vanishes under fractional sigma. "
        f"PPL byte-exact for both ({floorlock_ppl} / {globalflag_ppl} <= 2.42). "
        f"VERDICT: ship FLOOR-LOCK 161.70 (safe; the only flag is the conservative absolute-sigma tail). "
        f"GLOBAL-FLAG 234.47 is FASTER but NOT private-safe: byte-exact output does not buy speed-safety "
        f"because it re-inherits the very acceptance gap byte-exactness was meant to avoid.")
    verdict["reconcile_line"] = reconcile

    payload = {
        "pr": 486,
        "issue": 474,
        "author": "denken",
        "leg": "private re-run SPEED-validity (5% Delta) for the REAL fire candidates: floor-lock 161.70 + global-flag 234.47",
        "config": {
            "delta_gate": DELTA_GATE, "ppl_gate": PPL_GATE,
            "deployed_public_tps": DEPLOYED_PUBLIC_TPS, "deployed_private_tps": DEPLOYED_PRIVATE_TPS,
            "floorlock_public_tps": floorlock_tps, "globalflag_public_tps": globalflag_tps,
            "sigma_between_tps": sigma_between, "sigma_oneshot_tps": sigma_oneshot,
            "imports": {
                "deployed_gap_decomp": os.path.relpath(GAP_JSON, _root),
                "sigma_reconciliation": os.path.relpath(SIGMA_RECON, _root),
                "sigma_fresh_n10": os.path.relpath(SIGMA_JSON, _root),
                "floorlock_m1_floor": os.path.relpath(LITERAL_JSON, _root),
                "globalflag_bi_pin": os.path.relpath(CROSSCHECK_JSON, _root),
            },
            "note": "Private re-run validity for the REAL #474 candidates. CPU analysis only; no "
                    "kernel re-measure, no served change, no HF Job, no launch, no submission.",
        },
        "candidates": candidates,
        "sigma_fraction_table": fraction_table,
        "verdict": verdict,
        "self_test_conditions": st,
        "public_evidence_used": (
            "ubel #379 (5kpb73tb) public->private gap decomposition: 4.295% = 3.661% acceptance + 0.633% "
            "ctxlen; BASELINE.md 'top drafter stacks lose 4-9% TPS on the private set (prompt-distribution "
            "shift)' confirms the acceptance bucket is a DRAFTER property. denken #476 (p68oo5tj) "
            "literal_1p0_config_reachable: literal-1.0 reachable only at the M=1 AR floor 161.70. ubel #470 "
            "(ugqnytji) strict-frontier BI-pin cross-check: blanket VLLM_BATCH_INVARIANT=1 realizes 234.47 "
            "official, spec ALIVE E_accept~3.87 (does NOT collapse to the 161.70 floor), PPL 2.3770<=2.42. "
            "lawine #467 reconciliation: sigma_within 0.349 (same-session), sigma_between 4.864, sigma_oneshot "
            "4.876. Deployed pair 481.53->460.85 (Delta 4.3%) organizer cmpatino-verifier (BASELINE.md)."),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))

    print(f"[prv] FLOOR-LOCK 161.70: delta={fl['strict_private_delta_pct']:.3f}% (priv "
          f"{fl['predicted_private_tps']:.2f} TPS) headroom {fl['headroom_pp']:.2f}pp | breach "
          f"{100*fl['breach_prob_fractional']:.4f}% frac .. {100*fl['breach_prob_between_abs']:.2f}% abs | "
          f"safe={floorlock_safe} (abs-sigma flag={floorlock_abs_sigma_flag})", flush=True)
    print(f"[prv] GLOBAL-FLAG 234.47: delta={gf['strict_private_delta_pct']:.3f}% (priv "
          f"{gf['predicted_private_tps']:.2f} TPS) headroom {gf['headroom_pp']:.2f}pp | breach "
          f"{100*gf['breach_prob_oneshot_abs']:.1f}% (inherits acceptance) | safe={globalflag_safe_spechonest} "
          f"| byteexact-premise-safe={globalflag_safe_byteexact_premise}", flush=True)
    print(f"[prv] sigma_hw 4.864 fraction: deployed "
          f"{fraction_table['deployed_481p53']['sigma_between_frac_of_tps_pct']:.2f}% | global-flag "
          f"{gf['sigma_hw_frac_of_tps_pct']:.2f}% | floor-lock {fl['sigma_hw_frac_of_tps_pct']:.2f}% "
          f"-> zero-gap breach inflates to {100*fraction_table['floor_lock_161p70']['breach_byteexact_abs_sigma']:.2f}% "
          f"(abs) / {100*fraction_table['floor_lock_161p70']['breach_byteexact_frac_sigma']:.4f}% (frac)", flush=True)
    print(f"[prv] safest_ship={safest_ship} | self_test={self_test_passes}", flush=True)
    print(f"[prv] {reconcile}", flush=True)

    if not args.no_wandb:
        _log_wandb(args, payload)
    return 0 if self_test_passes else 1


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type=args.job_type, config=payload.get("config", {}))
    vd = payload["verdict"]
    run.summary.update({k: v for k, v in vd.items() if isinstance(v, (int, float, bool, str))})

    ct = wandb.Table(columns=["candidate", "public_tps", "is_speculative", "acceptance_gap_pct",
                              "systematic_delta_pct", "predicted_private_tps", "headroom_pp",
                              "sigma_hw_frac_pct", "breach_fractional", "breach_abs_oneshot", "ppl"])
    fl = payload["candidates"]["floor_lock_161p70"]
    gf = payload["candidates"]["global_flag_234p47"]
    ct.add_data("floor_lock_161p70", fl["public_tps"], False, fl["acceptance_gap_pct"],
                fl["strict_private_delta_pct"], fl["predicted_private_tps"], fl["headroom_pp"],
                fl["sigma_hw_frac_of_tps_pct"], fl["breach_prob_fractional"], fl["breach_prob_oneshot_abs"],
                fl["ppl"])
    ct.add_data("global_flag_234p47", gf["public_tps"], True, gf["acceptance_gap_pct"],
                gf["strict_private_delta_pct"], gf["predicted_private_tps"], gf["headroom_pp"],
                gf["sigma_hw_frac_of_tps_pct"], gf["breach_prob_fractional"], gf["breach_prob_oneshot_abs"],
                gf["ppl"])
    run.log({"candidate_verdict": ct})

    ft = wandb.Table(columns=["config", "public_tps", "sigma_between_frac_pct",
                              "breach_byteexact_abs_sigma", "breach_byteexact_frac_sigma"])
    for tag, r in payload["sigma_fraction_table"].items():
        ft.add_data(tag, r["public_tps"], r["sigma_between_frac_of_tps_pct"],
                    r["breach_byteexact_abs_sigma"], r["breach_byteexact_frac_sigma"])
    run.log({"sigma_fraction_table": ft})
    run.finish()
    print(f"[prv] logged W&B run {args.wandb_entity}/{args.wandb_project} "
          f"name={args.wandb_name} group={args.wandb_group}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
