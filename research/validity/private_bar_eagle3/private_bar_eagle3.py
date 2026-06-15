#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #300 (student lawine) -- Private-bar EAGLE-3: does a public-E[T] build clear private >=500?

WHAT THIS LEG ANSWERS
---------------------
The binding launch gate (land #245) is a MEASURED *private* >=500 TPS (at lambda_hat>=0.9780,
PPL<=2.42). Every EAGLE-3 acceptance target the fleet has derived is a *public* E[T]: fern #281's
4.966, wirbel #290's 4.9029 (step-banked), wirbel #293's 6.1245 (honest step-overhead-corrected).
ubel #263 measured the deployed LINEAR drafter's public->private acceptance collapse at priv/pub =
0.804 (raw width-1 spine). If EAGLE-3 inherits a similar collapse, a build that clears a public
target could still MISS the private bar. This CPU-analytic card prices the PRIVATE target and tests
the PR's alignment hypothesis: EAGLE-3's strength is DEEP-position fidelity (kanna #289 lifts j>=2
to ~0.91); does that close the private gap?

THE TWO PRIVATE NUMBERS (the crux, and why a scalar 0.804 is the wrong tax)
---------------------------------------------------------------------------
There are TWO private E[T] for the deployed linear drafter, and they are NOT the same haircut:

  (RAW width-1 spine)   E[T]_priv = 3.0898  (ratio 0.8037 ~ the banked 0.804; ubel #258/#263).
                        The per-position decomposition (ubel #258) shows this collapse is a
                        POSITION-1 cliff: the conditional drops 0.7287->0.5991 at j=1 and the deep
                        conditionals (j=5..7) actually IMPROVE on private. 3.0898 -> 387 TPS.

  (DEPLOYED tree-recovered)  E[T]_priv = 460.85/125.268 = 3.6791  (ratio 0.9553).  The deployed
                        system runs an M=8 verify TREE that salvages the rank-2+ matches the
                        width-1 spine rejects -- recovering the position-1 cliff and netting the
                        ORGANIZER-VERIFIED 460.85 private TPS (Delta 4.3% vs 481.53). This is the
                        anchor the PR requires step 1 to reproduce.

So the deployed-effective collapse is NOT a flat 0.804 multiplier. The tree recovers the position-1
loss, leaving a residual collapse CONCENTRATED ON j>=2 (the rank-2+ coverage that thins OOD). We
model exactly that: position-1 HELD (c_1=1.0), a single deep multiplier c_deep on j>=2 calibrated so
the deployed a_k reproduce 3.6791 -> 460.85. (This is the PR's prescribed fallback: "model the
collapse as concentrated on j>=2 and calibrate its magnitude to the aggregate.")

WHY THIS MATTERS FOR EAGLE-3 (the finding)
------------------------------------------
A scalar-0.804 tax says private-500 needs public E[T] = 3.9914/0.804 = 4.966 (= fern #281, by
construction -- wirbel #290 already noted 4.966*0.804=3.992). But that OVER-prices a DEEP-heavy
drafter: under the per-position deployed-effective collapse (position-1 held, mild c_deep~0.97 on
j>=2), a deep-flat EAGLE-3 profile keeps almost all of its deep survival, so private-500 needs only
public E[T] ~= 4.19 -- a far smaller "private tax" than the scalar model implies. EAGLE-3's deep
fidelity ALIGNS with where the (deployed) residual collapse is mild, so rho_priv_e3 climbs to ~0.94.

But the verdict HINGES on a modeling choice we cannot settle on CPU: does EAGLE-3 INHERIT the
deployed tree's rank-2+ recovery? We bracket it honestly:
  * DEPLOYED-EFFECTIVE (primary): EAGLE-3 feeds the SAME M=8 verify tree, so it inherits >= the
    linear drafter's recovery. Collapse = position-1 held, c_deep~0.97 on j>=2.
  * RAW / BINDING (sensitivity): EAGLE-3 gets NO tree recovery; apply ubel #258's per-position raw
    multiplier (position-1 cliff c_1~0.82). This is pessimistic (ubel #263 found tree recovery
    DEGRADES OOD but does not vanish, and a better drafter proposes better rank-2+ candidates).

LOCAL CPU-only analytic card over banked acceptance ladders. No GPU / vLLM / model forward / HF Job
/ submission / served-file change / train.py --launch. NOT a launch. NOT open2. BASELINE stays
481.53 (this leg adds 0 TPS). Greedy/PPL untouched. PRIMARY = self-test.

PRIMARY metric  private_bar_eagle3_self_test_passes
TEST    metric  rho_priv_e3  (EAGLE-3-adjusted private/public ratio, deployed-effective collapse)

Reproduce:
    cd target/ && .venv/bin/python research/validity/private_bar_eagle3/private_bar_eagle3.py \\
        --self-test --wandb_group private-bar-eagle3 --wandb_name lawine/private-bar-eagle3
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Imported fleet anchors (imported EXACTLY, UNCHANGED; this leg re-derives none)
# --------------------------------------------------------------------------- #
K_CAL = 125.268                 # composition anchor: official = K_cal * E[T] (kanna #269)
STEP_US = 1218.2                # deployed per-forward-pass time (us) (kanna #217)
TAU = 1.218                     # composition tau
OFFICIAL_PUBLIC = 481.53        # #52 deployed public TPS (PR #52, fa2sw_precache_kenyan)
CEILING_LAMBDA1 = 520.95        # lambda=1 ceiling (context)
E_T_ANCHOR = 3.844              # deployed linear served public E[T] (kanna #217)
K_SPEC = 7                      # num_speculative_tokens (manifest)
E_T_MAX = float(K_SPEC + 1)     # 8.0 theoretical ceiling at lambda=1
PUBLIC_PPL = 2.3772             # deployed public PPL
PRIVATE_VERIFIED_TPS = 460.85   # ORGANIZER-verified deployed private TPS (step-1 anchor)
PRIVATE_PPL = 2.3777            # organizer private PPL
PRIVATE_DELTA_PCT = 0.043       # (481.53 - 460.85)/481.53 = 4.295% ~ "Delta 4.3%"
LAMBDA_BAR = 0.9780112973731208 # validity bar (greedy-identity floor; construction-satisfied)
PRIVATE_BAR_TPS = 500.0         # the binding launch gate (land #245)
PPL_BAR = 2.42                  # launch PPL gate (construction-satisfied: greedy-exact verify)

# Deployed linear per-position conditional acceptance ladder a_k (kanna #289; PR lists rounded).
# lawine #282 / kanna #289 / denken #297 agree on [0.7293,0.7596,0.7930,0.8228,0.8349,0.8358,0.8465].
A_K = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]

# ubel #258 BANKED raw width-1 spine ladders (public x private), imported UNCHANGED. These define
# the per-position RAW collapse multiplier (the binding/no-tree-recovery sensitivity) and the 0.804
# aggregate consistency check.
A_PUB_RAW = [
    0.728739760479042, 0.7589764102641635, 0.7924989076194682, 0.821702519412012,
    0.8342716929825772, 0.8352594665096346, 0.8472621220149911,
]
A_PRIV_RAW = [
    0.5991304839661618, 0.6893085026081717, 0.7464222790849783, 0.7736626492721749,
    0.8477031613381011, 0.862379683274926, 0.885712826843481,
]
PRIV_RAW_ET_BANKED = 3.0898055282313597     # ubel #258 cached raw private E[T]
PRIV_RAW_RATIO_BANKED = 0.804               # PR-quoted aggregate priv/pub (raw spine)

# EAGLE-3 lift target profile (kanna #289 deep_flat_profile): keep a_1, lift j>=2 to a flat rate.
EAGLE3_DEEP_FLAT_RATE = 0.9144279172167558  # kanna #289 deep_flat_rate_a_d (-> public E[T]=4.966)
EAGLE3_A1 = A_K[0]                           # a_1 held (deep fusion does not move position-1)

# Public EAGLE-3 targets to price against the PRIVATE bar.
PUBLIC_TARGETS = {"fern281": 4.966, "wirbel290": 4.9029, "wirbel293": 6.1245}

ET_PRIVATE_500 = PRIVATE_BAR_TPS / K_CAL                  # 3.99146 direct inverse
ET_PRIVATE_500_PR = PRIVATE_BAR_TPS * (STEP_US / 1000.0) / (K_CAL * TAU)  # PR-formula reading
ET_PRIVATE_ANCHOR = PRIVATE_VERIFIED_TPS / K_CAL          # 3.67905 (460.85 -> E[T])

TOL_TPS = 1.0
TOL_ET = 1e-3
TOL_RATIO = 6e-3


# --------------------------------------------------------------------------- #
# Core: survival ladder, composition, collapse models.
# --------------------------------------------------------------------------- #
def survival(cond: list[float]) -> list[float]:
    """committed-survival S_d = prod_{j<=d} a_j for d=1..K (S_0=1 implicit)."""
    out, acc = [], 1.0
    for p in cond:
        acc *= float(p)
        out.append(acc)
    return out


def et_from_cond(cond: list[float]) -> float:
    """E[T] = 1 + sum_d S_d  (1 base token + expected accepted draft tokens)."""
    return 1.0 + sum(survival(cond))


def tps_at(et: float) -> float:
    """official = K_cal * E[T]  (125.268 * 3.844 = 481.53)."""
    return K_CAL * et


def raw_collapse_multiplier() -> list[float]:
    """per-position c_raw_k = a_priv_raw_k / a_pub_raw_k (ubel #258 banked spine collapse)."""
    return [A_PRIV_RAW[k] / A_PUB_RAW[k] for k in range(K_SPEC)]


def apply_raw_collapse(cond: list[float]) -> list[float]:
    """RAW/binding private map: per-position multiplier, clipped to a valid conditional [0,1]."""
    c = raw_collapse_multiplier()
    return [min(1.0, max(0.0, c[k] * cond[k])) for k in range(K_SPEC)]


def apply_deployed_collapse(cond: list[float], c_deep: float) -> list[float]:
    """DEPLOYED-EFFECTIVE private map: position-1 HELD (tree recovers the spine cliff), uniform
    deep multiplier c_deep on j>=2 (the residual rank-2+ collapse). Clipped to [0,1]."""
    out = [min(1.0, max(0.0, cond[0]))]
    out += [min(1.0, max(0.0, c_deep * cond[k])) for k in range(1, K_SPEC)]
    return out


def _bisect(f, lo: float, hi: float, target: float, iters: int = 200) -> float:
    """monotone-increasing f: return x in [lo,hi] with f(x)~=target."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_c_deep(cond_pub: list[float], et_target: float) -> float:
    """solve c_deep in (0,1] so the deployed-effective collapse of cond_pub hits et_target."""
    return _bisect(lambda c: et_from_cond(apply_deployed_collapse(cond_pub, c)), 0.0, 1.0, et_target)


# --------------------------------------------------------------------------- #
# Public profiles that realize a target public E[T].
# --------------------------------------------------------------------------- #
def deepflat_profile(a_d: float) -> list[float]:
    """EAGLE-3 family: a_1 held at EAGLE3_A1, a_{j>=2} = a_d (flat)."""
    return [EAGLE3_A1] + [a_d] * (K_SPEC - 1)


def uniform_profile(a_u: float) -> list[float]:
    """fallback family for targets above the deep-flat ceiling: all positions = a_u (lifts a_1)."""
    return [a_u] * K_SPEC


def deepflat_ceiling_et() -> float:
    return et_from_cond(deepflat_profile(1.0))  # a_d->1: 1 + EAGLE3_A1*K_SPEC


def profile_for_public_et(target: float) -> dict[str, Any]:
    """Smallest-deviation profile achieving public E[T]=target. Prefer the EAGLE-3 deep-flat family
    (deep fidelity only, a_1 held); if target exceeds its ceiling, fall back to a uniform profile
    (which necessarily LIFTS a_1 -- a denken #297 'a1-deepen' lever, NOT EAGLE-3 deep fusion)."""
    ceil = deepflat_ceiling_et()
    if target <= ceil + 1e-9:
        a_d = _bisect(lambda x: et_from_cond(deepflat_profile(x)), 0.0, 1.0, target)
        cond = deepflat_profile(a_d)
        return {"cond": cond, "family": "deepflat", "param": a_d, "a1_lifted": False,
                "deepflat_feasible": True, "public_et": et_from_cond(cond)}
    a_u = _bisect(lambda x: et_from_cond(uniform_profile(x)), 0.0, 1.0, target)
    cond = uniform_profile(a_u)
    return {"cond": cond, "family": "uniform", "param": a_u, "a1_lifted": True,
            "deepflat_feasible": False, "public_et": et_from_cond(cond)}


def price_target(name: str, target: float, c_deep: float) -> dict[str, Any]:
    """price one public target against the private bar under BOTH collapse regimes."""
    prof = profile_for_public_et(target)
    cond_pub = prof["cond"]
    et_pub = et_from_cond(cond_pub)

    cond_priv_dep = apply_deployed_collapse(cond_pub, c_deep)
    et_priv_dep = et_from_cond(cond_priv_dep)
    tps_priv_dep = tps_at(et_priv_dep)

    cond_priv_raw = apply_raw_collapse(cond_pub)
    et_priv_raw = et_from_cond(cond_priv_raw)
    tps_priv_raw = tps_at(et_priv_raw)

    return {
        "name": name, "public_target": target,
        "family": prof["family"], "param": prof["param"],
        "a1_lifted_above_073": prof["a1_lifted"],
        "deepflat_feasible": prof["deepflat_feasible"],
        "public_et": et_pub,
        # deployed-effective (primary)
        "private_et_deployed": et_priv_dep,
        "private_tps_deployed": tps_priv_dep,
        "clears_private_500_deployed": bool(tps_priv_dep >= PRIVATE_BAR_TPS),
        "residual_tps_deployed": tps_priv_dep - PRIVATE_BAR_TPS,
        "residual_et_deployed": et_priv_dep - ET_PRIVATE_500,
        "rho_deployed": et_priv_dep / et_pub,
        # raw / binding sensitivity
        "private_et_raw": et_priv_raw,
        "private_tps_raw": tps_priv_raw,
        "clears_private_500_raw": bool(tps_priv_raw >= PRIVATE_BAR_TPS),
        "residual_tps_raw": tps_priv_raw - PRIVATE_BAR_TPS,
        "rho_raw": et_priv_raw / et_pub,
        "cond_pub": cond_pub,
        "cond_priv_deployed": cond_priv_dep,
        "cond_priv_raw": cond_priv_raw,
    }


# --------------------------------------------------------------------------- #
# Synthesis (steps 1-3).
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    # ---- STEP 1: anchor the private model on ground truth (460.85). -------- #
    et_pub_ladder = et_from_cond(A_K)                     # ~3.8512 (ladder of deployed a_k)
    comp_anchor_tps = tps_at(E_T_ANCHOR)                  # 481.53 (composition round-trip)

    c_deep = calibrate_c_deep(A_K, ET_PRIVATE_ANCHOR)     # deployed-effective deep multiplier
    cond_priv_dep = apply_deployed_collapse(A_K, c_deep)
    et_priv_dep = et_from_cond(cond_priv_dep)
    tps_priv_dep = tps_at(et_priv_dep)                    # MUST ~= 460.85
    rho_deployed = et_priv_dep / et_pub_ladder            # ~0.955 (tree-recovered ratio)
    delta_pct_dep = (OFFICIAL_PUBLIC - tps_priv_dep) / OFFICIAL_PUBLIC

    # raw-spine consistency: applying ubel #258's per-position multiplier to a_k reproduces 0.804.
    cond_priv_raw_dep = apply_raw_collapse(A_K)
    et_priv_raw_dep = et_from_cond(cond_priv_raw_dep)
    rho_raw_deployed = et_priv_raw_dep / et_pub_ladder    # ~0.804
    et_raw_self = et_from_cond(A_PRIV_RAW)                # must == PRIV_RAW_ET_BANKED (3.0898)

    step1 = {
        "deployed_ak": A_K,
        "public_et_ladder": et_pub_ladder,
        "public_et_anchor_scalar": E_T_ANCHOR,
        "composition_anchor_tps": comp_anchor_tps,
        "c_deep_calibrated": c_deep,
        "private_et_deployed": et_priv_dep,
        "private_tps_deployed": tps_priv_dep,
        "private_tps_target": PRIVATE_VERIFIED_TPS,
        "private_tps_resid": tps_priv_dep - PRIVATE_VERIFIED_TPS,
        "delta_pct_deployed": delta_pct_dep,
        "rho_deployed_tree_recovered": rho_deployed,
        "rho_raw_spine_on_ak": rho_raw_deployed,
        "raw_ladder_selfcheck_et": et_raw_self,
        "private_et_anchor": ET_PRIVATE_ANCHOR,
        "note": ("position-1 held (tree recovers the spine cliff); residual collapse c_deep={:.5f} on "
                 "j>=2 reproduces 460.85 (Delta {:.2%}). The raw width-1 spine multiplier reproduces "
                 "the 0.804 aggregate (rho_raw={:.4f}).".format(c_deep, delta_pct_dep, rho_raw_deployed)),
    }

    # ---- STEP 2: invert for the private-500-equivalent PUBLIC E[T]. -------- #
    # within the EAGLE-3 deep-flat family (the build-relevant shape), find public E[T] whose
    # deployed-effective-collapsed private E[T] = 3.99146 (private 500 TPS).
    def priv_et_of_public(pub_et: float) -> float:
        prof = profile_for_public_et(pub_et)
        return et_from_cond(apply_deployed_collapse(prof["cond"], c_deep))

    public_etbar_for_private_500 = _bisect(priv_et_of_public, 1.0, E_T_MAX, ET_PRIVATE_500)
    # the scalar-0.804 model's answer (what fern #281 used) for contrast.
    public_etbar_scalar_0804 = ET_PRIVATE_500 / PRIV_RAW_RATIO_BANKED
    private_tax_perpos = public_etbar_for_private_500 - ET_PRIVATE_500
    private_tax_scalar = public_etbar_scalar_0804 - ET_PRIVATE_500

    step2 = {
        "private_bar_tps": PRIVATE_BAR_TPS,
        "et_private_500_needed": ET_PRIVATE_500,
        "et_private_500_pr_formula": ET_PRIVATE_500_PR,
        "public_etbar_for_private_500": public_etbar_for_private_500,
        "public_etbar_scalar_0804": public_etbar_scalar_0804,
        "private_tax_perpos_et": private_tax_perpos,
        "private_tax_scalar_et": private_tax_scalar,
        "scalar_overprices_tax_by": (private_tax_scalar / private_tax_perpos
                                     if private_tax_perpos > 1e-9 else float("inf")),
        "fern281_public_500_target": PUBLIC_TARGETS["fern281"],
        "note": ("per-position deployed-effective collapse needs PUBLIC E[T]~={:.3f} for private-500 "
                 "(deep-flat family); the scalar-0.804 model demands {:.3f} (~fern #281's 4.966). The "
                 "scalar OVER-prices the private tax {:.1f}x for a deep-heavy drafter."
                 .format(public_etbar_for_private_500, public_etbar_scalar_0804,
                         private_tax_scalar / max(private_tax_perpos, 1e-9))),
    }

    # ---- STEP 3: EAGLE-3 lift + alignment test. ---------------------------- #
    e3_cond = deepflat_profile(EAGLE3_DEEP_FLAT_RATE)     # kanna #289 deep-flat (public 4.966)
    e3_pub_et = et_from_cond(e3_cond)
    e3_priv_dep = et_from_cond(apply_deployed_collapse(e3_cond, c_deep))
    e3_priv_raw = et_from_cond(apply_raw_collapse(e3_cond))
    rho_priv_e3 = e3_priv_dep / e3_pub_et                 # PRIMARY TEST metric (deployed-effective)
    rho_priv_e3_raw = e3_priv_raw / e3_pub_et             # binding sensitivity

    priced = {name: price_target(name, tgt, c_deep) for name, tgt in PUBLIC_TARGETS.items()}

    verdicts = {
        # PR-required booleans (primary = deployed-effective collapse).
        "deep_position_fidelity_closes_private_gap": bool(rho_priv_e3 > PRIV_RAW_RATIO_BANKED),
        "eagle3_public_4p9029_clears_private_bar": priced["wirbel290"]["clears_private_500_deployed"],
        "eagle3_public_6p1245_clears_private_bar": priced["wirbel293"]["clears_private_500_deployed"],
        # binding-sensitivity mirror (no tree recovery for EAGLE-3).
        "deep_position_fidelity_closes_private_gap_raw": bool(rho_priv_e3_raw > PRIV_RAW_RATIO_BANKED),
        "eagle3_public_4p9029_clears_private_bar_raw": priced["wirbel290"]["clears_private_500_raw"],
        "eagle3_public_6p1245_clears_private_bar_raw": priced["wirbel293"]["clears_private_500_raw"],
        # is the decisive build target robust across the bracket?
        "wirbel290_robust_across_collapse_bracket": bool(
            priced["wirbel290"]["clears_private_500_deployed"]
            and priced["wirbel290"]["clears_private_500_raw"]),
    }

    step3 = {
        "eagle3_deep_flat_rate": EAGLE3_DEEP_FLAT_RATE,
        "eagle3_a1_held": EAGLE3_A1,
        "eagle3_public_et": e3_pub_et,
        "eagle3_private_et_deployed": e3_priv_dep,
        "eagle3_private_et_raw": e3_priv_raw,
        "rho_priv_e3": rho_priv_e3,
        "rho_priv_e3_raw": rho_priv_e3_raw,
        "deployed_linear_rho_raw_0804": PRIV_RAW_RATIO_BANKED,
        "priced_targets": priced,
        "verdicts": verdicts,
        "note": ("EAGLE-3 deep-flat lift (a_1 held, j>=2->{:.4f}) gives rho_priv_e3={:.4f} under the "
                 "deployed-effective collapse (> 0.804: deep fidelity CLOSES the gap) but {:.4f} under "
                 "the raw/no-tree-recovery sensitivity (< 0.804: it does NOT). The decisive build "
                 "target wirbel #290's 4.9029 is a CLEAR under deployed-effective ({:.0f} TPS) and a "
                 "MISS under raw ({:.0f} TPS) -- private success hinges on tree-recovery inheritance."
                 .format(EAGLE3_DEEP_FLAT_RATE, rho_priv_e3, rho_priv_e3_raw,
                         priced["wirbel290"]["private_tps_deployed"],
                         priced["wirbel290"]["private_tps_raw"])),
    }

    return {
        "step1_anchor": step1,
        "step2_invert": step2,
        "step3_eagle3": step3,
        "test_metric": {"rho_priv_e3": rho_priv_e3},
        "imported": {
            "K_cal": K_CAL, "step_us": STEP_US, "tau": TAU, "official_public": OFFICIAL_PUBLIC,
            "ceiling_lambda1": CEILING_LAMBDA1, "E_T_anchor": E_T_ANCHOR, "K_spec": K_SPEC,
            "E_T_max": E_T_MAX, "public_ppl": PUBLIC_PPL, "private_verified_tps": PRIVATE_VERIFIED_TPS,
            "private_ppl": PRIVATE_PPL, "lambda_bar": LAMBDA_BAR, "private_bar_tps": PRIVATE_BAR_TPS,
            "ppl_bar": PPL_BAR, "priv_raw_ratio_banked": PRIV_RAW_RATIO_BANKED,
            "priv_raw_et_banked": PRIV_RAW_ET_BANKED, "eagle3_deep_flat_rate": EAGLE3_DEEP_FLAT_RATE,
            "public_targets": PUBLIC_TARGETS,
            "provenance": ("deployed a_k lawine #282 / kanna #289 / denken #297; raw spine ladders "
                           "ubel #258 (3.0898/3.8445=0.804); deployed private 460.85 organizer-"
                           "verified; EAGLE-3 deep_flat_rate kanna #289; public targets fern #281 "
                           "4.966 / wirbel #290 4.9029 / wirbel #293 6.1245."),
        },
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY): >=10 checks of the machinery.
# --------------------------------------------------------------------------- #
def self_test(syn: dict[str, Any]) -> dict[str, Any]:
    s1, s2, s3 = syn["step1_anchor"], syn["step2_invert"], syn["step3_eagle3"]
    c = {}

    # (1) composition round-trips to 481.53 on the imported anchor.
    c["01_composition_roundtrips_481"] = abs(s1["composition_anchor_tps"] - OFFICIAL_PUBLIC) <= 0.5

    # (2) deployed-effective model reproduces 460.85 (by calibration; tight).
    c["02_deployed_reproduces_460p85"] = abs(s1["private_tps_deployed"] - PRIVATE_VERIFIED_TPS) <= TOL_TPS

    # (3) the reproduced drop is the organizer-verified ~4.3%.
    c["03_delta_is_4p3_pct"] = abs(s1["delta_pct_deployed"] - PRIVATE_DELTA_PCT) <= 0.01

    # (4) per-position decomposition consistent with the 0.804 aggregate (raw spine on a_k).
    c["04_raw_reproduces_0804"] = abs(s1["rho_raw_spine_on_ak"] - PRIV_RAW_RATIO_BANKED) <= TOL_RATIO

    # (5) raw ladder reconstructs ubel #258's banked private E[T] (3.0898) exactly.
    c["05_raw_ladder_matches_ubel258"] = abs(s1["raw_ladder_selfcheck_et"] - PRIV_RAW_ET_BANKED) <= TOL_ET

    # (6) c_deep calibration self-consistency (round-trip of the bisection target).
    c["06_cdeep_calibration_consistent"] = abs(s1["private_et_deployed"] - ET_PRIVATE_ANCHOR) <= TOL_ET

    # (7) rho_priv_e3 in [0.804, 1.0].
    c["07_rho_priv_e3_in_band"] = PRIV_RAW_RATIO_BANKED - 1e-9 <= s3["rho_priv_e3"] <= 1.0 + 1e-9

    # (8) private E[T] <= public E[T] for every priced profile, both collapse models.
    le_ok = True
    for t in s3["priced_targets"].values():
        le_ok &= (t["private_et_deployed"] <= t["public_et"] + 1e-9)
        le_ok &= (t["private_et_raw"] <= t["public_et"] + 1e-9)
    le_ok &= (s3["eagle3_private_et_deployed"] <= s3["eagle3_public_et"] + 1e-9)
    c["08_private_le_public_all"] = bool(le_ok)

    # (9) every conditional (public & private, every profile) in [0,1]; survival monotone non-incr.
    band_ok, mono_ok = True, True
    profs = [A_K, A_PUB_RAW, A_PRIV_RAW, deepflat_profile(EAGLE3_DEEP_FLAT_RATE)]
    for t in s3["priced_targets"].values():
        profs += [t["cond_pub"], t["cond_priv_deployed"], t["cond_priv_raw"]]
    for cond in profs:
        band_ok &= all(0.0 <= x <= 1.0 for x in cond)
        surv = survival(cond)
        mono_ok &= all(surv[i] >= surv[i + 1] - 1e-12 for i in range(len(surv) - 1))
    c["09_conditionals_in_unit_and_survival_monotone"] = bool(band_ok and mono_ok)

    # (10) private-bar residuals are sign-consistent (E[T] vs TPS) for all targets, both models.
    sign_ok = True
    for t in s3["priced_targets"].values():
        sign_ok &= (math.copysign(1, t["private_et_deployed"] - ET_PRIVATE_500)
                    == math.copysign(1, t["private_tps_deployed"] - PRIVATE_BAR_TPS))
        sign_ok &= (math.copysign(1, t["private_et_raw"] - ET_PRIVATE_500)
                    == math.copysign(1, t["private_tps_raw"] - PRIVATE_BAR_TPS))
    c["10_private_bar_residuals_sign_consistent"] = bool(sign_ok)

    # (11) inversion round-trips: public_etbar_for_private_500 collapses back to private-500.
    prof = profile_for_public_et(s2["public_etbar_for_private_500"])
    priv_back = et_from_cond(apply_deployed_collapse(prof["cond"], s1["c_deep_calibrated"]))
    c["11_private500_inversion_roundtrips"] = abs(tps_at(priv_back) - PRIVATE_BAR_TPS) <= TOL_TPS

    # (12) EAGLE-3 deep-flat reproduces the kanna #289 public 4.966 cross-check.
    c["12_deepflat_reproduces_4966"] = abs(s3["eagle3_public_et"] - PUBLIC_TARGETS["fern281"]) <= 0.01

    # (13) constants imported exact & unchanged.
    c["13_constants_imported_exact"] = (
        K_CAL == 125.268 and OFFICIAL_PUBLIC == 481.53 and PRIVATE_VERIFIED_TPS == 460.85
        and K_SPEC == 7 and A_K[3] == 0.8228 and PRIV_RAW_RATIO_BANKED == 0.804
        and EAGLE3_DEEP_FLAT_RATE == 0.9144279172167558)

    gate = all(bool(v) for v in c.values())
    return {"private_bar_eagle3_self_test_passes": gate, "checks": c}


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def assert_nan_clean(payload: dict) -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, "result")
    return bad


# --------------------------------------------------------------------------- #
# W&B logging (robust; never fatal; honors no-key/disabled).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args, payload: dict) -> str | None:
    if getattr(args, "no_wandb", False) or not getattr(args, "wandb_name", None):
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[private-bar-eagle3] wandb logging unavailable: {exc}", flush=True)
        return None

    syn = payload["synthesis"]
    s1, s2, s3 = syn["step1_anchor"], syn["step2_invert"], syn["step3_eagle3"]
    st = payload["self_test"]
    v = s3["verdicts"]

    run = init_wandb_run(
        job_type="validity-analytic", agent="lawine", name=args.wandb_name, group=args.wandb_group,
        tags=["private-bar-eagle3", "validity-analytic", "private-bar", "eagle3", "acceptance",
              "per-position-collapse", "bank-the-analysis"],
        config={
            "pr": 300, "K_cal": K_CAL, "official_public": OFFICIAL_PUBLIC,
            "private_verified_tps": PRIVATE_VERIFIED_TPS, "private_bar_tps": PRIVATE_BAR_TPS,
            "priv_raw_ratio_banked": PRIV_RAW_RATIO_BANKED, "lambda_bar": LAMBDA_BAR,
            "eagle3_deep_flat_rate": EAGLE3_DEEP_FLAT_RATE, "public_targets": PUBLIC_TARGETS,
            "imports": syn["imported"]["provenance"], "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[private-bar-eagle3] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return None

    summary: dict[str, Any] = {
        "private_bar_eagle3_self_test_passes": int(bool(st["private_bar_eagle3_self_test_passes"])),
        "rho_priv_e3": s3["rho_priv_e3"],
        "rho_priv_e3_raw": s3["rho_priv_e3_raw"],
        "deployed_linear_rho_0804": PRIV_RAW_RATIO_BANKED,
        "c_deep_calibrated": s1["c_deep_calibrated"],
        "private_tps_deployed_anchor": s1["private_tps_deployed"],
        "private_tps_resid_vs_460p85": s1["private_tps_resid"],
        "delta_pct_deployed": s1["delta_pct_deployed"],
        "rho_deployed_tree_recovered": s1["rho_deployed_tree_recovered"],
        "rho_raw_spine_on_ak": s1["rho_raw_spine_on_ak"],
        "public_etbar_for_private_500": s2["public_etbar_for_private_500"],
        "public_etbar_scalar_0804": s2["public_etbar_scalar_0804"],
        "private_tax_perpos_et": s2["private_tax_perpos_et"],
        "private_tax_scalar_et": s2["private_tax_scalar_et"],
        "scalar_overprices_tax_by": s2["scalar_overprices_tax_by"],
        "eagle3_public_et": s3["eagle3_public_et"],
        "eagle3_private_et_deployed": s3["eagle3_private_et_deployed"],
        "eagle3_private_et_raw": s3["eagle3_private_et_raw"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"verdict_{k}": int(bool(val)) for k, val in v.items()},
        **{f"selftest_{k}": int(bool(val)) for k, val in st["checks"].items()},
    }
    for name, t in s3["priced_targets"].items():
        summary[f"tgt_{name}_public_et"] = t["public_et"]
        summary[f"tgt_{name}_private_et_deployed"] = t["private_et_deployed"]
        summary[f"tgt_{name}_private_tps_deployed"] = t["private_tps_deployed"]
        summary[f"tgt_{name}_private_tps_raw"] = t["private_tps_raw"]
        summary[f"tgt_{name}_clears_deployed"] = int(t["clears_private_500_deployed"])
        summary[f"tgt_{name}_clears_raw"] = int(t["clears_private_500_raw"])
        summary[f"tgt_{name}_a1_lifted"] = int(t["a1_lifted_above_073"])
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="private_bar_eagle3_result", artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    print(f"[private-bar-eagle3] wandb run: {getattr(run, 'url', rid)}", flush=True)
    finish_wandb(run)
    return rid


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def print_report(payload: dict) -> None:
    syn = payload["synthesis"]
    s1, s2, s3 = syn["step1_anchor"], syn["step2_invert"], syn["step3_eagle3"]
    st = payload["self_test"]
    v = s3["verdicts"]
    print("\n" + "=" * 100, flush=True)
    print("PRIVATE-BAR EAGLE-3 (PR #300) -- does a public-E[T] build clear PRIVATE >=500?", flush=True)
    print("=" * 100, flush=True)
    print("STEP 1 -- anchor on ground truth:", flush=True)
    print(f"  composition: K_cal*E[T]_anchor = {s1['composition_anchor_tps']:.2f} (= 481.53)", flush=True)
    print(f"  deployed-effective collapse: c_1=1.0 (held), c_deep={s1['c_deep_calibrated']:.5f} on j>=2",
          flush=True)
    print(f"  -> private E[T]={s1['private_et_deployed']:.4f}  private TPS={s1['private_tps_deployed']:.2f}"
          f"  (target 460.85, Delta {s1['delta_pct_deployed']:.2%})", flush=True)
    print(f"  rho_deployed(tree-recovered)={s1['rho_deployed_tree_recovered']:.4f}   "
          f"rho_raw(spine)={s1['rho_raw_spine_on_ak']:.4f} (= 0.804)", flush=True)
    print("-" * 100, flush=True)
    print("STEP 2 -- private-500-equivalent PUBLIC E[T]:", flush=True)
    print(f"  per-position deployed-effective: public_etbar_for_private_500 = "
          f"{s2['public_etbar_for_private_500']:.4f}", flush=True)
    print(f"  scalar-0.804 model demands: {s2['public_etbar_scalar_0804']:.4f} (~fern #281 4.966)",
          flush=True)
    print(f"  private tax: per-position {s2['private_tax_perpos_et']:.4f} E[T] vs scalar "
          f"{s2['private_tax_scalar_et']:.4f} E[T]  (scalar over-prices {s2['scalar_overprices_tax_by']:.1f}x)",
          flush=True)
    print("-" * 100, flush=True)
    print("STEP 3 -- EAGLE-3 lift + alignment:", flush=True)
    print(f"  rho_priv_e3 = {s3['rho_priv_e3']:.4f} (deployed-eff)   {s3['rho_priv_e3_raw']:.4f} (raw)"
          f"   vs deployed 0.804", flush=True)
    print("  target     | public | priv_et(dep) | priv_TPS(dep) | clears(dep) | priv_TPS(raw) | clears(raw) | a1-lift",
          flush=True)
    for name in ("fern281", "wirbel290", "wirbel293"):
        t = s3["priced_targets"][name]
        print(f"  {name:<10} | {t['public_target']:.4f} |   {t['private_et_deployed']:.4f}    |"
              f"   {t['private_tps_deployed']:7.2f}    |    {str(t['clears_private_500_deployed']):<5}    |"
              f"   {t['private_tps_raw']:7.2f}    |    {str(t['clears_private_500_raw']):<5}    | "
              f"{t['a1_lifted_above_073']}", flush=True)
    print("-" * 100, flush=True)
    print("VERDICTS:", flush=True)
    for k, val in v.items():
        print(f"  {k}: {val}", flush=True)
    print("-" * 100, flush=True)
    print(f"(PRIMARY) private_bar_eagle3_self_test_passes = {st['private_bar_eagle3_self_test_passes']}",
          flush=True)
    for k, val in st["checks"].items():
        print(f"   - {k}: {val}", flush=True)
    print(f"nan_clean = {payload['nan_clean']}   peak_mem_mib = {payload['peak_mem_mib']}", flush=True)
    print("=" * 100 + "\n", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="private-bar-eagle3")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    syn = synthesize()
    st = self_test(syn)

    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "pr": 300, "agent": "lawine", "kind": "private-bar-eagle3",
        "private_bar_eagle3_analysis_only": True,
        "synthesis": syn, "self_test": st,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[private-bar-eagle3] WARNING non-finite at: {nan_paths}", flush=True)
    # fold nan-clean into the gate.
    gate = bool(st["private_bar_eagle3_self_test_passes"] and payload["nan_clean"])
    st["private_bar_eagle3_self_test_passes"] = gate

    print_report(payload)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "private_bar_eagle3_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[private-bar-eagle3] wrote {out_path}", flush=True)

    rid = maybe_log_wandb(args, payload)
    payload["wandb_run_id"] = rid
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print(f"  PRIMARY private_bar_eagle3_self_test_passes = {gate}", flush=True)
    print(f"  TEST rho_priv_e3 = {syn['step3_eagle3']['rho_priv_e3']:.4f}", flush=True)
    print(f"  public_etbar_for_private_500 = {syn['step2_invert']['public_etbar_for_private_500']:.4f}",
          flush=True)
    print(f"  wandb run = {rid}", flush=True)

    if args.self_test:
        print(f"[private-bar-eagle3] self-test {'PASS' if gate else 'FAIL'}", flush=True)
        return 0 if gate else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
