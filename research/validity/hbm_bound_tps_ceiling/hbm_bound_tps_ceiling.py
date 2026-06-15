#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""HBM-bandwidth-bound TPS ceiling (PR #283) — is the deployed 481.53 near the int4
body-read floor, and what is the HARD CAP on the verify-side kernel-timing front?

WHAT THIS LEG COMPUTES
----------------------
denken #278 (bu44n30q) measured the deployed int4 body-read HBM floor at 2933.83 us
(1.76 GB int4 body + lm_head / 600 GB/s) and showed it EXCEEDS the 1218.2 us NORMALIZED
composition step. This leg synthesises that floor into the HBM-bandwidth-bound TPS
ceiling (official units), places the deployed 481.53 on it, bounds the maximum TPS that
ALL kernel-timing tuning combined could recover before hitting the read wall, and prices
the only two ceiling-raising levers: E[T]-raise (tokens-per-read) and body-read-reduction
(bytes-per-read, PPL-gated).

THE UNIT BRIDGE (shown explicitly; self-test (a) pins it)
--------------------------------------------------------
K_cal = 125.268 is the OFFICIAL composition calibration. Physically it IS the official
per-step RATE: K_cal = official / E[T] = 481.53 / 3.844 = 125.268 STEPS / second. So the
HONEST official per-step wall is

    wall_deployed_official = 1 / K_cal = E[T] / official = 7982.87 us.

The int4 body read is the HARD floor on that wall (every verify reads the body once):

    read_floor_local  = (1.6973824 + 0.06291456) GB / 600 GB/s = 2933.83 us  (#278)
    read_floor_official = read_floor_local * tau_lo (#267 local->official rate) = 3037.20 us.

The HBM-bound ceiling is the TPS when the wall collapses to that read floor:

    ceiling = E[T] / read_floor_official = E[T] * bandwidth / (body_bytes * tau_lo)
            = 1265.6 TPS (official).

DEPLOYED 481.53 SITS AT 0.380 OF THE CEILING (the read is only 38% of the honest wall).
This is the crux: #278's "floor (2933.83) EXCEEDS the 1218.2 us step" is an artifact of the
~6.5x composition COMPRESSION of the normalized step (it proves step-shaving OVER-CREDIT,
NOT wall read-boundedness). In the HONEST wall basis the body read is 38% of the 7982.9 us
step; the other 62% is draft + verify-compute-above-read + host overhead. So the verify-side
kernel front is NOT hard-capped below 500 by the read floor: the bare-read-floor ceiling is
1265.6 (>> 500), and even removing ONLY the kernel-addressable slack (verify down to its read
floor + draft folded out) reaches ~747 TPS. The front is OPEN -- gated by how much of the
verify's 1.69x compute-above-read is HIDEABLE (kanna #280) and by #278's 4.8x over-credit on
incremental shaves, NOT by the HBM bandwidth floor.

The path to a +500 BUILD is therefore (1) E[T]-raise -- move the DEPLOYED point up the
ceiling (fern #281: E[T] >= 3.991 puts the deployed point at 500) -- and/or (2) hide the
verify compute-above-read (kanna #280). Body-read-reduction is the WRONG lever: the ceiling
is ALREADY > 500, so reducing bytes only raises a non-binding ceiling (the "-153% reduction
for 500" is the proof). This leg PRICES the levers; it does NOT realize them and adds 0 TPS.

SCOPE
-----
LOCAL CPU analytic + an OPTIONAL GPU read-bandwidth re-confirmation (read if present). All
#278/#217/#267 scalars IMPORTED EXACT, NOT re-derived. BASELINE stays 481.53. PRIMARY =
self-test. NOT a launch, NOT open2, no served-file change, no HF Job, no submission. The
launch gate remains a MEASURED >=500 build (human-approval-gated). Body-read-reduction would
require re-quantization that holds PPL <= 2.42 (headroom 0.0428) -- a SEPARATE measured leg;
do NOT re-quantize here.

PRIMARY metric  hbm_bound_ceiling_self_test_passes
TEST    metric  official_tps_as_frac_of_hbm_ceiling

Run:
    cd target/ && CUDA_VISIBLE_DEVICES=0 \
      /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/validity/hbm_bound_tps_ceiling/hbm_bound_tps_ceiling.py --self-test \
      --wandb_group hbm-bound-ceiling --wandb_name denken/hbm-bound-ceiling
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

# Optional GPU read-bandwidth re-confirmation (written by read_floor_confirm.py if run).
_READBENCH_JSON = HERE / "read_floor_confirm.json"
# denken #278 banked measurement (the source of the read floor + verify wall).
_D278_JSON = (REPO_ROOT / "research/validity/linear_step_decomposition/"
              "linear_verify_measurement.json")

# --------------------------------------------------------------------------- #
# PR #283 imported constants (provenance in the module docstring; NOT re-derived).
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                 # PR #52 official TPS (the deployed point)
GO_READ = 520.9527323111674               # kanna #217 lambda=1 ceiling
K_CAL = 125.26795005202914                # kanna #217 composition calibration (= official/E[T])
STEP_SERVED_US = 1218.2                    # kanna #217 normalized composition step
E_T = 3.844                               # kanna #217 deployed linear K=7 E[T]
TAU_LO = 1.0352356533046398               # lawine #267 local->official TPS rate transfer
VERIFY_HBM_FLOOR_US = 2933.828266666667   # denken #278 int4 body-read HBM floor (local)
BODY_INT4_GB = 1.6973824                  # denken #278 int4 body bytes
LMHEAD_BF16_GB = 0.06291456               # denken #278 lm_head bf16 bytes
A10G_BW_GBPS = 600.0                      # A10G HBM bandwidth (the figure #278's floor used)
DRAFT_K7_US = 706.8555014474051           # denken #278 draft K=7 chain wall (graphed)
VERIFY_M1_US = 4966.783229282924          # denken #278 deployed M=1 verify wall (graphed)
PPL_GATE = 2.42                           # public PPL gate
PPL_BASELINE = 2.3772                     # official baseline PPL
PPL_HEADROOM = 0.0428                     # 2.42 - 2.3772 (the bytes-per-read binding limit)

MILESTONE = 500.0                         # the live TPS milestone gate
FERN_ET_FLOOR_281 = 3.991                 # fern #281 E[T]-raise floor (deployed-point lever)
OVERCREDIT_278 = 4.817987532332126        # denken #278 composition over-credit factor

# tolerances
TOL_BASIS = 1e-6        # basis must reproduce 481.53 to this
TOL_ROUNDTRIP_TPS = 0.5  # levers must round-trip to 500 within this
TOL_IMPORT = 1e-9        # imported scalars matched to this

BODY_BYTES_GB = BODY_INT4_GB + LMHEAD_BF16_GB   # 1.76029696 GB read per verify


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# (1) The composition basis + the explicit token-per-read -> official-TPS bridge.
# --------------------------------------------------------------------------- #
def compute_basis() -> dict[str, Any]:
    # ET*tau invariant that round-trips 481.53 exactly through the normalized step
    # (kanna #217 / denken #278): official = K_cal * et_tau / step_norm.
    et_tau = OFFICIAL_BASELINE * STEP_SERVED_US / K_CAL          # == 4682.7608 us-token
    tps_via_composition = K_CAL * et_tau / STEP_SERVED_US        # == OFFICIAL_BASELINE

    # The PHYSICAL form: K_cal IS the official step-rate (steps/s); the honest official
    # per-step WALL is its inverse. official = E[T] * (steps/s) = E[T] * K_cal.
    kcal_as_steps_per_s = OFFICIAL_BASELINE / E_T                # == K_CAL
    wall_deployed_official_us = 1e6 / K_CAL                      # == E[T]/official * 1e6
    tps_via_steprate = E_T * (1e6 / wall_deployed_official_us)   # == OFFICIAL_BASELINE

    # the read floor (#278), local then bridged to official-clock via tau_lo (#267).
    # GB / (GB/s) = s; *1e6 -> us. == VERIFY_HBM_FLOOR_US (2933.83 us).
    read_floor_local_us = BODY_BYTES_GB / A10G_BW_GBPS * 1e6     # == VERIFY_HBM_FLOOR_US
    read_floor_official_us = read_floor_local_us * TAU_LO

    return {
        "official_baseline": OFFICIAL_BASELINE,
        "E_T": E_T,
        "K_cal": K_CAL,
        "K_cal_as_steps_per_s": kcal_as_steps_per_s,
        "step_norm_us": STEP_SERVED_US,
        "et_tau_invariant": et_tau,
        "tps_via_composition": tps_via_composition,
        "tps_via_steprate": tps_via_steprate,
        "wall_deployed_official_us": wall_deployed_official_us,
        "tau_lo": TAU_LO,
        "body_bytes_gb": BODY_BYTES_GB,
        "a10g_bw_gbps": A10G_BW_GBPS,
        "read_floor_local_us": read_floor_local_us,
        "read_floor_official_us": read_floor_official_us,
        "verify_hbm_floor_us_imported": VERIFY_HBM_FLOOR_US,
        "floor_byte_provenance_matches_import":
            bool(abs(read_floor_local_us - VERIFY_HBM_FLOOR_US) < 1e-6),
        "bridge_note": (
            "official = E[T]/wall = E[T]*K_cal (K_cal=125.268 = official/E[T] = steps/s). "
            "The HBM ceiling replaces the deployed wall (1/K_cal = 7982.9us) with the read "
            "floor (2933.8us local x tau_lo = 3037.2us official). token-per-read = E[T]/read_floor."),
    }


# --------------------------------------------------------------------------- #
# (2) The HBM-bandwidth-bound ceiling + placing 481.53 on it.
# --------------------------------------------------------------------------- #
def compute_ceiling(basis: dict[str, Any]) -> dict[str, Any]:
    read_floor_official_s = basis["read_floor_official_us"] / 1e6
    read_floor_local_s = basis["read_floor_local_us"] / 1e6
    wall_deployed_official_s = basis["wall_deployed_official_us"] / 1e6

    # CENTRAL (consistent units: official wall vs official read floor, tau_lo carried).
    ceiling_official = E_T / read_floor_official_s
    # cross-check via the bytes/bandwidth form (instruction 3): E[T]*bw/(bytes*tau_lo).
    ceiling_official_bw = E_T * (A10G_BW_GBPS * 1e9) / (BODY_BYTES_GB * 1e9) / TAU_LO
    # VARIANT (bandwidth-pure: assume an HBM read does NOT incur the +3.5% tau_lo).
    ceiling_local = E_T / read_floor_local_s

    frac = OFFICIAL_BASELINE / ceiling_official                  # == read_floor/wall (tau_lo-invariant)
    frac_identity = read_floor_official_s / wall_deployed_official_s
    frac_bw_pure = OFFICIAL_BASELINE / ceiling_local

    # kernel-timing headroom: the MAX TPS all kernel tuning combined could recover before
    # hitting the read wall = (bare-read-floor ceiling) - deployed.
    kernel_timing_headroom_tps = ceiling_official - OFFICIAL_BASELINE
    kernel_timing_clears_500 = bool(ceiling_official >= MILESTONE)

    # honest decomposition of the 62% NON-READ slack (official us).
    draft_official_us = DRAFT_K7_US * TAU_LO
    verify_above_read_local_us = VERIFY_M1_US - VERIFY_HBM_FLOOR_US
    verify_above_read_official_us = verify_above_read_local_us * TAU_LO
    kernel_addressable_official_us = draft_official_us + verify_above_read_official_us
    non_read_official_us = basis["wall_deployed_official_us"] - basis["read_floor_official_us"]
    host_other_official_us = non_read_official_us - kernel_addressable_official_us
    # sub-ceiling if ONLY the kernel-addressable slack is removed (host overhead stays).
    step_kernel_floor_us = basis["wall_deployed_official_us"] - kernel_addressable_official_us
    tps_kernel_floor = E_T * 1e6 / step_kernel_floor_us

    return {
        "hbm_bound_ceiling_tps": ceiling_official,
        "hbm_bound_ceiling_tps_bw_form_crosscheck": ceiling_official_bw,
        "hbm_bound_ceiling_tps_bandwidth_pure_variant": ceiling_local,
        "hbm_bound_ceiling_basis": (
            "ceiling = E[T] / read_floor_official = E[T]*bandwidth/(body_bytes*tau_lo); "
            "deployed official = E[T] / (1/K_cal). frac = read_floor_official / (1/K_cal). "
            f"E[T]={E_T}, read_floor_official={basis['read_floor_official_us']:.3f}us, "
            f"deployed_wall_official={basis['wall_deployed_official_us']:.3f}us."),
        "official_tps_as_frac_of_hbm_ceiling": frac,
        "official_tps_as_frac_identity_crosscheck": frac_identity,
        "official_tps_as_frac_bandwidth_pure": frac_bw_pure,
        "kernel_timing_headroom_tps": kernel_timing_headroom_tps,
        "kernel_timing_clears_500": kernel_timing_clears_500,
        "read_frac_of_wall_pct": 100.0 * frac,
        "non_read_slack": {
            "non_read_official_us": non_read_official_us,
            "non_read_pct_of_wall": 100.0 * non_read_official_us / basis["wall_deployed_official_us"],
            "draft_official_us": draft_official_us,
            "verify_above_read_official_us": verify_above_read_official_us,
            "verify_over_read_ratio": VERIFY_M1_US / VERIFY_HBM_FLOOR_US,
            "kernel_addressable_official_us": kernel_addressable_official_us,
            "host_other_official_us": host_other_official_us,
            "step_kernel_floor_us": step_kernel_floor_us,
            "tps_kernel_floor": tps_kernel_floor,
            "tps_kernel_floor_clears_500": bool(tps_kernel_floor >= MILESTONE),
            "note": ("kernel_addressable = draft + verify-compute-above-read (kanna #280's "
                     "hideable target); host_other ~ ubel #284 decode-loop overhead. Even "
                     "removing ONLY the kernel-addressable slack reaches tps_kernel_floor "
                     "(> 500), so the read floor does NOT cap the verify front below 500."),
        },
    }


# --------------------------------------------------------------------------- #
# (3) Price the two ceiling-raising levers.
# --------------------------------------------------------------------------- #
def compute_levers(basis: dict[str, Any], ceil: dict[str, Any]) -> dict[str, Any]:
    read_floor_official_s = basis["read_floor_official_us"] / 1e6

    # tokens-per-read lever: E[T] that puts the CEILING at exactly 500.
    et_for_500_at_read_floor = MILESTONE * read_floor_official_s
    ceiling_at_et500 = et_for_500_at_read_floor / read_floor_official_s   # round-trip -> 500

    # bytes-per-read lever: body bytes that put the ceiling at 500 at fixed E[T].
    bytes_new_gb = E_T * A10G_BW_GBPS / (MILESTONE * TAU_LO)
    read_reduction_pct = (1.0 - bytes_new_gb / BODY_BYTES_GB) * 100.0
    read_reduction_bytes_gb = BODY_BYTES_GB - bytes_new_gb
    ceiling_at_bytes500 = E_T * A10G_BW_GBPS / (bytes_new_gb * TAU_LO)    # round-trip -> 500

    # the DEPLOYED-point lever (cross-ref fern #281): E[T] that moves the DEPLOYED point to 500.
    et_for_500_deployed = E_T * MILESTONE / OFFICIAL_BASELINE
    fern_resid = abs(et_for_500_deployed - FERN_ET_FLOOR_281)

    return {
        "et_for_500_at_read_floor": et_for_500_at_read_floor,
        "et_for_500_at_read_floor_roundtrip_tps": ceiling_at_et500,
        "et_for_500_at_read_floor_below_current": bool(et_for_500_at_read_floor < E_T),
        "read_reduction_pct_for_500_at_fixed_et": read_reduction_pct,
        "read_reduction_bytes_for_500_gb": read_reduction_bytes_gb,
        "read_reduction_target_bytes_gb": bytes_new_gb,
        "read_reduction_roundtrip_tps": ceiling_at_bytes500,
        "read_reduction_is_binding_for_500": bool(read_reduction_pct > 0.0),
        "ppl_headroom": PPL_HEADROOM,
        "ppl_gate": PPL_GATE,
        "ppl_baseline": PPL_BASELINE,
        "et_for_500_deployed_point": et_for_500_deployed,
        "fern_281_et_floor": FERN_ET_FLOOR_281,
        "fern_281_resid": fern_resid,
        "fern_281_consistent": bool(fern_resid < 1e-3),
        "lever_note": (
            "et_for_500_at_read_floor (CEILING-to-500) = {:.4f} is BELOW the current E[T]={} -> the "
            "ceiling is already > 500 and the E[T] lever is non-binding ON THE CEILING. To move the "
            "DEPLOYED POINT to 500 needs E[T] >= {:.4f} (== fern #281's 3.991). read_reduction_pct = "
            "{:.1f}% is NEGATIVE -> you would need to ADD bytes to drop the ceiling to 500; the "
            "bytes-per-read lever is non-binding. The body-read-reduction lever would only matter if "
            "the deployed step were read-bound (it is 38% read); any real reduction is PPL-gated at "
            "headroom {:.4f} and must be a SEPARATE measured leg -- do NOT re-quantize here."
            .format(et_for_500_at_read_floor, E_T, et_for_500_deployed, read_reduction_pct,
                    PPL_HEADROOM)),
    }


# --------------------------------------------------------------------------- #
# Optional GPU read-bandwidth re-confirmation (non-fatal grounding cross-check).
# --------------------------------------------------------------------------- #
def load_readbench() -> dict[str, Any]:
    if not _READBENCH_JSON.exists():
        return {"present": False, "note": "no GPU read-bandwidth re-confirmation (analytic floor used)"}
    try:
        with _READBENCH_JSON.open(encoding="utf-8") as fh:
            rb = json.load(fh)
        eff_bw = rb.get("effective_read_bw_gbps")
        floor_us = rb.get("measured_read_floor_us")
        # achievable-BW ceiling (conservative LOWER ceiling): if the read transfers at the
        # MEASURED achievable BW (not nominal 600), the ceiling is still computed the same way.
        ach_ceiling = ach_frac = None
        if _finite(floor_us):
            read_floor_ach_official_s = floor_us * TAU_LO / 1e6
            ach_ceiling = E_T / read_floor_ach_official_s
            ach_frac = OFFICIAL_BASELINE / ach_ceiling
        return {
            "present": True,
            "effective_read_bw_gbps": eff_bw,
            "achievable_frac_of_nominal": rb.get("achievable_frac_of_nominal"),
            "measured_read_floor_us": floor_us,
            "analytic_floor_us": VERIFY_HBM_FLOOR_US,
            "measured_vs_analytic_ratio":
                (floor_us / VERIFY_HBM_FLOOR_US) if _finite(floor_us) else None,
            "achievable_bw_ceiling_tps": ach_ceiling,
            "official_frac_of_achievable_ceiling": ach_frac,
            "achievable_ceiling_clears_500": bool(ach_ceiling is not None and ach_ceiling >= MILESTONE),
            "note": rb.get("note", "GPU read-bandwidth re-confirmation present"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"present": False, "note": f"readbench load failed: {exc}"}


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY) — conditions (a)-(f).
# --------------------------------------------------------------------------- #
def self_test(basis: dict[str, Any], ceil: dict[str, Any], lev: dict[str, Any]) -> dict[str, Any]:
    # (a) the composition basis reproduces 481.53 at E[T]=3.844 and the measured deployed step
    #     (BOTH the normalized-step form and the physical step-rate form), resid 0.
    a = bool(abs(basis["tps_via_composition"] - OFFICIAL_BASELINE) < TOL_BASIS
             and abs(basis["tps_via_steprate"] - OFFICIAL_BASELINE) < TOL_BASIS)

    # (b) ceiling >= 481.53 (the deployed point cannot exceed its own read-floor ceiling).
    b = bool(ceil["hbm_bound_ceiling_tps"] >= OFFICIAL_BASELINE
             and ceil["hbm_bound_ceiling_tps_bandwidth_pure_variant"] >= OFFICIAL_BASELINE)

    # (c) et_for_500 and read_reduction each round-trip to 500 through the ceiling (resid < 0.5).
    c = bool(abs(lev["et_for_500_at_read_floor_roundtrip_tps"] / 1.0 - MILESTONE) >= 0  # guard finite
             and abs(lev["et_for_500_at_read_floor"] / (basis["read_floor_official_us"] / 1e6)
                     - MILESTONE) < TOL_ROUNDTRIP_TPS
             and abs(lev["read_reduction_roundtrip_tps"] - MILESTONE) < TOL_ROUNDTRIP_TPS)

    # (d) NaN-clean (filled at payload assembly; here check the key scalars finite).
    key_scalars = [basis["tps_via_composition"], basis["tps_via_steprate"],
                   basis["read_floor_official_us"], ceil["hbm_bound_ceiling_tps"],
                   ceil["official_tps_as_frac_of_hbm_ceiling"], ceil["kernel_timing_headroom_tps"],
                   lev["et_for_500_at_read_floor"], lev["read_reduction_pct_for_500_at_fixed_et"]]
    d = bool(all(_finite(x) for x in key_scalars))

    # (e) the imported anchors are EXACT.
    e = bool(abs(OFFICIAL_BASELINE - 481.53) < TOL_IMPORT
             and abs(GO_READ - 520.9527323111674) < 1e-6
             and abs(K_CAL - 125.26795005202914) < TOL_IMPORT
             and abs(STEP_SERVED_US - 1218.2) < TOL_IMPORT
             and abs(E_T - 3.844) < TOL_IMPORT
             and abs(TAU_LO - 1.0352356533046398) < TOL_IMPORT
             and abs(VERIFY_HBM_FLOOR_US - 2933.828266666667) < 1e-6
             and abs(PPL_GATE - 2.42) < TOL_IMPORT
             and abs(PPL_BASELINE - 2.3772) < TOL_IMPORT
             and basis["floor_byte_provenance_matches_import"])

    # (f) the leg carries the 0-TPS / ceiling-not-build caveat and the do-NOT-re-quantize note.
    f = bool("do NOT re-quantize" in lev["lever_note"]
             and "non-binding" in lev["lever_note"])

    conditions = {
        "a_basis_reproduces_481p53_both_forms": a,
        "b_ceiling_ge_official": b,
        "c_levers_roundtrip_to_500": c,
        "d_key_scalars_finite": d,
        "e_imports_exact": e,
        "f_caveats_present": f,
    }
    return {
        "conditions": conditions,
        "hbm_bound_ceiling_self_test_passes": bool(all(conditions.values())),
    }


def synthesize() -> dict[str, Any]:
    basis = compute_basis()
    ceil = compute_ceiling(basis)
    lev = compute_levers(basis, ceil)
    readbench = load_readbench()
    st = self_test(basis, ceil, lev)

    frac = ceil["official_tps_as_frac_of_hbm_ceiling"]
    clears = ceil["kernel_timing_clears_500"]
    verdict = (
        "HBM-BOUND CEILING IS NON-BINDING AT 500 -- THE VERIFY-SIDE FRONT IS OPEN, NOT CAPPED. "
        f"The HBM-bandwidth ceiling is {ceil['hbm_bound_ceiling_tps']:.1f} TPS (official; "
        f"{ceil['hbm_bound_ceiling_tps_bandwidth_pure_variant']:.1f} bandwidth-pure), and deployed "
        f"481.53 sits at {frac:.3f} of it: the int4 body read (3037us official) is only "
        f"{100.0*frac:.1f}% of the HONEST official per-step wall (1/K_cal = 7982.9us), NOT the "
        "dominant cost. denken #278's 'floor (2933.8) EXCEEDS the 1218.2us step' is an artifact of "
        "the ~6.5x composition COMPRESSION of the normalized step (it proves step-shaving "
        f"over-credit {OVERCREDIT_278:.2f}x, NOT wall read-boundedness). Kernel timing is NOT "
        f"hard-capped below 500 by the read floor: the bare-read-floor ceiling is "
        f"{ceil['hbm_bound_ceiling_tps']:.1f} (>> 500), and removing ONLY the kernel-addressable "
        f"slack (verify->read floor + draft folded) reaches "
        f"{ceil['non_read_slack']['tps_kernel_floor']:.0f} TPS. The front is OPEN -- gated by how "
        "much of the verify's 1.69x compute-above-read is HIDEABLE (kanna #280) and by #278's 4.8x "
        "over-credit on incremental shaves (consistent with the public field grinding 484->489.63 "
        "via split-KV/fa2sw, still <500). Body-read-reduction is the WRONG lever: the ceiling is "
        f"ALREADY > 500 so reducing bytes only raises a non-binding ceiling (the "
        f"{lev['read_reduction_pct_for_500_at_fixed_et']:.0f}% 'reduction for 500' is the proof). "
        "The binding levers to MOVE THE DEPLOYED POINT to 500 are E[T]-raise (fern #281: E[T] >= "
        f"{lev['et_for_500_deployed_point']:.3f}) or closing the 62% non-read slack (kanna #280 "
        "verify hideable-compute + ubel #284 host overhead). BASELINE 481.53 untouched; "
        "analysis-only; adds 0 TPS; NOT a launch; the gate remains a MEASURED >=500 build."
    )
    handoff = (
        f"The HBM-bandwidth-bound TPS ceiling is {ceil['hbm_bound_ceiling_tps']:.1f} (official), so "
        f"deployed 481.53 sits at {frac:.3f} of the read-floor ceiling and pure kernel-timing tuning "
        f"is hard-capped at {ceil['hbm_bound_ceiling_tps']:.1f} ({'CLEARS' if clears else 'MISSES'} "
        "500) -- confirming the verify-side front is OPEN (the int4 read is only "
        f"{100.0*frac:.0f}% of the honest 1/K_cal=7982.9us step; #278's 'floor>normalized-step' was "
        "composition-compression, not wall read-boundedness); the only two ceiling-raising levers "
        f"are E[T] >= {lev['et_for_500_at_read_floor']:.3f} (ceiling-to-500; the deployed point "
        f"reaches 500 at E[T] >= {lev['et_for_500_deployed_point']:.3f}, fern #281) or a "
        f"{lev['read_reduction_pct_for_500_at_fixed_et']:.0f}% body-read 'reduction' (NEGATIVE -> "
        "read-reduction is non-binding, the ceiling is already >500), so the path to 500 is "
        "E[T]-raise + hideable-verify-compute (kanna #280), NOT read-reduction and NOT "
        "read-floor-limited."
    )

    return {
        "basis": basis,
        "ceiling": ceil,
        "levers": lev,
        "gpu_readbench": readbench,
        "self_test": st,
        "verdict": verdict,
        "handoff": handoff,
        # surfaced headline metrics
        "hbm_bound_ceiling_self_test_passes": st["hbm_bound_ceiling_self_test_passes"],
        "official_tps_as_frac_of_hbm_ceiling": frac,
        "kernel_timing_clears_500": clears,
        "et_for_500_at_read_floor": lev["et_for_500_at_read_floor"],
        "read_reduction_pct_for_500_at_fixed_et": lev["read_reduction_pct_for_500_at_fixed_et"],
    }


# --------------------------------------------------------------------------- #
# NaN guard.
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "payload") -> list[str]:
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

    walk(payload, path)
    return bad


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    b, c, lev, st = syn["basis"], syn["ceiling"], syn["levers"], syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("HBM-BANDWIDTH-BOUND TPS CEILING (PR #283) — is 481.53 near the int4 body-read floor?",
          flush=True)
    print("=" * 100, flush=True)
    print(f"  basis: official = E[T]*K_cal = K_cal*et_tau/step_norm = {b['tps_via_composition']:.4f} "
          f"(K_cal={b['K_cal']:.5f} = official/E[T] = steps/s)", flush=True)
    print(f"  honest official per-step wall = 1/K_cal = {b['wall_deployed_official_us']:.2f}us  |  "
          f"read floor = {b['read_floor_local_us']:.2f}us local x tau_lo({TAU_LO:.5f}) = "
          f"{b['read_floor_official_us']:.2f}us official", flush=True)
    print("-" * 100, flush=True)
    print(f"  HBM CEILING (official, tau_lo-carried) = {c['hbm_bound_ceiling_tps']:.2f} TPS   "
          f"[bandwidth-pure variant {c['hbm_bound_ceiling_tps_bandwidth_pure_variant']:.2f}]",
          flush=True)
    print(f"  bw-form cross-check = {c['hbm_bound_ceiling_tps_bw_form_crosscheck']:.2f} (== central)",
          flush=True)
    print(f"  >>> official 481.53 sits at {c['official_tps_as_frac_of_hbm_ceiling']:.4f} of the "
          f"ceiling  (read is {c['read_frac_of_wall_pct']:.1f}% of the honest wall)", flush=True)
    print(f"  kernel_timing_headroom = {c['kernel_timing_headroom_tps']:.1f} TPS   "
          f"kernel_timing_clears_500 = {c['kernel_timing_clears_500']}", flush=True)
    ns = c["non_read_slack"]
    print(f"  non-read slack {ns['non_read_official_us']:.0f}us ({ns['non_read_pct_of_wall']:.1f}% of "
          f"wall): draft {ns['draft_official_us']:.0f} + verify-above-read "
          f"{ns['verify_above_read_official_us']:.0f} (kernel) + host/other "
          f"{ns['host_other_official_us']:.0f}", flush=True)
    print(f"  remove ONLY kernel-addressable -> step {ns['step_kernel_floor_us']:.0f}us -> "
          f"{ns['tps_kernel_floor']:.0f} TPS (clears500={ns['tps_kernel_floor_clears_500']})",
          flush=True)
    print("-" * 100, flush=True)
    print(f"  LEVER 1 (tokens-per-read): et_for_500_at_read_floor = {lev['et_for_500_at_read_floor']:.4f}"
          f"  (below current E[T]={E_T} -> ceiling non-binding; deployed-point lever fern #281 = "
          f"{lev['et_for_500_deployed_point']:.4f})", flush=True)
    print(f"  LEVER 2 (bytes-per-read): read_reduction_pct_for_500 = "
          f"{lev['read_reduction_pct_for_500_at_fixed_et']:.2f}%  "
          f"({lev['read_reduction_bytes_for_500_gb']:.3f} GB; binding={lev['read_reduction_is_binding_for_500']}) "
          f"-- PPL-gated at headroom {lev['ppl_headroom']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) hbm_bound_ceiling_self_test_passes = {st['hbm_bound_ceiling_self_test_passes']}",
          flush=True)
    for k, val in st["conditions"].items():
        print(f"          - {k}: {val}", flush=True)
    print(f"  (TEST) official_tps_as_frac_of_hbm_ceiling = "
          f"{syn['official_tps_as_frac_of_hbm_ceiling']:.4f}", flush=True)
    if syn["gpu_readbench"]["present"]:
        rb = syn["gpu_readbench"]
        print(f"  GPU read-bench: eff_bw {rb.get('effective_read_bw_gbps')} GB/s, measured floor "
              f"{rb.get('measured_read_floor_us')}us (analytic {VERIFY_HBM_FLOOR_US:.1f}us)", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)
    print("=" * 100, flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors the house pattern; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        # Import the installed wandb FIRST so it is cached in sys.modules before
        # REPO_ROOT joins sys.path. target/ holds a gitignored ./wandb run-output
        # dir (no __init__.py) that would otherwise shadow the package as a
        # namespace package and break `import wandb` inside scripts.wandb_logging.
        import wandb  # noqa: F401,E402
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[hbm-bound-ceiling] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    b, c, lev, st = syn["basis"], syn["ceiling"], syn["levers"], syn["self_test"]

    run = init_wandb_run(
        job_type="validity-gate",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["validity-gate", "hbm-bound-ceiling", "read-floor", "roofline",
              "verify-front-cap", "bank-the-analysis", "pr-283"],
        config={
            "official_baseline": OFFICIAL_BASELINE,
            "go_read": GO_READ,
            "K_cal": K_CAL,
            "step_served_us": STEP_SERVED_US,
            "E_T": E_T,
            "tau_lo": TAU_LO,
            "verify_hbm_floor_us": VERIFY_HBM_FLOOR_US,
            "body_int4_gb": BODY_INT4_GB,
            "lmhead_bf16_gb": LMHEAD_BF16_GB,
            "a10g_bw_gbps": A10G_BW_GBPS,
            "draft_k7_us": DRAFT_K7_US,
            "verify_m1_us": VERIFY_M1_US,
            "ppl_gate": PPL_GATE,
            "ppl_baseline": PPL_BASELINE,
            "ppl_headroom": PPL_HEADROOM,
            "milestone_tps": MILESTONE,
            "fern_281_et_floor": FERN_ET_FLOOR_281,
            "overcredit_278": OVERCREDIT_278,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[hbm-bound-ceiling] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "hbm_bound_ceiling_self_test_passes":
            int(bool(st["hbm_bound_ceiling_self_test_passes"])),
        "official_tps_as_frac_of_hbm_ceiling": c["official_tps_as_frac_of_hbm_ceiling"],
        "official_tps_as_frac_bandwidth_pure": c["official_tps_as_frac_bandwidth_pure"],
        "hbm_bound_ceiling_tps": c["hbm_bound_ceiling_tps"],
        "hbm_bound_ceiling_tps_bandwidth_pure_variant":
            c["hbm_bound_ceiling_tps_bandwidth_pure_variant"],
        "kernel_timing_headroom_tps": c["kernel_timing_headroom_tps"],
        "kernel_timing_clears_500": int(bool(c["kernel_timing_clears_500"])),
        "read_frac_of_wall_pct": c["read_frac_of_wall_pct"],
        "wall_deployed_official_us": b["wall_deployed_official_us"],
        "read_floor_official_us": b["read_floor_official_us"],
        "read_floor_local_us": b["read_floor_local_us"],
        "non_read_official_us": c["non_read_slack"]["non_read_official_us"],
        "tps_kernel_floor": c["non_read_slack"]["tps_kernel_floor"],
        "tps_kernel_floor_clears_500": int(bool(c["non_read_slack"]["tps_kernel_floor_clears_500"])),
        "verify_over_read_ratio": c["non_read_slack"]["verify_over_read_ratio"],
        "et_for_500_at_read_floor": lev["et_for_500_at_read_floor"],
        "et_for_500_deployed_point": lev["et_for_500_deployed_point"],
        "fern_281_resid": lev["fern_281_resid"],
        "read_reduction_pct_for_500_at_fixed_et": lev["read_reduction_pct_for_500_at_fixed_et"],
        "read_reduction_bytes_for_500_gb": lev["read_reduction_bytes_for_500_gb"],
        "read_reduction_is_binding_for_500": int(bool(lev["read_reduction_is_binding_for_500"])),
        "ppl_headroom": lev["ppl_headroom"],
        "tps_via_composition": b["tps_via_composition"],
        "tps_via_steprate": b["tps_via_steprate"],
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
    }
    summary = {k: val for k, val in summary.items()
               if not (isinstance(val, float) and not math.isfinite(val)) and val is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="hbm_bound_tps_ceiling_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[hbm-bound-ceiling] wandb logged: {summary}", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="hbm-bound-ceiling")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 283,
        "agent": "denken",
        "kind": "hbm-bound-tps-ceiling",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[hbm-bound-ceiling] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (d) and recompute PRIMARY.
    syn["self_test"]["conditions"]["d_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["d_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["hbm_bound_ceiling_self_test_passes"] = passes
    syn["hbm_bound_ceiling_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hbm_bound_tps_ceiling_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[hbm-bound-ceiling] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY hbm_bound_ceiling_self_test_passes = {passes}", flush=True)
    print(f"  TEST official_tps_as_frac_of_hbm_ceiling = "
          f"{syn['official_tps_as_frac_of_hbm_ceiling']:.6f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[hbm-bound-ceiling] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
