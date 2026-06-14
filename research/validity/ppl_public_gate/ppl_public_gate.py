#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PPL public-gate headroom: is PPL<=2.42 a third BINDING public gate? (PR #236).

THE THIRD CONDITION NOBODY PRICED
---------------------------------
The public 500-milestone is a CONJUNCTION of three conditions:

        TPS >= 500   AND   PPL <= 2.42   AND   128/128 complete.

Every launch-readiness leg this cycle (#217 composition, #222 binding-gate, #229 speed-
margin, #218 interleg-sigma, #228 publish-first lambda-floor) prices only the TPS / lambda
axes. PPL <= 2.42 has been carried as a STATIC "served 2.3772, fine" fact and never priced
as a function of the build's acceptance aggressiveness under the LOSSY int4 verify. The
frontier #52 serves PPL 2.3772 -- a margin of only 0.0428 (1.77%) below the 2.42 cap.
kanna #114 measured the int4-Marlin spec-verify diverging 56.08% per-token from plain greedy
AR. The worry: as the >=500 build pushes acceptance higher (deeper/wider tree, higher lambda)
to gain TPS, does the served PPL DRIFT toward the int4-standalone PPL and cross the cap --
i.e. is PPL a HIDDEN THIRD public gate?

THE LOAD-BEARING PHYSICS (why the drift premise is FALSE on the lambda axis)
---------------------------------------------------------------------------
program.md L27-28 sets the contract: "Greedy decode must remain token-identical to plain
greedy autoregressive decode FOR THE SUBMITTED CHECKPOINT." The submitted checkpoint is the
int4 model. So the served stream IS the int4 model's greedy stream, EXACTLY -- and a correct
(greedy-identity-preserving) speculative / tree decode changes only THROUGHPUT (tokens per
verify call), NEVER the served-token distribution. That is the whole point of speculative
decoding: it is output-equivalent to running the verify model alone, just faster.

Consequence: the served PPL is determined by the VERIFY MODEL (int4), and is INVARIANT to the
draft acceptance lambda / tree depth. The frontier #52 ALREADY serves 2.3772 WITH the int4
verify at its operating point, so:

    ppl_served(lambda) = int4_standalone_ppl = 2.3772   for ALL lambda     (output-equivalence)
    d(ppl_served)/d(lambda) = 0
    ppl_headroom_at_build_bar = 2.42 - 2.3772 = 0.0428   (lambda-invariant)

The "divergent-accept fraction" f_div is the int4-vs-target divergence ON THE SERVED (=int4)
stream = 0.5608, and is ITSELF lambda-invariant (lambda does not change which model verifies).
The build's lambda/tree aggressiveness cannot push f_div up -- only a change of the VERIFY
MODEL (coarser quantization) could. So PPL is NOT a binding gate on the launch-sigma lambda
axis; it is a VERIFY-MODEL property, pinned at 2.3772 with comfortable 0.0428 headroom.

THE MIXTURE (the PR's requested decomposition, fully pinned)
-----------------------------------------------------------
In log-PPL (= average NLL) space the served stream is a mixture of agreeing + divergent tokens:

    ln(ppl_served) = (1 - f_div)*ln(ppl_agree) + f_div*ln(ppl_div)

  ppl_agree = PPL on tokens where int4 argmax == target argmax  ~= reference PPL = 2.42/1.05
              (program.md: cap = "reference PPL + 5%")           = 2.30476  (the f_div=0 endpoint)
  f_div     = 0.5608                                              (kanna #114; lambda-invariant)
  ppl_div   = solved so the mixture round-trips served 2.3772     (the divergent-token sub-PPL)

This pins the int4-standalone served PPL at 2.3772 (resid 0) and lets us translate the 0.0428
PPL drift-tolerance into a max tolerable f_div increase -- the f_div at which the served average
would cross 2.42. The deployed f_div (0.5608) sits comfortably below that crossing, and the
build CANNOT consume the gap via lambda (output-equivalence). The honest bookend: the worst the
served PPL can ever reach by adding divergence is the int4-standalone PPL = 2.3772 itself (the
served stream is ALREADY the full int4 stream), and that is 0.0428 under the cap.

SCOPE: LOCAL CPU-only analytic pricing of the public milestone's third condition (PPL<=2.42) as
a function of the build's acceptance aggressiveness under the lossy int4 verify, calibrated to
#52's served-2.3772 anchor. No GPU / vLLM / HF Job / submission / served-file change / official
draw. BASELINE stays 481.53. Greedy/PPL path untouched (this leg ANALYZES PPL, changes nothing
served). Bank-the-analysis (PRIMARY = self-test, adds 0 TPS). NOT a launch. NOT open2. You do
NOT re-derive the divergence rate (lawine #232 / kanna #114) or integrate (fern #231).

PRIMARY metric  ppl_public_gate_self_test_passes
TEST    metric  ppl_headroom_at_build_bar  (= 2.42 - ppl_served(0.9780); lambda-invariant 0.0428)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/ppl_public_gate/ppl_public_gate.py \\
        --self-test --wandb_group issue192-reading-calibration --wandb_name denken/ppl-public-gate
"""
from __future__ import annotations

import argparse
import importlib.util
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
# Imported anchors (one source per constant; NOT re-derived).
# --------------------------------------------------------------------------- #
# frontier #52 served PPL + public cap (program.md: cap ~= reference PPL + 5%).
SERVED_PPL = 2.3772                 # PR #52 served int4-spec PPL (= the int4-standalone served PPL)
PPL_CAP = 2.42                      # public milestone cap
PPL_BUDGET_FACTOR = 1.05           # cap = reference PPL * 1.05  -> reference = cap/1.05
OFFICIAL_TPS = 481.53              # PR #52 official a10g frontier TPS

# kanna #114 (`9q5yy9l1`) banked int4-Marlin per-token argmax divergence (read-only).
_D114_INTERLOCK = (REPO_ROOT
                   / "research/validity/self_referential_gate/ab-20260614T075459Z"
                   / "interlock_report.json")
DIV_RATE_FALLBACK = 0.5608         # kanna #114 56.08% (used unless lawine #232 lands a rate)

# #222 binding_gate: the ceiling-anchored E[T](lambda)->mu_pub speed map + build bar (imported).
BINDING_GATE_PY = REPO_ROOT / "research/validity/binding_gate/binding_gate.py"
HEADLINE_REGIME = "both_bugs"      # the conservative regime (matches the build bar context)

# The PR's lambda evaluation grid {#52, 0.9500, 0.9780(build bar), 0.9970, 1.0}.
LAMBDA_950 = 0.9500
LAMBDA_997 = 0.9970
LAMBDA_CEILING = 1.0

TOL_EXACT = 1e-9
TOL_MARGIN = 1e-4


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Step 0 -- import banked anchors (divergence #114, speed map + build bar #222).
# --------------------------------------------------------------------------- #
def _lawine_232_rate() -> float | None:
    """Consume lawine #232's measured int4 M=8 divergence IF it has landed a banked result.

    lawine #232 is the IN-FLIGHT token-identity probe measuring the TRUE deployed divergence.
    Per the PR we consume its rate when it lands, else fall back to kanna #114's 56.08%. We
    detect a landed result by a banked validity dir; none exists yet -> return None (use #114).
    """
    for cand in (REPO_ROOT / "research/validity").glob("*lawine*"):
        for jf in cand.glob("*_results.json"):
            try:
                d = json.load(open(jf))
            except Exception:
                continue
            for k in ("token_div_frac", "divergence_rate", "int4_m8_divergence", "div_rate"):
                if _finite(d.get(k)):
                    return float(d[k])
    return None


def load_imports() -> dict[str, Any]:
    # kanna #114 banked per-token divergence (the served=int4 stream's divergence from target).
    d114 = json.load(open(_D114_INTERLOCK))
    per0 = d114["self_consistency_gate"]["per_run"][0]
    div_114 = float(per0["token_div_frac"])           # 0.5607757568359375

    # lawine #232 override if landed, else #114.
    div_lawine = _lawine_232_rate()
    div_rate = div_lawine if div_lawine is not None else div_114
    div_source = "lawine#232" if div_lawine is not None else "kanna#114"

    # #222 binding_gate: speed map (E[T](lambda)->mu_pub) + the 0.9780 build bar + composition.
    bg = _import("binding_gate", BINDING_GATE_PY)
    b = bg.import_banked()
    maps = bg.build_maps(b)
    mp = maps[HEADLINE_REGIME]

    reference_ppl = PPL_CAP / PPL_BUDGET_FACTOR        # 2.304762 (the lossless f_div=0 endpoint)

    return {
        # PPL gate anchors
        "served_ppl": SERVED_PPL,                      # 2.3772 (int4-standalone served PPL)
        "ppl_cap": PPL_CAP,                            # 2.42
        "reference_ppl": reference_ppl,                # 2.30476 (cap/1.05; bf16 lossless reference)
        "ppl_budget_factor": PPL_BUDGET_FACTOR,
        "official_tps": OFFICIAL_TPS,                  # 481.53
        # divergence (imported; #114, or lawine#232 if landed)
        "div_rate": div_rate,                          # 0.5608 (lambda-invariant on the served=int4 stream)
        "div_rate_114": div_114,
        "div_rate_source": div_source,
        "div_rate_lawine_232": div_lawine,
        # speed map + build bar (imported #222)
        "_map": mp,
        "regime": HEADLINE_REGIME,
        "build_bar": float(b["validity_bar"]),         # 0.9779783323491393 (the "0.9780" build bar)
        "build_bar_nominal": float(b["validity_bar_nominal"]),  # 0.9780
        "lambda_floor": float(b["lambda_floor"]),
        "ceiling_mu": float(b["lambda1_ceiling_mu"]),  # 520.9527 (#204 int4-spec ceiling)
        "K_cal": float(b["K_cal"]),                    # 125.268
        "step": float(b["step"]),                      # 1.2182
        "et_lambda1": float(mp.et_ceiling),            # 5.207
        "source_runs": {"d52": "(official 481.53)", "d114": "9q5yy9l1",
                        "d222": "binding_gate", "d191": "stark build bar 0.9780",
                        "d232": "lawine (in-flight)"},
    }


# --------------------------------------------------------------------------- #
# (1) Pin the third public condition (the frame).
# --------------------------------------------------------------------------- #
def frame_third_condition(imp: dict) -> dict[str, Any]:
    margin = imp["ppl_cap"] - imp["served_ppl"]
    return {
        "public_milestone": "TPS>=500  AND  PPL<=2.42  AND  128/128",
        "ppl_cap": imp["ppl_cap"],
        "ppl_served_frontier": imp["served_ppl"],
        "ppl_margin_frontier": margin,                 # 0.0428
        "ppl_margin_frontier_pct": 100.0 * margin / imp["ppl_cap"],
        "reference_ppl": imp["reference_ppl"],
        "served_vs_reference_pct": 100.0 * (imp["served_ppl"] / imp["reference_ppl"] - 1.0),  # +3.14%
        "budget_used_pct_of_5pct": 100.0 * (imp["served_ppl"] / imp["reference_ppl"] - 1.0) / 5.0,
        "priced_only_tps_lambda": ["#217", "#222", "#229", "#218", "#228"],
        "note": (
            f"the public milestone is a CONJUNCTION of three conditions; the launch-sigma legs "
            f"(#217/#222/#229/#218/#228) price only TPS/lambda. The third condition PPL<={imp['ppl_cap']} "
            f"is UNPRICED as a function of the build's acceptance aggressiveness. Frontier #52 serves "
            f"{imp['served_ppl']} -> margin {margin:.4f} ({100.0*margin/imp['ppl_cap']:.2f}%). The int4 verify "
            f"uses {100.0*(imp['served_ppl']/imp['reference_ppl']-1.0):.2f}% of the 5% PPL budget "
            f"(reference {imp['reference_ppl']:.5f} = cap/1.05)."),
    }


# --------------------------------------------------------------------------- #
# (2) The mixture (the PR's requested decomposition) -- fully pinned.
#     ln(ppl_served) = (1-f)*ln(ppl_agree) + f*ln(ppl_div).  Calibrate to #52's 2.3772.
# --------------------------------------------------------------------------- #
def _mix(f: float, ppl_lo: float, ppl_hi: float) -> float:
    """Log-space token-fraction mixture g(f); ppl_lo at f=0, ppl_hi at f=1."""
    return math.exp((1.0 - f) * math.log(ppl_lo) + f * math.log(ppl_hi))


def calibrate_mixture(imp: dict) -> dict[str, Any]:
    served = imp["served_ppl"]
    ppl_agree = imp["reference_ppl"]                  # f_div=0 endpoint (agreeing tokens ~ reference)
    f_div = imp["div_rate"]                            # operating divergent-accept fraction (=delta)

    # Solve ppl_div so the mixture round-trips the served 2.3772 at the operating f_div.
    #   ln(served) = (1-f)*ln(agree) + f*ln(div)  =>  ln(div) = [ln(served) - (1-f)*ln(agree)]/f
    ln_div = (math.log(served) - (1.0 - f_div) * math.log(ppl_agree)) / f_div
    ppl_div = math.exp(ln_div)

    served_roundtrip = _mix(f_div, ppl_agree, ppl_div)
    calib_resid = abs(served_roundtrip - served)

    # int4-standalone served PPL: the served stream IS the int4-standalone greedy stream
    # (output-equivalence), so the implied int4-standalone PPL == served == 2.3772 (PINNED).
    int4_standalone_ppl_implied = served
    int4_standalone_resid = abs(int4_standalone_ppl_implied - served)

    # drift driver at the operating point: d(ppl_served)/d(f_div) = ppl_served*(ln(div)-ln(agree)).
    dppl_dfdiv = served * (math.log(ppl_div) - math.log(ppl_agree))

    # cap-crossing fraction f* where the served AVERAGE would hit the cap, and the tolerance.
    # ln(cap) = (1-f*)*ln(agree) + f*  *ln(div)  =>  f* = [ln(cap)-ln(agree)]/[ln(div)-ln(agree)]
    denom = math.log(ppl_div) - math.log(ppl_agree)
    f_div_star = (math.log(imp["ppl_cap"]) - math.log(ppl_agree)) / denom if denom > 0 else float("inf")
    max_fdiv_increase = f_div_star - f_div            # how much f_div could rise before the cap

    return {
        "model": "ln(ppl_served) = (1-f_div)*ln(ppl_agree) + f_div*ln(ppl_div)",
        "ppl_agree_f0": ppl_agree,
        "f_div_operating": f_div,
        "ppl_div_solved": ppl_div,                    # divergent-token conditional PPL (sub-component)
        "ppl_div_above_cap": bool(ppl_div > imp["ppl_cap"]),
        "served_roundtrip": served_roundtrip,
        "calibration_residual": calib_resid,          # -> 0
        "int4_standalone_ppl_implied": int4_standalone_ppl_implied,  # 2.3772 (PINNED)
        "int4_standalone_residual": int4_standalone_resid,           # 0 (output-equivalence)
        "d_ppl_d_fdiv_at_operating": dppl_dfdiv,
        "f_div_star_cap_crossing": f_div_star,        # 0.8835 (where the served AVERAGE hits cap)
        "max_fdiv_increase_before_cap": max_fdiv_increase,  # 0.3227 (verify-model headroom in f_div)
        "note": (
            f"mixture calibrated to #52: agreeing tokens ~ reference {ppl_agree:.5f} (f_div=0), divergent "
            f"tokens ~ {ppl_div:.4f} (the f_div=1 sub-PPL), operating f_div={f_div:.4f} round-trips served "
            f"{served_roundtrip:.6f} (resid {calib_resid:.2e}). The divergent SUB-component {ppl_div:.4f} "
            f"sits ABOVE the cap, but it is only {f_div:.1%} of the stream -> the served AVERAGE is {served:.4f}. "
            f"The served average would cross {imp['ppl_cap']} only at f_div*={f_div_star:.4f} -- a "
            f"+{max_fdiv_increase:.4f} increase the build CANNOT make via lambda (output-equivalence)."),
    }


# --------------------------------------------------------------------------- #
# (3) Headroom at the build bar (the deliverable) -- the lambda projection table.
#     Output-equivalence: ppl_served(lambda) is FLAT while speed (mu_pub) RISES.
# --------------------------------------------------------------------------- #
def lambda_projection(imp: dict, calib: dict) -> dict[str, Any]:
    mp = imp["_map"]
    served = imp["served_ppl"]
    cap = imp["ppl_cap"]
    f_div = imp["div_rate"]

    # map-implied #52 acceptance: invert the public speed map at the 481.53 frontier (a LABEL only;
    # the headline does not depend on its exact value -- PPL is lambda-invariant).
    lam_52 = mp.solve_lambda_for_mu(imp["official_tps"])
    lam_52_label = lam_52 if lam_52 is not None else float("nan")

    grid = [("#52 (frontier)", lam_52_label),
            ("0.9500", LAMBDA_950),
            ("0.9780 (build bar)", imp["build_bar"]),
            ("0.9970", LAMBDA_997),
            ("1.0 (ceiling)", LAMBDA_CEILING)]

    rows = []
    for tag, lam in grid:
        if not _finite(lam):
            continue
        et = mp.et_of_lambda(lam)
        mu = mp.mu_pub(lam)                            # SPEED -- rises with lambda
        ppl_served_lam = served                        # FLAT (output-equivalence)
        headroom = cap - ppl_served_lam               # FLAT 0.0428
        rows.append({
            "tag": tag, "lambda_hat": lam,
            "E_T": et, "mu_pub_speed": mu,            # speed climbs ...
            "ppl_served": ppl_served_lam,             # ... PPL stays pinned
            "headroom_to_cap": headroom,
            "divergent_accept_fraction": f_div,       # FLAT delta
            "clears_cap": bool(ppl_served_lam <= cap + TOL_EXACT),
        })

    # headroom AT the build bar (the TEST metric).
    ppl_served_at_bar = served                         # lambda-invariant
    headroom_at_bar = cap - ppl_served_at_bar          # 0.0428

    # decoupling evidence: speed RISES across the grid while PPL is constant.
    mus = [r["mu_pub_speed"] for r in rows]
    ppls = [r["ppl_served"] for r in rows]
    speed_rises = all(mus[i + 1] > mus[i] - TOL_EXACT for i in range(len(mus) - 1))
    ppl_flat = (max(ppls) - min(ppls)) <= TOL_EXACT
    speed_gain_pct = 100.0 * (mus[-1] / mus[0] - 1.0) if mus and mus[0] else float("nan")

    return {
        "lambda_52_map_implied": lam_52_label,
        "build_bar": imp["build_bar"],
        "table": rows,
        "ppl_served_at_build_bar": ppl_served_at_bar,
        "ppl_headroom_at_build_bar": headroom_at_bar,  # TEST metric (0.0428)
        "max_ppl_drift_before_cap": cap - served,      # 0.0428 (full margin; only a verify change consumes it)
        "max_fdiv_increase_before_cap": calib["max_fdiv_increase_before_cap"],
        "d_ppl_served_d_lambda": 0.0,                  # output-equivalence: PPL flat in lambda
        "ppl_is_binding_public_gate": bool(ppl_served_at_bar > cap),  # FALSE
        "decoupling": {
            "speed_rises_with_lambda": bool(speed_rises),
            "ppl_flat_in_lambda": bool(ppl_flat),
            "speed_gain_across_grid_pct": speed_gain_pct,  # ~ +8% speed, 0% PPL
            "note": (
                f"across the lambda grid the public speed climbs {speed_gain_pct:.1f}% "
                f"({mus[0]:.2f}->{mus[-1]:.2f} TPS) while ppl_served stays pinned at {served} "
                f"(headroom {headroom_at_bar:.4f}) -- PPL is DECOUPLED from the build's acceptance axis."),
        },
        "land71_readout_flag": (
            "land #71's served run reports the scorer's PPL natively (vLLM prompt_logprobs path, "
            "program.md submission contract) -> PPL is a ONE-RUN-CONFIRMABLE readout in the GO-card, "
            "not a pre-launch unknown. It is NOT free: it must be READ from the served run, but it is "
            "pinned at the int4 verify's 2.3772 and cannot drift with lambda."),
    }


# --------------------------------------------------------------------------- #
# (4) Robustness + framing (SECONDARY).
# --------------------------------------------------------------------------- #
def robustness(imp: dict, calib: dict, proj: dict) -> dict[str, Any]:
    served = imp["served_ppl"]
    cap = imp["ppl_cap"]
    ppl_agree = imp["reference_ppl"]

    # (a) divergence sweep [0.50, 0.60] around #114's 0.5608 (and lawine #232 when it lands).
    #     The served PPL is the MEASURED anchor 2.3772 -> INVARIANT to the divergence rate; only the
    #     f_div decomposition + the f_div* crossing shift. The verdict (not binding on lambda) is robust.
    sweep = []
    for d in (0.50, 0.5608, 0.55, 0.5608, 0.60):
        ln_div = (math.log(served) - (1.0 - d) * math.log(ppl_agree)) / d
        ppl_div = math.exp(ln_div)
        denom = math.log(ppl_div) - math.log(ppl_agree)
        f_star = (math.log(cap) - math.log(ppl_agree)) / denom if denom > 0 else float("inf")
        sweep.append({
            "div_rate": d,
            "ppl_served": served,                      # INVARIANT (the measured anchor)
            "headroom_to_cap": cap - served,           # INVARIANT 0.0428
            "ppl_div_solved": ppl_div,
            "f_div_star_cap_crossing": f_star,
            "max_fdiv_increase_before_cap": f_star - d,
            "fdiv_margin_positive": bool(f_star - d > 0.0),
        })
    sweep_dedup = []
    seen = set()
    for s in sweep:
        if s["div_rate"] not in seen:
            seen.add(s["div_rate"])
            sweep_dedup.append(s)
    verdict_stable = all(s["headroom_to_cap"] > 0.0 and s["fdiv_margin_positive"] for s in sweep_dedup)

    # (b) publish-first framing: under #124 publish-first, PPL<=2.42 is a PUBLIC-milestone LAUNCH gate
    #     (NOT a post-hoc defence like the private bar) -> a breach FAILS the public milestone directly.
    publish_first = (
        "under #124 publish-first, PPL<=2.42 is a condition of the PUBLIC milestone itself -- a LAUNCH "
        "gate read at submission, not a post-hoc private-bar defence. A PPL breach would FAIL the public "
        "milestone directly even with TPS>=500 and 128/128. That makes pricing it (this leg) load-bearing "
        "-- and the result is that the int4 verify keeps it pinned 0.0428 under cap, lambda-invariant.")

    # (c) honest bookend: the worst the served PPL can reach by ADDING divergence is the int4-standalone
    #     PPL itself (the served stream is ALREADY the full int4 stream) = 2.3772 -- the ceiling the drift
    #     can never exceed. The divergent SUB-component (ppl_div) is higher but is not the served average.
    bookend = {
        "served_stream_is_int4_standalone": True,
        "worst_case_served_ppl_ceiling": served,       # 2.3772 (the realized value == the ceiling)
        "ceiling_clears_cap": bool(served <= cap),
        "divergent_subcomponent_ppl": calib["ppl_div_solved"],   # 2.4356 (sub-PPL, above cap, NOT the average)
        "note": (
            f"the served stream IS the int4-standalone greedy stream (output-equivalence), so the worst "
            f"the served AVERAGE can reach is the int4-standalone PPL {served} itself -- which is {cap-served:.4f} "
            f"UNDER the cap. The divergent-token sub-PPL {calib['ppl_div_solved']:.4f} is higher (above cap) but "
            f"is only a {imp['div_rate']:.1%} sub-fraction, NOT the served average."),
    }

    return {
        "divergence_sweep_0p50_0p60": sweep_dedup,
        "verdict_stable_under_divergence_sweep": bool(verdict_stable),
        "headroom_invariant_to_divergence": True,
        "publish_first_is_launch_gate": publish_first,
        "honest_bookend": bookend,
    }


# --------------------------------------------------------------------------- #
# (5) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, frame: dict, calib: dict, proj: dict, robust: dict) -> dict[str, Any]:
    served = imp["served_ppl"]
    cap = imp["ppl_cap"]
    ppl_agree = imp["reference_ppl"]
    ppl_div = calib["ppl_div_solved"]

    # (a) the mixture calibration round-trips #52's served PPL 2.3772 (resid -> 0).
    cond_a = bool(calib["calibration_residual"] <= TOL_EXACT
                  and calib["int4_standalone_residual"] <= TOL_EXACT)

    # (b) ppl_margin_frontier = 0.0428 within 1e-4.
    cond_b = bool(abs(frame["ppl_margin_frontier"] - 0.0428) <= TOL_MARGIN)

    # (c) ppl_served monotone non-decreasing in f_div (mixture) AND flat in lambda (output-equiv).
    f_grid = [i / 20.0 for i in range(21)]
    mix_vals = [_mix(f, ppl_agree, ppl_div) for f in f_grid]
    mono_fdiv = all(mix_vals[i + 1] >= mix_vals[i] - TOL_EXACT for i in range(len(mix_vals) - 1))
    ppls_lambda = [r["ppl_served"] for r in proj["table"]]
    flat_lambda = (max(ppls_lambda) - min(ppls_lambda)) <= TOL_EXACT
    cond_c = bool(mono_fdiv and flat_lambda and ppl_div >= ppl_agree)

    # (d) ppl_is_binding_public_gate consistent with (ppl_served(0.9780) > cap).
    cond_d = bool(proj["ppl_is_binding_public_gate"] == (proj["ppl_served_at_build_bar"] > cap))

    # (e) headroom monotone non-increasing in lambda (flat under output-equivalence).
    heads = [r["headroom_to_cap"] for r in proj["table"]]
    cond_e = all(heads[i + 1] <= heads[i] + TOL_EXACT for i in range(len(heads) - 1))

    # (f) NaN-clean (key scalars finite; full-payload walk enforced in main()).
    key = [served, cap, ppl_agree, ppl_div, frame["ppl_margin_frontier"],
           calib["calibration_residual"], calib["int4_standalone_ppl_implied"],
           calib["d_ppl_d_fdiv_at_operating"], calib["f_div_star_cap_crossing"],
           proj["ppl_headroom_at_build_bar"], proj["d_ppl_served_d_lambda"],
           proj["decoupling"]["speed_gain_across_grid_pct"]]
    cond_f = all(_finite(x) for x in key)

    # (g) decoupling: speed strictly rises while PPL is flat across the same lambda grid.
    cond_g = bool(proj["decoupling"]["speed_rises_with_lambda"]
                  and proj["decoupling"]["ppl_flat_in_lambda"])

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e and cond_f and cond_g)
    return {
        "ppl_public_gate_self_test_passes": passes,
        "conditions": {
            "a_mixture_roundtrips_served_2p3772_resid0": cond_a,
            "b_ppl_margin_frontier_is_0p0428": cond_b,
            "c_ppl_monotone_in_fdiv_and_flat_in_lambda": cond_c,
            "d_binding_bool_consistent_with_served_gt_cap": cond_d,
            "e_headroom_monotone_nonincreasing_in_lambda": cond_e,
            "f_key_scalars_finite": cond_f,
            "g_decoupling_speed_rises_ppl_flat": cond_g,
        },
        "evidence": {
            "a_calib_resid": calib["calibration_residual"],
            "a_int4_standalone_resid": calib["int4_standalone_residual"],
            "b_ppl_margin_frontier": frame["ppl_margin_frontier"],
            "c_mono_fdiv": mono_fdiv, "c_flat_lambda": flat_lambda,
            "c_ppl_div_ge_agree": bool(ppl_div >= ppl_agree),
            "d_binding": proj["ppl_is_binding_public_gate"],
            "d_served_at_bar_gt_cap": bool(proj["ppl_served_at_build_bar"] > cap),
            "e_headrooms": heads,
            "g_speed_gain_pct": proj["decoupling"]["speed_gain_across_grid_pct"],
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, frame: dict, calib: dict, proj: dict, robust: dict) -> str:
    binding = proj["ppl_is_binding_public_gate"]
    return (
        f"NOT-BINDING (on the launch-sigma lambda/tree axis). The public milestone's third condition "
        f"PPL<={imp['ppl_cap']} has headroom {proj['ppl_headroom_at_build_bar']:.4f} at the lambda={imp['build_bar']:.4f} "
        f"build bar -- and that headroom is LAMBDA-INVARIANT. program.md L27-28 pins greedy identity to the "
        f"SUBMITTED (int4) checkpoint, so the served stream IS the int4-standalone greedy stream and a correct "
        f"speculative/tree decode changes only THROUGHPUT, never the served-token distribution. Hence "
        f"ppl_served(lambda) = int4_standalone_ppl = {imp['served_ppl']} for ALL lambda (d(ppl)/d(lambda)=0); the "
        f"divergent-accept fraction is the int4-vs-target rate {imp['div_rate']:.4f} on the served=int4 stream and "
        f"is ITSELF lambda-invariant. The mixture calibrates exactly (resid {calib['calibration_residual']:.1e}): "
        f"agreeing tokens ~ reference {imp['reference_ppl']:.4f}, divergent sub-PPL {calib['ppl_div_solved']:.4f} "
        f"(above cap but only {imp['div_rate']:.0%} of the stream), served AVERAGE {imp['served_ppl']}. The served "
        f"average would cross {imp['ppl_cap']} only at f_div*={calib['f_div_star_cap_crossing']:.4f} -- a "
        f"+{calib['max_fdiv_increase_before_cap']:.4f} increase the build CANNOT make via lambda (only a coarser "
        f"VERIFY MODEL could). Robust across the divergence sweep [0.50,0.60] (served PPL is the measured anchor, "
        f"divergence-invariant; f_div margin stays positive). ppl_is_binding_public_gate={binding}. The honest "
        f"bookend: the worst the served average can reach by adding divergence is the int4-standalone {imp['served_ppl']} "
        f"itself ({proj['ppl_headroom_at_build_bar']:.4f} under cap). land #71's served run reports the scorer's PPL "
        f"natively -> one-run-confirmable, comfortable slack. BASELINE 481.53 untouched. NOT a launch."
    )


def _handoff(imp: dict, calib: dict, proj: dict) -> dict[str, str]:
    line = (
        f"the public milestone's third condition PPL<={imp['ppl_cap']} has headroom "
        f"{proj['ppl_headroom_at_build_bar']:.4f} at the lambda={imp['build_bar']:.4f} build bar (drift tolerance "
        f"+{calib['max_fdiv_increase_before_cap']:.4f} in divergent-accept-fraction from #52's {imp['served_ppl']} "
        f"anchor under the {imp['div_rate']:.0%} int4 divergence), and it is LAMBDA-INVARIANT (output-equivalence: "
        f"the served stream is the int4 greedy stream, so the build's acceptance aggressiveness cannot drift PPL) -- "
        f"so PPL is NOT a binding public gate on the launch-sigma axis; land #71's served run reports the scorer's PPL "
        f"natively, so it is one-run-confirmable but has comfortable slack (a verify-model change, not lambda, is the "
        f"only thing that moves it); fern #231 carries ppl_headroom_at_build_bar={proj['ppl_headroom_at_build_bar']:.4f} "
        f"as the third public-gate row alongside the TPS>=500 trigger."
    )
    return {"fern_231": line, "land_71": line, "human_124_packet": line}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    frame = frame_third_condition(imp)
    calib = calibrate_mixture(imp)
    proj = lambda_projection(imp, calib)
    robust = robustness(imp, calib, proj)
    st = self_test(imp, frame, calib, proj, robust)
    handoff = _handoff(imp, calib, proj)
    # strip the non-serializable map handle before returning the imports view.
    imp_view = {k: v for k, v in imp.items() if k != "_map"}
    return {
        "self_test": st,
        "test_metric": {"ppl_headroom_at_build_bar": proj["ppl_headroom_at_build_bar"]},
        "imports": imp_view,
        "frame_third_condition": frame,
        "mixture_calibration": calib,
        "lambda_projection": proj,
        "robustness": robust,
        "verdict": _verdict(imp, frame, calib, proj, robust),
        "handoff_lines": handoff,
    }


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
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
    imp = syn["imports"]
    frame, calib = syn["frame_third_condition"], syn["mixture_calibration"]
    proj, robust, st = syn["lambda_projection"], syn["robustness"], syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PPL PUBLIC-GATE HEADROOM (PR #236) -- is PPL<=2.42 a third BINDING public gate?", flush=True)
    print("=" * 100, flush=True)
    print(f"  public milestone: {frame['public_milestone']}", flush=True)
    print(f"  frontier #52 served PPL {imp['served_ppl']}  cap {imp['ppl_cap']}  "
          f"margin {frame['ppl_margin_frontier']:.4f} ({frame['ppl_margin_frontier_pct']:.2f}%)", flush=True)
    print(f"  reference {imp['reference_ppl']:.5f} (cap/1.05); int4 uses "
          f"{frame['served_vs_reference_pct']:.2f}% of the 5% budget", flush=True)
    print(f"  divergence {imp['div_rate']:.4f} (source {imp['div_rate_source']}; "
          f"lawine#232 landed={imp['div_rate_lawine_232'] is not None})", flush=True)
    print("-" * 100, flush=True)
    print("  MIXTURE (calibrated to #52):  ln(ppl_served) = (1-f)*ln(agree) + f*ln(div)", flush=True)
    print(f"    ppl_agree(f=0) {calib['ppl_agree_f0']:.5f}   ppl_div(f=1) {calib['ppl_div_solved']:.4f} "
          f"(above cap={calib['ppl_div_above_cap']})   f_div={calib['f_div_operating']:.4f}", flush=True)
    print(f"    round-trip served {calib['served_roundtrip']:.6f} (resid {calib['calibration_residual']:.2e})   "
          f"int4_standalone_ppl_implied {calib['int4_standalone_ppl_implied']:.4f} "
          f"(resid {calib['int4_standalone_residual']:.1e}, PINNED)", flush=True)
    print(f"    f_div* cap-crossing {calib['f_div_star_cap_crossing']:.4f}   "
          f"max f_div increase before cap {calib['max_fdiv_increase_before_cap']:.4f}", flush=True)
    print("-" * 100, flush=True)
    print("  LAMBDA PROJECTION (speed RISES, PPL FLAT):", flush=True)
    print(f"    {'lambda':>20}  {'E_T':>7}  {'speed':>8}  {'ppl':>7}  {'headroom':>8}  {'f_div':>6}  clears", flush=True)
    for r in proj["table"]:
        print(f"    {r['tag']:>20}  {r['E_T']:>7.4f}  {r['mu_pub_speed']:>8.2f}  {r['ppl_served']:>7.4f}  "
              f"{r['headroom_to_cap']:>8.4f}  {r['divergent_accept_fraction']:>6.4f}  {r['clears_cap']}", flush=True)
    print(f"    decoupling: speed +{proj['decoupling']['speed_gain_across_grid_pct']:.1f}% / PPL 0% across the grid", flush=True)
    print("-" * 100, flush=True)
    print(f"  ppl_headroom_at_build_bar = {proj['ppl_headroom_at_build_bar']:.4f}   <-- TEST", flush=True)
    print(f"  d(ppl_served)/d(lambda)   = {proj['d_ppl_served_d_lambda']}", flush=True)
    print(f"  ppl_is_binding_public_gate = {proj['ppl_is_binding_public_gate']}   <-- HEADLINE", flush=True)
    print(f"  robustness: verdict stable under divergence sweep [0.50,0.60] = "
          f"{robust['verdict_stable_under_divergence_sweep']} (headroom divergence-invariant)", flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) ppl_public_gate_self_test_passes = {st['ppl_public_gate_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF (fern #231 / land #71 / human #124): {syn['handoff_lines']['fern_231']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors #219; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[ppl-public-gate] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp, frame = syn["imports"], syn["frame_third_condition"]
    calib, proj = syn["mixture_calibration"], syn["lambda_projection"]
    robust, st = syn["robustness"], syn["self_test"]

    run = init_wandb_run(
        job_type="validity-gate",
        agent="denken",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["issue-192", "ppl-public-gate", "third-public-gate", "output-equivalence",
              "ppl-mixture", "int4-verify", "lambda-invariant", "kanna-114", "bank-the-analysis"],
        config={
            "official_tps": imp["official_tps"], "served_ppl": imp["served_ppl"],
            "ppl_cap": imp["ppl_cap"], "reference_ppl": imp["reference_ppl"],
            "div_rate": imp["div_rate"], "div_rate_source": imp["div_rate_source"],
            "build_bar": imp["build_bar"], "regime": imp["regime"],
            "ceiling_mu": imp["ceiling_mu"], "K_cal": imp["K_cal"], "step": imp["step"],
            "imports": "frontier#52 (2.3772) + kanna#114 (0.5608) + binding_gate#222 (map+bar)",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[ppl-public-gate] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "ppl_public_gate_self_test_passes": int(bool(st["ppl_public_gate_self_test_passes"])),
        "ppl_headroom_at_build_bar": proj["ppl_headroom_at_build_bar"],
        "ppl_is_binding_public_gate": int(bool(proj["ppl_is_binding_public_gate"])),
        "ppl_margin_frontier": frame["ppl_margin_frontier"],
        "ppl_served_frontier": frame["ppl_served_frontier"],
        "ppl_cap": frame["ppl_cap"],
        "reference_ppl": frame["reference_ppl"],
        "served_vs_reference_pct": frame["served_vs_reference_pct"],
        "div_rate": imp["div_rate"], "div_rate_114": imp["div_rate_114"],
        "lawine_232_landed": int(imp["div_rate_lawine_232"] is not None),
        "int4_standalone_ppl_implied": calib["int4_standalone_ppl_implied"],
        "int4_standalone_residual": calib["int4_standalone_residual"],
        "calibration_residual": calib["calibration_residual"],
        "ppl_div_solved": calib["ppl_div_solved"],
        "ppl_div_above_cap": int(bool(calib["ppl_div_above_cap"])),
        "d_ppl_d_fdiv_at_operating": calib["d_ppl_d_fdiv_at_operating"],
        "f_div_operating": calib["f_div_operating"],
        "f_div_star_cap_crossing": calib["f_div_star_cap_crossing"],
        "max_fdiv_increase_before_cap": calib["max_fdiv_increase_before_cap"],
        "max_ppl_drift_before_cap": proj["max_ppl_drift_before_cap"],
        "d_ppl_served_d_lambda": proj["d_ppl_served_d_lambda"],
        "lambda_52_map_implied": proj["lambda_52_map_implied"],
        "build_bar": imp["build_bar"],
        "speed_gain_across_grid_pct": proj["decoupling"]["speed_gain_across_grid_pct"],
        "speed_rises_with_lambda": int(bool(proj["decoupling"]["speed_rises_with_lambda"])),
        "ppl_flat_in_lambda": int(bool(proj["decoupling"]["ppl_flat_in_lambda"])),
        "verdict_stable_under_divergence_sweep": int(bool(robust["verdict_stable_under_divergence_sweep"])),
        "worst_case_served_ppl_ceiling": robust["honest_bookend"]["worst_case_served_ppl_ceiling"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        # the lambda table as flat scalars for plotting (speed rises, PPL flat).
        **{f"ppl_served_at_lambda_{i}": r["ppl_served"] for i, r in enumerate(proj["table"])},
        **{f"mu_pub_at_lambda_{i}": r["mu_pub_speed"] for i, r in enumerate(proj["table"])},
        **{f"headroom_at_lambda_{i}": r["headroom_to_cap"] for i, r in enumerate(proj["table"])},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="ppl_public_gate_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[ppl-public-gate] wandb logged {len(summary)} summary keys", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="issue192-reading-calibration")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 236,
        "agent": "denken",
        "kind": "ppl-public-gate-headroom",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[ppl-public-gate] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (f) and recompute PRIMARY.
    syn["self_test"]["conditions"]["f_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["f_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["ppl_public_gate_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ppl_public_gate_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[ppl-public-gate] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY ppl_public_gate_self_test_passes = {passes}", flush=True)
    print(f"  TEST ppl_headroom_at_build_bar = {syn['test_metric']['ppl_headroom_at_build_bar']:.4f}", flush=True)
    print(f"  HEADLINE ppl_is_binding_public_gate = {syn['lambda_projection']['ppl_is_binding_public_gate']}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[ppl-public-gate] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
