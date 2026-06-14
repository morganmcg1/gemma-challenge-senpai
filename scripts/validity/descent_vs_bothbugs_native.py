#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Descent-only vs both-bugs: NATIVE private-drop decision (PR #164).

THE PROBLEM
-----------
#156 pinned the tree's private drop at 1.80% (descent) / 1.86% (both-bugs) at the
organizer GT-4.3% LINEAR anchor, and concluded descent-only projects 510.6 / both-bugs
525.5. But that 1.80% was obtained by SHAPE-TRANSFER + INTERPOLATION: the deliberately
hard chat proxy (sglang-scored ~10.7% linear drop) was scaled toward public by a single
fraction `frac=0.40` (`build_calibrated_ladder`) so its AGGREGATE linear-E[T] drop hit
4.3%, and the tree drop was read off that ONE interpolated per-position shape. That bakes
in the assumption that the ladder SHAPE scales linearly between the hard tail and public.

The launch-topology decision turns on exactly this number:
  * if the tree's REAL private drop ~= the 4.3%-faithful 1.80%, descent-only (the simpler
    build, no spine) is private-safe and we can launch it;
  * if it sits materially higher, the both-bugs spine (wirbel #160 + lawine #161) becomes a
    HARD launch dependency.

WHAT THIS SCRIPT DOES (the native fix)
--------------------------------------
Removes the single-shape interpolation by propagating the tree drop under >=2 INDEPENDENT
organizer-faithful proxies, each NATIVELY ~4.3% on the LINEAR stack (not one hard tail
scaled by frac). For each native proxy we:
  1. take its per-position acceptance ladder measured under the sglang `vllm-chat` scored
     protocol (the organizer-matching protocol pinned in #156) -- either measured directly
     on a real 128-prompt set, or COUNT-POOLED from real measured component pools (pooling
     interpolates CUMULATIVES = the physically-realizable mixture, NOT conditionals like the
     removed `build_calibrated_ladder`);
  2. `relative_transfer` it onto the banked decode-frame public reference (the accepted
     harness-path bridge from #156 -- this is NOT the removed assumption; it only re-bases
     the protocol, it does not synthesize the 4.3% shape);
  3. feed the native ladder DIRECTLY into the banked descent-walk E[T] DP
     (`tree_private_acceptance_gap.project_one`) -- NO `frac` interpolation -- for both
     descent-only and both-bugs.
We then report the CI across the independent proxies on the binding input
(`tree_private_drop_pct`) and resolve descent-only-vs-both-bugs.

LADDER ALGEBRA (why pooling is native and interpolation is not)
---------------------------------------------------------------
The deployed drafter's per-draft accept events pool linearly at the COUNT level: a prompt
mixture's cumulative acceptance is C_mix[k] = sum_pool(C_pool[k]*drafts_pool)/sum(drafts).
That IS the exact ladder the drafter would produce on the combined real prompt set. The
conditional ladder p_mix[k]=C_mix[k]/C_mix[k-1] is then a NON-linear function of the mix.
#156's `build_calibrated_ladder` instead blends the CONDITIONALS linearly
(q=q_pub-frac*(q_pub-q_proxy)) -- an operation that is NOT realizable as any single prompt
distribution. So composing native proxies by count-pooling real measured pools removes
exactly the assumption #156 baked in, while letting us hit the 4.3% calibration anchor
exactly (continuous pool weight) and vary the SHAPE across genuinely different component
pools (length / domain / template axes).

LOCAL, CPU-ONLY (this file): consumes sglang server logs from `private_gap_probe.py` runs
(the single-A10G GPU step) + the banked #151/#156 ladders. No HF Job, no submission, no
served-file change. Greedy/PPL untouched, BASELINE unchanged (481.53).

OUTPUTS (PR #164)
-----------------
  native_proxies_reproduce_flagship_4p3 (PRIMARY, bool) -- AND across all constructed proxies
      that each reproduces GT-4.3% LINEAR drop natively to <=0.5pp (the calibration gate).
  tree_private_drop_pct_native_ci       (TEST)          -- CI-midpoint descent-only tree drop.
  descent_only_private_safe_native / both_bugs_required_private (bools) + TPS/drop margins,
      CI band, and the machinery-faithfulness cross-check (4.3% through the DP -> 510.6/525.5).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILER = ROOT / "scripts" / "profiler"
if str(PROFILER) not in sys.path:
    sys.path.insert(0, str(PROFILER))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Banked machinery -- one source of truth (do NOT re-derive). ------------------- #
_tpag = _load("tree_private_acceptance_gap", PROFILER / "tree_private_acceptance_gap.py")
_acc = _load("accept_calibration", PROFILER / "accept_calibration.py")
_recon = _load("tree_private_drop_reconcile", ROOT / "scripts/validity/tree_private_drop_reconcile.py")

DescentModel = _tpag.DescentModel
project_one = _tpag.project_one
breakeven_drop = _tpag.breakeven_drop
linear_et_from_q = _tpag.linear_et_from_q
official_tps_map = _tpag.official_tps_map
accept_length_for_official = _tpag.accept_length_for_official
run_self_test = _tpag.run_self_test
load_measured = _tpag.load_measured
load_rank_coverage = _tpag.load_rank_coverage
load_m32_topology = _tpag.load_m32_topology
BUG1_MULT = _tpag.BUG1_MULT
K_CAL = _tpag.K_CAL
TARGET_500 = _tpag.TARGET_500
TAU = _tpag.TAU
STEP_MEASURED_DEPTH9 = _tpag.STEP_MEASURED_DEPTH9
FRONTIER_OFFICIAL = _tpag.FRONTIER_OFFICIAL
relative_transfer = _recon.relative_transfer
parse_server_ladder = _recon.parse_server_ladder
parse_log_per_position = _acc.parse_log_per_position

# Organizer ground-truth anchor (BASELINE.md, VERIFIED 2026-06-13 23:04Z). ------- #
GT_PUBLIC_TPS = 481.53
GT_PRIVATE_TPS = 460.85
GT_DROP = 1.0 - GT_PRIVATE_TPS / GT_PUBLIC_TPS  # 0.04294...

# #156 canonical pinned 4.3% spine + projections (machinery-faithfulness anchors). #
PINNED_43_SPINE = [0.6906018151221658, 0.7483767282325214, 0.7826008065417849,
                   0.8303970184964822, 0.8533901004851394, 0.8308419924862088,
                   0.8683714482841856]
PINNED_43_DESCENT_TPS = 510.5849335058248
PINNED_43_BOTH_TPS = 525.4600047477221
PINNED_43_DESCENT_DROP = 1.801511061644668
PINNED_43_BOTH_DROP = 1.8626055407019635

PUBLIC_BANKED_JSON = "research/accept_calibration/accept_calibration_results.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"
# banked sglang public_cold reference (the protocol reference for relative_transfer).
BANKED_SGLANG_RUN = "research/validity/tree_private_drop_reconcile/sglang_ladder_run"
CALIB_TOL = 0.005  # <=0.5pp calibration gate (same as #156 harness_pin_reproduces_flagship_4p3)


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def _clean(x):
    """NaN/inf -> None (PR #164 requires NaN-clean metrics)."""
    if isinstance(x, (int, bool)):
        return x
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    return xf if np.isfinite(xf) else None


def conditional_to_cumulative(q: list[float]) -> list[float]:
    c, out = 1.0, []
    for qk in q:
        c *= qk
        out.append(c)
    return out


def cumulative_to_conditional(c: list[float]) -> list[float]:
    q = [c[0]]
    for k in range(1, len(c)):
        prev = c[k - 1]
        q.append(c[k] / prev if prev > 0 else 0.0)
    return q


def pool_cumulative(components: list[tuple[list[float], float]]) -> list[float]:
    """Exact native ladder of a prompt MIXTURE: draft-count-weighted cumulative blend.
    components = [(conditional_p, weight=num_drafts_or_fraction), ...]."""
    n = min(len(c) for c, _ in components)
    tot = sum(w for _, w in components)
    out = []
    for k in range(n):
        out.append(sum(conditional_to_cumulative(c)[k] * w for c, w in components) / tot)
    return out


def ladder_from_spec(spec: dict, run_dir_default: Path) -> dict[str, Any]:
    """Return {conditional_p, num_drafts, E_T} for a measured-or-explicit sglang ladder."""
    if "conditional_p_sglang" in spec:
        q = list(spec["conditional_p_sglang"])
        return {"conditional_p": q, "num_drafts": float(spec.get("num_drafts", 1.0)),
                "E_T": linear_et_from_q(q)}
    log = spec.get("server_log")
    if log:
        p = Path(log)
        if not p.is_absolute():
            p = (run_dir_default / log) if not p.exists() else p
        lad = parse_server_ladder(p)
        if lad is None:
            raise RuntimeError(f"could not parse per-position ladder from {p}")
        return {"conditional_p": lad["conditional_p"], "num_drafts": float(lad["num_drafts"]),
                "E_T": linear_et_from_q(lad["conditional_p"])}
    raise ValueError(f"ladder spec needs conditional_p_sglang or server_log: {spec}")


def native_sglang_ladder(proxy: dict, q_pub_sglang: list[float], run_dir: Path) -> dict[str, Any]:
    """Resolve a proxy's NATIVE sglang per-position ladder.

    mode 'measured' : parse a directly-measured real 128-set ladder.
    mode 'pooled'   : count-pool real component pools to hit a target sglang/decode drop
                      EXACTLY (continuous pool weight); the public pool is q_pub_sglang.
    """
    mode = proxy.get("mode", "measured")
    if mode == "measured":
        lad = ladder_from_spec(proxy, run_dir)
        return {"conditional_p": lad["conditional_p"], "num_drafts": lad["num_drafts"],
                "pool_weight": None}
    if mode == "pooled":
        comp = ladder_from_spec(proxy["component"], run_dir)   # the "hard" component pool
        q_hard = comp["conditional_p"]
        et_pub = linear_et_from_q(q_pub_sglang)
        # bisect the hard-pool weight so the POOLED sglang linear drop hits target.
        target = float(proxy.get("target_sglang_drop", GT_DROP))
        lo, hi = 0.0, 1.0
        for _ in range(80):
            w = 0.5 * (lo + hi)
            c = pool_cumulative([(q_pub_sglang, 1.0 - w), (q_hard, w)])
            drop = (et_pub - (1.0 + sum(c))) / et_pub
            if drop < target:
                lo = w
            else:
                hi = w
        w = 0.5 * (lo + hi)
        c = pool_cumulative([(q_pub_sglang, 1.0 - w), (q_hard, w)])
        return {"conditional_p": cumulative_to_conditional(c), "num_drafts": comp["num_drafts"],
                "pool_weight": w}
    raise ValueError(f"unknown proxy mode {mode}")


def propagate_native(model, q_public_banked, q_native_sglang, q_pub_sglang, rho_pub, step):
    """Feed a NATIVE proxy ladder DIRECTLY through the descent-walk DP (no frac).

    The native sglang ladder is `relative_transfer`-ed onto the banked decode-frame public
    (the #156 harness-path bridge), then fed as the private spine to project_one. Returns
    descent-only + both-bugs E[T], official TPS, drop% vs public, and the native linear drop.
    """
    et_pub_banked = linear_et_from_q(q_public_banked)
    # native sglang-frame linear drop (organizer-protocol reading).
    native_sglang_drop = (linear_et_from_q(q_pub_sglang) - linear_et_from_q(q_native_sglang)) \
        / linear_et_from_q(q_pub_sglang)
    # bridge to decode frame (the DP's anchor); this is re-basing, NOT shape synthesis.
    spine = relative_transfer(q_public_banked, q_pub_sglang, q_native_sglang)
    native_decode_drop = (et_pub_banked - linear_et_from_q(spine)) / et_pub_banked
    h = float(np.mean(spine)) / float(np.mean(q_public_banked[:len(spine)]))
    proj = project_one(model, spine, q_public_banked, rho_pub, h, step)
    et_pub_descent = model.et_tree(q_public_banked[0] * BUG1_MULT, q_public_banked, rho_cond=rho_pub)
    et_pub_both = model.et_tree(q_public_banked[0], q_public_banked, rho_cond=rho_pub)
    for topo, ref in (("descent_only", et_pub_descent), ("both_bugs", et_pub_both)):
        proj[topo]["et_drop_pct_vs_public"] = (ref - proj[topo]["et_central_rho_coupled"]) / ref * 100.0
    return {
        "spine_decode_frame": spine,
        "linear_E_T_decode": linear_et_from_q(spine),
        "native_sglang_linear_drop_pct": native_sglang_drop * 100.0,
        "native_decode_linear_drop_pct": native_decode_drop * 100.0,
        "spine_haircut_h": h,
        "projection": proj,
    }


def crosscheck_pinned_43(model, q_public_banked, rho_pub, step) -> dict[str, Any]:
    """Machinery faithfulness: feed the banked #156 pinned-4.3% spine through the SAME DP
    and confirm descent 510.6 / both-bugs 525.5 (drops 1.80% / 1.86%)."""
    h = float(np.mean(PINNED_43_SPINE)) / float(np.mean(q_public_banked[:len(PINNED_43_SPINE)]))
    proj = project_one(model, PINNED_43_SPINE, q_public_banked, rho_pub, h, step)
    et_pub_descent = model.et_tree(q_public_banked[0] * BUG1_MULT, q_public_banked, rho_cond=rho_pub)
    et_pub_both = model.et_tree(q_public_banked[0], q_public_banked, rho_cond=rho_pub)
    d = proj["descent_only"]; b = proj["both_bugs"]
    d_drop = (et_pub_descent - d["et_central_rho_coupled"]) / et_pub_descent * 100.0
    b_drop = (et_pub_both - b["et_central_rho_coupled"]) / et_pub_both * 100.0
    ok = (abs(d["official_central"] - PINNED_43_DESCENT_TPS) < 0.5
          and abs(b["official_central"] - PINNED_43_BOTH_TPS) < 0.5)
    return {
        "descent_tps": d["official_central"], "descent_tps_expected": PINNED_43_DESCENT_TPS,
        "both_bugs_tps": b["official_central"], "both_bugs_tps_expected": PINNED_43_BOTH_TPS,
        "descent_drop_pct": d_drop, "descent_drop_pct_expected": PINNED_43_DESCENT_DROP,
        "both_bugs_drop_pct": b_drop, "both_bugs_drop_pct_expected": PINNED_43_BOTH_DROP,
        "reproduces_156": bool(ok),
    }


def _band(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"min": None, "max": None, "mid": None, "mean": None, "n": 0}
    return {"min": min(vals), "max": max(vals), "mid": 0.5 * (min(vals) + max(vals)),
            "mean": statistics.fmean(vals), "n": len(vals)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxies-json", default=None,
                    help="manifest of native proxies (top-level q_pub_sglang spec + proxies[]). "
                         "Omit for the CPU cross-check / liveness pass (native_pending=true).")
    ap.add_argument("--public-banked", default=PUBLIC_BANKED_JSON)
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--banked-sglang-run", default=BANKED_SGLANG_RUN)
    ap.add_argument("--step", type=float, default=STEP_MEASURED_DEPTH9)
    ap.add_argument("--gt-drop", type=float, default=GT_DROP)
    ap.add_argument("--safe-margin", type=float, default=2.0,
                    help="descent-only is 'private-safe' iff its worst-proxy central TPS clears "
                         "500 by >= this margin across the CI band.")
    ap.add_argument("--output",
                    default="research/validity/descent_vs_bothbugs_private/results.json")
    ap.add_argument("--wandb-group", "--wandb_group", default="descent-vs-bothbugs-private-decision")
    ap.add_argument("--wandb-name", "--wandb_name", default="stark/descent-vs-bothbugs-native")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- banked model ----
    meas_pub = load_measured(args.public_banked, "server_log")
    q_public = list(meas_pub["q"])
    rc = load_rank_coverage(args.rankcov_json)
    rho_pub = rc["rho_cond"]
    parent = load_m32_topology(args.rho_opt_json)
    model = DescentModel(parent, rho_pub)
    st = run_self_test(model, q_public, rho_pub)
    assert st["passes"], "descent-walk DP does not reproduce banked anchors"
    et_pub_linear = linear_et_from_q(q_public)
    bar_500 = accept_length_for_official(TARGET_500, args.step, 1.0)

    # ---- machinery faithfulness cross-check (instruction 3) ----
    xcheck = crosscheck_pinned_43(model, q_public, rho_pub, args.step)
    assert xcheck["reproduces_156"], (
        f"DP does not reproduce #156 4.3% projection: {xcheck}")

    run_dir = Path(args.banked_sglang_run)
    results: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "constants": {
            "K_cal": K_CAL, "step": args.step, "bug1_mult": BUG1_MULT,
            "gt_drop": args.gt_drop, "gt_drop_pct": args.gt_drop * 100.0,
            "frontier_official": FRONTIER_OFFICIAL, "target_500": TARGET_500,
            "clear_500_bar_et": bar_500, "calib_tol_pp": CALIB_TOL * 100.0,
        },
        "self_test_passes": bool(st["passes"]),
        "public_ladder_banked": {"conditional_p": q_public, "linear_E_T": et_pub_linear},
        "machinery_faithfulness_xcheck": xcheck,
    }
    verdict: dict[str, Any] = {
        "self_test_passes": int(st["passes"]),
        "xcheck_reproduces_156_4p3": int(xcheck["reproduces_156"]),
        "xcheck_descent_tps": _clean(xcheck["descent_tps"]),
        "xcheck_both_bugs_tps": _clean(xcheck["both_bugs_tps"]),
        "gt_drop_pct": _clean(args.gt_drop * 100.0),
    }

    if not args.proxies_json or not Path(args.proxies_json).exists():
        results["native_pending"] = True
        results["note"] = ("no native-proxy manifest yet; CPU cross-check + liveness only. "
                           "Measure native proxies via private_gap_probe.py, then re-run with "
                           "--proxies-json.")
        results["verdict"] = verdict
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, default=_jd))
        print(f"[native] cross-check PASS (descent {xcheck['descent_tps']:.1f}, "
              f"both {xcheck['both_bugs_tps']:.1f}); wrote partial {args.output}", flush=True)
        if not args.no_wandb:
            try:
                _log_wandb(args, results, verdict, proxies_rows=[])
            except Exception as e:  # noqa: BLE001
                print(f"[native] W&B logging failed (non-fatal): {e!r}", flush=True)
        return 0

    # ---- native proxies ----
    manifest = json.loads(Path(args.proxies_json).read_text())
    pub_spec = manifest.get("q_pub_sglang")
    if pub_spec is None:
        # default: banked public_cold from the #156 sglang run
        q_pub_sglang = parse_server_ladder(run_dir / "server_public_cold.log")["conditional_p"]
    else:
        q_pub_sglang = ladder_from_spec(pub_spec, run_dir)["conditional_p"]

    proxies_out = []
    for proxy in manifest["proxies"]:
        native = native_sglang_ladder(proxy, q_pub_sglang, run_dir)
        prop = propagate_native(model, q_public, native["conditional_p"], q_pub_sglang,
                                rho_pub, args.step)
        d = prop["projection"]["descent_only"]; b = prop["projection"]["both_bugs"]
        calib_drop = prop["native_decode_linear_drop_pct"] / 100.0
        reproduces_4p3 = bool(abs(calib_drop - args.gt_drop) <= CALIB_TOL)
        proxies_out.append({
            "name": proxy["name"], "axis": proxy.get("axis"), "mode": proxy.get("mode", "measured"),
            "pool_weight": native.get("pool_weight"),
            "conditional_p_sglang": native["conditional_p"],
            "native_sglang_linear_drop_pct": _clean(prop["native_sglang_linear_drop_pct"]),
            "native_decode_linear_drop_pct": _clean(prop["native_decode_linear_drop_pct"]),
            "reproduces_flagship_4p3": reproduces_4p3,
            "spine_decode_frame": prop["spine_decode_frame"],
            "descent_only": {
                "tree_drop_pct": _clean(d["et_drop_pct_vs_public"]),
                "tps_central": _clean(d["official_central"]),
                "tps_taulow": _clean(d["official_taulow"]),
                "tps_band": [_clean(d["tps_band"][0]), _clean(d["tps_band"][1])],
                "clears_500_central": bool(d["clears_500_central"]),
                "clears_500_conservative": bool(d["clears_500_conservative"]),
                "margin_to_500": _clean(d["margin_to_500"]),
            },
            "both_bugs": {
                "tree_drop_pct": _clean(b["et_drop_pct_vs_public"]),
                "tps_central": _clean(b["official_central"]),
                "tps_taulow": _clean(b["official_taulow"]),
                "tps_band": [_clean(b["tps_band"][0]), _clean(b["tps_band"][1])],
                "clears_500_central": bool(b["clears_500_central"]),
                "clears_500_conservative": bool(b["clears_500_conservative"]),
                "margin_to_500": _clean(b["margin_to_500"]),
            },
        })
        print(f"[native] {proxy['name']:<16s} axis={proxy.get('axis'):<10s} "
              f"native_drop(decode)={prop['native_decode_linear_drop_pct']:.2f}% "
              f"reprod4.3={reproduces_4p3} -> descent {d['et_drop_pct_vs_public']:.2f}% "
              f"{d['official_central']:.1f}TPS | both {b['et_drop_pct_vs_public']:.2f}% "
              f"{b['official_central']:.1f}TPS", flush=True)

    # ---- CI across independent proxies ----
    desc_drop = _band([p["descent_only"]["tree_drop_pct"] for p in proxies_out])
    both_drop = _band([p["both_bugs"]["tree_drop_pct"] for p in proxies_out])
    desc_tps = _band([p["descent_only"]["tps_central"] for p in proxies_out])
    desc_tps_low = _band([p["descent_only"]["tps_taulow"] for p in proxies_out])
    both_tps = _band([p["both_bugs"]["tps_central"] for p in proxies_out])
    both_tps_low = _band([p["both_bugs"]["tps_taulow"] for p in proxies_out])

    # ---- PRIMARY self-validation: AND of per-proxy native 4.3% reproduction ----
    native_proxies_reproduce_flagship_4p3 = bool(
        len(proxies_out) >= 2 and all(p["reproduces_flagship_4p3"] for p in proxies_out))

    # ---- launch-topology decision ----
    # descent-only private-safe iff its WORST-proxy central TPS clears 500 by >= safe-margin
    # AND every proxy's conservative (tau-low) corner clears 500.
    desc_worst_central = desc_tps["min"]
    desc_worst_taulow = desc_tps_low["min"]
    descent_only_private_safe_native = bool(
        desc_worst_central is not None
        and desc_worst_central >= TARGET_500 + args.safe_margin
        and desc_worst_taulow is not None and desc_worst_taulow >= TARGET_500)
    both_worst_central = both_tps["min"]
    both_worst_taulow = both_tps_low["min"]
    both_bugs_clears_ci = bool(both_worst_taulow is not None and both_worst_taulow >= TARGET_500)
    both_bugs_required_private = bool((not descent_only_private_safe_native) and both_bugs_clears_ci)

    headline = {
        "native_proxies_reproduce_flagship_4p3": native_proxies_reproduce_flagship_4p3,  # PRIMARY
        "tree_private_drop_pct_native_ci": _clean(desc_drop["mid"]),                      # TEST
        "tree_private_drop_pct_native_ci_band": [_clean(desc_drop["min"]), _clean(desc_drop["max"])],
        "tree_private_drop_pct_native_mean": _clean(desc_drop["mean"]),
        "tree_private_drop_pct_native_both_bugs_ci": _clean(both_drop["mid"]),
        "tree_private_drop_pct_native_both_bugs_band": [_clean(both_drop["min"]), _clean(both_drop["max"])],
        "descent_only_tps_native_ci": [_clean(desc_tps["min"]), _clean(desc_tps["max"])],
        "descent_only_tps_native_mid": _clean(desc_tps["mid"]),
        "descent_only_tps_taulow_min": _clean(desc_worst_taulow),
        "both_bugs_tps_native_ci": [_clean(both_tps["min"]), _clean(both_tps["max"])],
        "descent_only_private_safe_native": descent_only_private_safe_native,
        "both_bugs_required_private": both_bugs_required_private,
        "descent_only_worst_margin_to_500": _clean(
            None if desc_worst_central is None else desc_worst_central - TARGET_500),
        "both_bugs_worst_margin_to_500": _clean(
            None if both_worst_central is None else both_worst_central - TARGET_500),
        "n_native_proxies": len(proxies_out),
        # vs the #156 transferred-interpolation result (does removing the interpolation move it?)
        "pinned_156_descent_drop_pct": PINNED_43_DESCENT_DROP,
        "pinned_156_descent_tps": PINNED_43_DESCENT_TPS,
        "native_vs_pinned_descent_drop_shift_pp": _clean(
            None if desc_drop["mid"] is None else desc_drop["mid"] - PINNED_43_DESCENT_DROP),
        "xcheck_reproduces_156_4p3": bool(xcheck["reproduces_156"]),
    }

    results["native_pending"] = False
    results["q_pub_sglang"] = q_pub_sglang
    results["proxies"] = proxies_out
    results["ci"] = {
        "descent_only_tree_drop_pct": desc_drop, "both_bugs_tree_drop_pct": both_drop,
        "descent_only_tps_central": desc_tps, "descent_only_tps_taulow": desc_tps_low,
        "both_bugs_tps_central": both_tps, "both_bugs_tps_taulow": both_tps_low,
    }
    results["headline"] = headline
    verdict.update({k: v for k, v in headline.items() if not isinstance(v, (list, dict))})
    results["verdict"] = verdict

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=_jd))
    print(f"[native] wrote {args.output}", flush=True)

    # ---- console summary ----
    print("\n" + "=" * 76, flush=True)
    print("PR #164 -- DESCENT-ONLY vs BOTH-BUGS, NATIVE PRIVATE-DROP DECISION", flush=True)
    print("=" * 76, flush=True)
    print(f"  machinery xcheck: 4.3% -> descent {xcheck['descent_tps']:.1f} / both "
          f"{xcheck['both_bugs_tps']:.1f}  (reproduces #156: {xcheck['reproduces_156']})", flush=True)
    print(f"  native proxies (n={len(proxies_out)}):", flush=True)
    for p in proxies_out:
        print(f"    - {p['name']:<16s} [{p['axis']}] native_drop={p['native_decode_linear_drop_pct']:.2f}% "
              f"reprod4.3={p['reproduces_flagship_4p3']}: descent {p['descent_only']['tree_drop_pct']:.2f}% "
              f"-> {p['descent_only']['tps_central']:.1f}TPS; both {p['both_bugs']['tree_drop_pct']:.2f}% "
              f"-> {p['both_bugs']['tps_central']:.1f}TPS", flush=True)
    print(f"  [PRIMARY] native_proxies_reproduce_flagship_4p3 : {native_proxies_reproduce_flagship_4p3}", flush=True)
    print(f"  [TEST]    tree_private_drop_pct_native_ci (descent): {_clean(desc_drop['mid'])} "
          f"band [{_clean(desc_drop['min'])}, {_clean(desc_drop['max'])}]", flush=True)
    print(f"  descent-only TPS band [{_clean(desc_tps['min'])}, {_clean(desc_tps['max'])}] "
          f"(worst tau-low {_clean(desc_worst_taulow)})", flush=True)
    print(f"  both-bugs   TPS band [{_clean(both_tps['min'])}, {_clean(both_tps['max'])}]", flush=True)
    print(f"  DECISION: descent_only_private_safe_native={descent_only_private_safe_native} ; "
          f"both_bugs_required_private={both_bugs_required_private}", flush=True)
    print("=" * 76 + "\n", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, results, verdict, proxies_rows=proxies_out)
        except Exception as e:  # noqa: BLE001
            print(f"[native] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[native] DONE", flush=True)
    return 0


def _log_wandb(args, results, verdict, proxies_rows):
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group=args.wandb_group, name=args.wandb_name, job_type="analysis",
        config={"step": args.step, "gt_drop": args.gt_drop, "K_cal": K_CAL,
                "bug1_mult": BUG1_MULT, "safe_margin": args.safe_margin,
                "native_pending": results.get("native_pending", True),
                "n_native_proxies": len(proxies_rows)})
    summ = {f"verdict/{k}": v for k, v in verdict.items() if not isinstance(v, (dict, list))}
    run.summary.update(summ)
    if proxies_rows:
        tbl = wandb.Table(columns=[
            "name", "axis", "mode", "native_decode_drop_pct", "reproduces_4p3",
            "descent_tree_drop_pct", "descent_tps", "descent_clears_500",
            "both_tree_drop_pct", "both_tps", "both_clears_500"])
        for p in proxies_rows:
            tbl.add_data(p["name"], p["axis"], p["mode"], p["native_decode_linear_drop_pct"],
                         int(p["reproduces_flagship_4p3"]), p["descent_only"]["tree_drop_pct"],
                         p["descent_only"]["tps_central"], int(p["descent_only"]["clears_500_central"]),
                         p["both_bugs"]["tree_drop_pct"], p["both_bugs"]["tps_central"],
                         int(p["both_bugs"]["clears_500_central"]))
        run.log({"native_proxy_sweep": tbl})
    run.summary["wandb_run_id"] = run.id
    print(f"[native] W&B run: {run.url}  (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
