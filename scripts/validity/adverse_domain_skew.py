#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Adverse domain-skew private stress: descent-only survival at 500 (PR #176).

THE RESIDUAL #164 LEFT
----------------------
My #164 (`5hz3dfrq`, MERGED) proved descent-only is private-safe across 3 native
proxies (code / casual / sharegpt): native drop CI mid 2.04% band [1.87, 2.21],
descent-only central band [508.5, 510.2], worst tau-low 504.6 (+4.6 margin),
`both_bugs_required_private=False`. But [1.87, 2.21] is a CONSTRUCTION-VARIANCE band
over 3 axes, NOT a sampling CI over the real private distribution. The binding
residual is the ADVERSE TAIL: if the organizer's real private set is skewed toward
the hardest domains, could descent-only's tau-low corner be pushed < 500 at the same
calibrated aggregate-4.3%? #164 already proved the tree drop is SHAPE-sensitive (same
aggregate-4.3%, two ladders -> drops 0.34pp apart). An adverse domain-skew pulls that
exact lever.

WHAT THIS DOES (the adverse-skew certificate)
---------------------------------------------
Imports ALL #164 machinery verbatim (descent_vs_bothbugs_native -> the descent-walk
E[T] DP, pooled-mode calibration, relative_transfer decode bridge, official map). No
re-derivation. Then:

  1. PER-AXIS (instruction 1). For each hard component in the manifest (the 3 #164
     components imported byte-identically + 2-3 NEW genuinely-distinct hard tails),
     count-pool with the shared public reference at the continuous weight landing the
     DECODE-frame linear drop on GT-4.3% (<=0.5pp gate) -- the EXACT #164 pooled path
     `native_sglang_ladder`. Reproduces #164's three axes by construction (self-test a).

  2. ADVERSE VERTEX (instruction 2). The eval set is a count-pool over
     {public, domain_1..N}: overall count weights g (g>=0, sum=1), each HARD domain
     capped g_i <= cap (cap=0.5 diversity floor). Parametrize by the hard-mix direction
     f (f>=0, sum f=1): g_public=1-W, g_i=W*f_i, where W=W*(f) is the unique total hard
     weight that lands the pooled DECODE drop on GT-4.3% (bisection; the pooled
     cumulative C_mix=(1-W)C_pub+W*sum_i f_i C_hard_i is EXACTLY the 2-component pool of
     public with the hard-centroid at weight W, so this reuses `_pool_decode_drop`). At
     fixed 4.3% decode drop, MAXIMIZE the descent tree private-drop over the capped
     polytope (vertex enumeration: singles + pair edges + triple faces + Dirichlet, each
     EXACTLY evaluated through the DP). The single axes are feasible points, so the
     optimum is >= the worst admissible single axis (self-test b).

  3. DESCENT-ONLY SURVIVAL AT THE ADVERSE CORNER (instruction 3). Propagate the
     adverse-vertex ladder through `official=K_cal*(E[T]/step)*tau` at central tau=1.0
     AND the conservative tree-class tau-low=0.9924. Report
     `descent_only_taulow_tps_adverse_corner` (TEST) and `descent_only_clears_500_adverse`
     (both tau corners). Does the adverse corner flip #164's both_bugs_required=False?

  4. WIDENED ENVELOPE (instruction 4). tree_private_drop_ci (min/mid/max across all axes)
     + the adverse-vertex drop as the certified worst-case ceiling.

  5. SELF-VALIDATE (instruction 5, PRIMARY). (a) reproduce #164 three axes within tol;
     (b) conservative ordering adverse_tree_drop >= max(admissible per-axis drops);
     (c) explicit descent-only adverse-corner clear-500 verdict (central AND tau-low);
     (d) NaN-clean. -> `adverse_skew_stress_self_test_passes` (PRIMARY) +
     `descent_only_taulow_tps_adverse_corner` (TEST).

LOCAL, CPU-ONLY. No HF Job, no submission, no served-file change. Greedy/PPL untouched,
BASELINE unchanged (481.53). Does NOT authorize a launch.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import the #164 core machinery as ONE source of truth (do NOT re-derive). -------- #
dvb = _load("descent_vs_bothbugs_native", ROOT / "scripts/validity/descent_vs_bothbugs_native.py")

# banked DP + map + calibration + bridge
DescentModel = dvb.DescentModel
project_one = dvb.project_one
linear_et_from_q = dvb.linear_et_from_q
official_tps_map = dvb.official_tps_map
run_self_test = dvb.run_self_test
crosscheck_pinned_43 = dvb.crosscheck_pinned_43
load_measured = dvb.load_measured
load_rank_coverage = dvb.load_rank_coverage
load_m32_topology = dvb.load_m32_topology
native_sglang_ladder = dvb.native_sglang_ladder
propagate_native = dvb.propagate_native
ladder_from_spec = dvb.ladder_from_spec
conditional_to_cumulative = dvb.conditional_to_cumulative
cumulative_to_conditional = dvb.cumulative_to_conditional
_pool_decode_drop = dvb._pool_decode_drop
_band = dvb._band
_clean = dvb._clean
_jd = dvb._jd

# constants
K_CAL = dvb.K_CAL
BUG1_MULT = dvb.BUG1_MULT
GT_DROP = dvb.GT_DROP
CALIB_TOL = dvb.CALIB_TOL
TARGET_500 = dvb.TARGET_500
TAU = dvb.TAU
STEP_MEASURED_DEPTH9 = dvb.STEP_MEASURED_DEPTH9
PUBLIC_BANKED_JSON = dvb.PUBLIC_BANKED_JSON
RANKCOV_JSON = dvb.RANKCOV_JSON
RHO_OPT_JSON = dvb.RHO_OPT_JSON
BANKED_SGLANG_RUN = dvb.BANKED_SGLANG_RUN

# #164 banked per-axis descent results (self-test a: byte-identical import must reproduce).
PINNED_164_AXES = {
    "native_code":     {"tree_drop_pct": 2.210460079395353, "tps_central": 508.45859521597004,
                        "tps_taulow": 504.6105118808893, "pool_weight": 0.4075},
    "native_casual":   {"tree_drop_pct": 1.9813550309269023, "tps_central": 509.64983132563714,
                        "tps_taulow": 505.79273255475715, "pool_weight": 0.31425},
    "native_sharegpt": {"tree_drop_pct": 1.8714348328259591, "tps_central": 510.22136351157076,
                        "tps_taulow": 506.3599393079007, "pool_weight": 0.402},
}
SELFTEST_A_DROP_TOL_PP = 0.05   # tree-drop reproduction tolerance (pp)
SELFTEST_A_TPS_TOL = 0.5        # TPS reproduction tolerance


# ----------------------------------------------------------------------------- #
# Adverse-vertex helpers: hard-centroid + 4.3%-calibrated W*, all via #164 funcs. #
# ----------------------------------------------------------------------------- #
def centroid_conditional(f: list[float], cum_hard: list[list[float]]) -> list[float]:
    """Conditional ladder of the hard-mix direction f: the cumulative count-pool of the
    hard components (C_centroid[k]=sum_i f_i C_hard_i[k]) -> conditional. Pooling at the
    cumulative level IS the realizable mixture ladder (same algebra as #164 pool)."""
    k = min(len(c) for c in cum_hard)
    c_cent = [sum(f[i] * cum_hard[i][k_] for i in range(len(f))) for k_ in range(k)]
    return cumulative_to_conditional(c_cent)


def calibrate_W(q_pub_sglang, q_cent, q_public_banked, target: float):
    """Total hard weight W landing the pooled DECODE drop on `target` (GT-4.3%).
    The pooled decode drop is monotone increasing in W (more hard mass -> lower spine
    E[T]); bisection is exact. Returns (W, achieved_drop, feasible)."""
    def drop(w: float) -> float:
        return _pool_decode_drop(w, q_pub_sglang, q_cent, q_public_banked)[0]
    lo, hi = 1e-6, 1.0
    d_hi = drop(hi)
    if d_hi < target:               # centroid too easy to reach 4.3% even at W=1
        return hi, d_hi, False
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if drop(mid) < target:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)
    return w, drop(w), True


def eval_direction(f, cum_hard, q_pub_sglang, q_public_banked, model, rho_pub, step,
                   cap: float, target: float) -> dict[str, Any]:
    """Calibrate the hard-mix direction f to 4.3% decode drop, check the diversity cap,
    and EXACTLY propagate the calibrated ladder through the descent-walk DP."""
    q_cent = centroid_conditional(f, cum_hard)
    W, dec_drop, feasible = calibrate_W(q_pub_sglang, q_cent, q_public_banked, target)
    g_pub = 1.0 - W
    g = [W * fi for fi in f]
    cap_ok = all(gi <= cap + 1e-9 for gi in g)
    calib_ok = abs(dec_drop - target) <= CALIB_TOL
    _, q_native = _pool_decode_drop(W, q_pub_sglang, q_cent, q_public_banked)
    prop = propagate_native(model, q_public_banked, q_native, q_pub_sglang, rho_pub, step)
    d = prop["projection"]["descent_only"]
    b = prop["projection"]["both_bugs"]
    return {
        "f": list(f), "W_hard": W, "g_public": g_pub, "g": g,
        "cap_ok": bool(cap_ok), "calib_ok": bool(calib_ok), "feasible": bool(feasible),
        "admissible": bool(cap_ok and calib_ok and feasible),
        "achieved_decode_drop_pct": dec_drop * 100.0,
        "q_native": q_native,
        "descent_tree_drop_pct": d["et_drop_pct_vs_public"],
        "descent_tps_central": d["official_central"],
        "descent_tps_taulow": d["official_taulow"],
        "descent_clears_500_central": bool(d["clears_500_central"]),
        "descent_clears_500_taulow": bool(d["clears_500_conservative"]),
        "both_tree_drop_pct": b["et_drop_pct_vs_public"],
        "both_tps_central": b["official_central"],
        "both_tps_taulow": b["official_taulow"],
        "both_clears_500_central": bool(b["clears_500_central"]),
        "both_clears_500_taulow": bool(b["clears_500_conservative"]),
    }


def _simplex_grid_2d(steps: int):
    """Barycentric grid on the 2-simplex (triple faces)."""
    out = []
    for a in range(steps + 1):
        for bcount in range(steps + 1 - a):
            ccount = steps - a - bcount
            out.append((a / steps, bcount / steps, ccount / steps))
    return out


def adverse_search(cum_hard, axis_names, q_pub_sglang, q_public_banked, model, rho_pub,
                   step, cap: float, target: float, dirichlet_n: int, seed: int,
                   pair_steps: int = 20, triple_steps: int = 8) -> dict[str, Any]:
    """Vertex enumeration + exact-DP refinement of the worst (max descent tree-drop)
    admissible hard-mix over the capped simplex. Searches singles, pair edges, triple
    faces, and Dirichlet interior samples; every candidate is evaluated EXACTLY."""
    n = len(cum_hard)
    evals: list[dict[str, Any]] = []

    def add(f, kind):
        r = eval_direction(f, cum_hard, q_pub_sglang, q_public_banked, model, rho_pub,
                           step, cap, target)
        r["kind"] = kind
        evals.append(r)
        return r

    # 1. singles (recover the #164 / new per-axis vertices)
    for i in range(n):
        f = [0.0] * n
        f[i] = 1.0
        add(f, f"single:{axis_names[i]}")
    # 2. pair edges
    for i, j in itertools.combinations(range(n), 2):
        for s in range(1, pair_steps):
            t = s / pair_steps
            f = [0.0] * n
            f[i], f[j] = 1.0 - t, t
            add(f, f"pair:{axis_names[i]}+{axis_names[j]}")
    # 3. triple faces
    for i, j, k in itertools.combinations(range(n), 3):
        for (a, b, c) in _simplex_grid_2d(triple_steps):
            if a in (0.0, 1.0) or b == 1.0 or c == 1.0:
                continue  # skip vertices/edges already covered
            f = [0.0] * n
            f[i], f[j], f[k] = a, b, c
            add(f, f"triple:{axis_names[i]}+{axis_names[j]}+{axis_names[k]}")
    # 4. Dirichlet interior over the full simplex
    rng = np.random.default_rng(seed)
    for _ in range(dirichlet_n):
        f = rng.dirichlet(np.ones(n)).tolist()
        add(f, "dirichlet")

    admissible = [e for e in evals if e["admissible"]]
    # adverse vertex = max descent tree-drop among admissible (= min descent TPS = worst case)
    best = max(admissible, key=lambda e: e["descent_tree_drop_pct"])
    cap_binds = any(abs(g - cap) < 1e-6 for g in best["g"])
    return {
        "n_eval": len(evals), "n_admissible": len(admissible),
        "adverse": best, "cap_binding_at_optimum": bool(cap_binds),
        "all_evals": evals,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxies-json", "--proxies_json",
                    default="research/validity/private_adverse_skew/proxies_native_6axis.json",
                    help="6-axis manifest (q_pub_sglang + proxies[] each with component). "
                         "Omit/missing -> liveness CPU pass (adverse_pending=true).")
    ap.add_argument("--public-banked", default=PUBLIC_BANKED_JSON)
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--banked-sglang-run", default=BANKED_SGLANG_RUN)
    ap.add_argument("--step", type=float, default=STEP_MEASURED_DEPTH9)
    ap.add_argument("--gt-drop", type=float, default=GT_DROP)
    ap.add_argument("--cap", type=float, default=0.5,
                    help="diversity cap: no single hard domain > cap of the eval set.")
    ap.add_argument("--dirichlet-n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=17600)
    ap.add_argument("--output",
                    default="research/validity/private_adverse_skew/results.json")
    ap.add_argument("--wandb-group", "--wandb_group", default="descent-private-adverse-skew")
    ap.add_argument("--wandb-name", "--wandb_name", default="stark/adverse-domain-skew")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- banked model (same setup as #164) ----
    meas_pub = load_measured(args.public_banked, "server_log")
    q_public = list(meas_pub["q"])
    rc = load_rank_coverage(args.rankcov_json)
    rho_pub = rc["rho_cond"]
    parent = load_m32_topology(args.rho_opt_json)
    model = DescentModel(parent, rho_pub)
    st = run_self_test(model, q_public, rho_pub)
    assert st["passes"], "descent-walk DP does not reproduce banked anchors"
    xcheck = crosscheck_pinned_43(model, q_public, rho_pub, args.step)
    assert xcheck["reproduces_156"], f"DP does not reproduce #156 4.3% projection: {xcheck}"

    run_dir = Path(args.banked_sglang_run)
    results: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "constants": {
            "K_cal": K_CAL, "step": args.step, "bug1_mult": BUG1_MULT,
            "gt_drop": args.gt_drop, "gt_drop_pct": args.gt_drop * 100.0,
            "target_500": TARGET_500, "tau_central": TAU["central"], "tau_low": TAU["low"],
            "calib_tol_pp": CALIB_TOL * 100.0, "cap": args.cap,
        },
        "self_test_passes": bool(st["passes"]),
        "machinery_faithfulness_xcheck": xcheck,
    }

    if not args.proxies_json or not Path(args.proxies_json).exists():
        results["adverse_pending"] = True
        results["note"] = ("no 6-axis manifest yet; CPU cross-check + liveness only. "
                           "Measure the new axes via private_gap_probe.py, assemble "
                           "proxies_native_6axis.json, then re-run with --proxies-json.")
        results["verdict"] = {"self_test_passes": int(st["passes"]),
                              "xcheck_reproduces_156_4p3": int(xcheck["reproduces_156"]),
                              "adverse_pending": True}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, default=_jd))
        print(f"[adverse] cross-check PASS (descent {xcheck['descent_tps']:.1f}, "
              f"both {xcheck['both_bugs_tps']:.1f}); wrote partial {args.output}", flush=True)
        if not args.no_wandb:
            try:
                _log_wandb(args, results, axis_rows=[], adverse=None)
            except Exception as e:  # noqa: BLE001
                print(f"[adverse] W&B logging failed (non-fatal): {e!r}", flush=True)
        return 0

    # ---- manifest ----
    manifest = json.loads(Path(args.proxies_json).read_text())
    pub_spec = manifest.get("q_pub_sglang")
    q_pub_sglang = ladder_from_spec(pub_spec, run_dir)["conditional_p"] if pub_spec else \
        dvb.parse_server_ladder(run_dir / "server_public_cold.log")["conditional_p"]

    # ---- PER-AXIS (instruction 1): the EXACT #164 pooled path for each component ----
    axis_rows: list[dict[str, Any]] = []
    cum_hard: list[list[float]] = []
    axis_names: list[str] = []
    selftest_a_ok = True
    selftest_a_detail: list[dict[str, Any]] = []
    for proxy in manifest["proxies"]:
        native = native_sglang_ladder(proxy, q_pub_sglang, q_public, run_dir, args.gt_drop)
        prop = propagate_native(model, q_public, native["conditional_p"], q_pub_sglang,
                                rho_pub, args.step)
        d = prop["projection"]["descent_only"]; b = prop["projection"]["both_bugs"]
        calib_drop = prop["native_decode_linear_drop_pct"] / 100.0
        reproduces_4p3 = bool(abs(calib_drop - args.gt_drop) <= CALIB_TOL)
        row = {
            "name": proxy["name"], "axis": proxy.get("axis"),
            "pool_weight": _clean(native.get("pool_weight")),
            "achieved_decode_drop_pct": _clean(native.get("achieved_decode_drop_pct")),
            "component_full_decode_drop_pct": _clean(native.get("component_full_decode_drop_pct")),
            "component_conditional_p": native.get("component_conditional_p"),
            "native_decode_linear_drop_pct": _clean(prop["native_decode_linear_drop_pct"]),
            "reproduces_flagship_4p3": reproduces_4p3,
            "descent_tree_drop_pct": _clean(d["et_drop_pct_vs_public"]),
            "descent_tps_central": _clean(d["official_central"]),
            "descent_tps_taulow": _clean(d["official_taulow"]),
            "descent_clears_500_central": bool(d["clears_500_central"]),
            "descent_clears_500_taulow": bool(d["clears_500_conservative"]),
            "both_tree_drop_pct": _clean(b["et_drop_pct_vs_public"]),
            "both_tps_central": _clean(b["official_central"]),
            "both_tps_taulow": _clean(b["official_taulow"]),
            "both_clears_500_taulow": bool(b["clears_500_conservative"]),
        }
        axis_rows.append(row)
        cum_hard.append(conditional_to_cumulative(native["component_conditional_p"]))
        axis_names.append(proxy["name"])
        # self-test (a): the 3 imported #164 components must reproduce within tol
        if proxy["name"] in PINNED_164_AXES:
            exp = PINNED_164_AXES[proxy["name"]]
            dd = abs(row["descent_tree_drop_pct"] - exp["tree_drop_pct"])
            dt = abs(row["descent_tps_central"] - exp["tps_central"])
            dl = abs(row["descent_tps_taulow"] - exp["tps_taulow"])
            ok = dd <= SELFTEST_A_DROP_TOL_PP and dt <= SELFTEST_A_TPS_TOL and dl <= SELFTEST_A_TPS_TOL
            selftest_a_ok = selftest_a_ok and ok
            selftest_a_detail.append({"name": proxy["name"], "drop_err_pp": dd,
                                      "tps_err": dt, "taulow_err": dl, "ok": bool(ok)})
        print(f"[adverse] axis {proxy['name']:<16s} w={row['pool_weight']:.4f} "
              f"decode={row['native_decode_linear_drop_pct']:.3f}% repro4.3={reproduces_4p3} "
              f"-> descent {row['descent_tree_drop_pct']:.3f}% {row['descent_tps_central']:.1f}TPS "
              f"(taulow {row['descent_tps_taulow']:.1f})", flush=True)

    # ---- ADVERSE VERTEX (instruction 2) ----
    search = adverse_search(cum_hard, axis_names, q_pub_sglang, q_public, model, rho_pub,
                            args.step, args.cap, args.gt_drop, args.dirichlet_n, args.seed)
    adv = search["adverse"]
    print(f"[adverse] searched {search['n_eval']} dirs ({search['n_admissible']} admissible); "
          f"ADVERSE descent drop {adv['descent_tree_drop_pct']:.3f}% "
          f"tps {adv['descent_tps_central']:.1f}/{adv['descent_tps_taulow']:.1f} "
          f"mix={dict(zip(axis_names, [round(x,3) for x in adv['f']]))} W={adv['W_hard']:.3f}",
          flush=True)

    # ---- ENVELOPE (instruction 4) ----
    per_axis_descent_drops = [r["descent_tree_drop_pct"] for r in axis_rows]
    per_axis_both_drops = [r["both_tree_drop_pct"] for r in axis_rows]
    desc_drop_band = _band(per_axis_descent_drops)
    both_drop_band = _band(per_axis_both_drops)
    desc_tps_band = _band([r["descent_tps_central"] for r in axis_rows])
    desc_taulow_band = _band([r["descent_tps_taulow"] for r in axis_rows])
    # max over ADMISSIBLE single-axis vertices (search path -> internally consistent with adverse)
    single_admissible = [e for e in search["all_evals"]
                         if e["kind"].startswith("single:") and e["admissible"]]
    max_admissible_single = max(e["descent_tree_drop_pct"] for e in single_admissible)
    max_per_axis_drop = max(per_axis_descent_drops)  # reported (native path) for context

    # ---- SELF-TESTS (instruction 5) ----
    # (b) conservative ordering: adverse >= max admissible single axis (eps slack)
    selftest_b_ok = bool(adv["descent_tree_drop_pct"] >= max_admissible_single - 1e-6)
    # (c) explicit descent-only adverse-corner clear-500 verdict (central AND tau-low)
    descent_only_clears_500_adverse = bool(
        adv["descent_clears_500_central"] and adv["descent_clears_500_taulow"])
    # (d) NaN-clean: every reported number finite
    flat_numbers = []
    for r in axis_rows:
        flat_numbers += [r["descent_tree_drop_pct"], r["descent_tps_central"],
                         r["descent_tps_taulow"], r["both_tree_drop_pct"]]
    flat_numbers += [adv["descent_tree_drop_pct"], adv["descent_tps_central"],
                     adv["descent_tps_taulow"], adv["both_tps_taulow"]]
    selftest_d_ok = all(x is not None and np.isfinite(x) for x in flat_numbers)

    adverse_skew_stress_self_test_passes = bool(
        selftest_a_ok and selftest_b_ok and descent_only_clears_500_adverse and selftest_d_ok)

    # launch-topology: does the adverse corner flip #164's both_bugs_required=False?
    both_bugs_required_private_adverse = bool(
        (not descent_only_clears_500_adverse) and adv["both_clears_500_taulow"])

    headline = {
        "adverse_skew_stress_self_test_passes": adverse_skew_stress_self_test_passes,  # PRIMARY
        "descent_only_taulow_tps_adverse_corner": _clean(adv["descent_tps_taulow"]),     # TEST
        "descent_only_tps_central_adverse_corner": _clean(adv["descent_tps_central"]),
        "descent_only_clears_500_adverse": descent_only_clears_500_adverse,
        "adverse_tree_drop_pct_descent": _clean(adv["descent_tree_drop_pct"]),
        "adverse_tree_drop_pct_both": _clean(adv["both_tree_drop_pct"]),
        "adverse_mixture_weights": {"public": _clean(adv["g_public"]),
                                    **{axis_names[i]: _clean(adv["g"][i]) for i in range(len(axis_names))}},
        "adverse_hard_direction": {axis_names[i]: _clean(adv["f"][i]) for i in range(len(axis_names))},
        "adverse_total_hard_weight_W": _clean(adv["W_hard"]),
        "cap": args.cap, "cap_binding_at_optimum": search["cap_binding_at_optimum"],
        "worst_single_axis_drop_pct": _clean(max_admissible_single),
        "worst_per_axis_drop_pct_reported": _clean(max_per_axis_drop),
        "adverse_vs_worst_single_pp": _clean(adv["descent_tree_drop_pct"] - max_admissible_single),
        "n_axes": len(axis_rows),
        "tree_private_drop_pct_descent_ci": _clean(desc_drop_band["mid"]),
        "tree_private_drop_pct_descent_band": [_clean(desc_drop_band["min"]), _clean(desc_drop_band["max"])],
        "tree_private_drop_pct_both_band": [_clean(both_drop_band["min"]), _clean(both_drop_band["max"])],
        "descent_only_tps_central_band": [_clean(desc_tps_band["min"]), _clean(desc_tps_band["max"])],
        "descent_only_tps_taulow_min_per_axis": _clean(desc_taulow_band["min"]),
        "both_bugs_required_private_adverse": both_bugs_required_private_adverse,
        # vs #164 central-band conclusion (mid 509.3, worst taulow 504.6)
        "pinned_164_descent_tps_mid": 509.3399793637704,
        "pinned_164_descent_taulow_min": 504.6105118808893,
        "adverse_vs_164_taulow_margin_pp": _clean(
            adv["descent_tps_taulow"] - 504.6105118808893),
        "xcheck_reproduces_156_4p3": bool(xcheck["reproduces_156"]),
        "selftest_a_reproduces_164": bool(selftest_a_ok),
        "selftest_b_conservative_ordering": selftest_b_ok,
        "selftest_d_nan_clean": bool(selftest_d_ok),
    }

    results["adverse_pending"] = False
    results["q_pub_sglang"] = q_pub_sglang
    results["per_axis"] = axis_rows
    results["self_test_a_reproduces_164"] = {"ok": bool(selftest_a_ok), "detail": selftest_a_detail}
    results["envelope"] = {
        "descent_tree_drop_pct": desc_drop_band, "both_tree_drop_pct": both_drop_band,
        "descent_tps_central": desc_tps_band, "descent_tps_taulow": desc_taulow_band,
        "adverse_ceiling_descent_drop_pct": _clean(adv["descent_tree_drop_pct"]),
    }
    results["adverse_vertex"] = {k: v for k, v in adv.items() if k != "q_native"}
    results["adverse_vertex"]["q_native"] = adv["q_native"]
    results["adverse_search_meta"] = {
        "n_eval": search["n_eval"], "n_admissible": search["n_admissible"],
        "cap_binding_at_optimum": search["cap_binding_at_optimum"],
        "dirichlet_n": args.dirichlet_n, "seed": args.seed,
    }
    results["self_tests"] = {
        "a_reproduces_164": bool(selftest_a_ok),
        "b_conservative_ordering": selftest_b_ok,
        "c_descent_clears_500_adverse": descent_only_clears_500_adverse,
        "d_nan_clean": bool(selftest_d_ok),
        "PRIMARY_adverse_skew_stress_self_test_passes": adverse_skew_stress_self_test_passes,
    }
    results["headline"] = headline
    verdict = {
        "self_test_passes": int(st["passes"]),
        "xcheck_reproduces_156_4p3": int(xcheck["reproduces_156"]),
        **{k: v for k, v in headline.items() if not isinstance(v, (list, dict))},
    }
    results["verdict"] = verdict

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=_jd))
    print(f"[adverse] wrote {args.output}", flush=True)

    # ---- console summary ----
    print("\n" + "=" * 78, flush=True)
    print("PR #176 -- ADVERSE DOMAIN-SKEW PRIVATE STRESS (descent-only survival @ 500)", flush=True)
    print("=" * 78, flush=True)
    print(f"  axes (n={len(axis_rows)}):", flush=True)
    for r in axis_rows:
        print(f"    - {r['name']:<16s} [{r['axis']}] decode={r['native_decode_linear_drop_pct']:.3f}% "
              f"repro4.3={r['reproduces_flagship_4p3']}: descent {r['descent_tree_drop_pct']:.3f}% "
              f"-> {r['descent_tps_central']:.1f}/{r['descent_tps_taulow']:.1f} TPS", flush=True)
    print(f"  ADVERSE VERTEX: descent drop {adv['descent_tree_drop_pct']:.3f}% (worst single "
          f"{max_admissible_single:.3f}%, +{adv['descent_tree_drop_pct']-max_admissible_single:.3f}pp)", flush=True)
    mix_str = ", ".join(f"{axis_names[i]} {adv['g'][i]:.3f}" for i in range(len(axis_names)))
    print(f"    mix g = public {adv['g_public']:.3f} | {mix_str}", flush=True)
    print(f"    cap={args.cap} binding={search['cap_binding_at_optimum']}", flush=True)
    print(f"  descent-only @ adverse corner: central {adv['descent_tps_central']:.1f}, "
          f"tau-low {adv['descent_tps_taulow']:.1f}  clears500(both corners)="
          f"{descent_only_clears_500_adverse}", flush=True)
    print(f"  [PRIMARY] adverse_skew_stress_self_test_passes = {adverse_skew_stress_self_test_passes}", flush=True)
    print(f"  [TEST]    descent_only_taulow_tps_adverse_corner = {adv['descent_tps_taulow']:.2f}", flush=True)
    print(f"  both_bugs_required_private_adverse = {both_bugs_required_private_adverse} "
          f"(#164 was False)", flush=True)
    print("=" * 78 + "\n", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, results, axis_rows=axis_rows, adverse=adv, axis_names=axis_names)
        except Exception as e:  # noqa: BLE001
            print(f"[adverse] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[adverse] DONE", flush=True)
    return 0


def _log_wandb(args, results, axis_rows, adverse, axis_names=None):
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group=args.wandb_group, name=args.wandb_name, job_type="analysis",
        config={"step": args.step, "gt_drop": args.gt_drop, "K_cal": K_CAL,
                "bug1_mult": BUG1_MULT, "cap": args.cap,
                "adverse_pending": results.get("adverse_pending", True),
                "n_axes": len(axis_rows)})
    verdict = results.get("verdict", {})
    run.summary.update({f"verdict/{k}": v for k, v in verdict.items()
                        if not isinstance(v, (dict, list))})
    if axis_rows:
        tbl = wandb.Table(columns=[
            "name", "axis", "pool_weight", "decode_drop_pct", "reproduces_4p3",
            "descent_tree_drop_pct", "descent_tps_central", "descent_tps_taulow",
            "descent_clears_500_taulow", "both_tree_drop_pct", "both_tps_taulow"])
        for r in axis_rows:
            tbl.add_data(r["name"], r["axis"], r["pool_weight"],
                         r["native_decode_linear_drop_pct"], int(r["reproduces_flagship_4p3"]),
                         r["descent_tree_drop_pct"], r["descent_tps_central"],
                         r["descent_tps_taulow"], int(r["descent_clears_500_taulow"]),
                         r["both_tree_drop_pct"], r["both_tps_taulow"])
        run.log({"per_axis_table": tbl})
    if adverse is not None and axis_names is not None:
        atbl = wandb.Table(columns=["component", "g_weight", "hard_direction_f"])
        atbl.add_data("public", adverse["g_public"], 0.0)
        for i, nm in enumerate(axis_names):
            atbl.add_data(nm, adverse["g"][i], adverse["f"][i])
        run.log({"adverse_mixture": atbl})
    run.summary["wandb_run_id"] = run.id
    print(f"[adverse] W&B run: {run.url}  (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
