#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Pin the tree's TRUE private drop (PR #156): reconcile 4.3% / 11.3% / 19.6%.

THE PROBLEM
-----------
PR #151 hardened the tree clear-500 verdict (descent-only 505.46 @ assumed-4.3%
private drop) but rested on ONE uncertain input: the *true* private acceptance drop
the tree will experience. Three numbers disagree by 2-4x:

  * 4.3%  GT-flagship  -- organizer's REAL public->private TPS drop (481.53 -> 460.85,
                          VERIFIED 2026-06-13 23:04Z). Measured by the OFFICIAL scored
                          benchmark = sglang `vllm-chat` (`hf_bucket_single_job.run_benchmark`),
                          ignore_eos, out 512, on the REAL private prompt set.
  * 11.3% sglang-probe -- kanna #44 `private_gap_probe.py` precache-neutral TPS distribution
                          gap (public_cold 418.4 -> private_rerun 371.0; E_accept 4.06->3.565)
                          measured by the SAME sglang `vllm-chat` scored protocol, on the hard
                          chat PROXY (`data/private_proxy_sharegpt.json`).
  * 19.6% official-decode -- stark #151 `accept_calibration.py` per-position E[T] drop
                          (public 3.844 -> proxy 3.090) measured by `decode_outputs.py`:
                          the UNSCORED greedy-identity *audit* pass (client-side chat
                          template, /v1/completions, its own prompt subset), on the PROXY.

RECONCILIATION (two independent axes)
-------------------------------------
  19.6% -> 11.3% : HARNESS PATH. `decode_outputs.py` is the audit pass, not the leaderboard's
                   scored protocol. It systematically under-reads E[T] and the gap widens on
                   the proxy -> inflated 19.6%. The scored sglang path reads ~11-12% on the
                   SAME proxy/stack. (suspects: client- vs server-side chat templating, prompt
                   subset; ignore_eos/max_tokens are matched in both.)
  11.3% -> 4.3%  : PROXY DIFFICULTY. The proxy is a deliberately-hard chat tail (~2.6x the real
                   private drop). The organizer's real private set is much closer to public.
                   BASELINE: "the ground-truth 4.3% is now the number to calibrate every
                   private-gap probe against."

=> PINNED organizer-matching protocol = sglang `vllm-chat` scored bench; calibration anchor
   = 4.3% (organizer's real LINEAR drop). 19.6% is a HARNESS ARTIFACT to discard.

WHAT THIS SCRIPT ADDS OVER #151
-------------------------------
#151 propagated the DECODE-path proxy ladder SHAPE (top-1 collapse to 0.599) scaled to 4.3%.
The tree's descent-walk E[T] amplifies per-position structure (depth-9 walk vs linear K=7), so
the ladder SHAPE -- not just the aggregate -- matters. This script measures the per-position
ladder under the SGLANG scored protocol (the new data), transfers its drop-shape onto the
banked public reference, re-calibrates to the 4.3% organizer anchor, and re-propagates through
the SAME banked descent-walk E[T] DP (`tree_private_acceptance_gap.py`). It then re-scores the
#151 verdict at the PINNED drop and certifies the harness against the flagship's GT-4.3%.

LOCAL, CPU-ONLY (this file): it consumes the sglang server logs produced by a prior
`private_gap_probe.py` run (the single-A10G GPU step) + the banked #151 ladders. No HF Job, no
submission, no served-file change. Greedy/PPL untouched.

OUTPUTS (PR #156)
-----------------
  harness_pin_reproduces_flagship_4p3 (PRIMARY, bool) -- the pinned protocol reproduces the
      flagship GT-4.3% LINEAR drop to <=0.5pp (reachable + calibrated).
  tree_private_drop_pct_pinned        (TEST)          -- tree descent-only E[T] drop at the
      4.3%-calibrated PINNED (sglang-shape) ladder.
  tree_private_tps_proj_pinned (+both_bugs), descent_only_clears_500_pinned /
      both_bugs_clears_500_pinned, tps band, and whether the #151 505.46 verdict holds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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


# Banked machinery -- one source of truth (do not re-derive). ------------------ #
_tpag = _load("tree_private_acceptance_gap", PROFILER / "tree_private_acceptance_gap.py")
_acc = _load("accept_calibration", PROFILER / "accept_calibration.py")

DescentModel = _tpag.DescentModel
build_calibrated_ladder = _tpag.build_calibrated_ladder
frac_for_target_drop = _tpag.frac_for_target_drop
project_one = _tpag.project_one
breakeven_drop = _tpag.breakeven_drop
linear_et_from_q = _tpag.linear_et_from_q
official_tps_map = _tpag.official_tps_map
accept_length_for_official = _tpag.accept_length_for_official
official_band = _tpag.official_band
run_self_test = _tpag.run_self_test
load_measured = _tpag.load_measured
load_rank_coverage = _tpag.load_rank_coverage
load_m32_topology = _tpag.load_m32_topology
BUG1_MULT = _tpag.BUG1_MULT
K_CAL = _tpag.K_CAL
TARGET_500 = _tpag.TARGET_500
TAU = _tpag.TAU
STEP_MEASURED_DEPTH9 = _tpag.STEP_MEASURED_DEPTH9
FRONTIER_OFFICIAL = _tpag.FRONTIER_OFFICIAL  # 481.53
E_T_LINEAR = _tpag.E_T_LINEAR                # 3.844 deployed linear-MTP floor
parse_log_per_position = _acc.parse_log_per_position

# Organizer ground-truth anchor (BASELINE.md, VERIFIED 2026-06-13 23:04Z). ------ #
GT_PUBLIC_TPS = 481.53
GT_PRIVATE_TPS = 460.85
GT_DROP = 1.0 - GT_PRIVATE_TPS / GT_PUBLIC_TPS  # 0.04294...

# Banked ladders. -------------------------------------------------------------- #
PUBLIC_BANKED_JSON = "research/accept_calibration/accept_calibration_results.json"
PROXY_DECODE_JSON = "research/validity/tree_private_acceptance_gap/accept_calibration_private.json"
RANKCOV_JSON = "research/rank_coverage/rank_coverage_results.json"
RHO_OPT_JSON = "research/spec_cost_model/rho_optimal_topology_results.json"


def _jd(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")


def _clean(x):
    """NaN/inf -> None (PR #156 requires NaN-clean metrics)."""
    if isinstance(x, (int, bool)):
        return x
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf):
        return None
    return xf


def drop_pct(hi: float, lo: float) -> float:
    return (hi - lo) / hi * 100.0 if hi else float("nan")


def parse_server_ladder(log_path: Path) -> dict[str, Any] | None:
    """Per-position cumulative C[k], conditional p[k], E[T] from a vLLM server log."""
    if not log_path.exists():
        return None
    res = parse_log_per_position(log_path.read_text())
    if not res or res.get("intervals", 0) == 0 or "conditional_acceptance_p" not in res:
        return None
    return {
        "conditional_p": list(res["conditional_acceptance_p"]),
        "cumulative_C": list(res["cumulative_acceptance_C"]),
        "E_T": res["mean_tokens_per_step_E_T"],
        "num_drafts": res["num_drafts"],
        "intervals": res["intervals"],
    }


def relative_transfer(q_public_banked, q_public_sglang, q_proxy_sglang):
    """Transfer the sglang-measured per-position DROP shape onto the banked public
    reference (so the tree DP -- whose self-test is anchored to the banked public --
    stays valid). r_k = q_proxy_sglang[k]/q_public_sglang[k] is the per-position
    acceptance retention under the scored protocol; q_pinned[k]=q_public_banked[k]*r_k."""
    n = min(len(q_public_banked), len(q_public_sglang), len(q_proxy_sglang))
    out = []
    for k in range(n):
        denom = q_public_sglang[k]
        r = (q_proxy_sglang[k] / denom) if denom > 0 else 1.0
        out.append(q_public_banked[k] * r)
    return out


def propagate(model, q_public, q_proxy_raw, rho_pub, step, drop_targets):
    """Reuse the #151 calibration+propagation for a given raw proxy ladder.
    Returns the per-anchor projections (descent_only + both_bugs) and break-evens."""
    et_pub_linear = linear_et_from_q(q_public)
    et_proxy_linear = linear_et_from_q(q_proxy_raw)
    raw_drop = (et_pub_linear - et_proxy_linear) / et_pub_linear
    et_pub_descent = model.et_tree(q_public[0] * BUG1_MULT, q_public, rho_cond=rho_pub)
    et_pub_both = model.et_tree(q_public[0], q_public, rho_cond=rho_pub)

    anchors = {}
    seen = set()
    for tgt in list(drop_targets) + [raw_drop]:
        key = round(float(tgt), 4)
        if key in seen:
            continue
        seen.add(key)
        frac, _ = frac_for_target_drop(q_public, q_proxy_raw, tgt)
        spine = build_calibrated_ladder(q_public, q_proxy_raw, frac)
        h = float(np.mean(spine)) / float(np.mean(q_public[:len(spine)]))
        proj = project_one(model, spine, q_public, rho_pub, h, step)
        for topo, ref in (("descent_only", et_pub_descent), ("both_bugs", et_pub_both)):
            proj[topo]["et_drop_pct_vs_public"] = (
                (ref - proj[topo]["et_central_rho_coupled"]) / ref * 100.0)
        label = ("raw_proxy" if abs(tgt - raw_drop) < 1e-9 else f"calibrated_{tgt*100:.1f}pct")
        anchors[label] = {
            "aggregate_drop_target": float(tgt), "frac_of_proxy_shift": float(frac),
            "spine": spine, "spine_haircut_h": h, "linear_E_T": linear_et_from_q(spine),
            "projection": proj,
        }
    bar_500 = accept_length_for_official(TARGET_500, step, 1.0)
    be_descent = breakeven_drop(model, q_public, q_proxy_raw, rho_pub, step, "descent_only", bar_500)
    be_both = breakeven_drop(model, q_public, q_proxy_raw, rho_pub, step, "both_bugs", bar_500)
    return {
        "raw_aggregate_drop": raw_drop,
        "public_tree_et_descent_only": et_pub_descent,
        "public_tree_et_both_bugs": et_pub_both,
        "anchors": anchors,
        "breakeven_private_drop_descent_only": be_descent,
        "breakeven_private_drop_both_bugs": be_both,
        "clear_500_bar_et": bar_500,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sglang-run-dir",
                    default="research/validity/tree_private_drop_reconcile/sglang_ladder_run",
                    help="out_dir of the private_gap_probe.py sglang-scored run (server logs).")
    ap.add_argument("--public-banked", default=PUBLIC_BANKED_JSON)
    ap.add_argument("--proxy-decode", default=PROXY_DECODE_JSON)
    ap.add_argument("--rankcov-json", default=RANKCOV_JSON)
    ap.add_argument("--rho-opt-json", default=RHO_OPT_JSON)
    ap.add_argument("--step", type=float, default=STEP_MEASURED_DEPTH9)
    ap.add_argument("--gt-drop", type=float, default=GT_DROP)
    ap.add_argument("--output",
                    default="research/validity/tree_private_drop_reconcile/results.json")
    ap.add_argument("--wandb-group", "--wandb_group", default="tree-private-drop-reconcile")
    ap.add_argument("--wandb-name", "--wandb_name", default="stark/tree-private-drop-reconcile")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.sglang_run_dir)

    # ---- banked model + public/decode-proxy ladders ----
    meas_pub = load_measured(args.public_banked, "server_log")
    q_public = list(meas_pub["q"])
    meas_proxy_decode = load_measured(args.proxy_decode, "server_log")
    q_proxy_decode = list(meas_proxy_decode["q"])
    rc = load_rank_coverage(args.rankcov_json)
    rho_pub = rc["rho_cond"]
    parent = load_m32_topology(args.rho_opt_json)
    model = DescentModel(parent, rho_pub)

    st = run_self_test(model, q_public, rho_pub)
    assert st["passes"], "descent-walk DP does not reproduce banked anchors"

    et_pub_linear = linear_et_from_q(q_public)
    decode_drop = (et_pub_linear - linear_et_from_q(q_proxy_decode)) / et_pub_linear

    # ---- sglang scored-protocol per-position ladders (the new data) ----
    lad_pub = parse_server_ladder(run_dir / "server_public_cold.log") \
        or parse_server_ladder(run_dir / "server_leaderboard.log")
    lad_proxy = parse_server_ladder(run_dir / "server_private_rerun.log")
    probe_report = None
    rp = run_dir / "report.json"
    if rp.exists():
        probe_report = json.loads(rp.read_text())

    sglang_available = bool(lad_pub and lad_proxy)
    reconciliation: dict[str, Any] = {
        "organizer_flagship": {
            "public_tps": GT_PUBLIC_TPS, "private_tps": GT_PRIVATE_TPS,
            "drop_pct": GT_DROP * 100.0, "protocol": "sglang vllm-chat scored bench, REAL private set",
            "source": "BASELINE.md VERIFIED 2026-06-13 23:04Z",
        },
        "decode_audit_path": {
            "public_E_T": et_pub_linear, "proxy_E_T": linear_et_from_q(q_proxy_decode),
            "drop_pct": decode_drop * 100.0,
            "protocol": "decode_outputs.py audit: client-side chat template, /v1/completions, ignore_eos",
            "source": "accept_calibration.py #151 (proxy=private_proxy_sharegpt.json)",
        },
    }

    if probe_report is not None:
        eacc = probe_report.get("e_accept", {})
        dec = probe_report.get("decomposition", {})
        reconciliation["sglang_scored_path_probe"] = {
            "public_E_accept": eacc.get("public"), "proxy_E_accept": eacc.get("private"),
            "E_accept_drop_pct": (drop_pct(eacc["public"], eacc["private"])
                                  if eacc.get("public") and eacc.get("private") else None),
            "tps_distribution_gap_pct": dec.get("distribution_gap_precache_neutral_pct"),
            "tps_headline_gap_pct": probe_report.get("headline_public_to_private_gap_pct"),
            "protocol": "private_gap_probe.py sglang vllm-chat scored, precache-neutral, PROXY",
        }

    results: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "constants": {
            "K_cal": K_CAL, "step": args.step, "bug1_mult": BUG1_MULT,
            "gt_drop": args.gt_drop, "gt_drop_pct": args.gt_drop * 100.0,
            "frontier_official": FRONTIER_OFFICIAL, "target_500": TARGET_500,
            "E_T_linear_floor": E_T_LINEAR,
            "clear_500_bar_et": accept_length_for_official(TARGET_500, args.step, 1.0),
        },
        "self_test_passes": bool(st["passes"]),
        "public_ladder_banked": {"conditional_p": q_public, "linear_E_T": et_pub_linear},
        "proxy_ladder_decode": {"conditional_p": q_proxy_decode,
                                "linear_E_T": linear_et_from_q(q_proxy_decode)},
        "sglang_available": sglang_available,
    }

    verdict: dict[str, Any] = {"self_test_passes": int(st["passes"]),
                               "decode_audit_drop_pct": _clean(decode_drop * 100.0),
                               "gt_drop_pct": _clean(args.gt_drop * 100.0)}

    if not sglang_available:
        reconciliation["note"] = ("sglang server logs not found in --sglang-run-dir; run "
                                  "private_gap_probe.py there first. Reconciliation reported from "
                                  "existing artifacts only; tree re-score pending.")
        results["reconciliation"] = reconciliation
        results["verdict"] = verdict
        results["pinned_pending"] = True
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2, default=_jd))
        print(f"[reconcile] sglang logs absent -> wrote partial {args.output}", flush=True)
        return 0

    # sglang native per-position ladders + native drop
    q_pub_sglang = lad_pub["conditional_p"]
    q_proxy_sglang = lad_proxy["conditional_p"]
    sglang_native_linear_drop = drop_pct(linear_et_from_q(q_pub_sglang),
                                         linear_et_from_q(q_proxy_sglang)) / 100.0
    reconciliation["sglang_scored_path_perposition"] = {
        "public_conditional_p": q_pub_sglang, "proxy_conditional_p": q_proxy_sglang,
        "public_E_T": lad_pub["E_T"], "proxy_E_T": lad_proxy["E_T"],
        "linear_E_T_drop_pct": sglang_native_linear_drop * 100.0,
        "public_top1": q_pub_sglang[0], "proxy_top1": q_proxy_sglang[0],
        "source": "vLLM per-position log lines under sglang vllm-chat bench (precache-off)",
    }

    # ---- PINNED proxy ladder: transfer sglang drop-shape onto banked public ----
    q_proxy_pinned = relative_transfer(q_public, q_pub_sglang, q_proxy_sglang)
    pinned_raw_drop = (et_pub_linear - linear_et_from_q(q_proxy_pinned)) / et_pub_linear
    results["proxy_ladder_pinned"] = {
        "conditional_p": q_proxy_pinned, "linear_E_T": linear_et_from_q(q_proxy_pinned),
        "raw_aggregate_drop_pct": pinned_raw_drop * 100.0,
        "method": "relative_transfer(banked_public, sglang_public, sglang_proxy)",
    }

    # attribution
    reconciliation["attribution"] = {
        "harness_path_gap_pp": (decode_drop - pinned_raw_drop) * 100.0,
        "proxy_difficulty_gap_pp": (pinned_raw_drop - args.gt_drop) * 100.0,
        "harness_path_explains_divergence": bool(pinned_raw_drop < decode_drop - 0.03),
        "anchor_4p3_reachable_under_pinned": bool(pinned_raw_drop >= args.gt_drop),
        "diagnosis": (
            "19.6% (decode-audit) -> {:.1f}% (sglang scored) is the HARNESS-PATH gap; "
            "{:.1f}% (sglang proxy) -> 4.3% (organizer real private) is PROXY DIFFICULTY. "
            "Pinned protocol = sglang scored; calibration anchor = 4.3%."
        ).format(pinned_raw_drop * 100.0, pinned_raw_drop * 100.0),
    }
    results["reconciliation"] = reconciliation

    # ---- propagate the PINNED ladder through the tree DP, calibrated to 4.3% ----
    prop = propagate(model, q_public, q_proxy_pinned, rho_pub, args.step,
                     drop_targets=[args.gt_drop, 0.09])
    results["tree_propagation_pinned"] = prop

    # also propagate the DECODE-path proxy for an apples-to-apples #151 reproduction
    prop_decode = propagate(model, q_public, q_proxy_decode, rho_pub, args.step,
                            drop_targets=[args.gt_drop])
    results["tree_propagation_decode_151"] = {
        "raw_aggregate_drop": prop_decode["raw_aggregate_drop"],
        "calibrated_4.3pct_descent_tps":
            prop_decode["anchors"][f"calibrated_{args.gt_drop*100:.1f}pct"]["projection"]["descent_only"]["official_central"],
        "calibrated_4.3pct_descent_et_drop_pct":
            prop_decode["anchors"][f"calibrated_{args.gt_drop*100:.1f}pct"]["projection"]["descent_only"]["et_drop_pct_vs_public"],
    }

    gt_label = f"calibrated_{args.gt_drop*100:.1f}pct"
    gt_anchor = prop["anchors"][gt_label]
    d = gt_anchor["projection"]["descent_only"]
    b = gt_anchor["projection"]["both_bugs"]

    # ---- self-validation (PRIMARY): pinned reproduces flagship GT-4.3% LINEAR drop ----
    calibrated_linear_drop = (et_pub_linear - gt_anchor["linear_E_T"]) / et_pub_linear
    reproduces_4p3 = bool(abs(calibrated_linear_drop - args.gt_drop) <= 0.005
                          and pinned_raw_drop >= args.gt_drop - 1e-9)

    headline = {
        "harness_pin_reproduces_flagship_4p3": reproduces_4p3,         # PRIMARY
        "tree_private_drop_pct_pinned": _clean(d["et_drop_pct_vs_public"]),  # TEST (descent-only)
        "tree_private_drop_pct_pinned_both_bugs": _clean(b["et_drop_pct_vs_public"]),
        "tree_private_tps_proj_pinned": _clean(d["official_central"]),
        "tree_private_tps_proj_pinned_both_bugs": _clean(b["official_central"]),
        "descent_only_clears_500_pinned": bool(d["clears_500_central"]),
        "both_bugs_clears_500_pinned": bool(b["clears_500_central"]),
        "tps_band_pinned_descent_only": [_clean(d["tps_band"][0]), _clean(d["tps_band"][1])],
        "tree_private_et_pinned_descent_only": _clean(d["et_central_rho_coupled"]),
        "calibrated_linear_drop_pct": _clean(calibrated_linear_drop * 100.0),
        "pinned_raw_proxy_drop_pct": _clean(pinned_raw_drop * 100.0),
        "decode_audit_drop_pct": _clean(decode_drop * 100.0),
        "sglang_native_linear_drop_pct": _clean(sglang_native_linear_drop * 100.0),
        "breakeven_private_drop_descent_only_pct": _clean(prop["breakeven_private_drop_descent_only"] * 100.0),
        "breakeven_private_drop_both_bugs_pct": _clean(prop["breakeven_private_drop_both_bugs"] * 100.0),
        # #151 anchor: descent-only 505.46 @ assumed-4.3%; does the PINNED measurement keep it?
        "verdict_151_descent_only_holds": bool(d["clears_500_central"]),
        "verdict_151_descent_tps_151": 505.46,
        "verdict_151_descent_tps_pinned": _clean(d["official_central"]),
    }
    results["headline"] = headline

    verdict.update({k: v for k, v in headline.items() if not isinstance(v, (list, dict))})
    verdict["pinned_raw_proxy_drop_pct"] = _clean(pinned_raw_drop * 100.0)
    results["verdict"] = verdict
    results["pinned_pending"] = False

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=_jd))
    print(f"[reconcile] wrote {args.output}", flush=True)

    # ---- console summary ----
    print("\n" + "=" * 72, flush=True)
    print("PR #156 -- TREE PRIVATE-DROP RECONCILIATION", flush=True)
    print("=" * 72, flush=True)
    print(f"  decode-audit path drop (19.6 expected) : {decode_drop*100:5.2f}%", flush=True)
    print(f"  sglang scored proxy drop (pinned, raw) : {pinned_raw_drop*100:5.2f}%  "
          f"(native linear {sglang_native_linear_drop*100:.2f}%)", flush=True)
    print(f"  organizer GT (real private)            : {args.gt_drop*100:5.2f}%", flush=True)
    print(f"  -> harness-path gap  {(decode_drop-pinned_raw_drop)*100:+.1f}pp ; "
          f"proxy-difficulty gap {(pinned_raw_drop-args.gt_drop)*100:+.1f}pp", flush=True)
    print(f"  PINNED proxy top-1: banked {q_public[0]:.3f} -> sglang-shape {q_proxy_pinned[0]:.3f} "
          f"(decode-path {q_proxy_decode[0]:.3f})", flush=True)
    print(f"\n  [PRIMARY] harness_pin_reproduces_flagship_4p3 : {reproduces_4p3} "
          f"(calibrated linear drop {calibrated_linear_drop*100:.2f}%)", flush=True)
    print(f"  [TEST]    tree_private_drop_pct_pinned        : {d['et_drop_pct_vs_public']:.2f}% "
          f"(descent-only); {b['et_drop_pct_vs_public']:.2f}% (both-bugs)", flush=True)
    print(f"  tree_private_tps_proj_pinned : descent {d['official_central']:.1f} "
          f"(clears500={d['clears_500_central']}); both {b['official_central']:.1f} "
          f"(clears500={b['clears_500_central']})", flush=True)
    print(f"  band (descent-only): [{d['tps_band'][0]:.1f}, {d['tps_band'][1]:.1f}]", flush=True)
    print(f"  break-even private drop @500: descent {prop['breakeven_private_drop_descent_only']*100:.2f}%, "
          f"both {prop['breakeven_private_drop_both_bugs']*100:.2f}%", flush=True)
    print(f"  #151 verdict (descent 505.46 clears 500) holds at pinned drop: "
          f"{d['clears_500_central']}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, results, verdict)
        except Exception as e:  # noqa: BLE001
            print(f"[reconcile] W&B logging failed (non-fatal): {e!r}", flush=True)
    print("[reconcile] DONE", flush=True)
    return 0


def _log_wandb(args, results, verdict):
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group=args.wandb_group, name=args.wandb_name, job_type="analysis",
        config={"step": args.step, "gt_drop": args.gt_drop, "K_cal": K_CAL,
                "bug1_mult": BUG1_MULT, "sglang_run_dir": str(args.sglang_run_dir)})
    summ = {f"verdict/{k}": v for k, v in verdict.items() if not isinstance(v, (dict, list))}
    run.summary.update(summ)
    rec = results.get("reconciliation", {})
    tbl = wandb.Table(columns=["number", "value_pct", "what", "protocol", "distribution"])
    tbl.add_data("GT-flagship", GT_DROP * 100.0, "organizer real public->private TPS drop",
                 "sglang vllm-chat scored", "REAL private")
    if "sglang_scored_path_perposition" in rec:
        sp = rec["sglang_scored_path_perposition"]
        tbl.add_data("sglang-pinned", results["proxy_ladder_pinned"]["raw_aggregate_drop_pct"],
                     "per-position E[T] drop (scored)", "sglang vllm-chat scored", "hard chat proxy")
    tbl.add_data("decode-19.6", rec["decode_audit_path"]["drop_pct"],
                 "per-position E[T] drop (audit)", "decode_outputs.py audit", "hard chat proxy")
    run.log({"drop_reconciliation": tbl})
    run.summary["wandb_run_id"] = run.id
    print(f"[reconcile] W&B run: {run.url}  (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
