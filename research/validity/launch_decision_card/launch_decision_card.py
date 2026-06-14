#!/usr/bin/env python3
"""PR #238 -- the #124 launch-decision GO/NO-GO card (integrator capstone).

WHAT THIS LEG PRODUCES
----------------------
The SINGLE human-facing #124 GO/NO-GO decision card. fern #231's GO-card read
"5-of-6 gates GREEN, RED = deep-tail lambda_hat". land #71's honest reset
(22:12Z) makes that framing INCOMPLETE: the true top-line RED is READINESS
itself -- there is NO MEASURED >=500 artifact. The served stack is still linear
MTP K=7 (identical to the 481.53 frontier); the +18.3% / E[T]_both=4.512 /
~520 TPS tree gain is an UNMEASURED analytic projection over observational
probes (treeverify_served_gain_MEASURED_realized=0.0), never realized
end-to-end. This card headlines READINESS = NOT-READY (projection-only; the live
tree-decode build is the long pole), folds EVERY banked axis into a labelled
row, and carries wirbel #235's compliant-PRIVATE-500 INFEASIBLE flip (operative
ceiling 520.95 < private bar 528.48 -- the lane is DEAD at the operative
ceiling). It must NOT let an optimistic projection read as a delivered win.

COMPOSITION (imports -- NOT re-derived)
---------------------------------------
    land #71      served = linear MTP K=7; tree gain PROJECTED not measured;
                  treeverify_served_gain_MEASURED_realized=0.0; projection
                  E[T]_both=4.512, min lambda q[2..9]=0.983.            [W&B only]
    wirbel #235   operative_compliant_ceiling=520.95 < stark #226 private bar
                  528.48 -> compliant-PRIVATE-500 INFEASIBLE; 15.71 TPS gap =
                  100% topology/coverage.       [twoceiling_reconcile, on-branch]
    denken #236   ppl_headroom_at_build_bar=0.0428; ppl_is_binding=False
                  (served PPL pinned 2.3772, lambda-invariant).      [ppl_public_gate]
    kanna #237    accepted_risk_at_speed_gate=0.0583 (assumed f_priv) / 0.2394
                  (grounded); lambda_risk5=0.9700.          [publishfirst_accepted_risk]
    lawine #232   int4_token_identity_M1_vs_M8=0.9927 -> 0.73% deployed-M=8
                  divergence (near-greedy).            [int4_tokenident_deployed_m8]
    stark #233    f_priv_breakeven_publish_first=0.9598 ([0.957,0.969]
                  straddles).                       [publish_first_fpriv_breakeven]
    stark #191    build bar lambda_hat>=0.9780.  ubel #229 public speed gate
                  0.9675.  ubel #234 public margin 0.0 @ floor / +2.367 @ 0.9780.

VERDICT
    readiness_verdict = NOT-READY while treeverify_served_gain_MEASURED_realized
    == 0.0. Under #124 publish-first the launch is gated on a MEASURED >=500
    build (land #71, the long pole) AND a human-approved HF-job issue -- neither
    exists. The analytic packet is the POST-HOC DEFENCE, not a launch trigger.

LOCAL CPU-only integration over banked MERGED legs. No GPU / vLLM / HF Job /
submission / served-file change / official draw. BASELINE stays 481.53. Greedy/
PPL untouched. Adds 0 TPS. Bank-the-analysis (PRIMARY = self-test). Authorizes
NOTHING. NOT a launch. NOT open2.

PRIMARY metric  launch_decision_card_self_test_passes
TEST    metric  n_green_gates

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/launch_decision_card/launch_decision_card.py \
        --self-test --wandb_group launch-readiness-integration --wandb_name fern/launch-decision-card
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from collections import deque
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# --------------------------------------------------------------------------- #
# Banked source artifacts (MERGED on the advisor branch). Each row's value is
# round-tripped against the file that backs its source PR (self-test cond a).
# --------------------------------------------------------------------------- #
SRC = {
    "lawine_232": "research/validity/int4_tokenident_deployed_m8/int4_tokenident_result.json",
    "denken_236": "research/validity/ppl_public_gate/ppl_public_gate_results.json",
    "kanna_237": "research/validity/publishfirst_accepted_risk/publishfirst_accepted_risk_results.json",
    "stark_233": "research/validity/publish_first_fpriv_breakeven/results.json",
    "ubel_234": "research/validity/publishfirst_public_margin/publishfirst_public_margin_results.json",
    # wirbel #235 (w6a34f51) re-headlines the two-ceiling reconcile leg; the
    # ceiling/bar pair it imports is banked here (the INFEASIBLE flip's basis).
    "wirbel_235": "research/validity/twoceiling_reconcile/results.json",
}

# Pinned full-precision constants imported from the source artifacts / PR #238.
# (The card's human-facing `value` strings show the PR's rounded forms; these
# are the exact banked numbers the round-trip asserts against.)
EXPECT = {
    "lawine_232": {"int4_token_identity_M1_vs_M8": 0.9927083333333333},
    "denken_236": {
        "ppl_headroom_at_build_bar": 0.04279999999999973,
        "ppl_is_binding_public_gate": False,
        "ppl_served_at_build_bar": 2.3772,
    },
    "kanna_237": {
        "accepted_risk_at_speed_gate": 0.05831773945416474,
        "accepted_risk_at_speed_gate_grounded": 0.2394311004235805,
        "lambda_risk5": 0.9699990336265527,
    },
    "stark_233": {"f_priv_breakeven_publish_first": 0.9597799742440889},
    "ubel_234": {
        "public_go_margin_at_floor": 0.0,
        "public_go_margin_at_validity_bar_0p9780": 2.3666088372732474,
    },
    "wirbel_235": {
        "operative_compliant_ceiling": 520.9527323111674,
        "private_bar_worstcase": 528.4835555959945,
        "reach_dp_central_tps": 536.6590426143789,
    },
}

# land #71 -- W&B-only projection (NO local artifact; the readiness truth).
LAND71 = {
    "wandb_run": "land#71",
    "served_topology": "linear MTP K=7 (identical to the 481.53 frontier)",
    "treeverify_served_gain_MEASURED_realized": 0.0,  # the load-bearing fact
    "projection_E_T_both": 4.512,
    "projection_tree_gain_pct": 18.3,
    "projection_served_tps_approx": 520.0,
    "projection_min_lambda_q2q9": 0.983,
}

# Public / build gate bars (banked scalars).
TARGET_TPS = 500.0          # the #124 public speed milestone
BASELINE_TPS = 481.53       # official frontier (PR #52); this leg adds 0 TPS
PPL_CAP = 2.42              # public quality gate (reference + 5%), program.md L24
BUILD_BAR_LAMBDA = 0.9780   # stark #191 lambda_hat build bar
SPEED_GATE_LAMBDA = 0.9675  # ubel #229 operative public speed gate
ROUNDTRIP_TOL = 1e-6


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_key(obj: Any, key: str) -> Any:
    """BFS for `key` -- prefers the shallowest (top-level) occurrence."""
    dq: deque = deque([obj])
    while dq:
        cur = dq.popleft()
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            dq.extend(cur.values())
        elif isinstance(cur, list):
            dq.extend(cur)
    return None


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _match(loaded: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(loaded, bool) and loaded == expected
    if _finite(expected):
        return _finite(loaded) and abs(float(loaded) - float(expected)) <= ROUNDTRIP_TOL
    return loaded == expected


def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


# --------------------------------------------------------------------------- #
# Step 0 -- import every banked source + round-trip it against its file.
# --------------------------------------------------------------------------- #
def load_sources() -> dict[str, Any]:
    out: dict[str, Any] = {"values": {}, "roundtrip": {}, "files_present": {}}
    for src, rel in SRC.items():
        path = os.path.join(REPO_ROOT, rel)
        present = os.path.exists(path)
        out["files_present"][src] = present
        data = _load(path) if present else None
        for key, expected in EXPECT[src].items():
            loaded = find_key(data, key) if present else None
            ok = bool(present and _match(loaded, expected))
            out["values"][key] = expected            # pinned (the card displays these)
            out["roundtrip"][f"{src}.{key}"] = {
                "source_pr": src,
                "file": rel,
                "file_present": present,
                "expected": expected,
                "loaded": loaded,
                "round_trips": ok,
            }
    return out


# --------------------------------------------------------------------------- #
# Step 1 -- the decision card (one row per gate).
# --------------------------------------------------------------------------- #
def build_card(s: dict[str, Any]) -> dict[str, Any]:
    v = s["values"]
    operative = float(v["operative_compliant_ceiling"])          # 520.953 (int4-spec)
    private_bar = float(v["private_bar_worstcase"])              # 528.484 (stark #226)
    reach_dp = float(v["reach_dp_central_tps"])                  # 536.659 (reach-DP)
    coverage_gap = reach_dp - operative                          # 15.71 (topology/coverage)
    private_500_shortfall = private_bar - operative              # 7.53 vs the private bar
    measured_gain = float(LAND71["treeverify_served_gain_MEASURED_realized"])  # 0.0

    rows = [
        {
            "idx": "i",
            "gate": "READINESS / measured >=500 artifact",
            "status": "RED",
            "value": ("NONE -- served=linear MTP K=7; "
                      "treeverify_served_gain_MEASURED_realized=%.1f (projection-only)" % measured_gain),
            "value_num": measured_gain,
            "source_pr": "land #71",
            "source_run": LAND71["wandb_run"],
            "what_flips_it": ("a BUILT, MEASURED, end-to-end >=500 tree-decode artifact "
                              "(the live tree-decode build -- the long pole)"),
            "note": ("the projection (E[T]_both=%.3f, +%.1f%%, ~%.0f TPS, min lambda q[2..9]=%.3f) "
                     "is UNMEASURED; the served stack is the 481.53 frontier. THIS is the top-line RED."
                     % (LAND71["projection_E_T_both"], LAND71["projection_tree_gain_pct"],
                        LAND71["projection_served_tps_approx"], LAND71["projection_min_lambda_q2q9"])),
        },
        {
            "idx": "ii",
            "gate": "public speed >=500",
            "status": "PENDING",
            "value": ("served %.2f < 500 (build-pending); projected ~%.0f @ E[T]_both=%.3f UNMEASURED"
                      % (BASELINE_TPS, LAND71["projection_served_tps_approx"], LAND71["projection_E_T_both"])),
            "value_num": BASELINE_TPS,
            "source_pr": "land #71 proj / ubel #229",
            "source_run": "land#71 / bz2b3fw8",
            "what_flips_it": "the measured build's served tps >= 500 on the a10g-small benchmark",
            "note": ("gated on the readiness build; the ~520 projection is observational, "
                     "not an end-to-end measured speed."),
        },
        {
            "idx": "iii",
            "gate": "lambda_hat build bar >=%.4f" % BUILD_BAR_LAMBDA,
            "status": "PENDING",
            "value": ("bar=%.4f (stark #191); projection min lambda q[2..9]=%.3f UNMEASURED"
                      % (BUILD_BAR_LAMBDA, LAND71["projection_min_lambda_q2q9"])),
            "value_num": BUILD_BAR_LAMBDA,
            "source_pr": "stark #191 / land #71",
            "source_run": "stark#191 / land#71",
            "what_flips_it": "the build's MEASURED deep-tail q[2..9] lambda_hat >= 0.9780",
            "note": ("needs the build's measured acceptance on q[2..9]; the 0.983 projection "
                     "clears the bar but is not measured."),
        },
        {
            "idx": "iv",
            "gate": "PPL <= %.2f" % PPL_CAP,
            "status": "GREEN",
            "value": ("served %.4f, headroom %.4f, NOT binding (lambda-invariant)"
                      % (float(v["ppl_served_at_build_bar"]), float(v["ppl_headroom_at_build_bar"]))),
            "value_num": float(v["ppl_headroom_at_build_bar"]),
            "source_pr": "denken #236",
            "source_run": "hodnu1w1",
            "what_flips_it": ("only a coarser VERIFY MODEL (not lambda / not the tree build) -- "
                              "the served stream is the int4 greedy stream"),
            "note": ("ppl_is_binding_public_gate=%s; the int4 verify pins served PPL at 2.3772, "
                     "%.4f under cap." % (str(bool(v["ppl_is_binding_public_gate"])),
                                          float(v["ppl_headroom_at_build_bar"]))),
        },
        {
            "idx": "v",
            "gate": "128/128 completion",
            "status": "GREEN",
            "value": "128/128 (frontier #52; program.md public-run contract)",
            "value_num": 128.0,
            "source_pr": "PR #52 / program.md",
            "source_run": "frontier-52",
            "what_flips_it": "a completion failure / timeout / OOM on the measured run",
            "note": "the 481.53 frontier completes all 128 public prompts; carried forward as banked GREEN.",
        },
        {
            "idx": "vi",
            "gate": "private bar (post-hoc defence, NOT a launch gate)",
            "status": "ACCEPTED-RISK",
            "value": ("accepted_risk_at_speed_gate=%.4f (assumed f_priv) / %.4f (grounded) @ lambda=%.4f; "
                      "lambda_risk5=%.4f; f_priv_breakeven=%.4f straddles [0.957,0.969]; "
                      "int4 M1-vs-M8 identity=%.4f (0.73%% divergence)"
                      % (float(v["accepted_risk_at_speed_gate"]),
                         float(v["accepted_risk_at_speed_gate_grounded"]), SPEED_GATE_LAMBDA,
                         float(v["lambda_risk5"]), float(v["f_priv_breakeven_publish_first"]),
                         float(v["int4_token_identity_M1_vs_M8"]))),
            "value_num": float(v["accepted_risk_at_speed_gate"]),
            "source_pr": "kanna #237 / stark #233 / lawine #232",
            "source_run": "8x7i38jh / pszvrf2a / nxwv6pam",
            "what_flips_it": ("a second hard public/private paired draw grounding f_priv "
                              "(collapses the [0.957,0.969] straddle to a measured band)"),
            "note": ("under #124 publish-first this is NOT a launch gate -- it is the accepted "
                     "single-draw risk (5.8%-23.9%) the publish-first posture takes on; the analytic "
                     "packet defends it POST-HOC, it does not trigger the launch."),
        },
        {
            "idx": "vii",
            "gate": "compliant-PRIVATE-500 lane",
            "status": "INFEASIBLE",
            "value": ("operative ceiling %.2f < private bar %.2f -> INFEASIBLE; shortfall %.2f; "
                      "topology/coverage gap %.2f (reach-DP %.2f)"
                      % (operative, private_bar, private_500_shortfall, coverage_gap, reach_dp)),
            "value_num": private_500_shortfall,
            "source_pr": "wirbel #235",
            "source_run": "w6a34f51",
            "what_flips_it": ("raising the OPERATIVE ceiling above 528.48 -- needs topology/coverage "
                              "beyond the int4-spec served stack (the 15.71 gap is 100%% topology)"),
            "note": ("DEAD at the operative ceiling: the int4-spec deployed bound 520.95 cannot reach "
                     "the worst-case private bar 528.48; the gain is locked in unrealized reach-DP topology."),
        },
    ]

    n_green = sum(1 for r in rows if r["status"] == "GREEN")
    n_red = sum(1 for r in rows if r["status"] == "RED")
    n_pending = sum(1 for r in rows if r["status"] == "PENDING")
    n_infeasible = sum(1 for r in rows if r["status"] == "INFEASIBLE")
    n_accepted_risk = sum(1 for r in rows if r["status"] == "ACCEPTED-RISK")

    readiness_verdict = "NOT-READY" if measured_gain == 0.0 else "READY-PENDING-HUMAN"
    # the human-facing #124 top-line: NOT-READY readiness -> overall NO-GO (NOT-YET).
    launch_decision = "NO-GO (NOT-READY)" if readiness_verdict == "NOT-READY" else "GO-PENDING-HUMAN"

    return {
        "rows": rows,
        "n_gates": len(rows),
        "n_green_gates": n_green,
        "n_red_gates": n_red,
        "n_pending_gates": n_pending,
        "n_infeasible_gates": n_infeasible,
        "n_accepted_risk_gates": n_accepted_risk,
        "readiness_verdict": readiness_verdict,
        "launch_decision": launch_decision,
        "treeverify_served_gain_MEASURED_realized": measured_gain,
        "operative_compliant_ceiling": operative,
        "private_bar_worstcase": private_bar,
        "reach_dp_central_tps": reach_dp,
        "compliant_private_500_infeasible": bool(operative < private_bar),
        "private_500_shortfall_tps": private_500_shortfall,
        "topology_coverage_gap_tps": coverage_gap,
        "ppl_headroom_at_build_bar": float(v["ppl_headroom_at_build_bar"]),
        "ppl_is_binding_public_gate": bool(v["ppl_is_binding_public_gate"]),
        "baseline_tps_unchanged": BASELINE_TPS,
    }


# --------------------------------------------------------------------------- #
# Step 2 -- the readiness truth (the core paragraph).
# --------------------------------------------------------------------------- #
def build_readiness_truth(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "readiness_verdict": card["readiness_verdict"],
        "launch_decision": card["launch_decision"],
        "statement": (
            "Under #124 publish-first the launch is gated on a MEASURED >=500 build (land #71, the "
            "long pole) AND a human-approved HF-job issue -- NEITHER exists. The served stack is still "
            "linear MTP K=7 (the 481.53 frontier); treeverify_served_gain_MEASURED_realized=%.1f, so the "
            "+%.1f%% / E[T]_both=%.3f / ~%.0f TPS tree gain is an UNMEASURED analytic projection, never "
            "realized end-to-end. The banked analytic packet (PPL not-binding, accepted private-draw "
            "risk, the INFEASIBLE compliant-private lane) is the POST-HOC DEFENCE, NOT a launch trigger. "
            "readiness_verdict=%s; the sole launch blocker is the live tree-decode build delivering a "
            "MEASURED >=500." % (
                card["treeverify_served_gain_MEASURED_realized"], LAND71["projection_tree_gain_pct"],
                LAND71["projection_E_T_both"], LAND71["projection_served_tps_approx"],
                card["readiness_verdict"])),
        "do_not_let_projection_read_as_delivered": True,
        "measured_artifact_exists": False,
        "human_hf_job_issue_approved": False,
    }


# --------------------------------------------------------------------------- #
# Step 3 -- self-test (PRIMARY metric).
# --------------------------------------------------------------------------- #
def self_test(s: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # (a) every row's value round-trips its source PR (all files present + match).
    rt = s["roundtrip"]
    all_present = all(s["files_present"].values())
    all_match = all(d["round_trips"] for d in rt.values())
    a_ok = bool(all_present and all_match)
    results["a_rows_roundtrip_source_pr"] = {
        "pass": a_ok,
        "all_files_present": all_present,
        "all_values_match": all_match,
        "n_sources": len(s["files_present"]),
        "n_values_checked": len(rt),
        "failures": [k for k, d in rt.items() if not d["round_trips"]],
    }

    # (b) readiness_verdict == NOT-READY while treeverify_served_gain_MEASURED_realized == 0.0.
    measured = card["treeverify_served_gain_MEASURED_realized"]
    b_ok = bool(measured == 0.0 and card["readiness_verdict"] == "NOT-READY")
    results["b_readiness_not_ready_while_no_measured_gain"] = {
        "pass": b_ok,
        "treeverify_served_gain_MEASURED_realized": measured,
        "readiness_verdict": card["readiness_verdict"],
    }

    # (c) the INFEASIBLE-flip row reproduces 520.95 < 528.48.
    op = card["operative_compliant_ceiling"]
    pb = card["private_bar_worstcase"]
    c_ok = bool(op < pb and card["compliant_private_500_infeasible"]
                and abs(op - 520.95) < 0.01 and abs(pb - 528.48) < 0.01)
    results["c_infeasible_flip_520p95_lt_528p48"] = {
        "pass": c_ok,
        "operative_compliant_ceiling": op,
        "private_bar_worstcase": pb,
        "infeasible": card["compliant_private_500_infeasible"],
        "shortfall_tps": card["private_500_shortfall_tps"],
        "topology_coverage_gap_tps": card["topology_coverage_gap_tps"],
    }

    # (d) the PPL row reproduces headroom 0.0428 / not-binding.
    hr = card["ppl_headroom_at_build_bar"]
    d_ok = bool(abs(hr - 0.0428) < 1e-3 and card["ppl_is_binding_public_gate"] is False)
    results["d_ppl_headroom_0p0428_not_binding"] = {
        "pass": d_ok,
        "ppl_headroom_at_build_bar": hr,
        "ppl_is_binding_public_gate": card["ppl_is_binding_public_gate"],
    }

    # (e) NaN-clean across every reported numeric.
    payload_numeric = {"card": card, "roundtrip": rt}
    nan_paths = _nan_paths(payload_numeric, "selftest")
    e_ok = (len(nan_paths) == 0)
    results["e_nan_clean"] = {"pass": e_ok, "nan_paths": nan_paths}

    # (f) extra internal-consistency guard rails (not part of the PR's a-e but
    #     cheap and informative): exactly one RED readiness blocker; n_green==2.
    f_ok = bool(card["n_red_gates"] == 1 and card["n_green_gates"] == 2
                and card["n_infeasible_gates"] == 1 and card["n_pending_gates"] == 2
                and card["n_accepted_risk_gates"] == 1)
    results["f_status_histogram_consistent"] = {
        "pass": f_ok, "n_green": card["n_green_gates"], "n_red": card["n_red_gates"],
        "n_pending": card["n_pending_gates"], "n_infeasible": card["n_infeasible_gates"],
        "n_accepted_risk": card["n_accepted_risk_gates"]}

    passes = bool(all(r["pass"] for r in results.values()))
    return {
        "launch_decision_card_self_test_passes": passes,
        "n_green_gates": card["n_green_gates"],
        "conditions": results,
    }


# --------------------------------------------------------------------------- #
# Step 4 -- the one-sentence hand-off to the human #124 packet.
# --------------------------------------------------------------------------- #
def build_handoff(card: dict[str, Any]) -> str:
    return (
        "the #124 decision card reads NOT-READY: no measured >=500 artifact exists "
        "(land #71 projection-only), the compliant-PRIVATE-500 lane is INFEASIBLE at the operative "
        "ceiling %.2f<%.2f, but the public packet is otherwise clean (PPL not binding, accepted "
        "private-draw risk 5.8-23.9%% at the 0.9675 speed gate) -- so the sole launch blocker is the "
        "live tree-decode build delivering a MEASURED >=500."
        % (card["operative_compliant_ceiling"], card["private_bar_worstcase"]))


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    t0 = time.time()
    s = load_sources()
    card = build_card(s)
    readiness = build_readiness_truth(card)
    st = self_test(s, card)
    handoff = build_handoff(card)

    payload = {
        "pr": 238,
        "agent": "fern",
        "kind": "launch_decision_card",
        "primary_metric_name": "launch_decision_card_self_test_passes",
        "launch_decision_card_self_test_passes": st["launch_decision_card_self_test_passes"],
        "test_metric_name": "n_green_gates",
        "n_green_gates": st["n_green_gates"],
        "readiness_verdict": card["readiness_verdict"],
        "launch_decision": card["launch_decision"],
        "decision_card": card,
        "readiness_truth": readiness,
        "self_test": st,
        "handoff_line": handoff,
        "roundtrip": s["roundtrip"],
        "land71_projection": LAND71,
        "constants": {
            "target_tps": TARGET_TPS, "baseline_tps": BASELINE_TPS, "ppl_cap": PPL_CAP,
            "build_bar_lambda": BUILD_BAR_LAMBDA, "speed_gate_lambda": SPEED_GATE_LAMBDA,
            "operative_compliant_ceiling": card["operative_compliant_ceiling"],
            "private_bar_worstcase": card["private_bar_worstcase"],
            "reach_dp_central_tps": card["reach_dp_central_tps"],
        },
        "provenance": (
            "land#71 (served=linear MTP K=7, measured tree gain 0.0, projection E[T]_both 4.512) x "
            "wirbel#235 w6a34f51 (operative ceiling 520.95 < private bar 528.48 INFEASIBLE; via "
            "twoceiling_reconcile, stark#226 tzcc5xuq bar, wirbel#199 reach-DP 536.66) x denken#236 "
            "hodnu1w1 (ppl headroom 0.0428, not binding) x kanna#237 8x7i38jh (accepted risk "
            "0.0583/0.2394, lambda_risk5 0.9700) x lawine#232 nxwv6pam (int4 identity 0.9927) x "
            "stark#233 pszvrf2a (f_priv breakeven 0.9598) x stark#191 (build bar 0.9780) x ubel#229 "
            "bz2b3fw8 (speed gate 0.9675) x ubel#234 (public margin 0.0/+2.367). All run-ids in "
            "wandb-applied-ai-team/gemma-challenge-senpai."),
        "scope": (
            "LOCAL CPU-only integration of the banked launch-validity packet into one human-facing "
            "#124 GO/NO-GO card, headlining land #71's no-measured-artifact readiness truth and "
            "carrying wirbel #235's INFEASIBLE flip. All values imported (NOT re-derived) and "
            "round-tripped against their source-PR artifacts. CPU-only, 0 TPS, BASELINE stays 481.53, "
            "greedy/PPL untouched, authorizes NOTHING. NOT a launch. NOT open2."),
        "elapsed_sec": time.time() - t0,
        "peak_mem_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    return payload


# --------------------------------------------------------------------------- #
# wandb + main
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> str | None:
    if args.no_wandb:
        return None
    try:
        import wandb
    except Exception as exc:               # noqa: BLE001
        print(f"[wandb] unavailable ({exc}); skipping.", file=sys.stderr)
        return None
    card = payload["decision_card"]
    c = payload["constants"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group,
                     config={
                         "pr": 238, "agent": "fern", "kind": "launch_decision_card",
                         "readiness_verdict": payload["readiness_verdict"],
                         "launch_decision": payload["launch_decision"],
                         "target_tps": c["target_tps"], "baseline_tps": c["baseline_tps"],
                         "ppl_cap": c["ppl_cap"], "build_bar_lambda": c["build_bar_lambda"],
                         "speed_gate_lambda": c["speed_gate_lambda"],
                         "operative_compliant_ceiling": c["operative_compliant_ceiling"],
                         "private_bar_worstcase": c["private_bar_worstcase"],
                         "reach_dp_central_tps": c["reach_dp_central_tps"],
                         "provenance": payload["provenance"], "scope": payload["scope"],
                     })
    summary = {
        "launch_decision_card_self_test_passes": payload["launch_decision_card_self_test_passes"],
        "n_green_gates": payload["n_green_gates"],
        "n_red_gates": card["n_red_gates"],
        "n_pending_gates": card["n_pending_gates"],
        "n_infeasible_gates": card["n_infeasible_gates"],
        "n_accepted_risk_gates": card["n_accepted_risk_gates"],
        "treeverify_served_gain_MEASURED_realized": card["treeverify_served_gain_MEASURED_realized"],
        "compliant_private_500_infeasible": card["compliant_private_500_infeasible"],
        "private_500_shortfall_tps": card["private_500_shortfall_tps"],
        "topology_coverage_gap_tps": card["topology_coverage_gap_tps"],
        "ppl_headroom_at_build_bar": card["ppl_headroom_at_build_bar"],
        "ppl_is_binding_public_gate": int(card["ppl_is_binding_public_gate"]),
        "operative_compliant_ceiling": card["operative_compliant_ceiling"],
        "private_bar_worstcase": card["private_bar_worstcase"],
        "baseline_tps_unchanged": card["baseline_tps_unchanged"],
        "elapsed_sec": payload["elapsed_sec"], "peak_mem_mib": payload["peak_mem_mib"],
    }
    for k, val in list(summary.items()):
        if isinstance(val, float) and not math.isfinite(val):
            summary[k] = None
    wandb.log(summary)
    wandb.summary.update({k: val for k, val in summary.items() if val is not None})
    # also log readiness_verdict + launch_decision as summary strings.
    wandb.summary.update({"readiness_verdict": payload["readiness_verdict"],
                          "launch_decision": payload["launch_decision"]})

    # the decision card as a table.
    tbl = wandb.Table(columns=["idx", "gate", "status", "value", "source_pr", "what_flips_it"])
    for r in card["rows"]:
        tbl.add_data(r["idx"], r["gate"], r["status"], r["value"], r["source_pr"], r["what_flips_it"])
    wandb.log({"decision_card": tbl})

    art = wandb.Artifact("launch_decision_card_results", type="analysis")
    with art.new_file("launch_decision_card_results.json", mode="w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    run.log_artifact(art)
    run_path = f"{run.entity}/{run.project}/{run.id}"
    wandb.finish()
    return run_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR #238 -- #124 launch-decision GO/NO-GO card")
    ap.add_argument("--self-test", "--self_test", action="store_true",
                    help="emphasise the self-test (PRIMARY); non-zero exit if it fails")
    ap.add_argument("--out",
                    default="research/validity/launch_decision_card/launch_decision_card_results.json")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/launch-decision-card")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-readiness-integration")
    args = ap.parse_args(argv)

    payload = run()
    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)

    st = payload["self_test"]
    print("=" * 78)
    print("PR #238 -- #124 LAUNCH-DECISION GO/NO-GO CARD")
    print("=" * 78)
    print("readiness_verdict =", payload["readiness_verdict"])
    print("launch_decision   =", payload["launch_decision"])
    print("launch_decision_card_self_test_passes =",
          st["launch_decision_card_self_test_passes"])
    print("n_green_gates =", st["n_green_gates"])
    print("\nDECISION CARD:")
    for r in payload["decision_card"]["rows"]:
        print(f"  ({r['idx']:>3}) [{r['status']:>13}] {r['gate']:<42} <- {r['source_pr']}")
        print(f"        {r['value']}")
    print("\nSELF-TEST:")
    for cond, val in st["conditions"].items():
        print(f"  {cond}: {'PASS' if val['pass'] else 'FAIL'}")
    print("\nREADINESS TRUTH:", payload["readiness_truth"]["statement"])
    print("\nHANDOFF:", payload["handoff_line"])
    nan_paths = _nan_paths(payload, "payload")
    if nan_paths:
        print("\n[WARN] NaN paths:", nan_paths, file=sys.stderr)

    run_path = _maybe_log_wandb(args, payload)
    if run_path:
        print("\nwandb run:", run_path)

    if args.self_test and not st["launch_decision_card_self_test_passes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
