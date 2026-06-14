#!/usr/bin/env python3
"""PR #231 -- Launch-readiness GO-card: pre-register the launch decision on the ONE unknown.

THE ONE NUMBER THIS LEG PRODUCES
--------------------------------
fern #185 (`launch_trigger_calculator`, MERGED) built the one-call GO/NO-GO + filled (un-filed)
approval block from land #71's FULL measured tuple. Since then the fleet has banked EVERY launch
gate GREEN except a SINGLE unmeasured input: land #71's built DEEP-TAIL acceptance lambda_hat on
the q[8..9] rungs (the shallow-mid spine q[2..7] is interim-pinned at 0.997; the deep tail is
UNMEASURED, budget 0.7875 from stark #215). This leg crosses the banked gates into a launch-
readiness SCORECARD (every gate GREEN vs the one RED) and a PRE-REGISTERED GO-card that sweeps
the deep-tail lambda_hat across [0.5, 1.0] and emits GO / HOLD / NO-GO for every value it can take
-- so the instant land #71 reports, the #124 publish-first launch decision is a TABLE LOOKUP, not
a fresh computation. The integrator capstone of the #185 calculator.

COMPOSITION (imports -- NOT re-derived)
---------------------------------------
    blended tree lambda  = W_shallow * lambda_spine + W_deep * lambda_deeptail   [stark #215 blend]
        W_shallow,W_deep = #208/#203 reach-weights, beta-extended to q[8..9]      [import #215]
        lambda_spine     = 0.997 (q[2..7] interim spine)                          [land #71]
    E[T](lambda)         = reach-DP pmf-mean on the #213 floor/ceiling spine      [#175/#184 via #222]
    official mu_pub(lam) = 520.953 * E[T](lam)/E[T](1)  (ceiling-anchored)        [ubel #222 / #204]
    private mean         = official * f_priv,  f_priv = 0.969107                  [#217]
    publish-first floor  = private MEAN >= 500  (the #124 gate)                   [kanna #224]
    kernel NO-GO floor   = blended lambda >= 0.8572                               [wirbel #216]
    GO/NO-GO + approval  = launch_decision(measured_tuple)                        [fern #185, reused]

VERDICT (per swept deep-tail lambda_hat -> one blended tree lambda)
    GO    : private MEAN >= 500                       (publish-first, the #124 launch gate)
    HOLD  : blended lambda in [0.8572, publish-first)  (valid kernel route exists, sub-publish-first)
    NO-GO : blended lambda < 0.8572                    (even a free kernel misses 500)

LOCAL CPU-only synthesis. No GPU / vLLM / HF Job / submission / served-file change / official draw.
Does NOT re-measure lambda (the deep tail is land #71's to measure; this leg pre-registers the
DECISION across its axis, not the value). BASELINE stays 481.53. Greedy/PPL untouched. Adds 0 TPS.
Bank-the-analysis (PRIMARY = self-test). Authorizes NOTHING. NOT open2. NOT a launch.

PRIMARY metric  launch_readiness_gocard_self_test_passes
TEST    metric  deeptail_lambda_for_publish_first_go

Run:
    python research/validity/launch_readiness_gocard/launch_readiness_gocard.py \
        --wandb_group launch-readiness-gocard --wandb_name fern/launch-readiness-gocard
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# ---- sibling engines (reuse merged-PR machinery verbatim via dynamic import) ---- #
LTC_PATH = os.path.join(REPO_ROOT, "research/validity/launch_trigger/launch_trigger_calculator.py")
DTB_PATH = os.path.join(REPO_ROOT, "research/validity/deeptail_bar_budget/deeptail_bar_budget.py")
BG_PATH = os.path.join(REPO_ROOT, "research/validity/binding_gate/binding_gate.py")
# ---- banked gate JSON (import-not-rederive) ---- #
KERNEL_216 = os.path.join(REPO_ROOT, "research/validity/kernel_feasibility/kernel_feasibility_results.json")
PRIV_224 = os.path.join(REPO_ROOT, "research/validity/private_bar_reachability/private_bar_reachability_results.json")
# ---- kanna #228 publish-first-lambda-floor (in-flight; import IF banked, else compute+note) ---- #
PUBFIRST_228 = os.path.join(REPO_ROOT,
                            "research/validity/publish_first_lambda_floor/publish_first_lambda_floor_results.json")

TARGET = 500.0                       # the #124 publish-first private-mean target
F_PRIV_PINNED = 0.969106920637722    # #217 vgovdrjc (the PR's pinned composition f_priv)
HEADLINE_REGIME = "both_bugs"        # conservative regime + the validity bar's own topology
DEEPTAIL_SWEEP_LO = 0.5              # PR step 2 axis
DEEPTAIL_SWEEP_HI = 1.0
ROUNDTRIP_TOL = 1e-9


def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _all_finite(obj: Any) -> bool:
    if isinstance(obj, bool):
        return True
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite(v) for v in obj)
    return True


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


# Engines (module-level: #185 builds its _Machinery at import; #215/#222 are pure functions).
LTC = _import("launch_trigger_calculator", LTC_PATH)
DTB = _import("deeptail_bar_budget", DTB_PATH)
BG = _import("binding_gate", BG_PATH)


# --------------------------------------------------------------------------- #
# Step 0 -- import every banked gate + the blend (NOT re-derived).
# --------------------------------------------------------------------------- #
def load_gates() -> dict[str, Any]:
    # --- ubel #222 / #218 / #204 / #213: the binding gate frame + ceiling-anchored mu_pub map. ---
    b = BG.import_banked()
    mp = BG.build_maps(b)[HEADLINE_REGIME]
    validity_bar = float(b["validity_bar"])              # 0.9779783323 (#208 worst-case private go-bar)
    validity_bar_nominal = float(b["validity_bar_nominal"])  # 0.9780 (#191)
    trigger_wc = float(b["trigger_worstcase"])           # 513.5575 (#218 grounded worst-case GO trigger)
    ceiling_mu = float(b["lambda1_ceiling_mu"])          # 520.9527 (#204 int4-spec ceiling)
    mu_at_bar = float(mp.mu_pub(validity_bar))           # public mu_pub at the validity bar
    speed_margin_at_bar = mu_at_bar - trigger_wc         # >0 => validity binds (speed auto-clears)
    validity_binds = bool(speed_margin_at_bar >= 0.0)

    # --- stark #215: the reach-weighted spine+deep-tail blend (W_shallow/W_deep + the 0.7875 budget). ---
    syn215 = DTB.synthesize(DTB.LAMBDA_SPINE_INTERIM)
    rwp = syn215["reach_weight_profile"]
    w_shallow = float(rwp["w_mass_shallow_q2q7"])        # ~0.9092 (q[2..7] reach mass)
    w_deep = float(rwp["w_mass_deeptail_q8q9"])          # ~0.0908 (q[8..9] reach mass)
    lambda_spine = float(syn215["lambda_spine_interim"]) # 0.997 (land #71 interim)
    deeptail_budget_bar = float(syn215["deeptail_budget"]["min_deeptail_lambda_q8q9_clears_bar"])  # 0.7875

    # --- wirbel #216: the kernel feasibility NO-GO floor (blended lambda below it => even free kernel misses). ---
    k216 = _load(KERNEL_216)["synthesis"]["headline"]
    kernel_floor = float(k216["lambda_min_kernel_feasible"])                 # 0.8571543
    lambda_crit_zero_overhead = float(k216["lambda_crit_clears_500_zero_overhead_both_bugs_tau1"])  # 0.8345

    # --- kanna #217 / #224: f_priv + the harder P95 private bar (mu_ceiling_needed). ---
    r224 = _load(PRIV_224)
    mu_ceiling_needed_p95 = float(r224["mu_ceiling_needed"])                 # 535.139 (P95 private bar)
    f_priv_grounded_point = float(r224["f_priv_grounded_point"])            # 0.95705 (#224 grounded)
    f_priv_grounded_band = [float(x) for x in r224["f_priv_grounded_band"]]  # [0.95, 0.969107]
    p95_reachable_at_ceiling = bool(r224["private_500_reachable_at_physical_ceiling"])  # False

    # --- kanna #228 publish-first-lambda-floor (in-flight): import IF banked, else compute + note. ---
    pubfirst_228 = None
    if os.path.exists(PUBFIRST_228):
        try:
            d228 = _load(PUBFIRST_228)
            for key in ("lambda_floor_publish_first", "publish_first_lambda_floor"):
                v = _dig(d228, key)
                if _finite(v):
                    pubfirst_228 = float(v)
                    break
        except Exception:  # noqa: BLE001
            pubfirst_228 = None

    return {
        "b": b, "mp": mp,
        "validity_bar": validity_bar, "validity_bar_nominal": validity_bar_nominal,
        "trigger_worstcase": trigger_wc, "ceiling_mu": ceiling_mu,
        "mu_pub_at_validity_bar": mu_at_bar, "speed_margin_at_validity_bar": speed_margin_at_bar,
        "validity_binds": validity_binds,
        "w_shallow": w_shallow, "w_deep": w_deep, "lambda_spine": lambda_spine,
        "deeptail_budget_bar_215": deeptail_budget_bar,
        "syn215_partial_passes": bool(syn215["self_test"]["partial_passes_a_to_d"]),
        "kernel_floor": kernel_floor, "lambda_crit_zero_overhead": lambda_crit_zero_overhead,
        "f_priv": F_PRIV_PINNED,
        "f_priv_grounded_point": f_priv_grounded_point, "f_priv_grounded_band": f_priv_grounded_band,
        "mu_ceiling_needed_p95": mu_ceiling_needed_p95,
        "p95_reachable_at_ceiling": p95_reachable_at_ceiling,
        "pubfirst_228_banked": pubfirst_228 is not None,
        "pubfirst_228_value": pubfirst_228,
        "provenance": (
            "ubel#222 (yw7i2ece) binding gate + mu_pub map x ubel#218 (0ug7vd7d) trigger 513.557 x "
            "ubel#204 ceiling 520.953 x wirbel#213 reach-DP spines x stark#215 deep-tail blend x "
            "wirbel#216 (pc8g6s04) kernel floor 0.8572 x kanna#217 (vgovdrjc) f_priv 0.969107 x "
            "kanna#224 (1081oc84) P95 bar 535.139 x land#71 spine 0.997. GO/NO-GO via fern#185."),
    }


def _dig(d: dict, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


# --------------------------------------------------------------------------- #
# The blend + the per-deep-tail evaluation (the GO-card row machine).
# --------------------------------------------------------------------------- #
def blended_lambda(deeptail: float, g: dict) -> float:
    """stark #215 reach-weighted aggregate: lambda_hat = W_shallow*spine + W_deep*deeptail."""
    return g["w_shallow"] * g["lambda_spine"] + g["w_deep"] * deeptail


def deeptail_for_blended(target_lambda: float, g: dict) -> float:
    """Invert the blend: the deep-tail lambda that lifts the blended tree lambda to `target`."""
    return (target_lambda - g["w_shallow"] * g["lambda_spine"]) / g["w_deep"]


def evaluate_deeptail(deeptail: float, g: dict) -> dict[str, Any]:
    """One GO-card row: blended lambda -> E[T] -> official/private TPS -> GO/HOLD/NO-GO verdict."""
    mp = g["mp"]
    lam = blended_lambda(deeptail, g)
    et = float(mp.et_of_lambda(lam))
    official = float(mp.mu_pub(lam))
    private_mean = official * g["f_priv"]
    publish_first_go = bool(private_mean >= TARGET)
    valid_kernel_route = bool(lam >= g["kernel_floor"])
    if publish_first_go:
        verdict = "GO"
    elif valid_kernel_route:
        verdict = "HOLD"
    else:
        verdict = "NO-GO"
    return {
        "deeptail_lambda": float(deeptail),
        "blended_tree_lambda": float(lam),
        "E_T": float(et),
        "official_tps": float(official),
        "private_mean_tps": float(private_mean),
        "publish_first_go": publish_first_go,
        "valid_kernel_route": valid_kernel_route,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Step 1 -- the launch-readiness scorecard.
# --------------------------------------------------------------------------- #
def build_scorecard(g: dict) -> dict[str, Any]:
    private_mean_at_ceiling = g["ceiling_mu"] * g["f_priv"]
    rows = [
        {
            "gate": "validity-binds",
            "status": "GREEN" if g["validity_binds"] else "RED",
            "banked_value": g["mu_pub_at_validity_bar"],
            "threshold": "mu_pub(0.9780) >= trigger %.3f (speed auto-clears once validity binds)" % g["trigger_worstcase"],
            "source": "ubel #222 (yw7i2ece) / #218",
            "is_red_blocker": not g["validity_binds"],
            "note": "validity gate (lambda_hat >= 0.9780) BINDS: speed margin +%.2f TPS at the bar."
                    % g["speed_margin_at_validity_bar"],
        },
        {
            "gate": "speed-clears-once-validity",
            "status": "GREEN",
            "banked_value": g["trigger_worstcase"],
            "threshold": "mu_pub at the validity bar (%.2f) >= worst-case GO trigger" % g["mu_pub_at_validity_bar"],
            "source": "ubel #222/#218 (0ug7vd7d)",
            "is_red_blocker": False,
            "note": "the grounded worst-case trigger 513.557 is cleared the moment validity binds.",
        },
        {
            "gate": "gate-2-one-run-confirmable",
            "status": "GREEN",
            "banked_value": "SPRT + AR-corrected ASN one-run-confirmable",
            "threshold": "a single confirmation run proves measured lambda_hat >= 0.9780",
            "source": "denken #225",
            "is_red_blocker": False,
            "note": "the validity gate is confirmable in one run (no fixed-N 30k blowup).",
        },
        {
            "gate": "valid-verify-int4-kernel-only",
            "status": "GREEN-CONDITIONAL",
            "banked_value": "int4-spec VERIFY kernel is the sole compliant 500-route",
            "threshold": "compliant-spec int4 kernel under budget (double gate: lambda>=lambda_crit AND kernel-under-budget)",
            "source": "wirbel #223/#220/#196/#216",
            "is_red_blocker": False,
            "note": "the valid-verify menu has COLLAPSED to int4-kernel-only; carries a PENDING #192 "
                    "human ruling (handed to #124/#211), GREEN-conditional not RED.",
        },
        {
            "gate": "publish-first-clears-at-ceiling",
            "status": "GREEN" if private_mean_at_ceiling >= TARGET else "RED",
            "banked_value": private_mean_at_ceiling,
            "threshold": "private MEAN at the ceiling (%.2f) >= 500 at f_priv=%.6f" % (private_mean_at_ceiling, g["f_priv"]),
            "source": "kanna #224 (1081oc84)",
            "is_red_blocker": not (private_mean_at_ceiling >= TARGET),
            "note": "at the pinned f_priv=0.969107 the ceiling private mean is 504.86 >= 500 (GREEN); "
                    "see honest-band: at the grounded f_priv=0.957 (#224) it falls to 498.58 < 500.",
        },
        {
            "gate": "deeptail-lambda-q8q9",
            "status": "RED",
            "banked_value": None,                       # UNMEASURED -- land #71's number
            "threshold": "deep-tail reach-weighted lambda_hat over q[8..9] >= %.4f (publish-first)" % g["deeptail_budget_bar_215"],
            "source": "land #71 (UNMEASURED)",
            "is_red_blocker": True,
            "note": "the SINGLE unmeasured launch input; the spine q[2..7]=0.997 is interim-pinned, "
                    "the deep tail q[8..9] is what land #71 measures. The whole GO-card sweeps THIS.",
        },
    ]
    n_red_blockers = sum(1 for r in rows if r["status"] == "RED")
    return {
        "rows": rows,
        "n_red_blockers": n_red_blockers,
        "n_gates": len(rows),
        "private_mean_at_ceiling": private_mean_at_ceiling,
        "all_green_except_red_blockers": bool(
            all(r["status"] in ("GREEN", "GREEN-CONDITIONAL") for r in rows if not r["is_red_blocker"])),
    }


# --------------------------------------------------------------------------- #
# Step 2 -- the deep-tail-lambda GO-card (the pre-registered decision).
# --------------------------------------------------------------------------- #
def build_gocard(g: dict) -> dict[str, Any]:
    mp = g["mp"]

    # --- the publish-first floor (the #124 gate) in blended-lambda space + its deep-tail. ---
    mu_publish_first = TARGET / g["f_priv"]                       # official needed for private mean = 500
    lambda_publish_first = mp.solve_lambda_for_mu(mu_publish_first)  # blended lambda at the floor
    if g["pubfirst_228_banked"] and _finite(g["pubfirst_228_value"]):
        lambda_publish_first = float(g["pubfirst_228_value"])    # reconcile to kanna #228 if banked
    deeptail_for_publish_first_go = deeptail_for_blended(lambda_publish_first, g)

    # --- the harder P95 private bar (mu_ceiling_needed 535.139): unreachable at the 520.953 ceiling. ---
    lambda_p95 = mp.solve_lambda_for_mu(g["mu_ceiling_needed_p95"])  # None => no lambda<=1 reaches it
    p95_reachable = lambda_p95 is not None
    deeptail_for_p95_private = (deeptail_for_blended(float(lambda_p95), g) if p95_reachable else None)
    official_at_deeptail_1 = float(mp.mu_pub(blended_lambda(1.0, g)))
    p95_shortfall_at_max_deeptail = g["mu_ceiling_needed_p95"] - official_at_deeptail_1

    # --- the PR-specified sweep [0.5, 1.0] (21 rungs) + the GO/HOLD boundary deep-tail. ---
    sweep = []
    n = 21
    for i in range(n):
        dt = DEEPTAIL_SWEEP_LO + (DEEPTAIL_SWEEP_HI - DEEPTAIL_SWEEP_LO) * i / (n - 1)
        sweep.append(evaluate_deeptail(dt, g))
    # pin the exact GO threshold + the interim spine point into the table.
    for dt in (deeptail_for_publish_first_go, g["lambda_spine"], g["deeptail_budget_bar_215"]):
        if DEEPTAIL_SWEEP_LO - 1e-9 <= dt <= DEEPTAIL_SWEEP_HI + 1e-9:
            sweep.append(evaluate_deeptail(dt, g))
    sweep.sort(key=lambda r: r["deeptail_lambda"])

    # --- extended diagnostic BELOW 0.5: confirm NO-GO is structurally unreachable at spine 0.997. ---
    extended = [evaluate_deeptail(dt, g) for dt in (0.0, 0.1, 0.2, 0.3, 0.4)]
    blend_floor = blended_lambda(0.0, g)                          # deep tail fully collapsed
    nogo_reachable_at_spine = bool(blend_floor < g["kernel_floor"])
    spine_below_which_nogo_reachable = g["kernel_floor"] / g["w_shallow"]

    verdicts = [r["verdict"] for r in sweep]
    go_present = "GO" in verdicts
    hold_present = "HOLD" in verdicts
    nogo_present = "NO-GO" in verdicts

    return {
        "axis": "deep-tail reach-weighted lambda_hat over q[8..9] (land #71's number)",
        "sweep_range": [DEEPTAIL_SWEEP_LO, DEEPTAIL_SWEEP_HI],
        "gocard_vs_deeptail_lambda": sweep,
        "extended_diagnostic_below_0p5": extended,
        # the headline thresholds.
        "deeptail_lambda_for_publish_first_go": deeptail_for_publish_first_go,   # TEST metric
        "lambda_publish_first_floor_blended": lambda_publish_first,
        "mu_publish_first_official": mu_publish_first,
        "deeptail_lambda_for_p95_private": deeptail_for_p95_private,             # None => unreachable
        "p95_private_reachable": p95_reachable,
        "lambda_p95_blended": lambda_p95,
        "p95_mu_ceiling_needed": g["mu_ceiling_needed_p95"],
        "official_at_max_deeptail": official_at_deeptail_1,
        "p95_shortfall_even_at_max_deeptail_tps": p95_shortfall_at_max_deeptail,
        # the structural finding: at spine 0.997 the blend floor sits ABOVE the kernel NO-GO floor.
        "blend_floor_deeptail_zero": blend_floor,
        "kernel_nogo_floor": g["kernel_floor"],
        "nogo_reachable_at_interim_spine": nogo_reachable_at_spine,
        "spine_below_which_nogo_reachable": spine_below_which_nogo_reachable,
        "verdict_regions_present": {"GO": go_present, "HOLD": hold_present, "NO-GO": nogo_present},
        "deeptail_budget_bar_215_cross_check": g["deeptail_budget_bar_215"],
        "publish_first_vs_215_bar_budget_delta": deeptail_for_publish_first_go - g["deeptail_budget_bar_215"],
    }


# --------------------------------------------------------------------------- #
# Step 3 -- the "one number" headline.
# --------------------------------------------------------------------------- #
def build_headline(g: dict, card: dict) -> dict[str, Any]:
    dt_go = card["deeptail_lambda_for_publish_first_go"]
    # HOLD band lower edge in deep-tail space = where blended lambda hits the kernel floor; with the
    # interim spine 0.997 this is < 0 (unreachable), so the physical HOLD band is [0, publish-first).
    dt_kernel_floor = deeptail_for_blended(g["kernel_floor"], g)
    hold_band_deeptail = [max(0.0, dt_kernel_floor), dt_go]
    return {
        "LAUNCH_GATED_ON_SINGLE_DEEPTAIL_LAMBDA": True,
        "the_one_number": "land #71's built deep-tail reach-weighted lambda_hat over q[8..9]",
        "deeptail_lambda_for_publish_first_go": dt_go,
        "publish_first_fires_iff_deeptail_ge": dt_go,
        "hold_band_deeptail_physical": hold_band_deeptail,
        "deeptail_at_kernel_floor_unclamped": dt_kernel_floor,
        "kernel_nogo_floor_blended_lambda": g["kernel_floor"],
        "nogo_reachable_at_interim_spine": card["nogo_reachable_at_interim_spine"],
        "statement": (
            "The launch is gated on EXACTLY ONE measurement -- land #71's deep-tail lambda_hat on "
            "q[8..9]. Every other launch gate is GREEN and banked. The #124 publish-first GO FIRES "
            "the moment that number lands >= %.4f (blended tree lambda reaches the publish-first "
            "floor %.4f, official %.2f, private mean >= 500). Below it (down to the kernel floor) "
            "the build is VALID-but-sub-publish-first (HOLD); below blended lambda %.4f it is NO-GO. "
            "With the interim spine 0.997, NO-GO is in fact UNREACHABLE for any deep tail (the blend "
            "floor %.4f sits above the kernel floor), so the live decision is GO-vs-HOLD on one "
            "number." % (dt_go, card["lambda_publish_first_floor_blended"],
                         card["mu_publish_first_official"], g["kernel_floor"],
                         card["blend_floor_deeptail_zero"])),
    }


# --------------------------------------------------------------------------- #
# Step 4 -- honest band.
# --------------------------------------------------------------------------- #
def build_honest_band(g: dict, card: dict) -> dict[str, Any]:
    # (a) spine-interim sensitivity: a lower re-measured spine RAISES the deep-tail threshold.
    spine_sensitivity = []
    for ls in (0.990, 0.995, 0.997, 0.999, 1.000):
        gg = dict(g, lambda_spine=ls)
        dt_go = deeptail_for_blended(card["lambda_publish_first_floor_blended"], gg)
        spine_sensitivity.append({
            "lambda_spine": ls,
            "deeptail_for_publish_first_go": dt_go,
            "blend_floor_deeptail_zero": blended_lambda(0.0, gg),
            "nogo_reachable": bool(blended_lambda(0.0, gg) < g["kernel_floor"]),
        })
    # (e) f_priv sensitivity: the publish-first GREEN flips under the #224 grounded f_priv.
    f_priv_sensitivity = []
    for fp, tag in ((g["f_priv_grounded_band"][0], "grounded_low_0.950"),
                    (g["f_priv_grounded_point"], "grounded_point_0.957_#224"),
                    (g["f_priv"], "pinned_0.969107_#217")):
        mu_pf = TARGET / fp
        lam_pf = g["mp"].solve_lambda_for_mu(mu_pf)
        reachable = lam_pf is not None
        f_priv_sensitivity.append({
            "f_priv": fp, "tag": tag,
            "mu_publish_first": mu_pf,
            "private_mean_at_ceiling": g["ceiling_mu"] * fp,
            "publish_first_reachable_below_ceiling": reachable,
            "deeptail_for_publish_first_go": (deeptail_for_blended(float(lam_pf), g) if reachable else None),
        })
    return {
        "a_spine_is_interim_land71": {
            "lambda_spine_interim": g["lambda_spine"],
            "rule": "the q[2..7] spine is INTERIM (land #71); if it re-measures LOWER the deep-tail "
                    "publish-first threshold RISES (and NO-GO can become reachable).",
            "spine_sensitivity": spine_sensitivity,
            "spine_below_which_nogo_reachable": card["spine_below_which_nogo_reachable"],
        },
        "b_et_map_imported_unchanged": (
            "the blended-lambda -> E[T] map is the #175/#184 reach-DP (via #213 floor/ceiling spines, "
            "ubel #222) imported UNCHANGED; the ceiling 520.953 (#204) anchors official mu_pub."),
        "c_deeptail_is_land71_to_measure": (
            "the deep-tail lambda_hat is what land #71 MEASURES; this leg pre-registers the DECISION "
            "across its [0.5,1.0] axis, not the value. The card is a table lookup once #71 reports."),
        "d_all_gate_values_imported": {
            "rule": "all gate values imported unchanged (no re-derivation).",
            "pubfirst_228_banked": g["pubfirst_228_banked"],
            "pubfirst_228_note": (
                "kanna #228 (publish-first-lambda-floor) has NOT banked a lambda_floor_publish_first; "
                "the publish-first floor is computed here from mu_pub^{-1}(500/f_priv) and the #215 "
                "blend. If #228 lands, reconcile to it (delta vs %.6f)."
                % card["lambda_publish_first_floor_blended"]),
        },
        "e_f_priv_sensitivity": {
            "rule": "the publish-first GREEN uses the PINNED f_priv=0.969107 (#217); under the #224 "
                    "GROUNDED f_priv=0.957 the ceiling private mean falls to 498.58 < 500 and the "
                    "publish-first GO becomes UNREACHABLE below the ceiling (a material caveat for #124).",
            "f_priv_sensitivity": f_priv_sensitivity,
            "p95_unreachable_note": "the harder P95 private bar (535.139) exceeds the 520.953 ceiling "
                                    "at EVERY f_priv -- unreachable without raising the public ceiling.",
        },
    }


# --------------------------------------------------------------------------- #
# Step 5 -- self-test (PRIMARY metric).
# --------------------------------------------------------------------------- #
def self_test(g: dict, scorecard: dict, card: dict) -> dict[str, Any]:
    results: dict[str, Any] = {}
    mp = g["mp"]

    # (a) at deep-tail = spine 0.997 the card reproduces the all-spine blended E[T]/official to tol.
    row = evaluate_deeptail(g["lambda_spine"], g)
    et_all_spine = float(mp.et_of_lambda(g["lambda_spine"]))
    official_all_spine = float(mp.mu_pub(g["lambda_spine"]))
    a_ok = (abs(row["blended_tree_lambda"] - g["lambda_spine"]) < 1e-12
            and abs(row["E_T"] - et_all_spine) < ROUNDTRIP_TOL
            and abs(row["official_tps"] - official_all_spine) < 1e-7)
    results["a_deeptail_eq_spine_reproduces_all_spine"] = {
        "pass": bool(a_ok), "blended_lambda": row["blended_tree_lambda"],
        "E_T": row["E_T"], "E_T_all_spine": et_all_spine,
        "official": row["official_tps"], "official_all_spine": official_all_spine}

    # (b) the GO verdict is monotone in deep-tail (the GO region is an upper interval).
    grid = [evaluate_deeptail(0.0 + i / 200.0, g) for i in range(201)]  # deeptail 0..1
    go_idx = [i for i, r in enumerate(grid) if r["verdict"] == "GO"]
    if go_idx:
        first_go = min(go_idx)
        go_is_upper_interval = all(grid[i]["verdict"] == "GO" for i in range(first_go, len(grid))) \
            and all(grid[i]["verdict"] != "GO" for i in range(0, first_go))
    else:
        go_is_upper_interval = True   # vacuously an (empty) upper interval
    blended_seq = [r["blended_tree_lambda"] for r in grid]
    blend_monotone = all(blended_seq[i + 1] >= blended_seq[i] - 1e-15 for i in range(len(blended_seq) - 1))
    b_ok = bool(go_is_upper_interval and blend_monotone)
    results["b_go_monotone_upper_interval"] = {
        "pass": b_ok, "go_is_upper_interval": bool(go_is_upper_interval),
        "blended_lambda_monotone_in_deeptail": bool(blend_monotone),
        "n_go_on_grid": len(go_idx)}

    # (c) deeptail_for_publish_first_go < deeptail_for_p95_private (point-estimate easier than P95).
    dt_pf = card["deeptail_lambda_for_publish_first_go"]
    dt_p95 = card["deeptail_lambda_for_p95_private"]      # None => unreachable (== +inf, strictly harder)
    c_ok = bool(_finite(dt_pf) and (dt_p95 is None or dt_pf < dt_p95))
    results["c_publish_first_easier_than_p95"] = {
        "pass": c_ok, "deeptail_for_publish_first_go": dt_pf,
        "deeptail_for_p95_private": dt_p95, "p95_reachable": card["p95_private_reachable"]}

    # (d) n_red_blockers == 1 (the single deep-tail lambda_hat).
    d_ok = bool(scorecard["n_red_blockers"] == 1
                and any(r["gate"] == "deeptail-lambda-q8q9" and r["status"] == "RED"
                        for r in scorecard["rows"]))
    results["d_one_red_blocker"] = {
        "pass": d_ok, "n_red_blockers": scorecard["n_red_blockers"]}

    # (e) round-trip fern #185 GO/NO-GO at fully-specified tuples (GO at lambda=1, NO-GO at the floor).
    d_go = LTC.launch_decision(LTC.synth_land71_tuple("gocard-roundtrip-lambda1", 1.0),
                               step_override=LTC._M.shipped_step)
    d_nogo = LTC.launch_decision(LTC.synth_land71_tuple("gocard-roundtrip-floor", LTC._M.lam_hat_liveprobe),
                                 step_override=LTC._M.shipped_step)
    blk = d_go.get("filled_approval_block")
    e_ok = (d_go["verdict"] == "GO" and isinstance(blk, str)
            and "Approval request: HF job for" in blk and "human must approve" in blk
            and d_nogo["verdict"] == "NO-GO" and d_go["headline_topology"] == "both_bugs")
    results["e_roundtrip_185_go_nogo"] = {
        "pass": bool(e_ok), "verdict_at_lambda1": d_go["verdict"],
        "verdict_at_floor": d_nogo["verdict"],
        "approval_block_wellformed": bool(isinstance(blk, str) and "Approval request: HF job for" in (blk or "")),
        "authorizes_nothing": bool(d_go["launch_authorized"]["authorized"] is False)}

    # (f) NaN-clean across every reported numeric (strings/None excluded).
    payload_numeric = {"scorecard": scorecard, "gocard": card,
                       "gates": {k: v for k, v in g.items() if k not in ("b", "mp", "provenance")}}
    nan_paths = _nan_paths(payload_numeric, "selftest")
    f_ok = (len(nan_paths) == 0)
    results["f_nan_clean"] = {"pass": bool(f_ok), "nan_paths": nan_paths}

    passes = bool(all(v["pass"] for v in results.values()))
    return {
        "launch_readiness_gocard_self_test_passes": passes,
        "deeptail_lambda_for_publish_first_go": card["deeptail_lambda_for_publish_first_go"],
        "conditions": results,
    }


# --------------------------------------------------------------------------- #
# Step 6 -- the one-sentence hand-off.
# --------------------------------------------------------------------------- #
def build_handoff(g: dict, scorecard: dict, card: dict) -> str:
    dt_go = card["deeptail_lambda_for_publish_first_go"]
    return (
        "the launch is GREEN on every gate except ONE -- land #71's deep-tail lambda_hat over "
        "q[8..9] (the only RED of %d gates); the GO-card says publish-first FIRES the moment it "
        "lands >= %.4f (blended tree lambda %.4f, official %.2f, private mean >= 500), HOLDS for "
        "the int4 kernel in [the kernel floor %.4f, that) -- and with the interim spine 0.997 the "
        "blend floor %.4f sits ABOVE the kernel floor so NO-GO is unreachable -- so we are waiting "
        "on exactly ONE number and the #124 publish-first decision is pre-registered for every "
        "value it can take." % (
            scorecard["n_gates"], dt_go, card["lambda_publish_first_floor_blended"],
            card["mu_publish_first_official"], g["kernel_floor"], card["blend_floor_deeptail_zero"]))


# --------------------------------------------------------------------------- #
# Synthesis orchestration.
# --------------------------------------------------------------------------- #
def run(args) -> dict[str, Any]:
    t0 = time.time()
    g = load_gates()
    scorecard = build_scorecard(g)
    card = build_gocard(g)
    headline = build_headline(g, card)
    honest = build_honest_band(g, card)
    st = self_test(g, scorecard, card)
    handoff = build_handoff(g, scorecard, card)

    payload = {
        "pr": 231,
        "agent": "fern",
        "kind": "launch_readiness_gocard",
        "primary_metric_name": "launch_readiness_gocard_self_test_passes",
        "launch_readiness_gocard_self_test_passes": st["launch_readiness_gocard_self_test_passes"],
        "test_metric_name": "deeptail_lambda_for_publish_first_go",
        "deeptail_lambda_for_publish_first_go": st["deeptail_lambda_for_publish_first_go"],
        "launch_scorecard": scorecard,
        "n_red_blockers": scorecard["n_red_blockers"],
        "gocard": card,
        "headline": headline,
        "honest_band": honest,
        "self_test": st,
        "handoff_line": handoff,
        "constants": {
            "validity_bar": g["validity_bar"], "validity_bar_nominal": g["validity_bar_nominal"],
            "trigger_worstcase": g["trigger_worstcase"], "ceiling_mu": g["ceiling_mu"],
            "mu_pub_at_validity_bar": g["mu_pub_at_validity_bar"],
            "speed_margin_at_validity_bar": g["speed_margin_at_validity_bar"],
            "f_priv_pinned": g["f_priv"], "f_priv_grounded_point": g["f_priv_grounded_point"],
            "kernel_floor": g["kernel_floor"], "lambda_spine": g["lambda_spine"],
            "w_shallow": g["w_shallow"], "w_deep": g["w_deep"],
            "deeptail_budget_bar_215": g["deeptail_budget_bar_215"],
            "mu_ceiling_needed_p95": g["mu_ceiling_needed_p95"],
            "target_private_mean": TARGET, "headline_regime": HEADLINE_REGIME,
        },
        "provenance": g["provenance"],
        "scope": ("LOCAL CPU-only integration of the banked launch gates into a readiness scorecard "
                  "+ a deep-tail-lambda-parameterized GO-card, reusing fern #185's GO/NO-GO logic; "
                  "all gate values and the E[T] blend imported unchanged; the deep-tail lambda stays "
                  "land #71's to measure; authorizes NOTHING. NOT a launch. NOT open2."),
        "elapsed_sec": time.time() - t0,
        "peak_mem_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    return payload


# --------------------------------------------------------------------------- #
# wandb + main.
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if args.no_wandb:
        return
    try:
        import wandb
    except Exception as exc:               # noqa: BLE001
        print(f"[wandb] unavailable ({exc}); skipping.", file=sys.stderr)
        return
    c = payload["constants"]
    card = payload["gocard"]
    sc = payload["launch_scorecard"]
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_name, group=args.wandb_group,
                     config={
                         "pr": 231, "agent": "fern", "kind": "launch_readiness_gocard",
                         "validity_bar": c["validity_bar"], "trigger_worstcase": c["trigger_worstcase"],
                         "ceiling_mu": c["ceiling_mu"], "f_priv_pinned": c["f_priv_pinned"],
                         "f_priv_grounded_point": c["f_priv_grounded_point"],
                         "kernel_floor": c["kernel_floor"], "lambda_spine": c["lambda_spine"],
                         "w_shallow": c["w_shallow"], "w_deep": c["w_deep"],
                         "deeptail_budget_bar_215": c["deeptail_budget_bar_215"],
                         "mu_ceiling_needed_p95": c["mu_ceiling_needed_p95"],
                         "target_private_mean": c["target_private_mean"],
                         "headline_regime": c["headline_regime"],
                         "pubfirst_228_banked": payload["honest_band"]["d_all_gate_values_imported"]["pubfirst_228_banked"],
                         "provenance": payload["provenance"], "scope": payload["scope"],
                     })
    summary = {
        "launch_readiness_gocard_self_test_passes": payload["launch_readiness_gocard_self_test_passes"],
        "deeptail_lambda_for_publish_first_go": payload["deeptail_lambda_for_publish_first_go"],
        "n_red_blockers": payload["n_red_blockers"],
        "lambda_publish_first_floor_blended": card["lambda_publish_first_floor_blended"],
        "mu_publish_first_official": card["mu_publish_first_official"],
        "deeptail_lambda_for_p95_private_reachable": card["p95_private_reachable"],
        "p95_shortfall_even_at_max_deeptail_tps": card["p95_shortfall_even_at_max_deeptail_tps"],
        "blend_floor_deeptail_zero": card["blend_floor_deeptail_zero"],
        "nogo_reachable_at_interim_spine": card["nogo_reachable_at_interim_spine"],
        "spine_below_which_nogo_reachable": card["spine_below_which_nogo_reachable"],
        "private_mean_at_ceiling": sc["private_mean_at_ceiling"],
        "speed_margin_at_validity_bar": c["speed_margin_at_validity_bar"],
        "mu_pub_at_validity_bar": c["mu_pub_at_validity_bar"],
        "publish_first_vs_215_bar_budget_delta": card["publish_first_vs_215_bar_budget_delta"],
        "elapsed_sec": payload["elapsed_sec"], "peak_mem_mib": payload["peak_mem_mib"],
    }
    for k, v in list(summary.items()):
        if isinstance(v, float) and not math.isfinite(v):
            summary[k] = None
    wandb.log(summary)
    wandb.summary.update({k: v for k, v in summary.items() if v is not None})

    # the scorecard + GO-card sweep as tables.
    sc_tbl = wandb.Table(columns=["gate", "status", "threshold", "source", "is_red_blocker"])
    for r in sc["rows"]:
        sc_tbl.add_data(r["gate"], r["status"], r["threshold"], r["source"], bool(r["is_red_blocker"]))
    gc_tbl = wandb.Table(columns=["deeptail_lambda", "blended_tree_lambda", "E_T",
                                  "official_tps", "private_mean_tps", "verdict"])
    for r in card["gocard_vs_deeptail_lambda"]:
        gc_tbl.add_data(r["deeptail_lambda"], r["blended_tree_lambda"], r["E_T"],
                        r["official_tps"], r["private_mean_tps"], r["verdict"])
    wandb.log({"launch_scorecard": sc_tbl, "gocard_vs_deeptail_lambda": gc_tbl})

    art = wandb.Artifact("launch_readiness_gocard_results", type="analysis")
    with art.new_file("launch_readiness_gocard_results.json", mode="w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    run.log_artifact(art)
    wandb.finish()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR #231 launch-readiness GO-card")
    ap.add_argument("--out",
                    default="research/validity/launch_readiness_gocard/launch_readiness_gocard_results.json")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", "--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", "--wandb_entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-name", "--wandb_name", default="fern/launch-readiness-gocard")
    ap.add_argument("--wandb-group", "--wandb_group", default="launch-readiness-gocard")
    args = ap.parse_args(argv)

    payload = run(args)
    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)

    st = payload["self_test"]
    print("launch_readiness_gocard_self_test_passes =",
          st["launch_readiness_gocard_self_test_passes"])
    print("deeptail_lambda_for_publish_first_go =", st["deeptail_lambda_for_publish_first_go"])
    print("n_red_blockers =", payload["n_red_blockers"])
    for cond, v in st["conditions"].items():
        print(f"  {cond}: {'PASS' if v['pass'] else 'FAIL'}")
    print("\nSCORECARD:")
    for r in payload["launch_scorecard"]["rows"]:
        print(f"  [{r['status']:>16}] {r['gate']:<34} <- {r['source']}")
    print("\nHEADLINE:", payload["headline"]["statement"])
    print("\nHANDOFF:", payload["handoff_line"])
    nan_paths = _nan_paths(payload, "payload")
    if nan_paths:
        print("\n[WARN] NaN paths:", nan_paths, file=sys.stderr)
    _maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
