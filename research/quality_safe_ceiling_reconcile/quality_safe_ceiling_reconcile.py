#!/usr/bin/env python3
"""PR #570 — Reconcile the 3 quality-safe head-lever ceilings to #553's corrected 299.29.

A pure-CPU SYNTHESIS card (the #561/#565 pattern). It brings the capstone
(lawine #561, ``v74ad5jb``) numerically CURRENT by re-deriving the three banked
quality-safe head-lever ceilings into ONE reconciled record, using wirbel #553's
CORRECTED plain-int4-head number (299.29, up from the superseded 292.1 projection)
and fern #560's REALIZED candidate-verify number (291.36, down from the 305.4
projection #561 carried). NOTHING here is measured — every input is a constant
that is GROUNDED against an on-branch, already-merged, W&B-verified source
artifact (loaded + asserted at runtime; no asserted-but-unchecked numbers). No
GPU, no served job, no microbench, no HF launch, no submission, no served-file
change.

The verdict that falls out is UNCHANGED after the #553 upward correction: the
three quality-safe ceilings order as
  candidate-verify 291.36 (#319 PASS)  <  plain-int4-head 299.29 (#319 FAIL, PPL-safe)  <  magically-free floor 311.25 (upper bound)
and even the LOOSEST (311.25) sits 64.61 below the 375.857 official ship and
188.75 below the 500 gate -> ``two_gate_satisfiable = FALSE``, NO-FIRE.

Outputs: a W&B run (group ``quality-safe-ceiling-reconcile``) logging the
reconciled fields + a JSON artifact + a single-line SENPAI-RESULT.

Run under the wandb-capable venv (``.venv/bin/python``); imports wandb first so
the cached real module beats the ``./wandb`` run-data namespace shadow.
"""
from __future__ import annotations

# --- real-wandb-first (beats ./wandb namespace shadow); harmless if absent ---
try:  # pragma: no cover
    import wandb as _wandb_real  # noqa: F401  (cache the real module in sys.modules)
except Exception:  # pragma: no cover
    _wandb_real = None

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research" / "quality_safe_ceiling_reconcile"
OUT_JSON = HERE / "quality_safe_ceiling_reconcile.json"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # APPEND so site-packages wandb wins over ./wandb

# ======================================================================
# ABSOLUTE ANCHORS — program contract (constants, not measurements).
# ======================================================================
SHIP_OFFICIAL_TPS = 375.857   # the quality-safe served ship to beat for a FIRE (official a10g-small)
GATE_TPS = 500.0              # the 500 official gate
BASELINE_OFFICIAL_1_TPS = 481.53  # PR-registered official public #1 (untouched)
SIGMA_HW_TPS = 4.864          # wirbel #553 hardware sigma (run-to-run TPS noise)

# ======================================================================
# BANKED INPUTS — every value GROUNDED at runtime against its on-branch,
# already-merged, W&B-verified source artifact (see SOURCES below).
# ======================================================================

# --- (1) candidate-verify: identity-safe head lever, #319 PASS ----------
#     fern #560 (ufv4nk21), group candidate-verify-realize, MERGED 3b85e85.
#     Realized (re-projected on the 252.31 anchor); SUPERSEDES the 305.4
#     projection #561 carried. argmax-identity 1.0 IFF vocab-index tie-break.
CV_LOCAL_TPS = 291.36223894548687
CV_REALIZED_GAIN = 39.05223894548686
CV_IDENTITY_RATE = 1.0
CV_319_STATUS = "PASS"  # identity-safe by construction (exact bf16 verify of top-8)

# --- (2) plain-int4-head: PPL-safe but BREAKS #319 (the CORRECTION) ------
#     wirbel #553 (bo43du3w), group realized-anchor-tps, MERGED 578594c.
#     SUPERSEDES lawine #544's 292.1 projection: realized +46.6 (1.217x),
#     ceiling corrects UPWARD +7.19 (+1.48 sigma_hw ceiling-space /
#     +1.71 sigma_hw gain-space). PPL delta int4-bf16 = -0.003 (quality-safe).
PLAIN_LOCAL_TPS = 299.28668280432635
PLAIN_OFFICIAL_TPS = 309.8322445983153   # = local * tau_lo (source-banked)
PLAIN_REALIZED_GAIN = 46.59826440514672
PLAIN_REALIZED_VS_544 = 1.2166648669751103
PLAIN_PPL_DELTA = -0.002994660457457332
PLAIN_ARGMAX_FLIP_RATE = 0.0076297049847405905
PLAIN_319_STATUS = "FAIL"  # strict greedy identity broken (~0.76% free-run flip)
SUPERSEDED_544_CEILING_TPS = 292.1        # the number #561 carried; now corrected

# --- (3) magically-free floor: head bytes -> 0, body intact (UPPER bound)-
#     lawine #554 (fi8vr1nb), group fixed-overhead-ceiling; carried by the
#     #561 capstone (v74ad5jb). UNCHANGED by the #553 correction.
FLOOR_LOCAL_TPS = 311.2485991465399
FLOOR_319_STATUS = "N/A"  # unrealizable upper bound (no real lever reaches it)

# --- shared anchor + local->official transfer --------------------------
ANCHOR_LOCAL_TPS = 252.68841839917962     # wirbel #553 (83jiwjr9) true surgical-path served anchor
TAU_LO = 1.0352356533046398               # #267 stable local->official scalar (wirbel #553 banked)

# --- the capstone this card brings current -----------------------------
CAPSTONE_CEILING_TPS = 311.2485991465399  # lawine #561 base_fullhead_quality_safe_ceiling_tps
CAPSTONE_GAP_TO_SHIP_TPS = 64.61          # lawine #561 gap_to_current_ship_tps (rounded)

# --- on-branch source artifacts (merged; W&B-verified at merge time) ----
SOURCES: dict[str, dict[str, Any]] = {
    "candidate_verify": {
        "pr": "fern #560", "wandb": "ufv4nk21", "group": "candidate-verify-realize",
        "merged": "3b85e85",
        "path": "research/candidate_verify_realize/stage2_reproject.json",
        "asserts": {  # field -> expected constant
            "cv_realized_quality_safe_tps": CV_LOCAL_TPS,
            "cv_realized_tps_gain": CV_REALIZED_GAIN,
            "argmax_identity_rate": CV_IDENTITY_RATE,
            "identity_hard_gate_pass": True,
        },
    },
    "plain_int4_head": {
        "pr": "wirbel #553", "wandb": "bo43du3w", "group": "realized-anchor-tps",
        "merged": "578594c",
        "path": "research/realized_anchor_tps/summary.json",
        "asserts": {
            "precision_head_served_tps": PLAIN_LOCAL_TPS,
            "precision_head_implied_official_tps": PLAIN_OFFICIAL_TPS,
            "realized_precision_gain": PLAIN_REALIZED_GAIN,
            "realized_vs_lawine38": PLAIN_REALIZED_VS_544,
            "ppl_delta_int4_minus_bf16": PLAIN_PPL_DELTA,
            "precision_head_argmax_flip_rate": PLAIN_ARGMAX_FLIP_RATE,
            "precision_head_strict_identity": False,
            "ppl_quality_safe": True,
            "lawine544_ceiling_confirmed": False,
            "anchor_served_tps": ANCHOR_LOCAL_TPS,
            "tau_lo": TAU_LO,
        },
    },
    "magically_free_floor": {
        "pr": "lawine #554", "wandb": "fi8vr1nb", "group": "fixed-overhead-ceiling",
        "merged": "(carried by #561)",
        "path": "research/speed/fixed_overhead_ceiling/fixed_overhead_ceiling.json",
        "asserts_nested": {  # ("a","b") -> expected
            ("verdict", "fixed_overhead_bounded_ceiling_tps"): FLOOR_LOCAL_TPS,
            ("verdict", "corrected_quality_safe_hard_ceiling"): FLOOR_LOCAL_TPS,
            ("verdict", "quality_safe_ship_can_beat_442"): False,
        },
    },
    "capstone_561": {
        "pr": "lawine #561", "wandb": "v74ad5jb", "group": "two-gate-fire-decision",
        "merged": "3bd83e1",
        "path": "research/two_gate_fire_decision/two_gate_fire_decision.json",
        "asserts_nested": {
            ("stage2_gaps", "base_fullhead_quality_safe_ceiling_tps"): CAPSTONE_CEILING_TPS,
            ("stage2_gaps", "base_fullhead_quality_safe_ceiling_tps_strict"): SUPERSEDED_544_CEILING_TPS,
            ("stage1_truth_table", "two_gate_satisfiable"): False,
        },
    },
}

TOL = 1e-9


def _get_nested(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = d
    for k in keys:
        cur = cur[k]
    return cur


def ground_against_sources() -> dict[str, Any]:
    """Load each merged source artifact and assert every cited constant matches.

    This is the #561 self-test discipline made literal: no asserted-but-unchecked
    numbers. Returns a per-source report; ``all_grounded`` is the conjunction.
    """
    report: dict[str, Any] = {}
    all_ok = True
    for key, spec in SOURCES.items():
        path = ROOT / spec["path"]
        checks: dict[str, bool] = {}
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover
            report[key] = {"loaded": False, "error": str(exc),
                           "pr": spec["pr"], "wandb": spec["wandb"]}
            all_ok = False
            continue
        for field, expected in spec.get("asserts", {}).items():
            got = data.get(field, None)
            ok = (got == expected) if isinstance(expected, bool) else (
                got is not None and math.isclose(float(got), float(expected), abs_tol=TOL))
            checks[field] = bool(ok)
        for keys, expected in spec.get("asserts_nested", {}).items():
            try:
                got = _get_nested(data, keys)
            except Exception:
                got = None
            ok = (got == expected) if isinstance(expected, bool) else (
                got is not None and math.isclose(float(got), float(expected), abs_tol=TOL))
            checks[".".join(keys)] = bool(ok)
        src_ok = all(checks.values())
        all_ok = all_ok and src_ok
        report[key] = {"loaded": True, "pr": spec["pr"], "wandb": spec["wandb"],
                       "group": spec["group"], "path": spec["path"],
                       "checks": checks, "all_match": src_ok}
    report["all_grounded"] = all_ok
    return report


def build_packet() -> dict[str, Any]:
    grounding = ground_against_sources()

    # ---- the three reconciled quality-safe head-lever ceilings ----------
    def official_proxy(local: float) -> float:
        return local * TAU_LO

    def row(name: str, local: float, status_319: str, *, role: str,
            source_pr: str, wandb: str, official_override: float | None = None) -> dict[str, Any]:
        off = official_override if official_override is not None else official_proxy(local)
        return {
            "ceiling": name,
            "role": role,
            "served_tps_local": local,
            "official_proxy": off,
            "official_proxy_method": ("source-banked (local*tau_lo)" if official_override is not None
                                      else "local*tau_lo"),
            "#319_status": status_319,
            "gap_to_ship": SHIP_OFFICIAL_TPS - local,        # 375.857 - local (PR definition)
            "gap_to_500_gate": GATE_TPS - local,             # 500 - local (PR definition)
            "source_pr": source_pr,
            "source_wandb": wandb,
        }

    candidate_verify = row(
        "candidate-verify", CV_LOCAL_TPS, CV_319_STATUS,
        role="identity-safe head lever (cheap int4 top-8 nominator -> exact bf16 verify -> re-argmax; "
             "identity 1.0 IFF vocab-index tie-break). Realized; SUPERSEDES the 305.4 projection #561 carried.",
        source_pr="fern #560", wandb="ufv4nk21")
    plain_int4_head = row(
        "plain-int4-head", PLAIN_LOCAL_TPS, PLAIN_319_STATUS,
        role="PPL-safe (ppl delta -0.003) but BREAKS #319 (~0.76% greedy flip). THE CORRECTION: "
             "realized +46.6 (1.217x), ceiling corrects UPWARD from the superseded 292.1 projection.",
        source_pr="wirbel #553", wandb="bo43du3w", official_override=PLAIN_OFFICIAL_TPS)
    magically_free_floor = row(
        "magically-free-floor", FLOOR_LOCAL_TPS, FLOOR_319_STATUS,
        role="body+overhead floor, head bytes -> 0 (UNREALIZABLE upper bound). UNCHANGED by the #553 correction.",
        source_pr="lawine #554", wandb="fi8vr1nb")

    ceilings = [candidate_verify, plain_int4_head, magically_free_floor]

    # ---- the loosest ceiling drives the tightest (smallest) gap ---------
    loosest = max(ceilings, key=lambda r: r["served_tps_local"])  # 311.25
    gap_to_ship_min = loosest["gap_to_ship"]                      # ~64.61
    gap_to_500_min = loosest["gap_to_500_gate"]                   # ~188.75

    # ---- the two-gate NO-FIRE conjunction (PR step 3) -------------------
    # two_gate_satisfiable iff SOME quality-safe ceiling >= the ship.
    any_ceiling_clears_ship = any(r["served_tps_local"] >= SHIP_OFFICIAL_TPS for r in ceilings)
    any_ceiling_clears_gate = any(r["served_tps_local"] >= GATE_TPS for r in ceilings)
    two_gate_satisfiable = bool(any_ceiling_clears_ship and any_ceiling_clears_gate)
    fire_decision = "FIRE" if two_gate_satisfiable else "NO-FIRE"

    # ---- the reconciliation deltas vs the superseded #561 numbers -------
    plain_correction_delta = PLAIN_LOCAL_TPS - SUPERSEDED_544_CEILING_TPS         # +7.19 (UP)
    plain_correction_sigma = plain_correction_delta / SIGMA_HW_TPS                # +1.48 sigma_hw
    correction_is_upward = plain_correction_delta > 0
    # the conjunction would only flip if the corrected ceiling reached the ship:
    conjunction_flip_needs_tps = SHIP_OFFICIAL_TPS - PLAIN_LOCAL_TPS             # ~76.57 still short
    correction_changes_conjunction = (PLAIN_LOCAL_TPS >= SHIP_OFFICIAL_TPS)      # False

    # ---- ordering insight the correction produces ----------------------
    # pre-#553: strict precision 292.1 < cand-verify-proj 305.4; post: cand-verify
    # realized 291.36 < plain-int4 299.29 (the +8 TPS verify pass is now visible).
    ordering_ok = CV_LOCAL_TPS < PLAIN_LOCAL_TPS < FLOOR_LOCAL_TPS

    # ---- self-tests (deterministic; grounding + arithmetic + conjunction) ----
    self_tests = {
        # --- grounding: every cited number matched its source artifact ---
        "all_numbers_grounded_against_sources": grounding["all_grounded"],
        "source_candidate_verify_matches": grounding["candidate_verify"]["all_match"],
        "source_plain_int4_head_matches": grounding["plain_int4_head"]["all_match"],
        "source_magically_free_floor_matches": grounding["magically_free_floor"]["all_match"],
        "source_capstone_561_matches": grounding["capstone_561"]["all_match"],
        # --- the #553 correction is genuinely upward and supersedes 292.1 -
        "correction_is_upward": correction_is_upward,
        "corrected_above_superseded_292": PLAIN_LOCAL_TPS > SUPERSEDED_544_CEILING_TPS,
        "correction_delta_is_7_19": math.isclose(plain_correction_delta, 7.18668280432636, abs_tol=1e-6),
        "correction_sigma_is_1_48": math.isclose(plain_correction_sigma, 1.4775252475999845, abs_tol=1e-6),
        "plain_realized_over_realizes_544": PLAIN_REALIZED_VS_544 > 1.0,
        # --- ordering after correction: CV < plain-int4 < floor ----------
        "ordering_cv_lt_plain_lt_floor": ordering_ok,
        # --- all three ceilings below the ship and the gate --------------
        "candidate_verify_below_ship": CV_LOCAL_TPS < SHIP_OFFICIAL_TPS,
        "plain_int4_head_below_ship": PLAIN_LOCAL_TPS < SHIP_OFFICIAL_TPS,
        "loosest_floor_below_ship": FLOOR_LOCAL_TPS < SHIP_OFFICIAL_TPS,
        "loosest_floor_below_gate": FLOOR_LOCAL_TPS < GATE_TPS,
        # --- gap_to_ship_min is the loosest ceiling's gap (~64.61) -------
        "gap_to_ship_min_from_loosest": math.isclose(gap_to_ship_min, SHIP_OFFICIAL_TPS - FLOOR_LOCAL_TPS, abs_tol=TOL),
        "gap_to_ship_min_is_64_61": math.isclose(round(gap_to_ship_min, 2), 64.61, abs_tol=1e-9),
        "gap_to_500_min_is_188_75": math.isclose(round(gap_to_500_min, 2), 188.75, abs_tol=1e-9),
        "gap_to_ship_min_matches_capstone": math.isclose(round(gap_to_ship_min, 2), CAPSTONE_GAP_TO_SHIP_TPS, abs_tol=1e-9),
        # --- the two-gate conjunction: NO-FIRE, UNCHANGED ----------------
        "no_ceiling_clears_ship": (not any_ceiling_clears_ship),
        "no_ceiling_clears_gate": (not any_ceiling_clears_gate),
        "two_gate_unsatisfiable": (not two_gate_satisfiable),
        "fire_is_nofire": fire_decision == "NO-FIRE",
        "correction_does_not_change_conjunction": (not correction_changes_conjunction),
        # --- #319 statuses are as banked ---------------------------------
        "candidate_verify_is_319_pass": CV_319_STATUS == "PASS",
        "plain_int4_head_is_319_fail": PLAIN_319_STATUS == "FAIL",
        "plain_int4_head_is_ppl_safe": PLAIN_PPL_DELTA < 0.01,
        # --- official proxies exceed local but stay below the ship -------
        "official_proxy_above_local": all(r["official_proxy"] > r["served_tps_local"] for r in ceilings),
        "loosest_official_proxy_below_ship": magically_free_floor["official_proxy"] < SHIP_OFFICIAL_TPS,
        "plain_official_proxy_matches_source": math.isclose(
            plain_int4_head["official_proxy"], PLAIN_OFFICIAL_TPS, abs_tol=TOL),
        # --- NaN-clean ---------------------------------------------------
        "nan_clean": all(math.isfinite(x) for x in [
            CV_LOCAL_TPS, PLAIN_LOCAL_TPS, FLOOR_LOCAL_TPS, PLAIN_OFFICIAL_TPS,
            gap_to_ship_min, gap_to_500_min, plain_correction_delta, plain_correction_sigma,
            candidate_verify["official_proxy"], magically_free_floor["official_proxy"],
        ]),
    }
    self_tests["self_test_passes"] = all(self_tests.values())
    self_det = bool(self_tests["self_test_passes"])

    packet = {
        "pr": 570,
        "card": "quality-safe-ceiling-reconcile",
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
        "no_hf_job": True,
        "no_submission": True,
        "peak_gpu_gib": 0.0,
        "baseline_official_1_tps": BASELINE_OFFICIAL_1_TPS,
        "ship_official_tps": SHIP_OFFICIAL_TPS,
        "gate_tps": GATE_TPS,
        "tau_lo": TAU_LO,
        # ---- the reconciled 3-ceiling table ----
        "reconciled_ceilings": {
            "candidate_verify": candidate_verify,
            "plain_int4_head": plain_int4_head,
            "magically_free_floor": magically_free_floor,
        },
        "loosest_ceiling": loosest["ceiling"],
        "gap_to_ship_min": gap_to_ship_min,
        "gap_to_500_min": gap_to_500_min,
        # ---- the #553 correction reconciliation ----
        "correction": {
            "superseded_544_ceiling_tps": SUPERSEDED_544_CEILING_TPS,
            "corrected_plain_int4_head_tps": PLAIN_LOCAL_TPS,
            "correction_delta_tps": plain_correction_delta,
            "correction_delta_sigma_hw": plain_correction_sigma,
            "correction_is_upward": correction_is_upward,
            "realized_gain_tps": PLAIN_REALIZED_GAIN,
            "realized_vs_544_projection": PLAIN_REALIZED_VS_544,
            "lawine544_ceiling_confirmed": False,
            "correction_changes_conjunction": correction_changes_conjunction,
            "conjunction_flip_needs_tps_over_corrected": conjunction_flip_needs_tps,
            "note": ("plain-int4-head corrects UPWARD 292.1->299.29 (+7.19 = +1.48 sigma_hw "
                     "ceiling-space); candidate-verify corrects DOWNWARD 305.4-proj->291.36-realized. "
                     "Both remain << the 375.857 ship -> conjunction UNCHANGED."),
        },
        # ---- the two-gate conjunction ----
        "two_gate_satisfiable": two_gate_satisfiable,
        "fire_decision": fire_decision,
        "any_ceiling_clears_ship": any_ceiling_clears_ship,
        "any_ceiling_clears_gate": any_ceiling_clears_gate,
        # ---- grounding report ----
        "grounding": grounding,
        "sources": {k: {kk: vv for kk, vv in v.items() if kk not in ("asserts", "asserts_nested")}
                    for k, v in SOURCES.items()},
        # ---- self-tests ----
        "self_tests": self_tests,
        "self_det": self_det,
        # ---- KEY OUTPUTS (PR-named) ----
        "quality_safe_319_safe_ceiling_tps": CV_LOCAL_TPS,
        "quality_safe_ppl_only_ceiling_tps": PLAIN_LOCAL_TPS,
        "magically_free_floor_tps": FLOOR_LOCAL_TPS,
        "self_tests_passed": sum(1 for v in self_tests.values() if v),
        "self_tests_total": len(self_tests),
        "primary_metric_name": "quality_safe_ppl_only_ceiling_tps",
        "primary_metric_value": PLAIN_LOCAL_TPS,
    }
    return packet


def wandb_summary(p: dict[str, Any]) -> dict[str, Any]:
    rc = p["reconciled_ceilings"]
    cv, pl, fl = rc["candidate_verify"], rc["plain_int4_head"], rc["magically_free_floor"]
    cor = p["correction"]
    return {
        # --- KEY OUTPUTS ---
        "quality_safe_319_safe_ceiling_tps": p["quality_safe_319_safe_ceiling_tps"],
        "quality_safe_ppl_only_ceiling_tps": p["quality_safe_ppl_only_ceiling_tps"],
        "magically_free_floor_tps": p["magically_free_floor_tps"],
        "gap_to_ship_min": p["gap_to_ship_min"],
        "gap_to_500_min": p["gap_to_500_min"],
        "two_gate_satisfiable": p["two_gate_satisfiable"],
        "two_gate_satisfiable_int": int(p["two_gate_satisfiable"]),
        "fire_decision": p["fire_decision"],
        "fire_is_nofire_int": int(p["fire_decision"] == "NO-FIRE"),
        # --- per-ceiling table ---
        "cv_served_tps_local": cv["served_tps_local"],
        "cv_official_proxy": cv["official_proxy"],
        "cv_319_status": cv["#319_status"],
        "cv_gap_to_ship": cv["gap_to_ship"],
        "cv_gap_to_500_gate": cv["gap_to_500_gate"],
        "plain_served_tps_local": pl["served_tps_local"],
        "plain_official_proxy": pl["official_proxy"],
        "plain_319_status": pl["#319_status"],
        "plain_gap_to_ship": pl["gap_to_ship"],
        "plain_gap_to_500_gate": pl["gap_to_500_gate"],
        "floor_served_tps_local": fl["served_tps_local"],
        "floor_official_proxy": fl["official_proxy"],
        "floor_319_status": fl["#319_status"],
        "floor_gap_to_ship": fl["gap_to_ship"],
        "floor_gap_to_500_gate": fl["gap_to_500_gate"],
        # --- the correction ---
        "superseded_544_ceiling_tps": cor["superseded_544_ceiling_tps"],
        "correction_delta_tps": cor["correction_delta_tps"],
        "correction_delta_sigma_hw": cor["correction_delta_sigma_hw"],
        "correction_is_upward_int": int(cor["correction_is_upward"]),
        "correction_changes_conjunction_int": int(cor["correction_changes_conjunction"]),
        "realized_vs_544_projection": cor["realized_vs_544_projection"],
        # --- grounding / meta ---
        "all_numbers_grounded_int": int(p["grounding"]["all_grounded"]),
        "self_det": p["self_det"],
        "self_det_int": int(p["self_det"]),
        "self_tests_passed": p["self_tests_passed"],
        "self_tests_total": p["self_tests_total"],
        "peak_gpu_gib": p["peak_gpu_gib"],
        "analysis_only": True,
        "official_tps": 0,
        "primary_metric": p["primary_metric_value"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="lawine/quality-safe-ceiling-reconcile")
    ap.add_argument("--wandb-group", default="quality-safe-ceiling-reconcile")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    p = build_packet()
    HERE.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(p, fh, indent=2)
    print(f"[reconcile] wrote {OUT_JSON}", flush=True)

    rc = p["reconciled_ceilings"]
    line = "=" * 8 + " PR #570 — QUALITY-SAFE CEILING RECONCILE (synthesis) " + "=" * 8
    print("\n" + line, flush=True)
    print(f"  grounding: all_numbers_grounded = {p['grounding']['all_grounded']}", flush=True)
    print("  reconciled 3-ceiling table  [local | official_proxy | #319 | gap_to_ship | gap_to_500]:", flush=True)
    for key in ("candidate_verify", "plain_int4_head", "magically_free_floor"):
        r = rc[key]
        print(f"    {r['ceiling']:>22s} : {r['served_tps_local']:7.2f} | {r['official_proxy']:7.2f} "
              f"| {r['#319_status']:>4s} | {r['gap_to_ship']:6.2f} | {r['gap_to_500_gate']:6.2f}  "
              f"({r['source_pr']} {r['source_wandb']})", flush=True)
    cor = p["correction"]
    print(f"  CORRECTION : plain-int4-head 292.1 -> {p['quality_safe_ppl_only_ceiling_tps']:.2f} "
          f"(+{cor['correction_delta_tps']:.2f} = +{cor['correction_delta_sigma_hw']:.2f} sigma_hw, UPWARD)", flush=True)
    print(f"  loosest ceiling = {p['loosest_ceiling']} -> gap_to_ship_min = {p['gap_to_ship_min']:.2f} "
          f"(ship {SHIP_OFFICIAL_TPS}) ; gap_to_500_min = {p['gap_to_500_min']:.2f}", flush=True)
    print(f"  two_gate_satisfiable = {p['two_gate_satisfiable']}  ->  {p['fire_decision']}  "
          f"(UNCHANGED after the #553 correction)", flush=True)
    print(f"  self_det = {p['self_det']}  ({p['self_tests_passed']}/{p['self_tests_total']} self-tests)", flush=True)
    if not p["self_det"]:
        failed = [k for k, v in p["self_tests"].items() if not v]
        print(f"  !! FAILED self-tests: {failed}", flush=True)
    print("=" * len(line), flush=True)

    rid = None
    if not args.no_wandb:
        try:
            from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                               log_json_artifact, log_summary)
            run = init_wandb_run(
                job_type="systems-profile",
                agent="lawine",
                name=args.wandb_name,
                group=args.wandb_group,
                tags=["quality-safe", "ceiling-reconcile", "synthesis", "analysis-only",
                      "no-fire", "two-gate", "local-a10g", "pr570"],
                notes="PR #570 reconcile the 3 quality-safe head-lever ceilings to wirbel #553's corrected 299.29 "
                      "(candidate-verify 291.36 #319-PASS < plain-int4-head 299.29 #319-FAIL < magically-free 311.25); "
                      "two_gate_satisfiable=FALSE UNCHANGED",
                config={
                    "synthesis_only": True,
                    "no_gpu": True,
                    "cited_prs": ["fern #560", "wirbel #553", "lawine #554", "lawine #561"],
                    "cited_runs": ["ufv4nk21", "bo43du3w", "fi8vr1nb", "v74ad5jb"],
                    "ship_official_tps": SHIP_OFFICIAL_TPS,
                    "gate_tps": GATE_TPS,
                    "baseline_official_1_tps": BASELINE_OFFICIAL_1_TPS,
                    "tau_lo": TAU_LO,
                },
            )
            if run is not None:
                log_summary(run, wandb_summary(p), step=0)
                log_json_artifact(run, name="quality-safe-ceiling-reconcile",
                                  artifact_type="ceiling-reconcile-synthesis", data=p)
                rid = getattr(run, "id", None)
                finish_wandb(run)
                p["wandb_run_id"] = rid
                with OUT_JSON.open("w") as fh:
                    json.dump(p, fh, indent=2)
                print(f"[reconcile] wandb run id = {rid}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[reconcile] wandb unavailable: {exc}", flush=True)

    # ---- single-line SENPAI-RESULT ----
    senpai = {
        "terminal": True,
        "status": "complete",
        "pending_arms": False,
        "analysis_only": True,
        "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": p["self_det"],
        "two_gate_satisfiable": p["two_gate_satisfiable"],
        "fire_decision": p["fire_decision"],
        "quality_safe_319_safe_ceiling_tps": round(p["quality_safe_319_safe_ceiling_tps"], 2),
        "quality_safe_ppl_only_ceiling_tps": round(p["quality_safe_ppl_only_ceiling_tps"], 2),
        "magically_free_floor_tps": round(p["magically_free_floor_tps"], 2),
        "gap_to_ship_min": round(p["gap_to_ship_min"], 2),
        "self_tests_passed": p["self_tests_passed"],
        "primary_metric": {"name": "quality_safe_ppl_only_ceiling_tps",
                           "value": round(p["quality_safe_ppl_only_ceiling_tps"], 2)},
        "test_metric": {"name": "gap_to_ship_min", "value": round(p["gap_to_ship_min"], 2)},
    }
    print("\nSENPAI-RESULT: " + json.dumps(senpai, separators=(",", ":")), flush=True)
    return 0 if p["self_det"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
