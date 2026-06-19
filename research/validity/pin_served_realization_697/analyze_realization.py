#!/usr/bin/env python3
"""PR #697 — served-stack speed-realization analysis.

Reads the two paired_tps_ab.py outputs (core pin A/B + AR anchor), computes the
served realized pin TPS, maps local->official-equiv via the #691 x0.870 basis,
and quantifies realization_gap_vs_model_frac against the modelled bi1 / fixed2d
tiers. Logs ONE canonical wandb run (agent=land) with all PR-required fields.

analysis_only — official_tps=0, no_hf_job=1, fires=0. No submission, served file
untouched.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # target/
sys.path.insert(0, str(ROOT))

# ---- basis / anchors (PR #697 body) ----------------------------------------
X = 0.870                     # local -> official-equiv (int4 regime, #691)
BAR = 136.378                 # +10 bar (= locked 126.378 + 10)
LOCKED_AR_OFFICIAL = 126.378  # int4_g128_lmhead official AR anchor
MODELLED = {"bi1": 162.32, "fixed2d": 187.35}
MODELLED_MARGIN = {"bi1": 25.94, "fixed2d": 50.97}
CORE = "research/walltps_ab/pin_served_realization_697/core/paired_ab.json"
ANCHOR = "research/walltps_ab/pin_served_realization_697/anchor/paired_ab.json"


def _arm(d: dict, side: str) -> dict:
    return (d.get("arms", {}) or {}).get(side, {}) or {}


def _med(arm: dict, key: str = "wall_tps"):
    s = arm.get(key) or {}
    return s.get("median")


def _cv(arm: dict, key: str = "wall_tps"):
    s = arm.get(key) or {}
    return s.get("cv_pct")


def _vals(arm: dict, key: str = "wall_tps"):
    s = arm.get(key) or {}
    return s.get("values")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default=CORE)
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--wandb", action="store_true", help="log canonical wandb run")
    ap.add_argument("--wandb-name", default="land/pin-served-realization-697")
    ap.add_argument("--wandb-group", default="pin-served-tps-realization-land")
    args = ap.parse_args()

    core = json.loads((ROOT / args.core).read_text())
    have_anchor = (ROOT / args.anchor).exists()
    anchor = json.loads((ROOT / args.anchor).read_text()) if have_anchor else None

    # --- arms ---------------------------------------------------------------
    unp = _arm(core, "baseline")     # unpinned spec (BI=0, K=5)
    pin = _arm(core, "candidate")    # bi1 pinned spec (BI=1, K=5)
    U = _med(unp); P = _med(pin)
    U_ea = _med(unp, "e_accept_exact"); P_ea = _med(pin, "e_accept_exact")

    A = P2 = A_ea = P2_ea = None
    if anchor:
        ar = _arm(anchor, "baseline")    # AR M=1 (K=0)
        px = _arm(anchor, "candidate")   # bi1 cross-check (BI=1, K=5)
        A = _med(ar); P2 = _med(px)
        A_ea = _med(ar, "e_accept_exact"); P2_ea = _med(px, "e_accept_exact")

    # --- official-equiv mapping --------------------------------------------
    U_off = U * X if U else None
    P_off = P * X if P else None
    A_off = A * X if A else None
    P2_off = P2 * X if P2 else None

    # --- self-calibration of the x0.870 basis (AR anchor) -------------------
    calib = {}
    if A:
        calib = {
            "ar_local": A,
            "ar_official_equiv_x0870": A_off,
            "locked_ar_official": LOCKED_AR_OFFICIAL,
            "implied_multiplier": LOCKED_AR_OFFICIAL / A,
            "abs_err_vs_locked": A_off - LOCKED_AR_OFFICIAL,
            "basis_holds_within_1tps": abs(A_off - LOCKED_AR_OFFICIAL) <= 1.0,
        }

    # --- realization vs modelled bi1 tier (primary, apples-to-apples) -------
    tier = "bi1"
    modelled_level = MODELLED[tier]
    modelled_margin = MODELLED_MARGIN[tier]
    measured_margin = (P_off - BAR) if P_off is not None else None
    gap_frac = ((modelled_margin - measured_margin) / modelled_margin
                if measured_margin is not None else None)
    level_real_frac = (P_off / modelled_level) if P_off is not None else None
    level_shortfall = (modelled_level - P_off) if P_off is not None else None

    # --- decomposition ------------------------------------------------------
    pin_delta_local = (P - U) if (P is not None and U is not None) else None
    pin_delta_off = (pin_delta_local * X) if pin_delta_local is not None else None
    spec_speedup_off = ((U - A) * X) if (U is not None and A is not None) else None

    # --- verdict ------------------------------------------------------------
    if gap_frac is None:
        verdict = "INCOMPLETE"
    elif gap_frac <= -0.10:
        verdict = "REALIZATION_EXCEEDS"
    elif gap_frac < 0.10:
        verdict = "REALIZATION_HOLDS"
    else:
        verdict = "REALIZATION_GAP"

    # --- report -------------------------------------------------------------
    def f(x, n=3):
        return "None" if x is None else f"{x:.{n}f}"

    print("=" * 72)
    print("PR #697 — SERVED-STACK SPEED REALIZATION (LOCAL A10G, analysis_only)")
    print("=" * 72)
    print(f"x0.870 basis | +10 bar={BAR} | locked AR official={LOCKED_AR_OFFICIAL}")
    print("-" * 72)
    print("MEASURED LOCAL wall_tps (median):")
    print(f"  AR M=1 (K=0)         A  = {f(A)}  (cv {f(_cv(_arm(anchor,'baseline')) if anchor else None)}%)  e_accept={f(A_ea)}")
    print(f"  unpinned spec K=5    U  = {f(U)}  (cv {f(_cv(unp))}%)  e_accept={f(U_ea)}")
    print(f"  bi1 pinned spec K=5  P  = {f(P)}  (cv {f(_cv(pin))}%)  e_accept={f(P_ea)}   <-- served_local_tps_pinned")
    print(f"  bi1 xcheck (anchor)  P2 = {f(P2)}  e_accept={f(P2_ea)}")
    print("-" * 72)
    print("OFFICIAL-EQUIV (x0.870):")
    print(f"  AR    {f(A_off)}   unpinned {f(U_off)}   bi1-pinned {f(P_off)}")
    print("-" * 72)
    if calib:
        print("SELF-CALIBRATION (AR anchor vs locked 126.378):")
        print(f"  A x0.870 = {f(A_off)}  vs locked {LOCKED_AR_OFFICIAL}  "
              f"(err {f(calib['abs_err_vs_locked'])}, implied_mult {f(calib['implied_multiplier'],4)}, "
              f"holds={calib['basis_holds_within_1tps']})")
        print("-" * 72)
    print(f"REALIZATION vs modelled {tier} tier {modelled_level} (margin +{modelled_margin}):")
    print(f"  served bi1 official-equiv      = {f(P_off)}")
    print(f"  measured margin over bar       = {f(measured_margin)}")
    print(f"  modelled margin                = {modelled_margin}")
    print(f"  realization_gap_vs_model_frac  = {f(gap_frac,4)}   <-- TEST METRIC")
    print(f"  level realization frac         = {f(level_real_frac,4)}  (shortfall {f(level_shortfall)})")
    print("-" * 72)
    print("DECOMPOSITION:")
    print(f"  spec speedup served (U-A)x0.870     = {f(spec_speedup_off)} official-equiv")
    print(f"  PIN delta served    (P-U)x0.870     = {f(pin_delta_off)} official-equiv")
    print(f"  bi1 e_accept (realized)             = {f(P_ea)}   unpinned e_accept = {f(U_ea)}")
    print(f"  fixed2d tier {MODELLED['fixed2d']} (+{MODELLED_MARGIN['fixed2d']}): NOT cleanly served")
    print(f"     (MIN_LAUNCH_GRID_SIZE_2D=0 threshold snaps to cudagraph capture size;")
    print(f"      only is_batch_invariant unconditionally forces 2D -> bi1 is the served pin)")
    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print("=" * 72)

    result = {
        "pr": 697, "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "verdict": verdict,
        "basis_x": X, "bar": BAR, "locked_ar_official": LOCKED_AR_OFFICIAL,
        "local": {"ar_m1": A, "unpinned_spec_k5": U, "bi1_pinned_k5": P, "bi1_xcheck": P2},
        "e_accept": {"ar_m1": A_ea, "unpinned_spec_k5": U_ea, "bi1_pinned_k5": P_ea,
                     "bi1_xcheck": P2_ea},
        "wall_tps_cv_pct": {"unpinned_spec_k5": _cv(unp), "bi1_pinned_k5": _cv(pin)},
        "official_equiv": {"ar_m1": A_off, "unpinned_spec_k5": U_off,
                           "bi1_pinned_k5": P_off, "bi1_xcheck": P2_off},
        "served_local_tps_pinned": P,
        "served_official_equiv_pinned": P_off,
        "self_calibration": calib,
        "modelled_tier_bi1": modelled_level, "modelled_margin_bi1": modelled_margin,
        "measured_margin_over_bar": measured_margin,
        "realization_gap_vs_model_frac": gap_frac,
        "level_realization_frac": level_real_frac,
        "level_shortfall": level_shortfall,
        "decomposition": {
            "spec_speedup_served_off": spec_speedup_off,
            "pin_delta_served_off": pin_delta_off,
            "pin_delta_served_local": pin_delta_local,
        },
        "fixed2d_modelled_tier": MODELLED["fixed2d"],
        "fixed2d_modelled_margin": MODELLED_MARGIN["fixed2d"],
        "fixed2d_served": "NOT_CLEANLY_REACHABLE_UNDER_CUDAGRAPH",
        "workload": {"num_prompts": 128, "output_len": 512, "seed": 1},
        "head": "full_vocab_proxy (pin head-independent per PR #697; 16k-head capstone "
                "NOT done — stark #690 cross-read blocked by launch isolation, needs human auth)",
    }
    out = ROOT / "research/validity/pin_served_realization_697/realization_result.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[697] wrote {out}")

    if args.wandb:
        from scripts import wandb_logging
        run = wandb_logging.init_wandb_run(
            job_type="walltps-realization", agent="land",
            name=args.wandb_name, group=args.wandb_group,
            tags=["pr697", "pin-served-realization", "analysis_only", verdict],
            config={k: result[k] for k in ("pr", "analysis_only", "official_tps",
                    "no_hf_job", "fires", "basis_x", "bar", "locked_ar_official",
                    "modelled_tier_bi1", "modelled_margin_bi1", "workload", "head")},
        )
        if run is None:
            print("[697] wandb disabled (no key); skipped")
            return
        flat = {
            "served_local_tps_pinned": P,
            "served_official_equiv_pinned": P_off,
            "realization_gap_vs_model_frac": gap_frac,
            "level_realization_frac": level_real_frac,
            "level_shortfall": level_shortfall,
            "measured_margin_over_bar": measured_margin,
            "ar_local": A, "ar_official_equiv": A_off,
            "unpinned_local": U, "unpinned_official_equiv": U_off,
            "bi1_local": P, "bi1_official_equiv": P_off, "bi1_xcheck_local": P2,
            "spec_speedup_served_off": spec_speedup_off,
            "pin_delta_served_off": pin_delta_off,
            "pin_delta_served_local": pin_delta_local,
            "e_accept_bi1": P_ea, "e_accept_unpinned": U_ea, "e_accept_ar": A_ea,
            "implied_multiplier": calib.get("implied_multiplier"),
            "basis_holds": 1.0 if calib.get("basis_holds_within_1tps") else 0.0,
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "verdict_holds": 1.0 if verdict == "REALIZATION_HOLDS" else 0.0,
            "verdict_gap": 1.0 if verdict == "REALIZATION_GAP" else 0.0,
        }
        flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        flat["verdict"] = verdict
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(run, name="realization_697",
                                        artifact_type="walltps-realization", data=result)
        wandb_logging.finish_wandb(run)
        print(f"[697] wandb logged: {args.wandb_name} (group {args.wandb_group})")


if __name__ == "__main__":
    main()
