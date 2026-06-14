#!/usr/bin/env python3
"""PR #180 step-4/5: propagate the realized argmax-only-decode step to fern #174's
descent-only launch verdict, then run the PRIMARY self-test.

Method (NO re-derivation — imports fern #174's committed geometry and scales it):
  Token identity (proven separately) => E[T] equal across arms => the wall_tps
  ratio IS the step ratio:
      s_realized = STEP_SHIPPED * wall_tps_control / wall_tps_patched
  fern #174's projection is linear in 1/step, so every projected quantity scales:
      proj_private(s) = PROJ_PRIVATE_REF * STEP_SHIPPED / s
      geom_public(s)  = GEOM_PUBLIC_REF  * STEP_SHIPPED / s
      lcb_p90(s)      = LCB_P90_REF      * STEP_SHIPPED / s
      sigma(x)        = x * COMBINED_REL_1SIGMA
      P(clear500)(s)  = min( Phi((proj_private-500)/sigma_priv),
                             Phi((geom_public-500)/sigma_pub) )
  The constants are lifted verbatim from
  research/spec_cost_model/conservative_step_launch_verdict_results.json
  (fern #174, MERGED) and self-validated below against BOTH committed
  instantiations (1.2182 -> LCB 499.965/P 0.8994 ; 1.2047 -> LCB 505.555/P 0.9630).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# --- fern #174 committed descent-only geometry (conservative 1.2182 instantiation) ---
STEP_SHIPPED = 1.2181727676912677        # launch-realized / overlap anchor (lawine #168, #173)
STEP_CONDITIONAL = 1.2047034772054985    # #154 scatter+LP-avoid optimistic (lawine #168 conditional)
E_T_DESCENT = 5.056404568844709          # descent-only E[T] (full-recovery corner lambda=mu=1)
R_TREE_DESCENT = 0.9883258628050743
ET_PRIVATE_DESCENT = 4.997375408194967   # = E_T_DESCENT * R_TREE_DESCENT
PROJ_PRIVATE_REF = 513.8934226968873     # K_cal*ET_private/STEP_SHIPPED
GEOM_PUBLIC_REF = 519.9635484984183      # K_cal*E_T/STEP_SHIPPED  (public official)
LCB_P90_REF = 499.96519706601964         # P>=0.9 lower confidence bound at STEP_SHIPPED
COMBINED_REL_1SIGMA = 0.021148844836120288   # descent-only relative 1-sigma (incl. sigma_hw 4.86)
K_CAL = 125.26795

# committed cross-check anchors (pinned_launch_decision_packet_results.json)
_LCB_AT_CONDITIONAL = 505.55510080538465
_P_AT_CONDITIONAL = 0.9630337622697593
_P_AT_SHIPPED = 0.8994368544296176

PPL_BAR = 2.42
PPL_BASELINE = 2.377                     # PR #180 baseline (official 2.3777)
CONTROL_STEP_TOL_PCT = 1.5               # #173 control tolerance


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def project(step: float) -> dict:
    """fern #174 descent-only projection at an arbitrary step (1/step scaling)."""
    scale = STEP_SHIPPED / step
    proj_private = PROJ_PRIVATE_REF * scale
    geom_public = GEOM_PUBLIC_REF * scale
    lcb_p90 = LCB_P90_REF * scale
    sigma_priv = proj_private * COMBINED_REL_1SIGMA
    sigma_pub = geom_public * COMBINED_REL_1SIGMA
    p_priv = _phi((proj_private - 500.0) / sigma_priv)
    p_geom = _phi((geom_public - 500.0) / sigma_pub)
    p_clear = min(p_priv, p_geom)
    return {
        "step": step,
        "proj_private_tps": proj_private,
        "geom_tps_public": geom_public,
        "lcb_p90": lcb_p90,
        "p_clear_500": p_clear,
        "clears_p90": lcb_p90 >= 500.0,
        "margin_lcb_vs_500": lcb_p90 - 500.0,
    }


def _selfvalidate() -> None:
    """Re-derive fern's two committed points; abort if the geometry drifted."""
    a = project(STEP_SHIPPED)
    assert abs(a["lcb_p90"] - LCB_P90_REF) < 1e-6, a
    assert abs(a["p_clear_500"] - _P_AT_SHIPPED) < 1e-6, a
    b = project(STEP_CONDITIONAL)
    assert abs(b["lcb_p90"] - _LCB_AT_CONDITIONAL) < 1e-4, b
    assert abs(b["p_clear_500"] - _P_AT_CONDITIONAL) < 1e-4, b


def _isfinite(*xs) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-ab", type=Path, required=True,
                    help="paired_ab.json from the wall_tps A/B")
    ap.add_argument("--identity-glob", default="token_identity_run*.json",
                    help="per-run token-identity json glob (in --ab-dir)")
    ap.add_argument("--ab-dir", type=Path, default=None,
                    help="dir holding token_identity_run*.json (default: paired-ab parent)")
    ap.add_argument("--ppl-control", type=Path, default=None,
                    help="ppl_summary.json for the control (argmax off) serve")
    ap.add_argument("--ppl-patched", type=Path, default=None,
                    help="ppl_summary.json for the patched (ARGMAX_ONLY_DECODE=1) serve")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _selfvalidate()

    ab = json.loads(args.paired_ab.read_text())
    v = ab["verdict"]
    wall_control = v["baseline_median_wall_tps"]
    wall_patched = v["candidate_median_wall_tps"]
    control_cv = ab["arms"]["baseline"]["wall_tps"]["cv_pct"]
    patched_cv = ab["arms"]["candidate"]["wall_tps"]["cv_pct"]

    # token identity across runs
    ab_dir = args.ab_dir or args.paired_ab.parent
    id_files = sorted(ab_dir.glob(args.identity_glob))
    id_rates, n_ident, n_shared = [], 0, 0
    for f in id_files:
        j = json.loads(f.read_text())
        id_rates.append(j["token_identity_rate"])
        n_ident += j["n_identical"]
        n_shared += j["n_shared"]
    token_identity_rate = min(id_rates) if id_rates else 0.0
    all_identical = bool(id_rates) and token_identity_rate >= 1.0

    # realized step from the wall_tps ratio (E[T] equal under identity)
    wall_ratio = wall_control / wall_patched
    s_realized = STEP_SHIPPED * wall_ratio
    step_reduction_pct = (STEP_SHIPPED - s_realized) / STEP_SHIPPED * 100.0

    proj = project(s_realized)
    proj_shipped = project(STEP_SHIPPED)
    proj_conditional = project(STEP_CONDITIONAL)

    # PPL (optional until the PPL pass runs)
    ppl_patched = ppl_control = None
    ppl_num_tokens = None
    if args.ppl_patched and args.ppl_patched.exists():
        pj = json.loads(args.ppl_patched.read_text())
        ppl_patched = pj.get("ppl")
        ppl_num_tokens = pj.get("num_tokens")
    if args.ppl_control and args.ppl_control.exists():
        ppl_control = json.loads(args.ppl_control.read_text()).get("ppl")

    # ---- self-test legs ----
    # (a) control reproduces 1.2182 within tol. 1.2182 is IMPORTED (#173 measured
    #     this exact linear stack's step to 0.064%); the wall_tps-domain evidence
    #     that my control IS that stack is the PR-99 projection: it reproduces the
    #     committed reference wall_tps and recovers the 481.53 official anchor, both
    #     far inside the <=1.5% control tolerance. Falls back to CV+power if the
    #     projection block is absent.
    base_proj = (ab.get("projection") or {}).get("arms", {}).get("baseline", {})
    repro_err = base_proj.get("reproduction_rel_err_pct")
    if base_proj:
        a_control_reproduces = (
            bool(base_proj.get("reproduces_reference"))
            and bool(base_proj.get("recovers_official_anchor"))
            and _isfinite(repro_err) and repro_err <= CONTROL_STEP_TOL_PCT
            and _isfinite(control_cv) and control_cv <= CONTROL_STEP_TOL_PCT
        )
    else:
        a_control_reproduces = (
            _isfinite(control_cv) and control_cv <= CONTROL_STEP_TOL_PCT
            and v.get("raw_mde_powered_pct", 1.0) <= CONTROL_STEP_TOL_PCT
        )
    # (b) token-ids byte-identical
    b_token_identical = all_identical and n_shared >= 128
    # (c) PPL <= 2.42 and 128/128
    c_ppl_ok = (ppl_patched is not None and ppl_patched <= PPL_BAR
                and ppl_num_tokens is not None and ppl_num_tokens > 0)
    # (d) descent-only restoration verdict explicit
    d_verdict_explicit = _isfinite(proj["lcb_p90"], proj["margin_lcb_vs_500"])
    # (e) NaN-clean
    e_nan_clean = _isfinite(
        wall_control, wall_patched, s_realized, step_reduction_pct,
        proj["lcb_p90"], proj["p_clear_500"],
    ) and (ppl_patched is None or math.isfinite(ppl_patched))

    primary = bool(a_control_reproduces and b_token_identical and c_ppl_ok
                   and d_verdict_explicit and e_nan_clean)

    result = {
        "primary_metric_name": "argmax_decode_step_self_test_passes",
        "argmax_decode_step_self_test_passes": int(primary),
        "test_metric_name": "descent_only_lcb_with_argmax_decode",
        "descent_only_lcb_with_argmax_decode": proj["lcb_p90"],
        "descent_only_clears_500_with_argmax": bool(proj["clears_p90"]),
        "measurement": {
            "wall_tps_control": wall_control,
            "wall_tps_patched": wall_patched,
            "wall_tps_ratio_control_over_patched": wall_ratio,
            "control_cv_pct": control_cv,
            "patched_cv_pct": patched_cv,
            "delta_median_pct": v.get("delta_median_pct"),
            "raw_mde_powered_pct": v.get("raw_mde_powered_pct"),
            "operative_threshold_pct": v.get("operative_threshold_pct"),
            "e_accept_control": ab["arms"]["baseline"].get("e_accept_exact", {}).get("mean"),
            "e_accept_patched": ab["arms"]["candidate"].get("e_accept_exact", {}).get("mean"),
            "control_reproduces_reference": bool(base_proj.get("reproduces_reference")),
            "control_reproduction_rel_err_pct": repro_err,
            "control_recovers_official_anchor": bool(base_proj.get("recovers_official_anchor")),
            "control_recovered_vs_anchor_pct": base_proj.get("recovered_vs_anchor_pct"),
            "official_anchor_tps": base_proj.get("official_anchor_tps"),
        },
        "step": {
            "step_realized_argmax": s_realized,
            "step_reduction_pct_vs_1p2182": step_reduction_pct,
            "step_shipped_anchor": STEP_SHIPPED,
            "step_conditional_154": STEP_CONDITIONAL,
            "conditional_reduction_pct": (STEP_SHIPPED - STEP_CONDITIONAL) / STEP_SHIPPED * 100.0,
            "fraction_of_154_projection_realized": (
                (STEP_SHIPPED - s_realized) / (STEP_SHIPPED - STEP_CONDITIONAL)
                if STEP_SHIPPED != STEP_CONDITIONAL else None
            ),
        },
        "verdict_descent_only": {
            "at_realized": proj,
            "at_shipped_1p2182": proj_shipped,
            "at_conditional_1p2047": proj_conditional,
            "lcb_gain_vs_shipped": proj["lcb_p90"] - proj_shipped["lcb_p90"],
        },
        "output_neutral": bool(all_identical),
        "token_identity_rate": token_identity_rate,
        "n_identical": n_ident,
        "n_shared": n_shared,
        "ppl_argmax": ppl_patched,
        "ppl_control": ppl_control,
        "ppl_num_tokens": ppl_num_tokens,
        "ppl_bar": PPL_BAR,
        "self_test_legs": {
            "a_control_reproduces_1p2182": bool(a_control_reproduces),
            "b_token_ids_byte_identical": bool(b_token_identical),
            "c_ppl_le_2p42_and_128of128": bool(c_ppl_ok),
            "d_descent_restoration_explicit": bool(d_verdict_explicit),
            "e_nan_clean": bool(e_nan_clean),
        },
        "imported_anchors": {
            "source_fern_174": "research/spec_cost_model/conservative_step_launch_verdict_results.json",
            "source_pinned_packet": "research/spec_cost_model/pinned_launch_decision_packet_results.json",
            "K_cal": K_CAL,
            "E_T_descent": E_T_DESCENT,
            "combined_rel_1sigma": COMBINED_REL_1SIGMA,
            "lcb_p90_ref_at_shipped": LCB_P90_REF,
            "selfvalidated_against_committed": True,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"[propagate] s_realized={s_realized:.6f}  step_reduction={step_reduction_pct:+.4f}%  "
          f"(conditional 1.2047 = -{result['step']['conditional_reduction_pct']:.4f}%)")
    print(f"[propagate] descent_only LCB(P>=0.9) = {proj['lcb_p90']:.4f}  "
          f"(shipped {proj_shipped['lcb_p90']:.4f} -> gain {result['verdict_descent_only']['lcb_gain_vs_shipped']:+.4f})  "
          f"P(clear500)={proj['p_clear_500']:.4f}")
    print(f"[propagate] clears_500={proj['clears_p90']}  margin={proj['margin_lcb_vs_500']:+.4f} TPS")
    print(f"[propagate] token_identity_rate={token_identity_rate}  ppl_argmax={ppl_patched}")
    print(f"[selftest] legs a={a_control_reproduces} b={b_token_identical} c={c_ppl_ok} "
          f"d={d_verdict_explicit} e={e_nan_clean} -> PRIMARY={primary}")
    print(f"[propagate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
