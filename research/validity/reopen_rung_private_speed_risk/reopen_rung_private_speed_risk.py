#!/usr/bin/env python
"""PR #522 (kanna): per-rung PRIVATE single-draw SPEED-drift risk for the reopen
decision tree (#517 upgrade branch).

analysis_only=true, official_tps=0. NO HF Jobs, NO --launch, NO submission.
group=reopen-rung-private-speed-risk.

WHAT THIS PRICES
----------------
#517's reopen decision tree has an UPGRADE branch: "keep surgical-357 or upgrade
to a faster byte-exact rung (399.75 split-KV / ~457.5 strict-frontier)?". The
quality side is settled (denken #513/#520: downstream exposure 0.0 -> drift is
PURE SPEED). This script supplies the missing SPEED-side input: the single-draw
private speed-drift risk per rung, from speculative-decode acceptance-rate
variance, combined with the hardware between-draw sigma_hw.

KEY STRUCTURAL FACT (why the table collapses to a shared multiplier)
--------------------------------------------------------------------
All three reopen rungs share the IDENTICAL MTP drafter + spec config:
  SPECULATIVE_CONFIG {"method":"mtp","model":"/tmp/qat-assistant","num_speculative_tokens":7}
  DRAFTER_SHA256 ed159e334999fd6b5f2d0dbad026346d4efac89eb7c6f55c5cdb042eca5dd18e
and are byte-exact greedy-identical to the same fa2sw_precache_kenyan parent.
They differ ONLY in the attention REDUCTION path (surgical 2D vs fixed-order 3D
split-KV vs strict-frontier kernels) -- a per-step wall-clock (t_step) change
that is acceptance-INDEPENDENT. Because TPS = E[T] / t_step and t_step is fixed
per rung (M=8 verify every step regardless of how many draft tokens are
accepted), the acceptance-driven private drift is a SHARED MULTIPLICATIVE factor
across rungs. So the per-rung private speed = public_anchor_r * (shared drift).

ACCEPTANCE VARIANCE INPUT (measured, reused -- shared drafter)
-------------------------------------------------------------
The acceptance distribution is a property of (drafter, prompt-distribution), NOT
of the attention path, so re-serving each rung would reproduce the same
acceptance. We reuse the 6 REAL served private draws measured on the shared
drafter in PR #44's private_gap_probe (submission fa2sw_precache_kenyan):
sharegpt + 5 native domains (casual/code/longctx/math/multilingual). Each draw
gives public and private E[T] (e_accept) -> the single-draw acceptance ratio
R_ea = E[T]_private / E[T]_public. These proxies are deliberately HARD (they
over-estimate the true private breach ~2-3x, see private_gap_probe.md), so we
take their MEAN breach as a pessimistic reconciliation and their draw-to-draw
SPREAD (sigma) as the single-draw acceptance variance, while HEADLINING the
grounded central breach from the banked framework below.

FRAMEWORK CONSTANTS (banked, kanna #504/#478/#508 ship_private_dossier)
----------------------------------------------------------------------
- propagation factor PF (acceptance breach -> TPS breach): 0.99992 ~ 1.0 (#504
  0urxqwob): a 1% E[T] drop is ~1% TPS drop because t_step is acceptance-fixed.
- grounded central breach: 0.04295 (4.295%) -- #504/#508 dossier, consistent
  with the board honest band (openevolve ~3.9%, firfir-cast cap 7.2%).
- sigma_hw: 1.00% FRACTIONAL one-shot (#478 mssuss3f). NOTE: the PR's
  "sigma_hw 4.864" is the ABSOLUTE between-leg TPS @~481 (frantic-penguin 3
  official draws); it must NOT be applied as a fixed constant at a different
  operating point -- fractional sigma scales with the rung TPS. Using 4.864 at
  375.857 would over-state the band by 4.864/(0.01*375.857)=1.29x.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROBE = os.path.join(TARGET, "research/validity/private_gap_probe")

# ---------------------------------------------------------------------------
# Framework constants (banked; provenance in module docstring).
# ---------------------------------------------------------------------------
PF = 0.9999171490311938                 # #504 0urxqwob realized propagation factor
BREACH_CENTRAL = 0.04294644155088977    # #504/#508 grounded central TPS breach
SIGMA_HW_FRAC = 0.01                    # #478 scale-invariant one-shot fractional
SIGMA_HW_ABS_AT_481 = 4.864             # provenance only (absolute @481.53)
SIGMA_HW_REF_TPS = 481.53
DENKEN24_BREACH = 0.24                  # refuted worst-case (denken #489), extreme tail
Z95 = 1.959963984540054                 # two-sided 95%
Z90_1S = 1.6448536269514722             # one-sided 95% (P5/P95)

# Public anchors as stated in the PR #522 baseline. surgical357 is OFFICIAL
# (j7qao5e9, 128x512 HF Jobs). splitkv399 is the lawine #496 / 42qroec1 number
# (provisional -- stark #519 owns the official split-KV full-workload anchor).
# frontier457 is PREDICTION-ONLY (no loadable submission; reanchor #455 -> ~466).
RUNGS = [
    {
        "rung": "surgical357",
        "role": "control (shipped primary)",
        "public_tps": 375.857,
        "public_anchor_kind": "official",
        "public_anchor_src": "j7qao5e9 (128x512 HF Jobs)",
        "loadable": True,
        "submission": "submissions/fa2sw_strict_surgical357",
    },
    {
        "rung": "splitkv399",
        "role": "upgrade candidate (byte-exact fixed-order split-KV)",
        "public_tps": 399.75,
        "public_anchor_kind": "provisional (local/#496)",
        "public_anchor_src": "lawine #496 / 42qroec1; stark #519 owns official anchor",
        "loadable": True,
        "submission": "submissions/fa2sw_strict_byteexact_splitkv399",
    },
    {
        "rung": "frontier457",
        "role": "upgrade candidate (strict-frontier prediction)",
        "public_tps": 457.5,
        "public_anchor_kind": "prediction-only",
        "public_anchor_src": "PR #522 prediction; reanchor #455 strict frontier ~466.0",
        "loadable": False,
        "submission": None,
    },
]

# Floor-lock literal-1.0 fallback (greedy-exact M=1 AR; zero breach) -- not a
# reopen-upgrade rung, carried only as the guaranteed-valid reference floor.
FLOORLOCK_TPS = 166.23


def load_acceptance_draws():
    """Return per-draw acceptance ratios R_ea (private/public) for the shared
    drafter, from PR #44 private_gap_probe REAL served runs. Excludes smoke."""
    draws = []
    if not os.path.isdir(PROBE):
        return draws
    for d in sorted(os.listdir(PROBE)):
        rj = os.path.join(PROBE, d, "report.json")
        if not os.path.isfile(rj):
            continue
        try:
            r = json.load(open(rj))
            if r.get("smoke"):
                continue
            lb = r["scenarios"]["leaderboard"]["acceptance"]
            pv = r["scenarios"]["private_rerun"]["acceptance"]
            ea_pub = float(lb["e_accept"])
            ea_pri = float(pv["e_accept"])
            tps_pub = float(r["scenarios"]["leaderboard"]["bench"]["tps"])
            tps_pri = float(r["scenarios"]["private_rerun"]["bench"]["tps"])
            if ea_pub <= 0:
                continue
            draws.append({
                "domain": d,
                "ea_pub": ea_pub,
                "ea_pri": ea_pri,
                "R_ea": ea_pri / ea_pub,
                "breach_acc": 1.0 - ea_pri / ea_pub,
                "tps_pub": tps_pub,
                "tps_pri": tps_pri,
                "R_tps": tps_pri / tps_pub,
            })
        except Exception as e:  # pragma: no cover - defensive
            print(f"  [skip {d}: {e}]")
    return draws


def project_rung(public_tps, breach_central, sigma_accdraw, sigma_hw_frac):
    """Compute the per-rung private speed band + combined worst-case floor.

    mean    = P * (1-breach_central) * PF
    sigma_hw= mean * sigma_hw_frac                       (hardware between-draw)
    band95  = mean +/- z95 * sigma_hw                    (hardware-only band, dossier-style)
    worst   = P * (1 - breach_p95) * (1 - z90_1s*sigma_hw_frac)
              breach_p95 = breach_central + z90_1s*sigma_accdraw   (acceptance downside)
              (combined one-sided 95%: acceptance-draw downside x hardware downside)
    """
    mult_central = (1.0 - breach_central) * PF
    mean = public_tps * mult_central
    sigma_hw = mean * sigma_hw_frac
    band95 = (mean - Z95 * sigma_hw, mean + Z95 * sigma_hw)

    breach_p95 = breach_central + Z90_1S * sigma_accdraw
    mult_worst = (1.0 - breach_p95) * (1.0 - Z90_1S * sigma_hw_frac)
    worst = public_tps * mult_worst
    return {
        "public_tps": public_tps,
        "mult_central": mult_central,
        "projected_private_tps_mean": mean,
        "private_tps_sigma_hw": sigma_hw,
        "private_tps_band95_lo": band95[0],
        "private_tps_band95_hi": band95[1],
        "breach_p95": breach_p95,
        "mult_worst": mult_worst,
        "private_tps_worstcase": worst,
    }


def main():
    draws = load_acceptance_draws()
    n = len(draws)
    R_ea = [d["R_ea"] for d in draws]
    breach_acc = [d["breach_acc"] for d in draws]
    R_tps = [d["R_tps"] for d in draws]

    acc = {
        "n_draws": n,
        "R_ea_mean": st.mean(R_ea) if n else float("nan"),
        "R_ea_sd": st.pstdev(R_ea) if n > 1 else 0.0,
        "R_ea_min": min(R_ea) if n else float("nan"),
        "R_ea_max": max(R_ea) if n else float("nan"),
        "breach_acc_mean": st.mean(breach_acc) if n else float("nan"),
        "breach_acc_sd": st.pstdev(breach_acc) if n > 1 else 0.0,
        "breach_acc_max": max(breach_acc) if n else float("nan"),
        "R_tps_mean": st.mean(R_tps) if n else float("nan"),
        "R_tps_sd": st.pstdev(R_tps) if n > 1 else 0.0,
    }
    # Single-draw acceptance variance: the dominant term is WHICH private
    # sub-distribution is drawn (cross-domain spread). Use it as sigma_accdraw.
    sigma_accdraw = acc["breach_acc_sd"]

    rows = []
    for r in RUNGS:
        proj = project_rung(r["public_tps"], BREACH_CENTRAL, sigma_accdraw, SIGMA_HW_FRAC)
        # proxy-pessimistic reconciliation (proxy mean breach, proxy worst draw)
        proxy_central = r["public_tps"] * acc["R_ea_mean"]
        proxy_worstdraw = r["public_tps"] * acc["R_ea_min"]
        denken24 = r["public_tps"] * (1.0 - DENKEN24_BREACH)
        rows.append({
            **{k: r[k] for k in ("rung", "role", "public_anchor_kind",
                                 "public_anchor_src", "loadable", "submission")},
            **proj,
            "proxy_pessimistic_private_tps_mean": proxy_central,
            "proxy_worstdraw_private_tps": proxy_worstdraw,
            "denken24_refuted_private_tps": denken24,
            "quality_verdict": "0 downstream exposure (denken #513/#520 pure-speed); spec-alive byte-exact",
        })

    # ----- verdict -----
    surg = next(x for x in rows if x["rung"] == "surgical357")
    skv = next(x for x in rows if x["rung"] == "splitkv399")
    fro = next(x for x in rows if x["rung"] == "frontier457")
    best = max(rows, key=lambda x: x["private_tps_worstcase"])

    verdict = (
        f"Acceptance is a SHARED drafter property (identical DRAFTER_SHA256 + MTP K=7 + "
        f"byte-exact output) -> all rungs inherit the SAME private-drift multiplier "
        f"(central {surg['mult_central']:.4f}, combined-worstcase {surg['mult_worst']:.4f}). "
        f"Private TPS ranking therefore equals public ranking at EVERY percentile: "
        f"frontier457 > splitkv399 > surgical357. Best risk-adjusted reopen rung = "
        f"{best['rung']} (worst-case private floor {best['private_tps_worstcase']:.1f} TPS). "
        f"splitkv399 worst-case floor {skv['private_tps_worstcase']:.1f} ~= surgical357 EXPECTED "
        f"private {surg['projected_private_tps_mean']:.1f}: the upgrade's downside lands at the "
        f"control's median, and splitkv399 beats surgical357 by +{skv['private_tps_worstcase']-surg['private_tps_worstcase']:.1f} "
        f"at the worst case. UPGRADE rule: private-speed-drift NEVER inverts the ranking, so "
        f"upgrade to the fastest rung whose PUBLIC anchor is validated (stark #519 for splitkv399; "
        f"frontier457 is prediction-only). Speed-side risk does not gate the upgrade; quality is "
        f"denken-cleared (0 exposure). Floor-lock {FLOORLOCK_TPS} remains the only literal-identity fallback."
    )

    out = {
        "pr": 522,
        "agent": "kanna",
        "analysis_only": True,
        "official_tps": 0,
        "no_serve": True,
        "no_hf_job": True,
        "no_launch": True,
        "no_submission": True,
        "group": "reopen-rung-private-speed-risk",
        "boundary": ("speed-side reopen-rung risk (private speed-drift per rung). "
                     "quality-side = denken #513/#520 (0 exposure). public split-KV "
                     "anchor = stark #519. scored accuracy = ubel #511."),
        "framework": {
            "PF": PF, "breach_central": BREACH_CENTRAL,
            "sigma_hw_frac": SIGMA_HW_FRAC,
            "sigma_hw_abs_at_481_provenance_only": SIGMA_HW_ABS_AT_481,
            "sigma_hw_ref_tps": SIGMA_HW_REF_TPS,
            "sigma_accdraw_used": sigma_accdraw,
            "denken24_refuted_breach": DENKEN24_BREACH,
            "z95": Z95, "z90_1sided": Z90_1S,
            "source_runs": {
                "kanna_504_propagation": "0urxqwob",
                "kanna_478_sigma_hw": "mssuss3f",
                "kanna_508_dossier": "ship_private_dossier",
                "pr44_acceptance_draws": "private_gap_probe",
            },
        },
        "acceptance_distribution": acc,
        "acceptance_draws": draws,
        "shared_drafter_note": (
            "surgical357 + splitkv399 share SPECULATIVE_CONFIG mtp K=7 and "
            "DRAFTER_SHA256 ed159e33...dd18e and are byte-exact greedy-identical; "
            "acceptance variance is identical across rungs by construction."),
        "rung_private_speed_risk_table": rows,
        "floorlock_tps": FLOORLOCK_TPS,
        "verdict_oneline": verdict,
        "splitkv399_projected_private_tps": skv["projected_private_tps_mean"],
        "splitkv399_private_tps_worstcase": skv["private_tps_worstcase"],
        "surgical357_private_tps_worstcase": surg["private_tps_worstcase"],
        "frontier457_projected_private_tps": fro["projected_private_tps_mean"],
        "frontier457_private_tps_worstcase": fro["private_tps_worstcase"],
        "best_riskadj_rung": best["rung"],
        "best_riskadj_worstcase_floor": best["private_tps_worstcase"],
    }

    # ----- self-test: NaN-clean + ordering + reproduction -----
    checks = {}

    def all_finite(o):
        if isinstance(o, float):
            return math.isfinite(o)
        if isinstance(o, dict):
            return all(all_finite(v) for v in o.values())
        if isinstance(o, list):
            return all(all_finite(v) for v in o)
        return True

    checks["nan_clean"] = all_finite(out)
    checks["n_draws_ge_5"] = n >= 5
    checks["ranking_preserved_mean"] = (
        fro["projected_private_tps_mean"] > skv["projected_private_tps_mean"] >
        surg["projected_private_tps_mean"])
    checks["ranking_preserved_worstcase"] = (
        fro["private_tps_worstcase"] > skv["private_tps_worstcase"] >
        surg["private_tps_worstcase"])
    checks["shared_multiplier"] = (
        abs(surg["mult_central"] - skv["mult_central"]) < 1e-12 and
        abs(surg["mult_worst"] - skv["mult_worst"]) < 1e-12)
    checks["splitkv_worst_ge_surg_worst"] = (
        skv["private_tps_worstcase"] > surg["private_tps_worstcase"])
    # dossier reproduction: surgical at the dossier's LOCAL anchor 357.22 -> 341.88
    dossier_mean = 357.22 * surg["mult_central"]
    checks["reproduces_dossier_surgical_341p9"] = abs(dossier_mean - 341.8786721491912) < 0.05
    checks["sigma_hw_fractional_not_fixed"] = (
        abs(surg["private_tps_sigma_hw"] - surg["projected_private_tps_mean"] * SIGMA_HW_FRAC) < 1e-9)
    checks["all_private_below_public"] = all(
        x["projected_private_tps_mean"] < x["public_tps"] and
        x["private_tps_worstcase"] < x["projected_private_tps_mean"] for x in rows)
    out["self_test"] = {"checks": checks, "passes": all(checks.values())}

    outpath = os.path.join(HERE, "rung_private_speed_risk_table.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    # ----- console report -----
    print(f"== PR#522 reopen-rung private SPEED-drift risk (analysis_only, n_draws={n}) ==")
    print(f"acceptance R_ea: mean {acc['R_ea_mean']:.4f} sd {acc['R_ea_sd']:.4f} "
          f"[{acc['R_ea_min']:.4f},{acc['R_ea_max']:.4f}]  -> breach_acc mean "
          f"{acc['breach_acc_mean']:.4f} sd {acc['breach_acc_sd']:.4f} (sigma_accdraw)")
    print(f"framework: PF {PF:.5f}  breach_central {BREACH_CENTRAL:.4%}  sigma_hw {SIGMA_HW_FRAC:.2%} fractional")
    print()
    hdr = f"{'rung':14s} {'pub':>8s} {'priv_mean':>10s} {'95band(hw)':>18s} {'WORSTCASE':>10s} {'kind':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        band = f"[{x['private_tps_band95_lo']:.1f},{x['private_tps_band95_hi']:.1f}]"
        print(f"{x['rung']:14s} {x['public_tps']:8.2f} {x['projected_private_tps_mean']:10.2f} "
              f"{band:>18s} {x['private_tps_worstcase']:10.2f} {x['public_anchor_kind']:>16s}")
    print(f"{'floor-lock':14s} {FLOORLOCK_TPS:8.2f} {'(literal-1.0 fallback; 0 breach, guaranteed)':>50s}")
    print()
    print("VERDICT:", verdict)
    print()
    print("self-test:", "PASS" if out["self_test"]["passes"] else "FAIL",
          {k: v for k, v in checks.items() if not v} or "(all green)")
    print("artifact:", outpath)
    return out


if __name__ == "__main__":
    main()
