"""Spec-dec +10 robustness: error-bar the 136.711 official-equiv ceiling (PR #704).

DECISION-FORCING QUESTION
-------------------------
denken #698 (`nov6tc53`, config axis) + #701 (`j7q2oupr`, kernel axis) terminate the
drafter-side strict-#319 speed lever at a HARD ceiling of
``realized_fixed2d_official_equiv = 136.711`` official-equiv TPS. It clears the +10
fire bar (136.378) by only ``margin_over_plus10 = 0.333`` TPS, i.e. **0.24%**.

That 136.711 is NOT a measured official TPS. It is::

    136.711  =  specdec_local_ceiling (157.139 local TPS)  x  bi_tax (0.870)

a MODELLED number. The 0.870 local->official factor is the #683/#677 "bi_tax".
This card asks the single open question that decides whether the human's #481
"+10-speed-only" branch even *has* a viable frontier:

    Is the 0.333-TPS margin ROBUST to the bi_tax's uncertainty, or is the +10
    verdict INSIDE the conversion noise?

This is ANALYSIS-ONLY. NO build, NO serve, NO HF-Job. ``official_tps = 0``. It
re-analyses existing denken local numbers (#698/#701 op-bench + the #683/#677
bi_tax basis). It CANNOT be a fire (modelled, ``official_tps = 0``); even a
robust-clear verdict only tells the human whether *requesting* the spec-dec
HF-Job is worth it (a real HF-Job stays the only fire trigger).

WHAT THE bi_tax ACTUALLY IS (recovered provenance)
--------------------------------------------------
``LOCAL_TO_OFFICIAL = 0.870`` originates in denken #677
(`specdec_amortization_ceiling.py:661`, run `hj2afh4j`) as a **PR-specified**
strict-basis projection (the program's conservative "stark tax"), NOT an
empirically-fit multi-anchor regression. For the SPEC stack there is exactly ONE
basis point (basis_n = 1): the 0.870 projection itself. There is NO measured
official spec TPS behind it.

#677 records two *independent* cross-check anchors, and BOTH point the OPPOSITE
way (official FASTER than local, factor > 1.0):

  * AR hard anchor  : 126.378 official (locked `int4_g128_lmhead`, PR #4, W&B
    `905tbujn`) / 122.87 local  ->  factor = 1.02858.  HARD (real measured
    official) but a NON-SPEC AR config.
  * deployed anchor : 481.53 official / 454.19 local  ->  factor = 1.06019
    (`research/walltps_ab/local_official_projection/projection_cal.json`). A
    non-strict high-TPS MTP config; the LEAST transferable to a strict spec
    ceiling, reported only as an upside sensitivity.

So 0.870 is a deliberately CONSERVATIVE floor: #677 states "x0.870 understates
official-equiv". The three lineage factors {0.870, 1.02858, 1.06019} span a
config-dependence of std 0.083 (8.4% of their mean) -- ~39x larger than the
0.24% margin the +10 clearance rides on.

THE CRUX (phase / geometry dependence)
--------------------------------------
The 136.711 uses a SINGLE BLANKET x0.870 (see
`drafter_reduction_ceiling.py:154` official_tps()); it is NOT a phase-appropriate
factor. But the AR anchor (1.029) is calibrated on a non-spec AR forward, whereas
the spec stack is structurally different: K drafter M=1 forwards + a verify M=6
forward + an acceptance loop -- a far more host-/dispatch-overhead-heavy profile
(the most launch-bound config in the program). #683 measured the spec BI-tax
inflation at 1.376 vs the AR's 1.227 (ratio 1.121): the strict tax already hits
spec ~12% harder than AR. The host-heavy drafter-M=1 + accept-loop phases plausibly
convert WORSE than the blanket -> the spec-specific factor could sit BELOW 0.870,
below every measured anchor. We have zero official spec measurement to rule that
out. That phase-mix error (order ~10%) dwarfs the 0.24% margin in BOTH directions.

VERDICT (by definition on the propagated band)
----------------------------------------------
  PLUS10_CLEARED_ROBUST : specdec_official_equiv_lo > 136.378
  PLUS10_INSIDE_NOISE   : 136.378 in [lo, hi]          (HF-Job REQUIRED)
  PLUS10_NOT_CLEARED    : specdec_official_equiv_hi < 136.378

LOCAL re-analysis only -- no GPU required, no server, no submission, no HF job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "specdec_official_equiv_margin.json"

# --------------------------------------------------------------------------- #
# Recovered anchors (all denken-owned; provenance in the module docstring)     #
# --------------------------------------------------------------------------- #
REF_OFFICIAL_TPS = 126.378          # locked int4_g128_lmhead (PR #4, W&B 905tbujn)
PLUS10_BAR = 136.378                # +10 fire speed bar
BI_TAX_POINT = 0.870                # #677/#683 LOCAL_TO_OFFICIAL (PR-specified, conservative)

# #698 (nov6tc53) drafter-reduction-ceiling op-bench, STOCK_E drafter operating point.
STOCK_E = 3.33                      # in-scope stock-drafter e_accept (#677 INSCOPE)
STEP_MS_FIXED2D = 21.191464166558372       # realized_fixed2d_step_ms (#698 JSON)
REALIZED_OFFICIAL_EQUIV = 136.71070470778648  # realized_fixed2d_official_equiv (#698 JSON, carry-forward)
MARGIN_OVER_PLUS10_CARRY = 0.333    # carry-forward point margin (#698)

# Independent cross-check factors recovered from #677 (NON-spec; point the OTHER way).
F_AR_ANCHOR = 1.028583450966137     # 126.378 / 122.87  (AR hard official anchor)
F_DEPLOYED = 1.0601865051833779     # 481.53 / 454.19   (non-strict, least transferable)

# #683 measured BI-tax inflation (strict tax hits spec harder than AR) -> phase risk.
SPEC_BI_INFLATION = 1.3763249407389855   # strict319_enforcement_tax.json measured_relatives
AR_BI_INFLATION = 1.2267504619788836

# projection_cal.json: even the BEST-anchored multiplier in the program carries a
# +/-~2% envelope (the single official anchor's UNMEASURED run-to-run CV). A lower
# bound on ANY official-factor uncertainty; the 0.24% margin is far inside it.
DEPLOYED_ENVELOPE_CI95 = (1.0394063229912567, 1.080966687375499)  # around 1.06019


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #
def official_from_local(local_tps: float, factor: float) -> float:
    return local_tps * factor


def analyze() -> dict:
    # 1. The local ceiling behind 136.711 (invert the blanket x0.870).
    local_ceiling = 1000.0 * STOCK_E / STEP_MS_FIXED2D            # 157.139 local TPS
    # self-consistency: local_ceiling * 0.870 must reproduce 136.711
    recon = official_from_local(local_ceiling, BI_TAX_POINT)

    # 2. break-even bi_tax: the factor that maps the local ceiling to EXACTLY 136.378.
    breakeven_bi_tax = PLUS10_BAR / local_ceiling                 # 0.86788
    margin_factor = BI_TAX_POINT - breakeven_bi_tax              # +0.00212 (0.24%)
    margin_factor_pct = 100.0 * margin_factor / BI_TAX_POINT

    # 3. bi_tax spread: single spec basis point; sensitivity from the cross-config
    #    dispersion of the three denken lineage factors.
    factors = [BI_TAX_POINT, F_AR_ANCHOR, F_DEPLOYED]
    n = len(factors)
    mean = sum(factors) / n
    var_pop = sum((f - mean) ** 2 for f in factors) / n
    std_pop = var_pop ** 0.5
    std_sample = (sum((f - mean) ** 2 for f in factors) / (n - 1)) ** 0.5
    half_range = 0.5 * (max(factors) - min(factors))
    bi_tax_spread = std_pop                                       # headline spread

    # 4. propagate: symmetric band centered on the USED point (0.870) +/- dispersion.
    #    Centering on the used point (not the cross-config mean) is the honest choice:
    #    0.870 is the ONLY spec-specific factor; the >1.0 anchors are non-spec and
    #    cannot be assumed to transfer to the host-heavy spec stack. We must allow the
    #    spec factor to sit BELOW 0.870 (drafter-M=1 + accept-loop convert worse).
    factor_lo = BI_TAX_POINT - bi_tax_spread
    factor_hi = BI_TAX_POINT + bi_tax_spread
    official_lo = official_from_local(local_ceiling, factor_lo)
    official_hi = official_from_local(local_ceiling, factor_hi)

    # Upside sensitivities: the empirical anchors (all clear comfortably).
    official_at_ar = official_from_local(local_ceiling, F_AR_ANCHOR)
    official_at_deployed = official_from_local(local_ceiling, F_DEPLOYED)

    # Reference alt-construction (REJECTED for the verdict): if we centered on the
    # cross-config mean we'd assert the spec factor >= the non-spec anchors -> a
    # ROBUST clear. We reject it: no spec official measurement justifies ruling out a
    # sub-0.870 spec factor. Logged for transparency.
    factor_lo_meancentered = mean - bi_tax_spread
    official_lo_meancentered = official_from_local(local_ceiling, factor_lo_meancentered)

    # 5. verdict (by definition on the propagated band).
    plus10_cleared_robustly = 1 if official_lo > PLUS10_BAR else 0
    if official_hi < PLUS10_BAR:
        verdict = "PLUS10_NOT_CLEARED"
    elif official_lo > PLUS10_BAR:
        verdict = "PLUS10_CLEARED_ROBUST"
    else:
        verdict = "PLUS10_INSIDE_NOISE"

    # robustness headline: how many multiples of the margin is the dispersion?
    dispersion_to_margin_ratio = bi_tax_spread / margin_factor
    # even the best-anchored official multiplier's envelope half-width (% of its center)
    deployed_env_halfwidth_pct = 100.0 * 0.5 * (
        DEPLOYED_ENVELOPE_CI95[1] - DEPLOYED_ENVELOPE_CI95[0]) / F_DEPLOYED
    # phase risk: spec strict-tax is this much harsher than AR (multiplicative)
    spec_vs_ar_tax_ratio = SPEC_BI_INFLATION / AR_BI_INFLATION

    return {
        "anchors": {
            "ref_official_tps": REF_OFFICIAL_TPS,
            "plus10_bar": PLUS10_BAR,
            "bi_tax_point": BI_TAX_POINT,
            "stock_e": STOCK_E,
            "step_ms_fixed2d": STEP_MS_FIXED2D,
            "realized_official_equiv_carry": REALIZED_OFFICIAL_EQUIV,
        },
        "local_ceiling": {
            "specdec_local_ceiling": local_ceiling,
            "reconstructed_official_equiv": recon,
            "reconstruction_resid": recon - REALIZED_OFFICIAL_EQUIV,
        },
        "bi_tax_basis": {
            "bi_tax_point": BI_TAX_POINT,
            "bi_tax_basis_n": 1,
            "bi_tax_basis_note": "single PR-specified spec-basis projection; no measured official spec TPS",
            "cross_check_factors": {
                "spec_basis_conservative": BI_TAX_POINT,
                "ar_hard_anchor": F_AR_ANCHOR,
                "deployed_nonstrict": F_DEPLOYED,
            },
            "factor_mean": mean,
            "bi_tax_spread_std_pop": std_pop,
            "bi_tax_spread_std_sample": std_sample,
            "bi_tax_spread_half_range": half_range,
            "bi_tax_spread": bi_tax_spread,
            "all_anchors_above_point": min(F_AR_ANCHOR, F_DEPLOYED) > BI_TAX_POINT,
        },
        "phase_dependence": {
            "uses_blanket_factor": True,
            "phase_appropriate_factor_used": False,
            "spec_bi_inflation": SPEC_BI_INFLATION,
            "ar_bi_inflation": AR_BI_INFLATION,
            "spec_vs_ar_tax_ratio": spec_vs_ar_tax_ratio,
            "note": "spec strict-tax ~12% harsher than AR; host-heavy drafter-M=1 + "
                    "accept-loop phases plausibly convert below the blanket 0.870",
        },
        "breakeven": {
            "breakeven_bi_tax": breakeven_bi_tax,
            "margin_factor": margin_factor,
            "margin_factor_pct": margin_factor_pct,
            "margin_over_plus10_carry": MARGIN_OVER_PLUS10_CARRY,
        },
        "propagation": {
            "factor_lo": factor_lo,
            "factor_hi": factor_hi,
            "specdec_official_equiv_lo": official_lo,
            "specdec_official_equiv_hi": official_hi,
            "official_at_ar_anchor": official_at_ar,
            "official_at_deployed": official_at_deployed,
            "mean_centered_lo_REJECTED": official_lo_meancentered,
            "dispersion_to_margin_ratio": dispersion_to_margin_ratio,
            "deployed_envelope_halfwidth_pct": deployed_env_halfwidth_pct,
        },
        "verdict": {
            "verdict": verdict,
            "plus10_cleared_robustly": plus10_cleared_robustly,
            "breakeven_inside_band": factor_lo <= breakeven_bi_tax <= factor_hi,
        },
        "guards": {
            "analysis_only": 1,
            "official_tps": 0,
            "no_hf_job": 1,
            "fires": 0,
        },
    }


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #
def self_test() -> int:
    r = analyze()
    c = []

    lc = r["local_ceiling"]
    c.append(("local_ceiling reconstructs 136.711 (x0.870)",
              abs(lc["reconstruction_resid"]) < 1e-6))
    c.append(("local_ceiling ~ 157.139",
              abs(lc["specdec_local_ceiling"] - 157.1387) < 0.01))

    bk = r["breakeven"]
    c.append(("breakeven_bi_tax ~ 0.86788",
              abs(bk["breakeven_bi_tax"] - 0.867883) < 1e-4))
    c.append(("breakeven < point (margin > 0)", bk["margin_factor"] > 0))
    c.append(("margin_factor_pct ~ 0.24%", abs(bk["margin_factor_pct"] - 0.243) < 0.02))
    c.append(("carry margin reproduced",
              abs((REALIZED_OFFICIAL_EQUIV - PLUS10_BAR) - bk["margin_over_plus10_carry"]) < 0.001))

    bt = r["bi_tax_basis"]
    c.append(("bi_tax_basis_n == 1 (single spec point)", bt["bi_tax_basis_n"] == 1))
    c.append(("both empirical anchors > 0.870", bt["all_anchors_above_point"] is True))
    c.append(("bi_tax_spread ~ 0.083", abs(bt["bi_tax_spread"] - 0.0832) < 0.001))

    pr = r["propagation"]
    c.append(("dispersion >> margin (>=30x)", pr["dispersion_to_margin_ratio"] >= 30.0))
    c.append(("lo < plus10 < hi (straddle)",
              pr["specdec_official_equiv_lo"] < PLUS10_BAR < pr["specdec_official_equiv_hi"]))
    c.append(("AR-anchor upside clears comfortably (>155)",
              pr["official_at_ar_anchor"] > 155.0))
    c.append(("deployed upside clears comfortably (>160)",
              pr["official_at_deployed"] > 160.0))
    c.append(("break-even factor inside propagated band",
              r["verdict"]["breakeven_inside_band"] is True))
    c.append(("best-anchored official envelope (>1%) already exceeds 0.24% margin",
              pr["deployed_envelope_halfwidth_pct"] > bk["margin_factor_pct"]))

    vd = r["verdict"]
    c.append(("verdict == PLUS10_INSIDE_NOISE", vd["verdict"] == "PLUS10_INSIDE_NOISE"))
    c.append(("plus10_cleared_robustly == 0", vd["plus10_cleared_robustly"] == 0))

    g = r["guards"]
    c.append(("guards: analysis_only/official_tps/no_hf_job/fires",
              g["analysis_only"] == 1 and g["official_tps"] == 0
              and g["no_hf_job"] == 1 and g["fires"] == 0))

    ok = sum(1 for _, p in c if p)
    for name, p in c:
        print(f"  [{'PASS' if p else 'FAIL'}] {name}", flush=True)
    print(f"[self-test] {ok}/{len(c)} passed", flush=True)
    return 0 if ok == len(c) else 1


# --------------------------------------------------------------------------- #
# W&B                                                                          #
# --------------------------------------------------------------------------- #
def log_wandb(r: dict, wandb_name: str | None, wandb_group: str | None) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None

    bt, bk, pr, vd, g = (r["bi_tax_basis"], r["breakeven"], r["propagation"],
                         r["verdict"], r["guards"])
    scalars = {
        # ---- honesty guards (NOT a fire) ----
        "analysis_only": g["analysis_only"], "official_tps": g["official_tps"],
        "no_hf_job": g["no_hf_job"], "fires": g["fires"],
        # ---- PRIMARY + TEST metrics ----
        "specdec_official_equiv_lo": pr["specdec_official_equiv_lo"],   # PRIMARY
        "plus10_cleared_robustly": vd["plus10_cleared_robustly"],       # TEST
        # ---- required scalars ----
        "bi_tax_point": bt["bi_tax_point"],
        "bi_tax_basis_n": bt["bi_tax_basis_n"],
        "bi_tax_spread": bt["bi_tax_spread"],
        "breakeven_bi_tax": bk["breakeven_bi_tax"],
        "specdec_official_equiv_hi": pr["specdec_official_equiv_hi"],
        "specdec_local_ceiling": r["local_ceiling"]["specdec_local_ceiling"],
        "margin_over_plus10": bk["margin_over_plus10_carry"],
        "verdict": vd["verdict"],
        # ---- supporting ----
        "bi_tax_spread_std_sample": bt["bi_tax_spread_std_sample"],
        "bi_tax_spread_half_range": bt["bi_tax_spread_half_range"],
        "factor_ar_anchor": bt["cross_check_factors"]["ar_hard_anchor"],
        "factor_deployed": bt["cross_check_factors"]["deployed_nonstrict"],
        "factor_mean": bt["factor_mean"],
        "factor_lo": pr["factor_lo"], "factor_hi": pr["factor_hi"],
        "margin_factor": bk["margin_factor"], "margin_factor_pct": bk["margin_factor_pct"],
        "official_at_ar_anchor": pr["official_at_ar_anchor"],
        "official_at_deployed": pr["official_at_deployed"],
        "mean_centered_lo_rejected": pr["mean_centered_lo_REJECTED"],
        "dispersion_to_margin_ratio": pr["dispersion_to_margin_ratio"],
        "deployed_envelope_halfwidth_pct": pr["deployed_envelope_halfwidth_pct"],
        "spec_vs_ar_tax_ratio": r["phase_dependence"]["spec_vs_ar_tax_ratio"],
        "uses_blanket_factor": int(r["phase_dependence"]["uses_blanket_factor"]),
        "breakeven_inside_band": int(vd["breakeven_inside_band"]),
        "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR, "stock_e": STOCK_E,
        "realized_official_equiv_carry": REALIZED_OFFICIAL_EQUIV,
    }
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        name=wandb_name or "denken/specdec-official-equiv-margin",
        group=wandb_group or "specdec-official-equiv-margin-denken",
        config={"pr": 704, "card": "specdec_official_equiv_margin",
                "analysis_only": True, "no_hf_job": 1,
                "depends_on": ["#683", "#677", "#698", "#701"]},
    )
    wandb.log(scalars)
    wandb.summary.update(scalars)
    wandb.summary.update({"result": json.dumps(r, default=str)})
    rid = run.id
    run.finish()
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument("--wandb-group", type=str, default="specdec-official-equiv-margin-denken")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    r = analyze()
    OUT.write_text(json.dumps(r, indent=2, default=str))

    pr, bk, vd = r["propagation"], r["breakeven"], r["verdict"]
    print("\n=== SPEC-DEC +10 ROBUSTNESS (PR #704) ===", flush=True)
    print(f"  specdec_local_ceiling   = {r['local_ceiling']['specdec_local_ceiling']:.3f} local TPS", flush=True)
    print(f"  bi_tax_point            = {BI_TAX_POINT}  (basis_n=1, spread={r['bi_tax_basis']['bi_tax_spread']:.4f})", flush=True)
    print(f"  breakeven_bi_tax        = {bk['breakeven_bi_tax']:.5f}  (margin {bk['margin_factor']:.5f} = {bk['margin_factor_pct']:.3f}%)", flush=True)
    print(f"  official_equiv [lo, hi] = [{pr['specdec_official_equiv_lo']:.2f}, {pr['specdec_official_equiv_hi']:.2f}]  (bar {PLUS10_BAR})", flush=True)
    print(f"  anchors: AR -> {pr['official_at_ar_anchor']:.2f}, deployed -> {pr['official_at_deployed']:.2f}", flush=True)
    print(f"  dispersion/margin ratio = {pr['dispersion_to_margin_ratio']:.1f}x", flush=True)
    print(f"  VERDICT                 = {vd['verdict']}  (plus10_cleared_robustly={vd['plus10_cleared_robustly']})", flush=True)

    if not args.no_wandb:
        rid = log_wandb(r, args.wandb_name, args.wandb_group)
        if rid:
            r["wandb_run_id"] = rid
            OUT.write_text(json.dumps(r, indent=2, default=str))
            print(f"[wandb] run id = {rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
