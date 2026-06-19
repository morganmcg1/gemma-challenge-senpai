#!/usr/bin/env python3
"""PR #725 (RE-SCOPED) -- local->official-equiv calibration + private-haircut risk
model for the stark K=5 MTP-spec fire (`int4_mtp_batchinv`, stark #727).

THE DECISION-GRADE QUESTION (advisor's re-scoped card, 2026-06-19 11:06)
-----------------------------------------------------------------------
Convert stark's K=5 *local* wall_tps to an *official-equiv* estimate, overlay the
documented 4-9% private-verify haircut, and report P(the config clears the 126.378
official bar) + a (a) comfortable / (b) knife-edge / (c) likely-fails verdict.
`analysis_only` -- NO HF Job, NO fire. Guard flags live in wandb.summary.

HONEST PAIR ACCOUNTING (card step 1: "be honest about how many real pairs exist")
---------------------------------------------------------------------------------
A "pair" is (local TPS, official a10g-small TPS) for the same config. The trap is
the LOCAL METER: this programme has carried >=3 incompatible local meters
(`single_stream`, canonical `wall_tps`, retired `steady`) that read the SAME stack
5-7% apart. A pair only converts a wall_tps input cleanly if its local leg is also
wall_tps. stark's K=5 number is wall_tps (paired_ab.json metric=wall_tps).

  local meter   pair (local, official)            transfer=off/local   precision
  ------------  --------------------------------  ------------------   ---------
  wall_tps      flagship (454.09, 481.53)         1.0604               bf16   <-- meter-matched
  single_stream int4_g128_lmhead (128.13,126.378) 0.9863               int4   <-- precision-matched
  single_stream lmhead12k diff-cfg (131.60, --)   0.960 (vs 126.378)   int4
  steady-16p    bf16 (~44.01, 44.018)             ~1.000               bf16
  ?             kenyan-420 kanna#44 (423.63,421.12)0.9941              bf16-frontier

=> EXACTLY ONE pair is meter-matched to a wall_tps input (flagship, 1.0604). The
int4 pairs are precision-matched but on a DIFFERENT meter, so their transfer
conflates the hardware factor with the meter gap. You CANNOT fit a cross-precision
line: meter and precision are confounded. We SAY SO (card step 2) and BRACKET
instead of fitting.

THE TRANSFER FACTOR IS THE WHOLE BALL GAME -- and the in-branch pairs disagree by
direction across the confound:
  * bf16/wall_tps says local is SLOW -> official HIGHER -> transfer 1.06 (favorable)
  * int4/single_stream says local runs HOT -> official LOWER -> transfer 0.96-0.99
The advisor's own back-of-envelope ("~159 local / ~139.9 official-equiv") implies
transfer 139.9/159 = 0.880 -- BELOW every measured pair. So we don't assert one
number; we MAP transfer -> verdict and report the BREAK-EVEN transfer (the cleanest
decision-grade quantity: how far the factor must collapse before the fire fails).

NAMED, SOURCED TRANSFER POINTS (all in-branch):
  T_flagship    1.0604  lone meter-matched pair (== K=5's wall_tps meter); lawine #99 fit
  T_definition  1.000   wall_tps == official output_throughput; floor unless local FASTER
  T_int4_match  0.9863  int4_g128_lmhead 126.378/128.13 -- SAME precision as K=5
  T_int4_pess   0.960   lmhead12k 126.378/131.60 -- most pessimistic MEASURED
  T_advisor     0.880   advisor's 139.9/159 -- below all measured (sanity target)
  MEASURED-ANCHORED BAND = [0.960, 1.0604]  (int4-pessimistic .. flagship)

THE BAR HAS NO HAIRCUT
----------------------
126.378 = int4_g128_lmhead (PR#4), a pure-AR rung: NO drafter -> no E_accept
distribution shift -> private == public. So we haircut ONLY the candidate (a
drafter stack, the haircut-prone kind per kanna #44) and hold the bar fixed.

TWO DIFFERENT "AFTER-HAIRCUT" QUESTIONS (disambiguated)
------------------------------------------------------
(1) the card's literal ask: official-equiv*(1-haircut) > 126.378 ? -> comfortable.
(2) the SEPARATE 5% private-repro VALIDITY gate: a submission is DQ'd if private
    drift > 5%. The 4-9% band straddles 5%, so a drafter stack can clear the BAR
    yet fail the GATE. We report both; (2) -- not the bar margin -- is the real risk.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/senpai/target")
HERE = Path(__file__).resolve().parent
K5_PAIRED = ROOT / "research/walltps_ab/optionb_bi1_stock_int4/ksweep/k5/paired_ab.json"
PROJ_CAL = ROOT / "research/walltps_ab/local_official_projection/projection_cal.json"

BAR_OFFICIAL = 126.378            # int4_g128_lmhead PR#4, pure-AR (private==public, no haircut)
ADVISOR_SECONDARY_LOCAL = 159.0   # advisor's verbal "~159" for stark #727 (provenance TBD)
ADVISOR_OFFICIAL_EQUIV = 139.9    # advisor's quoted official-equiv for the 159 -> implies 0.880

# named, sourced transfer points (all in-branch)
T_DEFINITION = 1.000              # wall_tps == official output_throughput
T_INT4_MATCH = 126.378 / 128.13   # 0.9863 int4_g128_lmhead (same precision as K=5)
T_INT4_PESS = 126.378 / 131.60    # 0.9603 lmhead12k (most pessimistic measured)
T_ADVISOR = ADVISOR_OFFICIAL_EQUIV / ADVISOR_SECONDARY_LOCAL  # 0.880 (below all measured)
# T_flagship (1.0604) read from projection_cal.json at runtime

# MEASURED-ANCHORED MC band: int4-pessimistic floor .. flagship ceiling
MC_BAND_LO = T_INT4_PESS          # 0.9603
# MC_BAND_HI = T_flagship (runtime)

# documented private-verify haircut band (BASELINE.md lines 36-44)
HAIRCUT_LO = 0.04
HAIRCUT_HI = 0.09
PRIVATE_REPRO_GATE = 0.05         # validity gate: DQ if private drift > 5%

N_MC = 200_000
SEED = 725


def _load_inputs():
    k5 = json.loads(K5_PAIRED.read_text())
    local_k5 = float(k5["verdict"]["candidate_median_wall_tps"])
    local_k7 = float(k5["verdict"]["baseline_median_wall_tps"])
    e_accept = None
    for path in (("arms", "candidate", "e_accept_exact", "median"),
                 ("arms", "candidate", "e_accept_exact", "mean")):
        cur = k5
        try:
            for key in path:
                cur = cur[key]
            e_accept = float(cur)
            break
        except (KeyError, TypeError):
            continue
    cal = json.loads(PROJ_CAL.read_text())["calibration"]
    mult_flagship = float(cal["local_to_official_multiplier"])  # 1.06019 (lawine #99)
    mult_env = cal.get("multiplier_ci95_envelope", [None, None])
    return {
        "local_k5_wall_tps": local_k5,
        "local_k7_wall_tps": local_k7,
        "k5_e_accept_exact": e_accept,
        "mult_flagship_measured": mult_flagship,
        "mult_envelope_lo": float(mult_env[0]) if mult_env[0] is not None else None,
        "mult_envelope_hi": float(mult_env[1]) if mult_env[1] is not None else None,
    }


def _breakeven_mult(local, haircut):
    """Transfer at which official-equiv*(1-haircut) == bar; below this -> fire fails."""
    return BAR_OFFICIAL / (local * (1.0 - haircut))


def _breakeven_haircut(local, mult):
    """Haircut at which official-equiv*(1-haircut) == bar at a given transfer."""
    return 1.0 - BAR_OFFICIAL / (local * mult)


def _point(local, mult, haircut, name):
    val = local * mult * (1 - haircut)
    return {"name": name, "local": local, "mult": mult, "haircut": haircut,
            "official_equiv_private": val, "margin_pct": 100 * (val / BAR_OFFICIAL - 1),
            "clears": bool(val > BAR_OFFICIAL)}


def _mc_band(local, mult_lo, mult_hi, rng):
    m = rng.uniform(mult_lo, mult_hi, N_MC)
    h = rng.uniform(HAIRCUT_LO, HAIRCUT_HI, N_MC)
    pub = local * m
    priv = pub * (1 - h)
    return {
        "mult_band": [mult_lo, mult_hi], "haircut_band": [HAIRCUT_LO, HAIRCUT_HI],
        "p_clears_bar": float((priv > BAR_OFFICIAL).mean()),
        "official_equiv_public_p50": float(np.percentile(pub, 50)),
        "official_equiv_private_p05": float(np.percentile(priv, 5)),
        "official_equiv_private_p50": float(np.percentile(priv, 50)),
        "official_equiv_private_p95": float(np.percentile(priv, 95)),
    }


def analyze():
    inp = _load_inputs()
    rng = np.random.default_rng(SEED)
    T_FLAGSHIP = inp["mult_flagship_measured"]      # 1.0604
    L172 = inp["local_k5_wall_tps"]                 # 172.74 in-branch
    L159 = ADVISOR_SECONDARY_LOCAL                  # 159 advisor

    transfer_points = {
        "T_flagship_metermatched": T_FLAGSHIP,
        "T_definitional_floor": T_DEFINITION,
        "T_int4_precision_matched": T_INT4_MATCH,
        "T_int4_most_pessimistic_measured": T_INT4_PESS,
        "T_advisor_implied_139p9_over_159": T_ADVISOR,
    }

    # deterministic grid: each named transfer x worst (9%) and best (4%) haircut, both inputs
    grid = {}
    for in_lab, local in [("k5_172", L172), ("adv_159", L159)]:
        grid[in_lab] = {}
        for t_lab, t in transfer_points.items():
            grid[in_lab][t_lab] = {
                "worst_haircut_9pct": _point(local, t, HAIRCUT_HI, f"{in_lab}/{t_lab}/h9"),
                "best_haircut_4pct": _point(local, t, HAIRCUT_LO, f"{in_lab}/{t_lab}/h4"),
                "no_haircut": _point(local, t, 0.0, f"{in_lab}/{t_lab}/h0"),
            }

    # measured-anchored MC band [0.960, 1.0604] for both inputs
    mc = {
        "k5_172_measured_anchored": _mc_band(L172, MC_BAND_LO, T_FLAGSHIP, rng),
        "adv_159_measured_anchored": _mc_band(L159, MC_BAND_LO, T_FLAGSHIP, rng),
    }

    # break-even transfers (THE decision-grade number)
    breakeven = {
        "k5_172_at_worst_9pct_haircut": _breakeven_mult(L172, HAIRCUT_HI),
        "adv_159_at_worst_9pct_haircut": _breakeven_mult(L159, HAIRCUT_HI),
        "k5_172_no_haircut": _breakeven_mult(L172, 0.0),
        "adv_159_no_haircut": _breakeven_mult(L159, 0.0),
        # at the advisor's own 0.880 transfer, how big a haircut can each input survive?
        "max_survivable_haircut_at_advisor_transfer_172": _breakeven_haircut(L172, T_ADVISOR),
        "max_survivable_haircut_at_advisor_transfer_159": _breakeven_haircut(L159, T_ADVISOR),
    }

    # SEPARATE 5% private-repro validity gate
    p_drift_gt_gate = max(0.0, (HAIRCUT_HI - PRIVATE_REPRO_GATE)) / (HAIRCUT_HI - HAIRCUT_LO)

    # 159 vs 172.74 reconciliation (159 ~= 172.74*0.92 -> maybe already private-equiv)
    ratio_159_172 = L159 / L172
    implied_haircut_if_159_private = 1.0 - ratio_159_172

    # VERDICT: bucket on the measured-anchored worst corner (most defensible floor 0.960)
    worst_meas_172 = grid["k5_172"]["T_int4_most_pessimistic_measured"]["worst_haircut_9pct"]
    worst_meas_159 = grid["adv_159"]["T_int4_most_pessimistic_measured"]["worst_haircut_9pct"]
    worst_meas_margin = min(worst_meas_172["margin_pct"], worst_meas_159["margin_pct"])
    p_clears_meas = min(mc["k5_172_measured_anchored"]["p_clears_bar"],
                        mc["adv_159_measured_anchored"]["p_clears_bar"])
    # the single knife-edge corner: advisor's unanchored 0.880 x full 9% x lower input
    knife_corner = grid["adv_159"]["T_advisor_implied_139p9_over_159"]["worst_haircut_9pct"]

    if p_clears_meas >= 0.99 and worst_meas_margin >= 5.0:
        bucket = "a_comfortable"
    elif p_clears_meas >= 0.40:
        bucket = "b_knife_edge"
    else:
        bucket = "c_likely_fails"

    one_number = (
        "FIRE (comfortable). Across EVERY in-branch-measured transfer factor "
        "[0.960 int4-pessimistic .. 1.060 flagship], K=5 clears 126.378 after the full "
        "9%% haircut for both the 172.74 in-branch and 159 advisor inputs: worst "
        "measured-anchored corner = %.1f TPS (159 x 0.960 x 0.91, +%.1f%%). P(clears)=1.00. "
        "The fire fails ONLY if the true int4 wall_tps->official transfer drops below %.3f "
        "(159) / %.3f (172.74) -- more pessimistic than any measured pair INCLUDING the int4 "
        "rungs. Even the advisor's own 139.9 (implied transfer 0.880, below all measured) "
        "still clears the post-9%%-haircut bar by +%.1f%% (%.1f TPS). The genuine downside is "
        "NOT the bar margin but the SEPARATE 5%% private-repro validity gate, which the 4-9%% "
        "haircut band straddles -- a drafter stack can beat 126.378 yet be DQ'd."
    ) % (
        worst_meas_159["official_equiv_private"], worst_meas_159["margin_pct"],
        breakeven["adv_159_at_worst_9pct_haircut"], breakeven["k5_172_at_worst_9pct_haircut"],
        knife_corner["margin_pct"], knife_corner["official_equiv_private"],
    )

    return {
        "pr": 725, "student": "kanna",
        "card": "local->official-equiv calibration + private-haircut risk for stark K=5 MTP fire",
        "guard_flags": {"analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0},
        "bar_official": BAR_OFFICIAL,
        "bar_config": "int4_g128_lmhead PR#4 (pure-AR; private==public; no haircut on the bar)",
        "inputs": inp,
        "pair_accounting": {
            "n_clean_meter_matched_pairs": 1,
            "the_meter_matched_pair": {"config": "flagship fa2sw_precache_kenyan",
                                       "local_wall_tps": 454.09, "official": 481.53,
                                       "transfer": 1.0604, "meter": "wall_tps (== K=5 input meter)"},
            "precision_matched_pair": {"config": "int4_g128_lmhead", "local_single_stream": 128.13,
                                       "official": 126.378, "transfer": T_INT4_MATCH,
                                       "caveat": "different meter (single_stream != wall_tps)"},
            "fit_possible": False,
            "honest_statement": ("Too few meter-matched pairs to FIT a line; meter and precision "
                                 "are confounded (bf16->wall_tps vs int4->single_stream). We BRACKET "
                                 "with named sourced transfer points and report the break-even."),
        },
        "transfer_points": transfer_points,
        "measured_anchored_band": [MC_BAND_LO, T_FLAGSHIP],
        "haircut_band": {"lo": HAIRCUT_LO, "hi": HAIRCUT_HI, "bar_haircut": 0.0,
                         "note": "applied to candidate only; bar is pure-AR"},
        "deterministic_grid": grid,
        "monte_carlo_measured_anchored": mc,
        "breakeven": breakeven,
        "private_repro_gate_SEPARATE_risk": {
            "gate_threshold_pct": PRIVATE_REPRO_GATE * 100,
            "haircut_band_straddles_gate": HAIRCUT_LO < PRIVATE_REPRO_GATE < HAIRCUT_HI,
            "p_drift_exceeds_gate_naive_uniform": p_drift_gt_gate,
            "interpretation": ("clearing the 126.378 BAR is comfortable; the 5%% private-repro "
                               "VALIDITY gate is the real residual risk. drafter stacks drift most "
                               "(kanna#44). flagship real drift was 4.3%% (passed). K=5 is a drafter "
                               "stack -> its private drift is UNMEASURED in-branch."),
        },
        "input_reconciliation_159_vs_172": {
            "k5_inbranch_wall_tps": L172, "advisor_secondary": L159,
            "ratio": ratio_159_172, "implied_haircut_if_159_already_private": implied_haircut_if_159_private,
            "advisor_official_equiv_quoted": ADVISOR_OFFICIAL_EQUIV,
            "advisor_implied_transfer": T_ADVISOR,
            "note": ("159 ~= 172.74*0.92 (8.0%% below) -> 159 may already be a haircut-applied "
                     "private-equiv figure; if so, re-applying the haircut double-counts. SEPARATELY "
                     "the advisor's 139.9 implies a 0.880 transfer, BELOW all in-branch pairs -- ask "
                     "the advisor where 159 and 139.9 came from. stark #727 is outside kanna's launch "
                     "scope; decision is robust to either input so this is non-blocking."),
        },
        "verdict": {
            "bucket": bucket,
            "p_clears_bar_measured_anchored": p_clears_meas,
            "worst_measured_anchored_corner_margin_pct": worst_meas_margin,
            "knife_edge_corner": knife_corner,
            "one_number_for_the_advisor": one_number,
        },
    }


def _print(out):
    print("WROTE", out.get("_outpath"))
    print("VERDICT BUCKET:", out["verdict"]["bucket"])
    print("ONE NUMBER:", out["verdict"]["one_number_for_the_advisor"])
    print("-- deterministic grid (official-equiv AFTER 9% worst haircut) --")
    for in_lab in out["deterministic_grid"]:
        for t_lab, cell in out["deterministic_grid"][in_lab].items():
            p = cell["worst_haircut_9pct"]
            print(f"  {in_lab:8s} {t_lab:36s} t={p['mult']:.4f} -> {p['official_equiv_private']:7.1f} "
                  f"({p['margin_pct']:+6.1f}%) clears={p['clears']}")
    print("-- break-even transfers --")
    for k, v in out["breakeven"].items():
        print(f"  {k} = {v:.4f}")
    print("-- MC measured-anchored P(clears) --")
    for k, v in out["monte_carlo_measured_anchored"].items():
        print(f"  {k}: P={v['p_clears_bar']:.4f} pub_p50={v['official_equiv_public_p50']:.1f} "
              f"priv_p05={v['official_equiv_private_p05']:.1f}")
    print("  5% private-repro gate P(drift>5%) naive:",
          out["private_repro_gate_SEPARATE_risk"]["p_drift_exceeds_gate_naive_uniform"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--name", default="kanna/official-equiv-k5-fire-haircut-risk")
    ap.add_argument("--out", default=str(HERE / "results/k5_fire_haircut_risk.json"))
    args = ap.parse_args()

    out = analyze()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    out["_outpath"] = str(outp)
    _print(out)

    if args.wandb:
        import wandb
        v = out["verdict"]
        g = out["deterministic_grid"]
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group="kanna-official-equiv-calibration", name=args.name, job_type="analysis",
            config={
                "pr": 725, "student": "kanna",
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "bar_official": BAR_OFFICIAL,
                "measured_anchored_band": out["measured_anchored_band"],
                "haircut_band": [HAIRCUT_LO, HAIRCUT_HI],
                "n_clean_meter_matched_pairs": 1,
                "k5_local_wall_tps": out["inputs"]["local_k5_wall_tps"],
                "advisor_secondary_local": ADVISOR_SECONDARY_LOCAL,
                "transfer_points": out["transfer_points"],
            },
            tags=["pr725", "kanna", "analysis_only", "official-equiv-calibration", "k5-fire", v["bucket"]],
        )
        summary = {
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "verdict_bucket": v["bucket"],
            "primary_metric_name": "p_clears_126378_after_haircut_measured_anchored",
            "primary_metric_value": v["p_clears_bar_measured_anchored"],
            "test_metric_name": "worst_measured_anchored_corner_margin_pct",
            "test_metric_value": v["worst_measured_anchored_corner_margin_pct"],
            "n_clean_meter_matched_pairs": 1,
            "mult_flagship_measured": out["inputs"]["mult_flagship_measured"],
            "T_int4_precision_matched": T_INT4_MATCH,
            "T_int4_most_pessimistic_measured": T_INT4_PESS,
            "T_advisor_implied": T_ADVISOR,
            "k5_local_wall_tps": out["inputs"]["local_k5_wall_tps"],
            "k5_e_accept_exact": out["inputs"]["k5_e_accept_exact"],
            "k5_official_equiv_public_p50": out["monte_carlo_measured_anchored"]["k5_172_measured_anchored"]["official_equiv_public_p50"],
            "p_clears_172_measured": out["monte_carlo_measured_anchored"]["k5_172_measured_anchored"]["p_clears_bar"],
            "p_clears_159_measured": out["monte_carlo_measured_anchored"]["adv_159_measured_anchored"]["p_clears_bar"],
            "breakeven_transfer_172_at_9pct_haircut": out["breakeven"]["k5_172_at_worst_9pct_haircut"],
            "breakeven_transfer_159_at_9pct_haircut": out["breakeven"]["adv_159_at_worst_9pct_haircut"],
            "worst_meas_corner_159_official_equiv": g["adv_159"]["T_int4_most_pessimistic_measured"]["worst_haircut_9pct"]["official_equiv_private"],
            "knife_corner_advisor_159_official_equiv": v["knife_edge_corner"]["official_equiv_private"],
            "knife_corner_advisor_159_margin_pct": v["knife_edge_corner"]["margin_pct"],
            "private_repro_gate_pct": PRIVATE_REPRO_GATE * 100,
            "p_private_drift_gt_5pct_naive": out["private_repro_gate_SEPARATE_risk"]["p_drift_exceeds_gate_naive_uniform"],
            "implied_haircut_if_159_is_private": out["input_reconciliation_159_vs_172"]["implied_haircut_if_159_already_private"],
        }
        run.summary.update(summary)
        wandb.log(summary)
        print("WANDB_RUN_ID", run.id)
        print("WANDB_RUN_URL", run.url)
        wandb.finish()


if __name__ == "__main__":
    main()
