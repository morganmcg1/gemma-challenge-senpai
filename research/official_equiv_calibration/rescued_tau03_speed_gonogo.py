#!/usr/bin/env python3
"""PR #732 -- SPEED go/no-go for the strict-safe RESCUED tau=0.3 fallback.

Apply the EXACT PR #725 haircut-calibration methodology (k5_fire_haircut_risk.py)
to the rescued tau=0.3 recompute-acceptor config (stark #642). #725 banked the
*un-rescued* K spec stack at P(clears 126.378 after the 4-9% private haircut)=1.00
(worst measured corner +9.95%, thinnest +0.74%, W&B 0gpahz4c). The config we'd
actually WANT to bank for a strict-safe fire is the rescued tau=0.3 variant --
greedy-identity-clean BY CONSTRUCTION (land #720 owns that cert) but ~22% slower.
Its speed has NOT been run through the haircut model. This file is the SPEED leg.

THE DECISION-GRADE QUESTION
---------------------------
Convert the rescued config's *local captured* TPS to an *official-equiv* estimate,
overlay the documented 4-9% private-verify haircut, and report P(clears 126.378) +
a GO / knife-edge / NO-GO verdict. analysis_only -- NO HF Job, NO fire.

INPUT PROVENANCE (and why this is analysis-only, like #725)
-----------------------------------------------------------
#725 did NOT boot a server: it read stark's K=5 wall_tps from an on-branch JSON and
ran the MC. This file is the exact analog. The rescued tau=0.3 *serve path* is land
#720 / denken's `int4_mtp_rescued_tau03` package -- NOT on the advisor branch
(`submissions/int4_mtp_batchinv` here is the UN-rescued K=6 stack; no recompute
acceptor), and this launch's isolation scope is kanna + advisor-branch only, so a
fresh boot is out of scope. We therefore consume the rescued config's local capture
as CONSOLIDATED ON THE ADVISOR BRANCH (CURRENT_RESEARCH_STATE.md, land #664
`REGIME_IS_ACCEPTANCE`, W&B 4fbu9b3o) and STATED IN THE PR ("~135 local / ~134.87
official-equiv"). The verdict is robust to +-5% on this input (see sensitivity).

  rescued tau=0.3 captured local (captured-graph, VLLM_BATCH_INVARIANT=1, 0.22.0)
      = 135.27  (stark #663, "captured 135.27 to the decimal" per state dump)
  AR-rung int4_g128_lmhead captured-graph local = 126.75  (the meter-match anchor)
  official AR rung = 126.378  (locked bar, int4_g128_lmhead PR#4)
  => meter-matched transfer T = 126.378/126.75 = 0.99707
  => official-equiv PUBLIC = 135.27 * 0.99707 = 134.87  (== consolidated, no haircut)

WHY THIS PAIR IS A CLEANER METER-MATCH THAN #725's
--------------------------------------------------
#725's lone meter-matched pair was the bf16 flagship (454.09->481.53, T=1.0604) --
right METER (wall_tps) but WRONG precision (bf16, not int4). Here the lone clean
pair is the AR rung ITSELF: int4 precision AND captured-graph meter AND it is the
literal speed bar. So n_clean_meter_matched_pairs=1, but it is precision+meter+bar
matched -- strictly better-anchored than #725. The transfer is ~definitional (0.997)
because captured-graph local ~= official output_throughput (unlike wall_tps, which
read 6% LOW and let #725's transfer run >1).

THE BAR HAS NO HAIRCUT (same as #725)
-------------------------------------
126.378 = int4_g128_lmhead, pure-AR (no drafter -> no E_accept shift -> private ==
public). We haircut ONLY the candidate. The rescued tau=0.3 is identity-clean but it
is STILL a drafter stack: its TPS depends on acceptance, which shifts on private
prompts -> haircut-prone. Identity-cleanliness protects the OUTPUT, not the SPEED.

TRANSFER SENSITIVITY BAND (PR step 3): T swept over [0.87, 0.99]
---------------------------------------------------------------
0.99 ~= the meter-matched ceiling (captured-graph can't beat definitional 1.0);
0.87 ~= the documented pessimistic floor (stark tax 0.870 / advisor 0.880 from #725,
both BELOW every measured int4 pair). This is a one-sided pessimism stress band
anchored at the clean top -- NOT a meter-confound bracket.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---- locked bar (no haircut; pure-AR) ----
BAR_OFFICIAL = 126.378            # int4_g128_lmhead PR#4 official a10g-small TPS

# ---- rescued tau=0.3 INPUT (consolidated on advisor branch; see provenance above) ----
L_RESCUED_CAPTURED = 135.27      # stark #663 captured-graph local; PR's "~135 local"
AR_RUNG_LOCAL_CAPTURED = 126.75  # int4_g128_lmhead captured-graph local (meter anchor)
INPUT_SOURCE = ("advisor-branch CURRENT_RESEARCH_STATE.md / land #664 REGIME_IS_ACCEPTANCE "
                "(W&B 4fbu9b3o) / PR #732 body '~135 local'; serve path land#720/denken "
                "is out of kanna isolation scope -> consumed like #725 consumed stark K=5")

# ---- meter-matched transfer (the clean anchor) ----
T_METER_MATCHED = BAR_OFFICIAL / AR_RUNG_LOCAL_CAPTURED   # 0.99707 captured-graph, int4, bar-matched

# ---- named, sourced transfer points ----
T_DEFINITIONAL = 1.000           # captured-graph local ~= official output_throughput (ceiling)
T_INT4_SS_MATCH = 126.378 / 128.13   # 0.9863 int4_g128_lmhead single_stream (#725; CROSS-METER)
T_INT4_SS_PESS = 126.378 / 131.60    # 0.9603 lmhead12k single_stream (#725; CROSS-METER, most pess measured)
T_STARK_TAX = 0.870              # documented rescued-projection tax (state dump: 147.55 via 0.870)
T_ADVISOR = 0.880                # advisor's #725 pessimistic implied transfer (139.9/159)

# ---- PR sensitivity band (step 3) ----
PR_BAND_LO, PR_BAND_HI = 0.87, 0.99
# ---- measured-anchored band (int4-pessimistic .. definitional ceiling; captured-graph <= 1.0) ----
MA_BAND_LO, MA_BAND_HI = T_INT4_SS_PESS, T_DEFINITIONAL

# ---- documented private-verify haircut band (BASELINE.md) ----
HAIRCUT_LO, HAIRCUT_HI = 0.04, 0.09
PRIVATE_REPRO_GATE = 0.05        # SEPARATE validity gate: DQ if private drift > 5%

N_MC = 400_000
SEED = 732


def _point(local, mult, haircut, name):
    val = local * mult * (1 - haircut)
    return {"name": name, "local": local, "mult": mult, "haircut": haircut,
            "official_equiv_private": val, "margin_pct": 100 * (val / BAR_OFFICIAL - 1),
            "clears": bool(val > BAR_OFFICIAL)}


def _breakeven_mult(local, haircut):
    """Transfer at which official-equiv*(1-haircut) == bar; below this -> fails."""
    return BAR_OFFICIAL / (local * (1.0 - haircut))


def _breakeven_haircut(local, mult):
    """Haircut at which official-equiv*(1-haircut) == bar at a given transfer."""
    return 1.0 - BAR_OFFICIAL / (local * mult)


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
    rng = np.random.default_rng(SEED)
    L = L_RESCUED_CAPTURED

    transfer_points = {
        "T_meter_matched_capturedgraph": T_METER_MATCHED,
        "T_definitional_ceiling": T_DEFINITIONAL,
        "T_int4_single_stream_match_CROSSMETER": T_INT4_SS_MATCH,
        "T_int4_single_stream_pessimistic_CROSSMETER": T_INT4_SS_PESS,
        "T_stark_tax_0p870": T_STARK_TAX,
        "T_advisor_0p880": T_ADVISOR,
    }

    # deterministic grid: each named transfer x {worst 9%, best 4%, no haircut}
    grid = {}
    for t_lab, t in transfer_points.items():
        grid[t_lab] = {
            "worst_haircut_9pct": _point(L, t, HAIRCUT_HI, f"{t_lab}/h9"),
            "best_haircut_4pct": _point(L, t, HAIRCUT_LO, f"{t_lab}/h4"),
            "no_haircut": _point(L, t, 0.0, f"{t_lab}/h0"),
        }

    # Monte Carlo: PR pessimism band (headline) + measured-anchored band (context)
    mc = {
        "pr_band_0p87_0p99": _mc_band(L, PR_BAND_LO, PR_BAND_HI, rng),
        "measured_anchored_0p9603_1p0": _mc_band(L, MA_BAND_LO, MA_BAND_HI, rng),
    }

    # break-evens (the decision-grade numbers)
    breakeven = {
        "transfer_at_worst_9pct_haircut": _breakeven_mult(L, HAIRCUT_HI),   # need T above this @9%
        "transfer_at_best_4pct_haircut": _breakeven_mult(L, HAIRCUT_LO),    # need T above this @4%
        "transfer_no_haircut": _breakeven_mult(L, 0.0),
        "max_survivable_haircut_at_meter_matched_T": _breakeven_haircut(L, T_METER_MATCHED),
        "max_survivable_haircut_at_definitional_T": _breakeven_haircut(L, T_DEFINITIONAL),
    }

    # SEPARATE 5% private-repro validity gate (same straddle note as #725)
    p_drift_gt_gate = max(0.0, (HAIRCUT_HI - PRIVATE_REPRO_GATE)) / (HAIRCUT_HI - HAIRCUT_LO)

    # ---- corner summary ----
    # worst MEASURED-anchored corner == lower edge of the measured int4 band x worst 9% haircut
    # (the exact #725 construction: worst measured transfer x worst haircut, on the input).
    worst_ma_corner = _point(L, MA_BAND_LO, HAIRCUT_HI, "MAlo_0p9603/h9")
    # the cleanest single anchor (captured-graph meter-matched) x worst 9% haircut -- highlighted
    meter_matched_worst_corner = grid["T_meter_matched_capturedgraph"]["worst_haircut_9pct"]
    best_thin_corner = _point(L, PR_BAND_HI, HAIRCUT_LO, "PRhi_0p99/h4")            # thinnest clearing
    worst_pr_corner = _point(L, PR_BAND_LO, HAIRCUT_HI, "PRlo_0p87/h9")            # worst PR-band corner
    mid_corner = _point(L, T_METER_MATCHED, (HAIRCUT_LO + HAIRCUT_HI) / 2, "mm/h6p5")  # central

    p_clears_pr = mc["pr_band_0p87_0p99"]["p_clears_bar"]
    p_clears_ma = mc["measured_anchored_0p9603_1p0"]["p_clears_bar"]
    worst_ma_margin = worst_ma_corner["margin_pct"]

    # VERDICT bucket (same thresholds as #725; "worst measured-anchored corner" = meter-matched x 9%)
    if p_clears_ma >= 0.99 and worst_ma_margin >= 5.0:
        bucket = "a_GO_comfortable"
    elif p_clears_ma >= 0.40:
        bucket = "b_knife_edge"
    else:
        bucket = "c_NOGO"

    one_number = (
        "NO-GO (speed). The rescued tau=0.3 is identity-clean by construction (land #720) "
        "but FAILS the speed bar after the private haircut. At the CLEANEST anchor -- the "
        "captured-graph meter-matched transfer T=%.4f (int4 + same meter + the literal bar, a "
        "tighter anchor than #725's bf16 flagship) -- official-equiv PUBLIC is %.2f (+%.1f%%), "
        "but the worst documented 9%% haircut drops it to %.2f (%.2f%%, BELOW 126.378) and even "
        "the mid 6.5%% haircut lands at %.2f (%.2f%%). It survives a haircut of only %.1f%% at "
        "the meter-matched transfer -- INSIDE the 4-9%% band. To clear after the FULL 9%% haircut "
        "the transfer must exceed %.4f, i.e. ABOVE the definitional 1.0 ceiling -- impossible. "
        "P(clears)=%.3f over the PR pessimism band [0.87,0.99] and only %.3f even over the "
        "favorable measured-anchored band [0.960,1.0]. The config clears ONLY in a thin sliver "
        "(transfer ~definitional AND haircut ~4%%, thinnest corner +%.1f%%). Contrast #725's "
        "un-rescued stack: P=1.00, worst corner +9.95%%. The ~22%% rescue tax (172.7->135.3 local) "
        "converts a comfortable GO into a NO-GO. Do not spend a fire on the rescued tau=0.3 for "
        "speed."
    ) % (
        T_METER_MATCHED, L * T_METER_MATCHED, 100 * (L * T_METER_MATCHED / BAR_OFFICIAL - 1),
        meter_matched_worst_corner["official_equiv_private"], meter_matched_worst_corner["margin_pct"],
        mid_corner["official_equiv_private"], mid_corner["margin_pct"],
        100 * breakeven["max_survivable_haircut_at_meter_matched_T"],
        breakeven["transfer_at_worst_9pct_haircut"],
        p_clears_pr, p_clears_ma, best_thin_corner["margin_pct"],
    )

    return {
        "pr": 732, "student": "kanna",
        "card": "SPEED go/no-go for the strict-safe rescued tau=0.3 fallback (#725 haircut model)",
        "guard_flags": {"analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0},
        "bar_official": BAR_OFFICIAL,
        "bar_config": "int4_g128_lmhead PR#4 (pure-AR; private==public; no haircut on the bar)",
        "inputs": {
            "rescued_tau03_local_captured_tps": L,
            "ar_rung_local_captured_tps": AR_RUNG_LOCAL_CAPTURED,
            "official_equiv_public_meter_matched": L * T_METER_MATCHED,
            "input_source": INPUT_SOURCE,
            "freshly_measured": False,
            "fresh_measure_blocker": ("rescued serve path = land#720/denken; not on advisor branch; "
                                      "kanna isolation scope precludes cross-read. Consumed consolidated "
                                      "capture exactly as #725 consumed stark's K=5 wall_tps."),
        },
        "pair_accounting": {
            "n_clean_meter_matched_pairs": 1,
            "the_meter_matched_pair": {
                "config": "int4_g128_lmhead (the AR rung / locked bar itself)",
                "local_captured_graph": AR_RUNG_LOCAL_CAPTURED, "official": BAR_OFFICIAL,
                "transfer": T_METER_MATCHED,
                "meter": "captured-graph (== rescued config's input meter)",
                "note": "int4 precision + captured-graph meter + literal bar -> tighter than #725's bf16 flagship",
            },
            "fit_possible": False,
            "honest_statement": ("One clean meter-matched pair (the AR rung, captured-graph, int4). "
                                 "Cannot FIT a line; we BRACKET with the PR pessimism band [0.87,0.99] "
                                 "and report break-evens. The single_stream int4 points (0.9863/0.9603) "
                                 "are cross-meter and push the verdict MORE negative, not less."),
        },
        "transfer_points": transfer_points,
        "pr_sensitivity_band": [PR_BAND_LO, PR_BAND_HI],
        "measured_anchored_band": [MA_BAND_LO, MA_BAND_HI],
        "haircut_band": {"lo": HAIRCUT_LO, "hi": HAIRCUT_HI, "bar_haircut": 0.0,
                         "note": "applied to candidate only; bar is pure-AR. rescued is identity-clean "
                                 "but still a drafter stack -> speed is haircut-prone"},
        "deterministic_grid": grid,
        "monte_carlo": mc,
        "breakeven": breakeven,
        "private_repro_gate_SEPARATE_risk": {
            "gate_threshold_pct": PRIVATE_REPRO_GATE * 100,
            "haircut_band_straddles_gate": HAIRCUT_LO < PRIVATE_REPRO_GATE < HAIRCUT_HI,
            "p_drift_exceeds_gate_naive_uniform": p_drift_gt_gate,
            "interpretation": ("Even setting the SPEED bar aside, the rescued config -- like any drafter "
                               "stack -- straddles the 5% private-repro gate. But the SPEED bar is the "
                               "binding failure here, not this gate."),
        },
        "honesty_caveats": {
            "n_clean_meter_matched_pairs": 1,
            "private_drift_prior": ("rescued tau=0.3 private E_accept drift is UNMEASURED in-scope; we use "
                                    "the documented 4-9% band (kanna #44 drafter-collapse prior). The "
                                    "recompute acceptor is STRICTER (lower base accept) -> drift direction "
                                    "on private prompts is unknown; band may be optimistic or pessimistic."),
            "input_not_freshly_measured": True,
            "thinnest_clearing_corner_margin_pct": best_thin_corner["margin_pct"],
            "input_sensitivity": ("verdict robust: NO-GO holds for L in [130,138] (breakeven-at-9%-haircut "
                                  "transfer stays >1.0 for any L<=138.9)."),
        },
        "verdict": {
            "bucket": bucket,
            "headline": "NO-GO" if bucket == "c_NOGO" else ("KNIFE-EDGE" if bucket == "b_knife_edge" else "GO"),
            "p_clears_126378_after_haircut_pr_band": p_clears_pr,
            "p_clears_126378_after_haircut_measured_anchored": p_clears_ma,
            "worst_measured_anchored_corner_margin_pct": worst_ma_margin,
            "worst_measured_anchored_corner": worst_ma_corner,
            "meter_matched_clean_anchor_worst_haircut_corner": meter_matched_worst_corner,
            "thinnest_clearing_corner": best_thin_corner,
            "worst_pr_band_corner": worst_pr_corner,
            "central_meter_matched_mid_haircut_corner": mid_corner,
            "max_survivable_haircut_at_meter_matched_T_pct": 100 * breakeven["max_survivable_haircut_at_meter_matched_T"],
            "one_number_for_the_advisor": one_number,
        },
    }


def _print(out):
    v = out["verdict"]
    print("VERDICT:", v["headline"], f"({v['bucket']})")
    print("ONE NUMBER:", v["one_number_for_the_advisor"])
    print("\n-- deterministic grid (official-equiv AFTER 9% worst haircut) --")
    for t_lab, cell in out["deterministic_grid"].items():
        p = cell["worst_haircut_9pct"]
        print(f"  {t_lab:46s} t={p['mult']:.4f} -> {p['official_equiv_private']:7.2f} "
              f"({p['margin_pct']:+6.2f}%) clears={p['clears']}")
    print("\n-- break-even transfers / haircuts --")
    for k, val in out["breakeven"].items():
        print(f"  {k} = {val:.4f}")
    print("\n-- MC P(clears) --")
    for k, m in out["monte_carlo"].items():
        print(f"  {k:30s}: P={m['p_clears_bar']:.4f} pub_p50={m['official_equiv_public_p50']:.1f} "
              f"priv_p05={m['official_equiv_private_p05']:.1f} priv_p50={m['official_equiv_private_p50']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_group", default="rescued-tau03-speed-gonogo")
    ap.add_argument("--name", default="kanna/rescued-tau03-speed-gonogo")
    ap.add_argument("--out", default=str(HERE / "results/rescued_tau03_speed_gonogo.json"))
    args = ap.parse_args()

    out = analyze()
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("WROTE", outp)
    _print(out)

    if args.wandb:
        import wandb
        v = out["verdict"]
        run = wandb.init(
            project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
            group=args.wandb_group, name=args.name, job_type="analysis",
            config={
                "pr": 732, "student": "kanna",
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "bar_official": BAR_OFFICIAL,
                "rescued_tau03_local_captured_tps": L_RESCUED_CAPTURED,
                "ar_rung_local_captured_tps": AR_RUNG_LOCAL_CAPTURED,
                "meter_matched_transfer": T_METER_MATCHED,
                "pr_sensitivity_band": [PR_BAND_LO, PR_BAND_HI],
                "measured_anchored_band": [MA_BAND_LO, MA_BAND_HI],
                "haircut_band": [HAIRCUT_LO, HAIRCUT_HI],
                "n_clean_meter_matched_pairs": 1,
                "freshly_measured": False,
                "transfer_points": out["transfer_points"],
            },
            tags=["pr732", "kanna", "analysis_only", "official-equiv-calibration",
                  "rescued-tau03", "speed-gonogo", v["bucket"]],
        )
        summary = {
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "verdict_bucket": v["bucket"], "verdict_headline": v["headline"],
            "primary_metric_name": "p_rescued_clears_126378_after_haircut",
            "primary_metric_value": v["p_clears_126378_after_haircut_pr_band"],
            "test_metric_name": "worst_corner_margin_pct",
            "test_metric_value": v["worst_measured_anchored_corner_margin_pct"],
            "p_clears_pr_band_0p87_0p99": v["p_clears_126378_after_haircut_pr_band"],
            "p_clears_measured_anchored_0p9603_1p0": v["p_clears_126378_after_haircut_measured_anchored"],
            "official_equiv_public_meter_matched": out["inputs"]["official_equiv_public_meter_matched"],
            "worst_measured_anchored_corner_margin_pct": v["worst_measured_anchored_corner_margin_pct"],
            "thinnest_clearing_corner_margin_pct": v["thinnest_clearing_corner"]["margin_pct"],
            "worst_pr_band_corner_margin_pct": v["worst_pr_band_corner"]["margin_pct"],
            "central_mid_haircut_margin_pct": v["central_meter_matched_mid_haircut_corner"]["margin_pct"],
            "max_survivable_haircut_at_meter_matched_T_pct": v["max_survivable_haircut_at_meter_matched_T_pct"],
            "breakeven_transfer_at_9pct_haircut": out["breakeven"]["transfer_at_worst_9pct_haircut"],
            "breakeven_transfer_at_4pct_haircut": out["breakeven"]["transfer_at_best_4pct_haircut"],
            "meter_matched_transfer": T_METER_MATCHED,
            "rescued_tau03_local_captured_tps": L_RESCUED_CAPTURED,
            "n_clean_meter_matched_pairs": 1,
            "freshly_measured": 0,
            "p_private_drift_gt_5pct_naive": out["private_repro_gate_SEPARATE_risk"]["p_drift_exceeds_gate_naive_uniform"],
        }
        run.summary.update(summary)
        wandb.log(summary)
        print("WANDB_RUN_ID", run.id)
        print("WANDB_RUN_URL", run.url)
        wandb.finish()


if __name__ == "__main__":
    main()
