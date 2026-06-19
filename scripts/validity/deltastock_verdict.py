#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Turn a delta_stock_measure report.json into the #749 FAITHFUL G1 fire-gate verdict.

#739 measured the stock-drafter delta_stock per prompt-subset and reported a
composition-weighted central of -5.94% (a net speedup). But that headline applied
the WHOLE 114-MCQ public weight to the MMLU-Pro held-out proxy, while the public
MCQ block is actually HALF GPQA-Diamond (57 MMLU-Pro + 57 GPQA-D), which is harder
and accepts worse. So -5.94% is OPTIMISTIC.

#749 removes that bias. We measure a held-out GPQA-Diamond proxy on the same exact
fire config and assemble the FAITHFUL public composition (57 MMLU-Pro + 57 GPQA-D
+ 14 free-response math):

    delta_stock(faithful) = W_MMLU*d_know + W_GPQA*d_gpqa + W_MATH*d_math
    W_MMLU = W_GPQA = 57/128,  W_MATH = 14/128

The measured per-subset delta_stock distribution still BRACKETS the real private gap:

  same-family held-out (knowledge_mmlupro + gpqa_diamond + reasoning_math) -> LOWER
      bracket (fresh instances of the SAME MMLU-Pro/GPQA/AIME suite the public set
      is drawn from; the faithful realization of "private = held-out same-benchmark")
  pure-chat / code / multilingual                                          -> UPPER
      bracket (OOD worst corners; pure-chat reconciles with kanna #44's 12.4%)

The strongest external anchor is flagship #52: +4.3% measured on the ACTUAL private
set (VALID). The faithful central is calibrated against it.

primary_metric = p_dq_g1_faithful_mix = P(delta_stock > 5%)  (lower=better)
test_metric    = delta_stock_faithful_central_pct  (faithful 57/57/14 held-out central)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from math import erf, sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Faithful public composition (57 MMLU-Pro + 57 GPQA-Diamond + 14 free-response math).
W_MMLU_F, W_GPQA_F, W_MATH_F = 57 / 128, 57 / 128, 14 / 128
# #739 OPTIMISTIC composition: the whole 114-MCQ weight on the MMLU-Pro proxy only.
W_MCQ_739, W_MATH_739 = 114 / 128, 14 / 128

SAME_FAMILY = ["knowledge_mmlupro", "gpqa_diamond", "reasoning_math"]
ANCHOR_52 = 4.3   # flagship #52, measured on REAL private, VALID
ANCHOR_44 = 12.4  # kanna #44 chat-proxy, pessimistic upper bound
G1 = 5.0

# #739 OOD-corner e_accept (chat/code/multilingual), same engine + same data, not
# re-measured this session. For the all-shifted / empirical sensitivity band we
# carry these over and recompute their delta against THIS session's public anchor
# so every delta is relative to one consistent e_accept_public.
OOD_739_EACCEPT = {
    "chat_casual": 4.022935220202513,
    "code": 3.2762425447316104,
    "multilingual": 4.043339472068753,
}


def gauss_tail(mu: float, sigma: float, thr: float = G1) -> float:
    if sigma <= 0:
        return 1.0 if mu > thr else 0.0
    return 0.5 * (1 - erf((thr - mu) / sigma / sqrt(2)))


def log_to_wandb(verdict: dict, report: dict, *, wandb_group: str, wandb_name: str,
                 report_path: str) -> str | None:
    """Log the authoritative #749 faithful-verdict run (separate from the driver's
    per-subset 'deltastock-measure' run). Returns the run id or None."""
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_file_artifact, log_summary)
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] unavailable: {e}", flush=True)
        return None
    c = verdict["centrals"]
    cor = verdict["corners"]
    pdq = verdict["p_dq_readings"]
    dec = verdict["decision"]
    summary = {
        # required #749 analysis-only flags
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # headline decision metrics
        "p_dq_g1_faithful_mix": dec["primary_metric_p_dq_g1_faithful_mix"],
        "delta_stock_faithful_central_pct": dec["test_metric_delta_stock_faithful_central_pct"],
        "delta_stock_optimistic_739_central_pct": c["optimistic_739_114_14"],
        "gpqa_correction_shift_pct": c["gpqa_correction_shift"],
        "gpqa_minus_mmlu_corner_gap_pct": c["gpqa_minus_mmlu_corner_gap"],
        # the GPQA-D-only corner (the load-bearing new measurement)
        "gpqa_diamond_delta_stock_pct": cor["gpqa_diamond_delta_pct"],
        "mmlu_pro_delta_stock_pct": cor["mmlu_pro_delta_pct"],
        "reasoning_math_delta_stock_pct": cor["reasoning_math_delta_pct"],
        "breakeven_private_math_fraction_to_5pct": cor["breakeven_private_math_fraction_to_5pct"],
        "breakeven_gpqa_share_of_mcq_to_5pct": cor["breakeven_gpqa_share_of_mcq_to_5pct"],
        # sensitivity band
        "p_dq_calibrated_headline": pdq["calibrated_headline"],
        "p_dq_empirical_frac_gt5_all_shifted": pdq["empirical_frac_gt5_all_shifted"],
        "p_dq_gaussian_all_shifted_pessimistic": pdq["gaussian_all_shifted_pessimistic"],
        "p_dq_gaussian_equal_weight_family": pdq["gaussian_equal_weight_family_artifact"],
        "cal_mu": pdq["calibrated_model"]["mu"], "cal_sigma": pdq["calibrated_model"]["sigma"],
        # decision flags
        "clears_g1_central": int(bool(dec["clears_g1_central"])),
        "faithful_g1_safe": int(bool(dec["faithful_g1_safe_le2_and_pdq_lt_0p13"])),
        "margin_to_5pct": dec["margin_to_5pct"],
        "e_accept_public": verdict.get("e_accept_public"),
        **{f"e_accept_delta__{k}": v for k, v in verdict["per_subset_delta_pct"].items()},
    }
    run = init_wandb_run(
        job_type="deltastock-faithful-verdict", agent="senpai", name=wandb_name,
        group=wandb_group, tags=["deltastock-faithful-verdict", wandb_group],
        notes=f"#749 faithful 57/57/14 held-out delta_stock; verdict={dec['verdict']}",
        config={"verdict": dec["verdict"], "analysis_only": True, "no_hf_job": True,
                "fires": 0, "faithful_weights": c["faithful_weights"],
                "anchor_52_pct": ANCHOR_52, "g1_threshold_pct": G1,
                "ood_carryover_used": pdq["ood_carryover_used"]})
    if run is None:
        print("[wandb] run not created; verdict json is the record", flush=True)
        return None
    log_summary(run, summary, step=0)
    run.summary["verdict"] = dec["verdict"]
    try:
        log_file_artifact(run, path=Path(report_path), name="deltastock_measure_report",
                          artifact_type="deltastock-measure-report")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] artifact log failed (non-fatal): {e!r}", flush=True)
    rid = run.id
    finish_wandb(run)
    print(f"[wandb] logged {wandb_name} id={rid} (group={wandb_group})", flush=True)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="path to delta_stock_measure report.json")
    ap.add_argument("--no-ood-carryover", action="store_true",
                    help="do NOT inject the #739 chat/code/multilingual corners into "
                         "the all-shifted sensitivity readings (default: inject)")
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name",
                    default="kanna/g1-faithful-tighten")
    args = ap.parse_args()
    r = json.loads(Path(args.report).read_text())
    d = dict(r["delta_stock_pct_by_eaccept"])  # name -> delta% (public=0)
    e_pub = r.get("e_accept_public")

    d_know = d.get("knowledge_mmlupro")
    d_gpqa = d.get("gpqa_diamond")
    d_math = d.get("reasoning_math")
    if d_know is None or d_gpqa is None or d_math is None:
        raise SystemExit("report missing one of knowledge_mmlupro / gpqa_diamond / "
                         "reasoning_math -- cannot assemble the faithful 57/57/14 mix")

    # --- the two centrals -------------------------------------------------------
    faithful_central = round(W_MMLU_F * d_know + W_GPQA_F * d_gpqa + W_MATH_F * d_math, 3)
    optimistic_739_central = round(W_MCQ_739 * d_know + W_MATH_739 * d_math, 3)
    gpqa_correction_shift = round(faithful_central - optimistic_739_central, 3)
    # equivalently: the GPQA-D half replaces MMLU-Pro for 57/128 of the weight:
    #   shift = (57/128) * (d_gpqa - d_know)
    gpqa_minus_mmlu = round(d_gpqa - d_know, 3)

    # Math-weight breakeven for the faithful MCQ blend: holding the MCQ slot at its
    # 50/50 MMLU/GPQA faithful split, what free-response-math fraction f pushes the
    # blend to the 5% line?  blend(f) = (1-f)*d_mcq + f*d_math.
    d_mcq_blend = 0.5 * d_know + 0.5 * d_gpqa
    breakeven_math_frac = (round((G1 - d_mcq_blend) / (d_math - d_mcq_blend), 4)
                           if d_math != d_mcq_blend else None)
    # GPQA-share breakeven: holding math at 14/128, how much of the 114-MCQ block
    # would have to be GPQA-D (vs MMLU-Pro) to push the central to 5%?
    # central(g) = W_MATH_F*d_math + (114/128)*((1-g)*d_know + g*d_gpqa) = G1
    mcq_w = 114 / 128
    rhs = G1 - W_MATH_F * d_math
    base = mcq_w * d_know
    breakeven_gpqa_share = (round((rhs - base) / (mcq_w * (d_gpqa - d_know)), 4)
                            if d_gpqa != d_know else None)

    # --- sensitivity band: shifted-subset distribution --------------------------
    shifted = {k: v for k, v in d.items() if k != "public"}
    if not args.no_ood_carryover and e_pub:
        for k, e in OOD_739_EACCEPT.items():
            shifted.setdefault(k, round((1 - e / e_pub) * 100, 3))
    same_fam = {k: d[k] for k in SAME_FAMILY if k in d}

    emp_all = (round(sum(1 for v in shifted.values() if v > G1) / len(shifted), 3)
               if shifted else None)
    sh_vals = list(shifted.values())
    p_pessimistic = round(gauss_tail(statistics.mean(sh_vals),
                          statistics.pstdev(sh_vals) if len(sh_vals) > 1 else 0.0), 3) if sh_vals else None
    sf_vals = list(same_fam.values())
    p_equalweight_family = round(gauss_tail(statistics.mean(sf_vals),
                                 statistics.pstdev(sf_vals) if len(sf_vals) > 1 else 0.0), 3) if sf_vals else None

    # --- calibrated headline: bracket faithful_central against real-private #52 --
    # #52 (+4.3%, VALID) disagrees in SIGN with the proxy, so proxy representativeness
    # is the dominant uncertainty. Model delta_stock ~ Normal spanning the bracket:
    #   mu = midpoint(faithful_central, #52);  sigma = half-separation.
    cal_mu = round((faithful_central + ANCHOR_52) / 2, 3)
    cal_sigma = round(abs(ANCHOR_52 - faithful_central) / 2, 3)
    p_dq_calibrated = round(gauss_tail(cal_mu, cal_sigma), 3)

    p_dq = p_dq_calibrated  # primary metric
    central = faithful_central

    margin = round(G1 - central, 3)
    clears = central < G1
    # #749 verdicts
    faithful_safe = (central <= 2.0 + 1e-9) and (p_dq < 0.13)
    if faithful_safe:
        verdict_str = "FAITHFUL_G1_SAFE"
    else:
        verdict_str = "FAITHFUL_G1_KNIFE_EDGE"

    verdict = {
        "e_accept_public": e_pub,
        "per_subset_delta_pct": d,
        "centrals": {
            "faithful_57_57_14_headline": faithful_central,
            "optimistic_739_114_14": optimistic_739_central,
            "gpqa_correction_shift": gpqa_correction_shift,
            "gpqa_minus_mmlu_corner_gap": gpqa_minus_mmlu,
            "faithful_weights": {"mmlu_pro": W_MMLU_F, "gpqa_diamond": W_GPQA_F,
                                 "free_response_math": W_MATH_F},
        },
        "corners": {
            "gpqa_diamond_delta_pct": d_gpqa,
            "mmlu_pro_delta_pct": d_know,
            "reasoning_math_delta_pct": d_math,
            "breakeven_private_math_fraction_to_5pct": breakeven_math_frac,
            "breakeven_gpqa_share_of_mcq_to_5pct": breakeven_gpqa_share,
        },
        "anchors": {"flagship_52_real_private_pct": ANCHOR_52,
                    "kanna_44_chat_upper_pct": ANCHOR_44, "g1_threshold_pct": G1},
        "p_dq_readings": {
            "calibrated_headline": p_dq_calibrated,
            "calibrated_model": {"mu": cal_mu, "sigma": cal_sigma,
                                 "bracket": [faithful_central, ANCHOR_52]},
            "empirical_frac_gt5_all_shifted": emp_all,
            "gaussian_all_shifted_pessimistic": p_pessimistic,
            "gaussian_equal_weight_family_artifact": p_equalweight_family,
            "ood_carryover_used": (not args.no_ood_carryover),
            "shifted_subsets_used": shifted,
        },
        "decision": {
            "test_metric_delta_stock_faithful_central_pct": central,
            "primary_metric_p_dq_g1_faithful_mix": p_dq,
            "margin_to_5pct": margin,
            "clears_g1_central": clears,
            "faithful_g1_safe_le2_and_pdq_lt_0p13": faithful_safe,
            "verdict": verdict_str,
        },
    }
    print(json.dumps(verdict, indent=2))

    run_id = None
    if args.wandb_group:
        try:
            run_id = log_to_wandb(verdict, r, wandb_group=args.wandb_group,
                                  wandb_name=args.wandb_name, report_path=args.report)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] logging failed (non-fatal): {e!r}", flush=True)

    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [run_id] if run_id else [],
        "primary_metric": {"name": "p_dq_g1_faithful_mix", "value": p_dq},
        "test_metric": {"name": "delta_stock_faithful_central_pct", "value": central},
    }
    print("\nSENPAI-RESULT:")
    print(json.dumps(senpai_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
