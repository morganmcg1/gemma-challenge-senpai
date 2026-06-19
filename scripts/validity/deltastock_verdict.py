#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Turn a delta_stock_measure report.json into the #739 G1 fire-gate verdict.

The measured per-subset delta_stock distribution BRACKETS the real private gap:

  same-family held-out (knowledge_mmlupro + reasoning_math)  -> LOWER bracket
      (fresh instances of the SAME MMLU-Pro/GPQA/AIME benchmark suite the public
       set is drawn from; the most faithful realization of "private = held-out
       instances of the same benchmark")
  pure-chat / code / multilingual                            -> UPPER bracket
      (out-of-distribution worst corners; pure-chat reconciles with kanna #44's
       12.4% chat-proxy upper bound)

The strongest external anchor is flagship #52: 4.3% measured on the ACTUAL private
set (VALID, i.e. it passed the 5% rule). The real private delta_stock sits inside
the bracket, near #52's 4.3%.

primary_metric = p_dq_g1_at_measured_delta_stock = P(delta_stock > 5%)  (lower=better)
test_metric    = delta_stock_central_pct  (same-family held-out central)
"""
from __future__ import annotations

import argparse
import json
import statistics
from math import erf, sqrt
from pathlib import Path

SAME_FAMILY = ["knowledge_mmlupro", "reasoning_math"]
ANCHOR_52 = 4.3   # flagship #52, measured on REAL private, VALID
ANCHOR_44 = 12.4  # kanna #44 chat-proxy, pessimistic upper bound
G1 = 5.0
# Public benchmark composition (confirmed from official EVAL_PROMPTS templates):
# 114 MCQ-knowledge (MMLU-Pro + GPQA, "ANSWER: $LETTER") + 14 free-response math
# ("Solve ... ANSWER: $ANSWER"). The private re-run most plausibly draws held-out
# instances of the SAME benchmark, so the faithful central is the composition-
# weighted blend of the two template-exact same-family proxies, NOT the equal-
# weight family mean (which inflates the 11% math corner to 50%).
W_MCQ, W_MATH = 114 / 128, 14 / 128


def gauss_tail(mu: float, sigma: float, thr: float = G1) -> float:
    if sigma <= 0:
        return 1.0 if mu > thr else 0.0
    return 0.5 * (1 - erf((thr - mu) / sigma / sqrt(2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="path to delta_stock_measure report.json")
    args = ap.parse_args()
    r = json.loads(Path(args.report).read_text())
    d = r["delta_stock_pct_by_eaccept"]  # name -> delta% (public=0)

    shifted = {k: v for k, v in d.items() if k != "public"}
    same_fam = {k: d[k] for k in SAME_FAMILY if k in d}
    ood = {k: v for k, v in shifted.items() if k not in SAME_FAMILY}

    # Composition-weighted central (headline): faithful to the 114/14 public mix.
    d_know = d.get("knowledge_mmlupro")
    d_math = d.get("reasoning_math")
    comp_central = None
    breakeven_math_frac = None
    if d_know is not None and d_math is not None:
        comp_central = round(W_MCQ * d_know + W_MATH * d_math, 3)
        # delta(f) = d_know + f*(d_math - d_know); solve delta(f)=G1 for the private
        # free-response-math fraction f that would push delta_stock to the 5% line.
        if d_math != d_know:
            breakeven_math_frac = round((G1 - d_know) / (d_math - d_know), 4)

    central = comp_central  # headline test_metric: composition-weighted
    equal_weight_family_mean = round(statistics.mean(same_fam.values()), 3) if same_fam else None
    all_shift_mean = round(statistics.mean(shifted.values()), 3) if shifted else None
    worst = round(max(shifted.values()), 3) if shifted else None
    best = round(min(shifted.values()), 3) if shifted else None

    # P(delta>5%) readings
    # primary (assumption-light): fraction of SAMPLED distribution-shifts breaching 5%.
    emp_all = round(sum(1 for v in shifted.values() if v > G1) / len(shifted), 3) if shifted else None
    # sensitivity: Gaussian over the equal-weight same-family pair (knowledge,math).
    # This inflates the 11% math corner to 50%, so it reads ~coin-flip; it is the
    # PESSIMISTIC artifact, NOT the faithful private case.
    sf_vals = list(same_fam.values())
    p_equalweight_family = round(gauss_tail(statistics.mean(sf_vals),
                                            statistics.pstdev(sf_vals) if len(sf_vals) > 1 else 0.0), 3) if sf_vals else None
    # sensitivity: Gaussian over ALL shifted subsets (uniform-over-distributions, incl OOD).
    sh_vals = list(shifted.values())
    p_pessimistic = round(gauss_tail(statistics.mean(sh_vals),
                                     statistics.pstdev(sh_vals) if len(sh_vals) > 1 else 0.0), 3) if sh_vals else None

    # Calibrated P(DQ): the realized private delta_stock is BRACKETED by my
    # composition-faithful proxy (comp_central, optimistic: a net speedup, but it
    # assumes the private mix == public mix and that my held-out proxies are
    # representative) and the single REAL-private anchor, flagship #52's +4.3%
    # (VALID). #52 disagrees in SIGN with my proxy, so proxy representativeness is
    # the dominant uncertainty. Model delta_stock ~ Normal spanning that bracket:
    #   mu = midpoint(comp_central, #52);  sigma = half-separation.
    # P(delta>5%) is then the principled headline that uses BOTH my measurement and
    # the strongest external evidence, instead of asserting ~0 from the proxy alone.
    p_dq_calibrated = None
    cal_mu = cal_sigma = None
    if comp_central is not None:
        cal_mu = round((comp_central + ANCHOR_52) / 2, 3)
        cal_sigma = round(abs(ANCHOR_52 - comp_central) / 2, 3)
        p_dq_calibrated = round(gauss_tail(cal_mu, cal_sigma), 3)

    # Primary metric = calibrated P(DQ). emp_all (fraction of sampled shifts >5%)
    # and the gaussians are reported as the transparent sensitivity band.
    p_dq = p_dq_calibrated if p_dq_calibrated is not None else emp_all

    margin = round(G1 - central, 3) if central is not None else None
    comfortable = central is not None and central < 3.5
    clears = central is not None and central < G1

    verdict = {
        "e_accept_public": r.get("e_accept_public"),
        "per_subset_delta_pct": d,
        "centrals": {
            "composition_weighted_headline": comp_central,
            "equal_weight_same_family": equal_weight_family_mean,
            "all_shifted_mean": all_shift_mean,
            "public_composition": {"mcq_knowledge": W_MCQ, "math_freeform": W_MATH},
        },
        "bracket": {
            "same_family_held_out": same_fam, "ood_corners": ood,
            "best_corner": best, "upper_bracket_worst_corner": worst,
            "breakeven_private_math_fraction_to_5pct": breakeven_math_frac,
        },
        "anchors": {"flagship_52_real_private_pct": ANCHOR_52,
                    "kanna_44_chat_upper_pct": ANCHOR_44, "g1_threshold_pct": G1},
        "p_dq_readings": {
            "calibrated_headline": p_dq_calibrated,
            "calibrated_model": {"mu": cal_mu, "sigma": cal_sigma,
                                 "bracket": [comp_central, ANCHOR_52]},
            "empirical_frac_gt5_all_shifted": emp_all,
            "gaussian_equal_weight_family_artifact": p_equalweight_family,
            "gaussian_all_shifted_pessimistic": p_pessimistic,
        },
        "decision": {
            "test_metric_delta_stock_central_pct": central,
            "primary_metric_p_dq_g1": p_dq,
            "margin_to_5pct": margin,
            "clears_g1_central": clears,
            "comfortable_margin_lt_3p5": comfortable,
            "recommendation": (
                "FIRE" if (clears and comfortable) else
                "FIRE-MARGINAL" if clears else "HOLD"),
        },
    }
    print(json.dumps(verdict, indent=2))

    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [],  # fill from wandb after logging
        "primary_metric": {"name": "p_dq_g1_at_measured_delta_stock", "value": p_dq},
        "test_metric": {"name": "delta_stock_central_pct", "value": central},
    }
    print("\nSENPAI-RESULT (fill wandb_run_ids):")
    print(json.dumps(senpai_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
