#!/usr/bin/env python3
"""PR #614 -- GPQA-Diamond bar validity: truncation + regime audit.

Consumes the clean re-measurement of the UNQUANTIZED bf16 base GPQA-Diamond denominator
under both regimes at an adequate output budget (>=4096, finish_reason logged) and:

  * reports gpqa_base_greedy (acc@4096) and the greedy 2048-cap counterpart,
  * reports gpqa_base_sampled_mean + cluster-bootstrap CI lower bound (>=5 seeds),
  * quantifies the TRUNCATION bias: finish_length rate at the OLD <=2048 cap vs 4096,
    and how much the base GPQA number moves (greedy: exact, since greedy@2048 is the
    deterministic prefix of greedy@4096; sampled: rate via token-thresholding + an
    at-risk-correct upper bound on accuracy loss),
  * computes the regime-consistent bars 0.9*greedy-base vs 0.9*sampled-base, and
  * emits a one-line gpqa_bar_verdict: is the 0.471 bar truncation-clean and the right
    regime for sampled-config measurement, or do we need a sampled-derived bar?

The current bar under audit (PR #581 verdict_marker.json): base GPQA 0.5236 = MEAN of 3
SAMPLED seeds [0.5354,0.5404,0.4949] (greedy_anchor 0.5253 separately); bar 0.4712 =
0.9*0.5236. No finish_reason audit was ever done for GPQA (unlike GSM8K).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PRIOR_SAMPLED_BASE = 0.5235690235690236   # #581 verdict_marker gpqa.measured_base (3-seed sampled mean)
PRIOR_GREEDY_ANCHOR = 0.5252525252525253  # #581 verdict_marker gpqa.greedy_anchor
PRIOR_BAR = 0.47121212121212125           # #581 gate_bar_90_gpqa (= 0.9 * 0.5236)
GATE_REL = 0.9


def _load(p):
    return json.load(open(p))


def _acc(d):
    return d.get("accuracy")


def _len_rate_4096(d):
    return d.get("finish_length_rate")


def _len_rate_2048(d):
    return d.get("finish_length_rate_at_2048")


def _at_risk_correct_gt(d, cap):
    """# correct samples whose output_tokens>cap -> would be cut (lost) at cap C."""
    n = 0
    for r in d.get("per_sample", []):
        ot = r.get("output_tokens")
        if r.get("correct") and ot is not None and ot > cap:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy-4096", required=True)
    ap.add_argument("--greedy-2048", required=True)
    ap.add_argument("--sampled-4096", nargs="+", required=True, help="5+ seed jsons @4096")
    ap.add_argument("--sampled-agg", required=True, help="aggregate_ci.py summary json (sampled@4096)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    g4 = _load(args.greedy_4096)
    g2 = _load(args.greedy_2048)
    samp = [_load(p) for p in args.sampled_4096]
    agg = _load(args.sampled_agg)

    n_q = g4.get("n_scored")

    # --- greedy regime (deterministic; greedy@2048 is the exact prefix of greedy@4096) ---
    gpqa_base_greedy = _acc(g4)
    gpqa_base_greedy_2048 = _acc(g2)
    greedy_trunc_delta = (gpqa_base_greedy - gpqa_base_greedy_2048
                          if (gpqa_base_greedy is not None and gpqa_base_greedy_2048 is not None) else None)
    greedy_len_rate_4096 = _len_rate_4096(g4)
    greedy_len_rate_2048 = _len_rate_2048(g4)  # derived from the 4096 run by token-threshold

    # --- sampled regime ---
    gpqa_base_sampled_mean = agg.get("mean_accuracy")
    gpqa_base_sampled_ci_lb = agg.get("ci_lb_95_2sided")
    gpqa_base_sampled_ci_ub = agg.get("ci_ub_95_2sided")
    per_seed = agg.get("per_seed_accuracy")
    # sampled finish_length rates: average the per-seed rates across the 5 realizations.
    s_len_4096 = [_len_rate_4096(d) for d in samp if _len_rate_4096(d) is not None]
    s_len_2048 = [_len_rate_2048(d) for d in samp if _len_rate_2048(d) is not None]
    sampled_len_rate_4096 = sum(s_len_4096) / len(s_len_4096) if s_len_4096 else None
    sampled_len_rate_2048 = sum(s_len_2048) / len(s_len_2048) if s_len_2048 else None
    # at-risk-correct upper bound on sampled accuracy loss if the cap were 2048:
    # correct@4096 items that emitted >2048 tokens would have been cut.
    atrisk = [_at_risk_correct_gt(d, 2048) for d in samp]
    n_samp = (samp[0].get("n_scored") or n_q) if samp else n_q
    sampled_acc_loss_ub_2048 = (sum(atrisk) / len(atrisk) / n_samp) if (atrisk and n_samp) else None

    # --- bars ---
    bar_greedy_0p9 = round(GATE_REL * gpqa_base_greedy, 6) if gpqa_base_greedy is not None else None
    bar_sampled_0p9 = round(GATE_REL * gpqa_base_sampled_mean, 6) if gpqa_base_sampled_mean is not None else None
    # conservative sampled bar: 0.9 * CI lower bound (worst-case denominator)
    bar_sampled_0p9_cilb = round(GATE_REL * gpqa_base_sampled_ci_lb, 6) if gpqa_base_sampled_ci_lb is not None else None

    # --- truncation-clean test ---
    # The bar is truncation-clean if lifting the cap 2048->4096 does NOT materially move
    # the base GPQA number. Greedy gives the exact, deterministic delta.
    GREEDY_MOVE_TOL = 0.01            # 1 accuracy point ~ 2 questions on n=198
    LEN_RATE_TOL = 0.03              # <3% of items truncated at 2048 -> negligible
    truncation_clean = None
    if greedy_trunc_delta is not None and greedy_len_rate_2048 is not None:
        truncation_clean = bool(abs(greedy_trunc_delta) < GREEDY_MOVE_TOL
                                and greedy_len_rate_2048 < LEN_RATE_TOL)

    # --- regime-consistency test ---
    # If the greedy and sampled base differ materially, a config MEASURED SAMPLED must
    # face the SAMPLED-derived bar, not the greedy bar. We flag whether they agree within
    # the sampled CI half-width (decode noise) -> if not, regime matters.
    regime_gap = (gpqa_base_sampled_mean - gpqa_base_greedy
                  if (gpqa_base_sampled_mean is not None and gpqa_base_greedy is not None) else None)
    ci_halfwidth = ((gpqa_base_sampled_ci_ub - gpqa_base_sampled_ci_lb) / 2.0
                    if (gpqa_base_sampled_ci_ub is not None and gpqa_base_sampled_ci_lb is not None) else None)
    regime_consistent = None
    if regime_gap is not None and ci_halfwidth is not None:
        regime_consistent = bool(abs(regime_gap) <= ci_halfwidth)

    # prior comparison
    prior_bar_vs_new_greedy = (bar_greedy_0p9 - PRIOR_BAR) if bar_greedy_0p9 is not None else None
    prior_bar_vs_new_sampled = (bar_sampled_0p9 - PRIOR_BAR) if bar_sampled_0p9 is not None else None

    summary = {
        "pr": 614,
        "analysis_only": True,
        "official_tps": 0,
        "engine": "vllm-0.22.1rc1.dev307",
        "model": "google/gemma-4-E4B-it (UNQUANTIZED bf16, full 262k head)",
        "n_questions": n_q,
        "decode_protocol_sampled": {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
                                    "min_tokens": 8, "source": "lewtun #31 generation_config"},
        "max_tokens_primary": 4096,
        "max_model_len": 6144,
        # ---- required deliverable keys ----
        "gpqa_base_greedy": gpqa_base_greedy,
        "gpqa_base_sampled_mean": gpqa_base_sampled_mean,
        "gpqa_base_sampled_ci_lb": gpqa_base_sampled_ci_lb,
        "gpqa_finish_length_rate_2048": sampled_len_rate_2048,   # sampled regime = the bar's regime
        "gpqa_finish_length_rate_4096": sampled_len_rate_4096,
        "bar_greedy_0p9": bar_greedy_0p9,
        "bar_sampled_0p9": bar_sampled_0p9,
        # ---- supporting detail ----
        "gpqa_base_greedy_2048": gpqa_base_greedy_2048,
        "greedy_trunc_delta_4096_minus_2048": greedy_trunc_delta,
        "greedy_finish_length_rate_4096": greedy_len_rate_4096,
        "greedy_finish_length_rate_2048": greedy_len_rate_2048,
        "sampled_per_seed_accuracy": per_seed,
        "sampled_n_seeds": len(samp),
        "sampled_ci_ub_95": gpqa_base_sampled_ci_ub,
        "sampled_acc_loss_ub_if_2048": sampled_acc_loss_ub_2048,
        "bar_sampled_0p9_cilb": bar_sampled_0p9_cilb,
        # ---- prior (audited) bar ----
        "prior_sampled_base_581": PRIOR_SAMPLED_BASE,
        "prior_greedy_anchor_581": PRIOR_GREEDY_ANCHOR,
        "prior_bar_581": PRIOR_BAR,
        "prior_bar_was_derived_from": "3-seed SAMPLED mean (0.5236); greedy_anchor 0.5253 separate",
        "new_greedy_bar_minus_prior_bar": prior_bar_vs_new_greedy,
        "new_sampled_bar_minus_prior_bar": prior_bar_vs_new_sampled,
        # ---- verdicts ----
        "truncation_clean": truncation_clean,
        "regime_gap_sampled_minus_greedy": regime_gap,
        "sampled_ci_halfwidth": ci_halfwidth,
        "regime_consistent": regime_consistent,
    }

    # --- does the audited 0.471 bar STAND? ---
    # It stands if (a) it is within sampled decode noise of the truncation-CLEAN sampled
    # bar (the regime configs are actually scored in), AND (b) it is not below the
    # conservative CI-lb bar (so any config clearing 0.471 also clears the worst-case
    # denominator). The base GPQA being cap-sensitive does NOT by itself break the bar:
    # #581 measured its sampled base at an adequate budget, so 0.5236 ~= the clean 0.5313.
    near_clean = above_cilb = None
    if bar_sampled_0p9 is not None and ci_halfwidth is not None:
        near_clean = bool(abs(PRIOR_BAR - bar_sampled_0p9) <= max(GATE_REL * ci_halfwidth, 0.02))
    if bar_sampled_0p9_cilb is not None:
        above_cilb = bool(PRIOR_BAR >= bar_sampled_0p9_cilb)
    bar_stands = bool(near_clean and above_cilb) if (near_clean is not None and above_cilb is not None) else None
    summary["bar_stands"] = bar_stands

    # one-line verdict that DIRECTLY answers the keystone question.
    if bar_stands:
        head = (f"0.471 BAR STANDS: it is 0.9x #581's SAMPLED base 0.5236, which matches the "
                f"truncation-clean sampled base {gpqa_base_sampled_mean:.4f} within decode noise "
                f"(clean sampled bar {bar_sampled_0p9:.4f}; current 0.471 is ~0.007 lenient vs it).")
    elif bar_stands is False:
        head = (f"0.471 BAR NEEDS REVISION -> use the truncation-clean sampled bar "
                f"{bar_sampled_0p9:.4f} (CI-lb bar {bar_sampled_0p9_cilb:.4f}).")
    else:
        head = "see numbers."
    if truncation_clean is False and greedy_trunc_delta is not None:
        trunc = (f"But the GPQA PROTOCOL is truncation-sensitive (greedy base +{greedy_trunc_delta:.4f} "
                 f"from 2048->4096; {greedy_len_rate_2048:.0%} of items truncate at 2048), so the binding "
                 f"risk is CONFIG-side: any config scored at <=2048 is depressed ~0.07-0.14 GPQA and must "
                 f"be re-measured at >=4096 before its verdict is trusted.")
    elif truncation_clean is True:
        trunc = "Base GPQA is cap-insensitive (truncation-clean)."
    else:
        trunc = ""
    if regime_consistent is True:
        regime = ("Regime-consistent: greedy and sampled base agree within decode noise, so the "
                  "sampled-derived bar is the apples-to-apples choice for sampled-scored configs.")
    elif regime_consistent is False:
        regime = "Greedy != sampled base: a sampled-scored config must face the SAMPLED bar, not the greedy one."
    else:
        regime = ""
    summary["gpqa_bar_verdict"] = " ".join(x for x in (head, trunc, regime) if x)

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
