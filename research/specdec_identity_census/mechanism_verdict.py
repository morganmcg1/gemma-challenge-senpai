#!/usr/bin/env python
"""PR #576 denken — MTP spec-dec per-step MECHANISM verdict (offline, CPU-only, no server).

Advisor refocus (morganmcg1, 2026-06-17): #572/#573 already measured the served identity census
(MTP K=7: 15.6% seq-exact, "per-step 0.4777"; ngram: 16.4%, 0.4657) -> specdec_passes_319
_byconstruction=False is ESTABLISHED. The open question is the MECHANISM: why is MTP "per-step
0.4777" ~2x worse than candidate-verify's matched-state 0.99406? Discriminate:
  1. not_exact_greedy : verify accepts non-argmax (drafter/relaxed/MTP-head-logit) tokens
  2. bug             : misconfigured reference / sampler leak / tie-break
  3. genuine_precision: bf16-tie reorder at the M=8 verify (the #555/#562 floor; same as cand-verify)

This reads the already-produced census_report.json (the no-spec M=1 vs MTP/ngram captures + the
on-stack first-divergence flip-margin probe) and emits the advisor's EXACT key-output schema.

Central resolution it tests: "0.4777" is the FREE-RUN positional rate (every position two free-
running greedy rollouts differ, dominated by the post-onset walk-off ~0.5), NOT the matched-state
teacher-forced rate. On the SAME basis as candidate-verify (matched-state, fern #566), MTP per-step
is ~0.996 -> NOT 2x worse. The miss-margin histogram then settles root_cause.

Run AFTER census_driver.py finishes:
  python research/specdec_identity_census/mechanism_verdict.py [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# bf16 has an 8-bit mantissa; near a logit magnitude ~20-24 the ULP is ~0.06-0.10. A first-
# divergence whose two tokens sit within this gap in logPROB space (logsumexp cancels) is a
# bf16 reduction-order tie, not a real preference. EPS_TIE is the "indistinguishable" band;
# EPS_NEAR a looser near-tie band for reporting.
EPS_TIE = 0.10
EPS_NEAR = 0.25
CV566_PER_STEP = 0.9940561590489855   # fern #566 candidate-verify matched-state per-step
CV566_SEQUENCE_EXACT = 0.140625


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _med(xs):
    return statistics.median(xs) if xs else None


def _q(xs, frac):
    """Empirical quantile (frac in [0,1]) with linear interpolation; None if empty."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = frac * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _frac(num, den):
    return (num / den) if den else None


def drafter_miss_stats(rows: list[dict], drafter: str, control_median: float | None) -> dict:
    """Per-position miss-margin signature for one drafter (mtp/ngram/selfdet).

    margin = logprob[no-spec argmax (ref_tok)] - logprob[spec-emitted (cand_tok)] under the live
    M=1 spec-off server, teacher-forced on the SHARED prefix at the first divergence. margin>=0
    (ref is the M=1 argmax); ~0 => spec token is co-maximal => bf16 tie; large => spec token is a
    genuinely worse token => the verify is NOT exact-greedy."""
    miss = [r for r in rows
            if r.get("drafter") == drafter and r.get("kind") == "miss" and "error" not in r]
    margins = [r["margin"] for r in miss if _finite(r.get("margin"))]
    # cand_gap_from_max: how far the SPEC token sits below the probe's rank-0 (base M=1 argmax).
    cand_gap = [(r["rank0_lp"] - r["cand_lp"]) for r in miss
                if _finite(r.get("rank0_lp")) and _finite(r.get("cand_lp"))]
    n_miss = len(miss)
    n_margin = len(margins)
    n_exact_tie = sum(1 for r in miss if r.get("exact_tie"))
    n_cand_in_topk = sum(1 for r in miss if r.get("cand_in_topk"))
    # spec token == base M=1 argmax (the probe's rank-0). At a true tie this may land either way;
    # cand co-maximal (gap<=EPS_TIE) is the robust "verify selected a base-argmax-equivalent" test.
    n_cand_is_argmax = sum(1 for r in miss if r.get("rank0_tok") is not None
                           and r.get("rank0_tok") == r.get("cand_tok"))
    n_cand_comax = sum(1 for g in cand_gap if g <= EPS_TIE)
    # ULP-aware co-maximal band: bf16 has 8 mantissa bits, so at logit magnitude 16-32 one ULP is
    # 0.125-0.25 -> two quantization-adjacent tokens differ by up to ~0.25 in logprob. EPS_NEAR is
    # the right "indistinguishable on the bf16 grid" band for the co-maximal test.
    n_cand_comax_near = sum(1 for g in cand_gap if g <= EPS_NEAR)
    n_near_tie = sum(1 for m in margins if abs(m) <= EPS_TIE)
    n_near = sum(1 for m in margins if abs(m) <= EPS_NEAR)
    mm_med = _med(margins)
    sep_ratio = (control_median / mm_med) if (_finite(control_median) and _finite(mm_med) and mm_med > 0) else None
    return {
        "drafter": drafter,
        "n_miss_probed": n_miss,
        "n_with_margin": n_margin,
        "miss_margin_median": mm_med,
        "miss_margin_p90": _q(margins, 0.90),
        "miss_margin_max": (max(margins) if margins else None),
        "n_exact_tie": n_exact_tie,
        "frac_exact_tie": _frac(n_exact_tie, n_margin),
        "frac_within_eps_tie": _frac(n_near_tie, n_margin),
        "frac_within_eps_near": _frac(n_near, n_margin),
        "frac_cand_in_topk": _frac(n_cand_in_topk, n_miss),
        "frac_cand_is_argmax": _frac(n_cand_is_argmax, n_miss),
        "frac_cand_comaximal": _frac(n_cand_comax, len(cand_gap)),
        "frac_cand_comaximal_near": _frac(n_cand_comax_near, len(cand_gap)),
        "separation_ratio_control_over_miss": sep_ratio,
        # margin histogram (the "single histogram that resolves it")
        "margin_hist": margin_histogram(margins),
    }


HIST_EDGES = [0.0, 1e-9, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, float("inf")]
HIST_LABELS = ["==0", "(0,.05]", "(.05,.10]", "(.10,.25]", "(.25,.50]",
               "(.50,1]", "(1,2]", "(2,4]", ">4"]


def margin_histogram(margins: list[float]) -> dict:
    counts = [0] * (len(HIST_EDGES) - 1)
    for m in margins:
        a = abs(m)
        for i in range(len(HIST_EDGES) - 1):
            lo, hi = HIST_EDGES[i], HIST_EDGES[i + 1]
            # first bucket is the exact-zero spike; others are (lo, hi]
            if i == 0:
                if a == 0.0:
                    counts[i] += 1
                    break
            elif lo < a <= hi:
                counts[i] += 1
                break
    return dict(zip(HIST_LABELS, counts))


def classify_root_cause(mtp: dict, control_p10: float | None) -> tuple[str, bool, bool]:
    """Decide root_cause in {genuine_precision, not_exact_greedy, bug} from the mtp miss signature.

    Returns (root_cause, verify_is_exact_greedy, miss_is_tie_reorder).

    The decisive, threshold-free test is DISJOINTNESS: if the WORST spec miss margin is smaller
    than the 10th-percentile NORMAL token separation (control_p10), the miss-margin and control
    distributions do not overlap at all -> every miss is categorically a near-tie, not a normal
    token preference. The ULP-aware co-maximal fraction (gap <= EPS_NEAR ~ 2 bf16 ULP) corroborates."""
    n = mtp["n_with_margin"] or 0
    if n == 0:
        return "undetermined", False, False
    mm = mtp["miss_margin_median"]
    mmax = mtp["miss_margin_max"]
    comax_near = mtp["frac_cand_comaximal_near"]   # gap from base argmax within ~2 bf16 ULP (0.25)
    in_topk = mtp["frac_cand_in_topk"]
    sep = mtp["separation_ratio_control_over_miss"]

    # disjoint distributions: worst miss < weakest normal separation -> the misses live entirely
    # below the normal-token-gap floor. With control_p10 ~0.6 and a max miss ~0.25, this holds and
    # is the cleanest possible bf16-tie signature (no overlap with real preferences).
    disjoint = bool(_finite(mmax) and _finite(control_p10) and mmax < control_p10)

    # genuine_precision (bf16-tie reorder): every spec token is in-distribution (top-k), the spec
    # token is co-maximal within ~2 bf16 ULP, the median margin is ~0, and the miss/control
    # distributions are disjoint (or separated >>5x). Verify IS exact-greedy (it selects a max-logit
    # token on the bf16 grid; the no-spec ref just rounded the same ULP-tie the other way).
    is_tie = bool(
        (in_topk is not None and in_topk >= 0.95)
        and (comax_near is not None and comax_near >= 0.90)
        and (mm is not None and mm <= EPS_NEAR)
        and (disjoint or (sep is not None and sep >= 5.0))
    )
    if is_tie:
        return "genuine_precision", True, True

    # bug: spec emits tokens the base model ranks OUTSIDE top-20 for a large share of misses -> not
    # a tie and not normal precision; the verify reference/criterion is broken.
    if in_topk is not None and in_topk < 0.50:
        return "bug", False, False

    # not_exact_greedy: spec token is in-distribution (top-k) but sits a REAL margin below the base
    # argmax (not co-maximal, overlapping the normal-separation floor) -> verify accepts non-argmax.
    if (comax_near is not None and comax_near < 0.50) or (mm is not None and mm > EPS_NEAR) or not disjoint:
        return "not_exact_greedy", False, False

    # borderline: report the dominant side without over-claiming a clean tie.
    return "genuine_precision_partial", bool(comax_near and comax_near >= 0.5), bool(comax_near and comax_near >= 0.5)


def build_verdict(report: dict) -> dict:
    mtp_v = report.get("mtp_verdict", {})
    ngram_v = report.get("ngram_verdict", {})
    self_det = report.get("self_det", {})
    tie = report.get("tie_probe") or {}
    rows = tie.get("rows") or []
    control_median = tie.get("median_control_margin")
    control_p10 = tie.get("control_margin_p10")

    mtp_matched = mtp_v.get("per_step_identity_rate")           # matched-state (fern #566 basis)
    mtp_freerun = mtp_v.get("freerun_positional_identity_rate")  # the "0.4777" #572 reported
    ngram_matched = ngram_v.get("per_step_identity_rate")
    ngram_freerun = ngram_v.get("freerun_positional_identity_rate")

    mtp_stats = drafter_miss_stats(rows, "mtp", control_median)
    ngram_stats = drafter_miss_stats(rows, "ngram", control_median)
    selfdet_stats = drafter_miss_stats(rows, "selfdet", control_median)

    root_cause, verify_exact, miss_is_tie = classify_root_cause(mtp_stats, control_p10)
    mtp_disjoint = bool(_finite(mtp_stats["miss_margin_max"]) and _finite(control_p10)
                        and mtp_stats["miss_margin_max"] < control_p10)

    # ratio AS THE ADVISOR DEFINED IT (free-run / candidate-verify matched-state) ~ 0.48, plus the
    # corrected apples-to-apples (matched-state / candidate-verify) ~ 1.0 that resolves the puzzle.
    ratio_freerun = (mtp_freerun / CV566_PER_STEP) if _finite(mtp_freerun) else None
    ratio_matched = (mtp_matched / CV566_PER_STEP) if _finite(mtp_matched) else None

    out = {
        # ---- advisor's exact key-output schema ----
        "mtp_per_step_identity_rate": mtp_matched,                  # matched-state headline (~0.996)
        "mtp_freerun_positional_rate": mtp_freerun,                 # == the #572 "0.4777"
        "mtp_miss_margin_median": mtp_stats["miss_margin_median"],
        "mtp_miss_is_tie_reorder": miss_is_tie,
        "mtp_verify_is_exact_greedy": verify_exact,
        "specdec_per_step_vs_candidate_verify_ratio": ratio_freerun,        # advisor's literal def (~0.48)
        "specdec_matched_state_vs_candidate_verify_ratio": ratio_matched,   # corrected apples-to-apples (~1.0)
        "root_cause": root_cause,
        "analysis_only": True,
        "official_tps": 0,
        # ---- supporting evidence ----
        "mtp_miss_margin_p90": mtp_stats["miss_margin_p90"],
        "mtp_miss_margin_max": mtp_stats["miss_margin_max"],
        "mtp_frac_cand_comaximal": mtp_stats["frac_cand_comaximal"],
        "mtp_frac_cand_comaximal_near": mtp_stats["frac_cand_comaximal_near"],
        "mtp_frac_cand_in_topk": mtp_stats["frac_cand_in_topk"],
        "mtp_frac_within_eps_tie": mtp_stats["frac_within_eps_tie"],
        "mtp_frac_within_eps_near": mtp_stats["frac_within_eps_near"],
        "mtp_n_exact_tie": mtp_stats["n_exact_tie"],
        "mtp_n_miss_probed": mtp_stats["n_miss_probed"],
        "mtp_separation_ratio": mtp_stats["separation_ratio_control_over_miss"],
        "mtp_miss_max_below_control_p10": mtp_disjoint,   # disjointness proof: worst miss < weakest normal gap
        "control_margin_median": control_median,
        "control_margin_p10": control_p10,
        "mtp_margin_hist": mtp_stats["margin_hist"],
        # cross-check: ngram + the no-spec chaos floor share the SAME near-tie signature
        "ngram_per_step_identity_rate": ngram_matched,
        "ngram_freerun_positional_rate": ngram_freerun,
        "ngram_miss_margin_median": ngram_stats["miss_margin_median"],
        "ngram_frac_cand_comaximal": ngram_stats["frac_cand_comaximal"],
        "selfdet_miss_margin_median": selfdet_stats["miss_margin_median"],
        "selfdet_frac_cand_comaximal": selfdet_stats["frac_cand_comaximal"],
        "chaos_floor_sequence_exact_rate": self_det.get("sequence_exact_rate"),
        "chaos_floor_matched_state_per_step": self_det.get("per_step_identity_rate"),
        # candidate-verify anchors (fern #566)
        "cv566_per_step_identity": CV566_PER_STEP,
        "cv566_sequence_exact": CV566_SEQUENCE_EXACT,
        # sequence-level (already established by #572/#573; echoed for completeness)
        "mtp_sequence_exact_rate": mtp_v.get("sequence_exact_rate"),
        "ngram_sequence_exact_rate": ngram_v.get("sequence_exact_rate"),
        "_per_drafter_stats": {"mtp": mtp_stats, "ngram": ngram_stats, "selfdet": selfdet_stats},
        "census_wandb_run_id": report.get("wandb_run_id"),
    }
    return out


def _print(v: dict) -> None:
    print("\n" + "=" * 12 + " PR #576 — MTP SPEC-DEC PER-STEP MECHANISM " + "=" * 12, flush=True)
    print(f"  matched-state per-step (fern#566 basis)  MTP={v['mtp_per_step_identity_rate']}  "
          f"ngram={v['ngram_per_step_identity_rate']}  (cand-verify {CV566_PER_STEP:.5f})", flush=True)
    print(f"  free-run positional rate ('0.4777')      MTP={v['mtp_freerun_positional_rate']}  "
          f"ngram={v['ngram_freerun_positional_rate']}", flush=True)
    print(f"  ratio vs cand-verify: free-run={v['specdec_per_step_vs_candidate_verify_ratio']}  "
          f"matched-state={v['specdec_matched_state_vs_candidate_verify_ratio']}", flush=True)
    print(f"  MTP miss margin median/p90/max           "
          f"{v['mtp_miss_margin_median']}/{v['mtp_miss_margin_p90']}/{v['mtp_miss_margin_max']}  "
          f"(control median {v['control_margin_median']}, p10 {v['control_margin_p10']})", flush=True)
    print(f"  DISJOINT (worst miss < weakest normal gap) = {v['mtp_miss_max_below_control_p10']}  "
          f"[{v['mtp_miss_margin_max']} < {v['control_margin_p10']}]", flush=True)
    print(f"  MTP cand co-maximal <=2ULP / <=0.10 / in-topk  "
          f"{v['mtp_frac_cand_comaximal_near']} / {v['mtp_frac_cand_comaximal']} / {v['mtp_frac_cand_in_topk']}  "
          f"(exact-tie {v['mtp_n_exact_tie']}/{v['mtp_n_miss_probed']})", flush=True)
    print(f"  MTP miss-margin histogram                {v['mtp_margin_hist']}", flush=True)
    print(f"  >> root_cause = {v['root_cause']}   verify_is_exact_greedy={v['mtp_verify_is_exact_greedy']}   "
          f"miss_is_tie_reorder={v['mtp_miss_is_tie_reorder']}", flush=True)


def log_wandb(v: dict) -> str | None:
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # pragma: no cover
        print(f"[mechanism] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile", agent="denken",
        name="denken/specdec-mtp-mechanism",
        group="base-fullhead-specdec-identity-census",
        tags=["specdec", "identity-census", "319", "mechanism", "mtp", "base-fullhead",
              "local-a10g", "analysis-only", "pr576"],
        notes="PR #576: MTP spec-dec per-step MECHANISM verdict — resolves the '0.4777 vs 0.994' "
              "puzzle (free-run vs matched-state basis) and classifies root_cause via the first-"
              "divergence M=1 flip-margin histogram.",
        config={"eps_tie": EPS_TIE, "eps_near": EPS_NEAR,
                "cv566_per_step": CV566_PER_STEP, "census_run": v.get("census_wandb_run_id")},
    )
    if run is None:
        return None
    summary = {k: val for k, val in v.items()
               if _finite(val) or isinstance(val, (bool, str))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="specdec-mtp-mechanism-verdict",
                      artifact_type="specdec-mechanism", data=v)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=HERE / "census_report.json")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if not args.report.exists():
        print(f"[mechanism] {args.report} missing — run census_driver.py first", flush=True)
        return 1
    report = json.loads(args.report.read_text())
    v = build_verdict(report)
    _print(v)

    out_file = HERE / "mechanism_verdict.json"
    out_file.write_text(json.dumps(v, indent=2, sort_keys=True, default=str))
    print(f"\n[mechanism] -> {out_file}", flush=True)

    if not args.no_wandb:
        rid = log_wandb(v)
        if rid:
            v["wandb_run_id"] = rid
            out_file.write_text(json.dumps(v, indent=2, sort_keys=True, default=str))
            print(f"[mechanism] wandb run id={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
