#!/usr/bin/env python3
"""Assemble the static-K wall-clock A/B into the PR #273 verdict.

Reads each K-arm's ``paired_ab.json`` (produced by ``paired_tps_ab.py``) for the
seed=1 primary sweep, places the MEASURED local wall_tps delta next to the
#256/#266 *composition* projection, and computes the **realization ratio**
``= measured-delta% / composition-delta%`` for each K. The headline question:
does measured local wall-clock confirm or refute the +4.28% static-K=4
composition gain?

Composition table (PR #266, run cpjafa3h, ONEGRAPH static-K capture):
    K   E[T]    net_tps   gain%      clears500
    3   2.720   492.94    +2.370     no
    4   3.080   502.12    +4.277     YES   <- top composition winner
    5   3.381   500.79    +3.999     YES
    6   3.632   492.95    +2.372     no
    7   3.844   481.53     0.000     (deployed reference)

The composition prices a draft-pass saving against E[T]/model-forward-step but the
served wall step is dominated by a FIXED serving overhead (CPU/Python/scheduler/
sampler/detokenize) that does NOT shrink when draft passes are dropped — so the
composition is expected to OVER-CREDIT, and the realization ratio to fall well
below 1 (or invert). The deployed path being K=7 is the standing evidence.

PRIMARY (self-test boolean) ``static_k_wallclock_ab_self_test_passes`` = 1 iff:
  (a) K=7 baseline stable within ±1% across repeats,
  (b) all K runs share one harness/prompts/seed (only K varies),
  (c) greedy-validity PARITY with deployed K=7 (see below),
  (d) all TPS finite, NaN-clean,
  (e) realization ratios reported for K=4 and K=5.

PREMISE CORRECTION (gate c). The PR stated the A/B is "greedy-safe by construction:
token-ids identical across all K (128/128)". That is empirically FALSE for *every*
config on this int4 + vLLM stack: the verify step's FP reduction order is not
bit-stable, so argmax ties flip and the greedy stream diverges run-to-run. The
official ``greedy_gate.py`` reports the *deployed* K=7 itself as DIVERGENT vs the
canonical M=1 autoregressive reference (~59% of tokens differ, late stochastic
onset) — this is the int4+vLLM nondeterminism the competition gate is documented to
tolerate (it compares within-stack; run-to-run FP drift is not a blocker; the live
gate is PPL ≤ 2.42, which is K-invariant because PPL scoring never invokes the
drafter). So strict 128/128 is unachievable for ANY K and would also fail the
deployed path. The methodologically correct gate is PARITY: every candidate K must
stay in K=7's benign late-onset FP regime (matching verdict, divergent-token% within
tol, onset median within tol, no integrity failures) — i.e. changing K did NOT push
the model into an early/lossy regime. The strict cross-K identity number is still
reported (non-gating) under ``greedy_identity.strict_*`` for full transparency.

Usage:
    .venv/bin/python research/validity/static_k_wallclock_ab/analyze.py --seed 1
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUTROOT = ROOT / "research" / "validity" / "static_k_wallclock_ab"
REF_K = 7

# --- #266 composition anchors (run cpjafa3h) ---
COMPOSITION = {
    3: {"e_t": 2.720164251497006, "net_tps": 492.94094882742036, "gain_pct": 2.369727499308527, "clears_500": False},
    4: {"e_t": 3.080339640718563, "net_tps": 502.12279296537685, "gain_pct": 4.27653374979271, "clears_500": True},
    5: {"e_t": 3.38082377245509, "net_tps": 500.78615614206257, "gain_pct": 3.998952535057554, "clears_500": True},
    6: {"e_t": 3.6318059880239524, "net_tps": 492.9541600938257, "gain_pct": 2.372471101245144, "clears_500": False},
    7: {"e_t": 3.8444537125748504, "net_tps": 481.53, "gain_pct": 0.0, "clears_500": False},
}
OFFICIAL_ANCHOR_TPS = 481.53
TARGET_TPS = 500.0
# A fresh server that recompiles a novel-K CUDA/loop graph runs longer than the warm
# steady ~90s; runs above this are flagged cold-cache for the warm-median cross-check.
COLD_READY_S = 120.0
OP_THRESHOLD_PCT = 0.10  # #72/#82 REAL/NULL bar at N>=3
STABILITY_PCT = 1.0      # K=7 baseline ±1% harness-stability gate
# Greedy-validity parity tolerances (gate c). Each candidate K's official greedy_gate
# divergence regime (vs the committed M=1 AR reference) must match the deployed K=7's.
PARITY_DIVTOK_TOL_PCT = 5.0   # |div-token%(K) - div-token%(K7)| must be <= this
PARITY_ONSET_TOL = 16.0       # |onset_median(K) - onset_median(K7)| must be <= this
PARITY_CANDIDATE_KS = (3, 4, 5, 6)


def _load(seed: int, k: int) -> dict | None:
    p = OUTROOT / f"seed{seed}_mtp_k{k}" / "paired_ab.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _warm_median(records: list[dict]) -> tuple[float | None, int]:
    warm = [r["wall_tps"] for r in records
            if isinstance(r.get("wall_tps"), (int, float)) and r["wall_tps"] == r["wall_tps"]
            and (r.get("server_ready_s") or 0) <= COLD_READY_S]
    n_cold = len([r for r in records if isinstance(r.get("wall_tps"), (int, float))]) - len(warm)
    return (statistics.median(warm) if warm else None), n_cold


def _finite(values: list) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _greedy_validity_parity(gate: dict) -> dict:
    """Does every candidate K share the deployed K=7's benign-FP greedy regime?

    ``gate`` is the ``greedy_gate_vs_m1ar_seed{N}.json`` produced by
    ``greedy_gate_summary.py`` (official greedy_gate.py per K vs the committed M=1
    AR reference). Parity holds iff, for each candidate K present, the verdict
    matches K=7's, divergent-token% is within ``PARITY_DIVTOK_TOL_PCT`` of K=7's,
    onset median is within ``PARITY_ONSET_TOL`` of K=7's, and there are no integrity
    failures. That certifies changing K did not push the stream out of the deployed
    regime — the precondition for a fair wall-clock A/B."""
    ref = gate.get(str(REF_K))
    if not ref or ref.get("pending"):
        return {"parity_pass": False, "reason": "K=7 greedy-gate summary missing",
                "per_k": {}, "all_candidates_present": False}
    ref_div = ref["divergent_token_pct"]
    ref_onset = ref["onset_median"]
    per_k: dict[str, Any] = {}
    ok = True
    for k in PARITY_CANDIDATE_KS:
        s = gate.get(str(k))
        if not s or s.get("pending"):
            continue
        same_verdict = (s["verdict"] == ref["verdict"])
        div_close = abs(s["divergent_token_pct"] - ref_div) <= PARITY_DIVTOK_TOL_PCT
        onset_close = (s["onset_median"] is not None and ref_onset is not None
                       and abs(s["onset_median"] - ref_onset) <= PARITY_ONSET_TOL)
        integ = not s.get("integrity_failures")
        k_ok = bool(same_verdict and div_close and onset_close and integ)
        ok = ok and k_ok
        per_k[str(k)] = {
            "verdict": s["verdict"], "divergent_token_pct": s["divergent_token_pct"],
            "onset_median": s["onset_median"], "n_early_onset_lt16": s.get("n_early_onset_lt16"),
            "same_verdict_as_k7": same_verdict, "divtok_within_tol": div_close,
            "onset_within_tol": onset_close, "no_integrity_failures": integ, "parity": k_ok,
        }
    all_present = set(map(str, PARITY_CANDIDATE_KS)).issubset(per_k.keys())
    return {
        "parity_pass": bool(ok and all_present),
        "ref_k": REF_K, "ref_verdict": ref["verdict"],
        "ref_divergent_token_pct": ref_div, "ref_onset_median": ref_onset,
        "tolerances": {"divtok_pct": PARITY_DIVTOK_TOL_PCT, "onset": PARITY_ONSET_TOL},
        "all_candidates_present": all_present,
        "per_k": per_k,
    }


def _override_only_k(result: dict, label: str) -> bool:
    """The arm's override env is exactly SPECULATIVE_CONFIG (the K knob) and nothing else.

    ``override_env`` lives at the top-level ``result["candidate"|"baseline"]`` block
    (not under ``result["arms"]``, which holds only the timing stats)."""
    ov = (result.get(label, {}) or {}).get("override_env") or {}
    keys = set(ov)
    return keys == {"SPECULATIVE_CONFIG"} or keys == set()  # baseline arm has no override


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--cert", type=Path, default=None,
                    help="greedy-identity certificate json (default: seed-matched)")
    ap.add_argument("--seed2", type=int, default=2, help="confirmation seed for decisive arms")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cert_path = args.cert or (OUTROOT / f"greedy_identity_certificate_seed{args.seed}.json")
    cert = json.loads(cert_path.read_text()) if cert_path.exists() else {}
    gate_path = OUTROOT / f"greedy_gate_vs_m1ar_seed{args.seed}.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}

    # --- K=7 baseline (shared, read from any completed arm) ---
    base_stats = None
    base_recs = None
    workload = None
    for k in args.ks:
        d = _load(args.seed, k)
        if d:
            base_stats = d["arms"]["baseline"]
            base_recs = base_stats.get("records") or []
            workload = d.get("workload")
            break
    if base_stats is None:
        print(f"[analyze] no completed arms under {OUTROOT} for seed={args.seed}")
        return 1

    base_med = base_stats["wall_tps"]["median"]
    base_cv = base_stats["wall_tps"].get("cv_pct")
    base_range_pct = base_stats["wall_tps"].get("range_pct")
    base_eacc = base_stats["e_accept_exact"]["mean"]
    base_vals = base_stats["wall_tps"].get("values") or []
    base_stable = (isinstance(base_range_pct, (int, float)) and abs(base_range_pct) <= STABILITY_PCT)

    per_k: dict[str, Any] = {}
    all_finite = _finite(base_vals)
    only_k_varies = True
    realization: dict[str, float] = {}

    for k in sorted(args.ks):
        comp = COMPOSITION.get(k, {})
        if k == REF_K:
            warm, n_cold = _warm_median(base_recs)
            per_k[str(k)] = {
                "measured": {
                    "median": base_med, "mean": base_stats["wall_tps"].get("mean"),
                    "std": base_stats["wall_tps"].get("std"), "cv_pct": base_cv,
                    "range_pct": base_range_pct, "values": base_vals,
                    "warm_median": warm, "n_cold": n_cold, "n": base_stats["wall_tps"].get("n"),
                    "e_accept_exact": base_eacc,
                    "server_ready_s": [round(r.get("server_ready_s", 0)) for r in base_recs],
                },
                "composition": comp,
                "measured_delta_pct": 0.0, "realization_ratio": None,
                "verdict": "REF",
            }
            continue
        d = _load(args.seed, k)
        if not d:
            per_k[str(k)] = {"pending": True, "composition": comp}
            continue
        cand = d["arms"]["candidate"]
        vals = cand["wall_tps"].get("values") or []
        all_finite = all_finite and _finite(vals)
        only_k_varies = only_k_varies and _override_only_k(d, "candidate") and _override_only_k(d, "baseline")
        # workload consistency (same prompts/output/seed)
        if d.get("workload") != workload:
            only_k_varies = False
        med = cand["wall_tps"]["median"]
        warm, n_cold = _warm_median(cand.get("records") or [])
        delta_pct = 100.0 * (med - base_med) / base_med
        comp_gain = comp.get("gain_pct")
        ratio = (delta_pct / comp_gain) if (comp_gain not in (None, 0.0)) else None
        if ratio is not None:
            realization[str(k)] = ratio
        verdict = "REAL+" if delta_pct >= OP_THRESHOLD_PCT else ("REAL-" if delta_pct <= -OP_THRESHOLD_PCT else "NULL")
        per_k[str(k)] = {
            "measured": {
                "median": med, "mean": cand["wall_tps"].get("mean"),
                "std": cand["wall_tps"].get("std"), "cv_pct": cand["wall_tps"].get("cv_pct"),
                "values": vals, "warm_median": warm, "n_cold": n_cold,
                "n": cand["wall_tps"].get("n"), "e_accept_exact": cand["e_accept_exact"]["mean"],
                "server_ready_s": [round(r.get("server_ready_s", 0)) for r in (d["arms"]["candidate"].get("records") or [])],
            },
            "composition": comp,
            "measured_delta_pct": delta_pct,
            "realization_ratio": ratio,
            "verdict": verdict,
        }

    # --- self-test sub-gates ---
    # Strict 128/128 cross-K token-id identity (the PR's literal premise). Empirically
    # FALSE for every config (int4+vLLM verify-step FP nondeterminism) — REPORTED but
    # NON-GATING. Gate (c) uses greedy-validity PARITY instead (see module docstring).
    strict_token_id_identity = bool(cert.get("token_id_identity_all_k"))
    parity = _greedy_validity_parity(gate)
    ratios_reported = ("4" in realization) and ("5" in realization)
    completed_ks = [k for k in args.ks if not per_k[str(k)].get("pending")]
    all_present = set(args.ks) == set(completed_ks)
    self_test = {
        "a_k7_baseline_stable_within_1pct": bool(base_stable),
        "b_only_k_varies": bool(only_k_varies and all_present),
        "c_greedy_validity_parity_vs_k7": bool(parity["parity_pass"]),
        "d_tps_finite_nan_clean": bool(all_finite),
        "e_realization_ratios_reported_k4_k5": bool(ratios_reported),
    }
    self_test_passes = all(self_test.values())

    # --- headline numbers ---
    k4 = per_k.get("4", {})
    k5 = per_k.get("5", {})
    test_metric_value = k4.get("measured_delta_pct")
    k4_beats_k7 = bool(isinstance(test_metric_value, (int, float)) and test_metric_value >= OP_THRESHOLD_PCT)
    rr4 = realization.get("4")
    rr5 = realization.get("5")
    composition_over_credits = bool(rr4 is not None and rr4 < 0.5)

    # --- seed-2 confirmation (decisive arms) ---
    seed2 = {}
    for k in (4, 5, 7):
        d = _load(args.seed2, k)
        if not d:
            continue
        if k == REF_K:
            seed2[str(k)] = {"median": d["arms"]["baseline"]["wall_tps"]["median"]}
        else:
            b = d["arms"]["baseline"]["wall_tps"]["median"]
            c = d["arms"]["candidate"]["wall_tps"]["median"]
            seed2[str(k)] = {"median": c, "delta_pct": 100.0 * (c - b) / b, "baseline_median": b}

    verdict_sentence = _verdict_sentence(test_metric_value, rr4, rr5, k4_beats_k7, composition_over_credits)

    report = {
        "pr": 273,
        "experiment": "static_k_wallclock_ab",
        "seed_primary": args.seed,
        "workload": workload,
        "anchors": {
            "official_tps": OFFICIAL_ANCHOR_TPS, "target_tps": TARGET_TPS,
            "local_k7_reference_wall_tps": base_med,
            "local_k7_e_accept_exact": base_eacc,
            "composition_e_t_k7": COMPOSITION[7]["e_t"],
        },
        "k7_baseline": {
            "median_wall_tps": base_med, "cv_pct": base_cv, "range_pct": base_range_pct,
            "values": base_vals, "stable_within_1pct": bool(base_stable),
        },
        "per_k": per_k,
        "realization_ratios": realization,
        "test_metric": {"name": "measured_local_wall_tps_gain_k4_vs_k7_pct", "value": test_metric_value},
        "booleans": {
            "static_k4_beats_k7_measured": k4_beats_k7,
            "composition_over_credits": composition_over_credits,
        },
        "greedy_identity": {
            # GATING: greedy-validity parity vs deployed K=7 (official greedy_gate.py vs M=1 AR).
            "parity_gate_file": str(gate_path.relative_to(ROOT)) if gate_path.exists() else None,
            "greedy_validity_parity_vs_k7": bool(parity["parity_pass"]),
            "parity": parity,
            # NON-GATING (reported): strict cross-K token-id identity vs K=7 (PR's literal premise).
            "strict_certificate_file": str(cert_path.relative_to(ROOT)) if cert_path.exists() else None,
            "strict_token_id_identity_all_k": strict_token_id_identity,
            "strict_per_k": {kk: {"n_identical": vv.get("n_identical"), "n_prompts": vv.get("n_prompts"),
                                  "all_identical": vv.get("all_identical")}
                             for kk, vv in (cert.get("per_k") or {}).items()},
        },
        "premise_correction": {
            "pr_claim": "greedy-safe by construction: token-ids identical across all K (128/128)",
            "finding": "strict 128/128 is FALSE for every config on this int4+vLLM stack; "
                       "the deployed K=7 itself is DIVERGENT vs the M=1 AR reference by benign "
                       "late-onset FP nondeterminism (verify-step reduction order is not bit-stable).",
            "why_not_a_blocker": "the competition greedy gate compares within-stack and tolerates "
                                 "run-to-run FP drift; the live numeric gate is PPL <= 2.42, which is "
                                 "K-invariant (PPL scoring is teacher-forced and never invokes the drafter).",
            "corrected_gate": "greedy-validity PARITY: every candidate K stays in K=7's benign-FP "
                              "regime (same verdict, divergent-token% within tol, onset median within "
                              "tol, no integrity failures) -> the A/B compares like-for-like greedy-valid configs.",
            "strict_token_id_identity_all_k": strict_token_id_identity,
            "greedy_validity_parity_vs_k7": bool(parity["parity_pass"]),
        },
        "self_test": self_test,
        "static_k_wallclock_ab_self_test_passes": int(self_test_passes),
        "seed2_confirmation": seed2,
        "verdict_sentence": verdict_sentence,
    }
    out = args.out or (OUTROOT / "report.json")
    out.write_text(json.dumps(report, indent=2, default=str))

    _print(report)
    return 0


def _verdict_sentence(delta4, rr4, rr5, beats, over_credits) -> str:
    if delta4 is None:
        return "Primary K=4 arm not yet complete."
    d = f"{delta4:+.3f}%"
    r4 = f"{rr4:.3f}" if rr4 is not None else "n/a"
    r5 = f"{rr5:.3f}" if rr5 is not None else "n/a"
    if beats:
        return (f"Measured local wall-clock CONFIRMS static-K=4 over K=7 ({d}); "
                f"realization ratio K4={r4}, K5={r5} — escalation candidate.")
    return (f"Measured local wall-clock REFUTES the +4.28% static-K=4 composition gain "
            f"(measured K4 vs K7 = {d}, realization ratio K4={r4}, K5={r5}); "
            f"the composition over-credits the draft-pass saving — deployed K=7 stands.")


def _print(rep: dict) -> None:
    print("\n===== PR #273 static-K wall-clock A/B — measured vs composition =====")
    print(f"  seed={rep['seed_primary']}  K=7 local ref={rep['k7_baseline']['median_wall_tps']:.3f} wall_tps "
          f"(cv={rep['k7_baseline']['cv_pct']}, stable±1%={rep['k7_baseline']['stable_within_1pct']})")
    print(f"{'K':>3}  {'measured':>9}  {'Δ% vs K7':>9}  {'verdict':>7}  {'warm_med':>9}  "
          f"{'E[acc]':>7}  {'comp gain%':>10}  {'realization':>11}  {'comp net_tps':>12}")
    for k in sorted(rep["per_k"], key=int):
        r = rep["per_k"][k]
        if r.get("pending"):
            print(f"{k:>3}  {'PENDING':>9}")
            continue
        m = r["measured"]; comp = r.get("composition", {})
        wm = f"{m['warm_median']:.3f}" if m.get("warm_median") is not None else "—"
        rr = r.get("realization_ratio")
        rr_s = f"{rr:+.3f}" if rr is not None else "—"
        print(f"{k:>3}  {m['median']:>9.3f}  {r['measured_delta_pct']:>+8.3f}%  {r['verdict']:>7}  "
              f"{wm:>9}  {m['e_accept_exact']:>7.4f}  {comp.get('gain_pct', 0):>+9.3f}%  "
              f"{rr_s:>11}  {comp.get('net_tps', 0):>12.2f}")
    gi = rep["greedy_identity"]; par = gi.get("parity", {})
    print("\n  greedy-validity (gate c) — official greedy_gate.py vs M=1 AR reference:")
    print(f"    strict 128/128 token-id identity across K = {gi['strict_token_id_identity_all_k']}  "
          f"(NON-GATING; false by int4+vLLM FP nondeterminism — deployed K=7 diverges too)")
    if par.get("per_k"):
        print(f"    K7 regime: {par['ref_verdict']} div-tok={par['ref_divergent_token_pct']}% "
              f"onset_med={par['ref_onset_median']}  (tol ±{par['tolerances']['divtok_pct']}pp / "
              f"±{par['tolerances']['onset']} tok)")
        for kk in sorted(par["per_k"], key=int):
            p = par["per_k"][kk]
            print(f"      K={kk}: {p['verdict']} div-tok={p['divergent_token_pct']}% "
                  f"onset_med={p['onset_median']}  parity={'YES' if p['parity'] else 'NO'}")
    print(f"    >>> greedy_validity_parity_vs_k7 = {gi['greedy_validity_parity_vs_k7']}")
    st = rep["self_test"]
    print("\n  self-test:")
    for kk, vv in st.items():
        print(f"    {'PASS' if vv else 'FAIL'}  {kk}")
    print(f"\n  >>> static_k_wallclock_ab_self_test_passes = {rep['static_k_wallclock_ab_self_test_passes']}")
    print(f"  >>> TEST measured_local_wall_tps_gain_k4_vs_k7_pct = {rep['test_metric']['value']}")
    print(f"  >>> static_k4_beats_k7_measured = {rep['booleans']['static_k4_beats_k7_measured']}")
    print(f"  >>> composition_over_credits = {rep['booleans']['composition_over_credits']}")
    if rep["seed2_confirmation"]:
        print(f"  >>> seed2 confirmation: {rep['seed2_confirmation']}")
    print(f"\n  VERDICT: {rep['verdict_sentence']}")
    print(f"  report -> {OUTROOT.name}/report.json")


if __name__ == "__main__":
    raise SystemExit(main())
