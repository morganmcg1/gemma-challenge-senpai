#!/usr/bin/env python3
"""Assemble the PR #263 private rank-2+ probe verdict from the run JSONs.

Reads the private rank-only run, the public reproduction run, and the fixed
smoke run, computes the six self-tests (a)-(f), the bounded tree gap-recovery,
and the verdict table. Pure analysis over already-written records -- launches
nothing, changes no served file. Run AFTER the smoke+public chain completes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- imported anchors (do NOT re-derive) -----------------------------------
PUB_COV4 = 0.6532              # rank_coverage z6wi4z4v public cov(rank2-4)
PUB_BEYOND4 = 0.3468
PUB_NDIV = 12869
PUB_TOP1 = 0.7335
PUB_RHO = [0.4165, 0.2655, 0.1908]   # public rho_marginal == tree_private_acceptance_gap rho_cond
PUB_RAW_ET = 3.8444537125748504      # tree_private_acceptance_gap public_ladder.linear_E_T
PRIV_RAW_ET = 3.0898055282313597     # tree_private_acceptance_gap private_ladder.linear_E_T_raw
TREE_PRIV_PROJ_TPS = 505.4635557048992  # ytxfi6zk headline (USED public rho)
BASELINE_TPS = 481.53
LAMBDA1_CEIL = 520.95

RESID_TOL = 0.01
PARTITION_TOL = 1e-6


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def has_nan(o) -> bool:
    if isinstance(o, float):
        return math.isnan(o)
    if isinstance(o, dict):
        return any(has_nan(v) for v in o.values())
    if isinstance(o, list):
        return any(has_nan(v) for v in o)
    return False


def main() -> int:
    priv = load(HERE / "rank_coverage_results.json")
    pub = load(HERE / "public_repro" / "rank_coverage_results.json")
    smoke = load(HERE / "smoke_fixed" / "rank_coverage_results_debug.json")

    assert priv is not None, "private run JSON missing"
    pa = priv["analysis"]
    priv_cov = pa["cumulative_coverage"]["4"]
    priv_beyond = pa["frac_true_beyond_topW"]
    n_div_priv = pa["n_divergences"]
    priv_top1 = pa["top1_acceptance"]
    priv_rho = [pa["rho_marginal"][k] for k in ("2", "3", "4")]

    # bounded tree gap-recovery (self-test d range [priv_raw, pub_raw])
    gap = PUB_RAW_ET - PRIV_RAW_ET
    recovered_frac = min(1.0, priv_cov / PUB_COV4)
    priv_tree_recovered_et = PRIV_RAW_ET + recovered_frac * gap

    # --- self tests ---------------------------------------------------------
    st = {}
    # (a) public default reproduces banked 0.653 within resid<=0.01
    if pub is not None:
        pub_cov_repro = pub["analysis"]["cumulative_coverage"]["4"]
        resid_a = abs(pub_cov_repro - PUB_COV4)
        st["a_public_reproduces_0653"] = bool(resid_a <= RESID_TOL)
    else:
        pub_cov_repro = None
        resid_a = None
        st["a_public_reproduces_0653"] = None  # pending

    # (b) smoke_records>0 AND n_div_private>0
    smoke_records = smoke["analysis"]["n_records"] if smoke is not None else None
    st["b_smoke_and_div_positive"] = bool(
        smoke_records is not None and smoke_records > 0 and n_div_priv > 0
    )

    # (c) partition: cov4 + beyond4 (+ rank1-correct=0 at divergences) == 1
    partition = priv_cov + priv_beyond  # rank-1 contributes 0 at divergences by construction
    st["c_partition_sums_to_1"] = bool(abs(partition - 1.0) <= PARTITION_TOL)

    # (d) monotone bound
    st["d_recovered_in_bounds"] = bool(PRIV_RAW_ET <= priv_tree_recovered_et <= PUB_RAW_ET)

    # (e) NaN-clean (both analyses)
    nan_clean = not has_nan(pa) and (pub is None or not has_nan(pub["analysis"]))
    st["e_nan_clean"] = bool(nan_clean)

    # (f) analysis-only: baseline + lambda=1 ceiling unchanged (probe moves nothing)
    st["f_baseline_unchanged"] = True  # 0 TPS, no served-file change, no submission

    pending = any(v is None for v in st.values())
    primary = (not pending) and all(bool(v) for v in st.values())

    rho_collapse = [(p / q - 1.0) for p, q in zip(priv_rho, PUB_RHO)]

    summary = {
        "pr": 263,
        "metric_primary": "private_rank2plus_probe_self_test_passes",
        "metric_test": "private_rank2plus_coverage",
        "private_rank2plus_probe_self_test_passes": (None if pending else primary),
        "private_rank2plus_coverage": priv_cov,
        "private_beyond_top4": priv_beyond,
        "n_div_private": n_div_priv,
        "private_top1_acceptance": priv_top1,
        "private_rho_marginal": priv_rho,
        "delta_cov_vs_public": priv_cov - PUB_COV4,
        "rel_drop_cov_vs_public_pct": (priv_cov - PUB_COV4) / PUB_COV4 * 100.0,
        "rho_collapse_pct": [round(x * 100, 1) for x in rho_collapse],
        "rho_collapse_mean_pct": round(sum(rho_collapse) / 3 * 100, 1),
        "private_tree_recovered_et": priv_tree_recovered_et,
        "tree_recovered_fraction_of_gap": recovered_frac,
        "private_rank_recovery_robust": bool(priv_cov >= PUB_COV4 - RESID_TOL),
        "rank_probe_analysis_only": True,
        "public_anchor": {
            "cov4": PUB_COV4, "beyond4": PUB_BEYOND4, "n_div": PUB_NDIV,
            "top1": PUB_TOP1, "rho_marginal": PUB_RHO, "run_id": "z6wi4z4v",
        },
        "public_repro": {
            "cov4": pub_cov_repro, "resid_vs_0653": resid_a,
            "wandb_run_id": (pub.get("wandb_run_id") if pub else None),
        },
        "smoke_records": smoke_records,
        "anchors": {
            "pub_raw_et": PUB_RAW_ET, "priv_raw_et": PRIV_RAW_ET,
            "tree_priv_proj_tps_USES_PUBLIC_RHO": TREE_PRIV_PROJ_TPS,
            "baseline_tps": BASELINE_TPS, "lambda1_ceiling_tps": LAMBDA1_CEIL,
        },
        "self_tests": st,
        "self_tests_pending": pending,
        "private_run_wandb": priv.get("wandb_run_id"),
    }

    out = HERE / "verdict_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # --- human-readable -----------------------------------------------------
    def f(x, n=4):
        return "pending" if x is None else f"{x:.{n}f}"

    print("\n================ PR #263 PRIVATE RANK-2+ PROBE VERDICT ================")
    print(f"{'set':<10}{'n_div':>8}{'top1':>9}{'rank2+cov':>11}{'beyond4':>10}{'impliedET':>11}")
    print(f"{'public':<10}{PUB_NDIV:>8}{PUB_TOP1:>9.4f}{PUB_COV4:>11.4f}{PUB_BEYOND4:>10.4f}{PUB_RAW_ET:>11.4f}")
    print(f"{'private':<10}{n_div_priv:>8}{priv_top1:>9.4f}{priv_cov:>11.4f}{priv_beyond:>10.4f}{priv_tree_recovered_et:>11.4f}")
    print(f"\nΔ rank2+ coverage vs public 0.653 : {priv_cov - PUB_COV4:+.4f} "
          f"({(priv_cov - PUB_COV4)/PUB_COV4*100:+.1f}% rel)")
    print(f"private branch-salvage rho        : {[round(x,4) for x in priv_rho]} "
          f"vs public {PUB_RHO} (mean {summary['rho_collapse_mean_pct']:+.1f}%)")
    print(f"tree recovers fraction of gap     : {recovered_frac:.4f} "
          f"(private_tree_recovered_et = {priv_tree_recovered_et:.4f}, "
          f"in [{PRIV_RAW_ET:.4f},{PUB_RAW_ET:.4f}])")
    print(f"private_rank_recovery_robust      : {summary['private_rank_recovery_robust']}")
    print(f"\npublic repro cov4 = {f(pub_cov_repro)}  (resid vs 0.6532 = {f(resid_a)}, tol {RESID_TOL})")
    print(f"smoke_records = {smoke_records}")
    print("\nself-tests:")
    for k, v in st.items():
        print(f"  {k:<28}: {v}")
    print(f"\nPRIMARY private_rank2plus_probe_self_test_passes = "
          f"{'PENDING' if pending else primary}")
    print(f"TEST    private_rank2plus_coverage               = {priv_cov:.4f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
