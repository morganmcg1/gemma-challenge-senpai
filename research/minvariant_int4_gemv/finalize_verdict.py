#!/usr/bin/env python
"""PR #736 wirbel — assemble the route-(a) verdict from the faithful microbenches
+ banked #728 anchors, and log to W&B (analysis_only, official_tps=0).

HEADLINE (refutes route (a)'s premise):
  The stock int4 Marlin GEMV is ALREADY bit-exactly M-invariant on A10G across
  every served shape and every M in 1..16 (faithful real-weight microbench). It
  is therefore NOT the source of the #607/#616/#728 spec-vs-AR divergence, and
  "building an M-invariant GEMV" is both unnecessary (it exists) and insufficient
  (not the locus). Under VLLM_BATCH_INVARIANT=1 the served Triton attention is
  also M-invariant for this comparison (both AR-decode and the K+1 verify take the
  2D single-segment path: use_3d=False via is_batch_invariant / max_seqlen_q>1).
  The residual divergence is a sub-ULP argmax tie-break at exact int4-grid logit
  ties (banked: onset-gap median 0.0, 100% <=0.3 nat, 0 confident flips @ tau=0.3),
  not a lossy GEMM divergence -> consistent with the GEMM emitting bit-identical
  values at M=1 vs M=K+1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent / "spec_achievable_ceiling" / "runs" / "sweep"
ANCHOR_OFFICIAL = 126.378


def load(p):
    return json.loads(Path(p).read_text())


def build_verdict():
    locus = load(HERE / "locus_real_report.json")
    msweep = load(HERE / "locus_msweep_report.json")
    micro = load(HERE / "microbench_report.json")
    rep728 = load(SWEEP / "report.json")
    rescue_k6 = load(SWEEP / "spec_k6.rescue.json")

    # --- locus: max bitdiff across ALL served shapes, ALL M in the sweep ---
    max_bitdiff = 0
    for r in msweep["results"]:
        for M, d in r["per_m"].items():
            max_bitdiff = max(max_bitdiff, d["n_bitdiff"])
        max_bitdiff = max(max_bitdiff, r["distinct8_total_bitdiff_vs_own_m1"])
    # also the M=6/7 single-activation locus_real pass
    for r in locus["results"]:
        for M, d in r.get("per_m", {}).items():
            if "n_bitdiff_vs_m1" in d:
                max_bitdiff = max(max_bitdiff, d["n_bitdiff_vs_m1"])
    faithful = {r["name"]: r["faithfulness"]["median_relerr_vs_fp32dequant"]
                for r in locus["results"] if r.get("faithfulness")}

    # --- overhead: stock-kernel M-flatness (timing is value-independent) ---
    mflat = {}
    for r in micro["results"]:
        t1 = r["timings"]["1"]["median_s"]
        t7 = r["timings"]["7"]["median_s"]
        mflat[r["name"]] = round(t7 / t1, 4)

    # --- #728 anchors (existing strict-DIVERGENT spec speed; NOT strict-clean) ---
    anchored = {}
    for r in rep728.get("results", []):
        k = r.get("k")
        if k in (5, 6, 7):
            anchored[k] = {
                "wall_tps_local": r.get("wall_tps_local"),
                "official_equiv_anchored": r.get("official_equiv_anchored"),
                "strict_verdict": r.get("strict_verdict"),
            }

    verdict = {
        "pr": 736, "analysis_only": True, "official_tps": 0,
        "group": "wirbel-minvariant-int4-gemv",
        "device": locus["device"], "vllm": "0.22.0",
        "anchor_submission": "int4_g128_lmhead",
        "anchor_official_tps": ANCHOR_OFFICIAL, "anchor_ppl": 2.019, "anchor_wandb": "905tbujn",

        # ---- JOB 1: locus ----
        "job1_locus": {
            "gemv_is_m_dependent": False,
            "max_bitdiff_any_served_shape_M_1to16": max_bitdiff,
            "faithfulness_median_relerr_vs_fp32dequant": faithful,
            "m1_run2run_bitexact": all(r.get("m1_run2run_bitexact", True) for r in locus["results"]),
            "distinct_rows_verify_bitexact": all(
                r["distinct8_total_bitdiff_vs_own_m1"] == 0 for r in msweep["results"]),
            "attention_m_invariant_under_BI": True,
            "attention_evidence": "triton_unified_attention use_3d=False for both AR-decode "
                                  "(is_batch_invariant) and K+1 verify (max_seqlen_q>1); same "
                                  "2D single-segment TILE_SIZE_PREFILL reduction.",
            "conclusion": "int4 Marlin GEMV is NOT the M-dependence locus; bit-exact "
                          "M-invariant across all served shapes. Corrects the prior "
                          "'M-dependent Marlin GEMM' attribution (#607/#616 inference).",
        },

        # ---- JOB 2: buildability ----
        "job2_buildability": {
            "m_invariant_gemv_already_exists_as_stock_kernel": True,
            "cheap_python_knob_to_alter_reduction_schedule": False,
            "strict_clean_lever_go_nogo": "NO-GO",
            "go_nogo_binary": 0,
            "reason": "The stock int4 Marlin GEMV is already maximally M-invariant "
                      "(bit-exact) AND is not the divergence source; making it 'more "
                      "invariant' cannot remove a divergence it does not cause.",
        },

        # ---- JOB 3: overhead ----
        "job3_overhead": {
            "pct_added_by_m_invariant_gemv": 0.0,
            "reason": "no kernel change is made; the stock kernel IS the M-invariant GEMV.",
            "stock_kernel_m_flat_ratio_t_m7_over_t_m1": mflat,
            "memory_or_launch_bound_at_M_le_8": True,
        },

        # ---- JOB 4: net official-equiv ----
        "job4_net_official_equiv": {
            "route_a_strict_clean_official_equiv_tps": 0.0,
            "margin_over_anchor_strict_clean_tps": 0.0 - 0.0,
            "speed_preserved_not_clean_anchored_official_equiv": {
                str(k): anchored.get(k, {}).get("official_equiv_anchored") for k in (5, 6)},
            "existing_divergent_spec_pct_over_anchor": {
                str(k): (anchored[k]["official_equiv_anchored"] / ANCHOR_OFFICIAL - 1) * 100
                for k in (5, 6) if k in anchored},
            "note": "route (a) makes NO change, so 'overhead 0%' preserves the EXISTING "
                    "spec speed (~260 TPS anchored at K=6) -- but that speed is strict-"
                    "DIVERGENT, and route (a) cannot confer strict-cleanness. So the "
                    "route-(a) STRICT-CLEAN official-equiv is 0 (no such config).",
        },

        # ---- divergence nature (banked #728, reconciles the refutation) ----
        "divergence_characterization": {
            "onset_gap_median_nat": rescue_k6["onset_gap_median"],
            "onset_gap_max_nat": rescue_k6["onset_gap_max"],
            "frac_onset_le_0p3nat": rescue_k6["onset_gap_frac_le_0.3"],
            "confident_genuine_flips_at_tau0p3": rescue_k6["tau_sweep"]["tau_0.3"]["confident_genuine_flips"],
            "nature": "sub-ULP argmax tie-break at exact int4-grid logit ties; benign; "
                      "100% rescued at tau=0.3 (downstream-quality-neutral).",
        },

        # ---- route split feedback for denken #733 (via advisor branch only) ----
        "route_split": {
            "route_a_minvariant_gemv": "REFUTED (GEMV already invariant; not the locus).",
            "route_b_tree_verify_at_m1": "remains the strict-clean lever (re-decode accepted "
                                         "prefix at M=1 -> byte-exact by construction) at a speed "
                                         "cost denken #733 quantifies.",
            "third_option": "since the divergence is benign exact-tie tie-breaks (0 confident "
                            "flips @ tau=0.3), relaxing strict #319 to a tau-tolerance / "
                            "downstream-quality gate keeps the full spec speed with no kernel work.",
        },
        "primary_metric": {"name": "route_a_strict_clean_official_equiv_tps", "value": 0.0},
        "test_metric": {"name": "buildability_go_nogo", "value": 0},
    }
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--name", default="wirbel/minvariant-int4-gemv-strictclean")
    args = ap.parse_args()

    v = build_verdict()
    out = HERE / "route_a_verdict.json"
    out.write_text(json.dumps(v, indent=2, default=str))
    print(f"[verdict] -> {out}")
    print(json.dumps({
        "job1_gemv_is_locus": v["job1_locus"]["gemv_is_m_dependent"],
        "job1_max_bitdiff_all_M": v["job1_locus"]["max_bitdiff_any_served_shape_M_1to16"],
        "job2_go_nogo": v["job2_buildability"]["strict_clean_lever_go_nogo"],
        "job3_overhead_pct": v["job3_overhead"]["pct_added_by_m_invariant_gemv"],
        "job4_route_a_strict_clean_tps": v["job4_net_official_equiv"]["route_a_strict_clean_official_equiv_tps"],
    }, indent=2))

    if args.no_wandb:
        return
    import wandb
    run = wandb.init(
        entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
        group="wirbel-minvariant-int4-gemv", name=args.name,
        config={
            "pr": 736, "lever": "minvariant_int4_marlin_gemv", "analysis_only": True,
            "official_tps": 0, "no_served_file_change": True, "no_hf_job": True,
            "anchor_submission": "int4_g128_lmhead", "anchor_official_tps": ANCHOR_OFFICIAL,
            "anchor_ppl": 2.019, "anchor_wandb": "905tbujn",
            "gpu": "A10G_sm86", "vllm": "0.22.0", "attn_backend": "TRITON_ATTN",
            "verdict": "ROUTE_A_REFUTED_GEMV_ALREADY_M_INVARIANT",
        },
    )
    j1, j2, j3, j4 = v["job1_locus"], v["job2_buildability"], v["job3_overhead"], v["job4_net_official_equiv"]
    summary = {
        "job1/gemv_is_m_dependent": int(j1["gemv_is_m_dependent"]),
        "job1/max_bitdiff_all_served_shapes_M_1to16": j1["max_bitdiff_any_served_shape_M_1to16"],
        "job1/m1_run2run_bitexact": int(j1["m1_run2run_bitexact"]),
        "job1/distinct_rows_verify_bitexact": int(j1["distinct_rows_verify_bitexact"]),
        "job1/attention_m_invariant_under_BI": int(j1["attention_m_invariant_under_BI"]),
        "job2/buildability_go_nogo": j2["go_nogo_binary"],
        "job3/overhead_pct": j3["pct_added_by_m_invariant_gemv"],
        "job4/route_a_strict_clean_official_equiv_tps": j4["route_a_strict_clean_official_equiv_tps"],
        "job4/margin_over_anchor_strict_clean_tps": j4["margin_over_anchor_strict_clean_tps"],
        "divergence/onset_gap_median_nat": v["divergence_characterization"]["onset_gap_median_nat"],
        "divergence/confident_flips_at_tau0p3": v["divergence_characterization"]["confident_genuine_flips_at_tau0p3"],
        "primary_metric": j4["route_a_strict_clean_official_equiv_tps"],
        "test_metric": j2["go_nogo_binary"],
    }
    for nm, ratio in j3["stock_kernel_m_flat_ratio_t_m7_over_t_m1"].items():
        summary[f"mflat/{nm}_t_m7_over_m1"] = ratio
    for nm, rel in j1["faithfulness_median_relerr_vs_fp32dequant"].items():
        summary[f"faithful/{nm}_relerr"] = rel

    tbl = wandb.Table(columns=["shape", "size_k", "size_n", "atomic", "any_div_M_1to16",
                               "distinct8_bitdiff", "max_abs_M8"])
    for r in load(HERE / "locus_msweep_report.json")["results"]:
        tbl.add_data(r["name"], r["size_k"], r["size_n"], int(r["use_atomic_add"]),
                     int(r["any_divergence_vs_m1"]), r["distinct8_total_bitdiff_vs_own_m1"],
                     r["per_m"]["8"]["max_abs"])
    run.log({"gemv_m_sweep_locus": tbl, **summary})
    for k, val in summary.items():
        run.summary[k] = val
    print("WANDB_RUN_ID", run.id)
    run.finish()


if __name__ == "__main__":
    main()
