#!/usr/bin/env python
"""PR #513: log acceptance-rate-invariance of spec-dec distribution preservation
to W&B. group=private-quality-preservation. analysis_only; official_tps=0.

The QUALITY twin of denken #489 / kanna #504,#508 (which priced the SPEED side of
the PRIVATE leaderboard acceptance shift): does the private acceptance breach carry
ANY downstream-quality exposure, or is the deployed rejection sampler
distribution-exact at EVERY acceptance rate (a pure SPEED risk)?
"""

from __future__ import annotations

import argparse
import json

import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"
GROUP = "private-quality-preservation"

ACCEPTANCE_RULE = (
    "standard-rejection (greedy-draft / NO_DRAFT_PROBS): accept x_d w.p. p(x_d) "
    "[random kernel L926], else recover ~ p|{y!=x_d} [recovered kernel L1006]; "
    "output ~ p for ANY draft token => acceptance-rate-invariant preservation"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--name", default="denken/private-quality-preservation")
    args = ap.parse_args()

    blob = json.load(open(args.results))
    s = blob["summary"]
    st = blob["self_test"]
    syn = blob["leg1_synthetic"]
    real = blob.get("leg1_real", [])
    depth = blob["leg2_k7_depth"]
    natural = blob["leg2_k7_natural"]

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        group=GROUP,
        name=args.name,
        job_type="analysis",
        config={
            "analysis_only": True,
            "official_tps": 0,
            "ship": "surgical-357 (PR#499)",
            "spec_method": "mtp",
            "num_speculative_tokens": 7,
            "vllm": s["vllm"],
            "rejection_sample_method": "standard (default)",
            "draft_sample_method": "greedy (default) -> NO_DRAFT_PROBS",
            "gen_config_sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "do_sample": True},
            "public_accept_anchor": s["public_accept_anchor"],
            "private_breach_band": s["private_breach_band"],
            "k_sigma_band": s["k_sigma_band"],
            "R_redraws": s["R_redraws"],
            "n_synthetic_cases": len(syn),
            "n_real_reasoning_cases": len(real),
            "M_synthetic": syn[0]["M"] if syn else None,
            "M_real": real[0]["M"] if real else None,
            "B_k7": depth["B"],
            "baseline_pr505_run": "bg03bq0d",
            "noise_floor_mean": s["mean_iid_noise_floor"],
        },
    )

    invariant = bool(s["quality_acceptance_invariant"])
    no_accum = bool(s["k7_no_accumulation"])
    pure_speed = invariant and no_accum and s["private_quality_exposure"] < s["exc_tol"]

    wandb.summary.update({
        "acceptance_rule": ACCEPTANCE_RULE,
        # ---- KEY OUTPUTS (card) ----
        "max_tv_across_acceptance_sweep": s["max_tv_across_acceptance_sweep"],
        "quality_acceptance_invariant": invariant,
        "private_quality_exposure": s["private_quality_exposure"],
        "max_tv_over_k7_positions": s["max_tv_over_k7_positions"],
        "k7_no_accumulation": no_accum,
        "verdict": s["verdict"],
        # ---- #505-aligned distribution metrics (worst case across the sweep) ----
        "sampled_tv_base_vs_spec": s["max_tv_across_acceptance_sweep"],
        "sampled_tv_iid_noise_floor": s["mean_iid_noise_floor"],
        "max_z_over_floor_across_sweep": s["max_z_over_floor_across_sweep"],
        "mean_signed_excess_over_mu": s["mean_signed_excess_over_mu"],
        "accept_z_correlation": s["accept_z_correlation"],
        "n_acceptance_points_pooled": s["n_acceptance_points_pooled"],
        "n_band_exceed": s["n_band_exceed"],
        # ---- M-independent goodness-of-fit corroboration ----
        "gtest_bonferroni_global_pvalue": s["gtest_bonferroni_global_pvalue"],
        "gtest_frac_p_gt_05": s["gtest_frac_p_gt_05"],
        # ---- split detail ----
        "syn_max_tv_across_acceptance_sweep": s["syn_max_tv_across_acceptance_sweep"],
        "syn_max_z_over_floor": s["syn_max_z_over_floor"],
        "syn_mean_signed_excess": s["syn_mean_signed_excess"],
        "real_max_tv_across_acceptance_sweep": s["real_max_tv_across_acceptance_sweep"],
        "real_max_z_over_floor": s["real_max_z_over_floor"],
        "real_mean_signed_excess": s["real_mean_signed_excess"],
        # ---- K=7 chaining ----
        "max_tv_over_k7_positions_2to7": s["max_tv_over_k7_positions_2to7"],
        "k7_depth_max_z_over_floor": s["k7_depth_max_z_over_floor"],
        "k7_depth_tv_slope": s["k7_depth_tv_slope"],
        # ---- gates ----
        "self_tests_passed": st["n_pass"],
        "self_tests_total": st["n_tests"],
        "peak_gpu_mem_gb": s.get("peak_gpu_mem_gb", 0.0),
        "downstream_eval_exposure": 0.0 if pure_speed else s["private_quality_exposure"],
        "verdict_oneline": (
            "The spec-alive surgical-357 PRIVATE leaderboard acceptance shift is a "
            f"PURE SPEED risk with ZERO quality exposure: the deployed standard "
            f"rejection sampler reproduces target p at EVERY acceptance rate "
            f"(TV<=floor band at all {s['n_acceptance_points_pooled']} swept points, "
            f"0 exceedances; mean signed excess {s['mean_signed_excess_over_mu']:.5f}~0; "
            f"acc<->z corr {s['accept_z_correlation']:.3f}~0; G-test Bonferroni p="
            f"{s['gtest_bonferroni_global_pvalue']:.3f}, {100*s['gtest_frac_p_gt_05']:.0f}% of "
            f"p>0.05). K=7 spine: no error accumulation (max TV pos 2..7 "
            f"{s['max_tv_over_k7_positions_2to7']:.4f}, depth slope {s['k7_depth_tv_slope']:.1e}). "
            "So the ~4.3-24% private breach (denken #489 / kanna #504,#508) moves E[T]/TPS only."
            if pure_speed else
            f"QUALITY EXPOSURE DETECTED: private_quality_exposure={s['private_quality_exposure']:.5f}"
        ),
    })

    # Synthetic acceptance-sweep table (per case)
    syn_tbl = wandb.Table(columns=[
        "case", "vocab", "support", "M", "floor_mu", "floor_hi",
        "accept_min", "accept_max", "max_tv", "max_z", "mean_signed_excess",
        "all_within_band", "tv_accept_slope", "min_gtest_p"])
    for c in syn:
        syn_tbl.add_data(c["label"], c["vocab"], c["support"], c["M"],
                         c["floor_mu"], c["floor_hi"], c["accept_min"], c["accept_max"],
                         c["max_tv_deployed_vs_p"], c["max_z_over_floor"],
                         c["mean_signed_excess_over_mu"], c["all_within_band"],
                         c["tv_acceptance_slope"], c["min_gtest_pvalue"])

    # Per-draft ladder for the explicit private bracket (0.387 -> breach band):
    # shows TV flat as acceptance descends through the private OOD regime.
    ladder_tbl = wandb.Table(columns=[
        "case", "draft", "p_draft", "realized_accept", "tv_deployed_vs_p",
        "z_over_floor", "within_band", "gtest_pvalue"])
    for c in syn:
        if c["label"] in ("private_bracket", "geom_ladder_v48"):
            for r in c["rows"]:
                ladder_tbl.add_data(c["label"], r["draft"], r["p_draft"],
                                    r["realized_accept"], r["tv_deployed_vs_p"],
                                    r["z_over_floor"], r["within_band"], r["gtest_pvalue"])

    # Real reasoning summary table (top by max_z for transparency)
    real_tbl = wandb.Table(columns=[
        "id", "source", "support", "M", "floor_mu", "floor_hi",
        "max_tv", "max_z", "mean_signed_excess", "all_within_band", "min_gtest_p"])
    for c in sorted(real, key=lambda c: c["max_z_over_floor"], reverse=True)[:40]:
        m = c.get("meta", {})
        real_tbl.add_data(m.get("id", c["label"]), m.get("source", "?"), c["support"],
                          c["M"], c["floor_mu"], c["floor_hi"], c["max_tv_deployed_vs_p"],
                          c["max_z_over_floor"], c["mean_signed_excess_over_mu"],
                          c["all_within_band"], c["min_gtest_pvalue"])

    # K=7 depth-isolated table (TV at floor at every spine depth)
    k7_tbl = wandb.Table(columns=[
        "depth", "reached", "reach_frac", "tv_deployed_vs_p", "z_over_floor",
        "within_band", "gtest_pvalue"])
    for d in depth["per_depth"]:
        k7_tbl.add_data(d["depth"], d["reached"], d["reach_frac"], d["tv_deployed_vs_p"],
                        d["z_over_floor"], d["within_band"], d["gtest_pvalue"])

    wandb.log({
        "synthetic_acceptance_sweep": syn_tbl,
        "private_bracket_ladder": ladder_tbl,
        "real_reasoning_cases": real_tbl,
        "k7_depth_isolated": k7_tbl,
    })

    print("W&B run:", run.url)
    print("run id:", run.id)
    run.finish()


if __name__ == "__main__":
    main()
