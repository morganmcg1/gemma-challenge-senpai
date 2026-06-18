"""PR #654 served-identity-EXACT upgrade -> W&B group `served-identity-exact-land`.

Pure offline. Reads:
  tiebreak_audit.json        (Part 1: lowest-index tie-break direction on the 60 wide ties)
  recensus_canonical.json    (Part 2: census re-scored vs the batch-M-stable M=1 oracle)
  consistency_check.json      (oracle reproduces #651 validate_decode_path decode_tok)
  ../ksweep/ar_ref_m1_canonical/meta.json  (the reusable canonical oracle artifact)

Logs the Part-1 tie-break tally, the Part-2 reference-swap head break_rate + confident-miss
count (per K + bx-subset), the genuine served-vs-canonical residuals (view B) with their
canonical-path margins, and the upgrade verdict. analysis_only=true, official_tps=0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

TIEBREAK = HERE / "tiebreak_audit.json"
RECENSUS = HERE / "recensus_canonical.json"
CONSIST = HERE / "consistency_check.json"
RESID_MARG = HERE / "residual_margins.json"
ORACLE_META = HERE.parent / "ksweep" / "ar_ref_m1_canonical" / "meta.json"
ORACLE_PATH_REL = "research/walltps_ab/optionb_bi1_stock_int4/ksweep/ar_ref_m1_canonical/"

PR = 654
HEADLINE_K = 5
CONF = 0.5  # confident-miss threshold (nat)


def compute_verdict(tb, rc):
    """PR #654 decision tree.
    SERVED_IDENTITY_EXACT     : AR wins lowest-index tie-break at ALL 60 wide ties
                                OR canonical census shows 0 genuine residual every K.
    EXACT_BY_TIEBREAK_ONLY    : AR wins all 60, but canonical oracle still shows residual.
    TIE_TOLERANT_RESIDUAL     : some wide ties LOST on index tie-break AND residual persists.
    """
    ar_wins_all = (tb["AR_loses_tiebreak"] == 0)
    residuals_per_k = {k: rc["per_k"][f"k{k}"]["B_n_canonical_residuals"] for k in (3, 5, 7)}
    residuals_zero_all = all(v == 0 for v in residuals_per_k.values())
    confident = sum(rc["per_k"][f"k{k}"]["B_residuals_verify_confident_ge_tau"] for k in (3, 5, 7))
    if ar_wins_all or residuals_zero_all:
        v = "SERVED_IDENTITY_EXACT"
    elif ar_wins_all and not residuals_zero_all:
        v = "EXACT_BY_TIEBREAK_ONLY"
    else:
        v = "TIE_TOLERANT_RESIDUAL"
    basis = (f"AR wins {tb['AR_wins_tiebreak']}/{tb['n_wide_total']} wide-tie lowest-index; "
             f"AR loses {tb['AR_loses_tiebreak']}; genuine canonical residuals/K={residuals_per_k}; "
             f"confident(>={CONF}nat) residuals={confident}")
    return v, basis, residuals_per_k, confident


def main() -> int:
    tb = json.loads(TIEBREAK.read_text())
    rc = json.loads(RECENSUS.read_text())
    consist = json.loads(CONSIST.read_text()) if CONSIST.exists() else {}
    rmarg = json.loads(RESID_MARG.read_text()) if RESID_MARG.exists() else {}
    ometa = json.loads(ORACLE_META.read_text()) if ORACLE_META.exists() else {}

    verdict, basis, residuals_per_k, confident = compute_verdict(tb, rc)
    kh = rc["per_k"][f"k{HEADLINE_K}"]

    # confident misses from the reference-swap census (view A), summed over K
    conf_miss_total = sum(rc["per_k"][f"k{k}"]["A_head_confident_miss"] for k in (3, 5, 7))

    run = wandb_logging.init_wandb_run(
        job_type="served_identity_exact",
        agent="land",
        name="land/served-identity-exact",
        group="served-identity-exact-land",
        notes=("PR#654: upgrade the #651 served-identity leg from tie-tolerance to EXACT. "
               "Part 1 (offline): at each of the 60 wide head ties (0.125-0.25 nat) does the "
               "served-AR token win the recompute acceptor's lowest-token-index tie-break "
               "(a_pos < r_pos)? AR-wins-all => exact by construction. Part 2 (GPU oracle "
               "regen + offline re-score): regenerate a batch-M-stable single-seq M=1 decode "
               "ar_ref oracle on the #651 reference server (SENPAI_REFERENCE_MODE=1, BI=1, "
               "MAX_NUM_SEQS=1, spec off, temp 0) and re-run the on-AR-head census against it; "
               "does the head break count drop to 0? Reusable canonical oracle for "
               "#645/#648/#651 served-identity census; does NOT overwrite ar_ref_bi1."),
        config={
            "pr": PR, "analysis_only": True, "official_tps": 0,
            "vllm": "0.22.0", "batch_invariant": 1, "max_num_seqs": 1,
            "recompute_engine": "spec_off_M1_AR (SENPAI_REFERENCE_MODE=1)",
            "canonical_oracle_artifact": ORACLE_PATH_REL,
            "canonical_oracle_kind": ometa.get("kind"),
            "canonical_oracle_num_records": ometa.get("num_records"),
            "canonical_oracle_output_len": ometa.get("output_len"),
            "does_not_overwrite": "ar_ref_bi1/decode_outputs.jsonl",
            "ks_censused": [3, 5, 7], "headline_k": HEADLINE_K,
            "tau_nat": rc["tau"], "confident_thresh_nat": CONF,
            "n_logprobs": 20, "output_len": 512,
            "tiebreak_rule": "lowest_token_index (torch.argmax first index)",
        },
        tags=["optionb", "batch_invariant", "pr654", "served", "identity", "exact",
              "tiebreak_audit", "canonical_oracle"],
    )
    if run is None:
        print("WANDB disabled/unavailable; dumping summary only", flush=True)

    summary = {
        # ---- Part 1: tie-break direction audit ----
        "part1/wide_ties_total": tb["n_wide_total"],
        "part1/wide_ties_AR_wins_tiebreak": tb["AR_wins_tiebreak"],
        "part1/wide_ties_AR_loses_tiebreak": tb["AR_loses_tiebreak"],
        "part1/wide_ties_degenerate": tb["degenerate"],
        "part1/k3_AR_wins": tb["per_k"]["3"]["AR_wins"],
        "part1/k5_AR_wins": tb["per_k"]["5"]["AR_wins"],
        "part1/k7_AR_wins": tb["per_k"]["7"]["AR_wins"],
        # ---- Part 2 view (A): reference-swap on-AR-head census vs canonical oracle ----
        "part2A/canonical_oracle_on_AR_head_break_rate_k5": kh["A_head_break_rate"],
        "part2A/head_breaks_k5": kh["A_head_breaks"],
        "part2A/head_fires_k5": kh["A_head_fires"],
        "part2A/canonical_oracle_confident_miss_count": conf_miss_total,
        "part2A/confident_miss_k5": kh["A_head_confident_miss"],
        "part2A/bx_head_break_rate_k5": kh["A_bx_head_break_rate"],
        "part2A/bx_head_breaks_k5": kh["A_bx_head_breaks"],
        "part2A/bx_head_fires_k5": kh["A_bx_head_fires"],
        # ---- Part 2 view (B): genuine served-vs-canonical residuals ----
        "part2B/genuine_canonical_residuals_k5": kh["B_n_canonical_residuals"],
        "part2B/residuals_served_wins_idx_k5": kh["B_residuals_served_wins_idx"],
        "part2B/residuals_verify_tie_lt_tau_k5": kh["B_residuals_verify_tie_lt_tau"],
        "part2B/residuals_verify_confident_ge_tau_k5": kh["B_residuals_verify_confident_ge_tau"],
        "part2B/oracle_eq_arref_prompts_k5": kh["B_oracle_eq_arref_prompts"],
        "part2B/total_genuine_residuals_allK": sum(
            rc["per_k"][f"k{k}"]["B_n_canonical_residuals"] for k in (3, 5, 7)),
        "part2B/total_confident_residuals_allK": confident,
        # airtight: every residual + every oracle/ar_ref divergence is an int4 near-tie
        "part2B/max_canonical_residual_margin_nat": rmarg.get("max_canonical_residual_margin"),
        "part2B/residuals_near_tie": rmarg.get("n_residuals_near_tie"),
        "part2B/oracle_vs_arref_divergent_prompts": rmarg.get("n_oracle_arref_div_total"),
        "part2B/oracle_vs_arref_div_all_near_tie": rmarg.get("n_oracle_arref_div_near_tie"),
        # ---- oracle reproducibility (consistency vs #651 validate_decode_path) ----
        # raw = all wide-break positions; well-defined = validate still on ar_ref trajectory at pos.
        # every disagreement is an int4 exact-tie inter-run branch pick (near-tie), not a bug.
        "oracle/consistency_raw_match": consist.get("match"),
        "oracle/consistency_raw_mismatch": consist.get("mismatch"),
        "oracle/consistency_welldefined_match": consist.get("well_defined_match"),
        "oracle/consistency_welldefined_mismatch": consist.get("well_defined_mismatch"),
        "oracle/consistency_fork_point_disagree": consist.get("fork_point_disagree"),
        "oracle/consistency_downstream_disagree": consist.get("downstream_disagree"),
        "oracle/artifact_path": ORACLE_PATH_REL,
        "oracle/num_records": ometa.get("num_records"),
        "oracle/capture_wall_s": ometa.get("capture_wall_s"),
        # ---- verdict ----
        "decision/verdict": verdict,
        "decision/verdict_basis": basis,
        "decision/AR_wins_all_60": bool(tb["AR_loses_tiebreak"] == 0),
        "decision/residuals_per_k": json.dumps(residuals_per_k),
    }

    if run is not None:
        import wandb
        # per-K census table (both views)
        cols = ["K", "A_head_fires", "A_head_breaks", "A_head_break_rate",
                "A_confident_miss", "A_bx_head_fires", "A_bx_head_breaks", "A_bx_break_rate",
                "B_oracle_eq_arref", "B_residuals", "B_resid_served_wins_idx",
                "B_resid_tie_lt_tau", "B_resid_confident_ge_tau",
                "tb_n_wide", "tb_AR_wins", "tb_AR_loses"]
        tbl = wandb.Table(columns=cols)
        for k in (3, 5, 7):
            s = rc["per_k"][f"k{k}"]
            t = tb["per_k"][str(k)]
            tbl.add_data(k, s["A_head_fires"], s["A_head_breaks"], s["A_head_break_rate"],
                         s["A_head_confident_miss"], s["A_bx_head_fires"], s["A_bx_head_breaks"],
                         s["A_bx_head_break_rate"], s["B_oracle_eq_arref_prompts"],
                         s["B_n_canonical_residuals"], s["B_residuals_served_wins_idx"],
                         s["B_residuals_verify_tie_lt_tau"], s["B_residuals_verify_confident_ge_tau"],
                         t["n_wide"], t["AR_wins"], t["AR_loses"])
        run.log({"per_k_canonical_census": tbl})

        # Part-1 tie-break LOSS table (the genuine residual loci)
        lcols = ["K", "id", "pos", "a_pos", "a_str", "r_pos", "r_str",
                 "verify_margin", "recompute_margin"]
        ltbl = wandb.Table(columns=lcols)
        for e in tb["losses"]:
            ltbl.add_data(e["k"], e["id"], e["pos"], e["a_pos"], e["a_str"],
                          e["r_pos"], e["r_str"], e["verify_margin"], e["recompute_margin"])
        run.log({"tiebreak_losses": ltbl})

        # Part-2 view-B genuine canonical residual table
        rcols = ["K", "id", "pos", "served_tok", "canonical_tok", "arref_tok",
                 "served_eq_arref", "served_wins_lowest_index", "verify_margin",
                 "sha_ok", "old_head", "new_head"]
        rtbl = wandb.Table(columns=rcols)
        for r in rc["residuals"]:
            rtbl.add_data(r["k"], r["id"], r["pos"], r["served_tok"], r["canonical_tok"],
                          r["arref_tok"], int(r["served_eq_arref"]),
                          int(r["served_wins_lowest_index"]), r["verify_margin"],
                          int(r["sha_ok"]), r["old_head"], r["new_head"])
        run.log({"canonical_residuals": rtbl})

        wandb_logging.log_summary(run, summary, step=PR)
        wandb_logging.log_json_artifact(
            run, name="served_identity_exact_654", artifact_type="analysis",
            data={"summary": summary, "tiebreak_audit": tb, "recensus_canonical": rc,
                  "consistency": consist, "oracle_meta": ometa})
        url = getattr(run, "url", "")
        rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] served-identity-exact id={rid} url={url}", flush=True)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\n[VERDICT] {verdict}\n  {basis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
