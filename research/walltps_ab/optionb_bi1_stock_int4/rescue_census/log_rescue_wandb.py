"""PR #651 served recompute-RESCUE census -> W&B group `served-recompute-rescue-census-land`.

Reads rescue_census_result.json (no server, no recompute). Logs served_rescue_rate /
served_break_rate (overall + per K) with prompt-bootstrap CI, the on-AR-head (stark TF
analog) vs off-AR-tail decomposition, the tail-break r==s/r!=s split, the break-locus
severity (benign 0.0-nat ULP ties vs wider misses), and the byte-exact subset.
analysis_only=true, official_tps=0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

RESULT = HERE / "rescue_census_result.json"
FINAL = HERE / "rescue_census_final.json"

PR = 651
STARK_TF_BREAK_RATE = 0.0          # 0/14035 teacher-forced
N632_DIVERGENCE_PROMPT_FRAC = 0.844


def main() -> int:
    res = json.loads(RESULT.read_text())
    fin = json.loads(FINAL.read_text()) if FINAL.exists() else None
    hk = res["headline_k"]
    head = res["per_k"][f"k{hk}"]
    fhead = fin["per_k"][f"k{hk}"] if fin else {}
    fval = fhead.get("decode_path_validation") or {}

    run = wandb_logging.init_wandb_run(
        job_type="rescue_census",
        agent="land",
        name="land/served-recompute-rescue-census",
        group="served-recompute-rescue-census-land",
        notes=("PR#651: served-path recompute-RESCUE census. For every fired position (verify "
               "margin<0.5) in the #632 served Option-B BI=1 spec streams, execute stark #636's "
               "M=1 (spec-OFF) recompute in the SERVED context and check whether it lands on the "
               "served-AR reference token ar_ref_bi1[pos]. Correctness complement to #645/#648 "
               "coverage; served analog of stark TF rescued_break_rate=0/14035. Decisive = on-AR "
               "head break_rate (the population the online acceptor operates on)."),
        config={
            "pr": PR, "analysis_only": True, "official_tps": 0,
            "vllm": "0.22.0", "batch_invariant": 1, "max_num_seqs": 1,
            "recompute_engine": "spec_off_M1_AR (SENPAI_REFERENCE_MODE=1)",
            "tau_nat": res["tau"], "num_prompts": head["n_prompts"], "output_len": 512,
            "n_logprobs": 20, "served_reference": "ar_ref_bi1 served_spec_off_M1_AR (BASELINE.md L10)",
            "ks_censused": sorted(int(k[1:]) for k in res["per_k"]),
            "headline_k": hk, "stark_636_tf_rescued_break_rate": res["stark_tf_rescued_break_rate"],
        },
        tags=["optionb", "batch_invariant", "pr651", "rescue_census", "served", "identity"],
    )
    if run is None:
        print("WANDB disabled/unavailable; dumping summary only", flush=True)

    summary = {
        "rescue/headline_k": hk,
        "rescue/served_rescue_rate": head["served_rescue_rate"],
        "rescue/served_break_rate": head["served_break_rate"],
        "rescue/break_ci95_lo": head["served_break_rate_ci95_boot"][0],
        "rescue/break_ci95_hi": head["served_break_rate_ci95_boot"][1],
        "rescue/total_fires": head["total_fires"],
        "rescue/broken": head["broken"],
        # decisive identity number: on-AR head (stark TF analog 0/14035)
        "head/on_AR_break_rate": head["pre_div_break_rate"],
        "head/on_AR_breaks": head["pre_div_breaks"],
        "head/on_AR_fires": head["pre_div_fires"],
        # off-AR tail (counterfactual; dominated by prefix divergence)
        "tail/off_AR_break_rate": head["post_div_break_rate"],
        "tail/off_AR_breaks": head["post_div_breaks"],
        "tail/off_AR_fires": head["post_div_fires"],
        "tail/break_trajectory_divergence_r_eq_s": head["post_break_trajectory_divergence_r_eq_s"],
        "tail/break_genuine_flip_r_ne_s": head["post_break_genuine_flip_r_ne_s"],
        # break severity
        "locus/breaks_benign_ulp_tie": head["breaks_benign_ulp_tie"],
        "locus/breaks_wider_miss": head["breaks_wider_miss"],
        # byte-exact robustness
        "bx/break_rate": head["bx_break_rate"],
        "bx/breaks": head["bx_breaks"],
        "bx/fires": head["bx_fires"],
        "bx/n_sha_ok_prompts": head["n_sha_ok_prompts"],
        # literal strict-rule (recompute_margin>1e-6) verdict from rescue_analyze
        "decision/literal_strict_verdict": res["verdict"],
        "decision/literal_strict_basis": res["verdict_basis"],
    }

    if fin is not None:
        # refined head taxonomy + decode-path reinterpretation (the decisive numbers)
        summary.update({
            "decision/verdict": fin["verdict"],
            "decision/verdict_basis": fin["verdict_basis"],
            "decision/confident_off_AR_head_misses_all_K": fin["confident_off_AR_head_misses_all_K"],
            "head/confident_off_AR_miss_rate": fin["headline_on_AR_head_confident_miss_rate"],
            "head/confident_off_AR_misses": fhead.get("head_confident_off_AR_misses"),
            "head/ulp_ties_0nat": fhead.get("head_ulp_ties_0nat"),
            "head/wide_int4_quantum_ties": fhead.get("head_wide_int4_quantum_ties"),
            "head/a_pos_is_recompute_top2": fhead.get("head_a_pos_is_recompute_top2"),
            "head/a_pos_in_recompute_topN": fhead.get("head_a_pos_in_recompute_topN"),
            # decode-path validation of the wide head breaks (headline K)
            "validate/wide_decode_rescued_to_AR_artifact": fval.get("wide_decode_rescued_to_AR_artifact"),
            "validate/wide_decode_agrees_prefill": fval.get("wide_decode_agrees_prefill"),
            "validate/wide_decode_other": fval.get("wide_decode_other"),
            "validate/ar_ref_outlier_prompts": fval.get("ar_ref_outlier_prompts"),
            "validate/ar_ref_faithful_prompts": fval.get("ar_ref_faithful_prompts"),
        })
    else:
        summary["decision/verdict"] = res["verdict"]
        summary["decision/verdict_basis"] = res["verdict_basis"]

    if run is not None:
        import wandb
        cols = ["K", "total_fires", "rescued", "broken", "served_break_rate",
                "break_ci95_lo", "break_ci95_hi", "on_AR_head_break_rate",
                "off_AR_tail_break_rate", "tail_break_r_eq_s", "tail_break_r_ne_s",
                "breaks_benign_ulp_tie", "breaks_wider_miss", "bx_break_rate"]
        tbl = wandb.Table(columns=cols)
        for k in sorted(int(kk[1:]) for kk in res["per_k"]):
            r = res["per_k"][f"k{k}"]
            tbl.add_data(k, r["total_fires"], r["rescued"], r["broken"],
                         r["served_break_rate"], r["served_break_rate_ci95_boot"][0],
                         r["served_break_rate_ci95_boot"][1], r["pre_div_break_rate"],
                         r["post_div_break_rate"], r["post_break_trajectory_divergence_r_eq_s"],
                         r["post_break_genuine_flip_r_ne_s"], r["breaks_benign_ulp_tie"],
                         r["breaks_wider_miss"], r["bx_break_rate"])
        run.log({"per_k_rescue_census": tbl})

        # break-locus table (headline K) -- the audit trail for every break
        bcols = ["id", "pos", "pre_div", "verify_margin", "recompute_margin",
                 "a_pos", "r_pos", "s_pos", "r_eq_s", "benign_ulp_tie"]
        btbl = wandb.Table(columns=bcols)
        for b in head["break_loci"][:1000]:
            btbl.add_data(b["id"], b["pos"], int(b["pre_div"]), b["verify_margin"],
                          b["recompute_margin"], b["a_pos"], b["r_pos"], b["s_pos"],
                          int(b["r_eq_s"]), int(b["benign_ulp_tie"]))
        run.log({"break_loci_headline": btbl})

        # decode-path validation table: how each K's on-AR-head wide breaks resolve
        if fin is not None:
            vcols = ["K", "head_breaks", "head_fires", "ulp_ties_0nat",
                     "wide_int4_quantum_ties", "a_pos_is_recompute_top2",
                     "confident_off_AR_misses", "decode_rescued_artifact",
                     "decode_agrees_prefill", "decode_other", "ar_ref_outlier_prompts"]
            vtbl = wandb.Table(columns=vcols)
            for k in sorted(int(kk[1:]) for kk in fin["per_k"]):
                v = fin["per_k"][f"k{k}"]
                d = v.get("decode_path_validation") or {}
                vtbl.add_data(k, v["on_AR_head_breaks"], v["on_AR_head_fires"],
                              v["head_ulp_ties_0nat"], v["head_wide_int4_quantum_ties"],
                              v["head_a_pos_is_recompute_top2"], v["head_confident_off_AR_misses"],
                              d.get("wide_decode_rescued_to_AR_artifact"),
                              d.get("wide_decode_agrees_prefill"), d.get("wide_decode_other"),
                              d.get("ar_ref_outlier_prompts"))
            run.log({"decode_path_validation": vtbl})

        wandb_logging.log_summary(run, summary, step=PR)
        wandb_logging.log_json_artifact(
            run, name="served_rescue_census_651", artifact_type="analysis",
            data={"summary": summary, "result": res, "final": fin})
        url = getattr(run, "url", "")
        rid = getattr(run, "id", "")
        wandb_logging.finish_wandb(run)
        print(f"[wandb] rescue census id={rid} url={url}", flush=True)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
