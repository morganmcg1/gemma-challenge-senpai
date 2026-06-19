#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Log the PR #748 served byte-exact transfer result to W&B (0-GPU, wandb-capable venv).

Reads runs/analysis.json (from analyze.py) + the 4 arm_summary.json files and creates ONE
run in group ``strict-clean-served-byteexact-land``. Emits the deliverable: the served BI=1
batched-verify spec strict-#319 greedy identity (N/128 -> fraction = primary_metric), its
single-stream TPS (test_metric), the BI=0 control identity, and the BI=1-vs-BI=0 decode tax
(the price of byte-exactness on the batched-verify path). Required flags per the PR card:
analysis_only=1, official_tps=0, no_hf_job=1, fires=0, the verdict string, decode_tax_pct.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=HERE / "runs" / "analysis.json")
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="strict-clean-served-byteexact-land")
    ap.add_argument("--name", default="land/route-b-served-byteexact")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    a = json.loads(args.analysis.read_text())
    arms = a["arms"]
    bi1_spec = arms["bi1_spec"]

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config={
            "pr": 748, "phase": "strict_clean_served_byteexact",
            "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "served_model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "attn_backend": "TRITON_ATTN",
            "engine": "vllm-0.22.0 v1 api_server (online, CUDA graphs, NOT enforce_eager)",
            "spec_method": "ngram",
            "num_speculative_tokens": 6,
            "n_prompts": bi1_spec.get("n_prompts"),
            "output_len": bi1_spec.get("output_len"),
            "operative_lever": "VLLM_BATCH_INVARIANT (num_splits=1 attention)",
            "builds_on": "land#743 rwk498ve (offline byte-exact-fixable proof)",
            "locked_tps_anchor": a["tps"]["locked_tps"],
            "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        },
    )

    bi1 = a["bi1_identity"]
    bi0 = a["bi0_control_identity"]
    xc = a.get("ar_bi_xcheck_identity", {})
    det = a.get("determinism_control")
    eager = a.get("eager_mechanism")
    tps = a["tps"]
    flat = {
        # required PR flags
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "verdict": a["verdict"],
        # primary + test metrics
        "served_spec_bi1_greedy_identity": a["primary_metric"]["value"],
        "served_spec_bi1_greedy_identity_n_match": bi1["n_match"],
        "served_spec_bi1_greedy_identity_n_total": bi1["n_total"],
        "served_spec_bi1_tps": a["test_metric"]["value"],
        # control
        "served_spec_bi0_greedy_identity": bi0["frac"],
        "served_spec_bi0_greedy_identity_n_match": bi0["n_match"],
        "served_spec_bi0_greedy_identity_n_total": bi0["n_total"],
        "bi0_control_diverges": int(bi0["n_match"] < bi0["n_total"]),
        # per-token near-tie flip hazard (the physically meaningful quantity): BI=1 gives NO
        # measurable reduction vs BI=0 -> the 512-tok rollout is fragile to ANY reduction-order
        # perturbation, and BI=1 aligns only one reduction site.
        "bi1_spec_per_token_flip_hazard": bi1.get("per_token_flip_hazard"),
        "bi0_spec_per_token_flip_hazard": bi0.get("per_token_flip_hazard"),
        "ar_bi_toggle_per_token_flip_hazard": xc.get("per_token_flip_hazard"),
        "ar_bi_toggle_identity_n_match": xc.get("n_match"),
        "ar_bi_toggle_identity_frac": xc.get("frac"),
        "bi1_minus_bi0_hazard_delta": (
            (bi1.get("per_token_flip_hazard") or 0) - (bi0.get("per_token_flip_hazard") or 0)),
        # tps deliverables
        "tps_bi1_spec": tps["bi1_spec"], "tps_bi0_spec": tps["bi0_spec"],
        "tps_bi1_arref": tps["bi1_arref"], "tps_bi0_arref": tps["bi0_arref"],
        "decode_tax_pct_bi1_vs_bi0": tps["decode_tax_pct_bi1_vs_bi0"],
        "bi1_spec_clears_126378": int(tps["bi1_spec_clears_126378"]),
        "bi0_spec_clears_126378": int(tps["bi0_spec_clears_126378"]),
        "locked_tps": tps["locked_tps"],
        # peak mem
        "bi1_spec_peak_gpu_mem_mib": bi1_spec.get("peak_gpu_mem_mib"),
        "transfers": int(a["verdict"] == "SERVED_BYTEEXACT_TRANSFERS"),
    }
    if det is not None:
        di = det["identity"]
        flat.update({
            # determinism floor: same BI=1 AR config run twice. 128/128 => stack is bit-reproducible
            # within-config, so the 21/128 spec result is a REAL per-step reduction-order divergence,
            # not intrinsic run-to-run noise. <128/128 => served greedy is not reproducible at all.
            "determinism_floor_n_match": di["n_match"],
            "determinism_floor_n_total": di["n_total"],
            "determinism_floor_frac": di["frac"],
            "determinism_floor_per_token_flip_hazard": di.get("per_token_flip_hazard"),
            "stack_deterministic_within_config": int(det["deterministic"]),
            "determinism_rep_tps": det["tps_rep"],
        })
    if eager is not None:
        ei = eager["identity"]
        flat.update({
            "eager_bi1_spec_identity_n_match": ei["n_match"],
            "eager_bi1_spec_identity_n_total": ei["n_total"],
            "eager_bi1_spec_identity_frac": ei["frac"],
            "eager_bi1_spec_per_token_flip_hazard": ei.get("per_token_flip_hazard"),
            "eager_transfers": int(eager["transfers_eager"]),
            "residual_mechanism": eager["mechanism"],
            "eager_tps_spec": eager["tps_spec"], "eager_tps_arref": eager["tps_arref"],
        })
    # ADVISOR REFRAME: self-consistency (tau=0.3) + PPL neutrality => benign ULP-tie classification
    sc = a.get("self_consistency")
    ppl = a.get("ppl")
    rf = a.get("reframe", {})
    flat.update({
        "literal_served_byteexact": int(rf.get("literal_served_byteexact", 0)),
        "self_consistent_not_byteexact": int(rf.get("self_consistent_not_byteexact", 0)),
        "predominantly_benign_ulp_marginal_tail": int(
            rf.get("predominantly_benign_ulp_marginal_confident_tail", 0)),
        "confident_flip_frac": rf.get("confident_flip_frac"),
        "confident_flips_all_marginal_3ulp": int(rf.get("confident_flips_all_marginal_3ulp", 0)),
        "ppl_bi_neutral_reframe": (None if rf.get("ppl_bi_neutral") is None
                                   else int(bool(rf.get("ppl_bi_neutral")))),
        "residual_class": rf.get("residual_class"),
        "named_residual_subop": rf.get("named_residual_subop"),
        "honest_read": rf.get("honest_read"),
        "bi1_necessary_not_sufficient": int(rf.get("bi1_necessary_not_sufficient_for_literal_byteexact", 0)),
    })
    if sc is not None:
        flat.update({
            "selfconsist_tau": sc["tau"],
            "selfconsist_n_diverging": sc["n_diverging"],
            "selfconsist_n_probed": sc["n_probed"],
            "selfconsist_confident_genuine_flips": sc["confident_genuine_flips"],
            "selfconsist_max_gap_nat": sc["max_gap_nat"],
            "selfconsist_pass": int(sc["self_consistent_pass"]),
            "selfconsist_frac_pair_is_model_top2": sc["frac_pair_is_model_top2"],
        })
    if ppl is not None:
        flat.update({
            "ppl_bi1": ppl["ppl_bi1"],
            "ppl_bi0": ppl["ppl_bi0"],
            "ppl_abs_delta_pct_bi1_vs_bi0": ppl["ppl_abs_delta_pct_bi1_vs_bi0"],
            "ppl_neutral_bi": int(ppl["ppl_neutral_bi"]),
            "ppl_deployed_anchor": ppl["deployed_pruned_head_anchor"],
        })
    run.summary.update(flat)

    # arms table
    run.log({"arms": wandb.Table(
        columns=["arm", "bi", "spec", "output_tps", "n_prompts", "output_len", "peak_mem_mib"],
        data=[[k, v["bi"], v["spec"], v["output_tps"], v["n_prompts"], v["output_len"],
               v.get("peak_gpu_mem_mib")] for k, v in arms.items()])})

    # diverging-prompt forensic table (BI=1 primary + BI=0 control)
    div_cols = ["arm_pair", "id", "index", "first_div_pos", "len_spec", "len_ref",
                "spec_tok", "ref_tok"]
    div_rows = []
    for d in bi1["diverging"]:
        div_rows.append(["bi1_spec_vs_bi1_ar", d["id"], d["index"], d["first_div_pos"],
                         d["len_spec"], d["len_ref"], d["spec_tok"], d["ref_tok"]])
    for d in bi0["diverging"][:64]:
        div_rows.append(["bi0_spec_vs_bi0_ar", d["id"], d["index"], d["first_div_pos"],
                         d["len_spec"], d["len_ref"], d["spec_tok"], d["ref_tok"]])
    if div_rows:
        run.log({"diverging_prompts": wandb.Table(columns=div_cols, data=div_rows)})

    # self-consistency gap histogram (the bf16-ULP-quantized onset margins; all < tau=0.3 => benign)
    sc = a.get("self_consistency")
    if sc is not None and sc.get("gap_nat_histogram"):
        run.log({"selfconsist_gap_nat_hist": wandb.Table(
            columns=["gap_nat", "count"],
            data=[[float(k), v] for k, v in sc["gap_nat_histogram"].items()])})

    print(f"[wandb] run {run.id}  group={args.group}")
    print(f"[wandb] VERDICT={a['verdict']}  residual_class={rf.get('residual_class')}  "
          f"self_consistent_not_byteexact={rf.get('self_consistent_not_byteexact')}")
    print(f"[wandb] bi1_identity={bi1['n_match']}/{bi1['n_total']}  "
          f"confident_flips={(sc or {}).get('confident_genuine_flips')}  "
          f"ppl_delta%={(ppl or {}).get('ppl_abs_delta_pct_bi1_vs_bi0')}  "
          f"bi1_tps={tps['bi1_spec']:.3f}  decode_tax={tps['decode_tax_pct_bi1_vs_bi0']:.2f}%")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
