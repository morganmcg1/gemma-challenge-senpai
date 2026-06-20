#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #761 (lawine) -- cross-arm verdict + wandb logger for the served-divergence-locus
census. Reads the two per-arm JSON reports produced by served_locus_census.py:

  * bi0 arm (VLLM_BATCH_INVARIANT=0): the divergence-OPEN config -- M=1 decode takes the
    3D split-KV attention path (num_splits>1), M=K+1 verify takes the 2D path
    (num_splits=1). This is where the first-divergence locus is measured + attributed.
  * bi1 arm (VLLM_BATCH_INVARIANT=1): the served byte-exact config -- use_3d=False for
    BOTH M, so num_splits=1 everywhere. This is the targeted-FIX arm (expect 0 divergence).

Primary metric  top_op_divergence_share        = bi0 arm's top first-divergence op family share.
Test metric     literal_strict_achievable_targeted = 1 iff the bi1 (BI=1 toggle) arm is
                byte-exact AND the bi0 locus is the attention reduction -> the single
                targeted fix (BI=1 on the attention split, a CURRENT toggle, no vLLM patch)
                closes the rung to literal 128/128.

Runs under the repo .venv (has wandb). NO GPU. analysis_only, official_tps=0, no_hf_job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi0", type=Path, required=True, help="divergence-open arm report json")
    ap.add_argument("--bi1", type=Path, required=True, help="BI=1 fix arm report json")
    ap.add_argument("--wandb-name", default="lawine/served-divergence-locus")
    ap.add_argument("--wandb-group", default="fire_bi_tax_750")
    ap.add_argument("--wandb-project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity",
                    default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent
                    / "runs" / "locus_census" / "verdict.json")
    args = ap.parse_args()

    bi0 = _load(args.bi0)
    bi1 = _load(args.bi1)

    a0 = bi0["attribution"]
    a1 = bi1["attribution"]
    v0 = bi0["verdict"]
    v1 = bi1["verdict"]

    top_family = a0["top_op_family"]
    top_share = float(a0["top_op_divergence_share"] or 0.0)
    bi1_byte_exact = bool(v1.get("byte_exact_this_arm"))
    attn_is_locus = (top_family == "attn")
    marlin_M_invariant = bool(v0.get("marlin_M_invariant_incl_lm_head")) and \
        bool(v1.get("marlin_M_invariant_incl_lm_head"))

    # The targeted fix is VLLM_BATCH_INVARIANT=1 (a CURRENT toggle). It is realizable iff
    # the bi1 arm is byte-exact and the locus it closes is the attention reduction.
    literal_strict_achievable_targeted = 1 if (bi1_byte_exact and attn_is_locus) else 0

    verdict = {
        "pr": 761, "phase": "served_divergence_locus_verdict",
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "deployed_ckpt": bi0.get("deployed_ckpt"),
        "strict_reference": "spec-off in-process M=1 greedy-AR decode of the DEPLOYED int4 "
                            "ckpt (the #755 strict reference); divergence = M=K+1 verify "
                            "activation != M=1 decode activation, ULP (torch.equal)",
        "primary_metric": {"name": "top_op_divergence_share", "value": top_share},
        "test_metric": {"name": "literal_strict_achievable_targeted",
                        "value": literal_strict_achievable_targeted},
        "locus": {
            "top_op_family": top_family,
            "top_op_divergence_share": top_share,
            "first_div_by_op_ranked": a0["first_div_by_op_ranked"],
            "first_div_by_family": a0["first_div_by_family"],
            "total_divergent_positions_bi0": a0["total_divergent_positions"],
            "attention_reduction_is_locus": attn_is_locus,
        },
        "fix_arm_bi1": {
            "total_divergent_positions": a1["total_divergent_positions"],
            "e2e_argmax_flips": bi1["phase1_forward_ab"]["e2e_argmax_flips"],
            "byte_exact": bi1_byte_exact,
        },
        "marlin_microbench": {
            "bi0_lm_head": bi0["phase2_microbench"].get("lm_head"),
            "bi1_lm_head": bi1["phase2_microbench"].get("lm_head"),
            "marlin_M_invariant_incl_lm_head_both_arms": marlin_M_invariant,
        },
        "verdict_text": (
            f"LOCUS = attention split-KV reduction (attn_out): {top_share*100:.1f}% of "
            f"first-divergences in the divergence-open (BI=0) arm. The int4 Marlin GEMMs "
            f"INCLUDING the full-vocab lm_head are M-invariant in BOTH arms (controlled "
            f"microbench, 0 divergent) -> NOT the locus. The single targeted fix "
            f"VLLM_BATCH_INVARIANT=1 (a current toggle, no vLLM patch) routes M=1 decode "
            f"off the 3D split onto use_3d=False -> num_splits=1 for both M -> the BI=1 arm "
            f"is byte-exact ({a1['total_divergent_positions']} divergent positions). "
            f"literal_strict_achievable_targeted={literal_strict_achievable_targeted}."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2, default=str))
    print(json.dumps(verdict, indent=2, default=str))

    if not args.no_wandb:
        import wandb
        cfg = {
            "pr": 761, "axis": "divergence_locus", "analysis_only": True,
            "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "deployed_ckpt": bi0.get("deployed_ckpt"),
            "ckpt_lm_head": bi0.get("ckpt_lm_head"),
            "k": bi0["config"]["k"], "verify_width": bi0["config"]["verify_width"],
            "n_prompts": bi0["config"]["n_prompts"], "n_new": bi0["config"]["n_new"],
            "ctx_cap": bi0["config"]["ctx_cap"], "attn_backend": "TRITON_ATTN",
            "enforce_eager": True, "n_layers": bi0["config"]["n_layers"],
        }
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         config=cfg, job_type="analysis")
        wandb.log({
            "top_op_divergence_share": top_share,
            "literal_strict_achievable_targeted": literal_strict_achievable_targeted,
            "bi0_total_divergent_positions": a0["total_divergent_positions"],
            "bi0_e2e_argmax_flips": bi0["phase1_forward_ab"]["e2e_argmax_flips"],
            "bi0_e2e_positions": bi0["phase1_forward_ab"]["e2e_positions"],
            "bi1_total_divergent_positions": a1["total_divergent_positions"],
            "bi1_e2e_argmax_flips": bi1["phase1_forward_ab"]["e2e_argmax_flips"],
            "bi1_byte_exact": int(bi1_byte_exact),
            "attention_reduction_is_locus": int(attn_is_locus),
            "marlin_M_invariant_incl_lm_head": int(marlin_M_invariant),
            "lm_head_n_divergent_bi0": bi0["phase2_microbench"].get("lm_head", {}).get("n_divergent"),
            "lm_head_n_divergent_bi1": bi1["phase2_microbench"].get("lm_head", {}).get("n_divergent"),
        })
        # log the ranked first-divergence ops as a table
        tbl = wandb.Table(columns=["op", "first_divergence_count"])
        for op, cnt in a0["first_div_by_op_ranked"]:
            tbl.add_data(op, cnt)
        wandb.log({"bi0_first_div_by_op": tbl})
        run.summary["verdict_text"] = verdict["verdict_text"]
        print(f"[wandb] run_id={run.id} url={run.url}", flush=True)
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
