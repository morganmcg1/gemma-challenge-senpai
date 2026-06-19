#!/usr/bin/env python3
"""Log the PR #743 strict-clean attention-locus-pin A/B to wandb.

Two-phase: locus_pin.py runs on the vLLM-0.22.0 GPU venv and writes
locus_bi0.json (VLLM_BATCH_INVARIANT=0) and locus_bi1.json
(VLLM_BATCH_INVARIANT=1). This 0-GPU logger reads both, creates ONE run in
group ``strict-clean-locus-pin``, and emits the deliverable:

  locus (layer+op), magnitude (max bitdiff / max logits delta),
  per-position argmax-flip count, and the byte-exact-fixability verdict.

Classification (instr 4): bi0 = M=1 decode runs 3D split-KV, M=K verify runs
2D one-shot -> different reduction order. bi1 forces M=1 to num_splits=1 (2D),
matching the verify path. If attn_out bitdiff COLLAPSES to byte-exact under
bi1, the locus is a deterministic reduction-ORDER artifact (byte-exact
fixable). If it persists, it is a genuine numeric difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

OPS = ["qkv", "attn_q", "attn_k", "attn_v", "attn_out", "oproj_out"]


def _flat(prefix, d, out):
    for k, val in d.items():
        key = f"{prefix}{k}"
        if isinstance(val, dict):
            _flat(key + "/", val, out)
        elif isinstance(val, bool):
            out[key] = int(val)
        elif isinstance(val, (int, float)):
            out[key] = val
    return out


def _arm_summary(tag, res, flat):
    """Flatten the headline scalars for one arm (bi0/bi1)."""
    loc = res.get("first_divergent_locus") or {}
    flat[f"{tag}/first_locus_layer"] = loc.get("layer", -99)
    flat[f"{tag}/first_locus_op_attn_out"] = int(loc.get("op") == "attn_out")
    flat[f"{tag}/first_locus_frac_bitdiff"] = loc.get("frac_bitdiff", 0.0)
    flat[f"{tag}/first_locus_max_abs"] = loc.get("max_abs", 0.0)
    flat[f"{tag}/e2e_argmax_flips"] = res.get("e2e_argmax_flips", 0)
    flat[f"{tag}/e2e_argmax_flip_rate"] = res.get("e2e_argmax_flip_rate", 0.0)
    flat[f"{tag}/e2e_positions"] = res.get("e2e_positions", 0)
    flat[f"{tag}/e2e_logprob_bitdiff_rate"] = res.get("e2e_logprob_bitdiff_rate", 0.0)
    flat[f"{tag}/ar_vs_ar_token_identity"] = res.get("ar_vs_ar_token_identity", -1.0)
    _flat(f"{tag}/prelmhead/", res.get("prelmhead", {}), flat)
    for op in OPS:
        c = (res.get("layer0_chain") or {}).get(op, {})
        flat[f"{tag}/layer0/{op}/frac_bitdiff"] = c.get("frac_bitdiff", 0.0)
        flat[f"{tag}/layer0/{op}/max_abs"] = c.get("max_abs", 0.0)
    flat[f"{tag}/n_decode_positions"] = res.get("n_decode_positions", 0)
    flat[f"{tag}/peak_mem_mib"] = res.get("peak_mem_mib", 0.0)
    return flat


def _attn_out_layer_table(res):
    """Per-layer attn_out frac_bitdiff/max_abs across all layers (one arm)."""
    plo = res.get("per_layer_op") or {}
    rows = []
    for li in sorted((int(k) for k in plo if int(k) >= 0)):
        cell = plo[str(li)].get("attn_out", {})
        rows.append([li, cell.get("frac_bitdiff", 0.0), cell.get("max_abs", 0.0),
                     cell.get("n_bitdiff", 0), cell.get("n", 0)])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi0", type=Path, required=True)
    ap.add_argument("--bi1", type=Path, default=None)
    ap.add_argument("--project", default="gemma-challenge-senpai")
    ap.add_argument("--entity", default="wandb-applied-ai-team")
    ap.add_argument("--group", default="strict-clean-locus-pin")
    ap.add_argument("--name", default="land/strict-clean-attn-locus-pin")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    bi0 = json.loads(args.bi0.read_text())
    bi1 = json.loads(args.bi1.read_text()) if args.bi1 and args.bi1.exists() else None

    # ---- classification verdict (instr 4) -------------------------------
    bi0_loc = bi0.get("first_divergent_locus") or {}
    bi0_attn_out = (bi0.get("layer0_chain") or {}).get("attn_out", {})
    verdict = {
        "pr": 743,
        "locus_layer": bi0_loc.get("layer"),
        "locus_op": bi0_loc.get("op"),
        "bi0_attn_out_frac_bitdiff": bi0_attn_out.get("frac_bitdiff"),
        "bi0_attn_out_max_abs": bi0_attn_out.get("max_abs"),
        "bi0_prelmhead_max_abs": (bi0.get("prelmhead") or {}).get("max_abs"),
        "bi0_e2e_argmax_flips": bi0.get("e2e_argmax_flips"),
        "bi0_e2e_positions": bi0.get("e2e_positions"),
    }
    if bi1 is not None:
        bi1_attn_out = (bi1.get("layer0_chain") or {}).get("attn_out", {})
        collapses = (bi1_attn_out.get("frac_bitdiff", 1.0) == 0.0)
        verdict.update({
            "bi1_attn_out_frac_bitdiff": bi1_attn_out.get("frac_bitdiff"),
            "bi1_attn_out_max_abs": bi1_attn_out.get("max_abs"),
            "bi1_prelmhead_max_abs": (bi1.get("prelmhead") or {}).get("max_abs"),
            "bi1_e2e_argmax_flips": bi1.get("e2e_argmax_flips"),
            "attn_out_collapses_under_bi1": int(collapses),
            "byte_exact_fixable": int(collapses),
            "classification": ("reduction_order_artifact_byte_exact_fixable"
                               if collapses else "genuine_numeric_difference"),
        })
    else:
        verdict["classification"] = "bi1_arm_missing"

    run = wandb.init(
        project=args.project, entity=args.entity, group=args.group,
        name=args.name, id=args.run_id, resume=("allow" if args.run_id else None),
        config={
            "pr": 743, "phase": "strict_clean_attn_locus_pin",
            "attn_backend": bi0.get("attn_backend"),
            "verify_width": bi0.get("verify_width"),
            "n_prompts": bi0.get("n_prompts"), "n_new": bi0.get("n_new"),
            "n_layers": bi0.get("n_layers"),
            "model_dir": bi0.get("model_dir"),
            "margin_model_full_vocab": bi0.get("margin_model_full_vocab"),
            "measured_input_reconstructed": "wirbel#736 GEMV M-invariance (in-scope repro)",
        },
    )

    flat: dict = {}
    _arm_summary("bi0", bi0, flat)
    if bi1 is not None:
        _arm_summary("bi1", bi1, flat)
    for k, v in verdict.items():
        if isinstance(v, bool):
            flat[f"verdict/{k}"] = int(v)
        elif isinstance(v, (int, float)):
            flat[f"verdict/{k}"] = v
        else:
            flat[f"verdict/{k}"] = v  # strings land in summary as text
    run.summary.update(flat)

    # ---- per-layer attn_out tables --------------------------------------
    cols = ["layer", "frac_bitdiff", "max_abs", "n_bitdiff", "n"]
    run.log({"bi0/attn_out_by_layer": wandb.Table(
        columns=cols, data=_attn_out_layer_table(bi0))})
    if bi1 is not None:
        run.log({"bi1/attn_out_by_layer": wandb.Table(
            columns=cols, data=_attn_out_layer_table(bi1))})

    print(f"[wandb] run {run.id}  group={args.group}")
    print(f"[wandb] LOCUS layer={verdict['locus_layer']} op={verdict['locus_op']}  "
          f"bi0 attn_out frac_bitdiff={verdict['bi0_attn_out_frac_bitdiff']} "
          f"max_abs={verdict['bi0_attn_out_max_abs']}")
    print(f"[wandb] CLASSIFICATION = {verdict['classification']}")
    if bi1 is not None:
        print(f"[wandb] bi1 attn_out frac_bitdiff={verdict.get('bi1_attn_out_frac_bitdiff')} "
              f"-> byte_exact_fixable={verdict.get('byte_exact_fixable')}")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
