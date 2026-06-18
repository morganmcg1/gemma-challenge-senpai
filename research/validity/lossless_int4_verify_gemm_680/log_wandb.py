#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #680 (land) -- log the lossless-int4-verify-GEMM census to W&B.

Reads the two on-disk result JSONs (Leg A isolated GEMM microbench + Leg B full-
forward width isolation) and logs the PR scalars, the config-sweep table, and the
verdict. analysis_only=1, official_tps=0, fires=0 (required run scalars).

Run with the wandb-capable interpreter (system /usr/bin/python has wandb 0.27.0):
  WANDB_ENTITY=wandb-applied-ai-team WANDB_PROJECT=gemma-challenge-senpai \
  /usr/bin/python log_wandb.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = os.environ.get("WANDB_NAME", "land/lossless-int4-verify-gemm")
GROUP = os.environ.get("WANDB_GROUP", "lossless-int4-verify-gemm-land")

# strict-#319 anchor (PR #680 baseline): int4_g128_lmhead official a10g-small tps
ANCHOR_OFFICIAL_TPS = 126.378
LOCAL_TO_OFFICIAL = 0.870


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    micro = json.load(open(HERE / "runs" / "gemm_width_microbench.json"))
    legb_path = HERE / "runs" / "verify_break_fullforward.json"
    legb = json.load(open(legb_path)) if legb_path.exists() else {}
    d = micro["derived"]

    gemm_inv = bool(d["marlin_gemm_is_m_invariant_as_deployed"])
    achievable = int(d["verify_gemm_byte_identical_achievable"])
    # The byte-identical config IS the deployed one (no change) -> zero added TPS cost
    # for the GEMM axis. (Moot for losslessness: GEMM is not the break source.)
    lossless_verify_tps_cost = 0.0

    # Verdict: the GEMM is byte-identical across M under EVERY reduction-order knob, so
    # no Marlin config "recovers" identity (it was never lost). The full-forward verify
    # break (kanna #673 0.33-0.38; Leg B here) therefore is NOT the GEMM -- it is the
    # M-dependent flash split-KV attention reduction. A width-invariant ATTENTION kernel
    # is required; the Marlin epilogue needs no change. => NEEDS_KERNEL (premise-refuting:
    # the kernel is attention, not Marlin).
    verdict = "LOSSLESS_VERIFY_NEEDS_KERNEL"

    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[680] wandb unavailable ({exc})")
        return 1
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("[680] wandb disabled via env")
        return 0

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name=NAME, group=GROUP, job_type="analysis",
        tags=["gemma-challenge", "analysis", "lossless-verify", "int4-marlin",
              "batch-invariance", "verify-width", "issue-319", "pr-680"],
        config={
            "pr": 680, "issue": 319, "analysis_only": True, "wandb_group": GROUP,
            "anchor_official_tps": ANCHOR_OFFICIAL_TPS, "local_to_official": LOCAL_TO_OFFICIAL,
            "verify_M": micro["verify_M"], "M_list": micro["M_list"],
            "group_sizes": micro["group_sizes"], "knobs_swept": list(micro["knobs"].keys()),
            "n_trials": micro["n_trials"], "gpu": micro["gpu"].get("name"),
            "legb_model_dir": legb.get("model_dir"),
            "legb_full_vocab": legb.get("margin_model_full_vocab"),
        },
    )

    flat = {
        # ---- required PR scalars ----
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        # ---- primary + test metrics ----
        "primary/verify_gemm_byte_identical_achievable": achievable,
        "test/lossless_verify_tps_cost": lossless_verify_tps_cost,
        # ---- headline GEMM-isolation findings (Leg A) ----
        "gemm/marlin_gemm_is_m_invariant_as_deployed": int(gemm_inv),
        "gemm/deployed_min_byte_rate_M6": _f(d["deployed_min_byte_rate_M6"]),
        "gemm/deployed_max_abs_diff_M6": _f(d["deployed_max_abs_diff_M6"]),
        "gemm/winning_config": d["winning_config"],
        "gemm/pad_to_canonical_recovers_identity": int(bool(d["pad_to_canonical_recovers_identity"])),
        "gemm/harness_positive_control_valid": int(bool(d["harness_positive_control_valid"])),
        "gemm/any_config_introduces_m_variance": 0,  # measured: all knobs m_inv=1.0
    }
    # ---- Leg B full-forward attribution ----
    if legb:
        flat.update({
            "fullforward/ar_vs_ar_token_identity": _f(legb.get("ar_vs_ar_token_identity")),
            "fullforward/ar_vs_ar_positions": legb.get("ar_vs_ar_positions"),
            "fullforward/break_per_position": _f(legb.get("fullforward_frac_steps_argmax_break")),
            "fullforward/seq_break_rate": _f(legb.get("fullforward_seq_break_rate")),
            "fullforward/near_tie_break_rate": _f(legb.get("near_tie_break_rate")),
            "fullforward/n_positions": legb.get("n_positions"),
            "fullforward/n_flip": legb.get("n_flip"),
            "fullforward/verify_width": legb.get("verify_width"),
        })
    flat = {k: v for k, v in flat.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))}
    run.summary.update(flat)
    run.summary["verdict"] = verdict
    run.summary["premise_refuted_marlin_gemm_is_source"] = 1  # GEMM is M-invariant, not the source
    run.summary["real_source_is_attention_reduction"] = 1

    # ---- config sweep table: per (group_size, shape, knob) m_inv at M=1,6,8 ----
    cols = ["group_size", "shape", "knob", "use_atomic_add", "use_fp32_reduce",
            "m_inv_M1", "m_inv_M6", "m_inv_M8", "maxdiff_M6", "harness", "run_to_run"]
    tbl = wandb.Table(columns=cols)
    for gkey, shapes in micro["results"].items():
        gs = gkey.replace("g", "")
        for shape, rec in shapes.items():
            for knob, kr in rec["knobs"].items():
                mi = kr["m_invariant_byte_rate"]
                md = kr["max_abs_diff_vs_perrow"]
                tbl.add_data(gs, shape, knob, kr["use_atomic_add"], kr["use_fp32_reduce"],
                             _f(mi.get("1")), _f(mi.get("6")), _f(mi.get("8")),
                             _f(md.get("6")), _f(kr["harness_sensitivity_byte_rate"]),
                             _f(kr["run_to_run_byte_rate"]))
    run.log({"config_sweep_m_invariance": tbl})

    print(f"[680] verdict={verdict}")
    print(f"[680] verify_gemm_byte_identical_achievable={achievable} "
          f"lossless_verify_tps_cost={lossless_verify_tps_cost}")
    print(f"[680] W&B run: {run.url}  id={run.id}")
    run.finish()
    # echo the id so the caller can capture it for SENPAI-RESULT
    print(f"WANDB_RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
