#!/usr/bin/env python
"""PR #662 (lawine) -- log the lm_head-dtype AIME panel to W&B.

One run per arm under group `lmhead-dtype-aime-lawine`. LOCAL ONLY:
analysis_only=True, official_tps=0, NO HF Job, NO submission. The binding metric
is AIME maj@1 greedy n=60 @ gb6144 (BI=1, mintok=8, M=1 AR, no drafter, 0.22.0),
plus the single-stream decode-TPS proxy that prices the head read. The HEADLINE
(Delta_head = bf16head - shipped, McNemar + Newcombe95) and the calibration
residual (official_g32 - bf16head) ride on the bf16head run's summary.

Run under the repo .venv (has wandb), NOT the serve venv:
  ./.venv/bin/python research/validity/lmhead_dtype_aime_662/log_wandb_662.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import wandb

GROUP = "lmhead-dtype-aime-lawine"
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
PROJECT = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")

COMMON = {
    "vllm_version": "0.22.0",
    "engine": "manifest-pinned 0.22.0 (/tmp/vllm0220-srv)",
    "vllm_batch_invariant": 1,
    "max_model_len": 8192,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.90,
    "min_tokens": 8,
    "max_tokens": 6144,
    "decode": "greedy temp=0 maj@1, M=1 AR no drafter",
    "use_flashinfer_sampler": 0,
    "serve_path": "submissions/bf16_base_aime/serve.py (MODEL_ID repoint)",
    "analysis_only": True,
    "official_tps": 0,
    "anchor_bf16_base_aime": 0.4667,
    "anchor_bar_0p420": 0.420,
    "anchor_official_g32_aime": 0.4167,
    "anchor_shipped_g128_653": 0.3833,
}

ARM_RECIPE = {
    "shipped_g128": "our int4 g128-minmax body + untied int4 g128 lm_head (LIVE submission recipe)",
    "our_g128_int8head": "our int4 g128-minmax body (byte-identical) + untied int8 g128 lm_head",
    "our_g128_bf16head": "our int4 g128-minmax body (byte-identical) + untied bf16 lm_head",
}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    stats = load(RES / "stats_662.json")
    assert stats, "run analyze.py first to produce stats_662.json"
    cells = stats["cells"]
    head = stats["headline_delta_head_bf16head_minus_shipped"]
    resid = stats["calibration_residual_officialg32_minus_bf16head"]

    ids = {}
    for arm, cell in cells.items():
        # TPS filenames are inconsistent across launch paths: the standalone
        # bf16head run wrote tps_bf16head.json (short label), while the
        # orchestrator's run_aime_arm.sh writes tps_<ARM>.json with the full ARM
        # name (tps_shipped_g128.json, tps_our_g128_int8head.json). Try every
        # candidate so no arm's TPS is silently dropped.
        short = arm.replace("our_g128_", "").replace("shipped_g128", "shipped")
        tps = load(RES / f"tps_{short}.json") or load(RES / f"tps_{arm}.json")
        gpqa = load(RES / f"gpqa_{arm}.json")

        run = wandb.init(
            project=PROJECT, entity=ENTITY, name=f"lawine/{arm}",
            group=GROUP, job_type="lmhead-dtype-aime", reinit=True,
            config={**COMMON, "arm": arm, "recipe": ARM_RECIPE.get(arm, arm)},
        )
        log = {
            "aime/maj1_acc": cell["maj1_acc"],
            "aime/n_correct": cell["n_correct"],
            "aime/n": cell["n"],
            "aime/wilson95_lo": cell["wilson95"][0],
            "aime/wilson95_hi": cell["wilson95"][1],
            "aime/pct_of_bf16_base": cell["pct_of_bf16_base"],
            "aime/clears_0p420": int(cell["clears_0p420"]),
        }
        if tps:
            log["tps_proxy/decode_tps_median"] = tps.get("decode_tps_median_across_reps")
            log["tps_proxy/inter_token_ms_median"] = tps.get("inter_token_ms_median_across_reps")
        if gpqa:
            log["gpqa_d/acc"] = gpqa.get("accuracy")
            log["gpqa_d/n_correct"] = gpqa.get("n_correct")
            log["gpqa_d/n"] = gpqa.get("n_scored") or gpqa.get("n")
        wandb.log(log)

        run.summary.update({
            "arm": arm,
            "aime_maj1_acc": cell["maj1_acc"],
            "aime_pct_of_bf16_base": cell["pct_of_bf16_base"],
            "aime_clears_0p420": cell["clears_0p420"],
            "decode_tps_proxy": (tps or {}).get("decode_tps_median_across_reps"),
        })
        if arm == "our_g128_bf16head":
            run.summary.update({
                "HEADLINE_delta_head_bf16head_minus_shipped": head["delta_acc"],
                "headline_newcombe95_lo": head["newcombe95"][0],
                "headline_newcombe95_hi": head["newcombe95"][1],
                "headline_mcnemar_b_bf16only": head["mcnemar_b"],
                "headline_mcnemar_c_shippedonly": head["mcnemar_c"],
                "headline_mcnemar_exact_p": head["mcnemar_exact_p"],
                "headline_significant_0p05": head["significant_0p05"],
                "calibration_residual_officialg32_minus_bf16head": resid,
                "surface_to_human": True,
            })
        print(f"[wandb] {arm}: aime={cell['maj1_acc']:.4f} tps={(tps or {}).get('decode_tps_median_across_reps')} id={run.id}")
        ids[arm] = run.id
        run.finish()

    print(f"[wandb] group={GROUP} project={ENTITY}/{PROJECT}")
    print(f"[wandb] run_ids={json.dumps(ids)}")
    (RES / "wandb_run_ids.json").write_text(json.dumps(ids, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
