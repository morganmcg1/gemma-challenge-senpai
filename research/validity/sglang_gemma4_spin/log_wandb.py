#!/usr/bin/env python
"""PR #745 — log the SGLang gemma-4-E4B spin investigation to W&B.

This is a capability/blocker investigation, not a served benchmark: SGLang
cannot be brought up for `google/gemma-4-E4B-it` on a10g-small in any version,
so there is no TPS/PPL to log. We record the empirical version-matrix and
kernel-arch findings so the negative result is durable and searchable.
"""
from __future__ import annotations

import wandb

run = wandb.init(
    entity="wandb-applied-ai-team",
    project="gemma-challenge-senpai",
    name="wirbel/sglang-gemma4-spin",
    group="sglang-gemma4-spin",
    job_type="capability-probe",
    config={
        "pr": 745,
        "hypothesis": "SGLang single-stream decode faster than vLLM int4 floor (126.378 TPS) for gemma-4-E4B on a10g-small",
        "model": "google/gemma-4-E4B-it",
        "model_arch": "Gemma4ForConditionalGeneration",
        "model_type": "gemma4",
        "gpu": "NVIDIA A10G",
        "compute_capability": "sm_86",
        "regime": "single-stream batch=1 greedy, 128 prompts x 512 tokens, ignore_eos",
        "vllm_int4_floor_tps": 126.378,
        # --- L1: harness-pinned bench stack ---
        "L1_sglang_version": "0.5.2",
        "L1_sglang_pins_transformers": "4.56.1",
        "L1_unsatisfiable_with_transformers_5_9_0": True,
        "L1_transformers_4561_parses_gemma4": False,
        "L1_sglang_052_has_gemma4_class": False,
        "L1_sglang_052_gemma3n_class": True,
        "L1_sglang_052_fallback": "TransformersForCausalLM (text-only, no multimodal)",
        # --- L2: gemma4-capable sglang ---
        "L2_native_gemma4_min_sglang": "0.5.11",
        "L2_all_gemma4_versions_hard_dep": "flash-attn-4>=4.0.0b9 (Blackwell FA4)",
        "L2_sglang_0511_0512_pins_transformers": "5.6.0",
        "L2_sglang_0513_pins_transformers": "5.8.1",
        "L2_no_sglang_coexists_with_transformers_5_9_0": True,
        "L2_torch_pulled": "2.11.0+cu130",
        "L2_torch_cuda_avail_on_a10g": True,
        "L2_transformers_581_parses_gemma4": True,
        "L2_sgl_kernel_common_ops_archs": "sm90, sm100 (NO sm86)",
        "L2_sgl_kernel_loads_on_a10g": False,
        "L2_sglang_registry_nonempty_on_a10g": False,
        "L2_missing_system_lib": "libnuma.so.1",
    },
)

# Headline scalar metrics. 0.0 TPS == SGLang could not be served on the A10G.
run.summary["sglang_a10g_single_stream_tps"] = 0.0
run.summary["sglang_served_tokens"] = 0
run.summary["vllm_int4_floor_tps"] = 126.378
run.summary["verdict"] = "NO-GO: SGLang cannot serve gemma-4 on a10g-small (sm_86) in any version"
run.summary["blocker_L1"] = "sglang0.5.2->transformers4.56.1 cannot parse model_type gemma4"
run.summary["blocker_L2"] = "sglang>=0.5.11 sgl_kernel ships sm90/sm100 only, no sm86 binary"

# A small table summarizing the version matrix for the dashboard.
tbl = wandb.Table(columns=["sglang", "pins_transformers", "has_gemma4_class", "runs_on_sm86", "blocker"])
tbl.add_data("0.5.2", "4.56.1", "no", "n/a", "transformers 4.56.1 cannot parse gemma4; text-only fallback")
tbl.add_data("0.5.11/0.5.12", "5.6.0", "yes", "no", "flash-attn-4 (Blackwell); sgl_kernel sm90/sm100 only")
tbl.add_data("0.5.13", "5.8.1", "yes", "no", "sgl_kernel ships sm90/sm100 only, no sm86; libnuma missing")
run.log({"version_matrix": tbl})

print("WANDB_RUN_ID", run.id)
print("WANDB_RUN_URL", run.url)
run.finish()
