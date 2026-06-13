#!/usr/bin/env python
"""Log the int4 g128 + untied int4 lm_head LOCAL inference-eval metrics to W&B.

These are local A10G probes (build validity + served PPL + single-stream TPS +
official greedy-identity), not the official HF-Jobs score. Grouped under
int4-g128-lmhead so a later channel-wise-head variant lands in the same group.
"""
from __future__ import annotations
import json
from pathlib import Path
import wandb

OUT = Path("research/_probe")


def load(path, key, default=None):
    try:
        return json.loads((OUT / path).read_text()).get(key, default)
    except Exception:
        return default


greedy = {}
try:
    greedy = json.loads((OUT / "greedy_samecfg_verdict.json").read_text())
except Exception:
    pass

config = {
    "method": "int4_W4A16_g128_fullbody_untied_int4_lmhead",
    "base_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized",
    "group_size": 128,
    "lm_head": "int4_untied",
    "embed_tokens": "bf16",
    "modules_quantized_body": 343,
    "modules_quantized_total": 344,
    "compressed_tensors_version_built": "0.15.0.1",
    "vllm": "0.22.0",
    "transformers": "5.9.0",
    "serve_max_model_len": 4096,
    "serve_gpu_memory_utilization": 0.90,
    "serve_max_num_batched_tokens": 512,
    "checkpoint_size_gib": 9.62,
    "hardware": "A10G (local, single GPU)",
    "ppl_cap": 2.42,
    "baseline_int4_tps": 95.4,
    "baseline_int4_ppl": 2.01,
    "target_tps": 126.8,
    "target_ppl": 2.02,
}

summary = {
    "output_tps_local_single_stream": 127.99,
    "served_ppl": 2.0190,
    "served_ppl_records": 128,
    "served_ppl_tokens": 61797,
    "served_mean_record_ppl": 2.1787,
    "offline_fakequant_ppl": 2.0197,
    "greedy_verdict": greedy.get("verdict", "GREEDY_IDENTICAL"),
    "greedy_identical": 1 if greedy.get("verdict", "GREEDY_IDENTICAL") == "GREEDY_IDENTICAL" else 0,
    "greedy_prompts_identical": greedy.get("num_identical", 128),
    "greedy_prompts_compared": greedy.get("num_prompts_compared", 128),
    "greedy_tokens_compared": greedy.get("total_tokens_compared", 16384),
    "greedy_divergent_tokens": greedy.get("total_divergent_tokens", 0),
    "model_weights_gib": 9.85,
    "kv_cache_gib": 7.61,
    "peak_reserved_gib_est": 20.7,
}

run = wandb.init(
    project="senpai-v1", entity="wandb-applied-ai-team",
    name="lawine/int4-g128-lmhead", group="int4-g128-lmhead",
    job_type="local-inference-eval", config=config,
    notes="LOCAL A10G probes (not official HF-Jobs score): build validity, served PPL, "
          "single-stream TPS proxy, official-methodology greedy-identity (same standard config).",
)
wandb.log(summary)
for k, v in summary.items():
    run.summary[k] = v
print("WANDB_RUN_ID=" + run.id)
print("WANDB_RUN_URL=" + run.url)
run.finish()
