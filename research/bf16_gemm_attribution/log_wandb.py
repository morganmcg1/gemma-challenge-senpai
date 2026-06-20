"""Log the PR #812 Step-1 attribution to W&B (group bf16-gemm-attribution).

Finding: the #809 "17% bf16 aten::mm" is the bf16 TIED LM_HEAD full-vocab
logits GEMM. The #809 run (wsr5i6qb) was profiled on w4a16-ct (bf16 lm_head),
NOT the int4head. On the firing int4head the lm_head is int4 Marlin and the
bf16 GEMM is gone (0.5% / 0.0%). The conversion IS the int4head +17.053% gain.

Run with .venv/bin/python (has wandb) from a dir without ./wandb shadow."""
from __future__ import annotations

import os
import wandb

ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"

run = wandb.init(
    entity=ENTITY, project=PROJECT,
    name="lawine/bf16-gemm-attribution-step1",
    group="bf16-gemm-attribution",
    job_type="attribution",
    config={
        "pr": 812,
        "model_int4head": "/workspace/gemma_build/int4_g32_lmhead (int4 body + int4 g32 lm_head)",
        "model_w4a16ct": "google/gemma-4-E4B-it-qat-w4a16-ct (int4 body + bf16 TIED lm_head)",
        "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "num_speculative_tokens": 6,
        "graphs": True, "uniproc": True,
        "step": 1, "decision": "STOP",
    },
)

# w4a16-ct profile (== #809 wsr5i6qb / audit.log; the trace that motivated the PR)
run.summary["w4a16ct/gpu_busy_ms"] = 674.1
run.summary["w4a16ct/matmul_gemm_pct"] = 64.2
run.summary["w4a16ct/aten_mm_ms"] = 114.75
run.summary["w4a16ct/aten_mm_pct"] = 17.0
run.summary["w4a16ct/aten_mm_count"] = 59
run.summary["w4a16ct/ampere_bf16_ms"] = 111.83
run.summary["w4a16ct/ampere_bf16_pct"] = 16.6
run.summary["w4a16ct/ampere_bf16_count"] = 40
run.summary["w4a16ct/ampere_bf16_grid_x"] = 4096   # 4096*64 = 262144 = full vocab
run.summary["w4a16ct/full_vocab_N"] = 262144

# int4head profile (the firing submission; my Step-1 re-profile)
run.summary["int4head/gpu_busy_ms"] = 412.4
run.summary["int4head/matmul_gemm_pct"] = 76.4
run.summary["int4head/marlin_ms"] = 290.19
run.summary["int4head/marlin_pct"] = 70.4
run.summary["int4head/ampere_bf16_ms"] = 2.09
run.summary["int4head/ampere_bf16_pct"] = 0.5
run.summary["int4head/aten_mm_ms"] = 0.13
run.summary["int4head/aten_mm_pct"] = 0.0
run.summary["int4head/lm_head_has_packed_int4"] = 1

# bf16 lm_head fingerprint (cross-confirmed: 9tcygwjf + dpc36210 + my trace)
run.summary["lmhead_bf16/bytes_gb"] = 1.3422          # 262144*2560*2
run.summary["lmhead_bf16/gemv_ms_per_tok"] = 2.777    # ~2.724 (dpc36210), ~2.795 (trace)
run.summary["lmhead_bf16/roofline_gbps"] = 1.3422 / (2.777e-3)

# the lever (already pulled in int4head, landed #801)
run.summary["lever/tps_bf16_lmhead_control"] = 219.336   # 9tcygwjf
run.summary["lever/tps_int4_lmhead_head"] = 256.740      # 9tcygwjf -> int4head
run.summary["lever/tps_gain_pct"] = 17.053
run.summary["lever/ppl_bf16_control"] = 2.005655
run.summary["lever/ppl_int4_head"] = 2.002908
run.summary["lever/ppl_int4_within_gate"] = 1
run.summary["lever/gsm8k_acc_control"] = 0.92

# decision metrics
run.summary["primary/int4head_ampere_bf16_pct"] = 0.5
run.summary["primary/bf16_gemm_owner"] = "bf16_tied_lm_head"
run.summary["primary/decision"] = "STOP_already_converted_on_int4head"

print(f"[wandb] logged run id={run.id} name={run.name} url={run.url}")
run.finish()

# verify before citing
import wandb as _w
chk = _w.Api().run(f"{ENTITY}/{PROJECT}/{run.id}")
print(f"[wandb] VERIFIED api.run found id={chk.id} state={chk.state} group={chk.group}")
