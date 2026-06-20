#!/usr/bin/env python3
"""Log the PR #781 bi0 Marlin-GEMM kernel-speed diagnostic to W&B (group bi0-marlin-gemm).

One run: bi0 control reproduction + conc=1 bandwidth diagnostic + sm_86 kernel
inventory + empirical lever probes (atomic_add / machete). Terminal NEGATIVE.
Run with the SYSTEM python (has a real wandb with .init); the submission venv's
wandb is a namespace shim without .init.
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path

sys.path.insert(0, ".")
os.environ.setdefault("WANDB_PROJECT", "gemma-challenge-senpai")
os.environ.setdefault("WANDB_ENTITY", "wandb-applied-ai-team")
from scripts.wandb_logging import (  # noqa: E402
    init_wandb_run, log_summary, finish_wandb, log_json_artifact, log_file_artifact,
)


def grep(path: str, pats: list[str]) -> list[str]:
    try:
        txt = Path(path).read_text(errors="replace")
    except OSError:
        return []
    return [ln for ln in txt.splitlines() if any(re.search(p, ln, re.I) for p in pats)]


AT = "research/_localrun/lever_probes/atomic_add_on.server.log"
MA = "research/_localrun/lever_probes/quant_machete.server.log"
atomic_marlin = bool(grep(AT, [r"Using MarlinLinearKernel"]))
machete_marlin = bool(grep(MA, [r"Using MarlinLinearKernel"]))
machete_err = bool(grep(MA, [r"Unknown quantization", r"ValueError", r"ValidationError", r"Traceback", r"not supported"]))
machete_config_mismatch = bool(grep(MA, [r"does not match the quantization method", r"validation error for ModelConfig"]))
atomic_kernel_line = (grep(AT, [r"Using .*Kernel for CompressedTensorsWNA16"]) or ["<none>"])[0].strip()
machete_kernel_line = (grep(MA, [r"Using .*Kernel for CompressedTensorsWNA16"]) or ["<none>"])[0].strip()

config = {
    "pr": 781,
    "submission": "submissions/int4_mtp_bi0_surgattn",
    "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
    "stack": "vllm==0.22.0, torch 2.11.0+cu130",
    "gpu": "A10G sm_86, 23.68GB, 600 GB/s peak HBM",
    "hbm_peak_gbs": 600.0,
    "profile_method": "official gemma_decode_profiler (torch.profiler graph+eager, conc=1 spec-off)",
    "verdict": "NEGATIVE: no numerics-equivalent faster W4A16 kernel/config on sm_86",
}

summary = {
    # --- control reproduction (bi0 as shipped, spec-on M=7) ---
    "control_decode_tps_aggregate": 218.49,     # 65536 tok / 299.95 s  (matches 218.02 official anchor)
    "control_probe_tps_degenerate": 2767.12,    # harness two-call probe; non-physical for spec-on; NOT trusted
    "ppl": 2.005655,                            # anchor 2.0058
    "ppl_cap": 2.42,
    "completed": 128,
    "official_gate_pass": 1,
    "all_modalities_loaded": 1,
    "greedy_divergent_vs_specoff_prompts": 108, # documented M=7-vs-M=1 int4 split-K batch-non-invariance; NOT an official gate
    "greedy_identical_vs_specoff_prompts": 20,
    # --- conc=1 decode-step bandwidth diagnostic (THE key diagnostic) ---
    "gemm_share_pct_of_gpu_busy_conc1": 91.4,
    "int4_marlin_per_token_ms_conc1": 6.28,
    "lmhead_gemv_per_token_ms_conc1": 2.776,
    "int4_marlin_bytes_per_token_GB": 2.18825,
    "lmhead_bytes_per_token_GB": 1.34218,
    "int4_marlin_hbm_util_pct_conc1": 58.0,
    "lmhead_hbm_util_pct_conc1": 80.6,
    "combined_weight_gemm_hbm_util_pct_conc1": 65.0,
    # --- sm_86 kernel inventory (code-traced + empirically confirmed) ---
    "marlin_is_sole_servable_w4a16_kernel_sm86": 1,
    "n_alternate_numerics_equiv_kernels_sm86": 0,
    "atomic_add_is_noop_sm86_bf16": 1,
    "machete_available_sm86": 0,
    # --- empirical lever probes (this run) ---
    "probe_atomic_add_on_selects_marlin": int(atomic_marlin),
    "probe_machete_selects_marlin_silently": int(machete_marlin),
    "probe_machete_errored": int(machete_err),
    "probe_machete_rejected_config_mismatch": int(machete_config_mismatch),
    "atomic_kernel_selection_line": atomic_kernel_line,
    "machete_kernel_selection_line": machete_kernel_line,
    # --- gate outcome ---
    "fire_worthy_variant_found": 0,
    "best_variant_tps_delta_vs_control": 0.0,
}

run = init_wandb_run(
    job_type="kernel-diagnostic",
    agent="ubel",
    name="ubel/bi0-marlin-gemm",
    group="bi0-marlin-gemm",
    notes=("PR #781 terminal NEGATIVE — int4 W4A16 Marlin GEMM kernel-speed on bi0 (A10G sm_86). "
           "MarlinLinearKernel is the sole servable W4A16 kernel; the one toggle "
           "(VLLM_MARLIN_USE_ATOMIC_ADD=1) still selects Marlin and is HW-gated off on Ampere+bf16 "
           "(empirically confirmed); no tile/split knobs; machete is sm_90-only AND --quantization "
           "machete is hard-rejected by ModelConfig validation (config-method mismatch vs the "
           "checkpoint's compressed-tensors, ValidationError, server fails to boot). Decode is "
           "BW-bound: conc=1 int4 Marlin 58% HBM, lm_head 80.6%, combined 65%; serving runs the verify "
           "GEMM at M=7 near the one-wave HBM wall. No greedy-safe faster kernel/config exists on "
           "sm_86 -> kernel-swap lever does not exist."),
    tags=["bi0-marlin-gemm", "kernel-speed", "negative-result", "sm86", "w4a16", "bandwidth-bound"],
    config=config,
)
log_summary(run, summary, step=0)
for name, path, atype in [
    ("bandwidth_diagnostic", "research/bi0_marlin_gemm/bandwidth_diagnostic.json", "diagnostic"),
    ("kernel_inventory_sm86", "research/bi0_marlin_gemm/kernel_inventory_sm86.md", "inventory"),
    ("graph_profile", "research/_localrun/profile-bi0-marlin/graph_profile.json", "profile"),
    ("eager_profile_breakdown", "research/_localrun/profile-bi0-marlin/profile_breakdown.json", "profile"),
    ("control_evidence", "research/_localrun/control-bi0/evidence.json", "validation"),
]:
    p = Path(path)
    if p.suffix == ".json" and p.exists():
        try:
            log_json_artifact(run, name=name, artifact_type=atype, data=json.loads(p.read_text()))
            continue
        except Exception:
            pass
    log_file_artifact(run, path=p, name=name, artifact_type=atype)

print("WANDB_RUN_ID:", getattr(run, "id", None))
print("WANDB_RUN_URL:", getattr(run, "url", None))
print("LEVER_PROBE: atomic_add_on selects Marlin =", atomic_marlin, "| machete selects Marlin =", machete_marlin, "| machete errored =", machete_err)
finish_wandb(run)
