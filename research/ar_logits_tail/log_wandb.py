#!/usr/bin/env python
"""Log the AR logits->token tail decomposition to W&B (#604).

ANALYSIS-ONLY: official_tps=0, no served-file change, no HF Job. Group
ar-logits-tail-overhead. Pulls the measured artifacts (tpot, trace decomp,
sampler microbench, served summary) and records the headline tail metrics +
verdict.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default if default is not None else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fern/ar-logits-tail-overhead")
    ap.add_argument("--group", default="ar-logits-tail-overhead")
    ap.add_argument("--verdict", default="NEGATIVE")
    base = "research/ar_logits_tail"
    ap.add_argument("--tpot", default=f"{base}/tpot_result.json")
    ap.add_argument("--decomp", default=f"{base}/trace_decomp.json")
    ap.add_argument("--sampler", default=f"{base}/sampler_microbench.json")
    ap.add_argument("--served", default=f"{base}/reference/ref_int4g128_ar.summary.json")
    args = ap.parse_args()

    tpot = load(args.tpot)
    decomp = load(args.decomp)
    sampler = load(args.sampler)
    served = load(args.served)

    tau = 1.03524
    served_wall_tps = None
    if served.get("num_completion_tokens") and served.get("duration_s"):
        served_wall_tps = served["num_completion_tokens"] / served["duration_s"]

    tail = decomp.get("TAIL") or {}
    # ANCHOR the served decode step to the streaming-TPOT measurement (CUDA-graph
    # ON, async-scheduled, served path) -- NOT the offline trace wall, which was
    # captured ENFORCE_EAGER=1 + torch.profiler + in-process (V1 multiproc OFF) and
    # is therefore ~9x inflated (74 ms/step, 88% "idle"). Kernel COMPUTE times in
    # the trace are graph-invariant, so the GPU-busy COMPOSITION is valid; the trace
    # absolute wall / host-idle are an eager+profiler artifact, not the served tail.
    served_step_ms = tpot.get("median_tpot_ms")            # 7.818 ms (real served step)
    decode_only_tps = tpot.get("decode_only_steady_tps_median")
    sampling_gpu_ms = tail.get("sampling_gpu_ms_step")     # 0.00854 ms = the only post-logits GPU op
    # worst-case host-side argmax incl a blocking .item() D->H sync (standalone microbench)
    tail_ub_host_ms = sampler.get("argmax_plus_item_sync_fp32_ms")  # 0.0316 ms
    sampling_pct = (sampling_gpu_ms / served_step_ms) if (sampling_gpu_ms and served_step_ms) else None
    tail_ub_pct = (tail_ub_host_ms / served_step_ms) if (tail_ub_host_ms and served_step_ms) else None
    # decode-only -> served gap on THIS substrate (NOT the cross-substrate 413/206)
    decode_vs_served_ratio = (served_wall_tps / decode_only_tps) if (served_wall_tps and decode_only_tps) else None

    summary = {
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "no_submission": True,
        "no_served_file_change": True,
        "pr": 604,
        "agent": "fern",
        "substrate": "int4_g128_lmhead (operative, plain stock vllm AR, MNS=1, spec-OFF, temp=0, FI_SAMPLER=0)",
        "operative_official_tps_anchor": 126.378,
        "tau_local_to_official": tau,
        # served & decode-only (all on THIS substrate)
        "served_wall_tps_128x512_local": served_wall_tps,
        "decode_only_steady_tps_tpot": decode_only_tps,
        "stream_wall_tps_single": tpot.get("stream_wall_tps_mean"),
        "ttft_ms": tpot.get("ttft_ms_mean"),
        "median_tpot_ms": served_step_ms,
        "served_step_ms_anchor": served_step_ms,
        # 2x-gap attribution: there is NONE on this AR substrate
        "decode_only_over_served_ratio": decode_vs_served_ratio,
        "two_x_gap_present_on_this_substrate": False,
        "ttft_amortization_pct_of_decode": (tpot.get("ttft_ms_mean") / (512 * served_step_ms)) if (tpot.get("ttft_ms_mean") and served_step_ms) else None,
        # the HEADLINE tail numbers (anchored on the SERVED step)
        "tail_ms_per_token": sampling_gpu_ms,
        "tail_pct_of_step": sampling_pct,
        "tail_ub_ms_per_token_host_sync": tail_ub_host_ms,
        "tail_ub_pct_of_step_host_sync": tail_ub_pct,
        "sampling_gpu_ms_step": sampling_gpu_ms,
        "sampling_gpu_pct_of_served_step": sampling_pct,
        # trace GPU-busy COMPOSITION (valid; share of GPU-busy compute)
        "trace_gpu_busy_ms_step": decomp.get("gpu_busy_ms_step"),
        "trace_busy_component_ms_step": decomp.get("busy_component_ms_step"),
        "trace_gemm_quant_ms_step": (decomp.get("busy_component_ms_step") or {}).get("gemm_quant"),
        # trace absolute wall/idle = EAGER+PROFILER ARTIFACT (NOT served tail)
        "trace_eager_step_ms_ARTIFACT": decomp.get("step_ms_mean"),
        "trace_eager_host_idle_ms_ARTIFACT": tail.get("host_idle_ms_step"),
        "trace_is_eager_profiler_inprocess": True,
        # sampler microbench
        "sampler_argmax_fp32_ms": sampler.get("argmax_fp32_ms"),
        "sampler_argmax_plus_item_sync_ms": sampler.get("argmax_plus_item_sync_fp32_ms"),
        "sampler_topk1_fp32_ms": sampler.get("topk1_fp32_ms"),
        "torch_argmax_picks_lowest_index": (sampler.get("flashinfer") or {}).get("torch_picks_lowest_index"),
        "flashinfer_sampler_buildable": not bool((sampler.get("flashinfer") or {}).get("flashinfer_call_error")),
        # Phase-2 lever verdict
        "lever_a_faster_argmax": "DEAD: FlashInfer sampler won't JIT (curand.h missing); native torch.argmax IS the byte-exact reference (lowest-index tie-break)",
        "lever_b_deferred_detok": "OFF critical path: vLLM V1 detok runs in front-end process, not EngineCore GPU stream (DETOK A/B was NULL)",
        "lever_c_unused_outputs": "NONE: plain greedy completions emit no logprobs/extra fields to disable",
        "projected_best_byte_identical_official_proxy_tps": 126.378,
        "beats_126_378": False,
        "verdict": args.verdict,
    }
    print(json.dumps(summary, indent=2))

    if os.environ.get("NO_WANDB") == "1":
        print("[no-wandb] skipping upload")
        return
    import wandb
    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
                     entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
                     name=args.name, group=args.group, job_type="analysis",
                     config={"pr": 604, "substrate": summary["substrate"],
                             "analysis_only": True, "official_tps": 0})
    run.summary.update({f"summary/{k}": v for k, v in summary.items() if v is not None})
    # attach raw artifacts
    art = wandb.Artifact("ar_logits_tail_decomp", type="analysis")
    for p in [args.tpot, args.decomp, args.sampler, args.served]:
        if Path(p).exists():
            art.add_file(p)
    run.log_artifact(art)
    run.finish()
    print(f"[wandb] logged run {run.id}")


if __name__ == "__main__":
    main()
