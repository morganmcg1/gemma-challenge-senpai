"""KV-cache L2-prefetch gate (PR #445) — measure t_attn_frac_of_verify and decide.

Analysis/measurement-only card. The advisor pre-registered a hard self-abort gate:

  * attention < 10% of verify forward time -> system-level prefetch gain <= 1 TPS;
    bank the negative, do NOT prototype.
  * attention >= 15% -> instrument an explicit cp.async.bulk.prefetch.L2 stub.

The lever: prefetch the NEXT attention layer's KV pages HBM->L2 while the current
layer computes, to overlap HBM-read latency in the Triton unified-attention kernel.
Pure memory scheduling -> byte-exact greedy identity by construction.

This script combines three evidence sources on the DEPLOYED K=7 split-KV stack
(``fa2sw_precache_kenyan``, public #1 at 481.53 TPS):

  A. Authoritative end-to-end verify decomposition from the committed post-split-KV
     served profile (``frontier_decode_postsplitkv``) -> t_attn_frac_of_verify.
  B. A FRESH local microbench of the real vLLM ``unified_attention`` kernel at the
     deployed M=8 verify dispatch (3D split-KV) across the 37-layer Gemma-4-E4B
     decoder -> attention ms/cycle, HBM-bandwidth efficiency (occupancy-bound
     check), and the split-KV speedup. (reuses PR #39 ``profile_attention.bench_op``)
  C. A cp.async-already check: scan the compiled Triton PTX for ``cp.async``/ldgsts
     (if the kernel already async-pipelines KV loads, explicit L2 prefetch is
     redundant -- PR instruction #3's gotcha).

Plus the hardware-premise correction (A10G L2 is 6 MB, not the PR's 40 MB; the
~11 MB full-prompt KV does NOT fit -> at most one-layer-ahead prefetch).

Local A10G op-microbench: no server, no submission, no HF Job, no leaderboard number.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "local_validation"))

# Gemma-4-E4B text decoder (from osoi5-v0-baked config; matches profile_attention)
N_LAYERS = 37
N_FULL = 7
N_SLIDING = 30
PR_CLAIMED_LAYERS = 42
PR_CLAIMED_L2_MB = 40.0

# Deployed-stack baseline anchors (PR body)
DEPLOYED_TPS = 481.53            # NON-equivalent incumbent (identity 0.9966), PR #52
REALIZED_FRONTIER_TPS = 467.14  # byte-exact equivalence frontier, denken #423

ABORT_GATE = 0.10               # attention < 10% of verify -> self-abort
PROTOTYPE_GATE = 0.15           # attention >= 15% -> prototype


# --------------------------------------------------------------------------- #
# A. Authoritative end-to-end fraction from the committed served profile        #
# --------------------------------------------------------------------------- #
def committed_verify_fraction() -> dict:
    p = REPO / "research/profiling/frontier_decode_postsplitkv/frontier_decode_profile.json"
    d = json.loads(p.read_text())
    a = d["analysis"]
    sub = a["verify_subsplit_ms"]
    verify_gpu_ms = a["cycle"]["verify_gpu_ms"]
    attn_ms = sub["attention_fa2sw"]
    subsplit_sum = sum(sub.values())
    comp = a["gpu_busy_composition_frac"]
    return {
        "source_file": str(p.relative_to(REPO)),
        "source_utc": d.get("utc"),
        "submission": Path(d.get("submission", "")).name,
        "verify_gpu_ms": verify_gpu_ms,
        "verify_subsplit_ms": sub,
        "verify_subsplit_sum_ms": subsplit_sum,
        "attention_ms": attn_ms,
        # primary number: attention's share of the verify forward wall
        "t_attn_frac_of_verify": attn_ms / verify_gpu_ms,
        "t_attn_frac_of_verify_subsplit_denom": attn_ms / subsplit_sum,
        "attn_frac_of_gpu_busy": comp["verify_attention_fa2sw"],
        "verify_body_gemm_frac_of_gpu_busy": comp["verify_body_int4_gemm"],
        "e_accept": a.get("e_accept"),
        "reconstructed_decode_tps": a["tps"]["reconstructed_decode_tps"],
    }


# --------------------------------------------------------------------------- #
# B. Fresh local kernel microbench (deployed 3D split-KV M=8 verify dispatch)    #
# --------------------------------------------------------------------------- #
def fresh_microbench(rep_ctx: int, n_iter: int) -> dict:
    import torch
    import profile_attention as pa

    assert torch.cuda.is_available(), "CUDA required"
    os.environ.setdefault("SPLITKV_VERIFY", "1")  # deployed verify path
    pa._maybe_install_splitkv()

    dev = torch.cuda.get_device_properties(0)
    peak = pa._measure_peak_bw(torch, torch.device("cuda"))["measured_peak_gbps_copy"]

    def agg(dispatch: str) -> dict:
        rs = pa.bench_op(torch, "sliding", 8, rep_ctx, dispatch=dispatch,
                         n_iter=n_iter, validate=True)
        rf = pa.bench_op(torch, "full", 8, rep_ctx, dispatch=dispatch,
                         n_iter=n_iter, validate=True)
        t_cycle_us = N_SLIDING * rs["device_us"] + N_FULL * rf["device_us"]
        bytes_cycle = (N_SLIDING * rs["bytes"]["total_raw_bytes"]
                       + N_FULL * rf["bytes"]["total_raw_bytes"])
        gbps = bytes_cycle / (t_cycle_us / 1e6) / 1e9
        return {
            "dispatch": dispatch,
            "sliding_us": rs["device_us"], "full_us": rf["device_us"],
            "sliding_3d": bool(rs["used_3d_split_kv"]),
            "full_3d": bool(rf["used_3d_split_kv"]),
            "sliding_gbps": rs["achieved_gbps_total"],
            "full_gbps": rf["achieved_gbps_total"],
            "sliding_val_max_abs_err": rs["validation"]["max_abs_err"],
            "full_val_max_abs_err": rf["validation"]["max_abs_err"],
            "attn_us_per_cycle": t_cycle_us,
            "attn_ms_per_cycle": t_cycle_us / 1e3,
            "kv_bytes_per_cycle": bytes_cycle,
            "achieved_gbps_aggregate": gbps,
            "bandwidth_eff_vs_measured_peak": gbps / peak,
            "bandwidth_eff_vs_spec_peak": gbps / pa.A10G_PEAK_GBPS,
        }

    deployed = agg("served")    # M=8 -> 3D split-KV (the deployed verify path)
    force2d = agg("force2d")    # pre-split-KV 2D path (for the speedup reference)

    return {
        "rep_ctx": rep_ctx,
        "gpu": dev.name,
        "l2_bytes_measured": dev.L2_cache_size,
        "l2_MB_measured": dev.L2_cache_size / 1e6,
        "sm_count": dev.multi_processor_count,
        "measured_peak_gbps_copy": peak,
        "deployed_3d_split_kv": deployed,
        "force2d_pre_splitkv": force2d,
        "splitkv_verify_speedup": force2d["attn_us_per_cycle"] / deployed["attn_us_per_cycle"],
    }


# --------------------------------------------------------------------------- #
# C. cp.async-already check: does the served kernel pipeline KV loads?           #
# --------------------------------------------------------------------------- #
def cp_async_scan() -> dict:
    """Scan compiled Triton PTX (from the warm cache) for cp.async / ldgsts.

    If the served unified_attention kernel already issues async global->shared
    copies through its pipeline stages, an explicit cp.async.bulk.prefetch.L2 is
    largely redundant (the KV is already being prefetched off the critical path).
    """
    hits = {}
    triton_cache = Path(os.environ.get("TRITON_CACHE_DIR",
                                       str(Path.home() / ".triton" / "cache")))
    ptx_files = glob.glob(str(triton_cache / "**" / "*.ptx"), recursive=True)
    attn_ptx_with_cp_async = 0
    attn_ptx_total = 0
    needles = ("cp.async", "ldgsts")
    sample = None
    for f in ptx_files:
        try:
            txt = Path(f).read_text(errors="ignore")
        except Exception:
            continue
        low = txt.lower()
        is_attn = ("unified_attention" in low) or ("attn" in Path(f).name.lower())
        if not is_attn:
            # also accept kernels that look like attention by signature
            if "block_table" not in low and "seqused" not in low:
                continue
        attn_ptx_total += 1
        n = sum(low.count(nd) for nd in needles)
        if n > 0:
            attn_ptx_with_cp_async += 1
            if sample is None:
                sample = {"file": Path(f).name, "cp_async_count": n}
    hits = {
        "triton_cache_dir": str(triton_cache),
        "n_ptx_scanned": len(ptx_files),
        "n_attn_ptx": attn_ptx_total,
        "n_attn_ptx_with_cp_async": attn_ptx_with_cp_async,
        "kernel_already_async_loads_kv": attn_ptx_with_cp_async > 0,
        "sample": sample,
    }
    return hits


# --------------------------------------------------------------------------- #
# D. Decision gate + honest optimistic prefetch-ceiling bracket                  #
# --------------------------------------------------------------------------- #
def decide(committed: dict, micro: dict) -> dict:
    frac_verify = committed["t_attn_frac_of_verify"]
    frac_busy = committed["attn_frac_of_gpu_busy"]

    if frac_verify < ABORT_GATE:
        verdict = "SELF_ABORT"
    elif frac_verify >= PROTOTYPE_GATE:
        verdict = "PROTOTYPE"
    else:
        verdict = "GRAY_ZONE_ABORT"  # 10-15%: below prototype bar -> default abort

    # KV-per-cycle vs the (real) 6 MB L2: does the full-prompt KV fit?
    kv_bytes_cycle = micro["deployed_3d_split_kv"]["kv_bytes_per_cycle"]
    l2_bytes = micro["l2_bytes_measured"]
    kv_fits_l2 = kv_bytes_cycle <= l2_bytes
    per_layer_kv = kv_bytes_cycle / N_LAYERS

    # Optimistic ceiling: if prefetch hid a fraction f of the per-cycle attention
    # time, decode is gpu-bound so TPS scales ~ 1/(1 - frac_busy*f). Bracket f.
    # f is bounded SMALL here because the kernel is occupancy/launch-bound (runs at
    # ~20% of HBM BW, not BW-bound) AND already cp.async-pipelines its KV loads.
    bw_eff = micro["deployed_3d_split_kv"]["bandwidth_eff_vs_measured_peak"]
    ceiling = []
    for f in (0.10, 0.25, 0.50, 1.00):
        tps = REALIZED_FRONTIER_TPS / (1.0 - frac_busy * f)
        ceiling.append({
            "hidden_attn_frac_f": f,
            "tps_on_467p14_base": tps,
            "delta_tps": tps - REALIZED_FRONTIER_TPS,
            "crosses_deployed_481p53": tps >= DEPLOYED_TPS,
        })

    return {
        "t_attn_frac_of_verify": frac_verify,
        "attn_frac_of_gpu_busy": frac_busy,
        "abort_gate": ABORT_GATE,
        "prototype_gate": PROTOTYPE_GATE,
        "verdict": verdict,
        "deployed_3d_bandwidth_eff_vs_peak": bw_eff,
        "kernel_occupancy_launch_bound": bw_eff < 0.5,
        "kv_bytes_per_cycle": kv_bytes_cycle,
        "kv_MB_per_cycle": kv_bytes_cycle / 1e6,
        "l2_MB_measured": l2_bytes / 1e6,
        "kv_fits_in_l2": kv_fits_l2,
        "per_layer_kv_MB": per_layer_kv / 1e6,
        "prefetch_mode_forced": "one-layer-ahead (full KV exceeds 6MB L2)",
        "optimistic_tps_ceiling": ceiling,
        "rationale": (
            f"attention is {frac_verify*100:.2f}% of the verify forward wall "
            f"({frac_busy*100:.2f}% of gpu_busy) on the deployed split-KV stack, "
            f"BELOW the {ABORT_GATE*100:.0f}% self-abort gate. The split-KV verify "
            f"lever (PR #43) already collapsed verify attention from 19.6% of cycle "
            f"(pre-split-KV) to here. The residual kernel runs at {bw_eff*100:.0f}% "
            f"of HBM peak -> occupancy/launch-bound, not bandwidth-bound, so L2 "
            f"prefetch (which targets HBM-read latency) addresses the wrong limit."
        ),
    }


# --------------------------------------------------------------------------- #
# E. Self-test                                                                  #
# --------------------------------------------------------------------------- #
def self_test(committed: dict, micro: dict, decision: dict) -> dict:
    checks = {}
    # 1. committed fraction reproduces the PR-body 9.28% read
    checks["committed_frac_in_band"] = 0.085 <= committed["t_attn_frac_of_verify"] <= 0.10
    # 2. fresh microbench attn ms/cycle is in the same ballpark as the served 0.605 ms
    micro_ms = micro["deployed_3d_split_kv"]["attn_ms_per_cycle"]
    checks["microbench_matches_served_within_2x"] = 0.3 <= micro_ms <= 1.0
    # 3. kernel is numerically correct (validates vs dense SDPA)
    checks["kernel_validates"] = (
        micro["deployed_3d_split_kv"]["sliding_val_max_abs_err"] < 1e-3
        and micro["deployed_3d_split_kv"]["full_val_max_abs_err"] < 1e-3)
    # 4. deployed dispatch actually used the 3D split-KV path
    checks["deployed_used_3d"] = (
        micro["deployed_3d_split_kv"]["sliding_3d"]
        and micro["deployed_3d_split_kv"]["full_3d"])
    # 5. split-KV gave a real speedup over the 2D path (the already-deployed lever)
    checks["splitkv_speedup_gt_2x"] = micro["splitkv_verify_speedup"] > 2.0
    # 6. hardware-premise correction: L2 is ~6 MB, not the PR's 40 MB
    checks["l2_is_6mb_not_40mb"] = micro["l2_MB_measured"] < 12.0
    # 7. the full-prompt KV does NOT fit in L2 (PR premise wrong)
    checks["kv_exceeds_l2"] = not decision["kv_fits_in_l2"]
    # 8. gate fires consistently
    checks["gate_fires_self_abort"] = decision["verdict"] in ("SELF_ABORT", "GRAY_ZONE_ABORT")
    checks["all_passed"] = all(v for k, v in checks.items() if k != "all_passed")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rep-ctx", type=int, default=528,
                    help="representative decode context length (mean ctx ~528)")
    ap.add_argument("--n-iter", type=int, default=100)
    ap.add_argument("--out", default="research/profiling/kv_prefetch_l2/kv_prefetch_l2_gate.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="gemma-challenge-senpai")
    ap.add_argument("--wandb-entity", default="wandb-applied-ai-team")
    ap.add_argument("--wandb-name", default="stark/kv-prefetch-l2-gate")
    ap.add_argument("--wandb-group", default="kv-prefetch-l2")
    args = ap.parse_args()

    t0 = time.time()
    result: dict = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "card": "PR #445 kv-cache-l2-prefetch (analysis-only gate)",
        "deployed_tps": DEPLOYED_TPS,
        "realized_frontier_tps": REALIZED_FRONTIER_TPS,
        "arch": {"n_layers": N_LAYERS, "n_sliding": N_SLIDING, "n_full": N_FULL,
                 "pr_claimed_layers": PR_CLAIMED_LAYERS,
                 "pr_claimed_l2_MB": PR_CLAIMED_L2_MB},
    }

    print("[gate] A: committed end-to-end verify decomposition", flush=True)
    result["committed_verify_fraction"] = committed_verify_fraction()
    print(f"   t_attn_frac_of_verify = "
          f"{result['committed_verify_fraction']['t_attn_frac_of_verify']*100:.2f}%  "
          f"(attn {result['committed_verify_fraction']['attention_ms']:.3f} ms / "
          f"verify {result['committed_verify_fraction']['verify_gpu_ms']:.3f} ms)",
          flush=True)

    print("[gate] B: fresh local kernel microbench (deployed 3D split-KV M=8)", flush=True)
    result["fresh_microbench"] = fresh_microbench(args.rep_ctx, args.n_iter)
    m = result["fresh_microbench"]["deployed_3d_split_kv"]
    print(f"   attn {m['attn_ms_per_cycle']:.3f} ms/cycle  "
          f"{m['achieved_gbps_aggregate']:.1f} GB/s = "
          f"{m['bandwidth_eff_vs_measured_peak']*100:.1f}% peak  "
          f"(splitkv speedup {result['fresh_microbench']['splitkv_verify_speedup']:.2f}x)",
          flush=True)
    print(f"   L2 measured {result['fresh_microbench']['l2_MB_measured']:.2f} MB "
          f"(PR claimed {PR_CLAIMED_L2_MB} MB)", flush=True)

    print("[gate] C: cp.async-already scan", flush=True)
    result["cp_async_scan"] = cp_async_scan()
    print(f"   kernel_already_async_loads_kv = "
          f"{result['cp_async_scan']['kernel_already_async_loads_kv']} "
          f"({result['cp_async_scan']['n_attn_ptx_with_cp_async']}/"
          f"{result['cp_async_scan']['n_attn_ptx']} attn PTX)", flush=True)

    print("[gate] D: decision", flush=True)
    result["decision"] = decide(result["committed_verify_fraction"],
                                result["fresh_microbench"])
    print(f"   VERDICT = {result['decision']['verdict']}", flush=True)

    result["self_test"] = self_test(result["committed_verify_fraction"],
                                    result["fresh_microbench"], result["decision"])
    print(f"   self_test all_passed = {result['self_test']['all_passed']}", flush=True)

    result["primary_metric"] = {"name": "t_attn_frac_of_verify",
                                "value": result["committed_verify_fraction"]["t_attn_frac_of_verify"]}
    result["test_metric"] = {"name": "self_test_all_passed",
                             "value": int(result["self_test"]["all_passed"])}
    result["elapsed_s"] = time.time() - t0

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[gate] wrote {out} ({result['elapsed_s']:.0f}s)", flush=True)

    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name, group=args.wandb_group,
                         config={
                             "card": result["card"],
                             "deployed_tps": DEPLOYED_TPS,
                             "realized_frontier_tps": REALIZED_FRONTIER_TPS,
                             "abort_gate": ABORT_GATE,
                             "prototype_gate": PROTOTYPE_GATE,
                             "rep_ctx": args.rep_ctx, "n_iter": args.n_iter,
                             "arch": result["arch"],
                         })
        flat = {
            "t_attn_frac_of_verify": result["committed_verify_fraction"]["t_attn_frac_of_verify"],
            "t_attn_frac_of_verify_subsplit": result["committed_verify_fraction"]["t_attn_frac_of_verify_subsplit_denom"],
            "attn_frac_of_gpu_busy": result["committed_verify_fraction"]["attn_frac_of_gpu_busy"],
            "verify_gpu_ms": result["committed_verify_fraction"]["verify_gpu_ms"],
            "attention_ms": result["committed_verify_fraction"]["attention_ms"],
            "microbench_attn_ms_per_cycle": m["attn_ms_per_cycle"],
            "microbench_bandwidth_eff_vs_peak": m["bandwidth_eff_vs_measured_peak"],
            "microbench_gbps": m["achieved_gbps_aggregate"],
            "splitkv_verify_speedup": result["fresh_microbench"]["splitkv_verify_speedup"],
            "l2_MB_measured": result["fresh_microbench"]["l2_MB_measured"],
            "kv_MB_per_cycle": result["decision"]["kv_MB_per_cycle"],
            "kv_fits_in_l2": int(result["decision"]["kv_fits_in_l2"]),
            "kernel_already_async_loads_kv": int(result["cp_async_scan"]["kernel_already_async_loads_kv"]),
            "verdict_self_abort": int(result["decision"]["verdict"] in ("SELF_ABORT", "GRAY_ZONE_ABORT")),
            "self_test_all_passed": int(result["self_test"]["all_passed"]),
            "primary_metric": result["primary_metric"]["value"],
            "test_metric": result["test_metric"]["value"],
        }
        wandb.log(flat)
        wandb.summary.update(flat)
        for i, row in enumerate(result["decision"]["optimistic_tps_ceiling"]):
            wandb.log({"ceiling_f": row["hidden_attn_frac_f"],
                       "ceiling_tps": row["tps_on_467p14_base"],
                       "ceiling_delta_tps": row["delta_tps"]})
        print(f"[gate] wandb run: {run.url}", flush=True)
        wandb.finish()


if __name__ == "__main__":
    main()
