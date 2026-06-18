"""Attention 2D(num_splits=1) vs 3D(split-KV) device-time probe for the
Option-B BI=1 attribution (PR #623, instruction 4).

WHY: the A/B (paired_tps_ab) measures that VLLM_BATCH_INVARIANT=1 costs ~40%
single-stream TPS on the int4-body + gemma4_assistant MTP-K7 spec stack. The PR's
stated prime suspect is "num_splits=1 serializing the KV reduction". But the served
vLLM TRITON_ATTN kernel gates the 3D split-KV path off whenever ``max_seqlen_q > 1``
*independently* of batch-invariance (triton_unified_attention.py:929), so the M=8
spec-verify forward is already 2D in BOTH arms; BI=1's num_splits=1 forcing
(line 931) only changes the M=1 forwards.

This probe quantifies that: it drives the *real* served kernel
(``vllm.v1.attention.ops.triton_unified_attention.unified_attention``, the same one
the server loads) at M in {1, 8} (decode vs verify width) for both layer types
(sliding hd=256, full hd=512) at representative decode contexts, comparing
``dispatch='served'`` (3D split-KV where the kernel allows it -> the BI=0 path) vs
``dispatch='force2d'`` (num_splits=1 always -> the BI=1 path). Device-side kernel
time via torch.profiler (excludes launch overhead).

Output: per-(layer,M,ctx) device_us for 2D and 3D, the split-KV speedup, and the
37-layer target-cycle attention time under each, so the BI=1 attention-only delta
can be separated from the deterministic-GEMM tax (the A/B per-step residual).

LOCAL op-microbench: no server, no submission, no leaderboard number. Reuses the
validated bench_op from scripts/local_validation/profile_attention.py verbatim.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation.profile_attention import (  # noqa: E402
    N_FULL,
    N_SLIDING,
    bench_op,
)

OUT = Path(__file__).resolve().parent / "attn_2d_vs_3d_probe.json"

# Representative single-stream decode contexts for the 128 sharegpt eval prompts.
# Sliding window is 512, so >=512 saturates the sliding-layer KV read; sweep a
# short and a saturated ctx to show the 3D win is ctx-robust.
CTX_SWEEP = (256, 512, 1024)
M_VALUES = (1, 8)  # 1 = MTP-draft decode width ; 8 = K=7+1 spec-verify width
N_ITER = 200


def main() -> int:
    import torch

    assert torch.cuda.is_available(), "CUDA required"
    t0 = time.time()
    result: dict = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "purpose": "BI=1 attention num_splits=1 attribution (PR #623)",
        "kernel": "vllm.v1.attention.ops.triton_unified_attention.unified_attention",
        "n_iter": N_ITER,
        "target_cycle_layers": {"sliding": N_SLIDING, "full": N_FULL},
        "ops": [],
    }

    print(f"[probe] {result['gpu']}  ctx={CTX_SWEEP}  M={M_VALUES}", flush=True)
    for ctx in CTX_SWEEP:
        for lt in ("sliding", "full"):
            for M in M_VALUES:
                f3 = bench_op(torch, lt, M, ctx, dispatch="served",
                              n_iter=N_ITER, validate=(M == 1))
                f2 = bench_op(torch, lt, M, ctx, dispatch="force2d",
                              n_iter=N_ITER)
                row = {
                    "ctx": ctx, "layer_type": lt, "M": M,
                    "served_3d_used": bool(f3["used_3d_split_kv"]),
                    "served_us": f3["device_us"],
                    "force2d_us": f2["device_us"],
                    "bi1_attn_delta_us": f2["device_us"] - f3["device_us"],
                    "split_kv_speedup": (f3["device_us"] and
                                         f2["device_us"] / f3["device_us"]),
                    "served_gbps": f3["achieved_gbps_total"],
                    "force2d_gbps": f2["achieved_gbps_total"],
                    "validation_max_abs_err": (f3.get("validation") or {}).get("max_abs_err"),
                }
                result["ops"].append(row)
                print(f"   ctx={ctx:<5d} {lt:8s} M={M}: "
                      f"3D(used={int(row['served_3d_used'])})={f3['device_us']:7.1f}us "
                      f"2D={f2['device_us']:7.1f}us "
                      f"speedup={row['split_kv_speedup']:.2f}x "
                      f"Δ_bi1={row['bi1_attn_delta_us']:+.1f}us", flush=True)

    # ---- 37-layer target-cycle attention time: 2D (BI=1) vs 3D-where-allowed (BI=0)
    # at the saturated ctx=512. The target VERIFY forward is M=8 (already 2D both
    # arms); the M=1 column is the MTP-draft-head decode width.
    cyc = {}
    for ctx in CTX_SWEEP:
        rows = {(r["layer_type"], r["M"]): r for r in result["ops"] if r["ctx"] == ctx}
        for M in M_VALUES:
            s = rows[("sliding", M)]
            f = rows[("full", M)]
            served_cycle = N_SLIDING * s["served_us"] + N_FULL * f["served_us"]
            force2d_cycle = N_SLIDING * s["force2d_us"] + N_FULL * f["force2d_us"]
            cyc[f"ctx{ctx}_M{M}"] = {
                "served_3d_cycle_us": served_cycle,
                "force2d_cycle_us": force2d_cycle,
                "bi1_attn_cycle_delta_us": force2d_cycle - served_cycle,
                "bi1_attn_cycle_delta_ms": (force2d_cycle - served_cycle) / 1e3,
            }
    result["target_cycle_attn_us"] = cyc
    result["interpretation"] = (
        "BI=1 forces attention 2D (num_splits=1). The served TRITON_ATTN kernel "
        "already runs 2D for max_seqlen_q>1 (line 929), so the M=8 spec-verify "
        "attention is identical in both arms (Δ~0). Only the M=1 forwards (the "
        "gemma4_assistant MTP draft head, 4 attn layers) lose 3D split-KV under "
        "BI=1. Compare bi1_attn_cycle_delta_ms@M8 (target verify) ~0 vs the A/B "
        "measured per-step forward delta to isolate the deterministic-GEMM tax."
    )
    result["elapsed_s"] = time.time() - t0
    OUT.write_text(json.dumps(result, indent=2))
    print(f"[probe] wrote {OUT}  ({result['elapsed_s']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
