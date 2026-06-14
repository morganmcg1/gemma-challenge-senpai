# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Dual-stream HBM-contention probe (PR #94 Step-2 confirmation) -- LOCAL, no HF Job.

THE crux of the draft-verify overlap gate: at conc=1 both the verify forward and
the drafter forward are *memory-bandwidth-bound* (verify GEMM 77% HBM #68, drafter
chain 47% HBM #75).  Saguaro-style overlap puts the drafter on a *secondary CUDA
stream* concurrent with verify on the primary stream -- but they SHARE one A10G
HBM bus.  Does a drafter-sized memory-bound GEMM actually HIDE behind a
verify-sized one, or do the two streams contend on the bus and serialize?

Faithful proxy = skinny GEMM at M=8 (the deployed verify width): few query rows ->
SM-light, but streams a big weight matrix once -> bandwidth-bound, exactly the
decode regime.  We size:
  - "verify"  weight ~ the int4 base weight bytes/step (~2.25 GB)
  - "drafter" weight ~ the drafter chain bytes/step   (~0.16 GB)
and measure solo vs concurrent wall time on two streams.

overlap_efficiency = (t_verify_solo + t_drafter_solo - t_both) / t_drafter_solo
   1.0 -> drafter fully hidden (overlap works; bus has real slack)
   0.0 -> fully serialized (bus contention; no hiding)  <-- the A10G catch
"""
import argparse
import json
import time

import torch


def skinny_gemm_bytes(M, K, N, dtype_bytes):
    # bandwidth-bound: read W[K,N] once (dominant) + x[M,K] + write out[M,N]
    return dtype_bytes * (K * N + M * K + M * N)


def time_ops(ops, iters):
    """ops: list of (callable, stream). Enqueue all per iter across streams,
    device-sync, return mean wall ms/iter (= max stream time if overlapped,
    sum if serialized)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for fn, s in ops:
            with torch.cuda.stream(s):
                fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8, help="verify width (deployed=8)")
    ap.add_argument("--verify-gb", type=float, default=2.25,
                    help="verify weight bytes/step target (GB), #68")
    ap.add_argument("--drafter-gb", type=float, default=0.16,
                    help="drafter chain bytes/step target (GB), #75")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--output",
                    default="research/draft_verify_overlap/dual_stream_contention.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device"
    dev = "cuda"
    dt = torch.float16
    dtb = 2
    M = args.M
    name = torch.cuda.get_device_name(0)

    # pick square-ish K=N so K*N*2 ~= target bytes
    def kn_for_gb(gb):
        kn = gb * 1e9 / dtb
        k = int((kn) ** 0.5)
        k = (k // 64) * 64
        return k, k

    Kv, Nv = kn_for_gb(args.verify_gb)
    Kd, Nd = kn_for_gb(args.drafter_gb)

    xv = torch.randn(M, Kv, device=dev, dtype=dt)
    Wv = torch.randn(Kv, Nv, device=dev, dtype=dt)
    ov = torch.empty(M, Nv, device=dev, dtype=dt)
    xd = torch.randn(M, Kd, device=dev, dtype=dt)
    Wd = torch.randn(Kd, Nd, device=dev, dtype=dt)
    od = torch.empty(M, Nd, device=dev, dtype=dt)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    def verify_op():
        torch.mm(xv, Wv, out=ov)

    def drafter_op():
        torch.mm(xd, Wd, out=od)

    vbytes = skinny_gemm_bytes(M, Kv, Nv, dtb)
    dbytes = skinny_gemm_bytes(M, Kd, Nd, dtb)

    # warmup
    for _ in range(args.warmup):
        verify_op(); drafter_op()
    torch.cuda.synchronize()

    t_verify_solo = time_ops([(verify_op, s1)], args.iters)
    t_drafter_solo = time_ops([(drafter_op, s2)], args.iters)
    # symmetric: two verify-sized streams (clean bus-contention factor)
    Wv2 = torch.randn(Kv, Nv, device=dev, dtype=dt)
    ov2 = torch.empty(M, Nv, device=dev, dtype=dt)
    def verify_op2():
        torch.mm(xv, Wv2, out=ov2)
    for _ in range(args.warmup):
        verify_op2()
    torch.cuda.synchronize()
    t_verify2_solo = time_ops([(verify_op2, s2)], args.iters)
    t_two_verify = time_ops([(verify_op, s1), (verify_op2, s2)], args.iters)

    # asymmetric: drafter concurrent with verify (the real overlap question)
    t_both = time_ops([(verify_op, s1), (drafter_op, s2)], args.iters)

    overlap_eff = (t_verify_solo + t_drafter_solo - t_both) / t_drafter_solo
    # symmetric overlap speedup: 2.0 perfect, 1.0 serialized
    sym_speedup = (t_verify_solo + t_verify2_solo) / t_two_verify
    # bus-contention factor: combined achieved BW / additive solo BW (1=no contention,0.5=serialized)
    v_bw_solo = vbytes / (t_verify_solo * 1e-3) / 1e9
    v2_bw_solo = vbytes / (t_verify2_solo * 1e-3) / 1e9
    two_v_bw = 2 * vbytes / (t_two_verify * 1e-3) / 1e9
    contention_factor = two_v_bw / (v_bw_solo + v2_bw_solo)

    out = {
        "device": name,
        "config": {"M": M, "Kv": Kv, "Nv": Nv, "Kd": Kd, "Nd": Nd,
                   "verify_bytes_gb": round(vbytes / 1e9, 4),
                   "drafter_bytes_gb": round(dbytes / 1e9, 4),
                   "iters": args.iters, "warmup": args.warmup},
        "t_verify_solo_ms": round(t_verify_solo, 4),
        "t_verify2_solo_ms": round(t_verify2_solo, 4),
        "t_drafter_solo_ms": round(t_drafter_solo, 4),
        "t_two_verify_ms": round(t_two_verify, 4),
        "t_both_ms": round(t_both, 4),
        "verify_solo_gbs": round(v_bw_solo, 1),
        "two_verify_combined_gbs": round(two_v_bw, 1),
        "drafter_overlap_efficiency": round(overlap_eff, 4),
        "symmetric_overlap_speedup": round(sym_speedup, 4),
        "bus_contention_factor": round(contention_factor, 4),
        "interpretation": {
            "drafter_overlap_efficiency": "1.0=drafter fully hidden behind verify; 0.0=serialized (bus contention)",
            "symmetric_overlap_speedup": "2.0=perfect dual-stream overlap; 1.0=full serialization",
            "bus_contention_factor": "1.0=no HBM contention (additive BW); 0.5=full serialization on the bus",
        },
    }
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[dual-stream] wrote {args.output}")


if __name__ == "__main__":
    main()
