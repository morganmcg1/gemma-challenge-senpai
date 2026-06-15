#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""GPU read-bandwidth re-confirmation for the PR #283 HBM-bound ceiling (denken).

Light CUDA-event micro-benchmark that grounds denken #278's int4 body-read HBM floor
(1.76029696 GB / 600 GB/s = 2933.83 us) by MEASURING the A10G's achievable read bandwidth.
A reduction over a ~1.76 GB tensor forces a full read of every byte once -- the same physical
operation that bounds a single int4 body forward. Writes read_floor_confirm.json, which the
analytic harness reads (non-fatal) as a grounding cross-check. The analytic CENTRAL floor stays
the IMPORTED nominal-600 figure (matching #278); this only confirms the floor is realizable.

NOT a launch, NOT a submission, no served-file change. Local GPU profiling only.

Run:
    cd target/ && CUDA_VISIBLE_DEVICES=0 \
      /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
      research/validity/hbm_bound_tps_ceiling/read_floor_confirm.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "read_floor_confirm.json"

BODY_BYTES_GB = 1.76029696        # denken #278 int4 body + lm_head bytes (one verify read)
NOMINAL_BW_GBPS = 600.0           # A10G nominal HBM bandwidth (the #278 floor figure)
ANALYTIC_FLOOR_US = 2933.828266666667


def main() -> int:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        OUT.write_text(json.dumps({"present": False, "note": f"torch import failed: {exc}"}, indent=2))
        print(f"[read-floor-confirm] torch unavailable: {exc}")
        return 0

    if not torch.cuda.is_available():
        OUT.write_text(json.dumps({"present": False, "note": "cuda not available"}, indent=2))
        print("[read-floor-confirm] cuda not available")
        return 0

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(dev)
    # model-loading smoke test: a representative large allocation + a tiny matmul to warm the context.
    _ = (torch.randn(2048, 2048, device=dev, dtype=torch.float16)
         @ torch.randn(2048, 2048, device=dev, dtype=torch.float16))
    torch.cuda.synchronize()

    # allocate ~BODY_BYTES_GB of fp16 (2 bytes/elt) -> one .sum() reads every byte once.
    n_elt = int(BODY_BYTES_GB * 1e9 / 2)
    x = torch.ones(n_elt, device=dev, dtype=torch.float16)
    bytes_read = n_elt * 2
    torch.cuda.synchronize()

    warmup, iters = 10, 50
    for _ in range(warmup):
        x.sum()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        x.sum()
        e1.record()
        torch.cuda.synchronize()
        times_ms.append(e0.elapsed_time(e1))

    med_ms = statistics.median(times_ms)
    eff_bw_gbps = bytes_read / 1e9 / (med_ms / 1e3)
    measured_floor_us = BODY_BYTES_GB / eff_bw_gbps * 1e6   # GB/(GB/s)=s -> us
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    out = {
        "present": True,
        "kind": "read-floor-confirm",
        "pr": 283,
        "agent": "denken",
        "gpu": name,
        "body_bytes_gb": BODY_BYTES_GB,
        "bytes_read": bytes_read,
        "iters": iters,
        "median_read_ms": med_ms,
        "effective_read_bw_gbps": eff_bw_gbps,
        "nominal_bw_gbps": NOMINAL_BW_GBPS,
        "achievable_frac_of_nominal": eff_bw_gbps / NOMINAL_BW_GBPS,
        "measured_read_floor_us": measured_floor_us,
        "analytic_floor_us": ANALYTIC_FLOOR_US,
        "measured_vs_analytic_ratio": measured_floor_us / ANALYTIC_FLOOR_US,
        "peak_gpu_gb": peak_gb,
        "note": ("A reduction over a 1.76 GB fp16 tensor measures achievable READ bandwidth; the "
                 "analytic floor uses NOMINAL 600 GB/s (matching #278). Achievable < nominal is "
                 "expected; the analytic central floor stays the imported figure."),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[read-floor-confirm] {name}: eff_read_bw {eff_bw_gbps:.1f} GB/s "
          f"({100*eff_bw_gbps/NOMINAL_BW_GBPS:.0f}% of nominal), measured floor "
          f"{measured_floor_us:.1f}us vs analytic {ANALYTIC_FLOOR_US:.1f}us, peak {peak_gb:.2f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
