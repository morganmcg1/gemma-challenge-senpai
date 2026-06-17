"""PR #544 -- base_fullhead TPS-ceiling: lm_head verify-tax micro-benchmark.

LOCAL, GPU, analysis-only. Isolates the TARGET-model verify head cost (the
"262k-head argmax + verify tax" the PR asks to quantify) from the body/attention.

The base_fullhead verify head is the STOCK dense bf16 ``lm_head.weight`` of
``google/gemma-4-E4B-it-qat-w4a16-ct`` ([262144, 2560]). The osoi5 ship head is an
int4-Marlin-packed [16384, 320] head (~16k rows). At MAX_NUM_SEQS=1 with K=7 spec
verify, the head is applied to M+1=8 query positions per step (we also bench 1 row
= the pure-AR/draft case).

We measure the REAL bf16 head matmul (hidden @ head.T) + greedy argmax with CUDA
events, at vocab in {262144, 16384(slice)} and rows in {1, 8}. The int4 cost is
modeled by the weight-read bandwidth ratio (decode head matmul is weight-bandwidth
bound: FLOPs at 8x2560x262144 ~ 10.7 GFLOP/0.15ms-compute vs 1.34GB/~2.2ms-read),
with the effective HBM bandwidth derived from the measured bf16 timing so the int4
projection is anchored to this exact GPU, not a datasheet number.

Outputs head_results.json:
  h_262k_bf16_ms[rows]   measured dense bf16 262k head (base_fullhead's actual head)
  h_16k_bf16_ms[rows]    measured dense bf16 16k-slice (pure vocab-size effect)
  argmax_only_ms[V]      argmax-over-V reduction cost alone
  eff_hbm_gbps           effective bandwidth back-solved from the 262k bf16 matmul
  h_262k_int4_ms[rows]   int4-262k projection (full head kept, weights int4) = recoverable lever
  h_16k_int4_ms[rows]    int4-16k projection (= osoi5 head, cross-check vs serve residual)

Run under the SERVE venv (needs torch + CUDA), GPU must be free::

    /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
        research/validity/base_fullhead_tps_ceiling/microbench_head.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch


def _resolve_qat_snapshot() -> Path:
    base = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    snaps = sorted(p for p in base.glob("*") if (p / "config.json").exists())
    if not snaps:
        raise RuntimeError(f"no qat-w4a16-ct snapshot under {base}")
    return snaps[0]


def _load_lm_head_bf16(snapshot: Path) -> torch.Tensor:
    """Load lm_head.weight [V, H] bf16 from the safetensors via offset slice."""
    from safetensors import safe_open

    with safe_open(str(snapshot / "model.safetensors"), framework="pt", device="cpu") as f:
        w = f.get_tensor("lm_head.weight")
    return w.to(torch.bfloat16)


def _time_ms(fn, *, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]  # median ms


def main() -> int:
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot = _resolve_qat_snapshot()
    print(f"[mbhead] snapshot={snapshot}", flush=True)

    head = _load_lm_head_bf16(snapshot).to(dev)  # [262144, 2560]
    V_full, H = head.shape
    assert V_full == 262144 and H == 2560, f"unexpected head shape {head.shape}"
    print(f"[mbhead] head={tuple(head.shape)} dtype={head.dtype} "
          f"bytes_full={head.numel()*2/1e9:.3f}GB", flush=True)

    # Vocab sizes to bench: full 262k (base_fullhead's actual head), 16384 (osoi5-v0
    # baked), 12288 (the ACTUAL served osoi5 ship head after LM_HEAD_PRUNE -> 12k).
    vocabs = [V_full, 16384, 12288]
    rows_set = [1, 8]
    int4_byte_per_elt = 0.5  # 4-bit packed weights (group-scale overhead negligible)
    head_t = {V: (head if V == V_full else head[:V]).t().contiguous() for V in vocabs}

    results: dict = {
        "schema": "base_fullhead_microbench_head_v2",
        "analysis_only": True,
        "snapshot": str(snapshot),
        "vocab_full": V_full, "vocab_osoi5_baked": 16384, "vocab_osoi5_served": 12288,
        "hidden": H, "rows_set": rows_set, "vocabs": vocabs,
        "int4_byte_per_elt": int4_byte_per_elt,
        "device": torch.cuda.get_device_name(0),
        # keyed "V/rows" -> ms
        "matmul_bf16_ms": {}, "head_bf16_ms": {}, "argmax_ms": {}, "head_int4_ms": {},
    }

    for V in vocabs:
        Wt = head_t[V]
        for rows in rows_set:
            x = torch.randn(rows, H, device=dev, dtype=torch.bfloat16)
            logits = x @ Wt

            mm_ms = _time_ms(lambda: x @ Wt)
            head_ms = _time_ms(lambda: (x @ Wt).argmax(dim=-1))
            amax_ms = _time_ms(lambda: logits.argmax(dim=-1))
            results["matmul_bf16_ms"][f"{V}/{rows}"] = mm_ms
            results["head_bf16_ms"][f"{V}/{rows}"] = head_ms
            results["argmax_ms"][f"{V}/{rows}"] = amax_ms
            print(f"[mbhead] V={V} rows={rows}: matmul={mm_ms:.4f} head={head_ms:.4f} "
                  f"argmax={amax_ms:.4f} ms", flush=True)

    # Effective HBM bandwidth back-solved from the 8-row 262k bf16 MATMUL (weight-read
    # bound: 1.342 GB weight read dominates the ~0.15ms compute). Anchors the int4
    # projection to THIS GPU rather than a datasheet number.
    t_mm_full_8 = results["matmul_bf16_ms"][f"{V_full}/8"] / 1e3  # s
    eff_bw_gbps = (V_full * H * 2 / 1e9) / t_mm_full_8
    results["eff_hbm_gbps"] = eff_bw_gbps
    for V in vocabs:
        for rows in rows_set:
            amax_ms = results["argmax_ms"][f"{V}/{rows}"]
            mm_int4 = (V * H * int4_byte_per_elt / 1e9) / eff_bw_gbps * 1e3
            results["head_int4_ms"][f"{V}/{rows}"] = mm_int4 + amax_ms

    print(f"[mbhead] eff_hbm_gbps={eff_bw_gbps:.1f} | "
          f"head262k_bf16(8)={results['head_bf16_ms'][f'{V_full}/8']:.4f} "
          f"head12k_int4(8)={results['head_int4_ms']['12288/8']:.4f} "
          f"head262k_int4(8)={results['head_int4_ms'][f'{V_full}/8']:.4f} ms", flush=True)

    out = Path(__file__).resolve().parent / "head_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[mbhead] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
