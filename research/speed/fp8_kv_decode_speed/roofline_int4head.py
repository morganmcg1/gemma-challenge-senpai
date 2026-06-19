#!/usr/bin/env python
"""PR #729 -- corrected HBM roofline for the int4_g128_lmhead submission.

The prior kv_read_fp8_lever roofline used a bf16 lm_head (3.51 GB weight read).
This submission has an INT4 untied lm_head + int4 body, and 67.6% of the .safetensors
is Per-Layer-Embedding tables that are GATHERED per token (negligible per-step read).
So the real per-step decode weight read is only ~2.396 GB, which makes KV reads a
slightly LARGER fraction. This recomputes kv_read_frac(L) and the OPTIMISTIC roofline
fp8-KV TPS uplift on the correct step-byte basis. Pure HBM roofline => upper bound;
the realized lever is smaller (fixed/compute overhead floor + fp8 dequant cost).

Architecture (text decoder, google/gemma-4-E4B-it):
  42 layers = 35 sliding (window 512) + 7 full; 2 KV heads; head_dim 256;
  kv bytes/pos/layer (bf16) = 2 (K+V) * 2 heads * 256 * 2 B = 2048.
"""
from __future__ import annotations

import json
from pathlib import Path

N_LAYERS = 42
N_SLIDING = 35
N_FULL = 7
WINDOW = 512
KV_BYTES_POS_LAYER_BF16 = 2 * 2 * 256 * 2  # 2048

# per-step decode reads on THIS submission (int4 body + int4 head; towers idle;
# embeddings/PLE gathered per token -> negligible)
WEIGHT_READ = 2.396e9          # int4 body 2.050 + int4 lm_head 0.346 GB
ACT_BYTES = 4.489e6
EFF_HBM_GBPS = 500.0           # measured eff. HBM ~500 GB/s (kv_read_fp8_lever / #555)

# benchmark trajectory facts (from kv_read_fp8_lever, official 128x512 sharegpt run)
BENCH_KV_MEAN = 527.66
BENCH_KV_MAX = 2938
DEPLOYED_MAX_MODEL_LEN = 4096


def kv_bytes_bf16(L: int) -> int:
    local_pos = min(L, WINDOW)
    return (N_SLIDING * local_pos + N_FULL * L) * KV_BYTES_POS_LAYER_BF16


def kv_bytes_fp8(L: int) -> float:
    # fp8 halves KV storage/read; scales are negligible
    return kv_bytes_bf16(L) / 2.0


def step_bytes(L: int, fp8: bool) -> float:
    kv = kv_bytes_fp8(L) if fp8 else kv_bytes_bf16(L)
    return WEIGHT_READ + ACT_BYTES + kv


def roofline_tps(L: int, fp8: bool) -> float:
    return EFF_HBM_GBPS * 1e9 / step_bytes(L, fp8)


def main():
    Ls = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    rows = {}
    crossover_10pct = None
    for L in Ls:
        sb_bf16 = step_bytes(L, False)
        sb_fp8 = step_bytes(L, True)
        kvb = kv_bytes_bf16(L)
        frac = kvb / sb_bf16
        # optimistic roofline fp8 uplift: TPS ~ 1/step_bytes
        uplift_pct = 100.0 * (sb_bf16 - sb_fp8) / sb_fp8
        rows[str(L)] = {
            "L": L,
            "kv_bytes_bf16_MB": kvb / 1e6,
            "kv_read_frac_bf16": frac,
            "roofline_tps_bf16": roofline_tps(L, False),
            "roofline_tps_fp8": roofline_tps(L, True),
            "fp8_uplift_pct_roofline_optimistic": uplift_pct,
            "reachable_in_deployed_cap": (L + 300) <= DEPLOYED_MAX_MODEL_LEN,
        }
        if frac >= 0.10 and crossover_10pct is None:
            crossover_10pct = L

    out = {
        "pr": 729, "basis": "int4_head_corrected_step_bytes",
        "weight_read_gib": WEIGHT_READ / 1024**3,
        "note": "OPTIMISTIC pure-HBM roofline upper bound; realized lever is smaller.",
        "scored_output_len": 512,
        "deployed_max_model_len": DEPLOYED_MAX_MODEL_LEN,
        "rows": rows,
        "fp8_uplift_pct_at_512_optimistic": rows["512"]["fp8_uplift_pct_roofline_optimistic"],
        "fp8_uplift_pct_at_2048_optimistic": rows["2048"]["fp8_uplift_pct_roofline_optimistic"],
        "fp8_uplift_pct_at_8192_optimistic": rows["8192"]["fp8_uplift_pct_roofline_optimistic"],
        "kv_read_frac_at_512": rows["512"]["kv_read_frac_bf16"],
        "kv_material_crossover_10pct_position": crossover_10pct,
        "crossover_beyond_deployed_cap": (crossover_10pct or 10**9) > DEPLOYED_MAX_MODEL_LEN,
        "crossover_beyond_8192": (crossover_10pct or 10**9) > 8192,
    }
    Path("research/speed/fp8_kv_decode_speed/roofline_int4head.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        "weight_read_gib", "fp8_uplift_pct_at_512_optimistic",
        "fp8_uplift_pct_at_2048_optimistic", "fp8_uplift_pct_at_8192_optimistic",
        "kv_read_frac_at_512", "kv_material_crossover_10pct_position",
        "crossover_beyond_deployed_cap")}, indent=2))


if __name__ == "__main__":
    main()
