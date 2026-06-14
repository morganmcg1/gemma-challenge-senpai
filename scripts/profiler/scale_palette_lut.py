#!/usr/bin/env python
"""Phase-1 lossless scale-PALETTE/LUT construction + bit-exactness proof (PR #110).

WHAT THIS MEASURES
------------------
The deployed frontier (`fa2sw_precache_kenyan` -> PLE-folded `osoi5-v0-baked`)
serves an int4 W4A16 (compressed-tensors / Marlin) body at group_size g=128.
Every verify-GEMM streams packed int4 weights PLUS an FP16 `weight_scale` tensor
(one scale per group of 128 weights, symmetric int4). The core-7 projection
scales total 26,849,280 values = **53.70 MB FP16** (227 tensors).

My PR #104 (KILL-but-banked) proved that *requantizing* those FP16 scales to
INT8 (QLoRA double-quant) is info-theoretically lossy here (bit-exact frac 13.1%
<< 98%). But it surfaced the successor: the core-7 scales take only ~1,009
DISTINCT FP16 values globally (per-tensor median ~427). That is a PALETTE: store
the distinct FP16 values once in a small LUT and replace each scale with a short
INDEX into that table. **Bit-exact by construction** -- the LUT holds the exact
original FP16 values, no requantization, no precision loss -> greedy-identity
preserved with zero risk (program.md L27-28).

THE LOAD-BEARING NUMBERS
------------------------
- `palette_bit_identical` = fraction of scales for which `palette[index[i]] == scale[i]`
  comparing raw FP16 bit patterns. MUST be **1.0** by construction (if not, the
  distinct-count is wrong -- a bug, not a lossy result).
- `palette_scale_byte_saving_pct` (PRIMARY metric) = honest byte saving of the
  best lossless palette encoding vs the 53.70 MB FP16 baseline:
  (LUT table bytes + packed index bytes) vs original 2*N FP16 bytes.

ENCODINGS ACCOUNTED (all lossless)
----------------------------------
- GLOBAL palette: one shared table of the U distinct FP16 values across all 227
  core-7 tensors; every scale -> ceil(log2 U)-bit index (10-bit for U~=1009).
    bytes = U*2  +  N*ceil(log2 U)/8
- PER-TENSOR palette: each tensor t gets its own table of u_t distinct values and
  a ceil(log2 u_t)-bit index (9-bit nominal; falls back per-tensor to wider when
  u_t > 512). Smaller indices, but T small tables.
    bytes = sum_t [ u_t*2  +  n_t*ceil(log2 u_t)/8 ]

This is a PURE CPU scan: it reads only the small `weight_scale` tensors from the
safetensors file (never the 9 GB of packed weights), builds palettes in numpy,
and never touches the served token stream -> lossless by construction. JSON dump
+ optional W&B (group scale-palette-lut).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
from safetensors import safe_open

# Deployed frontier (PLE-folded g=128 W4A16). Single-file safetensors in /tmp.
DEFAULT_CKPT = "/tmp/osoi5-v0-baked/model.safetensors"
CORE7 = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Baseline scale payload (PR #104 banked, confirmed g128 osoi5-v0-baked).
BASELINE_SCALE_MB = 53.70
INT4_BODY_MB = 1754.7          # #104: int4 language_model body bytes
# Decode-step framing (informational): verify-GEMM share of the conc=1 step.
VERIFY_GEMM_FRAC = 0.532       # #30: verify-GEMM = 53.2% of decode step
WALL_TPS = 454.338             # lawine #90 locked linear-chain local reference


def role_of(name: str) -> str:
    for r in CORE7:
        if f".{r}." in name:
            return r
    return "other_lm"


def iter_scale_tensors(ckpt: str):
    """Yield (name, fp16_ndarray) for every language_model .weight_scale tensor."""
    files = [ckpt] if os.path.isfile(ckpt) else sorted(glob.glob(os.path.join(ckpt, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors at {ckpt}")
    for fp in files:
        with safe_open(fp, framework="numpy") as f:
            for k in f.keys():
                if k.endswith(".weight_scale") and ".language_model." in k:
                    yield k, f.get_tensor(k)


def index_bits(n_distinct: int) -> int:
    """Minimum bits to index `n_distinct` palette entries (>=1)."""
    return max(1, math.ceil(math.log2(max(n_distinct, 2))))


def build_palette(scales_u16: np.ndarray):
    """Lossless palette of an fp16 array (passed as its uint16 bit pattern).

    Returns (palette_u16, index_array_int64). `np.unique(..., return_inverse=True)`
    gives BOTH the sorted distinct values AND the index that reconstructs the
    input exactly: palette[index] == scales_u16 elementwise, by construction.
    """
    palette, index = np.unique(scales_u16, return_inverse=True)
    return palette, index.astype(np.int64)


def verify_bit_exact(scales_u16: np.ndarray, palette_u16: np.ndarray, index: np.ndarray) -> int:
    """Count elements where palette[index[i]] == scale[i] on raw fp16 bit patterns."""
    recon = palette_u16[index]
    return int((recon == scales_u16).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--per-tensor-nominal-bits", type=int, default=9,
                    help="nominal per-tensor index width; falls back wider when a "
                         "tensor exceeds 2**bits distinct values")
    ap.add_argument("--output", default="research/scale_palette_lut/scale_palette_lut.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="scale-palette-lut")
    ap.add_argument("--wandb_name", default="wirbel/scale-palette-lut-phase1")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    print(f"[palette] scanning {args.checkpoint}", flush=True)
    t0 = time.time()

    # ---- enumerate core-7 weight_scale tensors as raw uint16 bit patterns ------
    per_tensor = []            # list of dict(name, role, n, u16)
    core7_chunks = []
    for name, arr in iter_scale_tensors(args.checkpoint):
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        u16 = np.ascontiguousarray(arr.reshape(-1)).view(np.uint16)
        role = role_of(name)
        if role not in CORE7:
            continue
        per_tensor.append({"name": name, "role": role, "n": int(u16.size), "u16": u16})
        core7_chunks.append(u16)

    if not per_tensor:
        raise RuntimeError("no core-7 weight_scale tensors found")
    cat = np.concatenate(core7_chunks)
    N = int(cat.size)
    n_tensors = len(per_tensor)
    orig_bytes = 2 * N
    print(f"[palette] {n_tensors} core-7 scale tensors | {N:,} scales "
          f"({orig_bytes/1e6:.2f} MB FP16) loaded in {time.time()-t0:.1f}s", flush=True)

    # ====================== GLOBAL palette (one shared LUT) =====================
    g_palette, g_index = build_palette(cat)
    U = int(g_palette.size)
    g_bits = index_bits(U)
    g_be = verify_bit_exact(cat, g_palette, g_index)
    g_bit_identical = g_be / N
    g_lut_bytes = U * 2
    g_index_bytes = math.ceil(N * g_bits / 8)
    g_total_bytes = g_lut_bytes + g_index_bytes
    g_saved_bytes = orig_bytes - g_total_bytes
    g_saved_pct = 100.0 * g_saved_bytes / orig_bytes

    print(f"\n[palette] === GLOBAL palette ===", flush=True)
    print(f"[palette]   distinct FP16 values (core7 global) = {U:,} -> {g_bits}-bit index", flush=True)
    print(f"[palette]   bit_identical = {g_bit_identical:.6f}  ({g_be:,}/{N:,})", flush=True)
    print(f"[palette]   bytes: LUT {g_lut_bytes/1e6:.3f} MB + index {g_index_bytes/1e6:.2f} MB "
          f"= {g_total_bytes/1e6:.2f} MB vs {orig_bytes/1e6:.2f} MB FP16", flush=True)
    print(f"[palette]   saved {g_saved_bytes/1e6:+.2f} MB ({g_saved_pct:+.2f}% of scale bytes)", flush=True)

    # ==================== PER-TENSOR palettes (adaptive bits) ===================
    nominal_bits = args.per_tensor_nominal_bits
    nominal_cap = 2 ** nominal_bits
    pt_rows = []
    pt_lut_bytes = 0
    pt_index_bytes = 0
    pt_be_total = 0
    pt_uniq_counts = []
    n_fallback = 0
    for rec in per_tensor:
        u16 = rec["u16"]
        pal, idx = build_palette(u16)
        u = int(pal.size)
        be = verify_bit_exact(u16, pal, idx)
        bits = index_bits(u)                       # exact min bits for this tensor
        used_bits = max(nominal_bits, bits) if u > nominal_cap else nominal_bits
        # honest: use the minimum bits that index this tensor (<= nominal unless it
        # genuinely overflows -> fallback wider). A tensor with u<=512 uses 9 bits.
        used_bits = bits if bits > nominal_bits else nominal_bits
        if bits > nominal_bits:
            n_fallback += 1
        lut_b = u * 2
        idx_b = math.ceil(rec["n"] * used_bits / 8)
        pt_lut_bytes += lut_b
        pt_index_bytes += idx_b
        pt_be_total += be
        pt_uniq_counts.append(u)
        pt_rows.append({"name": rec["name"], "role": rec["role"], "n": rec["n"],
                        "distinct": u, "index_bits": used_bits,
                        "bit_identical": be / rec["n"]})

    pt_total_bytes = pt_lut_bytes + pt_index_bytes
    pt_saved_bytes = orig_bytes - pt_total_bytes
    pt_saved_pct = 100.0 * pt_saved_bytes / orig_bytes
    pt_bit_identical = pt_be_total / N
    pt_uniq = np.asarray(pt_uniq_counts)
    print(f"\n[palette] === PER-TENSOR palettes (nominal {nominal_bits}-bit, "
          f"fallback when distinct>{nominal_cap}) ===", flush=True)
    print(f"[palette]   per-tensor distinct: min={pt_uniq.min()} median={int(np.median(pt_uniq))} "
          f"p90={int(np.percentile(pt_uniq,90))} max={pt_uniq.max()}  "
          f"({n_fallback}/{n_tensors} tensors exceed {nominal_cap} -> wider index)", flush=True)
    print(f"[palette]   bit_identical = {pt_bit_identical:.6f}", flush=True)
    print(f"[palette]   bytes: LUT {pt_lut_bytes/1e6:.3f} MB + index {pt_index_bytes/1e6:.2f} MB "
          f"= {pt_total_bytes/1e6:.2f} MB", flush=True)
    print(f"[palette]   saved {pt_saved_bytes/1e6:+.2f} MB ({pt_saved_pct:+.2f}% of scale bytes)", flush=True)

    # ---- per-role distinct breakdown ------------------------------------------
    role_agg = defaultdict(lambda: {"n": 0, "tensors": 0, "distinct_vals": []})
    for r in pt_rows:
        role_agg[r["role"]]["n"] += r["n"]
        role_agg[r["role"]]["tensors"] += 1
        role_agg[r["role"]]["distinct_vals"].append(r["distinct"])
    role_rows = []
    for role, d in sorted(role_agg.items(), key=lambda kv: -kv[1]["n"]):
        dv = np.asarray(d["distinct_vals"])
        role_rows.append({"role": role, "n": d["n"], "tensors": d["tensors"],
                          "distinct_median": int(np.median(dv)), "distinct_max": int(dv.max())})
    print(f"\n[palette] === per-role distinct (median / max) ===", flush=True)
    for r in role_rows:
        print(f"[palette]   {r['role']:>10s} | tensors={r['tensors']:3d} n={r['n']:>9,} | "
              f"distinct median={r['distinct_median']:4d} max={r['distinct_max']:4d}", flush=True)

    # ---- pick the best lossless scheme ----------------------------------------
    best_scheme = "per_tensor" if pt_saved_bytes >= g_saved_bytes else "global"
    best_saved_pct = max(pt_saved_pct, g_saved_pct)
    best_saved_mb = max(pt_saved_bytes, g_saved_bytes) / 1e6
    palette_bit_identical = min(g_bit_identical, pt_bit_identical)   # both must be 1.0

    # ---- analytical Phase-2 CEILING (modeled, NOT the empirical gate) ----------
    # If scale bytes are FULLY on the verify-GEMM DRAM critical path AND the GEMM
    # is purely bandwidth-bound, the wall_tps ceiling = scale_saving / total_stream
    # * verify_gemm_frac. The int4 body (1754.7 MB) is the weight stream; scales are
    # 53.70 MB on top. This is the OPTIMISTIC upper bound; the group-size sweep
    # (scale_palette_bw_probe.py) measures the REALIZED fraction (cache/overlap).
    stream_mb = INT4_BODY_MB + BASELINE_SCALE_MB
    bw_saved_pct_of_stream = 100.0 * best_saved_mb / stream_mb
    modeled_ceiling_tps_pct = bw_saved_pct_of_stream * VERIFY_GEMM_FRAC

    print(f"\n[palette] ===================== SUMMARY =====================", flush=True)
    print(f"[palette]   palette_bit_identical = {palette_bit_identical:.6f}  "
          f"(MUST be 1.0 -- lossless by construction)", flush=True)
    print(f"[palette]   BEST scheme = {best_scheme}: saved {best_saved_mb:+.2f} MB "
          f"= {best_saved_pct:+.2f}% of scale bytes", flush=True)
    print(f"[palette]   modeled BW-ceiling TPS lift (IF fully on critical path) "
          f"~= {modeled_ceiling_tps_pct:+.3f}%  (the gate measures the realized fraction)", flush=True)

    payload = {
        "config": {
            "checkpoint": args.checkpoint, "group_size": 128,
            "n_core7_scale_tensors": n_tensors, "n_core7_scales": N,
            "core7_scale_MB_fp16": orig_bytes / 1e6,
            "baseline_scale_MB": BASELINE_SCALE_MB, "int4_body_MB": INT4_BODY_MB,
            "verify_gemm_frac": VERIFY_GEMM_FRAC, "wall_tps": WALL_TPS,
            "per_tensor_nominal_bits": nominal_bits,
            "note": "CPU-only scan of deployed FP16 weight_scale tensors; never reads "
                    "packed weights or the token stream. Palette holds EXACT original "
                    "FP16 values -> lossless / greedy-identity-preserving by construction.",
        },
        "global_palette": {
            "distinct": U, "index_bits": g_bits, "bit_identical": g_bit_identical,
            "lut_bytes": g_lut_bytes, "index_bytes": g_index_bytes,
            "total_bytes": g_total_bytes, "orig_bytes": orig_bytes,
            "saved_bytes": g_saved_bytes, "saved_pct": g_saved_pct,
        },
        "per_tensor_palette": {
            "nominal_bits": nominal_bits, "n_fallback": n_fallback,
            "distinct_min": int(pt_uniq.min()), "distinct_median": int(np.median(pt_uniq)),
            "distinct_p90": int(np.percentile(pt_uniq, 90)), "distinct_max": int(pt_uniq.max()),
            "bit_identical": pt_bit_identical,
            "lut_bytes": pt_lut_bytes, "index_bytes": pt_index_bytes,
            "total_bytes": pt_total_bytes, "orig_bytes": orig_bytes,
            "saved_bytes": pt_saved_bytes, "saved_pct": pt_saved_pct,
        },
        "per_role": role_rows,
        "per_tensor_rows": pt_rows,
        "verdict": {
            "palette_bit_identical": palette_bit_identical,
            "best_scheme": best_scheme,
            "palette_scale_byte_saving_pct": best_saved_pct,
            "palette_scale_byte_saving_mb": best_saved_mb,
            "modeled_bw_ceiling_tps_pct": modeled_ceiling_tps_pct,
            "bw_saved_pct_of_verify_stream": bw_saved_pct_of_stream,
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[palette] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:   # noqa: BLE001
            print(f"[palette] W&B logging failed: {exc!r}", flush=True)


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    # per-role table
    rtbl = wandb.Table(columns=["role", "tensors", "n", "distinct_median", "distinct_max"])
    for r in payload["per_role"]:
        rtbl.add_data(r["role"], r["tensors"], r["n"], r["distinct_median"], r["distinct_max"])
    run.log({"per_role_table": rtbl})
    # per-tensor distinct histogram (as a table of distinct counts)
    dtbl = wandb.Table(columns=["name", "role", "n", "distinct", "index_bits"])
    for r in payload["per_tensor_rows"]:
        dtbl.add_data(r["name"], r["role"], r["n"], r["distinct"], r["index_bits"])
    run.log({"per_tensor_table": dtbl})

    v = payload["verdict"]
    g = payload["global_palette"]
    pt = payload["per_tensor_palette"]
    run.summary.update({
        "palette_bit_identical": v["palette_bit_identical"],
        "palette_scale_byte_saving_pct": v["palette_scale_byte_saving_pct"],
        "palette_scale_byte_saving_mb": v["palette_scale_byte_saving_mb"],
        "palette_best_scheme_pertensor": int(v["best_scheme"] == "per_tensor"),
        "modeled_bw_ceiling_tps_pct": v["modeled_bw_ceiling_tps_pct"],
        "bw_saved_pct_of_verify_stream": v["bw_saved_pct_of_verify_stream"],
        "global_distinct": g["distinct"], "global_index_bits": g["index_bits"],
        "global_saved_pct": g["saved_pct"], "global_bit_identical": g["bit_identical"],
        "pertensor_distinct_median": pt["distinct_median"],
        "pertensor_distinct_max": pt["distinct_max"],
        "pertensor_n_fallback": pt["n_fallback"],
        "pertensor_saved_pct": pt["saved_pct"], "pertensor_bit_identical": pt["bit_identical"],
        "n_core7_scales": payload["config"]["n_core7_scales"],
        "core7_scale_MB_fp16": payload["config"]["core7_scale_MB_fp16"],
    })
    run.finish()
    print(f"[palette] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
