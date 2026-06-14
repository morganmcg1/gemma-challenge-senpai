#!/usr/bin/env python
"""CPU bit-exact round-trip scan for INT8 double-quant of the verify-GEMM FP16
group-scales (PR #104, build-or-kill gate).

WHAT THIS MEASURES
------------------
The deployed frontier (`fa2sw_precache_kenyan` -> PLE-folded `osoi5-v0-baked`)
serves an int4 W4A16 (compressed-tensors / Marlin) body. Every verify-GEMM
streams packed int4 weights PLUS an FP16 `weight_scale` tensor (one scale per
group of `g=128` weights, symmetric int4). The hypothesis: double-quantize those
FP16 scales to INT8 (QLoRA arXiv:2305.14314 §2.2 style), keeping ONLY the scales
whose INT8 round-trip reconstructs the original FP16 scale **bit-for-bit**, and
storing the rest as FP16 "sparse exceptions" (SqueezeLLM dense-and-sparse,
arXiv:2306.07629). Bit-exact-by-construction => zero flipped greedy tokens
(program.md L27-28).

THE LOAD-BEARING NUMBER
-----------------------
`dq_scale_roundtrip_bitexact_frac` = fraction of FP16 scales for which
`dequant(quant(s_i)) == s_i` comparing raw FP16 bit patterns (NOT allclose).

GATE
----
- GREEN if bit-exact frac > 98% AND the lossless hybrid is net-byte-positive.
- KILL  if the FP16 sparse-exception set wipes out the saving (hybrid bytes
  >= original FP16 bytes), i.e. too few scales round-trip bit-exactly.

BIT-EXACTNESS THEORY (sanity backstop for the measured number)
--------------------------------------------------------------
Asymmetric INT8 over a secondary block spanning [min,max]: step c=(max-min)/255,
max reconstruction error c/2. FP16 (1s/5e/10m) ULP at value v is 2^(floor(log2 v)
-10); a scale round-trips bit-exactly iff |s_hat-s|<1/2 ULP(s). A whole block is
bit-exact-by-guarantee when c < ULP(min), i.e. relative spread (max-min)/min <
255/1024 ~= 0.249. So bit-exactness is gated by the WITHIN-secondary-block
dynamic range of the scales -- a purely empirical property of this checkpoint.

This is a PURE CPU scan: it reads only the small `weight_scale` tensors from the
safetensors file (never the 9 GB of packed weights), simulates the round-trip in
numpy, and never touches the served token stream -> lossless by construction.
JSON dump + optional W&B (group dq-verify-gemm-scales).
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

# A10G HBM for the (informational) bandwidth->TPS framing if GREEN.
A10G_HBM_GBS = 600.0
VERIFY_GEMM_FRAC = 0.532       # #30: verify-GEMM = 53.2% of conc=1 decode step
WALL_TPS = 454.338             # lawine #90 locked linear-chain local reference


# --------------------------------------------------------------------------- #
# round-trip primitives
# --------------------------------------------------------------------------- #
def _u16(a: np.ndarray) -> np.ndarray:
    """Raw bit pattern of an fp16 array as uint16 (for bit-exact comparison)."""
    return a.view(np.uint16)


def roundtrip_int8_asym(s_fp16: np.ndarray, block: int) -> np.ndarray:
    """Asymmetric INT8 double-quant round-trip, per secondary block of `block`.

    Per block: offset=min, c=(max-min)/255 (FP32), q=round((s-min)/c) in [0,255],
    s_hat=q*c+min, cast back to FP16. Degenerate (constant) blocks pass through.
    Returns the reconstructed FP16 array (same shape as input).
    """
    s = np.ascontiguousarray(s_fp16, dtype=np.float16)
    n = s.size
    rec = np.empty_like(s)

    def _do(arr2d: np.ndarray) -> np.ndarray:        # arr2d: [B, block] fp16
        b = arr2d.astype(np.float32)
        mn = b.min(axis=1, keepdims=True)
        mx = b.max(axis=1, keepdims=True)
        c = (mx - mn) / 255.0
        zero = c == 0.0
        cc = np.where(zero, 1.0, c)
        q = np.clip(np.round((b - mn) / cc), 0.0, 255.0)
        out = q * c + mn
        out = np.where(zero, b, out)
        return out.astype(np.float16)

    nfull = n // block
    if nfull:
        head = s[: nfull * block].reshape(nfull, block)
        rec[: nfull * block] = _do(head).reshape(-1)
    if n % block:
        tail = s[nfull * block :].reshape(1, -1)
        rec[nfull * block :] = _do(tail).reshape(-1)
    return rec


def block_relspread_stats(s_fp16: np.ndarray, block: int) -> dict:
    """Within-secondary-block relative spread (max-min)/min distribution and the
    fraction of blocks below the 0.249 bit-exact-guarantee threshold."""
    s = np.ascontiguousarray(s_fp16, dtype=np.float32)
    n = s.size
    nfull = n // block
    spreads = []
    if nfull:
        b = s[: nfull * block].reshape(nfull, block)
        mn = b.min(axis=1)
        mx = b.max(axis=1)
        ok = mn > 0
        spreads = ((mx[ok] - mn[ok]) / mn[ok]).tolist()
    if not spreads:
        return {"median_relspread": None, "p90_relspread": None, "frac_blocks_below_0p249": None}
    arr = np.asarray(spreads)
    return {
        "median_relspread": float(np.median(arr)),
        "p90_relspread": float(np.percentile(arr, 90)),
        "frac_blocks_below_0p249": float((arr < 255.0 / 1024.0).mean()),
    }


# --------------------------------------------------------------------------- #
# checkpoint scale enumeration
# --------------------------------------------------------------------------- #
def role_of(name: str) -> str:
    for r in CORE7:
        if f".{r}." in name or name.endswith(f".{r}.weight_scale"):
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


# --------------------------------------------------------------------------- #
# byte accounting
# --------------------------------------------------------------------------- #
def hybrid_bytes(n_total: int, n_exc: int, n_blocks: int) -> dict:
    """Bytes for the lossless hybrid scale buffer under three exception encodings.

    We credit the MOST byte-favorable lossless encoding (best chance for the lever).
    n_be = n_total - n_exc bit-exact scales (INT8), n_exc FP16 exceptions.

    original       : 2*N                                     (FP16)
    csr_dense      : N*1     + n_exc*6 + ovh   (SqueezeLLM dense+sparse: INT8 all + FP16 val + int32 idx)
    bitmap_dense   : N*1     + n_exc*2 + N/8 + ovh   (dense INT8 all + FP16 patch + 1 presence bit/scale)
    bitmap_sparse  : n_be*1  + n_exc*2 + N/8 + ovh   (INT8 only for bit-exact + FP16 exc + bitmap)  <- usually best
    ovh (secondary scale+offset per block, FP32 each) = n_blocks*8 (0.03 B/scale @ block256; negligible)
    """
    ovh = n_blocks * 8
    bitmap = math.ceil(n_total / 8)
    orig = 2 * n_total
    n_be = n_total - n_exc
    csr_dense = n_total * 1 + n_exc * 6 + ovh
    bitmap_dense = n_total * 1 + n_exc * 2 + bitmap + ovh
    bitmap_sparse = n_be * 1 + n_exc * 2 + bitmap + ovh
    best = min(csr_dense, bitmap_dense, bitmap_sparse)
    return {
        "orig_bytes": orig, "csr_dense_bytes": csr_dense,
        "bitmap_dense_bytes": bitmap_dense, "bitmap_sparse_bytes": bitmap_sparse,
        "best_bytes": best, "overhead_bytes": ovh,
        "saved_bytes_best": orig - best,
        "saved_frac_of_scale_bytes_best": (orig - best) / orig if orig else 0.0,
    }


# --------------------------------------------------------------------------- #
# main scan
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="safetensors file or dir of the deployed W4A16 checkpoint")
    ap.add_argument("--group-size", type=int, default=128,
                    help="primary weight group size (informational; read from shapes)")
    ap.add_argument("--secondary-block", type=int, default=256,
                    help="QLoRA secondary block size for the scale-of-scales quant")
    ap.add_argument("--block-sweep", default="256,128,64,32,16",
                    help="secondary block sizes to sweep for the best-chance frontier")
    ap.add_argument("--green-threshold", type=float, default=0.98)
    ap.add_argument("--output", default="research/dq_verify_gemm_scales/dq_scale_roundtrip.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="dq-verify-gemm-scales")
    ap.add_argument("--wandb_name", default="wirbel/dq-roundtrip-g128")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    primary_block = args.secondary_block
    sweep = [int(x) for x in args.block_sweep.split(",") if x.strip()]
    if primary_block not in sweep:
        sweep = [primary_block] + sweep

    print(f"[dq] scanning {args.checkpoint}", flush=True)
    t0 = time.time()

    # Load all language_model scale tensors into a per-tensor record.
    per_tensor = []          # list of dict(name, role, n, scales_fp16, in_feat?, n_groups)
    scope_scales = {"core7": [], "all": []}     # concatenated fp16 arrays
    for name, arr in iter_scale_tensors(args.checkpoint):
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        flat = arr.reshape(-1)
        role = role_of(name)
        rec = {"name": name, "role": role, "n": int(flat.size),
               "shape": list(arr.shape), "scales": flat}
        per_tensor.append(rec)
        scope_scales["all"].append(flat)
        if role in CORE7:
            scope_scales["core7"].append(flat)

    if not per_tensor:
        raise RuntimeError("no language_model weight_scale tensors found")
    n_tensors = len(per_tensor)
    cat = {k: (np.concatenate(v) if v else np.zeros(0, np.float16)) for k, v in scope_scales.items()}
    print(f"[dq] {n_tensors} scale tensors loaded in {time.time()-t0:.1f}s | "
          f"core7={cat['core7'].size:,} scales ({cat['core7'].size*2/1e6:.2f} MB FP16) | "
          f"all={cat['all'].size:,} scales ({cat['all'].size*2/1e6:.2f} MB FP16)", flush=True)

    # ---- PRIMARY: core7 verify-GEMM body, asym INT8, primary block ----------
    def scan_scope(scales: np.ndarray, block: int) -> dict:
        rec = roundtrip_int8_asym(scales, block)
        be = _u16(scales) == _u16(rec)
        n_be = int(be.sum()); n = int(scales.size); n_exc = n - n_be
        n_blocks = math.ceil(n / block)
        by = hybrid_bytes(n, n_exc, n_blocks)
        out = {"block": block, "n": n, "n_bitexact": n_be,
               "bitexact_frac": n_be / n if n else 0.0,
               "exception_frac": n_exc / n if n else 0.0,
               "n_blocks": n_blocks}
        out.update(by)
        out.update(block_relspread_stats(scales, block))
        return out

    primary = scan_scope(cat["core7"], primary_block)
    primary_all = scan_scope(cat["all"], primary_block)

    print(f"\n[dq] === PRIMARY (core7 verify-GEMM body, asym INT8, block={primary_block}) ===", flush=True)
    print(f"[dq]   bitexact_frac = {primary['bitexact_frac']:.4f}  "
          f"(exception_frac = {primary['exception_frac']:.4f})", flush=True)
    print(f"[dq]   median within-block relspread = {primary['median_relspread']:.3f}; "
          f"frac blocks < 0.249 = {primary['frac_blocks_below_0p249']:.4f}", flush=True)
    print(f"[dq]   best hybrid = {primary['best_bytes']/1e6:.2f} MB vs FP16 "
          f"{primary['orig_bytes']/1e6:.2f} MB -> saved {primary['saved_bytes_best']/1e6:+.2f} MB "
          f"({100*primary['saved_frac_of_scale_bytes_best']:+.2f}% of scale bytes)", flush=True)

    # ---- block-size sweep (best-chance frontier) ----------------------------
    sweep_rows = [scan_scope(cat["core7"], b) for b in sweep]
    print("\n[dq] === secondary-block sweep (core7) ===", flush=True)
    print("[dq]  block | bitexact | excpt | medspread | best_hybrid_MB | saved_MB | saved%scale", flush=True)
    for r in sweep_rows:
        print(f"[dq]  {r['block']:5d} | {r['bitexact_frac']:.4f} | {r['exception_frac']:.3f} | "
              f"{(r['median_relspread'] or 0):8.3f} | {r['best_bytes']/1e6:9.2f} | "
              f"{r['saved_bytes_best']/1e6:+8.3f} | {100*r['saved_frac_of_scale_bytes_best']:+7.2f}%", flush=True)

    # ---- per-role breakdown (primary block) ---------------------------------
    role_agg = defaultdict(lambda: {"n": 0, "n_be": 0})
    for rec in per_tensor:
        s = rec["scales"]
        rc = roundtrip_int8_asym(s, primary_block)
        nbe = int((_u16(s) == _u16(rc)).sum())
        role_agg[rec["role"]]["n"] += rec["n"]
        role_agg[rec["role"]]["n_be"] += nbe
    role_rows = []
    for role, d in sorted(role_agg.items(), key=lambda kv: -kv[1]["n"]):
        role_rows.append({"role": role, "n": d["n"],
                          "bitexact_frac": d["n_be"] / d["n"] if d["n"] else 0.0})
    print("\n[dq] === per-role bitexact_frac (block=%d) ===" % primary_block, flush=True)
    for r in role_rows:
        print(f"[dq]   {r['role']:>10s} | n={r['n']:>9,} | bitexact={r['bitexact_frac']:.4f}", flush=True)

    # ---- BF16-storage variant (coarser target grid) ------------------------
    # Caution: storing scales as BF16 is itself LOSSY vs the deployed FP16 scales
    # (3 fewer mantissa bits) -> only greedy-safe where FP16==BF16(FP16). We report
    # (a) what fraction of FP16 scales are exactly BF16-representable, and (b) the
    # INT8 double-quant bit-exact frac measured on the BF16 grid.
    import torch
    s32 = torch.from_numpy(cat["core7"].astype(np.float32))
    s_bf16 = s32.to(torch.bfloat16)
    bf16_lossless_frac = float((s_bf16.to(torch.float32) == s32).float().mean())
    # INT8 asym roundtrip on the bf16 values, compared on bf16 bit patterns.
    bf16_np = s_bf16.to(torch.float32).numpy()
    rec_bf = roundtrip_int8_asym(bf16_np.astype(np.float16), primary_block)  # quant grid in fp32, compare via bf16
    rec_bf_bf16 = torch.from_numpy(rec_bf.astype(np.float32)).to(torch.bfloat16)
    bf16_dq_bitexact = float((rec_bf_bf16.view(torch.int16) == s_bf16.view(torch.int16)).float().mean())
    print(f"\n[dq] === BF16-storage variant ===", flush=True)
    print(f"[dq]   FP16 scales exactly BF16-representable: {bf16_lossless_frac:.4f} "
          f"(BF16 storage is lossless only on this fraction)", flush=True)
    print(f"[dq]   INT8 double-quant bit-exact on BF16 grid: {bf16_dq_bitexact:.4f}", flush=True)

    # ---- distinct-value / palette feasibility (follow-up lever) -------------
    uniq_core7 = int(np.unique(cat["core7"].view(np.uint16)).size)
    palette_bits = math.ceil(math.log2(max(uniq_core7, 2)))
    palette_saved_frac = 1.0 - palette_bits / 16.0
    # per-tensor distinct (median) -- local palettes could be cheaper
    pt_uniq = [int(np.unique(r["scales"].view(np.uint16)).size) for r in per_tensor if r["role"] in CORE7]
    print(f"\n[dq] === palette/LUT follow-up (NOT this PR's lever) ===", flush=True)
    print(f"[dq]   distinct FP16 scale values (core7 global): {uniq_core7:,} -> "
          f"{palette_bits}-bit index, lossless, saves {100*palette_saved_frac:.1f}% of scale bytes", flush=True)
    print(f"[dq]   per-tensor distinct values (median): {int(np.median(pt_uniq)) if pt_uniq else 0}", flush=True)

    # ---- total int4 body weight bytes (for %-of-weight-bandwidth framing) ---
    # weight_packed is int32 [out, in/8]; 0.5 byte/int4 weight. Read shapes lazily.
    int4_weight_bytes = 0
    files = [args.checkpoint] if os.path.isfile(args.checkpoint) else sorted(
        glob.glob(os.path.join(args.checkpoint, "*.safetensors")))
    for fp in files:
        with safe_open(fp, framework="numpy") as f:
            for k in f.keys():
                if k.endswith(".weight_packed") and ".language_model." in k:
                    sl = f.get_slice(k)
                    sh = sl.get_shape()
                    int4_weight_bytes += int(np.prod(sh)) * 4   # int32 storage bytes
    saved_pct_of_weight = (100.0 * primary["saved_bytes_best"] / int4_weight_bytes
                           if int4_weight_bytes else 0.0)

    # ---- verdict ------------------------------------------------------------
    green = (primary["bitexact_frac"] > args.green_threshold and primary["saved_bytes_best"] > 0)
    # achievable TPS lift: scale-bandwidth saving as a share of the verify-GEMM
    # weight+scale stream, scaled by the verify-GEMM share of the decode step.
    stream_bytes = int4_weight_bytes + primary["orig_bytes"]
    dq_bandwidth_saved_pct = (100.0 * primary["saved_bytes_best"] / stream_bytes) if stream_bytes else 0.0
    dq_tps_lift_est_pct = dq_bandwidth_saved_pct * VERIFY_GEMM_FRAC   # achievable (>=0; ~0 if KILL)

    verdict = {
        "green": bool(green),
        "primary_bitexact_frac": primary["bitexact_frac"],
        "green_threshold": args.green_threshold,
        "reason": ("bit-exact > threshold and net-byte-positive" if green else
                   f"bit-exact {primary['bitexact_frac']:.4f} <= {args.green_threshold} "
                   f"(exception set {primary['exception_frac']:.2%} -> hybrid saves only "
                   f"{primary['saved_bytes_best']/1e6:+.2f} MB of {primary['orig_bytes']/1e6:.1f} MB)"),
        "int4_weight_bytes": int4_weight_bytes,
        "saved_pct_of_int4_weight_bytes": saved_pct_of_weight,
        "dq_scale_bandwidth_saved_pct": dq_bandwidth_saved_pct,
        "dq_tps_lift_est_pct": dq_tps_lift_est_pct,
    }
    print(f"\n[dq] ===================== VERDICT =====================", flush=True)
    print(f"[dq]   {'GREEN' if green else 'KILL'}: {verdict['reason']}", flush=True)
    print(f"[dq]   achievable scale-bytes saved (best encoding) = "
          f"{primary['saved_bytes_best']/1e6:+.2f} MB = {saved_pct_of_weight:+.3f}% of int4 weight bytes", flush=True)
    print(f"[dq]   => est achievable TPS lift ~= {dq_tps_lift_est_pct:+.3f}% "
          f"(vs wall_tps={WALL_TPS})", flush=True)

    payload = {
        "config": {
            "checkpoint": args.checkpoint, "group_size": args.group_size,
            "secondary_block": primary_block, "block_sweep": sweep,
            "green_threshold": args.green_threshold,
            "scheme": "asymmetric INT8 (min offset + (max-min)/255 step), FP32 secondary",
            "n_scale_tensors": n_tensors,
            "core7_scales": int(cat["core7"].size), "all_scales": int(cat["all"].size),
            "core7_scale_MB_fp16": cat["core7"].size * 2 / 1e6,
            "all_scale_MB_fp16": cat["all"].size * 2 / 1e6,
            "int4_weight_MB": int4_weight_bytes / 1e6,
            "A10G_HBM_GBS": A10G_HBM_GBS, "verify_gemm_frac": VERIFY_GEMM_FRAC,
            "wall_tps": WALL_TPS,
            "note": "CPU-only scan of deployed FP16 weight_scale tensors; never reads "
                    "packed weights or the token stream. Lossless by construction.",
        },
        "primary_core7": primary,
        "primary_all": primary_all,
        "block_sweep": sweep_rows,
        "per_role": role_rows,
        "bf16_variant": {
            "bf16_storage_lossless_frac": bf16_lossless_frac,
            "int8_dq_bitexact_on_bf16_grid": bf16_dq_bitexact,
        },
        "palette_followup": {
            "distinct_fp16_values_core7": uniq_core7,
            "palette_index_bits": palette_bits,
            "palette_saved_frac_of_scale_bytes": palette_saved_frac,
            "per_tensor_distinct_median": int(np.median(pt_uniq)) if pt_uniq else 0,
        },
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[dq] wrote {args.output}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:   # noqa: BLE001
            print(f"[dq] W&B logging failed: {exc!r}", flush=True)


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    # block-sweep table + line series
    cols = ["block", "bitexact_frac", "exception_frac", "median_relspread",
            "frac_blocks_below_0p249", "best_bytes", "saved_bytes_best",
            "saved_frac_of_scale_bytes_best"]
    tbl = wandb.Table(columns=cols)
    for r in payload["block_sweep"]:
        tbl.add_data(*[r.get(c) for c in cols])
    run.log({"block_sweep_table": tbl})
    for r in sorted(payload["block_sweep"], key=lambda x: x["block"]):
        run.log({"secondary_block": r["block"], "sweep_bitexact_frac": r["bitexact_frac"],
                 "sweep_exception_frac": r["exception_frac"],
                 "sweep_saved_MB": r["saved_bytes_best"] / 1e6})
    rtbl = wandb.Table(columns=["role", "n", "bitexact_frac"])
    for r in payload["per_role"]:
        rtbl.add_data(r["role"], r["n"], r["bitexact_frac"])
    run.log({"per_role_table": rtbl})

    p = payload["primary_core7"]
    v = payload["verdict"]
    run.summary.update({
        "dq_scale_roundtrip_bitexact_frac": p["bitexact_frac"],
        "dq_exception_frac": p["exception_frac"],
        "dq_scale_bytes_saved_mb": p["saved_bytes_best"] / 1e6,
        "dq_scale_bandwidth_saved_pct": v["dq_scale_bandwidth_saved_pct"],
        "dq_tps_lift_est_pct": v["dq_tps_lift_est_pct"],
        "dq_median_within_block_relspread": p["median_relspread"],
        "dq_frac_blocks_below_0p249": p["frac_blocks_below_0p249"],
        "dq_core7_scale_MB_fp16": payload["config"]["core7_scale_MB_fp16"],
        "dq_all_scale_MB_fp16": payload["config"]["all_scale_MB_fp16"],
        "dq_bf16_storage_lossless_frac": payload["bf16_variant"]["bf16_storage_lossless_frac"],
        "dq_distinct_fp16_values_core7": payload["palette_followup"]["distinct_fp16_values_core7"],
        "dq_palette_saved_frac": payload["palette_followup"]["palette_saved_frac_of_scale_bytes"],
        "dq_verdict_green": int(v["green"]),
        "dq_saved_pct_of_int4_weight_bytes": v["saved_pct_of_int4_weight_bytes"],
    })
    run.finish()
    print(f"[dq] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
