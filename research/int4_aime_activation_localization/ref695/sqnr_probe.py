#!/usr/bin/env python3
"""Per-layer g32->g128 SQNR localization probe (PR #695, instruction #1).

BASIS-INDEPENDENT, CPU-ONLY, ON-DISK. No HF Job, no GPU serve, no fetch.

Question (the card's hinge): is the int4 body's g32->g128 quantization-error
EXCESS concentrated in a small param-fraction of layers (TIGHT localization ->
a live SELECTIVE_GRID_CLEARS_SPEEDSAFE shot) or spread across most layers
(DIFFUSE -> SELECTIVE_GRID_NO_AIME_GAIN / TOO_COSTLY regardless of decode basis)?

Method.  Both on-disk int4 checkpoints quantize the SAME QAT-unquantized source:
  * official `gemma-4-E4B-it-qat-w4a16-ct`  -> group_size 32 (QAT-native grid)
  * int4_g128_lmhead anchor (PR #4)         -> group_size 128 (the byte-floor)
For each of the 343 body Linear modules we dequantize both and measure the
per-module relative L2 divergence

    rel_div_L = || dequant_g128_L - dequant_g32_L ||_F / || dequant_g32_L ||_F

= the EXACT excess quant error the coarser g128 grid adds over the QAT-native
g32 reference on module L -- i.e. exactly the error a selective-g32 recipe would
REMOVE if it put module L back on g32.  (dequant_g32 ~ source to rel 0.0666, so
rel_div ~ the g128 excess-over-source error; the small, ~uniform g32 residual
does not bias the per-module RANKING.)

Speed model (denken #676 byte-law, re-derived from ubel #679 endpoints):
    TPS(f) = 126.378 / (1 + 0.06005 * f),  f = body-param fraction put on g32.
f=0 -> 126.378 (anchor / byte-floor); f=1 -> 119.219 (#679 uniform g32).

Localization output: greedily add modules in descending excess-error-energy
PER PARAM (best AIME-error removed per speed-byte spent); plot cumulative
param-fraction f (= speed cost) vs cumulative excess-energy removed. f@{50,80,90}%
energy + projected TPS at those f are the decision scalars.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

G32_PATH = (
    "/senpai-run/home/student-ubel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors"
)
G128_PATH = "/workspace/gemma_build/int4_g128_lmhead/model.safetensors"

# Byte-law (advisor / denken #676), anchored on ubel #679's two measured endpoints.
TPS_FLOOR = 126.378
BYTE_K = 0.06005


def tps_at(f: float) -> float:
    return TPS_FLOOR / (1.0 + BYTE_K * f)


def read_header(path):
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr


def body_modules(h):
    mods = {k[: -len(".weight_packed")] for k in h if k.endswith(".weight_packed")}
    return sorted(m for m in mods if "lm_head" not in m)


def dequant(f, base):
    """Dequant a symmetric group-int4 module from an open safe_open handle."""
    packed = f.get_tensor(base + ".weight_packed")
    scale = f.get_tensor(base + ".weight_scale").float()
    shape = f.get_tensor(base + ".weight_shape").tolist()
    out_dim, in_dim = shape
    codes = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1).float()
    ng = scale.shape[1]
    gs = in_dim // ng
    w = (codes.reshape(out_dim, ng, gs) * scale[:, :, None]).reshape(out_dim, in_dim)
    return w, (out_dim, in_dim), gs, scale


def parse_name(base):
    """Return (layer_index:int|-1, proj:str) parsed from a module path."""
    m = re.search(r"\.layers\.(\d+)\.", base)
    layer = int(m.group(1)) if m else -1
    # projection / role token
    proj = base.split(".")[-1]
    return layer, proj


def intra_group_scale_cv(scale32, gs_ratio=4):
    """Mechanistic cause of g128 clip: per g128-group, how much do the 4 g32
    sub-block scales vary?  CV = std/mean across the gs_ratio sub-blocks, then
    take the per-module mean (weighted by group).  High CV -> g128 forces 1
    scale over disparate sub-blocks -> heavy clip on the near-tie weights."""
    out, ng = scale32.shape
    if ng % gs_ratio != 0:
        return float("nan")
    s = scale32.reshape(out, ng // gs_ratio, gs_ratio)  # (out, n128groups, 4)
    mean = s.mean(dim=-1)
    std = s.std(dim=-1)
    cv = (std / mean.clamp_min(1e-12))
    return float(cv.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "sqnr_probe.json"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N modules")
    args = ap.parse_args()

    t0 = time.time()
    h32 = read_header(G32_PATH)
    h128 = read_header(G128_PATH)
    mods = body_modules(h32)
    assert set(mods) == set(body_modules(h128)), "body module set mismatch"
    if args.limit:
        mods = mods[: args.limit]
    print(f"[probe] {len(mods)} body modules; g32 ref vs g128 anchor", flush=True)

    rows = []
    f32 = safe_open(G32_PATH, framework="pt", device="cpu")
    f128 = safe_open(G128_PATH, framework="pt", device="cpu")
    for i, base in enumerate(mods):
        w32, (out_dim, in_dim), gs32, scale32 = dequant(f32, base)
        w128, _, gs128, _ = dequant(f128, base)
        diff = w128 - w32
        n32 = w32.norm().item()
        ndiff = diff.norm().item()
        rel_div = ndiff / max(n32, 1e-9)
        energy = ndiff * ndiff  # ||diff||_F^2  absolute excess-error energy
        params = out_dim * in_dim
        layer, proj = parse_name(base)
        rows.append(
            {
                "module": base,
                "layer": layer,
                "proj": proj,
                "out_dim": out_dim,
                "in_dim": in_dim,
                "params": params,
                "gs32": gs32,
                "gs128": gs128,
                "w32_norm": n32,
                "diff_norm": ndiff,
                "rel_div": rel_div,
                "sqnr_db": (-20.0 * math.log10(rel_div)) if rel_div > 0 else float("inf"),
                "energy": energy,
                "energy_per_param": energy / params,
                "scale_cv_g128grp": intra_group_scale_cv(scale32),
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(mods)} last rel_div={rel_div:.4f} ({base})", flush=True)

    total_params = sum(r["params"] for r in rows)
    total_energy = sum(r["energy"] for r in rows)

    # ---- Localization curve: greedy by energy-per-param (best error removed per byte) ----
    order = sorted(rows, key=lambda r: r["energy_per_param"], reverse=True)
    cum_p = 0.0
    cum_e = 0.0
    curve = []
    f_at = {0.50: None, 0.80: None, 0.90: None, 0.95: None}
    for r in order:
        cum_p += r["params"]
        cum_e += r["energy"]
        fp = cum_p / total_params
        fe = cum_e / total_energy
        curve.append({"module": r["module"], "f_param": fp, "f_energy": fe,
                      "tps_proj": tps_at(fp)})
        for thr in f_at:
            if f_at[thr] is None and fe >= thr:
                f_at[thr] = fp

    # ---- Diffuse baseline: if every module had equal energy-per-param, the curve
    #      would be the identity f_energy == f_param.  Concentration = area between
    #      the greedy curve and that diagonal (Gini-like, 0=diffuse, ->1 tight). ----
    # trapezoidal area under f_energy vs f_param
    xs = [0.0] + [c["f_param"] for c in curve]
    ys = [0.0] + [c["f_energy"] for c in curve]
    area = 0.0
    for j in range(1, len(xs)):
        area += 0.5 * (ys[j] + ys[j - 1]) * (xs[j] - xs[j - 1])
    gini_like = 2.0 * (area - 0.5)  # area=0.5 -> 0 (diffuse); area->1 -> 1 (tight)

    # ---- per-param fragility ranking (scale-free) ----
    by_reldiv = sorted(rows, key=lambda r: r["rel_div"], reverse=True)

    # top-k energy share
    by_energy = sorted(rows, key=lambda r: r["energy"], reverse=True)
    def topk_share(k):
        return sum(r["energy"] for r in by_energy[:k]) / total_energy
    topk = {k: topk_share(k) for k in (1, 4, 8, 16, 32)}

    rel_vals = torch.tensor([r["rel_div"] for r in rows])
    summary = {
        "n_modules": len(rows),
        "total_params": total_params,
        "total_excess_energy": total_energy,
        "rel_div_mean": float(rel_vals.mean()),
        "rel_div_median": float(rel_vals.median()),
        "rel_div_min": float(rel_vals.min()),
        "rel_div_max": float(rel_vals.max()),
        "rel_div_std": float(rel_vals.std()),
        "rel_div_p90": float(rel_vals.quantile(0.90)),
        "rel_div_p99": float(rel_vals.quantile(0.99)),
        "f_param_at_50pct_energy": f_at[0.50],
        "f_param_at_80pct_energy": f_at[0.80],
        "f_param_at_90pct_energy": f_at[0.90],
        "f_param_at_95pct_energy": f_at[0.95],
        "tps_at_50pct_energy": tps_at(f_at[0.50]) if f_at[0.50] else None,
        "tps_at_80pct_energy": tps_at(f_at[0.80]) if f_at[0.80] else None,
        "tps_at_90pct_energy": tps_at(f_at[0.90]) if f_at[0.90] else None,
        "energy_gini_like": gini_like,
        "topk_energy_share": topk,
        "top8_modules_by_reldiv": [
            {"module": r["module"], "rel_div": r["rel_div"], "sqnr_db": r["sqnr_db"],
             "params": r["params"], "scale_cv": r["scale_cv_g128grp"]}
            for r in by_reldiv[:8]
        ],
        "top8_modules_by_energy": [
            {"module": r["module"], "energy": r["energy"], "rel_div": r["rel_div"],
             "params": r["params"]}
            for r in by_energy[:8]
        ],
        "elapsed_s": time.time() - t0,
    }

    out = {"summary": summary, "rows": rows, "loc_curve": curve}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[probe] wrote {args.out} in {summary['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
