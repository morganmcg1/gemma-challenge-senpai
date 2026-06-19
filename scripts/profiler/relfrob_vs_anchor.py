#!/usr/bin/env python
"""Offline weight-space divergence of a recovery config vs the int4-g128 anchor.

The #720 complement (advisor): "each config's divergence vs the int4-g128 anchor
(served + concrete rel-Frob), to sit beside the own-AR result." This is the
OFFLINE rel-Frob half -- dequantize each quantized module's weight for the config
and the anchor and report the relative Frobenius norm of the difference.

For g32-locus only the L14-27 modules differ (g32 vs g128 grouping of the same
dense weights); non-locus + lm_head are byte-identical (rel-Frob exactly 0). We
short-circuit byte-identical modules and dequantize only where they differ, so the
pass is cheap. Group size is derived from each tensor's scale shape, so this works
for g32-vs-g128 AND int8-vs-int4 (int8-locus) without recipe knowledge.

CPU only; does not touch the GPU (safe to run alongside a serve).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized import unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs, dequantize
from safetensors import safe_open

LOCUS_RE = re.compile(r"layers\.(1[4-9]|2[0-7])\.")


def _is_locus(name: str) -> bool:
    return bool(LOCUS_RE.search(name))


def dequant_module(f, mod: str) -> torch.Tensor:
    wp = f.get_tensor(mod + ".weight_packed")
    ws = f.get_tensor(mod + ".weight_scale")
    wsh = f.get_tensor(mod + ".weight_shape")
    out_f, in_f = int(wsh[0]), int(wsh[1])
    num_groups = ws.shape[1]
    gs = in_f // int(num_groups)
    num_bits = 4 if wp.dtype == torch.int32 and (in_f // wp.shape[1]) == 8 else (in_f // wp.shape[1] * 4 if False else 4)
    # robustly infer num_bits from packing density: int32 holds 32//num_bits values
    per_int32 = in_f // int(wp.shape[1])
    num_bits = 32 // per_int32
    unpacked = unpack_from_int32(wp, num_bits, torch.Size([out_f, in_f]), packed_dim=1)
    qa = QuantizationArgs(num_bits=num_bits, type="int", symmetric=True, strategy="group", group_size=gs)
    return dequantize(unpacked, ws, None, args=qa).float(), num_bits, gs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-dir", type=Path, required=True)
    ap.add_argument("--anchor-dir", type=Path, default=Path("/workspace/gemma_build/int4_g128_lmhead"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cst = args.config_dir / "model.safetensors"
    ast = args.anchor_dir / "model.safetensors"

    with safe_open(ast, framework="pt") as fa:
        a_keys = set(fa.keys())
    with safe_open(cst, framework="pt") as fc:
        c_keys = set(fc.keys())
    mods = sorted(k[: -len(".weight_packed")] for k in (a_keys & c_keys) if k.endswith(".weight_packed"))

    sum_dsq = 0.0       # Σ ||ΔW||²  (all modules)
    sum_wsq = 0.0       # Σ ||W_anchor||²
    sum_dsq_loc = 0.0   # locus only
    sum_wsq_loc = 0.0
    per_mod = []
    n_diff = 0
    fa = safe_open(ast, framework="pt")
    fc = safe_open(cst, framework="pt")
    try:
        for mod in mods:
            wp_a = fa.get_tensor(mod + ".weight_packed")
            wp_c = fc.get_tensor(mod + ".weight_packed")
            ws_a = fa.get_tensor(mod + ".weight_scale")
            ws_c = fc.get_tensor(mod + ".weight_scale")
            identical = (wp_a.shape == wp_c.shape and torch.equal(wp_a, wp_c)
                         and ws_a.shape == ws_c.shape and torch.equal(ws_a, ws_c))
            Wa, nb_a, gs_a = dequant_module(fa, mod)
            wa2 = float((Wa * Wa).sum())
            sum_wsq += wa2
            loc = _is_locus(mod)
            if loc:
                sum_wsq_loc += wa2
            if identical:
                continue
            n_diff += 1
            Wc, nb_c, gs_c = dequant_module(fc, mod)
            d2 = float(((Wc - Wa) ** 2).sum())
            sum_dsq += d2
            if loc:
                sum_dsq_loc += d2
            rel = (d2 ** 0.5) / (wa2 ** 0.5 + 1e-12)
            per_mod.append({"module": mod, "locus": loc, "rel_frob": rel,
                            "anchor_bits": nb_a, "anchor_gs": gs_a, "config_bits": nb_c, "config_gs": gs_c})
    finally:
        del fa, fc

    per_mod.sort(key=lambda r: r["rel_frob"], reverse=True)
    result = {
        "label": args.label,
        "config_dir": str(args.config_dir),
        "anchor_dir": str(args.anchor_dir),
        "n_modules_compared": len(mods),
        "n_modules_differ": n_diff,
        "global_rel_frob": (sum_dsq ** 0.5) / (sum_wsq ** 0.5 + 1e-12),
        "locus_rel_frob": (sum_dsq_loc ** 0.5) / (sum_wsq_loc ** 0.5 + 1e-12),
        "per_module_max_rel_frob": per_mod[0]["rel_frob"] if per_mod else 0.0,
        "per_module_mean_rel_frob": (sum(r["rel_frob"] for r in per_mod) / len(per_mod)) if per_mod else 0.0,
        "top_modules": per_mod[:8],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "label", "n_modules_compared", "n_modules_differ",
        "global_rel_frob", "locus_rel_frob", "per_module_max_rel_frob", "per_module_mean_rel_frob")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
