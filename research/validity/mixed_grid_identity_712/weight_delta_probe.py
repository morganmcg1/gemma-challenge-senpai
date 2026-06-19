#!/usr/bin/env python
"""PR #712 pre-flight: how big is the re-grid perturbation?

The faithful Route-B recovery weight is quant_g32(qat_unq); the anchor is
quant_g128(qat_unq). We do NOT have qat_unq on disk (only its g128 projection,
the locked int4_g128_lmhead). So the PR-literal in-memory Route-B is
requant_g32(dequant_g128) on the target modules. This probe MEASURES that
re-grid perturbation magnitude on real target modules, so the identity verdict
can be interpreted correctly (is the perturbation a faithful-magnitude proxy, or
near-lossless / understated?).

NO disk writes. Reads the on-disk g128 safetensors tensors only.
"""
from __future__ import annotations
import json, sys
import torch
from safetensors import safe_open
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

CKPT = "/workspace/gemma_build/int4_g128_lmhead/model.safetensors"


def qargs(gs):
    return QuantizationArgs(num_bits=4, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def dequant_ondisk(f, base):
    packed = f.get_tensor(base + ".weight_packed")
    scale = f.get_tensor(base + ".weight_scale")
    shape = f.get_tensor(base + ".weight_shape")
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1)
    w = dequantize(q, scale, None, qargs(128))
    return w.to(torch.float32), (out_dim, in_dim)


def requant_dequant(w, gs):
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    ng = in_dim // gs
    wg = w.reshape(out_dim, ng, gs)
    scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), qargs(gs))
    q = quantize(w, scale, zp, qargs(gs))
    deq = dequantize(q, scale, zp, qargs(gs))
    return deq.to(torch.float32)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    targets = [
        "model.language_model.layers.23.per_layer_input_gate",  # top1 PLIG
        "model.language_model.layers.0.per_layer_input_gate",   # PLIG
        "model.language_model.layers.0.self_attn.q_proj",       # attn q
        "model.language_model.layers.0.self_attn.k_proj",       # attn k
        "model.language_model.layers.0.self_attn.v_proj",       # attn v
        "model.language_model.layers.18.self_attn.k_proj",      # attn k (L18)
    ]
    out = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for base in targets:
            if base + ".weight_packed" not in keys:
                out[base] = {"error": "not_quantized_on_disk"}
                continue
            w128, shape = dequant_ondisk(f, base)
            w32 = requant_dequant(w128, 32)
            # element-wise relative change stats
            diff = (w32 - w128)
            denom = w128.abs().clamp_min(1e-9)
            elt_rel = (diff.abs() / denom)
            nz = w128.abs() > 0
            out[base] = {
                "shape": shape,
                "regrid_rel_frob": rel(w32, w128),            # ||w32-w128||/||w128||
                "regrid_max_abs": float(diff.abs().max()),
                "w128_rms": float(w128.pow(2).mean().sqrt()),
                "regrid_rms_abs": float(diff.pow(2).mean().sqrt()),
                "frac_elements_changed": float((diff != 0).float().mean()),
                "elt_rel_p50": float(elt_rel[nz].median()),
                "elt_rel_p99": float(elt_rel[nz].quantile(0.99)),
            }
            print(f"{base}\n   shape={shape} regrid_rel_frob={out[base]['regrid_rel_frob']:.5f} "
                  f"rms_abs={out[base]['regrid_rms_abs']:.3e} w_rms={out[base]['w128_rms']:.3e} "
                  f"frac_changed={out[base]['frac_elements_changed']:.4f}", flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else
                        "/workspace/senpai/target/research/validity/mixed_grid_identity_712/weight_delta.json", "w"),
              indent=2)
    # headline
    fr = [v["regrid_rel_frob"] for v in out.values() if "regrid_rel_frob" in v]
    print(f"\n=== re-grid rel_frob across targets: min={min(fr):.5f} max={max(fr):.5f} "
          f"mean={sum(fr)/len(fr):.5f} ===")
    print("Interpretation: int4 g128 quant error is typically ~3-5% rel_frob. "
          "If re-grid << that, requant_g32(dequant_g128) understates the true "
          "recovery delta (which is ~the g128 quant error).")


if __name__ == "__main__":
    main()
