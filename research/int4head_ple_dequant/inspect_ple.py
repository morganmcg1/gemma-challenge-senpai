#!/usr/bin/env python
"""Inspect per_layer_input_gate tensors across checkpoints to decide the bf16 source.

Answers the QAT-divergence question for PR #805: when de-quantizing
per_layer_input_gate back to bf16, do we source from google/gemma-4-E4B-it
(bf16 base) or dequantize the int4 weights already in the w4a16-ct checkpoint?

If QAT barely moved the gate, the google bf16 weights ARE the higher-precision
superset (advisor primary). If QAT diverged the gate substantially, the bf16
google weights are a DIFFERENT set of weights, and dequantizing the int4 keeps
the kernel-isolation clean (advisor fallback).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors import safe_open

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
W4A16 = HUB / "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
BF16 = HUB / "models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187"
QATUNQ = HUB / "models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/dfc5b925ddb1d41aaf1fe9679abdcfb0805e1aa6"

TARGET = "per_layer_input_gate"
SIBLING = "per_layer_projection"


def keys_for(st_path: Path, needle: str):
    out = {}
    with safe_open(str(st_path / "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            if needle in k:
                t = f.get_slice(k)
                out[k] = (tuple(t.get_shape()), t.get_dtype())
    return out


def dequant_int4_gate(st_path: Path, layer_prefix: str):
    """Dequantize one int4-packed per_layer_input_gate weight -> bf16-equivalent fp32."""
    with safe_open(str(st_path / "model.safetensors"), framework="pt", device="cpu") as f:
        packed = f.get_tensor(layer_prefix + ".weight_packed")
        scale = f.get_tensor(layer_prefix + ".weight_scale")
        shape = f.get_tensor(layer_prefix + ".weight_shape")
    out_dim, in_dim = int(shape[0]), int(shape[1])
    num_bits = 4
    group_size = in_dim // scale.shape[1] if scale.dim() == 2 and scale.shape[1] > 1 else -1
    q = unpack_from_int32(packed, num_bits, torch.Size([out_dim, in_dim]), packed_dim=1)
    if group_size != -1:
        qargs = QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                                 group_size=group_size, symmetric=True, observer="minmax")
    else:
        qargs = QuantizationArgs(num_bits=num_bits, type="int", strategy="channel",
                                 symmetric=True, observer="minmax")
    deq = dequantize(q.to(torch.int8), scale.to(torch.float32), None, qargs)
    return deq.to(torch.float32), (out_dim, in_dim), group_size


def get_bf16_weight(st_path: Path, key: str):
    with safe_open(str(st_path / "model.safetensors"), framework="pt", device="cpu") as f:
        if key in f.keys():
            return f.get_tensor(key).to(torch.float32)
    return None


def main():
    print("=" * 70)
    print("per_layer_input_gate keys in w4a16-ct (int4 source):")
    g = keys_for(W4A16, TARGET)
    for k in sorted(g)[:6]:
        print(f"  {k}: {g[k]}")
    print(f"  ... total {len(g)} keys ({len(g)//3 if g else 0} layers x 3 tensors)")

    print("\nper_layer_projection keys in w4a16-ct (SIBLING, stays int4):")
    s = keys_for(W4A16, SIBLING)
    for k in sorted(s)[:3]:
        print(f"  {k}: {s[k]}")
    print(f"  ... total {len(s)} keys")

    print("\nper_layer_input_gate keys in google bf16 base:")
    gb = keys_for(BF16, TARGET)
    for k in sorted(gb)[:3]:
        print(f"  {k}: {gb[k]}")
    print(f"  ... total {len(gb)} keys")

    # Pick a representative packed weight key
    packed_keys = sorted([k for k in g if k.endswith(".weight_packed")])
    print(f"\nFound {len(packed_keys)} per_layer_input_gate.weight_packed tensors")

    # Divergence on a sample of layers
    print("\n" + "=" * 70)
    print("QAT-DIVERGENCE: dequant(int4 w4a16-ct gate) vs bf16-google vs qat-unq")
    print("=" * 70)
    sample = packed_keys[:: max(1, len(packed_keys) // 6)][:6]
    for pk in sample:
        prefix = pk[: -len(".weight_packed")]
        deq, (od, idim), gs = dequant_int4_gate(W4A16, prefix)
        bf16_key = prefix + ".weight"
        wbf = get_bf16_weight(BF16, bf16_key)
        wqu = get_bf16_weight(QATUNQ, bf16_key)
        line = f"{prefix}  shape=({od},{idim}) gs={gs}"
        if wbf is not None and wbf.shape == deq.shape:
            rel_bf = ((deq - wbf).norm() / wbf.norm().clamp_min(1e-9)).item()
            cos_bf = torch.nn.functional.cosine_similarity(deq.flatten(), wbf.flatten(), dim=0).item()
            line += f" | bf16-google rel={rel_bf:.4f} cos={cos_bf:.5f}"
        else:
            line += f" | bf16-google MISSING/shape-mismatch (got {None if wbf is None else tuple(wbf.shape)})"
        if wqu is not None and wqu.shape == deq.shape:
            rel_qu = ((deq - wqu).norm() / wqu.norm().clamp_min(1e-9)).item()
            line += f" | qat-unq rel={rel_qu:.4f}"
        print(line)

    # also compare the two bf16 sources to each other on one layer
    print("\nbf16-google vs qat-unquantized (same gate, one layer):")
    pk = packed_keys[len(packed_keys) // 2]
    prefix = pk[: -len(".weight_packed")]
    wbf = get_bf16_weight(BF16, prefix + ".weight")
    wqu = get_bf16_weight(QATUNQ, prefix + ".weight")
    if wbf is not None and wqu is not None:
        rel = ((wbf - wqu).norm() / wqu.norm().clamp_min(1e-9)).item()
        print(f"  {prefix}: rel(google,qat-unq)={rel:.4f}")


if __name__ == "__main__":
    main()
