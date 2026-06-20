#!/usr/bin/env python
"""Quantize ONLY the bf16 lm_head of the official int4 W4A16 Gemma-4-E4B checkpoint.

Base = `google/gemma-4-E4B-it-qat-w4a16-ct` (compressed-tensors, int4 body
group_size=32, lm_head served bf16 because it is in `ignore` + tied to
embed_tokens). This builder produces a variant whose ONLY delta is the lm_head:

  * the int4 body (343 `*.weight_packed/scale/shape` tensors) is copied
    BYTE-FOR-BYTE from the source -- the body is not re-quantized, so the
    experiment isolates a single variable (lm_head bytes),
  * `embed_tokens` (input embedding, a cheap gather) stays bf16,
  * `lm_head.weight` (262144 x 2560 bf16, 1.342 GB/token at M=1, run once per
    accepted token and NOT amortized by speculation) is untied and quantized to
    int4 or int8 W*A16 -> `lm_head.weight_packed/scale/shape`,
  * config.json: `tie_word_embeddings=false` (both levels), `lm_head` dropped
    from `ignore`, and a new `group_1` carrying the head's quant scheme.

Quant math + packing use compressed-tensors' OWN primitives so the on-disk
`pack-quantized` layout is exactly what vLLM 0.22.0 repacks (Marlin for W4 /
W8-group; AllSpark for W8-channelwise on Ampere) at load. LOCAL build only --
does NOT launch any HF Job.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import compressed_tensors
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)

LM_HEAD_WEIGHT = "lm_head.weight"
ASSET_FILES = [
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "preprocessor_config.json",
]


def make_qargs(num_bits: int, group_size: int) -> QuantizationArgs:
    if group_size == -1:
        return QuantizationArgs(
            num_bits=num_bits, type="int", strategy="channel", symmetric=True,
            observer="minmax",
        )
    return QuantizationArgs(
        num_bits=num_bits, type="int", strategy="group", group_size=group_size,
        symmetric=True, observer="minmax",
    )


def quantize_weight(w: torch.Tensor, num_bits: int, group_size: int):
    """Return (weight_packed[int32], weight_scale[bf16], weight_shape[int64], rel_err)."""
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    qargs = make_qargs(num_bits, group_size)
    if group_size != -1:
        assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by {group_size}"
        ng = in_dim // group_size
        wg = w.reshape(out_dim, ng, group_size)
        min_vals = wg.amin(dim=-1)
        max_vals = wg.amax(dim=-1)
    else:
        min_vals = w.amin(dim=-1, keepdim=True)
        max_vals = w.amax(dim=-1, keepdim=True)
    scale, zp = calculate_qparams(min_vals, max_vals, qargs)
    q = quantize(w, scale, zp, qargs)            # integer-valued, clamped to the int range
    q_int8 = q.to(torch.int8)
    packed = pack_to_int32(q_int8, num_bits, packed_dim=1)
    scale_bf16 = scale.to(torch.bfloat16)
    shape = torch.tensor([out_dim, in_dim], dtype=torch.int64)

    # --- self-check: pack round-trips exactly, reconstruction error sane ---
    unpacked = unpack_from_int32(packed, num_bits, torch.Size([out_dim, in_dim]), packed_dim=1)
    assert torch.equal(unpacked, q_int8), "pack/unpack mismatch"
    deq = dequantize(q_int8, scale, zp, qargs)
    rel = (w - deq).norm() / w.norm().clamp_min(1e-9)
    return packed, scale_bf16, shape, float(rel)


def build_quant_config(src_qc: dict, num_bits: int, group_size: int) -> dict:
    """Copy the official quant config verbatim, add a group for the lm_head."""
    qc = json.loads(json.dumps(src_qc))  # deep copy
    head_weights = {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": group_size if group_size != -1 else None,
        "num_bits": num_bits,
        "observer": "minmax",
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "channel" if group_size == -1 else "group",
        "symmetric": True,
        "type": "int",
        "zp_dtype": None,
    }
    qc["config_groups"]["group_1"] = {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": ["re:.*lm_head"],
        "weights": head_weights,
    }
    qc["ignore"] = [m for m in qc["ignore"] if m != "lm_head"]
    qc["quantization_status"] = "compressed"
    qc["format"] = "pack-quantized"
    return qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True,
                    help="source checkpoint dir (official w4a16-ct snapshot)")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--num-bits", type=int, required=True, choices=[4, 8])
    ap.add_argument("--head-group-size", type=int, required=True,
                    help="-1 channelwise (W8->AllSpark), or 32/64/128 group (->Marlin)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    st_path = src / "model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    n_copy = 0
    head_done = False

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name == LM_HEAD_WEIGHT:
                packed, scale, shape, rel = quantize_weight(
                    t, args.num_bits, args.head_group_size
                )
                tensors["lm_head.weight_packed"] = packed
                tensors["lm_head.weight_scale"] = scale
                tensors["lm_head.weight_shape"] = shape
                head_done = True
                bf16_bytes = t.numel() * 2
                packed_bytes = packed.numel() * 4 + scale.numel() * 2
                print(
                    f"[lm_head] num_bits={args.num_bits} group_size={args.head_group_size} "
                    f"rel_err={rel:.5f} packed={tuple(packed.shape)} scale={tuple(scale.shape)}",
                    flush=True,
                )
                print(
                    f"[lm_head] bytes bf16={bf16_bytes/1e9:.4f}GB -> "
                    f"quant={packed_bytes/1e9:.4f}GB ({bf16_bytes/packed_bytes:.2f}x reduction)",
                    flush=True,
                )
            else:
                tensors[name] = t
                n_copy += 1

    assert head_done, "lm_head.weight not found in source checkpoint"
    print(f"[copy] copied {n_copy} tensors byte-identical (int4 body + embeddings + mm towers)",
          flush=True)

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    cfg = json.load(open(src / "config.json"))
    cfg["tie_word_embeddings"] = False
    cfg["text_config"]["tie_word_embeddings"] = False
    cfg["quantization_config"] = build_quant_config(
        cfg["quantization_config"], args.num_bits, args.head_group_size
    )
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print("[write] wrote config.json (tie=false, lm_head dropped from ignore, +group_1)")

    for fn in ASSET_FILES:
        s = src / fn
        if s.exists():
            shutil.copy2(s, out / fn)
    copied = [f for f in ASSET_FILES if (src / f).exists()]
    print(f"[write] copied assets: {copied}")
    total_gb = sum(p.stat().st_size for p in out.glob("*")) / 1e9
    print(f"[done] checkpoint at {out} ({total_gb:.2f} GB)")


if __name__ == "__main__":
    main()
