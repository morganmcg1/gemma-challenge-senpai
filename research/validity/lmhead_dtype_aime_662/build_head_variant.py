#!/usr/bin/env python
"""Surgical lm_head-dtype variants of the shipped int4_g128_lmhead checkpoint (PR #662).

Holds the int4-g128-minmax BODY *byte-identical* to the shipped submission and
changes ONLY the `lm_head` representation, so the AIME delta isolates the head
dtype with our calibration fixed (the clean apples-to-apples #653 set up but
couldn't run; official_g32 differs in BOTH head and Google's QAT calibration).

We do NOT re-quantize the 343 body modules. We copy every shipped tensor verbatim
and replace only the three `lm_head.weight_*` tensors (and the matching config
group). The shipped checkpoint keeps `model.language_model.embed_tokens.weight` in
bf16, so the un-quantized head value is recovered from there (the QAT model tied
embed==lm_head; the shipped build untied + int4'd it).

Variants:
  * bf16head  -- lm_head left bf16 (untied, separate tensor == embed_tokens).
                 `lm_head` added to `ignore`, group_1 removed. Recovery ceiling.
  * int8head  -- lm_head re-quantized W8A16 (int8, group_size 128, symmetric,
                 minmax) from embed_tokens. group_1 num_bits 4->8. Cheaper read.

Both keep the body group_0 (int4 g128) untouched. LOCAL build only; no HF Job.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
import compressed_tensors

EMBED_TOKENS = "model.language_model.embed_tokens.weight"
LM_HEAD_PACKED = "lm_head.weight_packed"
LM_HEAD_SCALE = "lm_head.weight_scale"
LM_HEAD_SHAPE = "lm_head.weight_shape"
ASSET_FILES = [
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "preprocessor_config.json",
]


def quantize_weight_nbits(w: torch.Tensor, num_bits: int, group_size: int):
    """Symmetric weight-only quant matching build_quant.py, parametrized on num_bits.

    Returns (weight_packed[int32], weight_scale[bf16], weight_shape[int64], rel_err).
    """
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by {group_size}"
    qargs = QuantizationArgs(
        num_bits=num_bits, type="int", strategy="group", group_size=group_size,
        symmetric=True, observer="minmax",
    )
    ng = in_dim // group_size
    wg = w.reshape(out_dim, ng, group_size)
    min_vals = wg.amin(dim=-1)
    max_vals = wg.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, qargs)
    q = quantize(w, scale, zp, qargs)            # integer-valued, clamped to range
    q_int8 = q.to(torch.int8)
    packed = pack_to_int32(q_int8, num_bits, packed_dim=1)
    scale_bf16 = scale.to(torch.bfloat16)
    shape = torch.tensor([out_dim, in_dim], dtype=torch.int64)

    # self-check: pack round-trips exactly, reconstruction error sane
    unpacked = unpack_from_int32(packed, num_bits, torch.Size([out_dim, in_dim]), packed_dim=1)
    assert torch.equal(unpacked, q_int8), "pack/unpack mismatch"
    deq = dequantize(q_int8, scale, zp, qargs)
    rel = (w - deq).norm() / w.norm().clamp_min(1e-9)
    return packed, scale_bf16, shape, float(rel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shipped", default="/workspace/gemma_build/int4_g128_lmhead",
                    help="shipped int4_g128_lmhead checkpoint dir (body source, byte-identical)")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--head", choices=["bf16", "int8"], required=True)
    ap.add_argument("--head-group-size", type=int, default=128,
                    help="group size for the int8 head (ignored for bf16)")
    args = ap.parse_args()

    shipped = Path(args.shipped)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st_path = shipped / "model.safetensors"

    tensors: dict[str, torch.Tensor] = {}
    embed_weight = None
    n_body = 0
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            if name in (LM_HEAD_PACKED, LM_HEAD_SCALE, LM_HEAD_SHAPE):
                continue  # drop the shipped int4 head; we rewrite it below
            t = f.get_tensor(name)
            if name == EMBED_TOKENS:
                embed_weight = t
            tensors[name] = t
            n_body += 1
    assert embed_weight is not None, "embed_tokens.weight not found in shipped checkpoint"
    print(f"[build] copied {n_body} body tensors verbatim (byte-identical body)", flush=True)
    print(f"[build] embed_tokens dtype={embed_weight.dtype} shape={tuple(embed_weight.shape)}", flush=True)

    cfg = json.load(open(shipped / "config.json"))
    qc = cfg["quantization_config"]

    if args.head == "bf16":
        # untied bf16 head == embed_tokens; lm_head -> ignore, group_1 removed
        tensors["lm_head.weight"] = embed_weight.clone()
        qc["config_groups"].pop("group_1", None)
        ig = list(qc.get("ignore", []))
        if "lm_head" not in ig:
            ig.append("lm_head")
        qc["ignore"] = ig
        print("[build] head=bf16: added lm_head.weight (bf16), lm_head -> ignore, removed group_1", flush=True)
    else:  # int8
        packed, scale, shape, rel = quantize_weight_nbits(embed_weight, 8, args.head_group_size)
        tensors[LM_HEAD_PACKED] = packed
        tensors[LM_HEAD_SCALE] = scale
        tensors[LM_HEAD_SHAPE] = shape
        # flip ONLY num_bits on the existing head group (group_size/strategy/etc unchanged)
        hw = qc["config_groups"]["group_1"]["weights"]
        hw["num_bits"] = 8
        hw["group_size"] = args.head_group_size if args.head_group_size != -1 else None
        hw["strategy"] = "channel" if args.head_group_size == -1 else "group"
        print(f"[build] head=int8 g{args.head_group_size}: rel_err={rel:.4f} packed={tuple(packed.shape)} "
              f"scale={tuple(scale.shape)}", flush=True)

    qc["version"] = compressed_tensors.__version__
    cfg["quantization_config"] = qc

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    for fn in ASSET_FILES:
        s = shipped / fn
        if s.exists():
            shutil.copy2(s, out / fn)
    total = sum(p.stat().st_size for p in out.glob("*")) / 1e9
    print(f"[done] {args.head} head checkpoint at {out} ({total:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
