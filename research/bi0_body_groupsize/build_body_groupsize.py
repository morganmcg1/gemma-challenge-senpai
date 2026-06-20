#!/usr/bin/env python
"""Build an int4head variant whose ONLY delta vs the shipped int4head fire
candidate is the BODY weight group_size (PR #814).

int4head = official ``gemma-4-E4B-it-qat-w4a16-ct`` body (int4 W4A16, group_size
32, 343 language-model Linear modules) + an untied int4 g32 ``lm_head`` + bf16
``embed_tokens`` + bf16 multimodal towers. This builder reproduces that exactly
EXCEPT it re-quantizes the 343 body modules at a chosen ``--body-group-size``
(32 / 64 / 128) from the QAT high-precision bf16 weights.

Clean-isolation contract (so the ONLY variable across {g32,g64,g128} arms, and
vs int4head, is the body group_size):

  * BODY (343 modules): re-quantized from
    ``gemma-4-E4B-it-qat-q4_0-unquantized`` (the bf16 QAT weights whose g32
    min-max quant IS the shipped w4a16-ct body -- verified: PTQ-g32 reproduces
    the shipped packed body to within 0.0086% of values / identical rel_err).
    Same min-max symmetric compressed-tensors recipe at every group size.
  * lm_head: int4 g32, quantized from the shipped w4a16-ct ``lm_head.weight``
    (byte-identical source + recipe to the int4head build) -- HELD FIXED.
  * Everything else (embed_tokens bf16, norms, layer_scalar, vision_tower,
    audio_tower, multimodal projectors, buffers): copied BYTE-FOR-BYTE from the
    shipped w4a16-ct checkpoint -- HELD FIXED, all modalities intact.

Quant math + packing use compressed-tensors' OWN primitives so the on-disk
``pack-quantized`` layout is exactly what vLLM 0.22.0 repacks to Marlin at load
(MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]; verified in the installed
.venv). LOCAL build only -- does NOT launch any HF Job and does NOT touch the
network (reads the official quant config + module set from the local cache).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
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


def hub_snapshot(model: str) -> str:
    hits = glob.glob(
        os.path.expanduser(
            f"~/.cache/huggingface/hub/models--google--{model}/snapshots/*/"
        )
    )
    if not hits:
        raise SystemExit(f"cached snapshot not found for google/{model}")
    return hits[0]


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
    """Return (weight_packed[int32], weight_scale[bf16], weight_shape[int64], rel_err).

    Identical math to submissions/int4_mtp_bi0_int4head/build_lmhead_quant.py.
    """
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
    q = quantize(w, scale, zp, qargs)
    q_int8 = q.to(torch.int8)
    packed = pack_to_int32(q_int8, num_bits, packed_dim=1)
    scale_bf16 = scale.to(torch.bfloat16)
    shape = torch.tensor([out_dim, in_dim], dtype=torch.int64)
    unpacked = unpack_from_int32(packed, num_bits, torch.Size([out_dim, in_dim]), packed_dim=1)
    assert torch.equal(unpacked, q_int8), "pack/unpack mismatch"
    deq = dequantize(q_int8, scale, zp, qargs)
    rel = (w - deq).norm() / w.norm().clamp_min(1e-9)
    return packed, scale_bf16, shape, float(rel)


def build_quant_config(src_qc: dict, body_group_size: int, head_group_size: int) -> dict:
    """w4a16-ct quant config, with body group_size bumped + an lm_head group."""
    qc = json.loads(json.dumps(src_qc))  # deep copy
    qc["config_groups"]["group_0"]["weights"]["group_size"] = body_group_size
    head_weights = dict(qc["config_groups"]["group_0"]["weights"])
    head_weights["group_size"] = head_group_size if head_group_size != -1 else None
    head_weights["strategy"] = "channel" if head_group_size == -1 else "group"
    qc["config_groups"]["group_1"] = {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": ["re:.*lm_head"],
        "weights": head_weights,
    }
    qc["ignore"] = [m for m in qc["ignore"] if m != "lm_head"]
    qc["version"] = compressed_tensors.__version__
    qc["quantization_status"] = "compressed"
    qc["format"] = "pack-quantized"
    return qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ct-src", default=None,
                    help="shipped w4a16-ct snapshot (config + non-body + lm_head source); "
                         "default: local hub cache")
    ap.add_argument("--uq-src", default=None,
                    help="qat-q4_0-unquantized snapshot (bf16 body source); "
                         "default: local hub cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--body-group-size", type=int, required=True, choices=[32, 64, 128])
    ap.add_argument("--head-group-size", type=int, default=32,
                    help="lm_head group size, HELD FIXED at int4head's 32")
    args = ap.parse_args()

    ct_dir = Path(args.ct_src or hub_snapshot("gemma-4-E4B-it-qat-w4a16-ct"))
    uq_dir = Path(args.uq_src or hub_snapshot("gemma-4-E4B-it-qat-q4_0-unquantized"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[src] ct (template + non-body + head) = {ct_dir}", flush=True)
    print(f"[src] uq (bf16 body)                  = {uq_dir}", flush=True)
    print(f"[cfg] body_group_size={args.body_group_size} head_group_size={args.head_group_size}", flush=True)

    ct_st = ct_dir / "model.safetensors"
    uq_st = uq_dir / "model.safetensors"

    fct = safe_open(str(ct_st), framework="pt", device="cpu")
    fuq = safe_open(str(uq_st), framework="pt", device="cpu")
    ct_keys = list(fct.keys())
    uq_keys = set(fuq.keys())

    body_bases = sorted(k[: -len(".weight_packed")] for k in ct_keys if k.endswith(".weight_packed"))
    assert len(body_bases) == 343, f"expected 343 body modules, got {len(body_bases)}"
    body_tensor_names = set()
    for b in body_bases:
        for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
            body_tensor_names.add(b + suf)

    tensors: dict[str, torch.Tensor] = {}
    rel_errs = []

    # 1) re-quantize the 343 body modules from the bf16 QAT weights
    for b in body_bases:
        wkey = b + ".weight"
        assert wkey in uq_keys, f"bf16 source missing for {b}"
        w = fuq.get_tensor(wkey)
        packed, scale, shape, rel = quantize_weight(w, 4, args.body_group_size)
        tensors[b + ".weight_packed"] = packed
        tensors[b + ".weight_scale"] = scale
        tensors[b + ".weight_shape"] = shape
        rel_errs.append(rel)
        if len(rel_errs) % 50 == 0:
            print(f"  body {len(rel_errs)}/343 (last rel_err={rel:.5f})", flush=True)

    # 2) lm_head: int4 g32 from the shipped w4a16-ct lm_head.weight (int4head parity)
    assert LM_HEAD_WEIGHT in set(ct_keys), "lm_head.weight not in w4a16-ct"
    lm_w = fct.get_tensor(LM_HEAD_WEIGHT)
    packed, scale, shape, rel = quantize_weight(lm_w, 4, args.head_group_size)
    tensors["lm_head.weight_packed"] = packed
    tensors["lm_head.weight_scale"] = scale
    tensors["lm_head.weight_shape"] = shape
    print(f"[lm_head] int4 g{args.head_group_size} rel_err={rel:.5f} packed={tuple(packed.shape)}", flush=True)

    # 3) everything else: copy byte-for-byte from the shipped w4a16-ct (held fixed)
    n_copy = 0
    for name in ct_keys:
        if name in body_tensor_names or name == LM_HEAD_WEIGHT:
            continue
        tensors[name] = fct.get_tensor(name)
        n_copy += 1

    print(f"[quant] body={len(body_bases)} (rel_err min={min(rel_errs):.5f} "
          f"max={max(rel_errs):.5f} mean={sum(rel_errs)/len(rel_errs):.5f}) "
          f"copied_verbatim={n_copy}", flush=True)

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    cfg = json.load(open(ct_dir / "config.json"))
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    cfg["quantization_config"] = build_quant_config(
        cfg["quantization_config"], args.body_group_size, args.head_group_size
    )
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print("[write] wrote config.json (tie=false, body g%d, lm_head g%d)"
          % (args.body_group_size, args.head_group_size), flush=True)

    for fn in ASSET_FILES:
        s = ct_dir / fn
        if s.exists():
            shutil.copy2(s, out / fn)
    copied = [f for f in ASSET_FILES if (ct_dir / f).exists()]
    print(f"[write] copied assets: {copied}", flush=True)
    total_gb = sum(p.stat().st_size for p in out.glob("*")) / 1e9
    print(f"[done] checkpoint at {out} ({total_gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
