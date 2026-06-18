#!/usr/bin/env python
"""Re-quantize the Gemma-4-E4B QAT-unquantized checkpoint to int4 W4A16.

Two deltas vs the official `...-qat-w4a16-ct` checkpoint:
  1. group_size 32 -> 128 on the *same* 343 language-model Linear modules the
     official checkpoint quantizes (derived from its safetensors header, incl. the
     MatFormer per_layer_input_gate/per_layer_projection/per_layer_model_projection).
  2. Untie `lm_head` (== embed_tokens, 262144x2560) and quantize it to int4 too.
     `embed_tokens` stays bf16.

Everything else (vision_tower, audio_tower, multimodal projectors, norms, embeddings,
buffers) is copied byte-for-byte from the source so all modalities stay intact.

Quantization math + packing use compressed-tensors' OWN primitives so the on-disk
`pack-quantized` layout is exactly what vLLM 0.22.0 repacks to Marlin at load.
LOCAL build only -- does not launch any HF Job.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import urllib.request
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

OFFICIAL_W4A16 = "google/gemma-4-E4B-it-qat-w4a16-ct"
EMBED_TOKENS = "model.language_model.embed_tokens.weight"
# Copy these non-tensor config/asset files verbatim from the source checkpoint.
ASSET_FILES = [
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "preprocessor_config.json",
]


def make_qargs(group_size: int, observer: str = "minmax") -> QuantizationArgs:
    if group_size == -1:
        return QuantizationArgs(
            num_bits=4, type="int", strategy="channel", symmetric=True,
            observer=observer,
        )
    return QuantizationArgs(
        num_bits=4, type="int", strategy="group", group_size=group_size,
        symmetric=True, observer=observer,
    )


def _mse_clipped_symmetric(w: torch.Tensor, group_size: int,
                           n_grid: int = 40, max_shrink: float = 0.45):
    """MSE *observer*: pick the symmetric clip per (channel/group) that minimizes
    int4 round-trip MSE, instead of the raw amin/amax (minmax observer).

    compressed_tensors 0.15.0.1 ships NO observer implementation -- the `observer`
    field is metadata only, and calculate_qparams derives scale = max_abs/7.5 from
    whatever min/max it is handed. So a genuine MSE observer must compute the clipped
    max here. Grid includes ratio=1.0 (== minmax), so MSE error <= minmax error always.
    Returns (min_vals, max_vals) shaped exactly like the minmax amin/amax branch.
    """
    out_dim, in_dim = w.shape
    if group_size == -1:
        wg = w.unsqueeze(1)                       # (out_dim, 1, in_dim)
    else:
        ng = in_dim // group_size
        wg = w.reshape(out_dim, ng, group_size)   # (out_dim, ng, gs)
    amax = wg.abs().amax(dim=-1)                   # (out_dim, ng) symmetric magnitude
    best_max = amax.clone()
    best_err = torch.full_like(amax, float("inf"))
    for i in range(n_grid):
        ratio = 1.0 - max_shrink * i / (n_grid - 1)   # 1.0 .. (1-max_shrink)
        cand = amax * ratio
        scale = (cand / 7.5).clamp_min(1e-12).unsqueeze(-1)
        q = torch.clamp(torch.round(wg / scale), -8, 7)
        err = ((wg - q * scale) ** 2).mean(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_max = torch.where(better, cand, best_max)
    return -best_max, best_max                    # symmetric: max(|min|,|max|) = best_max


def quantize_weight(w: torch.Tensor, group_size: int, observer: str = "minmax"):
    """Return (weight_packed[int32], weight_scale[bf16], weight_shape[int64])."""
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    qargs = make_qargs(group_size, observer)
    if group_size != -1:
        assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by {group_size}"
    if observer == "mse":
        min_vals, max_vals = _mse_clipped_symmetric(w, group_size)
    elif group_size == -1:
        min_vals = w.amin(dim=-1, keepdim=True)
        max_vals = w.amax(dim=-1, keepdim=True)
    else:
        ng = in_dim // group_size
        wg = w.reshape(out_dim, ng, group_size)
        min_vals = wg.amin(dim=-1)
        max_vals = wg.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, qargs)
    q = quantize(w, scale, zp, qargs)          # integer-valued, clamped [-8,7]
    q_int8 = q.to(torch.int8)
    packed = pack_to_int32(q_int8, 4, packed_dim=1)
    scale_bf16 = scale.to(torch.bfloat16)
    shape = torch.tensor([out_dim, in_dim], dtype=torch.int64)

    # --- self-check: pack round-trips exactly, reconstruction error sane ---
    unpacked = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1)
    assert torch.equal(unpacked, q_int8), "pack/unpack mismatch"
    deq = dequantize(q_int8, scale, zp, qargs)
    rel = (w - deq).norm() / w.norm().clamp_min(1e-9)
    return packed, scale_bf16, shape, float(rel)


def fetch_official_quant_config(token: str | None) -> dict:
    url = f"https://huggingface.co/{OFFICIAL_W4A16}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["quantization_config"]


def verify_official_module_set(token: str | None, expected: set[str]) -> None:
    url = f"https://huggingface.co/{OFFICIAL_W4A16}/resolve/main/model.safetensors"
    req = urllib.request.Request(url, headers={**({"Authorization": f"Bearer {token}"} if token else {}),
                                               "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=60) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={**({"Authorization": f"Bearer {token}"} if token else {}),
                                               "Range": f"bytes=8-{8 + n - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        hdr = json.loads(r.read().decode("utf-8"))
    live = {k[: -len(".weight_packed")] for k in hdr if k.endswith(".weight_packed")}
    if live != expected:
        raise SystemExit(
            f"Official module set drift: live={len(live)} expected={len(expected)} "
            f"only_live={sorted(live - expected)[:5]} only_exp={sorted(expected - live)[:5]}"
        )
    print(f"[verify] live official module set matches saved list ({len(live)} modules)")


def build_quant_config(group_size: int, head_group_size: int, token: str | None,
                       observer: str = "minmax") -> dict:
    qc = fetch_official_quant_config(token)
    # delta 1: bump body group_size
    qc["config_groups"]["group_0"]["weights"]["group_size"] = group_size
    qc["config_groups"]["group_0"]["weights"]["observer"] = observer
    # delta 2: untie + quantize lm_head -> its own group + drop from ignore
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
    ap.add_argument("--src", default="/workspace/gemma_build/qat_unq")
    ap.add_argument("--out", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--module-list",
                    default=str(Path(__file__).parent / "official_quantized_modules.json"))
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--head-group-size", type=int, default=128,
                    help="128 for g128 head, -1 for channel-wise head")
    ap.add_argument("--observer", choices=["minmax", "mse"], default="minmax",
                    help="scale-selection observer: minmax (raw amin/amax, the live "
                         "recipe) or mse (per-group clip-search minimizing int4 "
                         "round-trip MSE; zero on-disk-format/speed cost)")
    ap.add_argument("--no-verify-official", action="store_true",
                    help="skip live re-derivation of the official module set")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_modules = set(json.load(open(args.module_list)))
    assert len(quant_modules) == 343, f"expected 343 modules, got {len(quant_modules)}"
    if not args.no_verify_official:
        verify_official_module_set(token, quant_modules)
    quant_weight_names = {m + ".weight" for m in quant_modules}

    st_path = src / "model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    n_quant = 0
    n_copy = 0
    rel_errs = []
    embed_weight = None

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for name in keys:
            t = f.get_tensor(name)
            if name == EMBED_TOKENS:
                embed_weight = t
            if name in quant_weight_names:
                base = name[: -len(".weight")]
                packed, scale, shape, rel = quantize_weight(t, args.group_size, args.observer)
                tensors[base + ".weight_packed"] = packed
                tensors[base + ".weight_scale"] = scale
                tensors[base + ".weight_shape"] = shape
                rel_errs.append(rel)
                n_quant += 1
                if n_quant % 50 == 0:
                    print(f"  quantized {n_quant}/343 (last rel_err={rel:.4f})", flush=True)
            else:
                tensors[name] = t
                n_copy += 1

    assert n_quant == 343, f"quantized {n_quant} modules, expected 343"
    assert embed_weight is not None, "embed_tokens.weight not found"

    # untie + quantize lm_head from the (bf16) input embedding
    packed, scale, shape, rel = quantize_weight(embed_weight, args.head_group_size, args.observer)
    tensors["lm_head.weight_packed"] = packed
    tensors["lm_head.weight_scale"] = scale
    tensors["lm_head.weight_shape"] = shape
    print(f"[lm_head] quantized from embed_tokens, group_size={args.head_group_size}, "
          f"rel_err={rel:.4f}, packed={tuple(packed.shape)}")

    print(f"[quant] body modules quantized: {n_quant}, copied verbatim: {n_copy}")
    print(f"[quant] body rel_err: min={min(rel_errs):.4f} max={max(rel_errs):.4f} "
          f"mean={sum(rel_errs)/len(rel_errs):.4f}")

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    # config.json: tie_word_embeddings=false at both levels + new quant config
    cfg = json.load(open(src / "config.json"))
    cfg["tie_word_embeddings"] = False
    cfg["text_config"]["tie_word_embeddings"] = False
    cfg["quantization_config"] = build_quant_config(
        args.group_size, args.head_group_size, token, args.observer
    )
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print("[write] wrote config.json (tie_word_embeddings=false, g128 + lm_head group)")

    for fn in ASSET_FILES:
        s = src / fn
        if s.exists():
            shutil.copy2(s, out / fn)
    print(f"[write] copied assets: {[f for f in ASSET_FILES if (src/f).exists()]}")
    print(f"[done] checkpoint at {out} "
          f"({sum(p.stat().st_size for p in out.glob('*'))/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
