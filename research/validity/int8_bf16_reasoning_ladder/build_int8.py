#!/usr/bin/env python
"""PR #646 — quantize the PLAIN bf16 base google/gemma-4-E4B-it to int8 W8A16.

This builds the MIDDLE rung of the int4->int8->bf16 reasoning ladder. It is a
deliberate near-clone of ``submissions/int4_g128_lmhead/build_quant.py`` with
exactly TWO axes changed vs the live int4 body so the int4->int8 step isolates
*bit-width only*:

  * num_bits 4 -> 8 (W8A16). Everything else about the recipe is held fixed:
    same 343 language-model Linear modules, same group_size=128, same symmetric
    minmax, same compressed-tensors ``pack-quantized`` on-disk layout that vLLM
    0.22.0 repacks to Marlin at load, same untied int8 lm_head + bf16 embed_tokens.
  * source = the PLAIN bf16 base (``google/gemma-4-E4B-it`` snapshot fee6332c, the
    model ubel #628 served for the bf16 endpoints g3cig1xo / zoszxnb0) instead of
    the QAT-unquantized checkpoint. This is the PR #646 instruction: the int8->bf16
    rung must share the bf16 base so it is a clean 16->8 bit step on identical
    weights, and the verdict (int8 >= 90% of bf16) compares like-for-like.

Everything not in the 343-module quant set (vision_tower, audio_tower, multimodal
projectors, norms, embed_tokens, buffers) is copied byte-for-byte so all modalities
stay intact -- identical to the int4 build.

The quantization_config is constructed from the *live int4 body's* config (the
schema vLLM 0.22.0 is proven to load), flipping num_bits 4->8; no HF fetch needed.

ANALYSIS-ONLY local build. Does NOT launch any HF Job. analysis_only, official_tps=0.
"""
from __future__ import annotations

import argparse
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
NUM_BITS = 8
INT8_MIN, INT8_MAX = -128, 127


def make_qargs(group_size: int) -> QuantizationArgs:
    if group_size == -1:
        return QuantizationArgs(
            num_bits=NUM_BITS, type="int", strategy="channel", symmetric=True,
            observer="minmax",
        )
    return QuantizationArgs(
        num_bits=NUM_BITS, type="int", strategy="group", group_size=group_size,
        symmetric=True, observer="minmax",
    )


def quantize_weight(w: torch.Tensor, group_size: int):
    """Return (weight_packed[int32], weight_scale[bf16], weight_shape[int64], rel_err).

    int8 pack-quantized: pack_to_int32(..., 8) stores 4 int8 values per int32 word,
    the WNA16 layout vLLM's Marlin path reads for num_bits=8 weight-only."""
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    qargs = make_qargs(group_size)
    if group_size == -1:
        min_vals = w.amin(dim=-1, keepdim=True)
        max_vals = w.amax(dim=-1, keepdim=True)
    else:
        assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by {group_size}"
        ng = in_dim // group_size
        wg = w.reshape(out_dim, ng, group_size)
        min_vals = wg.amin(dim=-1)
        max_vals = wg.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, qargs)
    q = quantize(w, scale, zp, qargs)          # integer-valued, clamped [-128,127]
    q_int8 = q.to(torch.int8)
    assert int(q_int8.amin()) >= INT8_MIN and int(q_int8.amax()) <= INT8_MAX, "int8 range overflow"
    packed = pack_to_int32(q_int8, NUM_BITS, packed_dim=1)
    scale_bf16 = scale.to(torch.bfloat16)
    shape = torch.tensor([out_dim, in_dim], dtype=torch.int64)

    # --- self-check: pack round-trips exactly, reconstruction error sane ---
    unpacked = unpack_from_int32(packed, NUM_BITS, torch.Size([out_dim, in_dim]), packed_dim=1)
    assert torch.equal(unpacked, q_int8), "pack/unpack mismatch"
    deq = dequantize(q_int8, scale, zp, qargs)
    rel = (w - deq).norm() / w.norm().clamp_min(1e-9)
    return packed, scale_bf16, shape, float(rel)


def build_int8_quant_config(int4_cfg_path: Path, group_size: int, head_group_size: int) -> dict:
    """Flip the proven live-int4 quantization_config to int8 (only num_bits changes;
    group_size / strategy / pack-quantized format / ignore list all preserved so the
    int4->int8 step is bit-width-only and vLLM maps modules identically)."""
    qc = json.load(open(int4_cfg_path))["quantization_config"]
    g0 = qc["config_groups"]["group_0"]["weights"]
    g0["num_bits"] = NUM_BITS
    g0["group_size"] = group_size if group_size != -1 else None
    g0["strategy"] = "group" if group_size != -1 else "channel"
    g0["observer"] = "minmax"
    g1 = qc["config_groups"]["group_1"]["weights"]
    g1["num_bits"] = NUM_BITS
    g1["group_size"] = head_group_size if head_group_size != -1 else None
    g1["strategy"] = "group" if head_group_size != -1 else "channel"
    g1["observer"] = "minmax"
    qc["version"] = compressed_tensors.__version__
    qc["quantization_status"] = "compressed"
    qc["format"] = "pack-quantized"
    return qc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        default="/senpai-run/home/student-fern/.cache/huggingface/hub/"
        "models--google--gemma-4-E4B-it/snapshots/"
        "fee6332c1abaafb77f6f9624236c63aa2f1d0187",
        help="PLAIN bf16 base snapshot (single model.safetensors).",
    )
    ap.add_argument("--out", default="/workspace/gemma_build/int8_g128_lmhead")
    ap.add_argument(
        "--module-list",
        default="/workspace/senpai/target/submissions/int4_g128_lmhead/"
        "official_quantized_modules.json",
    )
    ap.add_argument(
        "--int4-config",
        default="/workspace/gemma_build/int4_g128_lmhead/config.json",
        help="live int4 body config.json — quantization_config schema template.",
    )
    ap.add_argument(
        "--int4-skeleton",
        default="/workspace/gemma_build/int4_g128_lmhead/model.safetensors",
        help="live int4 body safetensors — authoritative tensor-name SKELETON. "
        "gemma-4-E4B layers 24-41 are KV-sharing (they reuse an earlier layer's KV and "
        "carry NO own k_proj/v_proj/k_norm), so vLLM builds the module tree without those "
        "params. The PLAIN bf16 base still stores them, so a non-quantized bf16 tensor is "
        "copied only if its name appears in this proven-loadable layout; otherwise dropped.",
    )
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--head-group-size", type=int, default=128,
                    help="128 for g128 head (match int4 rung), -1 for channel-wise head")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_modules = set(json.load(open(args.module_list)))
    assert len(quant_modules) == 343, f"expected 343 modules, got {len(quant_modules)}"
    quant_weight_names = {m + ".weight" for m in quant_modules}

    # Authoritative skeleton = the proven-loadable live int4 body's tensor names. The
    # plain bf16 base carries k_proj/v_proj/k_norm for ALL 42 layers, but the 18
    # KV-sharing layers (24-41) have no such params in the vLLM module tree, so those
    # bf16 tensors must be DROPPED to match the int4 layout exactly (bit-width is the
    # only axis that may differ between the int4 and int8 rungs).
    with safe_open(str(args.int4_skeleton), framework="pt", device="cpu") as sk:
        skeleton = set(sk.keys())
    skeleton_plain = {
        k for k in skeleton
        if not k.endswith((".weight_packed", ".weight_scale", ".weight_shape"))
    }

    st_path = src / "model.safetensors"
    assert st_path.exists(), f"source safetensors missing: {st_path}"
    tensors: dict[str, torch.Tensor] = {}
    n_quant = 0
    n_copy = 0
    rel_errs = []
    dropped: list[str] = []
    embed_weight = None

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for name in keys:
            t = f.get_tensor(name)
            if name == EMBED_TOKENS:
                embed_weight = t
            if name in quant_weight_names:
                base = name[: -len(".weight")]
                packed, scale, shape, rel = quantize_weight(t, args.group_size)
                tensors[base + ".weight_packed"] = packed
                tensors[base + ".weight_scale"] = scale
                tensors[base + ".weight_shape"] = shape
                rel_errs.append(rel)
                n_quant += 1
                if n_quant % 50 == 0:
                    print(f"  quantized {n_quant}/343 (last rel_err={rel:.4f})", flush=True)
            elif name in skeleton_plain:
                tensors[name] = t
                n_copy += 1
            else:
                dropped.append(name)

    assert n_quant == 343, f"quantized {n_quant} modules, expected 343"
    assert embed_weight is not None, "embed_tokens.weight not found"

    # Defensive: only the KV-shared attn projections may be dropped — never silently
    # drop a load-bearing tensor (norm, embed, vision/audio tower, projector, ...).
    import re as _re
    bad_drop = [d for d in dropped
                if not _re.search(r"\.self_attn\.(k_proj|v_proj|k_norm)\.weight$", d)]
    assert not bad_drop, f"unexpected non-KV-shared tensors dropped: {bad_drop[:8]}"
    print(f"[skeleton] dropped {len(dropped)} KV-shared bf16 tensors "
          f"(k_proj/v_proj/k_norm on layers without own KV) to match int4 layout", flush=True)

    # untie + quantize lm_head from the (bf16) input embedding -> int8
    packed, scale, shape, rel = quantize_weight(embed_weight, args.head_group_size)
    tensors["lm_head.weight_packed"] = packed
    tensors["lm_head.weight_scale"] = scale
    tensors["lm_head.weight_shape"] = shape
    print(f"[lm_head] int8 quantized from embed_tokens, group_size={args.head_group_size}, "
          f"rel_err={rel:.4f}, packed={tuple(packed.shape)}")

    print(f"[quant] body modules quantized: {n_quant}, copied verbatim: {n_copy}")
    print(f"[quant] body rel_err: min={min(rel_errs):.4f} max={max(rel_errs):.4f} "
          f"mean={sum(rel_errs)/len(rel_errs):.4f}")

    # Strongest guarantee that the int4->int8 step is bit-width-only: the int8 tensor
    # NAME set must equal the proven-loadable int4 body's exactly (same modules present,
    # same quantized vs plain split, same KV-sharing layout). Values differ (8-bit pack
    # of the bf16 base); the skeleton does not.
    extra = sorted(set(tensors) - skeleton)
    missing = sorted(skeleton - set(tensors))
    assert not extra and not missing, (
        f"int8 tensor set != int4 skeleton: extra={extra[:5]} missing={missing[:5]}"
    )
    print(f"[skeleton] int8 tensor set == int4 skeleton ({len(skeleton)} tensors) OK",
          flush=True)

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    # config.json: untie embeddings + int8 quant config built off the live int4 schema
    cfg = json.load(open(src / "config.json"))
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    cfg["quantization_config"] = build_int8_quant_config(
        Path(args.int4_config), args.group_size, args.head_group_size
    )
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print("[write] wrote config.json (tie_word_embeddings=false, int8 g128 body + lm_head)")

    copied = []
    for fn in ASSET_FILES:
        s = src / fn
        if s.exists():
            shutil.copy2(s, out / fn)
            copied.append(fn)
    print(f"[write] copied assets: {copied}")
    print(f"[done] int8 checkpoint at {out} "
          f"({sum(p.stat().st_size for p in out.glob('*'))/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
