#!/usr/bin/env python
"""Build the lmhead12k_int4head checkpoint: int4-quantize the pruned 12k lm_head.

Single-variable delta vs the merged PR #14 ``lmhead12k_empirical`` rung: the body
(int4 language-model Linears + bf16 multimodal towers + bf16 full ``embed_tokens``)
is copied **byte-for-byte**; the ONLY change is the output ``lm_head``:

    bf16  ``lm_head.weight``            [12288, 2560]  (62.9 MB)
      ->  int4  ``lm_head.weight_packed`` [12288, 320] int32
          +    ``lm_head.weight_scale``  [12288, 2560/g] bf16
          +    ``lm_head.weight_shape``  [2] int64       (~15.7 MB + scales)

The head is quantized with compressed-tensors' OWN primitives (the same library
+ ``pack-quantized`` layout the body uses), so vLLM 0.22.0 repacks it to the same
Marlin int4 kernel at load -- a real int4 GEMV, not a dequant-to-bf16 GEMM. The
served checkpoint's ``quantization_config`` gains a ``group_1`` head group
(``targets=["re:.*lm_head"]``) and drops ``lm_head`` from ``ignore`` so vLLM
materializes a quantized linear method for the (rebuilt, pruned) head.

``embed_tokens`` stays full bf16 (the model must still embed any input id); only
the OUTPUT projection shrinks + quantizes, so the per-token decode lm_head GEMV
reads ~4x fewer head bytes. ``config.vocab_size`` stays 262144 (only
``lm_head.out_features`` is 12288); the custom vLLM class scatters the kept-row
logits back to full vocab so the sampler / prompt_logprobs path is unchanged.

LOCAL build only -- does not launch any HF Job.
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

FULL_VOCAB = 262144
HIDDEN = 2560
HEAD_KEY_BF16 = "lm_head.weight"
DEFAULT_SRC = "/workspace/gemma_build/lmhead12k_empirical"
DEFAULT_OUT = "/workspace/gemma_build/lmhead12k_int4head"


def make_qargs(group_size: int) -> QuantizationArgs:
    """int4 symmetric group quant args -- mirrors the body's group_0 scheme."""
    return QuantizationArgs(
        num_bits=4, type="int", strategy="group", group_size=group_size,
        symmetric=True, observer="minmax",
    )


def quantize_head(w: torch.Tensor, group_size: int):
    """Quantize the bf16 lm_head to int4 pack-quantized.

    Returns (weight_packed[int32], weight_scale[bf16], weight_shape[int64], rel_err).
    Uses the EXACT compressed-tensors primitives the body was packed with so the
    on-disk pack-quantized layout is what vLLM 0.22.0 repacks to Marlin at load.
    """
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} not divisible by {group_size}"
    qargs = make_qargs(group_size)
    ng = in_dim // group_size
    wg = w.reshape(out_dim, ng, group_size)
    min_vals = wg.amin(dim=-1)
    max_vals = wg.amax(dim=-1)
    scale, zp = calculate_qparams(min_vals, max_vals, qargs)
    q = quantize(w, scale, zp, qargs)          # integer-valued, clamped [-8, 7]
    q_int8 = q.to(torch.int8)
    packed = pack_to_int32(q_int8, 4, packed_dim=1)
    scale_bf16 = scale.to(torch.bfloat16)
    shape = torch.tensor([out_dim, in_dim], dtype=torch.int64)

    # self-check: pack round-trips exactly, reconstruction error sane
    unpacked = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1)
    assert torch.equal(unpacked, q_int8), "pack/unpack mismatch"
    deq = dequantize(q_int8, scale, zp, qargs)
    rel = float((w - deq).norm() / w.norm().clamp_min(1e-9))
    return packed, scale_bf16, shape, rel


def build_head_group(body_group0: dict, head_group_size: int) -> dict:
    """Derive the lm_head quant group from the body's group_0 (only group_size differs)."""
    head_weights = copy.deepcopy(body_group0["weights"])
    head_weights["group_size"] = head_group_size
    head_weights["strategy"] = "group"
    return {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": ["re:.*lm_head"],
        "weights": head_weights,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="merged lmhead12k_empirical checkpoint (bf16 12k head + int4 body)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--head-group-size", type=int, default=128,
                    help="128 (primary arm) or 64 (robustness arm)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = args.head_group_size

    st_path = src / "model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    head_bf16 = None
    n_copy = 0
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            if name == HEAD_KEY_BF16:
                head_bf16 = t
                continue  # replaced by int4 packed tensors below
            tensors[name] = t
            n_copy += 1

    if head_bf16 is None:
        raise SystemExit(f"{HEAD_KEY_BF16} not found in {st_path} -- src must be the bf16-head rung")
    if tuple(head_bf16.shape) != (12288, HIDDEN):
        raise SystemExit(f"unexpected head shape {tuple(head_bf16.shape)} (want [12288, {HIDDEN}])")
    if head_bf16.dtype != torch.bfloat16:
        raise SystemExit(f"head dtype {head_bf16.dtype} != bfloat16 -- src head already quantized?")

    bf16_head_bytes = head_bf16.numel() * 2
    packed, scale, shape, rel = quantize_head(head_bf16, g)
    tensors["lm_head.weight_packed"] = packed
    tensors["lm_head.weight_scale"] = scale
    tensors["lm_head.weight_shape"] = shape
    int4_head_bytes = packed.numel() * 4 + scale.numel() * 2 + shape.numel() * 8
    print(f"[lm_head] int4 g{g}: packed={tuple(packed.shape)} int32 "
          f"scale={tuple(scale.shape)} bf16 rel_err={rel:.4f}")
    print(f"[lm_head] head bytes: bf16 {bf16_head_bytes/1e6:.2f} MB -> "
          f"int4 {int4_head_bytes/1e6:.2f} MB ({bf16_head_bytes/int4_head_bytes:.2f}x cut)")
    print(f"[copy] body tensors copied byte-for-byte: {n_copy}")

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    # Copy kept_ids.json + every non-safetensors asset verbatim FIRST (this includes
    # the source config.json), then overwrite config.json below so the asset copy
    # cannot clobber our edited quant config.
    for extra in src.iterdir():
        if extra.suffix == ".safetensors" or extra.name.endswith(".safetensors.index.json"):
            continue
        if extra.name == "config.json":
            continue  # written below with the edited quant config
        dest = out / extra.name
        if extra.is_dir():
            shutil.copytree(extra, dest, dirs_exist_ok=True)
        elif extra.is_file():
            shutil.copy2(extra, dest)

    # config.json: un-ignore lm_head + add the head quant group; keep body untouched.
    cfg = json.loads((src / "config.json").read_text())
    qc = cfg["quantization_config"]
    if "lm_head" in qc.get("ignore", []):
        qc["ignore"] = [m for m in qc["ignore"] if m != "lm_head"]
    qc["config_groups"]["group_1"] = build_head_group(qc["config_groups"]["group_0"], g)
    qc["version"] = compressed_tensors.__version__
    cfg["tie_word_embeddings"] = False
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"]["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[write] config.json (lm_head un-ignored, group_1 head g{g}, vocab_size "
          f"{cfg.get('text_config', {}).get('vocab_size', cfg.get('vocab_size'))})")

    head_bytes = {
        "head_group_size": g,
        "kept_size": 12288,
        "hidden": HIDDEN,
        "bf16_head_bytes": bf16_head_bytes,
        "int4_head_bytes": int4_head_bytes,
        "int4_packed_bytes": packed.numel() * 4,
        "int4_scale_bytes": scale.numel() * 2,
        "byte_cut_x": round(bf16_head_bytes / int4_head_bytes, 3),
        "quant_rel_err": round(rel, 6),
        "weight_packed_shape": list(packed.shape),
        "weight_scale_shape": list(scale.shape),
        "scheme": "int4 W4A16 symmetric group pack-quantized (compressed-tensors)",
        "note": "on-disk packed bytes; vLLM repacks to Marlin int4 at load (real int4 GEMV)",
    }
    (out / "head_bytes_build.json").write_text(json.dumps(head_bytes, indent=2))
    print(f"[done] checkpoint at {out} "
          f"({sum(p.stat().st_size for p in out.glob('*'))/1e9:.2f} GB); "
          f"head_bytes_build.json written")


if __name__ == "__main__":
    main()
