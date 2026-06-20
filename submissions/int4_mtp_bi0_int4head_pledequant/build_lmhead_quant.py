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

Optional `--dequant-ple` adds ONE further delta on top of the int4 head: the
42 `per_layer_input_gate` projections (in_dim=2560 -> out_dim N=256, one per
layer) are de-quantized from int4 back to bf16 and `re:.*per_layer_input_gate`
is added to the quant-config `ignore` list, so vLLM serves them via
`UnquantizedLinearMethod` (cuBLAS) instead of int4-Marlin. At N=256 the
int4-Marlin tile grid is starved (stark #798: 20.8 us vs 5.8 us bf16-cuBLAS,
int4 3.6x SLOWER); the sibling `per_layer_projection` (N=2560) still wins at
int4 and is left untouched. Default `--ple-source dequant` reconstructs the
EXACT served weight values (q_int4 * scale -> bf16), isolating the kernel as
the only changed variable (quality stays neutral). `--ple-source bf16
--ple-bf16-src DIR` instead sources a separate bf16 `.weight`; note that
`google/gemma-4-E4B-it` is NOT the parent of this QAT int4 body (measured
rel~0.16-0.27), so its weights are a DIFFERENT set -- the true bf16 master is
`google/gemma-4-E4B-it-qat-q4_0-unquantized` (measured rel~0.067 == int4
rounding).
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
PLE_GATE_NEEDLE = ".per_layer_input_gate."
PLE_GATE_IGNORE = "re:.*per_layer_input_gate"
EXPECTED_PLE_GATES = 42
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


def dequantize_packed(
    packed: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor, num_bits: int
) -> torch.Tensor:
    """int4/int8 pack-quantized (packed[int32] + grouped scale) -> bf16 weight.

    Reconstructs the EXACT values vLLM's Marlin path serves (q_int * scale),
    so swapping to this bf16 weight changes only the kernel, not the math.
    """
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q = unpack_from_int32(packed, num_bits, torch.Size([out_dim, in_dim]), packed_dim=1)
    n_groups = scale.shape[1] if scale.dim() == 2 else 1
    if n_groups > 1 and n_groups != in_dim:
        group_size = in_dim // n_groups
        qargs = QuantizationArgs(
            num_bits=num_bits, type="int", strategy="group", group_size=group_size,
            symmetric=True, observer="minmax",
        )
    else:
        qargs = QuantizationArgs(
            num_bits=num_bits, type="int", strategy="channel", symmetric=True,
            observer="minmax",
        )
    deq = dequantize(q.to(torch.int8), scale.to(torch.float32), None, qargs)
    return deq


def build_quant_config(
    src_qc: dict, num_bits: int, group_size: int, dequant_ple: bool = False
) -> dict:
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
    if dequant_ple and PLE_GATE_IGNORE not in qc["ignore"]:
        # Force per_layer_input_gate to serve unquantized (bf16/cuBLAS), not
        # int4-Marlin. The matching bf16 `.weight` tensors are written below.
        qc["ignore"].append(PLE_GATE_IGNORE)
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
    ap.add_argument("--dequant-ple", action="store_true",
                    help="de-quantize the 42 per_layer_input_gate (N=256) projections "
                         "back to bf16 (served via cuBLAS, not int4-Marlin)")
    ap.add_argument("--ple-source", choices=["dequant", "bf16"], default="dequant",
                    help="dequant: reconstruct EXACT served values q_int*scale->bf16 "
                         "(clean kernel isolation, quality-neutral; default). "
                         "bf16: source a separate bf16 .weight from --ple-bf16-src.")
    ap.add_argument("--ple-bf16-src", default=None,
                    help="checkpoint dir holding bf16 per_layer_input_gate .weight "
                         "tensors (required when --ple-source bf16). NOTE: "
                         "google/gemma-4-E4B-it is NOT this QAT body's parent; the true "
                         "bf16 master is google/gemma-4-E4B-it-qat-q4_0-unquantized.")
    args = ap.parse_args()
    if args.ple_source == "bf16" and not args.dequant_ple:
        ap.error("--ple-source bf16 requires --dequant-ple")
    if args.ple_source == "bf16" and not args.ple_bf16_src:
        ap.error("--ple-source bf16 requires --ple-bf16-src DIR")

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Body quant params (group_0) -- used to dequantize per_layer_input_gate,
    # which is part of the int4 body (matched by group_0 targets ["Linear"]).
    src_cfg = json.load(open(src / "config.json"))
    body_w = src_cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    body_bits = int(body_w["num_bits"])
    body_gs = body_w.get("group_size")

    st_path = src / "model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    n_copy = 0
    head_done = False
    ple_buf: dict[str, dict[str, torch.Tensor]] = {}

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
            elif args.dequant_ple and PLE_GATE_NEEDLE in name:
                # Buffer the int4 per_layer_input_gate parts; emit bf16 .weight below.
                prefix, leaf = name.rsplit(".", 1)
                ple_buf.setdefault(prefix, {})[leaf] = t
            else:
                tensors[name] = t
                n_copy += 1

    assert head_done, "lm_head.weight not found in source checkpoint"
    print(f"[copy] copied {n_copy} tensors byte-identical (int4 body + embeddings + mm towers)",
          flush=True)

    if args.dequant_ple:
        assert len(ple_buf) == EXPECTED_PLE_GATES, (
            f"expected {EXPECTED_PLE_GATES} per_layer_input_gate modules, "
            f"found {len(ple_buf)}"
        )
        bf16_f = None
        if args.ple_source == "bf16":
            bf16_f = safe_open(
                str(Path(args.ple_bf16_src) / "model.safetensors"),
                framework="pt", device="cpu",
            )
        store_rels = []
        for prefix, parts in sorted(ple_buf.items()):
            if args.ple_source == "dequant":
                deq = dequantize_packed(
                    parts["weight_packed"], parts["weight_scale"],
                    parts["weight_shape"], body_bits,
                )  # fp32, == exact Marlin-served values
            else:
                deq = bf16_f.get_tensor(prefix + ".weight").to(torch.float32)
            w_bf16 = deq.to(torch.bfloat16)
            tensors[prefix + ".weight"] = w_bf16
            store_rels.append(
                ((deq - w_bf16.to(torch.float32)).norm() / deq.norm().clamp_min(1e-9)).item()
            )
        if bf16_f is not None:
            bf16_f.__exit__(None, None, None)
        sample = sorted(ple_buf)[0]
        sh = tuple(tensors[sample + ".weight"].shape)
        print(
            f"[ple-dequant] {len(ple_buf)} per_layer_input_gate -> bf16 "
            f"(source={args.ple_source}, shape={sh}, bf16-store rel "
            f"mean={sum(store_rels)/len(store_rels):.5f} max={max(store_rels):.5f}); "
            f"added re:.*per_layer_input_gate to ignore",
            flush=True,
        )

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    cfg = json.load(open(src / "config.json"))
    cfg["tie_word_embeddings"] = False
    cfg["text_config"]["tie_word_embeddings"] = False
    cfg["quantization_config"] = build_quant_config(
        cfg["quantization_config"], args.num_bits, args.head_group_size,
        dequant_ple=args.dequant_ple,
    )
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    msg = "[write] wrote config.json (tie=false, lm_head dropped from ignore, +group_1"
    msg += ", +per_layer_input_gate in ignore)" if args.dequant_ple else ")"
    print(msg)

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
