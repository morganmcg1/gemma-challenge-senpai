#!/usr/bin/env python
"""PR #659 — build a MIXED-precision body: int4 everywhere, int8 (or bf16) on a
named subset of decoder layers. Measures the AIME precision-allocation lower bound.

Design (holds the WEIGHT SOURCE fixed so the only axis that varies vs the live int4
body is the per-layer BIT-WIDTH on the upgraded layers):

  * SKELETON = the live operative int4 body (/workspace/gemma_build/int4_g128_lmhead).
    Every int4 module is copied BYTE-IDENTICAL, so N=0 (--upgrade-layers none) reproduces
    the operative 0.400-AIME body exactly, and every non-upgraded layer in every cell is
    identical to it.
  * UPGRADE source = the SAME qat-unquantized bf16 checkpoint the int4 body was built
    from (build_quant.py: /workspace/gemma_build/qat_unq). For each module in an upgraded
    layer we re-quantize that module's bf16 weight at int8 g128 (or keep it plain bf16),
    so the int4->int8 (or int4->bf16) step on those layers is BIT-WIDTH-ONLY on identical
    source weights.
  * lm_head stays int4 g128 (the LOCKED int4_g128_lmhead rung — never touched).

This isolates the question the card asks: starting from the shipped qat int4 body, what
is the MINIMUM layer-upgrade that recovers AIME, and its TPS price? N=all int8 is the
uniform-int8-from-qat anchor for THIS ladder (note: distinct from #646's int8 which was
built from the PLAIN bf16 base; see PR for the documented base choice).

vLLM 0.22.0 compressed-tensors loading of a MIXED 4/8 body: group_0 (int4) and a new
group_2 (int8) each carry an EXPLICIT module-name `targets` list (no overlap; every body
module name is in exactly one of group_0 / group_2 / ignore), group_1 keeps the int4
lm_head. bf16-upgraded modules are stored as a plain `.weight` and added to `ignore`.

ANALYSIS-ONLY local build. Does NOT launch any HF Job. analysis_only, official_tps=0.
"""
from __future__ import annotations

import argparse
import json
import re
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

ASSET_FILES = [
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "preprocessor_config.json",
]
BITS_RANGE = {4: (-8, 7), 8: (-128, 127)}


def make_qargs(num_bits: int, group_size: int) -> QuantizationArgs:
    if group_size == -1:
        return QuantizationArgs(num_bits=num_bits, type="int", strategy="channel",
                                symmetric=True, observer="minmax")
    return QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                            group_size=group_size, symmetric=True, observer="minmax")


def quantize_weight(w: torch.Tensor, num_bits: int, group_size: int):
    """Return (weight_packed[int32], weight_scale[bf16], weight_shape[int64], rel_err).

    pack_to_int32 with num_bits packs (32//num_bits) values per int32 word — the WNA16
    layout vLLM's Marlin path reads (8 int4/word for num_bits=4, 4 int8/word for 8)."""
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    qargs = make_qargs(num_bits, group_size)
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
    q = quantize(w, scale, zp, qargs)
    lo, hi = BITS_RANGE[num_bits]
    q_int = q.to(torch.int8 if num_bits == 8 else torch.int8)  # int4 still packs from int8
    assert int(q_int.amin()) >= lo and int(q_int.amax()) <= hi, f"{num_bits}-bit range overflow"
    packed = pack_to_int32(q_int, num_bits, packed_dim=1)
    shape = torch.tensor([out_dim, in_dim], dtype=torch.int64)
    unpacked = unpack_from_int32(packed, num_bits, torch.Size([out_dim, in_dim]), packed_dim=1)
    assert torch.equal(unpacked, q_int), "pack/unpack mismatch"
    deq = dequantize(q_int, scale, zp, qargs)
    rel = (w - deq).norm() / w.norm().clamp_min(1e-9)
    return packed, scale.to(torch.bfloat16), shape, float(rel)


def parse_layers(spec: str, all_layers: list[int]) -> list[int]:
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        return list(all_layers)
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--int4-body", default="/workspace/gemma_build/int4_g128_lmhead",
                    help="live int4 body = the SKELETON (int4 modules copied byte-identical)")
    ap.add_argument("--qat-src", default="/workspace/gemma_build/qat_unq",
                    help="qat-unquantized bf16 source the int4 body was built from; "
                         "upgraded layers are (re)quantized from THIS so base is held fixed")
    ap.add_argument("--module-list",
                    default="/workspace/senpai/target/submissions/int4_g128_lmhead/"
                            "official_quantized_modules.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--upgrade-layers", default="none",
                    help="comma/range list of decoder-layer indices to upgrade, or none/all")
    ap.add_argument("--upgrade-precision", default="int8", choices=["int8", "bf16"])
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    int4_body = Path(args.int4_body)
    qat_src = Path(args.qat_src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_modules = sorted(json.load(open(args.module_list)))
    assert len(quant_modules) == 343, f"expected 343 modules, got {len(quant_modules)}"
    all_layers = sorted({int(m.group(1)) for x in quant_modules
                         if (m := re.search(r"language_model\.layers\.(\d+)\.", x))})
    upgrade_layers = parse_layers(args.upgrade_layers, all_layers)
    bad = [l for l in upgrade_layers if l not in all_layers]
    assert not bad, f"layers not in model: {bad}"

    def layer_of(mod: str) -> int | None:
        m = re.search(r"language_model\.layers\.(\d+)\.", mod)
        return int(m.group(1)) if m else None

    upgrade_modules = [m for m in quant_modules if layer_of(m) in upgrade_layers]
    int4_modules = [m for m in quant_modules if m not in set(upgrade_modules)]
    print(f"[plan] upgrade_layers={upgrade_layers} ({len(upgrade_layers)} layers) "
          f"precision={args.upgrade_precision}", flush=True)
    print(f"[plan] upgrade_modules={len(upgrade_modules)} int4_modules={len(int4_modules)}",
          flush=True)

    # ---- load skeleton (live int4 body) tensors ----
    sk_path = int4_body / "model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(sk_path), framework="pt", device="cpu") as f:
        sk_keys = list(f.keys())
        for k in sk_keys:
            tensors[k] = f.get_tensor(k)
    print(f"[skeleton] loaded {len(tensors)} int4-body tensors", flush=True)

    # ---- upgrade the chosen modules from the qat-unquantized bf16 source ----
    upgraded_bf16_modules: list[str] = []
    upgraded_int8_modules: list[str] = []
    rel_errs: list[float] = []
    qat_st = qat_src / "model.safetensors"
    with safe_open(str(qat_st), framework="pt", device="cpu") as q:
        qkeys = set(q.keys())
        for mod in upgrade_modules:
            wname = mod + ".weight"
            assert wname in qkeys, f"qat source missing {wname}"
            w = q.get_tensor(wname)
            # remove the int4 tensors for this module from the skeleton
            for suf in (".weight_packed", ".weight_scale", ".weight_shape"):
                tensors.pop(mod + suf, None)
            if args.upgrade_precision == "bf16":
                tensors[wname] = w.to(torch.bfloat16)
                upgraded_bf16_modules.append(mod)
            else:
                packed, scale, shape, rel = quantize_weight(w, 8, args.group_size)
                tensors[mod + ".weight_packed"] = packed
                tensors[mod + ".weight_scale"] = scale
                tensors[mod + ".weight_shape"] = shape
                upgraded_int8_modules.append(mod)
                rel_errs.append(rel)
    if rel_errs:
        print(f"[upgrade:int8] {len(upgraded_int8_modules)} modules, rel_err "
              f"min={min(rel_errs):.4f} max={max(rel_errs):.4f} "
              f"mean={sum(rel_errs)/len(rel_errs):.4f}", flush=True)
    if upgraded_bf16_modules:
        print(f"[upgrade:bf16] {len(upgraded_bf16_modules)} modules kept plain bf16", flush=True)

    # ---- rebuild quantization_config ----
    # vLLM 0.22.0 names the running text-decoder modules `language_model.model.layers.<L>.…`
    # (the hf_to_vllm_mapper swaps `model.language_model.…` -> `language_model.model.…`),
    # and FUSES q/k/v -> self_attn.qkv_proj and gate/up -> mlp.gate_up_proj. find_matched_target
    # precedence is: (1) layer_name exact-or-`re:`regex, (2) class-name substring ("Linear"
    # is contained in "QKVParallelLinear"), (3) fused-component expansion. So a per-layer
    # REGEX matches the fused running name in stage (1) and WINS over the bare "Linear" class
    # match (stage 2) — explicit component paths do NOT (they never equal the fused name, and
    # stage-2 "Linear" fires before stage-3 expansion). int4 keeps the proven "Linear" class
    # target (the operative body's routing); the int8 layers get a layer-index regex.
    cfg = json.load(open(int4_body / "config.json"))
    qc = cfg["quantization_config"]
    base_ignore = list(qc.get("ignore", []))
    g0 = json.loads(json.dumps(qc["config_groups"]["group_0"]))  # Linear, num_bits=4
    g1 = json.loads(json.dumps(qc["config_groups"]["group_1"]))  # lm_head int4 (unchanged)

    def layer_regex(layers: list[int]) -> str:
        alt = "|".join(str(l) for l in sorted(layers))
        return rf"re:.*language_model\.model\.layers\.({alt})\."

    new_groups: dict[str, dict] = {}
    if upgraded_int8_modules:
        g8w = json.loads(json.dumps(g0["weights"]))
        g8w["num_bits"] = 8
        # int8 group FIRST (defensive: its regex is checked early in target_scheme_map).
        new_groups["group_int8"] = {"targets": [layer_regex(upgrade_layers)], "weights": g8w}
    new_groups["group_0"] = g0   # ["Linear"] num_bits=4 — unchanged operative routing
    new_groups["group_1"] = g1   # re:.*lm_head num_bits=4 — LOCKED int4 lm_head rung
    # bf16-upgraded layers: keep unquantized -> ignore via the same per-layer regex
    # (should_ignore_layer expands the fused qkv/gate_up to component shards and each
    # matches the regex prefix, so the whole layer is consistently unquantized).
    new_ignore = list(base_ignore)
    if upgraded_bf16_modules:
        bf16_layers = sorted({layer_of(m) for m in upgraded_bf16_modules})
        new_ignore.append(layer_regex(bf16_layers))

    qc["config_groups"] = new_groups
    qc["ignore"] = new_ignore
    qc["version"] = compressed_tensors.__version__
    qc["quantization_status"] = "compressed"
    qc["format"] = "pack-quantized"
    cfg["quantization_config"] = qc

    # ---- integrity checks ----
    # Every body module is routed to exactly one place (int4 / int8 / bf16-ignore).
    routed = set(int4_modules) | set(upgraded_int8_modules) | set(upgraded_bf16_modules)
    assert routed == set(quant_modules), "module routing lost/added a body module"
    # int8 modules must have packed tensors, bf16 modules a plain weight, neither both.
    for mod in upgraded_int8_modules:
        assert (mod + ".weight_packed") in tensors and (mod + ".weight") not in tensors
    for mod in upgraded_bf16_modules:
        assert (mod + ".weight") in tensors and (mod + ".weight_packed") not in tensors
    # int4 (non-upgraded) modules untouched: still packed, byte-identical to skeleton.
    with safe_open(str(sk_path), framework="pt", device="cpu") as f:
        for mod in int4_modules[:5] + int4_modules[-5:]:
            assert torch.equal(tensors[mod + ".weight_packed"],
                               f.get_tensor(mod + ".weight_packed")), f"int4 {mod} mutated"

    print(f"[config] groups={list(new_groups)} "
          f"ignore={len(new_ignore)} (base {len(base_ignore)} + bf16 {len(upgraded_bf16_modules)})",
          flush=True)
    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print(f"[write] wrote config.json (mixed {args.upgrade_precision} on "
          f"{len(upgrade_layers)} layers)", flush=True)

    copied = []
    for fn in ASSET_FILES:
        s = int4_body / fn
        if not s.exists():
            s = qat_src / fn
        if s.exists():
            shutil.copy2(s, out / fn)
            copied.append(fn)
    print(f"[write] copied assets: {copied}", flush=True)
    sz = sum(p.stat().st_size for p in out.glob('*')) / 1e9
    print(f"[done] mixed checkpoint at {out} ({sz:.2f} GB) "
          f"layers={args.upgrade_layers} prec={args.upgrade_precision}", flush=True)


if __name__ == "__main__":
    main()
