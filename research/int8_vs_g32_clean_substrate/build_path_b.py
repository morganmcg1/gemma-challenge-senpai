#!/usr/bin/env python
"""Path-B builder (PR #726 / ubel): DENSE bf16 fake-quant Gemma-4-E4B arms built
from ONE provenance-clean bf16 master (`qat_unq`), to decide whether fern #659's
int8-locus > full-g32 AIME edge REPLICATES on a substrate that can actually carry
int8's >int4 information.

This is the #702 `build_bf16_fakequant.py` re-sourced + extended:

  * SOURCE = the qat-unquantized bf16 master
    (`google/gemma-4-E4B-it-qat-q4_0-unquantized`). Every body Linear is read as a
    DENSE bf16 `<module>.weight` and fake-quantized (quant->dequant, bf16 matmul)
    per the arm recipe. (#702 instead dequantized an int4-g32 packed checkpoint,
    which made full_g32 == g32-direct and int8-of-that ~lossless == g32 -> the
    degenerate substrate this PR replaces.)
  * MODES added beyond #702's int4-only: int8 group-quant, and bf16 passthrough.

Arms (lm_head int4-g128 from bf16 embed, held CONSTANT across all arms so it
cancels in the paired test; locus = all body Linear in layers 14-27 = 118 mods):

  * full_g32   : int4 g32 on every body module                          (Arm1, baseline + substrate sanity; must reproduce #702 nqk9izab 0.3867)
  * int8_locus : int8 g128 on L14-27 (118) + int4 g32 elsewhere (225)   (Arm2, decisive; mirrors fern's upgrade_precision=int8, upgrade_layers=14-27)
  * bf16_locus : bf16 on L14-27 (118) + int4 g32 elsewhere (225)        (Arm3, ceiling; bf16 ⊇ int8 information)
  * int8_full  : int8 g128 on every body module                         (EXTEND, headroom bound)

LOCAL only: reads the on-disk bf16 master, no HF Job / fetch. analysis_only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams

REPO = "google/gemma-4-E4B-it-qat-q4_0-unquantized"
EMBED = "model.language_model.embed_tokens.weight"
LOCUS_LO, LOCUS_HI = 14, 27          # fern #659 upgrade_layers=14-27 (inclusive), 118 body modules
ASSETS = [
    "generation_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
    "preprocessor_config.json",
]


def resolve_src() -> str:
    """Path of the cached qat_unq snapshot (downloaded to the HF cache, no re-fetch)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(REPO, local_files_only=True)


def qargs(num_bits: int, gs: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def fake_quant_dense(w: torch.Tensor, num_bits: int, gs: int) -> tuple[torch.Tensor, float]:
    """quant->dequant a dense weight at int<num_bits> group size gs; returns (dense fp32, rel_err)."""
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    assert in_dim % gs == 0, f"in_dim {in_dim} not divisible by {gs}"
    a = qargs(num_bits, gs)
    ng = in_dim // gs
    wg = w.reshape(out_dim, ng, gs)
    scale, zp = calculate_qparams(wg.amin(dim=-1), wg.amax(dim=-1), a)
    q = quantize(w, scale, zp, a)
    deq = dequantize(q, scale, zp, a)
    rel = float((w - deq).norm() / w.norm().clamp_min(1e-9))
    return deq, rel


def layer_of(mod: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", mod)
    return int(m.group(1)) if m else None


def in_locus(mod: str) -> bool:
    L = layer_of(mod)
    return L is not None and LOCUS_LO <= L <= LOCUS_HI


def recipe(mod: str, arm: str) -> tuple[str, int | None]:
    """Per-module treatment -> ('int4',32) | ('int8',128) | ('bf16',None)."""
    if arm == "full_g32":
        return ("int4", 32)
    if arm == "int8_full":
        return ("int8", 128)
    if arm == "int8_locus":
        return ("int8", 128) if in_locus(mod) else ("int4", 32)
    if arm == "bf16_locus":
        return ("bf16", None) if in_locus(mod) else ("int4", 32)
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["full_g32", "int8_locus", "bf16_locus", "int8_full"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="qat_unq snapshot dir (default: resolve from HF cache)")
    ap.add_argument("--module-list",
                    default="submissions/int4_g128_lmhead/official_quantized_modules.json")
    ap.add_argument("--head-bits", type=int, default=4)
    ap.add_argument("--head-gs", type=int, default=128)
    args = ap.parse_args()

    src = Path(args.src or resolve_src())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_modules = set(json.load(open(args.module_list)))
    assert len(quant_modules) == 343, len(quant_modules)
    locus = sorted(m for m in quant_modules if in_locus(m))
    assert len(locus) == 118, f"locus L{LOCUS_LO}-{LOCUS_HI} = {len(locus)} (expected 118)"

    skip = {m + ".weight" for m in quant_modules}          # rebuilt below
    skip.add("lm_head.weight")                              # untied head synthesized

    tensors: dict[str, torch.Tensor] = {}
    embed_w = None
    n_copy = 0
    rel = {"int4": [], "int8": [], "bf16": []}
    n_kind = {"int4": 0, "int8": 0, "bf16": 0}

    src_st = src / "model.safetensors"
    with safe_open(str(src_st), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # 1) verbatim copies (norms, towers, embeddings, PLE, vision/audio, etc.)
        for k in keys:
            if k in skip:
                continue
            t = f.get_tensor(k)
            if k == EMBED:
                embed_w = t
            tensors[k] = t
            n_copy += 1
        # 2) body modules -> dense bf16 at the per-arm recipe
        for mod in sorted(quant_modules):
            w = f.get_tensor(mod + ".weight")              # DENSE bf16 from the master
            kind, gs = recipe(mod, args.arm)
            if kind == "bf16":
                out_w = w
                rel["bf16"].append(0.0)
            else:
                nb = 4 if kind == "int4" else 8
                deq, r = fake_quant_dense(w, nb, gs)
                out_w = deq
                rel[kind].append(r)
            n_kind[kind] += 1
            tensors[mod + ".weight"] = out_w.to(torch.bfloat16)

    assert embed_w is not None
    assert n_kind["int4"] + n_kind["int8"] + n_kind["bf16"] == 343

    # 2b) KV-shared k_norm gap. vLLM's Gemma4 registers a k_norm RMSNorm for every
    # attention layer, but the last num_kv_shared_layers never CALL it
    # (gemma4.py guards `if not is_kv_shared_layer: k = self.k_norm(k)`). The master
    # omits k_norm for those layers; the plain-bf16 loader is STRICT and raises
    # "weights were not initialized". Synthesize a shape-correct k_norm.weight (clone
    # of that layer's q_norm, identical head_dim) for each layer that has q_norm but no
    # k_norm -- numerically irrelevant because the module is never invoked.
    n_knorm = 0
    for k in list(tensors.keys()):
        if k.endswith(".self_attn.q_norm.weight"):
            kk = k[: -len("q_norm.weight")] + "k_norm.weight"
            if kk not in tensors:
                tensors[kk] = tensors[k].clone()
                n_knorm += 1
    print(f"[build:{args.arm}] synthesized {n_knorm} KV-shared k_norm.weight (== q_norm shape, unused in fwd)", flush=True)

    # 3) untied lm_head from bf16 embed_tokens, fake-quant int4 g128 (constant across arms)
    head, head_rel = fake_quant_dense(embed_w, args.head_bits, args.head_gs)
    tensors["lm_head.weight"] = head.to(torch.bfloat16)

    print(f"[build:{args.arm}] body modules=343  int4_g32={n_kind['int4']} "
          f"int8_g128={n_kind['int8']} bf16={n_kind['bf16']}  copied={n_copy}", flush=True)
    for kind in ("int4", "int8", "bf16"):
        rs = [x for x in rel[kind] if x > 0.0] if kind != "bf16" else rel[kind]
        if rel[kind]:
            mn = min(rel[kind]); mx = max(rel[kind]); me = sum(rel[kind]) / len(rel[kind])
            print(f"[build:{args.arm}]   {kind:4} rel_err n={len(rel[kind])} "
                  f"min={mn:.5f} max={mx:.5f} mean={me:.5f}", flush=True)
    print(f"[build:{args.arm}] lm_head rel_err={head_rel:.4f} (int{args.head_bits} g{args.head_gs})", flush=True)

    # config: drop any quant config -> plain bf16 model; untie head
    cfg = json.load(open(src / "config.json"))
    cfg.pop("quantization_config", None)
    cfg["tie_word_embeddings"] = False
    if "text_config" in cfg:
        cfg["text_config"]["tie_word_embeddings"] = False
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    print("[write] saving model.safetensors ...", flush=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    for fn in ASSETS:
        s = src / fn
        if s.exists():
            shutil.copy2(s, out / fn)
    sz = sum(p.stat().st_size for p in out.glob("*")) / 1e9
    print(f"[done] {args.arm} -> {out} ({sz:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
