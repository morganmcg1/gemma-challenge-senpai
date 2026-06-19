#!/usr/bin/env python
"""Build a DENSE bf16 fake-quant Gemma-4-E4B checkpoint for the selective-g32 AIME
test (PR #702 / ubel).

The body Linear weights are fake-quantized (quant->dequant) at a per-module group
size; everything else (towers, norms, embeddings, PLE) is copied verbatim from the
official g32 w4a16-ct source. lm_head is untied and fake-quant'd at g128 (the served
`int4_g128_lmhead` recipe), held CONSTANT across all three arms so the ONLY variable
is the body group-size grid.

Three arms:
  * full_g128  : every body module at g128  (re-quant of the g32-dequant ref)
  * full_g32   : every body module at g32   (== the g32-dequant ref directly = the
                 served uniform-g32 body, exact)
  * selective  : the 48-module activation-critical subset at g32, the other 295 at g128

g32 treatment = the g32-dequant reference used DIRECTLY (so full_g32 == the served
w4a16-ct body exactly). g128 treatment = fake_quant(g32-dequant, 128) (same source
the #140 PPL scan and the served int4_g128_lmhead build use). This isolates the body
group-size grid as the only difference between arms, served via one identical bf16
path (no mixed-grid compressed serialization).

LOCAL only: reads on-disk checkpoints, no HF Job / fetch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

G32_SRC = (
    "/senpai-run/home/student-ubel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
EMBED = "model.language_model.embed_tokens.weight"
ASSETS = [
    "generation_config.json", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
    "preprocessor_config.json",
]


def qargs(gs: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=4, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def dequant_packed(packed: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """int32-packed int4 + per-group bf16 scale -> fp32 dense (symmetric, zp=0)."""
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1).to(torch.float32)
    assert int(q.min()) >= -8 and int(q.max()) <= 7, f"int4 range [{q.min()},{q.max()}]"
    ng = scale.shape[1]
    gs = in_dim // ng
    qg = q.reshape(out_dim, ng, gs)
    return (qg * scale.float().unsqueeze(-1)).reshape(out_dim, in_dim)


def fake_quant_dense(w: torch.Tensor, gs: int) -> tuple[torch.Tensor, float]:
    """quant->dequant a dense fp32 weight at int4 group size gs; returns (dense fp32, rel_err)."""
    w = w.to(torch.float32)
    out_dim, in_dim = w.shape
    assert in_dim % gs == 0, f"in_dim {in_dim} not divisible by {gs}"
    a = qargs(gs)
    ng = in_dim // gs
    wg = w.reshape(out_dim, ng, gs)
    scale, zp = calculate_qparams(wg.amin(dim=-1), wg.amax(dim=-1), a)
    q = quantize(w, scale, zp, a)
    deq = dequantize(q, scale, zp, a)
    rel = float((w - deq).norm() / w.norm().clamp_min(1e-9))
    return deq, rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full_g128", "full_g32", "selective"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=G32_SRC)
    ap.add_argument("--module-list",
                    default="submissions/int4_g128_lmhead/official_quantized_modules.json")
    ap.add_argument("--subset",
                    default="research/int4_aime_selective_g32_build/subset48_manifest.json")
    ap.add_argument("--head-gs", type=int, default=128)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    quant_modules = set(json.load(open(args.module_list)))
    assert len(quant_modules) == 343, len(quant_modules)
    subset = set(json.load(open(args.subset))["modules"])
    assert len(subset) == 48 and subset <= quant_modules, (len(subset), len(subset & quant_modules))

    def gs_for(mod: str) -> int | None:
        """None => g32-direct (use dequant ref as-is); int => fake_quant at that gs."""
        if args.arm == "full_g32":
            return None
        if args.arm == "full_g128":
            return 128
        return None if mod in subset else 128  # selective

    skip = set()
    for m in quant_modules:
        for s in (".weight_packed", ".weight_scale", ".weight_shape"):
            skip.add(m + s)
    skip.add("lm_head.weight")  # replaced by untied fake-quant head

    tensors: dict[str, torch.Tensor] = {}
    embed_w = None
    n_direct = n_fq = n_copy = 0
    rel_body: list[float] = []
    n_g32 = n_g128 = 0

    with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # 1) verbatim copies (norms, towers, embeddings, PLE, etc.)
        for k in keys:
            if k in skip:
                continue
            t = f.get_tensor(k)
            if k == EMBED:
                embed_w = t
            tensors[k] = t
            n_copy += 1
        # 2) body modules -> dense bf16 at the per-arm grid
        for mod in sorted(quant_modules):
            packed = f.get_tensor(mod + ".weight_packed")
            scale = f.get_tensor(mod + ".weight_scale")
            shape = f.get_tensor(mod + ".weight_shape")
            ref = dequant_packed(packed, scale, shape)  # fp32 g32-dequant reference
            gs = gs_for(mod)
            if gs is None:
                w = ref  # g32-direct (== served uniform-g32 body)
                n_direct += 1
                n_g32 += 1
            else:
                w, rel = fake_quant_dense(ref, gs)
                rel_body.append(rel)
                n_fq += 1
                n_g128 += 1
            tensors[mod + ".weight"] = w.to(torch.bfloat16)

    assert embed_w is not None

    # 2b) KV-shared k_norm gap. vLLM's Gemma4 registers a k_norm RMSNorm for EVERY
    # attention layer, but the last `num_kv_shared_layers` (24-41 here) never CALL it
    # (gemma4.py:522 guards `if not is_kv_shared_layer: k = self.k_norm(k)`). The
    # official w4a16-ct checkpoint omits k_norm for those layers; the compressed-tensors
    # loader tolerates the gap but the plain-bf16 default loader is STRICT
    # (track_weights_loading raises "weights were not initialized"). Synthesize a
    # shape-correct k_norm.weight for each layer that has q_norm but no k_norm — value
    # is a clone of that layer's q_norm (identical head_dim), numerically irrelevant
    # because the module is never invoked in the KV-shared forward.
    n_knorm = 0
    for k in list(tensors.keys()):
        if k.endswith(".self_attn.q_norm.weight"):
            kk = k[: -len("q_norm.weight")] + "k_norm.weight"
            if kk not in tensors:
                tensors[kk] = tensors[k].clone()
                n_knorm += 1
    print(f"[build:{args.arm}] synthesized {n_knorm} KV-shared k_norm.weight (== q_norm shape, unused in fwd)", flush=True)

    # 3) untied lm_head from bf16 embed_tokens, fake-quant at head-gs (constant across arms)
    head, head_rel = fake_quant_dense(embed_w, args.head_gs)
    tensors["lm_head.weight"] = head.to(torch.bfloat16)

    n_q = n_direct + n_fq
    assert n_q == 343, n_q
    print(f"[build:{args.arm}] body modules={n_q} (g32-direct={n_g32} fake_quant_g128={n_g128}) "
          f"copied={n_copy}", flush=True)
    if rel_body:
        print(f"[build:{args.arm}] g128 body rel_err min={min(rel_body):.4f} "
              f"max={max(rel_body):.4f} mean={sum(rel_body)/len(rel_body):.4f}", flush=True)
    print(f"[build:{args.arm}] lm_head rel_err={head_rel:.4f} (head_gs={args.head_gs})", flush=True)

    # config: drop quant config -> plain bf16 model; untie head
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
