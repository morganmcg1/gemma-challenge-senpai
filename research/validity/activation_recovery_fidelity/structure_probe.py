#!/usr/bin/env python
"""PR #711 -- structure smoke for the activation-fidelity recovery probe. DISK-SAFE.

Validates the core mechanism cheaply BEFORE the full probe:
  1. Loads the bf16 QAT master (the source build_quant.py uses) read-only.
  2. Maps the 343 official quantized-module names -> live nn.Linear objects.
  3. Confirms output_hidden_states gives per-layer residual-stream hidden states.
  4. Verifies in-memory fake-quant(g128) reproduces the shipped int4_g128 codes
     EXACTLY (my quantizer == submissions/int4_g128_lmhead/build_quant.py), and
     that fake-quant(g32) reproduces the official w4a16-ct codes.
  5. Prints module taxonomy (PLIG / qkv counts) and peak HBM.

NO checkpoint write, NO dataset eval, NO generation. Run:
  CUDA_VISIBLE_DEVICES=0 uv run python \
    research/validity/activation_recovery_fidelity/structure_probe.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)

ROOT = Path("/workspace/senpai/target")
QAT_UNQ = Path("/workspace/gemma_build/qat_unq")                 # bf16 master (src)
SHIPPED_G128 = Path("/workspace/gemma_build/int4_g128_lmhead/model.safetensors")
OFFICIAL_G32 = Path(
    "/senpai-run/home/student-stark/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
)
MODULE_LIST = ROOT / "submissions/int4_g128_lmhead/official_quantized_modules.json"
DEV = "cuda"


def qargs(gs: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=4, type="int", strategy="group",
                            group_size=gs, symmetric=True, observer="minmax")


def fake_quant(W: torch.Tensor, gs: int) -> torch.Tensor:
    """dequant(quant(W, gs)) -- in-memory fake-quant, returns bf16-grade fp32."""
    W = W.to(torch.float32)
    out_dim, in_dim = W.shape
    qa = qargs(gs)
    wg = W.reshape(out_dim, in_dim // gs, gs)
    scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), qa)
    q = quantize(W, scale, zp, qa)
    return dequantize(q, scale, zp, qa)


def quant_codes(W: torch.Tensor, gs: int) -> torch.Tensor:
    """int32-packed codes for code-equality vs shipped checkpoints."""
    W = W.to(torch.float32)
    out_dim, in_dim = W.shape
    qa = qargs(gs)
    wg = W.reshape(out_dim, in_dim // gs, gs)
    scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), qa)
    q = quantize(W, scale, zp, qa).to(torch.int8)
    return pack_to_int32(q.cpu(), 4, packed_dim=1)


def mtype(name: str) -> str:
    for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
              "down_proj", "per_layer_input_gate", "per_layer_projection",
              "per_layer_model_projection"):
        if name.endswith(t):
            return t
    return "other"


def main() -> None:
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    modules = sorted(json.load(open(MODULE_LIST)))
    print(f"[smoke] {len(modules)} official quantized modules", flush=True)

    print("[smoke] loading bf16 QAT master ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(QAT_UNQ), dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEV).eval()
    torch.cuda.reset_peak_memory_stats()
    print(f"[smoke] loaded {type(model).__name__} in {time.time()-t0:.0f}s; "
          f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # -- name -> module map. named_modules() may or may not carry a leading 'model.'
    name2mod = dict(model.named_modules())
    have = sum(1 for m in modules if m in name2mod)
    # try stripping a leading 'model.' if the list keys don't match directly
    stripped = {m: m[len("model."):] if m.startswith("model.") else m for m in modules}
    have_stripped = sum(1 for m in modules if stripped[m] in name2mod)
    key_mode = "direct" if have >= have_stripped else "stripped"
    matched = have if key_mode == "direct" else have_stripped
    print(f"[smoke] module-name match: direct={have} stripped={have_stripped} "
          f"-> using {key_mode} ({matched}/{len(modules)})", flush=True)
    # show a couple resolved examples + their weight shapes
    ex = []
    for m in modules[:3] + [x for x in modules if x.endswith("per_layer_input_gate")][:1]:
        key = m if key_mode == "direct" else stripped[m]
        mod = name2mod.get(key)
        if mod is not None and hasattr(mod, "weight"):
            ex.append((m, tuple(mod.weight.shape), str(mod.weight.dtype)))
    for e in ex:
        print(f"    {e[0]}  shape={e[1]}  dtype={e[2]}", flush=True)

    # -- decoder layer count via the longest ModuleList ending in 'layers'
    import torch.nn as nn
    cands = [(n, mod) for n, mod in model.named_modules()
             if isinstance(mod, nn.ModuleList) and n.endswith("layers")]
    cands.sort(key=lambda x: len(x[1]))
    layer_stack_name, layers = cands[-1]
    print(f"[smoke] decoder stack '{layer_stack_name}' has {len(layers)} layers", flush=True)

    # -- forward with hidden states (tiny seq)
    tok = AutoTokenizer.from_pretrained(str(QAT_UNQ), trust_remote_code=True)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    print(f"[smoke] forward ok: {len(hs)} hidden tensors, each {tuple(hs[0].shape)}; "
          f"logits {tuple(out.logits.shape)}", flush=True)

    # -- faithfulness: fake-quant(g128) codes == shipped int4_g128 codes?
    print("[smoke] verifying fake-quant codes vs shipped g128 + official g32 ...", flush=True)
    test_names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.10.mlp.gate_proj",
        "model.language_model.layers.20.per_layer_input_gate",
    ]
    g32_snap = next(OFFICIAL_G32.glob("*/model.safetensors"), None)
    with safe_open(str(SHIPPED_G128), framework="pt", device="cpu") as fs, \
         safe_open(str(g32_snap), framework="pt", device="cpu") as fg:
        ship_keys = set(fs.keys())
        g32_keys = set(fg.keys())
        for nm in test_names:
            key = nm if key_mode == "direct" else stripped[nm]
            W = name2mod[key].weight.detach()
            my128 = quant_codes(W, 128)
            my32 = quant_codes(W, 32)
            pk = nm + ".weight_packed"
            r128 = r32 = None
            if pk in ship_keys:
                sp = fs.get_tensor(pk)
                r128 = (bool(torch.equal(my128, sp)),
                        round((my128 == sp).float().mean().item(), 4))
            if pk in g32_keys:
                gp = fg.get_tensor(pk)
                r32 = (bool(torch.equal(my32, gp)),
                       round((my32 == gp).float().mean().item(), 4))
            # also the fake-quant relative error at each gs (forward-relevant signal)
            e128 = (W.float() - fake_quant(W, 128)).norm() / W.float().norm()
            e32 = (W.float() - fake_quant(W, 32)).norm() / W.float().norm()
            print(f"    {nm.split('language_model.')[-1]:42s} "
                  f"g128_vs_ship={r128} g32_vs_official={r32} "
                  f"relerr_g128={e128:.4f} relerr_g32={e32:.4f}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[smoke] DONE peak={peak:.1f}GB total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
