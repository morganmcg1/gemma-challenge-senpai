#!/usr/bin/env python
"""Smoke: load on-disk g128 in transformers (decompressed bf16), run a text forward,
confirm target Linear weights are dense bf16 and writable. NO disk writes."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # inherited =7 is stale; force the assigned A10G
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

CKPT = "/workspace/gemma_build/int4_g128_lmhead"
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
cfg = AutoConfig.from_pretrained(CKPT)
print("config loaded; arch:", cfg.architectures, flush=True)

tok = AutoTokenizer.from_pretrained(CKPT)
# Load DECOMPRESSED to dense bf16 fake-quant (run_compressed=False): target Linears
# get a real dense .weight == dequant_g128(packed) == served-grid weight. This is the
# anchor; Route-B overwrites target .weight in place. The HF-vs-Marlin kernel diff
# CANCELS in the anchor-vs-RouteB differential (both use this same forward).
from transformers.utils.quantization_config import CompressedTensorsConfig
model = AutoModelForCausalLM.from_pretrained(
    CKPT, dtype=torch.bfloat16,
    quantization_config=CompressedTensorsConfig(run_compressed=False))
model.eval().to("cuda:0")
# lm_head produces the logits whose argmax defines greedy identity — verify it is dense.
lmh = model.get_submodule("lm_head") if any(n == "lm_head" for n, _ in model.named_modules()) else None
if lmh is not None:
    lw = getattr(lmh, "weight", None)
    print(f"[lm_head] {type(lmh).__name__} dense_weight={None if lw is None else tuple(lw.shape)} "
          f"dtype={None if lw is None else lw.dtype}", flush=True)
print(f"[load] {time.time()-t0:.1f}s  type={type(model).__name__}", flush=True)
print(f"[mem] alloc={torch.cuda.memory_allocated()/1e9:.2f}GB reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

# Find a target module and inspect its weight.
target = "model.language_model.layers.0.self_attn.q_proj"
mod = model.get_submodule(target)
print(f"[module] {target} -> {type(mod).__name__}", flush=True)
for n, p in mod.named_parameters(recurse=False):
    print(f"   param {n}: shape={tuple(p.shape)} dtype={p.dtype} device={p.device} requires_grad={p.requires_grad}")
for n, b in mod.named_buffers(recurse=False):
    print(f"   buffer {n}: shape={tuple(b.shape)} dtype={b.dtype}")

# Text-only forward on a tiny prompt. tf 5.9 apply_chat_template returns a BatchEncoding;
# use return_dict and splat so multimodal forward sees a real input_ids tensor.
enc = tok.apply_chat_template([{"role": "user", "content": "What is 2+2? Answer with a single number."}],
                              add_generation_prompt=True, tokenize=True, return_tensors="pt",
                              return_dict=True)
enc = {k: (v.to(0) if hasattr(v, "to") else v) for k, v in enc.items()}
with torch.no_grad():
    out = model(**enc)
logits = out.logits[0, -1]
top = torch.topk(logits.float(), 5)
print("[forward] last-token top5:", [(int(i), tok.decode([int(i)]), round(float(v), 3))
                                     for v, i in zip(top.values, top.indices)], flush=True)
print(f"[mem] peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
print("SMOKE_OK")
