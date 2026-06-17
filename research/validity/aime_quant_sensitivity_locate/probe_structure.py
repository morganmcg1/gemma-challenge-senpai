#!/usr/bin/env python
"""Structure probe: how does the int4-QAT compressed-tensors model load, and how
do we reach the 42 decoder layers? Resolves the weight-swap strategy before
building the profiler. CPU-only (no GPU OOM); inspects module types + param names.
"""
from __future__ import annotations

import sys
import torch
from transformers import AutoModelForCausalLM, AutoConfig

BF16 = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187"
INT4 = "google/gemma-4-E4B-it-qat-w4a16-ct"


def describe(tag, model):
    print(f"\n===== {tag} :: {type(model).__name__} =====", flush=True)
    # find the decoder layer container
    paths = [
        "model.layers", "model.model.layers", "language_model.model.layers",
        "model.language_model.layers", "language_model.layers",
    ]
    layers = None
    found = None
    for p in paths:
        obj = model
        ok = True
        for part in p.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            layers = obj
            found = p
            break
    print(f"decoder layers at: {found}  count={len(layers) if layers is not None else 'N/A'}", flush=True)
    if layers is None:
        # dump top-level children
        for n, _ in model.named_children():
            print("  child:", n, flush=True)
        return
    l0 = layers[0]
    print(f"layer[0] type: {type(l0).__name__}", flush=True)
    print("layer[0] named_modules with params:", flush=True)
    for name, mod in l0.named_modules():
        pkeys = [pn for pn, _ in mod.named_parameters(recurse=False)]
        bkeys = [bn for bn, _ in mod.named_buffers(recurse=False)]
        if pkeys or bkeys:
            info = []
            for pn, pp in mod.named_parameters(recurse=False):
                info.append(f"{pn}:{tuple(pp.shape)}:{pp.dtype}")
            for bn, bb in mod.named_buffers(recurse=False):
                info.append(f"[buf]{bn}:{tuple(bb.shape)}:{bb.dtype}")
            print(f"  {name} ({type(mod).__name__}): {info}", flush=True)
    # total params in layer 0
    np_ = sum(p.numel() for p in l0.parameters())
    print(f"layer[0] total params: {np_:,}", flush=True)


def main():
    cfg = AutoConfig.from_pretrained(INT4, trust_remote_code=True)
    print("config class:", type(cfg).__name__, flush=True)
    tc = getattr(cfg, "text_config", cfg)
    print("num_hidden_layers:", getattr(tc, "num_hidden_layers", getattr(cfg, "num_hidden_layers", "?")), flush=True)
    print("quantization_config present:", hasattr(cfg, "quantization_config"), flush=True)
    qc = getattr(cfg, "quantization_config", None)
    if qc is not None:
        print("quant cfg keys:", list(qc.keys()) if isinstance(qc, dict) else qc, flush=True)

    print("\nLoading int4 (CPU, bf16) ...", flush=True)
    m4 = AutoModelForCausalLM.from_pretrained(INT4, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    describe("INT4", m4)
    del m4

    print("\nLoading bf16 base (CPU, bf16) ...", flush=True)
    mb = AutoModelForCausalLM.from_pretrained(BF16, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    describe("BF16", mb)
    del mb
    return 0


if __name__ == "__main__":
    sys.exit(main())
