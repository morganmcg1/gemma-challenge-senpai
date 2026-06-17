#!/usr/bin/env python
"""Probe: does run_compressed=False decompress the int4-QAT checkpoint to dense
bf16 Linear.weight (= the library-exact dequant of the served int4 weights), so
per-layer bf16 weight-swaps are trivial? Also: finite forward + logit softcap.
"""
from __future__ import annotations
import sys, torch
from transformers import AutoModelForCausalLM

INT4 = "google/gemma-4-E4B-it-qat-w4a16-ct"

def main():
    try:
        from transformers import CompressedTensorsConfig
        qc = CompressedTensorsConfig(run_compressed=False)
        print("CompressedTensorsConfig importable; run_compressed=False", flush=True)
    except Exception as e:
        print("no CompressedTensorsConfig:", e, flush=True)
        qc = None

    m = AutoModelForCausalLM.from_pretrained(
        INT4, dtype=torch.bfloat16, trust_remote_code=True,
        **({"quantization_config": qc} if qc is not None else {}),
    )
    layer0 = m.model.language_model.layers[0]
    q = layer0.self_attn.q_proj
    pk = [pn for pn, _ in q.named_parameters(recurse=False)]
    print("q_proj param names:", pk, "| type:", type(q).__name__, flush=True)
    w = getattr(q, "weight", None)
    if w is not None:
        print(f"DENSE weight: shape={tuple(w.shape)} dtype={w.dtype} "
              f"finite={torch.isfinite(w).all().item()} "
              f"absmax={w.abs().max().item():.4f}", flush=True)
    else:
        print("NO dense weight -> still compressed", flush=True)

    # tiny GPU forward
    m = m.to("cuda:0").eval()
    ids = torch.tensor([[2, 105, 2364, 107, 1841, 603, 235248, 235274, 235340]], device="cuda:0")
    with torch.inference_mode():
        out = m(input_ids=ids)
    lg = out.logits
    print(f"forward logits: shape={tuple(lg.shape)} finite={torch.isfinite(lg).all().item()} "
          f"absmax={lg.abs().max().item():.4f} (softcap=30 => absmax<=30)", flush=True)
    print("argmax last:", int(lg[0, -1].argmax().item()), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
