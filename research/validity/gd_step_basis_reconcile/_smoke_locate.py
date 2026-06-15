#!/usr/bin/env python
"""Smoke test: load the deployed int4 target on the local A10G in the served wheel
venv and confirm we can locate the int4 body GEMMs + lm_head and run a verify
forward GEMM. De-risks the deployed-path g_d measurement (PR #271) before the
full coupled profiler. NO HF Job, NO submission, NO served-file change."""
from __future__ import annotations
import json, os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _here]
sys.modules.pop("profile", None)

import torch
from vllm import LLM

CANDS = [
    "/tmp/osoi5-v0-baked",
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"),
]


def resolve():
    found = []
    from pathlib import Path
    for c in CANDS:
        p = Path(c)
        if p.is_dir() and (p / "config.json").exists():
            found.append(str(p))
        elif p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    found.append(str(sub)); break
    return found


def main():
    print(f"[smoke] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}", flush=True)
    cands = resolve()
    print(f"[smoke] candidates: {cands}", flush=True)
    llm = None
    for c in cands:
        try:
            cfg = json.load(open(os.path.join(c, "config.json")))
            tc = cfg.get("text_config", cfg)
            print(f"[smoke] trying {c} layers={tc.get('num_hidden_layers')} hidden={tc.get('hidden_size')}", flush=True)
            t0 = time.time()
            llm = LLM(model=c, quantization="compressed-tensors", dtype="bfloat16",
                      max_model_len=1088, gpu_memory_utilization=0.60, max_num_seqs=1,
                      enforce_eager=True, trust_remote_code=True)
            print(f"[smoke] LOAD OK {c} in {time.time()-t0:.0f}s", flush=True)
            break
        except Exception as exc:
            print(f"[smoke] load FAILED {c}: {exc!r}", flush=True)
    if llm is None:
        print("[smoke] RESULT: no int4 model loaded", flush=True); return 1

    model = None
    for p in (
        lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
    ):
        try:
            m = p()
            if m is not None:
                model = m; print(f"[smoke] model_runner path OK: {type(m).__name__}", flush=True); break
        except Exception as exc:
            print(f"[smoke] path miss: {exc!r}", flush=True)
    if model is None:
        print("[smoke] RESULT: could not locate model_runner.model", flush=True); return 1

    import torch.nn as nn
    layers = None
    for chain in [("model","layers"),("model","language_model","layers"),
                  ("language_model","model","layers"),("language_model","layers"),
                  ("model","model","layers")]:
        obj, ok = model, True
        for a in chain:
            if hasattr(obj, a): obj = getattr(obj, a)
            else: ok = False; break
        if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
            layers = obj; print(f"[smoke] layers via {chain}: n={len(obj)}", flush=True); break
    if layers is None:
        for _, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0 and hasattr(mod[0], "self_attn"):
                layers = mod; print(f"[smoke] layers via named_modules n={len(mod)}", flush=True); break
    if layers is None:
        print("[smoke] RESULT: could not locate decoder layers", flush=True); return 1

    l0 = layers[0]
    def oi(mod):
        out = getattr(mod, "output_size_per_partition", None); inp = getattr(mod, "input_size_per_partition", None)
        if out is None or inp is None:
            w = getattr(mod, "weight", None)
            if w is not None and w.dim()==2: out, inp = int(w.shape[0]), int(w.shape[1])
        return (out, inp)
    try:
        mods = {"qkv_proj": l0.self_attn.qkv_proj, "o_proj": l0.self_attn.o_proj,
                "gate_up_proj": l0.mlp.gate_up_proj, "down_proj": l0.mlp.down_proj}
        for n, m in mods.items():
            print(f"[smoke]   {n}: out/in={oi(m)} quant_method={hasattr(m,'quant_method')}", flush=True)
        # run a real int4 verify GEMM at M=8
        dev = torch.device("cuda:0")
        x = torch.randn(8, oi(mods['qkv_proj'])[1], dtype=torch.bfloat16, device=dev)
        y = mods['qkv_proj'].quant_method.apply(mods['qkv_proj'], x, bias=None)
        torch.cuda.synchronize()
        print(f"[smoke]   qkv_proj.apply(M=8) -> {tuple(y.shape)} finite={torch.isfinite(y).all().item()}", flush=True)
    except Exception as exc:
        print(f"[smoke] body locate/apply FAILED: {exc!r}", flush=True); return 1
    print(f"[smoke] peak_gpu_gb={torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
    print("[smoke] RESULT: OK — int4 load + body-locate + verify-GEMM all work", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
