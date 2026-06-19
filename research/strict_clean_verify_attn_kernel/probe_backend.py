#!/usr/bin/env python
"""PR #747 wirbel — probe which attention backend the served int4 stack selects
under VLLM_BATCH_INVARIANT=1, and dump per-layer attention config (heads, kv
heads, head_dim, scale, sliding window, backend impl class). ANALYSIS ONLY.

Also confirms the stack loads + does a 1-token greedy generation, so this is a
legitimate model-loading smoke test on the assigned local A10G.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
# Local CUDA toolkit lacks curand.h -> flashinfer sampler JIT fails. We don't
# need the sampler for an attention-kernel measurement; use the native sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/workspace/gemma_build/int4_g128_lmhead/model.safetensors"
HERE = Path(__file__).resolve().parent


def main() -> int:
    model_dir = str(Path(MODEL).parent)
    print(f"[probe] VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT')}")
    print(f"[probe] loading {model_dir} enforce_eager=True ...", flush=True)
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Introspect the running model's attention layers.
    eng = llm.llm_engine
    try:
        model = eng.model_executor.driver_worker.worker.model_runner.model
    except AttributeError:
        # v1 engine path
        model = (
            eng.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model
        )

    info = []
    seen = set()
    for name, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if impl is None:
            continue
        rec = {
            "module": name,
            "module_cls": type(mod).__name__,
            "impl_cls": type(impl).__name__,
            "impl_mod": type(impl).__module__,
        }
        for attr in (
            "num_heads", "num_kv_heads", "head_size", "scale",
            "sliding_window", "logits_soft_cap", "kv_cache_dtype",
            "use_irope", "attn_type",
        ):
            if hasattr(impl, attr):
                rec[attr] = str(getattr(impl, attr))
        # backend class on the module
        be = getattr(mod, "backend", None)
        if be is not None:
            rec["backend"] = str(be)
        key = (rec["impl_cls"], rec.get("sliding_window"))
        if key not in seen:
            seen.add(key)
            info.append(rec)
        if len(info) >= 6:
            break

    print("[probe] === distinct attention impls ===")
    for rec in info:
        print(json.dumps(rec, indent=2))

    # tiny greedy smoke
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    out = llm.generate(["The capital of France is"], sp)
    print("[probe] greedy out:", repr(out[0].outputs[0].text))
    print("[probe] token_ids:", out[0].outputs[0].token_ids)

    (HERE / "probe_backend_report.json").write_text(
        json.dumps({"impls": info, "device": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0))}, indent=2)
    )
    print("[probe] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
