"""EAGLE-3 feature-export probe for google/gemma-4-E4B-it on vLLM 0.22.0.

Read-only feasibility probe (PR #15). Confirms on the REAL GPU-loaded model that
the multi-layer auxiliary hidden states EAGLE-3 needs are accessible via vLLM's
built-in SupportsEagle3 interface, with NO model-class surgery.

It does not launch an HF Job. It performs a single local model load on the
assigned A10G and exercises the interface via LLM.apply_model (runs inside the
worker on the real instantiated module).

Usage (from target/):
  HF_HOME=/senpai-run/home/student-fern/.cache/huggingface \
  /tmp/server-venv/bin/python research/eagle3_feasibility/probe_eagle3_export.py \
      2>&1 | tee research/eagle3_feasibility/probe.log
"""

import json
import os

# apply_model ships the probe fn across the V1 engine-core IPC boundary; allow
# the cloudpickle fallback so a __main__-defined function serializes by value.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3

MODEL = os.environ.get("PROBE_MODEL", "google/gemma-4-E4B-it")
OUT_JSON = os.path.join(os.path.dirname(__file__), "probe_result.json")


def probe(model):
    """Runs inside the vLLM worker on the real instantiated model module."""
    out = {}
    out["model_class"] = type(model).__name__
    out["supports_eagle3"] = bool(supports_eagle3(model))

    default_layers = tuple(model.get_eagle3_default_aux_hidden_state_layers())
    out["default_aux_layers"] = list(default_layers)

    # Locate the inner text model (the EagleModelMixin that collects aux states).
    lm = getattr(model, "language_model", None)
    inner = getattr(lm, "model", None) if lm is not None else None
    out["inner_model_class"] = type(inner).__name__ if inner is not None else None
    out["inner_is_EagleModelMixin"] = isinstance(inner, EagleModelMixin)
    out["num_decoder_layers"] = len(inner.layers) if inner is not None else None

    # Configure aux layers through the public interface, confirm the mutation
    # lands on the inner model's aux_hidden_state_layers tuple.
    model.set_aux_hidden_state_layers(default_layers)
    out["inner_aux_layers_after_set"] = list(
        getattr(inner, "aux_hidden_state_layers", ())
    )

    # Hidden size (Gemma-4 E4B text body).
    H = None
    try:
        H = int(model.config.text_config.hidden_size)
    except Exception:
        H = getattr(getattr(inner, "config", None), "hidden_size", None)
    out["hidden_size"] = H

    # Exercise the real inner model's _maybe_add_hidden_state collect logic with
    # synthetic residual-stream tensors, emulating the per-layer forward loop.
    if inner is not None and H:
        dev = next(inner.parameters()).device
        dt = next(inner.parameters()).dtype
        T = 5
        hs = torch.randn(T, H, device=dev, dtype=dt)
        res = torch.randn(T, H, device=dev, dtype=dt)
        aux = []
        for li in range(len(inner.layers) + 1):
            inner._maybe_add_hidden_state(aux, li, hs, res)
        out["synthetic_num_aux_collected"] = len(aux)
        out["synthetic_aux_shapes"] = [list(a.shape) for a in aux]
        out["synthetic_any_nan"] = any(bool(torch.isnan(a).any()) for a in aux)

    # Confirm the full multimodal towers are present (must stay complete).
    out["has_vision_tower"] = hasattr(model, "vision_tower") and (
        getattr(model, "vision_tower", None) is not None
    )
    out["has_audio_tower"] = getattr(model, "audio_tower", None) is not None
    return out


def main():
    print(f"[probe] loading {MODEL} (bf16, enforce_eager) ...", flush=True)
    llm = LLM(
        MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=float(os.environ.get("PROBE_GPU_UTIL", "0.85")),
        enforce_eager=True,
        max_model_len=2048,
    )

    results = llm.apply_model(probe)
    # apply_model returns one entry per TP rank; TP=1 -> single entry.
    result = results[0] if isinstance(results, list) else results

    print("PROBE_RESULT_JSON_START", flush=True)
    print(json.dumps(result, indent=2, default=str), flush=True)
    print("PROBE_RESULT_JSON_END", flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Sanity greedy decode: confirm the complete model produces sane output.
    gen = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    text = gen[0].outputs[0].text
    print("GEN_OUTPUT:", repr(text), flush=True)
    print("[probe] done.", flush=True)


if __name__ == "__main__":
    main()
