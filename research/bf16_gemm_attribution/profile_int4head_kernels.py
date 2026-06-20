"""DECISIVE Step-1 experiment for PR #812.

The saved #809 trace (research/cudagraph_sampling_capture/) was run on
`gemma-4-E4B-it-qat-w4a16-ct` -> that checkpoint keeps a *bf16 tied lm_head*.
Its 17% bf16 slice (aten::mm x59 / ampere_bf16_s16816gemm x40, grid=(4096,1,1)
=> N=262144 full vocab) is the bf16 lm_head full-vocab logits GEMM. On the
ACTUAL int4head (`int4_g32_lmhead`, lm_head quantized to int4 g32) that GEMM
should be served by Marlin, not ampere_bf16.

This profiles the real int4head decode (graphs ON, uniproc, K=6 spec) with
torch.profiler CUPTI self_device_time (sees graph-internal kernels) and reports
whether the ampere_bf16 full-vocab GEMM survives or is replaced by Marlin.

Run under the server venv."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import paths  # noqa: E402

OUT = ROOT / "research" / "bf16_gemm_attribution" / "int4head_kernels.json"


def categorize(name: str) -> str:
    n = name.lower()
    if "marlin" in n or "gemv" in n or "gemm" in n:
        return "matmul_gemm"
    if any(s in n for s in ("attn", "_fwd", "flash", "paged", "unified_attention",
                            "reshape_and_cache", "rotary", "rope")):
        return "attention"
    if any(s in n for s in ("log_softmax", "argmax", "topk", "softmax", "sample",
                            "logits", "cumsum", "sort")):
        return "sampling_lmhead"
    if any(s in n for s in ("rms", "layernorm", "layer_norm", "norm_kernel")):
        return "norm"
    if any(s in n for s in ("silu", "gelu", "swiglu", "act_and_mul")):
        return "activation"
    if any(s in n for s in ("elementwise", "copy", "cast", "convert", "memcpy",
                            "fill", "mul", "index", "vectorized")):
        return "elementwise_copy"
    return "other"


def self_dev(e) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[k] {note}", flush=True)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model_id = os.environ.get("MODEL_ID", "/workspace/gemma_build/int4_g32_lmhead")
    drafter = os.environ.get("DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6"))
    prof_tokens = int(os.environ.get("PROFILE_TOKENS", "192"))

    import torch  # noqa: E402
    from torch.profiler import profile, ProfilerActivity  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    print(f"[k] building int4head LLM (graphs ON, uniproc, K={num_spec}), model={model_id}", flush=True)
    llm = LLM(
        model=model_id, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.90, max_num_batched_tokens=512, max_num_seqs=1,
        trust_remote_code=True, enforce_eager=False, disable_log_stats=True,
        speculative_config={"model": drafter, "num_speculative_tokens": num_spec},
    )

    # confirm lm_head is int4 (has packed weights) on this checkpoint
    try:
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        for nm, mod in runner.model.named_modules():
            if nm.endswith("lm_head"):
                w = getattr(mod, "weight", None)
                wp = getattr(mod, "weight_packed", None)
                print(f"[k] lm_head {nm}: cls={type(mod).__name__} "
                      f"weight={(tuple(w.shape),str(w.dtype)) if w is not None else None} "
                      f"has_packed={wp is not None}", flush=True)
    except Exception as e:
        print(f"[k] lm_head introspect failed: {e}", flush=True)

    sp = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, seed=1)
    _ = llm.generate(["Hello there, tell me about gravity."], sp(16))
    torch.cuda.synchronize()

    print(f"[k] profiling {prof_tokens}-tok decode (CUPTI self-device) ...", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _ = llm.generate(["Explain why the sky is blue in one detailed paragraph."], sp(prof_tokens))
        torch.cuda.synchronize()

    rows = [(e.key, self_dev(e), int(getattr(e, "count", 0))) for e in prof.key_averages()]
    rows = [r for r in rows if r[1] > 0]
    rows.sort(key=lambda r: r[1], reverse=True)
    busy_us = sum(r[1] for r in rows)
    cats: dict[str, float] = {}
    for name, us, _ in rows:
        cats[categorize(name)] = cats.get(categorize(name), 0.0) + us

    def total_matching(sub):
        return sum(us for nm, us, _ in rows if sub in nm.lower())
    amp_bf16 = total_matching("ampere_bf16_s16816gemm")
    marlin = total_matching("marlin")
    aten_mm = total_matching("aten::mm")

    print("\n==== int4head GRAPH-MODE PROFILE (self-device) ====", flush=True)
    print(f"  GPU-busy: {busy_us/1000:.1f} ms", flush=True)
    print("  --- by category ---", flush=True)
    for c, us in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:18s} {us/1000:9.2f} ms  {100*us/busy_us:5.1f}%", flush=True)
    print("  --- KEY: bf16 vs marlin full-vocab GEMM ---", flush=True)
    print(f"    ampere_bf16_s16816gemm : {amp_bf16/1000:8.2f} ms  {100*amp_bf16/busy_us:5.1f}%", flush=True)
    print(f"    marlin (all)           : {marlin/1000:8.2f} ms  {100*marlin/busy_us:5.1f}%", flush=True)
    print(f"    aten::mm (host op)      : {aten_mm/1000:8.2f} ms  {100*aten_mm/busy_us:5.1f}%", flush=True)
    print("  --- top 20 kernels ---", flush=True)
    for name, us, cnt in rows[:20]:
        print(f"    {100*us/busy_us:5.1f}% {us/1000:8.2f}ms x{cnt:<6d} [{categorize(name)}] {name[:66]}", flush=True)

    OUT.write_text(json.dumps({
        "model": model_id, "gpu_busy_ms": busy_us / 1000,
        "category_ms": {k: v / 1000 for k, v in cats.items()},
        "category_pct": {k: 100 * v / busy_us for k, v in cats.items()},
        "ampere_bf16_ms": amp_bf16 / 1000, "ampere_bf16_pct": 100 * amp_bf16 / busy_us,
        "marlin_ms": marlin / 1000, "marlin_pct": 100 * marlin / busy_us,
        "aten_mm_ms": aten_mm / 1000,
        "top_kernels": [{"kernel": nm, "ms": us / 1000, "count": c,
                         "pct": 100 * us / busy_us, "category": categorize(nm)}
                        for nm, us, c in rows[:25]],
    }, indent=2, default=str))
    print(f"\n[k] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
