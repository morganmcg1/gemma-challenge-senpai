"""Definitive owner of the bf16 262144-wide aten::mm. Python-level torch.mm/
matmul/F.linear patching missed it (dispatched below the Python binding), so use
torch.profiler with record_shapes + with_stack: the dispatcher sees every
aten::mm regardless of how it was called, and with_stack gives the owning Python
frames.

Run under the server venv (same env as attribute_modules.py)."""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import paths  # noqa: E402

OUT = ROOT / "research" / "bf16_gemm_attribution" / "shapes_stack.json"
VOCAB = 262144


def shape_has(shapes, v):
    for s in shapes or []:
        if isinstance(s, (list, tuple)) and v in s:
            return True
    return False


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[ps] {note}", flush=True)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model_id = os.environ.get("MODEL_ID", "/workspace/gemma_build/int4_g32_lmhead")
    drafter = os.environ.get("DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6"))

    import torch  # noqa: E402
    from torch.profiler import profile, ProfilerActivity  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    enforce_eager = os.environ.get("ENFORCE_EAGER", "0") == "1"
    print(f"[ps] building LLM (enforce_eager={enforce_eager}, uniproc, K={num_spec}) ...", flush=True)
    llm = LLM(
        model=model_id, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.90, max_num_batched_tokens=512, max_num_seqs=1,
        trust_remote_code=True, enforce_eager=enforce_eager,
        speculative_config={"model": drafter, "num_speculative_tokens": num_spec},
    )

    sp = SamplingParams(temperature=0.0, max_tokens=40, seed=1)
    # warm up (exclude prefill/JIT noise from the profiled window as much as we can)
    _ = llm.generate(["Hello there, tell me about gravity."], sp)

    print("[ps] profiling 40-tok generate with record_shapes + with_stack ...", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True, with_modules=True) as prof:
        _ = llm.generate(["Explain why the sky is blue in one detailed paragraph."], sp)

    # Iterate raw events: find mm-family ops whose input shapes carry VOCAB.
    mm_names = {"aten::mm", "aten::addmm", "aten::matmul", "aten::bmm", "aten::linear"}
    by_shape = collections.defaultdict(lambda: {"count": 0, "name": None, "stack": None})
    allmm = collections.Counter()
    for ev in prof.events():
        nm = getattr(ev, "name", "")
        if nm not in mm_names:
            continue
        shapes = getattr(ev, "input_shapes", None)
        allmm[(nm, tuple(tuple(s) for s in (shapes or []) if isinstance(s, (list, tuple))))] += 1
        if not shape_has(shapes, VOCAB):
            continue
        key = (nm, tuple(tuple(s) for s in shapes if isinstance(s, (list, tuple))))
        rec = by_shape[key]
        rec["count"] += 1
        rec["name"] = nm
        if rec["stack"] is None:
            st = getattr(ev, "stack", None)
            if st:
                rec["stack"] = [f for f in st if "site-packages/torch/" not in f][-20:]

    print("\n[ps] === aten mm-family ops with a 262144 input dim ===", flush=True)
    out = []
    for (nm, shapes), rec in sorted(by_shape.items(), key=lambda kv: -kv[1]["count"]):
        print(f"\n  {nm}  shapes={list(shapes)}  count={rec['count']}", flush=True)
        for fr in (rec["stack"] or []):
            print(f"      {fr}", flush=True)
        out.append({"name": nm, "shapes": [list(s) for s in shapes], "count": rec["count"], "stack": rec["stack"]})
    if not out:
        print("  (none — 262144 op is not an aten mm-family op; will dump top mm shapes)", flush=True)

    # cross-check: top mm-family shapes by count (to find the ~40x verify-width GEMM)
    print("\n[ps] === top 15 mm-family (name, shapes) by count ===", flush=True)
    top = []
    for (nm, shapes), c in allmm.most_common(15):
        print(f"   x{c:4d}  {nm}  {list(shapes)}", flush=True)
        top.append({"name": nm, "shapes": [list(s) for s in shapes], "count": c})

    OUT.write_text(json.dumps({"vocab_hits": out, "top_mm_shapes": top}, indent=2, default=str))
    print(f"\n[ps] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
