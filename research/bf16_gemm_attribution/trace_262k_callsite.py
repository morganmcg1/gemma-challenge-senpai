"""Pin the exact call site of the bf16 262144-wide GEMM in the int4head decode
path. The module-forward hook only caught embed_tokens as M=1 gathers, so the
262k GEMM is a *functional* matmul. This wraps torch.mm / addmm / matmul /
F.linear and records a Python traceback whenever an op touches a 262144 dim,
deduped by call site.

Run under the server venv (same env as attribute_modules.py)."""
from __future__ import annotations

import collections
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import paths  # noqa: E402

OUT = ROOT / "research" / "bf16_gemm_attribution" / "callsite_262k.json"
VOCAB = 262144


def _dims(t):
    try:
        return tuple(int(x) for x in t.shape)
    except Exception:
        return None


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[cs] {note}", flush=True)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model_id = os.environ.get("MODEL_ID", "/workspace/gemma_build/int4_g32_lmhead")
    drafter = os.environ.get("DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6"))

    import torch  # noqa: E402
    import torch.nn.functional as F  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    hits = collections.defaultdict(lambda: {"count": 0, "shapes": collections.Counter(), "dtype": None, "stack": None})

    def involves_vocab(*tensors):
        for t in tensors:
            d = _dims(t) if hasattr(t, "shape") else None
            if d and VOCAB in d:
                return True
        return False

    def record(op, a, b, out):
        # filter to the heavy bf16 full-vocab GEMM: output last dim == VOCAB
        od = _dims(out)
        if not od or od[-1] != VOCAB:
            return
        # K of contraction & M
        ad = _dims(a)
        bd = _dims(b)
        key = (op, ad, bd, od)
        rec = hits[key]
        rec["count"] += 1
        rec["shapes"][f"a={ad} b={bd} out={od}"] += 1
        try:
            rec["dtype"] = str(out.dtype)
        except Exception:
            pass
        if rec["stack"] is None:
            # trim to the interesting frames (drop torch internals + this wrapper)
            st = traceback.extract_stack()
            frames = [f"{Path(fr.filename).name}:{fr.lineno} {fr.name}" for fr in st]
            rec["stack"] = [f for f in frames if "site-packages/torch" not in f][-18:]

    _mm = torch.mm
    _addmm = torch.addmm
    _matmul = torch.matmul
    _linear = F.linear

    def mm(a, b, *args, **kw):
        out = _mm(a, b, *args, **kw)
        try: record("torch.mm", a, b, out)
        except Exception: pass
        return out

    def addmm(bias, a, b, *args, **kw):
        out = _addmm(bias, a, b, *args, **kw)
        try: record("torch.addmm", a, b, out)
        except Exception: pass
        return out

    def matmul(a, b, *args, **kw):
        out = _matmul(a, b, *args, **kw)
        try: record("torch.matmul", a, b, out)
        except Exception: pass
        return out

    def linear(x, w, bias=None):
        out = _linear(x, w, bias)
        try: record("F.linear", x, w, out)
        except Exception: pass
        return out

    print(f"[cs] building LLM (eager, uniproc, K={num_spec}) ...", flush=True)
    llm = LLM(
        model=model_id, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.90, max_num_batched_tokens=512, max_num_seqs=1,
        trust_remote_code=True, enforce_eager=True,
        speculative_config={"model": drafter, "num_speculative_tokens": num_spec},
    )

    # inspect target lm_head module type (int4 vs bf16)
    try:
        runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
        m = runner.model
        for nm, mod in m.named_modules():
            if nm.endswith("lm_head") or nm.endswith("logits_processor"):
                w = getattr(mod, "weight", None)
                wp = getattr(mod, "weight_packed", None)
                print(f"[cs] target module {nm}: cls={type(mod).__name__} "
                      f"weight={(tuple(w.shape),str(w.dtype)) if w is not None else None} "
                      f"has_packed={wp is not None}", flush=True)
    except Exception as e:
        print(f"[cs] target lm_head introspect failed: {e}", flush=True)

    # install patches AFTER load so warmup/capture noise is excluded
    torch.mm, torch.addmm, torch.matmul, F.linear = mm, addmm, matmul, linear
    print("[cs] patches installed; running 24-tok generate ...", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=24, seed=1)
    _ = llm.generate(["Explain why the sky is blue in one sentence."], sp)
    torch.mm, torch.addmm, torch.matmul, F.linear = _mm, _addmm, _matmul, _linear

    out = []
    for (op, ad, bd, od), rec in sorted(hits.items(), key=lambda kv: -kv[1]["count"]):
        out.append({"op": op, "a": list(ad) if ad else None, "b": list(bd) if bd else None,
                    "out": list(od) if od else None, "count": rec["count"],
                    "dtype": rec["dtype"], "stack": rec["stack"]})
    print("\n[cs] === full-vocab (out[-1]==262144) GEMM call sites ===", flush=True)
    for r in out:
        print(f"\n  {r['op']}  a={r['a']} b={r['b']} out={r['out']} dtype={r['dtype']} count={r['count']}", flush=True)
        for fr in (r["stack"] or []):
            print(f"      {fr}", flush=True)
    if not out:
        print("  (no full-vocab GEMM observed in eager generate)", flush=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[cs] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
