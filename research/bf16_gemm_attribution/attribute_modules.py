"""Step-1 attribution for PR #812: which module owns the 17% bf16 `aten::mm`
(40x ampere_bf16_s16816gemm_64x64, N=262144 full-vocab, M~7) in the int4head
decode path?

The #809 trace already proves the bf16 GEMM is a full-vocab (262144-wide) bf16
matmul at the verify width (M~7), ~2.8 ms/call, ~once per decode step. The
target lm_head is int4-Marlin in int4head, and embed_tokens stays bf16. So the
bf16 262k GEMM must be a SEPARATE head -- almost certainly the gemma4_mtp
drafter's bf16 logits projection (tied to bf16 embed_tokens). This loads the
exact int4head stack in-process (uniproc, eager) and settles the owner
structurally + at runtime.

Run under the server venv from repo root with the submission dir on PYTHONPATH
(so its sitecustomize attention patches apply):

  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=submissions/int4_mtp_bi0_int4head:. \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  /tmp/senpai-venvs/20f658587e8a6643/bin/python \
    research/bf16_gemm_attribution/attribute_modules.py
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.local_validation import paths  # noqa: E402

OUT = ROOT / "research" / "bf16_gemm_attribution" / "module_attribution.json"
VOCAB = 262144


def scope_of(name: str) -> str:
    """Classify a qualified module/param name into a PR #812 owner bucket."""
    n = name.lower()
    if any(k in n for k in ("drafter", "proposer", "mtp", "eagle", "assistant")):
        return "drafter"
    if "lm_head" in n or "logits" in n or "unembed" in n:
        return "lm_head"
    if "embed_tokens_per_layer" in n or "per_layer" in n or "ple" in n:
        return "PLE"
    if "embed_tokens" in n or "embedding" in n:
        return "embeddings"
    if "altup" in n or "laurel" in n:
        return "altup_laurel"
    return "body"


def dump_tree(root, tag, sink):
    """Record every parameter-bearing leaf module: qualified name, class,
    weight dtype/shape, whether it carries packed int4 weights (Marlin) or a
    dense bf16 weight, and its declared vLLM `prefix`."""
    import torch

    for name, mod in root.named_modules():
        cls = type(mod).__name__
        prefix = getattr(mod, "prefix", None)
        # Direct params on this module only (recurse=False) so we attribute to leaves.
        params = dict(mod.named_parameters(recurse=False))
        if not params:
            continue
        has_packed = "weight_packed" in params
        w = params.get("weight")
        rec = {
            "tree": tag,
            "name": name,
            "cls": cls,
            "prefix": prefix,
            "scope": scope_of(prefix or name),
            "has_weight_packed": has_packed,
            "param_names": sorted(params.keys()),
        }
        if w is not None and isinstance(w, torch.Tensor):
            rec["weight_dtype"] = str(w.dtype)
            rec["weight_shape"] = list(w.shape)
            rec["is_bf16_dense"] = (w.dtype == torch.bfloat16) and not has_packed
            rec["is_vocab_wide"] = VOCAB in tuple(w.shape)
        if has_packed:
            ps = params["weight_packed"]
            rec["packed_shape"] = list(ps.shape)
            shp = params.get("weight_shape")
            if shp is not None:
                try:
                    rec["logical_shape"] = [int(x) for x in shp.tolist()]
                except Exception:
                    pass
        sink.append(rec)


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[attr] {note}", flush=True)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    model_id = os.environ.get("MODEL_ID", "gemma-challenge/gemma-4-e4b-it-int4-mtp-bi0-int4head")
    drafter = os.environ.get("DRAFTER_MODEL", "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant")
    num_spec = int(os.environ.get("NUM_SPECULATIVE_TOKENS", "6"))

    import torch  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    print(f"[attr] building LLM model={model_id} drafter={drafter} K={num_spec} (eager, uniproc)", flush=True)
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        trust_remote_code=True,
        enforce_eager=True,
        speculative_config={"model": drafter, "num_speculative_tokens": num_spec},
    )

    # --- locate the model runner + its model containers (target + drafter) ---
    runner = None
    for pathexpr in (
        "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner",
        "llm_engine.model_executor.driver_worker.model_runner",
    ):
        obj = llm
        ok = True
        for attr in pathexpr.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            runner = obj
            print(f"[attr] model_runner via llm.{pathexpr}", flush=True)
            break
    if runner is None:
        print("[attr] WARN could not locate model_runner via known paths; dumping attrs", flush=True)
        print([a for a in dir(llm.llm_engine) if not a.startswith("__")], flush=True)

    tree = []
    id2name = {}
    if runner is not None:
        # Target model
        m = getattr(runner, "model", None)
        if m is not None:
            dump_tree(m, "target", tree)
            for nm, mod in m.named_modules():
                id2name[id(mod)] = ("target", nm)
        # Drafter lives on the proposer/drafter attribute in vLLM V1 spec.
        for dattr in ("drafter", "proposer"):
            d = getattr(runner, dattr, None)
            if d is None:
                continue
            dm = getattr(d, "model", d)
            try:
                dump_tree(dm, f"drafter:{dattr}", tree)
                for nm, mod in dm.named_modules():
                    id2name.setdefault(id(mod), (f"drafter:{dattr}", nm))
            except Exception as e:
                print(f"[attr] drafter dump via {dattr} failed: {e}", flush=True)

    # --- highlight: every bf16 vocab-wide dense weight (the 262k GEMM candidates) ---
    vocab_bf16 = [r for r in tree if r.get("is_bf16_dense") and r.get("is_vocab_wide")]
    print("\n[attr] === bf16 DENSE vocab-wide (262144) weights (262k-GEMM owners) ===", flush=True)
    for r in vocab_bf16:
        print(f"   tree={r['tree']:16s} scope={r['scope']:12s} cls={r['cls']:28s} "
              f"shape={r.get('weight_shape')} name={r['name']} prefix={r['prefix']}", flush=True)
    if not vocab_bf16:
        print("   (none found by static walk -- 262k GEMM may use a tied embed via functional)", flush=True)

    # --- runtime hook: capture which modules actually run, at what M, how often ---
    runtime = collections.defaultdict(lambda: {"calls": 0, "in_rows": collections.Counter()})

    def hook(mod, inp, out):
        try:
            w = getattr(mod, "weight", None)
            packed = hasattr(mod, "weight_packed")
            wshape = list(w.shape) if (w is not None and hasattr(w, "shape")) else None
            wdtype = str(w.dtype) if (w is not None and hasattr(w, "dtype")) else None
            x = inp[0] if (isinstance(inp, tuple) and inp) else None
            rows = None
            if hasattr(x, "shape") and x.dim() >= 2:
                rows = int(x.shape[0]) if x.dim() == 2 else int(x.shape[-2])
            elif hasattr(x, "shape") and x.dim() == 1:
                rows = 1
            tname = id2name.get(id(mod))
            key = (type(mod).__name__, getattr(mod, "prefix", None), tuple(wshape) if wshape else None,
                   wdtype, packed, tname)
            rec = runtime[key]
            rec["calls"] += 1
            if rows is not None:
                rec["in_rows"][rows] += 1
        except Exception:
            pass

    handle = torch.nn.modules.module.register_module_forward_hook(hook)
    print("\n[attr] running short generate (32 tok) with hooks ...", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=32, seed=1)
    _ = llm.generate(["The capital of France is"], sp)
    handle.remove()

    # serialize runtime, focusing on bf16 vocab-wide + the heaviest movers
    rt = []
    for (cls, prefix, wshape, wdtype, packed, tname), rec in runtime.items():
        rt.append({
            "cls": cls, "prefix": prefix, "weight_shape": list(wshape) if wshape else None,
            "weight_dtype": wdtype, "packed": packed,
            "id2name": list(tname) if tname else None,
            "calls": rec["calls"], "in_rows": dict(rec["in_rows"]),
            "scope": scope_of((prefix or (tname[1] if tname else "")) or cls),
            "is_vocab_wide": bool(wshape and VOCAB in wshape),
            "is_bf16_dense": (wdtype == "torch.bfloat16" and not packed),
        })
    rt.sort(key=lambda r: (not (r["is_vocab_wide"] and r["is_bf16_dense"]), -r["calls"]))

    print("\n[attr] === runtime bf16-dense vocab-wide module calls (the 262k GEMM) ===", flush=True)
    for r in rt:
        if r["is_vocab_wide"] and r["is_bf16_dense"]:
            print(f"   scope={r['scope']:12s} cls={r['cls']:26s} shape={r['weight_shape']} "
                  f"calls={r['calls']} in_rows={r['in_rows']} prefix={r['prefix']} name={r['id2name']}",
                  flush=True)

    OUT.write_text(json.dumps({
        "model_id": model_id, "drafter": drafter, "num_spec": num_spec,
        "static_vocab_bf16_dense": vocab_bf16,
        "static_tree_n": len(tree),
        "runtime": rt,
    }, indent=2, default=str))
    print(f"\n[attr] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
