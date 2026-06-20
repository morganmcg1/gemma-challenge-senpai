#!/usr/bin/env python
"""PR #805 Step 2 — serve-verify the PLE-dequant kernel dispatch.

Two independent proofs that the de-quant took effect at RUNTIME:

  (A) STATIC predicate — vLLM's own ``should_ignore_layer`` run against the built
      checkpoint's ``quantization_config.ignore``. This is the exact function
      ``CompressedTensorsConfig.get_scheme`` calls to decide bf16 (return None ->
      UnquantizedLinearMethod / cuBLAS) vs quantized (CompressedTensorsWNA16 /
      Marlin). No GPU, no model load.

  (B) LIVE walk — actually load the built checkpoint through vLLM
      (quantization=compressed-tensors, bf16, the served target config) and walk
      the CONSTRUCTED model via ``LLM.apply_model``, reading each module's real
      ``quant_method`` class and (for quantized Linears) its resolved ``scheme``
      class. This is the constructed-model ground truth, not a prediction.

PASS contract (both proofs must agree):
  per_layer_input_gate  -> UnquantizedLinearMethod        (bf16, cuBLAS)   [TARGET]
  per_layer_projection  -> CompressedTensorsLinearMethod + CompressedTensorsWNA16 (int4, Marlin) [sibling stays int4]
  language-model body   -> CompressedTensorsLinearMethod + CompressedTensorsWNA16 (int4, Marlin)
  lm_head               -> CompressedTensorsLinearMethod + CompressedTensorsWNA16 (int4, Marlin)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BUILD = Path(os.environ.get("BUILD", "/workspace/gemma_build/bi0_int4head_pledequant"))

# Representative checkpoint module prefixes (one per dispatch class we care about).
PROBES = {
    "per_layer_input_gate L0 [TARGET->bf16]": "model.language_model.layers.0.per_layer_input_gate",
    "per_layer_input_gate L20 [TARGET->bf16]": "model.language_model.layers.20.per_layer_input_gate",
    "per_layer_projection L0 [sibling int4]": "model.language_model.layers.0.per_layer_projection",
    "self_attn.o_proj L0 [body int4]": "model.language_model.layers.0.self_attn.o_proj",
    "mlp.down_proj L0 [body int4]": "model.language_model.layers.0.mlp.down_proj",
    "lm_head [head int4]": "lm_head",
}


def static_predicate() -> dict[str, bool]:
    """(A) vLLM's should_ignore_layer against the built ignore list."""
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    qc = json.load(open(BUILD / "config.json"))["quantization_config"]
    ignore = qc["ignore"]
    print("=" * 78)
    print(f"(A) STATIC: should_ignore_layer vs ignore[{len(ignore)}] "
          f"(last entry: {ignore[-1]!r})")
    print(f"    config_groups: "
          + ", ".join(f"{g}={d['targets']} {d['weights']['num_bits']}b/g{d['weights'].get('group_size')}"
                      for g, d in qc["config_groups"].items()))
    print("-" * 78)
    out = {}
    for label, name in PROBES.items():
        ig = bool(should_ignore_layer(name, ignore=ignore, fused_mapping={}))
        disp = "bf16/Unquantized(cuBLAS)" if ig else "int4/WNA16(Marlin)"
        out[label] = ig
        print(f"  {label:40s} ignored={str(ig):5s} -> {disp}")
    return out


# --- (B) live walk (top-level fns so apply_model can ship them to the worker) ---
def _classify(name: str) -> str | None:
    n = name
    # only the LANGUAGE-MODEL decode path (vision/audio towers are bf16-ignored anyway)
    lm = "language_model" in n
    if n.endswith("per_layer_input_gate") and lm:
        return "per_layer_input_gate"
    if n.endswith("per_layer_projection") and lm:
        return "per_layer_projection"
    if n.endswith("self_attn.o_proj") and lm:
        return "self_attn.o_proj"
    if (n.endswith("mlp.down_proj") or n.endswith("mlp.gate_up_proj")) and lm:
        return "mlp"
    if n.endswith("lm_head"):
        return "lm_head"
    return None


def _walk_model(model):
    """Runs on the vLLM worker: collect first module per dispatch class."""
    seen: dict[str, dict] = {}
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        cls = _classify(name)
        if cls is None or cls in seen:
            continue
        scheme = getattr(mod, "scheme", None)
        seen[cls] = {
            "module_name": name,
            "module_cls": type(mod).__name__,
            "quant_method": type(qm).__name__,
            "scheme": type(scheme).__name__ if scheme is not None else None,
        }
    return seen


def live_walk() -> dict | None:
    """(B) load the built checkpoint and inspect the constructed modules."""
    from vllm import LLM

    print("\n" + "=" * 78)
    print(f"(B) LIVE: loading {BUILD} (compressed-tensors, bf16) for module walk")
    print("-" * 78)
    llm = LLM(
        model=str(BUILD),
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    results = llm.apply_model(_walk_model)
    seen = results[0] if isinstance(results, list) else results
    for cls in ("per_layer_input_gate", "per_layer_projection", "self_attn.o_proj", "mlp", "lm_head"):
        info = seen.get(cls)
        if info is None:
            print(f"  {cls:22s} <not found in named_modules>")
            continue
        print(f"  {cls:22s} {info['quant_method']:32s} scheme={info['scheme']}  "
              f"({info['module_cls']}  {info['module_name']})")
    return seen


def main() -> int:
    notes_ok = True
    stat = static_predicate()
    # static contract
    if not (stat["per_layer_input_gate L0 [TARGET->bf16]"]
            and stat["per_layer_input_gate L20 [TARGET->bf16]"]
            and not stat["per_layer_projection L0 [sibling int4]"]
            and not stat["self_attn.o_proj L0 [body int4]"]
            and not stat["mlp.down_proj L0 [body int4]"]
            and not stat["lm_head [head int4]"]):
        notes_ok = False
        print("  STATIC CONTRACT FAILED")

    if os.environ.get("SKIP_LIVE") == "1":
        print("\n[SKIP_LIVE=1] skipping live walk")
    else:
        seen = live_walk() or {}
        gate = seen.get("per_layer_input_gate", {})
        proj = seen.get("per_layer_projection", {})
        head = seen.get("lm_head", {})
        oproj = seen.get("self_attn.o_proj", {})
        live_ok = (
            gate.get("quant_method") == "UnquantizedLinearMethod"
            and proj.get("quant_method") == "CompressedTensorsLinearMethod"
            and proj.get("scheme") == "CompressedTensorsWNA16"
            and oproj.get("quant_method") == "CompressedTensorsLinearMethod"
            and head.get("quant_method") == "CompressedTensorsLinearMethod"
            and head.get("scheme") == "CompressedTensorsWNA16"
        )
        if not live_ok:
            notes_ok = False
            print("  LIVE CONTRACT FAILED")

    print("=" * 78)
    print("DISPATCH VERIFY:", "PASS" if notes_ok else "FAIL")
    return 0 if notes_ok else 1


if __name__ == "__main__":
    sys.exit(main())
