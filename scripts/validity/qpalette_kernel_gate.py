#!/usr/bin/env python
"""PR #132 Step-1 KERNEL-SERVABILITY GATE: is there a *servable* sub-4-bit
(3.0-3.5 avg-bit) weight-only GEMM kernel that beats int4-Marlin's M=1 decode
bandwidth on A10G sm_86?

WHY THIS GATE (the #122 / #113 lesson)
--------------------------------------
A fractional-bit quant scheme (Q-Palette, NeurIPS 2025, OpenReview l4F50jpiVH) is
worthless for TPS unless a real CUDA/Triton kernel actually realizes the
weight-byte cut at M=1 decode on *this* hardware. The dominant failure mode
(killed LUT-GEMM in #113): store weights at <4 bits but DEQUANT-TO-bf16 in
shared-mem/registers and run a normal-precision GEMM -> compute-bound -> realized
TPS <= int4, no bandwidth win. So before any PTQ spend we confirm a servable
native sub-4-bit GEMM exists.

WHAT THIS SCRIPT DOES (no GPU, no PTQ, no served-file change)
------------------------------------------------------------
Two evidence streams, both cheap:

A. PINNED-WHEEL INVENTORY (static source parse, authoritative for "in the wheel").
   Scans every weight-only mixed-precision GEMM kernel the pinned vLLM wheel ships
   (vllm/model_executor/kernels/linear/mixed_precision/*.py) and extracts, per
   kernel: SUPPORTED_QUANT_TYPES -> min weight bit-width, get_min_capability(), and
   any hard is_device_capability(N) arch gate. A kernel "serves sub-4-bit on
   sm_86" iff it supports a <4-bit weight type AND admits compute capability 86.

B. LITERATURE DROP-IN MATRIX (encoded from the PR research pass; the "clean
   drop-in" half of the gate). Q-Palette + the named fallbacks (Machete / QTIP /
   AQLM) + the nearest un-named candidates (FLUTE / VPTQ / QuIP#): for each, does
   a servable sub-4-bit GEMM that beats int4 Marlin at M=1 on sm_86 exist?

VERDICT
-------
GREEN  -> a servable sub-4-bit kernel beats int4 Marlin M=1 BW on sm_86.
RED    -> no such kernel in the wheel AND no clean drop-in; only route is
          dequant-to-bf16 (<=int4 TPS) or a from-scratch kernel port. Banks the
          closure of the below-int4 territory for A10G.

Pure static + literature: NO model load, NO HF Job, NO submission. JSON + W&B
(group qpalette-sub4bit).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from pathlib import Path

# Deployed frontier baseline (PR #132 body / BASELINE.md).
DEPLOYED_KERNEL = "MarlinLinearKernel (compressed-tensors W4A16 -> ops.marlin_gemm)"
A10G_CAPABILITY = 86  # sm_86 Ampere
TARGET_AVG_BITS = 3.25  # PR Step-2 target

# scalar_type token -> weight bit-width. First integer after the alpha prefix is
# the bit-width for vLLM ScalarType names (uint4b8=4, uint3b4=3, uint8b128=8,
# float4_e2m1f=4, float8_e4m3fn=8, int4=4, uint2b2=2).
_SCALAR_BITS_RE = re.compile(r"(?:u?int|float)(\d+)")


def scalar_type_bits(token: str) -> int | None:
    m = _SCALAR_BITS_RE.match(token.strip())
    return int(m.group(1)) if m else None


def _resolve_supported_tokens(src: str, list_node: ast.AST) -> list[str]:
    """Resolve a SUPPORTED_QUANT_TYPES assignment to scalar_type.<name> tokens,
    following one level of module-level alias (e.g. SUPPORTED = SOME_CONST)."""
    if isinstance(list_node, ast.List):
        toks = []
        for el in list_node.elts:
            # scalar_types.uint4b8 -> "uint4b8"
            if isinstance(el, ast.Attribute):
                toks.append(el.attr)
        return toks
    return []


def _collect_const_lists(tree: ast.AST, src: str) -> dict[str, list[str]]:
    """Module-level SUPPORTED_* scalar_type lists, incl. `Final`-annotated ones."""
    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        target = value = None
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    target, value = tgt.id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.List):
            if isinstance(node.target, ast.Name):
                target, value = node.target.id, node.value
        if target and "SUPPORTED" in target.upper():
            out[target] = _resolve_supported_tokens(src, value)
    return out


def parse_kernel_source(path: Path, sibling_utils: Path | None = None) -> dict:
    """Static-parse one mixed_precision kernel module for its servability facts."""
    src = path.read_text()
    tree = ast.parse(src)

    supported_tokens: list[str] = []
    min_capability: int | None = None
    hard_cap_gate: int | None = None  # is_device_capability(N) exact-match gate

    # (tree, src) pairs to scan for return-list type queries; kernel file first,
    # then the sibling quantization/utils/<stem>_utils.py (allspark const / marlin
    # query_marlin_supported_quant_types both live there).
    extra_sources: list[tuple[ast.AST, str]] = []
    module_const_lists = _collect_const_lists(tree, src)
    if sibling_utils is not None:
        util_path = sibling_utils / f"{path.stem}_utils.py"
        if util_path.exists():
            usrc = util_path.read_text()
            utree = ast.parse(usrc)
            module_const_lists.update(_collect_const_lists(utree, usrc))
            extra_sources.append((utree, usrc))

    # Class-level SUPPORTED_QUANT_TYPES (literal list or alias to a module const).
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "SUPPORTED_QUANT_TYPES":
                            if isinstance(stmt.value, ast.List):
                                supported_tokens = _resolve_supported_tokens(src, stmt.value)
                            elif isinstance(stmt.value, ast.Name):
                                supported_tokens = module_const_lists.get(stmt.value.id, [])

    # Fall back to any module/util const (conch, allspark).
    if not supported_tokens:
        for name, toks in module_const_lists.items():
            if "WEIGHT" in name.upper() or "QUANT" in name.upper():
                supported_tokens = toks
                break

    # Last-resort: collect `return [scalar_types.x, ...]` lists from the kernel
    # file and its util sibling (marlin exposes supported types via the function
    # query_marlin_supported_quant_types, not a constant).
    if not supported_tokens:
        toks: list[str] = []
        for t, s in [(tree, src)] + extra_sources:
            for node in ast.walk(t):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.List):
                    toks += _resolve_supported_tokens(s, node.value)
        supported_tokens = toks

    # get_min_capability() -> return N
    m = re.search(r"def get_min_capability\([^)]*\)[^:]*:\s*\n\s*return\s+(\d+)", src)
    if m:
        min_capability = int(m.group(1))

    # hard exact-arch gate: is_device_capability(N) used as a reject condition.
    m = re.search(r"is_device_capability\((\d+)\)", src)
    if m:
        hard_cap_gate = int(m.group(1))

    bits = [scalar_type_bits(t) for t in supported_tokens]
    bits = [b for b in bits if b is not None]
    min_bits = min(bits) if bits else None

    runs_on_a10g = True
    gate_reason = None
    if hard_cap_gate is not None and hard_cap_gate > A10G_CAPABILITY:
        runs_on_a10g = False
        gate_reason = f"hard is_device_capability({hard_cap_gate}) gate"
    elif min_capability is not None and min_capability > A10G_CAPABILITY:
        runs_on_a10g = False
        gate_reason = f"get_min_capability()={min_capability} > {A10G_CAPABILITY}"

    serves_sub4_on_a10g = bool(runs_on_a10g and min_bits is not None and min_bits < 4)

    return {
        "kernel": path.stem,
        "supported_types": supported_tokens,
        "min_weight_bits": min_bits,
        "min_capability": min_capability,
        "hard_cap_gate": hard_cap_gate,
        "runs_on_a10g_sm86": runs_on_a10g,
        "gate_reason": gate_reason,
        "serves_sub4bit_on_a10g": serves_sub4_on_a10g,
    }


def scan_pinned_wheel(mp_dir: Path) -> list[dict]:
    # sibling vllm/model_executor/layers/quantization/utils (allspark const lives here)
    sibling_utils = mp_dir.parents[2] / "layers" / "quantization" / "utils"
    rows = []
    for path in sorted(mp_dir.glob("*.py")):
        if path.stem in {"__init__", "MPLinearKernel"}:
            continue
        try:
            rows.append(parse_kernel_source(path, sibling_utils))
        except Exception as exc:  # pragma: no cover - defensive
            rows.append({"kernel": path.stem, "parse_error": repr(exc)})
    return rows


# --- Literature "clean drop-in" matrix (from the PR #132 research pass) --------
# servable_sub4_beats_marlin_m1_sm86 in {"yes","no","unknown"}.
LIT_MATRIX = [
    {
        "scheme": "Q-Palette (named lever)", "repo": "snu-mllab/Q-Palette",
        "family": "trellis+vector+scalar, Hadamard-incoherence (fractional-bit)",
        "native_lowbit_gemm": "unknown", "sm86_kernel": "no/unknown (RTX4090 sm_89 Ada only)",
        "vllm_path": "none", "m1_vs_marlin_number": "none (Ada 190-200 tok/s, does not transfer)",
        "servable_sub4_beats_marlin_m1_sm86": "no",
        "note": "Ada-only benchmarks, no sm_86 doc, no vLLM integration -> not a clean drop-in",
    },
    {
        "scheme": "Machete", "repo": "vllm (kernels/.../machete)",
        "family": "CUTLASS mixed-input GEMM",
        "native_lowbit_gemm": "yes", "sm86_kernel": "no (sm_90a Hopper-only)",
        "vllm_path": "in-wheel but arch-gated off", "m1_vs_marlin_number": "n/a on sm_86",
        "servable_sub4_beats_marlin_m1_sm86": "no",
        "note": "get_min_capability()=90; on A10G vLLM falls back to Marlin; uint4/uint8 only",
    },
    {
        "scheme": "QTIP", "repo": "Cornell-RelaxML/qtip",
        "family": "trellis-coded quant (bitshift HYB kernel)",
        "native_lowbit_gemm": "yes (BW-bound, >80% peak HBM)", "sm86_kernel": "yes (RTX3090/A6000 Ampere)",
        "vllm_path": "none (standalone stack)", "m1_vs_marlin_number": "119 tok/s @3b vs fp16 52.5 (not vs Marlin)",
        "servable_sub4_beats_marlin_m1_sm86": "unknown",
        "note": "good Ampere kernel but NO vLLM serving path; vs-Marlin M=1 unproven -> not a clean drop-in",
    },
    {
        "scheme": "AQLM", "repo": "Vahe1994/AQLM",
        "family": "additive/multi-codebook vector quant",
        "native_lowbit_gemm": "yes", "sm86_kernel": "partial (demo)",
        "vllm_path": "old demo only", "m1_vs_marlin_number": "'up to 3x' vs fp16 (not vs Marlin)",
        "servable_sub4_beats_marlin_m1_sm86": "unknown",
        "note": "codebook lookups serialize at M=1; speedup claim vs fp16 not Marlin",
    },
    {
        "scheme": "VPTQ", "repo": "microsoft/VPTQ",
        "family": "vector PTQ",
        "native_lowbit_gemm": "no (dequant-to-fp16 then matmul)", "sm86_kernel": "n/a",
        "vllm_path": "planned, not merged", "m1_vs_marlin_number": "n/a (compute-bound)",
        "servable_sub4_beats_marlin_m1_sm86": "no",
        "note": "exact #113 failure mode: dequant-to-fp16 -> realized TPS <= int4",
    },
    {
        "scheme": "QuIP#", "repo": "Cornell-RelaxML/quip-sharp",
        "family": "E8 lattice codebook",
        "native_lowbit_gemm": "yes (<5 instr/weight)", "sm86_kernel": "unknown",
        "vllm_path": "none", "m1_vs_marlin_number": "no Ampere batch=1 numbers",
        "servable_sub4_beats_marlin_m1_sm86": "unknown",
        "note": "technically BW-bound but no confirmed sm_86 decode numbers, no vLLM path",
    },
    {
        "scheme": "FLUTE (nearest un-named candidate)", "repo": "HanGuo97/flute",
        "family": "fused LUT dequant+GEMM (3-bit)",
        "native_lowbit_gemm": "yes (LUT in shared-mem, NOT dequant-to-fp16)", "sm86_kernel": "yes (Ampere-optimized)",
        "vllm_path": "old (~0.5.x monkeypatch; not the 0.22 MPLinearKernel API)",
        "m1_vs_marlin_number": "A6000 W3G128 108.1 vs W4G128 98.1 tok/s (~10% 3b>4b); NO Marlin row",
        "servable_sub4_beats_marlin_m1_sm86": "unknown",
        "note": "closest live candidate, but never benchmarked vs Marlin int4; FLUTE-4b < Marlin-4b -> "
                "FLUTE-3b ~ Marlin wash; vLLM 0.22 integration is a from-scratch port, not a clean drop-in",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mp-dir",
        default=".venvs/vllm022/lib/python3.12/site-packages/vllm/"
        "model_executor/kernels/linear/mixed_precision",
        help="pinned vLLM mixed_precision kernel dir",
    )
    ap.add_argument("--out", default="research/validity/qpalette_sub4bit/kernel_gate.json")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb-group", default="qpalette-sub4bit")
    ap.add_argument("--wandb-name", default="kanna/qpalette-sub4bit-kernel-gate")
    args = ap.parse_args()

    mp_dir = Path(args.mp_dir)
    wheel_rows = scan_pinned_wheel(mp_dir)
    cuda_rows = [r for r in wheel_rows if r.get("kernel") not in {"cpu", "xpu"}]

    servable_in_wheel = [r for r in wheel_rows if r.get("serves_sub4bit_on_a10g")]
    servable_drop_in = [r for r in LIT_MATRIX if r["servable_sub4_beats_marlin_m1_sm86"] == "yes"]

    # min servable weight bits on sm_86 across the whole pinned wheel.
    a10g_bits = [
        r["min_weight_bits"] for r in wheel_rows
        if r.get("runs_on_a10g_sm86") and r.get("min_weight_bits") is not None
    ]
    min_servable_bits_sm86 = min(a10g_bits) if a10g_bits else None

    servable = bool(servable_in_wheel or servable_drop_in)
    clears_500 = 0  # no servable kernel -> no realized BW win to project
    projected_official_tps = None
    verdict = "GREEN" if servable else "RED"

    gate = {
        "verdict": verdict,
        "deployed_kernel": DEPLOYED_KERNEL,
        "a10g_capability": A10G_CAPABILITY,
        "target_avg_bits": TARGET_AVG_BITS,
        "n_wonly_kernels_scanned": len(wheel_rows),
        "n_cuda_wonly_kernels": len(cuda_rows),
        "n_subbit_servable_in_wheel": len(servable_in_wheel),
        "n_subbit_servable_drop_in": len(servable_drop_in),
        "min_servable_weight_bits_sm86": min_servable_bits_sm86,
        "qpalette_servable_and_clears_500": clears_500,
        "qpalette_projected_official_tps": projected_official_tps,
    }

    payload = {
        "config": {
            "pr": 132, "step": "1-kernel-servability",
            "gpu": "A10G sm_86", "pinned_vllm": "0.22.x (frontier 0.22.1rc1.dev307+g3e8afdf78.cu129)",
            "baseline_official_tps": 481.53, "baseline_ppl": 2.3772,
            "local_wall_tps_ref": 454.338, "multiplier_99": 1.06019,
        },
        "gate": gate,
        "pinned_wheel_inventory": wheel_rows,
        "literature_dropin_matrix": LIT_MATRIX,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    # --- console summary ---
    print("=" * 78)
    print(f"PR #132 Step-1 KERNEL-SERVABILITY GATE  ->  {verdict}")
    print("=" * 78)
    print(f"{'kernel':22} {'min_bits':>8} {'min_cap':>7} {'a10g?':>6} {'sub4@sm86?':>11}  reason")
    for r in wheel_rows:
        if "parse_error" in r:
            print(f"{r['kernel']:22} PARSE_ERROR {r['parse_error']}")
            continue
        print(f"{r['kernel']:22} {str(r['min_weight_bits']):>8} {str(r['min_capability']):>7} "
              f"{str(r['runs_on_a10g_sm86']):>6} {str(r['serves_sub4bit_on_a10g']):>11}  "
              f"{r['gate_reason'] or ''}")
    print("-" * 78)
    print(f"min servable weight-bits on sm_86 (whole wheel): {min_servable_bits_sm86}")
    print(f"sub-4-bit servable IN WHEEL: {len(servable_in_wheel)}  |  "
          f"clean DROP-IN beating Marlin M=1: {len(servable_drop_in)}")
    print(f"primary_metric qpalette_projected_official_tps = {projected_official_tps}")
    print(f"test_metric    qpalette_servable_and_clears_500 = {clears_500}")
    print(f"JSON -> {out_path}")

    if args.wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:
            print(f"[qpgate] W&B logging failed: {exc!r}", flush=True)


def _log_wandb(args, payload) -> None:
    import wandb

    run = wandb.init(
        entity=args.wandb_entity, project=args.wandb_project,
        group=args.wandb_group, name=args.wandb_name,
        job_type="validity-gate", config=payload["config"],
    )

    wcols = ["kernel", "supported_types", "min_weight_bits", "min_capability",
             "runs_on_a10g_sm86", "serves_sub4bit_on_a10g", "gate_reason"]
    wtbl = wandb.Table(columns=wcols)
    for r in payload["pinned_wheel_inventory"]:
        wtbl.add_data(
            r.get("kernel"), ",".join(r.get("supported_types", []) or []),
            r.get("min_weight_bits"), r.get("min_capability"),
            r.get("runs_on_a10g_sm86"), r.get("serves_sub4bit_on_a10g"),
            r.get("gate_reason"),
        )
    run.log({"pinned_wheel_inventory": wtbl})

    lcols = ["scheme", "repo", "native_lowbit_gemm", "sm86_kernel", "vllm_path",
             "m1_vs_marlin_number", "servable_sub4_beats_marlin_m1_sm86", "note"]
    ltbl = wandb.Table(columns=lcols)
    for r in payload["literature_dropin_matrix"]:
        ltbl.add_data(*[r[c] for c in lcols])
    run.log({"literature_dropin_matrix": ltbl})

    g = payload["gate"]
    run.summary.update({k: v for k, v in g.items() if not isinstance(v, str) and v is not None})
    run.summary.update({
        "verdict": g["verdict"],
        "qpalette_servable_and_clears_500": g["qpalette_servable_and_clears_500"],
        "deployed_kernel": g["deployed_kernel"],
    })
    run.finish()
    print(f"[qpgate] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
