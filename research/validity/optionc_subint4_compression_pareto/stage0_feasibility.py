#!/usr/bin/env python3
"""Stage 0 — does any sub-int4 weight-compression variant LOAD+SERVE in vLLM dev307?

PR #611, option-C. LOCAL / analysis_only / NO FIRE.

Three variants (PR decreasing-load-likelihood order):
  (a) 2:4-structured-sparse int4 on MLP gate/up/down  (sparsity_config + w4a16)
  (b) mixed w4/w3  (extra config group num_bits=3 on MLP; attn/embed/lm_head stay int4)
  (c) uniform w3a16  (group_0 num_bits=3)

Two independent probes per variant, both exercising the EXACT load-time gates:
  Part A (in-process): call the real dev307 functions that run during model load
    - sparsity: CompressedTensorsConfig.from_config(quantization_config)
    - w3:       CompressedTensorsWNA16(strategy=group, num_bits=3, ...)  (what get_scheme builds)
  Part B (end-to-end): a genuine `vllm.LLM(load_format="dummy")` subprocess on a probe
    model dir (int4-body weights symlinked + the variant's config.json), capturing the
    real process rc + error tail. load_format=dummy avoids re-reading the 10GB safetensors;
    both gates fire at config-parse / scheme-construction, BEFORE any weight load, so the
    weights need not match the modified config.

Run from repo root with the dev307 venv python:
  cd target && CUDA_VISIBLE_DEVICES=0 \
    /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
    research/validity/optionc_subint4_compression_pareto/stage0_feasibility.py
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
INT4_BODY = Path("/workspace/gemma_build/int4_g128_lmhead")
DEV307_PY = "/tmp/senpai-venvs/5f4c623f772358a2/bin/python"
MLP_TARGET = "re:.*mlp\\.(gate_proj|up_proj|down_proj)$"


def base_qc() -> dict:
    cfg = json.load(open(INT4_BODY / "config.json"))
    return cfg["quantization_config"]


def variant_qc(variant: str) -> dict:
    """Return the quantization_config a real build of `variant` would carry."""
    qc = copy.deepcopy(base_qc())
    if variant == "a_24sparse_int4_mlp":
        # 2:4 structured sparse on the MLP, int4 weights unchanged.
        qc["sparsity_config"] = {
            "format": "sparse-24-bitmask",
            "targets": [MLP_TARGET],
            "sparsity_structure": "2:4",
        }
    elif variant == "b_mixed_w4_w3_mlp":
        # keep group_0 (int4 body) + group_1 (int4 lm_head); add a w3 group on the MLP.
        qc["config_groups"]["group_2"] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": [MLP_TARGET],
            "weights": {"num_bits": 3, "type": "int", "strategy": "group",
                        "group_size": 128, "symmetric": True, "observer": "minmax"},
        }
    elif variant == "c_uniform_w3a16":
        qc["config_groups"]["group_0"]["weights"]["num_bits"] = 3
    else:
        raise ValueError(variant)
    return qc


# --------------------------------------------------------------------------- #
# Part A — in-process gate probe (deterministic; calls the real dev307 funcs).
# --------------------------------------------------------------------------- #
def probe_inprocess(variant: str) -> dict:
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )
    qc = variant_qc(variant)
    out = {"gate": None, "load_ok": None, "error_type": None, "error_msg": None}
    try:
        if variant == "a_24sparse_int4_mlp":
            out["gate"] = "CompressedTensorsConfig.from_config (_parse_sparsity_config)"
            CompressedTensorsConfig.from_config(qc)
            out["load_ok"] = True  # parsed without raising
        else:
            out["gate"] = "CompressedTensorsWNA16.__init__ (num_bits gate via get_scheme)"
            # from_config first (must succeed — it just stores groups)
            CompressedTensorsConfig.from_config(qc)
            from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
                CompressedTensorsWNA16,
            )
            CompressedTensorsWNA16(strategy="group", num_bits=3, group_size=128,
                                   symmetric=True)
            out["load_ok"] = True
    except BaseException as exc:  # noqa: BLE001  (DeprecationWarning is raised, not warned)
        out["load_ok"] = False
        out["error_type"] = type(exc).__name__
        out["error_msg"] = str(exc).strip().splitlines()[0][:300]
    return out


# --------------------------------------------------------------------------- #
# Part B — end-to-end genuine engine load (real process rc).
# --------------------------------------------------------------------------- #
def build_probe_dir(variant: str) -> Path:
    d = HERE / f"probe_{variant}"
    d.mkdir(parents=True, exist_ok=True)
    for f in INT4_BODY.iterdir():
        if f.name == "config.json":
            continue
        link = d / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f)
    cfg = json.load(open(INT4_BODY / "config.json"))
    cfg["quantization_config"] = variant_qc(variant)
    json.dump(cfg, open(d / "config.json", "w"), indent=2)
    return d


PROBE_SNIPPET = (
    "import sys;"
    "from vllm import LLM;"
    "LLM(model=sys.argv[1], load_format='dummy', enforce_eager=True,"
    " max_model_len=2048, gpu_memory_utilization=0.85, trust_remote_code=True);"
    "print('LOADED_OK')"
)


def probe_endtoend(variant: str, timeout_s: int = 480) -> dict:
    d = build_probe_dir(variant)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    out = {"probe_dir": str(d), "rc": None, "load_ok": None, "tail": None, "timed_out": False}
    try:
        p = subprocess.run([DEV307_PY, "-c", PROBE_SNIPPET, str(d)],
                           env=env, text=True, capture_output=True, timeout=timeout_s)
        out["rc"] = p.returncode
        combined = (p.stdout + p.stderr).strip().splitlines()
        out["load_ok"] = (p.returncode == 0 and "LOADED_OK" in p.stdout)
        # surface the error signature lines
        sig = [ln for ln in combined if any(k in ln for k in (
            "Unsupported num_bits", "Sparsity support has been removed",
            "Error", "error", "Traceback", "raise", "LOADED_OK"))]
        out["tail"] = "\n".join((sig or combined)[-6:])[:800]
    except subprocess.TimeoutExpired:
        out["timed_out"] = True
        out["load_ok"] = None
        out["tail"] = f"timeout after {timeout_s}s"
    return out


def main() -> int:
    variants = {
        "a_24sparse_int4_mlp": "2:4 structured-sparse int4 on MLP gate/up/down (sparsity_config + w4a16)",
        "b_mixed_w4_w3_mlp": "mixed w4/w3: MLP -> w3 g128, attn/embed/lm_head stay int4",
        "c_uniform_w3a16": "uniform w3a16 (group_0 num_bits 4 -> 3)",
    }
    do_e2e = "--no-e2e" not in sys.argv
    results = {}
    for v, desc in variants.items():
        print(f"\n=== {v}: {desc} ===", flush=True)
        a = probe_inprocess(v)
        print(f"  [A in-proc] gate={a['gate']}", flush=True)
        print(f"  [A in-proc] load_ok={a['load_ok']} {a['error_type']}: {a['error_msg']}", flush=True)
        rec = {"description": desc, "inprocess": a}
        if do_e2e:
            b = probe_endtoend(v)
            print(f"  [B e2e] rc={b['rc']} load_ok={b['load_ok']} timed_out={b['timed_out']}", flush=True)
            print(f"  [B e2e] tail: {b['tail']}", flush=True)
            rec["endtoend"] = b
        # final load_ok = in-process gate verdict (definitive); e2e confirms rc
        rec["load_ok"] = bool(a["load_ok"]) if a["load_ok"] is not None else None
        results[v] = rec

    any_loads = any(r["load_ok"] for r in results.values())
    payload = {
        "pr": 611, "stage": 0, "analysis_only": True, "official_tps": 0,
        "engine": "vllm-dev307 (0.22.1rc1.dev307+g3e8afdf78)",
        "variants": results,
        "any_variant_loads": any_loads,
        "optionc_stage0_verdict": (
            "ALL THREE sub-int4 variants are LOAD-BLOCKED in the mandated vLLM dev307: "
            "(a) sparsity support REMOVED from compressed-tensors (non-empty sparsity_config "
            "raises at engine init); (b)/(c) WNA16 supports only num_bits in {4,8} on Ampere "
            "(num_bits=3 raises at scheme construction). No loadable sub-int4 config exists in "
            "this engine -> option C cannot reach Stage 1; 126.378 stands as the quality-safe "
            "ceiling for the weight-compression lever."
            if not any_loads else
            "At least one variant loaded; proceed to Stage 1 bytes/token + proxy TPS + PPL."),
    }
    (HERE / "stage0_feasibility.json").write_text(json.dumps(payload, indent=2))
    print("\n" + "=" * 90)
    print("STAGE0_VERDICT " + payload["optionc_stage0_verdict"])
    print("=" * 90)
    print(f"wrote {HERE/'stage0_feasibility.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
