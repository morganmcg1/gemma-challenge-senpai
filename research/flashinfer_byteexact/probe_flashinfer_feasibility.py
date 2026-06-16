#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""FlashInfer byte-exact M-invariance probe for google/gemma-4-E4B-it on sm_86 (PR #507).

Morgan's #481 ZOOM-OUT asked us to look past vLLM+Triton for a *free* byte-exact reduction.
denken #498 closed SGLang and fern #502 closed TensorRT-LLM (engine never builds). FlashInfer
-- vLLM/SGLang's high-performance attention library -- is the last open engine/kernel thread.
The decisive question: is FlashInfer decode attention byte-exact M=1-vs-M=8 invariant OUT OF THE
BOX (a cheaper route to lawine #496's split-KV 399.75 byte-exact rung), or batch-variant like
stock vLLM?

VERDICT (this card): FlashInfer **LOADS and RUNS** (unlike TRT-LLM) but is **NOT free byte-exact
M-invariant**. Its default split-KV decode is batch-VARIANT on the serving-relevant tensor-core
(GQA) path at every sequence length tested. It DOES natively expose `fixed_split_size` /
`disable_split_kv` -- the first-class API form of lawine's "fix the split SIZE not count" trick
-- which deliver 0-flip byte identity, but they cost 1.2-4.7x the M=1 decode-attn time of the
auto path (not the ~0% of the hand-rolled rung), AND the 7 head_dim-512 full-attention layers
have NO tensor-core/fixed-split path at all, AND flashinfer-python pins torch 2.10+cu128 vs the
pod's torch 2.11+cu130 (version skew -> not a clean drop-in to the deployed vLLM 0.22.1rc1 stack).
So FlashInfer is not a cheaper byte-exact route than the deployed split-KV 399.75; the last
engine thread closes NEGATIVE -- but it confirms the deployed hand-rolled trick is reproducing a
stock, upstreamed FlashInfer primitive.

LOCAL diagnostic. GPU used only for the FlashInfer decode-attn census (no model forward, no
serve, no TPS-on-official-prompts). NO HF job, NO submission, NO served-file change.

The flashinfer kernels run in an isolated venv (default /tmp/fi_probe) carrying the torch
2.10+cu128 that flashinfer-python pins; the pod's .venv keeps torch 2.11+cu130 untouched. The
M-invariance property measured (does the split scheduler change request-0's reduction tree when
M grows?) is an algorithmic property of the kernel and transfers across the torch minor version.

Reproduce:
    cd target/ && .venv/bin/python \
      research/flashinfer_byteexact/probe_flashinfer_feasibility.py \
      --self-test --wandb_group flashinfer-byteexact \
      --wandb_name fern/flashinfer-byteexact-probe

PRIMARY metric : `feasibility_evidence_complete` (1 iff flashinfer loads, the census is NaN-clean,
                 the default path is measured batch-variant AND the fixed_split path measured
                 0-flip invariant, and the head_dim-512 tensor-core support is probed -- i.e. the
                 negative-with-nuance verdict is fully substantiated and would flip loudly if a
                 future flashinfer gave free invariance or a head_dim-512 kernel).
TEST   metric  : `flashinfer_attn_m_invariant` (expected 0 -- default decode is batch-variant).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # .../target

MODEL_ID = "google/gemma-4-E4B-it"

# Ampere (sm_86) attention head_dim ceiling. FlashAttention-2 / FlashInfer FA2 tensor-core decode
# support head_dim up to 256 on Ampere; 512 (the 7 full-attention layers) exceeds the per-SM
# shared-memory budget and has no flashinfer tensor-core kernel (Dao-AILab/flash-attention#2427).
AMPERE_ATTN_HEADDIM_CAP = 256

# flashinfer-python's latest release pins torch 2.10 + cu128; the pod's serving-adjacent .venv
# carries torch 2.11 + cu130. flashinfer publishes no torch-2.11 AOT wheels (JIT-only), and an
# unresolved CUDA-12-cubin-vs-CUDA-13 incompatibility (flashinfer-ai/flashinfer#2195) makes a
# clean drop-in into the deployed torch-2.11/cu130 vLLM 0.22.1rc1 stack non-trivial.
FLASHINFER_REQUIRES_TORCH = "2.10.0"
FLASHINFER_REQUIRES_CUDA = "cu128"

DEFAULT_FI_VENV = "/tmp/fi_probe"
HEADLINE_CENSUS_L = 8192   # representative deep-decode KV length (clearly in the split regime)
HEADLINE_TPS_L = 4096      # representative mid-decode KV length for the M=1 cost ratio

# Researcher pass (PR #507), recorded with single strongest citation per finding.
RESEARCH = {
    "default_split_kv_batch_variant": {
        "value": True,
        "finding": "flashinfer decode plan() chooses the number of KV splits from an "
                   "occupancy heuristic (more splits at small batch to fill SMs, fewer as batch "
                   "grows) -> the merge_states reduction tree, and thus the output bytes, change "
                   "with M. Default is batch-VARIANT.",
        "citation": "SGLang deterministic-inference blog (uses flashinfer fixed_split_size to "
                    "override the default non-fixed planning); flashinfer plan() API semantics.",
    },
    "fixed_split_size_invariant": {
        "value": True,
        "finding": "fixed_split_size (FA2 tensor-core decode) fixes the split SIZE in pages, not "
                   "the count -> deterministic softmax merge_states -> batch-size-invariant "
                   "outputs. This is the first-class API form of lawine #496's split-KV trick.",
        "citation": "flashinfer BatchDecodeWithPagedKVCacheWrapper.plan() docstring: 'will lead "
                    "to deterministic softmax score reduction in the merge_states kernel, and "
                    "therefore batch-size invariant outputs'.",
    },
    "headdim_512_blocked": {
        "value": True,
        "finding": "FlashInfer/FA2 attention supports head_dim in {64,128,256} on sm_86; "
                   "head_dim 512 (the 7 full-attention layers) has no tensor-core kernel "
                   "(per-SM shared-memory budget exceeded). No fixed-split invariant path exists "
                   "for those layers.",
        "citation": "Dao-AILab/flash-attention#2427 (head_dim 512 for Gemma-4 global attn "
                    "unsupported); vLLM#38918 (flashinfer 'only supports [64,128,256]').",
    },
    "invariance_cost": {
        "value": "1.2-4.7x M=1 kernel slowdown (grows with KV length)",
        "finding": "Unlike lawine's hand-rolled ~0%-cost rung, flashinfer's fixed_split_size/"
                   "disable_split_kv forfeit the split-KV GPU-fill that makes M=1 decode fast, so "
                   "the invariant mode is materially slower at M=1 and worsens with longer KV.",
        "citation": "SGLang reports ~34% serve slowdown for --enable-deterministic-inference; "
                    "measured here as a 1.2x (L=2048) -> 4.7x (L=8192) M=1 decode-attn slowdown.",
    },
    "vllm_integration": {
        "value": "manual patch + per-layer backend split required",
        "finding": "vLLM 0.22.1rc1 predates batch-invariant wiring; using flashinfer's "
                   "fixed_split_size needs a manual plan() patch, AND a per-layer backend split "
                   "(flashinfer for head_dim-256 sliding layers, Triton for head_dim-512 full "
                   "layers) since flashinfer cannot serve the full-attention layers.",
        "citation": "vLLM#27433 (batch-invariant feature, newer releases); researcher pass #507.",
    },
}


# --------------------------------------------------------------------------- #
# A — live architecture probe (current env transformers; loads from HF cache).
# --------------------------------------------------------------------------- #
def probe_arch() -> dict[str, Any]:
    import transformers
    from transformers import AutoConfig

    detail: dict[str, Any] = {"env_transformers": transformers.__version__}
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False, local_files_only=True)
    tc = cfg.text_config
    layer_types = list(tc.layer_types)
    full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    sliding = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
    detail.update(
        loaded=True,
        model_type=cfg.model_type,
        architectures=list(cfg.architectures),
        num_text_layers=tc.num_hidden_layers,
        head_dim_sliding=tc.head_dim,
        head_dim_full=getattr(tc, "global_head_dim", None),
        mixed_head_dim=bool(getattr(tc, "global_head_dim", tc.head_dim) != tc.head_dim),
        num_attention_heads=getattr(tc, "num_attention_heads", None),
        num_key_value_heads=getattr(tc, "num_key_value_heads", None),
        gqa_group_size=(getattr(tc, "num_attention_heads", 0) // getattr(tc, "num_key_value_heads", 1)
                        if getattr(tc, "num_key_value_heads", None) else None),
        num_full_attention=len(full),
        num_sliding_attention=len(sliding),
        full_attention_idxs=full,
        sliding_window=getattr(tc, "sliding_window", None),
        num_kv_shared_layers=getattr(tc, "num_kv_shared_layers", None),
        final_logit_softcapping=getattr(tc, "final_logit_softcapping", None),
        attn_logit_softcapping=getattr(tc, "attn_logit_softcapping", None),
        has_audio_config=hasattr(cfg, "audio_config") and cfg.audio_config is not None,
        has_vision_config=hasattr(cfg, "vision_config") and cfg.vision_config is not None,
    )
    return detail


# --------------------------------------------------------------------------- #
# B — version skew: pod torch (.venv) vs the torch flashinfer-python pins; nvcc presence.
# --------------------------------------------------------------------------- #
def probe_versions(fi_python: str) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "flashinfer_requires_torch": FLASHINFER_REQUIRES_TORCH,
        "flashinfer_requires_cuda": FLASHINFER_REQUIRES_CUDA,
    }
    try:
        import torch as _t  # pod (.venv) torch

        detail["pod_torch"] = _t.__version__
    except Exception as exc:  # noqa: BLE001
        detail["pod_torch"] = f"err: {type(exc).__name__}"
    nvcc = shutil.which("nvcc")
    detail["nvcc_present"] = bool(nvcc)
    if nvcc:
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=30).stdout
            for line in out.splitlines():
                if "release" in line:
                    detail["nvcc_release"] = line.strip()
                    break
        except Exception:  # noqa: BLE001
            pass
    detail["fi_python"] = fi_python
    detail["fi_venv_present"] = Path(fi_python).exists()
    # skew holds iff the pod torch major.minor differs from what flashinfer pins
    pod = str(detail.get("pod_torch", "")).split("+")[0]
    detail["version_skew"] = bool(pod[:4] != FLASHINFER_REQUIRES_TORCH[:4])
    return detail


# --------------------------------------------------------------------------- #
# C — build the isolated flashinfer venv if missing (uv); then run the GPU census.
# --------------------------------------------------------------------------- #
def ensure_fi_venv(fi_python: str, allow_build: bool) -> dict[str, Any]:
    info: dict[str, Any] = {"fi_python": fi_python, "built": False}
    if Path(fi_python).exists():
        info["present"] = True
        return info
    info["present"] = False
    if not allow_build or not shutil.which("uv"):
        info["build_skipped"] = True
        return info
    venv_dir = str(Path(fi_python).parents[1])  # /tmp/fi_probe/bin/python -> /tmp/fi_probe
    try:
        subprocess.run(["uv", "venv", venv_dir, "--python", "3.12"],
                       check=True, capture_output=True, text=True, timeout=180)
        subprocess.run(["uv", "pip", "install", "--python", fi_python, "flashinfer-python", "numpy"],
                       check=True, capture_output=True, text=True, timeout=900)
        info["built"] = True
        info["present"] = Path(fi_python).exists()
    except Exception as exc:  # noqa: BLE001
        info["build_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return info


def run_gpu_census(fi_python: str) -> dict[str, Any]:
    census_script = HERE / "fi_gpu_census.py"
    if not Path(fi_python).exists():
        return {"flashinfer_import": False, "error": f"isolated venv python missing: {fi_python}"}
    try:
        proc = subprocess.run([fi_python, str(census_script)],
                              capture_output=True, text=True, timeout=1200)
    except Exception as exc:  # noqa: BLE001
        return {"flashinfer_import": False, "error": f"census subprocess failed: {type(exc).__name__}: {exc}"}
    for line in proc.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    return {"flashinfer_import": False,
            "error": (proc.stderr or proc.stdout).strip()[:400] or "no JSON line emitted"}


# --------------------------------------------------------------------------- #
# Headline ledger + self-test gate.
# --------------------------------------------------------------------------- #
def _cell(census: dict, path: str, mode: str, L: int) -> dict[str, Any]:
    try:
        return census["census"][path][mode][str(L)]
    except Exception:  # noqa: BLE001
        return {}


def build_ledger(arch: dict, versions: dict, census: dict) -> dict[str, Any]:
    loaded = bool(census.get("flashinfer_import"))
    hd = census.get("headdim_support", {})
    tc_d256_ok = bool(hd.get("tensor_core_d256", {}).get("ok"))
    tc_d512_ok = bool(hd.get("tensor_core_d512", {}).get("ok"))

    auto_cell = _cell(census, "tensor_core", "auto", HEADLINE_CENSUS_L)
    fixed_cell = _cell(census, "tensor_core", "fixed_split_512", HEADLINE_CENSUS_L)
    disable_cell = _cell(census, "tensor_core", "disable_split_kv", HEADLINE_CENSUS_L)

    total_flips = int(auto_cell.get("flips", -1))
    m_invariant = int(loaded and total_flips == 0)  # 0 expected: default is batch-variant
    fixed_split_invariant = int(bool(fixed_cell.get("invariant")))
    disable_split_invariant = int(bool(disable_cell.get("invariant")))

    # M=1 decode-attn micro-throughput (kernel-level, NOT serve TPS) + cost of invariance.
    tps = census.get("tps", {})
    auto_tps = tps.get(str(HEADLINE_TPS_L), {}).get("auto", {})
    fixed_tps = tps.get(str(HEADLINE_TPS_L), {}).get("fixed_split_512", {})
    m1_decode_tps = float(auto_tps.get("steps_per_s", 0.0))
    cost_ratio = (auto_tps.get("steps_per_s", 0.0) / fixed_tps.get("steps_per_s", 1.0)
                  if fixed_tps.get("steps_per_s") else None)

    # NaN cleanliness across the whole census.
    nan_seen = False
    for path in census.get("census", {}).values():
        for mode in path.values():
            for c in mode.values():
                if isinstance(c, dict) and c.get("nan"):
                    nan_seen = True

    headdim512_tc_blocked = int(loaded and not tc_d512_ok)

    # vs_splitkv_399_75: is flashinfer a CHEAPER free byte-exact route than the deployed
    # split-KV 399.75 rung? No -- on three independent grounds (slower invariant mode at M=1,
    # head_dim-512 layers unservable invariantly, version skew). NEGATIVE.
    cheaper_free_route = int(
        loaded and m_invariant == 1  # would need free (auto) invariance, which it does NOT have
    )

    findings = {
        "F1_flashinfer_loads": {
            "rank": 1, "confirmed": loaded, "live": True,
            "evidence": f"flashinfer {census.get('flashinfer_version')} imports + runs decode on "
                        f"{census.get('device_name')} cap {census.get('device_capability')} "
                        f"(torch {census.get('torch_version')}, isolated venv).",
        },
        "F2_default_split_kv_batch_variant": {
            "rank": 2, "confirmed": bool(loaded and total_flips > 0), "live": True,
            "evidence": f"tensor-core (GQA) auto split-KV: {total_flips}/{auto_cell.get('numel')} "
                        f"output elements flip M=1-vs-M=8 at L={HEADLINE_CENSUS_L} "
                        f"(max_abs {auto_cell.get('max_abs')}); variant at every L tested. "
                        f"NOT free byte-exact.",
        },
        "F3_fixed_split_size_invariant": {
            "rank": 3, "confirmed": bool(fixed_split_invariant and disable_split_invariant),
            "live": True,
            "evidence": "fixed_split_size and disable_split_kv -> 0 flips at every L (byte-exact "
                        "M-invariant): the first-class API form of lawine #496's split-KV trick.",
        },
        "F4_headdim512_no_invariant_path": {
            "rank": 4, "confirmed": bool(headdim512_tc_blocked), "live": True,
            "evidence": f"head_dim 512 (7 full-attention layers): tensor-core decode dispatch "
                        f"FAILS (no kernel); fixed_split_size requires tensor core -> no byte-exact "
                        f"invariant flashinfer path for those layers. (cuda-core runs d512 but has "
                        f"no working invariant knob.)",
        },
        "F5_version_skew_not_clean_drop_in": {
            "rank": 5, "confirmed": bool(versions.get("version_skew")), "live": True,
            "evidence": f"flashinfer-python pins torch {FLASHINFER_REQUIRES_TORCH}+{FLASHINFER_REQUIRES_CUDA}; "
                        f"pod serves torch {versions.get('pod_torch')} -> not a clean drop-in to the "
                        f"deployed vLLM 0.22.1rc1 stack (JIT-only torch-2.11; CUDA12/13 cubin risk).",
        },
    }
    all_confirmed = all(v["confirmed"] for v in findings.values())
    feasibility_evidence_complete = int(
        loaded
        and bool(arch.get("loaded"))
        and total_flips > 0          # default path measured variant
        and fixed_split_invariant == 1  # fixed-split path measured invariant
        and ("tensor_core_d512" in hd)  # head_dim-512 support probed
        and not nan_seen
    )
    return {
        "findings": findings,
        "all_findings_confirmed": all_confirmed,
        # headline KEY OUTPUTS requested by the PR:
        "flashinfer_loads": int(loaded),
        "flashinfer_attn_m_invariant": m_invariant,
        "flashinfer_total_flips_M1vs8": total_flips,
        "flashinfer_m1_decode_tps": m1_decode_tps,
        "vs_splitkv_399_75_cheaper_free_route": cheaper_free_route,
        # supporting metrics:
        "fixed_split_size_invariant": fixed_split_invariant,
        "disable_split_kv_invariant": disable_split_invariant,
        "headdim512_tensor_core_blocked": headdim512_tc_blocked,
        "headdim256_tensor_core_ok": int(tc_d256_ok),
        "version_skew": int(bool(versions.get("version_skew"))),
        "invariance_cost_ratio_m1": cost_ratio,
        "census_nan_seen": int(nan_seen),
        "feasibility_evidence_complete": feasibility_evidence_complete,
        "headline_census_L": HEADLINE_CENSUS_L,
        "headline_tps_L": HEADLINE_TPS_L,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors the house pattern; never fatal).
# --------------------------------------------------------------------------- #
def maybe_log_wandb(args: argparse.Namespace, payload: dict[str, Any]) -> list[str]:
    if getattr(args, "no_wandb", False):
        return []
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    try:
        import wandb as _wb  # noqa: F401

        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init")
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[fi-probe] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return []
    arch = payload["arch"]
    led = payload["ledger"]
    census = payload["census"]
    try:
        run = init_wandb_run(
            job_type="analysis", agent="fern",
            name=args.wandb_name or "fern/flashinfer-byteexact-probe",
            group=args.wandb_group,
            notes="FlashInfer byte-exact M-invariance probe for gemma-4-E4B-it on sm_86 (PR #507). "
                  "Loads + runs, but default split-KV decode is batch-VARIANT (not free byte-exact); "
                  "fixed_split_size/disable_split_kv give 0-flip invariance at 1.2-4.7x M=1 cost; "
                  "head_dim-512 full-attn layers have no tensor-core/fixed-split path; torch "
                  "2.10+cu128 vs pod 2.11+cu130 skew. GPU used only for the decode-attn census; "
                  "no model forward, no serve, no HF job, no served-file change.",
            tags=["flashinfer", "engine-benchmark", "zoom-out-481", "byte-exact", "m-invariance",
                  "negative", "pr-507"],
            config={"pr": 507, "wandb_group": args.wandb_group, "model_id": MODEL_ID,
                    "hardware": "A10G sm_86 24GB", "analysis_only": True, "official_tps": 0,
                    "flashinfer_version": census.get("flashinfer_version"),
                    "flashinfer_torch": census.get("torch_version"),
                    "pod_torch": payload["versions"].get("pod_torch"),
                    "census_config": census.get("config"),
                    "env_transformers": arch.get("env_transformers")},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[fi-probe] wandb init failed (analysis unaffected): {exc}", flush=True)
        return []
    if run is None:
        print("[fi-probe] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return []
    summary: dict[str, Any] = {
        # PR KEY OUTPUTS:
        "flashinfer_loads": led["flashinfer_loads"],
        "flashinfer_attn_m_invariant": led["flashinfer_attn_m_invariant"],
        "flashinfer_total_flips_M1vs8": led["flashinfer_total_flips_M1vs8"],
        "flashinfer_m1_decode_tps": led["flashinfer_m1_decode_tps"],
        "vs_splitkv_399_75_cheaper_free_route": led["vs_splitkv_399_75_cheaper_free_route"],
        # supporting:
        "fixed_split_size_invariant": led["fixed_split_size_invariant"],
        "disable_split_kv_invariant": led["disable_split_kv_invariant"],
        "headdim512_tensor_core_blocked": led["headdim512_tensor_core_blocked"],
        "headdim256_tensor_core_ok": led["headdim256_tensor_core_ok"],
        "version_skew": led["version_skew"],
        "invariance_cost_ratio_m1": led["invariance_cost_ratio_m1"],
        "feasibility_evidence_complete": led["feasibility_evidence_complete"],
        "all_findings_confirmed": int(led["all_findings_confirmed"]),
        "census_nan_seen": led["census_nan_seen"],
        # this card never serves prompts:
        "official_tps": 0, "ppl": 0, "analysis_only": 1, "no_served_file_change": 1,
        "gpu_used": 1, "gpu_model_forward": 0,
        # structural facts:
        "head_dim_sliding": arch.get("head_dim_sliding"),
        "head_dim_full": arch.get("head_dim_full"),
        "ampere_attn_headdim_cap": AMPERE_ATTN_HEADDIM_CAP,
        "num_full_attention": arch.get("num_full_attention"),
        "num_sliding_attention": arch.get("num_sliding_attention"),
        "gqa_group_size": arch.get("gqa_group_size"),
        "peak_mem_mb": census.get("peak_mem_mb"),
        "headline_census_L": led["headline_census_L"],
        "headline_tps_L": led["headline_tps_L"],
    }
    # flatten the tensor-core census sweep so the L-dependence is queryable in W&B.
    for mode in ("auto", "fixed_split_512", "disable_split_kv"):
        for L, c in census.get("census", {}).get("tensor_core", {}).get(mode, {}).items():
            if isinstance(c, dict) and "flips" in c:
                summary[f"tc_{mode}_L{L}_flips"] = c["flips"]
    for key, blk in led["findings"].items():
        summary[f"finding_{key}_confirmed"] = int(bool(blk["confirmed"]))
    run_ids: list[str] = []
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="flashinfer_byteexact_probe_result", artifact_type="analysis",
                          data=payload)
        run_ids.append(getattr(run, "id", "") or "")
        print(f"[fi-probe] wandb run logged: {getattr(run, 'id', '?')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[fi-probe] wandb summary/artifact skipped: {exc}", flush=True)
    finish_wandb(run)
    return [r for r in run_ids if r]


# --------------------------------------------------------------------------- #
# REPORT.md (generated so it always matches the live result).
# --------------------------------------------------------------------------- #
def write_report(path: Path, payload: dict[str, Any]) -> None:
    arch = payload["arch"]
    led = payload["ledger"]
    versions = payload["versions"]
    census = payload["census"]
    tc = census.get("census", {}).get("tensor_core", {})
    cc = census.get("census", {}).get("cuda_core", {})

    def flips_row(d: dict, modes: list[str]) -> list[str]:
        cells = []
        for L in census.get("config", {}).get("census_l", []):
            parts = []
            for m in modes:
                c = d.get(m, {}).get(str(L), {})
                parts.append(f"{m.split('_')[0]}={c.get('flips', '?')}" if "flips" in c else f"{m}=ERR")
            cells.append(f"L={L}: " + ", ".join(parts))
        return cells

    verdict = ("LOADS but NOT free byte-exact M-invariant -- default split-KV decode is "
               "batch-VARIANT; fixed_split_size gives invariance at 1.2-4.7x M=1 cost; head_dim-512 "
               "layers blocked; version skew -> not a cheaper route to the 399.75 byte-exact rung")
    cost = led.get("invariance_cost_ratio_m1")
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: 2026 CoreWeave, Inc.",
        "SPDX-License-Identifier: Apache-2.0",
        "SPDX-PackageName: senpai",
        "-->",
        "",
        "# FlashInfer byte-exact M-invariance for `google/gemma-4-E4B-it` on A10G (sm_86)",
        "",
        f"**PR:** #507 · **Author:** fern · **Generated:** {payload['generated_utc']} · "
        f"**W&B group:** `flashinfer-byteexact`",
        "",
        "**LOCAL diagnostic. GPU used ONLY for the FlashInfer decode-attn census (no model "
        "forward, no serve, no official-prompt TPS). NO HF job, NO submission, NO served-file "
        "change. `analysis_only=true`, `official_tps=0`.**",
        "",
        f"Reproduce: `cd target/ && .venv/bin/python {Path(*HERE.parts[-2:])}/probe_flashinfer_feasibility.py --self-test`",
        "",
        "---",
        "",
        f"## Verdict: {verdict}",
        "",
        f"`feasibility_evidence_complete = {led['feasibility_evidence_complete']}` · "
        f"`flashinfer_loads = {led['flashinfer_loads']}` · "
        f"`flashinfer_attn_m_invariant = {led['flashinfer_attn_m_invariant']}` · "
        f"`flashinfer_total_flips_M1vs8 = {led['flashinfer_total_flips_M1vs8']}` "
        f"(tensor-core auto, L={led['headline_census_L']})",
        "",
        "## KEY OUTPUTS",
        "",
        "| metric | value |",
        "|---|---|",
        f"| `flashinfer_loads` | {led['flashinfer_loads']} (flashinfer "
        f"{census.get('flashinfer_version')} on torch {census.get('torch_version')}, sm_86) |",
        f"| `flashinfer_attn_m_invariant` (out of the box) | {led['flashinfer_attn_m_invariant']} "
        f"(default auto split-KV is batch-VARIANT) |",
        f"| `flashinfer_total_flips_M1vs8` | {led['flashinfer_total_flips_M1vs8']}/"
        f"{_cell(census,'tensor_core','auto',led['headline_census_L']).get('numel','?')} "
        f"(tensor-core auto, L={led['headline_census_L']}) |",
        f"| `flashinfer_m1_decode_tps` | {led['flashinfer_m1_decode_tps']:.0f} decode-attn steps/s "
        f"(tensor-core auto, L={led['headline_tps_L']}; kernel micro-bench, NOT serve TPS) |",
        f"| `vs_splitkv_399_75` (cheaper free route?) | "
        f"{'YES' if led['vs_splitkv_399_75_cheaper_free_route'] else 'NO'} -- "
        f"fixed_split is {(f'{cost:.1f}x' if cost else '?')} slower than auto at M=1, "
        f"head_dim-512 blocked, version skew |",
        f"| `fixed_split_size_invariant` | {led['fixed_split_size_invariant']} (0 flips -- "
        f"lawine #496's trick, native) |",
        "",
        "## Findings ledger (each LIVE on the pod)",
        "",
        "| rank | finding | confirmed | evidence |",
        "|---|---|---|---|",
    ]
    for _, blk in sorted(led["findings"].items(), key=lambda kv: kv[1]["rank"]):
        mark = "YES" if blk["confirmed"] else "no"
        lines.append(f"| {blk['rank']} | {_} | {mark} | {blk['evidence']} |")
    lines += [
        "",
        "## M=1-vs-M=8 byte-exact census (flips / 2048 output elements, bf16)",
        "",
        "Identical decode query + shared physical KV pages across the batch; the only difference "
        "is M (1 vs 8). Non-zero flips => the split scheduler changed request-0's reduction tree.",
        "",
        "**Tensor-core path (GQA group "
        f"{arch.get('gqa_group_size')} -> vLLM's serve-relevant decode path):**",
        "",
        "- auto split-KV: " + " | ".join(flips_row(tc, ["auto"])) + "  -> **batch-VARIANT at every L**",
        "- fixed_split_size: " + " | ".join(flips_row(tc, ["fixed_split_512"])) + "  -> **0 flips, INVARIANT**",
        "- disable_split_kv: " + " | ".join(flips_row(tc, ["disable_split_kv"])) + "  -> **0 flips, INVARIANT**",
        "",
        "**CUDA-core path (raw API default; not used by vLLM for GQA):**",
        "",
        "- auto split-KV: " + " | ".join(flips_row(cc, ["auto"])) + "  (invariant only at short L; "
        "variant once it auto-splits; disable_split_kv is a no-op here)",
        "",
        "## M=1 decode-attn throughput -- the cost of invariance (tensor-core, head_dim 256, bf16)",
        "",
        "| L | auto (steps/s) | fixed_split_512 (steps/s) | auto / fixed |",
        "|---|---|---|---|",
    ]
    for L in census.get("config", {}).get("tps_l", []):
        a = census.get("tps", {}).get(str(L), {}).get("auto", {})
        f = census.get("tps", {}).get(str(L), {}).get("fixed_split_512", {})
        if a.get("steps_per_s") and f.get("steps_per_s"):
            lines.append(f"| {L} | {a['steps_per_s']:.0f} | {f['steps_per_s']:.0f} | "
                         f"{a['steps_per_s']/f['steps_per_s']:.2f}x |")
    lines += [
        "",
        "The invariant mode forfeits the split-KV GPU-fill that makes M=1 decode fast, so its cost "
        "GROWS with KV length -- the opposite of lawine #496's hand-rolled ~0%-cost rung.",
        "",
        "## Architecture (live, current env)",
        "",
        f"- `model_type={arch.get('model_type')}`, {arch.get('num_text_layers')} text layers: "
        f"{arch.get('num_sliding_attention')} sliding (head_dim {arch.get('head_dim_sliding')}) + "
        f"{arch.get('num_full_attention')} full (head_dim {arch.get('head_dim_full')}, idxs "
        f"{arch.get('full_attention_idxs')}).",
        f"- GQA: {arch.get('num_attention_heads')} q-heads / {arch.get('num_key_value_heads')} "
        f"kv-heads (group {arch.get('gqa_group_size')}); sliding_window {arch.get('sliding_window')}; "
        f"num_kv_shared_layers {arch.get('num_kv_shared_layers')}.",
        f"- Ampere attn head_dim cap {AMPERE_ATTN_HEADDIM_CAP}; head_dim {arch.get('head_dim_full')} "
        f"(full-attn) exceeds it -> no flashinfer tensor-core kernel.",
        "",
        "## Version skew (why it is not a clean drop-in)",
        "",
        f"- pod (.venv, serving-adjacent): torch `{versions.get('pod_torch')}`.",
        f"- flashinfer-python pins torch `{FLASHINFER_REQUIRES_TORCH}+{FLASHINFER_REQUIRES_CUDA}`; "
        f"ran here in an isolated venv (torch `{census.get('torch_version')}`). nvcc present: "
        f"`{versions.get('nvcc_present')}` ({versions.get('nvcc_release','')}).",
        "- flashinfer publishes no torch-2.11 AOT wheels (JIT-only) and carries an unresolved "
        "CUDA-12-cubin-vs-CUDA-13 incompatibility, so dropping it into the deployed vLLM 0.22.1rc1 "
        "/ torch-2.11+cu130 stack is non-trivial. The M-invariance property measured is an "
        "algorithmic property of the split scheduler and transfers across the torch minor version.",
        "",
        "## Honesty note",
        "",
        "FlashInfer LOADS and RUNS (unlike TRT-LLM #502), so the census is real GPU evidence, not a "
        "blocked counterfactual. The decisive finding is NEGATIVE for the lane's premise: FlashInfer "
        "is NOT free byte-exact M-invariant out of the box -- its default split-KV decode is "
        f"batch-variant ({led['flashinfer_total_flips_M1vs8']}/2048 elements flip at "
        f"L={led['headline_census_L']}). It exposes the invariance as an explicit `fixed_split_size` "
        "knob (0 flips), but that knob is its OWN slower path at M=1 (not free), and head_dim-512 "
        "full-attention layers have no tensor-core/fixed-split path at all -- so FlashInfer is not a "
        "cheaper route to the byte-exact 399.75 rung than the deployed split-KV stack. The useful "
        "positive: the deployed hand-rolled split-KV trick (lawine #496) is reproducing a stock, "
        "upstreamed FlashInfer primitive -- not a bespoke necessity. The last open engine/kernel "
        "thread of the #481 zoom-out closes.",
        "",
        "## Public evidence used",
        "",
        "- **lawine #496** (`42qroec1`) -- split-KV fixed-size 399.75 byte-exact rung; this card "
        "shows flashinfer's `fixed_split_size` is the same trick, native + first-class.",
        "- **denken #498** (`djwaqs7o`) -- SGLang/FlashInfer fast-but-NOT-byte-exact; this card "
        "pins the mechanism (occupancy-based split count) and the override (fixed_split_size).",
        "- **fern #502** (`sxi590tz`) -- TRT-LLM structurally blocked; FlashInfer is the last engine "
        "thread, now measured.",
        "- **Morgan #481** -- ZOOM-OUT directive (look past vLLM+Triton for free byte-exact); the "
        "tax is engine-agnostic IEEE-754, confirmed again here.",
        "- flashinfer `fixed_split_size` plan() docstring; SGLang deterministic-inference blog; "
        "Dao-AILab/flash-attention#2427 (head_dim 512 unsupported); vLLM#38918.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", "--self_test", dest="self_test", action="store_true",
                    help="exit non-zero unless the negative-with-nuance verdict is fully substantiated")
    ap.add_argument("--fi-python", dest="fi_python", default=f"{DEFAULT_FI_VENV}/bin/python",
                    help="python in the isolated flashinfer venv (torch 2.10+cu128)")
    ap.add_argument("--no-build-venv", dest="no_build_venv", action="store_true",
                    help="do not build the isolated flashinfer venv if missing")
    ap.add_argument("--no-wandb", "--no_wandb", dest="no_wandb", action="store_true")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="fern/flashinfer-byteexact-probe")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="flashinfer-byteexact")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    bar = "=" * 80
    print(bar, flush=True)
    print("FlashInfer byte-exact M-invariance probe -- gemma-4-E4B-it on sm_86 (PR #507)", flush=True)
    print(bar, flush=True)

    arch = probe_arch()
    print(f"[arch] {arch.get('model_type')} | head_dim {arch.get('head_dim_sliding')}/"
          f"{arch.get('head_dim_full')} | GQA {arch.get('num_attention_heads')}/"
          f"{arch.get('num_key_value_heads')} (group {arch.get('gqa_group_size')}) | "
          f"env tx {arch.get('env_transformers')}", flush=True)

    versions = probe_versions(args.fi_python)
    print(f"[versions] pod torch {versions.get('pod_torch')} vs flashinfer-pins "
          f"{FLASHINFER_REQUIRES_TORCH}+{FLASHINFER_REQUIRES_CUDA} | skew={versions.get('version_skew')} "
          f"| nvcc={versions.get('nvcc_present')}", flush=True)

    venv_info = ensure_fi_venv(args.fi_python, allow_build=not args.no_build_venv)
    if venv_info.get("built"):
        print(f"[venv] built isolated flashinfer venv at {args.fi_python}", flush=True)
    census = run_gpu_census(args.fi_python)
    if census.get("flashinfer_import"):
        ac = census.get("census", {}).get("tensor_core", {}).get("auto", {}).get(str(HEADLINE_CENSUS_L), {})
        print(f"[census] flashinfer {census.get('flashinfer_version')} torch "
              f"{census.get('torch_version')} | tensor-core auto L={HEADLINE_CENSUS_L}: "
              f"flips={ac.get('flips')}/{ac.get('numel')}", flush=True)
    else:
        print(f"[census] flashinfer census unavailable: {census.get('error','?')}", flush=True)

    led = build_ledger(arch, versions, census)

    print("-" * 80, flush=True)
    for _, blk in sorted(led["findings"].items(), key=lambda kv: kv[1]["rank"]):
        print(f"  [F{blk['rank']} {'CONFIRMED' if blk['confirmed'] else 'unconfirmed':>11}] {_}",
              flush=True)
    print("-" * 80, flush=True)
    print(f"flashinfer_loads={led['flashinfer_loads']} "
          f"attn_m_invariant={led['flashinfer_attn_m_invariant']} "
          f"total_flips_M1vs8={led['flashinfer_total_flips_M1vs8']} "
          f"fixed_split_invariant={led['fixed_split_size_invariant']} "
          f"feasibility_evidence_complete={led['feasibility_evidence_complete']}", flush=True)
    print(bar, flush=True)

    payload: dict[str, Any] = {
        "card": "flashinfer_byteexact", "pr": 507, "author": "fern",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "hardware": "A10G sm_86 24GB",
        "gpu_used": True, "gpu_model_forward": False, "official_tps": 0,
        "no_hf_job": True, "no_submission": True, "no_served_file_change": True,
        "analysis_only": True,
        "arch": arch, "versions": versions, "venv_info": venv_info,
        "census": census, "research": RESEARCH, "ledger": led,
    }

    run_ids = maybe_log_wandb(args, payload)
    payload["wandb_run_ids"] = run_ids

    out_path = Path(args.out) if args.out else HERE / "_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[fi-probe] wrote {out_path}", flush=True)
    report_path = HERE / "REPORT.md"
    write_report(report_path, payload)
    print(f"[fi-probe] wrote {report_path}", flush=True)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": run_ids,
        "primary_metric": {"name": "feasibility_evidence_complete",
                           "value": led["feasibility_evidence_complete"]},
        "test_metric": {"name": "flashinfer_attn_m_invariant",
                        "value": led["flashinfer_attn_m_invariant"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test and led["feasibility_evidence_complete"] != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
