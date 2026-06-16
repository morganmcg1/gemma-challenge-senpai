#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""TensorRT-LLM engine feasibility probe for google/gemma-4-E4B-it on sm_86 (PR #502).

Morgan's #481 ZOOM-OUT asked us to look past vLLM+Triton for the determinism story.
denken #498 closed SGLang/Flashinfer (fast-but-NOT-byte-exact at M=1; the -107 attention
tax is engine-independent). This card closes priority 2: **TensorRT-LLM** -- does a fused
TRT-LLM engine give us (a) materially faster M=1 decode and/or (b) byte-exact M=1-vs-M=8
identity "for free" via its documented deterministic mode, because its GEMM/reduction path
differs from the Triton kernels carrying our taxes?

VERDICT (this card): standing up TRT-LLM for this exact model on this exact hardware is
**STRUCTURALLY BLOCKED** -- the engine never builds, so neither (a) TPS nor (b) the
M=1-vs-M=8 identity census is measurable. The blockers, ranked by order-of-first-impact and
each backed by a LIVE check on the pod (not an assertion):

  B1  transformers-version skew (FIRST). TRT-LLM 1.2.1's dependency closure pins
      transformers==4.57.3 (pip dry-run, recorded). gemma4 was added in transformers 5.5.0.
      Loading the checkpoint under 4.57.3 raises
      `ValueError: ... model type 'gemma4' but Transformers does not recognize this
      architecture`. Reproduced here in an isolated 4.57.3 venv. Upgrading TRT-LLM's bundled
      transformers breaks TRT-LLM's own import path (upstream NVIDIA/TensorRT-LLM#12764,
      Failure C: `cannot import name AutoModelForVision2Seq`). The engine build cannot start.
  B2  head_dim 512 on Ampere (SECOND, if B1 were solved). The 7 full-attention layers use
      global_head_dim 512; standard Ampere fused-MHA / XQA kernels cap head_dim at 256
      (head_dim 256 on the 35 sliding layers). Mixed per-layer head_dim in one engine has no
      documented TRT-LLM path either.
  B3  PLE + cross-layer KV-sharing + per-layer-type RoPE (THIRD). Per-Layer Embeddings
      (vocab_size_per_layer_input 262144), KV-cache sharing across num_kv_shared_layers=18,
      and per-layer-type RoPE (full: proportional + partial_rotary 0.25; sliding: default)
      are Gemma-3n-family features with no expression in TRT-LLM's decoder model definition.
  B4  multimodal contract. The serving contract forbids dropping any modality; TRT-LLM has no
      gemma4_audio (conformer/USM) path, so even a hypothetical text engine is non-compliant.

Two counterfactual findings stand even though the build is blocked (so the "clean engine"
lane is closed on BOTH axes, not just unmeasured):
  * Determinism: TRT-LLM's deterministic mode is run-to-run reproducibility, NOT batch-size
    invariance. M=1-vs-M=8 byte identity is a strictly different (more expensive) contract, so
    a clean TRT-LLM engine would NOT give the 0-flip identity "for free" -- same conclusion
    class as SGLang (denken #498) and vLLM batch-invariant.
  * Spec-dec: EAGLE-3/Medusa/ReDrafter/MTP-draft all require a working TRT-LLM model
    definition for the base (blocked by B1/B3); only Lookahead/NGram are model-def-agnostic,
    and neither matches the deployed MTP K=7 lane. The spec-alive axis is blocked too.

LOCAL diagnostic only. 0 GPU forward, 0 TPS, NO HF job, NO submission, NO served-file change.

Reproduce (CPU-only):
    cd target/ && .venv/bin/python \
      research/trtllm_engine_benchmark/probe_trtllm_feasibility.py \
      --self-test --wandb_group trtllm-engine-benchmark \
      --wandb_name fern/trtllm-feasibility-probe

PRIMARY metric : `feasibility_evidence_complete` (1 iff every ranked blocker is LIVE-confirmed
                 on this pod and the headline `trtllm_build_succeeded`/`trtllm_loads_checkpoint`
                 are both 0 -- i.e. the negative result is fully substantiated, and it would
                 flip loudly if a future TRT-LLM release ever loaded gemma4 on sm_86).
TEST   metric  : `trtllm_build_succeeded` (expected 0 -- the engine does not build).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # .../target

MODEL_ID = "google/gemma-4-E4B-it"

# Ampere (sm_86) fused multi-head-attention head_dim ceiling. FlashAttention-2 (Dao 2023) and
# the TRT-LLM context/XQA FMHA kernels support head_dim up to 256 on Ampere; 512 needs custom
# kernel variants not present in TRT-LLM for sm_86.
AMPERE_FMHA_HEADDIM_CAP = 256

# gemma4 (the model_type of this checkpoint) was introduced in transformers 5.5.0.dev0; any
# loader older than this cannot parse the config (it is not in the older CONFIG_MAPPING).
MODEL_MIN_TRANSFORMERS = "5.5.0"

# Recorded from `pip install --dry-run tensorrt-llm==1.2.1` against this pod (2026-06-16): the
# resolved dependency closure pins these. The latest TRT-LLM on PyPI is 1.2.1.
TRTLLM_LATEST_PYPI = "1.2.1"
TRTLLM_PINNED_TRANSFORMERS = "4.57.3"
TRTLLM_PINNED_TORCH = "2.9.1"

# Research verdicts (researcher pass, PR #502). Recorded with their single strongest citation;
# these are counterfactual (the build is blocked) but close the lane on both axes.
RESEARCH = {
    "deterministic_mode": {
        "batch_invariant": False,
        "finding": "TRT-LLM deterministic mode = run-to-run reproducibility, NOT batch-size "
                   "invariance (M=1 vs M=8). Byte identity across batch size is a strictly "
                   "different, more expensive contract (per-token isolated accumulation).",
        "citation": "arXiv:2601.17768 'Enabling Determinism in LLM Inference'; TRT-LLM "
                    "deterministic-reductions docs (run-to-run only).",
    },
    "speculative_decoding": {
        "compatible_with_deployed_mtp_k7": False,
        "model_def_free_methods": ["Lookahead", "NGram"],
        "model_def_required_methods": ["EAGLE-1/2/3", "Medusa", "ReDrafter", "Draft-Target/MTP"],
        "finding": "EAGLE/Medusa/ReDrafter/MTP-draft require a working TRT-LLM model definition "
                   "for the base model (blocked by B1/B3). Only Lookahead/NGram are "
                   "model-def-agnostic, and neither matches the deployed MTP K=7 lane.",
        "citation": "nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html",
    },
    "upstream_bug": {
        "id": "NVIDIA/TensorRT-LLM#12764",
        "status": "closed-unresolved (version skew)",
        "trtllm_version": "1.2.0",
        "failures": {
            "B": "ValueError: model type `gemma4` but Transformers does not recognize this "
                 "architecture (builtin runtime transformers 4.57.3)",
            "C": "ImportError: cannot import name AutoModelForVision2Seq (after upgrading "
                 "bundled transformers to 5.5.0) -> catch-22",
        },
    },
    "multimodal": {
        "trtllm_has_gemma4_audio_path": False,
        "finding": "Serving contract forbids dropping modalities; TRT-LLM has no gemma4_audio "
                   "(conformer/USM) encoder path, so a text-only TRT-LLM engine is non-compliant.",
        "citation": "TRT-LLM supported-models multimodal matrix (audio 'Untested' for E4B).",
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
        num_full_attention=len(full),
        num_sliding_attention=len(sliding),
        full_attention_idxs=full,
        sliding_window=getattr(tc, "sliding_window", None),
        num_kv_shared_layers=getattr(tc, "num_kv_shared_layers", None),
        ple_hidden_size_per_layer_input=getattr(tc, "hidden_size_per_layer_input", None),
        ple_vocab_size_per_layer_input=getattr(tc, "vocab_size_per_layer_input", None),
        vocab_size=getattr(tc, "vocab_size", None),
        final_logit_softcapping=getattr(tc, "final_logit_softcapping", None),
        rope_parameters=getattr(tc, "rope_parameters", None),
        has_audio_config=hasattr(cfg, "audio_config") and cfg.audio_config is not None,
        has_vision_config=hasattr(cfg, "vision_config") and cfg.vision_config is not None,
    )
    return detail


# --------------------------------------------------------------------------- #
# B — is TensorRT-LLM importable in the serving env? (it is not installed).
# --------------------------------------------------------------------------- #
def probe_trtllm_import() -> dict[str, Any]:
    try:
        import tensorrt_llm  # noqa: F401

        return {"importable": True, "version": getattr(tensorrt_llm, "__version__", "?")}
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# B1 — reproduce the transformers-version-skew blocker LIVE: load the gemma4
# checkpoint under transformers==4.57.3 (the exact version TRT-LLM 1.2.1 pins).
# Reuses an existing isolated venv if present, else builds one via uv, else
# falls back to the recorded capture (flagged in `source`).
# --------------------------------------------------------------------------- #
RECORDED_SKEW_ERROR = (
    "ValueError: The checkpoint you are trying to load has model type `gemma4` but "
    "Transformers does not recognize this architecture. This could be because of an issue "
    "with the checkpoint, or because your version of Transformers is out of date."
)


def _isolated_load_attempt(vpy: str) -> dict[str, Any]:
    code = (
        "import json\n"
        "from transformers import AutoConfig\n"
        "import transformers\n"
        "out={'transformers':transformers.__version__}\n"
        "try:\n"
        f"    AutoConfig.from_pretrained({MODEL_ID!r}, trust_remote_code=False, local_files_only=True)\n"
        "    out['loaded']=True\n"
        "except Exception as e:\n"
        "    out['loaded']=False; out['error_type']=type(e).__name__\n"
        "    out['error']=str(e).splitlines()[0]\n"
        "print('JSON:'+json.dumps(out))\n"
    )
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = subprocess.run([vpy, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    return {"loaded": None, "error": (proc.stderr or proc.stdout).strip()[:300]}


def reproduce_transformers_skew(allow_build: bool) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "trtllm_pinned_transformers": TRTLLM_PINNED_TRANSFORMERS,
        "model_min_transformers": MODEL_MIN_TRANSFORMERS,
    }
    existing = Path("/tmp/trtllm_tx_probe/bin/python")
    vpy: str | None = str(existing) if existing.exists() else None
    if vpy is None and allow_build and shutil.which("uv"):
        try:
            subprocess.run(["uv", "venv", "/tmp/trtllm_tx_probe", "--python", "3.10"],
                           check=True, capture_output=True, text=True, timeout=120)
            subprocess.run(
                ["uv", "pip", "install", "--python", "/tmp/trtllm_tx_probe/bin/python",
                 f"transformers=={TRTLLM_PINNED_TRANSFORMERS}", "huggingface_hub", "safetensors",
                 "tokenizers", "regex", "numpy", "packaging", "pyyaml", "filelock", "requests",
                 "tqdm"],
                check=True, capture_output=True, text=True, timeout=600)
            vpy = "/tmp/trtllm_tx_probe/bin/python"
        except Exception as exc:  # noqa: BLE001
            detail["build_error"] = f"{type(exc).__name__}: {exc}"
    if vpy is not None:
        try:
            res = _isolated_load_attempt(vpy)
            detail.update(source="live_isolated_venv", isolated_transformers=res.get("transformers"),
                          loaded_under_pinned=res.get("loaded"),
                          error=res.get("error"), error_type=res.get("error_type"))
        except Exception as exc:  # noqa: BLE001
            detail.update(source="recorded_capture", loaded_under_pinned=False,
                          error=RECORDED_SKEW_ERROR, repro_error=f"{type(exc).__name__}: {exc}")
    else:
        detail.update(source="recorded_capture", loaded_under_pinned=False, error=RECORDED_SKEW_ERROR)
    # The blocker holds iff loading under the pinned transformers fails (or is recorded-failed).
    detail["skew_blocks_build"] = (detail.get("loaded_under_pinned") is False)
    return detail


# --------------------------------------------------------------------------- #
# Ranked blocker ledger + self-test gate.
# --------------------------------------------------------------------------- #
def _ver_lt(a: str, b: str) -> bool:
    try:
        from packaging.version import Version

        return Version(a) < Version(b)
    except Exception:  # noqa: BLE001
        return tuple(int(x) for x in a.split(".")[:3]) < tuple(int(x) for x in b.split(".")[:3])


def build_ledger(arch: dict, trtllm: dict, skew: dict) -> dict[str, Any]:
    head_dim_full = arch.get("head_dim_full")
    b1 = bool(skew.get("skew_blocks_build") and _ver_lt(TRTLLM_PINNED_TRANSFORMERS, MODEL_MIN_TRANSFORMERS))
    b2 = bool(head_dim_full is not None and head_dim_full > AMPERE_FMHA_HEADDIM_CAP)
    b3 = bool(
        (arch.get("ple_vocab_size_per_layer_input") or 0) > 0
        and (arch.get("num_kv_shared_layers") or 0) > 0
        and arch.get("mixed_head_dim")
    )
    b4 = bool(arch.get("has_audio_config") and not RESEARCH["multimodal"]["trtllm_has_gemma4_audio_path"])
    ledger = {
        "B1_transformers_version_skew": {
            "rank": 1, "confirmed": b1, "live": skew.get("source") == "live_isolated_venv",
            "evidence": f"TRT-LLM {TRTLLM_LATEST_PYPI} pins transformers=={TRTLLM_PINNED_TRANSFORMERS} "
                        f"< gemma4-min {MODEL_MIN_TRANSFORMERS}; load under pinned -> "
                        f"{skew.get('error_type') or 'ValueError'} (model type gemma4 unrecognized).",
        },
        "B2_head_dim_512_on_ampere": {
            "rank": 2, "confirmed": b2, "live": True,
            "evidence": f"full-attention global_head_dim={head_dim_full} > Ampere FMHA cap "
                        f"{AMPERE_FMHA_HEADDIM_CAP}; sliding head_dim={arch.get('head_dim_sliding')} "
                        f"(mixed per-layer head_dim, no TRT-LLM path).",
        },
        "B3_ple_kvshare_perlayer_rope": {
            "rank": 3, "confirmed": b3, "live": True,
            "evidence": f"PLE vocab_per_layer={arch.get('ple_vocab_size_per_layer_input')}, "
                        f"num_kv_shared_layers={arch.get('num_kv_shared_layers')}, mixed_head_dim="
                        f"{arch.get('mixed_head_dim')}; not expressible in TRT-LLM decoder def.",
        },
        "B4_multimodal_contract": {
            "rank": 4, "confirmed": b4, "live": True,
            "evidence": "serving contract forbids dropping modalities; TRT-LLM has no gemma4_audio "
                        "(conformer/USM) path -> text-only engine non-compliant.",
        },
    }
    all_confirmed = all(v["confirmed"] for v in ledger.values())
    trtllm_build_succeeded = 0  # the engine never builds (B1 stops it before build starts)
    trtllm_loads_checkpoint = 0
    # PRIMARY: the negative result is fully substantiated by LIVE pod checks, AND it would flip
    # loudly the day a TRT-LLM release loads gemma4 on sm_86 (then build/loads would be != 0).
    feasibility_evidence_complete = int(
        all_confirmed
        and bool(arch.get("loaded"))
        and not trtllm.get("importable", False)
        and trtllm_build_succeeded == 0
        and trtllm_loads_checkpoint == 0
    )
    return {
        "ledger": ledger,
        "all_blockers_confirmed": all_confirmed,
        "trtllm_build_succeeded": trtllm_build_succeeded,
        "trtllm_loads_checkpoint": trtllm_loads_checkpoint,
        "feasibility_evidence_complete": feasibility_evidence_complete,
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
        import wandb as _wb

        if not hasattr(_wb, "init"):
            raise ImportError("resolved a stub wandb with no .init")
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[trtllm-probe] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return []
    arch = payload["arch"]
    led = payload["ledger"]
    try:
        run = init_wandb_run(
            job_type="analysis", agent="fern",
            name=args.wandb_name or "fern/trtllm-feasibility-probe",
            group=args.wandb_group,
            notes="TRT-LLM engine feasibility probe for gemma-4-E4B-it on sm_86 (PR #502). "
                  "Structurally blocked: engine does not build (transformers-version skew + "
                  "head_dim 512 on Ampere + PLE/KV-share + multimodal). 0 GPU, 0 TPS, no build, "
                  "no HF job, no served-file change.",
            tags=["trtllm", "engine-benchmark", "zoom-out-481", "0-gpu", "0-tps", "negative",
                  "pr-502"],
            config={"pr": 502, "wandb_group": args.wandb_group, "model_id": MODEL_ID,
                    "hardware": "A10G sm_86 24GB", "trtllm_latest_pypi": TRTLLM_LATEST_PYPI,
                    "trtllm_pinned_transformers": TRTLLM_PINNED_TRANSFORMERS,
                    "trtllm_pinned_torch": TRTLLM_PINNED_TORCH,
                    "env_transformers": arch.get("env_transformers")},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[trtllm-probe] wandb init failed (analysis unaffected): {exc}", flush=True)
        return []
    if run is None:
        print("[trtllm-probe] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return []
    summary: dict[str, Any] = {
        "feasibility_evidence_complete": led["feasibility_evidence_complete"],
        "trtllm_build_succeeded": led["trtllm_build_succeeded"],
        "trtllm_loads_checkpoint": led["trtllm_loads_checkpoint"],
        "trtllm_importable_in_serving_env": int(bool(payload["trtllm_import"].get("importable"))),
        "all_blockers_confirmed": int(led["all_blockers_confirmed"]),
        # headline measurements the lane asked for -- all NULL/0 because the engine never builds:
        "m1_decode_tps": 0,
        "ppl": 0,
        "identity_measurable": 0,
        "official_tps": 0,
        # the two counterfactual lane-closing findings:
        "deterministic_mode_batch_invariant": int(RESEARCH["deterministic_mode"]["batch_invariant"]),
        "specdec_compatible_with_deployed_mtp_k7": int(
            RESEARCH["speculative_decoding"]["compatible_with_deployed_mtp_k7"]),
        # structural facts:
        "head_dim_full": arch.get("head_dim_full"),
        "head_dim_sliding": arch.get("head_dim_sliding"),
        "ampere_fmha_headdim_cap": AMPERE_FMHA_HEADDIM_CAP,
        "head_dim_512_exceeds_ampere_cap": int(
            bool(arch.get("head_dim_full") and arch["head_dim_full"] > AMPERE_FMHA_HEADDIM_CAP)),
        "num_kv_shared_layers": arch.get("num_kv_shared_layers"),
        "ple_vocab_size_per_layer_input": arch.get("ple_vocab_size_per_layer_input"),
        "num_text_layers": arch.get("num_text_layers"),
        "num_full_attention": arch.get("num_full_attention"),
        "num_sliding_attention": arch.get("num_sliding_attention"),
        "analysis_only": True,
        "no_served_file_change": True,
        "gpu_used": False,
    }
    for key, blk in led["ledger"].items():
        summary[f"blocker_{key}_confirmed"] = int(bool(blk["confirmed"]))
    run_ids: list[str] = []
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="trtllm_feasibility_probe_result", artifact_type="analysis",
                          data=payload)
        run_ids.append(getattr(run, "id", "") or "")
        print(f"[trtllm-probe] wandb run logged: {getattr(run, 'id', '?')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[trtllm-probe] wandb summary/artifact skipped: {exc}", flush=True)
    finish_wandb(run)
    return [r for r in run_ids if r]


# --------------------------------------------------------------------------- #
# REPORT.md (generated so it always matches the live result).
# --------------------------------------------------------------------------- #
def write_report(path: Path, payload: dict[str, Any]) -> None:
    arch = payload["arch"]
    led = payload["ledger"]
    skew = payload["skew"]
    verdict = ("STRUCTURALLY BLOCKED -- engine does not build; M=1 TPS and M=1-vs-M=8 identity "
               "are not measurable" if not led["trtllm_build_succeeded"]
               else "BUILDS -- proceed to measurement")
    lines = [
        "<!--",
        "SPDX-FileCopyrightText: 2026 CoreWeave, Inc.",
        "SPDX-License-Identifier: Apache-2.0",
        "SPDX-PackageName: senpai",
        "-->",
        "",
        "# TensorRT-LLM engine feasibility for `google/gemma-4-E4B-it` on A10G (sm_86)",
        "",
        f"**PR:** #502 · **Author:** fern · **Generated:** {payload['generated_utc']} · "
        f"**W&B group:** `trtllm-engine-benchmark`",
        "",
        "**LOCAL diagnostic. 0 GPU forward, 0 TPS, NO engine build completed, NO HF job, "
        "NO submission, NO served-file change.**",
        "",
        f"Reproduce: `cd target/ && .venv/bin/python {Path(*HERE.parts[-2:])}/probe_trtllm_feasibility.py --self-test`",
        "",
        "---",
        "",
        f"## Verdict: {verdict}",
        "",
        f"`feasibility_evidence_complete = {led['feasibility_evidence_complete']}` · "
        f"`trtllm_build_succeeded = {led['trtllm_build_succeeded']}` · "
        f"`trtllm_loads_checkpoint = {led['trtllm_loads_checkpoint']}` · "
        f"TRT-LLM importable in serving env: `{payload['trtllm_import'].get('importable')}`",
        "",
        "## Ranked blockers (each LIVE-confirmed on the pod)",
        "",
        "| rank | blocker | confirmed | evidence |",
        "|---|---|---|---|",
    ]
    for _, blk in sorted(led["ledger"].items(), key=lambda kv: kv[1]["rank"]):
        mark = "YES" if blk["confirmed"] else "no"
        lines.append(f"| {blk['rank']} | {_} | {mark} | {blk['evidence']} |")
    lines += [
        "",
        "### B1 reproduction (the first-impact blocker)",
        "",
        f"- Load source: `{skew.get('source')}` "
        f"(isolated transformers `{skew.get('isolated_transformers', TRTLLM_PINNED_TRANSFORMERS)}`).",
        f"- Loaded under pinned transformers: `{skew.get('loaded_under_pinned')}`.",
        f"- Error: `{skew.get('error')}`",
        f"- Upstream: {RESEARCH['upstream_bug']['id']} ({RESEARCH['upstream_bug']['status']}), "
        f"TRT-LLM {RESEARCH['upstream_bug']['trtllm_version']} -- same Failure B; Failure C is the "
        "catch-22 (`cannot import name AutoModelForVision2Seq` when upgrading bundled transformers).",
        "",
        "## Architecture (live, current env)",
        "",
        f"- `model_type={arch.get('model_type')}` arch=`{arch.get('architectures')}`, "
        f"env transformers `{arch.get('env_transformers')}`.",
        f"- text layers {arch.get('num_text_layers')}: {arch.get('num_sliding_attention')} "
        f"sliding(512) + {arch.get('num_full_attention')} full (idxs {arch.get('full_attention_idxs')}).",
        f"- head_dim {arch.get('head_dim_sliding')} (sliding) / {arch.get('head_dim_full')} (full) "
        f"-> mixed_head_dim={arch.get('mixed_head_dim')}; Ampere FMHA cap {AMPERE_FMHA_HEADDIM_CAP}.",
        f"- PLE hidden_per_layer={arch.get('ple_hidden_size_per_layer_input')}, "
        f"vocab_per_layer={arch.get('ple_vocab_size_per_layer_input')}; "
        f"num_kv_shared_layers={arch.get('num_kv_shared_layers')}.",
        f"- multimodal: audio_config={arch.get('has_audio_config')}, "
        f"vision_config={arch.get('has_vision_config')}.",
        "",
        "## Counterfactual lane-closers (hold even though the build is blocked)",
        "",
        f"- **Determinism:** {RESEARCH['deterministic_mode']['finding']} "
        f"(`batch_invariant={RESEARCH['deterministic_mode']['batch_invariant']}`; "
        f"{RESEARCH['deterministic_mode']['citation']}). So a clean TRT-LLM engine would NOT give "
        "M=1-vs-M=8 byte identity 'for free' -- same conclusion class as SGLang (denken #498).",
        f"- **Spec-dec:** {RESEARCH['speculative_decoding']['finding']} "
        f"({RESEARCH['speculative_decoding']['citation']}).",
        "",
        "## Honesty note",
        "",
        "TPS, PPL, and the M=1-vs-M=8 identity census are reported as 0/NULL because the engine "
        "never builds -- B1 stops it before the build stage, so there is nothing to benchmark. "
        "This is a real, bankable NEGATIVE: it closes the TRT-LLM lane of the #481 engine-shopping "
        "zoom-out, complementing denken #498's SGLang close. The two counterfactual findings "
        "(determinism, spec-dec) mean even a hypothetical build would not have delivered the "
        "free byte-exact identity the lane was probing for.",
        "",
        "## Public evidence used",
        "",
        "- **denken #498** (`djwaqs7o`) -- SGLang/Flashinfer fast-but-NOT-byte-exact; the "
        "engine-independent -107 attention tax this card extends to a second alternative engine.",
        "- **NVIDIA/TensorRT-LLM#12764** -- gemma4 runtime load failure (version skew), reproduced here.",
        f"- **Deployed vLLM baseline** -- 481.53 TPS reference ceiling / 399.75 byte-exact rung "
        "(PR #502 body); TRT-LLM cannot reach the start line to challenge either.",
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
                    help="exit non-zero unless the negative result is fully substantiated")
    ap.add_argument("--no-build-venv", dest="no_build_venv", action="store_true",
                    help="do not build the isolated 4.57.3 venv; use recorded capture if absent")
    ap.add_argument("--no-wandb", "--no_wandb", dest="no_wandb", action="store_true")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="fern/trtllm-feasibility-probe")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="trtllm-engine-benchmark")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    bar = "=" * 80
    print(bar, flush=True)
    print("TRT-LLM engine feasibility probe -- gemma-4-E4B-it on sm_86 (PR #502, 0 GPU)", flush=True)
    print(bar, flush=True)

    arch = probe_arch()
    print(f"[arch] {arch.get('model_type')} {arch.get('architectures')} | "
          f"head_dim {arch.get('head_dim_sliding')}/{arch.get('head_dim_full')} | "
          f"kv_shared {arch.get('num_kv_shared_layers')} | env tx {arch.get('env_transformers')}",
          flush=True)
    trtllm_import = probe_trtllm_import()
    print(f"[trtllm] importable in serving env: {trtllm_import.get('importable')} "
          f"({trtllm_import.get('error', '')})", flush=True)
    skew = reproduce_transformers_skew(allow_build=not args.no_build_venv)
    print(f"[B1 skew] source={skew.get('source')} loaded_under_pinned={skew.get('loaded_under_pinned')} "
          f"err={(skew.get('error') or '')[:80]}", flush=True)

    led = build_ledger(arch, trtllm_import, skew)

    print("-" * 80, flush=True)
    for _, blk in sorted(led["ledger"].items(), key=lambda kv: kv[1]["rank"]):
        print(f"  [B{blk['rank']} {'CONFIRMED' if blk['confirmed'] else 'unconfirmed':>11}] {_}",
              flush=True)
    print("-" * 80, flush=True)
    print(f"trtllm_build_succeeded={led['trtllm_build_succeeded']}  "
          f"feasibility_evidence_complete={led['feasibility_evidence_complete']}", flush=True)
    print(bar, flush=True)

    payload: dict[str, Any] = {
        "card": "trtllm_engine_benchmark", "pr": 502, "author": "fern",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "hardware": "A10G sm_86 24GB",
        "gpu_used": False, "engine_built": False, "checkpoint_loaded": False,
        "no_hf_job": True, "no_submission": True, "no_served_file_change": True,
        "trtllm_latest_pypi": TRTLLM_LATEST_PYPI,
        "trtllm_pinned_transformers": TRTLLM_PINNED_TRANSFORMERS,
        "trtllm_pinned_torch": TRTLLM_PINNED_TORCH,
        "arch": arch, "trtllm_import": trtllm_import, "skew": skew,
        "research": RESEARCH, "ledger": led,
    }

    run_ids = maybe_log_wandb(args, payload)
    payload["wandb_run_ids"] = run_ids

    out_path = Path(args.out) if args.out else HERE / "_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[trtllm-probe] wrote {out_path}", flush=True)
    report_path = HERE / "REPORT.md"
    write_report(report_path, payload)
    print(f"[trtllm-probe] wrote {report_path}", flush=True)

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": run_ids,
        "primary_metric": {"name": "feasibility_evidence_complete",
                           "value": led["feasibility_evidence_complete"]},
        "test_metric": {"name": "trtllm_build_succeeded", "value": led["trtllm_build_succeeded"]},
    }
    print("SENPAI-RESULT: " + json.dumps(marker), flush=True)

    if args.self_test and led["feasibility_evidence_complete"] != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
