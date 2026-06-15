#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""GPU measurement leg (PR #385) — served-stack identity ladder S0..S4 on the plain
full-vocab int4 M=1 decode of google/gemma-4-E4B-it-qat-w4a16-ct.

This is the #371 lineage (reuses measure_capture.py's structure + the same pinned
vLLM venv + the same eager (E) byte-exact reference). It ablates, one OFFLINE-reachable
lever at a time, the served-stack levers the 165.44 strict frontier (lawine #196,
submission fa2sw_nonspec_int4) banks, and tags each identity-safe vs identity-breaking:

  S0  capture anchor            mode=0 (no inductor) + cudagraph=FULL          == #371 C (~91)
  S1  + precache                S0 + enable_prefix_caching + prompt-prefill warm
                                (precache = serve_patch_precache: a PUBLIC-only prefix-cache
                                 warmup; offline we toggle vLLM prefix caching + warm prefill)
  S2  + split-KV                S1 + VLLM_ATTENTION_BACKEND=FLASH_ATTN (the FA2 path the
                                served FA_SLIDING routes sliding-window layers onto). NB the
                                named SPLITKV_VERIFY spec-lever is INERT on the M=1 non-spec
                                base (it routes only 1<q<=64 spec-verify steps; there are none).
  S3  + weight-residency        S2-base + served launch config: gpu_memory_utilization=0.90,
                                max_num_batched_tokens=512 (the fa2sw_nonspec_int4 manifest
                                values). residency is a cudagraph side-effect already in S0;
                                this measures the served launch-config delta.
  S4  + fusion                  S3 + inductor fusion (mode=3) == #371 F (~97) — the IDENTITY
                                BREAKER (#371: token_identity_rate 0.0703).

The identity reference is the SAME eager (E) full-vocab int4 plain-AR capture from #371
(decode_E.jsonl): E = no capture, no fusion, full-vocab => the spec-off NON-fused plain
greedy-AR reference for the full-vocab checkpoint. Each Sk's token_identity_rate vs E is
the real (non-self-referential) strict gate. M=1 offline greedy (temperature=0, ignore_eos)
IS plain non-spec AR by construction, so greedy output is seed-invariant: the 128x512
IDENTITY capture is run once per config; the slope decode-TPS probe is run under >=2 seeds
for timing robustness.

Writes, under --out-dir:
  decode_<cfg>.jsonl   official decode-output records (verifier-compatible).
  measure_<cfg>.json   decode TPS (slope, multi-seed), resolved compilation config, the
                       applied served-stack levers, peak GPU mem, load/gen walls.

Run (per config, SERVER venv):
  cd target/ && CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python \
    research/validity/served_stack_identity_ablation/measure_stack.py \
    --config S0 --out-dir research/validity/served_stack_identity_ablation/measured
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Determinism + container shims MUST precede CUDA/vLLM import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.local_validation import paths  # noqa: E402

# Canonical plain full-vocab int4 (= the full-vocab checkpoint vanilla LLM() loads;
# the byte-exact base #371 measured at eager 21.62 / capture 91.38). tie=True, body
# int4 (compressed-tensors W4A16), lm_head/embeddings bf16 FULL-vocab. NB: this is NOT
# the served lmhead12k-pruned osoi5 baked weights (those only load via serve.py's
# meta-path PCK-04 scatter hook); the lmhead12k read-reduction is attributed analytically
# in the synthesis card, not measured here.
DEFAULT_MODEL = (
    "/senpai-run/home/student-ubel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
DEFAULT_PROMPTS_REF = (
    REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/"
    "decode_outputs.offline.jsonl"
)

# The served-stack ladder. Each rung toggles ONE offline-reachable lever on top of the
# capture base. `comp_cfg` is the vLLM compilation_config; `attn_backend` sets
# VLLM_ATTENTION_BACKEND (None=default); `prefix_cache`+`prefill_warm` emulate precache;
# `max_num_batched_tokens`/`gpu_mem` are the served launch config.
CONFIGS: dict[str, dict[str, Any]] = {
    "S0": {
        "name": "S0-capture", "enforce_eager": False,
        "comp_cfg": {"mode": 0, "cudagraph_mode": "FULL"},
        "gpu_mem": 0.85, "prefix_cache": False, "prefill_warm": False,
        "attn_backend": None, "max_num_batched_tokens": None,
        "lever": "capture anchor (== #371 C): mode=0 (no inductor) + cudagraph=FULL",
    },
    "S1": {
        "name": "S1-precache", "enforce_eager": False,
        "comp_cfg": {"mode": 0, "cudagraph_mode": "FULL"},
        "gpu_mem": 0.85, "prefix_cache": True, "prefill_warm": True,
        "attn_backend": None, "max_num_batched_tokens": None,
        "lever": "+ precache: enable_prefix_caching + prompt-prefill warm (serve_patch_precache "
                 "is a PUBLIC-only prefix-cache warmup; identity-neutral, decode-step ~neutral)",
    },
    "S2": {
        "name": "S2-splitkv", "enforce_eager": False,
        "comp_cfg": {"mode": 0, "cudagraph_mode": "FULL"},
        "gpu_mem": 0.85, "prefix_cache": True, "prefill_warm": True,
        "attn_backend": "FLASH_ATTN", "max_num_batched_tokens": None,
        "lever": "+ split-KV: VLLM_ATTENTION_BACKEND=FLASH_ATTN (the FA2 path served FA_SLIDING "
                 "routes to). Named SPLITKV_VERIFY spec-lever is INERT at M=1 non-spec (no verify steps)",
    },
    "S3": {
        "name": "S3-residency", "enforce_eager": False,
        "comp_cfg": {"mode": 0, "cudagraph_mode": "FULL"},
        "gpu_mem": 0.90, "prefix_cache": True, "prefill_warm": True,
        "attn_backend": None, "max_num_batched_tokens": 512,
        "lever": "+ weight-residency / served launch config: gpu_mem 0.90 + max_num_batched_tokens 512 "
                 "(fa2sw_nonspec_int4 manifest); residency itself is a cudagraph side-effect already in S0",
    },
    "S4": {
        "name": "S4-fusion", "enforce_eager": False,
        "comp_cfg": {"mode": 3, "cudagraph_mode": "FULL"},
        "gpu_mem": 0.90, "prefix_cache": True, "prefill_warm": True,
        "attn_backend": None, "max_num_batched_tokens": 512,
        "lever": "+ fusion: inductor mode=3 (== #371 F) — the IDENTITY BREAKER (reduction reorder)",
    },
}

TPS_SEEDS = [1234, 5678]  # >=2 seeds for slope-TPS timing robustness (greedy is seed-invariant)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _device_mem_used_mib() -> float | None:
    """Device-0 memory used (MiB) via nvidia-smi (vLLM V1 engine is a subprocess, so the
    parent torch counter reads 0). Single-tenant pod on device 0 -> honest peak proxy."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def load_prompts(ref_path: Path, n: int) -> list[dict[str, Any]]:
    """Return [{id, prompt_token_ids}] for the first n eval records."""
    out: list[dict[str, Any]] = []
    with ref_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            pt = r.get("prompt_token_ids")
            if not isinstance(pt, list):
                continue
            out.append({"id": str(r["id"]), "prompt_token_ids": [int(t) for t in pt]})
            if len(out) >= n:
                break
    if not out:
        raise ValueError(f"no prompt_token_ids loaded from {ref_path}")
    return out


def resolved_compilation(llm) -> dict[str, Any]:
    try:
        cc = llm.llm_engine.vllm_config.compilation_config
        mode = getattr(cc, "mode", None)
        cache = llm.llm_engine.vllm_config.cache_config
        return {
            "mode": getattr(mode, "value", mode),
            "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
            "backend": getattr(cc, "backend", None),
            "cudagraph_capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", []) or []),
            "enable_prefix_caching": getattr(cache, "enable_prefix_caching", None),
            "attn_backend_env": os.environ.get("VLLM_ATTENTION_BACKEND"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def measure_decode_tps(llm, sampling_cls, probe_token_ids: list[int],
                       *, decode_tokens: int, repeats: int, seeds: list[int]) -> dict[str, Any]:
    """Steady-state single-stream decode TPS via the slope method (harness.probe_tps):
    (N-1)/(wall_N - wall_1). Warmup absorbs capture/compile. Median over `repeats` per
    seed and over all seeds (>=2 seeds for timing robustness; greedy is seed-invariant)."""
    from vllm import TokensPrompt

    def gen(max_tokens: int, seed: int | None) -> float:
        sp = sampling_cls(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                          ignore_eos=True, seed=seed)
        prompt = TokensPrompt(prompt_token_ids=probe_token_ids)
        t0 = time.time()
        llm.generate([prompt], sp, use_tqdm=False)
        return time.time() - t0

    # Warm up at FULL decode length so every shape-specialized kernel (attention across KV
    # block boundaries, sampler, capture/compile) is built BEFORE timing.
    gen(8, None)
    gen(decode_tokens, None)
    per_seed: dict[str, float] = {}
    all_samples: list[float] = []
    for seed in seeds:
        seed_samples = []
        for _ in range(repeats):
            w1 = gen(1, seed)
            wN = gen(decode_tokens, seed)
            if wN > w1:
                seed_samples.append((decode_tokens - 1) / (wN - w1))
        if seed_samples:
            per_seed[str(seed)] = statistics.median(seed_samples)
            all_samples.extend(seed_samples)
    tps = statistics.median(all_samples) if all_samples else float("nan")
    return {
        "decode_tps_single_stream": tps,
        "decode_tps_samples": all_samples,
        "decode_tps_per_seed_median": per_seed,
        "decode_tps_seed_spread": (max(per_seed.values()) - min(per_seed.values())
                                   if len(per_seed) > 1 else 0.0),
        "decode_tokens": decode_tokens,
        "repeats": repeats,
        "seeds": seeds,
        "method": "slope (N-1)/(wall_N - wall_1); harness.probe_tps; multi-seed median",
    }


def run(cfg_key: str, args) -> int:
    cfg = CONFIGS[cfg_key]
    # Attention-backend lever must be set before vLLM reads it at engine init.
    if cfg["attn_backend"]:
        os.environ["VLLM_ATTENTION_BACKEND"] = cfg["attn_backend"]
    notes = paths.prepare_local_gpu_env()
    for nte in notes:
        print(f"[measure:{cfg_key}] env: {nte}", flush=True)

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    from vllm import LLM, SamplingParams, TokensPrompt

    greedy_identity = paths.import_greedy_identity()

    prompts = load_prompts(Path(args.prompts_ref), args.num_prompts)
    print(f"[measure:{cfg_key}] {cfg['lever']} | n_prompts={len(prompts)} "
          f"out_len={args.output_len} attn_backend={cfg['attn_backend']} {_ts()}", flush=True)

    kwargs = dict(
        model=args.model, tokenizer=args.model, dtype="bfloat16",
        max_model_len=args.max_model_len, max_num_seqs=1,
        gpu_memory_utilization=cfg["gpu_mem"], enforce_eager=cfg["enforce_eager"],
        disable_log_stats=True, enable_prefix_caching=cfg["prefix_cache"],
        trust_remote_code=True,
    )
    if cfg["comp_cfg"] is not None:
        kwargs["compilation_config"] = cfg["comp_cfg"]
    if cfg["max_num_batched_tokens"] is not None:
        kwargs["max_num_batched_tokens"] = cfg["max_num_batched_tokens"]

    t_load = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t_load
    resolved = resolved_compilation(llm)
    print(f"[measure:{cfg_key}] loaded in {load_s:.1f}s resolved={resolved}", flush=True)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.output_len,
                        ignore_eos=True, seed=None)

    # precache emulation: warm the prompt-prefill KV (enable_prefix_caching makes the
    # shared chat-template prefix resident) by replaying a slice of the eval prompts at
    # short max_tokens, mirroring serve_patch_precache (PRECACHE_MAX_TOKENS=4).
    if cfg["prefill_warm"]:
        warm_n = min(args.precache_warm_prompts, len(prompts))
        warm_sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4,
                                 ignore_eos=True, seed=None)
        warm_prompts = [TokensPrompt(prompt_token_ids=p["prompt_token_ids"])
                        for p in prompts[:warm_n]]
        t_warm = time.time()
        llm.generate(warm_prompts, warm_sp, use_tqdm=False)
        print(f"[measure:{cfg_key}] precache prefill-warm: {warm_n} prompts in "
              f"{time.time()-t_warm:.1f}s", flush=True)

    # Decode TPS probe (uses the first eval prompt as the fixed probe), multi-seed.
    tps = measure_decode_tps(llm, SamplingParams, prompts[0]["prompt_token_ids"],
                             decode_tokens=args.tps_decode_tokens,
                             repeats=args.tps_repeats, seeds=TPS_SEEDS)
    print(f"[measure:{cfg_key}] decode_tps_single_stream={tps['decode_tps_single_stream']:.2f} "
          f"per_seed={ {k: round(v,1) for k,v in tps['decode_tps_per_seed_median'].items()} } "
          f"spread={tps['decode_tps_seed_spread']:.2f}", flush=True)

    # Bulk identity capture over the full eval (M=1, one sequence at a time). Greedy is
    # seed-invariant, so this single capture is the identity gate for the config.
    tokens_prompts = [TokensPrompt(prompt_token_ids=p["prompt_token_ids"]) for p in prompts]
    t0 = time.time()
    outs = llm.generate(tokens_prompts, sp, use_tqdm=False)
    gen_s = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decode_path = out_dir / f"decode_{cfg_key}.jsonl"
    total_out = 0
    with decode_path.open("w", encoding="utf-8") as fh:
        for p, o in zip(prompts, outs):
            ids = [int(t) for t in o.outputs[0].token_ids]
            total_out += len(ids)
            rec = {
                "id": p["id"],
                "prompt_token_ids": p["prompt_token_ids"],
                "completion_token_ids": ids,
                "completion_token_sha256": greedy_identity.sha256_tokens(ids),
                "num_prompt_tokens": len(p["prompt_token_ids"]),
                "num_completion_tokens": len(ids),
                "config": cfg_key,
                "reference_kind": f"offline_int4_{cfg['name']}",
            }
            fh.write(json.dumps(rec) + "\n")
    bulk_tps = total_out / gen_s if gen_s > 0 else float("nan")
    print(f"[measure:{cfg_key}] wrote {decode_path} | out_tok={total_out} "
          f"gen={gen_s:.1f}s bulk_tps~={bulk_tps:.1f}", flush=True)

    peak_mib = _device_mem_used_mib()
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    measure = {
        "config": cfg_key,
        "config_name": cfg["name"],
        "lever": cfg["lever"],
        "model": args.model,
        "num_prompts": len(prompts),
        "output_len": args.output_len,
        "resolved_compilation": resolved,
        "enforce_eager": cfg["enforce_eager"],
        "comp_cfg_requested": cfg["comp_cfg"],
        "gpu_mem": cfg["gpu_mem"],
        "prefix_cache": cfg["prefix_cache"],
        "prefill_warm": cfg["prefill_warm"],
        "attn_backend": cfg["attn_backend"],
        "max_num_batched_tokens": cfg["max_num_batched_tokens"],
        "decode_tps_single_stream": tps["decode_tps_single_stream"],
        "decode_tps_samples": tps["decode_tps_samples"],
        "decode_tps_per_seed_median": tps["decode_tps_per_seed_median"],
        "decode_tps_seed_spread": tps["decode_tps_seed_spread"],
        "tps_decode_tokens": args.tps_decode_tokens,
        "tps_seeds": TPS_SEEDS,
        "tps_method": tps["method"],
        "bulk_decode_tps_with_prefill": bulk_tps,
        "total_out_tokens": total_out,
        "load_s": load_s,
        "gen_s": gen_s,
        "peak_gpu_mib": peak_mib,
        "peak_gpu_mib_method": "nvidia-smi device-0 used (post-generate)",
        "peak_rss_mib": round(peak_rss_mib, 1),
        "decode_jsonl": str(decode_path),
        "created_at": _ts(),
    }
    measure_path = out_dir / f"measure_{cfg_key}.json"
    measure_path.write_text(json.dumps(measure, indent=2, sort_keys=True))
    print(f"[measure:{cfg_key}] wrote {measure_path}", flush=True)

    del llm
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=list(CONFIGS),
                    help="which single served-stack rung to measure (S0|S1|S2|S3|S4)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts-ref", default=str(DEFAULT_PROMPTS_REF),
                    help="jsonl with prompt_token_ids + id (the fixed eval)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tps-decode-tokens", type=int, default=256)
    ap.add_argument("--tps-repeats", type=int, default=3)
    ap.add_argument("--precache-warm-prompts", type=int, default=32)
    ap.add_argument("--out-dir", default=str(HERE / "measured"))
    args = ap.parse_args(argv)
    return run(args.config, args)


if __name__ == "__main__":
    raise SystemExit(main())
