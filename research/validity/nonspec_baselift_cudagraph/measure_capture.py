#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""GPU measurement leg (PR #371) — isolate CUDA-graph CAPTURE from inductor FUSION
on the plain int4 M=1 decode of google/gemma-4-E4B-it-qat-w4a16-ct.

ONE config per invocation (clean CUDA process isolation). Offline vLLM LLM(); the
single lever that changes across configs is the compilation_config, so identity is
attributable purely to capture/fusion:

  E (eager)             enforce_eager=True            -> mode=NONE,  cudagraph=NONE
  C (capture, NO fuse)  mode=0 (no inductor)          -> cudagraph=FULL  [capture only]
  F (capture + fuse)    mode=3 (VLLM_COMPILE inductor)-> cudagraph=FULL  [kanna #359 bundle]

E is the byte-exact greedy AR reference; the synthesis card compares C-vs-E and
F-vs-E with the OFFICIAL greedy_identity verifier. M=1 (max_num_seqs=1) offline
greedy (temperature=0, ignore_eos) IS plain non-spec AR by construction.

Writes, under --out-dir:
  decode_<cfg>.jsonl   official decode-output records (id, prompt/completion token ids,
                       completion_token_sha256) -> verifier-compatible.
  measure_<cfg>.json   decode TPS (slope), naive capture TPS, resolved compilation
                       config, peak GPU mem, load/gen walls.

Run (per config):
  cd target/ && CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python \
    research/validity/nonspec_baselift_cudagraph/measure_capture.py \
    --config C --out-dir research/validity/nonspec_baselift_cudagraph/measured
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

# Canonical plain full-vocab int4 (= lawine #196 "int4 M=1 plain AR"). tie=True,
# body int4 (compressed-tensors W4A16), lm_head/embeddings bf16 full-vocab.
DEFAULT_MODEL = (
    "/senpai-run/home/student-ubel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
# Source of aligned prompt_token_ids + stable record ids (the fixed 128-record eval,
# already tokenized with the gemma chat template). Model-independent prompts.
DEFAULT_PROMPTS_REF = (
    REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/"
    "decode_outputs.offline.jsonl"
)

# The three-config lever map (only compilation_config changes).
CONFIGS: dict[str, dict[str, Any]] = {
    "E": {"name": "E-eager", "enforce_eager": True, "comp_cfg": None,
          "desc": "eager: mode=NONE, cudagraph=NONE (byte-exact greedy AR reference)"},
    "C": {"name": "C-capture-nofuse", "enforce_eager": False,
          "comp_cfg": {"mode": 0, "cudagraph_mode": "FULL"},
          "desc": "capture only: mode=0 (no inductor) + cudagraph=FULL"},
    "F": {"name": "F-capture-fuse", "enforce_eager": False,
          "comp_cfg": {"mode": 3, "cudagraph_mode": "FULL"},
          "desc": "capture + inductor fusion: mode=3 + cudagraph=FULL (kanna #359 bundle)"},
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _device_mem_used_mib() -> float | None:
    """Device-0 memory used (MiB) via nvidia-smi.

    vLLM V1 runs the model in an EngineCore subprocess, so the parent's
    torch.cuda.max_memory_allocated() reads 0. This pod is single-tenant on
    physical device 0 (paths.prepare_local_gpu_env normalizes CUDA_VISIBLE_DEVICES
    -> 0), so device-level used is an honest peak proxy (weights + KV reservation +
    activations). Sampled right after the bulk generate.
    """
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
        return {
            "mode": getattr(mode, "value", mode),
            "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
            "backend": getattr(cc, "backend", None),
            "cudagraph_capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", []) or []),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def measure_decode_tps(llm, sampling_cls, probe_token_ids: list[int],
                       *, decode_tokens: int, repeats: int) -> dict[str, Any]:
    """Steady-state single-stream decode TPS via the slope method (harness.probe_tps):
    (N-1)/(wall_N - wall_1). Warmup absorbs capture/compile. Median of `repeats`."""
    from vllm import TokensPrompt

    def gen(max_tokens: int) -> float:
        sp = sampling_cls(temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                          ignore_eos=True, seed=None)
        prompt = TokensPrompt(prompt_token_ids=probe_token_ids)
        t0 = time.time()
        llm.generate([prompt], sp, use_tqdm=False)
        return time.time() - t0

    # Warm up at the FULL decode length so every shape-specialized kernel (attention
    # across KV-block boundaries, sampler, capture/compile) is built BEFORE timing.
    # A short gen(8) alone leaves a long-sequence JIT to fire mid-timed-gen and
    # artificially depress eager TPS (the eager floor must be trustworthy).
    gen(8)
    gen(decode_tokens)
    samples = []
    for _ in range(repeats):
        w1 = gen(1)
        wN = gen(decode_tokens)
        if wN > w1:
            samples.append((decode_tokens - 1) / (wN - w1))
    tps = statistics.median(samples) if samples else float("nan")
    return {
        "decode_tps_single_stream": tps,
        "decode_tps_samples": samples,
        "decode_tokens": decode_tokens,
        "repeats": repeats,
        "method": "slope (N-1)/(wall_N - wall_1); harness.probe_tps",
    }


def run(cfg_key: str, args) -> int:
    cfg = CONFIGS[cfg_key]
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
    print(f"[measure:{cfg_key}] {cfg['desc']} | n_prompts={len(prompts)} "
          f"out_len={args.output_len} {_ts()}", flush=True)

    kwargs = dict(
        model=args.model, tokenizer=args.model, dtype="bfloat16",
        max_model_len=args.max_model_len, max_num_seqs=1,
        gpu_memory_utilization=args.gpu_mem, enforce_eager=cfg["enforce_eager"],
        disable_log_stats=True, enable_prefix_caching=False, trust_remote_code=True,
    )
    if cfg["comp_cfg"] is not None:
        kwargs["compilation_config"] = cfg["comp_cfg"]

    t_load = time.time()
    llm = LLM(**kwargs)
    load_s = time.time() - t_load
    resolved = resolved_compilation(llm)
    print(f"[measure:{cfg_key}] loaded in {load_s:.1f}s resolved={resolved}", flush=True)

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.output_len,
                        ignore_eos=True, seed=None)

    # Decode TPS probe FIRST (uses the first eval prompt as the fixed probe), so its
    # warmup also captures the M=1 graph before the bulk identity capture.
    tps = measure_decode_tps(llm, SamplingParams, prompts[0]["prompt_token_ids"],
                             decode_tokens=args.tps_decode_tokens, repeats=args.tps_repeats)
    print(f"[measure:{cfg_key}] decode_tps_single_stream={tps['decode_tps_single_stream']:.2f} "
          f"(samples={[round(s,1) for s in tps['decode_tps_samples']]})", flush=True)

    # Bulk identity capture over the full eval (M=1, one sequence at a time).
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

    # Device-level peak (subprocess engine -> parent torch counter is 0); sampled
    # at peak right after the bulk generate.
    peak_mib = _device_mem_used_mib()
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    measure = {
        "config": cfg_key,
        "config_name": cfg["name"],
        "config_desc": cfg["desc"],
        "model": args.model,
        "num_prompts": len(prompts),
        "output_len": args.output_len,
        "resolved_compilation": resolved,
        "enforce_eager": cfg["enforce_eager"],
        "comp_cfg_requested": cfg["comp_cfg"],
        "decode_tps_single_stream": tps["decode_tps_single_stream"],
        "decode_tps_samples": tps["decode_tps_samples"],
        "tps_decode_tokens": args.tps_decode_tokens,
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
                    help="which single config to measure (E|C|F)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompts-ref", default=str(DEFAULT_PROMPTS_REF),
                    help="jsonl with prompt_token_ids + id (the fixed eval)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tps-decode-tokens", type=int, default=256)
    ap.add_argument("--tps-repeats", type=int, default=3)
    ap.add_argument("--out-dir", default=str(HERE / "measured"))
    args = ap.parse_args(argv)
    return run(args.config, args)


if __name__ == "__main__":
    raise SystemExit(main())
