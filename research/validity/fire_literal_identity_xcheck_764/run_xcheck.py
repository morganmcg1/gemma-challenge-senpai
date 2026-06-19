#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #764 (land) -- INDEPENDENT cross-validation of the fire's literal served-greedy identity.

Wirbel #751 (relayed FACT, his branch NOT read) measured the FULL fire config
(`submissions/int4_mtp_batchinv`, full VLLM_BATCH_INVARIANT=1, MTP drafter spec) at
108/128 divergent / 20/128 identical (literal greedy identity 0.156) vs HIS served
spec-off M=1 AR reference -- i.e. self-consistent-gate + PPL-clean but NOT literal-byte-
exact vs an independent AR reference. This card reproduces that number on a SECOND,
INDEPENDENT harness -- MY OWN #748 machinery (research/validity/strict_clean_served_byteexact_748).

ONE ARM = one live vLLM 0.22.0 api_server launched through the fire submission's OWN
serve.py (so the served config -- int4 W4A16 Marlin target + gemma4_assistant MTP drafter
+ the attention-group num_heads sitecustomize patch + every manifest env -- is byte-for-byte
the exact mode that yields denken's 128/128 self-consistent gate). Two arm kinds:

  spec_on   : default fire env -> NUM_SPECULATIVE_TOKENS=6, MTP drafter ON, BI=1. The CANDIDATE.
  spec_off  : same serve.py + SENPAI_REFERENCE_MODE=1 -> serve.py forces num_speculative_tokens=0,
              drafter OFF, plain int4 M=1 AR. MY INDEPENDENT reference (the fire's OWN documented
              reference-mode contract; reconstructed in-scope, NOT read from wirbel's #751).

Each arm is driven over HTTP exactly like land #748 / the official decode_outputs.py: the 128
public ShareGPT prompts (seed 1), integer-token prompts, temperature 0 (greedy, the #319
strict-identity protocol -- NOT generation_config.json), add_special_tokens=False, ignore_eos=True,
return_token_ids=True, output_len 512. We REUSE the merged #748 client + identity machinery so this
is genuinely my own harness, independent of wirbel's. Writes decode_outputs.jsonl (token streams +
sha256, #748 schema) + arm_summary.json. analyze_xcheck.py then compares spec_on vs spec_off.

LOCAL A10G only. analysis_only=1, official_tps=0, no_hf_job=1, fires=0. No served-file change, no
submission, no --launch. The locked int4_g128_lmhead@126.378 baseline is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
FIRE_SUB = ROOT / "submissions" / "int4_mtp_batchinv"
SERVE_PY = FIRE_SUB / "serve.py"
# stock vLLM 0.22.0 + transformers 5.9.0 -- matches the fire manifest deps exactly
# (the #743/#748 venv). The dev307 venvs are a different pin and are NOT the fire's.
VENV_PY = "/tmp/senpai-venvs/20f658587e8a6643/bin/python"

# Reuse the MERGED #748 harness (my own work): prompt construction byte-identical to the
# official decode_outputs.py, the HTTP completion client, sha256, and health wait.
P748 = ROOT / "research" / "validity" / "strict_clean_served_byteexact_748"
sys.path.insert(0, str(P748))
from run_arm import (  # noqa: E402
    GpuMemSampler,
    encode_prompt,
    extract_completion_ids,
    post_completion,
    read_sharegpt_prompts,
    sha_tokens,
    wait_health,
)

DATASET = (ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/"
           "eval_prompts_sharegpt.json")
TOKENIZER = "google/gemma-4-E4B-it"
SERVED_NAME = "gemma-4-e4b-it"


def fire_env(reference_mode: bool, port: int) -> dict[str, str]:
    """Exact fire-manifest serving env + LOCAL-ONLY greedy-neutral necessities.

    The manifest env block IS the fire config (BI=1, MTP drafter, num_spec=6, ...). We set it
    explicitly so the served numerics match denken's 128/128-gate config. The only additions are
    local-environment necessities that DO NOT change greedy tokens (applied identically to BOTH
    arms): CUDA_VISIBLE_DEVICES=0 (inherited =7 is stale), HF offline (weights are cached), and
    VLLM_USE_FLASHINFER_SAMPLER=0 (the flashinfer sampler's curand.h JIT fails on this A10G; at
    temperature 0 the sampler is argmax regardless, so this is greedy-token-neutral -- land #748).
    """
    env = dict(os.environ)
    # ---- the fire manifest env block (verbatim) -------------------------------------------
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["DRAFTER_MODEL"] = "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant"
    env["NUM_SPECULATIVE_TOKENS"] = "6"
    env["MAX_MODEL_LEN"] = "4096"
    env["GPU_MEMORY_UTILIZATION"] = "0.90"
    env["MAX_NUM_BATCHED_TOKENS"] = "512"
    env["MAX_NUM_SEQS"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # ---- serve.py consumes these from env -------------------------------------------------
    env["MODEL_ID"] = "google/gemma-4-E4B-it-qat-w4a16-ct"
    env["SERVED_MODEL_NAME"] = SERVED_NAME
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(port)
    # ---- LOCAL-ONLY, greedy-neutral, applied to BOTH arms ---------------------------------
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    # ---- the reference contract: the ONLY between-arm difference --------------------------
    if reference_mode:
        env["SENPAI_REFERENCE_MODE"] = "1"   # serve.py -> num_speculative_tokens=0, drafter OFF
    else:
        env.pop("SENPAI_REFERENCE_MODE", None)
    return env


def run_arm(tag: str, reference_mode: bool, n_prompts: int, output_len: int,
            seed: int, port: int, startup_timeout: int, req_timeout: int,
            outdir: Path) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "server.log"
    out_jsonl = outdir / "decode_outputs.jsonl"
    summary_path = outdir / "arm_summary.json"

    kind = "spec_off_M1_AR_reference" if reference_mode else "spec_on_fire_candidate"
    print(f"[arm {tag}] kind={kind} reference_mode={int(reference_mode)} "
          f"n_prompts={n_prompts} output_len={output_len} port={port}", flush=True)

    argv = [VENV_PY, str(SERVE_PY)]   # serve.py reads everything from env, then execvpe's api_server
    env = fire_env(reference_mode, port)
    print(f"[arm {tag}] serve argv: {' '.join(argv)}", flush=True)
    print(f"[arm {tag}] BI={env['VLLM_BATCH_INVARIANT']} num_spec={env['NUM_SPECULATIVE_TOKENS']} "
          f"REFERENCE_MODE={env.get('SENPAI_REFERENCE_MODE','<unset>')} "
          f"drafter={env['DRAFTER_MODEL']}", flush=True)

    logf = open(log_path, "w")
    proc = subprocess.Popen(argv, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    sampler = GpuMemSampler()
    boot_s = float("nan")
    wall_s = 0.0
    total_gen_s = 0.0
    total_comp_tok = 0
    total_prompt_tok = 0
    rows: list[dict[str, Any]] = []
    try:
        t_boot = time.time()
        wait_health(port, proc, startup_timeout)
        boot_s = time.time() - t_boot
        print(f"[arm {tag}] healthy after {boot_s:.0f}s", flush=True)
        sampler.start()

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        records = read_sharegpt_prompts(DATASET, num_prompts=n_prompts, seed=seed)
        assert len(records) == n_prompts, f"got {len(records)} prompts"

        t0 = time.time()
        for i, rec in enumerate(records):
            ptoks = encode_prompt(tok, rec["prompt_text"])
            t_req = time.time()
            resp = post_completion(port, ptoks, output_len, req_timeout)
            dt = time.time() - t_req
            comp = extract_completion_ids(resp, ptoks)
            total_gen_s += dt
            total_comp_tok += len(comp)
            total_prompt_tok += len(ptoks)
            rows.append({
                "id": rec["id"], "index": i, "dataset_index": rec["dataset_index"],
                "prompt_token_ids": ptoks, "prompt_token_sha256": sha_tokens(ptoks),
                "completion_token_ids": comp, "completion_token_sha256": sha_tokens(comp),
                "num_prompt_tokens": len(ptoks), "num_completion_tokens": len(comp),
                "gen_s": dt, "req_tps": (len(comp) / dt) if dt > 0 else 0.0,
            })
            if (i + 1) % 8 == 0 or i == len(records) - 1:
                run_tps = total_comp_tok / total_gen_s if total_gen_s else 0.0
                print(f"  [{i+1}/{len(records)}] comp={len(comp)} cum_tps={run_tps:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        wall_s = time.time() - t0
    finally:
        sampler.stop()
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=30)
        logf.close()

    # provenance: grep the resolved attention backend + spec config / reference-mode line out of the log
    backend_line = spec_line = refmode_line = ""
    try:
        for ln in log_path.read_text(errors="replace").splitlines():
            low = ln.lower()
            if "using" in low and "backend" in low and "attn" in low:
                backend_line = backend_line or ln.strip()
            if "speculativeconfig(" in low or "num_speculative_tokens" in low:
                spec_line = spec_line or ln.strip()
            if "reference_mode active" in low or "forcing num_speculative_tokens=0" in low:
                refmode_line = refmode_line or ln.strip()
    except Exception:  # noqa: BLE001
        pass

    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    output_tps = total_comp_tok / total_gen_s if total_gen_s else 0.0
    summary = {
        "phase": "fire_literal_identity_xcheck", "pr": 764, "tag": tag,
        "kind": kind, "reference_mode": bool(reference_mode),
        "batch_invariant": 1, "num_speculative_tokens": (0 if reference_mode else 6),
        "drafter_model": ("<off>" if reference_mode else env["DRAFTER_MODEL"]),
        "model_id": env["MODEL_ID"], "served_model_name": SERVED_NAME, "tokenizer": TOKENIZER,
        "dataset": str(DATASET), "n_prompts": len(rows), "output_len": output_len, "seed": seed,
        "served_via": "submissions/int4_mtp_batchinv/serve.py",
        "venv_python": VENV_PY,
        "env_local_neutral": {"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1",
                              "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "num_prompt_tokens": total_prompt_tok, "num_completion_tokens": total_comp_tok,
        "total_gen_s": round(total_gen_s, 3), "wall_s": round(wall_s, 3),
        "boot_s": round(boot_s, 1), "output_tps": round(output_tps, 4),
        "peak_gpu_mem_mib": sampler.peak_mib,
        "server_backend_line": backend_line, "server_spec_line": spec_line,
        "server_refmode_line": refmode_line,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72, flush=True)
    print(f"[ARM {tag}] kind={kind} output_tps={output_tps:.4f} comp_tok={total_comp_tok} "
          f"gen_s={total_gen_s:.1f} peak_mem={sampler.peak_mib}MiB boot={boot_s:.0f}s", flush=True)
    print(f"  backend_line: {backend_line}", flush=True)
    print(f"  spec_line:    {spec_line}", flush=True)
    print(f"  refmode_line: {refmode_line}", flush=True)
    print(f"  -> {out_jsonl}", flush=True)
    print("=" * 72, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reference-mode", type=int, required=True, choices=(0, 1),
                    help="1 = SENPAI_REFERENCE_MODE=1 (drafter OFF, M=1 AR reference); 0 = fire spec ON")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8033)
    ap.add_argument("--startup-timeout", type=int, default=900)
    ap.add_argument("--req-timeout", type=int, default=900)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    outdir = args.outdir or (HERE / "runs" / args.tag)
    return run_arm(args.tag, bool(args.reference_mode), args.n_prompts, args.output_len,
                   args.seed, args.port, args.startup_timeout, args.req_timeout, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
