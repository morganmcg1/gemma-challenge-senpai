#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #748 (land) -- does VLLM_BATCH_INVARIANT=1 make the SERVED spec byte-exact?

ONE ARM = one live vLLM 0.22.0 `vllm.entrypoints.openai.api_server` instance under a
chosen (VLLM_BATCH_INVARIANT, spec on/off) config, driven over HTTP exactly like the
official decode benchmark (`speed_benchmark/scripts/decode_outputs.py`): integer-token
prompts, temperature 0, add_special_tokens=False, ignore_eos=True, return_token_ids=True,
output_len 512, the 128 public ShareGPT prompts, seed 1.

This is the LIVE-online transfer test for land #743 (W&B rwk498ve). #743 proved OFFLINE
(enforce_eager, in-process LLM, a chunked-prefill prompt_logprobs proxy for the verify
shape) that BI=1 (num_splits=1) collapses the M=1-decode-vs-M=K-verify attention
reduction-order divergence to byte-exact. This card asks whether that holds on the REAL
served stack: api_server + CUDA graphs (NOT enforce_eager) + real v1 ngram batched-verify
spec-decode. The default v1 cudagraph behavior (M=1 decode graph-captured; the M=K+1=7
verify shape likely eager) is precisely the online condition #743's offline proxy missed.

We hold the attention backend fixed at TRITON_ATTN (the served unified-attention kernel
whose 2D/3D split-KV reduction is the #743 locus) so the only delta from #743 is
offline-proxy -> live-online. Each arm writes decode_outputs.jsonl (token streams +
sha256, identical schema to decode_outputs.py) + arm_summary.json (config, env, the
single-stream output_tps, peak GPU mem). The analyzer compares same-BI spec-vs-AR streams.

LOCAL A10G only. analysis_only -- NO HF Job, NO submission, NO served-file change. The
loadable full-vocab int4 QAT ckpt google/gemma-4-E4B-it-qat-w4a16-ct stands in for the
deployed pruned-16k-head int4_g128_lmhead (vanilla vLLM cannot load the pruned head); the
attention M-dependence is a kernel-occupancy/reduction-order property, head-vocab- and
weight-quant-independent (#743), so this is faithful to the served attention question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
DATASET = (ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/"
           "eval_prompts_sharegpt.json")
VENV_PY = "/tmp/senpai-venvs/20f658587e8a6643/bin/python"  # stock vLLM 0.22.0 (the #743 venv)
MODEL_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
TOKENIZER = "google/gemma-4-E4B-it"
SERVED_NAME = "gemma-4-e4b-it"

# ---- prompt construction: byte-identical to speed_benchmark/scripts/decode_outputs.py ----


def read_sharegpt_prompts(path: Path, *, num_prompts: int, seed: int) -> list[dict]:
    data = json.loads(path.read_text())
    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        first = conv[0]
        if not isinstance(first, dict):
            continue
        prompt = first.get("value")
        if not isinstance(prompt, str) or not prompt:
            continue
        records.append({"id": str(item.get("id", index)), "dataset_index": index,
                        "prompt_text": prompt})
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:num_prompts]


def _normalize_token_ids(value) -> list[int]:
    """Mirror decode_outputs.py:normalize_token_ids -- apply_chat_template(tokenize=True)
    can return a BatchEncoding/dict; extract input_ids (NOT the dict keys)."""
    from collections.abc import Mapping
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        for key in ("input_ids", "token_ids", "prompt_token_ids"):
            if key in value:
                return _normalize_token_ids(value[key])
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if value and all(isinstance(t, int) and t >= 0 for t in value):
            return value
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return _normalize_token_ids(value[0])
    raise ValueError(f"tokenization did not yield a list of int token ids: {type(value)}")


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                tokenize=True)
    except Exception:  # noqa: BLE001
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                  tokenize=False)
        encoded = tokenizer.encode(formatted, add_special_tokens=False)
    return _normalize_token_ids(encoded)


def sha_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


# ---- server lifecycle ----


def build_server_argv(bi: int, spec: int, port: int, args) -> list[str]:
    argv = [
        VENV_PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--served-model-name", SERVED_NAME,
        "--quantization", "compressed-tensors",
        "--dtype", "bfloat16",
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if getattr(args, "enforce_eager", 0):
        argv += ["--enforce-eager"]
    if spec:
        spec_cfg = {
            "method": "ngram",
            "num_speculative_tokens": args.num_spec,
            "prompt_lookup_max": args.prompt_lookup_max,
            "prompt_lookup_min": args.prompt_lookup_min,
        }
        argv += ["--speculative-config", json.dumps(spec_cfg)]
    return argv


def server_env(bi: int) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"            # inherited =7 is stale (gpu_env memory)
    env["VLLM_BATCH_INVARIANT"] = str(bi)        # the operative lever (num_splits=1 when 1)
    env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"  # the #743 served-locus kernel; hold fixed
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"     # curand.h JIT fails locally; greedy-neutral
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    return env


def wait_health(port: int, proc: subprocess.Popen, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server process exited early rc={proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise TimeoutError(f"server not healthy after {timeout_s}s")


class GpuMemSampler(threading.Thread):
    """Poll nvidia-smi GPU0 used-memory; keep the peak (MiB)."""

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                     "-i", "0"], text=True).strip().splitlines()
                self.peak_mib = max(self.peak_mib, int(out[0]))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def post_completion(port: int, token_ids: list[int], output_len: int, timeout_s: int) -> dict:
    payload = {
        "model": SERVED_NAME, "prompt": token_ids, "max_tokens": output_len,
        "temperature": 0.0, "stream": False, "add_special_tokens": False,
        "ignore_eos": True, "return_token_ids": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_completion_ids(resp: dict, prompt_ids: list[int]) -> list[int]:
    choice = resp["choices"][0]
    for v in (choice.get("token_ids"), choice.get("output_token_ids"),
              choice.get("completion_token_ids")):
        if isinstance(v, list) and v and isinstance(v[0], int):
            if len(v) >= len(prompt_ids) and v[: len(prompt_ids)] == prompt_ids:
                return v[len(prompt_ids):]
            return v
    raise ValueError(f"no completion token_ids; choice keys={list(choice.keys())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bi", type=int, required=True, choices=(0, 1))
    ap.add_argument("--spec", type=int, required=True, choices=(0, 1))
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-spec", type=int, default=6)
    ap.add_argument("--prompt-lookup-max", type=int, default=4)
    ap.add_argument("--prompt-lookup-min", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", type=int, default=0, choices=(0, 1),
                    help="1 = pass --enforce-eager (no CUDA graphs); replicates #743 offline condition")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--req-timeout", type=int, default=600)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    tag = args.tag or f"bi{args.bi}_spec{args.spec}"
    outdir = args.outdir or (HERE / "runs" / tag)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "server.log"
    out_jsonl = outdir / "decode_outputs.jsonl"
    summary_path = outdir / "arm_summary.json"

    print(f"[arm {tag}] bi={args.bi} spec={args.spec} n_prompts={args.n_prompts} "
          f"output_len={args.output_len} port={args.port}", flush=True)

    argv = build_server_argv(args.bi, args.spec, args.port, args)
    env = server_env(args.bi)
    print(f"[arm {tag}] server argv: {' '.join(argv)}", flush=True)
    print(f"[arm {tag}] env BI={env['VLLM_BATCH_INVARIANT']} "
          f"BACKEND={env['VLLM_ATTENTION_BACKEND']}", flush=True)

    logf = open(log_path, "w")
    proc = subprocess.Popen(argv, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    sampler = GpuMemSampler()
    server_meta: dict[str, Any] = {}
    try:
        t_boot = time.time()
        wait_health(args.port, proc, args.startup_timeout)
        boot_s = time.time() - t_boot
        print(f"[arm {tag}] healthy after {boot_s:.0f}s", flush=True)
        sampler.start()

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        records = read_sharegpt_prompts(DATASET, num_prompts=args.n_prompts, seed=args.seed)
        assert len(records) == args.n_prompts, f"got {len(records)} prompts"

        rows = []
        total_gen_s = 0.0
        total_comp_tok = 0
        total_prompt_tok = 0
        t0 = time.time()
        for i, rec in enumerate(records):
            ptoks = encode_prompt(tok, rec["prompt_text"])
            t_req = time.time()
            resp = post_completion(args.port, ptoks, args.output_len, args.req_timeout)
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
                print(f"  [{i+1}/{len(records)}] comp={len(comp)} "
                      f"cum_tps={run_tps:.2f} ({time.time()-t0:.0f}s)", flush=True)
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

    # grep the chosen attention backend + spec config out of the server log (provenance)
    backend_line = spec_line = ""
    try:
        for ln in log_path.read_text(errors="replace").splitlines():
            low = ln.lower()
            if "using" in low and "backend" in low and "attn" in low:
                backend_line = backend_line or ln.strip()
            if "speculativeconfig(" in low or ("method='ngram'" in low):
                spec_line = spec_line or ln.strip()
    except Exception:  # noqa: BLE001
        pass

    out_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    output_tps = total_comp_tok / total_gen_s if total_gen_s else 0.0
    summary = {
        "phase": "strict_clean_served_byteexact", "tag": tag,
        "batch_invariant": args.bi, "spec": args.spec,
        "model_id": MODEL_ID, "served_model_name": SERVED_NAME, "tokenizer": TOKENIZER,
        "dataset": str(DATASET), "n_prompts": len(rows), "output_len": args.output_len,
        "seed": args.seed,
        "spec_config": ({"method": "ngram", "num_speculative_tokens": args.num_spec,
                         "prompt_lookup_max": args.prompt_lookup_max,
                         "prompt_lookup_min": args.prompt_lookup_min} if args.spec else None),
        "env": {"VLLM_BATCH_INVARIANT": str(args.bi),
                "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
                "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "enforce_eager": bool(args.enforce_eager), "cudagraphs": not bool(args.enforce_eager),
        "num_prompt_tokens": total_prompt_tok, "num_completion_tokens": total_comp_tok,
        "total_gen_s": round(total_gen_s, 3), "wall_s": round(wall_s, 3),
        "boot_s": round(boot_s, 1),
        "output_tps": round(output_tps, 4),
        "peak_gpu_mem_mib": sampler.peak_mib,
        "server_backend_line": backend_line, "server_spec_line": spec_line,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72, flush=True)
    print(f"[ARM {tag}] output_tps={output_tps:.4f}  comp_tok={total_comp_tok}  "
          f"gen_s={total_gen_s:.1f}  peak_mem={sampler.peak_mib}MiB", flush=True)
    print(f"  backend_line: {backend_line}", flush=True)
    print(f"  spec_line:    {spec_line}", flush=True)
    print(f"  -> {out_jsonl}", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
