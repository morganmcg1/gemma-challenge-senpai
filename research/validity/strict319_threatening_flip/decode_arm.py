"""Decode the #694 flagged subset under one arm at the FULL 6144-tok NATURAL-EOS
budget so the answer actually commits.

Why natural EOS (ignore_eos=False) and not the official 512-tok ignore_eos speed
decode: #685/#626 showed the answer is emitted LAST (gpqa median 1843 / aime 3783
tok to `ANSWER:`), so the speed decode TRUNCATES before commit. The quality
contract (#626/#682) scores at natural EOS; this reproduces that so AR and the
spec stack both reach their answer line and we can compare the EXTRACTED answer
end to end.

Arms:
  ar       -- plain M=1 autoregressive reference (no speculative config)
  suffix6  -- {"method":"suffix","num_speculative_tokens":6}                (#678 primary)
  ngram5   -- {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_min":2,"prompt_lookup_max":6}

Served stack is identical to the strict anchor int4_g128_lmhead (#4): int4 g128 +
untied int4 lm_head, VLLM_BATCH_INVARIANT=1, TRITON_ATTN. analysis_only: this never
touches the served submission file and never launches an HF Job.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the OFFICIAL decode helpers (read-only mirror) so prompt encoding +
# token-id extraction are byte-identical to the organizer audit path.
OFFICIAL = Path("official/main_bucket/shared_resources/speed_benchmark/scripts").resolve()
sys.path.insert(0, str(OFFICIAL))
from decode_outputs import (  # noqa: E402
    encode_prompt,
    extract_generated_token_ids,
    choice_from_response,
    generated_text_from_choice,
    sha256_text,
    sha256_tokens,
)

SPEC = {
    "ar": None,
    "suffix6": {"method": "suffix", "num_speculative_tokens": 6},
    "ngram5": {"method": "ngram", "num_speculative_tokens": 5,
               "prompt_lookup_min": 2, "prompt_lookup_max": 6},
}

DEFAULT_MODEL = "/workspace/gemma_build/int4_g128_lmhead"
DEFAULT_PYTHON = os.environ.get(
    "SERVER_PYTHON", "/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python")
TOK = ("/senpai-run/home/student-kanna/.cache/huggingface/hub/"
       "models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187")


def start_server(arm, model, port, max_model_len, max_num_seqs, log_path):
    env = dict(os.environ)
    env["VLLM_BATCH_INVARIANT"] = "1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # The host sets CUDA_VISIBLE_DEVICES to the physical id (e.g. '3') but only one
    # GPU (index 0) is visible inside this container; vLLM's pynvml handle-by-index
    # then raises NVMLError_InvalidArgument. Normalize to the in-container index.
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and cvd != "0":
        env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        DEFAULT_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", "gemma-4-e4b-it",
        "--host", "127.0.0.1", "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code",
        "--no-enable-log-requests",
    ]
    spec = SPEC[arm]
    if spec is not None:
        cmd += ["--speculative-config", json.dumps(spec)]
    log = open(log_path, "w")
    log.write(f"[serve] arm={arm} spec={spec} model={model} max_model_len={max_model_len}\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    return proc, log


def wait_ready(port, proc, timeout_s):
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server not ready in {timeout_s}s")


def stop_server(proc, log):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    finally:
        log.close()


def request_decode(port, prompt_token_ids, max_tokens, timeout_s):
    payload = {
        "model": "gemma-4-e4b-it",
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "add_special_tokens": False,
        "ignore_eos": False,          # NATURAL EOS -- the answer must commit then stop
        "return_token_ids": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(SPEC))
    ap.add_argument("--subset", default="research/validity/strict319_threatening_flip/flagged_subset.json")
    ap.add_argument("--dataset", default="official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--max-model-len", type=int, default=9216)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout-s", type=int, default=600)
    ap.add_argument("--request-timeout-s", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=None, help="decode only first N flagged prompts (smoke)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK if os.path.isdir(TOK) else "google/gemma-4-E4B-it")

    subset = json.loads(Path(args.subset).read_text())
    flagged_ids = [p["id"] for p in subset["prompts"]]
    if args.limit:
        flagged_ids = flagged_ids[: args.limit]
    dataset = {r["id"]: r for r in json.loads(Path(args.dataset).read_text())}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".server.log")

    proc, log = start_server(args.arm, args.model, args.port,
                             args.max_model_len, args.max_num_seqs, log_path)
    records = []
    try:
        ready_s = wait_ready(args.port, proc, args.startup_timeout_s)
        print(f"[decode] arm={args.arm} server ready in {ready_s:.0f}s", flush=True)
        for i, pid in enumerate(flagged_ids):
            prompt_text = dataset[pid]["conversations"][0]["value"]
            prompt_token_ids = encode_prompt(tok, prompt_text)
            t0 = time.time()
            resp = request_decode(args.port, prompt_token_ids, args.max_tokens, args.request_timeout_s)
            dt = time.time() - t0
            choice = choice_from_response(resp)
            comp_ids, src, kind = extract_generated_token_ids(resp, choice, prompt_token_ids)
            text = generated_text_from_choice(choice)
            finish = choice.get("finish_reason")
            rec = {
                "id": pid,
                "arm": args.arm,
                "source": dataset[pid]["id"].split("-")[0],
                "prompt_token_ids": prompt_token_ids,
                "prompt_sha256": sha256_text(prompt_text),
                "generated_text": text,
                "completion_token_ids": comp_ids,
                "completion_token_sha256": sha256_tokens(comp_ids),
                "num_prompt_tokens": len(prompt_token_ids),
                "num_completion_tokens": len(comp_ids),
                "finish_reason": finish,
                "decode_s": round(dt, 2),
            }
            records.append(rec)
            print(f"[decode] {i+1}/{len(flagged_ids)} {pid} {args.arm} "
                  f"comp_tok={len(comp_ids)} finish={finish} {dt:.0f}s", flush=True)
    finally:
        stop_server(proc, log)

    with out_path.open("w", encoding="utf-8") as h:
        for rec in records:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[decode] wrote {out_path} ({len(records)} records)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
