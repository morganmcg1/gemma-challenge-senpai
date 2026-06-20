#!/usr/bin/env python
"""Acceptance A/B diagnostic (PR #814 reconciliation).

measure_arm.py (8 official eval prompts x512 tok) showed g128 = +2.79% wall TPS
over g32, with spec-decode acceptance PRESERVED (~34% both arms; from the server
logs). calibrate_arm.py (ONE hand-picked probe prompt, 256 forced tokens) showed
g128 = -46.8% vs g32. The only thing that can make g128 *slower* on a prompt --
its body GEMM reads FEWER bytes -- is MTP spec-decode acceptance: the
gemma4_assistant drafter is matched to the g32 QAT body, so changing the body to
g128 can drop draft acceptance on some prompts, and acceptance is the dominant
TPS lever (E_accept ~3.4x).

This script serves ONE arm through the int4head path (MAX_NUM_SEQS=1, same as
measure/calibrate) and, for each of several DIVERSE prompts, reads the vLLM
/metrics spec-decode counters before/after a fixed forced-token burst, so we get
PER-PROMPT acceptance + naive TPS. Run for g32 and g128; if g128's acceptance is
preserved on the eval-like prompts but collapses on the transformer probe prompt,
the calibrate -46.8% is a single-prompt artifact and the eval-prompt +2.79% is
the benchmark-relevant verdict.

LOCAL single-A10G only. No HF Job.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

INT4HEAD_DIR = ROOT / "submissions" / "int4_mtp_bi0_int4head"
SERVED_NAME = "gemma-4-e4b-it"

# The original calibrate/probe prompt (the one g128 was slow on) + diverse others.
PROMPTS = {
    "probe_transformer": "Explain step by step how a transformer decodes one token at a time.",
    "math_word": "A train travels 60 miles in 1.5 hours. If it keeps the same speed, how far does it travel in 4 hours? Show your reasoning.",
    "code_py": "Write a Python function that returns the n-th Fibonacci number using memoization, then explain how it works.",
    "story": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "factual_qa": "What were the main causes of the fall of the Western Roman Empire? Answer in a few sentences.",
}

_COUNTER_RE = {
    "accepted": re.compile(r"^vllm:spec_decode_num_accepted_tokens(?:_total)?(?:\{[^}]*\})?\s+([0-9eE+\-.]+)"),
    "draft": re.compile(r"^vllm:spec_decode_num_draft_tokens(?:_total)?(?:\{[^}]*\})?\s+([0-9eE+\-.]+)"),
    "drafts": re.compile(r"^vllm:spec_decode_num_drafts(?:_total)?(?:\{[^}]*\})?\s+([0-9eE+\-.]+)"),
}


def read_counters(base_url: str) -> dict[str, float]:
    out = {k: 0.0 for k in _COUNTER_RE}
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as exc:
        print(f"  [metrics] read failed: {exc}", flush=True)
        return out
    for line in text.splitlines():
        for key, rx in _COUNTER_RE.items():
            m = rx.match(line.strip())
            if m:
                out[key] += float(m.group(1))
    return out


def completion(base_url: str, prompt: str, max_tokens: int, timeout_s: int = 300) -> dict:
    payload = {"model": SERVED_NAME, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": False, "ignore_eos": True}
    req = urllib.request.Request(f"{base_url}/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        resp = json.loads(r.read().decode())
    resp["_wall_s"] = time.time() - t0
    return resp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--body-group-size", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "accept_diag.jsonl"))
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[accept] {note}", flush=True)

    ckpt = Path(args.ckpt).resolve()
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    manifest = harness.load_manifest(INT4HEAD_DIR)
    server_py = harness.ensure_server_venv(manifest["dependencies"])
    base_url = f"http://127.0.0.1:{args.port}"
    log_path = Path(__file__).resolve().parent / "logs" / f"accept_{args.arm}_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rec = {"arm": args.arm, "ckpt": str(ckpt), "body_group_size": args.body_group_size,
           "max_tokens": args.max_tokens, "t_start_utc": datetime.now(timezone.utc).isoformat(),
           "per_prompt": {}, "served_ok": False, "error": None}
    try:
        with harness.LocalServer(INT4HEAD_DIR, server_python=server_py, port=args.port,
                                 log_path=log_path, extra_env={"MODEL_ID": str(ckpt)}):
            rec["served_ok"] = True
            # warmup so compile/caches are hot
            completion(base_url, "Hello, how are you?", 16)
            for name, prompt in PROMPTS.items():
                c0 = read_counters(base_url)
                resp = completion(base_url, prompt, args.max_tokens)
                c1 = read_counters(base_url)
                usage = resp.get("usage") or {}
                n = usage.get("completion_tokens") or args.max_tokens
                d_acc = c1["accepted"] - c0["accepted"]
                d_drf = c1["draft"] - c0["draft"]
                d_drafts = c1["drafts"] - c0["drafts"]
                acc_rate = (d_acc / d_drf) if d_drf > 0 else None
                # mean accepted tokens per drafting step (+1 bonus token per accepted step)
                mean_acc_len = ((d_acc + d_drafts) / d_drafts) if d_drafts > 0 else None
                naive_tps = n / resp["_wall_s"] if resp["_wall_s"] else None
                rec["per_prompt"][name] = {
                    "n_out": n, "wall_s": round(resp["_wall_s"], 4),
                    "naive_tps": round(naive_tps, 2) if naive_tps else None,
                    "accepted": d_acc, "draft": d_drf, "drafts": d_drafts,
                    "accept_rate": round(acc_rate, 4) if acc_rate is not None else None,
                    "mean_accept_len": round(mean_acc_len, 3) if mean_acc_len is not None else None,
                }
                print(f"  {name:18} naive_tps={rec['per_prompt'][name]['naive_tps']} "
                      f"accept_rate={rec['per_prompt'][name]['accept_rate']} "
                      f"mean_accept_len={rec['per_prompt'][name]['mean_accept_len']} "
                      f"(acc={int(d_acc)}/drf={int(d_drf)}/drafts={int(d_drafts)})", flush=True)
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"[accept] ERROR: {exc}", flush=True)

    rec["t_end_utc"] = datetime.now(timezone.utc).isoformat()
    # aggregate
    pp = rec["per_prompt"]
    if pp:
        tot_acc = sum(v["accepted"] for v in pp.values())
        tot_drf = sum(v["draft"] for v in pp.values())
        tps_vals = [v["naive_tps"] for v in pp.values() if v["naive_tps"]]
        rec["agg_accept_rate"] = round(tot_acc / tot_drf, 4) if tot_drf > 0 else None
        rec["mean_naive_tps"] = round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"\n[accept] arm={args.arm} agg_accept_rate={rec.get('agg_accept_rate')} "
          f"mean_naive_tps={rec.get('mean_naive_tps')}", flush=True)
    return 0 if rec["served_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
