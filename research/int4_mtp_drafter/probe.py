#!/usr/bin/env python
"""Local exploratory probe: single-stream TPS + spec-decode acceptance.

Sends greedy (temp=0) integer-token completions to a local vLLM endpoint, times
output throughput, and reads vLLM's /metrics spec-decode counters to compute mean
accepted tokens/step and the per-position acceptance curve. Local AWS A10G only;
NOT the official a10g-small score.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def scrape_spec_metrics(base: str) -> dict:
    out: dict[str, float] = {}
    per_pos: dict[int, float] = {}
    try:
        text = get(f"{base}/metrics", timeout=10)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        for key in (
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
            "vllm:spec_decode_num_drafts",
            "vllm:spec_decode_num_spec_tokens",
        ):
            if line.startswith(key) and "_per_pos" not in line:
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        if "spec_decode_num_accepted_tokens_per_pos" in line:
            try:
                pos = int(line.split('position="', 1)[1].split('"', 1)[0])
                per_pos[pos] = per_pos.get(pos, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    if per_pos:
        out["per_pos"] = [per_pos[k] for k in sorted(per_pos)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    # A few fixed integer-token prompts (BOS=2 + a short instruction-ish seed).
    ap.add_argument("--prompt", default="2,2364,1841,603,573,2669,576,3777,235336")
    args = ap.parse_args()

    prompt_ids = [int(x) for x in args.prompt.split(",") if x.strip()]
    base = args.base.rstrip("/")

    before = scrape_spec_metrics(base)
    times = []
    ntok = []
    for i in range(args.reps):
        payload = {
            "model": args.model,
            "prompt": prompt_ids,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
            "add_special_tokens": False,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        t0 = time.time()
        resp = post(f"{base}/v1/completions", payload)
        dt = time.time() - t0
        ch = resp["choices"][0]
        toks = ch.get("token_ids") or ch.get("output_token_ids") or []
        gen = len(toks) - (len(prompt_ids) if toks[: len(prompt_ids)] == prompt_ids else 0) if toks else args.max_tokens
        times.append(dt)
        ntok.append(gen)
        print(f"rep{i}: {gen} tok in {dt:.2f}s -> {gen / dt:.2f} tok/s")

    after = scrape_spec_metrics(base)
    # Delta of counters over this probe.
    def d(k: str) -> float:
        return float(after.get(k, 0.0)) - float(before.get(k, 0.0))

    draft = d("vllm:spec_decode_num_draft_tokens")
    acc = d("vllm:spec_decode_num_accepted_tokens")
    drafts = d("vllm:spec_decode_num_drafts")
    best_dt = min(times)
    best_tok = ntok[times.index(best_dt)]
    print("\n=== SUMMARY ===")
    print(f"best single-stream TPS (local A10G, exploratory): {best_tok / best_dt:.2f} tok/s")
    print(f"spec counters delta: drafts={drafts} draft_tokens={draft} accepted_tokens={acc}")
    if drafts > 0:
        print(f"mean accepted tokens / step (incl. bonus): {1.0 + acc / drafts:.3f}")
    if draft > 0:
        print(f"overall acceptance rate: {acc / draft:.3f}")
    print(f"per_pos accepted (after): {after.get('per_pos')}")
    print(f"raw before: {json.dumps(before)}")
    print(f"raw after:  {json.dumps(after)}")


if __name__ == "__main__":
    main()
