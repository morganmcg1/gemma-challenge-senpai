#!/usr/bin/env python3
"""PR #563 linchpin check: with VLLM_USE_FLASHINFER_SAMPLER=0, does the native
torch sampler actually do temperature/top_p/top_k multinomial sampling?

Sends the SAME prompt to the running server (default :8000) under the mandated
gemma-4-E4B-it protocol (temp=1.0, top_p=0.95, top_k=64) with two different seeds,
then under greedy (temp=0) twice. Asserts:
  - sampling seeds s0 != s1  (sampling is live; not silently argmax)
  - greedy g0 == g1          (determinism sanity on the same stack)
  - top_k in extra body does not error the server

Pure stdlib (urllib) so it runs in any venv. Exit 0 on PASS, 1 on FAIL.
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
PROMPT = "List three prime numbers and then explain in one sentence what a prime number is."


def chat(temperature, seed, top_k=0, top_p=1.0, max_tokens=128):
    body = {
        "model": "gemma-4-e4b-it",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if top_k and top_k > 0:
        body["top_k"] = top_k  # vLLM SamplingParams extension (extra body field)
    req = urllib.request.Request(
        BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def main():
    print(f"[smoke] base={BASE}")
    s0 = chat(1.0, seed=11, top_k=64, top_p=0.95)
    s1 = chat(1.0, seed=22, top_k=64, top_p=0.95)
    g0 = chat(0.0, seed=0)
    g1 = chat(0.0, seed=0)

    sampling_live = s0 != s1
    greedy_det = g0 == g1
    print(f"[smoke] sampling_live (s0!=s1) = {sampling_live}")
    print(f"[smoke] greedy_det   (g0==g1) = {greedy_det}")
    print(f"[smoke] --- sample seed=11 (first 160c) ---\n{s0[:160]!r}")
    print(f"[smoke] --- sample seed=22 (first 160c) ---\n{s1[:160]!r}")
    print(f"[smoke] --- greedy (first 160c) ---\n{g0[:160]!r}")

    ok = sampling_live and greedy_det
    print(f"[smoke] {'PASS' if ok else 'FAIL'}: top_k+sampling honored, greedy deterministic")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
