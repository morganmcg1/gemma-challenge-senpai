"""PR #648: per-POSITION served recompute-fire census (extends #645 root-fork-only).

#645 read the M=K+1 verify top1-top2 margin at ROOT forks only. This card extends
the SAME faithful replay to EVERY served position: re-serve the exact #632 Option-B
BI=1 spec lane (stored prompt_token_ids, temp=0, add_special_tokens=false,
ignore_eos=true, return_token_ids=true) + logprobs=N, and at every output position
read the verify top1-top2 logprob gap. stark #636's tau=0.5 acceptor FIRES at a
position iff that gap < 0.5 nat, so this is the served analog of his teacher-forced
7.80% fire rate.

Faithful replay anchor (identical to #645): the greedy stream is deterministic, so
the re-served completion reproduces #632 byte-exact (sha256) except at perfect
0.0-nat ULP ties, where requesting logprobs can flip the argmax (sampler reduction
order). Those flip positions are themselves fires (gap 0), so they do not bias the
fire COUNT; we report sha_ok per prompt and recompute the census on the byte-exact
subset as a robustness check.

analysis_only=true, official_tps=0. No HF Job, no generation beyond the replay,
no served-file change. The locked int4_g128_lmhead @126.378 file is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "ksweep"
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-e4b-it"

# #632 reference streams per K (run00 == run01 sha-verified, 128 prompts each)
REF = {
    3: KS / "k3" / "k3" / "decode" / "run00.jsonl",
    4: KS / "k4" / "k4" / "decode" / "run00.jsonl",
    5: KS / "k5" / "k5" / "decode" / "run00.jsonl",
    6: KS / "k6" / "k6" / "decode" / "run00.jsonl",
    7: KS / "k3" / "k7" / "decode" / "run00.jsonl",
}


def sha256_tokens(tokens) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


def post(payload, timeout_s=600):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def load_ref(k):
    recs = {}
    with REF[k].open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                recs[d["id"]] = d
    return recs


def extract_completion_token_ids(choice, prompt_token_ids):
    """Mirror official decode_outputs.extract_generated_token_ids precedence."""
    for v in (choice.get("token_ids"), choice.get("output_token_ids"),
              choice.get("completion_token_ids")):
        if isinstance(v, list) and all(isinstance(t, int) and t >= 0 for t in v):
            if len(v) >= len(prompt_token_ids) and v[:len(prompt_token_ids)] == prompt_token_ids:
                return v[len(prompt_token_ids):]
            return v
    raise ValueError("no token_ids in choice")


def per_position_gaps(choice, n_out):
    """top1-top2 logprob gap at every output position, from vLLM completion logprobs.
    Robust to ties: gap = max(values) - 2nd-max(values) over the top_logprobs dict.
    Returns (gaps[list], n_le1_topN[count of positions where <2 entries returned])."""
    lp = choice["logprobs"]
    tops = lp["top_logprobs"]  # list[dict token_str->logprob], one per output position
    gaps = []
    degenerate = 0
    for pos in range(n_out):
        vals = sorted(tops[pos].values(), reverse=True)
        if len(vals) >= 2:
            gaps.append(vals[0] - vals[1])
        else:
            # only 1 token returned (vocab edge / N=1) -> treat as a wide, non-firing gap
            degenerate += 1
            gaps.append(float("inf"))
    return gaps, degenerate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="num_speculative_tokens of the running server")
    ap.add_argument("--n-logprobs", type=int, default=20)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only first N prompts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (HERE / f"per_prompt_margins_k{args.k}.jsonl")
    ref = load_ref(args.k)
    ordered = sorted(ref.values(), key=lambda d: d["index"])
    if args.smoke:
        ordered = ordered[: args.smoke]

    t0 = time.time()
    n_sha_ok = 0
    n_sha_bad = 0
    total_pos = 0
    total_degenerate = 0
    rows = []
    for i, rec in enumerate(ordered):
        pid = rec["id"]
        ptoks = rec["prompt_token_ids"]
        payload = {
            "model": MODEL, "prompt": ptoks,
            "max_tokens": rec["num_completion_tokens"],
            "temperature": 0.0, "stream": False,
            "add_special_tokens": False, "ignore_eos": True,
            "return_token_ids": True, "logprobs": args.n_logprobs,
        }
        resp = post(payload)
        choice = resp["choices"][0]
        comp = extract_completion_token_ids(choice, ptoks)
        sha = sha256_tokens(comp)
        sha_ok = (sha == rec["completion_token_sha256"])
        n_sha_ok += sha_ok
        n_sha_bad += (not sha_ok)

        gaps, degenerate = per_position_gaps(choice, len(comp))
        total_pos += len(comp)
        total_degenerate += degenerate
        rows.append({
            "id": pid, "index": rec["index"], "dataset_index": rec.get("dataset_index"),
            "sha_ok": bool(sha_ok), "n_pos": len(comp),
            "completion_token_ids": comp,
            "margins": [None if g == float("inf") else g for g in gaps],
        })
        if (i + 1) % 16 == 0 or args.smoke:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(ordered)}] {pid} sha_ok={sha_ok} n_pos={len(comp)} "
                  f"fires<0.5={sum(g < 0.5 for g in gaps)} ({dt:.0f}s)", flush=True)

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\n[K={args.k}] faithful replay: {n_sha_ok}/{n_sha_ok + n_sha_bad} completion shas match #632")
    print(f"[K={args.k}] total positions censused: {total_pos} "
          f"(degenerate <2-token positions: {total_degenerate})")
    print(f"[out] wrote {len(rows)} per-prompt margin rows -> {out} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
