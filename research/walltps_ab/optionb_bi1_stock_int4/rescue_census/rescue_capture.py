"""PR #651: served-path recompute-RESCUE census (correctness complement to #648 coverage).

#648 measured WHERE stark #636's tau=0.5 flag fires on the served stream (per-position
top1-top2 verify margin < 0.5). This card asks whether each of those served fires actually
RESCUES: at a fired position, does stark's M=1-recompute step -- a true single-seq M=1
(spec-OFF) greedy forward in the SERVED context -- land on the served-AR reference token
ar_ref_bi1[pos]? stark's teacher-forced analog is rescued_break_rate = 0/14035; this is the
SERVED-path analog, on the actual (potentially off-AR) trajectory.

Faithfulness: the M=1 recompute MUST be a single-seq M=1 decode on the spec-OFF reference
engine (the SAME engine that generated ar_ref_bi1), NOT a prompt_logprobs prefill -- BI=1 is
NOT M-invariant (the int4 Marlin GEMM is the lone M-dependent reduction, merged #122), so a
batched/prefill recompute would change the argmax at exactly the near-ties we probe. We send
one max_tokens=1 request per fired position (prompt + served_completion[:pos]); APC (on by
default in this V1 engine) caches the growing prefix when we walk pos ascending per prompt.

For each fired position we record three token ids:
  r_pos = M=1 recompute argmax (the acceptor's output)
  a_pos = ar_ref_bi1[pos]               (the served-AR reference token; PR step 1)
  s_pos = served_completion[pos]        (what the spec lane emitted = verify argmax)
rescue = (r_pos == a_pos). We also keep the recompute's own top1-top2 margin (id-free ULP-tie
signal) and, for BREAKS only, the raw top-N logprobs so the analyzer can classify a break as a
benign 0.0-nat ULP tie (a_pos tied with r_pos) vs a wider, genuine miss (PR step 3).

analysis_only=true, official_tps=0. No HF Job, no served-file change, locked 126.378 untouched.
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
MARGINS_DIR = HERE.parent / "fire_census"
AR_REF = KS / "ar_ref_bi1" / "decode_outputs.jsonl"
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-e4b-it"
TAU = 0.5  # stark #636 recompute-flag threshold (nat)

# #632 reference streams per K -> source of prompt_token_ids (same map as #648 fire_capture)
REF = {
    3: KS / "k3" / "k3" / "decode" / "run00.jsonl",
    4: KS / "k4" / "k4" / "decode" / "run00.jsonl",
    5: KS / "k5" / "k5" / "decode" / "run00.jsonl",
    6: KS / "k6" / "k6" / "decode" / "run00.jsonl",
    7: KS / "k3" / "k7" / "decode" / "run00.jsonl",
}


def post(payload, timeout_s=600):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def load_jsonl_by_id(path, keep):
    out = {}
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = {k: d.get(k) for k in keep}
    return out


def first_divergence(served_ids, ar_ids):
    n = min(len(served_ids), len(ar_ids))
    for i in range(n):
        if served_ids[i] != ar_ids[i]:
            return i
    return n  # identical up to compared length -> never diverges in this window


def recompute_one(prefix_ids, n_logprobs):
    """One true single-seq M=1 (spec-OFF) greedy step. Returns (r_pos id, recompute top1-top2
    margin, raw top_logprobs dict str->logprob)."""
    payload = {
        "model": MODEL, "prompt": prefix_ids,
        "max_tokens": 1, "temperature": 0.0, "stream": False,
        "add_special_tokens": False, "ignore_eos": True,
        "return_token_ids": True, "logprobs": n_logprobs,
    }
    resp = post(payload)
    choice = resp["choices"][0]
    # r_pos: the single generated token id (return_token_ids=true -> choices[0].token_ids)
    tids = None
    for v in (choice.get("token_ids"), choice.get("output_token_ids"),
              choice.get("completion_token_ids")):
        if isinstance(v, list) and v and isinstance(v[0], int):
            tids = v
            break
    if tids is None:
        raise ValueError(f"no generated token_ids in choice keys={list(choice.keys())}")
    r_pos = tids[0]
    # recompute top1-top2 margin (id-free): the gap between the argmax and runner-up logprob
    top = choice["logprobs"]["top_logprobs"][0]  # {token_str: logprob}
    vals = sorted(top.values(), reverse=True)
    rc_margin = (vals[0] - vals[1]) if len(vals) >= 2 else float("inf")
    return r_pos, rc_margin, top


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n-logprobs", type=int, default=20)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only first N prompts")
    ap.add_argument("--all-positions", action="store_true",
                    help="smoke/validation: recompute EVERY position (not just fires)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (HERE / f"rescue_k{args.k}.jsonl")

    margins = load_jsonl_by_id(
        MARGINS_DIR / f"per_prompt_margins_k{args.k}.jsonl",
        ("index", "sha_ok", "n_pos", "completion_token_ids", "margins"),
    )
    prompts = load_jsonl_by_id(REF[args.k], ("prompt_token_ids",))
    ar = load_jsonl_by_id(AR_REF, ("completion_token_ids",))

    ordered = sorted(margins.items(), key=lambda kv: kv[1]["index"])
    if args.smoke:
        ordered = ordered[: args.smoke]

    t0 = time.time()
    rows = []
    n_fire = n_rescue = n_break = 0
    pre_fire = pre_break = post_fire = post_break = 0
    for pi, (pid, m) in enumerate(ordered):
        served = m["completion_token_ids"]
        marg = m["margins"]
        ptoks = prompts[pid]["prompt_token_ids"]
        ar_ids = ar[pid]["completion_token_ids"]
        fdiv = first_divergence(served, ar_ids)
        n_pos = len(marg)
        if args.all_positions:
            positions = list(range(min(n_pos, len(ar_ids))))
        else:
            positions = [i for i, g in enumerate(marg)
                         if g is not None and g < TAU and i < len(ar_ids)]
        # ascending pos -> APC reuses the growing prefix for this prompt
        for pos in positions:
            prefix = ptoks + served[:pos]
            r_pos, rc_margin, top = recompute_one(prefix, args.n_logprobs)
            a_pos = ar_ids[pos]
            s_pos = served[pos]
            rescue = (r_pos == a_pos)
            is_fire = (marg[pos] is not None and marg[pos] < TAU)
            pre = pos < fdiv
            row = {
                "id": pid, "index": m["index"], "pos": pos, "pre_div": bool(pre),
                "fdiv": fdiv, "is_fire": bool(is_fire), "sha_ok": bool(m["sha_ok"]),
                "verify_margin": marg[pos], "recompute_margin": rc_margin,
                "r_pos": r_pos, "a_pos": a_pos, "s_pos": s_pos,
                "rescue": bool(rescue), "r_eq_s": bool(r_pos == s_pos),
            }
            if not rescue:
                # keep fat top-N only for breaks (ULP-tie vs wide classification offline)
                row["break_top"] = top
            rows.append(row)
            if is_fire:
                n_fire += 1
                if rescue:
                    n_rescue += 1
                else:
                    n_break += 1
                if pre:
                    pre_fire += 1
                    pre_break += (not rescue)
                else:
                    post_fire += 1
                    post_break += (not rescue)
        if (pi + 1) % 8 == 0 or args.smoke:
            dt = time.time() - t0
            br = (n_break / n_fire) if n_fire else 0.0
            print(f"  [{pi+1}/{len(ordered)}] {pid} fires={n_fire} rescued={n_rescue} "
                  f"broken={n_break} (break_rate={br*100:.3f}%) "
                  f"pre_brk={pre_break}/{pre_fire} post_brk={post_break}/{post_fire} "
                  f"({dt:.0f}s)", flush=True)

    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    br = (n_break / n_fire) if n_fire else 0.0
    print(f"\n[K={args.k}] fired positions censused: {n_fire}")
    print(f"[K={args.k}] rescued={n_rescue} broken={n_break} "
          f"served_break_rate={br*100:.4f}%")
    print(f"[K={args.k}] pre-div(on-AR head) breaks={pre_break}/{pre_fire} ; "
          f"post-div(off-AR tail) breaks={post_break}/{post_fire}")
    print(f"[out] wrote {len(rows)} rows -> {out} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
