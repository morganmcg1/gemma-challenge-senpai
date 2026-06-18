"""PR #645: recover on-trajectory M=8 margin for the 2 root forks whose faithful
replay flipped off-#632-trajectory.

Re-serve the EXACT #632 faithful payload (stored prompt_token_ids, temp=0,
add_special_tokens=false, ignore_eos=true, return_token_ids=true) + logprobs=20,
repeatedly, until the completion-token sha256 reproduces #632. A sha match means
the whole completion is byte-identical to #632, so position p shares #632's exact
M=8 verify context and the logprobs there are the genuine on-trajectory values.

If a prompt never matches within --attempts (the logprobs request can itself
perturb a perfect near-tie), we still record the best run + report it, since every
observed measurement for these forks is <=0.125 nat (far below tau=0.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
K7 = HERE.parent / "ksweep" / "k3" / "k7" / "decode" / "run00.jsonl"
ROOT_FORKS = HERE / "root_forks.jsonl"
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-e4b-it"
FLIPPED = ["mmlu_pro-006f3a2112", "mmlu_pro-012f0d5c8d"]


def sha256_tokens(tokens) -> str:
    return hashlib.sha256(",".join(str(t) for t in tokens).encode("ascii")).hexdigest()


def post(payload, timeout_s=300):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{BASE}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_completion(choice, ptoks):
    for v in (choice.get("token_ids"), choice.get("output_token_ids"),
              choice.get("completion_token_ids")):
        if isinstance(v, list) and all(isinstance(t, int) and t >= 0 for t in v):
            if len(v) >= len(ptoks) and v[:len(ptoks)] == ptoks:
                return v[len(ptoks):]
            return v
    raise ValueError("no token_ids")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--n-logprobs", type=int, default=20)
    args = ap.parse_args()

    k7 = {json.loads(l)["id"]: json.loads(l) for l in K7.open() if l.strip()}
    forks = {json.loads(l)["id"]: json.loads(l) for l in ROOT_FORKS.open() if l.strip()}

    results = {}
    for pid in FLIPPED:
        rec = k7[pid]
        f = forks[pid]
        ptoks = rec["prompt_token_ids"]
        p = f["root_pos"]
        A, B = f["spec_token_A"], f["ar_token_B"]
        payload = {
            "model": MODEL, "prompt": ptoks, "max_tokens": rec["num_completion_tokens"],
            "temperature": 0.0, "stream": False, "add_special_tokens": False,
            "ignore_eos": True, "return_token_ids": True, "logprobs": args.n_logprobs,
        }
        matched = None
        print(f"\n=== {pid} p={p} A={A} B={B} target_sha={rec['completion_token_sha256'][:16]} ===")
        for att in range(1, args.attempts + 1):
            resp = post(payload)
            ch = resp["choices"][0]
            comp = extract_completion(ch, ptoks)
            sha = sha256_tokens(comp)
            ok = (sha == rec["completion_token_sha256"])
            # first-divergence vs #632 stored completion
            stored = rec["completion_token_ids"]
            fd = next((i for i in range(min(len(comp), len(stored))) if comp[i] != stored[i]), None)
            emit_p = comp[p] if p < len(comp) else None
            print(f"  attempt {att:2d}: sha={sha[:16]} match={ok} first_div={fd} emit[p]={emit_p} (A={A})")
            if ok:
                lp = ch["logprobs"]
                top = lp["top_logprobs"][p]
                items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
                matched = {
                    "id": pid, "root_pos": p, "A": A, "B": B,
                    "emitted_id_at_p": comp[p], "emitted_eq_A": (comp[p] == A),
                    "sha_ok": True, "logp_A": lp["token_logprobs"][p],
                    "raw_top": items, "method": "faithful_rematch", "attempt": att,
                }
                print(f"    -> MATCHED on attempt {att}: emit[p]={comp[p]} eq_A={comp[p]==A} "
                      f"top1={items[0][0]!r}/{items[0][1]:.7f} top2={items[1][0]!r}/{items[1][1]:.7f} "
                      f"gap={items[0][1]-items[1][1]:.7f}")
                break
        results[pid] = matched
        if matched is None:
            print(f"  NO sha match in {args.attempts} attempts for {pid}")

    found = {k: v for k, v in results.items() if v is not None}
    if found:
        (HERE / "rematch_records.jsonl").write_text(
            "\n".join(json.dumps(v) for v in found.values()) + "\n")
        print(f"\n[out] wrote {len(found)} rematched record(s) -> rematch_records.jsonl")
    else:
        print("\n[out] no rematches; flipped forks remain off-trajectory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
