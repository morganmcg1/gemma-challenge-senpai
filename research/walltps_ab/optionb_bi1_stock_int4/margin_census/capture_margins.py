"""PR #645 step 2: re-serve the K=7 Option-B BI=1 spec lane with logprob capture,
read the M=8 verify-forward margin at each ROOT fork.

Faithful replay: feed the EXACT stored prompt_token_ids, identical /v1/completions
payload as the #632 capture (temp=0, add_special_tokens=false, ignore_eos=true,
return_token_ids=true) PLUS logprobs=N. Greedy is deterministic, so the token
stream reproduces #632 -- verified per-prompt via completion_token_sha256.

At each root position p (first divergence vs served M=1 AR):
  A = emitted spec token (M=8 argmax)         logp_A = top-1 logprob (token_logprobs[p])
  B = AR M=1 argmax token                     logp_B = B's logprob in the M=8 top-N
  C = M=8 runner-up (2nd highest)             logp_C = 2nd logprob
  margin_AB = logp_A - logp_B   (PR instruction #2: AR-token vs emitted-token gap)
  margin_AC = logp_A - logp_C   (the literal M=8 top1-top2 gap stark thresholds on)
At a root fork B should == C (see PR analysis); we measure both and report agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = HERE.parent / "ksweep"
K7 = KS / "k3" / "k7" / "decode" / "run00.jsonl"
ROOT_FORKS = HERE / "root_forks.jsonl"
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-e4b-it"


def sha256_tokens(tokens) -> str:
    body = ",".join(str(t) for t in tokens)
    return hashlib.sha256(body.encode("ascii")).hexdigest()


def post(payload, timeout_s=300):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def load_k7_records():
    recs = {}
    with K7.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            recs[d["id"]] = d
    return recs


def load_root_forks():
    forks = {}
    with ROOT_FORKS.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            forks[d["id"]] = d
    return forks


def extract_completion_token_ids(choice, prompt_token_ids):
    """Mirror official decode_outputs.extract_generated_token_ids precedence."""
    for v in (choice.get("token_ids"), choice.get("output_token_ids"),
              choice.get("completion_token_ids")):
        if isinstance(v, list) and all(isinstance(t, int) and t >= 0 for t in v):
            if len(v) >= len(prompt_token_ids) and v[:len(prompt_token_ids)] == prompt_token_ids:
                return v[len(prompt_token_ids):]
            return v
    raise ValueError("no token_ids in choice")


def top_logprobs_at(choice, pos):
    """Return (emitted_token_logprob, [(token_id_or_str, logprob), ...] sorted desc)
    for output position `pos`, from vLLM completion logprobs."""
    lp = choice["logprobs"]
    token_logprobs = lp["token_logprobs"]
    top = lp["top_logprobs"][pos]  # dict: token_str -> logprob (OpenAI shape)
    emitted_lp = token_logprobs[pos]
    items = sorted(top.items(), key=lambda kv: kv[1], reverse=True)
    return emitted_lp, items, lp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-logprobs", type=int, default=20)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only first N prompts + dump raw")
    ap.add_argument("--out", default=str(HERE / "margin_records.jsonl"))
    args = ap.parse_args()

    k7 = load_k7_records()
    forks = load_root_forks()
    # preserve #632 prompt ORDER (index field)
    ordered = sorted(k7.values(), key=lambda d: d["index"])
    if args.smoke:
        # smoke: pick first `smoke` DIVERGED prompts so we exercise root extraction
        diverged_ids = [d["id"] for d in ordered if d["id"] in forks]
        ordered = [k7[i] for i in diverged_ids[: args.smoke]]

    out_recs = []
    n_sha_ok = 0
    n_sha_bad = 0
    for rec in ordered:
        pid = rec["id"]
        ptoks = rec["prompt_token_ids"]
        payload = {
            "model": MODEL,
            "prompt": ptoks,
            "max_tokens": rec["num_completion_tokens"],
            "temperature": 0.0,
            "stream": False,
            "add_special_tokens": False,
            "ignore_eos": True,
            "return_token_ids": True,
            "logprobs": args.n_logprobs,
        }
        resp = post(payload)
        choice = resp["choices"][0]
        comp = extract_completion_token_ids(choice, ptoks)
        sha = sha256_tokens(comp)
        sha_ok = (sha == rec["completion_token_sha256"])
        n_sha_ok += sha_ok
        n_sha_bad += (not sha_ok)

        if args.smoke:
            print(f"\n=== {pid} (idx {rec['index']}) sha_ok={sha_ok} len={len(comp)} ===")
            lp = choice["logprobs"]
            print("logprobs keys:", list(lp.keys()))
            for k, v in lp.items():
                vlen = len(v) if isinstance(v, list) else "-"
                print(f"  {k}: {type(v).__name__} len={vlen}")
            # show raw top_logprobs at the root position
            if pid in forks:
                p = forks[pid]["root_pos"]
                print(f"  root_pos={p} A(spec)={forks[pid]['spec_token_A']} B(ar)={forks[pid]['ar_token_B']}")
                print(f"  token_logprobs[p]={lp['token_logprobs'][p]}")
                print(f"  tokens[p]={lp['tokens'][p]!r}" if 'tokens' in lp else "  (no tokens field)")
                print(f"  raw top_logprobs[p]={json.dumps(lp['top_logprobs'][p])[:600]}")
                # any token-id channel?
                for cand in ("token_ids", "top_token_ids"):
                    if cand in lp:
                        print(f"  lp[{cand}][p]={lp[cand][p]}")

        if pid not in forks:
            continue  # non-diverged: no root fork to score

        f = forks[pid]
        p = f["root_pos"]
        A, B = f["spec_token_A"], f["ar_token_B"]
        # faithfulness: emitted token id at p must equal A
        emitted_id_at_p = comp[p]
        emitted_lp, items, lp = top_logprobs_at(choice, p)
        rec_out = {
            "id": pid, "root_pos": p, "A": A, "B": B,
            "emitted_id_at_p": emitted_id_at_p,
            "emitted_eq_A": (emitted_id_at_p == A),
            "sha_ok": sha_ok,
            "logp_A": emitted_lp,
            "raw_top": items,  # list of [token_str, logprob] (or [id, lp] if vLLM gives ids)
        }
        out_recs.append(rec_out)

    print(f"\n[sha] faithful replay: {n_sha_ok}/{n_sha_ok + n_sha_bad} completion shas match #632")
    if not args.smoke:
        with open(args.out, "w") as fh:
            for r in out_recs:
                fh.write(json.dumps(r) + "\n")
        print(f"[out] wrote {len(out_recs)} root-fork logprob records -> {args.out}")
    return 0 if n_sha_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
