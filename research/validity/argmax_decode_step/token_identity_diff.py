#!/usr/bin/env python3
"""Token-identity diff for PR #180 (ubel #154 argmax-only decode realization).

Compares completion_token_ids between a control (ARGMAX_ONLY_DECODE off) and a
patched (ARGMAX_ONLY_DECODE=1) decode jsonl, keyed by prompt index. Greedy decode
must be byte-identical for the patch to be output-neutral.

Usage:
    token_identity_diff.py CONTROL.jsonl PATCHED.jsonl [--show 6]

Exit code 0 iff every shared prompt is token-identical.
"""
import argparse
import json
import sys


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("index")
            if idx is None:
                idx = len(rows)
            rows[idx] = r.get("completion_token_ids") or []
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("patched")
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ctrl = load(args.control)
    patch = load(args.patched)
    shared = sorted(set(ctrl) & set(patch))

    n_ident = 0
    first_div = {}
    for idx in shared:
        c, p = ctrl[idx], patch[idx]
        same = c == p
        n_ident += int(same)
        if not same:
            # find first divergent position
            pos = next((i for i in range(min(len(c), len(p))) if c[i] != p[i]),
                       min(len(c), len(p)))
            first_div[idx] = pos
        print(f"prompt {idx}: identical={same} len(c)={len(c)} len(p)={len(p)} "
              f"patched[:{args.show}]={p[:args.show]}"
              + ("" if same else f" control[:{args.show}]={c[:args.show]} first_div_pos={first_div[idx]}"))

    rate = n_ident / len(shared) if shared else 0.0
    print(f"IDENTICAL {n_ident}/{len(shared)}  token_identity_rate={rate:.6f}")

    if args.json_out:
        json.dump({
            "n_shared": len(shared),
            "n_identical": n_ident,
            "token_identity_rate": rate,
            "first_divergence_pos": first_div,
            "control": args.control,
            "patched": args.patched,
        }, open(args.json_out, "w"), indent=2)

    sys.exit(0 if (shared and n_ident == len(shared)) else 1)


if __name__ == "__main__":
    main()
