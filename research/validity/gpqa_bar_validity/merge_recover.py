#!/usr/bin/env python3
"""PR #614 -- splice a recovered single-item GPQA record into a base run json.

One GPQA-Diamond item (id recnTTKdBzfuoZ7w7, input ~2049 tok) overflows the 6144
context at max_tokens=4096 (2049+4096=6145>6144) and is force-scored WRONG via a
vLLM BadRequestError -- a context-length artifact, NOT a model failure. Leaving it
in would compound the exact downward bias this audit measures. We re-run JUST that
item at max_tokens=4095 (2049+4095=6144, the largest budget the 6144 context allows
for a 2049-tok input) on the SAME validated-clean serve, then splice its real record
in and recompute the summary fields run_eval.py reports. Honest + documented: this
one item is context-limited to 4095 output tokens; all others get the full 4096.

Usage: merge_recover.py --base runs/sampled_4096_s1.json --recover runs/_rec_s1.json \
        --out runs/sampled_4096_s1.merged.json
"""
from __future__ import annotations

import argparse
import json


def _recompute(d: dict) -> dict:
    ps = d["per_sample"]
    n_scored = sum(1 for r in ps if r.get("value") in ("C", "I"))
    n_correct = sum(1 for r in ps if r.get("correct"))
    n_error = sum(1 for r in ps if r.get("error"))
    n_empty = sum(1 for r in ps if r.get("empty"))
    n_length = sum(1 for r in ps if r.get("truncated"))
    have_ot = [r for r in ps if r.get("output_tokens") is not None]
    n_len_2048 = sum(1 for r in have_ot if r["output_tokens"] > 2048)
    n = len(ps)
    d["n_samples"] = n
    d["n_scored"] = n_scored
    d["n_correct"] = n_correct
    d["n_error"] = n_error
    d["n_empty"] = n_empty
    d["empty_rate"] = (n_empty / n) if n else None
    d["n_length"] = n_length
    d["finish_length_rate"] = (n_length / n) if n else None
    d["n_length_at_2048"] = n_len_2048 if have_ot else None
    d["finish_length_rate_at_2048"] = (n_len_2048 / len(have_ot)) if have_ot else None
    d["accuracy"] = (n_correct / n_scored) if n_scored else float("nan")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--recover", required=True, help="run_eval.py json over the single recovered id")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = json.load(open(args.base))
    rec = json.load(open(args.recover))
    rec_by_id = {r["id"]: r for r in rec["per_sample"]}
    if not rec_by_id:
        raise SystemExit("[merge] FATAL: recover json has no per_sample records")

    replaced = []
    new_ps = []
    for r in base["per_sample"]:
        if r["id"] in rec_by_id:
            rr = dict(rec_by_id[r["id"]])
            # carry the recovery provenance so the audit trail is explicit
            rr["recovered_at_max_tokens"] = rec.get("max_tokens")
            rr["recovered_from_error"] = r.get("error") is not None
            new_ps.append(rr)
            replaced.append(r["id"])
        else:
            new_ps.append(r)
    base["per_sample"] = sorted(new_ps, key=lambda r: r["id"])
    base["recovered_ids"] = replaced
    base = _recompute(base)

    with open(args.out, "w") as f:
        json.dump(base, f, indent=2)
    print(f"[merge] replaced={replaced} -> acc={base['accuracy']:.4f} "
          f"n_correct={base['n_correct']}/{base['n_scored']} n_error={base['n_error']} "
          f"len@cap={base['finish_length_rate']:.4f} len@2048={base['finish_length_rate_at_2048']:.4f} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
