#!/usr/bin/env python
"""Greedy-identity check for the lmhead12k empirical prune.

Two modes:

  static (default, CPU, no GPU):
      Proof by containment. For a row-pruned lm_head, the per-step greedy token is
      unchanged iff the full-vocab argmax token is in kept_ids: removing non-max
      rows cannot change which kept row is the maximum, so if the true argmax is
      kept it stays the argmax over the kept subset. The captured baseline decode
      outputs ARE the full-vocab argmax tokens, so verifying every emission is in
      kept_ids proves GREEDY_IDENTICAL on the captured prompts -- with no serving.
      Caveat: the bucket capture covers 31/128 benchmark prompts; the other 97 are
      unproven by this static check and must be confirmed by the served verifier.

  served (--candidate PATH):
      Delegate to the official verifier (greedy_identity.compare_files) with the
      baseline decode_outputs as REFERENCE and the pruned-served decode_outputs as
      CANDIDATE. This is the authoritative gate; run it after serving.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECODE_FILE = ROOT / "research/local_validation/vllm_baseline_128/decode_outputs.jsonl"
DECODE_FILE_FALLBACK = ROOT / "research/local_validation/vllm_baseline/decode_outputs_128.jsonl"
KEPT_IDS = ROOT / "submissions/lmhead12k_empirical/kept_ids.json"
OFFICIAL_VERIFIER_DIR = (
    ROOT / "official/main_bucket/shared_resources/"
    "gemma_greedy_identity_verifier_flowian-powers"
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def static_check(decode_file: Path, kept_ids_file: Path) -> int:
    decode = _read_jsonl(decode_file)
    kept = set(json.loads(kept_ids_file.read_text())["kept_ids"])
    divergent = []
    n_tokens = 0
    for rec in decode:
        toks = rec["completion_token_ids"]
        n_tokens += len(toks)
        bad = [(i, t) for i, t in enumerate(toks) if t not in kept]
        if bad:
            divergent.append({"id": rec["id"], "first_div_index": bad[0][0],
                              "n_clipped": len(bad)})
    verdict = "GREEDY_IDENTICAL" if not divergent else "DIVERGENT"
    n = len(decode)
    caveat = (
        f"Static proof over all {n}/128 benchmark prompts (full audit set)."
        if n >= 128 else
        f"Proves greedy identity only on the {n}/128 captured prompts; the rest "
        f"must be confirmed by the served verifier."
    )
    report = {
        "mode": "static-containment",
        "verdict": verdict,
        "prompts_checked": n,
        "prompts_checked_of_benchmark": 128,
        "tokens_checked": n_tokens,
        "divergent_prompts": divergent,
        "caveat": caveat,
    }
    print(json.dumps(report, indent=2))
    return 0 if verdict == "GREEDY_IDENTICAL" else 1


def served_check(reference: Path, candidate: Path) -> int:
    sys.path.insert(0, str(OFFICIAL_VERIFIER_DIR))
    import greedy_identity  # type: ignore

    report = greedy_identity.compare_files(str(reference), str(candidate))
    print(json.dumps(report.to_dict(), indent=2))
    return {"GREEDY_IDENTICAL": 0, "DIVERGENT": 1}.get(report.verdict, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decode-file", default=str(DECODE_FILE),
                    help="baseline greedy decode outputs (static mode reference)")
    ap.add_argument("--kept-ids", default=str(KEPT_IDS))
    ap.add_argument("--candidate", default=None,
                    help="pruned-served decode_outputs.jsonl; switches to the "
                         "authoritative served verifier")
    args = ap.parse_args()
    decode_file = Path(args.decode_file)
    if not decode_file.exists() and DECODE_FILE_FALLBACK.exists():
        decode_file = DECODE_FILE_FALLBACK
    if args.candidate:
        return served_check(decode_file, Path(args.candidate))
    return static_check(decode_file, Path(args.kept_ids))


if __name__ == "__main__":
    raise SystemExit(main())
