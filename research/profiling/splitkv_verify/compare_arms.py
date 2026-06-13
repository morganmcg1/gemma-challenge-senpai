"""Direct token-level A/B of the two served decode files (patched 3D split-KV
verify vs baseline 2D verify), matched by prompt hash.

The PR's Step-3 greedy gate compares the served decode against a *spec-off* M=1 AR
reference; but the unpatched baseline already diverges from that reference (int4
M=8-batched-verify vs M=1-AR rounding, ~0.33%/tok, cascades over 512 tokens). So
that gate cannot isolate THIS patch. The honest test of "does the 3D split-KV path
change the served output" is patched-decode vs baseline-decode directly — same
serving stack, only the verify-attention dispatch differs.

    python research/profiling/splitkv_verify/compare_arms.py \
        --baseline research/profiling/splitkv_verify/ab_verify_n128/decode_baseline.jsonl \
        --patched  research/profiling/splitkv_verify/ab_verify_n128/decode_patched.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["prompt_sha256"]] = d["completion_token_ids"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--patched", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    base = _load(args.baseline)
    pat = _load(args.patched)
    shared = sorted(set(base) & set(pat))

    identical = 0
    divergent = 0
    onsets: list[int] = []
    total_div_tokens = 0
    per_prompt = []
    for h in shared:
        b, p = base[h], pat[h]
        n = min(len(b), len(p))
        onset = None
        diffs = 0
        for i in range(n):
            if b[i] != p[i]:
                if onset is None:
                    onset = i
                diffs += 1
        # length mismatch also counts as divergence past the shorter length
        diffs += abs(len(b) - len(p))
        if onset is None and len(b) == len(p):
            identical += 1
        else:
            divergent += 1
            onsets.append(onset if onset is not None else n)
            total_div_tokens += diffs
        per_prompt.append({"prompt_sha256": h, "onset": onset,
                           "diff_tokens": diffs, "len_base": len(b), "len_pat": len(p)})

    result = {
        "baseline_file": str(args.baseline),
        "patched_file": str(args.patched),
        "num_shared_prompts": len(shared),
        "num_identical": identical,
        "num_divergent": divergent,
        "identical_frac": identical / len(shared) if shared else None,
        "total_divergent_tokens": total_div_tokens,
        "mean_divergent_tokens_per_divergent_prompt": (
            total_div_tokens / divergent if divergent else 0),
        "onset_min": min(onsets) if onsets else None,
        "onset_median": int(statistics.median(onsets)) if onsets else None,
        "onset_max": max(onsets) if onsets else None,
    }
    print(json.dumps(result, indent=2))
    out = args.out or args.patched.parent / "compare_arms.json"
    out.write_text(json.dumps({"summary": result, "per_prompt": per_prompt}, indent=2))
    print(f"[compare] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
