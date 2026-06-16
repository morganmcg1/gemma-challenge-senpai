"""PR #503 — pairwise operative-identity census for arbitrary config pairs.

analyze.py censuses every config vs the AR floor. That answers "does drafter X's
output match M=1 AR?" — but on the *deployed* (non-strict) stack the answer is
contaminated by two distinct effects we must separate to make an honest
strict-safety claim:

  1. cross-session nondeterminism (two separate server processes can differ even
     for the *same* config — bf16 attention / CUDA-graph capture, fixed seed=0);
  2. the M>1 verify reduction-order tax (spec-on verifies at M=k+1 in a wider
     captured graph than M=1 AR), which flips an occasional argmax that then
     cascades the rest of the greedy trajectory.

Neither is a *drafter* defect. To isolate the drafter we need pairwise censuses:

  * AR vs AR'  (ar_floor vs ar_floor2): pure cross-session noise floor.
  * ngram(M=8) vs MTP(M=8) at MATCHED verify width: isolates the drafter — a
    lossless greedy spec-dec method emits exactly the target's argmax at the
    accepted positions, so at matched width the *drafter* must be irrelevant and
    this should sit at (or below) the AR-vs-AR noise floor.
  * each-vs-AR: the shared M>1 tax — must be ~equal for ngram and MTP.

Reports prompt-level identity, token-level identity, and the first-divergence
position distribution (the honest metric: after one flip a greedy trajectory
cascades, so raw token mismatch over-counts a single root event).

Local A10G probe — NOT official a10g-small TPS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _per_prompt_map(rec: dict[str, Any]) -> dict[Any, list[int]]:
    out = {}
    for p in rec.get("per_prompt", []) or []:
        out[p.get("index")] = p.get("completion_token_ids") or []
    return out


def census_pair(cand: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    c = _per_prompt_map(cand)
    r = _per_prompt_map(ref)
    shared = sorted(set(c) & set(r), key=lambda x: (x is None, x))
    n = len(shared)
    identical = 0
    tot = 0
    mism = 0
    first_div = []
    for idx in shared:
        a = r[idx]
        b = c[idx]
        m = min(len(a), len(b))
        tot += m
        diffs = [i for i in range(m) if a[i] != b[i]]
        mism += len(diffs)
        if not diffs and len(a) == len(b):
            identical += 1
        elif diffs:
            first_div.append(diffs[0])
    return {
        "n_prompts": n,
        "identical_prompts": identical,
        "prompt_identity": (identical / n) if n else None,
        "tokens_compared": tot,
        "mismatch_tokens": mism,
        "token_identity": (1.0 - mism / tot) if tot else None,
        "token_flip_rate": (mism / tot) if tot else None,
        "n_divergent_prompts": len(first_div),
        "first_divergence_positions": sorted(first_div),
        "min_first_divergence": min(first_div) if first_div else None,
    }


def _get(results: dict[str, Any], name: str) -> dict[str, Any]:
    rec = results.get("configs", {}).get(name)
    if rec is None:
        raise KeyError(f"config {name!r} not in results (have: "
                       f"{sorted(results.get('configs', {}).keys())})")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="results JSON (may be merged across runs via --extra)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional results JSONs whose configs are merged in")
    ap.add_argument("--pair", action="append", default=[], metavar="CAND:REF",
                    help="census CAND vs REF (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    merged: dict[str, Any] = json.loads(Path(args.results).read_text())
    merged.setdefault("configs", {})
    for extra in args.extra:
        e = json.loads(Path(extra).read_text())
        merged["configs"].update(e.get("configs", {}))

    out: dict[str, Any] = {"pairs": {}}
    print(f"{'pair':28s} {'promptID':>9s} {'ident/N':>9s} {'tokID':>8s} "
          f"{'flip':>7s} {'minFDiv':>8s}")
    for spec in args.pair:
        cand_name, ref_name = spec.split(":", 1)
        cen = census_pair(_get(merged, cand_name), _get(merged, ref_name))
        out["pairs"][spec] = cen
        print(f"{spec:28s} {_f(cen['prompt_identity']):>9s} "
              f"{cen['identical_prompts']:>4d}/{cen['n_prompts']:<4d} "
              f"{_f(cen['token_identity']):>8s} {_f(cen['token_flip_rate']):>7s} "
              f"{str(cen['min_first_divergence']):>8s}")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[census_pairs] -> {args.out}")
    return 0


def _f(x: Any) -> str:
    return "—" if x is None else f"{float(x):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
