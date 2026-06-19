"""Merge several aime_eval.py output JSONs (year-split sub-cells of ONE body x budget
cell) into a single cell JSON the aggregator can consume unchanged.

Why: a full n=60 cell at the 12288 budget does not fit one SENPAI_TIMEOUT_MINUTES window
(per-problem wall is cap-gated by the longest of k samples). Running the 2024 (30) and
2025-I/2025-II (30) halves as separate aime_eval calls checkpoints the cell at the natural
year boundary; this concatenates their per_problem lists and recomputes the top-level
sampled/maj@k/finish-reason aggregates so {**meta, **result} stays schema-identical to a
single 60-problem run.

Pure-CPU, idempotent, order-preserving (problems are emitted in input-file order).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    return json.loads(Path(p).read_text())


def merge(parts: list[dict[str, Any]], label: str | None) -> dict[str, Any]:
    if not parts:
        raise SystemExit("merge: no input parts")
    # All parts must share the decode protocol (sampling) and k; guard it.
    base_sampling = parts[0].get("sampling")
    base_k = parts[0].get("maj_k")
    for d in parts[1:]:
        if d.get("sampling") != base_sampling:
            raise SystemExit(
                "merge: REFUSING to merge parts with different sampling configs:\n"
                f"  {base_sampling}\n  != {d.get('sampling')}"
            )
        if d.get("maj_k") != base_k:
            raise SystemExit(f"merge: parts disagree on maj_k ({base_k} != {d.get('maj_k')})")

    per: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for d in parts:
        for p in d.get("per_problem", []):
            pid = p.get("id")
            if pid in seen_ids:
                raise SystemExit(f"merge: duplicate problem id {pid!r} across parts (overlapping year sets?)")
            seen_ids.add(pid)
            per.append(p)

    n = len(per)
    n_correct_maj = sum(int(r.get("maj_correct", False)) for r in per)
    pass_rates = [r.get("pass_rate", 0.0) for r in per]
    total_samples = sum(r.get("k", len(r.get("finish_reasons", []))) for r in per)
    extract_fail = sum(1 for r in per for a in r.get("answers", []) if a is None)

    out = {**parts[0]}  # inherit meta (sampling, model, base_url, etc.) from first part
    out["label"] = label or (parts[0].get("label") or "merged")
    out["years"] = sorted({y for d in parts for y in (d.get("years") or [])})
    out["n_problems"] = n
    out["maj_k"] = base_k
    out["maj_k_accuracy"] = n_correct_maj / n if n else 0.0
    out["n_correct_maj"] = n_correct_maj
    out["mean_pass_rate"] = sum(pass_rates) / n if n else 0.0
    out["extract_fail_rate"] = extract_fail / total_samples if total_samples else 0.0
    out["total_samples"] = total_samples
    out["wall_s"] = sum(d.get("wall_s", 0.0) for d in parts)
    out["per_problem"] = per
    out["_merged_from"] = [d.get("label") for d in parts]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parts", nargs="+", type=Path, help="aime_eval JSONs to concatenate")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    merged = merge([_load(p) for p in args.parts], args.label)
    args.out.write_text(json.dumps(merged, indent=2))
    fin = [fr for p in merged["per_problem"] for fr in p.get("finish_reasons", [])]
    n_len = sum(1 for fr in fin if fr == "length")
    print(
        f"[merge] {len(args.parts)} parts -> {args.out}  n={merged['n_problems']} "
        f"years={merged['years']} maj@{merged['maj_k']}={merged['maj_k_accuracy']:.4f} "
        f"mean_pass_rate={merged['mean_pass_rate']:.4f} "
        f"trunc={n_len}/{len(fin)}={ (n_len/len(fin) if fin else 0):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
