#!/usr/bin/env python
"""PR #673 -- per-TOKEN greedy-divergence analysis of the drafter-free spec screen.

The run_drafterfree.py screen logs a per-PROMPT byte-identity break_rate (sha256 of
the whole completion). This script refines that into the per-TOKEN mechanism signal
-- divergence rate, onset position, cascade length -- by aligning each spec cell's
completion_token_ids against the AR anchor position-by-position (conc=1, ignore_eos,
max_tokens=512 => every completion is exactly OUTPUT_LEN tokens, so positions align).

This connects the drafter-FREE break to the documented drafter mechanism: PR #114
(spec-ON 56.1% tok divergence vs own M=1 AR, onset median ~121/512) and PR #5
(0.33-0.72%/tok near-tie argmax flip). Pure offline analysis of existing decode
jsonls -- NO GPU, NO server.

Usage::
    .venv/bin/python research/validity/drafterfree_specdec/analyze_token_divergence.py \
        --run-dir research/validity/drafterfree_specdec/_runs/screen_K56
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_completions(decode_jsonl: Path) -> dict[str, list[int]]:
    """prompt id -> completion_token_ids."""
    out: dict[str, list[int]] = {}
    with decode_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["id"])] = list(row["completion_token_ids"])
    return out


def compare_cell(ref: dict[str, list[int]], cell: dict[str, list[int]]) -> dict[str, Any]:
    common = sorted(set(ref) & set(cell))
    tot_tokens = 0
    div_tokens = 0
    onsets: list[int] = []          # first differing position (only for broken prompts)
    broken = 0
    per_prompt: list[dict[str, Any]] = []
    for pid in common:
        a, b = ref[pid], cell[pid]
        n = min(len(a), len(b))
        tot_tokens += n
        first = None
        ndiff = 0
        for i in range(n):
            if a[i] != b[i]:
                ndiff += 1
                if first is None:
                    first = i
        # length mismatch counts as divergence too
        ndiff += abs(len(a) - len(b))
        div_tokens += ndiff
        if first is not None or len(a) != len(b):
            broken += 1
            onsets.append(first if first is not None else n)
        per_prompt.append({"id": pid, "len_ref": len(a), "len_cell": len(b),
                           "onset": first, "n_div": ndiff})
    return {
        "n_prompts": len(common),
        "n_broken": broken,
        "prompt_break_rate": broken / len(common) if common else None,
        "total_tokens": tot_tokens,
        "divergent_tokens": div_tokens,
        "token_divergence_rate": div_tokens / tot_tokens if tot_tokens else None,
        "onset_median": statistics.median(onsets) if onsets else None,
        "onset_mean": statistics.fmean(onsets) if onsets else None,
        "onset_min": min(onsets) if onsets else None,
        "onset_max": max(onsets) if onsets else None,
        "per_prompt": per_prompt,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="dir with decode_ar_r0.jsonl + decode_<cell>_r0.jsonl")
    ap.add_argument("--ref", default="ar", help="reference cell name (default ar)")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="write per-cell JSON summary here (default <run-dir>/token_divergence.json)")
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    ref_jsonl = run_dir / f"decode_{args.ref}_r{args.repeat}.jsonl"
    if not ref_jsonl.exists():
        raise SystemExit(f"reference decode not found: {ref_jsonl}")
    ref = load_completions(ref_jsonl)

    results: dict[str, Any] = {}
    cell_jsonls = sorted(p for p in run_dir.glob(f"decode_*_r{args.repeat}.jsonl")
                         if p != ref_jsonl)
    print(f"[tokdiv] reference={ref_jsonl.name} ({len(ref)} prompts)")
    print(f"[tokdiv] {'cell':14s} {'prompt_break':>12s} {'tok_div_rate':>12s} "
          f"{'onset_med':>10s} {'onset_mean':>10s}")
    for cj in cell_jsonls:
        # decode_ngram6_r0.jsonl -> ngram6
        cell = cj.name[len("decode_"):-len(f"_r{args.repeat}.jsonl")]
        cmp = compare_cell(ref, load_completions(cj))
        results[cell] = cmp
        onset_mean_s = "None" if cmp["onset_mean"] is None else f"{cmp['onset_mean']:.1f}"
        print(f"[tokdiv] {cell:14s} "
              f"{cmp['prompt_break_rate']:>11.4f} "
              f"{cmp['token_divergence_rate']:>12.5f} "
              f"{str(cmp['onset_median']):>10s} "
              f"{onset_mean_s:>10s}")

    out = args.out or (run_dir / "token_divergence.json")
    # strip per_prompt from the on-disk summary's top echo to keep it readable but keep full detail
    out.write_text(json.dumps(results, indent=2))
    print(f"[tokdiv] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
