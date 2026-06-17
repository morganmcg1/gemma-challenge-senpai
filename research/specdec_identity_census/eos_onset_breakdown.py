#!/usr/bin/env python
"""PR #576 denken — EOS-aware onset breakdown (offline, CPU-only, no server).

The #319 audit forces ignore_eos + max_tokens=512, so every captured sequence free-runs
PAST the model's natural stop. This script answers the load-bearing objection to the
chaos-floor finding: are the no-spec self-det / spec-dec divergences in the REAL response
(before the model would have emitted EOS) or only in the forced FREE-RUN tail (after)?

For each prompt it finds the natural stop = first index in the no-spec ref_a completion
whose token id is an EOS id ({1 <eos>, 106 <end_of_turn>, 50} from generation_config), then
classifies every first-divergence onset as:
  * real-response  : onset <  natural_eos_pos  (a genuine output divergence)
  * free-run       : onset >= natural_eos_pos  (only in the ignore_eos audit tail)

Reuses the OFFICIAL greedy_identity verifier for the per-prompt onsets so the breakdown is
keyed to the exact #319 comparison. Run AFTER the census captures are final:
  python research/specdec_identity_census/eos_onset_breakdown.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "official" / "main_bucket" / "shared_resources"
                       / "gemma_greedy_identity_verifier_flowian-powers"))
import greedy_identity as gi  # noqa: E402

# gemma-4-E4B-it generation_config.json eos_token_id (the natural-stop tokens the model
# still EMITS under ignore_eos=True but would normally have stopped on).
EOS_IDS = {1, 106, 50}


def natural_eos_pos(ids: list[int]) -> int:
    """First index whose token is an EOS id = where greedy decode would naturally stop.
    len(ids) (=512) when the model never emits EOS within the forced window (no free-run)."""
    for i, t in enumerate(ids):
        if t in EOS_IDS:
            return i
    return len(ids)


def breakdown(ref_recs: dict, cand_path: str, label: str) -> dict:
    cand = gi.load_decode_outputs(cand_path)
    rep = gi.compare(ref_recs, cand)  # per_prompt populated over common keys even if INCOMPARABLE
    real, free = [], []
    deep_into_freerun = []  # onset - natural_eos for free-run onsets
    eos_positions = []
    has_freerun = 0
    for p in rep.per_prompt:
        ref_ids = ref_recs[p.key]["completion_token_ids"]
        eos = natural_eos_pos(ref_ids)
        eos_positions.append(eos)
        if eos < len(ref_ids):
            has_freerun += 1
        if p.identical or p.first_divergence_index is None:
            continue
        onset = p.first_divergence_index
        if onset < eos:
            real.append(onset)
        else:
            free.append(onset)
            deep_into_freerun.append(onset - eos)
    n_div = len(real) + len(free)
    return {
        "label": label,
        "verdict": rep.verdict,
        "n_compared": len(rep.per_prompt),
        "n_divergent": n_div,
        "n_onset_real_response": len(real),
        "n_onset_free_run": len(free),
        "frac_divergences_real_response": (len(real) / n_div) if n_div else None,
        "frac_divergences_free_run": (len(free) / n_div) if n_div else None,
        "real_onset_median": statistics.median(real) if real else None,
        "free_onset_median": statistics.median(free) if free else None,
        "median_tokens_into_freerun_at_onset": statistics.median(deep_into_freerun) if deep_into_freerun else None,
        "natural_eos_pos_median": statistics.median(eos_positions) if eos_positions else None,
        "natural_eos_pos_min": min(eos_positions) if eos_positions else None,
        "n_prompts_with_freerun": has_freerun,
        "n_prompts_no_eos_in_512": sum(1 for e in eos_positions if e >= 512),
    }


def main() -> int:
    ref_a = HERE / "decode_ref_a.jsonl"
    if not ref_a.exists():
        print(f"[eos] {ref_a} missing — run the census first", flush=True)
        return 1
    ref_recs = gi.load_decode_outputs(str(ref_a))

    rows = []
    for tag, label in [("mtp", "REF(no-spec) vs MTP K=7"),
                       ("ngram", "REF(no-spec) vs ngram"),
                       ("ref_b", "self-det ref_a vs ref_b (chaos floor)"),
                       ("ref_c", "self-det ref_a vs ref_c")]:
        cand = HERE / f"decode_{tag}.jsonl"
        if not cand.exists():
            print(f"[eos] skip {label}: {cand.name} missing", flush=True)
            continue
        row = breakdown(ref_recs, str(cand), label)
        rows.append(row)
        print(f"\n=== {label} ===  [{row['verdict']}]  n={row['n_compared']}", flush=True)
        print(f"  divergent prompts                = {row['n_divergent']}", flush=True)
        print(f"  onset REAL-RESPONSE / FREE-RUN   = {row['n_onset_real_response']} / {row['n_onset_free_run']}  "
              f"(real frac {row['frac_divergences_real_response']})", flush=True)
        print(f"  real / free onset medians        = {row['real_onset_median']} / {row['free_onset_median']}", flush=True)
        print(f"  median tokens into free-run      = {row['median_tokens_into_freerun_at_onset']}", flush=True)
        print(f"  natural EOS pos median/min       = {row['natural_eos_pos_median']} / {row['natural_eos_pos_min']}  "
              f"(prompts with free-run {row['n_prompts_with_freerun']}, no-EOS-in-512 {row['n_prompts_no_eos_in_512']})",
              flush=True)

    out = HERE / "eos_onset_breakdown.json"
    out.write_text(json.dumps({"eos_ids": sorted(EOS_IDS), "rows": rows}, indent=2, default=str))
    print(f"\n[eos] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
