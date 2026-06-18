#!/usr/bin/env python3
"""PR #631 supplement: completion-length (byte-proxy) determinism per arm on dev307.

The main sweep (run_sweep.py) measures determinism at the *parsed-answer* level
(`n_answer_unstable_union`). That under-counts engine nondeterminism: two reps
can produce byte-different completions that still parse to the same letter (or
both fail to parse -> both `answer=''` -> both "stable"). `completion_chars` is
a strictly finer signal: if it differs across reps for an item, the generated
text provably diverged. This script reports, per arm:

  - completion_chars pairwise-flip counts and union-unstable across the 3 reps
  - answer-flip union (echo of the main analysis, for side-by-side)
  - the loop-to-cap population (long completions) that dominates 0.22.0's surface

It reads the same conc{C}_rep{R}.json result files and writes
`completion_stability_summary.json`. Read-only w.r.t. the main artifact.
"""
from __future__ import annotations

import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CONCS = [1, 16]
REPEATS = 3
LOOP_CHAR_THRESHOLD = 12000  # ~loop-to-cap proxy; 0.22.0 loops ran ~25-32k chars (#618).
# On healthy dev307 (#615) loops should be rare -> n_long ~ 0. finish_length_rate
# (recorded natively by run_eval.py now) is the authoritative crater detector; this
# char-proxy is a coarser byte-level cross-check kept for #618 side-by-side.


def load(conc: int, rep: int) -> dict | None:
    p = RES / f"conc{conc}_rep{rep}.json"
    return json.loads(p.read_text()) if p.exists() and p.stat().st_size > 0 else None


def _common_ids(maps: list[dict]) -> set[str]:
    common = set(maps[0])
    for m in maps[1:]:
        common &= set(m)
    return common


def _pairwise(maps: list[dict], common: set[str]) -> dict[str, int]:
    return {
        f"{a}v{b}": sum(1 for i in common if maps[a][i] != maps[b][i])
        for a, b in itertools.combinations(range(len(maps)), 2)
    }


def _union_unstable(maps: list[dict], common: set[str]) -> list[str]:
    return sorted(i for i in common if len({m[i] for m in maps}) > 1)


def main() -> int:
    out: dict = {"pr": 631, "engine": "vllm-0.22.1rc1.dev307",
                 "metric": "completion_chars (byte-proxy) + answer", "arms": {}}
    for conc in CONCS:
        reps = [d for r in range(REPEATS) if (d := load(conc, r)) is not None]
        if len(reps) < 2:
            print(f"[supp] conc={conc}: only {len(reps)} reps present, skipping")
            continue
        cc_maps = [{s["id"]: s.get("completion_chars") for s in d["per_sample"]} for d in reps]
        an_maps = [{s["id"]: s.get("answer") for s in d["per_sample"]} for d in reps]
        common = _common_ids(cc_maps)
        cc_pw = _pairwise(cc_maps, common)
        an_pw = _pairwise(an_maps, common)
        cc_un = _union_unstable(cc_maps, common)
        an_un = _union_unstable(an_maps, common)
        # loop-to-cap population per rep (median/max chars; long-completion count)
        per_rep = []
        for r, d in enumerate(reps):
            ccs = [s["completion_chars"] for s in d["per_sample"]]
            per_rep.append({
                "rep": r, "accuracy": d["accuracy"],
                "median_chars": st.median(ccs), "max_chars": max(ccs),
                "n_long": sum(1 for c in ccs if c >= LOOP_CHAR_THRESHOLD),
            })
        n = len(common)
        out["arms"][f"conc{conc}"] = {
            "n_repeats": len(reps), "n_common": n,
            "completion_chars_flips_pairwise": cc_pw,
            "completion_chars_unstable_union": len(cc_un),
            "completion_chars_unstable_frac": round(len(cc_un) / n, 4) if n else None,
            "answer_flips_pairwise": an_pw,
            "answer_unstable_union": len(an_un),
            "answer_unstable_frac": round(len(an_un) / n, 4) if n else None,
            "byte_stable_but_answer_stable_gap": len(cc_un) - len(an_un),
            "completion_deterministic": len(cc_un) == 0,
            "per_rep": per_rep,
            "loop_char_threshold": LOOP_CHAR_THRESHOLD,
        }
        print(f"[supp] conc={conc}: cc_unstable={len(cc_un)}/{n} "
              f"(pw={cc_pw}) | answer_unstable={len(an_un)}/{n} "
              f"| completion_deterministic={len(cc_un) == 0}")
    (HERE / "completion_stability_summary.json").write_text(json.dumps(out, indent=2))
    print("[supp] wrote completion_stability_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
