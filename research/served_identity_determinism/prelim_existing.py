#!/usr/bin/env python
"""PR #596 — preliminary base_fullhead spec-OFF cross-process determinism from the EXISTING
run0/run1 captures (no GPU, no boot). Gives the advisor a first bit-stability number immediately.

Existing complete captures (base_fullhead, SENPAI_REFERENCE_MODE=1 M=1 AR greedy, 128x256):
  decode_run0_cold, decode_run0_warm, decode_run1_cold  (run1_warm was interrupted)

Measures:
  * cross-process COLD (run0_cold vs run1_cold): two genuinely fresh server boots, cold pass.
  * within-process cold->warm (run0_cold vs run0_warm): continuity with PR #576 chaos floor.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS_DIR = HERE.parents[1] / "research" / "specdec_identity_census"
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(CENSUS_DIR))

import census_driver as C  # noqa: E402
from scripts.local_validation import paths  # noqa: E402


def f(tag: str) -> str:
    return str(HERE / f"decode_{tag}.jsonl")


def main() -> int:
    pairs = {
        "xproc_cold_run0_vs_run1": (f("run0_cold"), f("run1_cold")),
        "within_cold_warm_run0": (f("run0_cold"), f("run0_warm")),
    }
    out = {}
    for label, (a, b) in pairs.items():
        if not (Path(a).exists() and Path(b).exists()):
            out[label] = {"error": "missing file"}
            continue
        v = C.verify_pair(paths, a, b)
        out[label] = {k: v[k] for k in (
            "verdict", "num_prompts_compared", "num_identical", "num_divergent",
            "sequence_exact_rate", "matched_state_per_step_identity_rate",
            "matched_state_per_step_hazard", "matched_state_trials", "matched_state_failures",
            "freerun_positional_identity_rate", "onset_min", "onset_median", "onset_max",
            "onset_count", "per_step_to_sequence_cascade_factor")}
    print(json.dumps(out, indent=2, default=str))
    (HERE / "prelim_existing.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
