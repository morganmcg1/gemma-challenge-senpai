#!/usr/bin/env python
"""PR #600 — OFFLINE preview of the base_mtp spec-ON vs spec-OFF greedy-identity census,
reusing the ALREADY-CAPTURED #596 warm decode files (128x256). No GPU, no server boot.

This is a directional read only: the canonical wirbel #588 predicate is 128x512, so the
headline number comes from the fresh 512 run. But both arms here are the SAME int4 W4A16
body + full BF16 262k head checkpoint, differing ONLY in spec mode:
  spec-OFF (reference) = decode_run{0,1}_warm.jsonl   (SENPAI_REFERENCE_MODE -> M=1 AR greedy)
  spec-ON  (candidate) = decode_base_mtp_run{0,1}_warm.jsonl  (MTP-K7 -> M=8 verify)

verify_pair(spec-OFF, spec-ON) answers: does the int4 BODY alone (BF16 head held constant)
break the spec-vs-AR greedy identity?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CENSUS_DIR = ROOT / "research" / "specdec_identity_census"
for p in (str(ROOT), str(CENSUS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import census_driver as C  # noqa: E402
from scripts.local_validation import paths  # noqa: E402

SID = ROOT / "research" / "served_identity_determinism"
PAIRS = {
    "run0_warm": (SID / "decode_run0_warm.jsonl", SID / "decode_base_mtp_run0_warm.jsonl"),
    "run1_warm": (SID / "decode_run1_warm.jsonl", SID / "decode_base_mtp_run1_warm.jsonl"),
}


def main() -> int:
    out = {}
    for name, (ref, cand) in PAIRS.items():
        if not ref.exists() or not cand.exists():
            print(f"[preview] {name}: MISSING ({ref.exists()=}, {cand.exists()=})", flush=True)
            continue
        v = C.verify_pair(paths, str(ref), str(cand))
        keep = {k: v[k] for k in (
            "verdict", "num_prompts_compared", "num_identical", "num_divergent",
            "sequence_exact_rate", "matched_state_per_step_identity_rate",
            "matched_state_trials", "matched_state_failures",
            "freerun_positional_identity_rate", "total_tokens_compared",
            "total_divergent_tokens", "onset_min", "onset_median", "onset_max", "onset_count")}
        out[name] = keep
        print(f"\n===== {name}: spec-OFF (M=1 AR) vs spec-ON (MTP-K7, M=8) =====", flush=True)
        print(f"  verdict                 = {keep['verdict']}", flush=True)
        print(f"  seq byte-exact rate     = {keep['sequence_exact_rate']} "
              f"({keep['num_identical']}/{keep['num_prompts_compared']})", flush=True)
        print(f"  matched-state per-step  = {keep['matched_state_per_step_identity_rate']} "
              f"(fail {keep['matched_state_failures']}/{keep['matched_state_trials']})", flush=True)
        print(f"  free-run positional     = {keep['freerun_positional_identity_rate']} "
              f"(div {keep['total_divergent_tokens']}/{keep['total_tokens_compared']})", flush=True)
        print(f"  divergent prompts       = {keep['num_divergent']}", flush=True)
        print(f"  onset min/median/max    = {keep['onset_min']}/{keep['onset_median']}/{keep['onset_max']}",
              flush=True)
    (Path(__file__).resolve().parent / "preview_256.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n[preview] wrote preview_256.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
