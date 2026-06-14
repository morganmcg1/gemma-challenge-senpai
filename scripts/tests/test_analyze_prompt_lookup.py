#!/usr/bin/env python3
"""CPU-only correctness tests for scripts/analyze_prompt_lookup.py (PR #81).

The Step-1 prompt-lookup gate rests entirely on the realized free-accept length q
being computed correctly, so these tests pin the three subtle points:

* the SOURCE-AVAILABILITY cap -- PLD may only copy already-generated tokens
  (indices < ap), so a recent match cannot "accept" not-yet-generated tokens;
* EARLIEST (vLLM-faithful) vs ORACLE-best occurrence selection;
* the PROMPT-region vs GENERATED-region hit classification.

Run: python -m pytest scripts/tests/test_analyze_prompt_lookup.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
import analyze_prompt_lookup as apl  # noqa: E402


def test_source_availability_cap_period1_run():
    """A period-1 run: match is at e = ap-1, so only ONE token may be copied.

    prompt=[100,101], generated=[5,5,5,5,5]. At the step that emits generated idx 3
    (ap=5) the n=2 suffix is [5,5], whose most-recent earlier occurrence ends at
    e=4 (ap-e == 1). PLD can copy exactly seq[4]=5 (q=1). Without the ap-e cap it
    would wrongly read seq[5] -- the token being emitted -- and report q=2.
    """
    res = apl.pld_per_position([100, 101], [5, 5, 5, 5, 5], n=2, max_draft=7)
    t = 3  # generated index 3, ap = P + 3 = 5
    assert res["q_earliest"][t] == 1, res["q_earliest"]
    assert res["q_oracle"][t] == 1, res["q_oracle"]


def test_earliest_vs_oracle_pick():
    """Two earlier [1,2] occurrences with different continuations; earliest diverges
    immediately (q=0) but a later occurrence accepts one token (oracle q=1)."""
    gen = [1, 2, 9, 1, 2, 8, 1, 2, 8, 5]
    res = apl.pld_per_position([], gen, n=2, max_draft=7)
    t = 8  # ap = 8, suffix seq[6:8] = [1,2]; earlier ends at e=2 (->9) and e=5 (->8)
    assert res["region_earliest"][t] >= 0, "should be a hit"
    assert res["q_earliest"][t] == 0, res["q_earliest"][t]  # earliest [1,2] -> 9 != 8
    assert res["q_oracle"][t] == 1, res["q_oracle"][t]      # later [1,2] -> 8 == 8


def test_prompt_vs_generated_region():
    """A suffix whose only earlier occurrence is inside the prompt is tagged region 0."""
    res = apl.pld_per_position([1, 2, 3], [3, 1, 2, 4], n=2, max_draft=7)
    t = 3  # ap = 6, suffix seq[4:6] = [1,2]; only earlier occ ends at e=2 (in prompt)
    assert res["region_earliest"][t] == 0, res["region_earliest"][t]
    assert res["has_prompt"][t] == 1
    assert res["has_gen"][t] == 0


def test_no_pld_means_no_augment():
    """With q == 0 everywhere the augment must add nothing (extra == 0, E[T] equal)."""
    cond_p = [0.729, 0.759, 0.792, 0.822, 0.834, 0.835, 0.847]
    q_zero = [[0] * 64 for _ in range(4)]
    sim = apl.simulate(q_zero, cond_p, trials=50, seed=0)
    assert sim["extra_tokens_per_step"] == 0.0
    assert abs(sim["ET_augment_sim"] - sim["ET_mtp_only_sim"]) < 1e-9


def test_corr_fullspan_floor_is_zero_at_aH_one():
    """Full-span correlation at a_H=1.0: MTP accepts every predictable span fully, so
    the augment headroom collapses to ~0 extra tokens/step."""
    cond_p = [0.729, 0.759, 0.792, 0.822, 0.834, 0.835, 0.847]
    q = [[3, 0, 2, 0] * 16 for _ in range(4)]  # some hits
    frac_hit = sum(sum(1 for v in row if v > 0) for row in q) / sum(len(r) for r in q)
    out = apl.simulate_corr(q, cond_p, frac_hit, a_H=1.0, trials=60, seed=3,
                            baseline_et=3.844, full_span=True)
    assert out["extra_tokens_per_step"] < 0.02, out["extra_tokens_per_step"]
