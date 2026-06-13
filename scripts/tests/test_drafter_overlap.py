#!/usr/bin/env python3
"""CPU-only validation for the SAM-Decoding drafter-overlap analysis (PR #13).

These tests exercise ``scripts/analyze_suffix_budget.py``'s ``--drafter-trace``
intersection logic with synthetic mock traces, so the moment kanna's MTP drafter
(#5) emits a real acceptance trace the net-headroom number is trustworthy.

The four advisor-required mocks (run end-to-end against the real cached
``decode_outputs_128.jsonl`` capture, no GPU, no model load):

* Mock A -- drafter accepts NOTHING   -> net_sam == base causal budget (~8.93%).
* Mock B -- drafter accepts EVERYTHING -> net_sam == 0 (overlap saturates).
* Mock C -- drafter accepts a random 50% of positions -> net_sam ~= 0.5 * base.
* Mock D -- drafter accepts exactly the SAM-run positions -> net_sam == 0.

Run: ``python -m pytest scripts/tests/test_drafter_overlap.py -v``
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

# import the analyzer module from scripts/
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
import analyze_suffix_budget as asb  # noqa: E402

MAX_CHECK = asb.DEFAULT_MAX_CHECK
K = asb.PRIMARY_K
# PR #10 (merged) realized causal SAM budget at K>8 on the 128 bench prompts.
PR10_CAUSAL_K8 = 0.0893


# ---------------------------------------------------------------------------
# shared fixtures: load the real decode capture once, precompute per-prompt
# causal SAM run positions (the set S the drafter overlap intersects against).
# ---------------------------------------------------------------------------
def _completion(rec: dict) -> list[int]:
    return rec.get("completion_token_ids") or rec.get("token_ids")


@pytest.fixture(scope="module")
def records() -> list[dict]:
    recs = asb.load_records(asb.DEFAULT_INPUT)
    assert recs, "decode capture is empty"
    return recs


@pytest.fixture(scope="module")
def sam_positions(records) -> list[set[int]]:
    """Per-prompt causal SAM run positions at K>8 (|S_p| == realized free tokens)."""
    out = []
    for rec in records:
        gen = _completion(rec)
        r_values = asb.realized_sam_lengths(rec["prompt_token_ids"], gen, MAX_CHECK)
        out.append(asb.greedy_run_positions(r_values, K))
    return out


@pytest.fixture(scope="module")
def total_tokens(records) -> int:
    return sum(len(_completion(r)) for r in records)


@pytest.fixture(scope="module")
def base_causal(sam_positions, total_tokens) -> float:
    """Base causal SAM K>8 budget computed straight from the run-position sets."""
    return sum(len(s) for s in sam_positions) / total_tokens


def _segments_from_positions(positions: set[int], generated_ids: list[int]) -> list[dict]:
    """Turn a set of accepted output positions into canonical drafter segments,
    coalescing consecutive positions into one segment (with output_start + the
    real token ids, so the analyzer's token-id consistency check sees no mismatch).
    """
    segs: list[dict] = []
    end = -1
    for p in sorted(positions):
        if segs and p == end:
            segs[-1]["accepted_token_ids"].append(generated_ids[p])
        else:
            segs.append({"output_start": p, "accepted_token_ids": [generated_ids[p]]})
        end = p + 1
    for step, s in enumerate(segs):
        s["step"] = step
        s["prompt_idx"] = None  # filled by caller when needed
        s["acceptance_len"] = len(s["accepted_token_ids"])
    return segs


def _run(records, drafter_by_prompt):
    return asb.compute_analysis(records, MAX_CHECK, drafter_by_prompt=drafter_by_prompt)


# ---------------------------------------------------------------------------
# unit tests for the intersection primitives
# ---------------------------------------------------------------------------
def test_greedy_run_positions_matches_free_count():
    # values where a run of length 5 starts at t=1 (> K when K small)
    values = [0, 5, 0, 0, 0, 0, 3, 0, 0]
    for k in (1, 2, 4):
        pos = asb.greedy_run_positions(values, k)
        free, _ = asb.greedy_free_tokens_and_runs(values, k)
        assert len(pos) == free


def test_drafter_positions_contiguous():
    gen = [10, 11, 12, 13]
    pos, mm = asb.drafter_accepted_positions(
        [{"accepted_token_ids": [10, 11]}, {"accepted_token_ids": [12]}], gen
    )
    assert pos == {0, 1, 2}
    assert mm == 0


def test_drafter_positions_output_start_gaps():
    gen = [10, 11, 12, 13]
    pos, mm = asb.drafter_accepted_positions(
        [
            {"output_start": 0, "accepted_token_ids": [10]},
            {"output_start": 2, "accepted_token_ids": [12]},
        ],
        gen,
    )
    assert pos == {0, 2}  # position 1 (a target/bonus token) is NOT drafter-covered
    assert mm == 0


def test_drafter_positions_token_mismatch_counted():
    gen = [10, 11]
    pos, mm = asb.drafter_accepted_positions(
        [{"output_start": 0, "accepted_token_ids": [99]}], gen
    )
    assert pos == {0}
    assert mm == 1


def test_drafter_positions_acceptance_len_only():
    # a trace that logs only the count, not the ids, still maps positions
    gen = [10, 11, 12]
    pos, mm = asb.drafter_accepted_positions(
        [{"output_start": 0, "acceptance_len": 2}], gen
    )
    assert pos == {0, 1}
    assert mm == 0


def test_drafter_positions_clamped_to_output():
    gen = [10, 11]
    pos, mm = asb.drafter_accepted_positions(
        [{"output_start": 1, "accepted_token_ids": [11, 12, 13]}], gen
    )
    assert pos == {1}  # positions 2,3 are past end-of-output, dropped
    assert mm == 0


# ---------------------------------------------------------------------------
# consistency: the drafter block's base must equal the merged-PR #10 budget
# ---------------------------------------------------------------------------
def test_base_causal_matches_pr10(records, base_causal):
    assert base_causal == pytest.approx(PR10_CAUSAL_K8, abs=5e-4)
    # and the empty-trace run must report the same sam_causal as the top-level causal value
    res = _run(records, {})
    assert res["causal_value_gt_K8"] == pytest.approx(base_causal, abs=1e-12)
    assert res["drafter_overlap"]["sam_causal_frac_gt_k8"] == pytest.approx(base_causal, abs=1e-12)


# ---------------------------------------------------------------------------
# Mock A -- 0% acceptance -> net == base causal budget
# ---------------------------------------------------------------------------
def test_mock_a_zero_acceptance(records, base_causal):
    res = _run(records, {})  # no drafter segments for any prompt
    do = res["drafter_overlap"]
    assert do["available"] is True
    assert do["drafter_accepted_frac"] == 0.0
    assert do["overlap_frac"] == 0.0
    assert do["token_id_mismatches"] == 0
    assert do["net_sam_beyond_drafter_frac"] == pytest.approx(base_causal, abs=1e-12)
    assert do["verdict"]["go"] is True  # 8.93% > 3% threshold
    assert do["verdict"]["band"] == "go"


# ---------------------------------------------------------------------------
# Mock B -- 100% acceptance of everything -> net == 0
# ---------------------------------------------------------------------------
def test_mock_b_full_acceptance(records, base_causal):
    trace = {}
    for i, rec in enumerate(records):
        gen = _completion(rec)
        trace[i] = _segments_from_positions(set(range(len(gen))), gen)
    do = _run(records, trace)["drafter_overlap"]
    assert do["drafter_accepted_frac"] == pytest.approx(1.0, abs=1e-12)
    assert do["overlap_frac"] == pytest.approx(base_causal, abs=1e-12)
    assert do["net_sam_beyond_drafter_frac"] == pytest.approx(0.0, abs=1e-12)
    assert do["token_id_mismatches"] == 0
    assert do["verdict"]["go"] is False
    assert do["verdict"]["band"] == "retire"


# ---------------------------------------------------------------------------
# Mock C -- random 50% acceptance -> net ~= 0.5 * base (proportionality)
# ---------------------------------------------------------------------------
def test_mock_c_half_acceptance(records, base_causal):
    rng = random.Random(20260613)
    trace = {}
    for i, rec in enumerate(records):
        gen = _completion(rec)
        marked = {p for p in range(len(gen)) if rng.random() < 0.5}
        trace[i] = _segments_from_positions(marked, gen)
    do = _run(records, trace)["drafter_overlap"]
    # independent Bernoulli(0.5) marking => E[overlap] = 0.5*|S|, very low variance
    assert do["drafter_accepted_frac"] == pytest.approx(0.5, abs=0.02)
    assert do["net_sam_beyond_drafter_frac"] == pytest.approx(0.5 * base_causal, abs=0.005)
    assert do["overlap_frac"] == pytest.approx(0.5 * base_causal, abs=0.005)
    assert do["token_id_mismatches"] == 0


# ---------------------------------------------------------------------------
# Mock D -- drafter accepts exactly the SAM-run positions -> net == 0 (worst case)
# ---------------------------------------------------------------------------
def test_mock_d_perfect_sam_overlap(records, sam_positions, base_causal):
    trace = {}
    for i, rec in enumerate(records):
        gen = _completion(rec)
        trace[i] = _segments_from_positions(sam_positions[i], gen)
    do = _run(records, trace)["drafter_overlap"]
    assert do["overlap_frac"] == pytest.approx(base_causal, abs=1e-12)
    assert do["net_sam_beyond_drafter_frac"] == pytest.approx(0.0, abs=1e-12)
    # drafter covered exactly S, so drafter_accepted_frac == base causal budget
    assert do["drafter_accepted_frac"] == pytest.approx(base_causal, abs=1e-12)
    assert do["token_id_mismatches"] == 0
    assert do["verdict"]["go"] is False
    assert do["verdict"]["band"] == "retire"


# ---------------------------------------------------------------------------
# load_drafter_trace round-trip through a real JSONL file (exercises the loader
# + step-ordering), reproducing Mock D's net == 0.
# ---------------------------------------------------------------------------
def test_load_drafter_trace_roundtrip(records, sam_positions, base_causal, tmp_path):
    trace_path = tmp_path / "drafter_trace.jsonl"
    with trace_path.open("w") as fh:
        for i, rec in enumerate(records):
            gen = _completion(rec)
            for seg in _segments_from_positions(sam_positions[i], gen):
                fh.write(json.dumps({
                    "prompt_idx": i,
                    "step": seg["step"],
                    "accepted_token_ids": seg["accepted_token_ids"],
                    "acceptance_len": seg["acceptance_len"],
                    "output_start": seg["output_start"],
                }) + "\n")
    by_prompt = asb.load_drafter_trace(trace_path)
    do = _run(records, by_prompt)["drafter_overlap"]
    assert do["net_sam_beyond_drafter_frac"] == pytest.approx(0.0, abs=1e-12)
    assert do["overlap_frac"] == pytest.approx(base_causal, abs=1e-12)


def test_per_dataset_breakdown_present(records):
    res = _run(records, {})
    pd = res["drafter_overlap"]["per_dataset"]
    # the 128-prompt capture spans three distributions
    assert set(pd) == {"aime2026", "gpqa_diamond", "mmlu_pro"}
    for d in pd.values():
        assert {"total_tokens", "drafter_accepted_frac", "sam_causal_frac_gt_k8",
                "overlap_frac", "net_sam_beyond_drafter_frac"} <= set(d)
