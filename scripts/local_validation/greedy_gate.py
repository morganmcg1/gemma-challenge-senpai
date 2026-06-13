"""Greedy token-identity gate — thin CLI over the official verifier.

Wires up ``check_greedy_identity.py`` (the flowian-powers shared resource) so a
candidate ``decode_outputs.jsonl`` can be compared against an exact-greedy AR
reference. All comparison logic lives in the official ``greedy_identity``
module; this only resolves the reference and surfaces the verdict.

Verdict / exit codes (inherited from the official rule):
  0  GREEDY_IDENTICAL  (valid)
  1  DIVERGENT         (invalid — a serving optimization changed token IDs)
  2  INCOMPARABLE      (prompt sets differ / integrity failure / error)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from . import paths

VERDICT_EXIT = {"GREEDY_IDENTICAL": 0, "DIVERGENT": 1, "INCOMPARABLE": 2}


def compare(reference: Path, candidate: Path) -> Any:
    """Return the official ComparisonReport for two decode_outputs.jsonl files."""
    gi = paths.import_greedy_identity()
    return gi.compare_files(str(reference), str(candidate))


def onset_summary(report: Any) -> dict[str, Any]:
    """Distribution of first-divergence onset over the divergent prompts.

    The official verdict is a single byte-exact yes/no, but *where* divergence
    starts tells you why. A lossy optimization (mis-verifying drafter, lossy
    kernel, wrong quant) diverges early and on most prompts; floating-point
    reduction non-determinism at long output_len diverges late and on a
    stochastic subset (argmax ties flip, then cascade). Surfacing the onset
    distribution lets a reviewer tell those two apart from one DIVERGENT
    verdict, without changing the official rule. ``onset_frac_*`` express the
    onset as a fraction of output_len so the read is length-independent.
    """
    onsets = sorted(
        p.first_divergence_index
        for p in report.per_prompt
        if not p.identical and p.first_divergence_index is not None
    )
    out: dict[str, Any] = {
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "onsets": onsets,
    }
    if onsets:
        out["onset_min"] = onsets[0]
        out["onset_median"] = int(statistics.median(onsets))
        out["onset_max"] = onsets[-1]
    return out


def onset_line(summary: dict[str, Any], output_len: int | None = None) -> str:
    """One-line human rendering of [[onset_summary]] for the evidence block."""
    nd = summary.get("num_divergent", 0)
    if not nd:
        return "divergence onset: none (all prompts identical)"
    lo, med, hi = summary.get("onset_min"), summary.get("onset_median"), summary.get("onset_max")
    frac = f" ({100 * lo / output_len:.0f}–{100 * hi / output_len:.0f}% of {output_len})" if output_len else ""
    return f"divergence onset (tok idx): min={lo} median={med} max={hi}{frac} over {nd} divergent prompt(s)"


def reference_for(model_id: str) -> Path:
    """Canonical reference path for a checkpoint's exact-greedy AR decode.

    This is the *served* spec-off reference (``gen_greedy_reference --mode
    served``), captured through the same api_server path as candidates so the
    gate isolates the optimization-under-test rather than cross-engine FP noise.
    """
    return paths.REFERENCE_ROOT / paths.model_tag(model_id) / "decode_outputs.jsonl"


def reference_kind(reference: Path) -> str:
    """Best-effort read of a reference's ``reference_kind`` from its sibling meta.

    Lets the evidence block state which anchor gated a candidate (``served_spec_off``
    is the trustworthy gate; ``plain_greedy_ar_offline_vllm`` is the diagnostic
    cross-check). Returns ``"unknown"`` if no meta is found.
    """
    reference = Path(reference)
    meta = reference.parent / ("meta.offline.json" if reference.name.endswith(".offline.jsonl") else "meta.json")
    try:
        return str(json.loads(meta.read_text()).get("reference_kind", "unknown"))
    except (OSError, ValueError):
        return "unknown"


def reference_num_records(reference: Path) -> int | None:
    """Best-effort count of how many prompt records a reference holds.

    The gate compares prompt-for-prompt, so a reference must hold at least as many
    records as the candidate decode or the gate reads INCOMPARABLE for the
    unmatched prompts (a confusing verdict if the record count is invisible). The
    count lives in the reference's sibling ``meta.json`` (written by
    gen_greedy_reference), falling back to ``decode_summary.json`` (written by the
    decode capture) — both carry ``num_records``. Returns ``None`` when neither
    file/field is available so callers can degrade quietly.
    """
    reference = Path(reference)
    for sibling in ("meta.json", "decode_summary.json"):
        try:
            data = json.loads((reference.parent / sibling).read_text())
        except (OSError, ValueError):
            continue
        value = data.get("num_records")
        if isinstance(value, int):
            return value
    return None


def _print_human(report: Any) -> None:
    suffix = {"GREEDY_IDENTICAL": " (valid)", "DIVERGENT": " (invalid)", "INCOMPARABLE": ""}
    print(f"VERDICT: {report.verdict}{suffix.get(report.verdict, '')}")
    print(f"  prompts compared:       {report.num_prompts_compared}")
    print(f"  identical:              {report.num_identical}")
    print(f"  divergent:              {report.num_divergent}")
    print(f"  total tokens compared:  {report.total_tokens_compared}")
    print(f"  total divergent tokens: {report.total_divergent_tokens}")
    if report.verdict == "DIVERGENT":
        print(f"  {onset_line(onset_summary(report))}")
        for pc in [p for p in report.per_prompt if not p.identical][:5]:
            print(f"    - {pc.key}: first divergence at index {pc.first_divergence_index}")
    elif report.verdict == "INCOMPARABLE":
        if report.missing_in_candidate:
            print(f"    missing in candidate: {', '.join(report.missing_in_candidate[:5])} ...")
        if report.missing_in_reference:
            print(f"    missing in reference: {', '.join(report.missing_in_reference[:5])} ...")
        if report.integrity_failures:
            print(f"    integrity failures: {', '.join(report.integrity_failures[:5])} ...")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="greedy_gate", description=__doc__)
    ap.add_argument("--candidate", required=True, type=Path, help="candidate decode_outputs.jsonl")
    ref = ap.add_mutually_exclusive_group(required=True)
    ref.add_argument("--reference", type=Path, help="explicit reference decode_outputs.jsonl")
    ref.add_argument("--model-id", help="resolve the canonical reference for this checkpoint")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = ap.parse_args(argv)

    reference = args.reference or reference_for(args.model_id)
    if not reference.exists():
        print(
            f"error: reference not found: {reference}\n"
            f"       generate it first: python -m scripts.local_validation.gen_greedy_reference "
            f"--mode served --model-id {args.model_id or '<id>'}",
            file=sys.stderr,
        )
        return 2
    try:
        report = compare(reference, args.candidate)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_human(report)
    return VERDICT_EXIT.get(report.verdict, 2)


if __name__ == "__main__":
    raise SystemExit(main())
