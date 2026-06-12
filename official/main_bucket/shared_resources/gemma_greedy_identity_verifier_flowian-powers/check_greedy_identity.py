#!/usr/bin/env python3
"""CLI wrapper for the Gemma greedy-identity verifier.

Thin command-line front-end over :mod:`greedy_identity`. It loads and compares
a CANDIDATE ``decode_outputs.jsonl`` against an EXACT-GREEDY REFERENCE and
reports a verdict, in either human-readable or JSON form.

Exit codes:
  0  GREEDY_IDENTICAL (valid)
  1  DIVERGENT (invalid)
  2  INCOMPARABLE, or any error (missing file, malformed/empty input, bad args)

Standard library only. All comparison logic lives in greedy_identity; this
module performs no comparison of its own.
"""

from __future__ import annotations

import argparse
import json
import sys

import greedy_identity


# Map each verdict to its process exit code.
_VERDICT_EXIT_CODES = {
    "GREEDY_IDENTICAL": 0,
    "DIVERGENT": 1,
    "INCOMPARABLE": 2,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_greedy_identity",
        description=(
            "Verify that a candidate decode_outputs.jsonl is token-identical "
            "to an exact-greedy reference."
        ),
    )
    parser.add_argument(
        "--reference",
        required=True,
        metavar="PATH",
        help="Path to the exact-greedy reference decode_outputs.jsonl.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        metavar="PATH",
        help="Path to the candidate decode_outputs.jsonl under test.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        metavar="N",
        help="Max number of divergent prompts to list in human output; "
        "a negative value shows all (default: 5).",
    )
    return parser


def _print_human(report, max_examples: int) -> None:
    """Print a human-readable summary of the comparison report to stdout."""
    banner_suffix = {
        "GREEDY_IDENTICAL": " (valid)",
        "DIVERGENT": " (invalid)",
        "INCOMPARABLE": "",
    }.get(report.verdict, "")
    print(f"VERDICT: {report.verdict}{banner_suffix}")

    # Left-justify labels to a fixed width so all values line up in a column.
    print(f"  {'prompts compared:':<24}{report.num_prompts_compared}")
    print(f"  {'identical:':<24}{report.num_identical}")
    print(f"  {'divergent:':<24}{report.num_divergent}")
    print(f"  {'total tokens compared:':<24}{report.total_tokens_compared}")
    print(f"  {'total divergent tokens:':<24}{report.total_divergent_tokens}")

    if report.verdict == "DIVERGENT":
        divergent = [pc for pc in report.per_prompt if not pc.identical]
        shown = divergent[:max_examples] if max_examples >= 0 else divergent
        print(f"  divergent prompts (showing {len(shown)} of "
              f"{len(divergent)}):")
        for pc in shown:
            print(f"    - {pc.key}: first divergence at index "
                  f"{pc.first_divergence_index}")
    elif report.verdict == "INCOMPARABLE":
        print("  reason:")
        if report.missing_in_candidate:
            print(f"    missing in candidate: "
                  f"{', '.join(report.missing_in_candidate)}")
        if report.missing_in_reference:
            print(f"    missing in reference: "
                  f"{', '.join(report.missing_in_reference)}")
        if report.integrity_failures:
            print(f"    stored-sha integrity failures: "
                  f"{', '.join(report.integrity_failures)}")


def main(argv=None) -> int:
    """Parse args, run the comparison, emit output, and return an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        report = greedy_identity.compare_files(args.reference, args.candidate)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_human(report, args.max_examples)

    return _VERDICT_EXIT_CODES.get(report.verdict, 2)


if __name__ == "__main__":
    raise SystemExit(main())
