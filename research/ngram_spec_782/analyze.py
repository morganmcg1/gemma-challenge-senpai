#!/usr/bin/env python3
"""PR #782 — offline analysis of the bi0 ngram-vs-MTP A/B cells.

``run_cell.py`` produces, per cell, a ``decode_outputs.jsonl`` (official greedy
decode, temp=0 seed=1) + ``cell_summary.json`` (wall_tps, spec-decode acceptance
counters, optional PPL). This script does the OFFLINE judging that ``run_cell``
deliberately leaves out, so the verdict logic is auditable and re-runnable
without touching the GPU:

  * greedy token-identity of every cell vs the committed bi0 plain-AR reference R
    (``scripts/local_validation/greedy_gate.compare``) — the official rule, plus
    the onset distribution so a reviewer can tell a lossy break (early, most
    prompts) from int4 near-tie FP residual (late, stochastic subset);
  * greedy token-identity of every ngram cell vs the MTP CONTROL cell C captured
    in the SAME harness — this is the apples-to-apples drafter-swap check the PR
    asks for (control and variant differ only in the proposer), and it factors
    out any control-vs-committed-R drift;
  * a TPS / acceptance(E[T]) / PPL comparison table vs the control.

The greedy-safety CLAIM is that at temp=0 vLLM's verifier emits the int4
target-argmax and rejects on the first draft!=target mismatch, so the accepted
token is independent of which drafter proposed it; therefore C-vs-R and
ngram-vs-C must share the SAME (tie-tolerant) divergence residual. This script
MEASURES that, it does not assume it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_validation import greedy_gate  # noqa: E402


def _gate(reference: Path, candidate: Path, output_len: int) -> dict[str, object]:
    """compare(reference, candidate) -> {verdict, counts, onset_line}."""
    if not reference.exists() or not candidate.exists():
        return {"verdict": "MISSING", "reference_exists": reference.exists(),
                "candidate_exists": candidate.exists()}
    report = greedy_gate.compare(reference, candidate)
    onset = greedy_gate.onset_summary(report)
    return {
        "verdict": report.verdict,
        "num_prompts_compared": report.num_prompts_compared,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
        "onset": onset,
        "onset_line": greedy_gate.onset_line(onset, output_len),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells-root", default=str(REPO / "research/ngram_spec_782/cells"))
    ap.add_argument(
        "--reference",
        default=str(
            REPO
            / "research/greedy_reference"
            / "workspace__senpai__target__submissions__int4_mtp_bi0_surgattn__google__gemma-4-E4B-it-qat-w4a16-ct"
            / "decode_outputs.jsonl"
        ),
        help="committed bi0 plain-AR greedy reference R",
    )
    ap.add_argument("--control-label", default="control_mtp_k6",
                    help="cell label that is the MTP control C (apples-to-apples anchor)")
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--out", default=str(REPO / "research/ngram_spec_782/analysis.json"))
    args = ap.parse_args()

    cells_root = Path(args.cells_root)
    reference = Path(args.reference)
    control_decode = cells_root / args.control_label / "decode_outputs.jsonl"

    cells = []
    for summary_path in sorted(cells_root.glob("*/cell_summary.json")):
        cell_dir = summary_path.parent
        summary = json.loads(summary_path.read_text())
        decode = cell_dir / "decode_outputs.jsonl"
        spec = summary.get("spec_metrics") or {}
        row: dict[str, object] = {
            "label": summary.get("label", cell_dir.name),
            "extra_env": summary.get("extra_env"),
            "wall_tps": summary.get("wall_tps"),
            "decode_num_records": summary.get("decode_num_records"),
            "acceptance_rate": spec.get("acceptance_rate"),
            "mean_tokens_per_step_ET": spec.get("mean_tokens_per_step_ET"),
            "spec_accepted": spec.get("accepted"),
            "spec_draft": spec.get("draft"),
            "spec_drafts": spec.get("drafts"),
            "ppl": summary.get("ppl"),
            "ppl_num_records": summary.get("ppl_num_records"),
            "vs_reference_R": _gate(reference, decode, args.output_len),
        }
        if control_decode.exists() and decode != control_decode:
            row["vs_control_C"] = _gate(control_decode, decode, args.output_len)
        cells.append(row)

    out = {
        "reference": str(reference),
        "control_label": args.control_label,
        "output_len": args.output_len,
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))

    # Human table.
    print(f"\n[analyze] reference R = {reference}")
    print(f"[analyze] control C    = {args.control_label}\n")
    hdr = f"{'label':28s} {'wall_tps':>9s} {'E[T]tok/stp':>11s} {'accept':>7s} {'ppl':>6s} {'vsR':>16s} {'vsC':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        et = c["mean_tokens_per_step_ET"]
        ar = c["acceptance_rate"]
        ppl = c["ppl"]
        vr = c["vs_reference_R"]
        vc = c.get("vs_control_C")
        vr_s = f"{vr['verdict'][:6]}/{vr.get('num_divergent','?')}" if isinstance(vr, dict) else str(vr)
        vc_s = (f"{vc['verdict'][:6]}/{vc.get('num_divergent','?')}"
                if isinstance(vc, dict) else "—")
        print(
            f"{str(c['label'])[:28]:28s} "
            f"{(c['wall_tps'] or 0):9.3f} "
            f"{(et if et is not None else float('nan')):11.4f} "
            f"{(ar if ar is not None else float('nan')):7.4f} "
            f"{(ppl if ppl is not None else float('nan')):6.3f} "
            f"{vr_s:>16s} {vc_s:>16s}"
        )
    print(f"\n[analyze] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
