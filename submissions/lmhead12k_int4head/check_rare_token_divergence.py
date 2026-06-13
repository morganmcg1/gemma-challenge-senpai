#!/usr/bin/env python
"""Quantify the rare-token divergence risk of the lmhead12k empirical prune.

The pruned lm_head can only emit kept_ids; any token the full model would emit
that is NOT in kept_ids is clipped -> a greedy-identity divergence and a PPL spike.
This script measures, on the captured benchmark decode outputs, how often that
would happen.

It reports two rates:
  * vs kept_ids (the shipped set): hard-includes the observed emissions, so this
    is ~0 by construction -- it confirms the shipped set covers what we have.
  * vs a leave-decode-out frequency top-K: the honest private-set RISK proxy --
    "if we had NOT seen these emissions, how many would a pure frequency cut clip?"

Pure-CPU; no model, no GPU.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECODE_FILE = ROOT / "research/local_validation/vllm_baseline_128/decode_outputs.jsonl"
DECODE_FILE_FALLBACK = ROOT / "research/local_validation/vllm_baseline/decode_outputs_128.jsonl"
GT_FILE = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
KEPT_IDS = ROOT / "submissions/lmhead12k_empirical/kept_ids.json"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decode-file", default=str(DECODE_FILE))
    ap.add_argument("--gt-file", default=str(GT_FILE))
    ap.add_argument("--kept-ids", default=str(KEPT_IDS))
    args = ap.parse_args()

    decode_file = Path(args.decode_file)
    if not decode_file.exists() and DECODE_FILE_FALLBACK.exists():
        decode_file = DECODE_FILE_FALLBACK
    decode = _read_jsonl(decode_file)
    gt = _read_jsonl(Path(args.gt_file))
    meta = json.loads(Path(args.kept_ids).read_text())
    kept = set(meta["kept_ids"])
    k = meta["K"]

    # Leave-decode-out frequency top-K: rank purely on the GT corpus, then ask how
    # many decode emissions a top-K cut would have clipped (private-set proxy).
    gt_freq: Counter[int] = Counter()
    for rec in gt:
        gt_freq.update(rec["target_token_ids"])
        gt_freq.update(rec["context_token_ids"])
    gt_topk = {t for t, _ in gt_freq.most_common(k)}

    n_clipped_vs_kept = 0
    n_clipped_vs_gt_topk = 0
    n_tokens = 0
    per_record = []
    for rec in decode:
        toks = rec["completion_token_ids"]
        c_kept = sum(1 for t in toks if t not in kept)
        c_gt = sum(1 for t in toks if t not in gt_topk)
        n_clipped_vs_kept += c_kept
        n_clipped_vs_gt_topk += c_gt
        n_tokens += len(toks)
        if c_gt:
            per_record.append({
                "id": rec["id"],
                "clipped_vs_gt_topk": c_gt,
                "n_tokens": len(toks),
            })

    # Finite-PPL coverage: every GT target token must be in kept.
    gt_target = set()
    for rec in gt:
        gt_target.update(rec["target_token_ids"])
    missing_gt_target = sorted(gt_target - kept)

    report = {
        "kept_size": len(kept),
        "K": k,
        "decode_records": len(decode),
        "decode_records_expected": 128,
        "decode_tokens_total": n_tokens,
        "rare_token_divergence_vs_kept": {
            "clipped": n_clipped_vs_kept,
            "rate": round(n_clipped_vs_kept / max(1, n_tokens), 6),
            "note": "hard-included, expected ~0; confirms shipped set covers captures",
        },
        "rare_token_divergence_vs_gt_topk_leave_decode_out": {
            "clipped": n_clipped_vs_gt_topk,
            "rate": round(n_clipped_vs_gt_topk / max(1, n_tokens), 6),
            "note": "private-set risk proxy: emissions a GT-only top-K would clip",
            "records_with_clips": per_record,
        },
        "finite_ppl_gt_target_missing": {
            "count": len(missing_gt_target),
            "sample": missing_gt_target[:20],
            "ppl_finite": len(missing_gt_target) == 0,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
