#!/usr/bin/env python3
"""PR #557 — ITEM-LEVEL convergence cross-check (advisor ask: "confirm the recovered
config actually converges the same items the broken serve truncated — don't trust the
aggregate alone"). LOCAL, NO FIRE, no GPU (re-reads existing .eval logs only).

For each axis we take the EXACT set of items the BROKEN #542 base truncated
(stop_reason in {max_tokens,model_length,length}, no parseable ANSWER) and ask what
the recovered ple_fold serve did on those SAME item ids:
  - still_truncated : ple_fold also truncates it (fold didn't help this item)
  - converged       : ple_fold now emits a parseable ANSWER (no longer truncates)
  - converged_correct: of the converged, how many ple_fold scores correct
Byte-identical item sets (prompt_sha asserted) make the id join exact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/vanilla_base_serve_regression"
F542 = ROOT / "research/validity/base_fullhead_shortchain_quality"
sys.path.insert(0, str(F542))
import failmodes  # noqa: E402

AXES = [
    ("mmlu_pro", F542 / "base_mmlu_pro.json", HERE / "ple_fold_mmlu_pro.json"),
    ("gpqa_diamond", F542 / "base_gpqa.json", HERE / "ple_fold_gpqa_diamond.json"),
]


def _per_sample_maps(json_path: Path):
    d = json.load(open(json_path))
    correct = {str(r["id"]): bool(r.get("correct")) for r in d["per_sample"]}
    answer = {str(r["id"]): r.get("answer") for r in d["per_sample"]}
    psha = {str(r["id"]): r.get("prompt_sha") for r in d["per_sample"]}
    return correct, answer, psha


def main() -> int:
    report = {}
    for axis, brk_json, rec_json in AXES:
        if not rec_json.exists():
            print(f"[xcheck] (skip {axis}: {rec_json.name} not present yet)", file=sys.stderr)
            continue
        bc = failmodes.classify_eval(brk_json)
        rc = failmodes.classify_eval(rec_json)
        brk_trunc = set(map(str, bc["truncation_ids"]))
        rec_trunc = set(map(str, rc["truncation_ids"]))
        rec_correct, rec_answer, rec_psha = _per_sample_maps(rec_json)
        _, _, brk_psha = _per_sample_maps(brk_json)

        # exact id join integrity (byte-identical prompts)
        common = set(brk_psha) & set(rec_psha)
        mism = [i for i in common if brk_psha[i] != rec_psha[i]]

        still_trunc = sorted(i for i in brk_trunc if i in rec_trunc)
        converged = sorted(i for i in brk_trunc if i not in rec_trunc
                           and rec_answer.get(i) not in (None, ""))
        converged_correct = sorted(i for i in converged if rec_correct.get(i))
        n_bt = len(brk_trunc)
        r = {
            "axis": axis,
            "n_broken_truncated": n_bt,
            "n_still_truncated_under_ple_fold": len(still_trunc),
            "n_converged": len(converged),
            "n_converged_and_correct": len(converged_correct),
            "convergence_rate": (len(converged) / n_bt) if n_bt else None,
            "converged_correct_rate": (len(converged_correct) / n_bt) if n_bt else None,
            "broken_truncation_rate": bc["truncation_rate"],
            "recovered_truncation_rate": rc["truncation_rate"],
            "n_prompt_mismatch": len(mism),
            "still_truncated_ids": still_trunc[:50],
        }
        report[axis] = r
        print(f"\n== {axis} ==")
        print(f"  broken base truncated {n_bt} items (trunc_rate {bc['truncation_rate']:.3f})")
        print(f"  under ple_fold: converged {len(converged)}/{n_bt} "
              f"({(len(converged)/n_bt*100 if n_bt else 0):.1f}%), of which correct "
              f"{len(converged_correct)} ; still truncated {len(still_trunc)}")
        print(f"  recovered trunc_rate {rc['truncation_rate']:.3f}  n_prompt_mismatch={len(mism)}")

    out = HERE / "converge_xcheck.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[xcheck] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
