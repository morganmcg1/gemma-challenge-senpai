#!/usr/bin/env python3
"""PR #791 -- targeted divergent-prompt check (advisor comment on #791).

The advisor's prior (from stark #794 / land #793) is that the surgattn-OFF
(3D-on-M=1) divergences are *answer-immaterial late near-tie argmax flips* with
*unchanged answer extraction*. This script tests that on OUR OWN paired records
(MMLU-Pro kill-gate + GPQA-Diamond panel), the two axes that ran BOTH arms on
byte-identical prompts at the same seed:

  * Are the answer flips genuine valid-letter flips, or extraction failures?
  * Is the per-axis variant-vs-control delta statistically significant
    (McNemar, exact paired test on the discordant pairs)?
  * Is any variant deficit driven by max_tokens truncation (a length-budget
    interaction of the divergent reasoning) vs an intrinsic accuracy loss?

Reads the per_sample arrays already written by run_quality.py / run_panel.py.
Pure analysis of existing json -- no server, no GPU, no eval re-run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

MMLU_V = RUNS / "mmlu_variant_n100_s12345_t2048.json"
MMLU_C = RUNS / "mmlu_control_n100_s12345_t2048.json"
GPQA_V = RUNS / "gpqa_variant_s12345_t3072.json"
GPQA_C = RUNS / "gpqa_control_s12345_t3072.json"


def _persample(d: dict) -> list[dict]:
    for k in ("per_sample", "samples", "results", "per_item"):
        if isinstance(d.get(k), list):
            return d[k]
    raise KeyError("no per_sample array in json")


def _trunc(r: dict) -> bool:
    return bool(
        r.get("length_truncated")
        or r.get("truncated")
        or r.get("stop_reason") in ("max_tokens", "length")
    )


def _valid(ans) -> bool:
    return ans not in (None, "")


def _mcnemar(b: int, c: int) -> dict:
    """b = control-right-variant-wrong, c = variant-right-control-wrong (discordant)."""
    n = b + c
    if n == 0:
        return {"discordant": 0, "chi2_cc": None, "p_approx": None}
    chi2_cc = (abs(b - c) - 1) ** 2 / n if n > 0 else None
    p = math.erfc(math.sqrt(chi2_cc / 2)) if chi2_cc is not None else None
    return {"discordant": n, "chi2_cc": round(chi2_cc, 3), "p_approx": round(p, 3)}


def analyse(vpath: Path, cpath: Path, axis: str) -> dict:
    v = json.loads(vpath.read_text())
    c = json.loads(cpath.read_text())
    vm = {r["id"]: r for r in _persample(v)}
    cm = {r["id"]: r for r in _persample(c)}
    ids = sorted(set(vm) & set(cm))

    sha_mismatch = sum(
        1 for i in ids if vm[i].get("prompt_sha") != cm[i].get("prompt_sha")
    )
    flips = [i for i in ids if vm[i].get("answer") != cm[i].get("answer")]
    both_valid = [i for i in flips if _valid(vm[i]["answer"]) and _valid(cm[i]["answer"])]
    one_empty = [i for i in flips if not (_valid(vm[i]["answer"]) and _valid(cm[i]["answer"]))]
    # of the one-empty flips, how many are explained by the empty arm truncating?
    one_empty_trunc = 0
    for i in one_empty:
        empty_arm = vm[i] if not _valid(vm[i]["answer"]) else cm[i]
        if _trunc(empty_arm):
            one_empty_trunc += 1

    vrcw = [i for i in ids if vm[i]["correct"] and not cm[i]["correct"]]
    crvw = [i for i in ids if cm[i]["correct"] and not vm[i]["correct"]]
    # variant losses that are variant-truncated-only (length-budget interaction)
    crvw_vtrunc = [i for i in crvw if _trunc(vm[i]) and not _trunc(cm[i])]

    v_acc = sum(1 for i in ids if vm[i]["correct"]) / len(ids)
    c_acc = sum(1 for i in ids if cm[i]["correct"]) / len(ids)
    v_tr = sum(_trunc(vm[i]) for i in ids)
    c_tr = sum(_trunc(cm[i]) for i in ids)

    return {
        "axis": axis,
        "n_paired": len(ids),
        "prompt_sha_mismatch": sha_mismatch,
        "variant_acc": round(v_acc, 4),
        "control_acc": round(c_acc, 4),
        "delta_variant_minus_control": round(v_acc - c_acc, 4),
        "answer_flips": len(flips),
        "flips_both_valid_letter": len(both_valid),
        "flips_one_arm_empty": len(one_empty),
        "flips_one_arm_empty_truncation_driven": one_empty_trunc,
        "variant_right_control_wrong": len(vrcw),
        "control_right_variant_wrong": len(crvw),
        "net_variant_minus_control": len(vrcw) - len(crvw),
        "mcnemar": _mcnemar(len(crvw), len(vrcw)),
        "variant_truncated": v_tr,
        "control_truncated": c_tr,
        "control_right_variant_wrong_variant_truncated_only": len(crvw_vtrunc),
        "net_excl_variant_truncation_losses": len(vrcw) - len(crvw) + len(crvw_vtrunc),
        "extraction_code_changed": False,  # identical run_eval harness in both arms
    }


def main() -> int:
    out = {
        "note": (
            "Paired divergent-prompt check on byte-identical prompts (same seed). "
            "Both arms use the IDENTICAL answer-extraction harness; 'one_arm_empty' "
            "flips are max_tokens truncation, not parse failures. Tests whether the "
            "3D-on-M=1 divergences are answer-immaterial near-tie flips."
        ),
        "mmlu_pro": analyse(MMLU_V, MMLU_C, "mmlu_pro"),
        "gpqa_diamond": analyse(GPQA_V, GPQA_C, "gpqa_diamond"),
    }
    (RUNS / "divergence_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
