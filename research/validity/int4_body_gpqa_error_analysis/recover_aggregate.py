#!/usr/bin/env python3
"""PR #619 -- aggregate the max_tokens=6144 recovery re-run into a before/after verdict.

Takes the stored #598 cells (max_tokens=3072) and overlays the recovery re-run cells
(max_tokens=6144) for the per-seed truncated-union ids, then recomputes the gpqa_main
deficit and McNemar BOTH ways:
  BEFORE = pure #598 (3072).
  AFTER  = #598 with the re-run (id,seed) cells substituted by their 6144 outcome.

A cell only changes if it truncated at 3072 (others are byte-identical pre-cap), so the
AFTER aggregate isolates the budget knob. Reports how many truncated cells RECOVERED
(wrong->right) per arm, how many still hit the 6144 cap, and the new int4-body deficit.
Self-contained; run with the inspect venv to read recovery stop_reasons:
  /tmp/land-inspect/bin/python recover_aggregate.py
"""
from __future__ import annotations

import glob
import json
import math
import random
import statistics as st
from pathlib import Path

from inspect_ai.log import read_eval_log

P598 = Path("/workspace/senpai/target/research/validity/gpqa_larger_instrument_ci/results")
HERE = Path("/workspace/senpai/target/research/validity/int4_body_gpqa_error_analysis")
RES = HERE / "recover_results"
Z = 1.959963984540054
QUALITY_FRAC = 0.90


def wilson_lo(k, n, z=Z):
    if n == 0:
        return float("nan")
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def _logpmf(k, n, p=0.5):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def binom_two_sided_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    m = max(a, b)
    return min(1.0, 2.0 * sum(math.exp(_logpmf(k, n)) for k in range(m, n + 1)))


def load598(prefix):
    """(id,seed)->correct AND (id,seed)->stop_reason from the #598 result JSONs + eval logs."""
    cell, stop = {}, {}
    for f in sorted(glob.glob(str(P598 / f"{prefix}_gpqa_main_mt8_s*.json"))):
        d = json.load(open(f))
        seed = d["sampling_seed"]
        el = d.get("eval_log")
        srmap = {}
        if el and Path(el).exists():
            log = read_eval_log(el)
            for s in (log.samples or []):
                o = getattr(s, "output", None)
                stp = None
                if o is not None:
                    ch = getattr(o, "choices", None)
                    if ch:
                        stp = getattr(ch[0], "stop_reason", None)
                    if stp is None:
                        stp = getattr(o, "stop_reason", None)
                srmap[str(s.id)] = stp
        for r in d["per_sample"]:
            if r.get("value") not in ("C", "I"):
                continue
            cell[(r["id"], seed)] = bool(r["correct"])
            stop[(r["id"], seed)] = srmap.get(r["id"])
    return cell, stop


def load_recover(prefix):
    """(id,seed)->{'correct':bool,'stop':..}; stop from the recovery eval logs."""
    cell = {}
    for f in sorted(glob.glob(str(RES / f"{prefix}_recover_s*.json"))):
        if Path(f).name.startswith("_smoke"):
            continue
        d = json.load(open(f))
        seed = d["sampling_seed"]
        stops = {}
        el = d.get("eval_log")
        if el and Path(el).exists():
            log = read_eval_log(el)
            for s in (log.samples or []):
                o = getattr(s, "output", None)
                stp = None
                if o is not None:
                    ch = getattr(o, "choices", None)
                    if ch:
                        stp = getattr(ch[0], "stop_reason", None)
                    if stp is None:
                        stp = getattr(o, "stop_reason", None)
                stops[str(s.id)] = stp
        for r in d["per_sample"]:
            if r.get("value") not in ("C", "I"):
                continue
            cell[(r["id"], seed)] = {"correct": bool(r["correct"]), "stop": stops.get(r["id"])}
    return cell


def mcnemar(b_cell, f_cell):
    shared = b_cell.keys() & f_cell.keys()
    n01 = sum(1 for k in shared if b_cell[k] and not f_cell[k])
    n10 = sum(1 for k in shared if (not b_cell[k]) and f_cell[k])
    return len(shared), n01, n10, binom_two_sided_p(n01, n10)


def seed_acc(cell):
    by = {}
    for (i, s), c in cell.items():
        by.setdefault(s, []).append(c)
    accs = [sum(v) / len(v) for s, v in sorted(by.items())]
    return accs, st.mean(accs)


def main():
    (b598, b_stop), (f598, f_stop) = load598("base"), load598("base_fullhead")
    brec, frec = load_recover("base"), load_recover("base_fullhead")

    # Overlay ONLY cells that were TRUNCATED at 3072 in #598. A non-truncated cell emits
    # the identical token stream at 6144 (the cap is never reached), so its outcome is
    # unchanged by the knob -- re-substituting it would only inject stack batch-variance
    # noise. Gating on the #598 stop_reason keeps the AFTER aggregate a clean budget A/B.
    bAfter = dict(b598)
    fAfter = dict(f598)
    n_overlay_b = n_overlay_f = 0
    for k, v in brec.items():
        if b_stop.get(k) == "max_tokens":
            bAfter[k] = v["correct"]
            n_overlay_b += 1
    for k, v in frec.items():
        if f_stop.get(k) == "max_tokens":
            fAfter[k] = v["correct"]
            n_overlay_f += 1

    # recovery accounting on the TRUNCATED re-run cells (the ones the knob can move)
    def recov(rec, before, stop598):
        trunc = {k: v for k, v in rec.items() if stop598.get(k) == "max_tokens"}
        recovered = sum(1 for k, v in trunc.items() if v["correct"])  # before were 100% wrong
        still_trunc = sum(1 for v in trunc.values() if v["stop"] == "max_tokens")
        # determinism witness: non-truncated re-run cells that flipped (should be ~0)
        nontrunc = {k: v for k, v in rec.items() if stop598.get(k) != "max_tokens"}
        flips = sum(1 for k, v in nontrunc.items() if v["correct"] != before.get(k))
        return {"truncated_cells_rerun": len(trunc),
                "recovered_wrong_to_right": recovered,
                "recovery_rate": (recovered / len(trunc)) if trunc else float("nan"),
                "still_truncated_at_6144": still_trunc,
                "nontruncated_rerun_cells": len(nontrunc),
                "nontruncated_flips_determinism_witness": flips}

    b_acc0, b_m0 = seed_acc(b598)
    f_acc0, f_m0 = seed_acc(f598)
    b_acc1, b_m1 = seed_acc(bAfter)
    f_acc1, f_m1 = seed_acc(fAfter)

    sh0, n01_0, n10_0, p0 = mcnemar(b598, f598)
    sh1, n01_1, n10_1, p1 = mcnemar(bAfter, fAfter)

    out = {
        "pr": 619, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "knob": "max_tokens 3072 -> 6144 (max_model_len 6144->8960)",
        "rerun_scope": "per-seed truncated-union cells, both arms, all re-run seeds",
        "overlay_cells_base": n_overlay_b, "overlay_cells_int4": n_overlay_f,
        "recovery_base": recov(brec, b598, b_stop),
        "recovery_int4": recov(frec, f598, f_stop),
        "BEFORE_3072": {
            "base_mean_acc": b_m0, "int4_mean_acc": f_m0, "deficit": b_m0 - f_m0,
            "base_per_seed": b_acc0, "int4_per_seed": f_acc0,
            "mcnemar_shared": sh0, "n01_base_right_int4_wrong": n01_0,
            "n10_base_wrong_int4_right": n10_0, "mcnemar_p_exact": p0,
        },
        "AFTER_6144": {
            "base_mean_acc": b_m1, "int4_mean_acc": f_m1, "deficit": b_m1 - f_m1,
            "base_per_seed": b_acc1, "int4_per_seed": f_acc1,
            "mcnemar_shared": sh1, "n01_base_right_int4_wrong": n01_1,
            "n10_base_wrong_int4_right": n10_1, "mcnemar_p_exact": p1,
        },
        "deficit_before": b_m0 - f_m0,
        "deficit_after": b_m1 - f_m1,
        "deficit_shrink": (b_m0 - f_m0) - (b_m1 - f_m1),
        "deficit_shrink_pct": 100.0 * (((b_m0 - f_m0) - (b_m1 - f_m1)) / (b_m0 - f_m0))
        if (b_m0 - f_m0) else float("nan"),
        "mcnemar_p_after_gt_0p05": bool(p1 > 0.05),
    }
    (HERE / "recover_aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
