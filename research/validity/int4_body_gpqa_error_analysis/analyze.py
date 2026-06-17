#!/usr/bin/env python3
"""PR #619 -- int4-body GPQA failure-mode analysis. LOCAL, NO FIRE, analysis_only.

Reuses the #598 gpqa_main K=5 sampling sweep (run n4ro7bzk) -- NO new eval pass for
the characterization step. Two arms on the identical faithful #589 stack; the ONLY
difference is the body checkpoint:
  base          = UNQUANTIZED bf16 google/gemma-4-E4B-it (gate denominator).
  base_fullhead = int4-W4A16-g32 QAT body + native 262k bf16 lm_head.

#598 found base_fullhead IS a gpqa_main regression (McNemar exact p=0.0009; n01=309
base-right/int4-wrong vs n10=231 base-wrong/int4-right over 2240 (id,seed) cells).
This script reads the McNemar-DISCORDANT cell that drives the regression -- the n01
cells (base correct AND int4-body wrong) -- and tags each by failure mode:

  (i)   truncated      = int4 hit max_tokens (stop_reason=max_tokens) -> no ANSWER line
  (ii)  first-tok-EOS  = empty completion (min_tokens=8 should make this 0)
  (iii) extraction     = model STATED the target letter but scorer extracted wrong
  (iv)  genuine        = model itself declared a wrong letter (real reasoning error)
  (v)   domain         = GPQA High-level domain clustering (Physics/Chem/Bio)

stop_reason + completion text come from the #598 inspect .eval logs (eval_log field
in each result JSON). Self-contained; run with the inspect client venv:
  /tmp/land-inspect/bin/python analyze.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics as st
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from inspect_ai.log import read_eval_log

P598 = Path("/workspace/senpai/target/research/validity/gpqa_larger_instrument_ci/results")
HERE = Path("/workspace/senpai/target/research/validity/int4_body_gpqa_error_analysis")
ANSWER_PAT = re.compile(r"ANSWER\s*[:=]\s*\(?([ABCD])\)?", re.I)


def load_cells(prefix: str):
    """(id,seed)->correct ; (id,seed)->per_sample row , from the #598 result JSONs."""
    cell, meta, logs = {}, {}, {}
    for f in sorted(glob.glob(str(P598 / f"{prefix}_gpqa_main_mt8_s*.json"))):
        d = json.load(open(f))
        seed = d["sampling_seed"]
        logs[seed] = d["eval_log"]
        for r in d["per_sample"]:
            if r.get("value") not in ("C", "I"):
                continue  # drop errors/unscored
            cell[(r["id"], seed)] = bool(r["correct"])
            meta[(r["id"], seed)] = r
    return cell, meta, logs


def read_stop_and_text(logs: dict[int, str]):
    """(id,seed)->{'stop':..,'text':completion} from the inspect .eval logs."""
    out = {}
    for seed, el in logs.items():
        log = read_eval_log(el)
        for s in (log.samples or []):
            o = getattr(s, "output", None)
            stop, text = None, ""
            if o is not None:
                text = o.completion or ""
                ch = getattr(o, "choices", None)
                if ch:
                    stop = getattr(ch[0], "stop_reason", None)
                if stop is None:
                    stop = getattr(o, "stop_reason", None)
            out[(str(s.id), seed)] = {"stop": stop, "text": text}
    return out


def domain_map() -> dict[str, str]:
    path = hf_hub_download(repo_id="Wanfq/gpqa", filename="gpqa_main.csv",
                           repo_type="dataset", token=os.environ.get("HF_TOKEN") or None)
    dom = {}
    with open(path) as f:
        r = csv.DictReader(f)
        idcol = [c for c in r.fieldnames if c.strip().lower() in ("record id", "record_id")][0]
        dcol = [c for c in r.fieldnames if "high-level domain" in c.strip().lower()][0]
        for row in r:
            dom[row[idcol].strip()] = row[dcol].strip()
    return dom


def main():
    b_cell, b_meta, b_logs = load_cells("base")
    f_cell, f_meta, f_logs = load_cells("base_fullhead")
    f_st = read_stop_and_text(f_logs)
    b_st = read_stop_and_text(b_logs)
    dom = domain_map()

    shared = b_cell.keys() & f_cell.keys()
    n01 = [k for k in shared if b_cell[k] and not f_cell[k]]   # base right, int4 WRONG
    n10 = [k for k in shared if (not b_cell[k]) and f_cell[k]]  # base wrong, int4 right

    # ---- categorize each n01 cell ----
    cats = {"i_truncated": [], "ii_empty": [], "iii_extraction": [], "iv_genuine": []}
    for k in n01:
        fr = f_meta[k]
        stop = f_st[k]["stop"]
        text = f_st[k]["text"]
        ans = fr["answer"]
        if fr["empty"]:
            cats["ii_empty"].append(k)
        elif stop == "max_tokens":
            cats["i_truncated"].append(k)
        else:
            # normal stop with an extracted/none answer
            m = ANSWER_PAT.findall(text)
            stated = m[-1].upper() if m else None
            tgt = fr["target"]
            if stated is not None and stated == tgt and ans != tgt:
                cats["iii_extraction"].append(k)
            else:
                cats["iv_genuine"].append(k)

    # truncation causality: every truncated cell is wrong (no ANSWER line). count net.
    all_f_trunc = [k for k in shared if f_st[k]["stop"] == "max_tokens"]
    all_b_trunc = [k for k in shared if b_st[k]["stop"] == "max_tokens"]
    f_trunc_wrong = sum(1 for k in all_f_trunc if not f_cell[k])
    b_trunc_wrong = sum(1 for k in all_b_trunc if not b_cell[k])
    n01_trunc = len(cats["i_truncated"])
    n10_b_trunc = sum(1 for k in n10 if b_st[k]["stop"] == "max_tokens")

    # verbosity: int4 vs base completion chars on n01 cells
    f_chars = [len(f_st[k]["text"]) for k in n01]
    b_chars = [len(b_st[k]["text"]) for k in n01]

    # domain clustering of the regression (n01 normalized by shared-cell prevalence)
    n01_dom = Counter(dom.get(k[0], "?") for k in n01)
    shared_dom = Counter(dom.get(k[0], "?") for k in shared)
    trunc_union_ids = sorted({k[0] for k in all_f_trunc} | {k[0] for k in all_b_trunc})
    trunc_dom = Counter(dom.get(i, "?") for i in trunc_union_ids)

    out = {
        "pr": 619, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "source_run": "n4ro7bzk", "instrument": "gpqa_main", "n_items": 448, "K_seeds": 5,
        "shared_cells": len(shared), "n01_base_right_int4_wrong": len(n01),
        "n10_base_wrong_int4_right": len(n10), "net_mcnemar_margin": len(n01) - len(n10),
        "category_counts_n01": {k: len(v) for k, v in cats.items()},
        "truncation": {
            "int4_truncated_cells_total": len(all_f_trunc),
            "base_truncated_cells_total": len(all_b_trunc),
            "int4_truncated_pct_wrong": 100.0 * f_trunc_wrong / max(1, len(all_f_trunc)),
            "base_truncated_pct_wrong": 100.0 * b_trunc_wrong / max(1, len(all_b_trunc)),
            "n01_int4_truncated": n01_trunc,
            "n10_base_truncated": n10_b_trunc,
            "net_truncation_regression_votes": n01_trunc - n10_b_trunc,
            "pct_of_net_margin": 100.0 * (n01_trunc - n10_b_trunc) / max(1, len(n01) - len(n10)),
            "full_truncation_asymmetry_cells": len(all_f_trunc) - len(all_b_trunc),
        },
        "verbosity_n01": {
            "int4_chars_median": int(st.median(f_chars)), "int4_chars_mean": int(st.mean(f_chars)),
            "base_chars_median": int(st.median(b_chars)), "base_chars_mean": int(st.mean(b_chars)),
        },
        "domain_n01_rate_per_1000_shared": {
            d: round(1000 * n01_dom[d] / shared_dom[d], 1) for d in ("Physics", "Chemistry", "Biology")
        },
        "domain_truncation_union": dict(trunc_dom),
        "trunc_union_n_ids": len(trunc_union_ids),
    }
    (HERE / "breakdown.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
