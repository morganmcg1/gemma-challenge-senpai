#!/usr/bin/env python3
"""PR #702 proof-of-life / live W&B logger.

Stands up a SINGLE resumable run in group `int4-aime-selective-g32-build-ubel`
and pushes whatever per-seed AIME results exist on disk so far as partial pooled
scalars + a partial arm table. Re-runnable: it reads/creates `_wandb_run_id.txt`
and resumes the same run each call, so the final aggregation (log_wandb.py, same
id) lands the verdict in this very run. LOCAL analysis only.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
GROUP = "int4-aime-selective-g32-build-ubel"
ARMS = ["full_g128", "selective", "full_g32"]
RID_FILE = HERE / "_wandb_run_id.txt"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def arm_partial(arm: str):
    files = sorted(glob.glob(str(HERE / "results" / f"{arm}_seed*.json")))
    if not files:
        return None
    nc = n = 0
    per = []
    for fp in files:
        d = json.load(open(fp))
        c, t = int(d["n_correct_maj"]), int(d["n_problems"])
        nc += c
        n += t
        per.append(c / t if t else float("nan"))
    lo, hi = wilson(nc, n)
    return {"n_seeds": len(files), "n_correct": nc, "n": n,
            "pooled_acc": nc / n if n else float("nan"),
            "wilson_lo": lo, "wilson_hi": hi,
            "perseed_min": min(per), "perseed_mean": sum(per) / len(per),
            "perseed_max": max(per)}


def main() -> None:
    rid = RID_FILE.read_text().strip() if RID_FILE.exists() else wandb.util.generate_id()
    RID_FILE.write_text(rid)

    config = {
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0, "pr": 702,
        "bar": 0.420, "bf16_base": 0.460, "g128_ref": 0.347, "g32_ref": 0.438,
        "subset_source_run": "vjhzcvmu", "subset_n_modules": 48,
        "subset_f_param": 0.013526888815572418, "byte_law_tps_projected": 126.275,
        "protocol": ("AIME #31 gate basis: years 2024,2025-I,2025-II n=60 k=1 "
                     "sampled T=1.0 top_p=0.95 top_k=64 max_tokens=12288 min_tokens=8 "
                     "no-thinking, 5-seed pooled n=300 Wilson z=1.96 (matches #693)."),
        "build_note": ("fake-quant bf16-dense serve; g128 = double-quant "
                       "(g32-dequant -> g128 RTN, rel_err mean 0.1036 ~= served "
                       "single-quant ~0.10); full_g32 = official w4a16-ct g32 direct."),
    }

    run = wandb.init(entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                     id=rid, resume="allow", group=GROUP, name="ubel/selective-g32-build",
                     job_type="analysis", config=config,
                     tags=["pr702", "int4-aime", "selective-g32", "analysis_only",
                           "quality-recovery", "proof-of-life"])

    flat = {"analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "verdict": "PENDING"}
    cols = ["arm", "n_seeds", "pooled_acc", "n_correct", "n",
            "wilson95_lo", "wilson95_hi", "perseed_min", "perseed_mean", "perseed_max"]
    tbl = wandb.Table(columns=cols)
    for a in ARMS:
        r = arm_partial(a)
        if not r:
            continue
        flat[f"{a}_pooled_acc"] = r["pooled_acc"]
        flat[f"{a}_wilson_lo"] = r["wilson_lo"]
        flat[f"{a}_wilson_hi"] = r["wilson_hi"]
        flat[f"{a}_n_correct"] = r["n_correct"]
        flat[f"{a}_n"] = r["n"]
        flat[f"{a}_n_seeds"] = r["n_seeds"]
        flat[f"{a}_perseed_mean"] = r["perseed_mean"]
        tbl.add_data(a, r["n_seeds"], r["pooled_acc"], r["n_correct"], r["n"],
                     r["wilson_lo"], r["wilson_hi"], r["perseed_min"],
                     r["perseed_mean"], r["perseed_max"])
    run.log({"aime_arm_table_partial": tbl})
    run.summary.update(flat)
    print(f"[live] run={run.id} url={run.url}")
    print(json.dumps(flat, indent=2, default=str))
    run.finish()


if __name__ == "__main__":
    main()
