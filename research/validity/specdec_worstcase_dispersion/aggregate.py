#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #577 — aggregate the per-drafter dispersion runs into the worst-case verdict.

Reads ``results_{nospec,mtp,ngram}.json`` (one served run per drafter, 3 strata
each), computes the DISPERSION + WORST-CASE net-TPS that the 3 mean speed legs
do NOT deliver, decides against the ship speed gate, logs everything to W&B
(group ``base-fullhead-specdec-dispersion``), and prints the single-line
SENPAI-RESULT.

The decision statistic is the WORST-CASE per-stratum net-TPS for MTP K=7 (the
ship's drafter), not the mean: acceptance is workload-dependent and the ship is
evaluated on the hard-reasoning mix where the quality gate lives.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research" / "validity" / "specdec_worstcase_dispersion"

# Anchors from the PR card.
SHIP = 375.857            # official ship speed bar (a10g-small output_throughput)
SIGMA_HW = 4.864          # hardware noise band
FLOOR_FREE = 311.25       # magically-free decode-overhead floor (lawine #554)
BASELINE = 481.53         # current public best (untouched)
NOSPEC_ANCHOR_CITED = 252.69  # wirbel #553 base_fullhead no-spec served anchor (FAIR, graphed)
# Local wall_tps -> official output_throughput transfer. My surgical357 served cert
# (run k8nqmc2b) measured the SHIP at local wall_tps 357.43 (target 357.64) on THIS
# A10G pod; the official ship is 375.857. So local wall_tps must be scaled by
# ~1.051 to compare against the official 375.857 gate. Equivalently the official
# 375.857 bar maps to a LOCAL wall_tps bar of 357.64. Both forms reported.
SHIP_LOCAL = 357.64       # local-wall_tps equivalent of the official ship bar
TRANSFER = SHIP / SHIP_LOCAL  # ~1.05095 local->official
# net_tps uses decode_outputs.py duration_s, which includes a per-prompt tax
# (prefill + tokenize + sha256 + jsonl write). Calibrated so the HARD stratum's
# decode-only tps == the surgical357 cert wall_tps (357.43, AIME/hard workload,
# run k8nqmc2b): hard 24576 tok / (79.30 - x*48) = 357.43 -> x = 0.2196 s/prompt.
# decode_only_tps = compl_tok / (duration_s - x*num_records). Because this tax is
# mostly prefill (scales with prompt length) and HARD has the SHORTEST prompts,
# the fixed-x estimate UNDER-corrects longer-prompt strata, so decode_only_tps is
# a conservative LOWER BOUND for easy/mix. The verdict holds under both lenses.
PER_PROMPT_OVERHEAD_S = 0.2196
STRATA = ["easy", "mix", "hard"]

# Tag -> file mapping (driver writes results_<tag>.json).
TAGS = {"nospec": "nospec_n16_l512", "mtp": "mtp_n48_l512", "ngram": "ngram_n48_l512"}


def load(drafter: str, tag: str | None = None) -> dict | None:
    t = tag or TAGS.get(drafter, drafter)
    p = OUT / f"results_{t}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def stratum_tps(res: dict) -> dict[str, float]:
    return {s: res["per_stratum"][s]["net_tps"]
            for s in STRATA if s in res.get("per_stratum", {})}


def stratum_accept(res: dict) -> dict[str, float | None]:
    return {s: res["per_stratum"][s].get("accept_rate")
            for s in STRATA if s in res.get("per_stratum", {})}


def stratum_meanlen(res: dict) -> dict[str, float | None]:
    return {s: res["per_stratum"][s].get("mean_accept_len")
            for s in STRATA if s in res.get("per_stratum", {})}


def stratum_decode_only(res: dict) -> dict[str, float | None]:
    """Prefill-corrected decode-only tps (conservative lower bound; see header)."""
    out: dict[str, float | None] = {}
    for s in STRATA:
        ps = res.get("per_stratum", {}).get(s)
        if not ps:
            continue
        ct = ps.get("num_completion_tokens")
        dur = ps.get("duration_s")
        n = ps.get("num_records")
        if not (ct and dur and n):
            out[s] = None
            continue
        denom = dur - PER_PROMPT_OVERHEAD_S * n
        out[s] = (ct / denom) if denom > 0 else None
    return out


def mmm(vals: list[float]) -> tuple[float, float, float]:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return (float("nan"), float("nan"), float("nan"))
    return (min(v), statistics.median(v), max(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="stark/specdec-worstcase-dispersion")
    ap.add_argument("--wandb-group", default="base-fullhead-specdec-dispersion")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    runs = {d: load(d) for d in ("nospec", "mtp", "ngram")}
    missing = [d for d, r in runs.items() if r is None]
    if missing:
        print(f"[agg] WARNING missing drafter results: {missing}", flush=True)

    nospec_tps = stratum_tps(runs["nospec"]) if runs["nospec"] else {}
    agg: dict = {
        "anchors": {"ship": SHIP, "sigma_hw": SIGMA_HW, "floor_free": FLOOR_FREE,
                    "baseline": BASELINE, "nospec_anchor_cited": NOSPEC_ANCHOR_CITED},
        "nospec_tps_per_stratum": nospec_tps,
        "drafters": {},
    }

    # self-det from the nospec run.
    sd = (runs["nospec"] or {}).get("self_det") if runs["nospec"] else None
    agg["self_det"] = sd.get("self_det") if sd else None
    agg["self_det_detail"] = sd

    for d in ("mtp", "ngram"):
        res = runs[d]
        if res is None:
            continue
        tps = stratum_tps(res)
        acc = stratum_accept(res)
        mlen = stratum_meanlen(res)
        dtps = stratum_decode_only(res)
        dtps_vals = [dtps[s] for s in STRATA if dtps.get(s) is not None]
        dtps_worst = min(dtps_vals) if dtps_vals else float("nan")
        tps_vals = [tps[s] for s in STRATA if s in tps]
        acc_vals = [acc[s] for s in STRATA if s in acc]
        tmin, tmed, tmax = mmm(tps_vals)
        amin, amed, amax = mmm(acc_vals)
        tmean = statistics.fmean([x for x in tps_vals if x is not None])
        worst_stratum = min(tps, key=tps.get) if tps else None
        worst_accept_stratum = (min({s: a for s, a in acc.items() if a is not None},
                                    key=lambda s: acc[s])
                                if any(a is not None for a in acc.values()) else None)
        # speedup vs the FAIR no-spec anchor (252.69, a standard graphed cudagraph
        # stack — wirbel #553). NOT this stack's own eager no-spec: the surgical357
        # ship runs the MAIN model eager by design, so its no-spec floor (~85 tps)
        # is an artifact, and net_tps/own_nospec overstates the "speedup" ~4x.
        speedup = {s: (tps[s] / NOSPEC_ANCHOR_CITED) if tps.get(s) else None
                   for s in tps}
        # within-stack amortization factor: how much MTP recovers over this stack's
        # OWN eager no-spec floor (the 4x artifact, reported for transparency only).
        amort_vs_eager = {s: (tps[s] / nospec_tps[s])
                          if (s in nospec_tps and nospec_tps[s]) else None
                          for s in tps}
        hard_vs_easy = (acc.get("hard") / acc.get("easy")
                        if acc.get("hard") not in (None, 0) and acc.get("easy")
                        else None)
        per_pos_hard = (res["per_stratum"].get("hard", {}).get("per_pos_accept"))
        per_pos_easy = (res["per_stratum"].get("easy", {}).get("per_pos_accept"))
        # Ship-gate decision is made in OFFICIAL output_throughput units: local
        # end-to-end net_tps is scaled by TRANSFER (~1.051, cert local 357.64 ->
        # official 375.857), then the hardware noise band sigma_hw is subtracted.
        tmin_off = TRANSFER * tmin
        tmean_off = TRANSFER * tmean
        agg["drafters"][d] = {
            "tps_per_stratum": tps,
            "accept_per_stratum": acc,
            "mean_accept_len_per_stratum": mlen,
            "decode_only_tps_per_stratum": dtps,
            "decode_only_tps_worstcase": dtps_worst,
            "decode_only_tps_worstcase_official": TRANSFER * dtps_worst,
            "speedup_per_stratum": speedup,
            "amort_vs_eager_per_stratum": amort_vs_eager,
            "tps_worstcase": tmin,
            "tps_median": tmed,
            "tps_max": tmax,
            "tps_mean": tmean,
            "tps_worstcase_official": tmin_off,
            "tps_mean_official": tmean_off,
            "tps_mean_minus_worstcase": tmean - tmin,
            "accept_min": amin,
            "accept_median": amed,
            "accept_max": amax,
            "worst_stratum": worst_stratum,
            "worst_accept_stratum": worst_accept_stratum,
            "hardreasoning_accept_vs_easy_ratio": hard_vs_easy,
            "per_pos_accept_hard": per_pos_hard,
            "per_pos_accept_easy": per_pos_easy,
            # local-bar forms (compare raw local net_tps to the 357.64 local bar)
            "worstcase_clears_ship_local": bool((tmin - SIGMA_HW) >= SHIP_LOCAL),
            "mean_clears_ship_local": bool((tmean - SIGMA_HW) >= SHIP_LOCAL),
            # official-transfer forms (canonical gate decision, in 375.857 units)
            "worstcase_clears_ship": bool((tmin_off - SIGMA_HW) >= SHIP),
            "worstcase_minus_ship": tmin_off - SHIP,
            "mean_clears_ship": bool((tmean_off - SIGMA_HW) >= SHIP),
            "mean_minus_sigma_clears_ship": bool((tmean_off - SIGMA_HW) >= SHIP),
        }

    # PRIMARY = MTP K=7 (the ship's drafter).
    mtp = agg["drafters"].get("mtp", {})
    ngram = agg["drafters"].get("ngram", {})
    key = {
        "specdec_mtp_tps_worstcase": mtp.get("tps_worstcase"),
        "specdec_mtp_tps_worstcase_official": mtp.get("tps_worstcase_official"),
        "specdec_mtp_decode_only_tps_worstcase": mtp.get("decode_only_tps_worstcase"),
        "specdec_mtp_decode_only_tps_worstcase_official":
            mtp.get("decode_only_tps_worstcase_official"),
        "specdec_ngram_tps_worstcase": ngram.get("tps_worstcase"),
        "specdec_ngram_tps_worstcase_official": ngram.get("tps_worstcase_official"),
        "specdec_mtp_accept_min": mtp.get("accept_min"),
        "specdec_mtp_accept_median": mtp.get("accept_median"),
        "specdec_mtp_accept_max": mtp.get("accept_max"),
        "specdec_tps_mean_minus_worstcase": mtp.get("tps_mean_minus_worstcase"),
        "specdec_worstcase_clears_ship": mtp.get("worstcase_clears_ship"),
        "specdec_mean_clears_ship": mtp.get("mean_clears_ship"),
        "specdec_hardreasoning_accept_vs_easy_ratio":
            mtp.get("hardreasoning_accept_vs_easy_ratio"),
        "worst_stratum": mtp.get("worst_stratum"),
        "ship_bar_official": SHIP,
        "ship_bar_local": SHIP_LOCAL,
        "sigma_hw": SIGMA_HW,
        "self_det": agg["self_det"],
        "analysis_only": True,
        "official_tps": 0,
    }
    agg["key_outputs"] = key

    (OUT / "dispersion_summary.json").write_text(json.dumps(agg, indent=2, default=str))
    print("[agg] wrote dispersion_summary.json", flush=True)
    print("KEY_OUTPUTS " + json.dumps(key, default=str), flush=True)

    if not args.no_wandb:
        try:
            from scripts import wandb_logging
            run = wandb_logging.init_wandb_run(
                job_type="specdec-dispersion", agent="stark",
                name=args.wandb_name, group=args.wandb_group,
                tags=["specdec", "dispersion", "worst-case", "base_fullhead"],
                config={
                    "substrate": "base_fullhead", "mtp_k": 7, "ngram_k": 7,
                    "num_prompts": (runs["nospec"] or {}).get("num_prompts"),
                    "output_len": (runs["nospec"] or {}).get("output_len"),
                    "ship": SHIP, "sigma_hw": SIGMA_HW, "floor_free": FLOOR_FREE,
                    "baseline": BASELINE, "nospec_anchor_cited": NOSPEC_ANCHOR_CITED,
                    "strata": STRATA, "analysis_only": True, "official_tps": 0,
                },
            )
            step = 0
            # per-stratum scalars (per drafter)
            for d in ("nospec", "mtp", "ngram"):
                res = runs[d]
                if res is None:
                    continue
                for s in STRATA:
                    ps = res.get("per_stratum", {}).get(s)
                    if not ps:
                        continue
                    metrics = {
                        f"stratum/{d}/{s}/net_tps": ps.get("net_tps"),
                        f"stratum/{d}/{s}/accept_rate": ps.get("accept_rate"),
                        f"stratum/{d}/{s}/mean_accept_len": ps.get("mean_accept_len"),
                    }
                    if d in agg["drafters"]:
                        sp = agg["drafters"][d]["speedup_per_stratum"].get(s)
                        metrics[f"stratum/{d}/{s}/speedup_vs_fair_anchor"] = sp
                        am = agg["drafters"][d]["amort_vs_eager_per_stratum"].get(s)
                        metrics[f"stratum/{d}/{s}/amort_vs_eager_nospec"] = am
                        dt = agg["drafters"][d]["decode_only_tps_per_stratum"].get(s)
                        metrics[f"stratum/{d}/{s}/decode_only_tps"] = dt
                    wandb_logging.log_event(run, f"stratum_{d}_{s}", step=step,
                                            metrics={k: v for k, v in metrics.items()
                                                     if v is not None})
                    step += 1
            # per-position decay (mtp hard + easy)
            for d in ("mtp", "ngram"):
                for s in ("easy", "hard"):
                    pp = (runs[d] or {}).get("per_stratum", {}).get(s, {}).get("per_pos_accept") if runs[d] else None
                    if pp:
                        for pos, val in pp.items():
                            if val is not None:
                                run.log({"global_step": step,
                                         f"perpos/{d}/{s}/pos_{pos}": val})
                        step += 1
            # summary KEY OUTPUTS
            for k, v in key.items():
                if isinstance(v, bool):
                    run.summary[f"summary/{k}"] = int(v)
                elif isinstance(v, (int, float)) and v is not None:
                    run.summary[f"summary/{k}"] = v
                else:
                    run.summary[f"summary/{k}"] = v
            for d in ("mtp", "ngram"):
                if d in agg["drafters"]:
                    for k, v in agg["drafters"][d].items():
                        if isinstance(v, (int, float, bool)) and v is not None:
                            run.summary[f"summary/{d}/{k}"] = (int(v) if isinstance(v, bool) else v)
            wandb_logging.log_json_artifact(
                run, name="specdec_dispersion_summary",
                artifact_type="dispersion", data=agg)
            print(f"WANDB_RUN_ID {run.id}", flush=True)
            agg["wandb_run_id"] = run.id
            (OUT / "dispersion_summary.json").write_text(
                json.dumps(agg, indent=2, default=str))
            wandb_logging.finish_wandb(run)
        except Exception as exc:  # noqa: BLE001
            print(f"[agg] W&B logging failed: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
