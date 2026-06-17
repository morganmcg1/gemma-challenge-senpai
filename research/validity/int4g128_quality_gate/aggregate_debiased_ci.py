#!/usr/bin/env python3
"""PR #610 aggregator: de-biased GPQA-Diamond gate verdict for int4_g128_lmhead.

Re-measures the GPQA-Diamond quality gate at land #598's de-biased model length
(--max-model-len 6144, vs #579's 4096) and deepens the sampled set from 3 to 10
seeds so the t-dist CI95 (df=9) is tight enough to give the human a DECISIVE
verdict on whether the locked-in submission (int4_g128_lmhead @ 126.378) clears
the GPQA-Diamond gate bar 0.4712 (= 0.90 x vanilla-base 0.5236).

Reads:
  * 6144 (de-biased, THIS PR):  results_6144/int4g128_gpqa_diamond_{greedy,sampled_s0..9}.json
  * 4096 (committed #579 base): results/int4g128_gpqa_diamond_{greedy,sampled_s0..2}.json

Emits the advisor's mandated deliverable keys to stdout + W&B group
``gpqa-diamond-debiased-ci``:
  gpqa_diamond_10seed_mean, gpqa_diamond_10seed_ci95, gpqa_diamond_10seed_verdict,
  gpqa_greedy_anchor_6144, gpqa_trunc_items_4096, gpqa_trunc_items_6144,
  gpqa_debias_delta_acc, analysis_only, official_tps.

analysis_only=True, official_tps=0 -- LOCAL served measurement, NO HF Job/fire.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

HERE = ROOT / "research" / "validity" / "int4g128_quality_gate"
RES_4096 = HERE / "results"          # committed #579 baseline (4096)
RES_6144 = HERE / "results_6144"     # de-biased this-PR sweep (6144)

# Gate bar: 0.90 x vanilla-base 0.5236 (W&B qi24h8zx/yokbmy9i), per PR #610 baseline.
GATE_BAR = 0.90 * 0.5236             # = 0.47124
GATE_BASE = 0.5236

# two-sided t_{0.975, df}
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
      7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t975(dof: int) -> float:
    return float("nan") if dof <= 0 else _T.get(dof, 1.96)


def _load(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def _stats(vals: list[float]):
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return {"mean": m, "n": 1, "sd": 0.0, "sem": 0.0, "ci95": None,
                "lo": m, "hi": m, "vals": vals}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    sem = sd / math.sqrt(n)
    half = _t975(n - 1) * sem
    return {"mean": m, "n": n, "sd": sd, "sem": sem, "ci95": half,
            "lo": m - half, "hi": m + half, "vals": vals}


def _verdict(stats: dict, bar: float) -> str:
    """Decisive read vs the bar from the t-dist CI95."""
    if stats is None or stats.get("ci95") is None:
        return "INSUFFICIENT-SEEDS"
    lo, hi = stats["lo"], stats["hi"]
    if hi < bar:
        return "DECISIVE-FAIL"      # CI entirely below the bar
    if lo > bar:
        return "PASS"              # CI entirely above the bar
    return "STILL-BORDERLINE"      # CI straddles -> n=198 cannot resolve


def _seed_files(res_dir: Path, seeds):
    out = {}
    for s in seeds:
        d = _load(res_dir / f"int4g128_gpqa_diamond_sampled_s{s}.json")
        if d is not None:
            out[s] = d
    return out


def _err_item_ids(d) -> set[str]:
    if not d:
        return set()
    return {r["id"] for r in d.get("per_sample", []) if r.get("error")}


def _scoremap(d) -> dict[str, bool]:
    """id -> bool(correct) for every scored sample."""
    if not d:
        return {}
    return {r["id"]: bool(r.get("correct")) for r in d.get("per_sample", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--peak-gpu-gb", type=float, default=None)
    ap.add_argument("--engine", default="vllm-0.22.1rc1.dev307")
    ap.add_argument("--seeds", default="0 1 2 3 4 5 6 7 8 9")
    args = ap.parse_args()

    seeds_10 = [int(x) for x in args.seeds.split()]

    # ---- 6144 de-biased: 10-seed sampled + greedy anchor ----
    s6144 = _seed_files(RES_6144, seeds_10)
    sampled_vals = []
    per_seed_6144 = {}
    for s in seeds_10:
        d = s6144.get(s)
        if d and d.get("accuracy") is not None and not math.isnan(d["accuracy"]):
            sampled_vals.append(d["accuracy"])
            per_seed_6144[s] = {
                "accuracy": d["accuracy"], "n_scored": d.get("n_scored"),
                "n_correct": d.get("n_correct"), "n_error": d.get("n_error"),
                "n_empty": d.get("n_empty"), "empty_rate": d.get("empty_rate"),
                "max_tokens": d.get("max_tokens"),
            }
    stats = _stats(sampled_vals)
    g6144 = _load(RES_6144 / "int4g128_gpqa_diamond_greedy.json")
    greedy_anchor = g6144.get("accuracy") if g6144 else None

    verdict = _verdict(stats, GATE_BAR)

    # ---- 4096 committed #579 baseline (for the truncation audit + debias delta) ----
    s4096 = _seed_files(RES_4096, [0, 1, 2])
    g4096 = _load(RES_4096 / "int4g128_gpqa_diamond_greedy.json")

    # ---- truncation audit: items force-failed (request error) at each model-len ----
    # vLLM 400s any request with prompt_tokens + max_tokens > max_model_len, and the
    # harness force-scores those WRONG (score_on_error). The set of erroring item ids
    # IS the structural truncation set: budget(4096)=4096-3072=1024, budget(6144)=3072.
    trunc_ids_4096 = set()
    for d in list(s4096.values()) + ([g4096] if g4096 else []):
        trunc_ids_4096 |= _err_item_ids(d)
    trunc_ids_6144 = set()
    for d in list(s6144.values()) + ([g6144] if g6144 else []):
        trunc_ids_6144 |= _err_item_ids(d)
    trunc_items_4096 = len(trunc_ids_4096)
    trunc_items_6144 = len(trunc_ids_6144)
    rescued_ids = sorted(trunc_ids_4096 - trunc_ids_6144)

    # ---- debias delta: acc(6144) - acc(4096) on the SAME seed (overlap 0,1,2 + greedy) ----
    debias = {}
    overlap_deltas = []
    for s in [0, 1, 2]:
        d6, d4 = s6144.get(s), s4096.get(s)
        if d6 and d4 and d6.get("accuracy") is not None and d4.get("accuracy") is not None:
            delta = d6["accuracy"] - d4["accuracy"]
            overlap_deltas.append(delta)
            # per-item flip decomposition (rigorous isolation of cause)
            m6, m4 = _scoremap(d6), _scoremap(d4)
            common = set(m6) & set(m4)
            flips = sorted(i for i in common if m6[i] != m4[i])
            debias[f"seed{s}"] = {
                "acc_4096": d4["accuracy"], "acc_6144": d6["accuracy"], "delta": delta,
                "n_items_flipped": len(flips),
                "flipped_ids": flips,
                "rescued_in_flips": [i for i in flips if i in set(rescued_ids)],
            }
    # greedy delta too (deterministic -> any flip is the rescued item or pure numerics)
    if g6144 and g4096 and g6144.get("accuracy") is not None and g4096.get("accuracy") is not None:
        m6, m4 = _scoremap(g6144), _scoremap(g4096)
        common = set(m6) & set(m4)
        flips = sorted(i for i in common if m6[i] != m4[i])
        debias["greedy"] = {
            "acc_4096": g4096["accuracy"], "acc_6144": g6144["accuracy"],
            "delta": g6144["accuracy"] - g4096["accuracy"],
            "n_items_flipped": len(flips), "flipped_ids": flips,
            "rescued_in_flips": [i for i in flips if i in set(rescued_ids)],
        }
    debias_delta_acc = (sum(overlap_deltas) / len(overlap_deltas)) if overlap_deltas else None

    # rescued-item score across 6144 seeds (did de-biasing convert forced-wrong -> right?)
    rescued_scores_6144 = {}
    for rid in rescued_ids:
        per = {}
        for s in seeds_10:
            d = s6144.get(s)
            if d:
                mm = _scoremap(d)
                if rid in mm:
                    per[f"s{s}"] = mm[rid]
        if g6144:
            mm = _scoremap(g6144)
            if rid in mm:
                per["greedy"] = mm[rid]
        rescued_scores_6144[rid] = per

    # ---- mandated deliverable keys ----
    ko = {
        "gpqa_diamond_10seed_mean": (stats["mean"] if stats else None),
        "gpqa_diamond_10seed_ci95": (stats["ci95"] if stats else None),
        "gpqa_diamond_10seed_verdict": verdict,
        "gpqa_greedy_anchor_6144": greedy_anchor,
        "gpqa_trunc_items_4096": trunc_items_4096,
        "gpqa_trunc_items_6144": trunc_items_6144,
        "gpqa_debias_delta_acc": debias_delta_acc,
        "analysis_only": True,
        "official_tps": 0,
    }
    detail = {
        **ko,
        "gate_bar": GATE_BAR,
        "gate_base_ref": GATE_BASE,
        "gate_frac": 0.90,
        "model_len_6144": 6144,
        "model_len_4096_baseline": 4096,
        "max_num_seqs_6144": 16,
        "n_dataset": (g6144.get("n_dataset") if g6144 else None),
        "n_seeds": (stats["n"] if stats else 0),
        "sd": (stats["sd"] if stats else None),
        "sem": (stats["sem"] if stats else None),
        "ci95_lo": (stats["lo"] if stats else None),
        "ci95_hi": (stats["hi"] if stats else None),
        "t975_df": (stats["n"] - 1 if stats else None),
        "per_seed_6144": per_seed_6144,
        "sampled_seeds_6144": (stats["vals"] if stats else []),
        "trunc_ids_4096": sorted(trunc_ids_4096),
        "trunc_ids_6144": sorted(trunc_ids_6144),
        "rescued_ids": rescued_ids,
        "rescued_scores_6144": rescued_scores_6144,
        "debias_per_seed": debias,
        "debias_overlap_seeds": [0, 1, 2],
        # #579 4096 reference numbers (for the report)
        "baseline_4096_3seed_mean": (sum(d["accuracy"] for d in s4096.values()) / len(s4096)
                                     if s4096 else None),
        "baseline_4096_greedy": (g4096.get("accuracy") if g4096 else None),
        "engine": args.engine,
        "checkpoint": "/workspace/gemma_build/int4_g128_lmhead",
        "decode_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
                            "source": "gemma-4-E4B-it/generation_config.json (lewtun #31)"},
        "min_tokens_guard": 8,
        "max_tokens": 3072,
        "dataset_seed": 12345,
        "peak_gpu_gb": args.peak_gpu_gb,
    }

    print("AGG610_SUMMARY " + json.dumps(detail, default=str))
    (HERE / "agg610_debiased_ci_summary.json").write_text(
        json.dumps(detail, indent=2, default=str))

    if args.wandb:
        run = wandb_logging.init_wandb_run(
            job_type="gpqa-diamond-debiased-ci",
            agent="kanna",
            name="kanna/gpqa-diamond-debiased-ci",
            group="gpqa-diamond-debiased-ci",
            notes=("PR #610: de-biased GPQA-Diamond gate for int4_g128_lmhead. "
                   "Re-measured at land #598's --max-model-len 6144 (vs #579's 4096) "
                   "with --max-num-seqs 16, deepened to 10 sampled seeds for a tight "
                   "t-dist CI95 (df=9). Decisive verdict vs bar 0.4712. dev307 engine, "
                   "min_tokens=8, T=1.0/top_p=0.95/top_k=64. LOCAL serve; "
                   "analysis_only, official_tps=0, NO FIRE."),
            tags=["quality-gate", "int4_g128_lmhead", "analysis-only", "pr-610",
                  "gpqa", "gpqa-diamond", "debiased", "10-seed-ci"],
            config={"pr": 610, "analysis_only": True, "official_tps": 0,
                    "engine": args.engine, "gate_bar": GATE_BAR, "gate_base_ref": GATE_BASE,
                    "model_len": 6144, "max_num_seqs": 16, "min_tokens_guard": 8,
                    "checkpoint": "int4_g128_lmhead", "n_seeds": (stats["n"] if stats else 0),
                    "dataset_seed": 12345, "max_tokens": 3072},
        )
        if run is None:
            print("[agg610] wandb disabled/unavailable; JSON only", flush=True)
            return 0
        for k, v in ko.items():
            run.summary[k] = v
        # per-seed accuracy series for the W&B plot
        for s in seeds_10:
            if s in per_seed_6144:
                run.log({"global_step": s, "gpqa_seed_accuracy": per_seed_6144[s]["accuracy"],
                         "gpqa_seed": s})
        wandb_logging.log_summary(run, detail, step=0)
        wandb_logging.log_json_artifact(
            run, name="gpqa_debiased_ci_detail", artifact_type="quality-eval", data=detail)
        run_id = getattr(run, "id", None)
        wandb_logging.finish_wandb(run)
        print(f"[agg610] wandb run_id={run_id}", flush=True)
        (HERE / "agg610_debiased_ci_summary.json").write_text(
            json.dumps({**detail, "wandb_run_id": run_id}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
