#!/usr/bin/env python3
"""PR #581: ground the unquantized-base MMLU-Pro / GPQA / GSM8K denominators.

Goal: the corrected >=90% gate bars (MMLU-Pro 0.601 / GPQA 0.423 / GSM8K 0.828)
are only as trustworthy as the cited base anchors they divide (0.668 / 0.470 /
0.920). This script grounds all three on OUR exact harness under the mandated
generation_config.json sampling (lewtun #31: do_sample T=1.0 top_p=0.95 top_k=64)
with the min_tokens=8 EOS-guard, then -- per task -- reports the measured base
number, whether it confirms or refutes the cited value, and the corrected 90% bar
(0.9 x measured).

Sources (all on this branch, identical harness/engine vLLM 0.22.1rc1.dev307):
  * MMLU-Pro / GPQA  -- #563 already served the fp16 *vanilla* (unquantized) base
    arm under the mandated sampling protocol, 3 seeds, on
    research/validity/downstream_quality_eval/run_eval.py, and PROVED min_tokens=8
    is a mechanical no-op on it (mt8_confirmatory.json: empty_rate=0). We reuse
    those banked means rather than burn redundant compute; the seed-0 mt8 cells
    are carried as the min_tokens=8 confirmation.
  * GSM8K            -- the genuinely missing cell. Measured fresh here on the
    bf16 base via research/downstream_quality_gsm8k/gsm8k_eval.py (8-shot CoT,
    strict last-anchor match), min_tokens=8 sampled across 3 seeds + a no-guard
    seed-1234 sampled/greedy anchor (subset-matched to the int4 base run).

analysis_only=true, official_tps=0. No HF Job, no submission.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
ROOT = HERE.parents[2]
K563 = ROOT / "research" / "validity" / "base_fullhead_quality_sampling" / "results"

# Cited base anchors to ground (PR #581 body). GSM8K "~0.920".
CITED = {"mmlupro": 0.668, "gpqa": 0.470, "gsm8k": 0.920}
GATE_REL = 0.90

# t(0.975, dof) for small-sample CIs.
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def _t975(dof: int) -> float:
    return _T.get(dof, 1.96) if dof > 0 else float("nan")


def _mean_ci(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return {"mean": None, "n": 0, "sd": None, "ci95": None, "lo": None, "hi": None, "vals": []}
    m = sum(vals) / n
    if n == 1:
        return {"mean": m, "n": 1, "sd": 0.0, "ci95": None, "lo": m, "hi": m, "vals": vals}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    half = _t975(n - 1) * sd / math.sqrt(n)
    return {"mean": m, "n": n, "sd": sd, "ci95": half, "lo": m - half, "hi": m + half, "vals": vals}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def cited_match(measured: dict, cited: float) -> dict:
    """Confirm/refute the cited anchor.

    within_ci95 : cited consistent with the measured base at 95% (cited in [lo,hi]);
                  the statistically defensible 'not refuted' test.
    rel_diff    : (measured_mean - cited) / cited; sign+magnitude of the gap.
    verdict     : 'confirm' if within_ci95 AND |rel_diff|<=0.03;
                  'refute-low'/'refute-high' if cited is outside CI;
                  'within-CI-but-off' if inside the (wide) CI yet |rel_diff|>0.03.
    """
    m = measured["mean"]
    lo, hi = measured.get("lo"), measured.get("hi")
    rel = (m - cited) / cited if cited else float("nan")
    within = (lo is not None and hi is not None and lo <= cited <= hi)
    if within and abs(rel) <= 0.03:
        verdict = "confirm"
    elif not within:
        verdict = "refute-low" if cited < (lo if lo is not None else m) else "refute-high"
    else:
        verdict = "within-CI-but-off"
    return {"within_ci95": bool(within), "rel_diff": rel, "verdict": verdict,
            "match": bool(verdict == "confirm")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--peak-gpu-gb", type=float, default=None)
    args = ap.parse_args()

    # ---- MMLU-Pro / GPQA: reuse #563 fp16 vanilla(=unquantized) base, mandated sampling ----
    v563 = _load(K563 / "verdict_marker.json")
    mt8 = _load(K563 / "mt8_confirmatory.json")
    if v563 is None:
        raise SystemExit("FATAL: #563 verdict_marker.json not found")
    ko = v563["key_outputs"]
    pt = v563["per_task"]

    mmlu_meas = {"mean": ko["mmlu_pro_vanilla_sampled"]["mean"],
                 "ci95": ko["mmlu_pro_vanilla_sampled"]["ci95"],
                 "n": ko["mmlu_pro_vanilla_sampled"]["n"],
                 "vals": pt["mmlu_pro"]["fp16_sampled"]["vals"],
                 "lo": pt["mmlu_pro"]["fp16_sampled"]["lo"],
                 "hi": pt["mmlu_pro"]["fp16_sampled"]["hi"],
                 "greedy": pt["mmlu_pro"]["fp16_greedy"],
                 "mt8_seed0": (mt8 or {}).get("cells", {}).get("fp16_mmlu_pro", {}).get("sampled_mt8"),
                 "source": "PR#563 fp16 vanilla-base, run_eval.py, n=500/seed, 3 seeds, mandated sampling"}
    gpqa_meas = {"mean": ko["gpqa_vanilla_sampled"]["mean"],
                 "ci95": ko["gpqa_vanilla_sampled"]["ci95"],
                 "n": ko["gpqa_vanilla_sampled"]["n"],
                 "vals": pt["gpqa"]["fp16_sampled"]["vals"],
                 "lo": pt["gpqa"]["fp16_sampled"]["lo"],
                 "hi": pt["gpqa"]["fp16_sampled"]["hi"],
                 "greedy": pt["gpqa"]["fp16_greedy"],
                 "mt8_seed0": (mt8 or {}).get("cells", {}).get("fp16_gpqa", {}).get("sampled_mt8"),
                 "source": "PR#563 fp16 vanilla-base, run_eval.py, GPQA-Diamond n=198, 3 seeds, mandated sampling"}

    # ---- GSM8K: fresh on bf16 unquantized base, min_tokens=8 sampled across seeds ----
    gsm_seeds = []
    gsm_vals = []
    for s in (1234, 1235, 1236):
        d = _load(RES / f"base_bf16_s{s}_mt8_sampled.json")
        if d and d.get("accuracy") is not None:
            gsm_vals.append(d["accuracy"])
            gsm_seeds.append({"seed": s, "acc": d["accuracy"], "n": d["n_problems"],
                              "n_correct": d["n_correct"], "strict_rate": d.get("strict_rate"),
                              "extract_fail_rate": d.get("extract_fail_rate"),
                              "truncation_rate": d.get("truncation_rate")})
    gsm_ci = _mean_ci(gsm_vals)
    ng = _load(RES / "base_bf16_s1234_noguard_sampled.json")
    gg = _load(RES / "base_bf16_s1234_noguard_greedy.json")
    gsm_meas = {"mean": gsm_ci["mean"], "ci95": gsm_ci["ci95"], "n": gsm_ci["n"],
                "vals": gsm_ci["vals"], "lo": gsm_ci["lo"], "hi": gsm_ci["hi"],
                "greedy": (gg or {}).get("accuracy"),
                "noguard_sampled_s1234": (ng or {}).get("accuracy"),
                "mt8_seed1234": next((x["acc"] for x in gsm_seeds if x["seed"] == 1234), None),
                "per_seed": gsm_seeds,
                "source": "PR#581 bf16 unquantized base, gsm8k_eval.py, 8-shot CoT n=500, mt8 sampled 3 seeds"}
    # GSM8K min_tokens=8 no-op delta on the clean base (s1234: mt8 vs no-guard, same subset)
    if gsm_meas["mt8_seed1234"] is not None and gsm_meas["noguard_sampled_s1234"] is not None:
        gsm_meas["mt8_noop_delta_s1234"] = gsm_meas["mt8_seed1234"] - gsm_meas["noguard_sampled_s1234"]

    tasks = {"mmlupro": mmlu_meas, "gpqa": gpqa_meas, "gsm8k": gsm_meas}

    summary = {
        "pr": 581, "analysis_only": True, "official_tps": 0,
        "engine": "vllm-0.22.1rc1.dev307",
        "decode_protocol": {"do_sample": True, "temperature": 1.0, "top_p": 0.95,
                            "top_k": 64, "min_tokens_guard": 8,
                            "source": "gemma-4-E4B-it/generation_config.json (lewtun #31)"},
        "model_unquantized_base": "google/gemma-4-E4B-it (bf16)",
        "gate_rel": GATE_REL,
        "per_task": {},
    }
    flat = {}
    for key, meas in tasks.items():
        cm = cited_match(meas, CITED[key])
        bar_measured = GATE_REL * meas["mean"]
        bar_cited = GATE_REL * CITED[key]
        rec = {
            "cited": CITED[key],
            "measured_base": meas["mean"],
            "measured_ci95": meas.get("ci95"),
            "measured_lo": meas.get("lo"), "measured_hi": meas.get("hi"),
            "measured_n_seeds": meas.get("n"),
            "measured_vals": meas.get("vals"),
            "greedy_anchor": meas.get("greedy"),
            "cited_match": cm["match"],
            "cited_verdict": cm["verdict"],
            "cited_within_ci95": cm["within_ci95"],
            "rel_diff_measured_vs_cited": cm["rel_diff"],
            "gate_bar_90_measured": bar_measured,
            "gate_bar_90_cited": bar_cited,
            "gate_bar_90_shift": bar_measured - bar_cited,
            "source": meas["source"],
        }
        for extra in ("mt8_seed0", "mt8_seed1234", "noguard_sampled_s1234",
                      "mt8_noop_delta_s1234", "per_seed"):
            if extra in meas:
                rec[extra] = meas[extra]
        summary["per_task"][key] = rec
        flat[f"unquantized_base_{key}"] = meas["mean"]
        flat[f"cited_match_{key}"] = cm["match"]
        flat[f"cited_verdict_{key}"] = cm["verdict"]
        flat[f"gate_bar_90_{key}"] = bar_measured
        flat[f"gate_bar_90_{key}_cited"] = bar_cited

    summary["deliverables"] = flat
    summary["peak_gpu_gb"] = args.peak_gpu_gb

    (RES / "verdict_marker.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.wandb:
        import wandb
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT"),
            name="land/base-quality-denominator-grounding",
            group="base-quality-denominator-grounding",
            job_type="analysis",
            config={
                "pr": 581, "analysis_only": True, "official_tps": 0,
                "model": "google/gemma-4-E4B-it (bf16 unquantized base)",
                "engine": "vllm-0.22.1rc1.dev307",
                "decode_protocol": summary["decode_protocol"],
                "gate_rel": GATE_REL, "cited_anchors": CITED,
                "gsm8k_n": 500, "gsm8k_n_shot": 8,
                "mmlupro_gpqa_source": "PR#563 fp16 vanilla-base banked (mandated sampling, mt8 no-op proven)",
            },
        )
        wb = dict(flat)
        wb["analysis_only"] = True
        wb["official_tps"] = 0
        for key in tasks:
            rec = summary["per_task"][key]
            wb[f"{key}_cited"] = rec["cited"]
            wb[f"{key}_measured_ci95"] = rec["measured_ci95"]
            wb[f"{key}_rel_diff"] = rec["rel_diff_measured_vs_cited"]
            wb[f"{key}_within_ci95"] = rec["cited_within_ci95"]
            wb[f"{key}_greedy_anchor"] = rec["greedy_anchor"]
            wb[f"{key}_gate_bar_90_shift"] = rec["gate_bar_90_shift"]
        if args.peak_gpu_gb is not None:
            wb["peak_gpu_gb"] = args.peak_gpu_gb
        wb["gsm8k_mt8_noop_delta_s1234"] = summary["per_task"]["gsm8k"].get("mt8_noop_delta_s1234")
        run.summary.update(wb)
        wandb.log({k: v for k, v in wb.items() if isinstance(v, (int, float, bool))})
        print(f"\n[wandb] run id: {run.id} name: {run.name}")
        run.finish()
        (RES / "_wandb_run_id.txt").write_text(run.id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
