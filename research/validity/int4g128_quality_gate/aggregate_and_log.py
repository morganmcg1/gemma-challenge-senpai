#!/usr/bin/env python3
"""PR #579 aggregator: int4_g128_lmhead downstream quality vs the corrected
>=90%-of-harness-base gate, across the 4-eval panel (AIME / MMLU-Pro / GPQA /
GSM8K). Emits the advisor's KEY OUTPUTS to stdout (SENPAI-RESULT-ready) and to
W&B group ``int4g128lmhead-characterization``.

CORRECTED gate bars (Morgan #579 11:59Z, grounded in #580 ubel AIME + #581 land
GPQA): MMLU-Pro >=0.605, GPQA-Diamond >=0.471, AIME >=0.090, GSM8K >=0.807.
Each is 0.90 x the harness-base reference (0.6727 / 0.5236 / 0.100 / 0.8967).

Protocol (PR #579, min_tokens=8 EOS-guard throughout):
  * AIME (n=60): greedy maj@1, #567 harness (aime_eval.py), --base-url.
  * MMLU-Pro / GPQA / GSM8K: native generation_config sampling (T=1.0, top_p=0.95,
    top_k=64), #563 harness for MMLU/GPQA (run_eval.py, 3 seeds + greedy anchor),
    #533 harness for GSM8K (gsm8k_eval.py, n=500).

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
RES = HERE / "results"

# --- CORRECTED >=90%-of-harness-base gate bars (Morgan #579 11:59Z) ---------- #
GATE = {
    "mmlu_pro": {"bar": 0.605, "base": 0.6727},
    "gpqa":     {"bar": 0.471, "base": 0.5236},
    "aime":     {"bar": 0.090, "base": 0.100},
    "gsm8k":    {"bar": 0.807, "base": 0.8967},
}
GATE_FRAC = 0.90

_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
      7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t975(dof: int) -> float:
    return float("nan") if dof <= 0 else _T.get(dof, 1.96)


def _load(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def _mean_ci(vals: list[float]):
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return {"mean": m, "n": 1, "sd": 0.0, "ci95": None, "lo": m, "hi": m, "vals": vals}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    half = _t975(n - 1) * sd / math.sqrt(n)
    return {"mean": m, "n": n, "sd": sd, "ci95": half, "lo": m - half, "hi": m + half, "vals": vals}


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mc_task(task_file: str, seeds=(0, 1, 2)):
    """MMLU-Pro / GPQA: sampled mean over seeds + greedy anchor (run_eval.py out)."""
    s_vals, per_seed = [], {}
    for s in seeds:
        d = _load(RES / f"int4g128_{task_file}_sampled_s{s}.json")
        if d and d.get("accuracy") is not None and not math.isnan(d["accuracy"]):
            s_vals.append(d["accuracy"])
            per_seed[s] = {"accuracy": d["accuracy"], "n_scored": d.get("n_scored"),
                           "empty_rate": d.get("empty_rate")}
    g = _load(RES / f"int4g128_{task_file}_greedy.json")
    return {
        "sampled": _mean_ci(s_vals),
        "per_seed": per_seed,
        "greedy": (g.get("accuracy") if g else None),
        "greedy_n_scored": (g.get("n_scored") if g else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--peak-gpu-gb", type=float, default=None)
    ap.add_argument("--engine", default="vllm-0.22.1rc1.dev307")
    args = ap.parse_args()

    # ---- per-task point estimates (the gate figures) ----
    mmlu = mc_task("mmlu_pro")
    gpqa = mc_task("gpqa_diamond")
    gsm = _load(HERE / "int4g128_sampled.json")            # gsm8k_eval.py out
    aime = _load(HERE / "aime_int4g128_min8_n60.json")     # aime_eval.py out

    mmlu_pt = mmlu["sampled"]["mean"] if mmlu["sampled"] else None
    gpqa_pt = gpqa["sampled"]["mean"] if gpqa["sampled"] else None
    gsm_pt = gsm.get("accuracy") if gsm else None
    aime_pt = aime.get("maj_k_accuracy") if aime else None

    points = {"mmlu_pro": mmlu_pt, "gpqa": gpqa_pt, "aime": aime_pt, "gsm8k": gsm_pt}

    # ---- gate adjudication vs CORRECTED bars ----
    passes, margins = {}, {}
    for k, pt in points.items():
        bar = GATE[k]["bar"]
        passes[k] = (None if pt is None else bool(pt >= bar))
        margins[k] = (None if pt is None else (pt - bar))
    measured = [k for k, p in passes.items() if p is not None]
    pass_count = sum(1 for k in measured if passes[k])

    order = ["mmlu_pro", "gpqa", "aime", "gsm8k"]
    pretty = {"mmlu_pro": "MMLU-Pro", "gpqa": "GPQA", "aime": "AIME", "gsm8k": "GSM8K"}
    parts = []
    for k in order:
        if passes[k] is None:
            parts.append(f"{pretty[k]}=NA")
        else:
            v = points[k]
            parts.append(f"{pretty[k]}={v:.4f}{'>=' if passes[k] else '<'}{GATE[k]['bar']:.3f} "
                         f"{'PASS' if passes[k] else 'FAIL'}")
    verdict = f"{pass_count}/{len(measured)} PASS | " + "; ".join(parts)

    # ---- AIME / GSM8K diagnostics ----
    aime_extra, gsm_extra = {}, {}
    if aime:
        nc, n = aime.get("n_correct_maj"), aime.get("n_problems")
        lo, hi = (wilson(nc, n) if (nc is not None and n) else (float("nan"), float("nan")))
        n_empty = sum(1 for p in aime.get("per_problem", [])
                      for t in (p.get("texts") or []) if not str(t).strip())
        aime_extra = {"aime_n_problems": n, "aime_n_correct": nc,
                      "aime_wilson95_lo": lo, "aime_wilson95_hi": hi,
                      "aime_extract_fail_rate": aime.get("extract_fail_rate"),
                      "aime_n_empty": n_empty, "aime_mean_pass_rate": aime.get("mean_pass_rate"),
                      "aime_max_tokens": (aime.get("sampling") or {}).get("max_tokens"),
                      "aime_min_tokens": (aime.get("sampling") or {}).get("min_tokens"),
                      "aime_years": ",".join(aime.get("years", []))}
    if gsm:
        gsm_extra = {"gsm8k_n_problems": gsm.get("n_problems"), "gsm8k_n_correct": gsm.get("n_correct"),
                     "gsm8k_strict_rate": gsm.get("strict_rate"),
                     "gsm8k_extract_fail_rate": gsm.get("extract_fail_rate"),
                     "gsm8k_truncation_rate": gsm.get("truncation_rate"),
                     "gsm8k_min_tokens": (gsm.get("sampling") or {}).get("min_tokens"),
                     "gsm8k_max_tokens": (gsm.get("sampling") or {}).get("max_tokens")}

    # ---- KEY OUTPUTS (advisor's exact deliverable names) ----
    ko = {
        "int4g128lmhead_mmlupro": mmlu_pt,
        "int4g128lmhead_gpqa": gpqa_pt,
        "int4g128lmhead_aime": aime_pt,
        "int4g128lmhead_gsm8k": gsm_pt,
        "int4g128lmhead_gate_pass_count": pass_count,
        "int4g128lmhead_gate_verdict": verdict,
        "analysis_only": True,
        "official_tps": 0,
    }
    detail = {
        **ko,
        "gate_bars_corrected": {k: GATE[k]["bar"] for k in GATE},
        "gate_base_refs": {k: GATE[k]["base"] for k in GATE},
        "gate_frac": GATE_FRAC,
        "passes": passes,
        "margins": margins,
        "mmlu_pro_sampled_ci95": (mmlu["sampled"] or {}).get("ci95"),
        "mmlu_pro_sampled_seeds": (mmlu["sampled"] or {}).get("vals"),
        "mmlu_pro_greedy": mmlu["greedy"],
        "gpqa_sampled_ci95": (gpqa["sampled"] or {}).get("ci95"),
        "gpqa_sampled_seeds": (gpqa["sampled"] or {}).get("vals"),
        "gpqa_greedy": gpqa["greedy"],
        "engine": args.engine,
        "checkpoint": "/workspace/gemma_build/int4_g128_lmhead",
        "decode_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
                            "source": "gemma-4-E4B-it/generation_config.json (lewtun #31)"},
        "min_tokens_guard": 8,
        "peak_gpu_gb": args.peak_gpu_gb,
        **aime_extra, **gsm_extra,
    }

    print("AGG579_SUMMARY " + json.dumps(detail, default=str))
    (HERE / "agg579_summary.json").write_text(json.dumps(detail, indent=2, default=str))

    if args.wandb:
        run = wandb_logging.init_wandb_run(
            job_type="int4g128lmhead-characterization",
            agent="kanna",
            name="kanna/int4g128lmhead-quality-gate",
            group="int4g128lmhead-characterization",
            notes=("PR #579: int4_g128_lmhead downstream quality vs the corrected "
                   ">=90%-of-harness-base gate (4 evals). AIME n=60 greedy maj@1 (#567); "
                   "MMLU-Pro/GPQA/GSM8K native-gen-config sampling, min_tokens=8 throughout. "
                   "LOCAL served on dev307; analysis_only, official_tps=0, NO FIRE."),
            tags=["quality-gate", "int4_g128_lmhead", "analysis-only", "pr-579",
                  "mmlu-pro", "gpqa", "aime", "gsm8k", "4-eval-panel"],
            config={"pr": 579, "analysis_only": True, "official_tps": 0,
                    "engine": args.engine,
                    "gate_bars_corrected": {k: GATE[k]["bar"] for k in GATE},
                    "gate_base_refs": {k: GATE[k]["base"] for k in GATE},
                    "checkpoint": "int4_g128_lmhead", "min_tokens_guard": 8},
        )
        if run is None:
            print("[agg] wandb disabled/unavailable; JSON only", flush=True)
            return 0
        # exact deliverable names at top level of summary
        for k, v in ko.items():
            run.summary[k] = v
        for k, v in {**margins}.items():
            run.summary[f"margin/{k}"] = v
        for k, v in passes.items():
            if v is not None:
                run.summary[f"pass/{k}"] = int(v)
        wandb_logging.log_summary(run, detail, step=0)
        for nm, obj in (("aime_int4g128_min8_n60", aime), ("gsm8k_int4g128_sampled", gsm)):
            if obj:
                slim = {kk: vv for kk, vv in obj.items() if kk != "per_problem"}
                slim["per_problem_no_texts"] = [
                    {a: b for a, b in p.items() if a not in ("texts", "text")}
                    for p in obj.get("per_problem", [])
                ]
                wandb_logging.log_json_artifact(run, name=nm, artifact_type="quality-eval", data=slim)
        run_id = getattr(run, "id", None)
        wandb_logging.finish_wandb(run)
        print(f"[agg] wandb run_id={run_id}", flush=True)
        (HERE / "agg579_summary.json").write_text(
            json.dumps({**detail, "wandb_run_id": run_id}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
