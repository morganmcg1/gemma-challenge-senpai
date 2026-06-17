#!/usr/bin/env python3
"""PR #563 Stage 3: compile the base_fullhead-quality-under-sampling A/B into the
gate verdict (+ optional W&B).

Reads results/<arm>_<task>_<decode>[_s<seed>].json from run_arm.sh and computes,
per task in {mmlu_pro, gpqa}:
  - base_fullhead (int4) sampled mean +- 95% CI over seeds, and greedy point
  - vanilla (fp16) sampled mean +- 95% CI over seeds, and greedy point
  - sampled_ratio = int4_sampled / fp16_sampled  (+ propagated 95% CI)
  - greedy_ratio  = int4_greedy  / fp16_greedy
  - greedy_vs_sampled_ratio_delta = sampled_ratio - greedy_ratio
  - gate_holds (point >= 0.90) and gate_holds_cilb (ratio 95% lower bound >= 0.90)

If the fp16 vanilla arm is degenerate (documented dev307 long-CoT serve regression,
stark #542/#557), its fresh ratio is INVALID; we additionally report the
protocol-mixed ratio against the documented ubel #511 greedy anchor (0.668/0.470),
clearly flagged. analysis_only; official_tps=0.

Usage: compile.py [--wandb]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# Documented ubel #511 greedy vanilla-base anchor (the gate denominator the program
# bound to after stark #542/#557 proved a fresh fp16 serve craters on dev307).
ANCHOR_GREEDY = {"mmlu_pro": 0.668, "gpqa": 0.470}
# kanna #547 greedy base_fullhead (int4 full head) on this exact harness/engine.
K547_GREEDY = {"mmlu_pro": 0.676, "gpqa": 0.4697}
GATE_REL = 0.90
# Degenerate-serve guard: an fp16 vanilla arm scoring below this on MMLU is the
# documented long-CoT attention-collapse regression, not a valid denominator.
DEGEN_MMLU = 0.50

# t(0.975, dof) for small samples; fall back to z=1.96 for dof>=30.
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
      7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _t975(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    return _T.get(dof, 1.96)


def _load(name: str):
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _mean_ci(vals: list[float]):
    """mean and 95% CI half-width (t) over a small list of accuracies."""
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    if n == 1:
        return {"mean": m, "n": 1, "sd": 0.0, "ci95": None, "lo": m, "hi": m, "vals": vals}
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    half = _t975(n - 1) * sd / math.sqrt(n)
    return {"mean": m, "n": n, "sd": sd, "ci95": half, "lo": m - half, "hi": m + half, "vals": vals}


def _sampled(arm: str, task_key: str, task_file: str, seeds=(0, 1, 2)):
    vals = []
    for s in seeds:
        d = _load(f"{arm}_{task_file}_sampled_s{s}.json")
        if d and d.get("accuracy") is not None and not math.isnan(d["accuracy"]):
            vals.append(d["accuracy"])
    return _mean_ci(vals)


def _greedy(arm: str, task_file: str):
    d = _load(f"{arm}_{task_file}_greedy.json")
    return None if d is None else d.get("accuracy")


def _ratio_ci(num, den):
    """ratio of two _mean_ci dicts with first-order error propagation."""
    if num is None or den is None or not den["mean"]:
        return None
    r = num["mean"] / den["mean"]
    # relative SEs (half-CI already includes the t-multiplier; convert back to SE-ish
    # by dividing by the per-arm t, then recombine with the larger dof's t). Keep it
    # simple+honest: propagate the relative half-CIs in quadrature and re-scale by r.
    rn = (num["ci95"] / num["mean"]) if num.get("ci95") and num["mean"] else 0.0
    rd = (den["ci95"] / den["mean"]) if den.get("ci95") and den["mean"] else 0.0
    half = r * math.sqrt(rn * rn + rd * rd)
    return {"ratio": r, "ci95": half, "lo": r - half, "hi": r + half}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--self-det", type=int, default=None,
                    help="1 if greedy decode was self-deterministic on this stack (smoke_sampling).")
    ap.add_argument("--peak-gpu-gb", type=float, default=None,
                    help="peak GPU memory in GiB during the runs.")
    args = ap.parse_args()

    TASKS = [("mmlu_pro", "mmlu_pro"), ("gpqa", "gpqa")]
    out: dict = {"pr": 563, "analysis_only": True, "official_tps": 0,
                 "engine": "vllm-0.22.1rc1.dev307", "decode_protocol": {
                     "do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
                     "source": "gemma-4-E4B-it/generation_config.json (lewtun #31)"}}
    per_task: dict = {}

    for task_key, task_file in TASKS:
        i_s = _sampled("int4", task_key, task_file)
        f_s = _sampled("fp16", task_key, task_file)
        i_g = _greedy("int4", task_file)
        f_g = _greedy("fp16", task_file)

        fp16_degen = (f_s is not None and f_s["mean"] < DEGEN_MMLU and task_key == "mmlu_pro") or \
                     (f_g is not None and task_key == "mmlu_pro" and f_g < DEGEN_MMLU)

        sampled_ratio = _ratio_ci(i_s, f_s)
        greedy_ratio = (i_g / f_g) if (i_g is not None and f_g) else None
        # protocol-mixed fallback: int4 sampled / documented greedy anchor
        anchor = ANCHOR_GREEDY[task_key]
        sampled_ratio_vs_anchor = (i_s["mean"] / anchor) if (i_s and anchor) else None
        # self-comparison: int4 sampled vs int4 greedy (#547) absolute delta
        self_abs_delta = (i_s["mean"] - (i_g if i_g is not None else K547_GREEDY[task_key])) if i_s else None

        gvs = None
        if sampled_ratio and greedy_ratio is not None:
            gvs = sampled_ratio["ratio"] - greedy_ratio

        per_task[task_key] = {
            "int4_sampled": i_s, "fp16_sampled": f_s,
            "int4_greedy": i_g, "fp16_greedy": f_g,
            "fp16_arm_degenerate": fp16_degen,
            "sampled_ratio_fresh": sampled_ratio,
            "greedy_ratio_fresh": greedy_ratio,
            "sampled_ratio_vs_documented_anchor": sampled_ratio_vs_anchor,
            "documented_anchor_greedy": anchor,
            "int4_sampled_minus_greedy_abs": self_abs_delta,
            "greedy_vs_sampled_ratio_delta": gvs,
        }

    def _choose_ratio(tk):
        """Prefer the fresh sampled ratio; if fp16 degenerate, use the protocol-mixed
        anchor ratio and flag it."""
        c = per_task[tk]
        if c["sampled_ratio_fresh"] and not c["fp16_arm_degenerate"]:
            return c["sampled_ratio_fresh"]["ratio"], c["sampled_ratio_fresh"].get("lo"), "fresh_both_arms_sampled"
        if c["sampled_ratio_vs_documented_anchor"] is not None:
            return c["sampled_ratio_vs_documented_anchor"], None, "protocol_mixed_vs_documented_greedy_anchor"
        return None, None, "unavailable"

    mmlu_ratio, mmlu_lo, mmlu_basis = _choose_ratio("mmlu_pro")
    gpqa_ratio, gpqa_lo, gpqa_basis = _choose_ratio("gpqa")

    def _holds(r, lo):
        if r is None:
            return None
        return bool(r >= GATE_REL)

    gate_point = None
    if mmlu_ratio is not None and gpqa_ratio is not None:
        gate_point = bool(mmlu_ratio >= GATE_REL and gpqa_ratio >= GATE_REL)
    gate_cilb = None
    if mmlu_lo is not None and gpqa_lo is not None:
        gate_cilb = bool(mmlu_lo >= GATE_REL and gpqa_lo >= GATE_REL)

    out["per_task"] = per_task
    out["verdict"] = {
        "mmlu_pro_sampled_ratio": mmlu_ratio, "mmlu_pro_ratio_basis": mmlu_basis,
        "gpqa_sampled_ratio": gpqa_ratio, "gpqa_ratio_basis": gpqa_basis,
        "gate_rel": GATE_REL,
        "gate_holds_under_sampling": gate_point,
        "gate_holds_under_sampling_cilb": gate_cilb,
        "mmlu_pro_greedy_vs_sampled_ratio_delta": per_task["mmlu_pro"]["greedy_vs_sampled_ratio_delta"],
        "gpqa_greedy_vs_sampled_ratio_delta": per_task["gpqa"]["greedy_vs_sampled_ratio_delta"],
    }

    def _mc(d):
        return None if not d else {"mean": d["mean"], "ci95": d.get("ci95"), "n": d["n"]}

    # KEY OUTPUTS in the PR's exact names (single source of truth for the report
    # comment + W&B summary). gen_config_* are the Stage-1 lewtun #31 protocol.
    ko = {
        "gen_config_do_sample": True,
        "gen_config_temperature": 1.0,
        "gen_config_top_p": 0.95,
        "gen_config_top_k": 64,
        "gen_config_is_greedy": False,
        "mmlu_pro_base_fullhead_sampled": _mc(per_task["mmlu_pro"]["int4_sampled"]),
        "mmlu_pro_vanilla_sampled": _mc(per_task["mmlu_pro"]["fp16_sampled"]),
        "gpqa_base_fullhead_sampled": _mc(per_task["gpqa"]["int4_sampled"]),
        "gpqa_vanilla_sampled": _mc(per_task["gpqa"]["fp16_sampled"]),
        "mmlu_pro_sampled_ratio": mmlu_ratio,
        "mmlu_pro_sampled_ratio_basis": mmlu_basis,
        "gpqa_sampled_ratio": gpqa_ratio,
        "gpqa_sampled_ratio_basis": gpqa_basis,
        "gate_holds_under_sampling": gate_point,
        "gate_holds_under_sampling_cilb": gate_cilb,
        "mmlu_pro_greedy_vs_sampled_ratio_delta": per_task["mmlu_pro"]["greedy_vs_sampled_ratio_delta"],
        "gpqa_greedy_vs_sampled_ratio_delta": per_task["gpqa"]["greedy_vs_sampled_ratio_delta"],
        "primary_metric_name": "mmlu_pro_sampled_ratio",
        "primary_metric": mmlu_ratio,
        "self_det": (None if args.self_det is None else bool(args.self_det)),
        "peak_gpu_gb": args.peak_gpu_gb,
    }
    out["key_outputs"] = ko

    (RES / "verdict_marker.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))

    if args.wandb:
        import wandb
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            group="base-fullhead-quality-sampling",
            name="kanna/base-fullhead-quality-sampling",
            job_type="analysis",
            config={"pr": 563, "analysis_only": True, "official_tps": 0,
                    "engine": "vllm-0.22.1rc1.dev307",
                    "decode_protocol": out["decode_protocol"], "gate_rel": GATE_REL},
        )
        flat = {}
        for tk in ("mmlu_pro", "gpqa"):
            c = per_task[tk]
            for arm in ("int4", "fp16"):
                s = c[f"{arm}_sampled"]
                if s:
                    flat[f"{tk}/{arm}_sampled_mean"] = s["mean"]
                    if s.get("ci95") is not None:
                        flat[f"{tk}/{arm}_sampled_ci95"] = s["ci95"]
                g = c[f"{arm}_greedy"]
                if g is not None:
                    flat[f"{tk}/{arm}_greedy"] = g
            if c["sampled_ratio_fresh"]:
                flat[f"{tk}/sampled_ratio_fresh"] = c["sampled_ratio_fresh"]["ratio"]
            if c["greedy_ratio_fresh"] is not None:
                flat[f"{tk}/greedy_ratio_fresh"] = c["greedy_ratio_fresh"]
            if c["sampled_ratio_vs_documented_anchor"] is not None:
                flat[f"{tk}/sampled_ratio_vs_anchor"] = c["sampled_ratio_vs_documented_anchor"]
            if c["greedy_vs_sampled_ratio_delta"] is not None:
                flat[f"{tk}/greedy_vs_sampled_ratio_delta"] = c["greedy_vs_sampled_ratio_delta"]
            flat[f"{tk}/fp16_arm_degenerate"] = int(bool(c["fp16_arm_degenerate"]))
        v = out["verdict"]
        if v["mmlu_pro_sampled_ratio"] is not None:
            flat["verdict/mmlu_pro_sampled_ratio"] = v["mmlu_pro_sampled_ratio"]
        if v["gpqa_sampled_ratio"] is not None:
            flat["verdict/gpqa_sampled_ratio"] = v["gpqa_sampled_ratio"]
        if v["gate_holds_under_sampling"] is not None:
            flat["verdict/gate_holds_under_sampling"] = int(v["gate_holds_under_sampling"])
        if v["gate_holds_under_sampling_cilb"] is not None:
            flat["verdict/gate_holds_under_sampling_cilb"] = int(v["gate_holds_under_sampling_cilb"])
        # PR-exact KEY OUTPUT names under key/* so a reviewer finds them directly.
        for k, val in ko.items():
            if isinstance(val, bool):
                flat[f"key/{k}"] = int(val)
            elif isinstance(val, (int, float)):
                flat[f"key/{k}"] = val
            elif isinstance(val, dict) and val.get("mean") is not None:
                flat[f"key/{k}_mean"] = val["mean"]
                if val.get("ci95") is not None:
                    flat[f"key/{k}_ci95"] = val["ci95"]
        wandb.log(flat)
        wandb.summary.update(flat)
        print(f"[wandb] logged run {run.id} ({run.url})")
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
