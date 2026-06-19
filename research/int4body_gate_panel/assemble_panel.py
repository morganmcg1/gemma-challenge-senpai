#!/usr/bin/env python3
"""PR #703 -- assemble the unified four-metric #31-basis #515 gate panel for the
int4_g128_lmhead checkpoint and decide AIME_SOLE_BLOCKER vs MULTI_LEG_FAIL.

Two legs are MEASURED here (GSM8K guarded + as-served, MMLU-Pro debiased) on the
exact #31 sampling basis (T=1.0 top_p=0.95 top_k=64, min_tokens=8 where guarded,
5 sampling-seeds, cluster-bootstrap CI). Two legs are CARRIED from #693 (6brpvz9x,
the banked #31 AIME/GPQA gate basis). Each leg is scored against its own
90%-of-vanilla-bf16-base bar; bases for GSM8K/MMLU come from the #590 canonical CI
(base_fullhead at the identical basis).

Writes panel.json and (unless --no-wandb) logs the panel + headline scalars to
W&B group int4body-gate-panel-completion-lawine. Run under .venv/bin/python so
wandb imports (serve venv has none).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

WD = Path("/workspace/senpai/target/research/int4body_gate_panel")
RUNS = WD / "runs"
SUMM = WD / "summaries"

# ---- carried #693 rows (banked #31 gate basis; run 6brpvz9x) ----------------
AIME_693 = {
    "metric": "AIME-2024 maj@8 (#31 sampled, n=60, 12288 tok)",
    "point": 0.3467, "ci_lo": None, "ci_hi": 0.4022,
    "base": 0.4600, "bar": 0.420, "source": "#693 6brpvz9x (banked 1b00c31, ubel #650/#672)",
    "candidate": "fail",  # point < bar -> use Wilson-hi as the clearing test
}
GPQA_693 = {
    "metric": "GPQA-Diamond (#31 sampled, n=198)",
    "point": 0.4747, "ci_lo": 0.4063, "ci_hi": 0.5441,
    "base": None, "bar": 0.471, "source": "#693 6brpvz9x (sampled_31)",
    "candidate": "pass",  # point >= bar -> use Wilson/CI-lo as the clearing test
}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return center - half, center + half


def gsm8k_arm(label: str) -> dict:
    """Aggregate the 5 seed files for one GSM8K arm: mean acc, empty_rate, per-seed."""
    files = sorted(RUNS.glob(f"{label}_sampled_s*.json"))
    if not files:
        return {}
    per_seed_acc, empties, totals = [], 0, 0
    n_q = None
    for f in files:
        d = json.loads(f.read_text())
        per_seed_acc.append(d["accuracy"])
        pp = d["per_problem"]
        n_q = len(pp)
        empties += sum(1 for r in pp if r.get("sample_chars", 1) == 0)
        totals += len(pp)
    mean_acc = sum(per_seed_acc) / len(per_seed_acc)
    return {
        "label": label, "n_seeds": len(files), "n_questions": n_q,
        "per_seed_accuracy": [round(a, 6) for a in per_seed_acc],
        "mean_accuracy": mean_acc,
        "empty_rate": empties / totals if totals else float("nan"),
        "n_empty": empties, "n_total": totals,
    }


def load_ci_summary(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {
        "mean_accuracy": d["mean_accuracy"],
        "ci_lo": d["ci_lb_95_2sided"], "ci_hi": d["ci_ub_95_2sided"],
        "bar": d["bar"], "per_seed_accuracy": d["per_seed_accuracy"],
        "n_questions": d["n_questions"], "n_seeds": d["n_seeds_samples_per_q"],
    }


def leg_call(point: float, ci_lo, ci_hi, bar: float, candidate: str) -> dict:
    """point-vs-bar headline + CI clearing test (PR #703 protocol)."""
    point_clears = point >= bar
    if candidate == "fail":
        # fail-candidate: does Wilson-hi clear the bar? if hi < bar -> confirmed FAIL.
        ci_clears = (ci_hi is not None) and (ci_hi >= bar)
    else:
        # pass-candidate: does CI-lo clear the bar? if lo >= bar -> clean PASS.
        ci_clears = (ci_lo is not None) and (ci_lo >= bar)
    return {
        "point_clears_bar": bool(point_clears),
        "ci_clears_bar": bool(ci_clears),
        # leg "fails" the gate when the point estimate is below the 90% bar
        "fails_gate": bool(point < bar),
        "clean_pass": bool(point_clears and ci_clears),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- GSM8K (measured) ----
    g_guard_raw = gsm8k_arm("int4g128_guard")
    g_noguard_raw = gsm8k_arm("int4g128_noguard")
    gsm_ci = load_ci_summary(SUMM / "gsm8k_guarded.json")
    GSM_BASE, GSM_BAR = 0.8952, gsm_ci["bar"]
    gsm_leg = {
        "metric": "GSM8K 8-shot (#31 sampled, n=500, guarded min_tokens=8)",
        "point": gsm_ci["mean_accuracy"], "ci_lo": gsm_ci["ci_lo"], "ci_hi": gsm_ci["ci_hi"],
        "base": GSM_BASE, "bar": GSM_BAR,
        "source": "this card (int4); base=#590 gsm8k_sampled.json",
        "candidate": "pass" if gsm_ci["mean_accuracy"] >= GSM_BAR else "fail",
        "empty_rate_guarded": g_guard_raw.get("empty_rate"),
        "empty_rate_asserved": g_noguard_raw.get("empty_rate"),
        "asserved_mean_accuracy": g_noguard_raw.get("mean_accuracy"),
        "asserved_vs_guarded_delta": (
            g_noguard_raw.get("mean_accuracy", float("nan")) - gsm_ci["mean_accuracy"]
        ),
        "per_seed_accuracy": gsm_ci["per_seed_accuracy"],
    }

    # ---- MMLU-Pro (measured, debiased) ----
    mmlu_ci = load_ci_summary(SUMM / "mmlu_debiased.json")
    MMLU_BASE, MMLU_BAR = 0.6695, mmlu_ci["bar"]
    mmlu_leg = {
        "metric": "MMLU-Pro (#31 sampled, n=2000, debiased@4096)",
        "point": mmlu_ci["mean_accuracy"], "ci_lo": mmlu_ci["ci_lo"], "ci_hi": mmlu_ci["ci_hi"],
        "base": MMLU_BASE, "bar": MMLU_BAR,
        "source": "this card (int4); base=#590 mmlu_debias.json",
        "candidate": "pass" if mmlu_ci["mean_accuracy"] >= MMLU_BAR else "fail",
        "per_seed_accuracy": mmlu_ci["per_seed_accuracy"],
    }

    # ---- assemble panel ----
    panel = []
    for leg in (AIME_693, GPQA_693, gsm_leg, mmlu_leg):
        call = leg_call(leg["point"], leg.get("ci_lo"), leg.get("ci_hi"),
                        leg["bar"], leg["candidate"])
        panel.append({**leg, **call})

    legs_failing = sum(1 for p in panel if p["fails_gate"])
    # GPQA marginal-tie: point over the gate but CI-lo under -> a pass for the
    # legs-failing count (per the #703 framing), flagged as not-clean.
    gpqa = next(p for p in panel if p["metric"].startswith("GPQA"))
    aime = next(p for p in panel if p["metric"].startswith("AIME"))

    if legs_failing <= 1 and aime["fails_gate"]:
        verdict = "AIME_SOLE_BLOCKER"
    elif legs_failing >= 2:
        verdict = "MULTI_LEG_FAIL"
    else:
        verdict = "NO_LEG_FAIL"  # defensive; shouldn't happen (AIME fails)

    gsm8k_serving_artifact = bool(
        gsm_leg["empty_rate_asserved"] is not None
        and gsm_leg["empty_rate_guarded"] is not None
        and gsm_leg["empty_rate_asserved"] > 0.02
        and gsm_leg["empty_rate_guarded"] <= 0.01
        and (gsm_leg["asserved_mean_accuracy"] < gsm_leg["bar"] <= gsm_leg["point"])
    )

    out = {
        "pr": 703, "verdict": verdict,
        "gate_panel_legs_failing": legs_failing,
        "gsm8k_guarded_compliant": gsm_leg["point"],
        "gsm8k_asserved": gsm_leg["asserved_mean_accuracy"],
        "gsm8k_empty_rate_guarded": gsm_leg["empty_rate_guarded"],
        "gsm8k_empty_rate_asserved": gsm_leg["empty_rate_asserved"],
        "gsm8k_asserved_vs_guarded_delta": gsm_leg["asserved_vs_guarded_delta"],
        "gsm8k_serving_artifact_confirmed": gsm8k_serving_artifact,
        "mmlu_pro_debiased": mmlu_leg["point"],
        "gpqa_marginal_tie": bool(gpqa["point_clears_bar"] and not gpqa["ci_clears_bar"]),
        "panel": panel,
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": False,
        "model": "int4_g128_lmhead (submissions/int4_g128_lmhead/model)",
        "serve_stack": "vllm==0.22.0 max_model_len=6144 max_num_batched_tokens=512 "
                       "flashinfer_sampler=OFF (replicates #693)",
    }
    (WD / "panel.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== PR #703 four-metric #31-basis gate panel ===")
    print(f"{'leg':<46}{'point':>8}{'CI-lo':>8}{'CI-hi':>8}{'bar':>8}  call")
    for p in panel:
        lo = f'{p["ci_lo"]:.4f}' if p["ci_lo"] is not None else '   -  '
        hi = f'{p["ci_hi"]:.4f}' if p["ci_hi"] is not None else '   -  '
        call = "FAIL" if p["fails_gate"] else ("PASS*" if not p["ci_clears_bar"] else "PASS")
        print(f'{p["metric"]:<46}{p["point"]:>8.4f}{lo:>8}{hi:>8}{p["bar"]:>8.3f}  {call}')
    print(f"\nlegs_failing={legs_failing}  verdict={verdict}")
    print(f"GSM8K guarded={gsm_leg['point']:.4f} as-served={gsm_leg['asserved_mean_accuracy']:.4f} "
          f"empty(guard/served)={gsm_leg['empty_rate_guarded']:.4f}/{gsm_leg['empty_rate_asserved']:.4f} "
          f"serving_artifact={gsm8k_serving_artifact}")

    if args.no_wandb:
        return 0
    try:
        import wandb
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] import failed ({e}); panel on disk at panel.json")
        return 0
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name="lawine/int4body-gate-panel-completion",
            group="int4body-gate-panel-completion-lawine",
            job_type="analysis",
            config={"pr": 703, "analysis_only": True, "official_tps": 0,
                    "no_hf_job": 1, "fires": False, "model": out["model"],
                    "serve_stack": out["serve_stack"]},
        )
        cols = ["leg", "point", "ci_lo", "ci_hi", "base", "bar",
                "point_clears_bar", "ci_clears_bar", "fails_gate", "source"]
        tbl = wandb.Table(columns=cols)
        for p in panel:
            tbl.add_data(p["metric"], p["point"], p.get("ci_lo"), p.get("ci_hi"),
                         p.get("base"), p["bar"], p["point_clears_bar"],
                         p["ci_clears_bar"], p["fails_gate"], p["source"])
            pfx = p["metric"].split()[0].replace("-", "_").lower()
            run.summary[f"leg__{pfx}__point"] = p["point"]
            run.summary[f"leg__{pfx}__ci_lo"] = p.get("ci_lo")
            run.summary[f"leg__{pfx}__ci_hi"] = p.get("ci_hi")
            run.summary[f"leg__{pfx}__bar"] = p["bar"]
            run.summary[f"leg__{pfx}__fails_gate"] = p["fails_gate"]
        run.log({"gate_panel": tbl})
        s = run.summary
        for k in ("verdict", "gate_panel_legs_failing", "gsm8k_guarded_compliant",
                  "gsm8k_asserved", "gsm8k_empty_rate_guarded", "gsm8k_empty_rate_asserved",
                  "gsm8k_asserved_vs_guarded_delta", "gsm8k_serving_artifact_confirmed",
                  "mmlu_pro_debiased", "gpqa_marginal_tie", "analysis_only",
                  "official_tps", "no_hf_job", "fires"):
            s[k] = out[k]
        # explicit decode-basis triples per measured arm (as in #693)
        s["gsm8k_guarded__eval_decode_basis"] = "generation_config_sampling"
        s["gsm8k_guarded__eval_sampling"] = "T=1.0,top_p=0.95,top_k=64"
        s["gsm8k_guarded__eval_min_tokens"] = 8
        s["gsm8k_asserved__eval_decode_basis"] = "generation_config_sampling"
        s["gsm8k_asserved__eval_sampling"] = "T=1.0,top_p=0.95,top_k=64"
        s["gsm8k_asserved__eval_min_tokens"] = 0
        s["mmlu_pro__eval_decode_basis"] = "generation_config_sampling"
        s["mmlu_pro__eval_sampling"] = "T=1.0,top_p=0.95,top_k=64"
        s["mmlu_pro__eval_min_tokens"] = 8
        run.finish()
        print(f"[wandb] logged run id={run.id}")
        (WD / "wandb_run_id.txt").write_text(run.id + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] logging failed ({e}); panel on disk at panel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
