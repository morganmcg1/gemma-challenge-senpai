#!/usr/bin/env python3
"""PR #814 Step-3 aggregator: g128 body-group-size fire-prep quality panel.

Reads the g128 arm files produced by run_panel_g128.sh + run_aime_g128.sh
against the local g128 server on :8021, computes the fire-gate verdicts vs the
Morgan quality floors, and logs one pooled run to W&B group
``bi0-int4head-body-groupsize`` (same group as the speed A/B).

g128 = int4 W4A16 body re-quantized at group_size=128 + int4 g32 lm_head + MTP
K=6 + surgattn force-2D, BI=0. The ONLY delta vs the int4head fire candidate
(#795) is the body group_size (g32 -> g128). So every axis is reported BOTH vs
the Morgan floor (the gate) AND vs the seed-paired #795 int4head reference (the
clean A/B isolating body group-size coarsening on downstream quality).

Gates (Morgan floors, same as #795):
  GPQA-Diamond pooled >= 0.4712 (90% of bf16 base 0.5236).  #795 int4head 0.5030.
  MMLU-Pro 5-choice 16-shot   >= 0.572.                     #795 int4head 0.6920.
  GSM8K 8-shot CoT            >= 0.807.                      #795/#788 int4head 0.915.
  AIME greedy maj@1 n=30      >= 0.090 (3/30).               #795 int4head 0.300.
LOCAL served measurement; analysis_only, official_tps=0, NO HF Job / NO FIRE.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

OUT = ROOT / "research" / "bi0_body_groupsize" / "firegate_g128"
RES = OUT / "results"
ARM = "g128_body"

# --- Morgan quality floors (the GATE) ------------------------------------- #
BASE_GPQA_BF16_REF = 0.5236   # bf16 base GPQA -> the >=90% gate DENOMINATOR
QUALITY_BAR_FRAC = 0.90
GPQA_PASS_BAR = round(BASE_GPQA_BF16_REF * QUALITY_BAR_FRAC, 4)  # 0.4712
MMLU_PASS_BAR = 0.572
GSM8K_PASS_BAR = 0.807
AIME_PASS_BAR = 0.090

# --- #795 int4head reference (the clean A/B; only delta = body g32->g128) -- #
INT4HEAD_GPQA_PER_SEED = {12345: 0.5455, 13579: 0.5101, 23456: 0.5152,
                          34567: 0.4848, 45678: 0.4596}
INT4HEAD_GPQA_POOLED = 0.5030
INT4HEAD_MMLU_4096 = 0.6920
INT4HEAD_MMLU_2048 = 0.6040
INT4HEAD_AIME_GREEDY = 9 / 30   # 0.300
INT4HEAD_AIME_SAMPLED = 12 / 30  # 0.400
INT4HEAD_GSM8K_GREEDY = 0.915   # #788

# --- PTQ-g32 same-recipe in-harness reference (speed A/B, this PR) --------- #
PTQ_G32_TPS = 234.08
G128_TPS = 240.62
G128_TPS_DELTA_PCT = 2.79

GPQA_SEEDS = [12345, 13579, 23456, 34567, 45678]
MMLU_BUDGETS = [4096, 2048]


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> int:
    summary: dict = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 814,
        "arm": ARM,
        "submission": "int4_mtp_bi0_int4head (MODEL_ID override -> g128 body)",
        "served_config": (
            "int4 W4A16 BODY group_size=128 + int4 W4A16 g32 lm_head + MTP K=6 + "
            "surgattn force-2D, VLLM_BATCH_INVARIANT=0"
        ),
        "only_delta_vs_int4head": "body group_size g32 -> g128",
        "sampling_gpqa_mmlu": "T=1.0 top_p=0.95 top_k=64 sampling_seed=0",
        "gpqa_max_tokens": 6144,
        "gpqa_n_per_seed": 198,
        "gpqa_seeds": GPQA_SEEDS,
        # ---- gate framing ---- #
        "gpqa_pass_bar_abs": GPQA_PASS_BAR,
        "mmlu_pass_bar_abs": MMLU_PASS_BAR,
        "gsm8k_pass_bar_abs": GSM8K_PASS_BAR,
        "aime_pass_bar_abs": AIME_PASS_BAR,
        "quality_bar_frac": QUALITY_BAR_FRAC,
        # ---- #795 int4head A/B references ---- #
        "int4head_gpqa_pooled_ref": INT4HEAD_GPQA_POOLED,
        "int4head_gpqa_per_seed_ref": INT4HEAD_GPQA_PER_SEED,
        "int4head_mmlu_4096_ref": INT4HEAD_MMLU_4096,
        "int4head_mmlu_2048_ref": INT4HEAD_MMLU_2048,
        "int4head_aime_greedy_ref": INT4HEAD_AIME_GREEDY,
        "int4head_aime_sampled_ref": INT4HEAD_AIME_SAMPLED,
        "int4head_gsm8k_greedy_ref": INT4HEAD_GSM8K_GREEDY,
        # ---- speed A/B (this PR) ---- #
        "ptq_g32_tps_ref": PTQ_G32_TPS,
        "g128_tps": G128_TPS,
        "g128_tps_delta_pct_vs_ptq_g32": G128_TPS_DELTA_PCT,
    }

    # ---- GPQA seeds (seed-paired vs #795) ---- #
    gpqa_arms: dict[int, dict] = {}
    for s in GPQA_SEEDS:
        d = load(RES / f"{ARM}_gpqa_s{s}.json")
        if d:
            gpqa_arms[s] = d

    passes_gpqa = None
    if gpqa_arms:
        per_seed_acc = {s: a["accuracy"] for s, a in gpqa_arms.items()}
        per_seed_delta = {
            s: round(per_seed_acc[s] - INT4HEAD_GPQA_PER_SEED[s], 4)
            for s in per_seed_acc if s in INT4HEAD_GPQA_PER_SEED
        }
        accs = list(per_seed_acc.values())
        n_seeds = len(accs)
        mean_acc = sum(accs) / n_seeds
        std_acc = math.sqrt(sum((x - mean_acc) ** 2 for x in accs) / (n_seeds - 1)) if n_seeds > 1 else 0.0
        pooled_correct = sum(a["n_correct"] for a in gpqa_arms.values())
        pooled_scored = sum(a["n_scored"] for a in gpqa_arms.values())
        pooled_acc = pooled_correct / pooled_scored if pooled_scored else float("nan")
        pooled_lo, pooled_hi = wilson(pooled_correct, pooled_scored)
        n_err = sum(a.get("n_error", 0) for a in gpqa_arms.values())
        n_trunc = sum(a.get("n_length_truncated", 0) for a in gpqa_arms.values())
        ctok_mean = (
            sum(a.get("completion_tokens_mean", 0.0) * a["n_scored"] for a in gpqa_arms.values()) / pooled_scored
            if pooled_scored else float("nan")
        )
        passes_gpqa = bool(pooled_acc >= GPQA_PASS_BAR)
        summary.update({
            "gpqa_diamond_pooled": pooled_acc,
            "gpqa_diamond_mean": mean_acc,
            "gpqa_diamond_std": std_acc,
            "gpqa_pooled_correct": pooled_correct,
            "gpqa_pooled_scored": pooled_scored,
            "gpqa_pooled_wilson95_lo": pooled_lo,
            "gpqa_pooled_wilson95_hi": pooled_hi,
            "gpqa_min_seed_acc": min(accs),
            "gpqa_max_seed_acc": max(accs),
            "gpqa_n_seeds": n_seeds,
            "gpqa_per_seed_acc": per_seed_acc,
            "gpqa_per_seed_delta_vs_int4head": per_seed_delta,
            "gpqa_n_error_total": n_err,
            "gpqa_n_length_truncated_total": n_trunc,
            "gpqa_completion_tokens_mean": ctok_mean,
            "gpqa_pct_of_bf16_base": pooled_acc / BASE_GPQA_BF16_REF,
            "gpqa_delta_vs_int4head": round(pooled_acc - INT4HEAD_GPQA_POOLED, 4),
            "passes_gate_wilson_lo": bool(pooled_lo >= GPQA_PASS_BAR),
            "passes_gpqa_gate": passes_gpqa,
        })

    # ---- MMLU-Pro at each budget ---- #
    mmlu_results: dict[int, dict] = {}
    for mt in MMLU_BUDGETS:
        m = load(RES / f"{ARM}_mmlu_n250_mt{mt}_s12345.json")
        if m:
            mmlu_results[mt] = m
            summary[f"mmlu_pro_mt{mt}"] = m.get("accuracy")
            summary[f"mmlu_pro_mt{mt}_n"] = m.get("n_scored")
            summary[f"mmlu_pro_mt{mt}_trunc"] = m.get("n_length_truncated")
    mmlu_clean = mmlu_results.get(4096) or (next(iter(mmlu_results.values())) if mmlu_results else {})
    passes_mmlu = None
    if mmlu_clean:
        mmlu_clean_acc = mmlu_clean.get("accuracy")
        passes_mmlu = bool(mmlu_clean_acc >= MMLU_PASS_BAR) if mmlu_clean_acc is not None else None
        summary["mmlu_pro_clean"] = mmlu_clean_acc
        summary["mmlu_pro_clean_budget"] = mmlu_clean.get("max_tokens")
        summary["mmlu_pro_delta_vs_int4head_4096"] = (
            round(mmlu_clean_acc - INT4HEAD_MMLU_4096, 4) if mmlu_clean_acc is not None else None
        )
        summary["passes_mmlu_gate"] = passes_mmlu

    # ---- GSM8K sampled (PRIMARY) + greedy (diagnostic) ---- #
    gsm_s = load(RES / f"{ARM}_sampled_s0.json")
    gsm_g = load(RES / f"{ARM}_greedy_s0.json")
    passes_gsm8k = None
    # PRIMARY gate axis: sampled (lewtun #31). If absent, fall back to greedy.
    gsm_primary = gsm_s or gsm_g
    if gsm_primary:
        gsm_acc = gsm_primary.get("accuracy")
        passes_gsm8k = bool(gsm_acc >= GSM8K_PASS_BAR) if gsm_acc is not None else None
        summary.update({
            "gsm8k_primary_regime": gsm_primary.get("regime"),
            "gsm8k_primary_acc": gsm_acc,
            "gsm8k_primary_n_correct": gsm_primary.get("n_correct"),
            "gsm8k_primary_n_problems": gsm_primary.get("n_problems"),
            "gsm8k_primary_trunc_rate": gsm_primary.get("truncation_rate"),
            "gsm8k_primary_extract_fail_rate": gsm_primary.get("extract_fail_rate"),
            "passes_gsm8k_gate": passes_gsm8k,
        })
    if gsm_s:
        summary["gsm8k_sampled_acc"] = gsm_s.get("accuracy")
        summary["gsm8k_sampled_n_correct"] = gsm_s.get("n_correct")
        summary["gsm8k_sampled_n_problems"] = gsm_s.get("n_problems")
    if gsm_g:
        summary["gsm8k_greedy_acc"] = gsm_g.get("accuracy")
        summary["gsm8k_greedy_n_correct"] = gsm_g.get("n_correct")
        summary["gsm8k_greedy_n_problems"] = gsm_g.get("n_problems")
        summary["gsm8k_greedy_delta_vs_int4head"] = (
            round(gsm_g.get("accuracy") - INT4HEAD_GSM8K_GREEDY, 4)
            if gsm_g.get("accuracy") is not None else None
        )

    # ---- AIME greedy maj@1 (PRIMARY) + sampled maj@8 (SUPPLEMENT) ---- #
    aime_g = load(RES / f"{ARM}_aime_greedy_n30.json")
    passes_aime = None
    if aime_g:
        g_acc = aime_g.get("maj_k_accuracy")
        trunc = sum(1 for p in aime_g.get("per_problem", []) for f in p.get("finish_reasons", []) if f == "length")
        passes_aime = bool(g_acc >= AIME_PASS_BAR) if g_acc is not None else None
        summary.update({
            "aime_greedy_maj1": g_acc,
            "aime_greedy_n_correct": aime_g.get("n_correct_maj"),
            "aime_greedy_n_problems": aime_g.get("n_problems"),
            "aime_greedy_extract_fail_rate": aime_g.get("extract_fail_rate"),
            "aime_greedy_length_trunc": trunc,
            "aime_greedy_total_samples": aime_g.get("total_samples"),
            "aime_greedy_delta_vs_int4head": (
                round(g_acc - INT4HEAD_AIME_GREEDY, 4) if g_acc is not None else None
            ),
            "passes_aime_gate": passes_aime,
        })
    aime_s = load(RES / f"{ARM}_aime_sampled_maj8_n30.json")
    if aime_s:
        summary.update({
            "aime_sampled_maj8": aime_s.get("maj_k_accuracy"),
            "aime_sampled_n_correct": aime_s.get("n_correct_maj"),
            "aime_sampled_mean_pass": aime_s.get("mean_pass_rate"),
            "aime_sampled_delta_vs_int4head": (
                round(aime_s.get("maj_k_accuracy") - INT4HEAD_AIME_SAMPLED, 4)
                if aime_s.get("maj_k_accuracy") is not None else None
            ),
        })

    # ---- overall fire verdict ---- #
    axes = {"gpqa": passes_gpqa, "mmlu": passes_mmlu, "gsm8k": passes_gsm8k, "aime": passes_aime}
    all_present = all(v is not None for v in axes.values())
    all_pass = all(v for v in axes.values() if v is not None)
    summary["axes_present"] = {k: (v is not None) for k, v in axes.items()}
    summary["all_axes_present"] = all_present
    summary["fire_worthy"] = bool(all_present and all_pass)
    summary["fire_worthy_so_far"] = bool(all_pass)

    print("G128_FIREGATE_SUMMARY " + json.dumps(summary, default=str))

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 814,
        "experiment": "bi0-int4head-body-groupsize",
        "arm": ARM,
        "body_group_size": 128,
        "head_group_size": 32,
        "substrate": "int4_w4a16_body_g128 + int4_w4a16_g32_lmhead + gemma4_mtp_K6 + surgattn_force2d + BI0",
        "control_substrate": "int4head (int4_w4a16_body_g32 + int4_w4a16_g32_lmhead)",
        "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "num_speculative_tokens": 6,
        "gpqa_seeds": GPQA_SEEDS,
        "mmlu_budgets": MMLU_BUDGETS,
        "gpqa_gate": GPQA_PASS_BAR,
        "mmlu_gate": MMLU_PASS_BAR,
        "gsm8k_gate": GSM8K_PASS_BAR,
        "aime_gate": AIME_PASS_BAR,
        "local_decode_tps": G128_TPS,
        "tps_lever_pct_vs_ptq_g32": G128_TPS_DELTA_PCT,
    }

    run = wandb_logging.init_wandb_run(
        job_type="bi0-body-groupsize-firegate",
        agent="wirbel",
        name="wirbel/body-groupsize-g128-firegate",
        group="bi0-int4head-body-groupsize",
        notes=(
            "PR #814: g128 body-group-size fire-prep quality panel. int4 W4A16 BODY "
            "group_size=128 (+2.79% local decode TPS vs PTQ-g32), only delta vs the "
            "#795 int4head fire candidate is body group_size. GPQA-Diamond n=198 x5 "
            "seeds (seed-paired vs #795) + MMLU-Pro n=250 @4096&2048 + GSM8K n=300 "
            "sampled&greedy + AIME n=30 greedy maj@1 & sampled maj@8. Gates: GPQA>=0.4712, "
            "MMLU>=0.572, GSM8K>=0.807, AIME>=0.090. #795 int4head refs: GPQA 0.5030 / "
            "MMLU 0.6920 / GSM8K 0.915 / AIME greedy 0.300. LOCAL served; analysis_only, "
            "official_tps=0, NO FIRE."
        ),
        tags=["gpqa", "mmlu-pro", "gsm8k", "aime", "5seed", "analysis-only", "pr-814",
              "body-groupsize", "g128", "firegate"],
        config=config,
    )
    if run is None:
        print("[agg] wandb disabled/unavailable; metrics above + JSON only", flush=True)
        (OUT / "firegate_g128_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return 0

    wandb_logging.log_summary(run, summary, step=0)
    for s, arm in gpqa_arms.items():
        slim = {k: v for k, v in arm.items() if k not in ("per_sample", "eval_log")}
        wandb_logging.log_json_artifact(run, name=f"{ARM}_gpqa_s{s}", artifact_type="gpqa-diamond", data=slim)
    for mt, m in mmlu_results.items():
        slim_m = {k: v for k, v in m.items() if k not in ("per_sample", "eval_log")}
        wandb_logging.log_json_artifact(run, name=f"{ARM}_mmlu_n250_mt{mt}", artifact_type="mmlu-pro", data=slim_m)
    for tag, gd in (("sampled", gsm_s), ("greedy", gsm_g)):
        if gd:
            slim_g = {k: v for k, v in gd.items() if k not in ("per_item", "samples")}
            wandb_logging.log_json_artifact(run, name=f"{ARM}_gsm8k_{tag}", artifact_type="gsm8k", data=slim_g)
    if aime_g:
        slim_a = {k: v for k, v in aime_g.items() if k != "per_problem"}
        wandb_logging.log_json_artifact(run, name=f"{ARM}_aime_greedy_n30", artifact_type="aime", data=slim_a)
    if aime_s:
        slim_as = {k: v for k, v in aime_s.items() if k != "per_problem"}
        wandb_logging.log_json_artifact(run, name=f"{ARM}_aime_sampled_maj8_n30", artifact_type="aime", data=slim_as)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    print(f"[agg] wandb run_id={run_id}", flush=True)
    (OUT / "firegate_g128_summary.json").write_text(
        json.dumps({**summary, "wandb_run_id": run_id}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
