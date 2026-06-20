#!/usr/bin/env python3
"""ubel #795 aggregator: int4head fire-prep quality panel.

Reads the int4head (int4 W4A16 g32 lm_head; body byte-identical to bi0) panel
arms produced by run_gpqa_mmlu.sh + run_aime.sh against the local int4head
server on :8020, computes the fire-gate verdicts, and logs one pooled run to
W&B group ``bi0-int4head-firegate``.

The control (bi0) is int4 W4A16 body + bf16 lm_head; #788 swaps only the head to
int4 g32. Gates (PR #795, Morgan quality floors):
  GPQA-Diamond pooled >= 0.4712 (90% of bf16 base 0.5236).  bi0 ref 0.4970.
  MMLU-Pro 5-choice 16-shot   >= 0.572.                     bi0 ref 0.644 (clean).
  AIME greedy maj@1 n=30      >= 0.090 (3/30).               bi0 ref 10/30=0.333.
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

OUT = ROOT / "research" / "validity" / "int4head_firegate_panel"
RES = OUT / "results"

# --- advisor anchors (PR #795 / #773 / #581 / #580 bodies) ----------------- #
BASE_GPQA_BF16_REF = 0.5236   # bf16 base GPQA -> the >=90% gate DENOMINATOR
BASE_GPQA_INT4_REF = 0.4798   # int4 base GPQA reference (ubel #538)
QUALITY_BAR_FRAC = 0.90
GPQA_PASS_BAR = round(BASE_GPQA_BF16_REF * QUALITY_BAR_FRAC, 4)  # 0.4712 ~ 0.471
MMLU_PASS_BAR = 0.572         # Morgan floor
AIME_PASS_BAR = 0.090         # Morgan floor (3/30)

BI0_GPQA_REF = 0.4970         # bi0 pooled GPQA (#773)
BI0_MMLU_CLEAN_REF = 0.644    # bi0 MMLU-Pro full-context (#762)
BI0_MMLU_2048_REF = 0.57      # bi0 MMLU-Pro n=100 @2048 (#773, 16% trunc)
BI0_AIME_SAMPLED_REF = 10 / 30  # bi0 AIME sampled maj@8 (#762)
BASE_AIME_GREEDY_REF = 6 / 60   # bf16 base AIME greedy (#580)

GPQA_SEEDS = [12345, 13579, 23456, 34567, 45678]
MMLU_BUDGETS = [4096, 2048]


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> int:
    # ---- GPQA seeds ---- #
    gpqa_arms: dict[int, dict] = {}
    for s in GPQA_SEEDS:
        d = load(RES / f"int4head_gpqa_s{s}.json")
        if d:
            gpqa_arms[s] = d

    summary: dict = {
        "analysis_only": True,
        "official_tps": 0,
        "submission": "int4_mtp_bi0_int4head",
        "served_config": (
            "int4 W4A16 g32 lm_head + int4 W4A16 body + MTP K=6 + surgattn "
            "force-2D, VLLM_BATCH_INVARIANT=0"
        ),
        "sampling_gpqa_mmlu": "T=1.0 top_p=0.95 top_k=64 sampling_seed=0",
        "gpqa_max_tokens": 6144,
        "gpqa_n_per_seed": 198,
        "gpqa_seeds": GPQA_SEEDS,
        "protocol": "Convention A (#652): vary dataset choice-shuffle seed, fix sampling_seed=0",
        # ---- gate framing ---- #
        "gpqa_pass_bar_abs": GPQA_PASS_BAR,
        "gpqa_base_bf16_reference_denominator": BASE_GPQA_BF16_REF,
        "gpqa_base_int4_reference": BASE_GPQA_INT4_REF,
        "quality_bar_frac": QUALITY_BAR_FRAC,
        "mmlu_pass_bar_abs": MMLU_PASS_BAR,
        "aime_pass_bar_abs": AIME_PASS_BAR,
        # ---- bi0 references ---- #
        "bi0_gpqa_ref": BI0_GPQA_REF,
        "bi0_mmlu_clean_ref": BI0_MMLU_CLEAN_REF,
        "bi0_mmlu_2048_ref": BI0_MMLU_2048_REF,
        "bi0_aime_sampled_ref": BI0_AIME_SAMPLED_REF,
        "base_aime_greedy_ref": BASE_AIME_GREEDY_REF,
    }

    gpqa_present = bool(gpqa_arms)
    passes_gpqa = None
    if gpqa_present:
        per_seed_acc = {s: a["accuracy"] for s, a in gpqa_arms.items()}
        accs = list(per_seed_acc.values())
        n_seeds = len(accs)
        mean_acc = sum(accs) / n_seeds
        std_acc = (
            math.sqrt(sum((x - mean_acc) ** 2 for x in accs) / (n_seeds - 1))
            if n_seeds > 1
            else 0.0
        )
        pooled_correct = sum(a["n_correct"] for a in gpqa_arms.values())
        pooled_scored = sum(a["n_scored"] for a in gpqa_arms.values())
        pooled_acc = pooled_correct / pooled_scored if pooled_scored else float("nan")
        pooled_lo, pooled_hi = wilson(pooled_correct, pooled_scored)
        min_seed_acc = min(accs)
        max_seed_acc = max(accs)
        n_err = sum(a.get("n_error", 0) for a in gpqa_arms.values())
        n_trunc = sum(a.get("n_length_truncated", 0) for a in gpqa_arms.values())
        ctok_mean = (
            sum(a.get("completion_tokens_mean", 0.0) * a["n_scored"] for a in gpqa_arms.values())
            / pooled_scored
            if pooled_scored
            else float("nan")
        )
        passes_gate_pooled = bool(pooled_acc >= GPQA_PASS_BAR)
        passes_gate_mean = bool(mean_acc >= GPQA_PASS_BAR)
        passes_gate_min_seed = bool(min_seed_acc >= GPQA_PASS_BAR)
        passes_gate_wilson_lo = bool(pooled_lo >= GPQA_PASS_BAR)
        # PR #795 gates GPQA on POOLED accuracy (>= 0.4712). A per-seed min gate is
        # stricter than the PR and the bf16-head control fails it too (bi0 min seed
        # 0.4697 < 0.4712, #773), so min-seed is reported as a diagnostic only; the
        # Wilson 95% lower bound is logged as a robustness corroboration.
        passes_gpqa = passes_gate_pooled
        summary.update(
            {
                "gpqa_diamond_pooled": pooled_acc,
                "gpqa_diamond_mean": mean_acc,
                "gpqa_diamond_std": std_acc,
                "gpqa_pooled_correct": pooled_correct,
                "gpqa_pooled_scored": pooled_scored,
                "gpqa_pooled_wilson95_lo": pooled_lo,
                "gpqa_pooled_wilson95_hi": pooled_hi,
                "gpqa_min_seed_acc": min_seed_acc,
                "gpqa_max_seed_acc": max_seed_acc,
                "gpqa_n_seeds": n_seeds,
                "gpqa_per_seed_acc": per_seed_acc,
                "gpqa_n_error_total": n_err,
                "gpqa_n_length_truncated_total": n_trunc,
                "gpqa_completion_tokens_mean": ctok_mean,
                "gpqa_pct_of_bf16_base": pooled_acc / BASE_GPQA_BF16_REF,
                "gpqa_pct_of_int4_base": pooled_acc / BASE_GPQA_INT4_REF,
                "gpqa_delta_vs_bi0": pooled_acc - BI0_GPQA_REF,
                "passes_gate_pooled": passes_gate_pooled,
                "passes_gate_mean": passes_gate_mean,
                "passes_gate_min_seed": passes_gate_min_seed,
                "passes_gate_wilson_lo": passes_gate_wilson_lo,
                "passes_gpqa_gate": passes_gpqa,
            }
        )

    # ---- MMLU-Pro at each budget ---- #
    mmlu_results: dict[int, dict] = {}
    for mt in MMLU_BUDGETS:
        m = load(RES / f"int4head_mmlu_n250_mt{mt}_s12345.json")
        if m:
            mmlu_results[mt] = m
            summary[f"mmlu_pro_mt{mt}"] = m.get("accuracy")
            summary[f"mmlu_pro_mt{mt}_n"] = m.get("n_scored")
            summary[f"mmlu_pro_mt{mt}_trunc"] = m.get("n_length_truncated")
            summary[f"mmlu_pro_mt{mt}_trunc_rate"] = (
                m.get("n_length_truncated", 0) / m["n_scored"] if m.get("n_scored") else None
            )
    # clean = the 4096 budget if present, else fall back to whatever exists
    mmlu_clean = mmlu_results.get(4096) or (next(iter(mmlu_results.values())) if mmlu_results else {})
    passes_mmlu = None
    if mmlu_clean:
        mmlu_clean_acc = mmlu_clean.get("accuracy")
        passes_mmlu = bool(mmlu_clean_acc >= MMLU_PASS_BAR) if mmlu_clean_acc is not None else None
        summary["mmlu_pro_clean"] = mmlu_clean_acc
        summary["mmlu_pro_clean_budget"] = mmlu_clean.get("max_tokens")
        summary["mmlu_pro_delta_vs_bi0_clean"] = (
            (mmlu_clean_acc - BI0_MMLU_CLEAN_REF) if mmlu_clean_acc is not None else None
        )
        summary["passes_mmlu_gate"] = passes_mmlu

    # ---- AIME greedy maj@1 (PRIMARY) + sampled maj@8 (SUPPLEMENT) ---- #
    aime_g = load(RES / "int4head_aime_greedy_n30.json")
    passes_aime = None
    if aime_g:
        g_acc = aime_g.get("maj_k_accuracy")
        trunc = sum(
            1 for p in aime_g.get("per_problem", []) for f in p.get("finish_reasons", []) if f == "length"
        )
        passes_aime = bool(g_acc >= AIME_PASS_BAR) if g_acc is not None else None
        summary.update(
            {
                "aime_greedy_maj1": g_acc,
                "aime_greedy_n_correct": aime_g.get("n_correct_maj"),
                "aime_greedy_n_problems": aime_g.get("n_problems"),
                "aime_greedy_extract_fail_rate": aime_g.get("extract_fail_rate"),
                "aime_greedy_length_trunc": trunc,
                "aime_greedy_total_samples": aime_g.get("total_samples"),
                "passes_aime_gate": passes_aime,
            }
        )
    aime_s = load(RES / "int4head_aime_sampled_maj8_n30.json")
    if aime_s:
        summary.update(
            {
                "aime_sampled_maj8": aime_s.get("maj_k_accuracy"),
                "aime_sampled_n_correct": aime_s.get("n_correct_maj"),
                "aime_sampled_mean_pass": aime_s.get("mean_pass_rate"),
                "aime_sampled_delta_vs_bi0": (
                    aime_s.get("maj_k_accuracy") - BI0_AIME_SAMPLED_REF
                    if aime_s.get("maj_k_accuracy") is not None
                    else None
                ),
            }
        )

    # ---- GSM8K already run in #788 (reference only, not re-run here) ---- #
    summary["gsm8k_788_ref"] = 0.915
    summary["gsm8k_pass_bar_abs"] = 0.807

    # ---- overall fire verdict ---- #
    axes_present = {
        "gpqa": passes_gpqa,
        "mmlu": passes_mmlu,
        "aime": passes_aime,
    }
    all_present = all(v is not None for v in axes_present.values())
    all_pass = all(v for v in axes_present.values() if v is not None)
    summary["axes_present"] = {k: (v is not None) for k, v in axes_present.items()}
    summary["all_axes_present"] = all_present
    summary["fire_worthy"] = bool(all_present and all_pass)
    summary["fire_worthy_so_far"] = bool(all_pass)  # of axes computed so far

    print("INT4HEAD_FIREGATE_SUMMARY " + json.dumps(summary, default=str))

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 795,
        "experiment": "bi0-int4head-firegate",
        "submission": "int4_mtp_bi0_int4head",
        "substrate": "int4_w4a16_g32_lmhead + int4_w4a16_body + gemma4_mtp_K6 + surgattn_force2d + BI0",
        "control_substrate": "bf16_lmhead + int4_w4a16_body (bi0)",
        "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "num_speculative_tokens": 6,
        "gpqa_seeds": GPQA_SEEDS,
        "mmlu_budgets": MMLU_BUDGETS,
        "gpqa_gate": GPQA_PASS_BAR,
        "mmlu_gate": MMLU_PASS_BAR,
        "aime_gate": AIME_PASS_BAR,
        "local_decode_tps": 256.74,
        "tps_lever_pct": 17.0,
    }

    run = wandb_logging.init_wandb_run(
        job_type="int4head-firegate-panel",
        agent="ubel",
        name="ubel/int4head-firegate-panel",
        group="bi0-int4head-firegate",
        notes=(
            "PR #795: int4head (int4 W4A16 g32 lm_head, body byte-identical to bi0) "
            "fire-prep quality panel. GPQA-Diamond n=198 x5 seeds (T=1/top_p=0.95/"
            "top_k=64, max_tokens=6144) + MMLU-Pro n=250 @4096 & @2048 + AIME n=30 "
            "greedy maj@1 (primary) & sampled maj@8 (supplement). Gates: GPQA>=0.4712, "
            "MMLU>=0.572, AIME>=0.090. bi0 refs: GPQA 0.4970 / MMLU 0.644 / AIME 10/30. "
            "LOCAL served; analysis_only, official_tps=0, NO FIRE."
        ),
        tags=["gpqa", "mmlu-pro", "aime", "5seed", "analysis-only", "pr-795", "int4head", "firegate"],
        config=config,
    )
    if run is None:
        print("[agg] wandb disabled/unavailable; metrics above + JSON only", flush=True)
        (OUT / "firegate_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return 0

    wandb_logging.log_summary(run, summary, step=0)
    for s, arm in gpqa_arms.items():
        slim = {k: v for k, v in arm.items() if k not in ("per_sample", "eval_log")}
        wandb_logging.log_json_artifact(
            run, name=f"int4head_gpqa_s{s}", artifact_type="gpqa-diamond", data=slim
        )
    for mt, m in mmlu_results.items():
        slim_m = {k: v for k, v in m.items() if k not in ("per_sample", "eval_log")}
        wandb_logging.log_json_artifact(
            run, name=f"int4head_mmlu_n250_mt{mt}", artifact_type="mmlu-pro", data=slim_m
        )
    if aime_g:
        slim_a = {k: v for k, v in aime_g.items() if k != "per_problem"}
        wandb_logging.log_json_artifact(
            run, name="int4head_aime_greedy_n30", artifact_type="aime", data=slim_a
        )
    if aime_s:
        slim_as = {k: v for k, v in aime_s.items() if k != "per_problem"}
        wandb_logging.log_json_artifact(
            run, name="int4head_aime_sampled_maj8_n30", artifact_type="aime", data=slim_as
        )
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    print(f"[agg] wandb run_id={run_id}", flush=True)
    (OUT / "firegate_summary.json").write_text(
        json.dumps({**summary, "wandb_run_id": run_id}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
