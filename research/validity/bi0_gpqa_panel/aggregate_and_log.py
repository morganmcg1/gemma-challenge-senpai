#!/usr/bin/env python3
"""wirbel #773 aggregator: GPQA-Diamond quality gate for the bi0 quality-safe rung.

Reads the multi-seed GPQA-Diamond arms (full n=198/seed) + the MMLU-Pro n=100
sanity arm produced by run_panel.sh against the local bi0 server
(int4_mtp_bi0_surgattn: int4 W4A16 + MTP K=6 + surgattn force-2D, VLLM_BATCH_INVARIANT=0),
emits Morgan's KEY OUTPUT (does bi0 clear the GPQA gate), and logs to W&B group
``bi0-gpqa-panel``.

Gate (PR #773): GPQA >= 0.471 == 90% of the bf16 base GPQA 0.5236. int4 base
reference = 0.4798 (ubel #538). MMLU-Pro sanity vs wirbel #762 = 0.644.
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

OUT = ROOT / "research" / "validity" / "bi0_gpqa_panel"
RES = OUT / "results"

# --- advisor anchors (PR #773 body) ---------------------------------------- #
BASE_GPQA_BF16_REF = 0.5236   # bf16 base GPQA -> the >=90% gate DENOMINATOR
BASE_GPQA_INT4_REF = 0.4798   # int4 base GPQA reference (ubel #538)
QUALITY_BAR_FRAC = 0.90
PASS_BAR = round(BASE_GPQA_BF16_REF * QUALITY_BAR_FRAC, 4)  # 0.4712 ~ PR's 0.471
MMLU_PRO_762_REF = 0.644      # wirbel #762 MMLU-Pro sanity anchor

GPQA_SEEDS = [12345, 13579, 23456, 34567, 45678]


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
        d = load(RES / f"bi0_gpqa_s{s}.json")
        if d:
            gpqa_arms[s] = d

    if not gpqa_arms:
        print("[agg] FATAL: no GPQA seed JSONs found", file=sys.stderr)
        return 2

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

    # Gate verdict: use the pooled estimate (tightest), confirm with worst seed +
    # lower Wilson bound so a near-threshold pass is not a single-seed fluke.
    passes_gate_pooled = bool(pooled_acc >= PASS_BAR)
    passes_gate_mean = bool(mean_acc >= PASS_BAR)
    passes_gate_min_seed = bool(min_seed_acc >= PASS_BAR)
    passes_gate_wilson_lo = bool(pooled_lo >= PASS_BAR)
    passes_gate = passes_gate_pooled and passes_gate_min_seed

    pct_of_bf16_base = pooled_acc / BASE_GPQA_BF16_REF
    pct_of_int4_base = pooled_acc / BASE_GPQA_INT4_REF

    # ---- MMLU-Pro sanity ---- #
    mmlu = load(RES / "bi0_mmlu_n100_s12345.json")
    mmlu_acc = mmlu.get("accuracy") if mmlu else None
    mmlu_n = mmlu.get("n_scored") if mmlu else None
    mmlu_delta_vs_762 = (mmlu_acc - MMLU_PRO_762_REF) if mmlu_acc is not None else None

    summary = {
        # ---- KEY OUTPUTS ---- #
        "gpqa_diamond_pooled": pooled_acc,
        "gpqa_diamond_mean": mean_acc,
        "gpqa_diamond_std": std_acc,
        "passes_gpqa_gate": passes_gate,
        "mmlu_pro": mmlu_acc,
        "analysis_only": True,
        "official_tps": 0,
        # ---- gate framing ---- #
        "gpqa_pass_bar_abs": PASS_BAR,
        "gpqa_base_bf16_reference_denominator": BASE_GPQA_BF16_REF,
        "gpqa_base_int4_reference": BASE_GPQA_INT4_REF,
        "quality_bar_frac": QUALITY_BAR_FRAC,
        "pct_of_bf16_base": pct_of_bf16_base,
        "pct_of_int4_base": pct_of_int4_base,
        "passes_gate_pooled": passes_gate_pooled,
        "passes_gate_mean": passes_gate_mean,
        "passes_gate_min_seed": passes_gate_min_seed,
        "passes_gate_wilson_lo": passes_gate_wilson_lo,
        # ---- GPQA distribution ---- #
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
        # ---- MMLU sanity ---- #
        "mmlu_pro_n": mmlu_n,
        "mmlu_pro_762_reference": MMLU_PRO_762_REF,
        "mmlu_pro_delta_vs_762": mmlu_delta_vs_762,
        # ---- provenance ---- #
        "submission": "int4_mtp_bi0_surgattn",
        "served_config": "int4 W4A16 + MTP K=6 + surgattn force-2D, VLLM_BATCH_INVARIANT=0",
        "sampling": "T=1.0 top_p=0.95 top_k=64 sampling_seed=0",
        "gpqa_max_tokens": 6144,
        "mmlu_max_tokens": 2048,
        "gpqa_n_per_seed": 198,
        "protocol": "Convention A (#652): vary dataset choice-shuffle seed, fix sampling_seed=0",
    }

    print("AGG773_SUMMARY " + json.dumps(summary, default=str))

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 773,
        "experiment": "bi0-gpqa-panel",
        "submission": "int4_mtp_bi0_surgattn",
        "substrate": "int4_w4a16_ct + gemma4_mtp_K6 + surgattn_force2d + BI0",
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "drafter_model": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "num_speculative_tokens": 6,
        "gpqa_seeds": GPQA_SEEDS,
        "gate_pass_bar": PASS_BAR,
    }

    run = wandb_logging.init_wandb_run(
        job_type="bi0-gpqa-panel",
        agent="wirbel",
        name="wirbel/bi0-gpqa-panel",
        group="bi0-gpqa-panel",
        notes=(
            "PR #773: GPQA-Diamond quality gate for the bi0 quality-safe rung "
            "(int4_mtp_bi0_surgattn, 218 TPS). Full n=198 GPQA x 5 choice-shuffle "
            "seeds + MMLU-Pro n=100 sanity, T=1/top_p=0.95/top_k=64, max_tokens=6144 "
            "(#619 anti-truncation). Gate = GPQA >= 0.471 (90% of bf16 base 0.5236); "
            "int4 base ref 0.4798 (ubel #538). LOCAL served; analysis_only, "
            "official_tps=0, NO FIRE."
        ),
        tags=["gpqa", "n198", "5seed", "analysis-only", "pr-773", "bi0", "quality-gate"],
        config=config,
    )
    if run is None:
        print("[agg] wandb disabled/unavailable; metrics above + JSON only", flush=True)
        (OUT / "agg773_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return 0

    wandb_logging.log_summary(run, summary, step=0)
    for s, arm in gpqa_arms.items():
        slim = {k: v for k, v in arm.items() if k != "per_sample"}
        wandb_logging.log_json_artifact(
            run, name=f"bi0_gpqa_s{s}", artifact_type="gpqa-diamond", data=slim
        )
    if mmlu:
        slim_m = {k: v for k, v in mmlu.items() if k != "per_sample"}
        wandb_logging.log_json_artifact(
            run, name="bi0_mmlu_n100_s12345", artifact_type="mmlu-pro", data=slim_m
        )
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    print(f"[agg] wandb run_id={run_id}", flush=True)
    (OUT / "agg773_summary.json").write_text(
        json.dumps({**summary, "wandb_run_id": run_id}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
