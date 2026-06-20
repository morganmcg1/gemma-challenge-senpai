#!/usr/bin/env python3
"""PR #791 W&B logger for the surgattn-OFF (3D-on-M=1) MMLU-Pro quality kill-gate.

Reads the paired ``mmlu_killgate_summary.json`` that run_quality.py produces
(variant = VLLM_SURGATTN=0 / 3D-on-M=1, control = shipped bi0 force-2D, on the
SAME n=100/seed=12345 byte-identical prompt set), frames it against the advisor's
#784 gate, and logs to W&B group ``bi0-surgattn-3d-qualregate``.

Anchor note (decisive): the PR quotes bi0 MMLU-Pro = 0.644, but that is the #762
anchor at a DIFFERENT sample/protocol. At THIS exact config (n=100/seed=12345/
max_tokens=2048/T=1.0-top_p=0.95-top_k=64) bi0 measured 0.57 in #773 (its own
aggregator hardcodes MMLU_PRO_762_REF=0.644 and reports the -0.074 delta). So the
absolute advisor thresholds (kill<0.572, pass>=0.612) are anchored to 0.644 and
are mis-calibrated for this config; the same-session PAIRED delta (variant vs the
fresh same-config control) is the load-bearing signal. We report both.

LOCAL analysis only; analysis_only=true, official_tps=0, publish=false, NO FIRE.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

HERE = ROOT / "research" / "validity" / "bi0_surgattn_3d_qualregate"
RUNS = HERE / "runs"

# --- advisor anchors (PR #791 / #784) ------------------------------------- #
BI0_ANCHOR_762 = 0.644      # PR-quoted bi0 MMLU-Pro (DIFFERENT sample/protocol)
BI0_SAMECONFIG_773 = 0.57   # bi0 MMLU-Pro at n=100/s12345/2048 (this config; #773)
KILL_FLOOR = 0.572          # advisor: variant < this => collapse/kill (anchored to .644)
PASS_BAR = 0.612            # advisor: variant >= this => within 5% of .644 => promising
BAND_FRAC = 0.95            # "within 5%" band


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p on discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> int:
    src = RUNS / "mmlu_killgate_summary.json"
    if not src.exists():
        print(f"[log] FATAL: {src} not found (run run_quality.py first)", file=sys.stderr)
        return 2
    kg = json.loads(src.read_text())
    arms = kg.get("arms", {})
    variant = arms.get("variant", {})
    control = arms.get("control", {})
    paired = kg.get("paired", {})

    v_acc = variant.get("accuracy")
    c_acc = control.get("accuracy")
    if v_acc is None:
        print("[log] FATAL: no variant accuracy in summary", file=sys.stderr)
        return 2

    # Paired discordants: b = control-right/variant-wrong, c = variant-right/control-wrong.
    b = paired.get("control_right_variant_wrong")
    c = paired.get("variant_right_control_wrong")
    mcnemar_p = mcnemar_exact_p(b, c) if (b is not None and c is not None) else None

    # Same-config band: "within 5% of bi0" interpreted against the same-session control
    # (the correct denominator) AND, for completeness, against the mis-anchored 0.644.
    band_vs_control = (BAND_FRAC * c_acc) if c_acc else None
    in_band_vs_control = bool(c_acc and v_acc >= band_vs_control)
    in_band_vs_762 = bool(v_acc >= BAND_FRAC * BI0_ANCHOR_762)  # == >=0.612 (advisor PASS_BAR)

    # Advisor absolute thresholds (anchored to 0.644).
    advisor_verdict = (
        "kill" if v_acc < KILL_FLOOR
        else "pass" if v_acc >= PASS_BAR
        else "ambiguous"
    )

    v_lo, v_hi = wilson(variant.get("n_correct", 0), variant.get("n_scored", 0))
    c_lo, c_hi = wilson(control.get("n_correct", 0), control.get("n_scored", 0))

    summary = {
        # ---- KEY OUTPUTS ---- #
        "mmlu_pro_variant": v_acc,
        "mmlu_pro_control": c_acc,
        "delta_variant_minus_control": kg.get("delta_variant_minus_control"),
        "in_band_vs_control_5pct": in_band_vs_control,
        "advisor_verdict_abs": advisor_verdict,
        "analysis_only": True,
        "official_tps": 0,
        "publish": False,
        # ---- paired (load-bearing signal) ---- #
        "paired_n": paired.get("n_paired"),
        "paired_prompt_sha_mismatch": paired.get("prompt_sha_mismatch"),
        "paired_same_answer_text": paired.get("same_answer_text"),
        "paired_answer_flip": paired.get("answer_flip"),
        "paired_control_right_variant_wrong": b,
        "paired_variant_right_control_wrong": c,
        "paired_net_variant_minus_control_correct": paired.get("net_variant_minus_control_correct"),
        "mcnemar_exact_p_two_sided": mcnemar_p,
        # ---- band framing ---- #
        "band_frac": BAND_FRAC,
        "band_floor_vs_control": band_vs_control,
        "in_band_vs_762anchor": in_band_vs_762,
        "bi0_anchor_762": BI0_ANCHOR_762,
        "bi0_sameconfig_773": BI0_SAMECONFIG_773,
        "advisor_kill_floor": KILL_FLOOR,
        "advisor_pass_bar": PASS_BAR,
        # ---- distribution ---- #
        "variant_correct": variant.get("n_correct"),
        "variant_scored": variant.get("n_scored"),
        "variant_n_error": variant.get("n_error"),
        "variant_wilson95_lo": v_lo,
        "variant_wilson95_hi": v_hi,
        "variant_ctok_mean": variant.get("ctok_mean"),
        "variant_length_stop_rate": variant.get("length_stop_rate"),
        "control_correct": control.get("n_correct"),
        "control_scored": control.get("n_scored"),
        "control_n_error": control.get("n_error"),
        "control_wilson95_lo": c_lo,
        "control_wilson95_hi": c_hi,
        "control_ctok_mean": control.get("ctok_mean"),
        # ---- provenance ---- #
        "task": "mmlu_pro",
        "n": kg.get("n"),
        "seed": kg.get("seed"),
        "max_tokens": kg.get("max_tokens"),
        "sampling": kg.get("sampling"),
        "served_config": kg.get("served_config"),
        "submission": "int4_mtp_bi0_surgattn",
        "variant_toggle": "VLLM_SURGATTN=0 (force-2D patch skipped; 3D split-KV on M=1)",
    }

    print("KILLGATE_SUMMARY " + json.dumps(summary, default=str))

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "publish": False,
        "pr": 791,
        "experiment": "bi0-surgattn-3d-qualregate",
        "submission": "int4_mtp_bi0_surgattn",
        "variant": "VLLM_SURGATTN=0 (3D-on-M=1)",
        "control": "shipped bi0 (surgattn force-2D)",
        "substrate": "int4_w4a16_ct + gemma4_mtp_K6 + BI0",
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "served_config": kg.get("served_config"),
        "n": kg.get("n"),
        "seed": kg.get("seed"),
        "max_tokens": kg.get("max_tokens"),
    }

    run = wandb_logging.init_wandb_run(
        job_type="bi0-surgattn-3d-qualregate",
        agent="wirbel",
        name="wirbel/bi0-surgattn-3d-qualregate-mmlu",
        group="bi0-surgattn-3d-qualregate",
        notes=(
            "PR #791: MMLU-Pro quality kill-gate for surgattn-OFF (3D-on-M=1). "
            "Paired variant (VLLM_SURGATTN=0) vs same-config control (shipped bi0 "
            "force-2D) on byte-identical n=100/seed=12345 prompts, T=1/top_p=0.95/"
            "top_k=64, max_tokens=2048. Read the PAIRED delta: PR's 0.644 anchor is "
            "the #762 sample; this config's bi0 = 0.57 (#773). LOCAL; analysis_only, "
            "official_tps=0, NO FIRE."
        ),
        tags=["mmlu-pro", "n100", "paired", "analysis-only", "pr-791", "bi0",
              "surgattn", "3d-on-m1", "quality-gate"],
        config=config,
    )
    if run is None:
        print("[log] wandb disabled/unavailable; metrics above + JSON only", flush=True)
        (HERE / "killgate_logged_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return 0

    wandb_logging.log_summary(run, summary, step=0)
    for arm_name, arm in (("variant", variant), ("control", control)):
        if arm:
            wandb_logging.log_json_artifact(
                run, name=f"mmlu_{arm_name}", artifact_type="mmlu-pro", data=arm
            )
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    print(f"[log] wandb run_id={run_id}", flush=True)
    (HERE / "killgate_logged_summary.json").write_text(
        json.dumps({**summary, "wandb_run_id": run_id}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
