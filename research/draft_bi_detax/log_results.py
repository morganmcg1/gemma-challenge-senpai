"""PR #633 draft-BI detax — leg-1/leg-2 analysis logger (analysis_only, official_tps=0).

Recomputes the byte-identity comparisons from the captured decode streams and logs
the verdict to W&B. No training/serving here — the four decode streams were captured
by gen_greedy_reference (served mode, int4 target, BI=1, greedy, K=6, 64x512):

  arm_x_qat      spec-ON, drafter=/tmp/qat-assistant            (default QAT MTP head)
  arm_x_qat_rep  spec-ON, drafter=/tmp/qat-assistant (re-boot)  (run-to-run FLOOR control)
  arm_y_alt      spec-ON, drafter=google/gemma-4-E4B-it-assistant (distinct published head)
  ref_m1ar       spec-OFF (SENPAI_REFERENCE_MODE=1) -> M=1 AR    (the #319 reference)
"""
from __future__ import annotations

import os
import re
import sys
import json
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scripts.local_validation import greedy_gate  # noqa: E402

ROOT = Path("research/draft_bi_detax")


def _decode(arm: str) -> Path:
    return ROOT / arm / "decode_outputs.jsonl"


def compare(ref_arm: str, cand_arm: str) -> dict:
    report = greedy_gate.compare(_decode(ref_arm), _decode(cand_arm))
    onset = greedy_gate.onset_summary(report)
    return {
        "verdict": report.verdict,
        "prompts_compared": report.num_prompts_compared,
        "prompts_identical": report.num_identical,
        "prompts_divergent": report.num_divergent,
        "tokens_compared": report.total_tokens_compared,
        "tokens_divergent": report.total_divergent_tokens,
        "tokens_divergent_frac": (report.total_divergent_tokens / report.total_tokens_compared)
        if report.total_tokens_compared else 0.0,
        "onset_min": onset.get("onset_min"),
        "onset_median": onset.get("onset_median"),
        "onset_max": onset.get("onset_max"),
    }


def mean_acceptance(arm: str) -> float | None:
    log = (ROOT / arm / "served_reference_server.log").read_text(errors="ignore")
    vals = [float(x) for x in re.findall(r"Mean acceptance length: ([0-9.]+)", log)]
    return round(sum(vals) / len(vals), 4) if vals else None


def main() -> int:
    floor = compare("arm_x_qat", "arm_x_qat_rep")          # same drafter, fresh boot
    draftswap = compare("arm_x_qat", "arm_y_alt")           # different drafter, fixed M/BI/target
    mconf_x = compare("ref_m1ar", "arm_x_qat")              # M=1 AR vs M=7 spec (qat)
    mconf_y = compare("ref_m1ar", "arm_y_alt")              # M=1 AR vs M=7 spec (alt)

    acc_x = mean_acceptance("arm_x_qat")
    acc_xrep = mean_acceptance("arm_x_qat_rep")
    acc_y = mean_acceptance("arm_y_alt")

    # Leg-1 verdict: the draft is output-invariant iff a draft swap (with a clean
    # run-to-run floor) leaves the byte-exact stream identical.
    floor_clean = floor["tokens_divergent"] == 0
    draft_output_invariant = floor_clean and draftswap["tokens_divergent"] == 0
    bi_draft_selectable = False  # leg 2: process-global ATen-dispatcher Library, no toggle

    if not floor_clean:
        verdict = "INCONCLUSIVE_NONZERO_FLOOR"
    elif not draft_output_invariant:
        verdict = "DRAFT_AFFECTS_OUTPUT"
    elif not bi_draft_selectable:
        verdict = "DRAFT_BI_GLOBAL_ONLY"
    else:
        verdict = "DRAFT_BI_DETAX_VIABLE"

    summary = {
        "verdict": verdict,
        "draft_output_invariant": draft_output_invariant,
        "bi_draft_selectable": bi_draft_selectable,
        "run_to_run_floor_clean": floor_clean,
        "floor_divergent_tokens": floor["tokens_divergent"],
        "draftswap_divergent_tokens": draftswap["tokens_divergent"],
        "draftswap_divergent_frac": round(draftswap["tokens_divergent_frac"], 5),
        "draftswap_divergent_prompts": draftswap["prompts_divergent"],
        "draftswap_onset_median": draftswap["onset_median"],
        "mconfound_x_divergent_frac": round(mconf_x["tokens_divergent_frac"], 5),
        "mconfound_x_onset_median": mconf_x["onset_median"],
        "mconfound_y_divergent_frac": round(mconf_y["tokens_divergent_frac"], 5),
        "acceptance_qat": acc_x,
        "acceptance_qat_rep": acc_xrep,
        "acceptance_alt": acc_y,
        "num_prompts": draftswap["prompts_compared"],
        "output_len": 512,
        "num_speculative_tokens": 6,
        "analysis_only": True,
        "official_tps": 0,
        "cmp_floor": floor,
        "cmp_draftswap": draftswap,
        "cmp_mconfound_x": mconf_x,
        "cmp_mconfound_y": mconf_y,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    try:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY"),
            name="wirbel/draft-bi-detax",
            group="optionb-draft-bi-detax",
            job_type="analysis",
            tags=["pr633", "specdec", "greedy-identity", "draft-bi-detax",
                  "int4-mtp", "draft-invariance", "analysis-only"],
            config={
                "submission": "submissions/int4_mtp_batchinv",
                "target_model": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "drafter_qat": "/tmp/qat-assistant",
                "drafter_alt": "google/gemma-4-E4B-it-assistant",
                "VLLM_BATCH_INVARIANT": 1,
                "num_speculative_tokens": 6,
                "num_prompts": 64,
                "output_len": 512,
                "seed": 1,
                "vllm": "0.22.0",
                "analysis_only": True,
                "official_tps": 0,
            },
        )
        wandb.log({k: v for k, v in summary.items() if not isinstance(v, dict)})
        wandb.run.summary.update({k: v for k, v in summary.items() if not isinstance(v, dict)})
        print(f"WANDB_RUN_ID={run.id}")
        wandb.finish()
    except Exception as exc:  # logging must never discard a finished analysis
        print(f"[wandb] non-fatal logging error: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
