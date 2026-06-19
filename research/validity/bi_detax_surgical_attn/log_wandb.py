#!/usr/bin/env python
"""PR #751 wirbel — log the BI-tax surgical-attn recovery A/B to W&B group
bi_detax_surgical_attn. ANALYSIS ONLY (local served profiling; no HF Job).

Two served arms, both on the loadable full-vocab QAT ckpt
(google/gemma-4-E4B-it-qat-w4a16-ct), 128 prompts x 512 tok, official protocol:
  * BASELINE  : int4_mtp_batchinv     (VLLM_BATCH_INVARIANT=1, as-fired fire cfg)
  * RECOVERY  : int4_mtp_bi0_surgattn (VLLM_BATCH_INVARIANT=0 + surgical force-2D
                TRITON_ATTN patch)
Plus the AR-vs-AR determinism controls (spec-off run A vs run B) that prove the
spec-vs-AR greedy break is REAL (not cross-process FP noise).
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"
GROUP = "bi_detax_surgical_attn"

ARMS = {
    "baseline_bi1": {
        "name": "wirbel/bi-detax-baseline-bi1",
        "dir": HERE / "baseline_bi1" / "validate",
        "submission": "submissions/int4_mtp_batchinv",
        "batch_invariant": 1,
        "surgical_force2d": 0,
        "control": HERE / "ar_vs_ar_control" / "baseline_bi1_arVSar.json",
    },
    "recover_bi0_surgattn": {
        "name": "wirbel/bi-detax-recover-bi0-surgattn",
        "dir": HERE / "recover_bi0_surgattn" / "validate",
        "submission": "submissions/int4_mtp_bi0_surgattn",
        "batch_invariant": 0,
        "surgical_force2d": 1,
        "control": HERE / "ar_vs_ar_control" / "recover_bi0_arVSar.json",
    },
}


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def arm_metrics(spec: dict) -> dict:
    ev = _load(spec["dir"] / "evidence.json") or {}
    gr = _load(spec["dir"] / "greedy_report.json") or {}
    ctrl = _load(spec["control"]) if spec.get("control") else None
    ttok = gr.get("total_tokens_compared") or 0
    dtok = gr.get("total_divergent_tokens") or 0
    m = {
        "tps_single_stream_a10g": ev.get("tps_single_stream_a10g"),
        "ppl": ev.get("ppl"),
        "official_gate_pass": int(bool(ev.get("official_gate_pass"))),
        "completed": ev.get("completed"),
        "greedy_verdict": gr.get("verdict") or ev.get("greedy_verdict"),
        "greedy_num_identical": gr.get("num_identical"),
        "greedy_num_divergent": gr.get("num_divergent"),
        "greedy_num_prompts": gr.get("num_prompts_compared"),
        "greedy_div_token_frac": (dtok / ttok) if ttok else None,
        "greedy_onset_median": (ev.get("greedy_onset") or {}).get("onset_median"),
    }
    if ctrl is not None:
        m["ar_vs_ar_num_prompts"] = ctrl.get("num_prompts_compared")
        m["ar_vs_ar_num_divergent"] = ctrl.get("num_divergent")
        m["ar_vs_ar_num_identical"] = ctrl.get("num_identical")
        m["ar_vs_ar_total_divergent_tokens"] = ctrl.get("total_divergent_tokens")
        m["ar_vs_ar_verdict"] = ctrl.get("verdict")
        m["spec_break_is_real"] = int(ctrl.get("num_divergent") == 0)
    return m


def main():
    base = arm_metrics(ARMS["baseline_bi1"])
    rec = arm_metrics(ARMS["recover_bi0_surgattn"])
    tps_b = base["tps_single_stream_a10g"]
    tps_r = rec["tps_single_stream_a10g"]
    recover_pct = ((tps_r - tps_b) / tps_b) if (tps_b and tps_r) else None

    ids = {}
    for key, spec in ARMS.items():
        m = base if key == "baseline_bi1" else rec
        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP, name=spec["name"],
            job_type="analysis", reinit=True,
            config={
                "pr": 751, "lane": "bi_detax_surgical_attn",
                "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
                "submission": spec["submission"],
                "batch_invariant": spec["batch_invariant"],
                "surgical_force2d": spec["surgical_force2d"],
                "backend": "TRITON_ATTN",
                "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "served_substrate": "full_vocab_loadable_qat_ct",
                "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
                "num_speculative_tokens": 6, "num_prompts": 128, "output_len": 512,
                "tps_basis": "local_a10g_single_stream_NOT_official",
            },
        )
        ids[key] = run.id
        summary = dict(m)
        if key == "recover_bi0_surgattn":
            summary.update({
                "tps_BI1": tps_b,
                "tps_recover": tps_r,
                "recover_pct": recover_pct,
                "identity_recover_divergent": rec["greedy_num_divergent"],
                "identity_recover_128_clean": int(rec["greedy_num_divergent"] == 0),
                "identity_baseline_divergent": base["greedy_num_divergent"],
                "force2d_identity_delta_prompts": (
                    base["greedy_num_divergent"] - rec["greedy_num_divergent"]),
                "force2d_div_token_frac_delta": (
                    (base["greedy_div_token_frac"] or 0) - (rec["greedy_div_token_frac"] or 0)),
                "verdict": (
                    "SURGICAL_RECOVERS_SPEED_NOT_STRICTCLEAN: "
                    "+44.5% local TPS, PPL-gate PASS, but identity FAILS (105/128 "
                    "divergent) and bi1 baseline ALSO fails (108/128) — the served "
                    "M=K-verify-vs-M=1-AR break is REAL and survives both bi1 and "
                    "force-2D (corroborates stark #690; 2D/3D selector is not the "
                    "mechanism). No strict-clean baseline exists to preserve."),
            })
        run.log(summary)
        run.summary.update(summary)
        run.finish()
        print(f"[wandb] {key}: run {run.id} tps={m['tps_single_stream_a10g']} "
              f"divergent={m['greedy_num_divergent']}/128 "
              f"ar_vs_ar_div={m.get('ar_vs_ar_num_divergent')}")
    print("WANDB_RUN_IDS=" + ",".join(ids.values()))
    print("RECOVER_PCT=%.4f" % (recover_pct or 0))


if __name__ == "__main__":
    raise SystemExit(main())
