"""PR #623 closeout -> W&B group `optionb-bi1-speed-cost`.

Logs a self-contained companion run alongside the A/B run (3igf1sq0): the
decision block (BI=1 TPS cost, does Arm B beat 126.378), the teacher-forced PPL
gate for both arms, and the 2D-vs-3D attention attribution. Reads the three local
artifacts; no server, no recompute.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

LOCKED_AR = 126.378  # strict-#319 rung to beat (int4_g128_lmhead, PR #4)
PPL_GATE = 2.42


def _load(p: Path):
    return json.loads(p.read_text())


def _ppl_for(bi: int):
    hits = sorted(glob.glob(str(HERE / f"ppl/ppl_summary_bi{bi}_*.json")))
    if not hits:
        return None, None
    s = _load(Path(hits[0]))
    tag = "spec" if "spec.json" in hits[0] else ("specoff" if "specoff" in hits[0] else "unknown")
    return float(s["ppl"]), tag


def main() -> int:
    ab = _load(HERE / "paired_ab.json")
    v = ab["verdict"]
    a_mean = ab["arms"]["baseline"]["wall_tps"]["mean"]
    a_med = ab["arms"]["baseline"]["wall_tps"]["median"]
    b_mean = ab["arms"]["candidate"]["wall_tps"]["mean"]
    b_med = ab["arms"]["candidate"]["wall_tps"]["median"]

    ppl_a, tag_a = _ppl_for(0)
    ppl_b, tag_b = _ppl_for(1)

    probe_p = HERE / "attn_2d_vs_3d_probe.json"
    probe = _load(probe_p) if probe_p.exists() else {}
    cyc = probe.get("target_cycle_attn_us", {})
    m8 = cyc.get("ctx512_M8", {}).get("bi1_attn_cycle_delta_ms")
    m1 = cyc.get("ctx512_M1", {}).get("bi1_attn_cycle_delta_ms")

    # per-spec-step budget from decode durations (e_accept ~3.81)
    durA = ab["arms"]["baseline"]["records"][0]["decode_duration_s"]
    durB = ab["arms"]["candidate"]["records"][0]["decode_duration_s"]
    ntok = 65536
    eacc = ab["arms"]["baseline"]["e_accept_exact"]["mean"]
    steps = ntok / eacc
    step_ms_a = durA * 1000 / steps
    step_ms_b = durB * 1000 / steps
    step_delta_ms = step_ms_b - step_ms_a
    attn_frac_upper = (m1 / step_delta_ms) if (m1 and step_delta_ms) else None

    b_beats = bool(b_med > LOCKED_AR)
    cost_pct = -v["delta_median_pct"]  # report as positive cost
    if b_beats and cost_pct < 50:
        verdict = "BI1_SPEED_VIABLE__beats_126"
    elif b_beats:
        verdict = "BI1_SPEED_VIABLE_but_steep_cost__beats_126"
    else:
        verdict = "BI1_SPEED_KILLS_OPTIONB"

    summary = {
        # decision (instruction #5 headline)
        "decision/tps_arm_a_no_bi_median": a_med,
        "decision/tps_arm_a_no_bi_mean": a_mean,
        "decision/tps_arm_b_bi1_median": b_med,
        "decision/tps_arm_b_bi1_mean": b_mean,
        "decision/bi1_tps_cost_pct_median": cost_pct,
        "decision/bi1_tps_cost_pct_mean": -v["delta_mean_pct"],
        "decision/locked_ar_to_beat": LOCKED_AR,
        "decision/arm_b_beats_126378": int(b_beats),
        "decision/arm_b_margin_tps": b_med - LOCKED_AR,
        "decision/arm_b_margin_pct": 100.0 * (b_med - LOCKED_AR) / LOCKED_AR,
        "decision/ab_verdict_real": int(v["verdict"] == "REAL"),
        "decision/verdict": verdict,
        # ppl gate (instruction #3)
        "ppl/arm_a_no_bi": ppl_a,
        "ppl/arm_b_bi1": ppl_b,
        "ppl/bi1_minus_bi0_delta": (ppl_b - ppl_a) if (ppl_a and ppl_b) else None,
        "ppl/gate_threshold": PPL_GATE,
        "ppl/arm_b_passes_gate": int(bool(ppl_b is not None and ppl_b <= PPL_GATE)),
        "ppl/arm_a_tag": tag_a,
        "ppl/arm_b_tag": tag_b,
        # attribution (instruction #4)
        "attribution/verify_m8_bi1_attn_delta_ms_ctx512": m8,
        "attribution/draft_m1_bi1_attn_cycle_delta_ms_ctx512_upperbound": m1,
        "attribution/ab_step_delta_ms": step_delta_ms,
        "attribution/attn_frac_of_cost_upperbound": attn_frac_upper,
        "attribution/gemm_tax_frac_of_cost_lowerbound": (1 - attn_frac_upper) if attn_frac_upper else None,
        "config/runtime_vllm": "0.22.0",
        "config/drafter": "/tmp/qat-assistant",
        "config/num_speculative_tokens": 7,
        "config/analysis_only": True,
        "config/official_tps": 0,
    }

    run = wandb_logging.init_wandb_run(
        job_type="bi_speed_closeout",
        agent="land",
        name="land/optionb-bi1-closeout",
        group="optionb-bi1-speed-cost",
        notes="PR#623 closeout: BI=1 speed-cost decision + PPL gate + 2D/3D attention attribution. Companion to A/B run 3igf1sq0.",
        config={"pr": 623, "ab_run": "3igf1sq0"},
        tags=["optionb", "batch_invariant", "pr623", "closeout"],
    )
    if run is None:
        print("WANDB disabled/unavailable; nothing logged", flush=True)
        print(json.dumps({k: vv for k, vv in summary.items()}, indent=2, default=str))
        return 1
    wandb_logging.log_summary(run, summary, step=0)
    wandb_logging.log_json_artifact(run, name="optionb_bi1_closeout", artifact_type="analysis",
                                    data={"summary": summary, "ab_verdict": v})
    url = getattr(run, "url", "")
    rid = getattr(run, "id", "")
    wandb_logging.finish_wandb(run)
    print(f"[wandb] closeout run id={rid} url={url}", flush=True)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
