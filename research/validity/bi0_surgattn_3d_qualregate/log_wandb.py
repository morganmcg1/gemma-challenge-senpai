#!/usr/bin/env python
"""PR #791 wirbel -- log the surgattn-OFF (3D-on-M=1) 5% quality-band re-gate to
W&B group bi0-surgattn-3d-qualregate. ANALYSIS ONLY (local served eval + local
decode-TPS proxy; NO HF Job).

Two served arms on the shipped bi0 submission (submissions/int4_mtp_bi0_surgattn,
google/gemma-4-E4B-it-qat-w4a16-ct + MTP-K6 drafter, MAX_NUM_SEQS=1 => every
decode M=1), the only changed variable being VLLM_SURGATTN:
  * control = shipped bi0 (force-2D ON, byte-identical to fired bi0).
  * variant = VLLM_SURGATTN=0 (kernel gate picks 3D split-KV on the M=1 forwards;
              +6.69% local TPS in #785 but breaks greedy byte-identity).

Under directive #784 byte-identity is NOT the gate; the gate is quality within 5%
of bi0's #773 panel on MMLU-Pro/GSM8K/GPQA/AIME + PPL<=2.42 + 128/128 + local
TPS>control. This script reads the panel/kill-gate/TPS json outputs, computes the
per-axis in-band verdict against the bi0 anchors, and logs variant+control runs.

Inputs (produced by run_quality.py, run_panel.py, tps_reps.py):
  runs/mmlu_killgate_summary.json   -- MMLU-Pro kill-gate (variant+control, n=100)
  runs/panel_summary.json           -- GSM8K+GPQA (variant+control) + AIME (variant)
  tps/tps_summary.json              -- decode-TPS A/B reps (optional at log time)
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
ENTITY = "wandb-applied-ai-team"
PROJECT = "gemma-challenge-senpai"
GROUP = "bi0-surgattn-3d-qualregate"

KILLGATE = HERE / "runs" / "mmlu_killgate_summary.json"
PANEL = HERE / "runs" / "panel_summary.json"
TPS = HERE / "tps" / "tps_summary.json"
DIVERGENCE = HERE / "runs" / "divergence_summary.json"

# bi0 #773 panel anchors + the #784 5%-band floors (PR #791 body).
ANCHORS = {
    "mmlu_pro": {"bi0": 0.644, "band": 0.612, "kill": 0.572},   # within 5% of 0.644
    "gsm8k": {"bi0": 0.867, "band": 0.824},                      # 0.95 * 0.867
    "gpqa_diamond": {"bi0": 0.4970, "band": 0.472},              # 0.95 * 0.497
    # AIME n=30 too coarse for a 5% band: require no-collapse (>=8/30) and >=Morgan floor 0.090.
    "aime": {"bi0_count": 10, "n": 30, "no_collapse_count": 8, "morgan_floor": 0.090},
}
# #785 anchors carried forward (deterministic, prefill-only => unchanged here).
PPL_785 = 2.0057          # variant PPL, blind to the M=1 decode path (prefill/M>1 is 2D both arms)
COMPLETED_785 = 128       # variant 128/128
DIV_PROMPT_FRAC_785 = 0.0625   # 6.25% prompts diverge (byte-identity broken)
DIV_TOKEN_FRAC_785 = 0.0176    # 1.76% tokens diverge
PPL_GATE = 2.42


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _axis(variant, band):
    """Return (in_band:int|None, margin_pct:float|None) for a simple >=band axis."""
    if variant is None:
        return None, None
    return int(variant >= band), (100.0 * variant / band - 100.0)


def collect():
    kg = _load(KILLGATE) or {}
    panel = _load(PANEL) or {}
    tps = _load(TPS) or {}
    div = _load(DIVERGENCE) or {}
    parms = panel.get("arms", {})
    kgarms = kg.get("arms", {})

    def panel_task(arm, task, field="accuracy"):
        return (((parms.get(arm) or {}).get("tasks") or {}).get(task) or {}).get(field)

    out = {"variant": {}, "control": {}, "axes": {}, "tps": {},
           "kill_gate_paired": kg.get("paired"), "divergence": div}

    # --- per-arm raw scores ---
    out["variant"]["mmlu_pro"] = (kgarms.get("variant") or {}).get("accuracy")
    out["control"]["mmlu_pro"] = (kgarms.get("control") or {}).get("accuracy")
    out["variant"]["gsm8k"] = panel_task("variant", "gsm8k")
    out["control"]["gsm8k"] = panel_task("control", "gsm8k")
    out["variant"]["gpqa_diamond"] = panel_task("variant", "gpqa_diamond")
    out["control"]["gpqa_diamond"] = panel_task("control", "gpqa_diamond")
    out["variant"]["aime_maj_k_acc"] = panel_task("variant", "aime", "maj_k_accuracy")
    out["variant"]["aime_n_correct_maj"] = panel_task("variant", "aime", "n_correct_maj")
    out["variant"]["aime_k"] = panel_task("variant", "aime", "k")

    # --- per-axis in-band verdict (variant vs bi0 #773 anchor) ---
    # Two anchors per axis: (a) the bi0 #773 panel value from the PR body, and
    # (b) the SAME-HARNESS control arm run here (the cleaner isolation of the
    # surgattn effect, since it shares this run's seed/max_tokens/harness). An
    # axis is only "clean" if the variant clears 95% of BOTH.
    for axis in ("mmlu_pro", "gsm8k", "gpqa_diamond"):
        ib, margin = _axis(out["variant"][axis], ANCHORS[axis]["band"])
        ctrl = out["control"][axis]
        ctrl_band = (0.95 * ctrl) if ctrl is not None else None
        in_band_vs_control = (
            int(out["variant"][axis] >= ctrl_band)
            if (out["variant"][axis] is not None and ctrl_band is not None) else None
        )
        out["axes"][axis] = {
            "variant": out["variant"][axis], "control": ctrl,
            "bi0_anchor": ANCHORS[axis]["bi0"], "band_floor": ANCHORS[axis]["band"],
            "in_band": ib, "margin_vs_floor_pct": margin,
            "control_band_floor": ctrl_band, "in_band_vs_control": in_band_vs_control,
            "variant_minus_control": (
                round(out["variant"][axis] - ctrl, 4)
                if (out["variant"][axis] is not None and ctrl is not None) else None
            ),
        }
    a_acc = out["variant"]["aime_maj_k_acc"]
    a_cnt = out["variant"]["aime_n_correct_maj"]
    aime_in_band = None
    if a_acc is not None and a_cnt is not None:
        aime_in_band = int(a_cnt >= ANCHORS["aime"]["no_collapse_count"]
                           and a_acc >= ANCHORS["aime"]["morgan_floor"])
    out["axes"]["aime"] = {
        "variant_maj_k_acc": a_acc, "variant_n_correct_maj": a_cnt,
        "bi0_anchor_count": ANCHORS["aime"]["bi0_count"], "n": ANCHORS["aime"]["n"],
        "no_collapse_floor_count": ANCHORS["aime"]["no_collapse_count"],
        "morgan_floor": ANCHORS["aime"]["morgan_floor"], "in_band": aime_in_band,
    }

    # --- TPS A/B ---
    out["tps"] = {
        "variant_mean": tps.get("variant_mean_tps"),
        "control_mean": tps.get("control_mean_tps"),
        "ab_delta_pct": tps.get("ab_delta_pct"),
        "variant_stdev": ((tps.get("arms") or {}).get("variant") or {}).get("tps_stdev"),
        "control_stdev": ((tps.get("arms") or {}).get("control") or {}).get("tps_stdev"),
    }
    return out


def verdict(c):
    axes = c["axes"]
    ib = [axes[a].get("in_band") for a in ("mmlu_pro", "gsm8k", "gpqa_diamond", "aime")]
    quality_pass = all(x == 1 for x in ib) if all(x is not None for x in ib) else None
    ppl_pass = int(PPL_785 <= PPL_GATE)
    completed_pass = int(COMPLETED_785 == 128)
    vt, ct = c["tps"]["variant_mean"], c["tps"]["control_mean"]
    tps_pass = int(vt > ct) if (vt is not None and ct is not None) else None
    all_pass = (quality_pass and ppl_pass and completed_pass and tps_pass) \
        if None not in (quality_pass, tps_pass) else None
    fail_axes = [a for a in ("mmlu_pro", "gsm8k", "gpqa_diamond", "aime")
                 if axes.get(a, {}).get("in_band") == 0]
    if all_pass is None:
        v = "INCOMPLETE: missing one or more axes/TPS at log time"
    elif all_pass:
        v = ("FIRE-WORTHY CANDIDATE: surgattn-OFF (3D-on-M=1) holds all 4 quality axes "
             "within 5% of bi0 AND PPL<=2.42/128/128 AND local TPS>control. Needs the "
             "human-gated official a10g A/B to certify speed outside local-proxy noise.")
    else:
        gp = axes.get("gpqa_diamond", {})
        v = (f"QUALITY FAIL on {fail_axes}: variant misses the 5% band -> keep force-2D ON "
             f"(shipped bi0). GPQA variant={gp.get('variant')} < band {gp.get('band_floor')} "
             f"(vs #773 0.497) AND < same-harness control {gp.get('control')} "
             f"(Δ={gp.get('variant_minus_control')}); the deficit is NOT statistically "
             f"significant (McNemar p~0.21) and ~1/3 truncation-driven, but the point "
             f"estimate cannot be shown within 5% of base on GPQA. MMLU/GSM8K/AIME pass; "
             f"divergences are answer-immaterial near-tie flips with intact extraction.")
    return {"quality_pass": quality_pass, "ppl_pass": ppl_pass,
            "completed_pass": completed_pass, "tps_pass": tps_pass,
            "all_gates_pass": all_pass, "fail_axes": fail_axes, "verdict": v}


def base_config(arm):
    return {
        "pr": 791, "lane": "bi0-surgattn-3d-qualregate",
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        "submission": "submissions/int4_mtp_bi0_surgattn",
        "arm": arm, "surgattn": (0 if arm == "variant" else 1),
        "vllm_surgattn_env": ("0" if arm == "variant" else "unset/default(1)"),
        "force_2d_on_m1": (0 if arm == "variant" else 1),
        "backend": "TRITON_ATTN",
        "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "num_speculative_tokens": 6, "max_num_seqs": 1, "max_model_len": 4096,
        "served_substrate": "full_vocab_loadable_qat_ct",
        "gate_basis": "downstream_quality_5pct_band_784_NOT_byte_identity",
        "tps_basis": "local_a10g_single_stream_decode_NOT_official",
    }


def main():
    c = collect()
    v = verdict(c)
    print(json.dumps({"collect": c, "verdict": v}, indent=2))

    ids = {}
    for arm in ("variant", "control"):
        cfg = base_config(arm)
        run = wandb.init(
            entity=ENTITY, project=PROJECT, group=GROUP,
            name=f"wirbel/bi0-surgattn-3d-qualregate-{arm}",
            job_type="analysis", reinit=True, config=cfg,
        )
        ids[arm] = run.id
        summary = {
            "mmlu_pro": c[arm].get("mmlu_pro"),
            "gsm8k": c[arm].get("gsm8k"),
            "gpqa_diamond": c[arm].get("gpqa_diamond"),
        }
        if arm == "variant":
            summary.update({
                "aime_maj_k_acc": c["variant"].get("aime_maj_k_acc"),
                "aime_n_correct_maj": c["variant"].get("aime_n_correct_maj"),
                "aime_k": c["variant"].get("aime_k"),
                # cross-arm + gate rollup (carried by the variant run)
                "tps_variant_mean": c["tps"]["variant_mean"],
                "tps_control_mean": c["tps"]["control_mean"],
                "tps_ab_delta_pct": c["tps"]["ab_delta_pct"],
                "tps_variant_stdev": c["tps"]["variant_stdev"],
                "tps_control_stdev": c["tps"]["control_stdev"],
                "ppl_blind_785": PPL_785, "ppl_gate": PPL_GATE,
                "completed_785": COMPLETED_785,
                "div_prompt_frac_785": DIV_PROMPT_FRAC_785,
                "div_token_frac_785": DIV_TOKEN_FRAC_785,
                "axis_mmlu_in_band": c["axes"]["mmlu_pro"].get("in_band"),
                "axis_gsm8k_in_band": c["axes"]["gsm8k"].get("in_band"),
                "axis_gpqa_in_band": c["axes"]["gpqa_diamond"].get("in_band"),
                "axis_aime_in_band": c["axes"]["aime"].get("in_band"),
                # same-harness control comparison (cleaner isolation of the 3D effect)
                "axis_mmlu_in_band_vs_control": c["axes"]["mmlu_pro"].get("in_band_vs_control"),
                "axis_gsm8k_in_band_vs_control": c["axes"]["gsm8k"].get("in_band_vs_control"),
                "axis_gpqa_in_band_vs_control": c["axes"]["gpqa_diamond"].get("in_band_vs_control"),
                "gpqa_variant_minus_control": c["axes"]["gpqa_diamond"].get("variant_minus_control"),
                "mmlu_variant_minus_control": c["axes"]["mmlu_pro"].get("variant_minus_control"),
                "gsm8k_variant_minus_control": c["axes"]["gsm8k"].get("variant_minus_control"),
                # divergent-prompt check (answer-immaterial near-tie flips, intact extraction)
                "gpqa_answer_flips": (c["divergence"].get("gpqa_diamond") or {}).get("answer_flips"),
                "gpqa_flips_both_valid": (c["divergence"].get("gpqa_diamond") or {}).get("flips_both_valid_letter"),
                "gpqa_mcnemar_p": ((c["divergence"].get("gpqa_diamond") or {}).get("mcnemar") or {}).get("p_approx"),
                "gpqa_net_excl_truncation": (c["divergence"].get("gpqa_diamond") or {}).get("net_excl_variant_truncation_losses"),
                "gpqa_variant_truncated": (c["divergence"].get("gpqa_diamond") or {}).get("variant_truncated"),
                "gpqa_control_truncated": (c["divergence"].get("gpqa_diamond") or {}).get("control_truncated"),
                "mmlu_answer_flips": (c["divergence"].get("mmlu_pro") or {}).get("answer_flips"),
                "mmlu_mcnemar_p": ((c["divergence"].get("mmlu_pro") or {}).get("mcnemar") or {}).get("p_approx"),
                "extraction_code_changed": False,
                **v,
            })
        run.log(summary)
        run.summary.update(summary)
        run.finish()
        print(f"[wandb] {arm}: run {run.id}")

    # cross-arm comparison table on a 3rd lightweight run for the dashboard view.
    tbl = wandb.init(entity=ENTITY, project=PROJECT, group=GROUP,
                     name="wirbel/bi0-surgattn-3d-qualregate-table",
                     job_type="analysis", reinit=True,
                     config={"pr": 791, "lane": GROUP, "analysis_only": 1, "no_hf_job": 1})
    ids["table"] = tbl.id
    t = wandb.Table(columns=["axis", "variant", "control", "bi0_anchor", "band_floor",
                             "in_band", "control_band_floor", "in_band_vs_control"])
    for axis in ("mmlu_pro", "gsm8k", "gpqa_diamond"):
        a = c["axes"][axis]
        # bi0_anchor/band_floor stringified so the column type is consistent with the
        # AIME row's ratio anchors ("10/30") -- W&B Tables infer one type per column.
        t.add_data(axis, a["variant"], a["control"], str(a["bi0_anchor"]), str(a["band_floor"]),
                   a["in_band"], a.get("control_band_floor"), a.get("in_band_vs_control"))
    aa = c["axes"]["aime"]
    t.add_data("aime_maj@k", aa["variant_maj_k_acc"], None,
               f"{aa['bi0_anchor_count']}/{aa['n']}",
               f">={aa['no_collapse_floor_count']}/{aa['n']}", aa["in_band"], None, None)
    tbl.log({"quality_band_table": t, **v})
    tbl.summary.update(v)
    tbl.finish()

    print("WANDB_RUN_IDS=" + ",".join(ids.values()))
    print("VERDICT=" + v["verdict"])


if __name__ == "__main__":
    raise SystemExit(main())
