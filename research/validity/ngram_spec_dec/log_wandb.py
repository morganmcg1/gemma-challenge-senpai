"""PR #503 — consolidated W&B logger for the n-gram prompt-lookup spec-dec probe.

Logs ONE analysis run with the full record:
  * per-config table (acceptance / e_accept / steady+walltime+probe TPS / lift /
    operative-identity vs AR / flip-rate) for the 11-config public screen;
  * private acceptance table (config x {public,easy,hard}) + per-config private-Δ;
  * pairwise operative-identity census (AR-vs-AR' noise floor, ngram/mtp-vs-AR,
    ngram-vs-mtp matched-width drafter isolation);
  * headline scalars in run.summary.

Local A10G probe — NOT official a10g-small TPS. analysis_only; official_tps=0.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

D = Path("/workspace/senpai/target/research/validity/ngram_spec_dec")


def _load(name: str) -> dict[str, Any]:
    p = D / name
    return json.loads(p.read_text()) if p.exists() else {}


def _safe(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _acc(cfgs: dict, name: str) -> dict:
    return (cfgs.get(name, {}) or {}).get("acceptance", {}) or {}


def _walltime(cfgs: dict, name: str):
    return (cfgs.get(name, {}) or {}).get("decode_tps_walltime")


def main() -> int:
    analysis = _load("analysis.json")
    census = _load("census_pairs_public.json")
    pub = _load("results_public_screen.json").get("configs", {})
    easy = _load("results_private_easy.json").get("configs", {})
    hard = _load("results_private_hard.json").get("configs", {})
    ctrl = _load("results_ctrl_ar2.json").get("configs", {})

    summary = analysis.get("summary", {})
    rows = summary.get("rows", {})
    pd_ngram = analysis.get("private_delta", {})
    pd_mtp = analysis.get("private_delta_mtp", {})

    # Clean AR floor: walltime agrees across two byte-identical sessions.
    ar_wall = _walltime(pub, "ar_floor")
    ar2_wall = _walltime(ctrl, "ar_floor2")
    ar_floor_wall = next((v for v in (ar_wall, ar2_wall) if v), None)

    # best ngram (by steady TPS and by acceptance)
    ng = {k: v for k, v in rows.items() if k.startswith("ngram")}
    best_tps_cfg = max(ng, key=lambda k: ng[k].get("steady_gen_tps") or 0)
    best_acc_cfg = max(ng, key=lambda k: ng[k].get("acceptance_rate") or 0)

    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable ({exc})")
        return 1

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        name="ubel/ngram-prompt-lookup-spec-dec",
        group="ngram-prompt-lookup-spec-dec",
        job_type="analysis",
        config={
            "analysis_only": True,
            "official_tps": 0,
            "stack": "fa2sw_precache_kenyan (deployed MTP-K7 base, int4/fa2sw/precache/splitkv/lmhead12k)",
            "verify_stack_byte_identical_across_configs": True,
            "only_drafter_varies": "SPECULATIVE_CONFIG",
            "prompts": "32 x 256 greedy; public ShareGPT + #497 easy/hard splits",
            "vllm": "0.22.1rc1.dev307",
            "ngram_async_scheduling_disabled": True,
            "note": "local A10G sm_86 probe; NOT official a10g-small TPS",
        },
    )

    # --- per-config public table ---
    t = wandb.Table(columns=[
        "config", "acceptance_rate", "e_accept", "steady_tps", "walltime_tps",
        "probe_tps", "lift_steady", "operative_identity_token_vs_ar",
        "semantic_flip_rate_vs_ar"])
    for name, r in rows.items():
        cen = r.get("census_vs_ar") or {}
        t.add_data(
            name, _safe(r.get("acceptance_rate")), _safe(r.get("e_accept")),
            _safe(r.get("steady_gen_tps")), _safe(_walltime(pub, name)),
            _safe(r.get("probe_decode_tps")), _safe(r.get("lift_steady")),
            _safe(cen.get("operative_identity_token_level")),
            _safe(cen.get("semantic_flip_rate_token_level")))
    run.log({"public_per_config": t})

    # --- private acceptance table (config x split) ---
    pt = wandb.Table(columns=[
        "config", "public_acc", "easy_acc", "hard_acc", "public_e_accept",
        "easy_e_accept", "hard_e_accept", "delta_pub_easy_pp", "delta_pub_hard_pp",
        "max_abs_private_delta_pp", "breach_shrinks"])
    for cfg, pdd in (("ngram_n2_k3", pd_ngram), ("mtp_k7", pd_mtp)):
        pt.add_data(
            cfg, _safe(pdd.get("public_acceptance_rate")), _safe(pdd.get("easy_acceptance_rate")),
            _safe(pdd.get("hard_acceptance_rate")), _safe(pdd.get("public_e_accept")),
            _safe(pdd.get("easy_e_accept")), _safe(pdd.get("hard_e_accept")),
            _safe(pdd.get("delta_public_minus_easy_pct")), _safe(pdd.get("delta_public_minus_hard_pct")),
            _safe(pdd.get("max_abs_private_delta_pct")), pdd.get("private_breach_shrinks"))
    # ngram_n2_k7 (computed inline)
    def _split_acc(cfgs, n):
        return _acc(cfgs, n).get("acceptance_rate")
    n27_pub, n27_easy, n27_hard = (_split_acc(pub, "ngram_n2_k7"),
                                   _split_acc(easy, "ngram_n2_k7"), _split_acc(hard, "ngram_n2_k7"))
    if None not in (n27_pub, n27_easy, n27_hard):
        de, dh = 100 * (n27_pub - n27_easy), 100 * (n27_pub - n27_hard)
        pt.add_data("ngram_n2_k7", n27_pub, n27_easy, n27_hard, None, None, None,
                    de, dh, max(abs(de), abs(dh)), max(abs(de), abs(dh)) < 4.295)
    run.log({"private_acceptance": pt})

    # --- pairwise census table ---
    ct = wandb.Table(columns=["pair", "prompt_identity", "token_identity",
                              "token_flip_rate", "min_first_divergence"])
    for pair, c in (census.get("pairs", {}) or {}).items():
        ct.add_data(pair, _safe(c.get("prompt_identity")), _safe(c.get("token_identity")),
                    _safe(c.get("token_flip_rate")), c.get("min_first_divergence"))
    run.log({"pairwise_census": ct})

    # --- headline scalars ---
    flat = {
        "ar_floor_walltime_tps": _safe(ar_floor_wall),
        "ar_xsession_operative_identity": _safe((census.get("pairs", {}).get("ar_floor2:ar_floor", {}) or {}).get("token_identity")),
        "ar_xsession_flip_rate": _safe((census.get("pairs", {}).get("ar_floor2:ar_floor", {}) or {}).get("token_flip_rate")),
        "mtp_steady_tps": _safe(rows.get("mtp_k7", {}).get("steady_gen_tps")),
        "mtp_walltime_tps": _safe(_walltime(pub, "mtp_k7")),
        "mtp_lift_steady": _safe(rows.get("mtp_k7", {}).get("lift_steady")),
        "mtp_acceptance_rate": _safe(rows.get("mtp_k7", {}).get("acceptance_rate")),
        "best_ngram_cfg_by_tps": best_tps_cfg,
        "best_ngram_steady_tps": _safe(ng[best_tps_cfg].get("steady_gen_tps")),
        "best_ngram_lift_steady": _safe(ng[best_tps_cfg].get("lift_steady")),
        "best_ngram_cfg_by_acc": best_acc_cfg,
        "best_ngram_acceptance_rate": _safe(ng[best_acc_cfg].get("acceptance_rate")),
        "ngram_beats_ar_floor": (ng[best_tps_cfg].get("lift_steady") or 0) > 1.0,
        # strict-safety: ngram vs MTP on the M>1 verify tax
        "ngram_flip_vs_ar": _safe((census["pairs"].get("ngram_n2_k3:ar_floor", {}) or {}).get("token_flip_rate")),
        "mtp_flip_vs_ar": _safe((census["pairs"].get("mtp_k7:ar_floor", {}) or {}).get("token_flip_rate")),
        "ngram_vs_mtp_matched_flip": _safe((census["pairs"].get("ngram_n2_k7:mtp_k7", {}) or {}).get("token_flip_rate")),
        # headline private robustness
        "ngram_max_private_accept_delta_pp": _safe(pd_ngram.get("max_abs_private_delta_pct")),
        "mtp_max_private_accept_delta_pp": _safe(pd_mtp.get("max_abs_private_delta_pct")),
        "mtp_reference_delta_pp_denken489": 4.295,
        "private_breach_shrinks": pd_ngram.get("private_breach_shrinks"),
    }
    run.summary.update(flat)
    print(f"[wandb] logged run {run.id} ({run.url})")
    run.finish()
    print(f"RUN_ID={run.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
