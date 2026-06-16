#!/usr/bin/env python
"""Log the PR #535 base-fullhead-fast-ship probe to W&B (analysis-only).

Reads the three local artifacts produced by this probe and logs one summary run
under group ``base-fullhead-fast-ship-probe`` with the KEY OUTPUTS the advisor
asked for: serve_ok, warm-median TPS, the TPS cost vs the osoi5 ship (both the
cited 357.06 and the same-pod local osoi5 number), AIME greedy maj@1 and its
fraction of the base anchor, served PPL vs the <=2.42 gate, self-determinism,
peak GPU, and the one-line ``quality_safe_fast_ship_exists`` / ``rebake_required``
verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

OUT = ROOT / "research" / "base_fullhead_fast_probe"

# Anchors (given by the probe instructions / prior A/B aime_ab_greedy.json).
BASE_AIME = 0.26666666666666666          # base int4 plain greedy maj@1 (n=30)
SHIP_AIME = 0.03333333333333333          # osoi5-12k ship greedy maj@1 (collapsed)
CITED_OSOI5_TPS = 357.06                 # cited surgical-357 official ship TPS
PPL_BASE = 2.0190133619612727            # base int4 plain served PPL (same dataset)
PPL_GATE = 2.42                          # probe quality gate


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    base = load(OUT / "phase1_base_fullhead.json")
    osoi5 = load(OUT / "phase1_osoi5_ship.json")
    # HEADLINE AIME is the protocol-matched conc=32 run (the base 0.267 anchor was
    # taken at MAX_NUM_SEQS=32). The conc=1 spec-on / conc=1 spec-off (M=1 AR) /
    # plain-int4 conc=1 runs are carried as supplementary attribution points.
    aime = load(OUT / "aime_base_fullhead_conc32.json")
    aime_c1_specon = load(OUT / "aime_base_fullhead_greedy.json")
    aime_c1_specoff = load(OUT / "aime_base_fullhead_specoff_greedy.json")
    aime_plain_c1 = load(OUT / "aime_int4_plain_conc1.json")

    serve_ok = bool(base.get("serve_ok"))
    tps = base.get("warm_median_tps")
    ppl = base.get("ppl")
    self_det = base.get("self_det")
    peak_gpu_mib = base.get("peak_gpu_mib")

    osoi5_local_tps = osoi5.get("warm_median_tps")
    osoi5_self_det = osoi5.get("self_det")

    aime_acc = aime.get("maj_k_accuracy")
    n_problems = aime.get("n_problems")
    n_correct = aime.get("n_correct_maj")
    aime_by_config = {
        "conc32_specon_PROTOCOL": aime.get("maj_k_accuracy"),
        "conc1_specon": aime_c1_specon.get("maj_k_accuracy"),
        "conc1_specoff_M1_AR": aime_c1_specoff.get("maj_k_accuracy"),
        "plain_int4_conc1_control": aime_plain_c1.get("maj_k_accuracy"),
        "plain_int4_conc32_anchor": BASE_AIME,
        "osoi5_ship_conc32_collapsed": SHIP_AIME,
    }
    aime_extract_fail = {
        "base_fullhead_conc32": aime.get("extract_fail_rate"),
        "base_fullhead_conc1_specon": aime_c1_specon.get("extract_fail_rate"),
        "base_fullhead_conc1_specoff": aime_c1_specoff.get("extract_fail_rate"),
        "plain_int4_conc1": aime_plain_c1.get("extract_fail_rate"),
    }

    # Derived KEY OUTPUTS.
    aime_pct_of_base = (aime_acc / BASE_AIME) if (aime_acc is not None) else None
    tps_cost_vs_cited = (tps - CITED_OSOI5_TPS) if (tps is not None) else None
    tps_frac_of_cited = (tps / CITED_OSOI5_TPS) if (tps is not None) else None
    tps_cost_vs_local = (
        (tps - osoi5_local_tps) if (tps is not None and osoi5_local_tps) else None
    )
    tps_frac_of_local = (
        (tps / osoi5_local_tps) if (tps is not None and osoi5_local_tps) else None
    )
    ppl_pass = (ppl is not None and ppl <= PPL_GATE)

    # CONCURRENCY-CONFOUND on the n=30 AIME metric. The plain stock base model,
    # unchanged, swings maj@1 from 0.100 (conc=1) to 0.267 (conc=32) — a 0.167
    # spread (>2se) driven purely by long-chain greedy batch-numerics. The fast
    # stack is conc-STABLE at 0.167 and sits INSIDE that band, so comparing
    # base_fullhead@conc1 vs the conc=32 anchor conflated the fast-stack effect
    # with concurrency. At n=30 the metric cannot adjudicate the >=90% bar.
    aime_plain_conc1 = aime_plain_c1.get("maj_k_accuracy")
    aime_plain_conc_spread = (
        (BASE_AIME - aime_plain_conc1) if (aime_plain_conc1 is not None) else None
    )
    se_var = (BASE_AIME * (1 - BASE_AIME) / n_problems) ** 0.5 if n_problems else None
    aime_metric_conc_confounded = bool(
        aime_plain_conc_spread is not None and se_var is not None
        and aime_plain_conc_spread > 2 * se_var
    )
    aime_base_fullhead_within_plain_conc_band = bool(
        aime_plain_conc1 is not None and aime_acc is not None
        and min(aime_plain_conc1, BASE_AIME) <= aime_acc <= max(aime_plain_conc1, BASE_AIME)
    )

    # The advisor's EXPLICIT verdict bar (#535 / Morgan #515 quality gate):
    # AIME maj@1 >= 90% of the base anchor. Point estimate, not a noise band.
    QUALITY_BAR_FRAC = 0.90
    aime_clears_90pct_bar = (
        aime_pct_of_base is not None and aime_pct_of_base >= QUALITY_BAR_FRAC
    )

    # 2-se band around the base anchor (binomial, n=30): the n=30 POWER caveat.
    # base_fullhead 0.167 is within 2se of 0.267 (Δ=0.10=1.24se) -> the eval can
    # neither confirm >=90% nor prove a real drop; it is underpowered for this call.
    se = (BASE_AIME * (1 - BASE_AIME) / n_problems) ** 0.5 if n_problems else None
    aime_within_noise = (
        (aime_acc is not None and se is not None and abs(aime_acc - BASE_AIME) <= 2 * se)
    )
    # NOT collapsed: categorically unlike the osoi5 16k-prune floor (0.033) — base
    # int4 native head stays coherent (extract_fail low) and >> 0.5*base.
    aime_not_collapsed = (aime_acc is not None and aime_acc >= 0.5 * BASE_AIME)

    # Verdict on the advisor's stated >=90%-of-base bar (point estimate). FALSE
    # here: serves + PPL-exact + not-collapsed, but AIME=0.167=62.5% misses 90%.
    quality_safe_fast_ship_exists = bool(serve_ok and ppl_pass and aime_clears_90pct_bar)

    # rebake_required is the FEASIBILITY gate the PR tied it to: do the surgical/
    # split-KV/MTP kernels NEED the osoi5 layout (i.e. fail to bind on base int4)?
    # They do NOT — serve_ok=True on the native 262k head — so rebake is NOT
    # required for the fast stack to RUN. (Distinct from the quality verdict.)
    rebake_required = not serve_ok

    # Serves + PPL-exact + not-collapsed, but the AIME point estimate misses the
    # 90% bar (within 2se at n=30): a full-262k QAT rebake (kernel-adapted native
    # head) OR a higher-n AIME eval is the recommended path to confirm/recover
    # native-head quality. Captures the third state the PR's binary didn't name.
    rebake_recommended_for_quality = bool(serve_ok and not aime_clears_90pct_bar)

    summary = {
        "serve_ok": serve_ok,
        "base_fullhead_surgical_warm_median_tps": tps,
        "tps_runs": base.get("tps_runs"),
        "osoi5_ship_local_warm_median_tps": osoi5_local_tps,
        "cited_osoi5_ship_tps": CITED_OSOI5_TPS,
        "tps_cost_vs_osoi5_ship_cited": tps_cost_vs_cited,
        "tps_frac_of_osoi5_ship_cited": tps_frac_of_cited,
        "tps_cost_vs_osoi5_ship_local": tps_cost_vs_local,
        "tps_frac_of_osoi5_ship_local": tps_frac_of_local,
        "aime_greedy_base_fullhead": aime_acc,
        "aime_n_correct_maj": n_correct,
        "aime_n_problems": n_problems,
        "aime_pct_of_base": aime_pct_of_base,
        "aime_base_anchor": BASE_AIME,
        "aime_osoi5_ship_collapsed": SHIP_AIME,
        "aime_quality_bar_frac": QUALITY_BAR_FRAC,
        "aime_clears_90pct_bar": aime_clears_90pct_bar,
        "aime_within_noise_of_base": aime_within_noise,
        "aime_not_collapsed": aime_not_collapsed,
        "aime_plain_conc1": aime_plain_conc1,
        "aime_plain_conc_spread": aime_plain_conc_spread,
        "aime_metric_conc_confounded": aime_metric_conc_confounded,
        "aime_base_fullhead_within_plain_conc_band": aime_base_fullhead_within_plain_conc_band,
        "aime_by_config": aime_by_config,
        "aime_extract_fail_by_config": aime_extract_fail,
        "ppl": ppl,
        "ppl_base_reference": PPL_BASE,
        "ppl_gate": PPL_GATE,
        "ppl_pass": ppl_pass,
        "self_det_base_fullhead": self_det,
        "self_det_osoi5_ship_control": osoi5_self_det,
        "peak_gpu_mib": peak_gpu_mib,
        "quality_safe_fast_ship_exists": quality_safe_fast_ship_exists,
        "rebake_required": rebake_required,
        "rebake_recommended_for_quality": rebake_recommended_for_quality,
    }

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 535,
        "experiment": "base-fullhead-fast-ship-probe",
        "substrate": "base_int4_native_262k_head",
        "submission": base.get("submission"),
        "base_serve_overrides": base.get("serve_overrides"),
        "fast_stack": "surgical_2d_attn + mtp_k7_spec + splitkv + onegraph + ple_fold",
    }

    print("PROBE535_SUMMARY " + json.dumps(summary, default=str))

    run = wandb_logging.init_wandb_run(
        job_type="base-fullhead-fast-ship-probe",
        agent="fern",
        name="fern/base-fullhead-fast-ship-probe",
        group="base-fullhead-fast-ship-probe",
        notes=(
            "PR #535: does a quality-safe FAST ship exist on base int4 (native 262k "
            "head)? Fast kernel stack served on stock w4a16 weights, no osoi5 bake, "
            "no head prune. Analysis-only; official_tps=0."
        ),
        tags=["aime", "tps", "ppl", "analysis-only", "pr-535", "base-fullhead"],
        config=config,
    )
    if run is None:
        print("[probe535] wandb disabled/unavailable; metrics above + JSON only", flush=True)
        (OUT / "probe535_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return 0

    wandb_logging.log_summary(run, summary, step=0)
    wandb_logging.log_json_artifact(run, name="phase1_base_fullhead", artifact_type="probe-535", data=base)
    if osoi5:
        wandb_logging.log_json_artifact(run, name="phase1_osoi5_ship", artifact_type="probe-535", data=osoi5)
    if aime:
        aime_slim = {k: v for k, v in aime.items() if k not in ("samples", "per_problem")}
        wandb_logging.log_json_artifact(run, name="aime_base_fullhead_greedy", artifact_type="probe-535", data=aime_slim)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    print(f"[probe535] wandb run_id={run_id}", flush=True)
    (OUT / "probe535_summary.json").write_text(json.dumps({**summary, "wandb_run_id": run_id}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
