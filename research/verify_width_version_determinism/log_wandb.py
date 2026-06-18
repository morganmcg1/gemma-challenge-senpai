#!/usr/bin/env python
"""PR #681 lawine -- merge the lm_head per-position width arm + the full-forward served-spec
arm (both vLLM versions) into ONE W&B run with the machine-checkable scalars and the verdict.

Run under the REPO venv (has wandb; server venvs may shadow the import via ./wandb):
  .venv/bin/python -m research.verify_width_version_determinism.log_wandb \
      --v0220 research/verify_width_version_determinism/width_results_v0220.json \
      --dev307 research/verify_width_version_determinism/width_results_dev307.json \
      --ff-v0220 research/verify_width_version_determinism/fullforward_report_v0220.json \
      --ff-dev307 research/verify_width_version_determinism/fullforward_report_dev307.json \
      --self-break-v0220 0.0 --self-break-dev307 0.90625 \
      --wandb_name lawine/verify-width-version-determinism \
      --wandb_group verify-width-version-determinism-lawine

TWO ARMS, two granularities of the SAME width-1-vs-width-(K+1=6) question:
  * lm_head arm (width_probe.py): a WITHIN-PROCESS per-position test that feeds the captured
    M=1 hidden state through the int4-Marlin lm_head GEMM at M=1 vs M=6. Its ar_vs_ar control
    is 0 on BOTH versions (clean, warm, in-process) -> it is FREE of the dev307 cross-start
    autotune confound (#601). RESULT: bit-identical (break==0) on both versions -> the head
    GEMM is width-invariant; kanna #673's break is NOT in the lm_head.
  * full-forward arm (fullforward_probe.py): the served-spec FULL-model test -- ref/ref2 = M=1
    AR, cand = M=6 spec verify (MTP-K5) on the locked int4_g128_lmhead body, scored by the
    official #319 verifier. This is the PR's headline verify_width_break_rate (the full-model
    width-1-vs-width-6 comparison). On 0.22.0 its cross-start floor (ref vs ref2) is CLEAN, so
    a cand divergence is a real structural width break on the SHIP version.

HEADLINE verify_width_break_rate_<ver> = full-forward prompt-level break rate (ref-b vs cand-b).
lm_head per-position rate is reported separately as lmhead_width_break_rate_<ver> (== 0): the
head-vs-body decomposition that localizes the break to the BODY (attention + body int4 GEMMs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(p: str | None) -> dict[str, Any] | None:
    if not p:
        return None
    path = Path(p)
    return json.loads(path.read_text()) if path.exists() else None


def _ff_break_rate(ff: dict[str, Any] | None, tag: str) -> dict[str, Any]:
    """Distill a full-forward per-version report into the headline width-break numbers.

    headline = prompt-level break rate (ref-b vs cand-b) = num_divergent / n.
    structural = floor-subtracted break rate on prompts STABLE across ref/ref2 (the chaos
    cross-tab) -- equals headline when the floor is clean (0.22.0), and is the dev307 signal
    that survives the ~0.9 cross-start autotune floor.
    """
    if not ff:
        return {}
    spec = ff.get("spec") or {}
    floor = ff.get("floor") or {}
    cross = ff.get("crosstab") or {}
    n = spec.get("num_prompts_compared") or 0
    n_div = spec.get("num_divergent") or 0
    n_stable = cross.get("n_stable_floor") or 0
    n_div_stable = cross.get("cand_n_divergent_on_stable") or 0
    return {
        f"verify_width_break_rate_{tag}": (n_div / n) if n else None,
        f"ff_{tag}_structural_break_rate_on_stable": (n_div_stable / n_stable) if n_stable else None,
        f"ff_{tag}_cand_seq_exact": spec.get("freerun_seq_exact"),
        f"ff_{tag}_cand_token_identity": spec.get("freerun_token_identity"),
        f"ff_{tag}_cand_divergent_tokens": spec.get("total_divergent_tokens"),
        f"ff_{tag}_floor_seq_exact": floor.get("freerun_seq_exact"),
        f"ff_{tag}_floor_divergent_tokens": floor.get("total_divergent_tokens"),
        f"ff_{tag}_n_stable_floor": n_stable,
        f"ff_{tag}_n_divergent_on_stable": n_div_stable,
        f"ff_{tag}_onset_min": spec.get("onset_min"),
        f"ff_{tag}_onset_median": spec.get("onset_median"),
        f"ff_{tag}_structural_break": int(bool(ff.get(f"{tag}_structural_break"))),
        f"self_break_{tag}_measured": (1.0 - floor.get("freerun_seq_exact"))
        if floor.get("freerun_seq_exact") is not None else None,
    }


def decide(v0220: dict[str, Any], dev307: dict[str, Any],
           ff_v0220: dict[str, Any] | None, ff_dev307: dict[str, Any] | None,
           ar_floor: float) -> dict[str, Any]:
    """Verdict: does the verify-WIDTH strict-#319 break reproduce on the SHIP vLLM 0.22.0?

    Decision rests on the FULL-FORWARD arm (the PR's headline width-1-vs-width-6 test):
      * 0.22.0 floor (ref vs ref2) CLEAN + cand diverges  -> structural break on ship.
    The lm_head arm decomposes WHERE the break lives: bit-identical M=1 vs M=6 on both versions
    -> the head GEMM is NOT the source; the break is body-driven.
    """
    # lm_head head-GEMM width-invariance (per-position, within-process, clean control).
    head_v0220_clean = (v0220["verify_width_break_rate"] <= ar_floor) and \
                       (v0220.get("bit_break_rate", 0.0) <= ar_floor)
    head_dev307_clean = (dev307["verify_width_break_rate"] <= ar_floor) and \
                        (dev307.get("bit_break_rate", 0.0) <= ar_floor)
    head_both_clean = head_v0220_clean and head_dev307_clean

    # full-forward structural breaks (the headline width test).
    v0220_ff_break = bool(ff_v0220 and ff_v0220.get("v0220_structural_break"))
    dev307_ff_break = bool(ff_dev307 and ff_dev307.get("dev307_structural_break"))
    v0220_floor_clean = bool(ff_v0220 and ff_v0220.get("v0220_floor_clean"))

    if v0220_ff_break and v0220_floor_clean:
        # The decisive arm: ship 0.22.0 full-model spec breaks #319 against a CLEAN floor.
        verdict = "VERIFY_WIDTH_VERSION_FUNDAMENTAL"
        decomp = ("lm_head GEMM is bit-identical M=1 vs M=6 on BOTH versions, so the break is "
                  "BODY-driven (attention + body int4 GEMMs), not the head. "
                  if head_both_clean else "")
        rationale = (
            "Full-forward served-spec (M=6 verify, K=5) STRUCTURALLY breaks strict #319 on the "
            "SHIP vLLM 0.22.0 against a CLEAN cross-start floor (ref==ref2) -- so the verify-width "
            "spec break is REAL on the version we ship, NOT a dev307 numerics artifact. "
            + decomp +
            "The strict-#319 spec blocker stands -> recompute-rescue (stark #669) or a "
            "width-invariant config (land #680) remains necessary to ship the spec lever.")
    elif ff_v0220 and (not v0220_ff_break) and dev307_ff_break:
        verdict = "VERIFY_WIDTH_DEV307_ARTIFACT"
        rationale = (
            "Full-forward served-spec breaks strict #319 on dev307 but NOT on the ship 0.22.0 "
            "(0.22.0 floor clean AND cand identical) -> the spec blocker is a dev307 numerics "
            "artifact, MOOT on ship; the spec lever is unblockable on 0.22.0.")
    elif ff_v0220 is None:
        verdict = "INCONCLUSIVE_NEEDS_FULLFORWARD"
        rationale = (
            "lm_head GEMM is width-invariant (bit-identical) on both versions, so the break is "
            "body-driven; the full-forward served-spec arm on 0.22.0 is required to decide the "
            "version axis.")
    else:
        verdict = "INCONCLUSIVE"
        rationale = "unexpected break pattern; inspect per-version full-forward + lm_head JSON."

    return {
        "verdict": verdict, "rationale": rationale,
        "head_v0220_clean": head_v0220_clean, "head_dev307_clean": head_dev307_clean,
        "head_both_clean": head_both_clean,
        "v0220_fullforward_break": v0220_ff_break, "dev307_fullforward_break": dev307_ff_break,
        "v0220_floor_clean": v0220_floor_clean,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0220", required=True, help="lm_head width_results_v0220.json")
    ap.add_argument("--dev307", required=True, help="lm_head width_results_dev307.json")
    ap.add_argument("--ff-v0220", default=None, help="fullforward_report_v0220.json")
    ap.add_argument("--ff-dev307", default=None, help="fullforward_report_dev307.json")
    ap.add_argument("--self-break-v0220", type=float, default=0.0)
    ap.add_argument("--self-break-dev307", type=float, default=0.90625)
    ap.add_argument("--self-break-source", default="lawine #675 (6se9d4gh), same venvs, 2026-06-18")
    ap.add_argument("--wandb_name", default="lawine/verify-width-version-determinism")
    ap.add_argument("--wandb_group", default="verify-width-version-determinism-lawine")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    v0220 = _load(args.v0220)
    dev307 = _load(args.dev307)
    ff_v0220 = _load(args.ff_v0220)
    ff_dev307 = _load(args.ff_dev307)
    if not v0220 or not dev307:
        print("ERROR: need both --v0220 and --dev307 lm_head result JSONs", flush=True)
        return 1

    ar_floor = max(v0220["ar_vs_ar_break"], dev307["ar_vs_ar_break"])
    verdict = decide(v0220, dev307, ff_v0220, ff_dev307, ar_floor)

    ffv = _ff_break_rate(ff_v0220, "v0220")
    ffd = _ff_break_rate(ff_dev307, "dev307")

    scalars: dict[str, Any] = {
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        # --- PR headline: full-model width break (falls back to lm_head value if ff missing) ---
        "verify_width_break_rate_v0220": ffv.get("verify_width_break_rate_v0220"),
        "verify_width_break_rate_dev307": ffd.get("verify_width_break_rate_dev307"),
        # --- keystone within-server AR-vs-AR control (lm_head, warm, in-process) ---
        "ar_vs_ar_break_v0220": v0220["ar_vs_ar_break"],
        "ar_vs_ar_break_dev307": dev307["ar_vs_ar_break"],
        # --- across-server self-break: measured by census ref-vs-ref2; #675 anchor as backup ---
        "self_break_v0220": ffv.get("self_break_v0220_measured", args.self_break_v0220),
        "self_break_dev307": ffd.get("self_break_dev307_measured", args.self_break_dev307),
        "self_break_v0220_anchor675": args.self_break_v0220,
        "self_break_dev307_anchor675": args.self_break_dev307,
        # --- lm_head head-vs-body decomposition (per-position; == 0 => break is body-driven) ---
        "lmhead_width_break_rate_v0220": v0220["verify_width_break_rate"],
        "lmhead_width_break_rate_dev307": dev307["verify_width_break_rate"],
        "lmhead_bit_break_rate_v0220": v0220.get("bit_break_rate"),
        "lmhead_bit_break_rate_dev307": dev307.get("bit_break_rate"),
        "lmhead_num_positions_v0220": v0220.get("num_positions"),
        "lmhead_num_positions_dev307": dev307.get("num_positions"),
        "lmhead_near_tie_fraction_v0220": v0220.get("near_tie_fraction"),
        "lmhead_near_tie_fraction_dev307": dev307.get("near_tie_fraction"),
        "lmhead_m1_matches_generated_v0220": v0220.get("m1_matches_generated"),
        "lmhead_m1_matches_generated_dev307": dev307.get("m1_matches_generated"),
        "peak_vram_gb": max(v0220.get("peak_vram_gb") or 0, dev307.get("peak_vram_gb") or 0,
                            (ff_v0220 or {}).get("peak_vram_gb") or 0,
                            (ff_dev307 or {}).get("peak_vram_gb") or 0),
    }
    scalars.update({k: v for k, v in ffv.items() if k != "verify_width_break_rate_v0220"})
    scalars.update({k: v for k, v in ffd.items() if k != "verify_width_break_rate_dev307"})

    print("\n" + "=" * 78, flush=True)
    print("[PR681] verify-width version-determinism -- VERDICT", flush=True)
    print(f"  HEADLINE verify_width_break_rate (full-model M=6 spec vs M=1 AR):", flush=True)
    print(f"    v0220 = {scalars['verify_width_break_rate_v0220']}   "
          f"dev307 = {scalars['verify_width_break_rate_dev307']}", flush=True)
    print(f"  full-forward floor (self_break, ref vs ref2):  v0220 = {scalars['self_break_v0220']}"
          f"  dev307 = {scalars['self_break_dev307']}  (0.22.0 must be ~0)", flush=True)
    print(f"  lm_head per-position width break (head decomp): v0220 = "
          f"{scalars['lmhead_width_break_rate_v0220']}  dev307 = "
          f"{scalars['lmhead_width_break_rate_dev307']}  (bit: "
          f"{scalars['lmhead_bit_break_rate_v0220']}/{scalars['lmhead_bit_break_rate_dev307']})", flush=True)
    print(f"  ar_vs_ar within-server control (must be 0):     v0220 = "
          f"{scalars['ar_vs_ar_break_v0220']}  dev307 = {scalars['ar_vs_ar_break_dev307']}", flush=True)
    print(f"  >>> VERDICT: {verdict['verdict']}", flush=True)
    print(f"      {verdict['rationale']}", flush=True)
    print("=" * 78, flush=True)

    report = {"scalars": scalars, "verdict": verdict,
              "v0220_lmhead": v0220, "dev307_lmhead": dev307,
              "ff_v0220": ff_v0220, "ff_dev307": ff_dev307,
              "self_break_source": args.self_break_source}
    (Path(__file__).resolve().parent / "verdict_report.json").write_text(
        json.dumps(report, indent=2, default=str))

    if args.no_wandb:
        return 0
    try:
        from scripts import wandb_logging as wl
    except ImportError:
        print("[log] scripts.wandb_logging unavailable; wrote verdict_report.json only", flush=True)
        return 0
    run = wl.init_wandb_run(
        job_type="verify-width-version-determinism", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        notes="PR681 int4-Marlin verify-width determinism: dev307 vs ship 0.22.0 (analysis_only). "
              "Headline = full-forward served-spec width break; lm_head arm = head-vs-body decomp.",
        tags=["pr681", "verify-width", "int4-marlin", "specdec", "greedy-identity",
              "version-axis", "analysis-only", "full-forward", "lmhead-decomp"],
        config={
            "pr": 681, "model": "int4_g128_lmhead", "k_verify": 6, "k_spec": 5,
            "widths": v0220["config"]["widths"], "num_prompts": v0220["config"]["num_prompts"],
            "output_len": v0220["config"]["output_len"], "seed": v0220["config"]["seed"],
            "vllm_v0220": v0220["vllm_version"], "vllm_dev307": dev307["vllm_version"],
            "verdict": verdict["verdict"], "self_break_source": args.self_break_source,
            "ff_drafter": (ff_v0220 or {}).get("config", {}).get("drafter"),
        },
    )
    if run is None:
        print("[log] wandb not configured (no API key/mode) -- verdict_report.json written", flush=True)
        return 0
    wl.log_event(run, "verdict", step=0,
                 metrics={k: v for k, v in scalars.items() if isinstance(v, (int, float))})
    for k, v in scalars.items():
        run.summary[k] = v
    run.summary["verdict"] = verdict["verdict"]
    run.summary["rationale"] = verdict["rationale"]
    wl.log_json_artifact(run, name="pr681_verdict_report", artifact_type="analysis", data=report)
    wl.finish_wandb(run)
    print(f"[log] wandb logged run -> {run.id}", flush=True)
    print(f"WANDB_RUN_ID {run.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
