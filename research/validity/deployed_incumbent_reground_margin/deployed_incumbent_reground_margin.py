#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #438 — the NARROW deploy verdict (bank-the-analysis, LOCAL pod A10G only).

THE ONE NUMBER THE HUMAN NEEDS
------------------------------
With 496.74 refuted (stark #433 ``0pg4bz25``: served Triton split-K realises
-5.82, not +13.998), the #407 packet reduces to a single honest question:

    Does ANY equivalence-respecting config actually BEAT the deployed 481.53 —
    with a confidence interval — or is the margin within measurement noise /
    contingent on cb3 fully realising?

The deployed 481.53 is NON-equivalent (served token-identity ~0.9966, ~3 M=8
reduction-order flips vs its own M=1 AR -> OUTSIDE the #407 feasible set). The
equivalence frontier is 467.14 (MEASURED floor, denken #423 ``5a6zq2yz``) ->
482.74 (MODELED +cb3, kanna #403 ``iv9i2wks``). The deployed 481.53 sits
BETWEEN them, so the verdict is knife-edge and the incumbent's banked numbers
(481.53 / 0.9966 / 3-flip, PR #52 HF job ``2x9fm2zx``) must be RE-GROUNDED FRESH
on THIS pod under the SAME local harness as the equivalence side.

WHAT THIS LEG OWNS (and what it CONSUMES)
-----------------------------------------
OWN  (measured fresh, this pod): the deployed incumbent — single-stream wall TPS
     (>=5 reps, mean +- CI), the M=8-vs-own-M=1 greedy-identity census
     (identity + flip count), PPL, and — for free, since the M=1 reference is a
     strictly-equivalent config BY CONSTRUCTION — the naive strict-equiv floor.
CONSUME (given in the PR body / public state, OUTSIDE this branch's boundary,
     NOT re-derived here): the equivalence-respecting frontier numbers 467.14
     (measured) and the +cb3 modelled delta (+15.60 -> 482.74), plus stark's
     cb3-realisation status. fern #357 owns the full composite integrator; kanna
     #416 owns the budget-exact 482.74 composition; stark owns cb3 realisation.

HONEST-HARNESS NOTE (stated plainly, not papered over): a TRUE same-harness A/B
would re-measure the equivalence configs locally too, but they live on other
students' branches outside this launch's isolation boundary, so they are
CONSUMED in official frame. The deployed side is re-grounded LOCALLY and bridged
to official frame via the validated tau_lo = 481.53/465.14 = 1.03524 (lawine
#267 ``nzqnd154``; stable to 0.135% over E[T] 3.5-4.5, hardware-dominated). So
the composed margin inherits tau_lo's small (sub-0.2%) regime-drift, which is
DWARFED by both the measured deficit and the cb3 modelling uncertainty.

PRIMARY metric  self_test_passes (0-GPU composition validation)
TEST    metric  honest_margin_tps = best_equivalent_tps - deployed_tps_reground

analysis_only=True; no HF Job / submission / served-file change; official_tps=0.

Run:
  # 0-GPU self-test gate (no model load):
  python research/validity/deployed_incumbent_reground_margin/deployed_incumbent_reground_margin.py \
      --self-test --no-wandb
  # full local re-grounding (serve deployed M=8 + M=1 reference on pod A10G):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      research/validity/deployed_incumbent_reground_margin/deployed_incumbent_reground_margin.py \
      --measure --wandb_group deployed-reground-margin \
      --wandb_name lawine/deployed-incumbent-reground-margin
  # tiny plumbing smoke first:  --measure --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Pre-import the REAL wandb BEFORE putting REPO_ROOT (= target/, which has a
# ./wandb run-output dir that shadows the installed package as a PEP-420
# namespace) on sys.path. Caches the real module in sys.modules.
try:
    import wandb as _wandb_preimport  # noqa: F401
except Exception:  # noqa: BLE001
    _wandb_preimport = None

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------- #
# Imported constants — provenance documented; NOT re-derived. Editing any of
# these trips the self-test constant guard.
# ---------------------------------------------------------------------------- #
# tau_lo anchor pair (lawine #267 nzqnd154; #246 0qc5lk4y local warm anchor).
LOCAL_ANCHOR_TPS = 465.14047160458415   # deployed cfg LOCAL warm wall_tps anchor
OFFICIAL_ANCHOR_TPS = 481.53            # deployed cfg OFFICIAL (PR #52 2x9fm2zx)
TAU_LO = OFFICIAL_ANCHOR_TPS / LOCAL_ANCHOR_TPS  # = 1.03524 local->official map

# Deployed incumbent banked greedy-identity (PR #52 HF job 2x9fm2zx) — the
# numbers we RE-GROUND fresh. ~3 reduction-order flips, served token-identity
# ~0.9966 vs the config's own M=1 AR. We re-measure; these are the prior.
DEPLOYED_IDENTITY_BANKED = 0.9966
DEPLOYED_NFLIPS_BANKED = 3

# Equivalence-respecting frontier — CONSUMED (PR body / public state), official
# frame, NOT re-derived on this branch.
EQUIV_FLOOR_MEASURED_TPS = 467.14   # blanket-strict measured floor, denken #423 5a6zq2yz
CB3_DELTA_MODELED_TPS = 15.60       # +cb3 modelled additive delta, kanna #403 iv9i2wks
EQUIV_PLUS_CB3_TPS = EQUIV_FLOOR_MEASURED_TPS + CB3_DELTA_MODELED_TPS  # = 482.74
# 496.74 pinned-K rung REFUTED by stark #433 0pg4bz25 (served split-K realised
# -5.82, not +13.998) — context only; pinned-K is NOT in this verdict.
PINNEDK_RUNG_REFUTED_TPS = 496.74

# PPL guardrail.
PPL_DEPLOYED_BANKED = 2.3772
PPL_EQUIV_BANKED = 2.378
PPL_GATE = 2.42

SUBMISSION_NAME = "fa2sw_precache_kenyan"
SUBMISSION = REPO_ROOT / "submissions" / SUBMISSION_NAME
OUT_ROOT = HERE
NUM_PROMPTS = 128
OUTPUT_LEN = 512
SEED = 1
REPS_DEFAULT = 5

TOL = 1e-9
TOL_RT = 1e-6

# 95% two-sided Student-t critical values by dof (n-1). Fallback to the
# large-sample 1.96 for dof > 30.
_T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _t_crit(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    if dof in _T_CRIT_95:
        return _T_CRIT_95[dof]
    if dof > 30:
        return 1.96
    # nearest tabulated dof below
    keys = [k for k in _T_CRIT_95 if k <= dof]
    return _T_CRIT_95[max(keys)] if keys else 2.776


def agg_ci(vals: list[float]) -> dict[str, Any]:
    """Mean / sample-std / 95% t-CI half-width for a list of measurements."""
    clean = [float(v) for v in vals if _finite(v)]
    n = len(clean)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(clean)
    std = statistics.stdev(clean) if n > 1 else 0.0
    ci_half = (_t_crit(n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
    s = sorted(clean)
    return {
        "n": n, "mean": mean, "std": std,
        "ci95_half": ci_half,
        "cv_pct": (100.0 * std / mean) if mean else float("nan"),
        "min": s[0], "max": s[-1], "median": statistics.median(clean),
        "values": clean,
    }


# ========================================================================== #
# Analytic core — the composition / verdict (the part the self-test validates).
# ========================================================================== #
def compose_verdict(
    *,
    deployed_reground_official: float,
    deployed_ci_official: float,
    best_equiv_measured: float,
    cb3_delta: float,
    equiv_ci: float = 0.0,
) -> dict[str, Any]:
    """The narrow deploy verdict: does the best equivalence-respecting config
    beat the freshly re-grounded deployed incumbent?

    Headline ``best_equivalent_tps`` is the best MEASURED equivalent (467.14):
    "actually beats" can only mean a number that actually exists. The modelled
    +cb3 (482.74) is reported as the cb3-realisation upside / sensitivity.

    The combined uncertainty for the noise test is the quadrature sum of the
    deployed CI and the (consumed) equivalence CI; the margin is "within noise"
    only if its magnitude is below that band. The DOMINANT uncertainty for the
    +cb3 path is cb3 realisation itself (the whole +15.60 is at risk per stark),
    captured separately by the robustness test, not by measurement noise.
    """
    best_equiv_with_cb3 = best_equiv_measured + cb3_delta
    combined_sigma = math.hypot(deployed_ci_official, equiv_ci)

    # MEASURED headline (cb3 = 0): does the best measured equiv beat deployed?
    margin_measured = best_equiv_measured - deployed_reground_official
    beats_measured = margin_measured > combined_sigma
    margin_within_noise = abs(margin_measured) < combined_sigma

    # cb3-FULL upside (cb3 = +15.60 fully realises).
    margin_with_cb3 = best_equiv_with_cb3 - deployed_reground_official
    beats_with_cb3 = margin_with_cb3 > combined_sigma

    # How much of the modelled cb3 must realise just to TIE the deployed config?
    cb3_required_to_tie = deployed_reground_official - best_equiv_measured
    cb3_required_frac = (cb3_required_to_tie / cb3_delta) if cb3_delta else float("inf")

    # robust to cb3 iff the verdict holds under BOTH cb3=0 and cb3=full.
    verdict_robust_to_cb3 = bool(beats_measured and beats_with_cb3)

    if beats_measured:
        headline = (
            f"the fastest strictly-equivalent config that has been MEASURED "
            f"({best_equiv_measured:.2f}) is FASTER than the freshly re-grounded "
            f"deployed non-equivalent incumbent ({deployed_reground_official:.2f}) "
            f"by {margin_measured:+.2f} TPS (+-{combined_sigma:.2f})."
        )
    else:
        headline = (
            f"the fastest strictly-equivalent config that has actually been "
            f"MEASURED ({best_equiv_measured:.2f} official-frame TPS) is SLOWER "
            f"than the freshly re-grounded deployed non-equivalent incumbent "
            f"({deployed_reground_official:.2f}) by {margin_measured:+.2f} TPS "
            f"(+-{combined_sigma:.2f}); only the MODELLED +cb3 config "
            f"({best_equiv_with_cb3:.2f}) edges ahead, by {margin_with_cb3:+.2f} "
            f"TPS — which needs cb3 to realise >= {100.0 * cb3_required_frac:.0f}% "
            f"of its modelled +{cb3_delta:.2f} and is NOT robust to a cb3 haircut "
            f"(stark already refuted the sibling pinned-K leg)."
        )

    return {
        "deployed_reground_official": deployed_reground_official,
        "deployed_ci_official": deployed_ci_official,
        "best_equivalent_tps": best_equiv_measured,
        "best_equivalent_is_measured": True,
        "best_equivalent_tps_ci": equiv_ci,
        "best_equiv_with_cb3_modeled": best_equiv_with_cb3,
        "combined_sigma_tps": combined_sigma,
        # headline (measured) verdict
        "beats_deployed_481": bool(beats_measured),
        "honest_margin_tps": margin_measured,
        "margin_within_noise": bool(margin_within_noise),
        # cb3 sensitivity
        "beats_deployed_481_if_cb3_full": bool(beats_with_cb3),
        "honest_margin_tps_if_cb3_full": margin_with_cb3,
        "cb3_required_to_tie_tps": cb3_required_to_tie,
        "cb3_required_frac_of_modeled": cb3_required_frac,
        "verdict_robust_to_cb3": verdict_robust_to_cb3,
        "one_line_verdict": headline,
    }


# ========================================================================== #
# Self-test (PRIMARY) — 0-GPU validation of the composition arithmetic.
# ========================================================================== #
def self_test() -> dict[str, Any]:
    # Synthetic re-grounding that reproduces the deployed anchor exactly.
    syn_local = LOCAL_ANCHOR_TPS
    syn_deployed_official = syn_local * TAU_LO          # == 481.53
    syn_ci = 0.20                                        # ~0.04% of 481.53
    v = compose_verdict(
        deployed_reground_official=syn_deployed_official,
        deployed_ci_official=syn_ci,
        best_equiv_measured=EQUIV_FLOOR_MEASURED_TPS,
        cb3_delta=CB3_DELTA_MODELED_TPS,
        equiv_ci=0.0,
    )

    # (a) tau_lo round-trip: local * tau_lo == official anchor.
    a = abs(syn_local * TAU_LO - OFFICIAL_ANCHOR_TPS) < 1e-6

    # (b) measured margin arithmetic: 467.14 - 481.53 = -14.39; beats=False;
    #     the deficit is FAR outside the noise band (not within noise).
    expect_margin = EQUIV_FLOOR_MEASURED_TPS - syn_deployed_official
    b = bool(abs(v["honest_margin_tps"] - expect_margin) < 1e-6
             and v["beats_deployed_481"] is False
             and v["margin_within_noise"] is False
             and expect_margin < -10.0)

    # (c) cb3-full upside: 482.74 > 481.53 -> beats_if_cb3_full True, margin
    #     small (+1.21), but NOT robust (fails the cb3=0 leg).
    expect_margin_cb3 = EQUIV_PLUS_CB3_TPS - syn_deployed_official
    c = bool(v["beats_deployed_481_if_cb3_full"] is True
             and abs(v["honest_margin_tps_if_cb3_full"] - expect_margin_cb3) < 1e-6
             and 0.0 < expect_margin_cb3 < 2.0
             and v["verdict_robust_to_cb3"] is False)

    # (d) cb3-required threshold: cb3 must realise >= ~92% of +15.60 to tie.
    expect_req = syn_deployed_official - EQUIV_FLOOR_MEASURED_TPS
    d = bool(abs(v["cb3_required_to_tie_tps"] - expect_req) < 1e-6
             and 0.85 < v["cb3_required_frac_of_modeled"] < 1.0)

    # (e) robustness symmetry: if the deployed config were SLOWER than the
    #     measured floor, the verdict would flip to beats=True under both legs.
    v_fast_equiv = compose_verdict(
        deployed_reground_official=450.0, deployed_ci_official=0.2,
        best_equiv_measured=EQUIV_FLOOR_MEASURED_TPS, cb3_delta=CB3_DELTA_MODELED_TPS,
    )
    e = bool(v_fast_equiv["beats_deployed_481"] is True
             and v_fast_equiv["verdict_robust_to_cb3"] is True)

    # (f) imported constants exact.
    f = bool(abs(OFFICIAL_ANCHOR_TPS - 481.53) < TOL
             and abs(LOCAL_ANCHOR_TPS - 465.14047160458415) < TOL
             and abs(EQUIV_FLOOR_MEASURED_TPS - 467.14) < TOL
             and abs(CB3_DELTA_MODELED_TPS - 15.60) < TOL
             and abs(EQUIV_PLUS_CB3_TPS - 482.74) < TOL_RT
             and abs(TAU_LO - OFFICIAL_ANCHOR_TPS / LOCAL_ANCHOR_TPS) < 1e-12
             and abs(round(TAU_LO, 5) - 1.03524) < 1e-9)

    # (g) NaN-clean over every composed float.
    floats = [v["honest_margin_tps"], v["honest_margin_tps_if_cb3_full"],
              v["cb3_required_to_tie_tps"], v["cb3_required_frac_of_modeled"],
              v["combined_sigma_tps"], TAU_LO]
    g = all(_finite(x) for x in floats)

    conditions = {
        "a_tau_lo_roundtrip": a,
        "b_measured_margin_is_deficit": b,
        "c_cb3_upside_not_robust": c,
        "d_cb3_required_threshold": d,
        "e_robustness_symmetry": e,
        "f_constants_exact": f,
        "g_nan_clean": g,
    }
    return {
        "conditions": conditions,
        "self_test_passes": bool(all(conditions.values())),
        "synthetic_verdict": v,
    }


# ========================================================================== #
# Measurement — local A10G re-grounding of the deployed incumbent.
# ========================================================================== #
def _gpu_mem_used_mib() -> float | None:
    import subprocess as sp
    try:
        out = sp.run(["nvidia-smi", "--query-gpu=memory.used",
                      "--format=csv,noheader,nounits", "-i", "0"],
                     capture_output=True, text=True, timeout=15)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def _preflight_gpu(mem_threshold_mib: int = 1500, timeout_s: int = 180) -> None:
    """Reap any lingering vLLM server + wait for VRAM to drain (single-GPU pod)."""
    import subprocess as sp
    reaped = False
    for pat in ("vllm.entrypoints.openai.api_server", "VLLM::EngineCore",
                "multiprocessing.resource_tracker"):
        r = sp.run(["pkill", "-9", "-f", pat], capture_output=True)
        reaped = reaped or (r.returncode == 0)
    if reaped:
        print("[reground] preflight: reaped lingering vLLM process(es)", flush=True)
        time.sleep(4)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_mem_used_mib()
        if used is None or used < mem_threshold_mib:
            return
        time.sleep(3)


def measure_deployed_m8(server_python: Path, run_dir: Path, *, reps: int,
                        num_prompts: int, output_len: int) -> dict[str, Any]:
    """Serve the deployed M=8 incumbent ONCE; run ``reps`` back-to-back decodes
    (one warm discard + ``reps`` measured) for a wall_tps CI; keep rep-0's token
    IDs as the M=8 census candidate; run one PPL pass after the timed loop."""
    from scripts.local_validation import harness  # noqa: E402

    wall_tps_vals: list[float] = []
    per_rep: list[dict[str, Any]] = []
    census_candidate = run_dir / "m8_candidate_decode.jsonl"
    ppl_summary: dict[str, Any] = {}
    peak_mib = {"v": 0.0}

    _preflight_gpu()
    log_path = run_dir / "server_m8.log"
    with harness.LocalServer(SUBMISSION, server_python=server_python,
                             log_path=log_path) as srv:
        served_model = srv.served_model_name
        # one warm discard pass so the headline reps are steady-state.
        warm_out = run_dir / "m8_warm.jsonl"
        warm_sum = run_dir / "m8_warm.summary.json"
        print("[reground] M=8 warm discard pass", flush=True)
        harness.capture_decode(server_python, base_url=srv.base_url,
                               model=served_model, out_file=warm_out,
                               summary_file=warm_sum, num_prompts=num_prompts,
                               output_len=output_len, seed=SEED)
        for i in range(reps):
            out_file = (census_candidate if i == 0
                        else run_dir / f"m8_rep{i:02d}.jsonl")
            sum_file = run_dir / f"m8_rep{i:02d}.summary.json"
            print(f"[reground] M=8 measured rep {i + 1}/{reps}", flush=True)
            s = harness.capture_decode(server_python, base_url=srv.base_url,
                                       model=served_model, out_file=out_file,
                                       summary_file=sum_file,
                                       num_prompts=num_prompts,
                                       output_len=output_len, seed=SEED)
            n_tok = int(s.get("num_completion_tokens", 0))
            dur = float(s.get("duration_s", 0.0) or 0.0)
            wtps = n_tok / dur if dur > 0 else float("nan")
            wall_tps_vals.append(wtps)
            per_rep.append({"rep": i, "num_completion_tokens": n_tok,
                            "duration_s": dur, "wall_tps": wtps})
            m = _gpu_mem_used_mib()
            if m:
                peak_mib["v"] = max(peak_mib["v"], m)
            print(f"[reground]   rep {i}: wall_tps={wtps:.3f} "
                  f"({n_tok} tok / {dur:.2f}s)", flush=True)
        # PPL after the timed loop so it never perturbs a timed run.
        try:
            print("[reground] M=8 PPL validity pass", flush=True)
            ppl_summary = harness.run_ppl(
                server_python, base_url=srv.base_url, model=served_model,
                out_file=run_dir / "m8_ppl.jsonl",
                summary_file=run_dir / "m8_ppl.summary.json")
        except Exception as exc:  # noqa: BLE001
            print(f"[reground] WARN PPL failed: {exc}", flush=True)
            ppl_summary = {"error": str(exc)}

    agg = agg_ci(wall_tps_vals)
    return {
        "served_model_name": served_model,
        "reps": reps, "num_prompts": num_prompts, "output_len": output_len,
        "wall_tps_local": agg,
        "per_rep": per_rep,
        "census_candidate": str(census_candidate),
        "ppl": ppl_summary.get("ppl"),
        "ppl_num_records": ppl_summary.get("num_records"),
        "peak_vram_gb": peak_mib["v"] / 1024.0 if peak_mib["v"] else None,
    }


def measure_reference_m1(server_python: Path, run_dir: Path, *,
                         num_prompts: int, output_len: int) -> dict[str, Any]:
    """Serve the SAME submission with SENPAI_REFERENCE_MODE=1 (speculation
    disabled -> plain M=1 AR greedy, the canonical strictly-equivalent reference)
    and capture ONE decode: the M=1 census reference AND the naive strict-equiv
    floor wall_tps (a strictly-equivalent config we can MEASURE directly)."""
    from scripts.local_validation import harness, paths  # noqa: E402

    _preflight_gpu()
    log_path = run_dir / "server_m1.log"
    ref_decode = run_dir / "m1_reference_decode.jsonl"
    ref_sum = run_dir / "m1_reference.summary.json"
    with harness.LocalServer(SUBMISSION, server_python=server_python,
                             log_path=log_path,
                             extra_env={paths.REFERENCE_MODE_ENV: "1"}) as srv:
        print("[reground] M=1 reference-mode decode (speculation OFF)", flush=True)
        s = harness.capture_decode(server_python, base_url=srv.base_url,
                                   model=srv.served_model_name,
                                   out_file=ref_decode, summary_file=ref_sum,
                                   num_prompts=num_prompts,
                                   output_len=output_len, seed=SEED)
    n_tok = int(s.get("num_completion_tokens", 0))
    dur = float(s.get("duration_s", 0.0) or 0.0)
    wtps = n_tok / dur if dur > 0 else float("nan")
    print(f"[reground] M=1 reference: wall_tps={wtps:.3f} "
          f"({n_tok} tok / {dur:.2f}s)", flush=True)
    return {
        "reference_decode": str(ref_decode),
        "num_completion_tokens": n_tok, "duration_s": dur,
        "wall_tps_local": wtps,
    }


def run_census(ref_decode: Path, cand_decode: Path) -> dict[str, Any]:
    """M=8 candidate vs M=1 reference greedy-identity census (official verifier,
    byte-exact token-id compare, NO numeric tolerance)."""
    from scripts.local_validation import paths  # noqa: E402
    verifier = paths.import_greedy_identity()
    ref_rec = verifier.load_decode_outputs(ref_decode)
    cand_rec = verifier.load_decode_outputs(cand_decode)
    # compare only the common key set (defensive; both should be the 128 prompts).
    common_ref = {k: v for k, v in ref_rec.items() if k in cand_rec}
    rep = verifier.compare(common_ref, cand_rec)
    n_prompts = rep.num_prompts_compared
    total = rep.total_tokens_compared
    div_tokens = rep.total_divergent_tokens
    identity_token = (1.0 - div_tokens / total) if total else float("nan")
    identity_prompt = (rep.num_identical / n_prompts) if n_prompts else float("nan")
    return {
        "verdict": rep.verdict,
        "num_prompts_compared": n_prompts,
        "num_identical_prompts": rep.num_identical,
        "num_divergent_prompts": rep.num_divergent,
        "total_tokens_compared": total,
        "total_divergent_tokens": div_tokens,
        "identity_rate_token": identity_token,
        "identity_rate_prompt": identity_prompt,
        "n_flips_token": div_tokens,
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.local_validation import harness, paths  # noqa: E402

    for note in paths.prepare_local_gpu_env():
        print(f"[reground] {note}", flush=True)
    if not SUBMISSION.exists():
        raise SystemExit(f"deployed submission not found: {SUBMISSION}")
    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    print(f"[reground] submission={SUBMISSION.name} server_python={server_python}", flush=True)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = OUT_ROOT / ("smoke" if args.smoke else "measured") / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    reps = 1 if args.smoke else args.reps
    num_prompts = 8 if args.smoke else args.num_prompts
    output_len = 64 if args.smoke else args.output_len

    m8 = measure_deployed_m8(server_python, run_dir, reps=reps,
                             num_prompts=num_prompts, output_len=output_len)
    m1 = measure_reference_m1(server_python, run_dir,
                              num_prompts=num_prompts, output_len=output_len)
    census = run_census(Path(m1["reference_decode"]), Path(m8["census_candidate"]))

    return {
        "run_dir": str(run_dir), "stamp": stamp, "smoke": bool(args.smoke),
        "workload": {"reps": reps, "num_prompts": num_prompts, "output_len": output_len},
        "m8_deployed": m8,
        "m1_reference": m1,
        "census": census,
    }


# ========================================================================== #
# Report assembly.
# ========================================================================== #
def build_report(measured: dict[str, Any] | None, st: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_tps": 0,
        "tau_lo": TAU_LO,
        "self_test": st,
        "self_test_passes": st["self_test_passes"],
        "consumed_equivalence": {
            "equiv_floor_measured_tps": EQUIV_FLOOR_MEASURED_TPS,
            "equiv_floor_provenance": "denken #423 5a6zq2yz (measured, official frame, CONSUMED)",
            "cb3_delta_modeled_tps": CB3_DELTA_MODELED_TPS,
            "cb3_provenance": "kanna #403 iv9i2wks (modelled +cb3, CONSUMED); stark owns realisation",
            "equiv_plus_cb3_tps": EQUIV_PLUS_CB3_TPS,
            "pinnedk_rung_refuted_tps": PINNEDK_RUNG_REFUTED_TPS,
            "same_harness_caveat": (
                "equivalence numbers are CONSUMED in official frame from other "
                "students' branches (outside this launch's isolation boundary); "
                "the deployed side is re-grounded LOCALLY and bridged via tau_lo."),
        },
        "deployed_banked": {
            "official_tps": OFFICIAL_ANCHOR_TPS,
            "local_anchor_tps": LOCAL_ANCHOR_TPS,
            "identity": DEPLOYED_IDENTITY_BANKED,
            "n_flips": DEPLOYED_NFLIPS_BANKED,
            "ppl": PPL_DEPLOYED_BANKED,
        },
    }

    if measured is None:
        # Self-test-only: surface the SYNTHETIC verdict (reproduces the anchor)
        # so the terminal fields are populated for a 0-GPU dry run.
        v = st["synthetic_verdict"]
        report["measured_present"] = False
        report["verdict"] = v
        report["terminal_fields"] = _terminal_fields(
            v, deployed_identity=DEPLOYED_IDENTITY_BANKED,
            deployed_n_flips=DEPLOYED_NFLIPS_BANKED, ppl=PPL_DEPLOYED_BANKED,
            self_test_passes=st["self_test_passes"], measured=False)
        return report

    m8 = measured["m8_deployed"]
    m1 = measured["m1_reference"]
    census = measured["census"]
    wall = m8["wall_tps_local"]
    deployed_reground_official = wall["mean"] * TAU_LO
    deployed_ci_official = wall["ci95_half"] * TAU_LO
    naive_equiv_floor_official = m1["wall_tps_local"] * TAU_LO

    v = compose_verdict(
        deployed_reground_official=deployed_reground_official,
        deployed_ci_official=deployed_ci_official,
        best_equiv_measured=EQUIV_FLOOR_MEASURED_TPS,
        cb3_delta=CB3_DELTA_MODELED_TPS,
        equiv_ci=0.0,
    )
    ppl = m8.get("ppl")
    report["measured_present"] = True
    report["measured"] = measured
    report["deployed_reground"] = {
        "local_wall_tps_mean": wall["mean"],
        "local_wall_tps_std": wall["std"],
        "local_wall_tps_ci95_half": wall["ci95_half"],
        "local_wall_tps_cv_pct": wall.get("cv_pct"),
        "n_reps": wall["n"],
        "official_frame_tps": deployed_reground_official,
        "official_frame_ci95_half": deployed_ci_official,
        "reproduces_banked_481": abs(deployed_reground_official - OFFICIAL_ANCHOR_TPS) < 5.0,
        "drift_vs_banked_481_tps": deployed_reground_official - OFFICIAL_ANCHOR_TPS,
    }
    report["measured_strict_equiv_floor"] = {
        "note": ("M=1 AR (SENPAI_REFERENCE_MODE) is strictly greedy-equivalent BY "
                 "CONSTRUCTION; this is a strict-equiv config we MEASURE directly"),
        "local_wall_tps": m1["wall_tps_local"],
        "official_frame_tps": naive_equiv_floor_official,
        "deficit_vs_deployed_tps": naive_equiv_floor_official - deployed_reground_official,
    }
    report["census"] = census
    report["ppl_measured_deployed"] = ppl
    report["ppl_passes_gate"] = (ppl is not None and ppl <= PPL_GATE)
    report["verdict"] = v
    # n_flips is the de-cascaded, banked-comparable flip count = number of
    # DIVERGENT PROMPTS, not total_divergent_tokens. Both M=8 and M=1 decode
    # freely to output_len, so a single first-divergence point cascades through
    # every later token of that prompt; total_divergent_tokens is therefore a
    # cascade count, not an event count. The banked "3 flips / identity 0.9966"
    # is only self-consistent as 3 divergent prompts (each cascading ~74 of 512
    # tokens -> ~223/65536 token-divergence -> token-identity 0.9966), so the
    # banked-comparable flip count is num_divergent_prompts.
    report["terminal_fields"] = _terminal_fields(
        v, deployed_identity=census["identity_rate_token"],
        deployed_n_flips=census["num_divergent_prompts"], ppl=ppl,
        self_test_passes=st["self_test_passes"], measured=True,
        n_divergent_tokens_cascaded=census["n_flips_token"])
    report["peak_vram_gb"] = m8.get("peak_vram_gb")
    return report


def _terminal_fields(v: dict[str, Any], *, deployed_identity: float,
                     deployed_n_flips: int, ppl: float | None,
                     self_test_passes: bool, measured: bool,
                     n_divergent_tokens_cascaded: int | None = None) -> dict[str, Any]:
    return {
        "deployed_tps_reground": v["deployed_reground_official"],
        "deployed_tps_ci": v["deployed_ci_official"],
        "deployed_identity_reground": deployed_identity,
        "deployed_n_flips": deployed_n_flips,
        "deployed_n_divergent_tokens_cascaded": n_divergent_tokens_cascaded,
        "best_equivalent_tps": v["best_equivalent_tps"],
        "best_equivalent_is_measured": v["best_equivalent_is_measured"],
        "best_equivalent_tps_ci": v["best_equivalent_tps_ci"],
        "beats_deployed_481": v["beats_deployed_481"],
        "honest_margin_tps": v["honest_margin_tps"],
        "margin_within_noise": v["margin_within_noise"],
        "verdict_robust_to_cb3": v["verdict_robust_to_cb3"],
        "beats_deployed_481_if_cb3_full": v["beats_deployed_481_if_cb3_full"],
        "honest_margin_tps_if_cb3_full": v["honest_margin_tps_if_cb3_full"],
        "cb3_required_frac_of_modeled": v["cb3_required_frac_of_modeled"],
        "ppl": ppl,
        "self_test_passes": self_test_passes,
        "analysis_only": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps": 0,
        "measured": measured,
    }


# ========================================================================== #
# NaN guard / print / wandb.
# ========================================================================== #
def _assert_nan_clean(payload: dict, path: str = "payload") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, val in node.items():
                walk(val, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, val in enumerate(node):
                walk(val, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(report: dict[str, Any]) -> None:
    line = "=" * 30 + " DEPLOY VERDICT (PR #438) " + "=" * 30
    print("\n" + line, flush=True)
    st = report["self_test"]
    print(f"  (PRIMARY) self_test_passes = {st['self_test_passes']}", flush=True)
    for k, val in st["conditions"].items():
        print(f"          - {k}: {val}", flush=True)
    if report.get("measured_present"):
        dr = report["deployed_reground"]
        ce = report["census"]
        sf = report["measured_strict_equiv_floor"]
        print("-" * len(line), flush=True)
        print(f"  deployed re-ground: LOCAL wall_tps {dr['local_wall_tps_mean']:.3f} "
              f"+- {dr['local_wall_tps_ci95_half']:.3f} (n={dr['n_reps']}, "
              f"CV {dr['local_wall_tps_cv_pct']:.3f}%) -> OFFICIAL "
              f"{dr['official_frame_tps']:.2f} +- {dr['official_frame_ci95_half']:.2f} "
              f"(drift vs banked 481.53 = {dr['drift_vs_banked_481_tps']:+.2f})", flush=True)
        print(f"  greedy census M=8 vs own M=1: verdict={ce['verdict']} "
              f"identity(token)={ce['identity_rate_token']:.4f} "
              f"n_flips(divergent_prompts)={ce['num_divergent_prompts']}/{ce['num_prompts_compared']} "
              f"[banked ~3] divergent_tokens_cascaded={ce['n_flips_token']}", flush=True)
        print(f"  measured strict-equiv floor (M=1 AR): "
              f"{sf['official_frame_tps']:.2f} official "
              f"(deficit vs deployed {sf['deficit_vs_deployed_tps']:+.2f})", flush=True)
        print(f"  PPL deployed (measured) = {report['ppl_measured_deployed']} "
              f"(gate <= {PPL_GATE}, passes={report['ppl_passes_gate']})", flush=True)
    v = report["verdict"]
    print("-" * len(line), flush=True)
    print(f"  best_equivalent_tps (measured) = {v['best_equivalent_tps']:.2f}", flush=True)
    print(f"  beats_deployed_481 = {v['beats_deployed_481']}  "
          f"honest_margin_tps = {v['honest_margin_tps']:+.2f}  "
          f"within_noise = {v['margin_within_noise']}", flush=True)
    print(f"  IF cb3 fully realises (+{CB3_DELTA_MODELED_TPS:.2f} -> "
          f"{EQUIV_PLUS_CB3_TPS:.2f}): beats = {v['beats_deployed_481_if_cb3_full']} "
          f"margin = {v['honest_margin_tps_if_cb3_full']:+.2f} "
          f"(cb3 must realise >= {100.0 * v['cb3_required_frac_of_modeled']:.0f}% to tie)", flush=True)
    print(f"  verdict_robust_to_cb3 = {v['verdict_robust_to_cb3']}", flush=True)
    print(f"\n  VERDICT: {v['one_line_verdict']}", flush=True)
    print("=" * len(line) + "\n", flush=True)


def _maybe_log_wandb(args: argparse.Namespace, payload: dict[str, Any]) -> str | None:
    if args.no_wandb or not getattr(args, "wandb_name", None):
        return None
    try:
        import wandb  # noqa: F401
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[reground] wandb unavailable: {exc}", flush=True)
        return None

    report = payload["report"]
    tf = report["terminal_fields"]
    v = report["verdict"]
    summary: dict[str, Any] = {
        "self_test_passes": int(bool(report["self_test_passes"])),
        "tau_lo": TAU_LO,
        "deployed_tps_reground": tf["deployed_tps_reground"],
        "deployed_tps_ci": tf["deployed_tps_ci"],
        "deployed_identity_reground": tf["deployed_identity_reground"],
        "deployed_n_flips": tf["deployed_n_flips"],
        "best_equivalent_tps": tf["best_equivalent_tps"],
        "best_equivalent_is_measured": int(bool(tf["best_equivalent_is_measured"])),
        "beats_deployed_481": int(bool(tf["beats_deployed_481"])),
        "honest_margin_tps": tf["honest_margin_tps"],
        "margin_within_noise": int(bool(tf["margin_within_noise"])),
        "verdict_robust_to_cb3": int(bool(tf["verdict_robust_to_cb3"])),
        "beats_deployed_481_if_cb3_full": int(bool(tf["beats_deployed_481_if_cb3_full"])),
        "honest_margin_tps_if_cb3_full": tf["honest_margin_tps_if_cb3_full"],
        "cb3_required_frac_of_modeled": v["cb3_required_frac_of_modeled"],
        "ppl": tf["ppl"],
        "official_tps": 0, "analysis_only": 1, "no_hf_job": 1,
        "no_served_file_change": 1, "nan_clean": int(bool(payload["nan_clean"])),
        "measured_present": int(bool(report.get("measured_present"))),
    }
    if report.get("measured_present"):
        dr = report["deployed_reground"]
        summary["deployed_local_wall_tps_mean"] = dr["local_wall_tps_mean"]
        summary["deployed_local_wall_tps_cv_pct"] = dr["local_wall_tps_cv_pct"]
        summary["deployed_drift_vs_banked_481"] = dr["drift_vs_banked_481_tps"]
        summary["measured_strict_equiv_floor_official_tps"] = \
            report["measured_strict_equiv_floor"]["official_frame_tps"]
        summary["ppl_passes_gate"] = int(bool(report["ppl_passes_gate"]))
        if report.get("peak_vram_gb"):
            summary["peak_vram_gb"] = report["peak_vram_gb"]
    summary = {k: val for k, val in summary.items()
               if val is not None and not (isinstance(val, float) and not math.isfinite(val))}

    run = init_wandb_run(
        job_type="validity-gate", agent="lawine",
        name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "deploy-verdict", "equivalence-frontier",
              "deployed-incumbent-reground", "tau-lo", "cb3-sensitivity",
              "bank-the-analysis", "pr-438"],
        notes="PR #438 narrow deploy verdict: does any equivalence config beat deployed 481.53?",
        config={
            "submission": SUBMISSION_NAME, "tau_lo": TAU_LO,
            "local_anchor_tps": LOCAL_ANCHOR_TPS, "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
            "equiv_floor_measured_tps": EQUIV_FLOOR_MEASURED_TPS,
            "cb3_delta_modeled_tps": CB3_DELTA_MODELED_TPS,
            "equiv_plus_cb3_tps": EQUIV_PLUS_CB3_TPS, "ppl_gate": PPL_GATE,
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[reground] wandb: no run (no API key / disabled) — skipping", flush=True)
        return None
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="deployed_incumbent_reground_margin_result",
                      artifact_type="validity", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[reground] wandb logged ({rid}): {summary}", flush=True)
    return rid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY 0-GPU composition self-validation and exit 0/1")
    ap.add_argument("--measure", action="store_true",
                    help="serve the deployed M=8 + M=1 reference on the pod A10G and re-ground")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny plumbing run (8 prompts x 64 tok, 1 rep) — validates serve/decode/census")
    ap.add_argument("--reps", type=int, default=REPS_DEFAULT,
                    help="measured wall_tps reps for the deployed M=8 CI (>=5)")
    ap.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="deployed-reground-margin")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    st = self_test()

    measured = None
    if args.measure or args.smoke:
        measured = run_measurement(args)

    report = build_report(measured, st)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 438, "agent": "lawine",
        "kind": "deployed-incumbent-reground-margin", "report": report,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[reground] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(report)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / "deployed_incumbent_reground_margin_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[reground] wrote {out_path}", flush=True)

    rid = _maybe_log_wandb(args, payload)
    if rid:
        report["wandb_run_id"] = rid
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=float)

    # single-line terminal marker for easy harvest
    print("TERMINAL_FIELDS " + json.dumps(report["terminal_fields"], sort_keys=True), flush=True)

    passes = bool(report["self_test_passes"]) and payload["nan_clean"]
    if args.self_test and not (args.measure or args.smoke):
        print(f"[reground] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
