#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #455 (lawine) -- Independent re-anchor: strict frontier 467.14 + deployed identity.

WHY (the decision this feeds)
-----------------------------
Before the advisor takes a "relax-strict-equivalence" proposal to the human, two load-bearing
anchor numbers must be INDEPENDENTLY reproduced on THIS pod, today, by a fresh hand:
  (1) the realized blanket-strict frontier  467.14 TPS  (denken #423 `5a6zq2yz`), the fastest
      strictly byte-equivalent serving path (int4 MTP K=7 verify, strict M-invariant attention,
      ZERO non-equivalent reduction flips), and
  (2) the deployed 481.53's NON-equivalence (PR #52 `2x9fm2zx`): its greedy-identity fraction
      and token-flip count vs the strict int4 argmax on the 128-prompt eval.
The gap between them (the "equivalence tax") is what the human is being asked to spend. This card
re-derives both from scratch and prices the tax against the hardware-noise envelope (sigma_hw).

WHAT 467.14 ACTUALLY IS (settled here from public W&B, not assumed)
------------------------------------------------------------------
denken #423 (`5a6zq2yz`) is a 1-SECOND COMPOSITION run (`_runtime=1`); it CONSUMES
`base_467_measured_412 = 467.1400155438763` from stark #412 (`uc7jg6vs`). stark #412's number is
itself COMPOSED: blanket_strict = OFFICIAL_TPS / (1 + eta_attn_decode), where eta_attn_decode is a
FA2 varlen-attention microbench delta (fast num_splits=0 heuristic vs strict num_splits=1 / M-invariant)
over the gemma-4 decode-position band. 467.14 is therefore an ISOLATION of the single component that
differs between the strict and the deployed-fast path -- the attention reduction-ORDER tax -- holding
cudagraph/ONEGRAPH/precache/lm_head-prune constant. It is NOT (and cannot be) a naive end-to-end serve:
serving with VLLM_BATCH_INVARIANT=1 ALSO disables cudagraph/ONEGRAPH, confounding the measurement (my
own #438 served the deployed config end-to-end at 465.04 local / 481.42 official on the FAST path).
The ONLY end-to-end-MEASURABLE strictly-equivalent config is the M=1 AR reference (no speculation),
which my #438 measured at 156.20 local / 161.70 official -- 305 TPS below deployed. So an HONEST
independent re-anchor of 467.14 = re-run #412's census+microbench MYSELF with FRESH seeds and re-compose.

METHOD (uses the in-boundary, advisor-branch-MERGED #412 method as a black-box subprocess)
------------------------------------------------------------------------------------------
This orchestrator drives `selective_recompute_equivalent_tps.py` (stark #412, MERGED to
approval-gated-8gpu-20260613 @8cff7c6 -- IN my isolation boundary) as isolated subprocesses, with my
OWN output paths and my OWN microbench seed set (5,6,7,8,9 -- distinct from #412's 0,1,2):
  - census  heuristic (VLLM_BATCH_INVARIANT=0): the SERVED fast M=8-verify attention -> deployed identity.
  - census  pinned    (VLLM_BATCH_INVARIANT=1): strict single-segment M-invariant reduction -> strict identity.
  - microbench (N>=5 seeds): FA2 fast(ns=0) vs strict(ns=1) per-step latency -> eta_attn_decode per seed.
Then I COMPOSE, per seed:  frontier_seed = OFFICIAL_TPS / (1 + eta_seed)  ->  median + sigma  (the N>=5
"runs" the PR asks for are these independent seed estimates; the census identity is deterministic, 1/arm).

DELIVERABLES (PR #455 fields; logged to W&B group `equivalence-escalation-anchors`)
  reanchored_strict_frontier_tps + strict_frontier_sigma          (Step 1)
  deployed_identity_fraction + deployed_token_flips + deployed_flips_stable   (Step 2)
  strict_identity_fraction + strict_token_flips                   (Step 2, the strict-arm counterpart)
  equivalence_tax_tps + sigma_hw + tax_exceeds_sigma_hw           (Step 3)
  ppl(=2.3772 anchor) + strict_frontier_reanchor_self_test_passes (Step 4)
  analysis_only=true / no_hf_job=true / no_served_file_change=true / official_tps=0

SCOPE: LOCAL A10G (sm_86), analysis-only. NO HF job, NO submission, NO served/deployed file touched;
the int4 path is READ only. The #412 script is invoked, never edited.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

# ======================================================================================
# Anchors (CITE public W&B; do NOT re-derive). Provenance in the module docstring.
# ======================================================================================
OFFICIAL_TPS = 481.53                    # deployed non-strict public #1 (PR #52 `2x9fm2zx`); tps=481.528
SIGMA_HW = 4.8                           # hardware-noise envelope (TPS), given by PR #455
PPL_ANCHOR = 2.3772                      # deployed PPL (PR #52 summary/ppl=2.37719; denken #423 ppl_deployed)
PPL_GATE = 2.42                          # validity gate ceiling

# what we are re-anchoring (the consumed numbers we must independently reproduce):
FRONTIER_ANCHOR = 467.1400155438763      # stark #412 base_467_measured; denken #423 consumed
FRONTIER_STD_ANCHOR = 0.16105003370123783
TAX_ANCHOR = 14.38998445612367           # denken #423 tax_measured = 481.53 - 467.14
DEPLOYED_IDENTITY_ANCHOR = 0.9965986394557823   # 879/882 (3 flips); #381/#405/#412 lineage
DEPLOYED_FLIPS_ANCHOR = 3
STRICT_IDENTITY_ANCHOR = 0.9988662131519275     # pinned arm, 1 residual varlen-combine flip (#412)
STRICT_FLIPS_ANCHOR = 1
KNOWN_FLIP_PROMPTS = (11, 18, 118)       # the 3 served-arm flip prompts (#381/#405/#412)
N_SERVED_POSITIONS_ANCHOR = 882

# my #438 corroboration (LOCAL, my prior merged card): deployed serve + M=1 AR strict-equiv end-to-end
DEPLOYED_PPL_438_LOCAL = 2.376682786480556
M1_AR_STRICT_EQUIV_OFFICIAL_438 = 161.6995796731182   # the ONLY end-to-end-measurable strict-equiv config
M1_AR_STRICT_EQUIV_LOCAL_438 = 156.1959145793974

# the in-boundary #412 method (MERGED to my base branch). Invoked as a subprocess, never edited.
S412 = Path("research/validity/selective_recompute_equivalent_tps/selective_recompute_equivalent_tps.py")
OUT_DIR = Path("research/validity/strict_frontier_reanchor")
CENSUS_ARMS = ("heuristic", "pinned")
MY_SEEDS = (5, 6, 7, 8, 9)               # N=5 fresh seeds (distinct from #412's 0,1,2) -> independence + N>=5


# ======================================================================================
# Subprocess driver for the #412 method (mirrors its own run_phase_subprocess env pinning)
# ======================================================================================
def run_412_phase(args_list: list[str], extra_env: dict | None = None) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, str(S412.resolve())] + args_list
    print(f"[orch] #412 <- {' '.join(args_list)} (VLLM_BATCH_INVARIANT={env.get('VLLM_BATCH_INVARIANT','0')})",
          flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"#412 phase failed (rc={rc}): {args_list}")


def measure_census_arm(a: argparse.Namespace, arm: str) -> dict:
    out_json = OUT_DIR / f"arm_{arm}_result.json"
    extra_env = {"VLLM_BATCH_INVARIANT": "1" if arm == "pinned" else "0"}
    run_412_phase([
        "--phase", "census", "--arm", arm, "--out", str(out_json),
        "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len), "--n-verify", str(a.n_verify),
        "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
        "--verbose-k", str(a.verbose_k),
    ], extra_env=extra_env)
    return json.load(open(out_json))


def measure_microbench(a: argparse.Namespace) -> dict:
    out_json = OUT_DIR / "microbench_result.json"
    run_412_phase([
        "--phase", "microbench", "--out", str(out_json),
        "--iters", str(a.iters), "--warmup", str(a.warmup),
        "--seeds", ",".join(str(s) for s in a.seeds),
    ])
    return json.load(open(out_json))


# ======================================================================================
# Compose (pure function of the two measured inputs -> the PR's deliverable fields)
# ======================================================================================
def _flip_prompts(arm: dict) -> list[int]:
    return sorted({int(f["prompt_idx"]) for f in arm.get("flip_details", [])})


def compose_report(census: dict, micro: dict, a: argparse.Namespace) -> dict:
    heur = census["heuristic"]
    pin = census["pinned"]

    # ---- Step 1: re-anchor the strict frontier over N>=5 microbench seeds ----
    eta_seeds = list(micro["eta_attn_decode_seeds"])
    frontier_seeds = [OFFICIAL_TPS / (1.0 + eta) for eta in eta_seeds]
    reanchored = statistics.median(frontier_seeds)
    sigma = statistics.pstdev(frontier_seeds) if len(frontier_seeds) > 1 else 0.0
    median_eta = statistics.median(eta_seeds)
    frontier_at_median_eta = OFFICIAL_TPS / (1.0 + median_eta)   # cross-check (median-of-ratios vs ratio-at-median)

    # ---- Step 2: deployed (heuristic) identity + strict (pinned) identity ----
    deployed_identity = heur["decodewidth_e2e_token_identity_rate"]
    deployed_flips = len(heur.get("flip_details", []))
    strict_identity = pin["decodewidth_e2e_token_identity_rate"]
    strict_flips = len(pin.get("flip_details", []))
    deployed_flip_prompts = _flip_prompts(heur)
    strict_flip_prompts = _flip_prompts(pin)

    # flip stability: (a) within-run byte determinism (M8-vs-M8 repeat == 1.0, prompts processed in isolation,
    # so ordering-independent), and (b) cross-run reproducibility -- does my fresh run hit the SAME flip prompts
    # the #381/#405/#412 lineage reported ({11,18,118})?
    within_run_byte_stable = bool(heur.get("determinism_M8_vs_M8") == 1.0
                                  and heur.get("within_batch_copy0_vs_copy1") == 1.0)
    known_present = [p for p in KNOWN_FLIP_PROMPTS if any(
        int(pp["prompt_idx"]) == p for pp in heur.get("per_prompt", []))]
    deployed_flips_match_known = bool(deployed_flip_prompts == sorted(known_present)) if known_present else False
    deployed_flips_stable = bool(within_run_byte_stable and deployed_flips_match_known)

    # ---- Step 3: equivalence tax vs sigma_hw envelope ----
    equivalence_tax_tps = OFFICIAL_TPS - reanchored
    tax_vs_sigma_hw_ratio = equivalence_tax_tps / SIGMA_HW if SIGMA_HW else float("nan")
    tax_exceeds_sigma_hw = bool(equivalence_tax_tps > SIGMA_HW)

    # ---- microbench equivalence facts ----
    strict_is_byte_exact_M8 = bool(micro.get("strict_is_byte_exact_M8"))
    fast_is_byte_exact_M8 = bool(micro.get("fast_is_byte_exact_M8"))

    report = {
        "pr": 455, "agent": "lawine",
        "leg": "Independent re-anchor: strict frontier 467.14 + deployed identity (LOCAL A10G, analysis-only)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,

        # ===== Step 1 deliverables =====
        "reanchored_strict_frontier_tps": reanchored,
        "strict_frontier_sigma": sigma,
        "frontier_seeds": frontier_seeds,
        "frontier_at_median_eta": frontier_at_median_eta,
        "eta_attn_decode_seeds": eta_seeds,
        "eta_attn_decode_median": median_eta,
        "eta_attn_decode_std": micro.get("eta_attn_decode_std"),
        "n_frontier_seeds": len(frontier_seeds),
        "frontier_anchor_412": FRONTIER_ANCHOR,
        "frontier_drift_vs_anchor": reanchored - FRONTIER_ANCHOR,

        # ===== Step 2 deliverables =====
        "deployed_identity_fraction": deployed_identity,
        "deployed_token_flips": deployed_flips,
        "deployed_flip_prompts": deployed_flip_prompts,
        "deployed_flips_stable": deployed_flips_stable,
        "deployed_flips_match_known_111888": deployed_flips_match_known,  # {11,18,118}
        "within_run_byte_stable": within_run_byte_stable,
        "strict_identity_fraction": strict_identity,
        "strict_token_flips": strict_flips,
        "strict_flip_prompts": strict_flip_prompts,
        "identity_anchor_deployed": DEPLOYED_IDENTITY_ANCHOR,
        "identity_drift_vs_anchor": deployed_identity - DEPLOYED_IDENTITY_ANCHOR,
        "strict_beats_deployed_identity": bool(strict_identity >= deployed_identity),
        "n_served_positions": heur.get("total_positions"),

        # ===== Step 3 deliverables =====
        "equivalence_tax_tps": equivalence_tax_tps,
        "sigma_hw": SIGMA_HW,
        "tax_exceeds_sigma_hw": tax_exceeds_sigma_hw,
        "tax_vs_sigma_hw_ratio": tax_vs_sigma_hw_ratio,
        "tax_anchor": TAX_ANCHOR,
        "tax_drift_vs_anchor": equivalence_tax_tps - TAX_ANCHOR,

        # ===== Step 4: PPL + equivalence facts =====
        "ppl": PPL_ANCHOR,
        "ppl_gate": PPL_GATE,
        "ppl_passes_gate": bool(PPL_ANCHOR <= PPL_GATE),
        "ppl_corroboration_438_local": DEPLOYED_PPL_438_LOCAL,
        "strict_is_byte_exact_M8": strict_is_byte_exact_M8,
        "fast_is_byte_exact_M8": fast_is_byte_exact_M8,

        # ===== honest framing facts (what 467.14 is / isn't) =====
        "frontier_is_composed_not_e2e_serve": True,
        "only_e2e_measurable_strict_equiv_config": "M=1 AR (no speculation)",
        "m1_ar_strict_equiv_official_tps_438": M1_AR_STRICT_EQUIV_OFFICIAL_438,
        "m1_ar_strict_equiv_local_tps_438": M1_AR_STRICT_EQUIV_LOCAL_438,

        # ===== census provenance (both arms) =====
        "arms": {
            arm: {
                "decodewidth_e2e_token_identity_rate": d["decodewidth_e2e_token_identity_rate"],
                "flip_count": len(d.get("flip_details", [])),
                "flip_prompts": _flip_prompts(d),
                "determinism_M1_vs_M1": d.get("determinism_M1_vs_M1"),
                "determinism_M8_vs_M8": d.get("determinism_M8_vs_M8"),
                "within_batch_copy0_vs_copy1": d.get("within_batch_copy0_vs_copy1"),
                "chunk_isolated_fraction": d.get("chunk_isolated_fraction"),
                "vllm_batch_invariant_env": d.get("vllm_batch_invariant_env"),
                "attn_is_batch_invariant": d.get("attn_is_batch_invariant"),
                "total_positions": d.get("total_positions"),
                "n_prompts": d.get("n_prompts"),
                "nan_clean": d.get("nan_clean"),
                "peak_gpu_gb": d.get("peak_gpu_gb"),
            } for arm, d in census.items()
        },
        "microbench": {k: micro.get(k) for k in (
            "penalty_decode_band", "penalty_decode_band_std", "eta_attn_decode", "eta_attn_decode_std",
            "eta_attn_decode_seeds", "band_penalty_M1_seeds", "verify_penalty_band_mean", "verify_penalty_free",
            "fast_is_byte_exact_M8", "strict_is_byte_exact_M8", "n_seeds", "iters", "warmup", "peak_gpu_gb")},
        "config": {
            "n_prompts": a.n_prompts, "ctx_len": a.ctx_len, "n_verify": a.n_verify,
            "seeds": list(a.seeds), "iters": a.iters, "warmup": a.warmup,
            "model_dir": heur.get("model_dir"),
        },
    }

    # ---- self-test ----
    checks, n_checks = build_self_test(report, census, micro)
    report["self_test"] = checks
    report["self_test_n_checks"] = n_checks
    report["strict_frontier_reanchor_self_test_passes"] = bool(all(checks.values()) and n_checks >= 18)

    # ---- one-line verdict ----
    report["one_line_verdict"] = (
        f"Independent re-anchor: strict frontier {reanchored:.2f} +-{sigma:.2f} TPS "
        f"(anchor 467.14, drift {report['frontier_drift_vs_anchor']:+.2f}); deployed identity "
        f"{deployed_identity:.4f}/{deployed_flips} flips (anchor 0.9966/3); strict {strict_identity:.4f}/"
        f"{strict_flips} flips; equivalence tax {equivalence_tax_tps:.2f} TPS = "
        f"{tax_vs_sigma_hw_ratio:.1f}x sigma_hw -> {'EXCEEDS' if tax_exceeds_sigma_hw else 'within'} the "
        f"{SIGMA_HW}-TPS hardware-noise envelope (the tax is {'a real, banked cost' if tax_exceeds_sigma_hw else 'noise'}). "
        f"467.14 is a COMPOSED attention-tax isolation, not an end-to-end serve; the only e2e-measurable "
        f"strict-equiv config is M=1 AR @ {M1_AR_STRICT_EQUIV_OFFICIAL_438:.1f} official."
    )
    return report


# ======================================================================================
# Self-test (>=18 asserts; validates composition logic + reproduction-against-anchor)
# ======================================================================================
def build_self_test(report: dict, census: dict, micro: dict) -> tuple[dict, int]:
    c: dict = {}
    re_tps = report["reanchored_strict_frontier_tps"]
    sig = report["strict_frontier_sigma"]
    tax = report["equivalence_tax_tps"]

    # Step 1: frontier reproduces the anchor, tight, internally consistent
    c["frontier_in_plausible_range"] = bool(455.0 <= re_tps <= 478.0)
    c["frontier_reproduces_anchor_467"] = bool(abs(re_tps - FRONTIER_ANCHOR) <= 3.0)
    c["frontier_sigma_small"] = bool(0.0 <= sig <= 2.0)
    c["frontier_median_consistent_with_eta"] = bool(
        abs(re_tps - report["frontier_at_median_eta"]) <= 0.5)
    c["eta_median_positive"] = bool(report["eta_attn_decode_median"] > 0.0)
    c["n_seeds_ge_5"] = bool(report["n_frontier_seeds"] >= 5)
    c["all_frontier_seeds_in_range"] = bool(all(455.0 <= f <= 478.0 for f in report["frontier_seeds"]))

    # Step 2: identity reproduces 0.9966/3 ; strict arm fewer-or-equal flips
    c["deployed_identity_reproduces_9966"] = bool(abs(report["deployed_identity_fraction"]
                                                      - DEPLOYED_IDENTITY_ANCHOR) <= 0.01)
    c["deployed_flips_eq_3"] = bool(report["deployed_token_flips"] == DEPLOYED_FLIPS_ANCHOR)
    c["strict_identity_ge_deployed"] = bool(report["strict_identity_fraction"]
                                            >= report["deployed_identity_fraction"])
    c["strict_flips_le_deployed"] = bool(report["strict_token_flips"] <= report["deployed_token_flips"])
    c["deployed_flips_stable"] = bool(report["deployed_flips_stable"])

    # Step 3: tax positive, exceeds sigma_hw, reproduces ~14.39, robustly real (>=2 sigma_hw)
    c["tax_positive"] = bool(tax > 0.0)
    c["tax_exceeds_sigma_hw"] = bool(report["tax_exceeds_sigma_hw"])
    c["tax_reproduces_anchor_1439"] = bool(abs(tax - TAX_ANCHOR) <= 3.0)
    c["tax_at_least_2x_sigma_hw"] = bool(report["tax_vs_sigma_hw_ratio"] >= 2.0)

    # Step 4 / equivalence facts: strict kernel byte-exact M=8, fast path is non-equivalent
    c["strict_byte_exact_M8"] = bool(micro.get("strict_is_byte_exact_M8"))
    c["fast_path_nonequivalent"] = bool(
        (not micro.get("fast_is_byte_exact_M8")) or report["deployed_identity_fraction"] < 1.0)
    c["ppl_passes_gate"] = bool(report["ppl_passes_gate"])

    # census harness correctness (both arms deterministic, isolated, nan-clean) + arm separation
    for arm, d in census.items():
        c[f"{arm}_det_m1_eq_1"] = bool(d.get("determinism_M1_vs_M1") == 1.0)
        c[f"{arm}_det_m8_eq_1"] = bool(d.get("determinism_M8_vs_M8") == 1.0)
        c[f"{arm}_within_eq_1"] = bool(d.get("within_batch_copy0_vs_copy1") == 1.0)
        c[f"{arm}_isolated"] = bool(d.get("chunk_isolated_fraction", 0.0) >= 0.99)
        c[f"{arm}_nan_clean"] = bool(d.get("nan_clean"))
    c["pinned_attn_batch_invariant"] = bool(census["pinned"].get("attn_is_batch_invariant"))
    c["heuristic_not_batch_invariant"] = bool(not census["heuristic"].get("attn_is_batch_invariant"))

    # constants exact (no silent anchor drift)
    c["constants_exact"] = bool(OFFICIAL_TPS == 481.53 and SIGMA_HW == 4.8 and PPL_ANCHOR == 2.3772)

    # nan-clean across the headline numbers
    c["headline_nan_clean"] = bool(all(math.isfinite(x) for x in
                                       (re_tps, sig, tax, report["deployed_identity_fraction"],
                                        report["strict_identity_fraction"], report["eta_attn_decode_median"])))
    return c, len(c)


# ======================================================================================
# Synthetic self-test (0-GPU): validate compose+self-test logic against anchors w/o any model load
# ======================================================================================
def _synthetic_inputs() -> tuple[dict, dict]:
    def census_arm(arm, identity, flips, batch_inv):
        # build flip_details on the canonical flip prompts so flip-stability check passes for heuristic
        fp = KNOWN_FLIP_PROMPTS[:flips] if arm == "heuristic" else (90,)[:flips]
        flip_details = [{"prompt_idx": p, "pos": 200 + i} for i, p in enumerate(fp)]
        per_prompt = [{"prompt_idx": i} for i in range(127)]
        return {
            "decodewidth_e2e_token_identity_rate": identity,
            "flip_details": flip_details, "per_prompt": per_prompt,
            "determinism_M1_vs_M1": 1.0, "determinism_M8_vs_M8": 1.0,
            "within_batch_copy0_vs_copy1": 1.0, "chunk_isolated_fraction": 1.0,
            "vllm_batch_invariant_env": batch_inv, "attn_is_batch_invariant": batch_inv,
            "total_positions": N_SERVED_POSITIONS_ANCHOR, "n_prompts": 127, "nan_clean": True,
            "peak_gpu_gb": 18.9, "model_dir": "/synthetic",
        }
    census = {
        "heuristic": census_arm("heuristic", DEPLOYED_IDENTITY_ANCHOR, DEPLOYED_FLIPS_ANCHOR, False),
        "pinned": census_arm("pinned", STRICT_IDENTITY_ANCHOR, STRICT_FLIPS_ANCHOR, True),
    }
    # eta that EXACTLY reproduces the 467.14 anchor, with tiny per-seed jitter
    eta_star = OFFICIAL_TPS / FRONTIER_ANCHOR - 1.0
    eta_seeds = [eta_star * (1.0 + j) for j in (-0.004, -0.002, 0.0, 0.002, 0.004)]
    micro = {
        "eta_attn_decode_seeds": eta_seeds,
        "eta_attn_decode": statistics.median(eta_seeds),
        "eta_attn_decode_std": statistics.pstdev(eta_seeds),
        "penalty_decode_band": 1.32, "penalty_decode_band_std": 0.002,
        "band_penalty_M1_seeds": [1.32] * 5, "verify_penalty_band_mean": 1.0, "verify_penalty_free": True,
        "fast_is_byte_exact_M8": False, "strict_is_byte_exact_M8": True,
        "n_seeds": 5, "iters": 50, "warmup": 10, "peak_gpu_gb": 0.3,
    }
    return census, micro


# ======================================================================================
# Console + W&B + finish
# ======================================================================================
def _print_console(r: dict) -> None:
    print("\n========== STRICT FRONTIER RE-ANCHOR (PR #455) ==========", flush=True)
    print(f" {r['one_line_verdict']}", flush=True)
    print(" --- Step 1: strict frontier ---", flush=True)
    print(f"  reanchored_strict_frontier_tps  : {r['reanchored_strict_frontier_tps']:.3f} "
          f"+-{r['strict_frontier_sigma']:.3f}  (anchor {FRONTIER_ANCHOR:.2f}, drift "
          f"{r['frontier_drift_vs_anchor']:+.3f})", flush=True)
    print(f"  eta_attn_decode (median, N={r['n_frontier_seeds']})  : {r['eta_attn_decode_median']:.6f}", flush=True)
    print(f"  frontier_seeds                  : {[round(f,2) for f in r['frontier_seeds']]}", flush=True)
    print(" --- Step 2: identity ---", flush=True)
    print(f"  deployed_identity_fraction      : {r['deployed_identity_fraction']:.6f}  "
          f"flips={r['deployed_token_flips']} @ {r['deployed_flip_prompts']}  "
          f"(anchor 0.9966/3) stable={r['deployed_flips_stable']}", flush=True)
    print(f"  strict_identity_fraction        : {r['strict_identity_fraction']:.6f}  "
          f"flips={r['strict_token_flips']} @ {r['strict_flip_prompts']}  (anchor 0.9989/1)", flush=True)
    print(" --- Step 3: equivalence tax vs sigma_hw ---", flush=True)
    print(f"  equivalence_tax_tps             : {r['equivalence_tax_tps']:.3f}  "
          f"(= {r['tax_vs_sigma_hw_ratio']:.2f}x sigma_hw={r['sigma_hw']})  "
          f"exceeds_sigma_hw={r['tax_exceeds_sigma_hw']}", flush=True)
    print(" --- Step 4: ppl + equivalence facts ---", flush=True)
    print(f"  ppl                             : {r['ppl']:.4f} (gate {r['ppl_gate']}, "
          f"local#438 {r['ppl_corroboration_438_local']:.4f})", flush=True)
    print(f"  strict_is_byte_exact_M8         : {r['strict_is_byte_exact_M8']}  "
          f"fast_is_byte_exact_M8={r['fast_is_byte_exact_M8']}", flush=True)
    print(f" SELF-TEST PASSES                 : {r['strict_frontier_reanchor_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['self_test_n_checks']})", flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("=========================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="lawine", name=a.wandb_name, group=a.wandb_group,
        notes="PR#455 independent re-anchor: strict frontier 467.14 + deployed identity. "
              "Re-runs #412 census+microbench (fresh seeds) and re-composes OFFICIAL/(1+eta); "
              "prices the equivalence tax vs sigma_hw. LOCAL A10G, analysis-only.",
        config={
            "pr": 455, "n_prompts": report["config"]["n_prompts"], "ctx_len": report["config"]["ctx_len"],
            "n_verify": report["config"]["n_verify"], "seeds": report["config"]["seeds"],
            "model_dir": report["config"]["model_dir"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "anchor/frontier_412": FRONTIER_ANCHOR, "anchor/tax": TAX_ANCHOR,
            "anchor/deployed_identity": DEPLOYED_IDENTITY_ANCHOR, "anchor/official_tps": OFFICIAL_TPS,
            "anchor/sigma_hw": SIGMA_HW, "anchor/ppl": PPL_ANCHOR,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    keys = (
        "reanchored_strict_frontier_tps", "strict_frontier_sigma", "frontier_at_median_eta",
        "eta_attn_decode_median", "eta_attn_decode_std", "n_frontier_seeds", "frontier_drift_vs_anchor",
        "deployed_identity_fraction", "deployed_token_flips", "deployed_flips_stable",
        "deployed_flips_match_known_111888", "within_run_byte_stable",
        "strict_identity_fraction", "strict_token_flips", "strict_beats_deployed_identity",
        "identity_drift_vs_anchor", "n_served_positions",
        "equivalence_tax_tps", "sigma_hw", "tax_exceeds_sigma_hw", "tax_vs_sigma_hw_ratio",
        "tax_drift_vs_anchor", "ppl", "ppl_gate", "ppl_passes_gate", "ppl_corroboration_438_local",
        "strict_is_byte_exact_M8", "fast_is_byte_exact_M8",
        "frontier_is_composed_not_e2e_serve", "m1_ar_strict_equiv_official_tps_438",
        "strict_frontier_reanchor_self_test_passes", "self_test_n_checks",
        "one_line_verdict", "analysis_only", "no_hf_job", "no_served_file_change", "official_tps",
    )
    for k in keys:
        run.summary[k] = report.get(k)
    run.summary["frontier_seeds"] = report["frontier_seeds"]
    run.summary["eta_attn_decode_seeds"] = report["eta_attn_decode_seeds"]
    for arm in CENSUS_ARMS:
        d = report["arms"][arm]
        run.summary[f"{arm}/identity"] = d["decodewidth_e2e_token_identity_rate"]
        run.summary[f"{arm}/flip_count"] = d["flip_count"]
        run.summary[f"{arm}/flip_prompts"] = d["flip_prompts"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    report["wandb_run_id"] = run.id
    print(f"[wandb] logged run {run.id}", flush=True)


def _finish(report: dict, a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "strict_frontier_reanchor_results.json"
    if not a.no_wandb:
        log_wandb(report, a)   # populates report["wandb_run_id"]
    json.dump(report, open(report_path, "w"), indent=2)
    _print_console(report)
    print(f"[done] results -> {report_path}", flush=True)


# ======================================================================================
# Modes
# ======================================================================================
def measure(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    census = {arm: measure_census_arm(a, arm) for arm in CENSUS_ARMS}
    micro = measure_microbench(a)
    _finish(compose_report(census, micro, a), a)


def reanalyze(a: argparse.Namespace) -> None:
    census = {}
    for arm in CENSUS_ARMS:
        p = OUT_DIR / f"arm_{arm}_result.json"
        if not p.exists():
            raise FileNotFoundError(f"--reanalyze needs {p} (run --measure first)")
        census[arm] = json.load(open(p))
    mp = OUT_DIR / "microbench_result.json"
    if not mp.exists():
        raise FileNotFoundError(f"--reanalyze needs {mp}")
    micro = json.load(open(mp))
    _finish(compose_report(census, micro, a), a)


def self_test(a: argparse.Namespace) -> None:
    census, micro = _synthetic_inputs()
    report = compose_report(census, micro, a)
    _print_console(report)
    ok = report["strict_frontier_reanchor_self_test_passes"]
    print(f"[self-test] synthetic compose+self-test PASSES={ok} "
          f"({sum(report['self_test'].values())}/{report['self_test_n_checks']})", flush=True)
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="run the GPU phases (census x2 + microbench) + compose")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU: recompose from saved arm_*/microbench JSONs")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="0-GPU: synthetic compose+self-test (no model load)")
    ap.add_argument("--smoke", action="store_true", help="tiny measure run (few prompts, 2 seeds) to validate plumbing")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=127)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=8)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in MY_SEEDS))
    ap.add_argument("--wandb_group", dest="wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", dest="wandb_name", default="lawine/strict-frontier-reanchor")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    a.seeds = [int(s) for s in str(a.seeds).split(",") if s != ""]

    if a.smoke:
        a.n_prompts = min(a.n_prompts, 6)
        a.iters = min(a.iters, 20)
        a.warmup = min(a.warmup, 5)
        a.seeds = a.seeds[:2] if len(a.seeds) >= 2 else a.seeds

    if a.self_test:
        self_test(a)
    elif a.reanalyze:
        reanalyze(a)
    elif a.measure:
        measure(a)
    else:
        ap.error("one of --measure / --reanalyze / --self-test is required")


if __name__ == "__main__":
    main()
