#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #471 (denken) -- Strict-submission identity certifier: a VALIDATED identity=1.0 oracle.

WHY (the instrument this builds)
--------------------------------
The human authorized a strict leaderboard submission (#407, 2026-06-16 07:26Z) for any config clearly above
161.70, demanded "honest & strict". The non-negotiable contract (#319) is `token_identity_rate = 1.0` --
byte-exact greedy vs the M=1 AR reference, ZERO flips. land #469 pre-stages the submission command + an
identity-1.0 gate; ubel/#466 realize the TPS. The MISSING piece is the AUTHORITATIVE, VALIDATED identity=1.0
certifier -- the instrument that PROVES the config we put on a public board is genuinely byte-exact strict,
not census-lucky.

THE STRONGER GATE (vs #464)
---------------------------
#464 (`1o7jwlw4`) built the census-anchored QUALITY gate: "are the flips that exist quality-neutral?" (answer:
the 3 deployed flips are bitwise ties, quality-neutral). The strict SUBMISSION needs the stronger, simpler
gate: "are there ZERO flips AT ALL?" `is_strict_1p0 = True` IFF n_flips == 0 -- a LITERAL byte-exact rule,
distinct from #464's quality-neutrality. This card hardens the census harness into that submission certifier
and VALIDATES it on configs whose answer we already know, so its verdict on the #466 config is trustworthy.

THE CENSUS MECHANISM (reused, NOT reimplemented)
------------------------------------------------
`token_identity_rate` is the WITHIN-arm decode-width identity: at a fixed prefix C, run M=1 AR greedy to get
the strict reference continuation, then read the M=8 chunked-verify argmax at each continuation position; a
position MATCHES iff M8-argmax == M1-AR token, a FLIP iff they differ. This is exactly the program.md strict
contract ("greedy decode token-identical to plain greedy AR for the submitted checkpoint"). The measurement
engine is stark #412's in-boundary census (`selective_recompute_equivalent_tps.py --phase census --arm <arm>`,
MERGED to approval-gated-8gpu-20260613); this certifier DRIVES it as a black-box subprocess (per-arm env pin,
process isolation -- VLLM_BATCH_INVARIANT must be set before vllm import) and applies the zero-flip rule.

CONFIG -> ARM map (the three known-answer anchors):
  deployed        481.53  -> arm=heuristic (VLLM_BATCH_INVARIANT=0)  served fast M=8 spec-verify. EXPECT REJECT
                                                                     (identity ~0.9966, 3 flips {11,18,118}).
  blanket_strict  467.14  -> arm=pinned    (VLLM_BATCH_INVARIANT=1)  batch-invariant attention everywhere.
                                                                     CENSUS its identity (genuine measurement).
  m1_ar           161.70  -> non-speculative M=1 AR. Verify width 1 -> NO chunked-verify reduction-order
                            divergence is possible -> identity 1.0 / 0 flips BY CONSTRUCTION (the served greedy
                            path IS the M=1 AR reference). Corroborated empirically by determinism_M1_vs_M1==1.0
                            measured in the heuristic census. EXPECT ACCEPT. This is the strict FLOOR (#438).

THE DECISIVE FINDING (honest, surfaced LOUD)
--------------------------------------------
Under the LITERAL zero-flip rule the certifier ACCEPTS only m1_ar (161.70). blanket_strict (467.14) carries
ONE residual flip @ prompt 90 (identity ~0.9989, a BITWISE TIE m1_self_gap=0.0; lawine #455, stark #429) ->
literal is_strict_1p0=FALSE. stark #429 (`blanket_strict_operative_identity`) showed that lone flip is a
fixed-point of the verify-arbiter (OPERATIVE identity 1.0) and punted "literal vs operative" to a
`human_contract_decision`. This certifier reports the LITERAL verdict (the #319 contract as worded) PLUS the
operative caveat, so the human gate decides with full information. It does NOT silently pass a 0.9989 config.

SCOPE: LOCAL A10G (sm_86), MEASUREMENT + analysis only. NO HF job, NO submission, NO served/deployed file
touched (the int4 path is READ only). analysis_only / no_hf_job / no_served_file_change / official_tps=0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# ======================================================================================
# Anchors (CITE merged cards / public W&B; do NOT re-derive)
# ======================================================================================
PPL_ANCHOR = 2.3772                  # deployed/strict PPL (teacher-forced; PR #52 / denken #423 / stark #429)
PPL_GATE = 2.42                      # validity gate ceiling
STRICT_FLOOR_TPS = 161.70            # M=1 AR strict floor, identity 1.0, 0 flips (lawine #438)

# deployed (heuristic) anchor -- the KNOWN non-strict config (lawine #455 `0r0ounl8`, denken #464 `1o7jwlw4`)
DEPLOYED_TPS = 481.53
DEPLOYED_IDENTITY_ANCHOR = 0.9965986394557823    # 879/882; 3 flips
DEPLOYED_FLIPS_ANCHOR = 3
DEPLOYED_FLIP_PROMPTS = (11, 18, 118)

# blanket-strict (pinned) anchor -- the config ubel/#466 realize for SPEED (denken #423, stark #429 `wvy2k7w7`/#412)
BLANKET_STRICT_TPS = 467.14
BLANKET_STRICT_IDENTITY_ANCHOR = 0.9988662131519275   # 881/882; 1 residual flip
BLANKET_STRICT_FLIPS_ANCHOR = 1
BLANKET_STRICT_FLIP_PROMPTS = (90,)
BLANKET_STRICT_OPERATIVE_IDENTITY = 1.0               # verify-arbiter fixed-point reading (stark #429)

IDENTITY_TOL = 0.01                  # reproduce-the-anchor tolerance (census denominator drifts +/- a few positions)
BAND_TOL = 1e-9                      # bitwise-tie tolerance (m1_self_gap <= BAND_TOL => bit-identical top-2 logits)

# The in-boundary census engine (MERGED). Driven as a subprocess; never edited.
S412 = Path("research/validity/selective_recompute_equivalent_tps/selective_recompute_equivalent_tps.py")
OUT_DIR = Path("research/validity/strict_submission_identity_certifier")
REPORT_JSON = OUT_DIR / "strict_submission_identity_certifier_results.json"

# CONFIG registry: name -> (arm, VLLM_BATCH_INVARIANT, by_construction, tps, note)
CONFIG_REGISTRY: dict[str, dict] = {
    "deployed": {
        "arm": "heuristic", "vbi": "0", "by_construction": None, "tps": DEPLOYED_TPS,
        "note": "served fast M=8 spec-verify (VLLM_BATCH_INVARIANT=0)",
    },
    "blanket_strict": {
        "arm": "pinned", "vbi": "1", "by_construction": None, "tps": BLANKET_STRICT_TPS,
        "note": "batch-invariant attention everywhere (VLLM_BATCH_INVARIANT=1)",
    },
    "m1_ar": {
        "arm": None, "vbi": None, "by_construction": "m1_ar", "tps": STRICT_FLOOR_TPS,
        "note": "non-speculative M=1 AR -- verify width 1, no chunked-verify divergence possible",
    },
}


# ======================================================================================
# Census subprocess driver (mirror #412/#464 env pinning; per-arm process isolation)
# ======================================================================================
def _pin_env(vbi: str) -> dict:
    """Pin the exact env #412/#464 use. VLLM_BATCH_INVARIANT chosen per-arm; set BEFORE vllm import (subprocess)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"                 # container exposes one A10G as index 0 (inherited =5 is wrong)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"          # avoid curand.h JIT failure (greedy uses argmax anyway)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["VLLM_BATCH_INVARIANT"] = vbi
    return env


def run_census_arm(arm: str, vbi: str, a: argparse.Namespace) -> dict:
    """Drive `selective_recompute_equivalent_tps.py --phase census --arm <arm>` (black box) and read its JSON."""
    out_json = OUT_DIR / f"census_{arm}_result.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(S412.resolve()),
           "--phase", "census", "--arm", arm, "--out", str(out_json),
           "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len), "--n-verify", str(a.n_verify),
           "--gpu-mem-util", str(a.gpu_mem_util), "--max-batched-tokens", str(a.max_batched_tokens),
           "--verbose-k", str(a.verbose_k)]
    print(f"[certify] census(arm={arm}, VLLM_BATCH_INVARIANT={vbi}) <- n_prompts={a.n_prompts} "
          f"ctx_len={a.ctx_len} n_verify={a.n_verify}", flush=True)
    rc = subprocess.run(cmd, env=_pin_env(vbi)).returncode
    if rc != 0:
        raise RuntimeError(f"census arm={arm} failed (rc={rc})")
    return json.load(open(out_json))


# ======================================================================================
# certify_strict_submission(config) -> the certifier verdict for ONE config
# ======================================================================================
def _flip_prompts(census: dict) -> list[int]:
    return sorted({int(f["prompt_idx"]) for f in census.get("flip_details", [])})


def certify_from_census(config_name: str, census: dict, tps: float, note: str) -> dict:
    """Apply the LITERAL zero-flip strict rule to a measured census. is_strict_1p0 IFF n_flips == 0."""
    identity = census.get("decodewidth_e2e_token_identity_rate")
    flips = census.get("flip_details", [])
    n_flips = len(flips)
    flip_prompts = _flip_prompts(census)
    n_positions = census.get("total_positions")
    n_prompts = census.get("n_prompts")
    det_m1 = census.get("determinism_M1_vs_M1")
    det_m8 = census.get("determinism_M8_vs_M8")
    within = census.get("within_batch_copy0_vs_copy1")

    # operative reading: are ALL residual flips bitwise ties? (stark #429: a tie-flip is a verify-arbiter
    # fixed point -> operative-1.0; literal-vs-operative is a human_contract_decision). LITERAL gate ignores this.
    all_residual_flips_bitwise_tie = bool(n_flips > 0 and all(bool(f.get("m1_is_bitwise_tie")) for f in flips))

    # the LITERAL strict-submission verdict: byte-exact, zero flips, on a deterministic census.
    census_is_sound = bool(det_m1 == 1.0 and det_m8 == 1.0 and within == 1.0
                           and isinstance(n_positions, int) and n_positions > 0)
    is_strict_1p0 = bool(n_flips == 0 and identity is not None and abs(identity - 1.0) <= BAND_TOL
                         and census_is_sound)

    return {
        "config": config_name, "by_construction": False, "tps_context": tps, "note": note,
        "identity_rate": identity, "n_flips": n_flips, "flip_prompts": flip_prompts,
        "is_strict_1p0": is_strict_1p0,
        "strict_gate_requires_zero_flips": True,
        "n_positions": n_positions, "n_prompts_censused": n_prompts,
        "determinism_M1_vs_M1": det_m1, "determinism_M8_vs_M8": det_m8, "within_batch_copy0_vs_copy1": within,
        "census_is_sound": census_is_sound,
        # operative caveat (REPORTED, non-gating for the literal verdict):
        "all_residual_flips_bitwise_tie": all_residual_flips_bitwise_tie,
        "operative_note": (
            "" if n_flips == 0 else
            ("all residual flips are bitwise ties (m1_self_gap=0.0): operative identity 1.0 under the "
             "verify-arbiter fixed-point reading (stark #429); literal-vs-operative is a human_contract_decision"
             if all_residual_flips_bitwise_tie else
             "WARNING: a residual flip is NOT a bitwise tie -- a confident-argmax divergence, not a tie-break")),
        "ppl": PPL_ANCHOR, "completion_note": (
            "identity certifier covers the PPL-ground-truth prompt set at the M=8 verify width (bounded "
            "decode window per prompt); FULL 128/128 free-running completion + PPL are measured by the HF "
            "benchmark (ubel/#466), not by this card"),
    }


def certify_m1_ar_by_construction(heuristic_census: dict | None) -> dict:
    """M=1 AR (161.70): verify width 1 -> the served greedy path IS the M=1 AR reference -> identity 1.0 / 0 flips
    BY CONSTRUCTION (no chunked-verify reduction order to diverge). Empirically corroborated by the measured
    determinism_M1_vs_M1==1.0 (the M=1 decode is deterministic run-to-run)."""
    det_m1 = heuristic_census.get("determinism_M1_vs_M1") if heuristic_census else None
    n_positions = heuristic_census.get("total_positions") if heuristic_census else None
    n_prompts = heuristic_census.get("n_prompts") if heuristic_census else None
    m1_deterministic = bool(det_m1 == 1.0)
    return {
        "config": "m1_ar", "by_construction": True, "tps_context": STRICT_FLOOR_TPS,
        "note": "non-speculative M=1 AR -- verify width 1, served greedy IS the M=1 AR reference",
        "identity_rate": 1.0, "n_flips": 0, "flip_prompts": [],
        "is_strict_1p0": bool(m1_deterministic),     # accept iff the empirical M=1 determinism corroborates it
        "strict_gate_requires_zero_flips": True,
        "n_positions": n_positions, "n_prompts_censused": n_prompts,
        "determinism_M1_vs_M1": det_m1, "determinism_M8_vs_M8": None, "within_batch_copy0_vs_copy1": None,
        "census_is_sound": m1_deterministic,
        "all_residual_flips_bitwise_tie": False,
        "operative_note": "",
        "m1_determinism_corroborates_construction": m1_deterministic,
        "ppl": PPL_ANCHOR, "completion_note": (
            "M=1 AR identity is 1.0 by construction (no verify-width divergence); corroborated by measured "
            "determinism_M1_vs_M1==1.0"),
    }


def certify_strict_submission(config_name: str, a: argparse.Namespace,
                              heuristic_census_for_m1: dict | None = None) -> dict:
    """PUBLIC oracle: certify ONE config. Drives the census engine (fresh GPU) unless by_construction."""
    spec = CONFIG_REGISTRY.get(config_name)
    if spec is None:
        raise ValueError(f"unknown config {config_name!r}; known: {sorted(CONFIG_REGISTRY)}")
    if spec["by_construction"] == "m1_ar":
        return certify_m1_ar_by_construction(heuristic_census_for_m1)
    census = run_census_arm(spec["arm"], spec["vbi"], a)
    return certify_from_census(config_name, census, spec["tps"], spec["note"])


# ======================================================================================
# Compose VALIDATE report (the three known-answer anchors + the gate it produces)
# ======================================================================================
def _classifies_reject(v: dict, identity_anchor: float, flips_anchor: int) -> bool:
    """A known NON-strict config is classified correctly iff: is_strict_1p0=False, identity reproduces the
    anchor (within tol), and the flip count is the expected non-zero class."""
    return bool(v["is_strict_1p0"] is False
                and v["identity_rate"] is not None and abs(v["identity_rate"] - identity_anchor) <= IDENTITY_TOL
                and v["n_flips"] == flips_anchor)


def _classifies_accept(v: dict) -> bool:
    """A known STRICT config is classified correctly iff: is_strict_1p0=True, identity==1.0, 0 flips."""
    return bool(v["is_strict_1p0"] is True and v["identity_rate"] == 1.0 and v["n_flips"] == 0)


def compose_validate(deployed: dict, blanket: dict, m1: dict, a: argparse.Namespace,
                     self_test_passes: bool) -> dict:
    # the two EXTERNALLY-known anchors (the falsifiable validation: must REJECT deployed, ACCEPT m1_ar)
    deployed_ok = _classifies_reject(deployed, DEPLOYED_IDENTITY_ANCHOR, DEPLOYED_FLIPS_ANCHOR)
    m1_ok = _classifies_accept(m1)
    # blanket-strict: the certifier must apply the rule CONSISTENTLY to its measured flips (no rubber stamp):
    #   is_strict_1p0 == (n_flips == 0). We do not pre-assert blanket's answer; we assert the rule is applied.
    blanket_rule_consistent = bool(blanket["is_strict_1p0"] == (blanket["n_flips"] == 0))
    certifier_validates_on_known_configs = bool(deployed_ok and m1_ok and blanket_rule_consistent)

    # full-128 coverage of the submission prompt set (all 128 PPL-gt prompts meet C+1; some may early-stop <8 tok)
    censused = deployed.get("n_prompts_censused")
    covers_full_128 = bool(isinstance(censused, int) and censused >= 127)   # >=127 effective of 128 (>=1 early-stop ok)

    # the staged #466 path: the instrument is built + validated -> ready to run once on the confirmed #466 config.
    certifier_ready_for_466 = bool(certifier_validates_on_known_configs and self_test_passes)
    one_line_call_for_466 = (
        ".venv/bin/python research/validity/strict_submission_identity_certifier/"
        "strict_submission_identity_certifier.py --certify <arm> --vbi <0|1> "
        f"--n-prompts {a.n_prompts} --no-wandb   # read is_strict_1p0 from the printed JSON")

    report = {
        "pr": 471, "agent": "denken",
        "leg": "Strict-submission identity certifier: validated identity=1.0 oracle for the #466 config "
               "(LOCAL A10G, measurement + analysis only)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
        "strict_contract": "token_identity_rate == 1.0 vs M=1 AR, ZERO flips (#319)",
        "strict_gate_requires_zero_flips": True,

        # ===== the three known-answer anchors =====
        "deployed_identity_rate": deployed["identity_rate"],
        "deployed_n_flips": deployed["n_flips"], "deployed_flip_prompts": deployed["flip_prompts"],
        "deployed_is_strict_1p0": deployed["is_strict_1p0"],
        "deployed_classifies_correctly_REJECT": deployed_ok,

        "m1_ar_identity_rate": m1["identity_rate"],
        "m1_ar_n_flips": m1["n_flips"], "m1_ar_is_strict_1p0": m1["is_strict_1p0"],
        "m1_ar_classifies_correctly_ACCEPT": m1_ok,
        "m1_determinism_corroborates": m1.get("m1_determinism_corroborates_construction"),

        "blanket_strict_identity_rate": blanket["identity_rate"],
        "blanket_strict_n_flips": blanket["n_flips"], "blanket_strict_flip_prompts": blanket["flip_prompts"],
        "blanket_strict_is_strict_1p0": blanket["is_strict_1p0"],
        "blanket_strict_all_residual_flips_bitwise_tie": blanket["all_residual_flips_bitwise_tie"],
        "blanket_strict_operative_note": blanket["operative_note"],
        "blanket_strict_operative_identity_anchor": BLANKET_STRICT_OPERATIVE_IDENTITY,
        "blanket_strict_rule_consistent": blanket_rule_consistent,

        # ===== the gate this card produces =====
        "certifier_validates_on_known_configs": certifier_validates_on_known_configs,   # PRIMARY metric
        "certifier_ready_for_466": certifier_ready_for_466,
        "certifier_self_test_passes": self_test_passes,
        "one_line_call_for_466": one_line_call_for_466,

        # ===== denominator reconciliation (instruction 1) =====
        "submission_set_n_positions": deployed.get("n_positions"),
        "submission_set_n_prompts_censused": censused,
        "submission_set_total_prompts": 128,
        "certifier_covers_full_128_prompts": covers_full_128,
        "denominator_note": (
            "n_positions is the decode-width census denominator: all censused prompts of the 128-prompt PPL "
            "ground-truth set, each contributing the M=8 verify-width positions after a fixed C=224 prefix. "
            "It covers every submission PROMPT (128/128 meet the C+1 length floor) but at a BOUNDED per-prompt "
            "decode window, NOT the full free-running completion. Flips are a property of the M=8 verify "
            "reduction order exercised at every decode step, so this is a strong representative identity census, "
            "not a literal byte-for-byte census of every emitted benchmark token."),

        # ===== anchors =====
        "ppl": PPL_ANCHOR, "ppl_gate": PPL_GATE, "ppl_passes_gate": bool(PPL_ANCHOR <= PPL_GATE),
        "strict_floor_tps": STRICT_FLOOR_TPS,
        "deployed_tps_context": DEPLOYED_TPS, "blanket_strict_tps_context": BLANKET_STRICT_TPS,

        # ===== full per-config verdicts =====
        "per_config": {"deployed": deployed, "blanket_strict": blanket, "m1_ar": m1},
        "config": {
            "n_prompts": a.n_prompts, "ctx_len": a.ctx_len, "n_verify": a.n_verify,
            "gpu_mem_util": a.gpu_mem_util,
            "deployed_peak_gpu_gb": deployed.get("_peak_gpu_gb"),
            "blanket_strict_peak_gpu_gb": blanket.get("_peak_gpu_gb"),
        },
    }

    checks, n_checks = build_self_test(report)
    report["self_test"] = checks
    report["self_test_n_checks"] = n_checks
    report["certifier_self_test_passes"] = bool(self_test_passes and all(checks.values()) and n_checks >= 20)

    report["one_line_verdict"] = (
        f"certifier_validates_on_known_configs={certifier_validates_on_known_configs}: "
        f"deployed REJECT (identity {deployed['identity_rate']:.4f}, {deployed['n_flips']} flips) | "
        f"m1_ar ACCEPT (identity {m1['identity_rate']}, 0 flips) | "
        f"blanket_strict is_strict_1p0={blanket['is_strict_1p0']} "
        f"(identity {blanket['identity_rate']:.4f}, {blanket['n_flips']} flip @ {blanket['flip_prompts']}, "
        f"all bitwise-tie={blanket['all_residual_flips_bitwise_tie']} -> operative-1.0 human_contract_decision). "
        f"Only m1_ar (161.70) passes the LITERAL zero-flip gate today; "
        f"certifier_ready_for_466={certifier_ready_for_466}.")
    return report


# ======================================================================================
# Self-test (>=20 asserts on the report + synthetic CASE A/B/C below validate the RULE)
# ======================================================================================
def build_self_test(report: dict) -> tuple[dict, int]:
    c: dict = {}
    dep = report["per_config"]["deployed"]
    blk = report["per_config"]["blanket_strict"]
    m1 = report["per_config"]["m1_ar"]

    # deployed: reproduces the 3-flip non-strict anchor and is REJECTED
    c["deployed_identity_reproduces_anchor"] = bool(
        dep["identity_rate"] is not None and abs(dep["identity_rate"] - DEPLOYED_IDENTITY_ANCHOR) <= IDENTITY_TOL)
    c["deployed_n_flips_eq_3"] = bool(dep["n_flips"] == DEPLOYED_FLIPS_ANCHOR)
    c["deployed_flip_prompts_match"] = bool(dep["flip_prompts"] == sorted(DEPLOYED_FLIP_PROMPTS))
    c["deployed_is_rejected"] = bool(dep["is_strict_1p0"] is False)
    c["deployed_classifies_correctly"] = bool(report["deployed_classifies_correctly_REJECT"])

    # m1_ar: identity 1.0, 0 flips, ACCEPTED, corroborated by measured determinism
    c["m1_identity_is_1p0"] = bool(m1["identity_rate"] == 1.0)
    c["m1_n_flips_eq_0"] = bool(m1["n_flips"] == 0)
    c["m1_is_accepted"] = bool(m1["is_strict_1p0"] is True)
    c["m1_determinism_corroborates"] = bool(m1.get("m1_determinism_corroborates_construction"))
    c["m1_classifies_correctly"] = bool(report["m1_ar_classifies_correctly_ACCEPT"])

    # blanket_strict: rule applied consistently (is_strict_1p0 == zero-flip); residual flip surfaced honestly
    c["blanket_rule_consistent"] = bool(report["blanket_strict_rule_consistent"])
    c["blanket_identity_in_range"] = bool(
        blk["identity_rate"] is not None and 0.0 <= blk["identity_rate"] <= 1.0)
    c["blanket_residual_flip_surfaced"] = bool(
        (blk["n_flips"] == 0) == (blk["is_strict_1p0"] is True))   # zero-flip <=> strict, honestly
    c["blanket_operative_note_present_iff_flips"] = bool(
        (blk["n_flips"] > 0) == bool(blk["operative_note"]))

    # the gate discriminates: it does NOT pass a config with flips, DOES pass zero-flip
    c["gate_rejects_any_nonzero_flips"] = bool(
        all((v["is_strict_1p0"] is False) for v in (dep, blk) if v["n_flips"] > 0))
    c["gate_accepts_only_zero_flips"] = bool(
        all((v["n_flips"] == 0) for v in (dep, blk, m1) if v["is_strict_1p0"] is True))

    # validation + readiness wiring
    c["validates_requires_both_anchors"] = bool(
        report["certifier_validates_on_known_configs"]
        == (report["deployed_classifies_correctly_REJECT"]
            and report["m1_ar_classifies_correctly_ACCEPT"]
            and report["blanket_strict_rule_consistent"]))
    c["ready_for_466_requires_validation"] = bool(
        (not report["certifier_ready_for_466"]) or report["certifier_validates_on_known_configs"])
    c["one_line_call_present"] = bool("strict_submission_identity_certifier.py --certify"
                                      in report["one_line_call_for_466"])

    # denominator reconciliation present + honest
    c["n_positions_positive"] = bool(isinstance(report["submission_set_n_positions"], int)
                                     and report["submission_set_n_positions"] > 0)
    c["covers_full_128_reported"] = bool(isinstance(report["certifier_covers_full_128_prompts"], bool))
    c["strict_gate_requires_zero_flips"] = bool(report["strict_gate_requires_zero_flips"] is True)

    # anchors + scope
    c["ppl_passes_gate"] = bool(report["ppl_passes_gate"])
    c["constants_exact"] = bool(PPL_ANCHOR == 2.3772 and STRICT_FLOOR_TPS == 161.70)
    c["analysis_only_flags"] = bool(report["analysis_only"] and report["no_served_file_change"]
                                    and report["official_tps"] == 0)
    return c, len(c)


# ======================================================================================
# Synthetic RULE self-test (0-GPU): the certifier is NOT a rubber stamp
# ======================================================================================
def _synthetic_census(identity, flip_prompts, total_positions, n_prompts, bitwise_tie=True):
    flips = [{"prompt_idx": p, "pos": 227, "m1_is_bitwise_tie": bool(bitwise_tie)} for p in flip_prompts]
    return {
        "decodewidth_e2e_token_identity_rate": identity, "flip_details": flips,
        "total_positions": total_positions, "n_prompts": n_prompts,
        "determinism_M1_vs_M1": 1.0, "determinism_M8_vs_M8": 1.0, "within_batch_copy0_vs_copy1": 1.0,
    }


def synthetic_rule_self_test() -> tuple[bool, dict]:
    """Validate the zero-flip RULE without any model load (analog of #464's CASE-B not-a-rubber-stamp).

    CASE-A strict      : 0 flips, identity 1.0           -> is_strict_1p0 = True  (ACCEPT)
    CASE-B non-strict  : 3 flips, identity 0.9966        -> is_strict_1p0 = False (REJECT; not a rubber stamp)
    CASE-C tie-flip    : 1 bitwise-tie flip, 0.9989      -> is_strict_1p0 = False BUT operative caveat surfaces
    CASE-D confident   : 1 NON-tie flip                  -> is_strict_1p0 = False AND operative WARNING (no tie)
    """
    A = certify_from_census("synthA", _synthetic_census(1.0, [], 896, 128), 0.0, "synthetic strict")
    B = certify_from_census("synthB", _synthetic_census(0.9966, [11, 18, 118], 882, 126), 0.0, "synthetic deployed")
    C = certify_from_census("synthC", _synthetic_census(0.9989, [90], 882, 126, bitwise_tie=True), 0.0, "synthetic blanket")
    D = certify_from_census("synthD", _synthetic_census(0.9989, [90], 882, 126, bitwise_tie=False), 0.0, "synthetic confident")
    checks = {
        "A_zero_flip_accepts": A["is_strict_1p0"] is True and A["n_flips"] == 0,
        "B_three_flip_rejects": B["is_strict_1p0"] is False and B["n_flips"] == 3,
        "C_one_tieflip_rejects_literal": C["is_strict_1p0"] is False and C["n_flips"] == 1,
        "C_tieflip_operative_caveat": C["all_residual_flips_bitwise_tie"] is True
                                      and "human_contract_decision" in C["operative_note"],
        "D_confident_flip_rejects": D["is_strict_1p0"] is False,
        "D_confident_flip_warns": D["all_residual_flips_bitwise_tie"] is False and "WARNING" in D["operative_note"],
        "rule_is_zero_flip": all((v["is_strict_1p0"] is True) == (v["n_flips"] == 0) for v in (A, B, C, D)),
    }
    return bool(all(checks.values())), checks


def self_test_mode() -> None:
    ok, checks = synthetic_rule_self_test()
    print("[self-test] synthetic zero-flip RULE (not-a-rubber-stamp):", flush=True)
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}", flush=True)
    print(f"[self-test] certifier RULE self-test PASSES={ok}", flush=True)
    if not ok:
        sys.exit(1)


# ======================================================================================
# Console + W&B + finish
# ======================================================================================
def _print_console(r: dict) -> None:
    print("\n========== STRICT-SUBMISSION IDENTITY CERTIFIER (PR #471) ==========", flush=True)
    print(f" {r['one_line_verdict']}", flush=True)
    print(" --- the three known-answer anchors ---", flush=True)
    print(f"  deployed       : identity={r['deployed_identity_rate']:.6f} n_flips={r['deployed_n_flips']} "
          f"@ {r['deployed_flip_prompts']} -> is_strict_1p0={r['deployed_is_strict_1p0']} "
          f"(REJECT correct={r['deployed_classifies_correctly_REJECT']})", flush=True)
    print(f"  m1_ar (161.70) : identity={r['m1_ar_identity_rate']} n_flips={r['m1_ar_n_flips']} "
          f"-> is_strict_1p0={r['m1_ar_is_strict_1p0']} (ACCEPT correct={r['m1_ar_classifies_correctly_ACCEPT']}) "
          f"[determinism_M1 corroborates={r['m1_determinism_corroborates']}]", flush=True)
    print(f"  blanket_strict : identity={r['blanket_strict_identity_rate']:.6f} "
          f"n_flips={r['blanket_strict_n_flips']} @ {r['blanket_strict_flip_prompts']} "
          f"-> is_strict_1p0={r['blanket_strict_is_strict_1p0']} "
          f"(all bitwise-tie={r['blanket_strict_all_residual_flips_bitwise_tie']})", flush=True)
    print(f"      operative: {r['blanket_strict_operative_note']}", flush=True)
    print(" --- the gate ---", flush=True)
    print(f"  certifier_validates_on_known_configs = {r['certifier_validates_on_known_configs']}  (PRIMARY)",
          flush=True)
    print(f"  certifier_ready_for_466              = {r['certifier_ready_for_466']}", flush=True)
    print(f"  certifier_self_test_passes           = {r['certifier_self_test_passes']} "
          f"({sum(r['self_test'].values())}/{r['self_test_n_checks']})", flush=True)
    print(f"  submission_set_n_positions           = {r['submission_set_n_positions']} "
          f"({r['submission_set_n_prompts_censused']}/{r['submission_set_total_prompts']} prompts; "
          f"covers_full_128={r['certifier_covers_full_128_prompts']})", flush=True)
    print(f"  one-line #466 call: {r['one_line_call_for_466']}", flush=True)
    print(f"  ppl={r['ppl']} <= {r['ppl_gate']} ({r['ppl_passes_gate']})", flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("===================================================================\n", flush=True)


def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="denken", name=a.wandb_name, group=a.wandb_group,
        notes="PR#471 strict-submission identity certifier: a VALIDATED identity=1.0 oracle. Hardens the #464 "
              "census harness into the submission's zero-flip gate (is_strict_1p0 IFF n_flips==0) and validates "
              "it on three known-answer configs: deployed (REJECT, 3 flips), m1_ar (ACCEPT, 0 flips), "
              "blanket-strict (1 residual tie-flip @ p90, 0.9989 literal / 1.0 operative). LOCAL A10G, "
              "measurement + analysis only.",
        config={
            "pr": 471, "n_prompts": a.n_prompts, "ctx_len": a.ctx_len, "n_verify": a.n_verify,
            "gpu_mem_util": a.gpu_mem_util, "analysis_only": True, "no_hf_job": True,
            "no_served_file_change": True, "official_tps": 0,
            "strict_contract": "token_identity_rate==1.0 vs M=1 AR, zero flips (#319)",
            "anchor/deployed_identity": DEPLOYED_IDENTITY_ANCHOR, "anchor/deployed_flips": DEPLOYED_FLIPS_ANCHOR,
            "anchor/blanket_strict_identity": BLANKET_STRICT_IDENTITY_ANCHOR,
            "anchor/blanket_strict_flips": BLANKET_STRICT_FLIPS_ANCHOR,
            "anchor/strict_floor_tps": STRICT_FLOOR_TPS, "anchor/ppl": PPL_ANCHOR,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    scalar_keys = (
        "deployed_identity_rate", "deployed_n_flips", "deployed_is_strict_1p0",
        "deployed_classifies_correctly_REJECT",
        "m1_ar_identity_rate", "m1_ar_n_flips", "m1_ar_is_strict_1p0", "m1_ar_classifies_correctly_ACCEPT",
        "m1_determinism_corroborates",
        "blanket_strict_identity_rate", "blanket_strict_n_flips", "blanket_strict_is_strict_1p0",
        "blanket_strict_all_residual_flips_bitwise_tie", "blanket_strict_rule_consistent",
        "blanket_strict_operative_identity_anchor",
        "certifier_validates_on_known_configs", "certifier_ready_for_466", "certifier_self_test_passes",
        "strict_gate_requires_zero_flips",
        "submission_set_n_positions", "submission_set_n_prompts_censused", "certifier_covers_full_128_prompts",
        "ppl", "ppl_gate", "ppl_passes_gate", "strict_floor_tps", "deployed_tps_context",
        "blanket_strict_tps_context", "self_test_n_checks",
        "one_line_verdict", "one_line_call_for_466", "blanket_strict_operative_note", "denominator_note",
        "analysis_only", "no_hf_job", "no_served_file_change", "official_tps",
    )
    for k in scalar_keys:
        run.summary[k] = report.get(k)
    run.summary["deployed_flip_prompts"] = report["deployed_flip_prompts"]
    run.summary["blanket_strict_flip_prompts"] = report["blanket_strict_flip_prompts"]
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    report["wandb_run_id"] = run.id
    print(f"[wandb] logged run {run.id}", flush=True)


def _finish(report: dict, a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not a.no_wandb:
        log_wandb(report, a)
    json.dump(report, open(REPORT_JSON, "w"), indent=2)
    _print_console(report)
    print(f"[done] results -> {REPORT_JSON}", flush=True)


# ======================================================================================
# Modes
# ======================================================================================
def validate(a: argparse.Namespace) -> None:
    """Drive heuristic + pinned census fresh on this pod; derive m1_ar; compose the validation verdict."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # run the RULE self-test first (0-GPU; gates certifier_ready_for_466)
    rule_ok, _ = synthetic_rule_self_test()
    print(f"[certify] synthetic RULE self-test passes={rule_ok}", flush=True)

    # deployed (heuristic) -- drives the census; also the source of the empirical M=1 determinism for m1_ar
    deployed_census = run_census_arm("heuristic", "0", a)
    deployed = certify_from_census("deployed", deployed_census, DEPLOYED_TPS,
                                   CONFIG_REGISTRY["deployed"]["note"])
    deployed["_peak_gpu_gb"] = deployed_census.get("peak_gpu_gb")

    # blanket_strict (pinned) -- the genuine new measurement on this pod
    blanket_census = run_census_arm("pinned", "1", a)
    blanket = certify_from_census("blanket_strict", blanket_census, BLANKET_STRICT_TPS,
                                  CONFIG_REGISTRY["blanket_strict"]["note"])
    blanket["_peak_gpu_gb"] = blanket_census.get("peak_gpu_gb")

    # m1_ar -- by construction, corroborated by the measured determinism_M1_vs_M1 from the heuristic census
    m1 = certify_m1_ar_by_construction(deployed_census)

    _finish(compose_validate(deployed, blanket, m1, a, rule_ok), a)


def certify_one(a: argparse.Namespace) -> None:
    """Single-config certification (the call land #469's gate invokes once #466 confirms its config)."""
    if a.config in CONFIG_REGISTRY:
        v = certify_strict_submission(a.config, a)
    else:
        # arbitrary arm/vbi (the generalized #466 path): arm name + VLLM_BATCH_INVARIANT supplied explicitly
        census = run_census_arm(a.arm, a.vbi, a)
        v = certify_from_census(a.config, census, 0.0, f"arm={a.arm} VLLM_BATCH_INVARIANT={a.vbi}")
        v["_peak_gpu_gb"] = census.get("peak_gpu_gb")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(v, open(OUT_DIR / f"certify_{a.config}_result.json", "w"), indent=2)
    print("\n========== SINGLE-CONFIG CERTIFICATION ==========", flush=True)
    print(json.dumps({k: v[k] for k in ("config", "identity_rate", "n_flips", "flip_prompts",
                                         "is_strict_1p0", "n_positions", "n_prompts_censused",
                                         "all_residual_flips_bitwise_tie", "operative_note")}, indent=2), flush=True)
    print(f"\nGATE VERDICT is_strict_1p0 = {v['is_strict_1p0']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="drive heuristic+pinned census + derive m1_ar + compose the validation verdict")
    ap.add_argument("--certify", dest="config", default=None,
                    help="certify ONE config (registry name, or an arbitrary label with --arm/--vbi)")
    ap.add_argument("--arm", default="pinned", help="census arm for an arbitrary --certify config")
    ap.add_argument("--vbi", default="1", help="VLLM_BATCH_INVARIANT for an arbitrary --certify config")
    ap.add_argument("--self-test", dest="self_test", action="store_true", help="0-GPU synthetic RULE self-test")
    ap.add_argument("--smoke", action="store_true", help="tiny census (few prompts) to validate plumbing")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)   # full 128-prompt PPL-gt set
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=8)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--wandb_group", dest="wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", dest="wandb_name", default="denken/strict-submission-identity-certifier")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.n_prompts = min(a.n_prompts, 20)

    if a.self_test:
        self_test_mode()
    elif a.config is not None:
        certify_one(a)
    elif a.validate:
        validate(a)
    else:
        ap.error("one of --validate / --certify <config> / --self-test is required")


if __name__ == "__main__":
    main()
