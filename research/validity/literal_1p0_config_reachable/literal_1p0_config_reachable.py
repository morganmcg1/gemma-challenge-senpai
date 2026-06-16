#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #476 (denken) -- Config-reachable literal-1.0? Close the p90 bitwise-tie WITHOUT a served-kernel rebuild.

THE QUESTION (GitHub PR #476)
-----------------------------
The strict submission firing via #474 is OPERATIVE-1.0: census 0.998875, ONE residual flip @ prompt 90, 0
semantic (stark #429 verify-arbiter fixed-point). The PR hypothesises that lone flip is a PURE argmax TIE-BREAK
("emitted 102643 vs M=1-AR ref 22355, BOTH with bit-identical logits -- the verify-arbiter and the M=1 AR
reference simply break the tie toward different indices"). IF so, a config-reachable tie-break alignment (env
flag / sampler determinism, NO served-kernel rebuild) would drive n_flips -> 0 and upgrade the flagship from
OPERATIVE-1.0 to LITERAL byte-exact 1.0 at the same ~457.5 TPS.

THE DECISIVE FINDING (refutes the premise, surfaced LOUD)
--------------------------------------------------------
The p90 flip is NOT a pure tie-break. It is a reduction-order VALUE artifact. The #471 certifier census
(this card re-reads it FRESH) reports at (prompt 90, pos 227):
    m8_gap       = 0.125   (M=8 verify ranks 102643 a full bf16 ULP ABOVE 22355 -- a strict VALUE preference)
    m1_self_gap  = 0.0      (M=1 AR has 22355 and 102643 bit-TIED, and breaks the tie to 22355 = lowest index)
    m8_top1=102643  m8_top2=22355=m1_tok_id   m1_in_m8_top2=True   m1_margin_in_m8=0.125
So the M=1 side is a bitwise tie (gap 0.0) but the M=8 side is NOT (gap 0.125): the M=8 reduction order nudges
102643's logit one bf16 ULP above 22355. The advisor's premise "BOTH with bit-identical logits" is FALSE for
the M=8 path. A lowest-index tie-break only re-orders among BIT-EQUAL logits; it CANNOT override a 0.125 value
preference. There is no tie at p90 for a sampler flag to re-break.

This is the SAME reduction-order class my #431 (uza2t8aq) pinned generally: every reduction-order divergence in
this model sits at gap = e* = 0.125 with the reference token as the M=8 top-2 -- `all_divergences_are_bitwise
_ties=False`. p90 is one instance.

THE LEVER TAXONOMY (instruction 3) -- which levers are config-reachable, which close p90
-----------------------------------------------------------------------------------------
  L1  VBI=1 + sampler/argmax determinism  CONFIG-REACHABLE (env)   does NOT close p90: no tie to break (m8_gap
                                                                   =0.125 value gap). M=1 AR already lowest-index
                                                                   (22355=min); the served verify argmax also
                                                                   resolves true ties to lowest index. No-op.
  L2  #427/#433 pinned-K split-K verify    NOT config-reachable     FA2 num_splits>1 raises NotImplementedError on
                                                                   sm_86 (#431 uza2t8aq verified fresh); #427/#408
                                                                   banked FEASIBILITY only, no built kernel ->
                                                                   needs a human-gated REBUILD. And even built it
                                                                   is SELF-REFERENTIAL (pinned-K M=1 != canonical
                                                                   M=1 AR, #431) + realized -5.82 TPS (#433
                                                                   0pg4bz25, closed-lever ledger supply-pinned-k).
  L3  #375 varlen-combine reassociation    NOT config-reachable     "the pin is a kernel-rebuild, NOT a served
                                                                   knob" (#375, wirbel). This IS the path that
                                                                   would make the M=8 varlen-combine M-invariant
                                                                   (bit-match M=1 -> close p90) -- but it is a
                                                                   SERVED-KERNEL REBUILD == requires_kernel_rebuild.
=> 0 config-reachable levers close p90. The only config-reachable LITERAL-1.0 is M=1 AR (161.70, by construction,
   no M=8 verify divergence): cost = 457.55 - 161.70 = 295.85 TPS (-64.6%). Literal-1.0 AT ~457.5 needs L3's
   served-kernel rebuild (gated). So: literal_1p0_config_reachable=False, requires_kernel_rebuild=True.

This does NOT gate the #474 submission: operative-1.0 ships now (the 1 flip is a quality-neutral bitwise tie,
stark #429 / denken #464). This card BOUNDS the cost of upgrading operative-1.0 -> literal-1.0.

SCOPE: LOCAL A10G (sm_86), measurement + analysis only. NO served-file change, NO kernel rebuild, NO HF job,
NO submission, NO --launch. Drives the MERGED #471 certifier (which drives the MERGED #412 census) as the
measurement engine; this card only re-reads its JSON and applies the literal-vs-config verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---- anchors (CITE; do NOT re-derive) -------------------------------------------------
PPL_ANCHOR = 2.3772                 # deployed/strict PPL teacher-forced (#52 / #429)
PPL_GATE = 2.42
SPEED_ANCHOR_TPS = 457.55           # operative-1.0 served speed (stark #472 wfggu51k)
M1_FLOOR_TPS = 161.70               # M=1 AR strict floor, literal-1.0 by construction (lawine #438)
EPS_STAR = 0.125                    # bf16 near-tie floor (one ULP at decode-logit scale)
BAND_TOL = 1e-9                     # bitwise-tie tolerance (gap <= BAND_TOL => bit-identical top-2)
FLIP_PROMPT = 90
FLIP_POS = 227

CERT_DIR = Path("research/validity/strict_submission_identity_certifier")
CERTIFIER = CERT_DIR / "strict_submission_identity_certifier.py"
CENSUS_PINNED_JSON = CERT_DIR / "census_pinned_result.json"          # written by the certifier's pinned arm
CERTIFY_JSON = CERT_DIR / "certify_blanket_strict_result.json"       # written by --certify blanket_strict
OUT_DIR = Path("research/validity/literal_1p0_config_reachable")
REPORT_JSON = OUT_DIR / "literal_1p0_config_reachable_results.json"

# ---- the lever taxonomy (instruction 3): config-reachable? closes p90? citations ------
LEVERS = [
    {
        "id": "L1_vbi1_sampler_tiebreak",
        "name": "VLLM_BATCH_INVARIANT=1 + sampler/argmax determinism setting",
        "config_reachable": True,
        "closes_p90": False,
        "is_kernel_rebuild": False,
        "mechanism": "lowest-index argmax tie-break among BIT-EQUAL logits",
        "why_fails": ("p90 is NOT a tie in the M=8 path (m8_gap=0.125; 102643 wins by one bf16 ULP). A tie-break "
                      "flag only re-orders bit-EQUAL logits; it cannot override a value preference. The M=1 AR "
                      "reference ALREADY breaks true ties to the lowest index (22355=min, m1_argmax_matches_token"
                      "=True), and the served verify argmax does too -- there is no tie at p90 to re-break."),
        "citation": "this census m8_gap=0.125; #405 (reduction-order flips at margin e*=0.125, M1 token is M8 top-2)",
    },
    {
        "id": "L2_pinned_k",
        "name": "#427/#433 pinned-K self-referential split-K verify GEMM",
        "config_reachable": False,
        "closes_p90": False,
        "is_kernel_rebuild": True,
        "mechanism": "pin the verify split-K layout to a canonical self-referential reduction",
        "why_fails": ("FA2 num_splits>1 raises NotImplementedError on sm_86 (#431 verified fresh); #427/#408 "
                      "banked feasibility only -- no built kernel -> needs a human-gated rebuild. Even built, "
                      "pinned-K M=1 != canonical M=1 AR (self_referential_only, #431) so it would NOT align M=8 "
                      "to the canonical reference the strict contract names; and it realized -5.82 TPS (#433)."),
        "citation": "closed-lever ledger supply-pinned-k (#433 0pg4bz25); #431 uza2t8aq",
    },
    {
        "id": "L3_varlen_combine",
        "name": "#375 varlen-combine reassociation (M-invariant verify attention)",
        "config_reachable": False,
        "closes_p90": True,           # it WOULD close p90 -- but only via a served-kernel rebuild
        "is_kernel_rebuild": True,
        "mechanism": ("determinize the FA2 varlen-combine so the M=8 verify output is bit-identical to M=1 "
                      "(removes the residual 0.125 value gap)"),
        "why_fails": ("'the pin is a kernel-rebuild, NOT a served knob' (#375). This is the requires_kernel_rebuild "
                      "path: it closes p90 but only by editing the served FA2 kernel, which is gated and explicitly "
                      "forbidden by this analysis-only card."),
        "citation": "#375 (wirbel) 'kernel-rebuild not a served knob'",
    },
]


# ======================================================================================
def run_certifier(n_prompts: int) -> None:
    """Drive the MERGED #471 certifier on the VBI=1 GO config (regenerates the census + certify JSONs)."""
    cmd = [sys.executable, str(CERTIFIER.resolve()),
           "--certify", "blanket_strict", "--n-prompts", str(n_prompts), "--no-wandb"]
    print(f"[literal-1p0] driving certifier: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}).returncode
    if rc != 0:
        raise RuntimeError(f"certifier --certify blanket_strict failed (rc={rc})")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the certifier first (--run-certify) "
                                f"or wait for the background census to finish")
    return json.load(open(path))


def find_flip(census: dict, prompt_idx: int) -> dict | None:
    for f in census.get("flip_details", []):
        if int(f.get("prompt_idx", -1)) == prompt_idx:
            return f
    flips = census.get("flip_details", [])
    return flips[0] if flips else None


def classify_flip(rec: dict | None) -> dict:
    """The literal-vs-config discriminator. A flip is a config-reachable PURE TIE-BREAK iff the M=8 path itself
    has the two tokens BIT-EQUAL (m8_gap <= BAND_TOL): only then can a lowest-index tie-break flip the pick.
    A non-zero m8_gap is a reduction-order VALUE artifact -- unreachable by any sampler/argmax config."""
    if rec is None:
        return {"present": False}
    m8_gap = float(rec.get("m8_gap"))
    m1_self_gap = rec.get("m1_self_gap")
    m1_self_gap = float(m1_self_gap) if m1_self_gap is not None else None
    m8_top1 = int(rec.get("m8_top1_id"))
    m8_top2 = int(rec.get("m8_top2_id"))
    m1_tok = int(rec.get("m1_tok_id"))
    m8_is_tie = bool(m8_gap <= BAND_TOL)
    m1_is_tie = bool(m1_self_gap is not None and m1_self_gap <= BAND_TOL)
    return {
        "present": True,
        "prompt_idx": int(rec.get("prompt_idx")), "pos": int(rec.get("pos")),
        "m8_top1_id": m8_top1, "m8_top2_id": m8_top2, "m1_tok_id": m1_tok,
        "m8_gap": m8_gap, "m1_self_gap": m1_self_gap, "m1_margin_in_m8": rec.get("m1_margin_in_m8"),
        "m1_in_m8_top2": bool(rec.get("m1_in_m8_top2")),
        # the literal-vs-config classification:
        "m8_path_has_value_preference": bool(not m8_is_tie),
        "is_pure_tiebreak": bool(m8_is_tie and m1_is_tie),          # config-reachable iff this is True
        "is_reduction_order_value_artifact": bool(not m8_is_tie),
        "m1_side_is_bitwise_tie": m1_is_tie,
        # tie-break rule check: M=1 AR picks the lowest index of the disputed pair
        "m1_tiebreak_is_lowest_index": bool(m1_tok == min(m8_top1, m8_top2)),
        "disputed_pair_min_id": min(m8_top1, m8_top2),
    }


def compose(census: dict, certify: dict | None, flip: dict, n_prompts_arg: int) -> dict:
    p90_is_pure_tiebreak = bool(flip.get("present") and flip.get("is_pure_tiebreak"))
    config_levers = [l for l in LEVERS if l["config_reachable"]]
    config_levers_that_close = [l for l in config_levers if l["closes_p90"]]
    closing_levers = [l for l in LEVERS if l["closes_p90"]]
    # the only lever that closes p90 is L3 (varlen-combine) and it is a kernel rebuild:
    requires_kernel_rebuild = bool(closing_levers and all(l["is_kernel_rebuild"] for l in closing_levers))
    literal_1p0_config_reachable = bool(len(config_levers_that_close) > 0)

    n_flips = census.get("flip_details", [])
    n_flips = len(n_flips)
    identity = census.get("decodewidth_e2e_token_identity_rate")
    # aligned_tiebreak_n_flips: the n_flips a config-reachable tie-break alignment achieves. Since L1 is a no-op
    # on a value gap, it stays at the measured n_flips (1).
    aligned_tiebreak_n_flips = n_flips

    # cost of literal-1.0 by config-reachable means: the ONLY config-reachable literal-1.0 is the M=1 AR floor.
    literal_config_tps_cost = round(SPEED_ANCHOR_TPS - M1_FLOOR_TPS, 2)   # 295.85 (drop to the M=1 floor)
    is_strict_1p0 = bool(certify.get("is_strict_1p0")) if certify else bool(n_flips == 0)

    census_sound = bool(census.get("determinism_M1_vs_M1") == 1.0 and census.get("determinism_M8_vs_M8") == 1.0
                        and census.get("within_batch_copy0_vs_copy1") == 1.0)

    report = {
        "pr": 476, "agent": "denken",
        "leg": "Config-reachable literal-1.0: close the p90 bitwise-tie WITHOUT a served-kernel rebuild "
               "(LOCAL A10G, measurement + analysis only)",
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "no_kernel_rebuild": True,
        "official_tps": 0,

        # ===== PRIMARY deliverables (PR-named) =====
        "literal_1p0_config_reachable": literal_1p0_config_reachable,   # PRIMARY bool
        "p90_is_pure_tiebreak": p90_is_pure_tiebreak,                   # bool
        "aligned_tiebreak_n_flips": aligned_tiebreak_n_flips,
        "literal_config_tps_cost": literal_config_tps_cost,
        "requires_kernel_rebuild": requires_kernel_rebuild,             # bool
        "ppl": PPL_ANCHOR,

        # ===== the p90 flip characterization (instruction 1) =====
        "p90_flip": flip,
        "p90_m8_gap": flip.get("m8_gap"),
        "p90_m1_self_gap": flip.get("m1_self_gap"),
        "p90_is_reduction_order_value_artifact": flip.get("is_reduction_order_value_artifact"),
        "p90_m8_path_has_value_preference": flip.get("m8_path_has_value_preference"),
        "p90_m1_side_is_bitwise_tie": flip.get("m1_side_is_bitwise_tie"),

        # ===== M=1 AR tie-break rule (instruction 2) =====
        "m1_ar_tiebreak_is_lowest_index": flip.get("m1_tiebreak_is_lowest_index"),
        "m1_ar_tiebreak_rule": "lowest_token_index_on_equal_logits (torch.argmax / vLLM greedy)",

        # ===== lever taxonomy (instruction 3) =====
        "levers": LEVERS,
        "n_config_reachable_levers": len(config_levers),
        "n_config_reachable_levers_that_close_p90": len(config_levers_that_close),
        "the_lever_that_closes_p90": (closing_levers[0]["id"] if closing_levers else None),
        "the_lever_that_closes_p90_is_kernel_rebuild": requires_kernel_rebuild,

        # ===== certifier verdict on the GO config (instruction 4) =====
        "certify_is_strict_1p0": is_strict_1p0,
        "certify_n_flips": n_flips,
        "certify_identity_rate": identity,
        "census_is_sound": census_sound,
        "census_determinism_M8_vs_M8": census.get("determinism_M8_vs_M8"),
        "census_n_prompts": census.get("n_prompts"),
        "census_total_positions": census.get("total_positions"),

        # ===== speed/PPL context =====
        "speed_anchor_tps": SPEED_ANCHOR_TPS, "m1_floor_tps": M1_FLOOR_TPS,
        "operative_1p0_holds": True,            # the 1 flip is a quality-neutral bitwise tie (stark #429 / #464)
        "operative_1p0_ships_via_474": True,    # this card does NOT gate the #474 submission
        "ppl_gate": PPL_GATE, "ppl_passes_gate": bool(PPL_ANCHOR <= PPL_GATE),
        "n_prompts_arg": n_prompts_arg,
    }

    checks, n = build_self_test(report)
    report["self_test"] = checks
    report["self_test_n_checks"] = n
    report["self_test_passes"] = bool(all(checks.values()) and n >= 18)

    report["one_line_verdict"] = (
        f"literal_1p0_config_reachable={literal_1p0_config_reachable}: p90 is a reduction-order VALUE artifact "
        f"(m8_gap={flip.get('m8_gap')} != 0; M=8 ranks 102643 one bf16 ULP above 22355) NOT a pure tie-break "
        f"(p90_is_pure_tiebreak={p90_is_pure_tiebreak}); M=1 AR is bit-tied (m1_self_gap={flip.get('m1_self_gap')}) "
        f"and breaks to lowest index 22355. 0/{len(config_levers)} config levers close it; the only closer is "
        f"#375 varlen-combine = a served-kernel rebuild (requires_kernel_rebuild={requires_kernel_rebuild}). "
        f"Config-reachable literal-1.0 exists ONLY at the M=1 floor 161.70 (cost {literal_config_tps_cost} TPS). "
        f"Operative-1.0 stands at {SPEED_ANCHOR_TPS}; #474 ships.")
    return report


# ======================================================================================
def build_self_test(r: dict) -> tuple[dict, int]:
    flip = r["p90_flip"]
    c: dict = {}
    # the flip exists and is at p90/pos227
    c["p90_flip_present"] = bool(flip.get("present"))
    c["p90_at_prompt_90"] = bool(flip.get("prompt_idx") == FLIP_PROMPT)
    c["p90_at_pos_227"] = bool(flip.get("pos") == FLIP_POS)
    c["p90_pair_is_102643_22355"] = bool(
        {flip.get("m8_top1_id"), flip.get("m8_top2_id")} == {102643, 22355})
    # the decisive classification: NOT a pure tie-break, IS a value artifact
    c["m8_gap_is_value_gap"] = bool(flip.get("m8_gap", 0.0) > BAND_TOL)
    c["m8_gap_equals_eps_star"] = bool(abs(float(flip.get("m8_gap")) - EPS_STAR) <= 1e-6)
    c["p90_not_pure_tiebreak"] = bool(r["p90_is_pure_tiebreak"] is False)
    c["p90_is_value_artifact"] = bool(r["p90_is_reduction_order_value_artifact"] is True)
    c["m1_side_is_bitwise_tie"] = bool(flip.get("m1_self_gap") == 0.0)
    # M=1 AR tie-break = lowest index, and 22355 = min of the pair
    c["m1_tiebreak_lowest_index"] = bool(r["m1_ar_tiebreak_is_lowest_index"] is True)
    c["m1_tok_is_min_of_pair"] = bool(flip.get("m1_tok_id") == min(flip.get("m8_top1_id"), flip.get("m8_top2_id")))
    # the verdict wiring
    c["literal_not_config_reachable"] = bool(r["literal_1p0_config_reachable"] is False)
    c["zero_config_levers_close_p90"] = bool(r["n_config_reachable_levers_that_close_p90"] == 0)
    c["requires_kernel_rebuild_true"] = bool(r["requires_kernel_rebuild"] is True)
    c["closer_is_varlen_combine_kernel"] = bool(r["the_lever_that_closes_p90"] == "L3_varlen_combine")
    c["aligned_tiebreak_stays_one_flip"] = bool(r["aligned_tiebreak_n_flips"] == r["certify_n_flips"])
    c["literal_config_cost_is_floor_drop"] = bool(
        abs(r["literal_config_tps_cost"] - (SPEED_ANCHOR_TPS - M1_FLOOR_TPS)) <= 1e-6)
    # GO-config certify is consistent (n_flips>0 <=> not strict-1.0)
    c["certify_rule_consistent"] = bool(r["certify_is_strict_1p0"] == (r["certify_n_flips"] == 0))
    c["census_sound"] = bool(r["census_is_sound"] is True)
    # operative-1.0 ships, this card does not gate it; ppl ok; scope flags
    c["operative_1p0_not_gated"] = bool(r["operative_1p0_ships_via_474"] is True)
    c["ppl_passes_gate"] = bool(r["ppl_passes_gate"] is True)
    c["scope_analysis_only"] = bool(r["analysis_only"] and r["no_served_file_change"]
                                    and r["no_kernel_rebuild"] and r["official_tps"] == 0)
    # at least one lever is the genuine closer but it is a kernel rebuild (so the answer is NO)
    c["a_closer_exists_but_is_kernel"] = bool(
        any(l["closes_p90"] for l in r["levers"]) and r["requires_kernel_rebuild"] is True)
    return c, len(c)


def synthetic_self_test() -> tuple[bool, dict]:
    """0-GPU: the classifier is NOT a rubber stamp. A pure tie-break (m8_gap=0) WOULD be config-reachable;
    a value gap (m8_gap=0.125, our p90) is NOT. A confident flip (m8_gap large) is also NOT."""
    pure = classify_flip({"prompt_idx": 1, "pos": 227, "m8_gap": 0.0, "m1_self_gap": 0.0,
                          "m8_top1_id": 102643, "m8_top2_id": 22355, "m1_tok_id": 22355,
                          "m1_in_m8_top2": True, "m1_margin_in_m8": 0.0})
    p90 = classify_flip({"prompt_idx": 90, "pos": 227, "m8_gap": 0.125, "m1_self_gap": 0.0,
                        "m8_top1_id": 102643, "m8_top2_id": 22355, "m1_tok_id": 22355,
                        "m1_in_m8_top2": True, "m1_margin_in_m8": 0.125})
    confident = classify_flip({"prompt_idx": 2, "pos": 227, "m8_gap": 2.0, "m1_self_gap": 1.5,
                              "m8_top1_id": 5, "m8_top2_id": 9, "m1_tok_id": 9,
                              "m1_in_m8_top2": True, "m1_margin_in_m8": 2.0})
    checks = {
        "pure_tiebreak_is_config_reachable": pure["is_pure_tiebreak"] is True
                                             and pure["is_reduction_order_value_artifact"] is False,
        "p90_value_gap_not_pure_tiebreak": p90["is_pure_tiebreak"] is False
                                           and p90["is_reduction_order_value_artifact"] is True,
        "p90_m1_lowest_index": p90["m1_tiebreak_is_lowest_index"] is True,
        "confident_flip_not_pure_tiebreak": confident["is_pure_tiebreak"] is False,
        "classifier_discriminates": (pure["is_pure_tiebreak"] is True)
                                    and (p90["is_pure_tiebreak"] is False),
    }
    return bool(all(checks.values())), checks


# ======================================================================================
def log_wandb(report: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="denken", name=a.wandb_name, group=a.wandb_group,
        notes="PR#476 config-reachable literal-1.0: the p90 residual flip is a reduction-order VALUE artifact "
              "(m8_gap=0.125, NOT a pure tie-break), so no env/sampler config closes it; the only closer is the "
              "#375 varlen-combine served-kernel rebuild. Config-reachable literal-1.0 exists only at the M=1 "
              "floor (161.70). Operative-1.0 (457.55) stands; #474 ships. LOCAL A10G, analysis only.",
        config={
            "pr": 476, "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
            "no_kernel_rebuild": True, "official_tps": 0,
            "anchor/speed_anchor_tps": SPEED_ANCHOR_TPS, "anchor/m1_floor_tps": M1_FLOOR_TPS,
            "anchor/ppl": PPL_ANCHOR, "anchor/eps_star": EPS_STAR,
            "go_config": "fa2sw_precache_kenyan + VLLM_BATCH_INVARIANT=1 (pinned arm)",
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    scalar_keys = (
        "literal_1p0_config_reachable", "p90_is_pure_tiebreak", "aligned_tiebreak_n_flips",
        "literal_config_tps_cost", "requires_kernel_rebuild", "ppl",
        "p90_m8_gap", "p90_m1_self_gap", "p90_is_reduction_order_value_artifact",
        "p90_m8_path_has_value_preference", "p90_m1_side_is_bitwise_tie",
        "m1_ar_tiebreak_is_lowest_index",
        "n_config_reachable_levers", "n_config_reachable_levers_that_close_p90",
        "the_lever_that_closes_p90", "the_lever_that_closes_p90_is_kernel_rebuild",
        "certify_is_strict_1p0", "certify_n_flips", "certify_identity_rate", "census_is_sound",
        "census_determinism_M8_vs_M8", "census_n_prompts", "census_total_positions",
        "speed_anchor_tps", "m1_floor_tps", "operative_1p0_holds", "operative_1p0_ships_via_474",
        "ppl_gate", "ppl_passes_gate", "self_test_passes", "self_test_n_checks", "one_line_verdict",
    )
    for k in scalar_keys:
        run.summary[k] = report.get(k)
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    run.summary["m1_ar_tiebreak_rule"] = report["m1_ar_tiebreak_rule"]
    finish_wandb(run)
    report["wandb_run_id"] = run.id
    print(f"[wandb] logged run {run.id}", flush=True)


def _print_console(r: dict) -> None:
    print("\n========== CONFIG-REACHABLE LITERAL-1.0 (PR #476) ==========", flush=True)
    print(f" {r['one_line_verdict']}", flush=True)
    f = r["p90_flip"]
    print(" --- p90 flip (instruction 1) ---", flush=True)
    print(f"  (prompt {f.get('prompt_idx')}, pos {f.get('pos')}): m8_top1={f.get('m8_top1_id')} "
          f"m8_top2={f.get('m8_top2_id')} m1_tok={f.get('m1_tok_id')}", flush=True)
    print(f"  m8_gap={f.get('m8_gap')} (M=8 VALUE preference, NOT a tie)  |  "
          f"m1_self_gap={f.get('m1_self_gap')} (M=1 bitwise TIE -> lowest index)", flush=True)
    print(f"  is_pure_tiebreak={r['p90_is_pure_tiebreak']}  "
          f"is_value_artifact={r['p90_is_reduction_order_value_artifact']}", flush=True)
    print(" --- levers (instruction 3) ---", flush=True)
    for l in r["levers"]:
        print(f"  {l['id']:22s} config_reachable={l['config_reachable']!s:5s} closes_p90={l['closes_p90']!s:5s} "
              f"kernel_rebuild={l['is_kernel_rebuild']}", flush=True)
    print(" --- verdict ---", flush=True)
    print(f"  literal_1p0_config_reachable = {r['literal_1p0_config_reachable']}  (PRIMARY)", flush=True)
    print(f"  requires_kernel_rebuild      = {r['requires_kernel_rebuild']}", flush=True)
    print(f"  aligned_tiebreak_n_flips     = {r['aligned_tiebreak_n_flips']}", flush=True)
    print(f"  literal_config_tps_cost      = {r['literal_config_tps_cost']} TPS (drop {SPEED_ANCHOR_TPS}->{M1_FLOOR_TPS})",
          flush=True)
    print(f"  certify is_strict_1p0={r['certify_is_strict_1p0']} n_flips={r['certify_n_flips']} "
          f"identity={r['certify_identity_rate']}", flush=True)
    print(f"  ppl={r['ppl']} <= {r['ppl_gate']} ({r['ppl_passes_gate']})", flush=True)
    print(f"  self_test_passes={r['self_test_passes']} ({sum(r['self_test'].values())}/{r['self_test_n_checks']})",
          flush=True)
    fails = [k for k, v in r["self_test"].items() if not v]
    if fails:
        print(f"   self-test FAILS: {fails}", flush=True)
    print("============================================================\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-certify", action="store_true",
                    help="drive the #471 certifier on VBI=1 first (GPU); else re-read existing JSONs")
    ap.add_argument("--self-test", dest="self_test", action="store_true", help="0-GPU synthetic classifier self-test")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--wandb_group", dest="wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", dest="wandb_name", default="denken/literal-1p0-config-reachable")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        ok, checks = synthetic_self_test()
        for k, v in checks.items():
            print(f"   {'PASS' if v else 'FAIL'}  {k}", flush=True)
        print(f"[self-test] synthetic classifier passes={ok}", flush=True)
        sys.exit(0 if ok else 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rule_ok, _ = synthetic_self_test()
    print(f"[literal-1p0] synthetic classifier self-test passes={rule_ok}", flush=True)
    if not rule_ok:
        raise RuntimeError("synthetic classifier self-test FAILED -- aborting")

    if a.run_certify:
        run_certifier(a.n_prompts)

    census = load_json(CENSUS_PINNED_JSON)
    certify = load_json(CERTIFY_JSON) if CERTIFY_JSON.exists() else None
    flip = classify_flip(find_flip(census, FLIP_PROMPT))
    report = compose(census, certify, flip, a.n_prompts)

    if not a.no_wandb:
        log_wandb(report, a)
    json.dump(report, open(REPORT_JSON, "w"), indent=2)
    _print_console(report)
    print(f"[done] results -> {REPORT_JSON}", flush=True)


if __name__ == "__main__":
    main()
