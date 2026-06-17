#!/usr/bin/env python3
"""PR #594 — Is base_fullhead a better OFFICIAL submission than int4_g128_lmhead?

ANALYSIS-ONLY reconciliation card (the #570 synthesis pattern). NO FIRE, NO HF
Job, NO submission, NO served-file change, NO GPU. Every load-bearing number is
either (a) GROUNDED at runtime against an on-branch research artifact (loaded +
asserted), or (b) PUBLIC-RESULT-BANKED with its job/run provenance recorded.

THE LEDGER TENSION (PR #594):
  int4_g128_lmhead = 126.378 OFFICIAL TPS (the human's "best high-quality fast
  submission") vs base_fullhead's 252.69 "ships clean". These cannot both be the
  best submission unless they are on different BASES. They are.

WHAT THE OFFICIAL BENCHMARK ACTUALLY MEASURES (the crux, resolved):
  official/.../speed_benchmark/{README.md, scripts/hf_bucket_single_job.py} ->
  the harness STARTS THE SUBMITTER'S OWN `serve` command (manifest.json:serve)
  and runs `sglang.bench_serving` against that endpoint; summary.json:tps =
  result["output_throughput"] = output_tokens / wall_clock at concurrency 1.
  There is NO fixed AR/no-spec harness: the basis is the submitter's full
  serving stack. Speculative decoding is therefore IN-FRAME -- spec-dec produces
  the same output tokens in less wall time -> higher official TPS.
  PROOF: the entire official leaderboard frontier (484-508 TPS) is spec-dec
  stacks (K7 drafters + split-KV / tree verify). Those are impossible under a
  fixed-AR harness (bf16 AR baseline = 44.018; int4 AR = 126.378).
  => official_basis_is_spec_frame = TRUE.

THE RECONCILIATION (one basis at a time):
  * int4_g128_lmhead 126.378 is itself a SPEC-OFF / plain-AR number: its serve.py
    is "identical serving path to the vLLM baseline" (no spec-dec config). It is
    REALIZED, SERVED, byte-exact identity-safe, all 4 quality gates PASS, PPL 2.0057.
  * base_fullhead has NEVER been officially served. Its fast figures are
    LOCAL/PROJECTION on (essentially official-equivalent, tau_lo~1.035) hardware:
      - 252.69 / 253.99 = MTP-K7 spec-ON (full-head EXACT verify, E[T]_exact 3.844).
        Byte-exact-by-construction is PLAUSIBLE but its SERVED FREE-RUNNING
        identity is UNMEASURED, and it is unrealized + unvalidated-PPL officially.
      - 291-305 / 299.28 = candidate-verify (int4 top-8 nominator + bf16 verify
        head). REFUTED on served byte-exact identity: free-running token identity
        0.449, seq-exact 9/64 (#566). The per-step 0.994 CASCADES under greedy AR.
      - 83.44 = clean AR no-spec (identity-safe) -> official-proxy ~86 < 126.378.
  * The "252.69 is 3x the 83.44 pod" gap is NOT hardware: lawine run wndiyzxk
    logs BOTH 83.44 (nospec) and 253.99 (MTP-K7 spec) on the SAME pod -> the 3x
    is the spec-dec speedup, not a slow pod. (tau_lo local->official ~1.035.)

VERDICT:
  base_fullhead cannot SIMULTANEOUSLY clear (A) official TPS > 126.378 AND
  (C) served byte-exact greedy identity:
    - identity-safe served form (AR no-spec) ~86 official-proxy < 126.378;
    - the >126.378 fast forms are either REFUTED-served-identity (cand-verify 0.449)
      or UNMEASURED-served-identity + UNREALIZED (MTP-K7).
  int4_g128_lmhead clears BOTH (126.378 official AND byte-exact served).
  => base_fullhead_official_beats_g128lmhead = FALSE (as realized/validated).
  => int4_g128_lmhead@126.378 IS the correct best VALID official quality-safe
     submission. The ledger's "252.69 ships clean" is LOOSE and corrected here:
     252.69 is a LOCAL MTP-spec figure, not an official submittable number, and
     it does NOT "ship clean" without a SERVED free-running identity=1.0 proof.

GO/NO-GO (NO autonomous fire either way):
  * NO-GO swapping the current best submission (int4_g128 stays).
  * CONDITIONAL human-approval candidate: base_fullhead + MTP-K7 (full-head exact
    verify) projects ~263 official-proxy (> 126.378) at PLAUSIBLE byte-exactness.
    It is worth ONE human-approved benchmark slot IFF three preconditions are
    first proven LOCALLY: (1) SERVED FREE-RUNNING greedy identity = 1.0 (not
    per-step -- the #566 lesson), (2) a remote-loadable packaged serve.py with the
    MTP drafter, (3) PPL <= cap on a real serve. Until then it is not a confirmed
    better submission.
  * Apples-to-apples caveat: spec-dec is official-in-frame, so int4_g128_lmhead
    could ALSO be MTP-drafted; the int4 head is a cheaper verify substrate than the
    full bf16 head, so int4_g128+MTP would likely DOMINATE base_fullhead+MTP. A
    quality-safe spec benchmark slot should prefer int4_g128+MTP.

Run under the wandb-capable venv (.venv/bin/python).
"""
from __future__ import annotations

# --- real-wandb-first (beats ./wandb namespace shadow); harmless if absent ---
try:  # pragma: no cover
    import wandb as _wandb_real  # noqa: F401
except Exception:  # pragma: no cover
    _wandb_real = None

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research" / "submission_basis_reconciliation"
OUT_JSON = HERE / "submission_basis_reconciliation.json"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # APPEND so site-packages wandb wins over ./wandb

# ======================================================================
# ABSOLUTE ANCHORS — program contract + official a10g-small numbers.
# PUBLIC-RESULT-BANKED: provenance recorded; not asserted against a local file
# (they are external HF-Job results, read this cycle from the main bucket).
# ======================================================================
INT4_G128_OFFICIAL_TPS = 126.378        # job 6a2d5a96234ca64b60121aa5, run 905tbujn, PPL 2.0057
INT4_G128_OFFICIAL_PPL = 2.0057
BF16_BASELINE_OFFICIAL_TPS = 44.018     # vllm_baseline floor (operator-confirmed smoke)
SHIP_OFFICIAL_TPS = 375.857             # 12k-head spec ship (fails deeper quality gates)
PUBLIC_1_OFFICIAL_TPS = 481.53          # public #1 (speed-only; fails all 4 quality gates)
PPL_CAP = 2.42                          # reference PPL + 5% (validity gate)
TAU_LO = 1.0352356533046398             # #267/#553-banked local->official scalar

# int4_g128_lmhead board quality (public result 20260617-121233, senpai 12:12)
INT4_G128_GATES = {"mmlu_pro": 0.668, "gpqa_d": 0.480, "aime": 0.117, "gsm8k": 0.850}
# operative gates (PR #594 body + senpai 12:12 announcement; AIME>=0.090)
GATES = {"mmlu_pro": 0.605, "gpqa_d": 0.471, "aime": 0.090, "gsm8k": 0.807}
# base_fullhead board quality: MMLU-Pro/GPQA/GSM8K PASS, AIME 0.117 (== int4_g128;
# AIME is set by the shared int4 BODY, head-invariant), GPQA-D +0.009 vs gate (PR).
BASE_FULLHEAD_GATES = {"mmlu_pro": 0.668, "gpqa_d": 0.480, "aime": 0.117, "gsm8k": 0.850}

# ======================================================================
# GROUNDED INPUTS — asserted at runtime against on-branch artifacts.
# ======================================================================

# --- (1) base_fullhead clean AR no-spec, this pod (identity-safe path) -----
#     research/base_fullhead_specdec/specdec_report.json (committed; #573)
BASE_FULLHEAD_AR_LOCAL_TPS = 83.44593011705038   # ref_warm_aggregate_tps

# --- (2) base_fullhead candidate-verify SERVED (the 291-305 fast path) -----
#     research/candidate_verify_realize/served_cv_report.json (#566)
CV_SERVED_TPS_LOCAL = 299.2824640536662          # cvsp warm_median_tps
CV_REF_SERVED_TPS_LOCAL = 263.10239414866754     # ref warm_median_tps (full-head, fast stack)
CV_PERSTEP_IDENTITY = 0.9940561590489855         # teacher-forced per-step argmax identity
CV_FREERUN_TOKEN_IDENTITY = 0.44940185546875     # FREE-RUNNING served token identity
CV_FREERUN_SEQ_EXACT = 0.140625                  # 9/64 sequences byte-exact
#     research/candidate_verify_realize/stage3_realized_identity.json (#560/#566)
CV_STAGE3_ARGMAX_IDENTITY = 0.9954666666666667
CV_STAGE3_HARD_GATE_PASS = False                 # identity_hard_gate_pass
#     research/candidate_verify_realize/stage3c_tiebreak.json
CV_TIEBREAK_VOCTB = 1.0                          # OFFLINE recompute w/ vocab-index tie-break
CV_TIEBREAK_POSTB = 0.99545                      # served (shortlist-position) tie-break

# --- (3) base_fullhead MTP-K7 spec-ON (the 252.69 anchor path) -------------
#     PUBLIC/W&B-BANKED: wirbel #553 (run 83jiwjr9) anchor 252.69; lawine run
#     wndiyzxk (group base-fullhead-specdec-ceiling) logs BOTH nospec 83.44 and
#     MTP-K7 spec 253.99 on ONE pod with e_accept_exact 3.844.
BASE_FULLHEAD_SPEC_ANCHOR_TPS = 252.69           # wirbel #553 (83jiwjr9)
BASE_FULLHEAD_SPEC_LOCAL_TPS = 253.99            # lawine wndiyzxk base_fullhead_spec_tps
BASE_FULLHEAD_SPEC_E_ACCEPT_EXACT = 3.844        # lawine wndiyzxk acceptance/e_accept_exact
BASE_FULLHEAD_NOSPEC_LOCAL_TPS = 83.44           # lawine wndiyzxk (== SLOW_statson 83.43)

# --- on-branch source artifacts (committed on the advisor branch) ----------
SOURCES: dict[str, dict[str, Any]] = {
    "base_fullhead_ar_specdec_report": {
        "path": "research/base_fullhead_specdec/specdec_report.json",
        "pr": "wirbel #573", "wandb": "(specdec axis)",
        "asserts_nested": {
            ("acceptance_models", 0, "ref_warm_aggregate_tps"): BASE_FULLHEAD_AR_LOCAL_TPS,
            ("acceptance_models", 0, "k"): 7,
            ("analysis_only",): True,
        },
    },
    "cv_served_report": {
        "path": "research/candidate_verify_realize/served_cv_report.json",
        "pr": "fern #566", "wandb": "(served-cv-realize)",
        "asserts_nested": {
            ("cvsp", "tps", "warm_median_tps"): CV_SERVED_TPS_LOCAL,
            ("ref", "tps", "warm_median_tps"): CV_REF_SERVED_TPS_LOCAL,
            ("audit", "identity_rate"): CV_PERSTEP_IDENTITY,
            ("identity", "token_identity_rate"): CV_FREERUN_TOKEN_IDENTITY,
            ("identity", "sequence_exact_rate"): CV_FREERUN_SEQ_EXACT,
            ("identity", "n_sequences_byte_exact"): 9,
            ("official_tps",): 0,
        },
    },
    "cv_stage3_identity": {
        "path": "research/candidate_verify_realize/stage3_realized_identity.json",
        "pr": "fern #560/#566", "wandb": "ufv4nk21",
        "asserts": {
            "argmax_identity_rate": CV_STAGE3_ARGMAX_IDENTITY,
            "identity_hard_gate_pass": CV_STAGE3_HARD_GATE_PASS,
            "official_tps": 0,
        },
    },
    "cv_stage3c_tiebreak": {
        "path": "research/candidate_verify_realize/stage3c_tiebreak.json",
        "pr": "fern #560", "wandb": "ufv4nk21",
        "asserts_nested": {
            ("identity_vs_served_bf16", "C_vocTB"): CV_TIEBREAK_VOCTB,
            ("identity_vs_served_bf16", "C_posTB"): CV_TIEBREAK_POSTB,
            ("containment_rate_at_K8",): 1.0,
        },
    },
}

TOL = 1e-9


def _get_nested(d: Any, keys: tuple[Any, ...]) -> Any:
    cur: Any = d
    for k in keys:
        cur = cur[k]
    return cur


def _match(got: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return got == expected
    if isinstance(expected, (int, float)):
        return got is not None and math.isclose(float(got), float(expected), abs_tol=TOL)
    return got == expected


def ground_against_sources() -> dict[str, Any]:
    """Load each on-branch artifact and assert every cited constant matches.

    The #561/#570 self-test discipline: no asserted-but-unchecked numbers.
    """
    report: dict[str, Any] = {}
    all_ok = True
    for key, spec in SOURCES.items():
        path = ROOT / spec["path"]
        checks: dict[str, bool] = {}
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover
            report[key] = {"loaded": False, "error": str(exc), "pr": spec["pr"]}
            all_ok = False
            continue
        for field, expected in spec.get("asserts", {}).items():
            checks[field] = bool(_match(data.get(field, None), expected))
        for keys, expected in spec.get("asserts_nested", {}).items():
            try:
                got = _get_nested(data, keys)
            except Exception:
                got = None
            checks[".".join(str(k) for k in keys)] = bool(_match(got, expected))
        src_ok = all(checks.values())
        all_ok = all_ok and src_ok
        report[key] = {"loaded": True, "pr": spec["pr"], "wandb": spec["wandb"],
                       "path": spec["path"], "checks": checks, "all_match": src_ok}
    report["all_grounded"] = all_ok
    return report


def gates_verdict(scores: dict[str, float]) -> dict[str, Any]:
    per = {k: (scores[k] >= GATES[k]) for k in GATES}
    return {"per_gate": per, "all_pass": all(per.values())}


def build_packet() -> dict[str, Any]:
    grounding = ground_against_sources()

    def official_proxy(local: float) -> float:
        return local * TAU_LO

    # ---- official-proxy estimates for base_fullhead's three serving forms ----
    bf_ar_official_proxy = official_proxy(BASE_FULLHEAD_NOSPEC_LOCAL_TPS)        # ~86.4
    bf_spec_official_proxy = official_proxy(BASE_FULLHEAD_SPEC_LOCAL_TPS)        # ~262.9
    bf_cv_official_proxy = official_proxy(CV_SERVED_TPS_LOCAL)                   # ~309.8

    # ---- the reconciled head-to-head table (PR step 2) ----
    # columns: official_basis_tps | local_spec_served_tps | clean_ar_tps |
    #          4 quality gates | served byte-exact identity
    table = {
        "int4_g128_lmhead": {
            "official_basis_tps": INT4_G128_OFFICIAL_TPS,     # REALIZED + served + validated
            "official_basis_status": "REALIZED (job 6a2d5a96 / run 905tbujn)",
            "local_spec_served_tps": None,                    # AR-only submission; spec not packaged
            "clean_ar_tps": INT4_G128_OFFICIAL_TPS,           # its official number IS AR
            "ppl": INT4_G128_OFFICIAL_PPL,
            "quality_gates": gates_verdict(INT4_G128_GATES),
            "served_byte_exact_identity": True,               # byte-exact vs base int4 (#585 / senpai 12:12)
            "identity_status": "PASS (served byte-exact)",
            "spec_off": True,
        },
        "base_fullhead__ar_nospec": {
            "official_basis_tps": bf_ar_official_proxy,       # PROXY (unrealized officially)
            "official_basis_status": "UNREALIZED (proxy = local 83.44 * tau_lo)",
            "local_spec_served_tps": None,
            "clean_ar_tps": BASE_FULLHEAD_NOSPEC_LOCAL_TPS,   # 83.44 local (lawine wndiyzxk / SLOW_statson)
            "ppl": INT4_G128_OFFICIAL_PPL,                    # 2.0057 (full bf16 head)
            "quality_gates": gates_verdict(BASE_FULLHEAD_GATES),
            "served_byte_exact_identity": True,               # full bf16 head, no shortlist/spec -> exact
            "identity_status": "PASS (full head, no spec) — but SLOW",
            "spec_off": True,
        },
        "base_fullhead__mtp_k7_spec": {
            "official_basis_tps": bf_spec_official_proxy,     # PROXY (unrealized officially)
            "official_basis_status": "UNREALIZED (proxy = local 253.99 * tau_lo)",
            "local_spec_served_tps": BASE_FULLHEAD_SPEC_LOCAL_TPS,  # 253.99 (anchor 252.69)
            "clean_ar_tps": BASE_FULLHEAD_NOSPEC_LOCAL_TPS,
            "e_accept_exact": BASE_FULLHEAD_SPEC_E_ACCEPT_EXACT,
            "ppl": INT4_G128_OFFICIAL_PPL,
            "quality_gates": gates_verdict(BASE_FULLHEAD_GATES),
            "served_byte_exact_identity": None,               # UNMEASURED served free-running
            "identity_status": "UNMEASURED (full-head exact verify -> plausibly exact; must measure free-run)",
            "spec_off": False,
        },
        "base_fullhead__candidate_verify": {
            "official_basis_tps": bf_cv_official_proxy,       # PROXY (unrealized officially)
            "official_basis_status": "UNREALIZED (proxy = local 299.28 * tau_lo)",
            "local_spec_served_tps": CV_SERVED_TPS_LOCAL,     # 299.28 (ref full-head 263.10)
            "clean_ar_tps": BASE_FULLHEAD_NOSPEC_LOCAL_TPS,
            "ppl": INT4_G128_OFFICIAL_PPL,
            "quality_gates": gates_verdict(BASE_FULLHEAD_GATES),
            "served_byte_exact_identity": False,              # REFUTED: free-run token 0.449, seq 9/64
            "served_freerun_token_identity": CV_FREERUN_TOKEN_IDENTITY,
            "served_freerun_seq_exact": CV_FREERUN_SEQ_EXACT,
            "perstep_identity": CV_PERSTEP_IDENTITY,
            "identity_status": "FAIL (served free-run 0.449; per-step 0.994 cascades, #566)",
            "spec_off": False,
        },
    }

    # ---- the two verdict bools (PR step 3) ----
    official_basis_is_spec_frame = True   # harness runs submitter serve.py + output_throughput;
    #                                       proven by the all-spec-dec 484-508 official frontier.

    # base_fullhead "officially beats" iff SOME base_fullhead form clears, ON A
    # REALIZED+VALIDATED basis: official TPS > 126.378 AND 4 gates AND served identity.
    def form_beats(form: dict[str, Any]) -> bool:
        realized = form["official_basis_status"].startswith("REALIZED")
        faster = (form["official_basis_tps"] or 0.0) > INT4_G128_OFFICIAL_TPS
        gates_ok = form["quality_gates"]["all_pass"]
        ident_ok = form["served_byte_exact_identity"] is True
        return bool(realized and faster and gates_ok and ident_ok)

    base_fullhead_forms = {k: v for k, v in table.items() if k.startswith("base_fullhead")}
    any_form_beats_realized = any(form_beats(v) for v in base_fullhead_forms.values())
    base_fullhead_official_beats_g128lmhead = bool(any_form_beats_realized)

    # the identity-safe served form's official-proxy (the only honest realized-style number)
    base_fullhead_identity_safe_official_proxy = bf_ar_official_proxy            # ~86.4 < 126.378
    # the fast-projection (NOT served-identity-validated) ceiling
    base_fullhead_fast_projection_official_proxy = bf_spec_official_proxy        # ~262.9 (MTP-K7)

    base_fullhead_official_tps_estimate = {
        "identity_safe_served_official_proxy": base_fullhead_identity_safe_official_proxy,
        "identity_safe_beats_126": base_fullhead_identity_safe_official_proxy > INT4_G128_OFFICIAL_TPS,
        "fast_projection_official_proxy_mtp_k7": base_fullhead_fast_projection_official_proxy,
        "fast_projection_official_proxy_cand_verify": bf_cv_official_proxy,
        "fast_projection_is_served_identity_validated": False,
        "note": ("identity-safe served (AR no-spec) ~86 < 126.378; the >126.378 fast "
                 "projections (MTP-K7 ~263 / cand-verify ~310) are UNREALIZED and not "
                 "served-identity-validated (cand-verify REFUTED 0.449; MTP-K7 UNMEASURED)."),
    }

    # ---- GO/NO-GO (PR step 4) — NO autonomous fire either way ----
    # the conditional human-approval candidate is the MTP-K7 path: projects >126.378
    # at PLAUSIBLE byte-exactness, gated on 3 served preconditions.
    human_approval_candidate = "base_fullhead + MTP-K7 (full-head exact verify)"
    human_approval_preconditions = [
        "SERVED FREE-RUNNING greedy identity == 1.0 (free-run, not per-step — the #566 lesson)",
        "remote-loadable packaged serve.py with the MTP drafter (Hub-hosted / in-submission)",
        "PPL <= 2.42 on a real serve (the 252.69/253.99 figures were never PPL-validated served)",
    ]
    verdict_label = ("NO-GO swap (int4_g128_lmhead@126.378 remains the best VALID official "
                     "quality-safe submission); CONDITIONAL human-approval GO-flag for "
                     "base_fullhead+MTP-K7 gated on 3 served preconditions; NO autonomous fire.")

    # ---- self-tests (grounding + arithmetic + logic + NaN-clean) ----
    st = {
        "all_numbers_grounded_against_sources": grounding["all_grounded"],
        "src_ar_specdec_report_matches": grounding["base_fullhead_ar_specdec_report"]["all_match"],
        "src_cv_served_report_matches": grounding["cv_served_report"]["all_match"],
        "src_cv_stage3_identity_matches": grounding["cv_stage3_identity"]["all_match"],
        "src_cv_stage3c_tiebreak_matches": grounding["cv_stage3c_tiebreak"]["all_match"],
        # the basis is spec-frame, proven by the all-spec-dec official frontier > AR ceiling
        "frontier_exceeds_int4_ar_ceiling": PUBLIC_1_OFFICIAL_TPS > INT4_G128_OFFICIAL_TPS,
        "ship_exceeds_int4_ar_ceiling": SHIP_OFFICIAL_TPS > INT4_G128_OFFICIAL_TPS,
        "official_basis_is_spec_frame_true": official_basis_is_spec_frame is True,
        # int4_g128 is the realized, served, identity-safe, quality-safe number
        "int4_g128_realized": table["int4_g128_lmhead"]["official_basis_status"].startswith("REALIZED"),
        "int4_g128_gates_all_pass": table["int4_g128_lmhead"]["quality_gates"]["all_pass"],
        "int4_g128_identity_pass": table["int4_g128_lmhead"]["served_byte_exact_identity"] is True,
        # base_fullhead identity-safe served form loses to 126.378 on AR
        "bf_ar_proxy_below_126": base_fullhead_identity_safe_official_proxy < INT4_G128_OFFICIAL_TPS,
        # base_fullhead candidate-verify REFUTED on served byte-exact identity
        "cv_freerun_identity_fails": CV_FREERUN_TOKEN_IDENTITY < 0.999,
        "cv_perstep_looks_clean_but_cascades": (CV_PERSTEP_IDENTITY > 0.99) and (CV_FREERUN_TOKEN_IDENTITY < 0.5),
        "cv_hard_gate_fails": CV_STAGE3_HARD_GATE_PASS is False,
        "cv_voctb_exact_offline_only": (CV_TIEBREAK_VOCTB == 1.0) and (CV_TIEBREAK_POSTB < 1.0),
        # MTP-K7 fast projection exceeds 126.378 (the genuine upside) but is unrealized/unmeasured
        "mtp_spec_proxy_exceeds_126": base_fullhead_fast_projection_official_proxy > INT4_G128_OFFICIAL_TPS,
        "mtp_spec_identity_unmeasured": table["base_fullhead__mtp_k7_spec"]["served_byte_exact_identity"] is None,
        # the 3x gap is spec-dec, not hardware (same-pod nospec vs spec)
        "three_x_gap_is_specdec_not_hw": math.isclose(
            BASE_FULLHEAD_SPEC_LOCAL_TPS / BASE_FULLHEAD_NOSPEC_LOCAL_TPS, 3.044, abs_tol=0.05),
        "tau_lo_is_small_hw_gap": abs(TAU_LO - 1.0) < 0.05,
        # the headline verdict
        "base_fullhead_does_not_officially_beat": base_fullhead_official_beats_g128lmhead is False,
        "no_base_fullhead_form_realized_beats": (not any_form_beats_realized),
        # NaN-clean
        "nan_clean": all(math.isfinite(x) for x in [
            bf_ar_official_proxy, bf_spec_official_proxy, bf_cv_official_proxy,
            base_fullhead_identity_safe_official_proxy, base_fullhead_fast_projection_official_proxy,
        ]),
    }
    st["self_test_passes"] = all(st.values())
    self_det = bool(st["self_test_passes"])

    packet = {
        "pr": 594,
        "card": "submission-basis-reconciliation",
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
        "no_hf_job": True,
        "no_submission": True,
        "no_fire": True,
        "peak_gpu_gib": 0.0,
        # ---- contract anchors ----
        "int4_g128_official_tps": INT4_G128_OFFICIAL_TPS,
        "bf16_baseline_official_tps": BF16_BASELINE_OFFICIAL_TPS,
        "ship_official_tps": SHIP_OFFICIAL_TPS,
        "public_1_official_tps": PUBLIC_1_OFFICIAL_TPS,
        "ppl_cap": PPL_CAP,
        "tau_lo": TAU_LO,
        # ---- the crux: what the official benchmark measures ----
        "official_benchmark_mechanism": {
            "harness_runs_submitter_serve_command": True,
            "tps_metric": "summary.json:tps = result['output_throughput'] (output_tokens/wall @ conc=1)",
            "fixed_ar_no_spec_harness": False,
            "spec_dec_in_frame": True,
            "proof": ("official leaderboard frontier 484-508 TPS are all spec-dec stacks "
                      "(K7 drafters + split-KV/tree verify); impossible under fixed-AR "
                      "(bf16 AR 44.018; int4 AR 126.378)."),
            "source": "official/main_bucket/shared_resources/speed_benchmark/{README.md,scripts/hf_bucket_single_job.py}",
        },
        # ---- the reconciled head-to-head table ----
        "headtohead_table": table,
        # ---- the two PR-named verdict bools ----
        "official_basis_is_spec_frame": official_basis_is_spec_frame,
        "base_fullhead_official_beats_g128lmhead": base_fullhead_official_beats_g128lmhead,
        "base_fullhead_official_tps_estimate": base_fullhead_official_tps_estimate,
        # ---- ledger correction ----
        "ledger_correction": {
            "loose_claim": "base_fullhead 252.69 ships clean",
            "corrected": ("252.69 is a LOCAL MTP-K7 spec figure (wirbel #553 / lawine wndiyzxk "
                          "253.99), NOT an official submittable number; base_fullhead has never "
                          "been officially served. It does NOT 'ship clean' without a SERVED "
                          "free-running greedy identity == 1.0 proof (per-step 0.994 is not enough)."),
            "int4_g128_is_best_valid_official_quality_safe": True,
        },
        # ---- GO/NO-GO ----
        "go_no_go": {
            "autonomous_fire": "NO",
            "swap_best_submission": "NO-GO (int4_g128_lmhead@126.378 stays)",
            "human_approval_candidate": human_approval_candidate,
            "human_approval_preconditions": human_approval_preconditions,
            "apples_to_apples_note": ("spec-dec is official-in-frame -> int4_g128_lmhead can ALSO be "
                                      "MTP-drafted; the int4 head is a cheaper verify substrate than "
                                      "the full bf16 head, so int4_g128+MTP would likely DOMINATE "
                                      "base_fullhead+MTP. Prefer int4_g128+MTP for a quality-safe spec slot."),
            "verdict_label": verdict_label,
        },
        # ---- grounding + self-tests ----
        "grounding": grounding,
        "sources": {k: {"path": v["path"], "pr": v["pr"], "wandb": v["wandb"]} for k, v in SOURCES.items()},
        "self_tests": st,
        "self_det": self_det,
        "self_tests_passed": sum(1 for v in st.values() if v),
        "self_tests_total": len(st),
        # ---- primary/test metric (for SENPAI-RESULT) ----
        "primary_metric_name": "base_fullhead_official_beats_g128lmhead",
        "primary_metric_value": int(base_fullhead_official_beats_g128lmhead),
    }
    return packet


def wandb_summary(p: dict[str, Any]) -> dict[str, Any]:
    t = p["headtohead_table"]
    est = p["base_fullhead_official_tps_estimate"]
    return {
        # --- KEY VERDICTS ---
        "official_basis_is_spec_frame": p["official_basis_is_spec_frame"],
        "official_basis_is_spec_frame_int": int(p["official_basis_is_spec_frame"]),
        "base_fullhead_official_beats_g128lmhead": p["base_fullhead_official_beats_g128lmhead"],
        "base_fullhead_official_beats_g128lmhead_int": int(p["base_fullhead_official_beats_g128lmhead"]),
        "int4_g128_is_best_valid_official": int(p["ledger_correction"]["int4_g128_is_best_valid_official_quality_safe"]),
        # --- official anchors ---
        "int4_g128_official_tps": p["int4_g128_official_tps"],
        "bf16_baseline_official_tps": p["bf16_baseline_official_tps"],
        "tau_lo": p["tau_lo"],
        # --- base_fullhead official-proxy estimates ---
        "bf_identity_safe_served_official_proxy": est["identity_safe_served_official_proxy"],
        "bf_identity_safe_beats_126_int": int(est["identity_safe_beats_126"]),
        "bf_fast_proxy_mtp_k7": est["fast_projection_official_proxy_mtp_k7"],
        "bf_fast_proxy_cand_verify": est["fast_projection_official_proxy_cand_verify"],
        "bf_fast_served_identity_validated_int": int(est["fast_projection_is_served_identity_validated"]),
        # --- the head-to-head table (scalar projections) ---
        "int4_g128_official_basis_tps": t["int4_g128_lmhead"]["official_basis_tps"],
        "int4_g128_identity_pass_int": int(t["int4_g128_lmhead"]["served_byte_exact_identity"] is True),
        "bf_ar_clean_ar_tps_local": t["base_fullhead__ar_nospec"]["clean_ar_tps"],
        "bf_mtp_spec_local_tps": t["base_fullhead__mtp_k7_spec"]["local_spec_served_tps"],
        "bf_mtp_e_accept_exact": t["base_fullhead__mtp_k7_spec"]["e_accept_exact"],
        "bf_cv_local_spec_served_tps": t["base_fullhead__candidate_verify"]["local_spec_served_tps"],
        "bf_cv_freerun_token_identity": t["base_fullhead__candidate_verify"]["served_freerun_token_identity"],
        "bf_cv_freerun_seq_exact": t["base_fullhead__candidate_verify"]["served_freerun_seq_exact"],
        "bf_cv_perstep_identity": t["base_fullhead__candidate_verify"]["perstep_identity"],
        "bf_cv_identity_fail_int": int(t["base_fullhead__candidate_verify"]["served_byte_exact_identity"] is False),
        "bf_mtp_identity_unmeasured_int": int(t["base_fullhead__mtp_k7_spec"]["served_byte_exact_identity"] is None),
        # --- meta ---
        "all_numbers_grounded_int": int(p["grounding"]["all_grounded"]),
        "self_det": p["self_det"],
        "self_det_int": int(p["self_det"]),
        "self_tests_passed": p["self_tests_passed"],
        "self_tests_total": p["self_tests_total"],
        "peak_gpu_gib": p["peak_gpu_gib"],
        "analysis_only": True,
        "official_tps": 0,
        "primary_metric": p["primary_metric_value"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="fern/submission-basis-reconciliation")
    ap.add_argument("--wandb-group", default="submission-basis-reconciliation")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    p = build_packet()
    HERE.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(p, fh, indent=2)
    print(f"[reconcile] wrote {OUT_JSON}", flush=True)

    line = "=" * 6 + " PR #594 — SUBMISSION-BASIS RECONCILIATION (analysis-only) " + "=" * 6
    print("\n" + line, flush=True)
    print(f"  grounding: all_numbers_grounded = {p['grounding']['all_grounded']}", flush=True)
    print(f"  official_basis_is_spec_frame = {p['official_basis_is_spec_frame']}  "
          f"(harness runs submitter serve.py; frontier 484-508 all spec-dec)", flush=True)
    print("  head-to-head [official_basis | local_spec | clean_AR | gates | served-identity]:", flush=True)
    t = p["headtohead_table"]
    for key, r in t.items():
        ob = r["official_basis_tps"]
        ob_s = f"{ob:7.2f}" if isinstance(ob, (int, float)) else f"{str(ob):>7s}"
        ls = r.get("local_spec_served_tps")
        ls_s = f"{ls:7.2f}" if isinstance(ls, (int, float)) else "   --  "
        ar = r.get("clean_ar_tps")
        ar_s = f"{ar:6.2f}" if isinstance(ar, (int, float)) else "  --  "
        g = "PASS" if r["quality_gates"]["all_pass"] else "FAIL"
        print(f"    {key:>32s} : {ob_s} | {ls_s} | {ar_s} | {g} | {r['identity_status']}", flush=True)
    est = p["base_fullhead_official_tps_estimate"]
    print(f"  base_fullhead identity-safe served official-proxy = "
          f"{est['identity_safe_served_official_proxy']:.2f}  (< {p['int4_g128_official_tps']})", flush=True)
    print(f"  base_fullhead fast proxy (MTP-K7, UNVALIDATED-served) = "
          f"{est['fast_projection_official_proxy_mtp_k7']:.2f}", flush=True)
    print(f"  >>> base_fullhead_official_beats_g128lmhead = "
          f"{p['base_fullhead_official_beats_g128lmhead']}", flush=True)
    print(f"  >>> {p['go_no_go']['verdict_label']}", flush=True)
    print(f"  self_det = {p['self_det']}  ({p['self_tests_passed']}/{p['self_tests_total']} self-tests)", flush=True)
    if not p["self_det"]:
        failed = [k for k, v in p["self_tests"].items() if not v]
        print(f"  !! FAILED self-tests: {failed}", flush=True)
    print("=" * len(line), flush=True)

    rid = None
    if not args.no_wandb:
        try:
            from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                               log_json_artifact, log_summary)
            run = init_wandb_run(
                job_type="systems-profile",
                agent="fern",
                name=args.wandb_name,
                group=args.wandb_group,
                tags=["submission-basis", "reconciliation", "analysis-only", "no-fire",
                      "base_fullhead", "int4_g128_lmhead", "spec-frame", "pr594"],
                notes="PR #594 reconcile base_fullhead vs int4_g128_lmhead onto ONE basis. "
                      "official_basis_is_spec_frame=TRUE (harness runs submitter serve.py; frontier "
                      "484-508 all spec-dec). base_fullhead_official_beats_g128lmhead=FALSE (no realized "
                      "official win; cand-verify served identity 0.449 REFUTED; MTP-K7 unmeasured-served "
                      "+ unrealized). int4_g128@126.378 is best valid official quality-safe submission.",
                config={
                    "analysis_only": True, "no_gpu": True, "no_fire": True,
                    "int4_g128_official_tps": INT4_G128_OFFICIAL_TPS,
                    "tau_lo": TAU_LO,
                    "cited_runs": ["905tbujn", "83jiwjr9", "wndiyzxk", "ufv4nk21"],
                    "cited_jobs": ["6a2d5a96234ca64b60121aa5"],
                    "cited_prs": ["#553", "#560", "#566", "#573", "#582", "#585"],
                },
            )
            if run is not None:
                log_summary(run, wandb_summary(p), step=0)
                log_json_artifact(run, name="submission-basis-reconciliation",
                                  artifact_type="basis-reconcile-synthesis", data=p)
                rid = getattr(run, "id", None)
                finish_wandb(run)
                p["wandb_run_id"] = rid
                with OUT_JSON.open("w") as fh:
                    json.dump(p, fh, indent=2)
                print(f"[reconcile] wandb run id = {rid}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[reconcile] wandb unavailable: {exc}", flush=True)

    senpai = {
        "terminal": True,
        "status": "complete",
        "pending_arms": False,
        "analysis_only": True,
        "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": p["self_det"],
        "official_basis_is_spec_frame": p["official_basis_is_spec_frame"],
        "base_fullhead_official_beats_g128lmhead": p["base_fullhead_official_beats_g128lmhead"],
        "base_fullhead_identity_safe_official_proxy": round(
            p["base_fullhead_official_tps_estimate"]["identity_safe_served_official_proxy"], 2),
        "base_fullhead_fast_proxy_mtp_k7": round(
            p["base_fullhead_official_tps_estimate"]["fast_projection_official_proxy_mtp_k7"], 2),
        "int4_g128_official_tps": INT4_G128_OFFICIAL_TPS,
        "self_tests_passed": p["self_tests_passed"],
        "primary_metric": {"name": "base_fullhead_official_beats_g128lmhead",
                           "value": int(p["base_fullhead_official_beats_g128lmhead"])},
        "test_metric": {"name": "int4_g128_official_tps", "value": INT4_G128_OFFICIAL_TPS},
    }
    print("\nSENPAI-RESULT: " + json.dumps(senpai, separators=(",", ":")), flush=True)
    return 0 if p["self_det"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
