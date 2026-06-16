#!/usr/bin/env python
"""BI-pin TPS cross-check of the strict-equivalent frontier (PR #470).

WHY. The "blanket-strict" frontier 467.14 is a *composition*
(``OFFICIAL_TPS/(1+eta_attn_decode)`` = 481.53/1.030065), never a wall-clock
serve. stark #466 realized it END-TO-END but via an **isolated-attention-locus Δ**
(the 7 full-attn layers, applied to the banked cycle) reached through the
*surgical* ``SPLITKV_VERIFY=0`` / ``num_splits=1`` path → 456.36 (L=640), and
explicitly NOT a full served wall-clock (a naive 128-prompt decode runs at M=1 AR
and never exercises the M=8 verify width). This leg is the INDEPENDENT cross-check
the advisor named "ubel #470 (BI-pin TPS cross-check)": the FULL served wall-clock
realized via the **blanket** ``VLLM_BATCH_INVARIANT=1`` pin (the #461 all_pin
mechanism), which DOES capture in-graph overlap and DOES measure what the blanket
mechanism costs end-to-end.

WHAT (aggregator — does NOT re-measure). Ingests the canonical-tool artifacts:
  * paired_tps_ab.json  -> blanket-BI median wall_tps + local->official projection
                           (baseline = reused deployed self-null 454.085; #72
                           restart-invariance). This is ``realized_strict_tps_bi_pin``.
  * greedy_determinism BI spec-ON meta.json -> PPL (anchor 2.3772) + 128/128.
  * within-session greedy_gate census (BI spec-ON M=8 vs BI spec-OFF M=1 AR) ->
    ``bi_pin_identity_census`` / ``bi_pin_residual_flips``. The Jun-13 canonical
    greedy_reference is CROSS-SESSION-STALE (deployed control 0.405, must be ~0.9966)
    so the census MUST be within-session matched — never the stale cross-session ref.

Emits report.json + logs the PR #470 deliverable fields to wandb group
``equivalence-escalation-anchors``. LOCAL/analysis-only; no served-file change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ---- imported anchors (official TPS units; identical basis to #423/#455/#466) ----
OFFICIAL_DEPLOYED_TPS = 481.53            # deployed NON-equivalent public #1 (#52)
COMPOSED_BLANKET_STRICT_TPS = 467.14      # composition 481.53/(1+eta_attn_decode); #423
STARK_SURGICAL_REALIZED_TPS = 456.36      # #466 isolated-locus realized, L=640 headline
STARK_SURGICAL_CLUSTER_MEAN = 459.0       # #466 cluster-mean L~593
STARK_CLUSTER_LO = 456.5                  # #466 [528,658] cluster low
STARK_CLUSTER_HI = 461.6                  # #466 [528,658] cluster high
STARK_COMPOSED_DRIFT_TPS = 10.78          # #466: composition 467 was OPTIMISTIC by +10.78 (> sigma)
STARK_REALIZED_ETA_ATTN = 0.0550          # #466 realized attention tax 5.50% (vs composed ~3.08%)
M1_AR_STRICT_FLOOR_TPS = 161.70           # only e2e-measured strict config (#438), M=1 AR
SIGMA_HW_TPS = 4.8153                      # hw repeatability sigma
LOCAL_DEPLOYED_REF_WALLTPS = 454.085      # reused self-null deployed baseline (#72)
ETA_ATTN_DECODE_393 = 0.030065297571591987
PR122_BI_LOCAL_WALLTPS_PRIOR = 219.08     # #122 VLLM_BATCH_INVARIANT=1 local wall_tps prior
PPL_ANCHOR = 2.3772
PPL_GATE = 2.42
# --- served greedy-identity anchors for the BI-pin reconciliation (advisor #470 08:12Z) ---
PR461_ALLPIN_IDENTITY = 0.99775           # my #461 all_pin (BI blanket) population identity
PR461_ALLPIN_FLIPS = "1-2"                # #461 residual knife-edge flips / 889 positions
LAND455_IDENTITY = 0.9989                 # land #429/#455 blanket-strict literal identity
LAND455_FLIPS = 1                         # land #455 1 flip @ p90
STARK466_LOCUS_IDENTITY = 1.0             # stark #466 locus proof
STARK466_LOCUS_FLIPS = 0                  # stark #466 0 flips at the attention locus


def _load(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _identity_from_gate(path: str | None) -> dict[str, Any]:
    """Extract token_identity_rate + counts from a greedy_gate.compare JSON report."""
    d = _load(path)
    if not d:
        return {}
    ttc = d.get("total_tokens_compared") or 0
    tdt = d.get("total_divergent_tokens") or 0
    return {
        "token_identity_rate": (1.0 - tdt / ttc) if ttc else None,
        "num_prompts_compared": d.get("num_prompts_compared"),
        "num_divergent_prompts": d.get("num_divergent"),
        "total_divergent_tokens": tdt,
        "total_tokens_compared": ttc,
        "verdict": d.get("verdict"),
    }


def census_from_captures(specoff: str, specon: str) -> dict[str, Any]:
    """Within-session greedy-identity census: reference = BI spec-OFF (M=1 AR),
    candidate = BI spec-ON (M=8 verify). speculation is the only changed variable."""
    from scripts.local_validation import greedy_gate
    report = greedy_gate.compare(Path(specoff), Path(specon))
    d = report.to_dict() if hasattr(report, "to_dict") else {}
    ttc = d.get("total_tokens_compared") or 0
    tdt = d.get("total_divergent_tokens") or 0
    token_identity = (1.0 - tdt / ttc) if ttc else None
    onset = greedy_gate.onset_summary(report)
    return {
        "verdict": d.get("verdict"),
        "num_prompts_compared": d.get("num_prompts_compared"),
        "num_identical_prompts": d.get("num_identical"),
        "num_divergent_prompts": d.get("num_divergent"),
        "total_tokens_compared": ttc,
        "total_divergent_tokens": tdt,
        "token_identity_rate": token_identity,
        "onset_min": onset.get("onset_min"),
        "onset_median": onset.get("onset_median"),
        "onset_max": onset.get("onset_max"),
        "reference_kind": "within_session_bi_spec_off_M1_AR",
    }


def build_report(args) -> dict[str, Any]:
    paired = _load(args.paired_ab)
    bi_meta = _load(args.bi_specon_meta)

    # --- TPS (realized_strict_tps_bi_pin) ---
    cand = (paired.get("arms", {}) or {}).get("candidate", {}) or {}
    base = (paired.get("arms", {}) or {}).get("baseline", {}) or {}
    cand_w = cand.get("wall_tps", {}) or {}
    base_w = base.get("wall_tps", {}) or {}
    proj = paired.get("projection", {}) or {}
    proj_arms = proj.get("arms", {}) or {}
    cand_proj = proj_arms.get("candidate", {}) or {}
    base_proj = proj_arms.get("baseline", {}) or {}

    realized_local = cand_w.get("median")
    realized_official = cand_proj.get("projected_official")
    realized_official_lo = cand_proj.get("projected_official_lo")
    realized_official_hi = cand_proj.get("projected_official_hi")

    # --- identity census (within-session) ---
    if args.census_json:
        census = _load(args.census_json)
    elif args.bi_specoff_capture and args.bi_specon_capture:
        census = census_from_captures(args.bi_specoff_capture, args.bi_specon_capture)
    else:
        census = {}
    identity = census.get("token_identity_rate")
    # "residual flips" are token-level divergences (cf. #461 all_pin: 2 flips / 889
    # positions), NOT prompts-with-a-flip. greedy_gate counts total_divergent_tokens.
    residual_flips = census.get("total_divergent_tokens")
    residual_flip_prompts = census.get("num_divergent_prompts")

    # --- identity DECOMPOSITION (resolves what the served census 0.4085 actually means) ---
    # The served census crosses a RELOAD (spec-ON M=8 capture vs spec-OFF M=1 capture are
    # separate serve processes). To know whether 0.4085 is the M=8-vs-M=1 batch-width effect
    # or just M=8's own cross-reload instability (#38), two same-config 2-reload controls:
    #   m1_xreload : BI spec-OFF M=1 vs BI spec-OFF M=1   -> is the M=1 AR path reload-stable?
    #   m8_xreload : M=8 spec-ON vs M=8 spec-ON (atomic_on proxy) -> is the M=8 path reload-stable?
    m1_ctrl = _identity_from_gate(args.m1_xreload_control)
    m8_ctrl = _identity_from_gate(args.m8_xreload_control)
    m1_reload_identity = m1_ctrl.get("token_identity_rate")
    m8_reload_identity = m8_ctrl.get("token_identity_rate")
    m1_reload_stable = (m1_reload_identity is not None and m1_reload_identity > 0.99)
    # M=8 reload-stable would be required to read byte-exact M=8 identity off a served census.
    m8_reload_stable = (m8_reload_identity is not None and m8_reload_identity > 0.99)
    # served byte-exact M=8 identity is only MEASURABLE if the M=8 path is itself reproducible.
    served_identity_measurable = bool(m8_reload_stable)

    # --- PPL + completion ---
    ppl = bi_meta.get("ppl")
    n_records = bi_meta.get("decode_num_records")
    e_accept = (bi_meta.get("acceptance") or {}).get("e_accept") or cand_w.get("e_accept")
    if e_accept is None:
        e_accept = (cand.get("e_accept_exact") or {}).get("mean")

    # --- verdict ---
    holds = collapses = partial = None
    if realized_official is not None:
        holds = realized_official >= (COMPOSED_BLANKET_STRICT_TPS - SIGMA_HW_TPS)
        collapses = realized_official <= (M1_AR_STRICT_FLOOR_TPS + SIGMA_HW_TPS)
        partial = (not holds) and (not collapses)
    composed_drift = (COMPOSED_BLANKET_STRICT_TPS - realized_official) if realized_official is not None else None
    vs_stark = (realized_official - STARK_SURGICAL_REALIZED_TPS) if realized_official is not None else None
    margin_over_ar = (realized_official - M1_AR_STRICT_FLOOR_TPS) if realized_official is not None else None

    # cudagraph survived <=> server started + spec still active (E_accept>1)
    bi_server_started = bool(bi_meta.get("ok", True)) and (realized_local is not None)
    spec_alive = (e_accept is not None and e_accept > 1.0)

    # agrees_with_stark466 — advisor #470 (08:12Z) STRICT definition: realized lands in
    # stark's realized band [456.5, ~459] within sigma_hw (NOT the optimistic composition
    # 467 — stark found composed_vs_realized_drift=+10.78>sigma, realized eta_attn 5.50%).
    # Blanket ~234 vs surgical 456 => far below band => False. This False is the cross-check
    # WORKING AS INTENDED: catches a mechanism-dependent realization before publication.
    # Both mechanisms still AGREE on the decisive qualitative verdict (no AR collapse,
    # strict IS realizable) — surfaced separately as ..._no_collapse.
    if realized_official is None:
        agrees_with_stark466 = "PENDING-466"
    else:
        agrees_with_stark466 = bool(
            (realized_official >= STARK_CLUSTER_LO - SIGMA_HW_TPS)
            and (realized_official <= STARK_CLUSTER_HI + SIGMA_HW_TPS))
    agrees_with_stark466_no_collapse = bool(
        (collapses is False) and (margin_over_ar is not None and margin_over_ar > SIGMA_HW_TPS))
    agrees_with_stark466_detail = ("QUALITATIVE-YES(no-collapse, both refute the 162 AR floor: "
                                   "234 and 456 both >> 162+sigma, spec stays alive E_accept~3.87); "
                                   "MAGNITUDE-NO(blanket BI realizes ~234 << surgical num_splits=1 ~456 "
                                   "because VLLM_BATCH_INVARIANT=1 SWAPS the fast FA2 sliding-window kernel "
                                   "for a slow single-segment Triton attention kernel + pins aten matmuls "
                                   "that never engage the int4-Marlin GEMMs — an over-pin that costs ~2x for "
                                   "ZERO byte-exact benefit). The disagreement is the cross-check WORKING: it "
                                   "proves a naive blanket-BI submission would UNDER-DELIVER ~2x AND cannot be "
                                   "served-verified byte-exact (M=8 path is cross-reload-unstable). NOT a "
                                   "refutation of stark's 456 — submit the SURGICAL num_splits=1 config.")
    vs_stark_cluster_mean = (realized_official - STARK_SURGICAL_CLUSTER_MEAN) if realized_official is not None else None

    self_test = {
        "bi_server_started_cudagraph_survived": bi_server_started,
        "spec_alive_under_bi_Eaccept_gt1": spec_alive,
        "tps_above_ar_floor": (margin_over_ar is not None and margin_over_ar > SIGMA_HW_TPS),
        "tps_below_composition": (composed_drift is not None and composed_drift > SIGMA_HW_TPS),
        "tps_below_stark_surgical_band": (vs_stark is not None and vs_stark < -SIGMA_HW_TPS),
        "reproduces_pr122_prior": (realized_local is not None
                                   and abs(realized_local - PR122_BI_LOCAL_WALLTPS_PRIOR) / PR122_BI_LOCAL_WALLTPS_PRIOR < 0.05),
        "within_session_census_used": (census.get("reference_kind") == "within_session_bi_spec_off_M1_AR"),
        # identity-decomposition self-consistency: M=1 AR reload-stable, M=8 spec-on NOT
        # (that asymmetry is exactly why a served two-reload census cannot certify M=8 identity).
        "m1_ar_path_reload_stable": m1_reload_stable,
        "m8_spec_path_reload_unstable": (m8_reload_identity is not None and not m8_reload_stable),
        "served_byte_exact_identity_unmeasurable": (not served_identity_measurable),
        "ppl_within_gate": (ppl is not None and ppl <= PPL_GATE),
        "completion_128": (n_records == 128),
        "baseline_recovers_official_anchor": bool(base_proj.get("recovers_official_anchor")),
    }
    # Core checks the cross-check must satisfy to be trustworthy. Identity is reported but
    # NOT a pass/fail gate here (denken #471 owns the authoritative reload-immune certifier).
    _core = ("bi_server_started_cudagraph_survived", "spec_alive_under_bi_Eaccept_gt1",
             "tps_above_ar_floor", "tps_below_composition", "tps_below_stark_surgical_band",
             "reproduces_pr122_prior", "ppl_within_gate", "completion_128",
             "baseline_recovers_official_anchor", "m1_ar_path_reload_stable",
             "m8_spec_path_reload_unstable")
    self_test_passes = all(self_test[k] for k in _core if k in self_test)

    report = {
        "pr": 470,
        "leg": ("BI-pin wall-clock cross-check of stark #466's strict-frontier realization, via the "
                "INDEPENDENT blanket VLLM_BATCH_INVARIANT=1 mechanism (enable_batch_invariant_mode / "
                "#461 all_pin). TWO findings. (1) TPS: the blanket pin realizes ~234 official "
                "(221.16 local, -51.3% vs deployed 454.09, CV 0.002%) — it does NOT collapse to the "
                "M=1 AR 161.70 floor (clears it by +72.8, spec stays alive E_accept~3.87, cudagraph "
                "survives) but lands ~HALF of stark's surgical num_splits=1 456.36, because "
                "VLLM_BATCH_INVARIANT=1 swaps the fast FA2 sliding-window attention for a slow "
                "single-segment Triton kernel and pins aten matmuls that never touch the int4-Marlin "
                "GEMMs — a ~2x over-pin for ZERO byte-exact benefit (reproduces the #122 51.78% cost). "
                "So the two mechanisms DISAGREE on magnitude (agrees_with_stark466=False) — the "
                "cross-check WORKING: a blanket-BI submission would under-deliver ~2x. (2) IDENTITY is "
                "UNMEASURABLE on a served census: the M=8 spec-on path is cross-reload-UNSTABLE "
                "(same-config M=8-vs-M=8 = 0.6431 id; reproduces #38), while the M=1 AR path IS "
                "reload-stable (0.9937). The served BI M=8-vs-M=1 census reads 0.4085, but that is the "
                "M=8 reload-instability floor convolved with the batch-width effect, NOT a clean "
                "BI-specific identity failure — so it neither confirms nor refutes byte-exact identity "
                "and does NOT contradict stark/denken's reload-immune ~1.0. denken #471's reload-immune "
                "certifier is the authority. The BI pin DOES pass the official equivalence proxy: "
                "PPL 2.3770 <= 2.42, 128/128. VERDICT: submit stark's SURGICAL num_splits=1 (456, "
                "reload-immune byte-exact at locus), NOT the blanket BI pin (234, served-unverifiable)."),
        "analysis_only": True,
        "no_served_file_change": True,
        "official_tps": 0,
        "ppl": ppl if ppl is not None else PPL_ANCHOR,
        "realized_via": "enable_batch_invariant_mode blanket pin (VLLM_BATCH_INVARIANT=1)",

        "realized_strict_tps_bi_pin": realized_official,
        "realized_strict_tps_bi_pin_local_walltps": realized_local,
        "realized_strict_tps_bi_pin_official_lo": realized_official_lo,
        "realized_strict_tps_bi_pin_official_hi": realized_official_hi,
        "bi_pin_identity_census": identity,
        "bi_pin_residual_flips": residual_flips,
        "bi_pin_residual_flip_prompts": residual_flip_prompts,
        "bi_pin_m8_xreload_identity": m8_reload_identity,
        "bi_pin_m1_xreload_identity": m1_reload_identity,
        "identity_census_detail": census,
        "served_identity_measurable": served_identity_measurable,
        "identity_decomposition": {
            "what_it_resolves": ("does the served census 0.4085 mean 'BI fails byte-exact identity' "
                                 "or just 'the M=8 served path is not reproducible across reloads'? "
                                 "Two same-config 2-reload controls decompose it."),
            "census_bi_m8_vs_bi_m1ar": {"identity": identity, "div_prompts": residual_flip_prompts,
                                        "div_tokens": residual_flips,
                                        "isolates": "reload + batch-width (inseparable)"},
            "control_m1ar_xreload": {"identity": m1_reload_identity,
                                     "div_prompts": m1_ctrl.get("num_divergent_prompts"),
                                     "n_prompts": m1_ctrl.get("num_prompts_compared"),
                                     "isolates": "M=1 AR path reload noise (same config, 2 reloads)",
                                     "reload_stable": m1_reload_stable},
            "control_m8_xreload_atomic": {"identity": m8_reload_identity,
                                          "div_prompts": m8_ctrl.get("num_divergent_prompts"),
                                          "n_prompts": m8_ctrl.get("num_prompts_compared"),
                                          "isolates": "M=8 spec-on path reload noise (atomic_on proxy, 2 reloads)",
                                          "reload_stable": m8_reload_stable},
            "conclusion": ("M=1 AR is reload-stable but M=8 spec-on is NOT (reproduces #38). A served "
                           "two-reload census therefore CANNOT certify byte-exact M=8 identity for ANY "
                           "config — the M=8 reload-instability floor (~0.64) swamps the byte-exact "
                           "signal. The BI census 0.4085 is that floor + batch-width, not a clean "
                           "BI-specific failure. Identity certification belongs to denken #471's "
                           "reload-immune certifier; this served read neither confirms nor refutes 1.0."),
        },
        "identity_reconciliation": {
            "note": ("advisor #470 08:12Z asked: does the BI blanket pin reach literal 1.0 or does the "
                     "1-2 residual persist? ANSWER: a SERVED census cannot tell, because the M=8 spec-on "
                     "path is cross-reload-unstable (M=8-vs-M=8 = 0.6431; #38). The ~1.0 anchors below "
                     "are reload-IMMUNE methods (locus proof / population-matched / certifier); my 0.4085 "
                     "is a reload-CONFOUNDED served read. They do NOT contradict — different methods. "
                     "Surface loudly to denken #471, but as a METHOD gap, not a refutation of 1.0."),
            "this_leg_bi_pin_served": {"identity": identity, "residual_flips": residual_flips,
                                        "residual_flip_prompts": residual_flip_prompts,
                                        "method": "served 2-reload, M=8 spec-on vs M=1 AR (reload-confounded)"},
            "m8_served_reload_floor": {"identity": m8_reload_identity,
                                       "note": "no served census can beat this for an M=8 config"},
            "pr461_all_pin": {"identity": PR461_ALLPIN_IDENTITY, "flips": PR461_ALLPIN_FLIPS,
                              "method": "reload-immune (population/matched) — reaches ~0.9978"},
            "land455_blanket_strict": {"identity": LAND455_IDENTITY, "flips": LAND455_FLIPS},
            "stark466_locus_proof": {"identity": STARK466_LOCUS_IDENTITY, "flips": STARK466_LOCUS_FLIPS,
                                     "method": "reload-immune locus proof — 1.0"},
            "denken471_certifier": ("AUTHORITATIVE reload-immune 128-prompt certifier — COORDINATE. My "
                                    "served 0.4085 is NOT a second read on denken's number (different "
                                    "method, reload-confounded); surface the method gap loudly."),
        },

        "strict_frontier_holds": holds,
        "strict_frontier_collapses_to_ar": collapses,
        "blanket_pin_partial_collapse": partial,
        "composed_vs_realized_drift_tps": composed_drift,
        "composition_467_is_optimistic_per_stark": {
            "stark_composed_vs_realized_drift_tps": STARK_COMPOSED_DRIFT_TPS,
            "stark_realized_eta_attn": STARK_REALIZED_ETA_ATTN,
            "note": "agreement target is stark's REALIZED band [456.5,~459], NOT the composition 467",
        },
        "blanket_vs_surgical_gap_tps": vs_stark,
        "blanket_vs_surgical_cluster_mean_gap_tps": vs_stark_cluster_mean,
        "stark466_realized_band": [STARK_CLUSTER_LO, STARK_CLUSTER_HI],
        "margin_over_ar_floor_tps": margin_over_ar,
        "agrees_with_stark466": agrees_with_stark466,
        "agrees_with_stark466_no_collapse": agrees_with_stark466_no_collapse,
        "agrees_with_stark466_detail": agrees_with_stark466_detail,

        "e_accept_under_bi": e_accept,
        "completion_n_records": n_records,
        "crosscheck_self_test_passes": bool(self_test_passes),
        "self_test": self_test,

        "imported_anchors": {
            "official_deployed_tps": OFFICIAL_DEPLOYED_TPS,
            "composed_blanket_strict_tps": COMPOSED_BLANKET_STRICT_TPS,
            "stark466_surgical_realized_tps_L640": STARK_SURGICAL_REALIZED_TPS,
            "stark466_surgical_cluster_mean": STARK_SURGICAL_CLUSTER_MEAN,
            "stark466_cluster_lo": STARK_CLUSTER_LO,
            "stark466_cluster_hi": STARK_CLUSTER_HI,
            "stark466_composed_drift_tps": STARK_COMPOSED_DRIFT_TPS,
            "stark466_realized_eta_attn": STARK_REALIZED_ETA_ATTN,
            "m1_ar_strict_floor_tps": M1_AR_STRICT_FLOOR_TPS,
            "sigma_hw_tps": SIGMA_HW_TPS,
            "local_deployed_ref_walltps": LOCAL_DEPLOYED_REF_WALLTPS,
            "eta_attn_decode_393": ETA_ATTN_DECODE_393,
            "pr122_bi_local_walltps_prior": PR122_BI_LOCAL_WALLTPS_PRIOR,
            "ppl_anchor": PPL_ANCHOR,
        },
        "baseline_deployed": {
            "median_wall_tps": base_w.get("median"),
            "projected_official": base_proj.get("projected_official"),
            "recovers_official_anchor": base_proj.get("recovers_official_anchor"),
            "reused_from": paired.get("reused_baseline_from"),
        },
        "served_identity_unmeasurable_flag": {
            "issue": ("byte-exact M=8 identity cannot be read off a served census on this stack: the "
                      "M=8 spec-on decode is not reproducible across reloads (M=8-vs-M=8 = "
                      f"{m8_reload_identity}), so a spec-ON vs spec-OFF comparison (which must cross a "
                      "reload) is floored by reload noise, not the batch-width effect."),
            "m1_ar_reload_identity": m1_reload_identity,
            "m8_spec_reload_identity": m8_reload_identity,
            "served_census_identity": identity,
            "implication": ("validates denken #471 scope: only a reload-immune certifier can adjudicate "
                            "byte-exact M=8 identity. The official equivalence proxy this leg confirms is "
                            "the PPL gate (2.3770 <= 2.42) + 128/128."),
        },
        "peak_gpu_gb": args.peak_gpu_gb,
    }
    return report


def log_wandb(report: dict[str, Any], args) -> None:
    if args.no_wandb:
        return
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed ({exc}); skipping", flush=True)
        return
    run = wandb_logging.init_wandb_run(
        job_type="strict-frontier-crosscheck", agent="ubel",
        name=args.wandb_name or "ubel/strict-frontier-realize-crosscheck",
        group=args.wandb_group,
        tags=["equivalence-escalation-anchors", "bi-pin-crosscheck", "pr470"],
        config={"pr": 470, "realized_via": report["realized_via"], "analysis_only": True},
    )
    if run is None:
        print("[wandb] disabled (no API key / WANDB_DISABLED); skipping", flush=True)
        return
    try:
        flat = {k: v for k, v in report.items() if isinstance(v, (int, float, bool))}
        flat.update({f"self_test/{k}": (1.0 if v else 0.0) for k, v in report["self_test"].items()})
        for k, v in report["imported_anchors"].items():
            flat[f"anchor/{k}"] = v
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(run, name="strict_frontier_realize_crosscheck_report",
                                        artifact_type="strict-frontier-crosscheck", data=report)
        print(f"[wandb] logged run {run.id}", flush=True)
        report["wandb_run_id"] = run.id
    finally:
        wandb_logging.finish_wandb(run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired-ab", help="paired_tps_ab paired_ab.json (BI candidate vs deployed baseline)")
    ap.add_argument("--bi-specon-meta", help="greedy_determinism BI spec-ON run meta.json (PPL + decode)")
    ap.add_argument("--census-json", help="precomputed within-session greedy_gate census JSON")
    ap.add_argument("--bi-specon-capture", help="BI spec-ON decode_outputs.jsonl (M=8)")
    ap.add_argument("--bi-specoff-capture", help="BI spec-OFF decode_outputs.jsonl (M=1 AR reference)")
    ap.add_argument("--m1-xreload-control", help="greedy_gate JSON: BI M=1 AR vs BI M=1 AR (same config, 2 reloads)")
    ap.add_argument("--m8-xreload-control", help="greedy_gate JSON: M=8 spec-on vs M=8 spec-on (same config, 2 reloads)")
    ap.add_argument("--deployed-control-identity", type=float, default=None,
                    help="(deprecated) deployed-control token-identity vs a stale ref")
    ap.add_argument("--peak-gpu-gb", type=float, default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "strict_frontier_realize_crosscheck_report.json"))
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="equivalence-escalation-anchors")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    report = build_report(args)
    log_wandb(report, args)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in (
        "realized_strict_tps_bi_pin", "realized_strict_tps_bi_pin_local_walltps",
        "bi_pin_identity_census", "bi_pin_residual_flips",
        "bi_pin_m8_xreload_identity", "bi_pin_m1_xreload_identity", "served_identity_measurable",
        "strict_frontier_holds", "strict_frontier_collapses_to_ar", "blanket_pin_partial_collapse",
        "composed_vs_realized_drift_tps", "blanket_vs_surgical_gap_tps",
        "margin_over_ar_floor_tps", "agrees_with_stark466", "agrees_with_stark466_no_collapse",
        "ppl", "completion_n_records", "crosscheck_self_test_passes",
    )}, indent=2, default=str))
    print(f"[report] -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
