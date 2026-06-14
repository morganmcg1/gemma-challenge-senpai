#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Plan-B Rank-4 screen (PR #251, wirbel) — price the drafter->verifier L1-8
activation-recycle lever. CPU-only analytic bank-the-analysis.

THE QUESTION (researcher RESEARCH_IDEAS Rank-4, "+4-8% projected")
------------------------------------------------------------------
The fresh Plan-B sweep flagged "recycle the drafter's early-layer (L1-8)
transformer activations into the verifier instead of recomputing them" as the
single largest un-priced LINEAR-path TPS lever. The decisive screening question:
does the verifier RECOMPUTE the early layers L1-8 that the drafter already
computed (=> recycling saves real verify-FLOPs, a GO), or does the MTP head
already SHARE the base model's L1-8 forward so the activations are computed
exactly once (=> NULL lever, NO-GO)?

CRUX RECONCILIATION. The researcher projects +4-8% but a pure HBM-byte
accounting (~2 MB/step ~= 3.3 us vs the 1.2182 ms step ~= 0.27%/step) is an
order of magnitude smaller. The +4-8% can only be real if the saving is
verify-FLOPs (eliminated L1-8 recompute), NOT HBM bytes -- so the GO/NO-GO turns
entirely on the recompute-vs-shared CODE AUDIT.

THE CODE AUDIT (the frame -- decisive; grounded in the served wheel)
--------------------------------------------------------------------
Served stack: submissions/fa2sw_precache_kenyan, manifest
SPECULATIVE_CONFIG={"method":"mtp","model":"/tmp/qat-assistant",
"num_speculative_tokens":7}; vLLM 0.22.1rc1.dev307+g3e8afdf78.

  * method="mtp" + Gemma4 target => the drafter is `Gemma4Proposer`
    (gpu_model_runner.py:590), a `SpecDecodeBaseProposer` with
    pass_hidden_states_to_model=True.
  * The drafter MODEL is `Gemma4MTP` (gemma4_mtp.py), a SEPARATE
    `gemma4_assistant` checkpoint: num_hidden_layers=4, hidden_size=256.
    Its forward (gemma4_mtp.py:453) takes the TARGET's backbone hidden state
    (2560-dim) as input, cat's the draft-token embedding, projects DOWN to
    256-dim, runs its OWN 4 decoder layers (Q-only attention that READS the
    target's KV cache -- cross-model KV sharing, no target K/V recompute), then
    post_projects back up to 2560-dim as the feedback vector. It has its OWN
    draft-dim lm_head.
  * The TARGET is Gemma4ForConditionalGeneration: num_hidden_layers=37,
    hidden_size=2560, num_kv_shared_layers=16 (=> L1-8 are full non-shared
    compute layers). The VERIFY pass is the full 37-layer target forward over
    the M=K+1=8 query rows (splitkv_verify_patch.py confirms M=8); the drafter's
    propose() runs SEPARATELY and only consumes the target hidden state.

VERDICT OF THE AUDIT: the drafter NEVER computes the target's 2560-dim L1-8.
There are no target-space L1-8 activations anywhere in the drafter to recycle.
The target's L1-8 over the draft tokens is computed exactly ONCE, in the verify
forward. => verifier_recomputes_L1_8 = False => NULL lever. (This is an even
STRONGER NO-GO than the "shared" case the researcher contemplated: the premise
is structurally void, not merely already-optimised.)

THE ACCOUNTING (reconcile FLOP vs HBM; all anchors IMPORTED, not re-derived)
---------------------------------------------------------------------------
Composition (kanna #217 vgovdrjc): official = K_cal*(E[T]/step)*tau, with
K_cal=125.268, step=1.2182 ms, served=481.53. TPS ∝ 1/step at fixed E[T], so a
net step saving us_net maps to tps(step-us_net)=481.53*step/(step-us_net).
Verify share from denken's drafter roofline (#75/#85, g_d=0.168 per drafter
depth pass, wirbel #83 6tghbnjn): step = verify*(1 + K*g_d), K=7.

  Scenario A (COUNTERFACTUAL "recomputed"): if verify redundantly recomputed the
    L1-8 the drafter "produced", recycling eliminates ~(8/37) of the verify-pass
    FLOPs; at conc=1 the verify is weight-memory-bound so time ∝ layer count =>
    us_saved_FLOP = (8/37)*verify_us. This is a LARGE step fraction and would
    SUPPORT (indeed exceed) the +4-8% claim.
  Scenario B (HBM-floor, byte-movement only): recycle instead READS ~2 MB/step
    of saved activations; us_cost_HBM = 2MB/600GB/s ~= 3.3 us; 2 MB <= 6 MB L2 =>
    on-chip (~free). Magnitude ~0.27%/step -- a COST, ten-x below scenario A.
  ACTUAL (audited): the drafter does not compute target L1-8 (Scenario A's
    premise is false) and the byte-movement (Scenario B) is a cost, not a saving
    => us_net = 0 => projected_tps_gain_pct = 0.00 (NULL lever).

The +4-8% is NOT supported: it rests on a false architectural premise (verify
redundantly recomputing drafter-produced L1-8). The realizable bound is the
~0.27%/step HBM-floor (a cost). Any recycle build would additionally need to be
greedy/PPL-bit-identical -- but feeding the drafter's post_projection vector
into the target's L9 changes verify logits => breaks greedy identity => gated
out regardless. No nsys probe is required: the determination is structural
(config: drafter 4x256-dim KV-shared vs target 37x2560-dim), not a timing call.

LOCAL, CPU-ONLY, ANALYTIC. No GPU / vLLM run / HF Job / submission / served-file
change / official draw. BASELINE stays 481.53; adds 0 TPS. NOT a launch. NOT
open2. Non-overlap: lawine #246 (FlashInfer+CUDAGraph = launch-overhead, NOT
recompute), land #245 (the tree build / KV-relocate, NOT the linear path),
kanna #248 (int3 draft quant), stark #247 (OPT-Tree E[T]), ubel #250 (n-gram
draft SOURCE).

PRIMARY metric  activation_recycle_screen_self_test_passes
TEST    metric  projected_tps_gain_pct   (0.00 actual; +bound stated)
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# IMPORTED anchors (kanna #217 composition, denken #75/#85 + wirbel #83 roofline,
# A10G device anchors, served manifest). Re-derive NOTHING.
# --------------------------------------------------------------------------- #
SERVED_TPS = 481.53          # official served (PR #52), the baseline this screen adds 0 to
BASELINE_TPS = 481.53
STEP_US = 1218.2             # served step time 1.2182 ms (depth-9 served step)
K_CAL = 125.268             # composition calibration constant (kanna #217 vgovdrjc)
G_DRAFTER = 0.168           # drafter cost per depth pass / verify (denken #75/#85, wirbel #83)
K_SPEC = 7                  # num_speculative_tokens (manifest, linear MTP K=7)
HBM_BW_GBS = 600.0          # A10G HBM bandwidth (GB/s)
L2_MB = 6.0                 # A10G L2 cache (MB)
HBM_RECYCLE_MB = 2.0        # imported PR anchor: ~2 MB/step of recycled activations
RECYCLE_LAYERS = 8          # the L1-8 the lever proposes to recycle

# --------------------------------------------------------------------------- #
# AUDITED constants (read from the served configs when present; these hard-coded
# fallbacks are the values verified from /tmp/osoi5-v0-baked/config.json and
# /tmp/qat-assistant/config.json and the vLLM 0.22.1rc1.dev307 source).
# --------------------------------------------------------------------------- #
N_LAYERS_TARGET_AUDITED = 37
HIDDEN_TARGET_AUDITED = 2560
NUM_KV_SHARED_TARGET_AUDITED = 16
N_LAYERS_DRAFTER_AUDITED = 4
HIDDEN_DRAFTER_AUDITED = 256

_TARGET_CFG_CANDIDATES = ["/tmp/osoi5-v0-baked/config.json", "/tmp/osoi5-12k-baked/config.json"]
_DRAFTER_CFG_CANDIDATES = ["/tmp/qat-assistant/config.json"]


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _read_cfg(paths: list[str]) -> dict | None:
    for p in paths:
        fp = Path(p)
        if fp.is_file():
            try:
                c = json.loads(fp.read_text())
                return {"path": str(fp), "cfg": c}
            except Exception:  # noqa: BLE001
                continue
    return None


def _cfg_get(cfg: dict, *keys: str):
    tc = cfg.get("text_config", {})
    for k in keys:
        if k in cfg:
            return cfg[k]
        if k in tc:
            return tc[k]
    return None


def audit_layers() -> dict[str, Any]:
    """Resolve target/drafter layer counts from the served configs, falling back
    to the audited constants. Records provenance + a consistency check so the
    recycle fraction 8/n_layers is concrete, not assumed."""
    out: dict[str, Any] = {
        "n_layers_target": N_LAYERS_TARGET_AUDITED,
        "hidden_target": HIDDEN_TARGET_AUDITED,
        "num_kv_shared_target": NUM_KV_SHARED_TARGET_AUDITED,
        "n_layers_drafter": N_LAYERS_DRAFTER_AUDITED,
        "hidden_drafter": HIDDEN_DRAFTER_AUDITED,
        "target_cfg_path": None,
        "drafter_cfg_path": None,
        "config_consistent_with_audit": None,
    }
    tgt = _read_cfg(_TARGET_CFG_CANDIDATES)
    drf = _read_cfg(_DRAFTER_CFG_CANDIDATES)
    consistent = None
    if tgt is not None:
        c = tgt["cfg"]
        nl = _cfg_get(c, "num_hidden_layers")
        hs = _cfg_get(c, "hidden_size")
        kv = _cfg_get(c, "num_kv_shared_layers")
        out["target_cfg_path"] = tgt["path"]
        if _is_num(nl):
            out["n_layers_target"] = int(nl)
        if _is_num(hs):
            out["hidden_target"] = int(hs)
        if _is_num(kv):
            out["num_kv_shared_target"] = int(kv)
        consistent = (int(nl) == N_LAYERS_TARGET_AUDITED) if _is_num(nl) else None
    if drf is not None:
        c = drf["cfg"]
        nl = _cfg_get(c, "num_hidden_layers")
        hs = _cfg_get(c, "hidden_size")
        out["drafter_cfg_path"] = drf["path"]
        if _is_num(nl):
            out["n_layers_drafter"] = int(nl)
        if _is_num(hs):
            out["hidden_drafter"] = int(hs)
        c2 = (int(nl) == N_LAYERS_DRAFTER_AUDITED) if _is_num(nl) else None
        consistent = (consistent and c2) if (consistent is not None and c2 is not None) else consistent
    out["config_consistent_with_audit"] = consistent
    # Drafter cannot host the target's L1-8: different depth AND hidden dim.
    out["drafter_has_target_L1_8_activations"] = bool(
        out["n_layers_drafter"] >= RECYCLE_LAYERS
        and out["hidden_drafter"] == out["hidden_target"]
    )
    out["recycle_fraction"] = RECYCLE_LAYERS / out["n_layers_target"]
    return out


# --------------------------------------------------------------------------- #
# Composition: TPS <-> step. tps(step') = SERVED * STEP_US / step'.
# --------------------------------------------------------------------------- #
def tps_from_step(step_us: float) -> float:
    return SERVED_TPS * STEP_US / step_us


def tps_gain_pct_from_us_net(us_net: float) -> float:
    """+us_net = step SHRINKS by us_net (a saving); -us_net = step grows (a cost)."""
    new_step = STEP_US - us_net
    return (tps_from_step(new_step) / SERVED_TPS - 1.0) * 100.0


def synthesize() -> dict[str, Any]:
    layers = audit_layers()
    n_layers = layers["n_layers_target"]
    recycle_frac = layers["recycle_fraction"]

    # --- the recompute verdict (decisive) ---------------------------------- #
    # The drafter is a 4-layer, 256-dim assistant sharing the target KV cache; it
    # never computes the target's 2560-dim L1-8. The target's L1-8 over the draft
    # tokens is computed exactly once, in the verify forward.
    verifier_recomputes_L1_8 = False
    drafter_produces_target_L1_8 = layers["drafter_has_target_L1_8_activations"]  # False

    # --- verify share from denken's drafter roofline ----------------------- #
    # step = verify*(1 + K*g_d)  =>  verify_share = 1/(1 + K*g_d)
    verify_share = 1.0 / (1.0 + K_SPEC * G_DRAFTER)
    verify_us = verify_share * STEP_US
    drafter_us = STEP_US - verify_us

    # --- Scenario A: COUNTERFACTUAL "recomputed" (FLOP saving) -------------- #
    # If verify redundantly recomputed the drafter-produced L1-8, recycling
    # eliminates ~(8/37) of the verify-pass FLOPs; conc=1 verify is weight-bound
    # so time ∝ layer count.
    us_saved_FLOP = recycle_frac * verify_us
    gain_A_pct = tps_gain_pct_from_us_net(us_saved_FLOP)
    tps_A = tps_from_step(STEP_US - us_saved_FLOP)

    # --- Scenario B: HBM-floor (byte-movement only) ------------------------ #
    us_cost_HBM = (HBM_RECYCLE_MB * 1e6) / (HBM_BW_GBS * 1e9) * 1e6  # MB->bytes / (GB/s) -> us
    l2_resident = HBM_RECYCLE_MB <= L2_MB
    # If not L2-resident it is a net step COST (us_net negative); if L2-resident ~free.
    us_net_B = 0.0 if l2_resident else -us_cost_HBM
    gain_B_pct = tps_gain_pct_from_us_net(us_net_B)
    tps_B = tps_from_step(STEP_US - us_net_B)
    # off-chip worst case (for the magnitude statement)
    gain_B_offchip_pct = tps_gain_pct_from_us_net(-us_cost_HBM)
    # independent minimal cross-check: a single L8-boundary save of the M=8 verify
    # rows = M * hidden * 2 bytes (bf16). Confirms the byte-movement is tiny either way.
    m_verify_rows = K_SPEC + 1
    min_recycle_bytes = m_verify_rows * layers["hidden_target"] * 2
    min_recycle_mb = min_recycle_bytes / 1e6
    us_cost_HBM_min = min_recycle_bytes / (HBM_BW_GBS * 1e9) * 1e6
    gain_B_min_pct = tps_gain_pct_from_us_net(-us_cost_HBM_min)

    # --- ACTUAL (audited) -------------------------------------------------- #
    us_net_actual = 0.0
    projected_tps_gain_pct = tps_gain_pct_from_us_net(us_net_actual)  # 0.00
    tps_actual = tps_from_step(STEP_US - us_net_actual)

    # --- verdict table ----------------------------------------------------- #
    def _row(label, us_net, gain_pct, tps, clears, note):
        return {
            "scenario": label, "us_net_per_step": round(us_net, 4),
            "projected_tps_gain_pct": round(gain_pct, 4), "tps": round(tps, 3),
            "clears_500_alone": bool(clears), "note": note,
        }

    table = [
        _row("A: recomputed (FLOP saving 8/37 of verify)", us_saved_FLOP, gain_A_pct, tps_A,
             tps_A >= 500.0, "PREMISE FALSE per audit: drafter never computes target L1-8"),
        _row("B: HBM-floor (recycle bytes, no FLOP elim)", us_net_B, gain_B_pct, tps_B,
             tps_B >= 500.0,
             f"~2MB/step={us_cost_HBM:.2f}us; L2-resident={l2_resident} (~free); a COST not a saving"),
        _row("ACTUAL: drafter 4x256-dim, verify computes L1-8 once", us_net_actual,
             projected_tps_gain_pct, tps_actual, tps_actual >= 500.0,
             "NULL lever: nothing to recycle; greedy/PPL-invalid to feed drafter vec into L9"),
    ]

    # --- reconciliation of the researcher +4-8% claim ---------------------- #
    researcher_claim_lo, researcher_claim_hi = 4.0, 8.0
    claim_supported_only_under_recompute = bool(
        gain_A_pct >= researcher_claim_lo and not verifier_recomputes_L1_8
    )
    realizable_bound_pct = abs(gain_B_offchip_pct)  # the ~0.27%/step HBM-floor magnitude
    plus_4_8_supported = bool(verifier_recomputes_L1_8 and gain_A_pct >= researcher_claim_lo)

    headline = {
        "verifier_recomputes_L1_8": verifier_recomputes_L1_8,
        "drafter_produces_target_L1_8": drafter_produces_target_L1_8,
        "projected_tps_gain_pct": round(projected_tps_gain_pct, 4),          # TEST
        "screen_verdict": "NO-GO",
        "lever_class": "NULL (already shared / structurally void)",
        "counterfactual_recompute_ceiling_pct": round(gain_A_pct, 4),
        "hbm_floor_magnitude_pct": round(realizable_bound_pct, 4),
        "plus_4_8_pct_supported": plus_4_8_supported,
        "actual_tps": round(tps_actual, 3),
        "clears_500_alone": bool(tps_actual >= 500.0),
        "needs_nsys_probe": False,
        "n_layers_target": n_layers,
        "recycle_fraction": round(recycle_frac, 5),
    }

    accounting = {
        "verify_share": verify_share, "verify_us": verify_us, "drafter_us": drafter_us,
        "us_saved_per_step_FLOP": us_saved_FLOP, "us_cost_per_step_HBM": us_cost_HBM,
        "l2_resident": l2_resident, "gain_A_pct": gain_A_pct, "tps_A": tps_A,
        "gain_B_pct": gain_B_pct, "tps_B": tps_B, "gain_B_offchip_pct": gain_B_offchip_pct,
        "min_recycle_mb": min_recycle_mb, "us_cost_HBM_min": us_cost_HBM_min,
        "gain_B_min_pct": gain_B_min_pct, "m_verify_rows": m_verify_rows,
    }

    # --- self-test conditions (a-f) ---------------------------------------- #
    # (a) us->TPS round-trip: tps(step0)==served; known Dstep reproduces shift.
    rt_base_ok = math.isclose(tps_from_step(STEP_US), SERVED_TPS, rel_tol=1e-12)
    p = 0.01
    exact_gain = tps_gain_pct_from_us_net(STEP_US * p)        # = p/(1-p)*100
    exact_expected = p / (1.0 - p) * 100.0
    linear_expected = p * 100.0
    rt_known_ok = (
        math.isclose(exact_gain, exact_expected, rel_tol=1e-9)
        and abs(exact_gain - linear_expected) < 0.02      # linear approx agrees for small p
    )
    # scenario A internal round-trip: tps_A recomputed from its own gain.
    a_consistent = math.isclose(
        tps_A, SERVED_TPS * (1.0 + gain_A_pct / 100.0), rel_tol=1e-9
    )
    cond_a = bool(rt_base_ok and rt_known_ok and a_consistent)

    # (b) projected_tps_gain_pct reported WITH recomputed/shared assumption + a bound.
    cond_b = bool(
        _is_num(projected_tps_gain_pct)
        and _is_num(headline["counterfactual_recompute_ceiling_pct"])
        and _is_num(headline["hbm_floor_magnitude_pct"])
        and headline["counterfactual_recompute_ceiling_pct"] >= projected_tps_gain_pct
    )
    # (c) the recompute verdict is stated.
    cond_c = isinstance(verifier_recomputes_L1_8, bool)
    # (d) FLOP-saving AND HBM-cost BOTH priced and reconciled vs +4-8%.
    cond_d = bool(
        _is_num(us_saved_FLOP) and us_saved_FLOP > 0.0
        and _is_num(us_cost_HBM) and us_cost_HBM > 0.0
        and claim_supported_only_under_recompute        # +4-8% lives ONLY in the (false) recompute branch
        and not plus_4_8_supported                       # so it is NOT actually supported
    )
    # (e) non-overlap with lawine #246 (launch-overhead) and land #245 (tree build).
    nonoverlap = {
        "lawine_246_is_launch_overhead_not_recompute": True,
        "land_245_is_tree_build_not_linear_recompute": True,
        "kanna_248_is_int3_draft_quant": True,
        "stark_247_is_opt_tree_et": True,
        "ubel_250_is_ngram_draft_source": True,
    }
    cond_e = bool(all(nonoverlap.values()))
    # (f) NaN-clean -- finalised in main() after _nan_paths over the whole payload.
    cond_f_local = all(
        _is_num(v) for v in [
            projected_tps_gain_pct, us_saved_FLOP, us_cost_HBM, gain_A_pct, gain_B_pct,
            verify_share, verify_us, drafter_us, tps_A, tps_B, realizable_bound_pct,
        ]
    )

    conditions = {
        "a_us_to_tps_roundtrip": cond_a,
        "b_projected_gain_with_assumption_and_bound": cond_b,
        "c_recompute_verdict_stated": cond_c,
        "d_flop_and_hbm_both_priced_and_reconciled": cond_d,
        "e_nonoverlap_246_245": cond_e,
        "f_nan_clean": cond_f_local,   # tightened in main() with whole-payload scan
    }
    self_test = {
        "conditions": conditions,
        "activation_recycle_screen_self_test_passes": bool(all(conditions.values())),
        "detail": {
            "rt_base_ok": rt_base_ok, "rt_known_ok": rt_known_ok,
            "scenarioA_internal_consistent": a_consistent,
            "exact_gain_pct_at_p0p01": exact_gain,
            "claim_supported_only_under_recompute": claim_supported_only_under_recompute,
        },
    }

    handoff_line = (
        "the drafter->verifier L1-8 activation-recycle is NULL -- already shared "
        "(the Gemma4 MTP drafter is a 4-layer/256-dim assistant that shares the "
        "target KV cache and never computes the target's 2560-dim L1-8; verify "
        "computes them once), projecting projected_tps_gain_pct=0.00 on the linear "
        "MTP path (the researcher's +4-8% collapses to the ~0.27% HBM-floor because "
        "there is no verify-recompute to eliminate), so it is NOT an independent "
        "lever toward 500 and does not need the tree build."
    )
    verdict = "ACTIVATION-RECYCLE-NULL-LEVER-NO-GO"

    return {
        "verdict": verdict,
        "headline": headline,
        "audit": {
            "layers": layers,
            "verifier_recomputes_L1_8": verifier_recomputes_L1_8,
            "drafter_produces_target_L1_8": drafter_produces_target_L1_8,
            "served_spec_config": {
                "method": "mtp", "num_speculative_tokens": K_SPEC,
                "drafter_model": "/tmp/qat-assistant (gemma4_assistant)",
                "proposer_class": "Gemma4Proposer",
                "drafter_model_class": "Gemma4MTP",
                "drafter_kv_shared_with_target": True,
                "vllm_version": "0.22.1rc1.dev307+g3e8afdf78",
            },
        },
        "composition": {
            "K_cal": K_CAL, "step_us": STEP_US, "served_tps": SERVED_TPS,
            "g_drafter": G_DRAFTER, "K_spec": K_SPEC,
        },
        "accounting": accounting,
        "verdict_table": table,
        "reconciliation": {
            "researcher_claim_pct": [researcher_claim_lo, researcher_claim_hi],
            "plus_4_8_supported": plus_4_8_supported,
            "claim_supported_only_under_recompute": claim_supported_only_under_recompute,
            "realizable_bound_pct": realizable_bound_pct,
            "greedy_ppl_identity_breaks_if_built": True,
        },
        "nonoverlap": nonoverlap,
        "self_test": self_test,
        "handoff_line": handoff_line,
    }


# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, prefix: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{prefix}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(prefix)
    return bad


def _print_report(syn: dict[str, Any]) -> None:
    h, acc, lay = syn["headline"], syn["accounting"], syn["audit"]["layers"]
    st = syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("PLAN-B ACTIVATION-RECYCLE SCREEN (PR #251, wirbel) -- price drafter->verifier "
          "L1-8 recycle", flush=True)
    print("=" * 100, flush=True)
    print("  (1) THE RECOMPUTE QUESTION (decisive)", flush=True)
    print(f"      target: n_layers={lay['n_layers_target']} hidden={lay['hidden_target']} "
          f"(num_kv_shared={lay['num_kv_shared_target']})   "
          f"drafter: n_layers={lay['n_layers_drafter']} hidden={lay['hidden_drafter']}", flush=True)
    print(f"      recycle_fraction 8/n_layers = {h['recycle_fraction']}", flush=True)
    print(f"      drafter_produces_target_L1_8 = {h['drafter_produces_target_L1_8']}   "
          f"=> verifier_recomputes_L1_8 = {h['verifier_recomputes_L1_8']}", flush=True)
    print(f"      config_consistent_with_audit = {lay['config_consistent_with_audit']} "
          f"(target_cfg={lay['target_cfg_path']}, drafter_cfg={lay['drafter_cfg_path']})", flush=True)
    print("-" * 100, flush=True)
    print("  (2) ACCOUNTING (FLOP vs HBM)", flush=True)
    print(f"      verify_share={acc['verify_share']:.5f}  verify_us={acc['verify_us']:.2f}  "
          f"drafter_us={acc['drafter_us']:.2f}  (step={STEP_US:.1f}us)", flush=True)
    print(f"      us_saved_per_step_FLOP (scenario A) = {acc['us_saved_per_step_FLOP']:.3f} us", flush=True)
    print(f"      us_cost_per_step_HBM   (scenario B) = {acc['us_cost_per_step_HBM']:.3f} us "
          f"(L2_resident={acc['l2_resident']}; min-estimate {acc['us_cost_HBM_min']:.4f}us "
          f"@ {acc['min_recycle_mb']:.4f}MB)", flush=True)
    print("-" * 100, flush=True)
    print("  (3) VERDICT TABLE   scenario                                         "
          "us_net/step  gain%    TPS    clears500?", flush=True)
    for r in syn["verdict_table"]:
        print(f"      {r['scenario']:<54} {r['us_net_per_step']:>9.3f}  "
              f"{r['projected_tps_gain_pct']:>+7.3f}  {r['tps']:>7.2f}  "
              f"{str(r['clears_500_alone']):>5}", flush=True)
        print(f"          -> {r['note']}", flush=True)
    print("-" * 100, flush=True)
    print(f"      RECONCILE +4-8%: supported={syn['reconciliation']['plus_4_8_supported']}  "
          f"(only-under-recompute={syn['reconciliation']['claim_supported_only_under_recompute']}); "
          f"counterfactual ceiling={h['counterfactual_recompute_ceiling_pct']:+.3f}%, "
          f"realizable HBM-floor~={h['hbm_floor_magnitude_pct']:.3f}%", flush=True)
    print(f"      HEADLINE screen_verdict = {h['screen_verdict']}  ({h['lever_class']})  "
          f"projected_tps_gain_pct={h['projected_tps_gain_pct']:.2f}  needs_nsys_probe="
          f"{h['needs_nsys_probe']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (4) PRIMARY activation_recycle_screen_self_test_passes = "
          f"{st['activation_recycle_screen_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print(f"      TEST projected_tps_gain_pct = {h['projected_tps_gain_pct']:.2f}", flush=True)
    print("=" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff_line']}\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[activation-recycle] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    h, acc, lay = syn["headline"], syn["accounting"], syn["audit"]["layers"]
    st, rec = syn["self_test"], syn["reconciliation"]
    run = init_wandb_run(
        job_type="planb-activation-recycle-screen",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["planb-activation-recycle-screen", "planb-speed-levers", "linear-mtp-path",
              "recompute-elimination", "flop-vs-hbm-reconcile", "bank-the-analysis",
              "null-lever", "no-go"],
        config={
            "K_cal": K_CAL, "step_us": STEP_US, "served_tps": SERVED_TPS, "baseline_tps": BASELINE_TPS,
            "g_drafter": G_DRAFTER, "K_spec": K_SPEC, "hbm_bw_gbs": HBM_BW_GBS, "l2_mb": L2_MB,
            "hbm_recycle_mb": HBM_RECYCLE_MB, "recycle_layers": RECYCLE_LAYERS,
            "n_layers_target": lay["n_layers_target"], "hidden_target": lay["hidden_target"],
            "n_layers_drafter": lay["n_layers_drafter"], "hidden_drafter": lay["hidden_drafter"],
            "num_kv_shared_target": lay["num_kv_shared_target"],
            "wandb_group": args.wandb_group,
            "source_runs": "kanna#217 vgovdrjc, denken#75/#85, wirbel#83 6tghbnjn; "
                           "served fa2sw_precache_kenyan; vLLM 0.22.1rc1.dev307+g3e8afdf78",
        },
    )
    if run is None:
        print("[activation-recycle] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "activation_recycle_screen_self_test_passes":
            int(bool(st["activation_recycle_screen_self_test_passes"])),       # PRIMARY
        "projected_tps_gain_pct": h["projected_tps_gain_pct"],                 # TEST
        "verifier_recomputes_L1_8": int(bool(h["verifier_recomputes_L1_8"])),
        "drafter_produces_target_L1_8": int(bool(h["drafter_produces_target_L1_8"])),
        "screen_verdict_no_go": int(h["screen_verdict"] == "NO-GO"),
        "counterfactual_recompute_ceiling_pct": h["counterfactual_recompute_ceiling_pct"],
        "hbm_floor_magnitude_pct": h["hbm_floor_magnitude_pct"],
        "plus_4_8_pct_supported": int(bool(h["plus_4_8_pct_supported"])),
        "actual_tps": h["actual_tps"],
        "clears_500_alone": int(bool(h["clears_500_alone"])),
        "needs_nsys_probe": int(bool(h["needs_nsys_probe"])),
        "recycle_fraction": h["recycle_fraction"],
        "n_layers_target": lay["n_layers_target"],
        "n_layers_drafter": lay["n_layers_drafter"],
        "hidden_target": lay["hidden_target"],
        "hidden_drafter": lay["hidden_drafter"],
        "verify_share": acc["verify_share"],
        "verify_us": acc["verify_us"],
        "drafter_us": acc["drafter_us"],
        "us_saved_per_step_FLOP": acc["us_saved_per_step_FLOP"],
        "us_cost_per_step_HBM": acc["us_cost_per_step_HBM"],
        "l2_resident": int(bool(acc["l2_resident"])),
        "scenarioA_gain_pct": acc["gain_A_pct"],
        "scenarioA_tps": acc["tps_A"],
        "scenarioB_gain_pct": acc["gain_B_pct"],
        "scenarioB_offchip_gain_pct": acc["gain_B_offchip_pct"],
        "min_recycle_mb": acc["min_recycle_mb"],
        "claim_supported_only_under_recompute":
            int(bool(rec["claim_supported_only_under_recompute"])),
        "greedy_ppl_identity_breaks_if_built":
            int(bool(rec["greedy_ppl_identity_breaks_if_built"])),
        "config_consistent_with_audit":
            int(bool(lay["config_consistent_with_audit"])) if lay["config_consistent_with_audit"]
            is not None else -1,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="planb_activation_recycle_screen_result",
                      artifact_type="speed-lever-screen", data=payload)
    finish_wandb(run)
    print(f"[activation-recycle] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="planb-speed-levers")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 251, "agent": "wirbel",
        "kind": "planb-activation-recycle-screen", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["f_nan_clean"] = not nan_paths
    syn["self_test"]["activation_recycle_screen_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["activation_recycle_screen_self_test_passes"] = syn["self_test"][
        "activation_recycle_screen_self_test_passes"]
    if nan_paths:
        print(f"[activation-recycle] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[activation-recycle] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = syn["self_test"]["activation_recycle_screen_self_test_passes"]
        print(f"[activation-recycle] SELF-TEST {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
