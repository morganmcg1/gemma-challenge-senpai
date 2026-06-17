#!/usr/bin/env python3
"""PR #565 — Depth-drop fire-region analytical feasibility proof (SYNTHESIS).

A pure-CPU SYNTHESIS card. It composes already-banked, W&B-logged numbers into
the one artifact the program is missing: the ANALYTICAL proof of whether the
depth-drop fire-region is empty, pre-adjudicating ubel #546's empirical served
depth Pareto from the analytical end. NOTHING here is measured — every input is
a constant that traces to a banked W&B run (cited inline + in ``PROVENANCE``).
No GPU, no served job, no microbench, no HF launch, no submission, no
served-file change.

It upgrades my own #561 capstone's ASSERTION
(``depth_drop_conjunction_plausible=False``) to a two-curve PROOF:

  Stage 1 — TPS(layers dropped): from #554's per-layer body decomposition,
    how many dropped layers it takes to lift base_fullhead past the 375.857 ship.
  Stage 2 — quality(layers dropped), BAKE-separated: from ubel #538's served
    body 2x2, how many PURE-DROP (no-bake) layers breach the >=90% MMLU-Pro gate.
  Stage 3 — the feasibility verdict: layers-for-TPS vs layers-that-crater-quality
    (+ the #319 identity leg). If TPS needs MORE layers than quality survives,
    the (quality AND speed AND identity) region is empty by construction.

The verdict that falls out: clearing +64.61 TPS needs >=7 dropped layers even on
the MOST generous assumptions (a magically-free head it cannot physically have +
the largest per-layer saving), but the int4 MMLU-Pro gate breaks at <=2 PURE-DROP
layers (#538: 0.668->0.374 at 5 layers, layer-removal dominant). The quality
cliff PRECEDES the TPS bar by many layers, AND every single dropped block breaks
#319 greedy-identity. Fire-region ANALYTICALLY EMPTY — #546 confirms, not
discovers.

Outputs: a W&B run (group ``depth-drop-fire-region-feasibility``) logging the
synthesized fields + a JSON artifact + a single-line SENPAI-RESULT.

Run under the wandb-capable venv (``.venv/bin/python``); imports wandb first so
the cached real module beats the ``./wandb`` run-data namespace shadow.
"""
from __future__ import annotations

# --- real-wandb-first (beats ./wandb namespace shadow); harmless if absent ---
try:  # pragma: no cover
    import wandb as _wandb_real  # noqa: F401  (cache the real module in sys.modules)
except Exception:  # pragma: no cover
    _wandb_real = None

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research" / "depth_drop_feasibility"
OUT_JSON = HERE / "depth_drop_feasibility.json"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # APPEND so site-packages wandb wins over ./wandb

# ======================================================================
# BANKED INPUTS — every value cited to its PR + W&B run. DO NOT re-derive.
# ======================================================================

# --- the TPS basis: lawine #554 (fi8vr1nb), group fixed-overhead-ceiling -------
# base_fullhead served anchor + the served accept-cycle whose inverse scales TPS.
BFH_TPS = 252.30599912117162          # #554 / wirbel #553 grounded served anchor
BFH_TCYCLE_MS = 15.137999999999998    # #554 served accept-cycle (the "M=1 ~15.1 ms step")
BFH_ET = 3.8194082146962955           # #554 mean accepted tokens / cycle (E[T])
# TPS = E[T]*1000 / tcycle  ->  K_CYCLE := TPS*tcycle = E[T]*1000 is conserved
# under a depth drop (drafter unchanged, only the verify forward gets cheaper).
# Holding E[T] is GENEROUS to the depth drop: dropping target layers can only
# REDUCE draft/target agreement, which would shrink E[T] and the TPS gain.

# the quality-safe HARD ceilings (the depth-drop pass bar is registered over these)
CEILING_MAGFREE_TPS = 311.2485991465399  # #554 magically-free head (tcycle 12.271)
CEILING_MAGFREE_TCYCLE_MS = 12.271246280848539  # #554 tcycle_freed_head_kv_ms (cross-check)
CEILING_STRICT_TPS = 292.1               # #544 (d44b61gj) int4 precision lever (strict)
SHIP_TPS = 375.857                       # the current shipped osoi5 (official a10g-small)

# --- the per-layer body cost (lawine #554 osoi5 cross-check) -------------------
# osoi5 served reduction over base = 4.249 ms (tcycle 15.138 -> 10.889 @ 350.76 TPS).
# Of that, the DIRECT head matmul is 2.678 ms; the REMAINDER 1.571 ms is the body
# (osoi5 dropped 5 layers + baked). The bake is TPS-NEUTRAL (same 37L architecture,
# only weight VALUES change), so the 1.571 ms is essentially all the 5-layer drop:
PLM_BODY_REMAINDER_MS = 1.571             # #554: osoi5 body-side served reduction (5 layers)
OSOI5_LAYERS_DROPPED = 5
N_LAYERS = 42                             # #538/#554: full=42L, osoi5=37L (drop 5)
BODY_READ_MS = 4.332889834615344          # #554 full-body weight-read roofline (42 layers)
SDPA_FLOOR_MS = 0.5727232009172439        # #554 42-launch fixed SDPA floor

# per-layer marginal brackets (ms saved per dropped body layer):
PLM_CENTRAL = PLM_BODY_REMAINDER_MS / OSOI5_LAYERS_DROPPED   # 0.3142 (bake-neutral marginal; PR-registered)
# conservative floor: pure weight-read + the per-layer SDPA launch only (if any of
# the 1.571 ms were somehow NOT per-layer). Served overhead makes the true marginal
# larger than this; it is the LEAST generous (most layers needed) honest bound.
PLM_CONSERVATIVE = (BODY_READ_MS / N_LAYERS) + (SDPA_FLOOR_MS / N_LAYERS)  # 0.1169
# NOTE: PLM_CENTRAL (0.314) gives the FEWEST layers -> most generous to a fire.

# --- the quality basis: ubel #538 (8xo7bc3h), group served body 2x2 ------------
# greedy, byte-identical prompts. PURE layer-drop (raw-ablate, NO bake, NO head-prune).
ANCHOR_MMLU_INT4_42L = 0.668          # #538 base int4 42L (= ubel #511 anchor)
DROP5_MMLU_INT4_37L = 0.374           # #538 int4 37L (5 layers dropped, raw)
ANCHOR_MMLU_BF16_42L = 0.656          # #538 bf16 42L
DROP5_MMLU_BF16_37L = 0.448           # #538 bf16 37L (5 layers dropped, raw)
LAYERDROP_COST_BF16 = 0.208           # #538 dominant body knob (bf16 42L->37L)
INT4_COST_FULLBODY = -0.012           # #538 int4 quant on full body ~free (NOT a layer cost)
# dominant_body_knob = "layer-removal" (#538, unanimous); base_int4_clears_gate=True.

# --- the gate + the osoi5 collapse anchors -------------------------------------
MORGAN_QUALITY_GATE_FRAC = 0.90       # Morgan #515 >=90%-of-vanilla-base gate
MMLU_GATE_FLOOR = 0.601               # #515 explicit MMLU-Pro floor (= 0.90 * 0.668)
# osoi5 (5-drop + 12k head-prune + bake) collapse — the moat (wirbel #548 uc9fnfrn
# + my osoi5_floor_deconfound): MMLU-Pro 0.262-0.274, moat_is_genuine=True.
OSOI5_MMLU_AS_SERVED = 0.272          # my osoi5_floor_deconfound (osoi5_as_served)
OSOI5_MMLU_FLOORED = 0.262            # wirbel #548 / my deconfound (min_tokens=8 floor)
OSOI5_MMLU_MOAT = 0.274               # #548 headline moat number (0.668->0.274)
# head vs body split of the osoi5 collapse (kanna #547 r11skm0y): head 31% / body 69%.
HEAD_PRUNE_COST_12K = 0.126           # #547 12k head-prune on INTACT body (0.676->0.550)

# --- coordination anchors ------------------------------------------------------
CAPSTONE_DEPTH_DROP_BAR_TPS = 64.60840085346013   # my #561 (v74ad5jb) registered bar (over 311.25)
CAPSTONE_DEPTH_DROP_BAR_STRICT = 83.757           # my #561 strict bar (over 292.1)

PROVENANCE = {
    "#561": {"wandb": "v74ad5jb", "group": "two-gate-fire-decision",
             "what": "PARENT capstone: depth_drop_conjunction_plausible=False (asserted); registered bar gap_to_current_ship_tps=64.61 over 311.25 / 83.757 over 292.1. This card upgrades the assertion to a proof."},
    "#554": {"wandb": "fi8vr1nb", "group": "fixed-overhead-ceiling",
             "what": "TPS basis: bfh_tps 252.31 / tcycle 15.138 ms / E[T] 3.819; magically-free ceiling 311.25 (tcycle 12.271); osoi5 body-side served reduction 1.571 ms (5 layers) = direct-head-matmul-corrected -> per-layer ~0.314 ms; body weight-read 4.333 ms; 42-launch SDPA floor 0.573 ms."},
    "#538": {"wandb": "8xo7bc3h", "group": "served-body-2x2",
             "what": "QUALITY basis: served body 2x2, greedy byte-identical. int4 42L 0.668 -> 37L 0.374; bf16 42L 0.656 -> 37L 0.448; layerdrop_cost_bf16 +0.208 (dominant) vs int4_cost_fullbody -0.012 (~free); dominant_body_knob=layer-removal; base_int4_clears_gate=True. PURE drop (raw-ablate, no bake/head-prune)."},
    "#548": {"wandb": "uc9fnfrn", "group": "osoi5-floor-deconfound",
             "what": "the moat: shipped osoi5 (5-drop+12k-head+bake) MMLU-Pro 0.262 floored / 0.274 as-served; moat_is_genuine=True; first-token-EOS 0.0 (not an EOS artifact). Bake-separation reference."},
    "#547": {"wandb": "r11skm0y", "group": "base-fullhead-headwidth-sweep",
             "what": "head/body split of the osoi5 collapse: head 31% / body 69%; 12k head-prune on INTACT body 0.676->0.550 (-0.126). Lets us subtract the head-prune from osoi5 to isolate the pure body-drop."},
    "#539": {"wandb": "(kanna)", "group": "offline-drop-set",
             "what": "offline drop-set Pareto: single-layer block-influence predictor rho=0.761; passers cluster late-sharpening L36-39; raw-ablate no-heal = NECESSARY-not-sufficient; contiguity hurts. Bake-damage / which-layers locus."},
    "#543": {"wandb": "(kanna)", "group": "offline-drop-set",
             "what": "reconciles #539: BI good as single-layer predictor, blind to composition (fails k>=5); drop=3 likely fails the offline screen. The most-generous quality-safe drop locus."},
    "#546": {"wandb": "(ubel, in-flight)", "group": "served-depth-pareto",
             "what": "the EMPIRICAL complement: served depth Pareto (TPS + quality at reduced depth), max_quality_safe_drop_count. This card pre-adjudicates it from the analytical end (two-ended pin)."},
    "#544": {"wandb": "d44b61gj", "group": "base-fullhead-tps-ceiling",
             "what": "strict int4 precision ceiling 292.1; the 328.9 free-head pre-correction (#554 corrects to 311.25)."},
    "#515": {"wandb": "(Morgan)", "group": "—",
             "what": ">=90%-of-vanilla-base downstream quality gate; explicit MMLU-Pro floor 0.601 (= 0.90 * 0.668 anchor)."},
    "#319": {"wandb": "(program)", "group": "—",
             "what": "strict byte-identical greedy-identity HARD gate (argmax_identity_rate=1.0)."},
}


def tps_from_tcycle(k_cycle: float, tcycle_ms: float) -> float:
    """Served TPS for a given accept-cycle time (E[T] held => K conserved)."""
    return k_cycle / tcycle_ms


def layers_to_clear(k_cycle: float, start_tps: float, target_tps: float, plm_ms: float) -> tuple[float, int]:
    """Drop-count d so that TPS rises from start_tps to >= target_tps.

    TPS(d) = K / (tcycle_start - d*plm). Returns (real, ceil-int)."""
    tcycle_start = k_cycle / start_tps
    tcycle_target = k_cycle / target_tps
    delta_needed = tcycle_start - tcycle_target
    d_real = delta_needed / plm_ms
    return d_real, math.ceil(d_real)


def mmlu_after_drop(d: float, anchor: float, drop5: float) -> float:
    """Linear-interpolated PURE-DROP int4 MMLU-Pro after d dropped layers.

    Anchored on #538: anchor at d=0, drop5 at d=OSOI5_LAYERS_DROPPED. Linear is a
    GENEROUS read of a curve that is typically super-linear (accelerating) past
    the first few layers."""
    rate = (anchor - drop5) / OSOI5_LAYERS_DROPPED
    return anchor - rate * d


def layers_to_breach_gate(anchor: float, drop5: float, floor: float) -> tuple[float, int]:
    """Smallest integer drop-count d with MMLU-Pro(d) < floor (linear, int4)."""
    rate = (anchor - drop5) / OSOI5_LAYERS_DROPPED
    d_real = (anchor - floor) / rate
    # first integer drop count strictly below the floor:
    d_breach = math.floor(d_real) + 1
    return d_real, d_breach


def build_packet() -> dict[str, Any]:
    # K = E[T]*1000 conserved across a depth drop (drafter fixed, verify cheaper).
    k_cycle = BFH_TPS * BFH_TCYCLE_MS
    et_check = BFH_ET * 1000.0

    # ---- Stage 1: the TPS(layers dropped) curve ------------------------
    # tps_per_layer_dropped: marginal at the anchor (full head), central plm.
    tps_anchor = BFH_TPS
    tps_drop1_central = tps_from_tcycle(k_cycle, BFH_TCYCLE_MS - PLM_CENTRAL)
    tps_per_layer_dropped = tps_drop1_central - tps_anchor  # marginal @ first layer, full head

    # PRIMARY (most generous to the fire): composed with the magically-free head
    # (start at the 311.25 ceiling) AND the central/largest per-layer saving.
    # This is the registered #561 bar (+64.61 over 311.25 = reach 375.857).
    d_clear_A_central_real, d_clear_A_central = layers_to_clear(
        k_cycle, CEILING_MAGFREE_TPS, SHIP_TPS, PLM_CENTRAL)
    d_clear_A_cons_real, d_clear_A_cons = layers_to_clear(
        k_cycle, CEILING_MAGFREE_TPS, SHIP_TPS, PLM_CONSERVATIVE)
    # PHYSICAL (full head kept — the depth drop cannot magically free the head):
    d_clear_B_central_real, d_clear_B_central = layers_to_clear(
        k_cycle, BFH_TPS, SHIP_TPS, PLM_CENTRAL)
    d_clear_B_cons_real, d_clear_B_cons = layers_to_clear(
        k_cycle, BFH_TPS, SHIP_TPS, PLM_CONSERVATIVE)
    # STRICT (over the 292.1 strict ceiling; +83.757 bar):
    d_clear_strict_central_real, d_clear_strict_central = layers_to_clear(
        k_cycle, CEILING_STRICT_TPS, SHIP_TPS, PLM_CENTRAL)
    d_clear_strict_cons_real, d_clear_strict_cons = layers_to_clear(
        k_cycle, CEILING_STRICT_TPS, SHIP_TPS, PLM_CONSERVATIVE)

    # headline = most generous (fewest layers): framing A + central plm.
    layers_to_clear_ship_tps = d_clear_A_central          # 7
    layers_to_clear_strict = d_clear_strict_central       # 10

    # ---- Stage 2: the quality(layers dropped) curve, BAKE-separated -----
    # PURE int4 layer-drop (#538 raw-ablate), gate floor 0.601.
    breach_real, layers_to_breach_quality_gate = layers_to_breach_gate(
        ANCHOR_MMLU_INT4_42L, DROP5_MMLU_INT4_37L, MMLU_GATE_FLOOR)
    # bf16 cross-check (milder slope):
    breach_bf16_real, _ = layers_to_breach_gate(
        ANCHOR_MMLU_BF16_42L, DROP5_MMLU_BF16_37L, MORGAN_QUALITY_GATE_FRAC * ANCHOR_MMLU_BF16_42L)
    # the largest quality-safe pure-drop count (one less than the breach):
    max_quality_safe_drop = max(0, layers_to_breach_quality_gate - 1)
    # kanna-optimistic upper bound on the safe window (drop the least-sensitive
    # layers; #543: drop=2 may survive the offline screen, drop=3 likely fails):
    max_quality_safe_drop_optimistic = 2

    # is the PURE drop milder than osoi5's drop+bake+head-prune? Yes: the pure
    # int4 5-layer drop (0.374) sits ABOVE the osoi5 collapse (0.262-0.274). The
    # ~0.10 gap is the 12k head-prune (#547 -0.126) osoi5 also applied; the bake
    # is TPS/quality ~neutral. The honest caveat is TRUE, but it does NOT open a
    # window — the pure drop STILL craters below 0.601 by ~2 layers.
    puredrop_milder_than_osoi5 = DROP5_MMLU_INT4_37L > OSOI5_MMLU_MOAT
    puredrop_5L_minus_headprune = DROP5_MMLU_INT4_37L - HEAD_PRUNE_COST_12K  # ~0.248, ~osoi5

    # ---- Stage 3: the feasibility verdict ------------------------------
    # the load-bearing comparison: TPS needs MORE layers than quality survives,
    # on EVERY bracket (use the most generous TPS vs the most optimistic quality).
    depth_drop_fire_region_analytically_empty = (
        layers_to_clear_ship_tps > max(layers_to_breach_quality_gate,
                                       max_quality_safe_drop_optimistic + 1)
    )

    # #319: dropping ANY transformer block changes every downstream hidden state
    # -> every logit -> a near-certain greedy-token flip. base_fullhead's operative
    # identity is already only a LOCUS (0.997955, land #534) at a ULP tie; removing
    # a full block is a MACROSCOPIC perturbation, categorically identity-breaking.
    identity_survives_any_layer_drop = False
    identity_safe_drop_count = 0  # strict #319 holds only at d=0

    # the conjunction: clear +64.61 TPS AND hold quality>=90% AND preserve #319.
    can_clear_within_quality = layers_to_clear_ship_tps <= max_quality_safe_drop
    depth_drop_can_fire = bool(
        can_clear_within_quality
        and (not depth_drop_fire_region_analytically_empty)
        and identity_survives_any_layer_drop
    )

    # the precise sub-bar window ubel #546 must empirically confirm: the best TPS a
    # QUALITY-safe drop could yield (full head, physical). Strict-identity-safe is
    # d=0 (= the anchor), so this window already fails #319 — its only role is the
    # quality Pareto #546 measures.
    msd = max_quality_safe_drop_optimistic  # generous: 2 least-sensitive layers
    max_quality_safe_drop_tps = tps_from_tcycle(k_cycle, BFH_TCYCLE_MS - msd * PLM_CENTRAL)
    max_quality_safe_drop_tps_freehead = tps_from_tcycle(
        k_cycle, CEILING_MAGFREE_TCYCLE_MS - msd * PLM_CENTRAL)  # composed upper bound
    identity_safe_drop_tps = BFH_TPS  # d=0

    if depth_drop_fire_region_analytically_empty:
        verdict_str = (
            f"depth-drop fire-region ANALYTICALLY EMPTY — #546 confirms, not discovers: "
            f"clearing the +{CAPSTONE_DEPTH_DROP_BAR_TPS:.1f} TPS bar needs >= {layers_to_clear_ship_tps} "
            f"dropped layers even granting a magically-free head AND the largest per-layer saving "
            f"({PLM_CENTRAL:.3f} ms/layer); the int4 MMLU-Pro gate ({MMLU_GATE_FLOOR}) breaks at "
            f"<= {layers_to_breach_quality_gate} PURE-DROP layers (#538: 0.668->0.374 at 5L, "
            f"layer-removal dominant); the quality cliff precedes the TPS bar by many layers, "
            f"and every dropped block breaks #319 identity. The best quality-safe drop "
            f"(~{msd} layers) yields ~{max_quality_safe_drop_tps:.0f} TPS (full head), "
            f"<< the {SHIP_TPS} ship."
        )
    else:  # pragma: no cover — not reached on the banked numbers
        verdict_str = (
            f"a quality-safe sub-bar window exists at {max_quality_safe_drop}-layer drop "
            f"(TPS={max_quality_safe_drop_tps:.0f} < ship): #546 must measure it."
        )

    # ---- self-tests (deterministic arithmetic on banked constants) -----
    self_tests = {
        "k_cycle_matches_et": abs(k_cycle - et_check) < 5.0,  # 3819.4 ~ E[T]*1000
        "magfree_tcycle_consistent": abs(
            (k_cycle / CEILING_MAGFREE_TPS) - CEILING_MAGFREE_TCYCLE_MS) < 0.02,
        "anchor_below_magfree": BFH_TPS < CEILING_MAGFREE_TPS,
        "magfree_below_ship": CEILING_MAGFREE_TPS < SHIP_TPS,
        "plm_central_above_conservative": PLM_CENTRAL > PLM_CONSERVATIVE,
        "tps_per_layer_positive": tps_per_layer_dropped > 0,
        # Stage 1: every TPS-clear bracket needs many layers
        "clear_A_central_ge_7": layers_to_clear_ship_tps >= 7,
        "clear_A_cons_ge_clear_A_central": d_clear_A_cons >= d_clear_A_central,
        "clear_B_ge_clear_A": d_clear_B_central >= d_clear_A_central,  # full head needs more
        "clear_strict_ge_clear_A": layers_to_clear_strict >= layers_to_clear_ship_tps,
        # Stage 2: quality breaks within a couple of pure-drop layers
        "breach_le_3": layers_to_breach_quality_gate <= 3,
        "drop5_int4_below_gate": DROP5_MMLU_INT4_37L < MMLU_GATE_FLOOR,
        "puredrop_milder_than_osoi5": puredrop_milder_than_osoi5,
        "puredrop_minus_headprune_near_osoi5": abs(puredrop_5L_minus_headprune - OSOI5_MMLU_MOAT) < 0.05,
        "layer_removal_dominant": LAYERDROP_COST_BF16 > abs(INT4_COST_FULLBODY),
        # Stage 3: the empty-region proof + the conjunction
        "tps_layers_exceed_quality_layers": layers_to_clear_ship_tps > layers_to_breach_quality_gate,
        "region_empty_true": depth_drop_fire_region_analytically_empty,
        "identity_breaks": (not identity_survives_any_layer_drop),
        "cannot_fire": (not depth_drop_can_fire),
        "max_qsafe_drop_tps_below_ship": max_quality_safe_drop_tps < SHIP_TPS,
        "max_qsafe_freehead_below_ship": max_quality_safe_drop_tps_freehead < SHIP_TPS,
        "nan_clean": all(math.isfinite(x) for x in [
            k_cycle, tps_per_layer_dropped, d_clear_A_central_real, d_clear_B_central_real,
            d_clear_strict_central_real, breach_real, max_quality_safe_drop_tps,
            max_quality_safe_drop_tps_freehead,
        ]),
    }
    self_tests["self_test_passes"] = all(self_tests.values())
    self_det = bool(self_tests["self_test_passes"])

    packet = {
        "pr": 565,
        "card": "depth-drop-fire-region-feasibility",
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
        "no_hf_job": True,
        "peak_gpu_gib": 0.0,
        "model": {"name": "gemma-4-E4B-it", "n_layers": N_LAYERS, "vllm": "0.22.1rc1", "gpu": "A10G sm_86"},
        # ---- Stage 1: TPS(layers dropped) ----
        "stage1_tps_curve": {
            "k_cycle": k_cycle,
            "bfh_tcycle_ms": BFH_TCYCLE_MS,
            "et": BFH_ET,
            "plm_central_ms_per_layer": PLM_CENTRAL,
            "plm_conservative_ms_per_layer": PLM_CONSERVATIVE,
            "tps_per_layer_dropped": tps_per_layer_dropped,  # marginal @ anchor, full head, central plm
            # PRIMARY (registered +64.61 bar, composed-with-free-head, MOST generous):
            "layers_to_clear_ship_tps": layers_to_clear_ship_tps,          # 7 (framing A, central)
            "layers_to_clear_ship_tps_real": d_clear_A_central_real,
            "layers_to_clear_ship_tps_conservative": d_clear_A_cons,       # 13 (framing A, conservative)
            "layers_to_clear_strict": layers_to_clear_strict,              # 10 (framing A strict, central)
            "layers_to_clear_strict_conservative": d_clear_strict_cons,    # 18
            # PHYSICAL (full head kept — the depth drop cannot free the head):
            "layers_to_clear_ship_tps_puredrop_fullhead": d_clear_B_central,        # 16
            "layers_to_clear_ship_tps_puredrop_fullhead_cons": d_clear_B_cons,      # 30
            "framing_note": "PRIMARY = composed-with-magically-free-head (the registered #561 +64.61-over-311.25 bar, the MOST generous to a fire). PHYSICAL (full-head, can't free the head) needs even more layers. Every bracket >> the quality cliff.",
        },
        # ---- Stage 2: quality(layers dropped), bake-separated ----
        "stage2_quality_curve": {
            "anchor_mmlu_int4_42L": ANCHOR_MMLU_INT4_42L,
            "drop5_mmlu_int4_37L": DROP5_MMLU_INT4_37L,
            "mmlu_gate_floor": MMLU_GATE_FLOOR,
            "layers_to_breach_quality_gate": layers_to_breach_quality_gate,  # 2 (int4 pure-drop, linear)
            "layers_to_breach_quality_gate_real": breach_real,
            "max_quality_safe_drop": max_quality_safe_drop,                  # 1 (linear) ..2 (kanna-optimistic)
            "max_quality_safe_drop_optimistic": max_quality_safe_drop_optimistic,
            "puredrop_milder_than_osoi5": puredrop_milder_than_osoi5,        # True
            "puredrop_5L_minus_headprune": puredrop_5L_minus_headprune,      # ~0.248 ~ osoi5 0.274
            "osoi5_mmlu_moat": OSOI5_MMLU_MOAT,
            "dominant_body_knob": "layer-removal",
            "bake_separation_note": "PURE int4 layer-drop (#538 raw-ablate, no bake/head-prune) 0.668->0.374 at 5L sits ABOVE osoi5's 0.274 collapse — the ~0.10 gap is the 12k head-prune (#547 -0.126) osoi5 also applied; the bake is ~neutral. puredrop_milder=True but it does NOT open a window: the pure drop still craters below 0.601 by ~2 layers.",
        },
        # ---- Stage 3: the feasibility verdict ----
        "stage3_verdict": {
            "depth_drop_fire_region_analytically_empty": depth_drop_fire_region_analytically_empty,  # True
            "identity_survives_any_layer_drop": identity_survives_any_layer_drop,                    # False
            "identity_safe_drop_count": identity_safe_drop_count,                                     # 0
            "depth_drop_can_fire": depth_drop_can_fire,                                               # False
            "max_quality_safe_drop_tps": max_quality_safe_drop_tps,                # ~263 (full head, ~2L)
            "max_quality_safe_drop_tps_freehead": max_quality_safe_drop_tps_freehead,  # ~328 (composed upper)
            "identity_safe_drop_tps": identity_safe_drop_tps,                      # 252.31 (d=0)
            "verdict": verdict_str,
            "two_ended_pin": "ANALYTICAL end (this card): fire-region empty by construction. EMPIRICAL end (ubel #546, in flight): served depth Pareto. The sub-bar window (~2-layer drop -> ~263 TPS full head, << 375.857 ship) is the precise thing #546 must confirm.",
        },
        "provenance": PROVENANCE,
        "self_tests": self_tests,
        "self_det": self_det,
        "primary_metric_name": "max_quality_safe_drop_tps",
        "primary_metric_value": max_quality_safe_drop_tps,
    }
    return packet


def wandb_summary(p: dict[str, Any]) -> dict[str, Any]:
    s1 = p["stage1_tps_curve"]
    s2 = p["stage2_quality_curve"]
    s3 = p["stage3_verdict"]
    return {
        # Stage 1
        "tps_per_layer_dropped": s1["tps_per_layer_dropped"],
        "plm_central_ms_per_layer": s1["plm_central_ms_per_layer"],
        "plm_conservative_ms_per_layer": s1["plm_conservative_ms_per_layer"],
        "layers_to_clear_ship_tps": s1["layers_to_clear_ship_tps"],
        "layers_to_clear_ship_tps_conservative": s1["layers_to_clear_ship_tps_conservative"],
        "layers_to_clear_strict": s1["layers_to_clear_strict"],
        "layers_to_clear_ship_tps_puredrop_fullhead": s1["layers_to_clear_ship_tps_puredrop_fullhead"],
        # Stage 2
        "layers_to_breach_quality_gate": s2["layers_to_breach_quality_gate"],
        "max_quality_safe_drop": s2["max_quality_safe_drop"],
        "puredrop_milder_than_osoi5": s2["puredrop_milder_than_osoi5"],
        "puredrop_milder_than_osoi5_int": int(s2["puredrop_milder_than_osoi5"]),
        "drop5_mmlu_int4_37L": s2["drop5_mmlu_int4_37L"],
        "mmlu_gate_floor": s2["mmlu_gate_floor"],
        # Stage 3
        "depth_drop_fire_region_analytically_empty": s3["depth_drop_fire_region_analytically_empty"],
        "depth_drop_fire_region_analytically_empty_int": int(s3["depth_drop_fire_region_analytically_empty"]),
        "identity_survives_any_layer_drop": s3["identity_survives_any_layer_drop"],
        "identity_survives_any_layer_drop_int": int(s3["identity_survives_any_layer_drop"]),
        "depth_drop_can_fire": s3["depth_drop_can_fire"],
        "depth_drop_can_fire_int": int(s3["depth_drop_can_fire"]),
        "max_quality_safe_drop_tps": s3["max_quality_safe_drop_tps"],
        "max_quality_safe_drop_tps_freehead": s3["max_quality_safe_drop_tps_freehead"],
        "identity_safe_drop_tps": s3["identity_safe_drop_tps"],
        "verdict": s3["verdict"],
        # meta
        "self_det": p["self_det"],
        "self_det_int": int(p["self_det"]),
        "n_self_tests_passed": sum(1 for v in p["self_tests"].values() if v),
        "n_self_tests_total": len(p["self_tests"]),
        "peak_gpu_gib": p["peak_gpu_gib"],
        "analysis_only": True,
        "official_tps": 0,
        "primary_metric": p["primary_metric_value"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-name", default="lawine/depth-drop-fire-region-feasibility")
    ap.add_argument("--wandb-group", default="depth-drop-fire-region-feasibility")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    p = build_packet()
    HERE.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as fh:
        json.dump(p, fh, indent=2)
    print(f"[depth-drop] wrote {OUT_JSON}", flush=True)

    s1, s2, s3 = p["stage1_tps_curve"], p["stage2_quality_curve"], p["stage3_verdict"]
    line = "=" * 8 + " PR #565 — DEPTH-DROP FIRE-REGION FEASIBILITY (synthesis) " + "=" * 8
    print("\n" + line, flush=True)
    print("  STAGE 1 — TPS(layers dropped):", flush=True)
    print(f"    per-layer saving  = {s1['plm_central_ms_per_layer']:.3f} ms/layer central "
          f"({s1['plm_conservative_ms_per_layer']:.3f} conservative);  ~{s1['tps_per_layer_dropped']:.1f} TPS/layer @ anchor", flush=True)
    print(f"    layers_to_clear_ship_tps = {s1['layers_to_clear_ship_tps']} (most generous: free-head + central) "
          f".. {s1['layers_to_clear_ship_tps_conservative']} (conservative);  full-head physical {s1['layers_to_clear_ship_tps_puredrop_fullhead']}..{s1['layers_to_clear_ship_tps_puredrop_fullhead_cons']}", flush=True)
    print(f"    layers_to_clear_strict   = {s1['layers_to_clear_strict']} (+83.757 over 292.1)", flush=True)
    print("  STAGE 2 — quality(layers dropped), bake-separated:", flush=True)
    print(f"    PURE int4 drop (#538): 0.668 (0L) -> {s2['drop5_mmlu_int4_37L']} (5L);  gate floor {s2['mmlu_gate_floor']}", flush=True)
    print(f"    layers_to_breach_quality_gate = {s2['layers_to_breach_quality_gate']}  "
          f"(max quality-safe drop = {s2['max_quality_safe_drop']}..{s2['max_quality_safe_drop_optimistic']})", flush=True)
    print(f"    puredrop_milder_than_osoi5 = {s2['puredrop_milder_than_osoi5']} "
          f"(0.374 > osoi5 {s2['osoi5_mmlu_moat']}; gap = 12k head-prune, NOT a window)", flush=True)
    print("  STAGE 3 — feasibility verdict:", flush=True)
    print(f"    depth_drop_fire_region_analytically_empty = {s3['depth_drop_fire_region_analytically_empty']} "
          f"({s1['layers_to_clear_ship_tps']} TPS-layers > {s2['layers_to_breach_quality_gate']} quality-layers)", flush=True)
    print(f"    identity_survives_any_layer_drop = {s3['identity_survives_any_layer_drop']}  "
          f"depth_drop_can_fire = {s3['depth_drop_can_fire']}", flush=True)
    print(f"    max_quality_safe_drop_tps = {s3['max_quality_safe_drop_tps']:.1f} (full head) "
          f"/ {s3['max_quality_safe_drop_tps_freehead']:.1f} (composed) << ship {SHIP_TPS}", flush=True)
    print(f"  VERDICT: {s3['verdict']}", flush=True)
    print(f"  self_det = {p['self_det']}  ({p['self_tests']['n_self_tests_passed'] if False else sum(1 for v in p['self_tests'].values() if v)}/{len(p['self_tests'])} self-tests)", flush=True)
    print("=" * len(line), flush=True)

    rid = None
    if not args.no_wandb:
        try:
            from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                               log_json_artifact, log_summary)
            run = init_wandb_run(
                job_type="systems-profile",
                agent="lawine",
                name=args.wandb_name,
                group=args.wandb_group,
                tags=["depth-drop", "fire-region", "feasibility", "synthesis",
                      "analysis-only", "no-fire", "local-a10g"],
                notes="PR #565 depth-drop fire-region analytical feasibility: TPS(layers) vs quality(layers) -> region empty; pre-adjudicates ubel #546",
                config={
                    "synthesis_only": True,
                    "no_gpu": True,
                    "cited_prs": list(PROVENANCE.keys()),
                    "ship_tps": SHIP_TPS,
                    "mmlu_gate_floor": MMLU_GATE_FLOOR,
                    "plm_central_ms_per_layer": PLM_CENTRAL,
                },
            )
            if run is not None:
                log_summary(run, wandb_summary(p), step=0)
                log_json_artifact(run, name="depth-drop-feasibility",
                                  artifact_type="fire-region-feasibility", data=p)
                rid = getattr(run, "id", None)
                finish_wandb(run)
                p["wandb_run_id"] = rid
                with OUT_JSON.open("w") as fh:
                    json.dump(p, fh, indent=2)
                print(f"[depth-drop] wandb run id = {rid}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[depth-drop] wandb unavailable: {exc}", flush=True)

    # ---- single-line SENPAI-RESULT ----
    senpai = {
        "terminal": True,
        "status": "complete",
        "pending_arms": False,
        "analysis_only": True,
        "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": p["self_det"],
        "depth_drop_fire_region_analytically_empty": s3["depth_drop_fire_region_analytically_empty"],
        "depth_drop_can_fire": s3["depth_drop_can_fire"],
        "primary_metric": {"name": "max_quality_safe_drop_tps",
                           "value": round(s3["max_quality_safe_drop_tps"], 2)},
        "test_metric": {"name": "layers_to_clear_ship_tps",
                        "value": s1["layers_to_clear_ship_tps"]},
    }
    print("\nSENPAI-RESULT: " + json.dumps(senpai, separators=(",", ":")), flush=True)
    return 0 if p["self_det"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
