#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Bridge basis-honesty: re-price the banked STEP levers to ONE coherent basis (PR #286).

THE QUESTION
------------
denken #278 (`bu44n30q`) proved the served `step = 1218.2us` is a NORMALIZED
(batch+acceptance amortized) unit, NOT a wall draft+verify+overhead sum: the M=1
linear verify ALONE is 4966.8us (its int4-body HBM read floor is 2934us, already
> the whole 1218.2us step). So a batch=1 WALL draft saving subtracted directly
from the normalized step OVER-CREDITS by 1/bridge ~ 4.82x, where

    bridge = step_norm / step_wall_microbuilt
           = 1218.2 / (draft_k7_wall 706.86 + verify_m1_wall 4966.78)
           = 1218.2 / 5673.64  =  0.21471.

denken re-priced ONE lever through this (kanna #269 fold: +4.39% -> +0.91%) and
flagged the OPEN question (fu#2): is bridge ~ 0.21 UNIVERSAL across all step
levers, or DRAFT-SIDE-SPECIFIC?

THE RESOLUTION (this leg)
-------------------------
The bridge is DRAFT-SIDE-SPECIFIC. Classify every banked step lever by the
MEASUREMENT BASIS of its saving:

  * draft-side, batch=1 WALL  -> the saving is an ABSOLUTE batch=1 GPU-op saving
    subtracted from the normalized step. It needs the bridge ~ 0.2147 discount.
    Levers: kanna #269 (MLP GeluAndMul fold), kanna #277 (io_projection, NULL),
            wirbel #270 (draft/tree attention autotune).
  * verify-side, deployed M=8 -> the saving was measured ON the deployed batch-8
    verify path and priced as a FRACTIONAL reduction of the verify-dominated
    step (step_norm 1218.2 ~ per-seq batch-8 verify 1129.6). It is ALREADY in the
    normalized basis -> bridge ~ 1.0 (no further discount).
    Levers: wirbel #279 (verify SDPA num_stages=2, deployed 3D split-KV),
            kanna #280 (the SAME SDPA lever, priced via the verify decomposition).

Re-price each lever to its basis-honest TPS, produce the consolidated portfolio
card, compose the disjoint basis-honest step stack, and reconcile denken #278's
bridge (linear-path batch-normalization) with fern #274's phi=0.603 (tree-path /
wall-clock fixed-overhead absorption) into ONE coherent basis.

Pure CPU analytic over banked W&B numbers (all imported VERBATIM). Analysis-only;
BASELINE 481.53 untouched (this RE-PRICES banked measurements; adds 0 TPS). NOT a
launch; no served-file change; no HF Job; no submission."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                      # .../target

# --------------------------------------------------------------------------- #
# Banked composition anchors (imported VERBATIM; never re-derived).
# --------------------------------------------------------------------------- #
OFFICIAL_BASELINE = 481.53                       # PR #52 official frontier TPS
LAMBDA1_CEIL = 520.9527323111674                 # #257 lambda=1 ceiling
K_CAL = 125.26795005202914                       # #257 K_cal
STEP_SERVED_US = 1218.2                           # kanna #217 served step (NORMALIZED unit)
E_T = 3.844                                       # PR #278 K=7-linear E[T] (literal split)
TAU = 1.218                                       # PR #278 composition round-trip tau

# ---- denken #278 bu44n30q: the bridge derivation (measured wall components) ----
D278_DRAFT_K7_WALL_US = 706.8555014474051        # K=7 draft chain, batch=1 wall (graphed)
D278_VERIFY_M1_WALL_US = 4966.783229282924       # deployed LINEAR M=1 verify, batch=1 wall
D278_STEP_WALL_MICROBUILT_US = D278_DRAFT_K7_WALL_US + D278_VERIFY_M1_WALL_US  # 5673.6387
D278_BRIDGE = STEP_SERVED_US / D278_STEP_WALL_MICROBUILT_US                    # 0.2147122962556323
D278_VERIFY_B8_TOTAL_US = 9036.682913643974      # deployed M=8 verify total (8 seqs)
D278_VERIFY_B8_PER_SEQ_US = 1129.5853642054967   # / 8 ~ step_norm  -> verify-side bridge ~ 1.0
D278_B8_OVER_B1_RATIO = 1.8194236584286443       # verify b8-total / b1  -> per-seq amort 1.819/8
# denken's own banked re-pricing of kanna #269 (the cross-check target for self-test a):
D278_MODEL_A_GAIN_PCT = 4.390896003290608        # raw (subtract WALL saving from NORM step)
D278_MODEL_B_GAIN_PCT = 0.9113547874137429       # honest (bridge the saving first)
D278_OVERCREDIT_FACTOR = 4.817987532332126       # model_A / model_B

# ---- fern #274 brnmnl60: the phi (tree-path / wall-clock fixed-overhead) discount ----
F274_PHI_SERVED_WALL = 0.6034589848288221        # model_forward / served_wall_clock_est (the "0.603")
F274_PHI_KCAL_CLEAN = 0.735133735318471          # model_forward / kcal_clean wall
F274_MODEL_FORWARD_US = 5868.490184545517        # draft 704.78 + verify_m8 5163.71 (fern's wall basis)
F274_SERVED_WALL_CLOCK_US = 9724.754013249434    # fern's wall step convention (incl fixed overhead)

# --------------------------------------------------------------------------- #
# The lever portfolio (every banked step-shaving lever). Each saving is recorded
# in the STEP basis that produced its banked raw composition gain:
#   * draft-side  -> delta_step_raw_us is the batch=1 WALL chain saving that was
#                    subtracted from step_norm (needs the bridge).
#   * verify-side -> delta_step_raw_us is the step-normalized saving already on the
#                    deployed-M8 fractional basis (bridge ~ 1.0; raw == honest).
# delta_step_raw_us reproduces the lever's banked raw gain via tps = base*step/(step-delta).
# --------------------------------------------------------------------------- #
def _delta_from_gain(gain_pct: float) -> float:
    """Step-basis saving that reproduces a composition gain% (verify-side back-out)."""
    return STEP_SERVED_US * (1.0 - 1.0 / (1.0 + gain_pct / 100.0))


LEVERS: list[dict[str, Any]] = [
    {
        "lever": "draft_mlp_activation_fold",
        "pr": 269, "run": "epl52mkq",
        "basis": "draft_b1_wall",
        "disjoint_group": "draft_mlp",
        "stack_member": True,
        "delta_step_raw_us": 51.24,                  # GeluAndMul fold, 7.32us/pass x K=7 chain, b1 wall
        "banked_raw_gain_pct": 4.390896003290608,    # == denken #278 model_A
        "same_lever_as": None,
        "justification": (
            "kanna #269 folds the SEPARATE GeluAndMul companion kernel into the draft MLP "
            "gate_up epilogue. The 7.32us/pass saving is a batch=1 WALL draft-GPU-op saving "
            "(the drafter runs the K=7 chain), subtracted as an ABSOLUTE 51.24us from the "
            "NORMALIZED step -> draft-side, needs the bridge."),
    },
    {
        "lever": "draft_io_projection",
        "pr": 277, "run": "ahw089yi",
        "basis": "draft_b1_wall",
        "disjoint_group": "draft_io",
        "stack_member": True,
        "delta_step_raw_us": 0.06,                   # recoverable_io_us (1 scheduling gap; immaterial)
        "banked_raw_gain_pct": 0.0,                  # NULL (recoverable_material=False)
        "same_lever_as": None,
        "justification": (
            "kanna #277 io_projection = 2 GEMVs at opposite ends of the 4-layer draft stack, "
            "intrinsic-M=1, NO companion kernel to fold -> NULL (0.0%). Measured at the batch=1 "
            "draft wall like the MLP -> draft-side. (Bridge irrelevant: saving is ~0.)"),
    },
    {
        "lever": "draft_tree_attention_autotune",
        "pr": 270, "run": "iwwcmvez",
        "basis": "draft_b1_wall",
        "disjoint_group": "draft_attn",
        "stack_member": True,
        "delta_step_raw_us": 15.411217212677002,     # draft_step_saving_us_bitident, 2.20us/pass x7, b1 wall
        "banked_raw_gain_pct": 1.281290400544255,    # upper bound (eager)
        "onegraph_residual_gain_pct": 0.05564302041867197,  # in-graph realistic (~null)
        "same_lever_as": None,
        "justification": (
            "wirbel #270 autotunes the draft/tree attention Triton kernel (28.5us attn term). The "
            "2.20us/pass bit-identical saving (x K=7 = 15.41us) is a batch=1 WALL draft saving -> "
            "draft-side, needs the bridge. NOTE: the realistic ONEGRAPH in-graph residual is only "
            "+0.056% (the 1.281% is an eager upper bound; ONEGRAPH already erased the launch slack)."),
    },
    {
        "lever": "verify_sdpa_num_stages2_deployed",
        "pr": 279, "run": "xme9snkv",
        "basis": "verify_deployed_m8",
        "disjoint_group": "verify_sdpa",
        "stack_member": True,
        "delta_step_raw_us": 15.554561614990178,     # verify_sdpa_saving_us (ALREADY step-normalized)
        "banked_raw_gain_pct": 1.2933622095534503,   # wirbel labels it "honest_projected_tps_after" 487.76
        "same_lever_as": None,
        "justification": (
            "wirbel #279 measures the verify SDPA num_stages=3->2 tune ON the deployed 3D split-KV "
            "M=8 path (realistic ctx 512), and prices it as a FRACTIONAL reduction of the verify-"
            "dominated step (saving expressed directly as a 15.55us step saving) -> ALREADY in the "
            "normalized deployed-M8 basis -> verify-side, bridge ~ 1.0 (no further discount)."),
    },
    {
        "lever": "verify_sdpa_num_stages2_component",
        "pr": 280, "run": "sdrerk5h",
        "basis": "verify_deployed_m8",
        "disjoint_group": "verify_sdpa",
        "stack_member": False,                       # SAME physical lever as #279 -> not double-counted
        "delta_step_raw_us": _delta_from_gain(1.1853573814381013),  # back-out from banked gain (14.27us)
        "banked_raw_gain_pct": 1.1853573814381013,
        "same_lever_as": 279,
        "justification": (
            "kanna #280 prices the SAME verify SDPA num_stages=2 lever via the verify-component "
            "decomposition: SDPA is 14.51% of the M=8 verify; num_stages=2 gives 1.097x on SDPA -> "
            "66.27us verify saving -> +1.185% composition. Deployed-M8 basis -> bridge ~ 1.0. It is "
            "the SAME physical kernel tune as wirbel #279 (verify SDPA num_stages=2) -> EXCLUDED "
            "from the composed stack to avoid double-counting."),
    },
]


# --------------------------------------------------------------------------- #
# Analytic core.
# --------------------------------------------------------------------------- #
def tps_at(new_step_us: float) -> float:
    """Composition TPS at a new normalized step (round-trips OFFICIAL_BASELINE at step_norm)."""
    return OFFICIAL_BASELINE * STEP_SERVED_US / new_step_us


def bridge_for(basis: str) -> float:
    if basis == "draft_b1_wall":
        return D278_BRIDGE                      # 0.2147 (batch=1 wall -> normalized step)
    if basis == "verify_deployed_m8":
        return 1.0                              # already deployed-M8 (per-seq) normalized basis
    raise ValueError(f"unknown basis {basis!r}")


def reprice_lever(lev: dict[str, Any]) -> dict[str, Any]:
    basis = lev["basis"]
    bridge = bridge_for(basis)
    d_raw = float(lev["delta_step_raw_us"])

    # raw (as banked): subtract the raw step saving directly from step_norm.
    new_step_raw = STEP_SERVED_US - d_raw
    tps_raw = tps_at(new_step_raw)
    raw_gain_pct = (tps_raw / OFFICIAL_BASELINE - 1.0) * 100.0

    # basis-honest: bridge the saving into the normalized basis first.
    d_honest = bridge * d_raw
    new_step_honest = STEP_SERVED_US - d_honest
    tps_honest = tps_at(new_step_honest)
    honest_gain_pct = (tps_honest / OFFICIAL_BASELINE - 1.0) * 100.0

    return {
        "lever": lev["lever"], "pr": lev["pr"], "run": lev["run"],
        "basis": basis, "bridge": bridge,
        "disjoint_group": lev["disjoint_group"],
        "stack_member": lev["stack_member"], "same_lever_as": lev.get("same_lever_as"),
        "delta_step_raw_us": d_raw, "delta_step_honest_us": d_honest,
        "raw_composition_gain_pct": raw_gain_pct,
        "banked_raw_gain_pct": lev["banked_raw_gain_pct"],
        "basis_honest_gain_pct": honest_gain_pct,
        "basis_honest_tps": tps_honest,
        "crosses_500_alone": bool(tps_honest >= 500.0),
        "onegraph_residual_gain_pct": lev.get("onegraph_residual_gain_pct"),
        "justification": lev["justification"],
    }


def synthesize() -> dict[str, Any]:
    # ---- (0) bridge derivation + per-basis proof ----
    # draft bridge: re-derive from the measured wall components (== denken #278).
    bridge_draft = D278_BRIDGE
    one_over_bridge = 1.0 / bridge_draft
    # verify bridge ~ 1.0 PROOF: the per-seq batch-8 verify ~ the normalized step, and the
    # b8-per-seq amortization factor ~ the draft bridge (both ~0.21..0.23), so a deployed-M8
    # verify saving priced fractionally is ALREADY in the normalized basis.
    verify_b8_per_seq = D278_VERIFY_B8_PER_SEQ_US
    per_seq_amort_factor = D278_B8_OVER_B1_RATIO / 8.0              # 1.819/8 = 0.2274
    step_vs_perseq_verify_ratio = STEP_SERVED_US / verify_b8_per_seq  # 1218.2/1129.6 = 1.078
    bridge_vs_amort_resid = abs(bridge_draft - per_seq_amort_factor)  # ~0.013 (bridge ~ per-seq amort)
    verify_bridge = 1.0

    # ---- (1)+(3) classify + re-price every lever -> the consolidated card ----
    card = [reprice_lever(l) for l in LEVERS]

    # ---- (5)-best: best single basis-honest lever ----
    best = max(card, key=lambda r: r["basis_honest_tps"])
    best_basis_honest_step_tps = best["basis_honest_tps"]

    # ---- (4) compose the DISJOINT basis-honest step stack ----
    # disjoint step sub-components (MLP companion kernel / io GEMVs / attn kernel / verify SDPA);
    # sum the honest step savings over the stack members (one verify SDPA row only).
    stack_rows = [r for r in card if r["stack_member"]]
    groups_used = sorted({r["disjoint_group"] for r in stack_rows})
    sum_delta_honest = sum(r["delta_step_honest_us"] for r in stack_rows)
    composed_new_step = STEP_SERVED_US - sum_delta_honest
    composed_tps = tps_at(composed_new_step)
    composed_gain_pct = (composed_tps / OFFICIAL_BASELINE - 1.0) * 100.0
    composed_step_stack_crosses_500 = bool(composed_tps >= 500.0)
    # headroom to the lambda=1 ceiling and to 500
    composed_to_500_gap = 500.0 - composed_tps
    composed_to_ceiling_gap = LAMBDA1_CEIL - composed_tps

    # ---- (2) draft-bridge caveat (batch-8 draft amortization not separately measured) ----
    draft_bridge_caveat = {
        "assumption": "carry denken #278's bridge=0.2147 (= step_norm/(draft_b1+verify_b1)) for "
                      "ALL draft-side levers.",
        "verify_side_confirmed": (
            f"the deployed-M8 verify per-seq cost {verify_b8_per_seq:.1f}us ~ step_norm "
            f"{STEP_SERVED_US:.1f}us (ratio {step_vs_perseq_verify_ratio:.3f}), and the b8-per-seq "
            f"amortization factor {per_seq_amort_factor:.4f} ~ bridge {bridge_draft:.4f} "
            f"(resid {bridge_vs_amort_resid:.4f}) -> a deployed-M8 verify saving is already "
            f"normalized -> verify bridge = 1.0 (banked-confirmed)."),
        "draft_side_open": (
            "the batch-8 DRAFT-chain per-seq amortization is NOT separately measured (denken #278 "
            "fu#1; denken #283 HBM-ceiling not yet landed on this branch). The draft GEMVs are "
            "intrinsic-M=1 under-saturated (kanna #269/#277): at batch=8 they would amortize the "
            "weight reads, so the TRUE draft per-seq saving is <= the bridged saving."),
        "bound_direction": (
            "if the draft amortizes MORE than verify, the draft bridge < 0.2147 -> the honest draft "
            "gain is EVEN SMALLER -> the draft-side card rows (and the composed stack) are UPPER "
            "BOUNDS. The closure (no step lever clears 500) is therefore robust / conservative."),
        "verify_b8_per_seq_us": verify_b8_per_seq,
        "per_seq_amort_factor": per_seq_amort_factor,
        "bridge_vs_amort_resid": bridge_vs_amort_resid,
    }

    # ---- (5) reconcile the two discounts: bridge (denken #278) vs phi (fern #274) ----
    reconcile = {
        "bridge_denken278": {
            "value": bridge_draft,
            "mechanism": "linear-path BATCH-NORMALIZATION: maps a batch=1 WALL ABSOLUTE saving into "
                         "the NORMALIZED composition step (step_norm 1218.2us).",
            "step_convention": "normalized composition step (1218.2us)",
            "applies_to": "linear-path DRAFT-side ABSOLUTE savings subtracted from step_norm "
                          "(kanna #269, kanna #277, wirbel #270).",
            "verify_side_value": verify_bridge,
            "verify_side_note": "deployed-M8 verify savings priced fractionally are already in this "
                                "basis -> bridge = 1.0.",
        },
        "phi_fern274": {
            "value_served_wall": F274_PHI_SERVED_WALL,
            "value_kcal_clean": F274_PHI_KCAL_CLEAN,
            "mechanism": "wall-clock FIXED-OVERHEAD absorption: phi = model_forward / wall_step is "
                         "the touchable model-forward fraction of the WALL step; the (1-phi) fixed "
                         "overhead is unaffected by a draft cut.",
            "step_convention": f"wall-clock step (served_wall_clock_est {F274_SERVED_WALL_CLOCK_US:.0f}us "
                               "or kcal_clean 7982.9us) -- NOT the normalized step.",
            "applies_to": "tree-path / static-K draft-pass CUTS in the WALL-CLOCK recompute "
                          "(stark #273 A/B; fern #274's static_k4/k5, tree_width). NOT in this portfolio.",
        },
        "verdict": (
            "bridge (0.2147) and phi (0.603) are DIFFERENT mechanisms on DIFFERENT step conventions: "
            "bridge maps a wall saving into the NORMALIZED step; phi is the model-forward fraction "
            "WITHIN the wall step. They must NOT be stacked on the same projection. This portfolio "
            "card is in the COMPOSITION (normalized) basis, so it uses the BRIDGE only (draft 0.2147, "
            "verify 1.0). fern #274's phi-discounted numbers live in the complementary wall-clock "
            "basis; both AGREE on the bottom line (no draft lever clears 500 honestly)."),
        "double_count_flag": (
            "DOUBLE-COUNT RISK: kanna #269 -- combining its phi-discounted WALL gain (fern's "
            "convention) with its bridge-discounted COMPOSITION gain (this card) double-discounts. "
            "Coherent rule: pick the discount matching the step you project against -- composition "
            "step -> bridge; wall-clock A/B -> phi. NEVER both."),
    }

    # ---- (6) self-test (PRIMARY) ----
    # (a) bridge reproduces denken #278's +4.39% -> +0.91% re-pricing of kanna #269.
    row_269 = next(r for r in card if r["pr"] == 269)
    a_raw_269 = abs(row_269["raw_composition_gain_pct"] - D278_MODEL_A_GAIN_PCT) <= 0.05
    a_honest_269 = abs(row_269["basis_honest_gain_pct"] - D278_MODEL_B_GAIN_PCT) <= 0.05
    a_overcredit = abs((row_269["raw_composition_gain_pct"] / row_269["basis_honest_gain_pct"])
                       - D278_OVERCREDIT_FACTOR) <= 0.05
    # (b) verify-side bridge ~ 1.0 reproduces wirbel #279's +1.29% as basis-honest (no discount).
    row_279 = next(r for r in card if r["pr"] == 279)
    b_verify = (abs(row_279["bridge"] - 1.0) < 1e-12
                and abs(row_279["basis_honest_gain_pct"] - row_279["raw_composition_gain_pct"]) < 1e-9
                and abs(row_279["basis_honest_gain_pct"] - 1.2933622095534503) <= 0.05)
    # (c) composition round-trips 481.53 at delta_step = 0.
    c_roundtrip = abs(tps_at(STEP_SERVED_US) - OFFICIAL_BASELINE) <= 0.1
    c_roundtrip_literal = abs(K_CAL * (E_T / (STEP_SERVED_US / 1e3)) * TAU - OFFICIAL_BASELINE) <= 0.1
    # (d) every lever classified with a stated justification.
    d_classified = all(r["basis"] in {"draft_b1_wall", "verify_deployed_m8"}
                       and bool(r["justification"]) for r in card)
    # (f) baseline constants imported EXACT.
    f_constants = (abs(OFFICIAL_BASELINE - 481.53) < 1e-9
                   and abs(LAMBDA1_CEIL - 520.9527323111674) < 1e-6
                   and abs(K_CAL - 125.26795005202914) < 1e-9
                   and abs(STEP_SERVED_US - 1218.2) < 1e-9
                   and abs(D278_BRIDGE - 0.2147122962556323) < 1e-9
                   and abs(E_T - 3.844) < 1e-9)

    cond = {
        "a_bridge_reproduces_denken278_269_repricing": bool(a_raw_269 and a_honest_269 and a_overcredit),
        "b_verify_bridge1_reproduces_wirbel279_129": bool(b_verify),
        "c_composition_roundtrips_481p53_at_delta0": bool(c_roundtrip and c_roundtrip_literal),
        "d_every_lever_classified_with_justification": bool(d_classified),
        "f_baseline_constants_imported_exact": bool(f_constants),
    }
    # (e) NaN-clean is checked on the full payload after assembly (added in main()).

    # ---- assemble headline + verdict ----
    pre_repricing_best = max(card, key=lambda r: r["raw_composition_gain_pct"])
    verdict = (
        f"The bridge is DRAFT-SIDE-SPECIFIC. Re-priced to ONE coherent (composition/normalized) "
        f"basis: draft-side levers carry bridge={bridge_draft:.4f} (the {one_over_bridge:.2f}x "
        f"over-credit), verify-side levers carry bridge=1.0 (already deployed-M8 normalized). The "
        f"re-pricing FLIPS the best lever: pre-repricing kanna #269's +{pre_repricing_best['raw_composition_gain_pct']:.2f}% "
        f"({tps_at(STEP_SERVED_US - pre_repricing_best['delta_step_raw_us']):.1f} TPS) LOOKED best and "
        f"the only one 'crossing 500'; basis-honest it is +{row_269['basis_honest_gain_pct']:.2f}% "
        f"({row_269['basis_honest_tps']:.1f} TPS, does NOT cross). The best single basis-honest step "
        f"lever is the VERIFY SDPA tune at {best_basis_honest_step_tps:.1f} TPS "
        f"(+{best['basis_honest_gain_pct']:.2f}%), and the full composed disjoint basis-honest step "
        f"stack reaches {composed_tps:.1f} TPS (+{composed_gain_pct:.2f}%) -- "
        f"{'crosses' if composed_step_stack_crosses_500 else 'does NOT cross'} 500 (gap "
        f"{composed_to_500_gap:.1f}). The step side is rigorously CLOSED; no step lever or composed "
        f"step stack reaches 500. BASELINE 481.53 untouched; analysis-only; NOT a launch.")

    handoff = (
        f"draft-side levers carry bridge~0.21 ({one_over_bridge:.1f}x over-credit) and verify-side "
        f"levers carry bridge~1.0 (already deployed-basis), so the basis-honest best single step "
        f"lever is {best_basis_honest_step_tps:.1f} TPS (verify SDPA num_stages=2) and the full "
        f"composed basis-honest step stack reaches {composed_tps:.1f} TPS "
        f"({'crosses' if composed_step_stack_crosses_500 else 'does not cross'} 500), rigorously "
        f"closing the step-side and confirming the E[T]-raise axis (fern #281) as the sole remaining "
        f">500 path.")

    return {
        "constants": {
            "official_baseline": OFFICIAL_BASELINE, "lambda1_ceil": LAMBDA1_CEIL, "K_cal": K_CAL,
            "step_served_us": STEP_SERVED_US, "E_T": E_T, "tau": TAU,
            "bridge_draft": bridge_draft, "bridge_verify": verify_bridge,
            "one_over_bridge_draft": one_over_bridge,
        },
        "bridge_derivation": {
            "draft_k7_wall_us": D278_DRAFT_K7_WALL_US,
            "verify_m1_wall_us": D278_VERIFY_M1_WALL_US,
            "step_wall_microbuilt_us": D278_STEP_WALL_MICROBUILT_US,
            "bridge_draft": bridge_draft, "one_over_bridge": one_over_bridge,
            "verify_b8_total_us": D278_VERIFY_B8_TOTAL_US,
            "verify_b8_per_seq_us": verify_b8_per_seq,
            "per_seq_amort_factor": per_seq_amort_factor,
            "step_vs_perseq_verify_ratio": step_vs_perseq_verify_ratio,
            "bridge_vs_amort_resid": bridge_vs_amort_resid,
        },
        "draft_bridge_caveat": draft_bridge_caveat,
        "lever_card": card,
        "best_single_lever": {
            "lever": best["lever"], "pr": best["pr"], "run": best["run"],
            "basis": best["basis"], "basis_honest_tps": best_basis_honest_step_tps,
            "basis_honest_gain_pct": best["basis_honest_gain_pct"],
            "crosses_500_alone": best["crosses_500_alone"],
        },
        "composed_step_stack": {
            "disjoint_groups_used": groups_used,
            "stack_members": [r["lever"] for r in stack_rows],
            "excluded_duplicates": [r["lever"] for r in card if not r["stack_member"]],
            "sum_delta_honest_us": sum_delta_honest,
            "composed_new_step_us": composed_new_step,
            "composed_step_stack_basis_honest_tps": composed_tps,
            "composed_gain_pct": composed_gain_pct,
            "composed_step_stack_crosses_500": composed_step_stack_crosses_500,
            "gap_to_500_tps": composed_to_500_gap,
            "gap_to_lambda1_ceiling_tps": composed_to_ceiling_gap,
        },
        "phi_bridge_reconciliation": reconcile,
        "self_test": {
            "conditions": cond,
            "row_269_raw_gain_pct": row_269["raw_composition_gain_pct"],
            "row_269_honest_gain_pct": row_269["basis_honest_gain_pct"],
            "denken278_model_A_gain_pct": D278_MODEL_A_GAIN_PCT,
            "denken278_model_B_gain_pct": D278_MODEL_B_GAIN_PCT,
            "row_279_honest_gain_pct": row_279["basis_honest_gain_pct"],
        },
        # ---- headline metrics ----
        "best_basis_honest_step_tps": best_basis_honest_step_tps,
        "composed_step_stack_basis_honest_tps": composed_tps,
        "composed_step_stack_crosses_500": composed_step_stack_crosses_500,
        "verdict": verdict, "handoff": handoff,
    }


# --------------------------------------------------------------------------- #
# W&B logging (mirrors denken #278; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.append(repo)
    _w = sys.modules.get("wandb")
    if _w is not None and not hasattr(_w, "init"):
        del sys.modules["wandb"]
    try:
        import wandb as _wb
        if not hasattr(_wb, "init"):
            raise ImportError(
                f"resolved a stub/namespace wandb at {list(getattr(_wb, '__path__', []) or [])} "
                "with no .init -> this venv lacks the wandb wheel")
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[bridge-card] wandb logging skipped (analysis unaffected): {exc}", flush=True)
        return

    syn = payload["synthesis"]
    cs = syn["composed_step_stack"]
    st = syn["self_test"]
    try:
        run = init_wandb_run(
            job_type="validity-gate", agent="kanna", name=args.wandb_name, group=args.wandb_group,
            tags=["bridge-repricing", "lever-card", "basis-honest", "composition-honesty",
                  "draft-vs-verify", "step-side-closure", "bank-the-analysis", "pr-286"],
            config={
                "official_baseline": OFFICIAL_BASELINE, "lambda1_ceil": LAMBDA1_CEIL, "K_cal": K_CAL,
                "step_served_us": STEP_SERVED_US, "E_T": E_T, "tau": TAU,
                "bridge_draft": D278_BRIDGE, "bridge_verify": 1.0,
                "n_levers": len(LEVERS),
                "imports": "denken#278(bu44n30q bridge=0.2147) x kanna#269(epl52mkq +4.39%) x "
                           "kanna#277(ahw089yi NULL) x wirbel#270(iwwcmvez +1.28%) x "
                           "wirbel#279(xme9snkv +1.29%) x kanna#280(sdrerk5h +1.19%) x "
                           "fern#274(brnmnl60 phi=0.603)",
                "wandb_group": args.wandb_group,
            },
        )
    except Exception as exc:
        print(f"[bridge-card] wandb init failed (analysis unaffected): {exc}", flush=True)
        return
    if run is None:
        print("[bridge-card] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "bridge_repricing_lever_card_self_test_passes":
            int(bool(payload["bridge_repricing_lever_card_self_test_passes"])),
        "best_basis_honest_step_tps": syn["best_basis_honest_step_tps"],
        "composed_step_stack_basis_honest_tps": syn["composed_step_stack_basis_honest_tps"],
        "composed_step_stack_crosses_500": int(bool(syn["composed_step_stack_crosses_500"])),
        "composed_gain_pct": cs["composed_gain_pct"],
        "composed_new_step_us": cs["composed_new_step_us"],
        "sum_delta_honest_us": cs["sum_delta_honest_us"],
        "gap_to_500_tps": cs["gap_to_500_tps"],
        "bridge_draft": D278_BRIDGE, "bridge_verify": 1.0,
        "one_over_bridge_draft": syn["constants"]["one_over_bridge_draft"],
        "best_lever_pr": syn["best_single_lever"]["pr"],
        "best_lever_gain_pct": syn["best_single_lever"]["basis_honest_gain_pct"],
        "row_269_raw_gain_pct": st["row_269_raw_gain_pct"],
        "row_269_honest_gain_pct": st["row_269_honest_gain_pct"],
        "row_279_honest_gain_pct": st["row_279_honest_gain_pct"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    # per-lever card rows as flat metrics
    for r in syn["lever_card"]:
        pr = r["pr"]
        summary[f"lever{pr}_basis_honest_tps"] = r["basis_honest_tps"]
        summary[f"lever{pr}_basis_honest_gain_pct"] = r["basis_honest_gain_pct"]
        summary[f"lever{pr}_raw_gain_pct"] = r["raw_composition_gain_pct"]
        summary[f"lever{pr}_bridge"] = r["bridge"]
        summary[f"lever{pr}_crosses_500_alone"] = int(bool(r["crosses_500_alone"]))

    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    try:
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="bridge_repricing_lever_card_result",
                          artifact_type="validity", data=payload)
        finish_wandb(run)
        print(f"[bridge-card] wandb logged {len(summary)} summary keys", flush=True)
    except Exception as exc:
        print(f"[bridge-card] wandb write failed (analysis unaffected): {exc}", flush=True)


def _assert_nan_clean(obj: Any, path: str = "") -> list[str]:
    bad = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad += _assert_nan_clean(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _assert_nan_clean(v, f"{path}[{i}]")
    return bad


def _print_human(syn: dict) -> None:
    print("\n" + "=" * 104, flush=True)
    print(" BRIDGE BASIS-HONESTY LEVER CARD (PR #286) — re-price step levers (draft 0.21 vs verify 1.0)",
          flush=True)
    print("=" * 104, flush=True)
    bd = syn["bridge_derivation"]
    print(f"  bridge_draft = step_norm/(draft_b1 {bd['draft_k7_wall_us']:.1f} + verify_b1 "
          f"{bd['verify_m1_wall_us']:.1f}) = {STEP_SERVED_US:.1f}/{bd['step_wall_microbuilt_us']:.1f} "
          f"= {bd['bridge_draft']:.4f}  (1/bridge = {bd['one_over_bridge']:.2f}x)", flush=True)
    print(f"  bridge_verify = 1.0  PROOF: verify b8 per-seq {bd['verify_b8_per_seq_us']:.1f}us ~ "
          f"step_norm {STEP_SERVED_US:.1f}us; b8-per-seq amort {bd['per_seq_amort_factor']:.4f} ~ "
          f"bridge {bd['bridge_draft']:.4f}", flush=True)
    print("-" * 104, flush=True)
    print(f"  {'lever':<34}{'basis':<18}{'raw%':>7}{'bridge':>8}{'honest%':>9}{'honest_tps':>12}"
          f"{'>500?':>7}", flush=True)
    for r in syn["lever_card"]:
        dup = "  (== #%d)" % r["same_lever_as"] if r["same_lever_as"] else ""
        print(f"  {r['lever']:<34}{r['basis']:<18}{r['raw_composition_gain_pct']:>6.2f}%"
              f"{r['bridge']:>8.4f}{r['basis_honest_gain_pct']:>8.2f}%{r['basis_honest_tps']:>12.2f}"
              f"{str(r['crosses_500_alone']):>7}{dup}", flush=True)
    print("-" * 104, flush=True)
    best = syn["best_single_lever"]
    cs = syn["composed_step_stack"]
    print(f"  BEST single basis-honest lever: {best['lever']} (#{best['pr']}) = "
          f"{best['basis_honest_tps']:.2f} TPS (+{best['basis_honest_gain_pct']:.2f}%)  "
          f"crosses_500={best['crosses_500_alone']}", flush=True)
    print(f"  COMPOSED disjoint stack {cs['stack_members']}:", flush=True)
    print(f"    sum_delta_honest = {cs['sum_delta_honest_us']:.2f}us -> step {cs['composed_new_step_us']:.1f}us "
          f"-> {cs['composed_step_stack_basis_honest_tps']:.2f} TPS (+{cs['composed_gain_pct']:.2f}%)  "
          f"crosses_500={cs['composed_step_stack_crosses_500']}  (gap_to_500 {cs['gap_to_500_tps']:.1f})",
          flush=True)
    st = syn["self_test"]
    print(f"  SELF-TEST: { {k: int(v) for k, v in st['conditions'].items()} }", flush=True)
    print("-" * 104, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print(f"\n  HAND-OFF: {syn['handoff']}\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="bridge-repricing-lever-card")
    args = ap.parse_args(argv)

    syn = synthesize()
    self_test_passes = all(syn["self_test"]["conditions"].values())

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 286, "agent": "kanna",
        "kind": "bridge-repricing-lever-card", "synthesis": syn,
        "bridge_repricing_lever_card_self_test_passes": self_test_passes,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    # fold NaN-clean (condition e) into the PRIMARY pass.
    payload["bridge_repricing_lever_card_self_test_passes"] = bool(self_test_passes and payload["nan_clean"])
    if nan_paths:
        print(f"[bridge-card] WARNING non-finite at: {nan_paths[:10]}", flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bridge_repricing_lever_card_results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    _print_human(syn)
    print(f"[bridge-card] wrote {out_path}", flush=True)
    print(f"[bridge-card] PRIMARY bridge_repricing_lever_card_self_test_passes = "
          f"{payload['bridge_repricing_lever_card_self_test_passes']}", flush=True)
    print(f"[bridge-card] TEST best_basis_honest_step_tps = {syn['best_basis_honest_step_tps']:.4f}", flush=True)
    print(f"[bridge-card] composed_step_stack_basis_honest_tps = "
          f"{syn['composed_step_stack_basis_honest_tps']:.4f}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = payload["bridge_repricing_lever_card_self_test_passes"]
        print(f"[bridge-card] PRIMARY self-test: {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
