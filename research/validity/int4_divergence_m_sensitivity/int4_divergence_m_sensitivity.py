#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Int4 divergence M-sensitivity: does the deployed M=8 0.73% identity hold at M=16? (PR #242).

THE MISSING M-AXIS OF THE TOKEN-IDENTITY STORY
----------------------------------------------
lawine #232 (`nxwv6pam`) measured the TRUE deployed greedy divergence at the spec verify
width M=8 -- int4 M=1-AR-vs-M=8-verify per-token argmax divergence = 0.007292 (identity
0.992708, near-greedy). But land #71's live TREE-decode build runs the verify at the tree
width M=16, not M=8. Every downstream validity leg consumes "0.73% divergence" as if it were
M-invariant: denken #236 prices the served PPL with it, stark #233 the private-draw fraction,
fern the GO-card. The build's M differs from the measured M. CRUX: does the 0.73% identity
HOLD, IMPROVE, or DEGRADE at the tree build's M=16?

THE MECHANISM (why divergence depends on M -- the frame)
--------------------------------------------------------
A GEMM C[m,n] = sum_k A[m,k]*B[k,n] reduces over K. Split-K partitions that K-reduction across
tiles; the floating-point accumulation ORDER (how partial sums are grouped/combined) is set by
the kernel's tiling, which the kernel selects from the problem shape (M, N, K). For a position
whose top-2 logits sit within FP-rounding distance (a "near-tie"), a different accumulation
order can FLIP the argmax. So the verify-vs-M1 divergence = (near-tie mass) x (order-instability
at width M). #114's interlock onset signature is literally "FP-reduction near-tie flips", and
its reference_kind is "unknown" (native-spec geometry, M unrecorded). This is the mechanism.

THE LOCUS CORRECTION (what #232 actually pinned -- load-bearing, do NOT skip)
----------------------------------------------------------------------------
The PR frames the root cause as "the int4 Marlin split-K reduction order = f(M)". #232's
in-process diagnostic CORRECTS the locus: all four int4-Marlin BODY GEMMs (qkv/o/gate_up/down)
are BIT-EXACT across M in {1,8} (max_abs_diff = 0.0 each) -- the int4 body's split-K schedule is
M-INVARIANT at the deployed widths, contributing ZERO batch-width divergence. The residual
0.73% is the bf16 tied lm_head + bf16 attention/norm accumulation being batch-variant (and it
sits BELOW the #221 bf16 floor 0.010559, 0.69x). So the M-dependence the build will see is a
bf16-lm_head property, not an int4-body property -- which makes the M=16 projection MORE
confident toward a plateau (the int4 body stays bit-exact; only the bf16 lm_head's K-reduction,
set by the hidden-dim K-partition, can move, and only if M=16 crosses a tiling-config boundary).

THE divergence(M) CURVE (the core)
----------------------------------
Two complementary readings, both anchored to the two hard measured points:
  (A) CLEAN per-token verify divergence -- the quantity the build runs. Boundary: div_A(1)=0
      (#232 determinism control det_M1_vs_M1=1.0: M=1-verify IS M=1-AR), measured div_A(8)=0.007292.
      Mechanism: order-instability q(M) is 0 at M=1 and SATURATES once the GEMM enters batched
      mode (M>=2), because each output row's K-reduction order is set by the K-partition (shared
      by all rows), not by M -- so div_A(M>=2) plateaus at div_A(8). The clean curve RISES from
      0 (M=1) to the batched floor (M=8) and HOLDS.
  (B) PR two-anchor ENVELOPE -- fit a monotone-decreasing law through (M=1, 0.560776 #114-native
      -spec) and (M=8, 0.007292 #232-clean). #114 and #232 measure DIFFERENT quantities (#114 is
      native-spec stochasticity incl. draft branching, an UPPER envelope; #232 is the isolated
      batch-width effect), so B treats 0.5608 as the M=1 envelope, not a clean point. The "drop"
      0.5608->0.0073 is the native-spec envelope collapsing to the clean batched floor.

Both readings agree on the projection direction: at M=16 the divergence does NOT grow past the
M=8 value except in the bounded pessimistic tail (M=16 crosses a split-K tiling boundary and
re-randomizes the near-tie flips, capped near 2x the M=8 mass / the bf16 floor regime). Central
call: PLATEAU. projected_divergence_at_M16 = 0.007292 (= the M=8 value), bound [B-decay tail,
2x-M8 tail]. Near-greedy (identity >= 0.99) SURVIVES at the central projection; the strict 0.99
line is at risk only in the pessimistic upper tail, which is one-probe-confirmable by land #71.

SCOPE: LOCAL CPU-only analytic projection of the int4 greedy divergence from the measured M=8
to the live tree build's M=16, via the FP-reduction-order mechanism + the two banked anchors,
with an explicit bound and a confirming-probe flag. No GPU / vLLM / draw / served-file change /
HF Job / submission. BASELINE stays 481.53. This leg adds 0 TPS (a projection). Bank-the-analysis
(PRIMARY = self-test). You do NOT re-measure (the M=16 confirm is land #71's build) and do NOT
re-derive the M=8/M=1 rates (import #232 / #114).

PRIMARY metric  int4_divergence_m_sensitivity_self_test_passes
TEST    metric  projected_divergence_at_M16   (= the plateau central projection, 0.007292)

Run:
    CUDA_VISIBLE_DEVICES="" python research/validity/int4_divergence_m_sensitivity/int4_divergence_m_sensitivity.py \\
        --self-test --wandb_group issue192-reading-calibration --wandb_name lawine/int4-divergence-m-sensitivity
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
# Imported anchors (one source per constant; NOT re-derived). Read from the
# banked JSONs where they exist so the anchors are authoritative, not re-typed.
# --------------------------------------------------------------------------- #
# lawine #232 (`nxwv6pam`): clean deployed M=1-AR-vs-M=8-verify int4 divergence + body bit-exactness.
_LAWINE_232_REPORT = REPO_ROOT / "research/validity/int4_tokenident_deployed_m8/int4_tokenident_report.json"
DIV_M8_FALLBACK = 0.007291666666666696       # #232 int4_divergence_M1_vs_M8
IDENT_M8_FALLBACK = 0.9927083333333333       # #232 int4_token_identity_M1_vs_M8

# kanna #114 (`9q5yy9l1`): native-spec-vs-M1 per-token divergence (reference_kind "unknown",
# onset "FP-reduction near-tie flips"). The M=1 ENVELOPE anchor (a different quantity than #232).
_KANNA_114_INTERLOCK = (REPO_ROOT
                        / "research/validity/self_referential_gate/ab-20260614T075459Z"
                        / "interlock_report.json")
DIV_M1_NATIVE_FALLBACK = 0.5607757568359375  # #114 token_div_frac (0.5608)

# #221 (`6m40u2bg`): the bf16 floor M1-vs-M8 divergence -- the LOCUS dtype (the residual #232
# divergence is the bf16 lm_head/attention, so the bf16 floor is the natural M=16 ceiling reference).
FP16_FLOOR_DIV_221 = 0.01055908203125        # #221 bf16 M1-vs-M8 divergence (imported via #232 report)

# land #71: the live tree-decode build runs the verify at the tree width M=16.
M_TREE_BUILD = 16                            # land #71 tree verify width (the projection target)
K_SPEC = 7                                   # num_speculative_tokens (manifest)
M_DEPLOYED = K_SPEC + 1                      # = 8 (the #232 measured spec verify width)

OFFICIAL_BASELINE = 481.53                   # PR #52 official TPS (this leg adds 0)

# The M grid the PR requests.
M_GRID = [1, 4, 8, 16]

# Near-greedy threshold: identity >= 0.99  <=>  divergence <= 0.01. The line denken #236's
# "0.73% near-greedy" framing rides on.
NEAR_GREEDY_DIV_MAX = 0.01
# Pessimistic-tail multiplier: if M=16 crosses a split-K tiling boundary and re-randomizes the
# near-tie flips vs M=1, the flip mass is bounded by ~2x the M=8 mass (a disjoint equal-size set).
PESSIMISTIC_MULT = 2.0

TOL_ANCHOR = 1e-9        # the two-anchor fit reproduces both anchors to ~machine precision
TOL_FINITE = 1e-12


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# Step 0 -- import the banked anchors.
# --------------------------------------------------------------------------- #
def load_imports() -> dict[str, Any]:
    # lawine #232: the clean M=8 divergence + identity + the int4-body bit-exactness locus fact.
    div_m8 = DIV_M8_FALLBACK
    ident_m8 = IDENT_M8_FALLBACK
    body_bitexact_m8 = True
    fp16_floor = FP16_FLOOR_DIV_221
    src_232 = "fallback-constant"
    try:
        r232 = json.load(open(_LAWINE_232_REPORT))
        if _finite(r232.get("int4_divergence_M1_vs_M8")):
            div_m8 = float(r232["int4_divergence_M1_vs_M8"])
        if _finite(r232.get("int4_token_identity_M1_vs_M8")):
            ident_m8 = float(r232["int4_token_identity_M1_vs_M8"])
        body_bitexact_m8 = bool(r232.get("int4_body_bitexact_decode_M8", True))
        anc = r232.get("imported_anchors", {})
        if _finite(anc.get("fp16_divergence_221")):
            fp16_floor = float(anc["fp16_divergence_221"])
        src_232 = str(_LAWINE_232_REPORT.relative_to(REPO_ROOT))
    except Exception as exc:  # never block the projection on a missing banked file
        print(f"[m-sensitivity] #232 report unavailable ({exc!r}); using fallback constants", flush=True)

    # kanna #114: the native-spec-vs-M1 M=1 envelope anchor.
    div_m1_native = DIV_M1_NATIVE_FALLBACK
    src_114 = "fallback-constant"
    onset_114 = "late/stochastic (FP-reduction near-tie flips)"
    refkind_114 = "unknown"
    try:
        r114 = json.load(open(_KANNA_114_INTERLOCK))
        per0 = r114["self_consistency_gate"]["per_run"][0]
        if _finite(per0.get("token_div_frac")):
            div_m1_native = float(per0["token_div_frac"])
        onset_114 = str(r114["self_consistency_gate"].get("onset_signature", onset_114))
        refkind_114 = str(r114["self_consistency_gate"].get("reference_kind", refkind_114))
        src_114 = str(_KANNA_114_INTERLOCK.relative_to(REPO_ROOT))
    except Exception as exc:
        print(f"[m-sensitivity] #114 interlock unavailable ({exc!r}); using fallback constant", flush=True)

    return {
        "div_m8_clean": div_m8,                 # 0.007292  (#232; the clean batch-width effect at M=8)
        "identity_m8_clean": ident_m8,          # 0.992708  (#232)
        "int4_body_bitexact_m8": body_bitexact_m8,  # True   (#232; the int4 body is M-invariant -> locus is bf16)
        "fp16_floor_div_221": fp16_floor,       # 0.010559  (#221 bf16 floor; the locus dtype ceiling reference)
        "div_m1_native": div_m1_native,         # 0.560776  (#114 native-spec envelope at M=1)
        "onset_signature_114": onset_114,       # "FP-reduction near-tie flips" (confirms the mechanism)
        "reference_kind_114": refkind_114,      # "unknown" (native-spec geometry; M unrecorded)
        "m_tree_build": M_TREE_BUILD,           # 16 (land #71)
        "m_deployed": M_DEPLOYED,               # 8
        "official_baseline": OFFICIAL_BASELINE, # 481.53
        "src_232": src_232, "src_114": src_114,
        "source_runs": {"lawine_232": "nxwv6pam", "kanna_114": "9q5yy9l1",
                        "fp16_221": "6m40u2bg", "land_71": "(tree build, M=16)"},
    }


# --------------------------------------------------------------------------- #
# (1) The mechanism (the frame) + the #232 locus correction.
# --------------------------------------------------------------------------- #
def frame_mechanism(imp: dict) -> dict[str, Any]:
    return {
        "mechanism": (
            "verify-vs-M1 per-token divergence = (near-tie mass) x (FP-reduction order-instability "
            "at width M). A GEMM's split-K partitions the K-reduction across tiles; the kernel picks "
            "the tiling (and thus the accumulation ORDER) from the problem shape (M,N,K). A near-tie "
            "(top-2 logits within FP-rounding distance) can flip argmax under a different order -> "
            "divergence is M-dependent."),
        "anchor_M1_native_spec": imp["div_m1_native"],   # 0.560776 (#114, envelope, native-spec)
        "anchor_M8_clean": imp["div_m8_clean"],           # 0.007292 (#232, clean batch-width effect)
        "drop_M1_to_M8": imp["div_m1_native"] - imp["div_m8_clean"],  # large DROP (0.5535)
        "direction_M1_to_M8": "DROP (native-spec envelope 0.5608 collapses to the clean batched floor 0.0073)",
        "onset_signature_114": imp["onset_signature_114"],
        "reference_kind_114": imp["reference_kind_114"],
        "locus_correction": (
            "#232 in-process diagnostic: all four int4-Marlin BODY GEMMs (qkv/o/gate_up/down) are "
            "BIT-EXACT across M in {1,8} (max_abs_diff=0 each) -> the int4 body split-K is M-INVARIANT "
            "and contributes ZERO batch-width divergence. The residual 0.73% is the bf16 tied lm_head "
            "+ bf16 attention/norm reduction being batch-variant (below the #221 bf16 floor 0.010559, "
            "0.69x). So the M-dependence is a bf16-lm_head property, not an int4-body property."),
        "int4_body_bitexact_m8": imp["int4_body_bitexact_m8"],
        "fp16_floor_div_221": imp["fp16_floor_div_221"],
        "locus_implication_for_M16": (
            "the int4 body stays bit-exact at M=16 (0 contribution); only the bf16 lm_head's "
            "K-reduction order can move, and only if M=16 crosses a tiling-config boundary -> the "
            "projection is biased toward PLATEAU, with a bounded tiling-boundary upper tail."),
        "two_quantities_caveat": (
            "the M=1 anchor (#114 0.5608) and the M=8 anchor (#232 0.0073) are DIFFERENT quantities: "
            "#114 is native-spec stochasticity (draft branching, reference_kind 'unknown') -- an UPPER "
            "envelope; #232 is the isolated 'hold weights, vary only M' batch-width effect. The clean "
            "curve's M=1 value is 0 (#232 determinism control), not 0.5608."),
    }


# --------------------------------------------------------------------------- #
# (2) The divergence(M) curve (the core) -- two readings.
# --------------------------------------------------------------------------- #
def _curveB_powerlaw(M: float, d1: float, d8: float) -> float:
    """Monotone-decreasing power law through (1, d1) and (8, d8): div = d1 * M^(-p)."""
    p = -math.log(d8 / d1) / math.log(8.0)
    return d1 * (M ** (-p))


def _curveB_exp(M: float, d1: float, d8: float) -> float:
    """Monotone-decreasing exponential through (1, d1) and (8, d8): div = d1 * exp(-lam*(M-1))."""
    lam = -math.log(d8 / d1) / 7.0
    return d1 * math.exp(-lam * (M - 1.0))


def _curveA_clean(M: float, d8: float) -> float:
    """Clean per-token verify divergence: 0 at M=1 (det control), saturates to d8 for M>=2 (plateau)."""
    if M <= 1.0:
        return 0.0
    return d8


def build_curves(imp: dict) -> dict[str, Any]:
    d1 = imp["div_m1_native"]     # 0.560776 (#114 envelope at M=1)
    d8 = imp["div_m8_clean"]      # 0.007292 (#232 clean at M=8)

    # Curve B fit parameters (the two-anchor envelope).
    p_pow = -math.log(d8 / d1) / math.log(8.0)        # ~2.089
    lam_exp = -math.log(d8 / d1) / 7.0                # ~0.6204

    # Reproduce both anchors (self-test (a)): the fits pass through (1,d1) and (8,d8) by construction.
    fitB_at1_pow = _curveB_powerlaw(1.0, d1, d8)
    fitB_at8_pow = _curveB_powerlaw(8.0, d1, d8)
    fitB_at1_exp = _curveB_exp(1.0, d1, d8)
    fitB_at8_exp = _curveB_exp(8.0, d1, d8)
    # Curve A reproduces its own M=8 anchor exactly and its M=1 boundary (the #232 det control = 0).
    fitA_at8 = _curveA_clean(8.0, d8)
    fitA_at1 = _curveA_clean(1.0, d8)

    return {
        "curveA_clean": {
            "form": "div_A(1)=0 (#232 det control); div_A(M>=2)=div_A(8)=d8 (saturating plateau)",
            "rationale": ("each output row's K-reduction order is set by the K-partition (shared across "
                          "rows), not by M; once batched (M>=2) the order saturates -> the clean curve "
                          "plateaus at the M=8 batched floor."),
            "fitA_at1": fitA_at1, "fitA_at8": fitA_at8,
            "monotone": "non-decreasing from M=1 (0) to the plateau d8 (M>=2)",
        },
        "curveB_envelope": {
            "form": "div_B(M) = d1 * M^(-p) [power]  and  d1 * exp(-lam*(M-1)) [exp], through (1,d1),(8,d8)",
            "rationale": ("the PR's two-anchor reading: 0.5608 is the M=1 native-spec envelope, 0.0073 "
                          "the M=8 clean floor; the 'drop' is the envelope collapsing. A decaying law "
                          "extrapolates 'more batch -> more stable' beyond M=8."),
            "p_powerlaw": p_pow, "lam_exp": lam_exp,
            "fitB_at1_powerlaw": fitB_at1_pow, "fitB_at8_powerlaw": fitB_at8_pow,
            "fitB_at1_exp": fitB_at1_exp, "fitB_at8_exp": fitB_at8_exp,
            "monotone": "decreasing in M (reproduces the PR's M=1->M=8 DROP direction)",
        },
        "anchors": {"d1_native_M1": d1, "d8_clean_M8": d8},
        "params": {"p_powerlaw": p_pow, "lam_exp": lam_exp},
    }


# --------------------------------------------------------------------------- #
# (3) Project M=16 with a bound + the M-grid table (the deliverable).
# --------------------------------------------------------------------------- #
def project_M16(imp: dict, curves: dict) -> dict[str, Any]:
    d1 = imp["div_m1_native"]
    d8 = imp["div_m8_clean"]
    M16 = float(imp["m_tree_build"])

    # Central call: PLATEAU (Curve A) -- the mechanism-faithful projection.
    central = _curveA_clean(M16, d8)                          # = d8 = 0.007292

    # Optimistic tail (Curve B "continued decay" reading): the exp fit gives the smallest M=16
    # value, the power fit a milder decay; take the exp as the lower bound endpoint.
    b16_pow = _curveB_powerlaw(M16, d1, d8)                   # ~0.001714
    b16_exp = _curveB_exp(M16, d1, d8)                        # ~5.09e-5
    lower = min(b16_exp, b16_pow, central)                   # continued-decay optimistic tail

    # Pessimistic tail (Curve A tiling-boundary re-randomization): M=16 crosses a split-K boundary
    # and re-randomizes the near-tie flips vs M=1; bounded by ~2x the M=8 mass (a disjoint equal set).
    # Reference the bf16 floor (#221) as the locus-dtype ceiling cross-check.
    upper = PESSIMISTIC_MULT * d8                             # ~0.014583
    upper_bf16_ref = imp["fp16_floor_div_221"]               # 0.010559 (bf16 M=8 floor; cross-check)

    projected_divergence_at_M16 = central
    projected_identity_at_M16 = 1.0 - central
    near_greedy_survives = bool(projected_divergence_at_M16 <= NEAR_GREEDY_DIV_MAX)
    # the strict 0.99-identity line is at risk only if the UPPER tail breaches it.
    strict_line_at_risk_upper = bool(upper > NEAR_GREEDY_DIV_MAX)

    # --- the M-grid table ---
    def row(M: int) -> dict[str, Any]:
        Mf = float(M)
        clean = _curveA_clean(Mf, d8)
        env_pow = _curveB_powerlaw(Mf, d1, d8)
        if M == 1:
            return {
                "M": 1,
                "divergence_clean": 0.0,                      # #232 det control (M=1-verify IS M=1-AR)
                "divergence_envelope_native": d1,            # #114 native-spec envelope
                "identity_clean": 1.0,
                "kind": "MEASURED (two quantities: clean det-control 0.0 #232; native-spec env 0.5608 #114)",
                "confidence": "high (both measured; flagged as different quantities)",
            }
        if M == 8:
            return {
                "M": 8,
                "divergence_clean": d8,                       # #232 MEASURED
                "divergence_envelope_native": env_pow,
                "identity_clean": 1.0 - d8,
                "kind": "MEASURED (#232 clean M=1-AR-vs-M=8-verify)",
                "confidence": "high (measured, controls 1.0)",
            }
        if M == 16:
            return {
                "M": 16,
                "divergence_clean": projected_divergence_at_M16,  # plateau central
                "divergence_envelope_native": env_pow,
                "identity_clean": projected_identity_at_M16,
                "kind": "PROJECTED (plateau central; bound below)",
                "confidence": ("medium-high (plateau; only risk is a split-K tiling-boundary crossing "
                               "between M=8 and M=16 -> one-probe-confirmable by land #71)"),
            }
        # M=4 (interior): plateau central, envelope-power as the decay reading.
        return {
            "M": M,
            "divergence_clean": clean,                        # plateau (= d8 for M>=2)
            "divergence_envelope_native": env_pow,
            "identity_clean": 1.0 - clean,
            "kind": "PROJECTED (interior; plateau central, no boundary crossing 1<M<8 expected)",
            "confidence": "medium (interpolated plateau; no measurement)",
        }

    table = [row(M) for M in M_GRID]

    return {
        "projected_divergence_at_M16": projected_divergence_at_M16,   # TEST metric (plateau central 0.007292)
        "projected_identity_at_M16": projected_identity_at_M16,        # 0.992708
        "bound_lower": lower,                  # continued-decay optimistic tail (~5e-5)
        "bound_upper": upper,                  # tiling-boundary re-randomization tail (~0.0146)
        "bound_upper_bf16_floor_ref": upper_bf16_ref,  # 0.010559 (bf16 M=8 floor cross-check)
        "central_model": "PLATEAU (Curve A: clean batch-width order saturates for M>=2)",
        "curveB_M16_powerlaw": b16_pow,
        "curveB_M16_exp": b16_exp,
        "direction_verdict": (
            "PLATEAU central -- M=16 CONTINUES no worse than M=8 (the clean batch-width order is "
            "M-invariant within a tiling bucket; M=8 and M=16 are both small and likely share it). "
            "Optimistic side: CONTINUE the improvement (Curve B decay). Pessimistic side: a bounded "
            "REVERSE only if M=16 crosses a split-K tiling boundary (capped ~2x M=8)."),
        "near_greedy_threshold_div": NEAR_GREEDY_DIV_MAX,
        "near_greedy_threshold_identity": 1.0 - NEAR_GREEDY_DIV_MAX,
        "near_greedy_survives_at_M16": near_greedy_survives,    # True (central 0.0073 <= 0.01)
        "near_greedy_verdict": "SURVIVES" if near_greedy_survives else "AT-RISK",
        "strict_line_at_risk_in_upper_tail": strict_line_at_risk_upper,  # True (0.0146 > 0.01)
        "table": table,
        "confirming_probe_flag": (
            "the M=16 divergence is a ONE-PROBE-CONFIRMABLE readout the live tree build (land #71) must "
            "report: re-run the #232 M=1-AR-vs-M=16-verify identity at the build's tree width. It is NOT "
            "free (needs the built tree-decode verify) but is cheap once the build runs -- it confirms "
            "plateau vs the bounded tiling-boundary tail."),
    }


# --------------------------------------------------------------------------- #
# (4) Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def self_test(imp: dict, frame: dict, curves: dict, proj: dict) -> dict[str, Any]:
    d1 = imp["div_m1_native"]
    d8 = imp["div_m8_clean"]
    cB = curves["curveB_envelope"]
    cA = curves["curveA_clean"]

    # (a) the curve passes through BOTH anchors within tolerance.
    #     Curve B reproduces (1,d1) and (8,d8); Curve A reproduces its M=8 anchor d8 and M=1 boundary 0.
    cond_a = bool(
        abs(cB["fitB_at1_powerlaw"] - d1) <= TOL_ANCHOR
        and abs(cB["fitB_at8_powerlaw"] - d8) <= TOL_ANCHOR
        and abs(cB["fitB_at1_exp"] - d1) <= TOL_ANCHOR
        and abs(cB["fitB_at8_exp"] - d8) <= TOL_ANCHOR
        and abs(cA["fitA_at8"] - d8) <= TOL_ANCHOR
        and abs(cA["fitA_at1"] - 0.0) <= TOL_ANCHOR
    )

    # (b) projected_divergence_at_M16 reported WITH an explicit bound: lower <= central <= upper,
    #     all finite, in (0,1).
    lo, ce, up = proj["bound_lower"], proj["projected_divergence_at_M16"], proj["bound_upper"]
    cond_b = bool(
        _finite(lo) and _finite(ce) and _finite(up)
        and 0.0 < lo <= ce <= up < 1.0
    )

    # (c) monotonicity/assumption stated AND the M=1->M=8 direction reproduced.
    #     Curve B: DROP (d1 > d8). Curve A: clean non-decreasing rise from 0 to d8.
    curveB_drop = (d1 > d8)
    curveA_rise = (_curveA_clean(1.0, d8) <= _curveA_clean(8.0, d8))
    # plateau assumption: div_A(16) == div_A(8) (the central model).
    plateau_assumption = abs(_curveA_clean(16.0, d8) - d8) <= TOL_ANCHOR
    cond_c = bool(curveB_drop and curveA_rise and plateau_assumption)

    # (d) the near-greedy verdict at M=16 is stated as survives/at-risk WITH the threshold,
    #     and is consistent with (central <= threshold).
    verdict = proj["near_greedy_verdict"]
    cond_d = bool(
        verdict in ("SURVIVES", "AT-RISK")
        and _finite(proj["near_greedy_threshold_div"])
        and proj["near_greedy_survives_at_M16"] == (ce <= proj["near_greedy_threshold_div"])
        and (verdict == "SURVIVES") == bool(proj["near_greedy_survives_at_M16"])
    )

    # (e) NaN-clean (key scalars finite; full-payload walk enforced in main()).
    key = [d1, d8, imp["identity_m8_clean"], imp["fp16_floor_div_221"],
           curves["params"]["p_powerlaw"], curves["params"]["lam_exp"],
           lo, ce, up, proj["projected_identity_at_M16"], proj["bound_upper_bf16_floor_ref"],
           proj["curveB_M16_powerlaw"], proj["curveB_M16_exp"]]
    cond_e = all(_finite(x) for x in key)

    passes = bool(cond_a and cond_b and cond_c and cond_d and cond_e)
    return {
        "int4_divergence_m_sensitivity_self_test_passes": passes,
        "conditions": {
            "a_curve_through_both_anchors": cond_a,
            "b_M16_projection_has_explicit_bound": cond_b,
            "c_monotonicity_and_M1_to_M8_direction": cond_c,
            "d_near_greedy_verdict_with_threshold": cond_d,
            "e_key_scalars_finite": cond_e,
        },
        "evidence": {
            "a_fitB1_pow": cB["fitB_at1_powerlaw"], "a_fitB8_pow": cB["fitB_at8_powerlaw"],
            "a_fitB1_exp": cB["fitB_at1_exp"], "a_fitB8_exp": cB["fitB_at8_exp"],
            "a_fitA8": cA["fitA_at8"], "a_fitA1": cA["fitA_at1"],
            "b_bound_lower": lo, "b_central": ce, "b_bound_upper": up,
            "c_curveB_drop_M1_gt_M8": bool(curveB_drop),
            "c_curveA_rise_0_to_d8": bool(curveA_rise),
            "c_plateau_assumption_div16_eq_div8": bool(plateau_assumption),
            "d_verdict": verdict, "d_threshold_div": proj["near_greedy_threshold_div"],
            "d_survives": bool(proj["near_greedy_survives_at_M16"]),
        },
    }


# --------------------------------------------------------------------------- #
# Verdict + hand-off.
# --------------------------------------------------------------------------- #
def _verdict(imp: dict, frame: dict, proj: dict) -> str:
    return (
        f"PLATEAU. The deployed 0.73% int4 divergence is an M=8 fact (#232); projecting the "
        f"FP-reduction-order mechanism to the tree build's M={imp['m_tree_build']} gives "
        f"projected_divergence_at_M16 = {proj['projected_divergence_at_M16']:.6f} "
        f"(identity {proj['projected_identity_at_M16']:.6f}), bound "
        f"[{proj['bound_lower']:.2e}, {proj['bound_upper']:.6f}]. The central call is PLATEAU because "
        f"#232's locus correction is load-bearing: the int4-Marlin BODY GEMMs are bit-exact across M "
        f"(M-invariant split-K, 0 contribution), so the M-dependence is a bf16-lm_head property whose "
        f"per-row K-reduction order is set by the hidden-dim K-partition (shared across rows), not by M "
        f"-- it saturates once batched (M>=2) and holds from M=8 to M=16 unless M=16 crosses a split-K "
        f"tiling boundary (the bounded upper tail ~{proj['bound_upper']:.4f} = 2x the M=8 mass, "
        f"cross-checked against the #221 bf16 floor {proj['bound_upper_bf16_floor_ref']:.6f}). "
        f"Near-greedy (identity >= {proj['near_greedy_threshold_identity']:.2f}) {proj['near_greedy_verdict']} "
        f"at the central projection; the strict line is at risk only in the pessimistic tiling-boundary "
        f"upper tail (one-probe-confirmable by land #71). Both readings agree on direction: Curve A "
        f"(clean) plateaus at 0.0073; Curve B (the PR two-anchor envelope through #114-native 0.5608 and "
        f"#232-clean 0.0073) decays further to [{proj['curveB_M16_exp']:.2e}, {proj['curveB_M16_powerlaw']:.4f}]. "
        f"BASELINE {imp['official_baseline']} untouched (this leg adds 0 TPS, a projection). NOT a launch."
    )


def _handoff(imp: dict, proj: dict) -> dict[str, str]:
    survives = proj["near_greedy_verdict"]
    line = (
        f"the deployed 0.73% int4 divergence is an M=8 fact; projecting the split-K FP-reduction-order "
        f"model (with #232's locus correction: int4 body bit-exact, bf16 lm_head is the M-variant locus) "
        f"to the tree build's M={imp['m_tree_build']} gives projected_divergence_at_M16="
        f"{proj['projected_divergence_at_M16']:.6f} (bound [{proj['bound_lower']:.2e}, "
        f"{proj['bound_upper']:.6f}], near-greedy {survives}), so denken #236's lambda-invariant-PPL "
        f"assumption holds at the build's M (PPL is output-equivalence-pinned regardless; even the upper "
        f"tail {proj['bound_upper']:.4f} keeps the served stream the int4 greedy stream) -- and land #71's "
        f"build MUST report the M=16 divergence as a one-probe confirm (plateau vs the bounded tiling tail)."
    )
    return {"denken_236": line, "fern_card": line, "land_71": line}


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    imp = load_imports()
    frame = frame_mechanism(imp)
    curves = build_curves(imp)
    proj = project_M16(imp, curves)
    st = self_test(imp, frame, curves, proj)
    handoff = _handoff(imp, proj)
    return {
        "self_test": st,
        "test_metric": {"projected_divergence_at_M16": proj["projected_divergence_at_M16"]},
        "imports": imp,
        "frame_mechanism": frame,
        "curves": curves,
        "projection_M16": proj,
        "verdict": _verdict(imp, frame, proj),
        "handoff_lines": handoff,
    }


# --------------------------------------------------------------------------- #
# NaN-clean walk.
# --------------------------------------------------------------------------- #
def _assert_nan_clean(payload: dict, path: str = "result") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


# --------------------------------------------------------------------------- #
# Console report.
# --------------------------------------------------------------------------- #
def _print_report(syn: dict) -> None:
    imp = syn["imports"]
    frame, curves = syn["frame_mechanism"], syn["curves"]
    proj, st = syn["projection_M16"], syn["self_test"]
    print("\n" + "=" * 100, flush=True)
    print("INT4 DIVERGENCE M-SENSITIVITY (PR #242) -- does the deployed M=8 0.73% hold at the tree M=16?", flush=True)
    print("=" * 100, flush=True)
    print(f"  anchors:  M=1 native-spec (#114) {imp['div_m1_native']:.6f}   "
          f"M=8 clean (#232) {imp['div_m8_clean']:.6f} (identity {imp['identity_m8_clean']:.6f})", flush=True)
    print(f"  mechanism: FP-reduction order = f(M); #114 onset '{imp['onset_signature_114']}' "
          f"(reference_kind {imp['reference_kind_114']})", flush=True)
    print(f"  LOCUS:    int4 body bit-exact across M = {imp['int4_body_bitexact_m8']}  -> M-variant locus is "
          f"the bf16 lm_head (#221 bf16 floor {imp['fp16_floor_div_221']:.6f})", flush=True)
    print("-" * 100, flush=True)
    print(f"  Curve A (clean): div(1)=0 (#232 det control) -> plateau div(M>=2)={imp['div_m8_clean']:.6f}", flush=True)
    print(f"  Curve B (envelope): power p={curves['params']['p_powerlaw']:.4f}, "
          f"exp lam={curves['params']['lam_exp']:.4f}  (through #114 M=1 and #232 M=8)", flush=True)
    print("-" * 100, flush=True)
    print(f"  {'M':>4}  {'div_clean':>11}  {'div_envelope':>13}  {'identity':>10}  kind", flush=True)
    for r in proj["table"]:
        print(f"  {r['M']:>4}  {r['divergence_clean']:>11.6f}  {r['divergence_envelope_native']:>13.6f}  "
              f"{r['identity_clean']:>10.6f}  {r['kind']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  projected_divergence_at_M16 = {proj['projected_divergence_at_M16']:.6f}   <-- TEST", flush=True)
    print(f"    bound [{proj['bound_lower']:.3e}, {proj['bound_upper']:.6f}]  "
          f"(bf16-floor cross-check {proj['bound_upper_bf16_floor_ref']:.6f})", flush=True)
    print(f"    central model: {proj['central_model']}", flush=True)
    print(f"    Curve B M=16: power {proj['curveB_M16_powerlaw']:.6f}  exp {proj['curveB_M16_exp']:.3e}", flush=True)
    print(f"  near-greedy (identity >= {proj['near_greedy_threshold_identity']:.2f}): "
          f"{proj['near_greedy_verdict']}   <-- HEADLINE", flush=True)
    print(f"    strict 0.99 line at risk in upper tail: {proj['strict_line_at_risk_in_upper_tail']}", flush=True)
    print("-" * 100, flush=True)
    print(f"  (PRIMARY) int4_divergence_m_sensitivity_self_test_passes = "
          f"{st['int4_divergence_m_sensitivity_self_test_passes']}", flush=True)
    for k, v in st["conditions"].items():
        print(f"          - {k}: {v}", flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)
    print(f"\n  HAND-OFF (denken #236 / fern card / land #71): {syn['handoff_lines']['denken_236']}\n", flush=True)


# --------------------------------------------------------------------------- #
# W&B logging (mirrors denken #236; never fatal).
# --------------------------------------------------------------------------- #
def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[m-sensitivity] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    imp, frame = syn["imports"], syn["frame_mechanism"]
    curves, proj, st = syn["curves"], syn["projection_M16"], syn["self_test"]

    run = init_wandb_run(
        job_type="validity-gate",
        agent="lawine",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["issue-192", "int4-divergence", "m-sensitivity", "token-identity", "split-k",
              "reduction-order", "tree-M16", "lawine-232", "kanna-114", "land-71", "bank-the-analysis"],
        config={
            "pr": 242,
            "div_m1_native_114": imp["div_m1_native"], "div_m8_clean_232": imp["div_m8_clean"],
            "identity_m8_232": imp["identity_m8_clean"], "int4_body_bitexact_m8": imp["int4_body_bitexact_m8"],
            "fp16_floor_div_221": imp["fp16_floor_div_221"],
            "m_tree_build": imp["m_tree_build"], "m_deployed": imp["m_deployed"],
            "official_baseline": imp["official_baseline"],
            "near_greedy_div_max": NEAR_GREEDY_DIV_MAX, "pessimistic_mult": PESSIMISTIC_MULT,
            "imports": "lawine#232 (0.007292) + kanna#114 (0.560776) + #221 bf16 floor (0.010559) + land#71 (M=16)",
            "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[m-sensitivity] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "int4_divergence_m_sensitivity_self_test_passes": int(bool(
            st["int4_divergence_m_sensitivity_self_test_passes"])),
        "projected_divergence_at_M16": proj["projected_divergence_at_M16"],
        "projected_identity_at_M16": proj["projected_identity_at_M16"],
        "bound_lower": proj["bound_lower"],
        "bound_upper": proj["bound_upper"],
        "bound_upper_bf16_floor_ref": proj["bound_upper_bf16_floor_ref"],
        "near_greedy_survives_at_M16": int(bool(proj["near_greedy_survives_at_M16"])),
        "strict_line_at_risk_in_upper_tail": int(bool(proj["strict_line_at_risk_in_upper_tail"])),
        "near_greedy_threshold_div": proj["near_greedy_threshold_div"],
        # anchors
        "div_m1_native_114": imp["div_m1_native"],
        "div_m8_clean_232": imp["div_m8_clean"],
        "identity_m8_232": imp["identity_m8_clean"],
        "int4_body_bitexact_m8": int(bool(imp["int4_body_bitexact_m8"])),
        "fp16_floor_div_221": imp["fp16_floor_div_221"],
        "drop_M1_to_M8": frame["drop_M1_to_M8"],
        # curve params + Curve B M=16 readings
        "curveB_p_powerlaw": curves["params"]["p_powerlaw"],
        "curveB_lam_exp": curves["params"]["lam_exp"],
        "curveB_M16_powerlaw": proj["curveB_M16_powerlaw"],
        "curveB_M16_exp": proj["curveB_M16_exp"],
        "m_tree_build": imp["m_tree_build"],
        "nan_clean": int(bool(payload["nan_clean"])),
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
        # the M-grid table as flat scalars for plotting.
        **{f"div_clean_at_M{r['M']}": r["divergence_clean"] for r in proj["table"]},
        **{f"identity_clean_at_M{r['M']}": r["identity_clean"] for r in proj["table"]},
        **{f"div_envelope_at_M{r['M']}": r["divergence_envelope_native"] for r in proj["table"]},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="int4_divergence_m_sensitivity_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[m-sensitivity] wandb logged {len(summary)} summary keys (run {run.id})", flush=True)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="issue192-reading-calibration")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at,
        "pr": 242,
        "agent": "lawine",
        "kind": "int4-divergence-m-sensitivity",
        "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }

    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[m-sensitivity] WARNING non-finite values at: {nan_paths}", flush=True)
    # fold nan-clean into self-test (e) and recompute PRIMARY.
    syn["self_test"]["conditions"]["e_key_scalars_finite"] = bool(
        syn["self_test"]["conditions"]["e_key_scalars_finite"] and payload["nan_clean"])
    passes = bool(all(syn["self_test"]["conditions"].values()))
    syn["self_test"]["int4_divergence_m_sensitivity_self_test_passes"] = passes

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "int4_divergence_m_sensitivity_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[m-sensitivity] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    print(f"  PRIMARY int4_divergence_m_sensitivity_self_test_passes = {passes}", flush=True)
    print(f"  TEST projected_divergence_at_M16 = {syn['test_metric']['projected_divergence_at_M16']:.6f}", flush=True)
    print(f"  HEADLINE near-greedy at M=16 = {syn['projection_M16']['near_greedy_verdict']}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        ok = passes and payload["nan_clean"]
        print(f"[m-sensitivity] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
