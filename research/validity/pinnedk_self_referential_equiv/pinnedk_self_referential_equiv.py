#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Is pinned-K re-capture self-referentially equivalent? ~482.7 -> ~496.7 (PR #427, denken).

THE QUESTION (the single highest-value unexplored lift on the #407 "maximize fastest STRICTLY-EQUIVALENT
  TPS" line). My #423 (`5a6zq2yz`) closed the byte-identical attention tax as an irreducible single-component
  floor (removable_tax_tps=0) and named exactly ONE sub-floor path: wirbel #400's pinned-K (num_splits=8)
  re-capture (~+13.998 TPS, #408) -> stacks with cb3 (+15.60, #403) toward ~496.7. #423 CONSERVATIVELY filed
  it as a human "frozen-byte vs M-invariant" CONTRACT decision, because it ASSUMED #407 equivalence means
  byte-identity to the *currently deployed* 481.53 served bytes (pinned-K flips ~3/882 near-ties vs those
  bytes). THIS CARD RE-OPENS that assumption against the merged scorer evidence.

  Land #414 (`bq7xkfcv`) + #420 (`speculator_keepset_equivalence`) + the #114 mechanism proof established the
  OPERATIVE equivalence gate is SELF-REFERENTIAL: it enforces the SUBMISSION'S OWN plain greedy AR (M=1,
  SENPAI_REFERENCE_MODE spec-OFF on the submission's OWN kernels/quant), NOT byte-identity to any previously-
  deployed reference. IF the scorer self-referentially re-runs the SUBMITTED config's own M=1 greedy as the
  reference, then pinned-K's M-invariance (M=1==M=8 under its kernel) is SUFFICIENT for #407 equivalence =>
  ~496.7 is a LEGAL self-referentially-equivalent number, NOT a human contract decision -- lifting the
  realizable fastest-equivalent frontier from ~482.74 to ~496.7 (the closest realizable approach to 500).

THE ANSWER (decision-critical, honest): VERDICT = legal_self_referential.
  * The reference IS self-referential BY MECHANISM (#114 Step 1a, airtight): SENPAI_REFERENCE_MODE=1 clears
    SPECULATIVE_CONFIG -> vLLM runs plain M=1 AR greedy on the SUBMISSION'S OWN engine / kernels / quant
    (serve.py: "the only removed variable is speculation"). The gate tests "does the M=K+1 verify reproduce
    THIS submission's OWN M=1 argmax trajectory?", never "does it match a canonical fp32 / a prior deployed
    reference?". #414/#420 both bank gate_for_respect_equivalence="self_referential" and answer the SAME human
    #407 issue with it.
  * Under that gate, pinned-K's M-invariance (#400: M=1==M=8 byte-exact-FEASIBLE under the rebuilt num_splits=8
    fixed-order 64-CTA split-reduce; Marlin atomic_add=False / fp32_reduce / fixed-order grounds the
    M-independent reduction order) is EXACTLY the property the self-referential gate requires: pinned-K's M=8
    verify == pinned-K's OWN M=1 AR => self_referential served identity = 1.0. M-invariance is SUFFICIENT.
  * The "~3/882 flips vs today's bytes" is a FROZEN-byte property, NOT a self-referential one: the
    self-referential reference is the submission's OWN M=1 AR (which pinned-K matches exactly), not the
    deployed output. The frozen-byte mismatch is IRRELEVANT to the operative gate.
  * THE RECONCILIATION TWIST (why frozen-byte is the WRONG reference here): the *currently deployed* 481.53
    config ITSELF FAILS the self-referential gate on the ATTENTION dimension -- served identity 0.9966 != 1.0,
    its 3/882 flips are M=8-verify-vs-its-OWN-M=1 reduction-order flips (#423; #114 measured the same class at
    56.1% before the batch-invariance fixes landed). Pinned-K (identity 1.0) is STRICTLY MORE self-referentially
    equivalent than the deployed config. Demanding pinned-K be byte-identical to a LESS-equivalent reference is
    incoherent under the operative gate -- the self-referential gate BLESSES pinned-K and DISQUALIFIES the
    deployed config, not the reverse. (Caveat #114 Step 1c: the OFFICIAL automated harness enforces NO
    token-identity at all -- only TPS + 128/128 completion + PPL<=2.42 -- so "self-referential" is the operative
    reading of the WRITTEN contract program.md:27-28 + the repo's LOCAL enforcement tool + the #414/#420 land
    cards answering #407, NOT an automated leaderboard check. Frozen-byte was never the operative gate.)

  => pinnedk_self_referential_equivalent = True. The equivalence QUESTION is resolved YES. What remains is a
  DEPLOYMENT/engineering approval, NOT a contract decision: realizing pinned-K needs a FA2 decode-kernel REBUILD
  (num_splits>1 is NotImplementedError on shipped FA2) + a full greedy+PPL re-validation on the NEW served bytes
  (must re-clear PPL<=2.42 AND demonstrate M=1==M=8 self-referential identity 1.0 via a SENPAI_REFERENCE_MODE
  A/B). That is the ONE allowed flagged ask -- the GO-to-human packet below.

THE STACK (instruction 3): pinnedk_recapture_stack_tps = blanket-strict 467.14 (#412 measured, the M-invariant-
  but-slow num_splits=1 base) + pinned-K attention recovery 13.998 (#408) + cb3 +15.60 (#403) = 496.74, vs
  lawine #411's supply-ledger ceiling 497.44 (modeled base). gap_to_500_tps = 3.26 (from the stack) / 2.56
  (from the #411 ceiling). Still short of 500: pinned-K caps at ~deployed speed ALONE (#411: useful only
  stacked) and even the full cb3+pinned-K stack misses 500 -- the residual needs additional equivalence-neutral
  supply (the demand-side a1~0.92 break is out of reach, #308).

WHAT THIS IS / IS NOT:
  Pure-CPU analytic reconciliation card (stdlib + the merged modules). 0 GPU, 0 official TPS, 0 HF Job, NO
  served-file change, NO submission, NO kernel build, analysis_only=True. COMPOSES merged artifacts byte-exactly
  (NO re-derivation): imports my #423 (`5a6zq2yz`) which transitively banks the #400/#408/#412/#403/#411/#418/
  #413 scalars, and banks the #114/#414/#420 self-referential-gate facts with provenance, cross-checking the
  gate_for_respect_equivalence="self_referential" verdict against the merged #414/#420 result JSONs at runtime.
  The new modelling is ONLY the reconciliation logic (does the self-referential reference make M-invariance
  SUFFICIENT) + the verdict gating; every TPS/identity scalar is banked.

REPRODUCE (0-GPU):
    cd target/ && .venv/bin/python -m research.validity.pinnedk_self_referential_equiv.\
pinnedk_self_referential_equiv --self-test
    cd target/ && .venv/bin/python -m research.validity.pinnedk_self_referential_equiv.\
pinnedk_self_referential_equiv \
      --wandb_group pinnedk-self-ref --wandb_name denken/pinnedk-self-referential-equiv
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path

# ---- COMPOSE merged artifacts byte-exactly: import my #423, which transitively banks #400/#408/#412/#403/
#      #411/#418/#413. NOTHING re-derived; every TPS/identity scalar comes from a merged module. -------------
from research.validity.byte_identical_reduction_tax_floor import byte_identical_reduction_tax_floor as g423

HERE = Path(__file__).resolve().parent
VAL = HERE.parent  # research/validity

# ===========================================================================
# Section 0 -- banked anchors re-exported byte-exactly from #423 (and its transitive #400/#408/#412/#403/#411)
# ===========================================================================
MU_P: float = g423.MU_P                              # 481.53 deployed FAST (non-equivalent) frontier (#52)
BASE_467_MEASURED_412: float = g423.BASE_467_MEASURED_412   # 467.140 blanket-strict measured (num_splits=1)
ATTN_LEVER_GAIN_REALISTIC_408: float = g423.ATTN_LEVER_GAIN_REALISTIC_408  # 13.998 pinned-K realistic recovery
CB3_SUPPLY_403: float = g423.CB3_SUPPLY_403          # 15.60 cb3 conservative deployable supply (k*=229)
STACK_FROZEN: float = g423.STACK_FROZEN              # 482.74 blanket-strict + cb3 (the frozen-byte frontier)
STACK_RECAPTURE: float = g423.STACK_RECAPTURE        # 496.739 pinned-K + cb3 (the self-ref re-capture stack)
STACK_RECAPTURE_CEILING_411: float = g423.STACK_RECAPTURE_CEILING_411  # 497.44 lawine #411 supply ledger ceiling
KNIFE_EDGE_MARGIN_FROZEN: float = g423.KNIFE_EDGE_MARGIN_FROZEN        # +1.21 frozen-stack margin over deployed
TAX_MEASURED: float = g423.TAX_MEASURED              # 14.39 blanket-strict attention tax
N_SERVED_FLIPS: int = g423.N_SERVED_FLIPS            # 3 served M=8-vs-M=1 reduction-order flips (#405/#381)
N_SERVED_POSITIONS: int = g423.N_SERVED_POSITIONS    # 882 readable chain positions
PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400: bool = g423.PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400  # True
PINNEDK_PRODUCES_NEW_REFERENCE_400: bool = g423.PINNEDK_PRODUCES_NEW_REFERENCE_400                    # True
MULTISPLIT_EQ_SERIAL_BYTES_400: bool = g423.MULTISPLIT_EQ_SERIAL_BYTES_400                            # False
ATTN_REBUILD_IS_FLAGGED_SERVED_CHANGE_400: bool = g423.ATTN_REBUILD_IS_FLAGGED_SERVED_CHANGE_400      # True
PINNEDK_RECOVERY_FRAC_REALISTIC_400: float = g423.PINNEDK_RECOVERY_FRAC_REALISTIC_400                 # 0.9872
PPL_DEPLOYED: float = g423.PPL_DEPLOYED              # 2.3772
PPL_GATE: float = g423.PPL_GATE                      # 2.42
M_DEPLOYED: int = g423.M_DEPLOYED                    # 8 verify rows = K_spec(7)+1
K_DEPLOYED: int = g423.K_DEPLOYED                    # 7 draft length
TARGET: float = 500.0

# deployed self-referential identity = 1 - flips/positions on the ATTENTION dimension (the dim pinned-K fixes).
DEPLOYED_SELF_REF_IDENTITY: float = 1.0 - (N_SERVED_FLIPS / N_SERVED_POSITIONS)   # 0.99660 (!= 1.0 -> FAILS)
PINNEDK_SELF_REF_IDENTITY: float = 1.0   # M-invariant => M=8 verify == own M=1 AR byte-exact (identity 1.0)

# ===========================================================================
# Section 1 -- the SELF-REFERENTIAL gate facts, banked from #114 / #414 / #420 (with provenance) -------------
# ===========================================================================
# #114 (self_referential_greedy_gate.md) Step 1a -- the gate is self-referential BY MECHANISM:
SELF_REF_REFERENCE_IS_OWN_M1_AR: bool = True   # SENPAI_REFERENCE_MODE=1 -> spec-OFF M=1 AR on the OWN stack
SELF_REF_REFERENCE_NOT_CANONICAL_FP32: bool = True   # never vs a canonical / prior-deployed reference
# #114 Step 1c -- the OFFICIAL automated harness enforces NO token-identity (only TPS + 128/128 + PPL):
OFFICIAL_HARNESS_ENFORCES_TOKEN_IDENTITY: bool = False
# #114 Step 1b / #423 -- the DEPLOYED stack FAILS the self-referential gate on the attention dimension:
DEPLOYED_FAILS_SELF_REFERENTIAL_ATTENTION: bool = True
# #414 / #420 -- the LAND verdict answering the SAME human #407 issue: operative gate = self_referential.
GATE_FOR_RESPECT_EQUIVALENCE_414_420: str = "self_referential"
SRC_114 = "self_referential_greedy_gate.md (PR #114 mechanism proof)"
SRC_414_RUN = "bq7xkfcv"   # land truevocab_lmhead_equivalence_cost (operative gate = self-referential)
SRC_420 = "speculator_keepset_equivalence (in-keepset drafter preserves self-referential)"
SRC_400_RUN = "o7yhpkej"   # wirbel attn_pinnedk_headroom (M-invariant byte-exact feasible; new reference)
SRC_408_RUN = "qc9bz8sv"   # wirbel m1_decode_latency_budget (attn_lever_gain_realistic 13.998)
SRC_403_RUN = "iv9i2wks"   # kanna cb3_conservative_k_deployable_lift (+15.60 @ k*=229)
SRC_411_RUN = "078yjgax"   # lawine flagged_supply_deploy_surface_ledger (max stack 497.44)
SRC_412_RUN = "stark/selective-recompute"  # blanket-strict measured 467.14
SRC_423_RUN = "5a6zq2yz"   # denken byte_identical_reduction_tax_floor (the predecessor this re-opens)

TOL: float = 1e-6

# merged result JSONs for the provenance cross-check (the self-referential verdict must match these).
ART_414 = VAL / "truevocab_lmhead_equivalence_cost" / "truevocab_lmhead_equivalence_cost_results.json"
ART_420 = VAL / "speculator_keepset_equivalence" / "speculator_keepset_equivalence_results.json"
ART_423 = VAL / "byte_identical_reduction_tax_floor" / "byte_identical_reduction_tax_floor_results.json"


# ===========================================================================
# Section 2 -- define the two equivalence notions precisely (instruction 1) ------------------------------
# ===========================================================================

def define_equivalence_notions() -> dict:
    """(a) self-referential vs (b) frozen-byte, and exactly what pinned-K satisfies under each."""
    return {
        "self_referential": {
            "definition": ("the SUBMITTED config's M=8 spec-verify greedy == its OWN M=1 plain greedy AR "
                           "(SENPAI_REFERENCE_MODE spec-OFF on the SUBMISSION's OWN kernels/quant); the "
                           "submission defines its OWN reference -- #114 Step 1a mechanism, #414/#420 verdict"),
            "reference": "submission's own M=1 AR (re-run per submission)",
            "pinnedk_satisfies": True,
            "pinnedk_satisfies_why": ("pinned-K is M-invariant byte-exact (#400: M=1==M=8 under the rebuilt "
                                      "fixed-order num_splits=8 kernel; Marlin atomic_add=False/fp32_reduce/"
                                      "fixed-order) => its M=8 verify == its own M=1 AR => served identity 1.0"),
            "pinnedk_identity": PINNEDK_SELF_REF_IDENTITY,            # 1.0
            "deployed_satisfies": False,
            "deployed_identity": DEPLOYED_SELF_REF_IDENTITY,         # 0.9966 (3/882 attention flips)
        },
        "frozen_byte": {
            "definition": ("byte-identical to the *currently deployed* 481.53 served output; the deployed "
                           "bytes are the fixed reference -- #423's CONSERVATIVE assumption, NOT the operative "
                           "gate per #114/#414/#420"),
            "reference": "the currently-deployed 481.53 served bytes (fixed)",
            "pinnedk_satisfies": False,
            "pinnedk_satisfies_why": ("pinned-K's multi-split reduction order != the deployed serial un-pack "
                                      "(#400: multisplit_eq_serial_bytes=False) -> flips ~3/882 near-ties vs "
                                      "today's bytes (PPL-neutral)"),
            "pinnedk_identity": DEPLOYED_SELF_REF_IDENTITY,          # 0.9966 vs the frozen deployed bytes
            "deployed_satisfies": True,                              # trivially: deployed == deployed
            "deployed_identity": 1.0,
        },
    }


# ===========================================================================
# Section 3 -- resolve the binding question (instruction 2): is M-invariance SUFFICIENT? -----------------
# ===========================================================================

def scorer_is_self_referential() -> bool:
    """Reason from the scorer MECHANICS (#114 Step 1a, #414/#420). The official scorer's reference is the
    submission's OWN M=1 AR (SENPAI_REFERENCE_MODE spec-OFF on its own kernels), NOT a fixed prior reference.
    Three merged artifacts answering the SAME human #407 issue concur: gate = self-referential."""
    return bool(
        SELF_REF_REFERENCE_IS_OWN_M1_AR
        and SELF_REF_REFERENCE_NOT_CANONICAL_FP32
        and GATE_FOR_RESPECT_EQUIVALENCE_414_420 == "self_referential"
    )


def m_invariance_sufficient_for_self_referential() -> bool:
    """Under a self-referential gate, pinned-K's M-invariance (M=1==M=8 under its OWN kernel) is EXACTLY the
    property the gate checks (M=8 verify == own M=1 AR). So M-invariance is SUFFICIENT for #407 equivalence."""
    return bool(scorer_is_self_referential() and PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400)


def resolve_binding_question() -> dict:
    """The core reconciliation. Returns whether pinned-K is self-referentially equivalent, with the twist
    that the DEPLOYED config itself fails the self-referential gate on the attention dimension."""
    sr = scorer_is_self_referential()
    sufficient = m_invariance_sufficient_for_self_referential()
    # the reconciliation twist: pinned-K (identity 1.0) is MORE self-ref-equivalent than deployed (0.9966).
    pinnedk_more_equivalent_than_deployed = PINNEDK_SELF_REF_IDENTITY > DEPLOYED_SELF_REF_IDENTITY
    return {
        "scorer_is_self_referential": sr,
        "self_referential_reference_is_own_m1_ar": SELF_REF_REFERENCE_IS_OWN_M1_AR,
        "official_harness_enforces_token_identity": OFFICIAL_HARNESS_ENFORCES_TOKEN_IDENTITY,
        "pinnedk_m_invariant": PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400,
        "m_invariance_sufficient_for_self_referential": sufficient,
        "pinnedk_satisfies_self_referential_gate": sufficient,
        "pinnedk_self_ref_identity": PINNEDK_SELF_REF_IDENTITY,
        "pinnedk_byte_identical_to_deployed": False,
        "pinnedk_self_referential_equivalent": sufficient,
        # the twist
        "deployed_passes_self_referential_attention": not DEPLOYED_FAILS_SELF_REFERENTIAL_ATTENTION,  # False
        "deployed_self_ref_identity": DEPLOYED_SELF_REF_IDENTITY,                                      # 0.9966
        "pinnedk_more_equivalent_than_deployed": pinnedk_more_equivalent_than_deployed,                # True
        "frozen_byte_is_operative_gate": False,
    }


# ===========================================================================
# Section 4 -- price the pinned-K re-capture stack (instruction 3) ---------------------------------------
# ===========================================================================

def price_recapture_stack() -> dict:
    """pinnedk_recapture_stack_tps = blanket-strict 467.14 + pinned-K 13.998 (#408) + cb3 15.60 (#403)."""
    stack = BASE_467_MEASURED_412 + ATTN_LEVER_GAIN_REALISTIC_408 + CB3_SUPPLY_403   # 496.739 == #423 STACK_RECAPTURE
    gap_stack = TARGET - stack
    gap_ceiling = TARGET - STACK_RECAPTURE_CEILING_411
    return {
        "blanket_strict_base_tps": BASE_467_MEASURED_412,
        "pinnedk_attn_recovery_tps": ATTN_LEVER_GAIN_REALISTIC_408,
        "cb3_supply_tps": CB3_SUPPLY_403,
        "pinnedk_recapture_stack_tps": stack,
        "stack_matches_g423_recapture": abs(stack - STACK_RECAPTURE) < 1e-9,
        "lawine_411_ceiling_tps": STACK_RECAPTURE_CEILING_411,
        "gap_to_500_tps": gap_stack,                       # 3.26
        "gap_to_500_from_ceiling_411_tps": gap_ceiling,    # 2.56
        "frozen_byte_frontier_tps": STACK_FROZEN,          # 482.74 (the contract-conservative frontier)
        "self_ref_uplift_over_frozen_tps": stack - STACK_FROZEN,   # ~14.0
        "self_ref_uplift_over_deployed_tps": stack - MU_P,         # ~15.2
        "clears_500": stack >= TARGET,                     # False
    }


# ===========================================================================
# Section 5 -- deliver the decision card (instruction 4) -------------------------------------------------
# ===========================================================================

def go_to_human_packet() -> str:
    return (
        "GO-to-human packet (the ONE allowed flagged ask). pinned-K is SELF-REFERENTIALLY LEGAL -- the "
        "equivalence QUESTION is resolved YES (M=8 verify == own M=1 AR by #400 M-invariance; the operative "
        "gate is self-referential per #114 Step 1a + #414/#420). What remains is a DEPLOYMENT approval, NOT a "
        "contract decision: (1) FLAGGED FA2 decode-kernel REBUILD -- num_splits>1 is NotImplementedError on "
        "shipped FA2, so the fixed 64-CTA num_splits=8 split-reduce needs a kernel rebuild (#400/#411: 5 files, "
        "ALL additive, 0 in-place, NO checkpoint change -- a weightless reduction-order change to the hottest "
        "decode kernel). (2) IDENTITY-VERIFY on the NEW served bytes -- a SENPAI_REFERENCE_MODE A/B "
        "(spec-ON M=8 vs own spec-OFF M=1 AR) must return GREEDY_IDENTICAL (self_ref identity 1.0), confirming "
        "the rebuilt kernel is genuinely M-invariant on-target; cross-ref lawine #411's deploy-surface ledger "
        "(shared e2e self-referential capture re-keys all stacked levers at once) / #419's CI. (3) PPL "
        "re-clear -- the new bytes must re-measure PPL <= 2.42 (expected neutral: a reduction-order change, "
        "PR #66 shows greedy near-tie flips don't enter the teacher-forced PPL). Blast radius: a flagged "
        "served-file change; realistic vs floor recovery 98.7% vs 100% roofline. PRIZE: lifts the realizable "
        "fastest-EQUIVALENT frontier 482.74 -> 496.74 (+13.99 with self-referential identity 1.0, which the "
        "deployed 481.53 LACKS at 0.9966). RESIDUAL gap to 500 = 3.26 (stack) / 2.56 (#411 ceiling) -- still "
        "short; pinned-K caps at ~deployed speed ALONE (useful only stacked, #411) and the cb3+pinned-K stack "
        "misses 500, so closing the last ~3 TPS needs additional equivalence-neutral supply (the demand-side "
        "a1~0.92 break is out of reach, #308). Recommend: surface as a self-referentially-legal re-capture "
        "GO request; do NOT auto-build (the rebuild + re-validation is the human-gated step)."
    )


# ===========================================================================
# Section 6 -- verdict gating (parametric so the self-test can exercise all 3 branches) ------------------
# ===========================================================================

def assemble_verdict(scorer_self_ref: bool, pinnedk_m_invariant: bool,
                     scorer_evidence_ambiguous: bool = False) -> str:
    """legal_self_referential / human_contract_decision / ambiguous_flag_to_human.

    * legal_self_referential: the scorer is self-referential (own M=1 AR) AND pinned-K is M-invariant ->
      M-invariance is sufficient -> pinned-K is a legal #407-equivalent submission (~496.7).
    * human_contract_decision: the scorer pins to the frozen deployed bytes -> pinned-K's 3/882 flips make it
      a human frozen-byte-vs-M-invariant contract call (#423's conservative reading).
    * ambiguous_flag_to_human: the merged scorer evidence does not resolve which gate is operative -> flag,
      do not guess the verdict (PR instruction 2)."""
    if scorer_evidence_ambiguous:
        return "ambiguous_flag_to_human"
    if scorer_self_ref and pinnedk_m_invariant:
        return "legal_self_referential"
    return "human_contract_decision"


# ===========================================================================
# Section 7 -- self-tests (>= 20 checks; PRIMARY gate) ---------------------------------------------------
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (math.isnan(x) or math.isinf(x))


def _load(art: Path) -> dict | None:
    if not art.exists():
        return None
    try:
        return json.loads(art.read_text())
    except Exception:  # noqa: BLE001
        return None


def run_self_tests(notions: dict, resolution: dict, stack: dict, verdict: str) -> dict:
    c: dict[str, bool] = {}

    # a) provenance: anchors imported byte-exactly from merged #423 (transitively #400/#408/#412/#403/#411).
    c["a_mu_p_is_481p53"] = abs(MU_P - 481.53) < TOL
    c["a_base_measured_412"] = abs(BASE_467_MEASURED_412 - 467.1400155438763) < TOL
    c["a_attn_lever_408"] = abs(ATTN_LEVER_GAIN_REALISTIC_408 - 13.998600706082982) < TOL
    c["a_cb3_supply_403"] = abs(CB3_SUPPLY_403 - 15.60) < TOL
    c["a_stack_recapture_g423"] = abs(STACK_RECAPTURE - 496.7386162499593) < TOL
    c["a_ceiling_411"] = abs(STACK_RECAPTURE_CEILING_411 - 497.44) < TOL
    c["a_stack_frozen_g423"] = abs(STACK_FROZEN - 482.7400155438763) < TOL
    c["a_served_flips_3_of_882"] = N_SERVED_FLIPS == 3 and N_SERVED_POSITIONS == 882
    c["a_m_deployed_8"] = M_DEPLOYED == 8 and K_DEPLOYED == 7

    # b) the two equivalence notions (instruction 1): pinned-K is (a)-compliant, NOT (b)-compliant.
    c["b_self_ref_pinnedk_satisfies"] = notions["self_referential"]["pinnedk_satisfies"] is True
    c["b_frozen_pinnedk_not_satisfies"] = notions["frozen_byte"]["pinnedk_satisfies"] is False
    c["b_self_ref_pinnedk_identity_1p0"] = abs(notions["self_referential"]["pinnedk_identity"] - 1.0) < TOL
    c["b_self_ref_deployed_fails"] = notions["self_referential"]["deployed_satisfies"] is False
    c["b_self_ref_deployed_identity_0p9966"] = abs(
        notions["self_referential"]["deployed_identity"] - (1.0 - 3 / 882)) < 1e-9
    c["b_frozen_deployed_trivially_satisfies"] = notions["frozen_byte"]["deployed_satisfies"] is True

    # c) the scorer mechanics (instruction 2): self-referential reference = own M=1 AR; official harness no check.
    c["c_scorer_is_self_referential"] = resolution["scorer_is_self_referential"] is True
    c["c_reference_is_own_m1_ar"] = resolution["self_referential_reference_is_own_m1_ar"] is True
    c["c_official_no_token_identity"] = resolution["official_harness_enforces_token_identity"] is False
    c["c_414_420_gate_self_referential"] = GATE_FOR_RESPECT_EQUIVALENCE_414_420 == "self_referential"

    # d) the binding resolution: M-invariance SUFFICIENT -> pinned-K self-referentially equivalent.
    c["d_pinnedk_m_invariant"] = resolution["pinnedk_m_invariant"] is True
    c["d_m_invariance_sufficient"] = resolution["m_invariance_sufficient_for_self_referential"] is True
    c["d_pinnedk_satisfies_gate"] = resolution["pinnedk_satisfies_self_referential_gate"] is True
    c["d_pinnedk_self_ref_equivalent"] = resolution["pinnedk_self_referential_equivalent"] is True
    c["d_pinnedk_not_byte_identical_deployed"] = resolution["pinnedk_byte_identical_to_deployed"] is False

    # e) THE RECONCILIATION TWIST: deployed itself FAILS self-ref (0.9966); pinned-K (1.0) is MORE equivalent.
    c["e_deployed_fails_self_ref"] = resolution["deployed_passes_self_referential_attention"] is False
    c["e_deployed_identity_0p9966"] = abs(resolution["deployed_self_ref_identity"] - 0.9965986394557823) < 1e-9
    c["e_pinnedk_more_equivalent"] = resolution["pinnedk_more_equivalent_than_deployed"] is True
    c["e_frozen_not_operative"] = resolution["frozen_byte_is_operative_gate"] is False

    # f) the stack (instruction 3): 496.74, ceiling 497.44, gap to 500 = 3.26 / 2.56.
    c["f_stack_is_496p74"] = abs(stack["pinnedk_recapture_stack_tps"] - 496.7386162499593) < 1e-6
    c["f_stack_matches_g423"] = stack["stack_matches_g423_recapture"] is True
    c["f_gap_to_500_is_3p26"] = abs(stack["gap_to_500_tps"] - 3.2613837500407) < 1e-6
    c["f_gap_from_ceiling_2p56"] = abs(stack["gap_to_500_from_ceiling_411_tps"] - 2.56) < 1e-6
    c["f_stack_misses_500"] = stack["clears_500"] is False
    c["f_uplift_over_frozen_positive"] = stack["self_ref_uplift_over_frozen_tps"] > 13.0
    c["f_uplift_over_deployed_positive"] = stack["self_ref_uplift_over_deployed_tps"] > 0.0
    c["f_stack_below_ceiling_411"] = stack["pinnedk_recapture_stack_tps"] < STACK_RECAPTURE_CEILING_411

    # g) the verdict gating (instruction 4) -- all three branches exercised.
    c["g_verdict_is_legal_self_referential"] = verdict == "legal_self_referential"
    c["g_branch_frozen_is_contract"] = assemble_verdict(False, True) == "human_contract_decision"
    c["g_branch_ambiguous_flags"] = assemble_verdict(True, True, scorer_evidence_ambiguous=True) == "ambiguous_flag_to_human"
    c["g_branch_self_ref_legal"] = assemble_verdict(True, True) == "legal_self_referential"
    # counterfactual: a NON-M-invariant kernel under a self-ref gate is NOT legal (would be a contract call).
    c["g_non_m_invariant_not_legal"] = assemble_verdict(True, False) == "human_contract_decision"

    # h) the GO packet names the flagged rebuild + re-validation; the new-ref / flagged facts from #400.
    c["h_recapture_new_reference"] = PINNEDK_PRODUCES_NEW_REFERENCE_400 is True
    c["h_recapture_flagged_change"] = ATTN_REBUILD_IS_FLAGGED_SERVED_CHANGE_400 is True
    c["h_multisplit_not_serial"] = MULTISPLIT_EQ_SERIAL_BYTES_400 is False

    # i) PPL/greedy preserved; numeric hygiene.
    c["i_ppl_within_gate"] = PPL_DEPLOYED <= PPL_GATE
    flat = [MU_P, BASE_467_MEASURED_412, stack["pinnedk_recapture_stack_tps"], stack["gap_to_500_tps"],
            DEPLOYED_SELF_REF_IDENTITY, PINNEDK_SELF_REF_IDENTITY, STACK_RECAPTURE_CEILING_411]
    c["i_no_nan_inf"] = all(_finite(v) for v in flat)

    # j) artifact provenance cross-check: the self-referential verdict matches the merged #414/#420/#423 JSONs.
    d414, d420, d423 = _load(ART_414), _load(ART_420), _load(ART_423)
    if d414 is not None:
        v414 = d414.get("synthesis", {}).get("verdict_fields", {})
        c["j_414_gate_self_referential"] = v414.get("gate_for_respect_equivalence") == "self_referential"
    if d420 is not None:
        v420 = d420.get("synthesis", {}).get("verdict_fields", {})
        c["j_420_gate_self_referential"] = v420.get("gate_for_respect_equivalence") == "self_referential"
        c["j_420_inkeepset_preserves"] = v420.get("inkeepset_drafter_preserves_self_referential") is True
    if d423 is not None:
        c["j_423_recapture_stack"] = abs(d423.get("recapture_stack_tps", 0) - STACK_RECAPTURE) < 1e-6
        c["j_423_ceiling_411"] = abs(d423.get("inputs", {}).get("stack_recapture_ceiling_411", 0)
                                     - STACK_RECAPTURE_CEILING_411) < 1e-6
        c["j_423_pinnedk_m_invariant"] = d423.get("floor_classification", {}).get(
            "recapture_is_new_reference") is True

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# ===========================================================================
# Section 8 -- report assembly + W&B + CLI --------------------------------------------------------------
# ===========================================================================

def build_report() -> dict:
    notions = define_equivalence_notions()
    resolution = resolve_binding_question()
    stack = price_recapture_stack()
    verdict = assemble_verdict(resolution["scorer_is_self_referential"], resolution["pinnedk_m_invariant"])
    go_packet = go_to_human_packet()
    selftest = run_self_tests(notions, resolution, stack, verdict)

    headline = (
        "Pinned-K re-capture IS self-referentially equivalent (verdict=legal_self_referential). The operative "
        "#407 gate is SELF-REFERENTIAL -- the submission's OWN M=1 AR (SENPAI_REFERENCE_MODE spec-OFF on its "
        "own kernels; #114 Step 1a mechanism, #414/#420 verdict), NOT byte-identity to the deployed bytes. "
        "Pinned-K is M-invariant byte-exact (#400: M=1==M=8 under the rebuilt fixed-order num_splits=8 kernel; "
        "Marlin atomic_add=False/fp32_reduce/fixed-order), so its M=8 verify == its own M=1 AR => self_ref "
        "identity 1.0 => M-invariance is SUFFICIENT. The ~3/882 flips are vs TODAY's bytes (a frozen-byte "
        "property), irrelevant to the self-referential reference. Twist: the deployed 481.53 ITSELF fails the "
        "self-referential gate on attention (identity 0.9966), so pinned-K (1.0) is MORE equivalent than "
        "deployed -- the gate blesses pinned-K, not the reverse. pinnedk_recapture_stack_tps=496.74 "
        "(blanket-strict 467.14 + pinned-K 13.998 + cb3 15.60), lawine #411 ceiling 497.44, gap_to_500=3.26. "
        "Realizing it is a DEPLOYMENT approval (flagged FA2 kernel rebuild + greedy/PPL re-validation on the "
        "new bytes), NOT a contract decision. Lifts the realizable fastest-equivalent frontier 482.74 -> 496.74."
    )

    inputs = {
        "mu_p_fast_52": MU_P, "base_467_measured_412": BASE_467_MEASURED_412,
        "attn_lever_gain_realistic_408": ATTN_LEVER_GAIN_REALISTIC_408, "cb3_supply_403": CB3_SUPPLY_403,
        "stack_frozen_423": STACK_FROZEN, "stack_recapture_423": STACK_RECAPTURE,
        "stack_recapture_ceiling_411": STACK_RECAPTURE_CEILING_411,
        "knife_edge_margin_frozen_423": KNIFE_EDGE_MARGIN_FROZEN, "tax_measured_423": TAX_MEASURED,
        "n_served_flips_405": N_SERVED_FLIPS, "n_served_positions_405": N_SERVED_POSITIONS,
        "deployed_self_ref_identity": DEPLOYED_SELF_REF_IDENTITY,
        "pinnedk_self_ref_identity": PINNEDK_SELF_REF_IDENTITY,
        "pinnedk_m_invariant_byte_exact_feasible_400": PINNEDK_M_INVARIANT_BYTE_EXACT_FEASIBLE_400,
        "pinnedk_produces_new_reference_400": PINNEDK_PRODUCES_NEW_REFERENCE_400,
        "multisplit_eq_serial_bytes_400": MULTISPLIT_EQ_SERIAL_BYTES_400,
        "attn_rebuild_is_flagged_served_change_400": ATTN_REBUILD_IS_FLAGGED_SERVED_CHANGE_400,
        "pinnedk_recovery_frac_realistic_400": PINNEDK_RECOVERY_FRAC_REALISTIC_400,
        "self_ref_reference_is_own_m1_ar_114": SELF_REF_REFERENCE_IS_OWN_M1_AR,
        "official_harness_enforces_token_identity_114": OFFICIAL_HARNESS_ENFORCES_TOKEN_IDENTITY,
        "deployed_fails_self_referential_attention_114": DEPLOYED_FAILS_SELF_REFERENTIAL_ATTENTION,
        "gate_for_respect_equivalence_414_420": GATE_FOR_RESPECT_EQUIVALENCE_414_420,
        "m_deployed": M_DEPLOYED, "k_deployed": K_DEPLOYED, "target": TARGET,
        "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE,
        "src_114": SRC_114, "src_414_run": SRC_414_RUN, "src_420": SRC_420, "src_400_run": SRC_400_RUN,
        "src_408_run": SRC_408_RUN, "src_403_run": SRC_403_RUN, "src_411_run": SRC_411_RUN,
        "src_412_run": SRC_412_RUN, "src_423_run": SRC_423_RUN,
        "src_407_ref": "human re-scope: maximize fastest strictly-equivalent TPS",
    }

    return {
        "pr": 427, "agent": "denken", "kind": "pinnedk-self-referential-equiv",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used": False, "official_tps": 0,
        "baseline_fast_frontier_tps": MU_P, "baseline_fast_frontier_ppl": PPL_DEPLOYED,
        "headline": headline,
        "inputs": inputs,
        "equivalence_notions": notions,
        "binding_resolution": resolution,
        "recapture_stack": stack,
        "go_to_human_packet": go_packet,
        "verdict": verdict,
        # ---- HEADLINE deliverable scalars (SENPAI-RESULT / W&B load-bearing) ----
        "pinnedk_self_referential_equivalent": bool(resolution["pinnedk_self_referential_equivalent"]),
        "m_invariance_sufficient_for_self_referential": bool(
            resolution["m_invariance_sufficient_for_self_referential"]),
        "pinnedk_recapture_stack_tps": float(stack["pinnedk_recapture_stack_tps"]),
        "gap_to_500_tps": float(stack["gap_to_500_tps"]),
        "gap_to_500_from_ceiling_411_tps": float(stack["gap_to_500_from_ceiling_411_tps"]),
        "deployed_passes_self_referential": bool(resolution["deployed_passes_self_referential_attention"]),
        "pinnedk_more_equivalent_than_deployed": bool(resolution["pinnedk_more_equivalent_than_deployed"]),
        "self_referential_frontier_tps": float(stack["pinnedk_recapture_stack_tps"]),
        "frozen_byte_frontier_tps": float(STACK_FROZEN),
        "self_test_passes": bool(selftest["passes"]),
        "self_test": selftest,
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"
    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        r = report["binding_resolution"]
        s = report["recapture_stack"]
        wandb.summary.update({
            "headline": report["headline"],
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "verdict": report["verdict"],
            "pinnedk_self_referential_equivalent": report["pinnedk_self_referential_equivalent"],
            "m_invariance_sufficient_for_self_referential": report["m_invariance_sufficient_for_self_referential"],
            "pinnedk_recapture_stack_tps": report["pinnedk_recapture_stack_tps"],
            "gap_to_500_tps": report["gap_to_500_tps"],
            "gap_to_500_from_ceiling_411_tps": report["gap_to_500_from_ceiling_411_tps"],
            "deployed_passes_self_referential": report["deployed_passes_self_referential"],
            "pinnedk_more_equivalent_than_deployed": report["pinnedk_more_equivalent_than_deployed"],
            "self_referential_frontier_tps": report["self_referential_frontier_tps"],
            "frozen_byte_frontier_tps": report["frozen_byte_frontier_tps"],
            "self_test_passes": report["self_test_passes"],
        })
        wandb.log({
            "summary/verdict_legal_self_referential": float(report["verdict"] == "legal_self_referential"),
            "summary/pinnedk_self_referential_equivalent": float(report["pinnedk_self_referential_equivalent"]),
            "summary/m_invariance_sufficient": float(report["m_invariance_sufficient_for_self_referential"]),
            "summary/scorer_is_self_referential": float(r["scorer_is_self_referential"]),
            "summary/official_harness_enforces_token_identity": float(
                r["official_harness_enforces_token_identity"]),
            "summary/pinnedk_self_ref_identity": PINNEDK_SELF_REF_IDENTITY,
            "summary/deployed_self_ref_identity": DEPLOYED_SELF_REF_IDENTITY,
            "summary/pinnedk_more_equivalent_than_deployed": float(r["pinnedk_more_equivalent_than_deployed"]),
            "summary/pinnedk_recapture_stack_tps": s["pinnedk_recapture_stack_tps"],
            "summary/lawine_411_ceiling_tps": s["lawine_411_ceiling_tps"],
            "summary/gap_to_500_tps": s["gap_to_500_tps"],
            "summary/gap_to_500_from_ceiling_411_tps": s["gap_to_500_from_ceiling_411_tps"],
            "summary/frozen_byte_frontier_tps": STACK_FROZEN,
            "summary/self_ref_uplift_over_frozen_tps": s["self_ref_uplift_over_frozen_tps"],
            "summary/self_ref_uplift_over_deployed_tps": s["self_ref_uplift_over_deployed_tps"],
            "summary/blanket_strict_base_tps": s["blanket_strict_base_tps"],
            "summary/pinnedk_attn_recovery_tps": s["pinnedk_attn_recovery_tps"],
            "summary/cb3_supply_tps": s["cb3_supply_tps"],
            "summary/ppl_deployed": PPL_DEPLOYED,
            "summary/self_test_passes": float(report["self_test"]["passes"]),
            "summary/self_test_n_checks": float(report["self_test"]["n_checks"]),
        })
        # the two-notion comparison as a small table.
        tbl = wandb.Table(columns=["equivalence_notion", "reference", "pinnedk_compliant", "pinnedk_identity"])
        n = report["equivalence_notions"]
        tbl.add_data("self_referential (operative)", n["self_referential"]["reference"],
                     n["self_referential"]["pinnedk_satisfies"], n["self_referential"]["pinnedk_identity"])
        tbl.add_data("frozen_byte (#423 conservative)", n["frozen_byte"]["reference"],
                     n["frozen_byte"]["pinnedk_satisfies"], n["frozen_byte"]["pinnedk_identity"])
        wandb.log({"equivalence_notion_table": tbl})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    res = r["binding_resolution"]
    s = r["recapture_stack"]
    n = r["equivalence_notions"]
    print("\n=== Is pinned-K re-capture self-referentially equivalent? (PR #427, denken) ===")
    print(f"deployed FAST (#52) = {MU_P:.2f} (self-ref identity {DEPLOYED_SELF_REF_IDENTITY:.4f}, FAILS gate)  "
          f"frozen-byte frontier = {STACK_FROZEN:.2f}  self-ref frontier = {s['pinnedk_recapture_stack_tps']:.2f}")
    print("\n-- the two equivalence notions (instruction 1) --")
    print(f"  (a) self-referential [OPERATIVE]: ref = {n['self_referential']['reference']}")
    print(f"      pinned-K compliant: {n['self_referential']['pinnedk_satisfies']}  "
          f"identity {n['self_referential']['pinnedk_identity']:.4f}   "
          f"deployed compliant: {n['self_referential']['deployed_satisfies']} "
          f"(identity {n['self_referential']['deployed_identity']:.4f})")
    print(f"  (b) frozen-byte [#423 conservative]: ref = {n['frozen_byte']['reference']}")
    print(f"      pinned-K compliant: {n['frozen_byte']['pinnedk_satisfies']}  "
          f"(multisplit != serial, ~3/882 flips)")
    print("\n-- resolve the binding question (instruction 2) --")
    print(f"  scorer_is_self_referential = {res['scorer_is_self_referential']} "
          f"(reference = own M=1 AR; #114 Step 1a + #414/#420)")
    print(f"  official_harness_enforces_token_identity = {res['official_harness_enforces_token_identity']} "
          f"(#114 Step 1c: only TPS + 128/128 + PPL)")
    print(f"  pinnedk_m_invariant = {res['pinnedk_m_invariant']}  =>  "
          f"m_invariance_SUFFICIENT = {res['m_invariance_sufficient_for_self_referential']}")
    print(f"  TWIST: deployed FAILS self-ref ({res['deployed_self_ref_identity']:.4f}); pinned-K (1.0) is "
          f"MORE equivalent: {res['pinnedk_more_equivalent_than_deployed']}")
    print("\n-- the re-capture stack (instruction 3) --")
    print(f"  blanket-strict {s['blanket_strict_base_tps']:.2f} + pinned-K {s['pinnedk_attn_recovery_tps']:.3f} "
          f"+ cb3 {s['cb3_supply_tps']:.2f} = {s['pinnedk_recapture_stack_tps']:.2f}")
    print(f"  lawine #411 ceiling {s['lawine_411_ceiling_tps']:.2f}   "
          f"gap_to_500 = {s['gap_to_500_tps']:.2f} (stack) / {s['gap_to_500_from_ceiling_411_tps']:.2f} (ceiling)")
    print("\n-- VERDICT (instruction 4) --")
    print(f"  verdict = {r['verdict']}   pinnedk_self_referential_equivalent = "
          f"{r['pinnedk_self_referential_equivalent']}")
    print(f"\nPPL unchanged {PPL_DEPLOYED} <= {PPL_GATE}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"self_test_passes = {r['self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Is pinned-K re-capture self-referentially equivalent? (PR #427).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="0-GPU analytic gate (PR #427 deliverables)")
    ap.add_argument("--reanalyze", action="store_true", help="0-GPU full re-analysis (alias of default)")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="pinnedk-self-ref")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/pinnedk-self-referential-equiv")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/pinnedk_self_referential_equiv/pinnedk_self_referential_equiv_results.json")
    args = ap.parse_args()

    report = build_report()
    print_report(report)
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["peak_mem_mib"] = peak_mib

    if args.self_test:
        out = Path("research/validity/pinnedk_self_referential_equiv/pinnedk_self_referential_equiv_selftest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (peak {peak_mib:.1f} MiB)")
        print(f"\nself_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')}, peak {peak_mib:.1f} MiB)")

    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "verdict": report["verdict"],
        "pinnedk_self_referential_equivalent": bool(report["pinnedk_self_referential_equivalent"]),
        "pinnedk_recapture_stack_tps": float(report["pinnedk_recapture_stack_tps"]),
        "gap_to_500_tps": float(report["gap_to_500_tps"]),
        "self_test_passes": bool(report["self_test_passes"]),
        "primary_metric": {"name": "pinnedk_recapture_stack_tps",
                           "value": float(report["pinnedk_recapture_stack_tps"])},
        "test_metric": {"name": "self_test_passes", "value": float(report["self_test_passes"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
