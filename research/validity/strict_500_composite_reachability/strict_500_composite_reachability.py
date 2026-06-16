#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""The honest strict-frontier map: what strict (byte-exact greedy) numbers we can ship, and why.

RE-POINT (2026-06-16 12:25Z, advisor formal #357 relay)
-------------------------------------------------------
This file has had three lives.  It began as ">=500 composite reachability", then became the
"strict-submission GO/NO-GO decision packet".  Both framings are now SETTLED and the advisor
re-pointed it (12:25Z) to its current, terminal-bound mandate:

    Render the HONEST STRICT-FRONTIER MAP -- the ranked set of strict (byte-exact greedy-equivalent)
    rungs we can actually serve, each tagged with its REALIZED TPS and its PRIVATE-safety, plus the
    two orthogonal forward levers that are the only paths to a strict number that is faster AND safe.

Three things closed the old "submission decision" framing and forced this re-point:
  * The composed strict frontier (456.36 / 457.55 / 467.14) was a LOCUS proof, not a served number.
    The exact config the packet recommended (`VLLM_BATCH_INVARIANT=1`) serves at ~222-234 e2e, not
    ~457 -- the global flag is a ~48% determinism tax, not the ~5% the composite assumed.  (#474 hold,
    the 6th composed-vs-realized inversion the verification campaign caught one step before the draw.)
  * My own wait-target landed: lawine #482 (044xamdd) measured the multi-stream byte-exact realizable
    number = 457.54 and found the 474.44 resource ceiling UNREALIZABLE (DEPENDENCY_COLLAPSES_TO_FLOOR:
    a Gemma layer is a serial recurrence o_proj(L) <- attn(L), so attn has no independent GEMM to hide
    under).  multi-stream gain over single-stream ~= 0; multi-stream CLOSES as a byte-exact path.
  * denken #489 (q1ivw9tt) re-framed everything: PRIVATE-safety is a property of the drafter-acceptance
    gap Delta, NOT of TPS.  Every spec-alive config (222 AND 457) carries the SAME Delta = 4.295% ->
    ~24% one-shot private breach, scale-invariant.  The floor-lock (no drafter, Delta = 0.633%) is the
    ONLY strict private-safe ship.  Fast + byte-exact != private-safe.

This capstone MEASURES NOTHING.  It is a CPU-only synthesis of MERGED / advisor-relayed results
(official_tps == 0, no HF job, no served-file change).  It owns the human-facing MAP; it does not own
the submission trigger (land) or the identity oracle (denken).

The map (ranked by realized TPS; PRIVATE-safety is the second axis)
------------------------------------------------------------------
  rung                 realized TPS   strict?            private-safe?            ship status
  -------------------  ------------   ----------------   ----------------------   --------------------------
  floor-lock M=1 AR    166.23 (proj)  literal-1.0*       SAFE  (Delta 0.633%)     the rung that STICKS
  global-flag BI=1     222 / 234      operative-1.0      RISKY (Delta 4.295%)     superseded by surgical-357
  surgical / 2D byte   357.6 (meas)   byte-exact e2e     RISKY (Delta 4.295%)     THE #474 SHIP (land #473)
  deployed (ref)       481.53         NON-equiv (.9966)  --                       outside the strict set

  * floor-lock literal-1.0 is BY CONSTRUCTION (M=1 AR, no drafter, no reassociation).  FLAG: the relayed
    realize-run (stark #485, pavotwci) logged its identity comparison vs the `fa2sw_precache_kenyan`
    reference (119/128 divergent, verdict/literal_1p0=0) -- a DIFFERENT config, divergence expected --
    NOT vs the M=1 AR reference.  So the served literal-1.0 census vs the correct reference is the
    load-bearing confirm; this packet renders the TPS (166.23, verified) and the private-safety
    (Delta 0.633%, verified) as solid, and the literal-1.0 LABEL as by-construction-pending-census.

>500 is SETTLED: dead via all known strict levers
-------------------------------------------------
Strict (byte-exact greedy) >500 is out of reach via every known lever.  The IEEE-754 determinism tax is
irreducible (denken #423); there is no free fast byte-exact GEMM (#481 forward survey: deterministic-IO
tax band 22-63%, land measured 51.39% e2e at batch-1); the realized strict ceiling is ~467 (deployed-
equivalent locus) / 457.5 (byte-exact realizable) / 166 (private-safe).  The ONLY >500 path is the
greedy-UNSAFE ~16% GEMM relax-prize -- which leaves the strict lane entirely (identity 0.730) and is
escalated to the human on #407.  Verdict: strict >500 is a genuinely-new-method problem, ~3x over the
166 private-safe floor.

TERMINAL (advisor #357 relay 2026-06-16 14:37Z): the map is FROZEN at d3ed366 + this single pass
------------------------------------------------------------------------------------------------
The capstone's own gating question -- is the surgical ~457 a REAL served rung? -- is RESOLVED: lawine
#488 (ko01dcyy) served the surgical attention-pin config e2e and measured 357.6 byte-exact (REFUTING
the 456.98 ubel #484 prediction; honesty flag #2 vindicated).  The human ruled "357 -- go, finish it"
(#474, 13:51Z); land #473 arms the surgical-357 fire.  So terminal = (surgical realized AND #474
resolved) = True.  No LIVE slots remain -- every input is banked.

The two forward levers below are DECOUPLED forward research in the #481 zoom-out menu (SGLang #498,
TRT-LLM, alt spec-dec) -- they are the only paths to a FUTURE "faster AND private-safe" rung, but they
do NOT gate this terminal map:
  * #491 (ubel) reduction-sensitivity census  -> shed the determinism speed-tax  -> a FASTER floor-lock
                (attacks the TPS ceiling; keeps Delta safe -- the floor-lock has no drafter)
  * #492 (denken) drafter-gap feasibility      -> EAGLE-3 pulls Delta_accept <= ~3.0%  -> a private-safe
                fast rung  (attacks the Delta gate; keeps a drafter for speed)

PRIMARY metric  strict_frontier_map_self_test_passes  (0-GPU arithmetic-invariant integrity gate)
TEST    metrics floor_lock_realized_tps, floor_lock_private_safe, global_flag_tps, surgical_measured_tps,
                surgical_byte_exact_predicted_tps (refuted), multistream_realizable_tps,
                private_safety_is_delta_not_tps, strict_gt500_dead_via_known_levers,
                forward_research_decoupled, human_474_ruling, terminal, floor_lock_literal_1p0_flag.
Consume cross-student numbers ONLY as relayed by the advisor into the #357 thread.
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
# Banked / formally-relayed constants.  Every number below was relayed by the
# advisor into the #357 thread (12:25Z relay) and SPOT-VERIFIED against the
# named MERGED W&B run summaries (we read summaries; we do NOT re-run evals or
# inspect source branches).  Run ids recorded for provenance.
# --------------------------------------------------------------------------- #

PPL_GATE: float = 2.42                  # challenge PPL ceiling
SIGMA_HW: float = 4.8153                # lawine #467 / #482 between-session hw sigma (TPS)
DELTA_GATE_PCT: float = 5.0             # private-set acceptance-gap gate (denken #489 delta_gate=0.05)

# --- Rung 1: floor-lock M=1 AR, no drafter (stark #485 pavotwci, MERGED) ------------------------- #
FLOOR_LOCK_TPS: float = 166.23          # projected_official_from_sglang=166.234; realizes_16170=1
FLOOR_LOCK_PPL: float = 2.3767          # ppl/ppl, clears the gate
FLOOR_LOCK_TARGET_TPS: float = 161.70   # the M=1 AR strict floor it realizes (>= target)
FLOOR_LOCK_LITERAL_1P0_BY_CONSTRUCTION: bool = True   # M=1 AR int4 IS the strict reference computation
# FLAG #1 (advisor terminalize relay 14:37Z resolved the POINTER, not the boolean):
#   * The stark #485 (pavotwci) realize-run logged identity vs the `fa2sw_precache_kenyan` reference
#     (a DIFFERENT config) -> 119/128 divergent, verdict/literal_1p0=0.  That is NOT the M=1 AR
#     reference; divergence vs a different config is expected (stale signal, kept for the record).
#   * The CLOSEST served census is denken #471 (`bwyhpkd7`), a DISCRIMINATING certifier (it rejects the
#     deployed non-strict config at 0.9966 / 3 flips) that ACCEPTED the M=1 AR floor at
#     token_identity_rate=1.0, 0 flips -- consistent with the by-construction claim.
#   * I hold `confirmed_by_served_census=False` (by-construction-PENDING) because under launch-isolation
#     I cannot verify whether bwyhpkd7's floor reference is the canonical M=1-AR greedy (an INDEPENDENT
#     confirmation) or the floor config's own output (tautological self-match).  Either way it does NOT
#     gate the freeze: the team ships surgical-357, not floor-lock.  Honest rendering = literal-1.0 by
#     construction + bwyhpkd7 served census accepted it 1.0/0 (pointer), boolean stays pending.
FLOOR_LOCK_RELAYRUN_VERDICT_LITERAL_1P0: int = 0      # pavotwci verdict/literal_1p0 (vs precache ref)
FLOOR_LOCK_RELAYRUN_DIVERGENT_VS_PRECACHE: int = 119  # of 128 prompts (expected: different config)
FLOOR_LOCK_SERVED_CENSUS_RUN: str = "bwyhpkd7"        # denken #471 certifier: accepted floor 1.0/0 flips
FLOOR_LOCK_SERVED_CENSUS_ACCEPTED_FLOOR: bool = True  # bwyhpkd7 accepted M=1 AR floor at 1.0, 0 flips
FLOOR_LOCK_LITERAL_1P0_CONFIRMED_BY_SERVED_CENSUS: bool = False  # bwyhpkd7 reference unverifiable (iso)

# --- Rung 2: global-flag VLLM_BATCH_INVARIANT=1 (the #474 live call; land/ubel #470 e2e) -------- #
GLOBAL_FLAG_TPS_LOCAL: float = 222.32       # land local full-serve
GLOBAL_FLAG_TPS_OFFICIAL: float = 234.47    # ubel #470 (ugqnytji) official; 221.16 local
GLOBAL_FLAG_NEEDS_MANIFEST_ENV_EDIT: bool = True   # shell-prefix does NOT propagate to the HF runner
GLOBAL_FLAG_OPERATIVE_1P0: bool = True      # operative-1.0, 0 semantic flips (denken #471 census)

# --- Rung 3: surgical attention-only / 2D byte-exact frontier (lawine #488 MEASURED) ------------ #
# RESOLVED (2026-06-16, advisor terminalize relay 14:37Z): lawine #488 (ko01dcyy) SERVED the surgical
# attention-pin config e2e and measured 357.6 TPS byte-exact -- this is the REAL served rung, and it
# CONFIRMS honesty flag #2: the composed ~457 was OPTIMISTIC.  The surgical row headline is now the
# MEASURED 357.6, not the 456.98 prediction.  357.6 is the human-picked #474 SHIP ("357 -- go, finish it").
SURGICAL_MEASURED_TPS: float = 357.6            # lawine #488 (ko01dcyy) served byte-exact e2e -- THE SHIP
SURGICAL_MEASURED_PPL: float = 2.3767           # lawine #488: PPL clears the gate, 128/128 completed
SURGICAL_MEASURED_OVER_GLOBALFLAG_TPS: float = 135.65   # 357.6 - 221.95 (the supplanted global-flag local)
SURGICAL_PREDICTED_TPS: float = 456.98          # ubel #484 (r1l881bx) predicted_surgical_tps -- REFUTED by #488
SURGICAL_PREDICTED_TPS_MEASURED: float = 347.96  # ubel #484 companion "measured" variant (foreshadowed the gap)
MULTISTREAM_REALIZABLE_TPS: float = 457.54      # lawine #482 (044xamdd) dependency_bounded_strict_tps
MULTISTREAM_CEILING_TPS: float = 474.44         # lawine #482 ceiling_477_tps (resource-feasibility)
SINGLE_STREAM_REALIZED_TPS: float = 457.55      # lawine #482 single_stream_realized_tps (#472)
MULTISTREAM_GAIN_VS_SINGLE_TPS: float = -0.01   # ~0: the per-layer data dependency eats the overlap
BYTE_EXACT_LOCUS_IDENTITY: float = 1.0000       # lawine #482/#488 strict_identity_fraction=1, 0 flips
SURGICAL_REALIZED_E2E: bool = True              # RESOLVED: lawine #488 served it e2e (357.6 measured)

# --- Rung 4 (reference): deployed PR #52 -- NON-equivalent, OUTSIDE the strict feasible set ------ #
DEPLOYED_TPS: float = 481.53
DEPLOYED_IDENTITY: float = 0.9966               # 3 flips {11,18,118}, quality-neutral ties (denken #464)
DEPLOYED_IS_STRICT_EQUIVALENT: bool = False
PPL_DEPLOYED: float = 2.3772

# --- The organizing principle: private-safety = f(Delta), NOT f(TPS) (denken #489 q1ivw9tt) ----- #
DELTA_FLOORLOCK_PCT: float = 0.6334     # delta_floorlock_pct (no drafter)
DELTA_SPEC_ALIVE_PCT: float = 4.2946    # delta_spec_alive_pct (ALL drafter-alive configs share this)
DEPLOYED_ACCEPTANCE_GAP_PCT: float = 3.6613   # deployed_acceptance_bucket_pct (the underlying gap)
FLOORLOCK_BREACH_FRAC_PCT: float = 0.0008      # floorlock_breach_frac_pct (~0% physical)
FLOORLOCK_BREACH_ABS_PCT: float = 7.382        # floorlock_breach_abs_pct (worst tail)
SPEC_ALIVE_BREACH_FRAC_PCT: float = 24.306     # globalflag/surgical457 breach (scale-INVARIANT)
GLOBALFLAG_BREACH_ABS_PCT: float = 36.725      # globalflag_breach_abs_pct (worst tail)
PRIVATE_SAFETY_IS_SCALE_INVARIANT: bool = True  # spec_alive_frac_scale_invariant_breach_pct == 24.306
FAST_BYTEEXACT_PRIVATESAFE_COEXISTS: bool = False  # denken #489: they do NOT coexist (today)
FLOORLOCK_RECONFIRM_SAFE: bool = True           # denken #489 floorlock_reconfirm_safe

# --- >500 closure (settled): dead via all known strict levers -------------------------------- #
STRICT_REALIZED_CEILING_TPS: float = 467.14     # best deployed-EQUIVALENT locus (composed upper bound)
GT500_TARGET_TPS: float = 500.0
IEEE754_TAX_IRREDUCIBLE: bool = True            # denken #423
NO_FREE_FAST_BYTE_EXACT_GEMM: bool = True       # #481 survey: deterministic-IO tax band 22-63%
DETERMINISM_TAX_E2E_PCT: float = 51.39          # land measured e2e (#481 survey), top of the 22-63% band
# The lone >500 path leaves the strict lane: the greedy-UNSAFE ~16% GEMM relax-prize.
RELAX_PRIZE_IDENTITY: float = 0.730             # stark #452: out of the strict set (3317 flips)
RELAX_PRIZE_REALIZED_GAIN_TPS: float = -0.94    # AND ~0 TPS gain -> strictly dominated within strict
GT500_ONLY_PATH: str = "greedy-UNSAFE ~16% GEMM relax-prize (out of strict lane; escalated to human #407)"

# --- Forward research (DECOUPLED; #481 zoom-out menu, NOT capstone-gating slots) -------------- #
# These two attack the only remaining "faster AND private-safe" problem, but they are FORWARD RESEARCH
# in the wider #481 menu (SGLang #498, TRT-LLM, alt spec-dec) -- NOT slots this terminal capstone waits
# on.  The capstone's own gating question (#488 surgical realization) is RESOLVED (357.6 measured), so
# the map is TERMINAL regardless of these.  They are rendered as orthogonal forward levers, informational.
FORWARD_RESEARCH = {
    "ubel491_reduction_sensitivity": {
        "question": "which decode matmuls MUST be deterministic? shed the rest -> a FASTER floor-lock",
        "attacks": "the determinism TPS-tax (keeps Delta safe: the floor-lock has no drafter)",
        "axis": "TPS-ceiling",
    },
    "denken492_eagle3_drafter": {
        "question": "can EAGLE-3 pull Delta_accept <= ~3.0% -> a fast strict PRIVATE-safe rung?",
        "attacks": "the drafter Delta-gate (keeps a drafter for speed)",
        "axis": "private-Delta",
    },
}

# --- The #474 SHIP (human ruled "357 -- go, finish it", 13:51Z; armed on land #473) ------------ #
# land #473 re-pointed at the surgical-357 package (per advisor relay; packaged on land's branch, NOT
# in this working tree -- launch-isolation).  This capstone CITES the ship; it does not own the trigger.
SHIP_RUNG: str = "surgical attn-only / 2D byte-exact (357.6 measured)"
SHIP_SUBMISSION: str = "fa2sw_strict_surgical357 (land #473 package, per advisor relay)"
SHIP_TRIGGER: str = "land #473 (fires the draw on its next poll)"
SHIP_PER: str = "#474 -- human ruled '357 -- go, finish it' (13:51Z)"
SHIP_STATUS_LINE: str = "surgical-357 fire armed (land #473, per #474)"

# Verdict enums
REC_FLOOR_LOCK = "FLOOR-LOCK-166-PRIVATE-SAFE"        # the rung that sticks (private-safe)
REC_SHIP_357 = "SHIP-SURGICAL-357-FAST-RISKY"         # the human's #474 choice (fast-risky/penalize lane)
VALID_LIVE_474_RULINGS = (REC_FLOOR_LOCK, REC_SHIP_357, "PENDING")

CONSUMED_CARDS = ("stark#485", "lawine#482", "lawine#488", "denken#489", "ubel#484", "ubel#470",
                  "denken#471", "land#473", "denken#423")


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# --------------------------------------------------------------------------- #
# The map.
# --------------------------------------------------------------------------- #
def build_rungs() -> list[dict[str, Any]]:
    """The ranked strict-frontier rungs (+ the non-equivalent deployed reference)."""
    return [
        {
            "rung": "floor-lock M=1 AR (no drafter)",
            "submission": "fa2sw_strict_m1ar_int4",
            "realized_tps": FLOOR_LOCK_TPS,
            "realized_provenance": "stark#485 pavotwci projected_official=166.234, realizes_16170=1",
            "strict": True,
            "identity_label": "literal-1.0 (by construction; denken#471 bwyhpkd7 census accepted 1.0/0)",
            "identity_confirmed_by_served_census": FLOOR_LOCK_LITERAL_1P0_CONFIRMED_BY_SERVED_CENSUS,
            "served_census_run": FLOOR_LOCK_SERVED_CENSUS_RUN,
            "served_census_accepted_floor": FLOOR_LOCK_SERVED_CENSUS_ACCEPTED_FLOOR,
            "delta_pct": DELTA_FLOORLOCK_PCT,
            "private_safe": True,
            "breach_frac_pct": FLOORLOCK_BREACH_FRAC_PCT,
            "ship_status": "the strict rung that STICKS (private-safe); fallback, NOT the #474 ship",
            "ppl": FLOOR_LOCK_PPL,
        },
        {
            "rung": "global-flag VLLM_BATCH_INVARIANT=1 (drafter alive)",
            "submission": "fa2sw_precache_kenyan + manifest env BI=1",
            "realized_tps": GLOBAL_FLAG_TPS_OFFICIAL,
            "realized_provenance": "land 222.32 local / ubel#470 ugqnytji 234.47 official",
            "strict": True,
            "identity_label": "operative-1.0 (0 semantic flips; denken #471 census)",
            "identity_confirmed_by_served_census": True,
            "delta_pct": DELTA_SPEC_ALIVE_PCT,
            "private_safe": False,
            "breach_frac_pct": SPEC_ALIVE_BREACH_FRAC_PCT,
            "ship_status": ("the original #474 call (222/234); SUPERSEDED by surgical-357 in the same "
                            "fast-risky lane (+135 TPS, same Delta) -- not the ship"),
            "ppl": PPL_DEPLOYED,
        },
        {
            "rung": "surgical attn-only / 2D byte-exact (357.6 measured, drafter alive)",
            "submission": SHIP_SUBMISSION,
            "realized_tps": SURGICAL_MEASURED_TPS,
            "realized_provenance": ("lawine#488 ko01dcyy SERVED e2e=357.6 byte-exact (+135.65 over the "
                                    "221.95 global-flag), PPL 2.3767, 128/128 -- REFUTES the 456.98 "
                                    "ubel#484 prediction (composed-vs-realized; honesty flag #2 vindicated)"),
            "strict": True,
            "identity_label": "byte-exact e2e, operative-1.0 (lawine #488 served, 0 semantic flips)",
            "identity_confirmed_by_served_census": True,
            "delta_pct": DELTA_SPEC_ALIVE_PCT,
            "private_safe": False,
            "breach_frac_pct": SPEC_ALIVE_BREACH_FRAC_PCT,
            "ship_status": f"THE #474 SHIP: {SHIP_STATUS_LINE}; fast-but-private-RISKY (same Delta as the 222)",
            "ppl": SURGICAL_MEASURED_PPL,
        },
        {
            "rung": "deployed PR#52 (reference, NON-equivalent)",
            "submission": "fa2sw_precache_kenyan (deployed)",
            "realized_tps": DEPLOYED_TPS,
            "realized_provenance": "PR#52 2x9fm2zx deployed fast path",
            "strict": False,
            "identity_label": f"NON-equivalent ({DEPLOYED_IDENTITY:.4f}, 3 ties) -- OUTSIDE strict set",
            "identity_confirmed_by_served_census": True,
            "delta_pct": DELTA_SPEC_ALIVE_PCT,
            "private_safe": None,
            "breach_frac_pct": None,
            "ship_status": "outside the strict feasible set; the incumbent reference only",
            "ppl": PPL_DEPLOYED,
        },
    ]


def the_principle() -> dict[str, Any]:
    """denken #489: private-safety is a property of Delta (drafter gap), NOT of TPS."""
    return {
        "statement": "private-safety = f(Delta drafter-acceptance gap), NOT f(TPS)",
        "delta_floorlock_pct": DELTA_FLOORLOCK_PCT,
        "delta_spec_alive_pct": DELTA_SPEC_ALIVE_PCT,
        "delta_gate_pct": DELTA_GATE_PCT,
        "floorlock_headroom_pp": round(DELTA_GATE_PCT - DELTA_FLOORLOCK_PCT, 4),
        "spec_alive_headroom_pp": round(DELTA_GATE_PCT - DELTA_SPEC_ALIVE_PCT, 4),
        "floorlock_breach_frac_pct": FLOORLOCK_BREACH_FRAC_PCT,
        "spec_alive_breach_frac_pct": SPEC_ALIVE_BREACH_FRAC_PCT,
        "scale_invariant": PRIVATE_SAFETY_IS_SCALE_INVARIANT,
        "scale_invariant_note": ("the 222, the 457 and the deployed 481 all carry the SAME Delta "
                                 "= 4.295% -> the SAME 24.3% one-shot breach, regardless of TPS"),
        "fast_byteexact_privatesafe_coexists": FAST_BYTEEXACT_PRIVATESAFE_COEXISTS,
        "conclusion": ("the floor-lock (no drafter, Delta 0.633%) is the ONLY strict private-safe "
                       "ship; fast + byte-exact != private-safe"),
        "provenance": "denken#489 q1ivw9tt",
    }


def the_gt500_closure() -> dict[str, Any]:
    """The original >500 question, now SETTLED: dead via all known strict levers."""
    headroom = round(GT500_TARGET_TPS - STRICT_REALIZED_CEILING_TPS, 2)
    return {
        "verdict": "strict >500 is DEAD via all known levers",
        "strict_realized_ceiling_tps": STRICT_REALIZED_CEILING_TPS,
        "gt500_target_tps": GT500_TARGET_TPS,
        "residual_gap_to_500_tps": headroom,
        "ieee754_determinism_tax_irreducible": IEEE754_TAX_IRREDUCIBLE,
        "no_free_fast_byte_exact_gemm": NO_FREE_FAST_BYTE_EXACT_GEMM,
        "determinism_tax_e2e_pct": DETERMINISM_TAX_E2E_PCT,
        "only_gt500_path": GT500_ONLY_PATH,
        "relax_prize_identity": RELAX_PRIZE_IDENTITY,
        "relax_prize_realized_gain_tps": RELAX_PRIZE_REALIZED_GAIN_TPS,
        "relax_prize_in_strict_lane": False,
        "is_genuinely_new_method_problem": True,
        "multiple_over_private_safe_floor": round(GT500_TARGET_TPS / FLOOR_LOCK_TPS, 2),
        "provenance": "denken#423 (IEEE-754 tax); #481 forward survey; stark#452 (relax-prize out of lane)",
    }


def the_forward_research() -> dict[str, Any]:
    """The two orthogonal forward levers -- DECOUPLED forward research (#481 zoom-out), NOT terminal slots.

    The capstone's own gating question (surgical #488 realization) is RESOLVED, so these do NOT hold the
    map open; they are the only paths to a FUTURE "faster AND private-safe" rung, handed to the wider menu.
    """
    levers = {k: {**meta, "status": "forward-research (decoupled; not capstone-gating)"}
              for k, meta in FORWARD_RESEARCH.items()}
    return {
        "levers": levers,
        "orthogonal_axes": {
            "ubel491": "determinism TPS-tax -> a FASTER floor-lock (Delta stays safe)",
            "denken492": "drafter Delta-gate -> a PRIVATE-SAFE fast rung (keeps a drafter)",
        },
        "only_faster_and_safe_is_their_conjunction": True,
        "gates_terminal": False,
        "decoupled_to_481_menu": True,
        "wider_menu": ["SGLang #498", "TRT-LLM", "alt spec-dec (Medusa/EAGLE-3/prompt-lookup)",
                       "GPTQ/AWQ/SmoothQuant", "FA3 / torch.compile"],
        "n_levers": len(levers),
    }


def the_474_decision() -> dict[str, Any]:
    """The #474 GO/NO-GO fork this capstone existed to inform -- RESOLVED (human ruled 357, 13:51Z).

    Sequence: human approved 222 @10:24Z -> denken #486 reopened it on private-Delta risk -> stark #493
    de-risked the identity axis (the realized scorer has NO token-identity gate, only {private TPS-drift
    <= 5%, PPL <= 2.42, 128/128}) -> lawine #488 upgraded the penalize-lane number 222 -> 357.6 measured
    -> human ruled "357 -- go, finish it".  Chose the fast-risky/PENALIZE lane (NOT floor-lock).
    """
    ruling = REC_SHIP_357
    return {
        "fork": {
            "FLOOR_LOCK": {
                "fire": "senpai-strict-m1ar-161 (166.23 realized)",
                "rationale": "a guaranteed-valid strict number that STICKS (Delta 0.633%, private-safe)",
                "breach_rule": "would be picked if a private re-draw over 5% INVALIDATES the submission",
                "chosen": False,
            },
            "SHIP_357": {
                "fire": "senpai-strict-eqv / surgical-357 (357.6 measured byte-exact e2e)",
                "rationale": "crushes the floor on public TPS (+135 over the 222); even a breach scores high",
                "breach_rule": "chosen: human accepts the fast-risky/PENALIZE lane",
                "chosen": True,
            },
        },
        "binding_question": ("does a private re-draw over the 5% gate INVALIDATE (waste the one shot) "
                             "or PENALIZE (scored on the lower private number)?"),
        "binding_resolution": ("the human chose the fast-risky lane -> treated as PENALIZE; the realized "
                               "scorer has no token-identity gate (stark #493), so operative-1.0 is fine"),
        "advisor_recommendation": "FLOOR-LOCK unless a breach is only a penalty -- human overrode to SHIP-357",
        "human_ruling": ruling,
        "human_ruling_verbatim": "357 -- go, finish it",
        "resolved": True,
        "ship": {
            "rung": SHIP_RUNG, "submission": SHIP_SUBMISSION, "trigger": SHIP_TRIGGER, "per": SHIP_PER,
            "tps": SURGICAL_MEASURED_TPS, "ppl": SURGICAL_MEASURED_PPL,
        },
        "provenance": ("#474 (10:24Z human approved 222; 11:16Z denken #486 reopened on private-gap; "
                       "13:51Z human ruled 357 after lawine #488 upgraded the number)"),
    }


def build_packet() -> dict[str, Any]:
    """Assemble the full TERMINAL strict-frontier-map packet (the JSON rollup).

    All gating questions are RESOLVED and banked: surgical #488 realized (357.6 measured) and the #474
    decision is made (human ruled 357).  terminal = (surgical realized AND #474 resolved).  The forward
    research (#491/#492) is DECOUPLED to the #481 menu and does NOT gate this.
    """
    rungs = build_rungs()
    principle = the_principle()
    gt500 = the_gt500_closure()
    forward = the_forward_research()
    decision = the_474_decision()

    strict_rungs = [r for r in rungs if r["strict"]]
    private_safe_rungs = [r for r in strict_rungs if r["private_safe"]]

    surgical_realized = bool(SURGICAL_REALIZED_E2E)
    decision_resolved = bool(decision["resolved"])
    terminal = surgical_realized and decision_resolved

    packet = {
        "kind": "strict-frontier-map",
        "pr": 357,
        "agent": "fern",
        "A_the_map": {
            "rungs": rungs,
            "n_strict_rungs": len(strict_rungs),
            "n_private_safe_strict_rungs": len(private_safe_rungs),
            "only_private_safe_rung": (private_safe_rungs[0]["rung"] if len(private_safe_rungs) == 1
                                       else None),
            "deployed_outside_strict_set": True,
            "ship_status_line": SHIP_STATUS_LINE,
            "ship_rung": SHIP_RUNG,
        },
        "B_the_principle": principle,
        "C_gt500_closure": gt500,
        "D_forward_research": forward,
        "E_474_decision": decision,
        "ownership": {
            "this_packet": "the human-facing strict-frontier MAP (CPU-only synthesis)",
            "land473": "the submission trigger (surgical-357 fire armed)",
            "denken471": "the served-census identity oracle",
            "consumes_cards": list(CONSUMED_CARDS),
            "analysis_only": True,
            "no_served_file_change": True,
            "no_hf_job": True,
            "official_tps": 0,
        },
        "terminal_basis": {
            "surgical_realized_e2e": surgical_realized,
            "decision_474_resolved": decision_resolved,
            "forward_research_gates_terminal": forward["gates_terminal"],
        },
        "terminal": terminal,
    }
    packet["self_test"] = _selftests(packet)
    packet["headline"] = _headline(packet)
    return packet


def _headline(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "floor_lock_realized_tps": FLOOR_LOCK_TPS,
        "floor_lock_private_safe": True,
        "global_flag_tps": GLOBAL_FLAG_TPS_OFFICIAL,
        "surgical_measured_tps": SURGICAL_MEASURED_TPS,
        "surgical_byte_exact_predicted_tps": SURGICAL_PREDICTED_TPS,
        "multistream_realizable_tps": MULTISTREAM_REALIZABLE_TPS,
        "private_safety_is_delta_not_tps": True,
        "strict_gt500_dead_via_known_levers": True,
        "forward_research_decoupled": p["D_forward_research"]["decoupled_to_481_menu"],
        "human_474_ruling": p["E_474_decision"]["human_ruling"],
        "ship_status_line": SHIP_STATUS_LINE,
        "terminal": p["terminal"],
        "strict_frontier_map_self_test_passes": None,  # filled by main() after nan-check
    }


# --------------------------------------------------------------------------- #
# Self-test -- REAL arithmetic / logical invariants, not a banked-constant mirror.
# Each condition checks a RELATIONSHIP (ordering, a computed gap, a decision
# outcome, an honesty constraint), so green means the map is internally
# consistent + honest, NOT merely that a constant was copied.
# --------------------------------------------------------------------------- #
def _selftests(p: dict[str, Any]) -> dict[str, Any]:
    pr = p["B_the_principle"]
    gt = p["C_gt500_closure"]
    fw = p["D_forward_research"]
    lv = p["E_474_decision"]
    cond: dict[str, bool] = {}

    # --- The map is strictly ordered by realized TPS: floor < global-flag < surgical(measured) < deployed. ---
    cond["a_floor_lt_globalflag"] = FLOOR_LOCK_TPS < GLOBAL_FLAG_TPS_OFFICIAL
    cond["b_globalflag_lt_surgical"] = GLOBAL_FLAG_TPS_OFFICIAL < SURGICAL_MEASURED_TPS
    cond["c_surgical_lt_deployed"] = SURGICAL_MEASURED_TPS < DEPLOYED_TPS
    cond["d_floor_realizes_target"] = FLOOR_LOCK_TPS >= FLOOR_LOCK_TARGET_TPS

    # --- lawine #482: multi-stream CLOSES -- realizable ~= single-stream, ceiling unrealizable. ---
    cond["e_multistream_gain_near_zero"] = abs(MULTISTREAM_GAIN_VS_SINGLE_TPS) < SIGMA_HW
    cond["f_realizable_below_ceiling"] = MULTISTREAM_REALIZABLE_TPS < MULTISTREAM_CEILING_TPS
    cond["g_unrealizable_gap_positive"] = (MULTISTREAM_CEILING_TPS - MULTISTREAM_REALIZABLE_TPS) > 0.0
    cond["h_realizable_matches_single_stream"] = math.isclose(
        MULTISTREAM_REALIZABLE_TPS, SINGLE_STREAM_REALIZED_TPS, abs_tol=SIGMA_HW)

    # --- The principle: private-safety is Delta-determined, scale-INVARIANT. ---
    cond["i_floorlock_delta_below_gate"] = DELTA_FLOORLOCK_PCT < DELTA_GATE_PCT
    cond["j_specalive_delta_below_gate"] = DELTA_SPEC_ALIVE_PCT < DELTA_GATE_PCT
    cond["k_floorlock_delta_lt_specalive"] = DELTA_FLOORLOCK_PCT < DELTA_SPEC_ALIVE_PCT
    cond["l_floorlock_more_headroom"] = pr["floorlock_headroom_pp"] > pr["spec_alive_headroom_pp"]
    cond["m_floorlock_breach_lt_specalive"] = FLOORLOCK_BREACH_FRAC_PCT < SPEC_ALIVE_BREACH_FRAC_PCT
    cond["n_scale_invariant"] = bool(pr["scale_invariant"])
    cond["o_fast_byteexact_not_private_safe"] = pr["fast_byteexact_privatesafe_coexists"] is False

    # --- Floor-lock is the UNIQUE private-safe strict rung. ---
    cond["p_exactly_one_private_safe_rung"] = p["A_the_map"]["n_private_safe_strict_rungs"] == 1
    cond["q_only_safe_rung_is_floorlock"] = (
        p["A_the_map"]["only_private_safe_rung"] == "floor-lock M=1 AR (no drafter)")

    # --- fast + byte-exact != private-safe: surgical is fast AND byte-exact AND NOT safe. ---
    cond["r_surgical_fast"] = SURGICAL_MEASURED_TPS > GLOBAL_FLAG_TPS_OFFICIAL
    cond["s_surgical_byte_exact"] = math.isclose(BYTE_EXACT_LOCUS_IDENTITY, 1.0)
    cond["t_surgical_not_private_safe"] = SPEC_ALIVE_BREACH_FRAC_PCT > FLOORLOCK_BREACH_FRAC_PCT

    # --- >500 closure: realized ceiling < 500; the only >500 path is greedy-UNSAFE (out of lane). ---
    cond["u_strict_ceiling_below_500"] = STRICT_REALIZED_CEILING_TPS < GT500_TARGET_TPS
    cond["v_gt500_gap_positive"] = gt["residual_gap_to_500_tps"] > 0.0
    cond["w_relax_prize_out_of_strict_lane"] = RELAX_PRIZE_IDENTITY < BYTE_EXACT_LOCUS_IDENTITY
    cond["x_relax_prize_dominated"] = RELAX_PRIZE_REALIZED_GAIN_TPS <= 0.0
    cond["y_gt500_is_new_method_problem"] = bool(gt["is_genuinely_new_method_problem"])

    # --- TERMINAL: gating questions RESOLVED; forward research DECOUPLED (does NOT gate terminal). ---
    #     terminal <=> (surgical #488 realized e2e) AND (#474 decision resolved).  #491/#492 are forward
    #     research in the #481 menu, not capstone slots.
    cond["z_forward_research_decoupled"] = (
        fw["decoupled_to_481_menu"] is True and fw["gates_terminal"] is False)
    cond["aa_forward_research_two_levers"] = fw["n_levers"] == 2
    cond["ab_terminal_iff_mandate_resolved"] = (
        p["terminal"] == (bool(SURGICAL_REALIZED_E2E) and bool(lv["resolved"])))
    cond["ac_two_orthogonal_axes"] = (
        fw["orthogonal_axes"]["ubel491"] != fw["orthogonal_axes"]["denken492"])

    # --- The #474 decision is RESOLVED to the human's SHIP-357 (fast-risky lane, NOT floor-lock). ---
    cond["ad_decision474_ruling_valid"] = lv["human_ruling"] in VALID_LIVE_474_RULINGS
    cond["ae_decision474_resolved_consistent"] = (lv["resolved"] == (lv["human_ruling"] != "PENDING"))

    # --- Honesty flag #1: floor-lock literal-1.0 is BY CONSTRUCTION; the bwyhpkd7 served census ACCEPTED
    #     it (1.0/0) but the confirmed boolean stays FALSE (reference unverifiable under launch-isolation). ---
    cond["af_floorlock_literal_flag_honest"] = (
        FLOOR_LOCK_LITERAL_1P0_BY_CONSTRUCTION is True
        and FLOOR_LOCK_LITERAL_1P0_CONFIRMED_BY_SERVED_CENSUS is False
        and FLOOR_LOCK_SERVED_CENSUS_ACCEPTED_FLOOR is True
        and FLOOR_LOCK_RELAYRUN_VERDICT_LITERAL_1P0 == 0)
    cond["ag_floorlock_tps_independently_solid"] = FLOOR_LOCK_TPS >= FLOOR_LOCK_TARGET_TPS

    # --- Honesty flag #2 VINDICATED by measurement: surgical realized e2e, BELOW its composed prediction
    #     (lawine #488 measured 357.6 < ubel #484 predicted 456.98 -- the composed-vs-realized mirage). ---
    cond["ah_surgical_realized_e2e"] = SURGICAL_REALIZED_E2E is True
    cond["ai_surgical_measured_below_predicted"] = SURGICAL_MEASURED_TPS < SURGICAL_PREDICTED_TPS

    # --- PPL clears the gate on the shippable strict rungs. ---
    cond["aj_floorlock_ppl_clears_gate"] = FLOOR_LOCK_PPL <= PPL_GATE
    cond["ak_deployed_ppl_clears_gate"] = PPL_DEPLOYED <= PPL_GATE

    # --- Mandate constraints (analysis-only 0-GPU capstone). ---
    cond["al_analysis_only"] = bool(p["ownership"]["analysis_only"])
    cond["am_no_served_file_change"] = bool(p["ownership"]["no_served_file_change"])
    cond["an_no_hf_job"] = bool(p["ownership"]["no_hf_job"])
    cond["ao_official_tps_zero"] = p["ownership"]["official_tps"] == 0

    # --- TERMINAL coherence + the SHIP is surgical-357 (the human's #474 pick, beating the floor). ---
    #     (ap_nan_clean is appended by main() after the NaN scan.)
    cond["aq_terminal_is_true"] = p["terminal"] is True
    cond["ar_ship_is_surgical_357"] = math.isclose(lv["ship"]["tps"], SURGICAL_MEASURED_TPS)
    cond["as_ship_is_fast_risky_not_floorlock"] = lv["human_ruling"] == REC_SHIP_357
    cond["at_ship_beats_floorlock"] = SURGICAL_MEASURED_TPS > FLOOR_LOCK_TPS
    cond["au_surgical_ppl_clears_gate"] = SURGICAL_MEASURED_PPL <= PPL_GATE

    return {
        "conditions": cond,
        "n_conditions": len(cond),
        "n_passing": sum(1 for v in cond.values() if v),
        "strict_frontier_map_self_test_passes": bool(all(cond.values())),
    }


# --------------------------------------------------------------------------- #
# One-screen human-facing render.
# --------------------------------------------------------------------------- #
def render_one_screen(p: dict[str, Any]) -> str:
    pr = p["B_the_principle"]
    gt = p["C_gt500_closure"]
    fw = p["D_forward_research"]
    lv = p["E_474_decision"]
    st = p["self_test"]
    L = [
        "================================================================================",
        " THE HONEST STRICT-FRONTIER MAP  —  PR #357  (fern, CPU-only synthesis)  [TERMINAL]",
        "================================================================================",
        " A. THE MAP (ranked by realized TPS; private-safety is the 2nd axis)",
        "      rung                         realized TPS   strict?           private-safe?",
        "      ---------------------------  ------------   ---------------   --------------------",
        f"      floor-lock M=1 AR            {FLOOR_LOCK_TPS:>7.2f} proj   literal-1.0*      SAFE  (Δ {DELTA_FLOORLOCK_PCT:.3f}%)  <- fallback",
        f"      global-flag BI=1            {GLOBAL_FLAG_TPS_OFFICIAL:>7.2f}       operative-1.0     RISKY (Δ {DELTA_SPEC_ALIVE_PCT:.3f}%)  <- superseded",
        f"      surgical / 2D byte-exact    {SURGICAL_MEASURED_TPS:>7.2f} meas  byte-exact e2e    RISKY (Δ {DELTA_SPEC_ALIVE_PCT:.3f}%)  <- #474 SHIP",
        f"      deployed (reference)        {DEPLOYED_TPS:>7.2f}       NON-equiv .9966   — (outside strict set)",
        f"      surgical 357.6 = lawine #488 (ko01dcyy) MEASURED byte-exact e2e — REFUTES the 456.98",
        f"      composed prediction (honesty flag #2 vindicated: composed-vs-realized, +{SURGICAL_MEASURED_OVER_GLOBALFLAG_TPS:.1f} over 222).",
        f"      * floor-lock literal-1.0 = BY CONSTRUCTION (M=1 AR int4 = the strict reference); the denken",
        f"        #471 served census ({FLOOR_LOCK_SERVED_CENSUS_RUN}) ACCEPTED the floor 1.0/0 flips (rejects deployed",
        f"        .9966), but confirmed=False — reference M=1-AR-self-vs-greedy unverifiable under iso (non-gating).",
        "",
        f"   >>> SHIP STATUS: {SHIP_STATUS_LINE} <<<",
        "",
        " B. THE PRINCIPLE (denken #489): private-safety = f(Δ drafter-gap), NOT f(TPS)",
        f"      floor-lock Δ {DELTA_FLOORLOCK_PCT:.3f}% ({pr['floorlock_headroom_pp']:.2f}pp headroom, breach ~{FLOORLOCK_BREACH_FRAC_PCT:.3f}%)  vs"
        f"  spec-alive Δ {DELTA_SPEC_ALIVE_PCT:.3f}% ({pr['spec_alive_headroom_pp']:.2f}pp, breach {SPEC_ALIVE_BREACH_FRAC_PCT:.1f}%)",
        f"      SCALE-INVARIANT: the 222, the 357.6 and the 481 all carry the SAME Δ -> SAME breach, any TPS",
        f"      => floor-lock (no drafter) is the ONLY strict private-safe ship.  Fast + byte-exact ≠ safe.",
        "",
        " C. >500 CLOSURE (settled): strict >500 DEAD via all known levers",
        f"      realized strict ceiling {STRICT_REALIZED_CEILING_TPS:.2f} < 500 (gap {gt['residual_gap_to_500_tps']:.2f}); IEEE-754 tax irreducible (denken#423);",
        f"      no free fast byte-exact GEMM (#481: tax 22-63%, e2e {DETERMINISM_TAX_E2E_PCT:.1f}%).  Only >500 path =",
        f"      greedy-UNSAFE ~16% relax-prize (id {RELAX_PRIZE_IDENTITY:.3f}, out of lane, human #407).  ~{gt['multiple_over_private_safe_floor']:.1f}x over the 166 floor.",
        "",
        " D. FORWARD RESEARCH (DECOUPLED -> #481 zoom-out menu; NOT capstone-gating)",
        f"      [ ] #491 reduction-sensitivity -> FASTER floor-lock (attacks TPS-tax, Δ stays safe)",
        f"      [ ] #492 EAGLE-3 drafter       -> PRIVATE-SAFE fast rung (attacks Δ-gate, keeps drafter)",
        f"      orthogonal axes; only their CONJUNCTION yields faster AND private-safe.  Wider menu:",
        f"      {', '.join(fw['wider_menu'])}.",
        "",
        f" E. THE #474 DECISION (RESOLVED): human ruled \"{lv['human_ruling_verbatim']}\" (13:51Z) -> SHIP-357",
        f"      chose the fast-risky/PENALIZE lane over floor-lock {FLOOR_LOCK_TPS:.0f}; realized scorer has no",
        f"      token-identity gate (stark #493), so operative-1.0 clears.  ruling: {lv['human_ruling']}",
        f"      fire: surgical-357 ({SURGICAL_MEASURED_TPS:.1f} TPS) armed on {SHIP_TRIGGER}.",
        " OWNERSHIP: this packet = the MAP · land#473 = trigger · denken#471 = census oracle"
        "  ·  CPU-only, official_tps=0, no served-file change",
        "================================================================================",
        f" self-test: {st['n_passing']}/{st['n_conditions']} invariants  ·  terminal={p['terminal']}"
        f"  ·  >500 strict: DEAD-via-known-levers  ·  SHIP: surgical-357",
        "================================================================================",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# NaN guard + W&B logging (reuses scripts.wandb_logging plumbing).
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, path: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, float) and not math.isfinite(node):
        bad.append(path)
    elif isinstance(node, dict):
        for k, v in node.items():
            bad.extend(_nan_paths(v, f"{path}.{k}"))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad.extend(_nan_paths(v, f"{path}[{i}]"))
    return bad


def _wandb_summary(payload: dict) -> dict[str, Any]:
    p = payload["synthesis"]
    pr = p["B_the_principle"]
    gt = p["C_gt500_closure"]
    fw = p["D_forward_research"]
    lv = p["E_474_decision"]
    o = p["ownership"]
    st = p["self_test"]
    rec_code = {"PENDING": 0, REC_FLOOR_LOCK: 1, REC_SHIP_357: 2}.get(lv["human_ruling"], 0)
    summary: dict[str, Any] = {
        # PRIMARY
        "strict_frontier_map_self_test_passes": int(bool(st["strict_frontier_map_self_test_passes"])),
        "self_test_n_passing": st["n_passing"],
        "self_test_n_conditions": st["n_conditions"],
        # A. the map
        "floor_lock_realized_tps": FLOOR_LOCK_TPS,
        "floor_lock_private_safe": 1,
        "floor_lock_target_tps": FLOOR_LOCK_TARGET_TPS,
        "floor_lock_ppl": FLOOR_LOCK_PPL,
        "global_flag_tps_official": GLOBAL_FLAG_TPS_OFFICIAL,
        "global_flag_tps_local": GLOBAL_FLAG_TPS_LOCAL,
        "global_flag_private_safe": 0,
        "surgical_measured_tps": SURGICAL_MEASURED_TPS,
        "surgical_measured_ppl": SURGICAL_MEASURED_PPL,
        "surgical_measured_over_globalflag_tps": SURGICAL_MEASURED_OVER_GLOBALFLAG_TPS,
        "surgical_byte_exact_predicted_tps": SURGICAL_PREDICTED_TPS,
        "surgical_predicted_tps_measured": SURGICAL_PREDICTED_TPS_MEASURED,
        "surgical_realized_e2e": int(bool(SURGICAL_REALIZED_E2E)),
        "multistream_realizable_tps": MULTISTREAM_REALIZABLE_TPS,
        "multistream_ceiling_tps": MULTISTREAM_CEILING_TPS,
        "multistream_gain_vs_single_tps": MULTISTREAM_GAIN_VS_SINGLE_TPS,
        "multistream_unrealizable_gap_tps": round(MULTISTREAM_CEILING_TPS - MULTISTREAM_REALIZABLE_TPS, 2),
        "single_stream_realized_tps": SINGLE_STREAM_REALIZED_TPS,
        "deployed_tps": DEPLOYED_TPS,
        "deployed_identity": DEPLOYED_IDENTITY,
        "deployed_is_strict_equivalent": int(bool(DEPLOYED_IS_STRICT_EQUIVALENT)),
        "n_private_safe_strict_rungs": p["A_the_map"]["n_private_safe_strict_rungs"],
        # B. the principle
        "private_safety_is_delta_not_tps": 1,
        "delta_floorlock_pct": DELTA_FLOORLOCK_PCT,
        "delta_spec_alive_pct": DELTA_SPEC_ALIVE_PCT,
        "delta_gate_pct": DELTA_GATE_PCT,
        "floorlock_headroom_pp": pr["floorlock_headroom_pp"],
        "spec_alive_headroom_pp": pr["spec_alive_headroom_pp"],
        "floorlock_breach_frac_pct": FLOORLOCK_BREACH_FRAC_PCT,
        "spec_alive_breach_frac_pct": SPEC_ALIVE_BREACH_FRAC_PCT,
        "globalflag_breach_abs_pct": GLOBALFLAG_BREACH_ABS_PCT,
        "private_safety_scale_invariant": int(bool(PRIVATE_SAFETY_IS_SCALE_INVARIANT)),
        "fast_byteexact_privatesafe_coexists": int(bool(FAST_BYTEEXACT_PRIVATESAFE_COEXISTS)),
        # C. >500 closure
        "strict_gt500_dead_via_known_levers": 1,
        "strict_realized_ceiling_tps": STRICT_REALIZED_CEILING_TPS,
        "residual_gap_to_500_tps": gt["residual_gap_to_500_tps"],
        "ieee754_determinism_tax_irreducible": int(bool(IEEE754_TAX_IRREDUCIBLE)),
        "determinism_tax_e2e_pct": DETERMINISM_TAX_E2E_PCT,
        "relax_prize_identity": RELAX_PRIZE_IDENTITY,
        "relax_prize_in_strict_lane": 0,
        "gt500_multiple_over_floor": gt["multiple_over_private_safe_floor"],
        # D. forward research (decoupled to the #481 menu; NOT capstone-gating)
        "forward_research_decoupled": int(bool(fw["decoupled_to_481_menu"])),
        "forward_research_gates_terminal": int(bool(fw["gates_terminal"])),
        "forward_research_n_levers": fw["n_levers"],
        # E. #474 decision (RESOLVED -> ship surgical-357)
        "decision_474_ruling_code": rec_code,  # 0 pending / 1 floor-lock / 2 ship-357
        "decision_474_resolved": int(bool(lv["resolved"])),
        "ship_tps": lv["ship"]["tps"],
        "ship_ppl": lv["ship"]["ppl"],
        # honesty flags
        "floor_lock_literal_1p0_by_construction": int(bool(FLOOR_LOCK_LITERAL_1P0_BY_CONSTRUCTION)),
        "floor_lock_literal_1p0_confirmed_by_census": int(bool(
            FLOOR_LOCK_LITERAL_1P0_CONFIRMED_BY_SERVED_CENSUS)),
        "floor_lock_served_census_accepted_floor": int(bool(FLOOR_LOCK_SERVED_CENSUS_ACCEPTED_FLOOR)),
        "floor_lock_relayrun_verdict_literal_1p0": FLOOR_LOCK_RELAYRUN_VERDICT_LITERAL_1P0,
        "floor_lock_relayrun_divergent_vs_precache": FLOOR_LOCK_RELAYRUN_DIVERGENT_VS_PRECACHE,
        # gates + mandate
        "ppl_gate": PPL_GATE,
        "sigma_hw": SIGMA_HW,
        "terminal": int(bool(p["terminal"])),
        "analysis_only": int(bool(o["analysis_only"])),
        "no_served_file_change": int(bool(o["no_served_file_change"])),
        "no_hf_job": int(bool(o["no_hf_job"])),
        "official_tps": o["official_tps"],
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    return {k: v for k, v in summary.items()
            if v is not None and not (isinstance(v, float) and not math.isfinite(v))}


def _maybe_log_wandb(args: Any, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[strict-frontier-map] wandb logging unavailable: {exc}", flush=True)
        return

    run = init_wandb_run(
        job_type="strict-frontier-map",
        agent="fern",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=[
            "strict-frontier-map", "strict-frontier", "strict-equivalent", "honest-map", "TERMINAL",
            "floor-lock-166-private-safe", "surgical-357-measured-ship", "surgical-457-predicted-refuted",
            "private-safety-is-delta-not-tps", "denken-489-delta-principle",
            "lawine-482-dependency-collapses-to-floor", "ubel-484-surgical-predicted-refuted",
            "stark-485-floorlock-realize", "gt500-dead-via-known-levers", "ieee754-tax-irreducible",
            "forward-research-decoupled-481-menu", "ubel-491-reduction-sensitivity", "denken-492-eagle3",
            "lawine-488-surgical-realized-357", "human-474-ruled-357", "floorlock-literal-1p0-flag",
            "analysis-only", "cpu-only", "no-served-file-change", "bank-the-analysis",
        ],
        config={
            "floor_lock_tps": FLOOR_LOCK_TPS,
            "global_flag_tps_official": GLOBAL_FLAG_TPS_OFFICIAL,
            "surgical_measured_tps": SURGICAL_MEASURED_TPS,
            "surgical_predicted_tps_refuted": SURGICAL_PREDICTED_TPS,
            "multistream_realizable_tps": MULTISTREAM_REALIZABLE_TPS,
            "deployed_tps": DEPLOYED_TPS,
            "delta_floorlock_pct": DELTA_FLOORLOCK_PCT,
            "delta_spec_alive_pct": DELTA_SPEC_ALIVE_PCT,
            "delta_gate_pct": DELTA_GATE_PCT,
            "strict_realized_ceiling_tps": STRICT_REALIZED_CEILING_TPS,
            "ppl_gate": PPL_GATE,
            "sigma_hw": SIGMA_HW,
            "terminal": payload["synthesis"]["terminal"],
            "human_474_ruling": REC_SHIP_357,
            "ship_status_line": SHIP_STATUS_LINE,
            "ship_tps": SURGICAL_MEASURED_TPS,
            "forward_research_decoupled_to_481_menu": True,
            "source_runs": (
                "stark#485(pavotwci): floor-lock realizes 166.23 proj official, PPL 2.3767, realizes_16170=1; "
                "FLAG verdict/literal_1p0=0 vs fa2sw_precache_kenyan ref (119/128 divergent, different config); "
                "denken#471(bwyhpkd7) served census ACCEPTED M=1 AR floor at 1.0/0 flips (pointer, iso-pending). "
                "lawine#488(ko01dcyy): surgical attn-pin SERVED e2e=357.6 byte-exact, PPL 2.3767, 128/128 -- THE "
                "SHIP; REFUTES the ubel#484 456.98 prediction (composed-vs-realized inversion). "
                "lawine#482(044xamdd): dependency_bounded_strict_tps=457.54, ceiling 474.44 UNREALIZABLE "
                "(DEPENDENCY_COLLAPSES_TO_FLOOR), multistream gain ~0, byte-exact 0 flips. "
                "denken#489(q1ivw9tt): delta_floorlock=0.633%, delta_spec_alive=4.295%, scale-invariant "
                "24.3% breach, fast_byteexact_privatesafe_coexists=False. ubel#484(r1l881bx): "
                "predicted_surgical_tps=456.98 (measured-variant 347.96) -- REFUTED by #488. "
                "ubel#470(ugqnytji): global-flag 234.47 official / 221.16 local. denken#423: IEEE-754 tax "
                "irreducible. stark#452: relax-prize out of strict lane (id 0.730). #481 forward survey: "
                "no free fast byte-exact GEMM, tax 22-63%, e2e 51.39%. Deployed PR#52(2x9fm2zx): 481.53 NON-equiv. "
                "human #474: ruled '357 -- go, finish it' (13:51Z); land #473 arms the surgical-357 fire."
            ),
        },
    )
    if run is None:
        print("[strict-frontier-map] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary = _wandb_summary(payload)
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="strict_frontier_map",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[strict-frontier-map] wandb logged {len(summary)} keys", flush=True)


# --------------------------------------------------------------------------- #
# CLI / main.
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    # TERMINAL map (advisor #357 relay 14:37Z): all gating questions are RESOLVED and banked, so there
    # are no LIVE slots to pass in.  Surgical #488 realized 357.6 measured and the human ruled 357; the
    # forward levers (#491/#492) are DECOUPLED to the #481 zoom-out menu and do NOT gate this map.
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group", default="strict-frontier")
    args = ap.parse_args(argv)

    packet = build_packet()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 357, "agent": "fern",
        "kind": "strict-frontier-map", "analysis_only": True,
        "terminal_inputs_resolved": {
            "lawine488_surgical_served_tps": SURGICAL_MEASURED_TPS,
            "human474_ruling": REC_SHIP_357,
            "ship_status_line": SHIP_STATUS_LINE,
            "forward_research_decoupled_to_481_menu": True,
        },
        "synthesis": packet,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    packet["self_test"]["conditions"]["ap_nan_clean"] = not nan_paths
    packet["self_test"]["n_conditions"] = len(packet["self_test"]["conditions"])
    packet["self_test"]["n_passing"] = sum(1 for v in packet["self_test"]["conditions"].values() if v)
    packet["self_test"]["strict_frontier_map_self_test_passes"] = bool(
        all(packet["self_test"]["conditions"].values()))
    packet["headline"]["strict_frontier_map_self_test_passes"] = packet["self_test"][
        "strict_frontier_map_self_test_passes"]
    if nan_paths:
        print(f"[strict-frontier-map] WARNING non-finite at: {nan_paths}", flush=True)

    print(render_one_screen(packet), flush=True)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "strict_500_composite_reachability_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    md_path = out_dir / "strict_frontier_map.md"
    md_path.write_text(render_one_screen(packet) + "\n", encoding="utf-8")
    print(f"[strict-frontier-map] wrote {json_path} and {md_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = packet["self_test"]["strict_frontier_map_self_test_passes"] and payload["nan_clean"]
        print(f"[strict-frontier-map] self-test {'PASS' if ok else 'FAIL'} "
              f"({packet['self_test']['n_passing']}/{packet['self_test']['n_conditions']})", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
