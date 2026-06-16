"""PR #517 -- Cross-axis reopen-action capstone (the unifying artifact above the dossiers).

analysis_only=true, official_tps=0, CPU-only. NO serve, NO HF job, NO --launch,
NO submission, NO served-file change, NO evals run.

This is a pure *composition* PR: it folds the three banked, decision-grade dossiers
into ONE recommended reopen action plus a decision tree keyed on the human's risk
preference. It re-derives NOTHING -- every cited number resolves to a banked W&B run;
the composer only re-states the banked verdicts and asserts they cohere into a single
recommendation.

The strategic context (PR #517): we hold three dossiers but no single act-on-it line.
When the organizer's quality pause lifts and the challenge reopens, the human needs ONE
read to decide what to fire. The three banked inputs:

  BANKED (composed from directly):
   - #508 speed/private dossier (kanna, fn2v5wox): surgical-357 private band, floor-lock
     portfolio price. VERDICT = "bracketed" (ship surgical under penalize / speed-invalidate
     E-value; keep floor-lock under a literal private-identity rule or maximin).
   - #512 quality dossier (kanna, 3fxrmc8u): surgical-357 downstream quality = base by
     construction; PASSES Morgan's MMLU-Pro/GPQA gate while pruned competitors collapse.
     VERDICT = "dominant" (quality-valid under all three organizer quality rules).
   - #500 byteexact-399 economics (lawine, m76qbs3l / feof8wtk / rvl5w50z): +42 local TPS
     over surgical-357, 0 semantic flips, PPL-identical, served-self-deterministic, BUT
     5 ULP near-ties e2e vs surgical's 1 -> operative / quality-equivalent, NOT byte-
     identical-token tight to surgical's <=1-flip census.

The three rungs differ on SPEED and TOKEN-IDENTITY STRICTNESS, not quality: the quality
pause selects FOR every one of our byte-faithful entries and AGAINST the pruned-substrate
frontier. So the reopen decision is a two-axis fork:
  Axis A (private validity rule / objective): literal-private-identity OR maximin -> floor-lock;
    penalize-breach OR speed-invalidate-with-E-value -> the fastest valid rung.
  Axis B (token-identity strictness of the speed rung): byte-identical-token (<=1-flip)
    required -> surgical-357 (shipped); operative / quality-equivalent (0-semantic)
    acceptable -> byteexact-399 (faster, +42 local).

The recommendation is TABLE-DRIVEN: it keys on each rung's shipped flag, official TPS,
label, and best-local TPS -- it does not hard-code a lean.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import wandb  # noqa: F401  (import first to win over any ./wandb shadow dir)

from scripts import wandb_logging

# --------------------------------------------------------------------------- #
# INPUTS (cited baselines -- every number resolves to a banked W&B run).       #
# Nothing here is re-measured; this composer only re-states banked outputs.     #
# --------------------------------------------------------------------------- #

# ---- Quality gate + anchors (Morgan #483; dixie-flatline #483 anchors) ----
GATE_MMLU_PRO = 0.60
GATE_GPQA_DIAMOND = 0.42
BASE_MMLU_PRO = 0.668            # base gemma-4-E4B-it greedy (dixie #483)
BASE_GPQA_DIAMOND = 0.470        # base gemma-4-E4B-it greedy (dixie #483)
PRUNED_MMLU_PRO = 0.330          # pruned-substrate competitor collapse (dixie #483)
PRUNED_GPQA_DIAMOND = 0.283      # pruned-substrate competitor collapse (near chance, dixie #483)
CHANCE_GPQA_DIAMOND = 0.25

# ---- Realized automated scorer (stark #493 xuvmnpav, advisor-verified) ----
# The realized scorer has NO token-identity gate: realized gate = {private TPS-drift
# <= 5%, PPL <= 2.42, 128/128}. Token-identity strictness is a HUMAN integrity-posture
# axis (the draw-standard question), not an automated-scorer gate.
PPL_CAP = 2.42
SCORER_DRIFT_CAP_FRAC = 0.05
SCORER_HAS_TOKEN_IDENTITY_GATE = False

# ---- Ship: surgical-357 (#508 fn2v5wox + #512 3fxrmc8u; fired stark #499 j7qao5e9) ----
SURGICAL_OFFICIAL_TPS = 375.857          # j7qao5e9 official a10g-small (stark #499); the SHIP
SURGICAL_LOCAL_TPS = 357.6               # local pod warm-median (the "357" rung name; #500 row)
SURGICAL_PRIVATE_ANCHOR_TPS = 357.22     # #508 private-band anchor (kanna #504 0urxqwob)
SURGICAL_PPL = 2.37673                   # <= 2.42 gate (j7qao5e9)
SURGICAL_COMPLETED = 128
# private band (composed verbatim from #508 dossier fn2v5wox -- anchored on 357.22 local):
SURGICAL_PRIVATE_MEAN = 341.8786721491912
SURGICAL_PRIVATE_95LO = 335.17797330424327
SURGICAL_PRIVATE_95HI = 348.5793709941391
SURGICAL_BREACH_FRAC = 0.04294644155088977     # linear 4.3% breach (#504, PF 1.00)
SURGICAL_P_BELOW_95PCT = 0.23055883245068903   # P(private < 0.95x public)
SURGICAL_P_BELOW_FLOORLOCK = 0.0               # 51sigma away (#508)
SURGICAL_WC24_95LO = 266.1661486573638         # refuted 24% worst-case 95%-downside (#508)
# token-identity census (stark #494 k8nqmc2b/5fxw18gu; same #461 locus as #500):
SURGICAL_CENSUS_FLIPS = 1
SURGICAL_SEMANTIC_FLIPS = 0
SURGICAL_MARGIN_NAT = 0.125              # 1 bf16-ULP near-tie
# quality (= base by construction, #512 3fxrmc8u):
SURGICAL_MMLU_PRO = 0.668
SURGICAL_GPQA_DIAMOND = 0.470
SURGICAL_QUALITY_WORSTCASE_DACC_PCT = 9.375    # #512 construction-refuted worst case (closed by ubel #511)
SURGICAL_LABEL = "operative-1.0"         # byte-identical-token at <=1-flip census; spec-alive
SURGICAL_SUBMISSION = "submissions/fa2sw_strict_surgical357"
SURGICAL_SHIPPED = True                   # fired + landed (stark #499)

# ---- Floor-lock 166.23 (#508 fn2v5wox; stark #485 pavotwci) ----
FLOORLOCK_TPS = 166.23                     # literal-1.0, zero breach (M=1 AR)
FLOORLOCK_PRIVATE_MEAN = 166.23
FLOORLOCK_PRIVATE_95LO = 162.97195186849905
FLOORLOCK_PRIVATE_95HI = 169.48804813150093
FLOORLOCK_BREACH_FRAC = 0.0
FLOORLOCK_CENSUS_FLIPS = 0
FLOORLOCK_SEMANTIC_FLIPS = 0
FLOORLOCK_MMLU_PRO = 0.668                 # = base by construction (literal greedy identity)
FLOORLOCK_GPQA_DIAMOND = 0.470
FLOORLOCK_LABEL = "literal-1.0"            # literal byte-identical greedy; private-safe
FLOORLOCK_SUBMISSION = "submissions/fa2sw_strict_m1ar_int4"
FLOORLOCK_SHIPPED = False                  # pre-staged invalidation-insurance fallback

# ---- Byteexact-399 (#500 m76qbs3l / feof8wtk / rvl5w50z; lawine) ----
BYTEEXACT_OFFICIAL_TPS = 0.0               # never launched (analysis_only) -- best-local only
BYTEEXACT_LOCAL_FIRETIME_TPS = 444.82227295018606   # 128x512 fire-time (m76qbs3l)
BYTEEXACT_LOCAL_MATCHED_TPS = 399.97443199055806    # 32x256 matched recert vs #496 proxy (feof8wtk)
BYTEEXACT_PPL = 2.3766643358900286         # <= 2.42 gate; PPL-identical to surgical
BYTEEXACT_COMPLETED = 128
BYTEEXACT_CENSUS_FLIPS = 5                  # rvl5w50z #461 locus
BYTEEXACT_SEMANTIC_FLIPS = 0               # quality-safe
BYTEEXACT_MARGIN_NAT = 0.25                # all 5 are sub-0.25-nat bf16-ULP near-ties
BYTEEXACT_SERVED_SELF_DET = 1.0            # warm r1-r2 served self-determinism
BYTEEXACT_MMLU_PRO = 0.668                 # = base prior (0 semantic -> 0 answer change)
BYTEEXACT_GPQA_DIAMOND = 0.470
BYTEEXACT_LABEL = "quality-equivalent"     # operative, 0-semantic, NOT <=1-flip token-tight
BYTEEXACT_SUBMISSION = "submissions/fa2sw_strict_byteexact_splitkv399"
BYTEEXACT_SHIPPED = False                  # staged + locally certified, pending human draw-standard ruling

# ---- Banked dossier verdicts (the three inputs we compose) ----
SPEED_VERDICT = "bracketed"     # #508 fn2v5wox
QUALITY_VERDICT = "dominant"    # #512 3fxrmc8u

SOURCE_RUNS = {
    "speed_private_508": "fn2v5wox",
    "quality_512": "3fxrmc8u",
    "byteexact_econ_500_firetime": "m76qbs3l",
    "byteexact_econ_500_matched": "feof8wtk",
    "byteexact_econ_500_census": "rvl5w50z",
    "ship_surgical357_official": "j7qao5e9",
    "surgical357_locus_cert_494": ["k8nqmc2b", "5fxw18gu"],
    "private_propagation_504": "0urxqwob",
    "floorlock_485": "pavotwci",
}


# --------------------------------------------------------------------------- #
# Quality-gate helper (carried for context: all three rungs are quality-valid) #
# --------------------------------------------------------------------------- #
def _gate(mmlu: float, gpqa: float) -> dict[str, Any]:
    pass_mmlu = mmlu >= GATE_MMLU_PRO
    pass_gpqa = gpqa >= GATE_GPQA_DIAMOND
    return {
        "mmlu": mmlu, "gpqa": gpqa,
        "pass_mmlu": pass_mmlu, "pass_gpqa": pass_gpqa,
        "passes_gate": bool(pass_mmlu and pass_gpqa),
        "margin_mmlu": mmlu - GATE_MMLU_PRO, "margin_gpqa": gpqa - GATE_GPQA_DIAMOND,
    }


# --------------------------------------------------------------------------- #
# (3) Per-rung risk/reward table (the structured heart of the capstone)        #
# --------------------------------------------------------------------------- #
def build_rungs() -> list[dict[str, Any]]:
    """Each rung folds official/best-local TPS, private band, quality, label, draw-readiness."""
    floorlock = {
        "name": "floor-lock-166.23",
        "submission": FLOORLOCK_SUBMISSION,
        "official_tps": FLOORLOCK_TPS,           # literal M=1 AR: private == public, no drift
        "best_local_tps": FLOORLOCK_TPS,
        "private_mean_tps": FLOORLOCK_PRIVATE_MEAN,
        "private_95lo_tps": FLOORLOCK_PRIVATE_95LO,
        "private_95hi_tps": FLOORLOCK_PRIVATE_95HI,
        "private_worstcase_95lo_tps": FLOORLOCK_PRIVATE_95LO,  # zero breach => no separate WC
        "breach_frac": FLOORLOCK_BREACH_FRAC,
        "census_flips": FLOORLOCK_CENSUS_FLIPS,
        "semantic_flips": FLOORLOCK_SEMANTIC_FLIPS,
        "quality": _gate(FLOORLOCK_MMLU_PRO, FLOORLOCK_GPQA_DIAMOND),
        "quality_basis": "base by construction (literal greedy identity, M=1 AR)",
        "label": FLOORLOCK_LABEL,
        "ppl": SURGICAL_PPL,                     # literal greedy => base PPL
        "shipped": FLOORLOCK_SHIPPED,
        "draw_ready": "pre-staged fallback (locally validated, never launched)",
        "private_safe": True,
        "dossier": "#508 fn2v5wox",
    }
    surgical = {
        "name": "surgical-357",
        "submission": SURGICAL_SUBMISSION,
        "official_tps": SURGICAL_OFFICIAL_TPS,   # 375.857 -- the only OFFICIAL number on the board
        "best_local_tps": SURGICAL_LOCAL_TPS,
        "private_mean_tps": SURGICAL_PRIVATE_MEAN,
        "private_95lo_tps": SURGICAL_PRIVATE_95LO,
        "private_95hi_tps": SURGICAL_PRIVATE_95HI,
        "private_worstcase_95lo_tps": SURGICAL_WC24_95LO,  # refuted 24% worst-case downside
        "breach_frac": SURGICAL_BREACH_FRAC,
        "p_below_95pct": SURGICAL_P_BELOW_95PCT,
        "p_below_floorlock": SURGICAL_P_BELOW_FLOORLOCK,
        "census_flips": SURGICAL_CENSUS_FLIPS,
        "semantic_flips": SURGICAL_SEMANTIC_FLIPS,
        "census_margin_nat": SURGICAL_MARGIN_NAT,
        "quality": _gate(SURGICAL_MMLU_PRO, SURGICAL_GPQA_DIAMOND),
        "quality_basis": "base by construction (greedy-faithful; locus cert 1 bf16-ULP near-tie, 0 semantic)",
        "quality_worstcase_dacc_pct": SURGICAL_QUALITY_WORSTCASE_DACC_PCT,
        "quality_measured_pending": "ubel #511 served base-vs-ship MMLU-Pro/GPQA A/B (decisive |dAcc|; prior ~0)",
        "label": SURGICAL_LABEL,
        "ppl": SURGICAL_PPL,
        "shipped": SURGICAL_SHIPPED,
        "draw_ready": "SHIPPED (fired + landed official 375.857, stark #499; board publish held for human --publish)",
        "private_safe": False,                   # spec-alive: diverges from plain greedy off-public
        "dossier": "#508 fn2v5wox + #512 3fxrmc8u",
    }
    byteexact = {
        "name": "byteexact-399",
        "submission": BYTEEXACT_SUBMISSION,
        "official_tps": BYTEEXACT_OFFICIAL_TPS,  # 0 -- never launched; best-local only
        "best_local_tps": BYTEEXACT_LOCAL_MATCHED_TPS,   # 399.97 matched 32x256 (rung-vs-rung)
        "best_local_firetime_tps": BYTEEXACT_LOCAL_FIRETIME_TPS,  # 444.82 128x512 fire-time
        "private_mean_tps": None,                # NOT separately propagated -- honest gap (see note); null, never faked
        "private_95lo_tps": None,
        "private_95hi_tps": None,
        "private_worstcase_95lo_tps": None,
        "breach_frac": None,
        "census_flips": BYTEEXACT_CENSUS_FLIPS,
        "semantic_flips": BYTEEXACT_SEMANTIC_FLIPS,
        "census_margin_nat": BYTEEXACT_MARGIN_NAT,
        "served_self_det": BYTEEXACT_SERVED_SELF_DET,
        "quality": _gate(BYTEEXACT_MMLU_PRO, BYTEEXACT_GPQA_DIAMOND),
        "quality_basis": "base prior (0 semantic flips => 0 answer change; PPL-identical 2.37666)",
        "label": BYTEEXACT_LABEL,
        "ppl": BYTEEXACT_PPL,
        "shipped": BYTEEXACT_SHIPPED,
        "draw_ready": "staged + locally certified (served r1-r2=1.0), pending human draw-standard ruling (#500 fu#3)",
        "private_safe": False,
        "private_band_note": ("byteexact-399 has NO separate private-propagation run; it shares "
                              "surgical's spec-alive breach mechanism, so its private band would "
                              "track surgical's ~4.3% off its OWN public anchor -- not composed "
                              "here to avoid reporting an unmeasured band. Its official TPS is "
                              "UNMEASURED (never launched); +42 is a LOCAL matched-workload delta."),
        "dossier": "#500 m76qbs3l/feof8wtk/rvl5w50z",
    }
    return [floorlock, surgical, byteexact]


def _rung_by_name(rungs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for r in rungs:
        if r["name"] == name:
            return r
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# (2) Decision tree keyed on the human's risk preference                       #
# --------------------------------------------------------------------------- #
def decision_tree(rungs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Three risk-preference branches; each RESOLVES to a rung by a rule on rung fields.

    The resolution rules are data-driven (label / shipped / local TPS), so self_test can
    assert each branch lands on the intended rung without a hard-coded answer.
    """
    # Branch A: resolve to the literal-1.0, private-safe rung (lowest breach / guaranteed floor).
    branch_a_rung = min(
        (r for r in rungs if r["label"] == "literal-1.0" and r["private_safe"]),
        key=lambda r: r["breach_frac"],
    )
    # Branch B: resolve to the SHIPPED operative-1.0 (byte-identical-token, <=1-flip) rung.
    branch_b_rung = max(
        (r for r in rungs if r["label"] == "operative-1.0" and r["shipped"]),
        key=lambda r: r["official_tps"],
    )
    # Branch C: resolve to the fastest-best-local quality-safe rung whose token bar is operative
    #           (0-semantic) but NOT <=1-flip byte-identical-token.
    branch_c_rung = max(
        (r for r in rungs if r["semantic_flips"] == 0 and r["label"] == "quality-equivalent"),
        key=lambda r: r["best_local_tps"],
    )
    return [
        {
            "risk_preference": "zero private-speed-risk required",
            "predicate": "organizer invalidates on LITERAL private greedy identity, OR objective is maximin/guaranteed-floor",
            "resolves_to": branch_a_rung["name"],
            "headline_tps": branch_a_rung["official_tps"],
            "why": ("floor-lock is literal-byte-identical (M=1 AR, zero breach, private == public) -> the "
                    "only guaranteed-valid rung under a literal private-identity rule; it is the maximin "
                    "choice (surgical is operative-1.0 and risks a 0 under such a rule). #508 verdict leg (b2)."),
        },
        {
            "risk_preference": "max quality-safe speed, byte-identical tokens preferred",
            "predicate": "penalize-breach OR speed-threshold-invalidate with an expected-value objective; integrity posture wants <=1-flip token census",
            "resolves_to": branch_b_rung["name"],
            "headline_tps": branch_b_rung["official_tps"],
            "why": ("surgical-357 is SHIPPED at 375.857 official, operative-1.0 (1 bf16-ULP near-tie, 0 "
                    "semantic -> byte-identical-token at a >=1-ULP tolerance), quality-dominant (#512), and the "
                    "#508 bracketed-ship choice under penalize / speed-invalidate-E-value. Forward upgrade to a "
                    "byteexact variant ONLY if lawine lands a LITERAL-matmul one that reaches the <=1-flip bar "
                    "(#500 fu#2)."),
        },
        {
            "risk_preference": "max quality-safe speed, operative (0-semantic) acceptable",
            "predicate": "byte-identical-token NOT required; the realized scorer has no token-identity gate (#493) so 0-semantic quality-equivalent is sufficient",
            "resolves_to": branch_c_rung["name"],
            "headline_tps": branch_c_rung["best_local_tps"],
            "why": ("byteexact-399 is +42 local TPS over surgical (399.97 vs 357.6 matched 32x256; 444.82 "
                    "128x512 fire-time), 0 semantic, PPL-identical 2.37666, served-self-deterministic (r1-r2=1.0); "
                    "its only gap is 5 ULP near-ties e2e vs surgical's 1 (matmul-induced, sub-0.25-nat, no meaning "
                    "change). Gated on the human's draw-standard ruling (#500 fu#3) and one official launch to get a "
                    "real number (its 399.97 is LOCAL; official is unmeasured)."),
        },
    ]


# --------------------------------------------------------------------------- #
# (1) Recommended primary action -- TABLE-DRIVEN (not a hard-coded lean)        #
# --------------------------------------------------------------------------- #
def recommended_action(rungs: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the primary action, the fallback, and the forward upgrade from rung fields.

    primary_now  = the quality-valid rung already validated on official hardware
                   (shipped=True AND official_tps>0) with the highest official TPS.
    fallback      = the literal-1.0, private-safe rung (invalidation insurance).
    forward_upgrade = the fastest-best-local quality-safe, not-yet-shipped rung,
                      gated on the human's draw-standard ruling.
    """
    quality_valid = [r for r in rungs if r["quality"]["passes_gate"] and r["semantic_flips"] == 0]

    primary_candidates = [r for r in quality_valid if r["shipped"] and r["official_tps"] > 0]
    primary = max(primary_candidates, key=lambda r: r["official_tps"])

    fallback = min(
        (r for r in quality_valid if r["label"] == "literal-1.0" and r["private_safe"]),
        key=lambda r: r["breach_frac"],
    )

    upgrade_candidates = [r for r in quality_valid if not r["shipped"] and r["best_local_tps"] > primary["best_local_tps"]]
    forward_upgrade = max(upgrade_candidates, key=lambda r: r["best_local_tps"]) if upgrade_candidates else None

    return {
        "primary_now": primary["name"],
        "primary_official_tps": primary["official_tps"],
        "primary_label": primary["label"],
        "primary_basis": (
            "surgical-357 is the only rung already FIRED + landed on official a10g-small "
            "(375.857, stark #499 j7qao5e9), it is quality-DOMINANT (#512: PASSES Morgan's "
            "gate while pruned competitors collapse), operative-1.0 (1 bf16-ULP near-tie, 0 "
            "semantic), and the #508 bracketed-SHIP choice under penalize / speed-invalidate-"
            "E-value. The score is already locked in the results bucket; the only remaining act "
            "is the human's --publish nod."),
        "fallback": fallback["name"],
        "fallback_official_tps": fallback["official_tps"],
        "fallback_basis": (
            "floor-lock-166.23 (literal-1.0, private-safe) is pre-staged invalidation insurance: "
            "swap in IFF the reopen rule invalidates on literal private greedy identity OR the "
            "objective is maximin. P(surgical < floor-lock on raw TPS) ~ 0 (51sigma), so floor-lock "
            "is NEVER a speed case -- purely a validity hedge (#508 leg b2)."),
        "forward_upgrade": forward_upgrade["name"] if forward_upgrade else None,
        "forward_upgrade_best_local_tps": forward_upgrade["best_local_tps"] if forward_upgrade else None,
        "forward_upgrade_gate": (
            "byteexact-399 (+42 local TPS, 0 semantic, PPL-identical, served r1-r2=1.0) is the forward "
            "speed-upgrade -- but it is NOT yet actionable: (1) it has no official number (one launch "
            "needed), and (2) it is operative/quality-equivalent (5 ULP e2e), not byte-identical-token "
            "tight to surgical's <=1-flip bar, so it requires the human's draw-standard ruling (#500 fu#3). "
            "If the human wants a strictly byte-identical-token rung >357, hold for lawine's literal-matmul "
            "variant (#500 fu#2) instead."),
        "decisive_fork": (
            "Reopen pivots on TWO axes: (A) the private validity rule/objective -- literal-private-identity "
            "or maximin -> floor-lock; penalize or speed-invalidate-E-value -> the fastest valid rung; and "
            "(B) token-identity strictness of the speed rung -- byte-identical-token required -> surgical-357 "
            "(shipped); 0-semantic operative acceptable -> byteexact-399. There is NO axis on which DOWNSTREAM "
            "QUALITY hurts us: every rung is quality = base, so the pause selects FOR us."),
    }


# --------------------------------------------------------------------------- #
# Build + self-test                                                            #
# --------------------------------------------------------------------------- #
def build_results() -> dict[str, Any]:
    rungs = build_rungs()
    pruned_gate = _gate(PRUNED_MMLU_PRO, PRUNED_GPQA_DIAMOND)
    tree = decision_tree(rungs)
    rec = recommended_action(rungs)

    surgical = _rung_by_name(rungs, "surgical-357")
    byteexact = _rung_by_name(rungs, "byteexact-399")
    floorlock = _rung_by_name(rungs, "floor-lock-166.23")

    # economics: byteexact local matched-workload delta over surgical (the +42 headline).
    byteexact_local_delta = BYTEEXACT_LOCAL_MATCHED_TPS - SURGICAL_LOCAL_TPS
    surgical_over_floorlock_official = SURGICAL_OFFICIAL_TPS - FLOORLOCK_TPS

    one_line = (
        "REOPEN ACTION: publish/fire surgical-357 (SHIPPED, 375.857 official j7qao5e9) as primary -- "
        "it is the only rung on official hardware, quality-DOMINANT (#512: MMLU-Pro %.3f / GPQA %.3f "
        "PASS Morgan's gate >= %.2f/%.2f while pruned competitors collapse %.3f/%.3f), operative-1.0 "
        "(1 bf16-ULP near-tie, 0 semantic), and the #508 bracketed-SHIP choice. HOLD floor-lock-166.23 "
        "(literal-1.0, private-safe) as invalidation insurance; STAGE byteexact-399 (+%.1f local, 0 "
        "semantic, 5 ULP e2e) as the forward speed-upgrade pending the human's draw-standard ruling "
        "(#500 fu#3). Decision tree: zero-private-risk -> floor-lock; byte-identical-token preferred -> "
        "surgical-357; operative 0-semantic acceptable -> byteexact-399. No reopen rule lets quality "
        "hurt us."
        % (SURGICAL_MMLU_PRO, SURGICAL_GPQA_DIAMOND, GATE_MMLU_PRO, GATE_GPQA_DIAMOND,
           PRUNED_MMLU_PRO, PRUNED_GPQA_DIAMOND, byteexact_local_delta))

    eli5 = (
        "We have three saved-up race cars and one race that just got a new safety inspection. "
        "(1) The SLOW-BUT-SPOTLESS car (floor-lock, 166 mph) is guaranteed to pass any inspection no "
        "matter how strict -- it's our 'never gets disqualified' backup. (2) The FAST car we ALREADY "
        "drove and that already posted a real lap time (surgical-357, 375.857) is clean enough to pass "
        "the new safety check easily (its answers match the original model exactly), and it's the one "
        "we recommend showing the moment the race reopens. (3) The FASTEST car in practice (byteexact-399, "
        "+42 in testing) gives the same answers but its internal math wobbles in the last decimal a few "
        "more times -- harmless, but we'd want the human to say 'that's fine' and let us drive it once for "
        "real before trusting that lap. The new safety inspection is GOOD for us: every one of our cars "
        "passes it, while the popular shortcut cars (pruned models) flunk it badly.")

    dossier_verdict = (
        "PRIMARY: publish surgical-357 (375.857 official, j7qao5e9) -- the only rung already fired on "
        "a10g-small, quality-DOMINANT (#512), operative-1.0, and the #508 bracketed-SHIP choice; its score "
        "is locked, only the human --publish nod remains. FALLBACK: floor-lock-166.23 (literal-1.0, "
        "private-safe) as invalidation insurance, swapped in IFF the reopen rule invalidates on literal "
        "private greedy identity or the objective is maximin (P(surgical < floor-lock on raw TPS) ~ 0). "
        "FORWARD UPGRADE: byteexact-399 (+%.1f local, 0 semantic, PPL-identical, served r1-r2=1.0) -- the "
        "fastest rung, but gated on (a) one official launch for a real number and (b) the human's "
        "draw-standard ruling (#500 fu#3), since it is operative/quality-equivalent (5 ULP e2e) not "
        "byte-identical-token tight. Decision tree keys on two axes -- private validity rule (literal/maximin "
        "-> floor-lock; penalize/E-value -> fastest valid) and token-identity strictness (byte-identical "
        "-> surgical; operative-ok -> byteexact). Across ALL of them downstream quality is base, so no "
        "organizer rule lets the quality pause hurt the ship."
        % byteexact_local_delta)

    results = {
        "pr": 517,
        "agent": "kanna",
        "analysis_only": True,
        "official_tps": 0,
        "no_serve": True,
        "no_hf_job": True,
        "no_launch": True,
        "no_submission": True,
        "no_served_file_change": True,
        "no_evals_run": True,
        "lane_discipline": ("pure cross-axis composition of three BANKED dossiers (#508 fn2v5wox speed/"
                            "private, #512 3fxrmc8u quality, #500 m76qbs3l byteexact economics); re-derives "
                            "NOTHING -- only re-states the banked verdicts and asserts they cohere into a "
                            "single table-driven recommendation + risk-keyed decision tree."),
        "source_runs": SOURCE_RUNS,
        # banked dossier verdicts folded
        "speed_private_verdict": SPEED_VERDICT,
        "quality_verdict": QUALITY_VERDICT,
        # (3) per-rung risk/reward table
        "rungs": rungs,
        "pruned_competitor_gate": pruned_gate,
        # (2) decision tree
        "decision_tree": tree,
        # (1) recommended action
        "recommendation": rec,
        # economics
        "economics": {
            "byteexact_local_matched_delta_over_surgical_tps": byteexact_local_delta,
            "byteexact_local_matched_tps": BYTEEXACT_LOCAL_MATCHED_TPS,
            "byteexact_local_firetime_tps": BYTEEXACT_LOCAL_FIRETIME_TPS,
            "surgical_local_tps": SURGICAL_LOCAL_TPS,
            "surgical_official_tps": SURGICAL_OFFICIAL_TPS,
            "surgical_over_floorlock_official_tps": surgical_over_floorlock_official,
            "official_faster_than_pod_note": ("surgical's 375.857 official is +5.2% ABOVE its ~357.6 local "
                                              "pod (official-faster-than-pod, NOT drift); the private band "
                                              "~342 is a SEPARATE local->private 4.3% breach anchored on the "
                                              "357.22 local public anchor (#508)."),
        },
        # one-pagers
        "one_line_summary": one_line,
        "eli5": eli5,
        "dossier_verdict": dossier_verdict,
    }
    results["self_test"] = self_test(results)
    return results


def self_test(r: dict[str, Any]) -> dict[str, Any]:
    rungs = r["rungs"]
    floorlock = _rung_by_name(rungs, "floor-lock-166.23")
    surgical = _rung_by_name(rungs, "surgical-357")
    byteexact = _rung_by_name(rungs, "byteexact-399")
    tree = r["decision_tree"]
    rec = r["recommendation"]
    pruned = r["pruned_competitor_gate"]
    checks: dict[str, bool] = {}

    # (A) PROVENANCE -- every headline number reproduces its banked dossier exactly.
    checks["surgical_official_is_375857"] = surgical["official_tps"] == 375.857
    checks["surgical_private_mean_matches_508"] = abs(surgical["private_mean_tps"] - SURGICAL_PRIVATE_MEAN) < 1e-9
    checks["surgical_private_95band_ordered"] = surgical["private_95lo_tps"] < surgical["private_mean_tps"] < surgical["private_95hi_tps"]
    checks["surgical_wc24_matches_508"] = abs(surgical["private_worstcase_95lo_tps"] - 266.1661486573638) < 1e-6
    checks["floorlock_is_16623"] = floorlock["official_tps"] == 166.23
    checks["floorlock_zero_breach"] = floorlock["breach_frac"] == 0.0
    checks["byteexact_matched_local_is_39997"] = abs(byteexact["best_local_tps"] - 399.97443199055806) < 1e-9
    checks["byteexact_firetime_is_44482"] = abs(byteexact["best_local_firetime_tps"] - 444.82227295018606) < 1e-9
    checks["byteexact_official_is_zero"] = byteexact["official_tps"] == 0.0  # never launched

    # (B) RUNG TABLE integrity: exactly 3 rungs, official-TPS ordering, distinct labels.
    checks["three_rungs"] = len(rungs) == 3
    checks["distinct_labels"] = len({rg["label"] for rg in rungs}) == 3
    checks["labels_are_the_three_axes"] = {rg["label"] for rg in rungs} == {"literal-1.0", "operative-1.0", "quality-equivalent"}
    checks["surgical_official_above_floorlock"] = surgical["official_tps"] > floorlock["official_tps"]
    checks["byteexact_local_above_surgical_local"] = byteexact["best_local_tps"] > surgical["best_local_tps"]

    # (C) QUALITY: all three rungs are quality-valid (= base, 0 semantic); pruned competitor FAILS.
    checks["all_rungs_pass_quality_gate"] = all(rg["quality"]["passes_gate"] for rg in rungs)
    checks["all_rungs_zero_semantic"] = all(rg["semantic_flips"] == 0 for rg in rungs)
    checks["pruned_competitor_fails_gate"] = pruned["passes_gate"] is False
    checks["pruned_gpqa_near_chance"] = abs(pruned["gpqa"] - CHANCE_GPQA_DIAMOND) < 0.05
    checks["surgical_gate_margins_positive"] = surgical["quality"]["margin_mmlu"] > 0 and surgical["quality"]["margin_gpqa"] > 0

    # (D) SPEED/PRIVATE: surgical dominates floor-lock on raw TPS under every draw (P~0).
    checks["surgical_p_below_floorlock_zero"] = surgical["p_below_floorlock"] == 0.0
    checks["surgical_95lo_beats_floorlock_95hi"] = surgical["private_95lo_tps"] > floorlock["private_95hi_tps"]
    checks["surgical_wc24_beats_floorlock_95hi"] = surgical["private_worstcase_95lo_tps"] > floorlock["private_95hi_tps"]
    checks["floorlock_private_safe"] = floorlock["private_safe"] is True and surgical["private_safe"] is False

    # (E) ECONOMICS: byteexact +42 local; the +42 is a LOCAL delta, official UNMEASURED.
    econ = r["economics"]
    checks["byteexact_local_delta_about_42"] = 41.0 < econ["byteexact_local_matched_delta_over_surgical_tps"] < 43.0
    checks["byteexact_private_band_is_null_not_faked"] = byteexact["private_mean_tps"] is None
    checks["official_faster_than_pod_positive"] = econ["surgical_official_tps"] > econ["surgical_local_tps"]

    # (F) DECISION TREE: 3 branches, each resolves to the INTENDED rung by its data-driven rule.
    checks["tree_has_three_branches"] = len(tree) == 3
    by_pref = {b["risk_preference"]: b for b in tree}
    checks["branch_zero_risk_to_floorlock"] = by_pref["zero private-speed-risk required"]["resolves_to"] == "floor-lock-166.23"
    checks["branch_byte_identical_to_surgical"] = by_pref["max quality-safe speed, byte-identical tokens preferred"]["resolves_to"] == "surgical-357"
    checks["branch_operative_ok_to_byteexact"] = by_pref["max quality-safe speed, operative (0-semantic) acceptable"]["resolves_to"] == "byteexact-399"
    checks["tree_branches_cover_all_three_rungs"] = {b["resolves_to"] for b in tree} == {rg["name"] for rg in rungs}

    # (G) RECOMMENDATION: primary=shipped surgical; fallback=literal floor-lock; forward=byteexact.
    checks["primary_is_surgical"] = rec["primary_now"] == "surgical-357"
    checks["primary_is_shipped_official"] = surgical["shipped"] is True and rec["primary_official_tps"] == 375.857
    checks["fallback_is_floorlock"] = rec["fallback"] == "floor-lock-166.23"
    checks["forward_upgrade_is_byteexact"] = rec["forward_upgrade"] == "byteexact-399"
    checks["primary_label_operative1p0"] = rec["primary_label"] == "operative-1.0"

    # (H) LABEL CONSISTENCY with each banked dossier verdict.
    checks["speed_verdict_bracketed"] = r["speed_private_verdict"] == "bracketed"
    checks["quality_verdict_dominant"] = r["quality_verdict"] == "dominant"
    checks["floorlock_literal_label"] = floorlock["label"] == "literal-1.0" and floorlock["census_flips"] == 0
    checks["surgical_operative_label"] = surgical["label"] == "operative-1.0" and surgical["census_flips"] == 1
    checks["byteexact_quality_equiv_label"] = byteexact["label"] == "quality-equivalent" and byteexact["census_flips"] == 5

    # (I) PR DISCIPLINE: this composer is itself analysis_only / official_tps==0 / PPL gate respected.
    checks["pr_analysis_only"] = r["analysis_only"] is True and r["official_tps"] == 0
    checks["all_rungs_ppl_under_cap"] = all(rg["ppl"] <= PPL_CAP for rg in rungs)
    checks["scorer_has_no_token_identity_gate"] = SCORER_HAS_TOKEN_IDENTITY_GATE is False

    # (J) finite over every numeric leaf (the unmeasured byteexact private band is null, not NaN).
    checks["all_numeric_leaves_finite"] = _all_finite(r)

    return {"checks": checks, "passes": all(checks.values())}


def _all_finite(obj: Any) -> bool:
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite(v) for v in obj)
    return True  # strings / None


# --------------------------------------------------------------------------- #
# Pretty-print + W&B                                                          #
# --------------------------------------------------------------------------- #
def _print(r: dict[str, Any]) -> None:
    rungs = r["rungs"]
    rec = r["recommendation"]
    print("\n[capstone] ===== CROSS-AXIS REOPEN-ACTION CAPSTONE (PR #517) =====", flush=True)
    print("  COMPOSITION: folds #508 (speed/private, %s) + #512 (quality, %s) + #500 (byteexact econ) into ONE action"
          % (r["speed_private_verdict"], r["quality_verdict"]), flush=True)
    print("  -- per-rung risk/reward table --", flush=True)
    print("    %-18s %10s %10s %14s %9s %18s %s" % (
        "rung", "off_tps", "local_tps", "private_mean", "sem/flips", "label", "draw_ready"), flush=True)
    for rg in rungs:
        pm = rg["private_mean_tps"]
        pm_s = "n/a" if pm is None else "%.1f" % pm
        print("    %-18s %10.3f %10.1f %14s %4d/%-4d %18s %s" % (
            rg["name"], rg["official_tps"], rg["best_local_tps"], pm_s,
            rg["semantic_flips"], rg["census_flips"], rg["label"],
            "SHIPPED" if rg["shipped"] else rg["draw_ready"][:34]), flush=True)
    print("  -- decision tree (keyed on risk preference) --", flush=True)
    for b in r["decision_tree"]:
        print("    [%s] -> %s (%.1f)" % (b["risk_preference"], b["resolves_to"], b["headline_tps"]), flush=True)
    print("  -- RECOMMENDATION --", flush=True)
    print("    PRIMARY  : %s (%.3f official)" % (rec["primary_now"], rec["primary_official_tps"]), flush=True)
    print("    FALLBACK : %s (%.2f, invalidation insurance)" % (rec["fallback"], rec["fallback_official_tps"]), flush=True)
    print("    FORWARD  : %s (%.1f best-local, gated on draw-standard ruling)" % (
        rec["forward_upgrade"], rec["forward_upgrade_best_local_tps"]), flush=True)
    print("  SELF-TEST passes = %s (%d checks)" % (r["self_test"]["passes"], len(r["self_test"]["checks"])), flush=True)
    if not r["self_test"]["passes"]:
        for k, v in r["self_test"]["checks"].items():
            if not v:
                print("    FAILED: %s" % k, flush=True)
    print("\n  ONE-LINE: %s" % r["one_line_summary"], flush=True)
    print("\n  ELI5: %s" % r["eli5"], flush=True)
    print("\n  VERDICT: %s" % r["dossier_verdict"], flush=True)


def _flat_summary(r: dict[str, Any]) -> dict[str, float | int]:
    rungs = r["rungs"]
    floorlock = _rung_by_name(rungs, "floor-lock-166.23")
    surgical = _rung_by_name(rungs, "surgical-357")
    byteexact = _rung_by_name(rungs, "byteexact-399")
    rec = r["recommendation"]
    econ = r["economics"]
    flat = {
        # KEY OUTPUTS (PR #517)
        "n_rungs": len(rungs),
        "primary_is_surgical": int(rec["primary_now"] == "surgical-357"),
        "primary_official_tps": rec["primary_official_tps"],
        "fallback_is_floorlock": int(rec["fallback"] == "floor-lock-166.23"),
        "forward_upgrade_is_byteexact": int(rec["forward_upgrade"] == "byteexact-399"),
        "speed_verdict_bracketed": int(r["speed_private_verdict"] == "bracketed"),
        "quality_verdict_dominant": int(r["quality_verdict"] == "dominant"),
        # per-rung official / local TPS
        "floorlock_official_tps": floorlock["official_tps"],
        "surgical_official_tps": surgical["official_tps"],
        "surgical_local_tps": surgical["best_local_tps"],
        "byteexact_local_matched_tps": byteexact["best_local_tps"],
        "byteexact_local_firetime_tps": byteexact["best_local_firetime_tps"],
        "byteexact_official_tps": byteexact["official_tps"],
        # private band (surgical + floor-lock)
        "surgical_private_mean_tps": surgical["private_mean_tps"],
        "surgical_private_95lo_tps": surgical["private_95lo_tps"],
        "surgical_private_95hi_tps": surgical["private_95hi_tps"],
        "surgical_private_worstcase_95lo_tps": surgical["private_worstcase_95lo_tps"],
        "surgical_breach_frac": surgical["breach_frac"],
        "surgical_p_below_floorlock": surgical["p_below_floorlock"],
        "floorlock_private_mean_tps": floorlock["private_mean_tps"],
        "floorlock_private_95hi_tps": floorlock["private_95hi_tps"],
        # token-identity census
        "floorlock_census_flips": floorlock["census_flips"],
        "surgical_census_flips": surgical["census_flips"],
        "byteexact_census_flips": byteexact["census_flips"],
        "floorlock_semantic_flips": floorlock["semantic_flips"],
        "surgical_semantic_flips": surgical["semantic_flips"],
        "byteexact_semantic_flips": byteexact["semantic_flips"],
        # quality
        "surgical_mmlu_pro": surgical["quality"]["mmlu"],
        "surgical_gpqa_diamond": surgical["quality"]["gpqa"],
        "gate_mmlu_pro": GATE_MMLU_PRO,
        "gate_gpqa_diamond": GATE_GPQA_DIAMOND,
        "pruned_mmlu_pro": PRUNED_MMLU_PRO,
        "pruned_gpqa_diamond": PRUNED_GPQA_DIAMOND,
        "all_rungs_pass_quality_gate": int(all(rg["quality"]["passes_gate"] for rg in rungs)),
        # economics
        "byteexact_local_delta_over_surgical_tps": econ["byteexact_local_matched_delta_over_surgical_tps"],
        "surgical_over_floorlock_official_tps": econ["surgical_over_floorlock_official_tps"],
        # ppl
        "surgical_ppl": surgical["ppl"],
        "byteexact_ppl": byteexact["ppl"],
        "ppl_cap": PPL_CAP,
        # discipline
        "analysis_only": int(r["analysis_only"]),
        "official_tps": r["official_tps"],
        "self_test_passes": int(r["self_test"]["passes"]),
        "self_test_n_checks": len(r["self_test"]["checks"]),
    }
    return {k: v for k, v in flat.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="kanna/reopen-action-capstone")
    ap.add_argument("--group", default="reopen-capstone")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    r = build_results()
    _print(r)

    out_path = Path(__file__).resolve().parent / "capstone.json"
    out_path.write_text(json.dumps(r, indent=2))
    print("\n[capstone] artifacts -> %s" % out_path, flush=True)

    if not r["self_test"]["passes"]:
        print("[capstone] SELF-TEST FAILED -- not logging to W&B", flush=True)
        return 1
    if args.no_wandb:
        return 0

    run = wandb_logging.init_wandb_run(
        job_type="reopen-action-capstone", agent="kanna",
        name=args.name, group=args.group,
        tags=["reopen-capstone", "cross-axis", "decision-tree", "reopen-decision",
              "surgical357", "floor-lock", "byteexact399", "speed-private", "downstream-quality",
              "analysis-only"],
        notes="Cross-axis reopen-action capstone: one recommended fire + a risk-keyed decision tree.",
        config={
            "pr": 517,
            "speed_verdict": SPEED_VERDICT,
            "quality_verdict": QUALITY_VERDICT,
            "surgical_official_tps": SURGICAL_OFFICIAL_TPS,
            "floorlock_tps": FLOORLOCK_TPS,
            "byteexact_local_matched_tps": BYTEEXACT_LOCAL_MATCHED_TPS,
            "gate_mmlu_pro": GATE_MMLU_PRO,
            "gate_gpqa_diamond": GATE_GPQA_DIAMOND,
            "analysis_only": True, "official_tps": 0,
            "source_runs": list(SOURCE_RUNS.values()),
        },
    )
    if run is None:
        print("[capstone] wandb disabled (no API key); skipping", flush=True)
        return 0
    wandb_logging.log_summary(run, _flat_summary(r), step=0)
    wandb_logging.log_json_artifact(
        run, name="reopen_action_capstone", artifact_type="reopen-action-capstone", data=r)
    wandb_logging.finish_wandb(run)
    print("[capstone] wandb_run_id=%s" % getattr(run, "id", None), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
