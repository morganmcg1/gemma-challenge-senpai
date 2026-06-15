#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""REST gate-free >500 feasibility (#348, kanna): a no-retrain method on OUR reasoning eval?

THE GOVERNING QUESTION (#319: every priced >500 path needs a human-gated GPU spend)
-----------------------------------------------------------------------------------
Every >500 lane priced so far needs an approval-gated cluster train/launch: the EAGLE-3 head
coverage retrain (wirbel #343) and the tree build (land #245) both demand it. The open question is
whether there is a >500 method we can deploy with NO gated retrain. The researcher-agent's non-EAGLE-3
top-5 flagged REST -- Retrieval-Based Speculative Decoding (He et al. 2024, arXiv:2311.08252) -- as the
one method needing NO draft training and NO cluster retrain: it drafts candidate continuations by
RETRIEVING from a datastore (a suffix array built OFFLINE on CPU from a corpus), so it sidesteps the
#319 demand/retrain blocker entirely. But REST shines on code/repetitive/in-context text and is weak on
novel reasoning, and OUR eval is 100% reasoning/STEM (lawine #330). So:

    Is there a no-retrain >500 method, and does REST's accepted-length hold on OUR eval?

This card prices that. It is a CPU-analytic SCREEN: no GPU, no datastore build, no model forward, no HF
Job, no launch, 0 official TPS. BASELINE stays 481.53; adds 0 TPS.

THE COMPARISON AXIS (E[T] = expected tokens emitted per target forward step, incl. the bonus token)
---------------------------------------------------------------------------------------------------
REST's published quantity is "mean generation length" M (a.k.a. mean accepted length) -- tokens emitted
per target forward step, with M=1.0 the no-draft floor (the always-accepted bonus token). That is the
SAME axis as the deployed spec head's E[T] (denken #289 fi34s269 decomposed E[T]=1+sum G(k)=3.8512 from
its per-position acceptance a_k). So REST's M is directly comparable to E[T]_eagle3.

  REST mean accepted length M (He et al. 2024, Tables 1-2; carried as RANGES, workload-dependent):
    * code  (HumanEval, LLaMA-2 7B/13B):  M in [2.53, 2.69]   <- REST's STRONG case (NOT our eval)
    * chat  (MT-Bench,  LLaMA-2 7B/13B):  M in [1.97, 2.01]   <- closest published analog to general text
    * reasoning / math / multi-step CoT:  NO published REST number; principled no-draft FLOOR ~1.0 as
                                          retrieval hit-rate falls on novel CoT vocabulary.
  OUR eval is 100% reasoning/STEM (mmlu_pro 57 / gpqa 57 / aime 14, lawine #330) -- REST's WEAK case.

The decisive structural fact: even REST's GLOBAL BEST published number (code, M=2.69) is BELOW the
deployed spec head's E[T]=3.8512. So REST under-accepts vs the EXISTING already-priced lane on EVERY
workload, and on OUR novel-reasoning eval it is far worse (band [1.0 floor, 2.01 generous chat upper]).

THE PPL-ONLY CEILING (step 3 formula, given EXACTLY by the PR)
-------------------------------------------------------------
    REST_ppl_only_ceiling = 520.953 * (E[T]_REST / E[T]_eagle3)
The deployed head's own PPL-only ceiling under this formula is 520.953*(E[T]_eagle3/E[T]_eagle3)=520.953
(the lambda=1 ceiling). REST beats it iff E[T]_REST > E[T]_eagle3 -- which it NEVER does (ratio<1 even
at the code sanity bound). The headline TEST ratio rest_etratio_vs_eagle3 uses the GENEROUS chat-analog
upper bound (M=2.01, a MEASURED number that OVER-states reasoning) as the steel-man: even there the
ratio is 0.522 and the ceiling 271.9 TPS -- below BOTH 500 and the deployed 481.53.

THE STRICT SUPPLY TAX (does REST inherit denken #332's batched-verify BW floor?)
-------------------------------------------------------------------------------
REST verifies a Trie of up to c=64 retrieved candidates in a SINGLE batched tree-attention forward
(m=10 max draft tokens/step; architecturally Medusa/SpecInfer-class). That IS a batched multi-token
verify forward, so REST inherits denken #332's (y5cl0ena) deterministic-attention bandwidth floor:
inherits_332_supply_tax=True. Under a STRICT greedy-identity gate its ceiling is additionally taxed to
520.953*ratio*(1-0.09103) -- but REST is already demand-capped far below 500, so the supply tax is moot.

THE OPERATIONAL VERDICT (the distinctive deliverable -- step 4)
--------------------------------------------------------------
REST IS gate-free deployable (offline CPU suffix array, NO gradient train, NO #319 gated cluster retrain,
exact Leviathan-2023 accept rule => PPL-preserving). That advantage is REAL but does NOT rescue it: on
OUR 100%-reasoning eval REST under-accepts so badly (E[T] band [1.0, 2.01] vs the deployed 3.8512) that
its PPL-only ceiling (135-272 TPS) sits far below the existing EAGLE-3 head's already-priced lane
(520.953 ceiling / 481.53 deployed). rest_is_gatefree_deployable=True; rest_is_viable_500_path=False.
The no-retrain advantage is irrelevant when the method is demand-capped below the existing lane.

LOCAL, CPU-ONLY, ANALYTIC. 0 GPU, no model forward, no datastore build, no training, no publish, no HF
Job, no submission, no served-file change, no official draw. Imports verbatim: denken #289 fi34s269
(a_k -> E[T]=3.8512 via 1+sum cumprod), denken #332 y5cl0ena (supply floor 0.09103, strict ceiling
473.5, lambda ceiling 520.953), lawine #330 hfrscdai (eval = 100% reasoning/STEM, 57/57/14, cov prior
0.8903). Literature: He et al. 2024 REST (arXiv:2311.08252); Leviathan et al. ICML 2023 (arXiv:2211.17192,
distributional guarantee). ORTHOGONAL to stark #345 (method-CLASS supply-tax screen; REST is a DISTINCT
retrieval class and this card owns the gate-free / no-retrain operational angle).

PRIMARY metric  rest_feasibility_self_test_passes
TEST    metric  rest_etratio_vs_eagle3        (float: E[T]_REST_generous_upper / E[T]_eagle3)
TEST    metric  rest_is_gatefree_deployable   (bool:  REST needs NO gated retrain -- True)
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
# Imported EXACT from banked W&B runs. Re-derive NOTHING. The PR's displayed forms (3.8512, 473.5,
# 520.953, 0.09103, 0.8903) are round-to-display values, asserted exact in the self-test.
# --------------------------------------------------------------------------- #
K_SPEC = 7                                     # deployed speculative depth (denken #289 / lawine #282)

# ---- denken #289 (fi34s269): deployed/existing spec-head per-position acceptance -> E[T]. ----
# #289 decomposed the DEPLOYED (already-priced) spec head; the PR labels it the "EAGLE-3 head".
# Whole-run spec counters (W&B 2j0e8xgg, banked by lawine #282); a_k = G(k)/G(k-1); E[T]=1+sum cumprod.
BANKED_ACCEPTED_PER_POS = [12452.0, 9458.0, 7500.0, 6171.0, 5152.0, 4306.0, 3645.0]
BANKED_NUM_DRAFTS = 17075.0
BANKED_NUM_ACCEPTED = 48684.0                  # == sum(BANKED_ACCEPTED_PER_POS)
A_K_289_DISPLAY = [0.72925, 0.75956, 0.79298, 0.82280, 0.83487, 0.83579, 0.84649]  # 5-dp display
E_T_EAGLE3_DISPLAY = 3.8512                    # 1 + 48684/17075 (denken #289 / lawine #282)

# ---- denken #332 (y5cl0ena): method-INDEPENDENT batched-verify supply floor + strict ceiling. ----
LAMBDA_CEIL = 520.9527323111674                # lambda=1 ceiling at E[T]_eagle3 (int4-spec BI verify)
SUPPLY_FLOOR_GEO = 0.09103155435261377         # forgone-parallelism fraction @ geometric phi (M=8 SDPA)
STRICT_CEILING_332 = 473.5295953446407         # LAMBDA_CEIL * (1 - SUPPLY_FLOOR_GEO)

# ---- lawine #330 (hfrscdai): official 128-prompt eval composition (100% reasoning/STEM). ----
EVAL_SRC_COUNTS = {"mmlu_pro": 57, "gpqa": 57, "aime": 14}   # by id prefix; n = 128
EVAL_N = 128
COV_PRIOR = 0.8903                             # fusion top-4 c_eff on the reasoning holdout (context)

# ---- REST literature (He et al. 2024, arXiv:2311.08252): mean accepted length M (tokens/step). ----
# M=1.0 is the no-draft floor (always-accepted bonus). Same axis as E[T]_eagle3. Carried as RANGES.
REST_M_CODE = (2.53, 2.69)        # HumanEval, LLaMA-2 7B/13B greedy/nucleus (Table 1) -- STRONG, not our eval
REST_M_CHAT = (1.97, 2.01)        # MT-Bench,  LLaMA-2 7B/13B greedy/nucleus (Table 1) -- closest analog
REST_M_REASONING_FLOOR = 1.0      # NO published REST reasoning M; principled no-draft floor (retrieval miss)
# Datastore-size ablation (Table 2, code domain, LLaMA-2 7B greedy): GB -> M. Log-linear, diminishing;
# out-of-domain collapses to ~1.0 (domain match dominates size).
REST_DATASTORE_SCALING_GB_M = [(0.9, 1.96), (4.4, 2.18), (8.7, 2.35), (14.0, 2.45), (27.0, 2.65)]
# REST verify structure: Trie of up to c=64 candidates, single batched tree-attention forward, m<=10.
REST_VERIFY_TRIE_CANDIDATES = 64
REST_VERIFY_MAX_DRAFT_TOKENS = 10

# REST E[T] estimates ON OUR 100%-reasoning eval (band; the chat upper is the GENEROUS steel-man).
REST_ET_OUR_EVAL_FLOOR = REST_M_REASONING_FLOOR          # 1.00  (pure retrieval miss on novel CoT)
REST_ET_OUR_EVAL_GENEROUS_UPPER = REST_M_CHAT[1]         # 2.01  (= MEASURED chat M; OVER-states reasoning)
REST_ET_GLOBAL_BEST_SANITY = REST_M_CODE[1]              # 2.69  (REST's best published M overall, code)

TARGET = 500.0
BASELINE_TPS = 481.53

TOL_EXACT = 1e-9
TOL_289 = 1e-6           # reproduce denken #289 E[T] from a_k
TOL_332 = 1e-6           # reproduce denken #332 strict ceiling
TOL_AK_DISPLAY = 1e-4    # a_k matches its 5-dp display
TOL_DISPLAY_C = 5e-5
TOL_DISPLAY_TPS = 5e-3


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _cumprod(xs: list[float]) -> list[float]:
    out, acc = [], 1.0
    for x in xs:
        acc *= x
        out.append(acc)
    return out


# --------------------------------------------------------------------------- #
# Core laws.
# --------------------------------------------------------------------------- #
def eagle3_et_from_ak() -> dict[str, Any]:
    """Reproduce the deployed/existing spec head's E[T] from denken #289's per-position acceptance.

    G(m)=P(j>=m)=accepted_per_pos[m-1]/num_drafts; a_k=G(k)/G(k-1); cumprod(a_k) == G(1..K) by
    construction; E[T] = 1 + sum cumprod(a_k) = 1 + sum G = 1 + num_accepted/num_drafts.
    """
    nd = BANKED_NUM_DRAFTS
    G = [1.0] + [BANKED_ACCEPTED_PER_POS[m - 1] / nd for m in range(1, K_SPEC + 1)]
    a_k = [G[m] / G[m - 1] for m in range(1, K_SPEC + 1)]
    cum = _cumprod(a_k)                                   # == G(1..K)
    e_t = 1.0 + sum(cum)
    e_t_counter = 1.0 + BANKED_NUM_ACCEPTED / nd          # independent check
    roundtrip_cum_vs_G = max(abs(cum[i] - G[i + 1]) for i in range(K_SPEC))
    ak_vs_display = max(abs(a_k[i] - A_K_289_DISPLAY[i]) for i in range(K_SPEC))
    return {
        "survival_G": G,
        "a_k": a_k,
        "cumprod_a_k": cum,
        "E_T_eagle3": e_t,
        "E_T_eagle3_from_counter": e_t_counter,
        "E_T_eagle3_display": E_T_EAGLE3_DISPLAY,
        "roundtrip_cumprod_eq_survival": roundtrip_cum_vs_G,
        "a_k_vs_289_display_maxabs": ak_vs_display,
        "reproduces_289_le_1e6": bool(
            abs(e_t - e_t_counter) <= TOL_EXACT
            and abs(e_t - E_T_EAGLE3_DISPLAY) <= 5e-5
            and roundtrip_cum_vs_G <= 1e-12
            and ak_vs_display <= TOL_AK_DISPLAY),
    }


def rest_ppl_only_ceiling(e_t_rest: float, e_t_eagle3: float) -> float:
    """PR step-3 formula, EXACTLY: 520.953 * (E[T]_REST / E[T]_eagle3)."""
    return LAMBDA_CEIL * (e_t_rest / e_t_eagle3)


def rest_strict_ceiling(e_t_rest: float, e_t_eagle3: float) -> float:
    """STRICT-gated: PPL-only ceiling further taxed by denken #332's supply floor (batched verify)."""
    return rest_ppl_only_ceiling(e_t_rest, e_t_eagle3) * (1.0 - SUPPLY_FLOOR_GEO)


# --------------------------------------------------------------------------- #
# (D1) REST accepted-length on our eval vs the deployed spec head's E[T].
# --------------------------------------------------------------------------- #
def deliverable1_rest_et_on_our_eval(e_t_eagle3: float) -> dict[str, Any]:
    band = {
        "reasoning_floor": REST_ET_OUR_EVAL_FLOOR,
        "generous_upper_chat_analog": REST_ET_OUR_EVAL_GENEROUS_UPPER,
        "global_best_sanity_code": REST_ET_GLOBAL_BEST_SANITY,
    }
    ratio_floor = REST_ET_OUR_EVAL_FLOOR / e_t_eagle3
    ratio_upper = REST_ET_OUR_EVAL_GENEROUS_UPPER / e_t_eagle3
    ratio_code = REST_ET_GLOBAL_BEST_SANITY / e_t_eagle3
    return {
        "rest_M_literature_ranges": {
            "code_humaneval": list(REST_M_CODE),
            "chat_mtbench": list(REST_M_CHAT),
            "reasoning_no_published_floor": REST_M_REASONING_FLOOR,
        },
        "our_eval_is_100pct_reasoning_stem": True,
        "rest_et_our_eval_band": band,
        "e_t_eagle3": e_t_eagle3,
        "ratio_floor_vs_eagle3": ratio_floor,
        "ratio_generous_upper_vs_eagle3": ratio_upper,            # == headline TEST metric
        "ratio_global_best_code_vs_eagle3": ratio_code,
        "rest_under_accepts_on_every_workload": bool(REST_ET_GLOBAL_BEST_SANITY < e_t_eagle3),
        "rest_lt_eagle3_even_generous_upper": bool(REST_ET_OUR_EVAL_GENEROUS_UPPER < e_t_eagle3),
        "note": ("REST's mean accepted length M is the SAME axis as E[T] (M=1.0 = no-draft bonus floor). "
                 "Code M in [2.53,2.69] (STRONG, NOT our eval); chat M in [1.97,2.01] (closest analog); "
                 "reasoning has NO published REST number -> principled floor ~1.0. OUR eval is 100% "
                 "reasoning/STEM (#330), so the relevant band is [1.0 floor, 2.01 generous chat upper]. "
                 "Even REST's GLOBAL BEST (code 2.69) < deployed E[T]=3.8512: REST under-accepts vs the "
                 "existing already-priced head on EVERY workload, and far worse on novel reasoning."),
    }


# --------------------------------------------------------------------------- #
# (D2) Strict supply tax: does REST inherit denken #332's batched-verify BW floor?
# --------------------------------------------------------------------------- #
def deliverable2_strict_supply_tax(e_t_eagle3: float) -> dict[str, Any]:
    inherits = True   # REST verifies a Trie of c=64 candidates in ONE batched tree-attention forward
    strict_ceiling_roundtrip = LAMBDA_CEIL * (1.0 - SUPPLY_FLOOR_GEO)
    rest_strict_upper = rest_strict_ceiling(REST_ET_OUR_EVAL_GENEROUS_UPPER, e_t_eagle3)
    return {
        "rest_uses_batched_multitoken_verify": True,
        "verify_structure": ("Trie of up to c={} retrieved candidates, single batched tree-attention "
                             "forward, m<={} draft tokens/step (Medusa/SpecInfer-class)".format(
                                 REST_VERIFY_TRIE_CANDIDATES, REST_VERIFY_MAX_DRAFT_TOKENS)),
        "inherits_332_supply_tax": inherits,             # expected True
        "supply_floor_geometric_phi": SUPPLY_FLOOR_GEO,
        "strict_ceiling_method_independent_473p5": strict_ceiling_roundtrip,
        "roundtrips_denken332_473p5": bool(abs(strict_ceiling_roundtrip - STRICT_CEILING_332) <= TOL_332),
        "rest_strict_ceiling_at_generous_upper": rest_strict_upper,
        "supply_tax_is_moot_for_rest": True,             # REST already demand-capped far below 500
        "note": ("REST IS a batched multi-token verify (tree attention over a c=64 Trie), so it inherits "
                 "denken #332's method-INDEPENDENT determinism BW floor -> strict-compliant ceiling "
                 "<= 520.953*(1-0.09103) = 473.5 (same as EAGLE-3). But REST's E[T] is so low it is "
                 "demand-capped far below 500 regardless, so the supply tax is moot here."),
    }


# --------------------------------------------------------------------------- #
# (D3) PPL-only ceiling: 520.953 * (E[T]_REST / E[T]_eagle3).
# --------------------------------------------------------------------------- #
def deliverable3_ppl_only_ceiling(e_t_eagle3: float) -> dict[str, Any]:
    ceil_floor = rest_ppl_only_ceiling(REST_ET_OUR_EVAL_FLOOR, e_t_eagle3)
    ceil_upper = rest_ppl_only_ceiling(REST_ET_OUR_EVAL_GENEROUS_UPPER, e_t_eagle3)
    ceil_code = rest_ppl_only_ceiling(REST_ET_GLOBAL_BEST_SANITY, e_t_eagle3)
    eagle3_ceiling = rest_ppl_only_ceiling(e_t_eagle3, e_t_eagle3)   # == LAMBDA_CEIL by construction
    return {
        "formula": "520.953 * (E[T]_REST / E[T]_eagle3)",
        "rest_ppl_only_ceiling_reasoning_floor": ceil_floor,
        "rest_ppl_only_ceiling_generous_upper": ceil_upper,
        "rest_ppl_only_ceiling_global_best_code": ceil_code,
        "eagle3_ppl_only_ceiling": eagle3_ceiling,
        "lambda_ceil": LAMBDA_CEIL,
        "deployed_tps": BASELINE_TPS,
        "rest_ceiling_beats_eagle3": bool(ceil_upper > eagle3_ceiling),         # False
        "rest_ceiling_reaches_500_even_code": bool(ceil_code >= TARGET),         # False
        "rest_ceiling_beats_deployed_even_code": bool(ceil_code > BASELINE_TPS), # False
        "note": ("PPL-only ceiling scales linearly with E[T]. The deployed head's own PPL-only ceiling is "
                 "520.953 (ratio 1.0). REST beats it iff E[T]_REST > E[T]_eagle3, which never holds. Even "
                 "REST's GLOBAL BEST (code 2.69) gives only {:.1f} TPS -- below BOTH 500 and the deployed "
                 "481.53. On our reasoning eval the band is {:.1f}-{:.1f} TPS.".format(
                     ceil_code, ceil_floor, ceil_upper)),
    }


# --------------------------------------------------------------------------- #
# (D4) The operational verdict: gate-free deployability vs reasoning under-acceptance.
# --------------------------------------------------------------------------- #
def deliverable4_operational_verdict(d1: dict, d3: dict) -> dict[str, Any]:
    is_gatefree = True   # offline CPU suffix array; NO gradient train; NO #319 gated cluster retrain
    reaches_500 = bool(d3["rest_ceiling_reaches_500_even_code"])
    is_viable = bool(reaches_500)   # gate-free is necessary-not-sufficient; demand decides viability
    verdict = (
        "REST IS gate-free deployable (offline CPU suffix-array datastore, NO gradient training, NO #319 "
        "gated cluster retrain, exact Leviathan-2023 accept rule => PPL-preserving) -- the deploy-today "
        "advantage is REAL. But on OUR 100%-reasoning eval REST under-accepts so badly (E[T] band "
        "[{floor:.2f}, {upper:.2f}] vs the deployed head's 3.8512) that its PPL-only ceiling "
        "({cfloor:.0f}-{cupper:.0f} TPS) sits far below the existing EAGLE-3 head's already-priced lane "
        "(520.953 ceiling / 481.53 deployed). Even REST's global best (code 2.69) reaches only "
        "{ccode:.0f} TPS. VERDICT: NOT a viable no-retrain >500 path on our eval -- the gate-free "
        "advantage is irrelevant when the method is demand-capped below the existing lane.".format(
            floor=REST_ET_OUR_EVAL_FLOOR, upper=REST_ET_OUR_EVAL_GENEROUS_UPPER,
            cfloor=d3["rest_ppl_only_ceiling_reasoning_floor"],
            cupper=d3["rest_ppl_only_ceiling_generous_upper"],
            ccode=d3["rest_ppl_only_ceiling_global_best_code"]))
    return {
        "rest_is_gatefree_deployable": is_gatefree,        # TEST bool -- True
        "rest_reaches_500_on_our_eval": reaches_500,        # False
        "rest_is_viable_500_path": is_viable,               # False
        "gatefree_advantage_real_but_irrelevant": bool(is_gatefree and not is_viable),
        "exactness_guarantee": "Leviathan et al. 2023 (arXiv:2211.17192) rejection-sampling rule -> PPL-preserving",
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# (D5) Honest caveats + the named (cheap, gated) measurement that would settle E[T]_REST.
# --------------------------------------------------------------------------- #
def deliverable5_caveats() -> dict[str, Any]:
    return {
        "caveats": [
            ("Literature accept rates are workload-dependent POINT estimates -> carried as RANGES: code "
             "[2.53,2.69], chat [1.97,2.01], reasoning has NO published REST number (principled floor "
             "~1.0). The headline ratio uses the GENEROUS chat-analog upper (2.01), which OVER-states "
             "reasoning; the verdict is range-robust (loses across the whole band)."),
            ("Datastore quality/size is a FREE parameter. REST's M scales log-linearly with datastore "
             "size on code (0.9GB->1.96 ... 27GB->2.65) BUT out-of-domain collapses to ~1.0 -- domain "
             "match dominates size. There is no large verbatim-reasoning-CoT corpus, so even a huge "
             "datastore will not lift novel-reasoning M much above the floor. Assumed regime: best-case "
             "in-domain reasoning datastore ~ chat hit-rate (the generous upper); realistic ~ floor."),
            ("A MEASURED REST E[T] on our eval needs a (cheap, GATED) run: build a reasoning datastore "
             "OFFLINE on CPU (suffix array over a math/STEM corpus), serve REST verify, measure mean "
             "accepted length on the 128-prompt eval. This is a gated local-GPU smoke or HF Job -- NOT "
             "drawn here. This card prices the SCREEN, not a build; no build is claimed beyond it."),
            ("'EAGLE-3 head' here = the DEPLOYED/existing already-priced spec head that denken #289 "
             "(fi34s269) decomposed (E[T]=3.8512). REST is compared to that existing lane, not to the "
             "unbuilt EAGLE-3 coverage-retrain candidate (which itself needs the #319 gated spend)."),
            ("CPU-analytic SCREEN. NOT a launch / build / datastore build / served-file change / HF Job "
             "/ submission. BASELINE 481.53 unchanged; adds 0 TPS."),
        ],
        "named_measurement_to_settle_et": (
            "offline CPU suffix-array datastore over a STEM/math corpus -> REST tree-verify on serve -> "
            "measure mean accepted length on the official 128-prompt reasoning eval (gated, cheap)."),
    }


# --------------------------------------------------------------------------- #
# (E) eval composition context (lawine #330) -- accept-rate context for REST's weak case.
# --------------------------------------------------------------------------- #
def eval_composition_context() -> dict[str, Any]:
    n = sum(EVAL_SRC_COUNTS.values())
    frac = sum(EVAL_SRC_COUNTS.get(k, 0) for k in ("mmlu_pro", "gpqa", "aime")) / n
    return {
        "src_counts": dict(EVAL_SRC_COUNTS),
        "n_prompts": n,
        "reasoning_stem_frac": frac,
        "cov_prior_top4": COV_PRIOR,
        "is_100pct_reasoning_stem": bool(abs(frac - 1.0) <= TOL_EXACT and n == EVAL_N),
        "note": ("official 128 eval is 100% reasoning/STEM (mmlu_pro 57 / gpqa 57 / aime 14, lawine "
                 "#330 hfrscdai): novel CoT (latex/math/science), the WORST case for verbatim retrieval. "
                 "This is the accept-rate context that puts REST's M near its no-draft floor."),
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY).
# --------------------------------------------------------------------------- #
def _selftests(e3: dict, d1: dict, d2: dict, d3: dict, d4: dict, comp: dict) -> dict[str, Any]:
    rest_M_finite = all(_finite(x) for x in (*REST_M_CODE, *REST_M_CHAT, REST_M_REASONING_FLOOR))
    rest_M_ordered = bool(
        REST_M_REASONING_FLOOR <= REST_M_CHAT[0] <= REST_M_CHAT[1] <= REST_M_CODE[0] <= REST_M_CODE[1])
    ratio_upper = d1["ratio_generous_upper_vs_eagle3"]
    conditions = {
        # (a) deployed/existing spec head E[T] reproduced from denken #289 a_k <= 1e-6.
        "a_eagle3_et_reproduces_289": bool(
            e3["reproduces_289_le_1e6"]
            and abs(e3["E_T_eagle3"] - e3["E_T_eagle3_from_counter"]) <= TOL_289
            and abs(e3["E_T_eagle3"] - E_T_EAGLE3_DISPLAY) <= 5e-5),
        # (b) denken #332 supply floor + 473.5 strict ceiling round-trip <= 1e-6.
        "b_supply_floor_473p5_roundtrips_332": bool(
            d2["roundtrips_denken332_473p5"]
            and abs(d2["strict_ceiling_method_independent_473p5"] - STRICT_CEILING_332) <= TOL_332),
        # (c) REST E[T] ranges cited + finite + ordered (NaN-clean).
        "c_rest_et_range_cited_finite": bool(rest_M_finite and rest_M_ordered),
        # (d) inherits_332_supply_tax computed (True -- batched tree-attention verify).
        "d_inherits_332_supply_tax": bool(d2["inherits_332_supply_tax"]),
        # (e) reasoning/STEM eval composition (#330) cited (100%, 57/57/14, n=128).
        "e_eval_composition_330_cited": bool(comp["is_100pct_reasoning_stem"]),
        # (f) TEST ratio in (0,1), == generous-upper/eagle3, and < 1 (REST under-accepts).
        "f_test_ratio_in_unit_and_lt1": bool(
            0.0 < ratio_upper < 1.0
            and abs(ratio_upper - REST_ET_OUR_EVAL_GENEROUS_UPPER / e3["E_T_eagle3"]) <= TOL_EXACT),
        # (g) PPL-only ceiling does NOT beat eagle3 and does NOT reach 500 (even code-sanity).
        "g_rest_ceiling_loses": bool(
            (not d3["rest_ceiling_beats_eagle3"])
            and (not d3["rest_ceiling_reaches_500_even_code"])
            and (not d3["rest_ceiling_beats_deployed_even_code"])),
        # (h) TEST bool: gate-free deployable True, but viable-500 False (gate-free yet not viable).
        "h_gatefree_true_viable_false": bool(
            d4["rest_is_gatefree_deployable"] and (not d4["rest_is_viable_500_path"])
            and d4["gatefree_advantage_real_but_irrelevant"]),
        # (i) structural: even REST's GLOBAL BEST published M (code 2.69) < E[T]_eagle3 (3.8512).
        "i_rest_under_accepts_globally": bool(d1["rest_under_accepts_on_every_workload"]),
        # (j) imports EXACT: constants round to displayed forms.
        "j_imports_exact": bool(
            abs(E_T_EAGLE3_DISPLAY - 3.8512) <= TOL_DISPLAY_C
            and abs(STRICT_CEILING_332 - 473.53) <= TOL_DISPLAY_TPS
            and abs(LAMBDA_CEIL - 520.95) <= TOL_DISPLAY_TPS
            and abs(SUPPLY_FLOOR_GEO - 0.09103) <= 1e-5
            and abs(COV_PRIOR - 0.8903) <= TOL_DISPLAY_C),
        # (k) NaN-clean (set by caller).
        "k_nan_clean": True,
    }
    return {
        "conditions": conditions,
        "rest_feasibility_self_test_passes": bool(all(conditions.values())),
        "n_checks": len(conditions),
    }


# --------------------------------------------------------------------------- #
# Synthesis.
# --------------------------------------------------------------------------- #
def synthesize() -> dict[str, Any]:
    e3 = eagle3_et_from_ak()
    e_t_eagle3 = e3["E_T_eagle3"]
    d1 = deliverable1_rest_et_on_our_eval(e_t_eagle3)
    d2 = deliverable2_strict_supply_tax(e_t_eagle3)
    d3 = deliverable3_ppl_only_ceiling(e_t_eagle3)
    d4 = deliverable4_operational_verdict(d1, d3)
    d5 = deliverable5_caveats()
    comp = eval_composition_context()
    st = _selftests(e3, d1, d2, d3, d4, comp)

    rest_etratio = d1["ratio_generous_upper_vs_eagle3"]
    handoff = (
        "REST (retrieval spec-decode) IS the one gate-free no-retrain >500 candidate (offline CPU "
        "suffix-array, no #319 gated spend, exact Leviathan-2023 accept rule => PPL-safe). But on OUR "
        "100%-reasoning eval (#330) its accepted length does NOT hold: E[T] band [{:.2f}, {:.2f}] (floor "
        "to generous chat-analog upper) vs the deployed head's 3.8512, so the PPL-only ceiling is "
        "{:.0f}-{:.0f} TPS -- and even REST's GLOBAL BEST (code 2.69) reaches only {:.0f} TPS, below "
        "BOTH 500 and the deployed 481.53. rest_etratio_vs_eagle3={:.4f} (<1 across the whole band). "
        "REST also inherits denken #332's batched-verify supply tax (Trie tree-attention verify). "
        "VERDICT: gate-free but NOT a viable >500 path on our eval -- the no-retrain advantage is "
        "irrelevant when REST is demand-capped below the existing already-priced lane. A measured "
        "E[T]_REST would need a cheap gated reasoning-datastore run (named, not drawn).".format(
            REST_ET_OUR_EVAL_FLOOR, REST_ET_OUR_EVAL_GENEROUS_UPPER,
            d3["rest_ppl_only_ceiling_reasoning_floor"], d3["rest_ppl_only_ceiling_generous_upper"],
            d3["rest_ppl_only_ceiling_global_best_code"], rest_etratio))

    headline = {
        "rest_feasibility_self_test_passes": bool(st["rest_feasibility_self_test_passes"]),  # PRIMARY
        "rest_etratio_vs_eagle3": rest_etratio,                                              # TEST float
        "rest_is_gatefree_deployable": bool(d4["rest_is_gatefree_deployable"]),              # TEST bool
        "e_t_eagle3": e_t_eagle3,
        "rest_et_our_eval_floor": REST_ET_OUR_EVAL_FLOOR,
        "rest_et_our_eval_generous_upper": REST_ET_OUR_EVAL_GENEROUS_UPPER,
        "rest_ppl_only_ceiling_generous_upper": d3["rest_ppl_only_ceiling_generous_upper"],
        "rest_ppl_only_ceiling_global_best_code": d3["rest_ppl_only_ceiling_global_best_code"],
        "eagle3_ppl_only_ceiling": d3["eagle3_ppl_only_ceiling"],
        "rest_ceiling_reaches_500_even_code": bool(d3["rest_ceiling_reaches_500_even_code"]),
        "inherits_332_supply_tax": bool(d2["inherits_332_supply_tax"]),
        "rest_is_viable_500_path": bool(d4["rest_is_viable_500_path"]),
        "rest_under_accepts_on_every_workload": bool(d1["rest_under_accepts_on_every_workload"]),
    }
    return {
        "headline": headline,
        "eagle3_et_from_ak": e3,
        "deliverable1_rest_et_on_our_eval": d1,
        "deliverable2_strict_supply_tax": d2,
        "deliverable3_ppl_only_ceiling": d3,
        "deliverable4_operational_verdict": d4,
        "deliverable5_caveats": d5,
        "eval_composition_context": comp,
        "self_test": st,
        "handoff": handoff,
        "imports": {
            "provenance": (
                "denken #289 fi34s269 (per-position a_k -> E[T]=3.8512 via 1+sum cumprod; whole-run "
                "counters W&B 2j0e8xgg / lawine #282) x denken #332 y5cl0ena (method-independent supply "
                "floor 0.09103 @ geometric phi, strict ceiling 473.5, lambda ceiling 520.953) x lawine "
                "#330 hfrscdai (official 128 eval = 100% reasoning/STEM, mmlu_pro 57 / gpqa 57 / aime 14, "
                "cov prior 0.8903). Literature: He et al. 2024 REST (arXiv:2311.08252, mean accepted "
                "length M: code [2.53,2.69], chat [1.97,2.01], datastore scaling Table 2, Trie c=64 "
                "tree-attention verify, offline CPU suffix array); Leviathan et al. ICML 2023 "
                "(arXiv:2211.17192, distributional-equivalence guarantee). All run-ids in "
                "wandb-applied-ai-team/gemma-challenge-senpai."),
            "caveats": d5["caveats"],
        },
    }


# --------------------------------------------------------------------------- #
# NaN guard + report + W&B.
# --------------------------------------------------------------------------- #
def _nan_paths(node: Any, p: str = "result") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            bad += _nan_paths(v, f"{p}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            bad += _nan_paths(v, f"{p}[{i}]")
    elif isinstance(node, float) and not math.isfinite(node):
        bad.append(p)
    return bad


def _print_report(syn: dict) -> None:
    h = syn["headline"]
    d1 = syn["deliverable1_rest_et_on_our_eval"]
    d2 = syn["deliverable2_strict_supply_tax"]
    d3 = syn["deliverable3_ppl_only_ceiling"]
    d4 = syn["deliverable4_operational_verdict"]
    st = syn["self_test"]
    print("\n" + "=" * 98, flush=True)
    print("REST GATE-FREE >500 FEASIBILITY (#348, kanna) — a no-retrain method on OUR reasoning eval?", flush=True)
    print("=" * 98, flush=True)
    print("  (D1) REST E[T] ON OUR EVAL vs DEPLOYED SPEC HEAD", flush=True)
    print(f"      E[T]_eagle3 (denken #289 a_k) = {h['e_t_eagle3']:.6f}  (== 1 + 48684/17075)", flush=True)
    print(f"      REST M lit: code [{REST_M_CODE[0]}, {REST_M_CODE[1]}]  chat [{REST_M_CHAT[0]}, "
          f"{REST_M_CHAT[1]}]  reasoning floor {REST_M_REASONING_FLOOR} (no published)", flush=True)
    print(f"      our-eval band: floor {REST_ET_OUR_EVAL_FLOOR:.2f} .. generous upper "
          f"{REST_ET_OUR_EVAL_GENEROUS_UPPER:.2f}  (code sanity {REST_ET_GLOBAL_BEST_SANITY:.2f})", flush=True)
    print(f"      rest_etratio_vs_eagle3 (TEST) = {h['rest_etratio_vs_eagle3']:.4f}  "
          f"(under-accepts globally: {d1['rest_under_accepts_on_every_workload']})", flush=True)
    print("-" * 98, flush=True)
    print("  (D2) STRICT SUPPLY TAX", flush=True)
    print(f"      inherits_332_supply_tax = {d2['inherits_332_supply_tax']}  (Trie c=64 tree-attention "
          f"verify)  strict 473.5 round-trips #332: {d2['roundtrips_denken332_473p5']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D3) PPL-ONLY CEILING = 520.953 * (E[T]_REST / E[T]_eagle3)", flush=True)
    print(f"      REST floor={d3['rest_ppl_only_ceiling_reasoning_floor']:.1f}  upper="
          f"{d3['rest_ppl_only_ceiling_generous_upper']:.1f}  code-best="
          f"{d3['rest_ppl_only_ceiling_global_best_code']:.1f}  |  eagle3="
          f"{d3['eagle3_ppl_only_ceiling']:.1f}  deployed={BASELINE_TPS:.1f}", flush=True)
    print(f"      beats eagle3: {d3['rest_ceiling_beats_eagle3']}   reaches 500 (even code): "
          f"{d3['rest_ceiling_reaches_500_even_code']}", flush=True)
    print("-" * 98, flush=True)
    print("  (D4) OPERATIONAL VERDICT", flush=True)
    print(f"      gate-free deployable (TEST) = {d4['rest_is_gatefree_deployable']}   "
          f"viable >500 path = {d4['rest_is_viable_500_path']}", flush=True)
    print(f"      >> {d4['verdict']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  HAND-OFF: {syn['handoff']}", flush=True)
    print("-" * 98, flush=True)
    print(f"  PRIMARY rest_feasibility_self_test_passes = "
          f"{st['rest_feasibility_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    for k, v in st["conditions"].items():
        print(f"        - {k}: {v}", flush=True)
    print("=" * 98 + "\n", flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:
        print(f"[rest-gatefree-500-feasibility] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    d1 = syn["deliverable1_rest_et_on_our_eval"]
    d2 = syn["deliverable2_strict_supply_tax"]
    d3 = syn["deliverable3_ppl_only_ceiling"]
    d4 = syn["deliverable4_operational_verdict"]
    comp = syn["eval_composition_context"]
    st = syn["self_test"]
    h = syn["headline"]
    run = init_wandb_run(
        job_type="rest-gatefree-500-feasibility",
        agent="kanna",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["rest-gatefree-500-feasibility", "rest", "retrieval-spec-decode", "issue-319",
              "no-retrain", "gate-free", "demand-envelope", "supply-tax", "compliant-500",
              "reasoning-eval", "validity-gate", "bank-the-analysis"],
        config={
            "K_spec": K_SPEC, "e_t_eagle3_display": E_T_EAGLE3_DISPLAY,
            "lambda_ceil": LAMBDA_CEIL, "supply_floor_geometric_phi": SUPPLY_FLOOR_GEO,
            "strict_ceiling_332": STRICT_CEILING_332, "cov_prior": COV_PRIOR,
            "rest_M_code": list(REST_M_CODE), "rest_M_chat": list(REST_M_CHAT),
            "rest_M_reasoning_floor": REST_M_REASONING_FLOOR,
            "rest_et_our_eval_floor": REST_ET_OUR_EVAL_FLOOR,
            "rest_et_our_eval_generous_upper": REST_ET_OUR_EVAL_GENEROUS_UPPER,
            "rest_et_global_best_sanity": REST_ET_GLOBAL_BEST_SANITY,
            "eval_src_counts": dict(EVAL_SRC_COUNTS), "eval_n": EVAL_N,
            "target": TARGET, "baseline_tps": BASELINE_TPS, "wandb_group": args.wandb_group,
            "source_runs": "denken#289(fi34s269), lawine#282(2j0e8xgg), denken#332(y5cl0ena), lawine#330(hfrscdai)",
            "literature": "He et al. 2024 REST arXiv:2311.08252; Leviathan et al. 2023 arXiv:2211.17192",
        },
    )
    if run is None:
        print("[rest-gatefree-500-feasibility] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return

    summary: dict[str, Any] = {
        "rest_feasibility_self_test_passes": int(bool(st["rest_feasibility_self_test_passes"])),  # PRIMARY
        "rest_etratio_vs_eagle3": h["rest_etratio_vs_eagle3"],                                    # TEST
        "rest_is_gatefree_deployable": int(bool(d4["rest_is_gatefree_deployable"])),              # TEST
        "e_t_eagle3": h["e_t_eagle3"],
        "rest_et_our_eval_floor": REST_ET_OUR_EVAL_FLOOR,
        "rest_et_our_eval_generous_upper": REST_ET_OUR_EVAL_GENEROUS_UPPER,
        "ratio_floor_vs_eagle3": d1["ratio_floor_vs_eagle3"],
        "ratio_generous_upper_vs_eagle3": d1["ratio_generous_upper_vs_eagle3"],
        "ratio_global_best_code_vs_eagle3": d1["ratio_global_best_code_vs_eagle3"],
        "rest_under_accepts_on_every_workload": int(bool(d1["rest_under_accepts_on_every_workload"])),
        "inherits_332_supply_tax": int(bool(d2["inherits_332_supply_tax"])),
        "strict_ceiling_method_independent_473p5": d2["strict_ceiling_method_independent_473p5"],
        "rest_strict_ceiling_at_generous_upper": d2["rest_strict_ceiling_at_generous_upper"],
        "rest_ppl_only_ceiling_reasoning_floor": d3["rest_ppl_only_ceiling_reasoning_floor"],
        "rest_ppl_only_ceiling_generous_upper": d3["rest_ppl_only_ceiling_generous_upper"],
        "rest_ppl_only_ceiling_global_best_code": d3["rest_ppl_only_ceiling_global_best_code"],
        "eagle3_ppl_only_ceiling": d3["eagle3_ppl_only_ceiling"],
        "rest_ceiling_beats_eagle3": int(bool(d3["rest_ceiling_beats_eagle3"])),
        "rest_ceiling_reaches_500_even_code": int(bool(d3["rest_ceiling_reaches_500_even_code"])),
        "rest_ceiling_beats_deployed_even_code": int(bool(d3["rest_ceiling_beats_deployed_even_code"])),
        "rest_is_viable_500_path": int(bool(d4["rest_is_viable_500_path"])),
        "gatefree_advantage_real_but_irrelevant": int(bool(d4["gatefree_advantage_real_but_irrelevant"])),
        "reasoning_stem_frac": comp["reasoning_stem_frac"],
        "eval_n_prompts": comp["n_prompts"],
        "baseline_tps": BASELINE_TPS,
        "peak_mem_mib": payload["peak_mem_mib"],
        **{f"selftest_{k}": int(bool(v)) for k, v in st["conditions"].items()},
    }
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v)) and v is not None}

    log_summary(run, summary, step=0)
    log_json_artifact(run, name="rest_gatefree_500_feasibility_result", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[rest-gatefree-500-feasibility] wandb logged {len(summary)} keys", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the PRIMARY self-validation")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    ap.add_argument("--wandb-group", "--wandb_group", dest="wandb_group",
                    default="rest-gatefree-500-feasibility")
    args = ap.parse_args(argv)

    syn = synthesize()

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 348, "agent": "kanna",
        "kind": "rest-gatefree-500-feasibility", "analysis_only": True, "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _nan_paths(payload)
    payload["nan_clean"] = not nan_paths
    syn["self_test"]["conditions"]["k_nan_clean"] = not nan_paths
    syn["self_test"]["rest_feasibility_self_test_passes"] = bool(
        all(syn["self_test"]["conditions"].values()))
    syn["headline"]["rest_feasibility_self_test_passes"] = syn["self_test"][
        "rest_feasibility_self_test_passes"]
    if nan_paths:
        print(f"[rest-gatefree-500-feasibility] WARNING non-finite at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rest_gatefree_500_feasibility_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"[rest-gatefree-500-feasibility] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    if args.self_test:
        ok = (syn["self_test"]["rest_feasibility_self_test_passes"] and payload["nan_clean"])
        print(f"[rest-gatefree-500-feasibility] self-test {'PASS' if ok else 'FAIL'}", flush=True)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
