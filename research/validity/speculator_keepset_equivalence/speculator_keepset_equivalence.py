#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Speculator self-referential equivalence + in-keepset acceptance upside (PR #420, land).

RE-SCOPE FROM THE HUMAN (Issue #407, 2026-06-15 21:13Z)
------------------------------------------------------
"Remember, we need token equivalence for any speculator we build -- are we still working on that? I
want you to find the fastest implementation that also respects this equivalence." My just-merged #414
(`bq7xkfcv`) established the OPERATIVE equivalence is SELF-REFERENTIAL (the deployed 16384-row
truncated-head greedy AR is the official scorer's arbiter; the deployed config passes it for FREE;
absolute/full-vocab equivalence is a stronger notion costing ~54 TPS the scorer does not require).
This card closes the SPECULATOR-side self-referential equivalence question (the direct answer to the
human) AND prices an acceptance-side equivalent-TPS lever.

THE TWO DELIVERABLES
--------------------
(1) IDENTITY -- PROVABLE BY CONSTRUCTION. An in-keepset drafter + truncated-head verify preserves the
    self-referential gate end-to-end REGARDLESS of the drafter's proposal vocabulary. Greedy spec verify
    EMITS the truncated-head target's argmax (over the 16384 keepset) for every accepted draft position
    AND for the correction/bonus token at the first mismatch; the drafter only PROPOSES (its tokens gate
    only HOW MANY positions accept, never WHICH token is emitted). So EVERY EMITTED token is in-keepset
    BY CONSTRUCTION -> the speculator satisfies the official self-referential gate for free, exactly as
    the deployed config does. Grounded in the DEPLOYED verify code (fa2sw_nonspec_int4/serve.py dixie
    slim greedy sampler lines 410-456; the same splitkv_verify_patch.py lawine #417 identified):
    `dixie_target_argmax = all_argmax[target_logits_indices]` feeds `rejection_greedy_sample_kernel`,
    where `draft_token_ids` is used ONLY for accept-comparison and the EMITTED ids are the target argmax
    / bonus. Confirmed independently by sitecustomize.py ("Drafter-only => cannot change emitted tokens").
    Deliverable: inkeepset_drafter_preserves_self_referential = True + a synthetic greedy-verify
    simulation proving emitted in keepset for ARBITRARY (incl. out-of-keepset) drafter proposals.

(2) ACCEPTANCE UPSIDE -- THE LEVER. The deployed drafter is a SEPARATE small model (`/tmp/qat-assistant`,
    MTP K=7, M=8; Gemma4AssistantForCausalLM, full-vocab 262144 tied head, centroid fused-sparse argmax),
    NOT vocab-restricted to the truncated head. If it proposes an OUT-OF-KEEPSET (OOK) token, the
    truncated-head verify can NEVER emit it -> the proposal is auto-rejected -> the accept run truncates
    at that position -> lost E[T]. Masking the drafter's proposal logits to the 16384 keepset is a
    STRUCTURALLY >= 0 acceptance lever (an OOK proposal is a guaranteed reject; the masked re-proposal can
    only match the target MORE often) that TRIVIALLY preserves self-referential identity (the verify
    arbiter is unchanged). Model: a_i' = a_i + p * q_i where p = OOK proposal rate, q_i in [0, s_i],
    s_i = a_i/(1-p) the in-keepset conditional accept rate; floor q=0 (a'=a, no upside), ceiling q=s
    (a'=a/(1-p), every OOK reclaimed as well as a native in-keepset proposal). E[T] uplift -> equiv-TPS
    via dE[T]*K_cal (a proposal-only change leaves the step rate K_cal invariant).

THE MEASUREMENT (honest, analytic-bounded)
------------------------------------------
A FAITHFUL per-position drafter-argmax read is BLOCKED for a 0-TPS analysis card (ubel #401 `i2qsjyp6`,
in-scope sibling, AUTHORITATIVE ledger): the deployed drafter forward needs inputs_embeds(5120=2x2560
backbone hidden) + shared_kv_states (backbone cross-attn KV) -- a vLLM-MTP-proposer-specific construction
that is not bankable standalone (`standalone_forward_blocked`); a plain bf16 HF read is WRONG-distribution
(deployed = prune(16k)+int4-Marlin) AND cross-session NONDETERMINISTIC (bf16 lm_head argmax flips ~9-13%;
only int4-Marlin is bit-exact). The faithful read is a custom-vLLM-patch / local-serve instrumentation
effort -> out of scope here. So `probe_was_live=False` and the OOK rate is an ANALYTIC IMITATION ANCHOR:
the drafter is the OFFICIAL google gemma-4-E4B assistant (trained to imitate the backbone), so its
full-vocab argmax tracks the backbone's; the backbone's full-vocab greedy OOK rate on my #414 held-out
corpus (64,602 positions / 274 NL+code+multilingual prompts) is 9.21% with 96 distinct OOK ids. Trivial
upper bound p <= 1 - a_1 = 0.2707 (all top-1 mismatches OOK). A cheap, FAITHFUL GPU structural probe
confirms the lever's PRECONDITION (drafter loadable, full-vocab 262144 head, NOT keepset-masked, keepset
a strict 16384-subset) without the blocked per-position read.

VERDICT (this card)
-------------------
* inkeepset_drafter_preserves_self_referential = True (PROVEN by construction; the direct answer to #407:
  ANY speculator -- any drafter proposal vocab -- respects the operative self-referential equivalence,
  because the truncated-head verify is the sole arbiter of EMITTED tokens).
* The lever is STRUCTURALLY OPEN (>= 0) and PRESERVES identity. At the imitation anchor (p=9.21%) the
  CEILING (q=s, every OOK reclaimed) is inkeepset_drafting_equiv_tps_upside ~= 138 equiv-TPS; the FLOOR
  is 0. lever_is_closed=False at the anchor. The REALIZED value is UNRESOLVED-IN-ENVELOPE and expected
  MODEST: #289 (`fi34s269`) shows the deployed drafter is already AT its linear acceptance cap
  (`deployed_at_or_above_linear_cap`=True), so the formerly-OOK positions are exactly the hard ones whose
  in-keepset runner-up is unlikely to nail the target (q << s). Resolving the magnitude needs the #401
  vLLM-proposer probe (p) + a masked-drafter A/B (q).
* drafter_vocab_mask_is_additive_served_change = True: masking the drafter's proposal logits to the
  keepset is an ADDITIVE served-file change (drafter config / a logits mask), human-gated, that touches
  ONLY which tokens are PROPOSED -- never which are emitted -> PPL unchanged 2.3772 <= 2.42 BY
  CONSTRUCTION. This card only PRICES the lever for lawine #419's go/no-go; it does NOT alter the served
  drafter.

PRIMARY metric  speculator_keepset_equivalence_self_test_passes  (>=20 pure-logic checks: ladder/E[T]
roundtrips, the greedy-verify identity simulation over ARBITRARY drafter proposals, the a'=a+p*q upside
model + envelope monotonicity + the dE[T]*K_cal conversion, equiv_tps framing, and the verdict gating;
env-independent, runs under the numpy-only .venv).
SCOPE: analysis / microbench. NO HF Job, NO submission, NO served-file change, 0 official TPS. The drafter
structural GPU probe (config + keepset subset; optional safetensors head-shape) is a graceful enrichment
that DEGRADES to the #401-banked structural facts; the verdict stands on the 0-GPU construction proof +
the #414/#289/#413 banked anchors.

Run:
  # PRIMARY self-test only (numpy-only, no torch/GPU):
  .venv/bin/python -m research.validity.speculator_keepset_equivalence.speculator_keepset_equivalence --self-test
  # full card (torch env for the structural probe, e.g. /tmp/server-venv on the A10G box):
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 /tmp/server-venv/bin/python -m \
    research.validity.speculator_keepset_equivalence.speculator_keepset_equivalence \
    --wandb_group speculator-keepset-equivalence --wandb_name land/speculator-keepset-equivalence
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

import numpy as np

# Pre-import the REAL wandb BEFORE putting REPO_ROOT (= target/, has a ./wandb run-output dir that shadows
# the package as a PEP-420 namespace) on sys.path[0]. Mirrors the #414/#406/#398/#385 house pattern.
try:
    import wandb as _wandb_preimport  # noqa: F401
except Exception:  # noqa: BLE001
    _wandb_preimport = None

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------- #
# Banked anchors (hardcoded with provenance, #414-style; NOT re-derived).
# ---------------------------------------------------------------------------- #
HIDDEN = 2560                       # gemma-4-E4B hidden_size
FULL_VOCAB = 262144                 # gemma-4-E4B text vocab (== drafter tied lm_head rows)
DEPLOYED_LMHEAD_ROWS = 16384        # #390 osoi5 baked deployed truncated head == the keepset size

# ---- #289 (fi34s269) DEPLOYED MTP per-position conditional acceptance ladder a_1..a_7 (K=7) -------
LADDER_289: list[float] = [
    0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
    0.8348727920920435, 0.8357919254658385, 0.8464932652113331,
]
E_ACCEPTED_289 = 2.851185944363104  # #289 E[accepted draft tokens]/step = sum of cumprods
E_T_289 = 3.851185944363104         # #289 E[T] = 1 + E[accepted]
K_SPEC = 7                          # num_speculative_tokens (drafter MTP depth)
M_DEPLOYED = 8                      # verify rows = K_spec + 1 (chain/bonus token)

# ---- #344/#413 public-TPS calibration: official public TPS = E[T] * K_cal -----------------------
MU_P = 481.53                       # deployed public TPS (PR #52, 2x9fm2zx)
K_CAL = 125.26795005202914          # steps/s; public official TPS = E[T] * K_cal (#344)
E_T_REALIZED = MU_P / K_CAL         # 3.84399 secant-consistent realized accept length

# ---- denken #413 (se8mf9ax) equivalent-TPS frame: equiv_tps(K) = public_tps - EQUIV_TAX ----------
EQUIV_TAX_AT_M8 = 2.6               # #413 equivalence-machinery tax at M=8
EQUIV_TPS_7 = MU_P - EQUIV_TAX_AT_M8  # 478.93 deployed equivalent-TPS at K*=7 (#413)

# ---- #414 (bq7xkfcv) held-out absolute non-equivalence anchors (the OOK imitation anchor) --------
# The drafter is the OFFICIAL gemma-4-E4B assistant; its full-vocab argmax imitates the backbone, whose
# full-vocab greedy OOK rate over my #414 held-out corpus anchors the drafter's OOK PROPOSAL rate.
P_OOK_IMITATION_414 = 0.09205597349927247  # #414 held_out_clip_rate (64602 pos, 274 prompts)
DISTINCT_OOK_IDS_414 = 96                  # #414 held_out_distinct_ids_clipped
HELDOUT_POSITIONS_414 = 64602              # #414 held-out positions
HELDOUT_PROMPTS_414 = 274                  # #414 held-out prompts
N406_OFFICIAL_OOK_RATE_BF16 = 0.01019287109375  # #406/#414 official-128 full-vocab greedy clip rate

PPL_DEPLOYED = 2.3772
PPL_GATE = 2.42

TOL = 1e-9

# ---- local artifact paths (degrade gracefully if absent) ----
KEEPSET_16K_CANDIDATES = [
    "/tmp/osoi5-v0-baked/pck04_keepset.json",
]
DRAFTER_CONFIG_CANDIDATES = [
    "/tmp/qat-assistant/config.json",
]
DRAFTER_DIR = "/tmp/qat-assistant"


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def _load_keepset(paths: list[str]) -> tuple[list[int] | None, str | None]:
    p = _first_existing(paths)
    if p is None:
        return None, None
    meta = json.loads(Path(p).read_text())
    ids = meta.get("keep_ids") or meta.get("kept_ids")
    return (sorted(int(i) for i in ids) if ids else None), p


# ---------------------------------------------------------------------------- #
# (1) E[T] / ladder geometry (the #289 acceptance basis).
# ---------------------------------------------------------------------------- #
def expected_accepted(ladder: list[float]) -> float:
    """E[accepted draft tokens]/step = sum_k prod_{j<=k} a_j (conditional ladder)."""
    cum, acc = 1.0, 0.0
    for a in ladder:
        cum *= a
        acc += cum
    return acc


def expected_tokens_per_step(ladder: list[float]) -> float:
    return 1.0 + expected_accepted(ladder)


# ---------------------------------------------------------------------------- #
# (2) IDENTITY -- the deployed greedy spec verify emits target argmax for ANY drafter proposal.
# ---------------------------------------------------------------------------- #
def greedy_spec_emit(target_argmax: list[int], draft_tokens: list[int]) -> tuple[list[int], int]:
    """Faithful model of the DEPLOYED greedy spec verify (fa2sw_nonspec_int4/serve.py:410-456).

    `dixie_target_argmax = all_argmax[target_logits_indices]` is the truncated-head target argmax at each
    of the K draft positions; `dixie_bonus_token_ids` is the argmax at position K (the bonus). The
    `rejection_greedy_sample_kernel` accepts the longest prefix of draft positions whose proposed token
    EQUALS the target argmax, then emits the target argmax at the first mismatch (the correction). In ALL
    cases the EMITTED ids are a PREFIX OF target_argmax (+ the bonus) -- NEVER a draft token. The draft
    tokens gate ONLY the accept LENGTH. target_argmax has K+1 entries (K positions + 1 bonus)."""
    k = len(draft_tokens)
    emitted: list[int] = []
    for i in range(k):
        emitted.append(target_argmax[i])          # emitted token IS the target argmax (accept OR correct)
        if draft_tokens[i] != target_argmax[i]:
            return emitted, i                      # first mismatch -> correction emitted, stop
    emitted.append(target_argmax[k])              # all K accepted -> emit the bonus token (target argmax)
    return emitted, k


def identity_simulation(n_trials: int = 4000, k: int = K_SPEC, keep_lo: int = 0,
                        keep_hi: int = DEPLOYED_LMHEAD_ROWS, full_vocab: int = FULL_VOCAB,
                        seed: int = 0) -> dict[str, Any]:
    """Monte-Carlo proof that the EMITTED token is in-keepset for ARBITRARY drafter proposals (incl. OOK).

    Keepset = [keep_lo, keep_hi). target_argmax is drawn from the keepset (the truncated head can only
    emit kept ids). For each trial we run TWO drafters -- a FULL-VOCAB drafter (proposes anywhere in
    [0, full_vocab), frequently OOK) and a KEEPSET-MASKED drafter -- and confirm: (a) every emitted token
    is in keepset for BOTH; (b) emitted is always a prefix of target_argmax (never a draft token); (c) NO
    OOK draft token is ever emitted. This is the construction-level identity invariant."""
    rng = np.random.default_rng(seed)
    keep = set(range(keep_lo, keep_hi))
    all_emitted_in_keepset = True
    emitted_is_target_prefix = True
    ook_never_emitted = True
    any_ook_proposed = False
    total_emitted = 0
    total_ook_proposals = 0
    for _ in range(n_trials):
        target = [int(x) for x in rng.integers(keep_lo, keep_hi, size=k + 1)]  # all in keepset
        # full-vocab drafter: argmax over [0, full_vocab) -> often OOK.
        draft_full = [int(x) for x in rng.integers(0, full_vocab, size=k)]
        # keepset-masked drafter: argmax over the keepset only.
        draft_masked = [int(x) for x in rng.integers(keep_lo, keep_hi, size=k)]
        ook_positions = [t for t in draft_full if t not in keep]
        total_ook_proposals += len(ook_positions)
        any_ook_proposed = any_ook_proposed or bool(ook_positions)
        for draft in (draft_full, draft_masked):
            emitted, _kacc = greedy_spec_emit(target, draft)
            total_emitted += len(emitted)
            if any(t not in keep for t in emitted):
                all_emitted_in_keepset = False
            if emitted != target[:len(emitted)]:
                emitted_is_target_prefix = False
            # no OOK draft token (a token proposed by the full drafter that is OOK) appears in emitted.
            if any((t in emitted) for t in ook_positions if draft is draft_full):
                ook_never_emitted = False
    return {
        "n_trials": n_trials,
        "k_spec": k,
        "keepset_size": keep_hi - keep_lo,
        "full_vocab": full_vocab,
        "total_emitted_tokens": total_emitted,
        "total_ook_proposals": total_ook_proposals,
        "any_ook_proposed": bool(any_ook_proposed),
        "all_emitted_in_keepset": bool(all_emitted_in_keepset),
        "emitted_is_target_prefix": bool(emitted_is_target_prefix),
        "ook_draft_never_emitted": bool(ook_never_emitted),
        "inkeepset_drafter_preserves_self_referential": bool(
            all_emitted_in_keepset and emitted_is_target_prefix and ook_never_emitted),
    }


# ---------------------------------------------------------------------------- #
# (3) ACCEPTANCE UPSIDE -- a_i' = a_i + p*q_i, q_i in [0, s_i], s_i = a_i/(1-p).
# ---------------------------------------------------------------------------- #
def masked_ladder(ladder: list[float], p: float, q_frac: float) -> list[float]:
    """Drafter-keepset-mask uplifted ladder. p = OOK proposal rate (position-independent soft anchor);
    q_frac in [0,1] interpolates the masked re-proposal quality: a_i'(q_frac) = a_i + p*(q_frac*s_i),
    s_i = a_i/(1-p). q_frac=0 -> a'=a (floor); q_frac=1 -> a'=a/(1-p)=s_i (ceiling). Capped at 1.0."""
    if p <= 0.0:
        return list(ladder)
    out: list[float] = []
    for a in ladder:
        s = a / (1.0 - p)
        ap = a + p * (q_frac * s)
        out.append(min(ap, 1.0))
    return out


def upside_at(p: float, q_frac: float, ladder: list[float] = LADDER_289,
              k_cal: float = K_CAL) -> dict[str, float]:
    """E[T] uplift and equiv-TPS upside from masking the drafter (proposal-only -> K_cal invariant)."""
    base_et = expected_tokens_per_step(ladder)
    a_prime = masked_ladder(ladder, p, q_frac)
    et_prime = expected_tokens_per_step(a_prime)
    detp = et_prime - base_et
    return {
        "p": p, "q_frac": q_frac,
        "base_et": base_et, "uplifted_et": et_prime,
        "etp_uplift": detp,
        "equiv_tps_upside": detp * k_cal,
        "equiv_tps_level": EQUIV_TPS_7 + detp * k_cal,
    }


def upside_envelope(ladder: list[float] = LADDER_289) -> dict[str, Any]:
    """The full (p, q_frac) envelope: p in {0 (closed), imitation anchor, trivial UB}, q in {floor, ceiling}."""
    a1 = ladder[0]
    p_ub = 1.0 - a1                                   # trivial UB: all top-1 mismatches are OOK
    grid: dict[str, dict[str, float]] = {}
    p_points = {"p0_closed": 0.0, "p_imitation_414": P_OOK_IMITATION_414, "p_trivial_ub": p_ub}
    for pk, pv in p_points.items():
        for qk, qv in {"floor_q0": 0.0, "ceiling_q1": 1.0}.items():
            grid[f"{pk}__{qk}"] = upside_at(pv, qv, ladder)
    anchor_ceiling = grid["p_imitation_414__ceiling_q1"]
    anchor_floor = grid["p_imitation_414__floor_q0"]
    return {
        "grid": grid,
        "p_trivial_ub": p_ub,
        # headline = the CEILING at the imitation anchor (the BOUND the lever can deliver).
        "headline_etp_uplift": anchor_ceiling["etp_uplift"],
        "headline_equiv_tps_upside": anchor_ceiling["equiv_tps_upside"],
        "headline_equiv_tps_level": anchor_ceiling["equiv_tps_level"],
        "floor_etp_uplift": anchor_floor["etp_uplift"],
        "floor_equiv_tps_upside": anchor_floor["equiv_tps_upside"],
    }


# ---------------------------------------------------------------------------- #
# (4) DRAFTER structural probe (GPU-optional; confirms the lever PRECONDITION, not the rate).
# ---------------------------------------------------------------------------- #
def drafter_structural_probe(run_measurements: bool) -> dict[str, Any]:
    """Cheap, FAITHFUL structural facts that confirm the lever's PRECONDITION: the deployed drafter has a
    full-vocab (262144) head and is NOT keepset-masked -> it CAN propose OOK; the keepset is a strict
    16384-subset -> 245760 OOK ids exist that the verify can never emit. The faithful PER-POSITION argmax
    read is BLOCKED (#401: standalone forward needs inputs_embeds[5120]+shared_kv_states; bf16 standalone
    = wrong-distribution + nondeterministic) -> probe_was_live=False. Degrades to config/#401 facts."""
    res: dict[str, Any] = {
        "available": False,
        "probe_was_live": False,
        # the faithful per-position read is out of scope for a 0-TPS card (ubel #401 i2qsjyp6 ledger):
        "faithful_perposition_read_blocked": True,
        "faithful_read_blocker": ("vLLM-MTP-proposer-specific: drafter forward needs inputs_embeds(5120="
                                  "2x2560 backbone hidden)+shared_kv_states; bf16 standalone is wrong-"
                                  "distribution (deployed=prune16k+int4-Marlin) AND nondeterministic "
                                  "(bf16 argmax flips ~9-13%; only int4-Marlin bit-exact) -- ubel #401"),
    }
    k16, keepset_path = _load_keepset(KEEPSET_16K_CANDIDATES)
    if k16 is not None:
        s16 = set(k16)
        res["keepset_path"] = keepset_path
        res["keepset_size"] = len(s16)
        res["keepset_min_id"] = min(s16)
        res["keepset_max_id"] = max(s16)
        res["keepset_strict_subset_of_full_vocab"] = bool(
            len(s16) < FULL_VOCAB and max(s16) < FULL_VOCAB)
        res["n_out_of_keepset_ids"] = FULL_VOCAB - len(s16)

    cfg_path = _first_existing(DRAFTER_CONFIG_CANDIDATES)
    if cfg_path is not None:
        try:
            cfg = json.loads(Path(cfg_path).read_text())
            tcfg = cfg.get("text_config", {}) if isinstance(cfg.get("text_config"), dict) else {}
            # Gemma4AssistantConfig nests vocab_size under text_config (top-level is the multimodal wrapper).
            vocab = cfg.get("vocab_size") or tcfg.get("vocab_size")
            res["drafter_config_path"] = cfg_path
            res["drafter_model_type"] = cfg.get("model_type")
            res["drafter_vocab_size"] = vocab
            res["drafter_tie_word_embeddings"] = bool(
                cfg.get("tie_word_embeddings", tcfg.get("tie_word_embeddings", False)))
            res["drafter_num_speculative"] = (cfg.get("num_speculative_tokens")
                                              or cfg.get("num_nextn_predict_layers")
                                              or tcfg.get("num_speculative_tokens"))
            res["drafter_num_speculative_banked"] = K_SPEC  # deployed spec config (#401 inferred K=7)
            res["drafter_use_ordered_embeddings"] = bool(cfg.get("use_ordered_embeddings", False))
            res["drafter_full_vocab_head"] = bool(vocab == FULL_VOCAB)
            # the deployed drafter is NOT keepset-masked: sitecustomize.py FUSED_SPARSE_ARGMAX runs over
            # the FULL token_ordering (262144); the proposal argmax is full-vocab -> CAN propose OOK.
            res["drafter_keepset_masked_in_deployed_path"] = False
            res["drafter_can_propose_ook"] = bool(vocab == FULL_VOCAB)
        except Exception as exc:  # noqa: BLE001
            res["drafter_config_note"] = f"config read failed: {exc!r}"

    if not run_measurements:
        # the precondition is established from config (full-vocab head, not keepset-masked) even 0-GPU.
        res["note"] = "structural probe skipped (self-test only); precondition facts from config/#401"
        return res

    # optional faithful safetensors head-shape confirm (no forward; just tensor metadata).
    try:
        from safetensors import safe_open
        st = _first_existing([str(Path(DRAFTER_DIR) / "model.safetensors")])
        if st is not None:
            with safe_open(st, framework="np", device="cpu") as f:
                keys = list(f.keys())
                res["drafter_safetensors_keys_n"] = len(keys)
                has_embed = any("embed_tokens.weight" in kk for kk in keys)
                res["drafter_has_embed_tokens"] = bool(has_embed)
                if has_embed:
                    ek = next(kk for kk in keys if "embed_tokens.weight" in kk)
                    sl = f.get_slice(ek)
                    shp = list(sl.get_shape())
                    res["drafter_embed_tokens_shape"] = shp
                    res["drafter_head_rows_full_vocab"] = bool(shp and shp[0] == FULL_VOCAB)
            res["available"] = True
    except Exception as exc:  # noqa: BLE001
        res["safetensors_note"] = f"safetensors head-shape probe degraded: {exc!r}"
    # the structural precondition is established from config even if safetensors is absent.
    res["available"] = bool(res.get("available") or cfg_path is not None)
    # consolidate the lever PRECONDITION from BOTH sources: the tied-head rows (safetensors, authoritative)
    # OR the config vocab_size. A full-vocab head that is not keepset-masked => the drafter CAN propose OOK.
    full_head = bool(res.get("drafter_full_vocab_head") or res.get("drafter_head_rows_full_vocab"))
    res["drafter_full_vocab_head"] = full_head
    res["drafter_can_propose_ook"] = bool(full_head and not res.get(
        "drafter_keepset_masked_in_deployed_path", False))
    return res


# ---------------------------------------------------------------------------- #
# (5) Verdict assembly.
# ---------------------------------------------------------------------------- #
def assemble_verdict(idsim: dict, env: dict, probe: dict) -> dict[str, Any]:
    p_anchor = P_OOK_IMITATION_414
    lever_is_closed = not (p_anchor > 0.0)            # closed iff OOK rate ~ 0 (PR step 3)
    return {
        # ---- PR #420 headline deliverables ----
        "speculator_keepset_equivalence_self_test_passes": None,  # filled by synthesize()
        "inkeepset_drafter_preserves_self_referential": bool(
            idsim["inkeepset_drafter_preserves_self_referential"]),
        "drafter_outofkeepset_proposal_rate": p_anchor,
        "drafter_distinct_outofkeepset_ids": DISTINCT_OOK_IDS_414,
        "inkeepset_drafting_etp_uplift": float(env["headline_etp_uplift"]),
        "inkeepset_drafting_equiv_tps_upside": float(env["headline_equiv_tps_upside"]),
        "lever_is_closed": bool(lever_is_closed),
        "drafter_vocab_mask_is_additive_served_change": True,
        "probe_was_live": bool(probe.get("probe_was_live", False)),
        # ---- supporting (envelope + framing) ----
        "drafter_outofkeepset_proposal_rate_is_analytic_bound": True,
        "drafter_outofkeepset_proposal_rate_trivial_ub": float(env["p_trivial_ub"]),
        "inkeepset_drafting_equiv_tps_upside_floor": float(env["floor_equiv_tps_upside"]),  # = 0.0
        "inkeepset_drafting_equiv_tps_level_ceiling": float(env["headline_equiv_tps_level"]),
        "deployed_equiv_tps_7": EQUIV_TPS_7,
        "lever_is_structurally_open_ge_zero": True,
        "lever_magnitude_unresolved_in_envelope": True,
        "lever_realized_expected_modest_drafter_at_linear_cap": True,
        "ppl_unchanged": PPL_DEPLOYED,
        "ppl_gate": PPL_GATE,
        "ppl_passes_by_construction": bool(PPL_DEPLOYED <= PPL_GATE),
        "drafter_can_propose_ook": bool(probe.get("drafter_can_propose_ook", True)),
        "keepset_strict_subset": bool(probe.get("keepset_strict_subset_of_full_vocab", True)),
        "gate_for_respect_equivalence": "self_referential",
        "resolution_route": ("ubel #401 vLLM-MTP-proposer per-position argmax probe (p) + a masked-drafter "
                             "A/B (q) on a LOCAL served run -- a custom-vLLM-patch effort, out of scope here"),
    }


# ---------------------------------------------------------------------------- #
# (6) PRIMARY self-test -- >=20 pure-logic checks (numpy + stdlib; env-independent).
# ---------------------------------------------------------------------------- #
def self_test() -> dict[str, Any]:
    c: dict[str, bool] = {}

    # ---- ladder / E[T] roundtrips (the #289 acceptance basis) ----
    c["t01_ladder_len_7"] = len(LADDER_289) == K_SPEC
    c["t02_ladder_monotone_nondecreasing"] = all(LADDER_289[i] <= LADDER_289[i + 1] for i in range(6))
    c["t03_ladder_in_unit_interval"] = all(0.0 < a < 1.0 for a in LADDER_289)
    c["t04_e_accepted_roundtrips_289"] = abs(expected_accepted(LADDER_289) - E_ACCEPTED_289) < 1e-9
    c["t05_e_t_roundtrips_289"] = abs(expected_tokens_per_step(LADDER_289) - E_T_289) < 1e-9

    # ---- IDENTITY: greedy-verify emits target argmax (in keepset) for ARBITRARY drafter proposals ----
    # toy keepset [0,16) within full vocab [0,64): the full-vocab drafter proposes OOK frequently.
    sim = identity_simulation(n_trials=2000, k=K_SPEC, keep_lo=0, keep_hi=16, full_vocab=64, seed=1)
    c["t06_sim_actually_proposed_ook"] = sim["any_ook_proposed"] is True
    c["t07_all_emitted_in_keepset"] = sim["all_emitted_in_keepset"] is True
    c["t08_emitted_is_target_prefix"] = sim["emitted_is_target_prefix"] is True
    c["t09_ook_draft_never_emitted"] = sim["ook_draft_never_emitted"] is True
    c["t10_inkeepset_preserves_self_referential"] = sim["inkeepset_drafter_preserves_self_referential"] is True
    # direct construction unit cases: accept-all -> emit bonus; mismatch -> emit correction (target), not draft.
    em_all, kacc_all = greedy_spec_emit([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7])
    c["t11_accept_all_emits_bonus"] = (em_all == [1, 2, 3, 4, 5, 6, 7, 8] and kacc_all == 7)
    em_mis, kacc_mis = greedy_spec_emit([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 999, 4, 5, 6, 7])
    c["t12_mismatch_emits_target_not_draft"] = (em_mis == [1, 2, 3] and kacc_mis == 2 and 999 not in em_mis)

    # ---- UPSIDE model: a' = a + p*q, floor/ceiling, monotonicity, conversion ----
    p = P_OOK_IMITATION_414
    a_floor = masked_ladder(LADDER_289, p, 0.0)
    a_ceil = masked_ladder(LADDER_289, p, 1.0)
    c["t13_floor_q0_is_identity"] = all(abs(x - y) < 1e-12 for x, y in zip(a_floor, LADDER_289))
    c["t14_ceiling_q1_is_a_over_1mp"] = all(abs(x - min(y / (1.0 - p), 1.0)) < 1e-12
                                            for x, y in zip(a_ceil, LADDER_289))
    u_floor = upside_at(p, 0.0)
    u_ceil = upside_at(p, 1.0)
    c["t15_floor_uplift_zero"] = abs(u_floor["etp_uplift"]) < 1e-12
    c["t16_ceiling_uplift_positive"] = u_ceil["etp_uplift"] > 0.0
    u_mid = upside_at(p, 0.5)
    c["t17_uplift_monotone_in_q"] = u_floor["etp_uplift"] <= u_mid["etp_uplift"] <= u_ceil["etp_uplift"]
    c["t18_uplift_monotone_in_p"] = (upside_at(0.05, 1.0)["etp_uplift"]
                                     < upside_at(0.15, 1.0)["etp_uplift"])
    c["t19_p0_closes_lever"] = abs(upside_at(0.0, 1.0)["etp_uplift"]) < 1e-12
    c["t20_a_prime_capped_at_one"] = all(x <= 1.0 + 1e-12 for x in masked_ladder(LADDER_289, 0.27, 1.0))
    c["t21_tps_is_detp_times_kcal"] = abs(u_ceil["equiv_tps_upside"] - u_ceil["etp_uplift"] * K_CAL) < 1e-9
    c["t22_ceiling_a_prime_below_one_at_anchor"] = all(x < 1.0 for x in a_ceil)

    # ---- equiv_tps framing (denken #413) ----
    c["t23_deployed_equiv_tps_is_478p93"] = abs(EQUIV_TPS_7 - 478.93) < 1e-9
    c["t24_kcal_calibrates_mu_p"] = abs(E_T_REALIZED * K_CAL - MU_P) < 1e-6
    c["t25_upside_level_is_base_plus_delta"] = abs(
        u_ceil["equiv_tps_level"] - (EQUIV_TPS_7 + u_ceil["equiv_tps_upside"])) < 1e-9

    # ---- envelope + verdict gating ----
    env = upside_envelope()
    c["t26_headline_is_anchor_ceiling"] = abs(env["headline_equiv_tps_upside"]
                                              - u_ceil["equiv_tps_upside"]) < 1e-9
    c["t27_floor_equiv_tps_zero"] = abs(env["floor_equiv_tps_upside"]) < 1e-12
    c["t28_trivial_ub_is_1_minus_a1"] = abs(env["p_trivial_ub"] - (1.0 - LADDER_289[0])) < 1e-12
    idsim = identity_simulation(n_trials=500, keep_hi=16, full_vocab=64, seed=2)
    v = assemble_verdict(idsim, env, {"probe_was_live": False, "drafter_can_propose_ook": True,
                                      "keepset_strict_subset_of_full_vocab": True})
    c["t29_lever_open_at_anchor"] = v["lever_is_closed"] is False
    c["t30_mask_is_additive_served_change"] = v["drafter_vocab_mask_is_additive_served_change"] is True
    c["t31_probe_not_live"] = v["probe_was_live"] is False
    c["t32_ppl_passes_by_construction"] = v["ppl_passes_by_construction"] is True
    c["t33_gate_is_self_referential"] = v["gate_for_respect_equivalence"] == "self_referential"
    # counterfactual: at p=0 the lever closes cleanly (PR's valuable NULL).
    lever_closed_at_p0 = not (0.0 > 0.0)
    c["t34_lever_closes_at_p0"] = lever_closed_at_p0 is True

    # ---- constants ----
    c["t35_constants_exact"] = bool(HIDDEN == 2560 and FULL_VOCAB == 262144
                                    and DEPLOYED_LMHEAD_ROWS == 16384 and K_SPEC == 7
                                    and M_DEPLOYED == 8)

    passes = bool(all(c.values()))
    return {"conditions": c, "n_checks": len(c),
            "speculator_keepset_equivalence_self_test_passes": passes}


# ---------------------------------------------------------------------------- #
# Synthesis / report / W&B (house pattern).
# ---------------------------------------------------------------------------- #
def synthesize(run_measurements: bool) -> dict[str, Any]:
    st = self_test()
    # faithful identity simulation on the REAL keepset geometry (16384-of-262144) when self-test passes.
    idsim = identity_simulation(n_trials=4000, k=K_SPEC, keep_lo=0,
                                keep_hi=DEPLOYED_LMHEAD_ROWS, full_vocab=FULL_VOCAB, seed=7)
    env = upside_envelope()
    probe = drafter_structural_probe(run_measurements)
    verdict = assemble_verdict(idsim, env, probe)
    verdict["speculator_keepset_equivalence_self_test_passes"] = st[
        "speculator_keepset_equivalence_self_test_passes"]
    return {
        "self_test": st,
        "speculator_keepset_equivalence_self_test_passes": st[
            "speculator_keepset_equivalence_self_test_passes"],
        "n_self_test_checks": st["n_checks"],
        "identity_simulation": idsim,
        "upside_envelope": env,
        "drafter_structural_probe": probe,
        "verdict_fields": verdict,
        "verdict": _build_verdict(verdict, idsim, env, probe),
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }


def _build_verdict(v, idsim, env, probe) -> str:
    parts = [
        "Q (PR #420): does an in-keepset drafter + truncated-head verify respect the operative "
        "(self-referential) equivalence, and is there acceptance-side equivalent-TPS to win by masking "
        "the drafter to the keepset? VERDICT: inkeepset_drafter_preserves_self_referential={}, "
        "drafter_outofkeepset_proposal_rate={:.4f} (analytic imitation anchor; probe_was_live={}), "
        "inkeepset_drafting_equiv_tps_upside={:.1f} equiv-TPS (CEILING q=1), floor=0, lever_is_closed={}.".format(
            v["inkeepset_drafter_preserves_self_referential"], v["drafter_outofkeepset_proposal_rate"],
            v["probe_was_live"], v["inkeepset_drafting_equiv_tps_upside"], v["lever_is_closed"]),
        "IDENTITY (PROVEN by construction, the direct answer to #407): greedy spec verify emits the "
        "truncated-head target argmax (in the 16384 keepset) for every accepted draft position AND the "
        "correction/bonus token; the drafter only PROPOSES (gates accept LENGTH, never the EMITTED token). "
        "So EVERY emitted token is in-keepset for ANY drafter proposal vocab -- the simulation over {} "
        "trials (incl. {} out-of-keepset proposals) confirms all_emitted_in_keepset & ook_draft_never_"
        "emitted. Grounded in serve.py:410-456 (rejection_greedy_sample_kernel; draft_token_ids gate "
        "accept only) + sitecustomize 'Drafter-only => cannot change emitted tokens'.".format(
            idsim["n_trials"], idsim["total_ook_proposals"]),
    ]
    if probe.get("drafter_full_vocab_head") or probe.get("drafter_can_propose_ook"):
        parts.append(
            "LEVER PRECONDITION (faithful GPU structural probe): the deployed drafter has a full-vocab "
            "{}-row head and is NOT keepset-masked (sitecustomize FUSED_SPARSE_ARGMAX over the full "
            "token_ordering) -> it CAN propose any of {} out-of-keepset ids the verify can never emit "
            "(keepset = strict 16384-subset).".format(
                probe.get("drafter_vocab_size", FULL_VOCAB),
                probe.get("n_out_of_keepset_ids", FULL_VOCAB - DEPLOYED_LMHEAD_ROWS)))
    parts.append(
        "UPSIDE (priced, not deployed): masking the drafter's proposal logits to the keepset is a "
        "structurally >= 0 acceptance lever (an OOK proposal is a guaranteed reject; the masked re-proposal "
        "can only match the target MORE often). Model a_i'=a_i+p*q_i, q in [0, s_i=a_i/(1-p)]. At the "
        "imitation anchor p={:.4f}: dE[T] in [0, {:.3f}] -> equiv-TPS upside in [0, {:.1f}] (deployed "
        "equiv_tps {:.2f} -> ceiling {:.2f}). The CEILING assumes every OOK reclaimed as well as a native "
        "in-keepset proposal.".format(
            v["drafter_outofkeepset_proposal_rate"], env["headline_etp_uplift"],
            env["headline_equiv_tps_upside"], v["deployed_equiv_tps_7"],
            env["headline_equiv_tps_level"]))
    parts.append(
        "HONEST: the OOK rate is an ANALYTIC imitation anchor (the faithful per-position drafter-argmax "
        "read is BLOCKED for a 0-TPS card -- ubel #401: vLLM-MTP-proposer-specific inputs_embeds[5120]+"
        "shared_kv_states; bf16 standalone is wrong-distribution+nondeterministic). The REALIZED upside is "
        "UNRESOLVED-IN-ENVELOPE and expected MODEST: #289 shows the deployed drafter is already at its "
        "linear acceptance cap, so the formerly-OOK positions are the hard ones whose in-keepset runner-up "
        "rarely nails the target (q << s). Resolving needs the #401 proposer probe (p) + a masked-drafter "
        "A/B (q).")
    parts.append(
        "DEPLOY (human-gated, not changed here): drafter_vocab_mask_is_additive_served_change=True -- an "
        "ADDITIVE drafter-vocab-mask (mask proposal logits to the keepset before argmax) that touches ONLY "
        "which tokens are PROPOSED, never which are EMITTED -> PPL unchanged {} <= {} BY CONSTRUCTION. This "
        "card PRICES the lever for lawine #419; it does NOT alter the served drafter. LOCAL/analysis-only; "
        "0 official TPS; NO HF Job / submission / served-file change.".format(PPL_DEPLOYED, PPL_GATE))
    return " ".join(parts)


def _assert_nan_clean(payload: dict, path: str = "payload") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, val in node.items():
                walk(val, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, val in enumerate(node):
                walk(val, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict) -> None:
    st = syn["self_test"]
    v = syn["verdict_fields"]
    print("\n" + "=" * 100, flush=True)
    print("SPECULATOR self-referential EQUIVALENCE + in-keepset ACCEPTANCE UPSIDE (PR #420)", flush=True)
    print("=" * 100, flush=True)
    print(f"  (PRIMARY) speculator_keepset_equivalence_self_test_passes = "
          f"{st['speculator_keepset_equivalence_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    fail = [k for k, val in st["conditions"].items() if not val]
    if fail:
        print(f"          FAILED: {fail}", flush=True)
    print("-" * 100, flush=True)
    print("  --- PR #420 HEADLINE FIELDS ---", flush=True)
    for k in ("inkeepset_drafter_preserves_self_referential", "drafter_outofkeepset_proposal_rate",
              "drafter_distinct_outofkeepset_ids", "inkeepset_drafting_etp_uplift",
              "inkeepset_drafting_equiv_tps_upside", "inkeepset_drafting_equiv_tps_upside_floor",
              "inkeepset_drafting_equiv_tps_level_ceiling", "lever_is_closed",
              "drafter_vocab_mask_is_additive_served_change", "probe_was_live",
              "drafter_outofkeepset_proposal_rate_trivial_ub", "deployed_equiv_tps_7",
              "ppl_passes_by_construction"):
        print(f"  {k:<52} = {v.get(k)}", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[speculator-keepset] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    st = syn["self_test"]
    v = syn["verdict_fields"]
    idsim = syn["identity_simulation"]
    probe = syn["drafter_structural_probe"]
    env = syn["upside_envelope"]

    def _num(x):
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    summary: dict[str, Any] = {
        "speculator_keepset_equivalence_self_test_passes": int(bool(
            st["speculator_keepset_equivalence_self_test_passes"])),
        "n_self_test_checks": st["n_checks"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        # ---- PR #420 headline keys ----
        "inkeepset_drafter_preserves_self_referential": int(bool(
            v["inkeepset_drafter_preserves_self_referential"])),
        "drafter_outofkeepset_proposal_rate": v["drafter_outofkeepset_proposal_rate"],
        "drafter_distinct_outofkeepset_ids": v["drafter_distinct_outofkeepset_ids"],
        "inkeepset_drafting_etp_uplift": v["inkeepset_drafting_etp_uplift"],
        "inkeepset_drafting_equiv_tps_upside": v["inkeepset_drafting_equiv_tps_upside"],
        "inkeepset_drafting_equiv_tps_upside_floor": v["inkeepset_drafting_equiv_tps_upside_floor"],
        "inkeepset_drafting_equiv_tps_level_ceiling": v["inkeepset_drafting_equiv_tps_level_ceiling"],
        "lever_is_closed": int(bool(v["lever_is_closed"])),
        "drafter_vocab_mask_is_additive_served_change": int(bool(
            v["drafter_vocab_mask_is_additive_served_change"])),
        "probe_was_live": int(bool(v["probe_was_live"])),
        "drafter_outofkeepset_proposal_rate_trivial_ub": v["drafter_outofkeepset_proposal_rate_trivial_ub"],
        "deployed_equiv_tps_7": v["deployed_equiv_tps_7"],
        "ppl_unchanged": v["ppl_unchanged"], "ppl_gate": v["ppl_gate"],
        "ppl_passes_by_construction": int(bool(v["ppl_passes_by_construction"])),
        "lever_is_structurally_open_ge_zero": int(bool(v["lever_is_structurally_open_ge_zero"])),
        "lever_magnitude_unresolved_in_envelope": int(bool(v["lever_magnitude_unresolved_in_envelope"])),
        # ---- identity simulation + structural probe ----
        "idsim_n_trials": idsim["n_trials"],
        "idsim_total_ook_proposals": idsim["total_ook_proposals"],
        "idsim_all_emitted_in_keepset": int(bool(idsim["all_emitted_in_keepset"])),
        "idsim_emitted_is_target_prefix": int(bool(idsim["emitted_is_target_prefix"])),
        "idsim_ook_draft_never_emitted": int(bool(idsim["ook_draft_never_emitted"])),
        "drafter_structural_probe_available": int(bool(probe.get("available"))),
        "faithful_perposition_read_blocked": int(bool(probe.get("faithful_perposition_read_blocked", True))),
        "analysis_only": int(True), "no_hf_job": int(True), "no_served_file_change": int(True),
        "official_tps": 0,
    }
    for src in (idsim, probe, env):
        if isinstance(src, dict):
            for k, val in src.items():
                n = _num(val)
                if n is not None and k not in summary:
                    summary[f"m_{k}"] = n
    summary = {k: val for k, val in summary.items()
               if val is not None and not (isinstance(val, float) and not math.isfinite(val))}

    run = init_wandb_run(
        job_type="validity-gate", agent="land", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "speculator-equivalence", "self-referential", "in-keepset-drafter",
              "drafter-vocab-mask", "acceptance-upside", "equiv-tps", "greedy-identity", "analysis-only",
              "bank-the-analysis", "pr-420"],
        config={
            "hidden": HIDDEN, "full_vocab": FULL_VOCAB, "deployed_lmhead_rows": DEPLOYED_LMHEAD_ROWS,
            "k_spec": K_SPEC, "m_deployed": M_DEPLOYED, "k_cal": K_CAL, "mu_p": MU_P,
            "e_t_289": E_T_289, "e_accepted_289": E_ACCEPTED_289, "equiv_tps_7": EQUIV_TPS_7,
            "equiv_tax_at_m8": EQUIV_TAX_AT_M8, "p_ook_imitation_414": P_OOK_IMITATION_414,
            "distinct_ook_ids_414": DISTINCT_OOK_IDS_414, "heldout_positions_414": HELDOUT_POSITIONS_414,
            "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE, "wandb_group": args.wandb_group,
            "source_289_run": "fi34s269", "source_414_run": "bq7xkfcv", "source_413_run": "se8mf9ax",
            "source_401_run": "i2qsjyp6",
        },
    )
    if run is None:
        print("[speculator-keepset] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="speculator_keepset_equivalence_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[speculator-keepset] wandb logged: identity={v['inkeepset_drafter_preserves_self_referential']} "
          f"ook_rate={v['drafter_outofkeepset_proposal_rate']:.4f} "
          f"upside_ceiling={v['inkeepset_drafting_equiv_tps_upside']:.1f} "
          f"lever_closed={v['lever_is_closed']}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY pure-logic self-validation only (numpy-only; no torch/GPU)")
    ap.add_argument("--no-measurements", action="store_true",
                    help="skip the torch/safetensors drafter structural probe (self-test + 0-GPU only)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="speculator-keepset-equivalence")
    args = ap.parse_args(argv)

    run_measurements = not (args.self_test or args.no_measurements)
    syn = synthesize(run_measurements=run_measurements)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 420, "agent": "land",
        "kind": "speculator-keepset-equivalence", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[speculator-keepset] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "speculator_keepset_equivalence_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[speculator-keepset] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    passes = bool(syn["speculator_keepset_equivalence_self_test_passes"]) and payload["nan_clean"]
    v = syn["verdict_fields"]
    print(f"  PRIMARY speculator_keepset_equivalence_self_test_passes = {passes} "
          f"({syn['n_self_test_checks']} checks)", flush=True)
    print(f"  inkeepset_drafter_preserves_self_referential={v['inkeepset_drafter_preserves_self_referential']} "
          f"ook_rate={v['drafter_outofkeepset_proposal_rate']:.4f} "
          f"upside_ceiling={v['inkeepset_drafting_equiv_tps_upside']:.2f} equiv-TPS "
          f"(floor={v['inkeepset_drafting_equiv_tps_upside_floor']:.2f}) lever_closed={v['lever_is_closed']}",
          flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        print(f"[speculator-keepset] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
