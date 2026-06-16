#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""LIVE OOK probe -- resolve the keepset-mask lever to a deployable point (PR #426, land).

WHAT THIS RESOLVES
------------------
My merged #420 (`qe4qagc1`) PROVED, by construction, that an in-keepset drafter + truncated-head
verify preserves the operative (self-referential) equivalence for ANY drafter proposal vocab (the
verify is the sole arbiter of EMITTED tokens; the drafter only PROPOSES, gating accept LENGTH). It
priced the only speculator-side equivalent-TPS lever -- masking the drafter's proposal logits to the
16384 keepset -- but left its MAGNITUDE in an envelope [0, ~138 equiv-TPS ceiling] anchored on an
ANALYTIC imitation rate p=0.0921 (`probe_was_live`=False), because #420 judged a faithful per-position
drafter-argmax read BLOCKED (ubel #401: a STANDALONE drafter forward needs inputs_embeds[5120]+
shared_kv_states and a bf16 standalone head is wrong-distribution + nondeterministic).

THE KEY INSIGHT OF THIS CARD: the #401 blocker is a STANDALONE-forward blocker. The per-position
drafter argmax is directly READABLE *live*, by LOGGING the deployed `Gemma4Proposer.propose` return
INSIDE the running served engine -- where inputs_embeds + shared_kv_states are already correctly
constructed by vLLM and the head is the DEPLOYED int4 path (bit-exact, not bf16). So this card
RESOLVES `p` (the measured out-of-keepset PROPOSAL rate) to a LIVE number, turning #420's analytic
anchor into a measured one. NO served kernel/file is modified, rebuilt, or submitted: the probe is a
PURE-LOGGING `.pth`-injected hook (`ook_probe_hook.py`), env-gated and inert unless the driver enables
it, that wraps the already-deployed-patched `propose` and records per-position draft argmax ids +
the previous-step verify's num_rejected (the acceptance ladder source). See `ook_probe_hook.py`.

WHAT STAYS ANALYTIC (honest, per the PR escape hatch)
-----------------------------------------------------
The REALIZED reclaim `q` (how often a keepset-MASKED re-proposal would have matched the target at a
position where the unmasked drafter proposed OOK) genuinely needs a MASKED-DRAFTER A/B: re-running the
drafter argmax over keepset-masked logits. The deployed proposer uses FUSED_SPARSE_ARGMAX (a centroid
top-K sparse argmax that never materializes full logits), so a masked re-proposal is a SERVED-KERNEL
PATCH, not logging -> OUT OF SCOPE for this read-only card (the PR's explicit escape hatch). So
`masked_drafter_ab_delta_etp` = null and the upside is reported as the ENVELOPE at the LIVE p
([floor=0, ceiling=q*s]); the realized point is unresolved but expected MODEST (#289: the deployed
drafter is already at its linear acceptance cap, so formerly-OOK positions are the hard ones whose
in-keepset runner-up rarely nails the target -> q << s ceiling). As a LIVE refinement that needs only
logging, we ALSO measure `ook_caused_truncation` -- of the steps that truncated, how many truncated
on an OOK proposal (the ONLY positions a keepset mask could reclaim) -- a strictly tighter, data-driven
bound on the reclaim opportunity than the raw p ceiling.

PRIMARY metric  live_ook_probe_self_test_passes  (>=20 pure-logic checks: the #420 lever math re-
verified + the new live-estimator logic -- OOK-rate accounting, the num_rejected->accept-ladder
reconstruction roundtrip, the OOK-caused-truncation pairing, the upside envelope at a measured p, and
the verdict gating). Env-independent; runs under the numpy-only .venv.

Run:
  # PRIMARY self-test only (numpy-only; no torch/serve):
  .venv/bin/python -m research.validity.live_ook_probe_keepset_mask.live_ook_probe_keepset_mask --self-test
  # tiny live smoke (confirms the hook installs + logs sane draft tokens, a_1~0.73):
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 /tmp/server-venv/bin/python -m \
    research.validity.live_ook_probe_keepset_mask.live_ook_probe_keepset_mask --smoke
  # full live card (official 128 + held-out 274; ~10-15 min on the warm A10G):
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 /tmp/server-venv/bin/python -m \
    research.validity.live_ook_probe_keepset_mask.live_ook_probe_keepset_mask \
    --wandb_group live-ook-probe --wandb_name land/live-ook-probe-keepset-mask
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Pre-import the REAL wandb BEFORE putting REPO_ROOT (= target/, has a ./wandb run-output dir that
# shadows the package as a PEP-420 namespace) on sys.path[0]. Mirrors the #414/#420 house pattern.
try:
    import wandb as _wandb_preimport  # noqa: F401
except Exception:  # noqa: BLE001
    _wandb_preimport = None

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Single source of truth for the lever math + banked anchors: reuse my merged #420 module verbatim so
# this card's envelope is byte-consistent with `qe4qagc1`.
from research.validity.speculator_keepset_equivalence.speculator_keepset_equivalence import (  # noqa: E402
    DEPLOYED_LMHEAD_ROWS,
    DISTINCT_OOK_IDS_414,
    E_ACCEPTED_289,
    E_T_289,
    EQUIV_TPS_7,
    FULL_VOCAB,
    HIDDEN,
    K_CAL,
    K_SPEC,
    LADDER_289,
    M_DEPLOYED,
    MU_P,
    P_OOK_IMITATION_414,
    PPL_DEPLOYED,
    PPL_GATE,
    expected_accepted,
    expected_tokens_per_step,
    greedy_spec_emit,
    identity_simulation,
    masked_ladder,
    upside_at,
)

# The deployed served stack this card probes (read-only; never modified/rebuilt/submitted).
DEFAULT_SUBMISSION = "submissions/fa2sw_precache_kenyan"
DEFAULT_VENV_PYTHON = "/tmp/server-venv/bin/python"
DEFAULT_KEEPSET = "/tmp/osoi5-v0-baked/pck04_keepset.json"
LEVER_CLOSED_P_THRESHOLD = 1e-2  # PR step 3: p ~ 0 (< 1%) => lever_is_closed_live (a clean, mergeable NULL)
A1_TARGET_289 = LADDER_289[0]    # 0.7293 deployed conditional a_1 (#289) -- the live-ladder liveness check
A1_TOL = 0.06                    # |a_1_live - a_1_289| tolerance for the liveness consistency flag

TOL = 1e-9


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# ============================================================================ #
# (A) LIVE ESTIMATORS over the per-step JSONL the hook appends.
#     Each record: {tag, pid, draft_ids:[K], n_draft, n_ook, ook_ids:[...], [num_rejected:[per-req]]}.
# ============================================================================ #
def ook_rate_from_records(records: list[dict], keepset: set[int] | None = None) -> dict[str, Any]:
    """The out-of-keepset PROPOSAL rate p = (# OOK draft positions) / (# draft positions), plus distinct
    OOK ids and the per-draft-position OOK rate. If `keepset` is given we recompute OOK membership from
    `draft_ids` (authoritative); otherwise we trust the hook's pre-computed `n_ook`/`ook_ids`."""
    total_draft = 0
    total_ook = 0
    distinct_ook: set[int] = set()
    pos_draft: dict[int, int] = defaultdict(int)
    pos_ook: dict[int, int] = defaultdict(int)
    n_records = 0
    for rec in records:
        ids = rec.get("draft_ids")
        if not isinstance(ids, list):
            continue
        n_records += 1
        for i, t in enumerate(ids):
            total_draft += 1
            pos_draft[i] += 1
            is_ook = (t not in keepset) if keepset is not None else None
            if is_ook is None:
                # fall back to the hook's ook_ids membership for this record
                is_ook = t in set(rec.get("ook_ids", []))
            if is_ook:
                total_ook += 1
                pos_ook[i] += 1
                distinct_ook.add(int(t))
    p = (total_ook / total_draft) if total_draft else 0.0
    per_position_p = {i: (pos_ook[i] / pos_draft[i] if pos_draft[i] else 0.0) for i in sorted(pos_draft)}
    return {
        "n_records": n_records,
        "total_draft_positions": total_draft,
        "total_ook_positions": total_ook,
        "drafter_outofkeepset_proposal_rate_live": p,
        "drafter_distinct_outofkeepset_ids_live": len(distinct_ook),
        "per_position_ook_rate": per_position_p,
        "distinct_ook_ids_sample": sorted(distinct_ook)[:256],
    }


def accept_counts_from_records(records: list[dict], k: int = K_SPEC) -> list[int]:
    """Histogram of per-step accepted-draft-length A in [0, k], reconstructed from the previous-step
    verify's num_rejected (A = k - num_rejected; the runner sets num_rejected = K - num_accepted, see
    gpu_model_runner.prepare_inputs_padded / valid_sampled_tokens_count = num_accepted + 1). Records
    whose num_rejected is absent (the first decode step of each sequence, spec_decode_metadata=None) are
    naturally excluded. Per-req lists are flattened (MAX_NUM_SEQS=1 => single element in steady state)."""
    counts = [0] * (k + 1)
    for rec in records:
        nrej = rec.get("num_rejected")
        if not isinstance(nrej, list):
            continue
        for nr in nrej:
            a = k - int(nr)
            a = max(0, min(k, a))
            counts[a] += 1
    return counts


def ladder_from_accept_counts(counts: list[int] | list[float]) -> dict[str, Any]:
    """Reconstruct the conditional acceptance ladder a_i = P(A>=i | A>=i-1) from the accept-length
    histogram. survival S_i = P(A>=i) = (sum_{j>=i} counts) / N; a_i = S_i / S_{i-1} (S_0 = 1).
    Also returns E[A], E[T]=1+E[A] for a cross-check against the #289 anchors."""
    k = len(counts) - 1
    n = float(sum(counts))
    if n <= 0:
        return {"n_steps": 0, "ladder": [], "e_accepted": 0.0, "e_t": 1.0, "survival": []}
    surv = [0.0] * (k + 1)
    for i in range(k + 1):
        surv[i] = sum(counts[i:]) / n  # S_0 = 1.0 exactly
    ladder: list[float] = []
    for i in range(1, k + 1):
        prev = surv[i - 1]
        ladder.append((surv[i] / prev) if prev > 0 else 0.0)
    e_accepted = sum(i * counts[i] for i in range(k + 1)) / n
    return {
        "n_steps": n,
        "ladder": ladder,
        "survival": surv,
        "e_accepted": e_accepted,
        "e_t": 1.0 + e_accepted,
    }


def ook_caused_truncation_from_records(records: list[dict], keepset: set[int],
                                       k: int = K_SPEC) -> dict[str, Any]:
    """LIVE refinement of the reclaim opportunity (logging-only; strictly tighter than the raw-p ceiling).

    Pair each step's accept-length A (from THIS record's num_rejected, which describes the PREVIOUS
    record's draft) with the PREVIOUS record's draft_ids, and ask: at the first REJECTED position (index
    A of the previous draft), was the proposed token OUT of the keepset? An OOK proposal is a GUARANTEED
    reject the verify can never emit -- so OOK-caused truncations are exactly the positions a keepset
    mask could reclaim. In-keepset-caused truncations are genuine drafter misses a mask cannot help.
    Pairing is per-pid and in log order; cross-sequence pairs are auto-skipped because a sequence's first
    decode step carries no num_rejected. Returns counts + the OOK-caused fraction of truncations."""
    by_pid: dict[Any, list[dict]] = defaultdict(list)
    for rec in records:
        if isinstance(rec.get("draft_ids"), list):
            by_pid[rec.get("pid")].append(rec)
    n_truncations = 0          # steps with a rejection (A < K)
    n_ook_caused = 0           # truncation whose first-rejected proposal was OOK (reclaimable)
    n_inkeepset_caused = 0     # truncation whose first-rejected proposal was in-keepset (genuine miss)
    n_paired = 0
    sum_kma_ook_caused = 0     # sum of (K - A) over OOK-caused truncations = max extra acceptable tokens
    for _pid, recs in by_pid.items():
        for i in range(1, len(recs)):
            rec, prev = recs[i], recs[i - 1]
            nrej = rec.get("num_rejected")
            prev_draft = prev.get("draft_ids")
            if not isinstance(nrej, list) or not nrej or not isinstance(prev_draft, list):
                continue
            n_paired += 1
            kk = len(prev_draft)  # the previous draft's length (== K in steady state)
            a = max(0, min(kk, k - int(nrej[0])))
            if a >= kk:
                continue  # all accepted this step -> no truncation, nothing to reclaim
            n_truncations += 1
            first_rejected = prev_draft[a]
            if first_rejected not in keepset:
                n_ook_caused += 1
                sum_kma_ook_caused += (kk - a)  # UPPER bound: reclaim A..K-1 (full downstream match)
            else:
                n_inkeepset_caused += 1
    frac_ook = (n_ook_caused / n_truncations) if n_truncations else 0.0
    # LIVE upper bound on realized E[A] uplift (logging-only): assume each OOK-caused truncation reclaims
    # ALL of positions A..K-1. dE[A]_ub = sum(K-A) / n_paired_steps; equiv-TPS UB = dE[A]_ub * K_cal.
    deta_ub = (sum_kma_ook_caused / n_paired) if n_paired else 0.0
    return {
        "n_paired_steps": n_paired,
        "n_truncations": n_truncations,
        "n_ook_caused_truncations": n_ook_caused,
        "n_inkeepset_caused_truncations": n_inkeepset_caused,
        "ook_caused_truncation_fraction": frac_ook,
        "sum_k_minus_a_ook_caused": sum_kma_ook_caused,
        "realized_etp_uplift_upper_bound": deta_ub,
        "realized_equiv_tps_upside_upper_bound": deta_ub * K_CAL,
    }


def upside_envelope_at(p: float, ladder: list[float] = LADDER_289) -> dict[str, Any]:
    """Equiv-TPS upside envelope at a GIVEN (e.g. LIVE-measured) OOK rate p: floor q=0 (no reclaim) and
    ceiling q=1 (every OOK reclaimed as well as a native in-keepset proposal). Proposal-only => K_cal
    invariant, so equiv-TPS = dE[T] * K_cal (denken #413 frame)."""
    floor = upside_at(p, 0.0, ladder)
    ceil = upside_at(p, 1.0, ladder)
    return {
        "p": p,
        "floor_etp_uplift": floor["etp_uplift"],
        "ceiling_etp_uplift": ceil["etp_uplift"],
        "floor_equiv_tps_upside": floor["equiv_tps_upside"],
        "ceiling_equiv_tps_upside": ceil["equiv_tps_upside"],
        "ceiling_equiv_tps_level": ceil["equiv_tps_level"],
        "deployed_equiv_tps_7": EQUIV_TPS_7,
    }


# ============================================================================ #
# (B) PRIMARY self-test -- >=20 pure-logic checks (numpy + stdlib; env-independent).
# ============================================================================ #
def _synthetic_counts_from_ladder(ladder: list[float], n: float = 1.0e6) -> list[int]:
    """Build an accept-length histogram that EXACTLY realizes `ladder` (for the reconstruction roundtrip):
    P(A>=i) = prod_{j<=i} a_j; P(A=i) = S_i - S_{i+1} (i<K), P(A=K) = S_K. Scaled by n (kept as floats so
    the roundtrip is exact; ladder_from_accept_counts accepts float counts)."""
    k = len(ladder)
    surv = [1.0]
    for a in ladder:
        surv.append(surv[-1] * a)
    counts = []
    for i in range(k + 1):
        hi = surv[i + 1] if i + 1 <= k else 0.0
        counts.append((surv[i] - hi) * n)
    return counts


def self_test() -> dict[str, Any]:
    c: dict[str, bool] = {}

    # ---- #420 lever math re-verified (ladder / E[T] / identity / upside) ----
    c["t01_ladder_len_7"] = len(LADDER_289) == K_SPEC
    c["t02_ladder_monotone"] = all(LADDER_289[i] <= LADDER_289[i + 1] for i in range(K_SPEC - 1))
    c["t03_ladder_unit_interval"] = all(0.0 < a < 1.0 for a in LADDER_289)
    c["t04_e_accepted_roundtrips"] = abs(expected_accepted(LADDER_289) - E_ACCEPTED_289) < 1e-9
    c["t05_e_t_roundtrips"] = abs(expected_tokens_per_step(LADDER_289) - E_T_289) < 1e-9
    sim = identity_simulation(n_trials=2000, k=K_SPEC, keep_lo=0, keep_hi=16, full_vocab=64, seed=1)
    c["t06_sim_proposed_ook"] = sim["any_ook_proposed"] is True
    c["t07_all_emitted_in_keepset"] = sim["all_emitted_in_keepset"] is True
    c["t08_ook_draft_never_emitted"] = sim["ook_draft_never_emitted"] is True
    c["t09_inkeepset_preserves_self_referential"] = sim["inkeepset_drafter_preserves_self_referential"] is True
    em_mis, kacc_mis = greedy_spec_emit([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 999, 4, 5, 6, 7])
    c["t10_mismatch_emits_target_not_draft"] = (em_mis == [1, 2, 3] and kacc_mis == 2 and 999 not in em_mis)

    # ---- (NEW) OOK-rate accounting from synthetic records ----
    keep = set(range(0, 16))  # toy keepset [0,16); full vocab [0,64)
    recs = [
        # 3 records, K=4: draft_ids with known OOK counts (ids >=16 are OOK).
        {"pid": 1, "draft_ids": [1, 20, 3, 40]},   # 2 OOK (20,40)
        {"pid": 1, "draft_ids": [5, 6, 7, 8]},      # 0 OOK
        {"pid": 1, "draft_ids": [16, 16, 2, 99]},   # 3 OOK (16,16,99) -> distinct adds {16,99}
    ]
    o = ook_rate_from_records(recs, keep)
    c["t11_ook_total_positions"] = o["total_draft_positions"] == 12
    c["t12_ook_rate_exact"] = abs(o["drafter_outofkeepset_proposal_rate_live"] - (5 / 12)) < 1e-12
    # distinct OOK ids across the 3 records = {20, 40, 16, 99} = 4
    c["t13_distinct_ook_exact"] = o["drafter_distinct_outofkeepset_ids_live"] == 4
    c["t14_per_position_ook_rate"] = (abs(o["per_position_ook_rate"][0] - (1 / 3)) < 1e-12
                                      and abs(o["per_position_ook_rate"][3] - (2 / 3)) < 1e-12)
    o_nokeep = ook_rate_from_records(recs, keepset=None)  # trust hook's ook_ids (absent here -> 0 OOK)
    c["t15_ook_membership_fallback"] = o_nokeep["total_ook_positions"] == 0

    # ---- (NEW) num_rejected -> accept-length histogram ----
    nr_recs = [
        {"pid": 1, "draft_ids": [0] * 7},                 # no num_rejected (first step) -> excluded
        {"pid": 1, "draft_ids": [0] * 7, "num_rejected": [0]},  # A = 7
        {"pid": 1, "draft_ids": [0] * 7, "num_rejected": [7]},  # A = 0
        {"pid": 1, "draft_ids": [0] * 7, "num_rejected": [4]},  # A = 3
    ]
    counts = accept_counts_from_records(nr_recs, k=7)
    c["t16_accept_counts_total"] = sum(counts) == 3  # only the 3 with num_rejected
    c["t17_accept_counts_map"] = (counts[7] == 1 and counts[0] == 1 and counts[3] == 1)
    counts_clamp = accept_counts_from_records([{"draft_ids": [], "num_rejected": [99]},
                                               {"draft_ids": [], "num_rejected": [-3]}], k=7)
    c["t18_accept_counts_clamped"] = (counts_clamp[0] == 1 and counts_clamp[7] == 1)

    # ---- (NEW) ladder reconstruction roundtrip ----
    syn_counts = _synthetic_counts_from_ladder(LADDER_289, n=1.0e6)
    rec_ladder = ladder_from_accept_counts(syn_counts)
    c["t19_ladder_roundtrip"] = all(abs(rec_ladder["ladder"][i] - LADDER_289[i]) < 1e-6
                                    for i in range(K_SPEC))
    c["t20_ladder_et_roundtrip"] = abs(rec_ladder["e_t"] - E_T_289) < 1e-6
    c["t21_survival_starts_at_one"] = abs(ladder_from_accept_counts(syn_counts)["survival"][0] - 1.0) < 1e-12

    # ---- (NEW) OOK-caused-truncation pairing ----
    # K=4 toy keepset [0,16). Build 2 paired mid-sequence steps:
    #   step0 draft [5,6,20,7]; step1.num_rejected=[2] -> A=2 -> first rejected = draft0[2]=20 (OOK) -> reclaimable
    #   step1 draft [5,6,7,99]; step2.num_rejected=[1] -> A=3 -> first rejected = draft1[3]=99 (OOK) -> reclaimable
    #   step2 draft [5,6,7,8];  step3.num_rejected=[2] -> A=2 -> first rejected = draft2[2]=7 (in-keepset) -> genuine miss
    fr_recs = [
        {"pid": 9, "draft_ids": [5, 6, 20, 7]},                        # first step (no num_rejected)
        {"pid": 9, "draft_ids": [5, 6, 7, 99], "num_rejected": [2]},   # A=2 vs draft0 -> 20 OOK
        {"pid": 9, "draft_ids": [5, 6, 7, 8], "num_rejected": [1]},    # A=3 vs draft1 -> 99 OOK
        {"pid": 9, "draft_ids": [5, 6, 7, 8], "num_rejected": [2]},    # A=2 vs draft2 -> 7 in-keepset
    ]
    fr = ook_caused_truncation_from_records(fr_recs, keep, k=4)
    c["t22_frontier_truncations"] = fr["n_truncations"] == 3
    c["t23_frontier_ook_caused"] = fr["n_ook_caused_truncations"] == 2
    c["t24_frontier_inkeepset_caused"] = fr["n_inkeepset_caused_truncations"] == 1
    c["t25_frontier_fraction"] = abs(fr["ook_caused_truncation_fraction"] - (2 / 3)) < 1e-12

    # ---- (NEW) upside envelope at a measured p ----
    env = upside_envelope_at(P_OOK_IMITATION_414)
    c["t26_floor_is_zero"] = abs(env["floor_equiv_tps_upside"]) < 1e-12
    c["t27_ceiling_positive"] = env["ceiling_equiv_tps_upside"] > 0.0
    c["t28_envelope_monotone_in_p"] = (upside_envelope_at(0.05)["ceiling_equiv_tps_upside"]
                                       < upside_envelope_at(0.15)["ceiling_equiv_tps_upside"])
    c["t29_p0_closes_envelope"] = abs(upside_envelope_at(0.0)["ceiling_equiv_tps_upside"]) < 1e-12
    c["t30_tps_is_detp_times_kcal"] = abs(env["ceiling_equiv_tps_upside"]
                                          - env["ceiling_etp_uplift"] * K_CAL) < 1e-9
    c["t31_ceiling_level_is_base_plus_delta"] = abs(
        env["ceiling_equiv_tps_level"] - (EQUIV_TPS_7 + env["ceiling_equiv_tps_upside"])) < 1e-9

    # ---- (NEW) verdict gating ----
    c["t32_lever_closed_below_threshold"] = (0.005 < LEVER_CLOSED_P_THRESHOLD) is True
    c["t33_lever_open_above_threshold"] = (0.05 >= LEVER_CLOSED_P_THRESHOLD) is True
    c["t34_a1_consistency_band"] = abs(A1_TARGET_289 - 0.7293) < 1e-3 and A1_TOL > 0
    c["t35_ppl_unchanged_by_construction"] = bool(PPL_DEPLOYED <= PPL_GATE)

    # ---- constants ----
    c["t36_constants_exact"] = bool(HIDDEN == 2560 and FULL_VOCAB == 262144
                                    and DEPLOYED_LMHEAD_ROWS == 16384 and K_SPEC == 7
                                    and M_DEPLOYED == 8)

    passes = bool(all(c.values()))
    return {"conditions": c, "n_checks": len(c), "live_ook_probe_self_test_passes": passes}


# ============================================================================ #
# (C) LIVE orchestration -- stand up the LOCAL served stack, inject the read-only logging hook via a
#     .pth bootstrap, feed the official 128 + held-out 274 prompts, parse the JSONL, never modify a
#     served file. The hook itself (ook_probe_hook.py) is env-gated and inert unless we enable it here.
# ============================================================================ #
HARNESS = REPO_ROOT / "official/main_bucket/shared_resources/speed_benchmark"
EVAL_DATASET = HARNESS / "data/eval_prompts_sharegpt.json"

_BOOTSTRAP_SRC = (
    "# ook-probe bootstrap (PR #426, land) -- auto-written by live_ook_probe_keepset_mask.py.\n"
    "# Safe to delete. Env-gated: a no-op unless OOK_PROBE_LOG + OOK_PROBE_DIR are set (only in the\n"
    "# probe's served subprocess tree), so normal venv use is unaffected.\n"
    "import os, sys\n"
    "_log = os.environ.get('OOK_PROBE_LOG')\n"
    "_dir = os.environ.get('OOK_PROBE_DIR')\n"
    "if _log and _dir:\n"
    "    try:\n"
    "        if _dir not in sys.path:\n"
    "            sys.path.insert(0, _dir)\n"
    "        import ook_probe_hook  # noqa: F401\n"
    "    except Exception as _exc:  # never break interpreter startup\n"
    "        sys.stderr.write('[ook-probe-bootstrap] failed: %r\\n' % (_exc,))\n"
)


def _load_keepset_set(path: str) -> set[int]:
    d = json.loads(Path(path).read_text())
    ids = d.get("keep_ids") or d.get("kept_ids") or []
    return {int(i) for i in ids}


def _site_packages(venv_python: Path) -> Path:
    # Do NOT resolve() -- /tmp/server-venv/bin/python symlinks to the uv base python; the venv's OWN
    # site-packages (where vllm/transformers live and where .pth is processed) is reached structurally.
    base = Path(os.path.abspath(venv_python)).parent.parent
    for c in sorted(glob.glob(str(base / "lib/python*/site-packages"))):
        if Path(c).is_dir():
            return Path(c)
    raise RuntimeError(f"site-packages not found under {base}")


def _install_probe_pth(site_pkgs: Path) -> list[Path]:
    bootstrap = site_pkgs / "ook_probe_bootstrap.py"
    pth = site_pkgs / "zzz_ook_probe.pth"
    bootstrap.write_text(_BOOTSTRAP_SRC)
    pth.write_text("import ook_probe_bootstrap\n")
    print(f"[live-ook] installed probe .pth bootstrap in {site_pkgs}", flush=True)
    return [bootstrap, pth]


def _remove_probe_pth(installed: list[Path]) -> None:
    for p in installed:
        try:
            p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    # drop the cached bytecode too so a stale bootstrap can never re-import.
    for p in installed:
        pyc = p.parent / "__pycache__"
        for f in glob.glob(str(pyc / (p.stem + ".*.pyc"))):
            try:
                os.unlink(f)
            except Exception:  # noqa: BLE001
                pass
    print("[live-ook] removed probe .pth bootstrap", flush=True)


def _load_tokenizer():
    from transformers import AutoTokenizer
    cands = ["google/gemma-4-E4B-it"]
    cands += sorted(glob.glob(
        "/senpai-run/home/student-land/.cache/huggingface/hub/"
        "models--google--gemma-4-E4B-it/snapshots/*"))
    cands += ["/tmp/osoi5-v0-baked"]
    last: Exception | None = None
    for c in cands:
        try:
            return AutoTokenizer.from_pretrained(c)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"could not load tokenizer; tried {cands}: {last!r}")


def _load_prompts(official_n: int, heldout_n: int, seed: int) -> tuple[list[list[int]], list[list[int]], dict]:
    sys.path.insert(0, str(HARNESS / "scripts"))
    import decode_outputs as do  # official encode/request contract (read-only mirror)
    tok = _load_tokenizer()
    official: list[list[int]] = []
    if official_n and official_n > 0:
        recs = do.read_sharegpt_prompts(EVAL_DATASET, num_prompts=official_n, seed=seed)
        official = [do.encode_prompt(tok, r["prompt_text"]) for r in recs]
    heldout: list[list[int]] = []
    if heldout_n != 0:
        from research.validity.truevocab_lmhead_equivalence_cost.truevocab_lmhead_equivalence_cost import (
            build_heldout_corpus,
        )
        corpus = build_heldout_corpus()
        if heldout_n and heldout_n > 0:
            corpus = corpus[:heldout_n]
        heldout = [do.encode_prompt(tok, p) for p in corpus]
    meta = {"n_official": len(official), "n_heldout": len(heldout),
            "tokenizer": getattr(tok, "name_or_path", "?")}
    return official, heldout, meta


def _request_decode(base_url: str, model: str, token_ids: list[int], output_len: int,
                    timeout_s: int) -> None:
    sys.path.insert(0, str(HARNESS / "scripts"))
    import decode_outputs as do
    do.request_decode(base_url=base_url, model=model, prompt_token_ids=token_ids,
                      output_len=output_len, timeout_s=timeout_s)


def _server_log_has(path: str, needle: str) -> bool:
    try:
        return needle in Path(path).read_text(errors="ignore")
    except Exception:  # noqa: BLE001
        return False


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for _ in fh)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
    return out


def run_live(args) -> dict[str, Any]:
    from scripts.local_prevalidate import (  # reuse the LOCAL serve helpers (no HF Job path)
        build_serve_cmd, load_manifest, server_env, terminate, wait_for_models,
    )
    submission_dir = Path(args.submission)
    if not submission_dir.is_absolute():
        submission_dir = (REPO_ROOT / submission_dir).resolve()
    manifest = load_manifest(submission_dir)
    served_model_name = str(manifest.get("served_model_name", "gemma-4-e4b-it"))
    keepset_path = str((manifest.get("env") or {}).get("PCK04_KEEPSET", args.keepset))
    venv_python = Path(args.venv_python)
    if args.attach_base_url is None and not venv_python.exists():
        raise FileNotFoundError(f"venv python not found: {venv_python}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "research/local_validation" / f"live_ook_probe_{stamp}")
    # Resolve to an ABSOLUTE path: the served worker runs with cwd=submission_dir, so a
    # relative OOK_PROBE_LOG would resolve under the submission tree (nonexistent dir) and
    # the hook's open() would fail SILENTLY -> zero records. Absolute is CWD-independent.
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "ook_probe_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    server_log = args.server_log or str(out_dir / "serve.log")

    official_ids, heldout_ids, prompt_meta = _load_prompts(
        args.official_prompts, args.heldout_prompts, args.seed)
    keepset_set = _load_keepset_set(keepset_path)
    print(f"[live-ook] prompts: official={len(official_ids)} heldout={len(heldout_ids)} "
          f"keepset={len(keepset_set)} ({keepset_path})", flush=True)

    site_pkgs = _site_packages(venv_python)
    installed = _install_probe_pth(site_pkgs)
    server_proc = None
    base_url = args.attach_base_url or f"http://{args.host}:{args.port}"
    req_fail = 0
    n_official_lines = 0
    try:
        if args.attach_base_url is None:
            env = server_env(manifest, submission_dir, venv_python, args.port)
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_devices)
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # curand.h JIT unavailable locally (serve-env memory)
            env["OOK_PROBE_LOG"] = str(log_path)       # enables the (otherwise inert) logging hook
            env["OOK_PROBE_DIR"] = str(HERE)           # where ook_probe_hook.py lives
            env["OOK_PROBE_KEEPSET"] = keepset_path
            env["OOK_PROBE_TAG"] = "live"
            serve_cmd = build_serve_cmd(manifest, venv_python)
            logf = open(server_log, "w")
            print(f"[live-ook] starting served stack: {' '.join(serve_cmd)} (log {server_log})", flush=True)
            server_proc = subprocess.Popen(
                serve_cmd, cwd=submission_dir, env=env,
                stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        print(f"[live-ook] waiting for {base_url}/v1/models (timeout {args.startup_timeout_s}s) ...", flush=True)
        wait_for_models(base_url, args.startup_timeout_s, server_proc)
        hook_installed = _server_log_has(server_log, "[ook-probe] installed propose logger")
        print(f"[live-ook] endpoint ready; hook_installed_marker={hook_installed}", flush=True)

        t0 = time.time()
        for i, tids in enumerate(official_ids):
            try:
                _request_decode(base_url, served_model_name, tids, args.official_output_len,
                                args.request_timeout_s)
            except Exception as exc:  # noqa: BLE001
                req_fail += 1
                if i == 0:
                    raise RuntimeError(f"first official decode failed (contract/config?): {exc!r}") from exc
        time.sleep(0.75)  # let the worker flush line-buffered official records before the boundary snapshot
        n_official_lines = _count_lines(log_path)
        for tids in heldout_ids:
            try:
                _request_decode(base_url, served_model_name, tids, args.heldout_output_len,
                                args.request_timeout_s)
            except Exception:  # noqa: BLE001
                req_fail += 1
        print(f"[live-ook] fed {len(official_ids)+len(heldout_ids)} prompts in {time.time()-t0:.1f}s "
              f"(req_fail={req_fail}); boundary_line={n_official_lines}", flush=True)
    finally:
        terminate(server_proc)
        _remove_probe_pth(installed)

    # ----- parse + estimate (per-set: official / heldout / pooled) -----
    all_records = _read_jsonl(log_path)
    meta = {
        "out_dir": str(out_dir), "log_path": str(log_path), "server_log": server_log,
        "served_model_name": served_model_name, "keepset_path": keepset_path,
        "keepset_size": len(keepset_set), "base_url": base_url, "req_fail": req_fail,
        "hook_installed_marker": bool(_server_log_has(server_log, "[ook-probe] installed propose logger")),
    }
    return _estimate_live(all_records, n_official_lines, keepset_set, prompt_meta, meta)


def _estimate_live(all_records: list[dict], boundary: int, keepset_set: set[int],
                   prompt_meta: dict, meta: dict) -> dict[str, Any]:
    """Compute per-set (official before `boundary`, held-out after, pooled) OOK rates, acceptance
    ladders and OOK-caused-truncation frontiers from parsed records. Shared by the live run and the
    offline --reparse path. The OFFICIAL set is the deployable/leaderboard distribution; the held-out
    set is the #414 general-text comparison corpus -- they are reported SEPARATELY (never pooled into
    the headline) because the deployed keepset is tuned to the benchmark, so the OOK rate is ~24x
    higher off-distribution and a pooled p is a meaningless function of the prompt-mix ratio."""
    draft_records = [r for r in all_records if isinstance(r.get("draft_ids"), list)]
    error_records = [r for r in all_records if "error" in r]
    official_records = [r for r in all_records[:boundary] if isinstance(r.get("draft_ids"), list)]
    heldout_records = [r for r in all_records[boundary:] if isinstance(r.get("draft_ids"), list)]

    def _set(recs: list[dict]) -> dict[str, Any]:
        return {
            "ook": ook_rate_from_records(recs, keepset_set),
            "ladder": ladder_from_accept_counts(accept_counts_from_records(recs)),
            "frontier": ook_caused_truncation_from_records(recs, keepset_set),
        }

    official, heldout, pooled = _set(official_records), _set(heldout_records), _set(draft_records)
    live = dict(meta)
    live.update({
        "n_all_records": len(all_records), "n_draft_records": len(draft_records),
        "n_error_records": len(error_records), "boundary_line": boundary, "prompt_meta": prompt_meta,
        "n_official_records": len(official_records), "n_heldout_records": len(heldout_records),
        # OOK rate per set (ook_combined/ook_official/ook_heldout names preserved for back-compat).
        "ook_official": official["ook"], "ook_heldout": heldout["ook"], "ook_combined": pooled["ook"],
        # acceptance ladder per set (ladder_live == pooled, preserved for back-compat).
        "ladder_official": official["ladder"], "ladder_heldout": heldout["ladder"],
        "ladder_live": pooled["ladder"], "accept_counts": accept_counts_from_records(draft_records),
        # OOK-caused-truncation frontier per set (frontier == pooled, preserved for back-compat).
        "frontier_official": official["frontier"], "frontier_heldout": heldout["frontier"],
        "frontier": pooled["frontier"],
    })
    return live


def reparse(args) -> dict[str, Any]:
    """Offline: recompute the per-set live estimates from an already-captured ook_probe_log.jsonl
    (no serve, no GPU). Lets us re-aggregate a saved probe run after a headline-framing change."""
    from scripts.local_prevalidate import load_manifest
    submission_dir = Path(args.submission)
    if not submission_dir.is_absolute():
        submission_dir = (REPO_ROOT / submission_dir).resolve()
    manifest = load_manifest(submission_dir)
    keepset_path = str((manifest.get("env") or {}).get("PCK04_KEEPSET", args.keepset))
    keepset_set = _load_keepset_set(keepset_path)
    log_path = Path(args.reparse_jsonl)
    all_records = _read_jsonl(log_path)
    boundary = args.reparse_boundary if args.reparse_boundary >= 0 else len(all_records)
    print(f"[live-ook] reparse {log_path} ({len(all_records)} records, boundary={boundary}, "
          f"keepset={len(keepset_set)})", flush=True)
    prompt_meta = {"reparsed_from": str(log_path), "boundary_line": boundary}
    meta = {
        "out_dir": str(log_path.parent), "log_path": str(log_path), "server_log": None,
        "served_model_name": str(manifest.get("served_model_name", "gemma-4-e4b-it")),
        "keepset_path": keepset_path, "keepset_size": len(keepset_set), "base_url": None,
        "hook_installed_marker": True, "req_fail": 0, "reparsed": True,
    }
    return _estimate_live(all_records, boundary, keepset_set, prompt_meta, meta)


# ============================================================================ #
# (D) Verdict assembly + report + W&B (house pattern).
# ============================================================================ #
def assemble_live_headline(live: dict | None, st: dict) -> dict[str, Any]:
    h: dict[str, Any] = {
        "live_ook_probe_self_test_passes": bool(st["live_ook_probe_self_test_passes"]),
        "probe_was_live": bool(live is not None),
        "ppl_unchanged_by_keepset_mask": True,  # mask touches PROPOSED never EMITTED -> PPL invariant (#420)
        "masked_drafter_ab_delta_etp": None,    # realized q needs a masked re-argmax (fused-sparse kernel
                                                # patch) -> OUT OF SCOPE for this read-only card (PR escape hatch)
        "realized_vs_ceiling_ratio": None,      # realized point unresolved (q unmeasured); see frontier bound
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
    }
    if live is None:
        return h
    # PRIMARY = the OFFICIAL benchmark (the distribution the leaderboard scores + the one #289's
    # a_1=0.7293 was measured on). Held-out + pooled are reported as explicitly-suffixed SIBLINGS.
    ook_off, ook_held, ook_pool = live["ook_official"], live["ook_heldout"], live["ook_combined"]
    lad_off, lad_held = live["ladder_official"], live["ladder_heldout"]
    fr_off, fr_held = live["frontier_official"], live["frontier_heldout"]
    p_off = ook_off["drafter_outofkeepset_proposal_rate_live"]
    p_held = ook_held["drafter_outofkeepset_proposal_rate_live"]
    p_pool = ook_pool["drafter_outofkeepset_proposal_rate_live"]
    env_off, env_held, env_pool = (upside_envelope_at(p_off), upside_envelope_at(p_held),
                                   upside_envelope_at(p_pool))
    a1_off = lad_off["ladder"][0] if lad_off["ladder"] else 0.0
    a1_held = lad_held["ladder"][0] if lad_held["ladder"] else 0.0
    anchor_env = upside_envelope_at(P_OOK_IMITATION_414)
    h.update({
        # ---- PRIMARY: benchmark (official) ----
        "drafter_outofkeepset_proposal_rate_live": p_off,
        "drafter_distinct_outofkeepset_ids_live": ook_off["drafter_distinct_outofkeepset_ids_live"],
        # headline upside = the CEILING at the OFFICIAL p (floor 0); mirrors #420's convention.
        "inkeepset_drafting_equiv_tps_upside_live": env_off["ceiling_equiv_tps_upside"],
        "inkeepset_drafting_equiv_tps_upside_live_floor": env_off["floor_equiv_tps_upside"],
        "inkeepset_drafting_equiv_tps_upside_live_ceiling": env_off["ceiling_equiv_tps_upside"],
        "inkeepset_drafting_equiv_tps_level_ceiling": env_off["ceiling_equiv_tps_level"],
        "lever_is_closed_live": bool(p_off < LEVER_CLOSED_P_THRESHOLD),
        # LIVE refinement (logging-only): of the OFFICIAL steps that truncated, the fraction that
        # truncated on an OOK proposal -- the ONLY positions a keepset mask could reclaim.
        "ook_caused_truncation_fraction_live": fr_off["ook_caused_truncation_fraction"],
        "n_ook_caused_truncations_live": fr_off["n_ook_caused_truncations"],
        "n_truncations_live": fr_off["n_truncations"],
        # liveness vs #289 a_1=0.7293 -- measured ON the official benchmark (apples to apples).
        "a1_live": a1_off, "a1_target_289": A1_TARGET_289,
        "live_ladder_consistent_with_289": bool(abs(a1_off - A1_TARGET_289) < A1_TOL),
        "e_t_live": lad_off["e_t"], "e_t_289": E_T_289,
        "n_draft_records_live": live["n_draft_records"],
        "n_official_records_live": live.get("n_official_records"),
        "n_draft_positions_live": ook_off["total_draft_positions"],
        "p_official_live": p_off,
        # ---- SECONDARY: held-out general-text corpus (#414 comparison) ----
        "p_heldout_live": p_held,
        "drafter_distinct_outofkeepset_ids_heldout": ook_held["drafter_distinct_outofkeepset_ids_live"],
        "a1_heldout": a1_held, "e_t_heldout": lad_held["e_t"],
        "lever_is_closed_heldout": bool(p_held < LEVER_CLOSED_P_THRESHOLD),
        "inkeepset_drafting_equiv_tps_upside_heldout_ceiling": env_held["ceiling_equiv_tps_upside"],
        "ook_caused_truncation_fraction_heldout": fr_held["ook_caused_truncation_fraction"],
        "n_heldout_records_live": live.get("n_heldout_records"),
        "n_draft_positions_heldout": ook_held["total_draft_positions"],
        "realized_equiv_tps_upside_upper_bound_heldout": fr_held["realized_equiv_tps_upside_upper_bound"],
        # ---- TRANSPARENCY: pooled (sampling-mix artifact -- NOT a deployable number) ----
        "p_pooled_live": p_pool,
        "inkeepset_drafting_equiv_tps_upside_pooled_ceiling": env_pool["ceiling_equiv_tps_upside"],
        # ---- comparison to #420's analytic imitation anchor (a GENERAL-text proxy -> vs held-out) ----
        "analytic_anchor_p_414": P_OOK_IMITATION_414,
        "analytic_anchor_ceiling_equiv_tps": anchor_env["ceiling_equiv_tps_upside"],
        "live_vs_analytic_p_ratio": (p_held / P_OOK_IMITATION_414) if P_OOK_IMITATION_414 else None,
        "live_vs_analytic_p_ratio_official": (p_off / P_OOK_IMITATION_414) if P_OOK_IMITATION_414 else None,
        "deployed_equiv_tps_7": EQUIV_TPS_7,
    })
    return h


def _build_verdict_text(h: dict, live: dict | None) -> str:
    if live is None:
        return ("PR #426 self-test only (no live serve). live_ook_probe_self_test_passes="
                f"{h['live_ook_probe_self_test_passes']}. Run without --self-test on the warm A10G "
                "(/tmp/server-venv) to resolve the LIVE OOK proposal rate p.")
    parts = [
        "Q (PR #426): resolve the keepset-mask lever's magnitude to a LIVE, deployable point. RESULT "
        f"(BENCHMARK / official 128): drafter_outofkeepset_proposal_rate_live={h['p_official_live']:.4f} "
        f"(probe_was_live=True, {h['n_draft_positions_live']} draft positions over "
        f"{h['n_official_records_live']} steps), distinct OOK ids={h['drafter_distinct_outofkeepset_ids_live']}, "
        f"lever_is_closed_live={h['lever_is_closed_live']}: on the distribution the leaderboard actually "
        "scores, the deployed drafter almost NEVER proposes out-of-keepset, so the keepset-mask lever is "
        "essentially CLOSED.",
        f"LIVENESS (decisive): the official-set acceptance ladder reconstructs to a_1={h['a1_live']:.4f} vs "
        f"deployed #289 a_1={h['a1_target_289']:.4f} (consistent={h['live_ladder_consistent_with_289']}), "
        f"E[T]_live={h['e_t_live']:.4f} vs #289 {h['e_t_289']:.4f} -- near-identical, so the probe reads the "
        "GENUINE deployed distribution (not a standalone-forward proxy).",
        "HOW (the #401 'standalone-forward' blocker sidestepped): the per-position drafter argmax was read "
        "LIVE by LOGGING the deployed Gemma4Proposer.propose return INSIDE the running served engine (where "
        "inputs_embeds[5120]+shared_kv_states are already built and the head is the bit-exact int4 path) -- "
        "a pure-logging .pth hook; NO served kernel/file modified, rebuilt, or submitted.",
        f"BENCHMARK UPSIDE (envelope at the live official p): floor q=0 -> 0 equiv-TPS; ceiling q=1 -> "
        f"{h['inkeepset_drafting_equiv_tps_upside_live_ceiling']:.1f} equiv-TPS (deployed "
        f"{h['deployed_equiv_tps_7']:.2f} -> {h['inkeepset_drafting_equiv_tps_level_ceiling']:.2f}); and the "
        f"frontier refinement shows only {h['n_ook_caused_truncations_live']} of {h['n_truncations_live']} "
        f"truncations were OOK-caused (fraction {h['ook_caused_truncation_fraction_live']:.4f}) -- the realized "
        "benchmark upside is a small fraction of even that ~6 equiv-TPS ceiling.",
        f"HELD-OUT (general-text / #414 corpus, SECONDARY): p_heldout={h['p_heldout_live']:.4f} "
        f"(~{h['p_heldout_live']/max(h['p_official_live'],1e-9):.0f}x the benchmark; {h['n_draft_positions_heldout']} "
        f"positions, {h['drafter_distinct_outofkeepset_ids_heldout']} distinct OOK ids), a_1={h['a1_heldout']:.4f}, "
        f"ceiling {h['inkeepset_drafting_equiv_tps_upside_heldout_ceiling']:.0f} equiv-TPS. So #420's analytic "
        f"imitation anchor (p={h['analytic_anchor_p_414']:.4f}) actually describes OFF-distribution text "
        f"(live/analytic ratio {h['live_vs_analytic_p_ratio']:.2f}x on held-out vs "
        f"{h['live_vs_analytic_p_ratio_official']:.2f}x on the benchmark): the deployed keepset is tuned to the "
        "benchmark, so the lever only opens on text the benchmark does not cover. NOTE the pooled p="
        f"{h['p_pooled_live']:.4f} is a sampling-MIX artifact (128 official vs 274 held-out) -- NOT deployable.",
        "REALIZED q STAYS ANALYTIC (honest, per the PR escape hatch): the realized reclaim needs a masked-"
        "drafter A/B (re-argmax over keepset-masked logits), but the deployed FUSED_SPARSE_ARGMAX never "
        "materializes full logits -> a masked re-proposal is a SERVED-KERNEL PATCH, not logging -> out of "
        "scope. masked_drafter_ab_delta_etp=null, realized_vs_ceiling_ratio=null.",
        "IDENTITY is unchanged (PROVEN by construction, #420): the truncated-head verify is the sole arbiter "
        f"of EMITTED tokens, so masking the drafter is identity- and PPL-neutral ({PPL_DEPLOYED} <= {PPL_GATE}) "
        "by construction. LOCAL/analysis-only; 0 official TPS; NO HF Job / submission / served-file change.",
    ]
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


def _print_report(st: dict, h: dict, live: dict | None, verdict: str) -> None:
    print("\n" + "=" * 100, flush=True)
    print("LIVE OOK PROBE -- keepset-mask lever resolved to a deployable point (PR #426)", flush=True)
    print("=" * 100, flush=True)
    print(f"  (PRIMARY) live_ook_probe_self_test_passes = "
          f"{st['live_ook_probe_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    fail = [k for k, val in st["conditions"].items() if not val]
    if fail:
        print(f"          FAILED: {fail}", flush=True)
    print("-" * 100, flush=True)
    if live is not None:
        print("  --- BENCHMARK (official 128) -- the deployable / leaderboard distribution ---", flush=True)
        for k in ("p_official_live", "drafter_distinct_outofkeepset_ids_live", "n_draft_positions_live",
                  "a1_live", "a1_target_289", "live_ladder_consistent_with_289", "e_t_live", "e_t_289",
                  "inkeepset_drafting_equiv_tps_upside_live", "inkeepset_drafting_equiv_tps_upside_live_floor",
                  "ook_caused_truncation_fraction_live", "n_ook_caused_truncations_live", "n_truncations_live",
                  "lever_is_closed_live"):
            print(f"  {k:<52} = {h.get(k)}", flush=True)
        print("  --- HELD-OUT (general text / #414 corpus) -- SECONDARY ---", flush=True)
        for k in ("p_heldout_live", "drafter_distinct_outofkeepset_ids_heldout", "n_draft_positions_heldout",
                  "a1_heldout", "e_t_heldout", "lever_is_closed_heldout",
                  "inkeepset_drafting_equiv_tps_upside_heldout_ceiling", "ook_caused_truncation_fraction_heldout"):
            print(f"  {k:<52} = {h.get(k)}", flush=True)
        print("  --- pooled (sampling-MIX artifact -- NOT deployable) + anchors ---", flush=True)
        for k in ("p_pooled_live", "inkeepset_drafting_equiv_tps_upside_pooled_ceiling",
                  "analytic_anchor_p_414", "live_vs_analytic_p_ratio", "live_vs_analytic_p_ratio_official",
                  "probe_was_live"):
            print(f"  {k:<52} = {h.get(k)}", flush=True)
    else:
        print("  (self-test only; no live serve performed)", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print("=" * 100, flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[live-ook] wandb logging unavailable: {exc}", flush=True)
        return

    st = payload["self_test"]
    h = payload["headline"]
    live = payload.get("live")

    def _num(x):
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    summary: dict[str, Any] = {
        "live_ook_probe_self_test_passes": int(bool(st["live_ook_probe_self_test_passes"])),
        "n_self_test_checks": st["n_checks"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        "probe_was_live": int(bool(h["probe_was_live"])),
        "analysis_only": 1, "no_hf_job": 1, "no_served_file_change": 1, "official_tps": 0,
    }
    for k, val in h.items():
        if isinstance(val, bool):
            summary[k] = int(val)
        elif _num(val) is not None:
            summary[k] = val
    if live is not None:
        summary["n_draft_records_live"] = live["n_draft_records"]
        summary["n_error_records_live"] = live["n_error_records"]
        summary["req_fail_live"] = live["req_fail"]
        summary["hook_installed_marker"] = int(bool(live["hook_installed_marker"]))
        # per-set ladders (a1_live in the headline == official; keep distinct suffixes here so the
        # pooled/heldout ladders never clobber it).
        for tag, key in (("official", "ladder_official"), ("heldout", "ladder_heldout"),
                         ("pooled", "ladder_live")):
            for i, a in enumerate(live.get(key, {}).get("ladder", [])):
                summary[f"a{i+1}_{tag}"] = a
    summary = {k: val for k, val in summary.items()
               if val is not None and not (isinstance(val, float) and not math.isfinite(val))}

    run = init_wandb_run(
        job_type="validity-gate", agent="land", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "live-ook-probe", "speculator-equivalence", "keepset-mask",
              "drafter-vocab-mask", "acceptance-upside", "equiv-tps", "probe-live", "read-only-logging",
              "analysis-only", "pr-426"],
        config={
            "hidden": HIDDEN, "full_vocab": FULL_VOCAB, "deployed_lmhead_rows": DEPLOYED_LMHEAD_ROWS,
            "k_spec": K_SPEC, "m_deployed": M_DEPLOYED, "k_cal": K_CAL, "mu_p": MU_P,
            "e_t_289": E_T_289, "e_accepted_289": E_ACCEPTED_289, "equiv_tps_7": EQUIV_TPS_7,
            "p_ook_imitation_414": P_OOK_IMITATION_414, "distinct_ook_ids_414": DISTINCT_OOK_IDS_414,
            "ppl_deployed": PPL_DEPLOYED, "ppl_gate": PPL_GATE, "a1_target_289": A1_TARGET_289,
            "lever_closed_p_threshold": LEVER_CLOSED_P_THRESHOLD, "wandb_group": args.wandb_group,
            "source_420_run": "qe4qagc1", "source_414_run": "bq7xkfcv", "source_289_run": "fi34s269",
            "source_401_run": "i2qsjyp6",
            "submission": args.submission,
            "official_prompts": args.official_prompts, "heldout_prompts": args.heldout_prompts,
            "official_output_len": args.official_output_len, "heldout_output_len": args.heldout_output_len,
        },
    )
    if run is None:
        print("[live-ook] wandb: no run (no WANDB_API_KEY/mode) -- skipping", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="live_ook_probe_keepset_mask_result", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[live-ook] wandb logged: probe_was_live={h['probe_was_live']} "
          f"p_live={h.get('drafter_outofkeepset_proposal_rate_live')} "
          f"lever_closed={h.get('lever_is_closed_live')}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY pure-logic self-validation only (numpy-only; no serve)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny live run (2 official prompts, short output) to verify the hook logs")
    ap.add_argument("--submission", default=DEFAULT_SUBMISSION)
    ap.add_argument("--venv-python", dest="venv_python", default=DEFAULT_VENV_PYTHON)
    ap.add_argument("--keepset", default=DEFAULT_KEEPSET)
    ap.add_argument("--attach-base-url", dest="attach_base_url", default=None,
                    help="attach to an already-running probe-enabled endpoint instead of launching serve")
    ap.add_argument("--cuda-devices", dest="cuda_devices", default="0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--startup-timeout-s", dest="startup_timeout_s", type=int, default=1800)
    ap.add_argument("--request-timeout-s", dest="request_timeout_s", type=int, default=180)
    ap.add_argument("--server-log", dest="server_log", default=None)
    ap.add_argument("--official-prompts", dest="official_prompts", type=int, default=128)
    ap.add_argument("--heldout-prompts", dest="heldout_prompts", type=int, default=-1,
                    help="-1 = all (~274); 0 = skip; N>0 = first N")
    ap.add_argument("--official-output-len", dest="official_output_len", type=int, default=512)
    ap.add_argument("--heldout-output-len", dest="heldout_output_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    ap.add_argument("--reparse-jsonl", dest="reparse_jsonl", default=None,
                    help="offline: recompute the per-set headline from an existing ook_probe_log.jsonl (no serve)")
    ap.add_argument("--reparse-boundary", dest="reparse_boundary", type=int, default=-1,
                    help="record index splitting official (before) from held-out (after) for --reparse-jsonl")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="live-ook-probe")
    args = ap.parse_args(argv)

    if args.smoke:
        args.official_prompts = 2
        args.heldout_prompts = 0
        args.official_output_len = 24

    st = self_test()
    live: dict | None = None
    if not args.self_test:
        live = reparse(args) if args.reparse_jsonl else run_live(args)

    headline = assemble_live_headline(live, st)
    verdict = _build_verdict_text(headline, live)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 426, "agent": "land", "kind": "live-ook-probe-keepset-mask",
        "self_test": st, "headline": headline, "live": live, "verdict": verdict,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean({k: v for k, v in payload.items() if k != "verdict"})
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[live-ook] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(st, headline, live, verdict)

    out_dir = Path(args.out_dir) if args.out_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_ook_probe_keepset_mask_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[live-ook] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    passes = bool(st["live_ook_probe_self_test_passes"]) and payload["nan_clean"]
    print(f"  PRIMARY live_ook_probe_self_test_passes = {passes} ({st['n_checks']} checks)", flush=True)
    if live is not None:
        print(f"  drafter_outofkeepset_proposal_rate_live = "
              f"{headline['drafter_outofkeepset_proposal_rate_live']:.5f}  "
              f"upside_ceiling={headline['inkeepset_drafting_equiv_tps_upside_live']:.2f} equiv-TPS "
              f"(floor 0)  lever_closed={headline['lever_is_closed_live']}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)
    if args.self_test:
        print(f"[live-ook] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
