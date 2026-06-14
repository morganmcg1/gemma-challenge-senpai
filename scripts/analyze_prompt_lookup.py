#!/usr/bin/env python3
"""Offline prompt-lookup / n-gram free-token measurement for PR #81 (Step-1 gate).

Prompt-lookup decoding (PLD) proposes draft tokens FOR FREE -- zero model forward
-- whenever the current generation suffix (last ``n`` tokens) recurs earlier in
``prompt + generated-so-far``: it copies the continuation of that earlier match
and lets the verifier accept the longest greedy-correct prefix. This is the
``ngram`` speculative proposer in vLLM / HF ``prompt_lookup_num_tokens``.

This script is the OFFLINE, FAIL-FAST viability gate for PR #81. It runs no model
and launches no HF Job; it reads the deployed greedy decode trace (the validity
contract makes that trace token-identical across all valid submissions) and
measures, per n-gram size ``n in {2,3,4}``:

* HIT-RATE -- fraction of decode positions where the length-``n`` suffix has an
  earlier occurrence, split PROMPT-region vs GENERATED-region (the latter is the
  real signal for reasoning outputs, which are generated, not copied).
* FREE-ACCEPT-LENGTH | HIT -- of the tokens following the most-recent earlier
  match, how many equal the actual greedy continuation (= what the verifier would
  accept), capped at ``--max-draft``.

It then settles the decision-relevant question -- the value of PLD as an AUGMENT
on TOP of the deployed trained MTP drafter (E[T] = 3.844 tok/step, K=7), not as a
replacement -- two ways:

* REPLACE head-to-head (context): PLD-only E[T] (renewal walk advancing by the
  realized PLD accept). Expected to lose badly to MTP-only; reported for the
  Step-2 "augment vs replace" verdict the advisor asked for.
* AUGMENT (the gate): a renewal Monte-Carlo simulation of MTP + PLD tree-verify.
  At each step from position ``pos`` the MTP chain accepts ``m`` draft tokens
  (drawn from the MEASURED per-position conditional acceptance) and PLD proposes
  ``q = q_pld[pos]`` greedy-correct tokens; the run extends to ``max(m, q)`` (one
  verifier forward, composes with land #71 tree-verify), so the step yields
  ``max(0, q - m)`` EXTRA accepted tokens. Aggregated over the renewal this gives
  the augment E[T'] and TPS uplift. The "rescue an MTP rejection" cross-cut --
  steps where ``q > m``, bucketed by ``m`` (``m = 0`` = full MTP miss rescued) --
  is reported because free tokens that rescue an MTP rejection are worth far more
  than free tokens MTP already accepts.

LIMITATION (stated honestly): no per-position MTP accept/reject trace exists (the
PR #13 drafter-overlap was always a template awaiting a real trace), so ``m`` is
drawn from the measured marginal acceptance distribution INDEPENDENTLY of
``q_pld[pos]``. If PLD hits correlate POSITIVELY with MTP success (both fire on
predictable/repetitive spans) the augment value is OVER-estimated here; if they
anti-correlate it is UNDER-estimated. The replace-mode PLD-only realized free
fraction is reported as the assumption-free upper bracket, and the longest-suffix
SAM estimate from ``analyze_suffix_budget.py`` is the ultimate upper bound
(longest-suffix match dominates any fixed-n match).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / (
    "research/greedy_reference/"
    "workspace__senpai__target__submissions__fa2sw_precache_kenyan__google__gemma-4-E4B-it/"
    "decode_outputs.jsonl"
)
DEFAULT_ACCEPT = ROOT / "research/accept_calibration/accept_calibration_results.json"
DEFAULT_OUTPUT = ROOT / "research/local_validation/prompt_lookup/prompt_lookup_analysis.json"

NGRAMS = (2, 3, 4)
DEFAULT_MAX_DRAFT = 7  # PLD continuation length; matches deployed MTP chain K=7
BASELINE_ET = 3.844131736526946
# lawine #72 (MERGED): the robust local decode metric is wall_tps (CV 0.035%, MDE
# >=0.2% @ N=1) -- ~454. The old "428.37 steady" was a fragile estimator point.
# Augment uplift here is an E[T]-ratio (baseline-invariant); TPS_LOCAL is record-only.
BASELINE_TPS_LOCAL = 454.0  # robust wall_tps (was 428.37 fragile steady-meter)


def distribution_of(rec_id: str) -> str:
    return str(rec_id).split("-", 1)[0]


def load_records(path: Path) -> list[dict[str, Any]]:
    recs = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# ---------------------------------------------------------------------------
# Per-position PLD measurement on a single (prompt, generated) sequence.
# ---------------------------------------------------------------------------
def _accept_len(seq: list[int], e: int, ap: int, max_draft: int, total: int) -> int:
    """Greedy-correct accepted length copying seq[e:e+L] against target seq[ap:ap+L].

    PLD can only copy ALREADY-GENERATED source tokens (indices < ap), so L is capped
    by (ap - e) as well as the draft budget and the target tail. Mirrors the source-
    availability cap in analyze_suffix_budget.realized_sam_lengths."""
    cap = min(max_draft, ap - e, total - ap)
    r = 0
    while r < cap and seq[e + r] == seq[ap + r]:
        r += 1
    return r


def pld_per_position(
    prompt_ids: list[int],
    generated_ids: list[int],
    n: int,
    max_draft: int,
) -> dict[str, list[int]]:
    """Per generated position t, the fixed-n PLD result under two occurrence picks.

    seq = prompt_ids + generated_ids; ap = P + t is the absolute index of the next
    token to emit at step t; the length-n suffix is seq[ap-n:ap]. A rolling map
    ngram -> sorted list of END indices e (occurrence seq[e-n:e]) lets us pick:
      * earliest occurrence  -- what vLLM v1 ngram_proposer actually copies from
        (longest copy room ap-e, but the continuation context differed most);
      * oracle-best occurrence -- the earlier occurrence maximizing accepted length
        (gives PLD its best shot; an upper bound on realized accept for this n).

    Returns dict of per-position lists:
      q_earliest, q_oracle  -- realized free-accept length (0 on miss), <= max_draft
      region_earliest       -- -1 miss, 0 earliest match ends in prompt, 1 in generated
      has_prompt, has_gen   -- 1 if any earlier occurrence ends in prompt / generated
    """
    seq = list(prompt_ids) + list(generated_ids)
    P = len(prompt_ids)
    N = len(generated_ids)
    total = len(seq)
    occ: dict[tuple[int, ...], list[int]] = defaultdict(list)
    q_earliest = [0] * N
    q_oracle = [0] * N
    region_earliest = [-1] * N
    has_prompt = [0] * N
    has_gen = [0] * N

    # seed with prompt-internal windows ending at e in [n .. P-1] (ascending => sorted)
    for e in range(n, P):
        occ[tuple(seq[e - n : e])].append(e)

    for t in range(N):
        ap = P + t
        self_key = tuple(seq[ap - n : ap]) if ap - n >= 0 else None
        if self_key is not None:
            ends = occ.get(self_key)
            if ends:  # all entries are < ap (ap appended only after)
                e_first = ends[0]
                q_earliest[t] = _accept_len(seq, e_first, ap, max_draft, total)
                region_earliest[t] = 0 if e_first <= P else 1
                has_prompt[t] = 1 if e_first <= P else 0
                has_gen[t] = 1 if ends[-1] > P else 0
                # oracle: scan occurrences most-recent-first, early-stop at max_draft
                best = 0
                for e in reversed(ends):
                    r = _accept_len(seq, e, ap, max_draft, total)
                    if r > best:
                        best = r
                        if best >= max_draft:
                            break
                q_oracle[t] = best
            occ[self_key].append(ap)

    return {
        "q_earliest": q_earliest,
        "q_oracle": q_oracle,
        "region_earliest": region_earliest,
        "has_prompt": has_prompt,
        "has_gen": has_gen,
    }


def combined_best_n_q(
    per_n: dict[int, dict[str, list[int]]],
    ngrams: tuple[int, ...],
    N: int,
    field: str,
) -> list[int]:
    """Realistic vLLM PLD prefers the LARGEST n with a match (longest context =
    highest precision). q_combined[t] = q from the largest n that has a hit."""
    out = [0] * N
    order = sorted(ngrams, reverse=True)
    for t in range(N):
        for n in order:
            if per_n[n][field][t] > 0:
                out[t] = per_n[n][field][t]
                break
    return out


# ---------------------------------------------------------------------------
# Renewal Monte-Carlo: MTP-only, PLD-only (replace), and MTP+PLD (augment).
# ---------------------------------------------------------------------------
def draw_m(cond_p: list[float], rng: random.Random) -> int:
    """Accepted MTP draft-token count m in [0, K] from measured conditional accept."""
    m = 0
    for pj in cond_p:
        if rng.random() < pj:
            m += 1
        else:
            break
    return m


def simulate_corr(
    q_by_prompt: list[list[int]],
    cond_p: list[float],
    frac_hit: float,
    a_H: float,
    trials: int,
    seed: int,
    baseline_et: float,
    full_span: bool = False,
) -> dict[str, float]:
    """Correlation-aware augment sim under the conservation constraint.

    The measured aggregate MTP top-1 acceptance is cond_p[0]; we redistribute it
    between PLD-HIT positions (top-1 = a_H) and PLD-MISS positions (top-1 = a_M)
    so the frac_hit-weighted average is preserved:
        frac_hit * a_H + (1 - frac_hit) * a_M = cond_p[0].
    a_H = cond_p[0] is the independence point; a_H -> 1.0 is maximal positive
    correlation. Only HIT positions can contribute augment extra (q = 0 elsewhere),
    so this isolates the decision-critical sensitivity to the UNMEASURED q/m
    correlation.

    Two correlation models bracket reality:
      * top1-only (full_span=False): lift only the first-token acceptance to a_H,
        hold the deeper conditionals cond_p[1:] fixed. CONSERVATIVE -- it lets MTP
        still drop the chain at depth>=2 on a hit, so PLD can extend partial chains.
        An UPPER bracket on the augment at a given a_H.
      * full-span (full_span=True): a repetitive span MTP predicts is predictable at
        EVERY depth, so lift the WHOLE conditional chain by the same factor
        boost = (a_H - p0)/(1 - p0): cond_hit[j] = p[j] + (1-p[j])*boost. At a_H=1.0
        the whole chain -> 1 (m=K>=q), augment -> ~0. The LOWER bracket, matching the
        literature floor (SAM-Decoding ~-0.05x on math reasoning over EAGLE-2).
    """
    top1 = cond_p[0]
    a_M = (top1 - frac_hit * a_H) / (1.0 - frac_hit) if frac_hit < 1.0 else top1
    if full_span:
        boost = (a_H - top1) / (1.0 - top1) if top1 < 1.0 else 0.0
        chain_hit = [pj + (1.0 - pj) * boost for pj in cond_p]
    else:
        chain_hit = [a_H] + list(cond_p[1:])
    chain_miss = [max(0.0, a_M)] + list(cond_p[1:])
    rng = random.Random(seed)
    tot_tokens = sum(len(q) for q in q_by_prompt)
    steps_aug = steps_mtp = 0
    extra_tokens = 0
    for _ in range(trials):
        for q_pld in q_by_prompt:
            N = len(q_pld)
            # MTP-only baseline under the SAME position-dependent chains, so the uplift
            # is self-consistent. (top1-only preserves E[T]~=baseline; full-span raises
            # the modelled MTP-only E[T], correctly shrinking the augment headroom.)
            pos = 0
            while pos < N:
                m = draw_m(chain_hit if q_pld[pos] > 0 else chain_miss, rng)
                pos += m + 1
                steps_mtp += 1
            pos = 0
            while pos < N:
                q = q_pld[pos]
                m = draw_m(chain_hit if q > 0 else chain_miss, rng)
                if q > m:
                    extra_tokens += q - m
                pos += max(m, q) + 1
                steps_aug += 1
    et_aug = tot_tokens * trials / steps_aug
    et_mtp = tot_tokens * trials / steps_mtp
    return {
        "a_H": a_H,
        "a_M": a_M,
        "extra_tokens_per_step": extra_tokens / steps_aug,
        "ET_augment": et_aug,
        "ET_mtp_only_model": et_mtp,
        "tps_uplift_pct": (et_aug / et_mtp - 1.0) * 100.0,
    }


def simulate(
    q_pld_by_prompt: list[list[int]],
    cond_p: list[float],
    trials: int,
    seed: int = 1,
) -> dict[str, Any]:
    """Renewal simulation over all prompts; returns step counts and augment stats.

    Each prompt has fixed total length N; the three policies differ only in how far
    each step advances:
      MTP-only : advance = m + 1
      PLD-only : advance = q + 1            (PLD replaces MTP)
      augment  : advance = max(m, q) + 1    (MTP + PLD tree-verify)
    All emit exactly N greedy-correct tokens, so E[T] = N / steps.

    The MTP-only and augment walks use COMMON RANDOM NUMBERS (two RNGs seeded
    identically): the augment step at a given renewal position draws the same
    ``m`` the MTP-only walk would. Where ``q == 0`` the two walks then advance
    identically, so ``ET_augment == ET_mtp`` exactly (no spurious MC gap), and the
    headline difference ``delta_ET_augment`` is estimated with CRN variance
    reduction. They diverge only at positions where PLD actually adds tokens
    (``q > m``), which is exactly the effect being measured.
    """
    rng_mtp = random.Random(seed)
    rng_aug = random.Random(seed)
    K = len(cond_p)
    tot_tokens = sum(len(q) for q in q_pld_by_prompt)

    steps_mtp = steps_pld = steps_aug = 0
    extra_tokens = 0  # sum over augment steps of max(0, q-m)
    rescue_steps = 0  # augment steps with q > m
    rescue_from_m0 = 0  # rescue steps where MTP fully missed (m == 0)
    extra_from_m0 = 0  # extra tokens contributed when m == 0
    aug_steps_total = 0

    for _ in range(trials):
        for q_pld in q_pld_by_prompt:
            N = len(q_pld)
            # MTP-only
            pos = 0
            while pos < N:
                m = draw_m(cond_p[:K], rng_mtp)
                pos += m + 1
                steps_mtp += 1
            # PLD-only (replace): advance by realized PLD accept (q), +1 bonus
            pos = 0
            while pos < N:
                q = q_pld[pos] if pos < N else 0
                pos += q + 1
                steps_pld += 1
            # augment: MTP + PLD, run reaches max(m, q)
            pos = 0
            while pos < N:
                m = draw_m(cond_p[:K], rng_aug)
                q = q_pld[pos]
                ex = q - m
                if ex > 0:
                    extra_tokens += ex
                    rescue_steps += 1
                    if m == 0:
                        rescue_from_m0 += 1
                        extra_from_m0 += ex
                pos += max(m, q) + 1
                steps_aug += 1
                aug_steps_total += 1

    et_mtp = tot_tokens * trials / steps_mtp
    et_pld = tot_tokens * trials / steps_pld
    et_aug = tot_tokens * trials / steps_aug
    return {
        "trials": trials,
        "tot_tokens_per_trial": tot_tokens,
        "ET_mtp_only_sim": et_mtp,
        "ET_pld_only_sim": et_pld,
        "ET_augment_sim": et_aug,
        "delta_ET_augment": et_aug - et_mtp,
        "extra_tokens_per_step": extra_tokens / aug_steps_total,
        "tps_uplift_augment": et_aug / et_mtp,
        "rescue_step_frac": rescue_steps / aug_steps_total,
        "rescue_from_m0_step_frac": rescue_from_m0 / aug_steps_total,
        "share_extra_from_m0": (extra_from_m0 / extra_tokens) if extra_tokens else 0.0,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def compute(
    records: list[dict[str, Any]],
    accept_path: Path,
    ngrams: tuple[int, ...],
    max_draft: int,
    trials: int,
) -> dict[str, Any]:
    accept = json.loads(accept_path.read_text())
    cond_p = accept["headline"]["conditional_acceptance_p"]
    et_measured = accept["headline"]["deployed_chain_mean_tokens_per_step"]

    # position buckets (early/mid/late within the 512-token completion)
    BUCKETS = [(0, 64), (64, 128), (128, 256), (256, 512)]

    def bucket_of(t: int) -> str:
        for lo, hi in BUCKETS:
            if lo <= t < hi:
                return f"{lo}-{hi}"
        return f"{BUCKETS[-1][0]}-{BUCKETS[-1][1]}"

    total_pos = 0
    # per-n accumulators (hit-rate from vLLM-faithful EARLIEST pick)
    hits = {n: 0 for n in ngrams}
    hits_prompt = {n: 0 for n in ngrams}
    hits_generated = {n: 0 for n in ngrams}
    any_gen = {n: 0 for n in ngrams}  # positions with ANY generated-region occurrence
    q_sum_e = {n: 0 for n in ngrams}  # earliest q over ALL positions
    q_sum_o = {n: 0 for n in ngrams}  # oracle q over ALL positions
    q_sum_hit_e = {n: 0 for n in ngrams}  # earliest q over hits
    q_sum_hit_o = {n: 0 for n in ngrams}  # oracle q over hits
    qhist = {n: Counter() for n in ngrams}  # oracle q | hit distribution
    bucket_pos: Counter = Counter()
    bucket_hits = {n: Counter() for n in ngrams}
    dist_pos: Counter = Counter()
    dist_hits = {n: Counter() for n in ngrams}
    dist_qsum_o = {n: Counter() for n in ngrams}

    q_comb_earliest: list[list[int]] = []
    q_comb_oracle: list[list[int]] = []

    for rec in records:
        prompt_ids = rec["prompt_token_ids"]
        gen = rec.get("completion_token_ids") or rec.get("token_ids")
        dist = distribution_of(rec.get("id", ""))
        N = len(gen)
        total_pos += N
        for t in range(N):
            bucket_pos[bucket_of(t)] += 1
            dist_pos[dist] += 1
        per_n_rec: dict[int, dict[str, list[int]]] = {}
        for n in ngrams:
            res = pld_per_position(prompt_ids, gen, n, max_draft)
            per_n_rec[n] = res
            region = res["region_earliest"]
            qe = res["q_earliest"]
            qo = res["q_oracle"]
            for t in range(N):
                if region[t] >= 0:
                    hits[n] += 1
                    if region[t] == 0:
                        hits_prompt[n] += 1
                    else:
                        hits_generated[n] += 1
                    q_sum_hit_e[n] += qe[t]
                    q_sum_hit_o[n] += qo[t]
                    qhist[n][qo[t]] += 1
                    bucket_hits[n][bucket_of(t)] += 1
                    dist_hits[n][dist] += 1
                if res["has_gen"][t]:
                    any_gen[n] += 1
                q_sum_e[n] += qe[t]
                q_sum_o[n] += qo[t]
                dist_qsum_o[n][dist] += qo[t]
        q_comb_earliest.append(combined_best_n_q(per_n_rec, ngrams, N, "q_earliest"))
        q_comb_oracle.append(combined_best_n_q(per_n_rec, ngrams, N, "q_oracle"))

    per_n = {}
    for n in ngrams:
        per_n[str(n)] = {
            "hit_rate": hits[n] / total_pos,
            "hit_rate_prompt_region": hits_prompt[n] / total_pos,
            "hit_rate_generated_region": hits_generated[n] / total_pos,
            "frac_any_generated_occurrence": any_gen[n] / total_pos,
            "mean_q_given_hit_earliest": (q_sum_hit_e[n] / hits[n]) if hits[n] else 0.0,
            "mean_q_given_hit_oracle": (q_sum_hit_o[n] / hits[n]) if hits[n] else 0.0,
            "mean_q_all_earliest": q_sum_e[n] / total_pos,
            "mean_q_all_oracle": q_sum_o[n] / total_pos,
            "q_given_hit_histogram_oracle": {str(k): qhist[n][k] for k in sorted(qhist[n])},
            "per_bucket_hit_rate": {
                b: (bucket_hits[n][b] / bucket_pos[b] if bucket_pos[b] else 0.0)
                for b in sorted(bucket_pos)
            },
            "per_distribution": {
                d: {
                    "hit_rate": (dist_hits[n][d] / dist_pos[d] if dist_pos[d] else 0.0),
                    "mean_q_all_oracle": (
                        dist_qsum_o[n][d] / dist_pos[d] if dist_pos[d] else 0.0
                    ),
                }
                for d in sorted(dist_pos)
            },
        }

    def comb_stats(qbp: list[list[int]]) -> dict[str, float]:
        qt = sum(sum(q) for q in qbp)
        hh = sum(sum(1 for v in q if v > 0) for q in qbp)
        return {
            "hit_rate": hh / total_pos,
            "mean_q_over_all_positions": qt / total_pos,
            "mean_q_given_hit": (qt / hh) if hh else 0.0,
        }

    combined = {
        "ngrams": list(ngrams),
        "policy": "prefer largest n with a match (vLLM ngram min..max)",
        "earliest_pick_vllm_faithful": comb_stats(q_comb_earliest),
        "oracle_pick_upper_bound": comb_stats(q_comb_oracle),
    }

    sim_e = simulate(q_comb_earliest, cond_p, trials, seed=1)
    sim_o = simulate(q_comb_oracle, cond_p, trials, seed=2)

    # Correlation-aware sweep (THE decision-critical analysis): the independence sims
    # above assume MTP top-1 acceptance on PLD-hit positions == 0.729. In reality PLD
    # fires on repetitive/predictable spans where MTP also wins, so a_H > 0.729 and the
    # augment shrinks. Sweep a_H from independence to full correlation under the
    # conservation constraint, on the OPTIMISTIC oracle q (PLD best shot).
    frac_hit_oracle = combined["oracle_pick_upper_bound"]["hit_rate"]
    a_H_grid = [round(cond_p[0], 3), 0.80, 0.85, 0.90, 0.95, 1.0]
    corr_sweep = [
        simulate_corr(q_comb_oracle, cond_p, frac_hit_oracle, a_H, trials, 10 + i, et_measured)
        for i, a_H in enumerate(a_H_grid)
    ]
    corr_sweep_fullspan = [
        simulate_corr(q_comb_oracle, cond_p, frac_hit_oracle, a_H, trials, 20 + i, et_measured,
                      full_span=True)
        for i, a_H in enumerate(a_H_grid)
    ]
    NOISE_FLOOR_PCT = 4.4  # lawine #72 +/-4.4% TPS noise floor
    # smallest a_H at which the optimistic-linear uplift falls below the noise floor
    a_H_break = next((s["a_H"] for s in corr_sweep if s["tps_uplift_pct"] < NOISE_FLOOR_PCT), None)
    a_H_break_fullspan = next(
        (s["a_H"] for s in corr_sweep_fullspan if s["tps_uplift_pct"] < NOISE_FLOOR_PCT), None
    )

    # gate on the OPTIMISTIC oracle/independence pick: if even PLD's best shot fails, robust no-go.
    extra_oracle_ub = sim_o["extra_tokens_per_step"]
    uplift_pct = (sim_o["tps_uplift_augment"] - 1.0) * 100.0
    # the REPORTED primary is the vLLM-faithful EARLIEST mechanism (still an independence UB).
    extra = sim_e["extra_tokens_per_step"]
    if extra_oracle_ub >= 0.3:
        gate = "go (independence upper bound only)"
    elif extra_oracle_ub >= 0.15:
        gate = "borderline (independence upper bound only)"
    else:
        gate = "no-go"

    return {
        "n_prompts": len(records),
        "total_positions": total_pos,
        "max_draft": max_draft,
        "baseline_ET_measured": et_measured,
        "baseline_tps_local": BASELINE_TPS_LOCAL,
        "mtp_conditional_acceptance_p": cond_p,
        "per_ngram": per_n,
        "combined_best_n": combined,
        "augment_simulation_earliest": sim_e,
        "augment_simulation_oracle": sim_o,
        "augment_correlation_sweep_oracle_top1only": corr_sweep,
        "augment_correlation_sweep_oracle_fullspan": corr_sweep_fullspan,
        "noise_floor_pct": NOISE_FLOOR_PCT,
        "a_H_break_below_noise_floor_top1only": a_H_break,
        "a_H_break_below_noise_floor_fullspan": a_H_break_fullspan,
        "primary_metric_name": "promptlookup_extra_accept_tokens_per_step",
        "primary_metric_value": extra,  # vLLM-faithful EARLIEST pick, independence UB
        "primary_metric_value_oracle_upper_bound": extra_oracle_ub,
        "augment_tps_uplift_pct": uplift_pct,
        "gate": gate,
        "gate_note": (
            f"AUGMENT (ORACLE occurrence, PLD best shot, INDEPENDENCE) extra accepted tokens/step = "
            f"{extra:.4f} (E[T] {sim_o['ET_mtp_only_sim']:.3f} -> {sim_o['ET_augment_sim']:.3f}, "
            f"+{uplift_pct:.2f}% TPS optimistic-linear) -- this is an UPPER BOUND. vLLM-faithful "
            f"EARLIEST pick (independence) extra/step = {sim_e['extra_tokens_per_step']:.4f} "
            f"(+{(sim_e['tps_uplift_augment']-1)*100:.2f}%). DECISION-CRITICAL: the correlation sweep "
            f"shows the +TPS falls below the {NOISE_FLOOR_PCT}% noise floor once MTP top-1 acceptance "
            f"on PLD-hit positions a_H >= {a_H_break} (independence a_H={cond_p[0]:.3f}). PLD fires on "
            "repetitive/predictable spans where MTP already wins (positive q/m correlation), and the "
            "literature (SAM-Decoding -0.05x on math reasoning over EAGLE-2) puts the realistic point "
            "at the high-a_H / low-gain end. PLD-only replace E[T]="
            f"{sim_o['ET_pld_only_sim']:.3f} loses to MTP {sim_o['ET_mtp_only_sim']:.3f} (augment, not "
            "replace, confirmed). Net: upside is real but unmeasured-net and not composable in stock "
            "vLLM 0.22 (ngram XOR mtp); see report."
        ),
    }


def _log_wandb(result: dict[str, Any], args: argparse.Namespace) -> None:
    """Record the Step-1 measurement to W&B (no training; one summary row + tables)."""
    import wandb  # local import: only needed when --wandb is passed

    run = wandb.init(
        name=args.wandb_name,
        group=args.wandb_group,
        job_type="offline-analysis",
        tags=["prompt-lookup", "step1-gate", "pr81", "augment", "measurement-only"],
        config={
            "ngrams": list(NGRAMS),
            "max_draft": args.max_draft,
            "trials": args.trials,
            "input_trace": args.input,
            "accept_arrays": args.accept,
            "baseline_ET_measured": result["baseline_ET_measured"],
            "baseline_tps_local": result["baseline_tps_local"],
            "noise_floor_pct": result["noise_floor_pct"],
            "n_prompts": result["n_prompts"],
            "total_positions": result["total_positions"],
        },
    )

    sim_e = result["augment_simulation_earliest"]
    sim_o = result["augment_simulation_oracle"]
    comb = result["combined_best_n"]
    summary: dict[str, float] = {
        "promptlookup_extra_accept_tokens_per_step": result["primary_metric_value"],
        "promptlookup_extra_accept_tokens_per_step_oracle_ub": result[
            "primary_metric_value_oracle_upper_bound"
        ],
        "augment_tps_uplift_pct_earliest_independence_ub": (sim_e["tps_uplift_augment"] - 1) * 100,
        "augment_tps_uplift_pct_oracle_independence_ub": (sim_o["tps_uplift_augment"] - 1) * 100,
        "combined_hit_rate_earliest": comb["earliest_pick_vllm_faithful"]["hit_rate"],
        "combined_hit_rate_oracle": comb["oracle_pick_upper_bound"]["hit_rate"],
        "ET_mtp_only_sim": sim_e["ET_mtp_only_sim"],
        "ET_augment_earliest_sim": sim_e["ET_augment_sim"],
        "ET_augment_oracle_sim": sim_o["ET_augment_sim"],
        "ET_pld_only_replace_sim": sim_o["ET_pld_only_sim"],
        "rescue_step_frac_oracle": sim_o["rescue_step_frac"],
        "rescue_from_m0_step_frac_oracle": sim_o["rescue_from_m0_step_frac"],
        "share_extra_from_m0_oracle": sim_o["share_extra_from_m0"],
        "a_H_break_below_noise_floor_top1only": result["a_H_break_below_noise_floor_top1only"] or 0.0,
        "a_H_break_below_noise_floor_fullspan": result["a_H_break_below_noise_floor_fullspan"] or 0.0,
    }
    for n in NGRAMS:
        pn = result["per_ngram"][str(n)]
        summary[f"hit_rate_n{n}"] = pn["hit_rate"]
        summary[f"hit_rate_prompt_region_n{n}"] = pn["hit_rate_prompt_region"]
        summary[f"hit_rate_generated_region_n{n}"] = pn["hit_rate_generated_region"]
        summary[f"mean_q_given_hit_oracle_n{n}"] = pn["mean_q_given_hit_oracle"]
        summary[f"mean_q_given_hit_earliest_n{n}"] = pn["mean_q_given_hit_earliest"]
    run.summary.update(summary)

    per_n_tbl = wandb.Table(
        columns=["n", "hit_rate", "hit_prompt", "hit_generated",
                 "mean_q|hit_earliest", "mean_q|hit_oracle", "mean_q_all_oracle"]
    )
    for n in NGRAMS:
        pn = result["per_ngram"][str(n)]
        per_n_tbl.add_data(n, pn["hit_rate"], pn["hit_rate_prompt_region"],
                           pn["hit_rate_generated_region"], pn["mean_q_given_hit_earliest"],
                           pn["mean_q_given_hit_oracle"], pn["mean_q_all_oracle"])

    corr_tbl = wandb.Table(
        columns=["a_H", "top1_extra_per_step", "top1_uplift_pct",
                 "fullspan_extra_per_step", "fullspan_uplift_pct"]
    )
    for s, sf in zip(
        result["augment_correlation_sweep_oracle_top1only"],
        result["augment_correlation_sweep_oracle_fullspan"],
    ):
        corr_tbl.add_data(s["a_H"], s["extra_tokens_per_step"], s["tps_uplift_pct"],
                          sf["extra_tokens_per_step"], sf["tps_uplift_pct"])

    run.log({"per_ngram_table": per_n_tbl, "correlation_sweep_table": corr_tbl,
             "gate": result["gate"]})
    print(f"wandb run: {run.url} (id={run.id})")
    run.finish()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--accept", default=str(DEFAULT_ACCEPT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--max-draft", type=int, default=DEFAULT_MAX_DRAFT)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--wandb", action="store_true", help="log the Step-1 measurement to W&B")
    ap.add_argument("--wandb_name", default="denken/prompt-lookup-step1")
    ap.add_argument("--wandb_group", default="prompt-lookup-drafter")
    args = ap.parse_args()

    records = load_records(Path(args.input))
    if not records:
        raise SystemExit(f"no records in {args.input}")
    result = compute(records, Path(args.accept), NGRAMS, args.max_draft, args.trials)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"n_prompts={result['n_prompts']} total_positions={result['total_positions']}")
    for n in NGRAMS:
        pn = result["per_ngram"][str(n)]
        print(
            f"n={n}: hit_rate={pn['hit_rate']:.4f} "
            f"(prompt={pn['hit_rate_prompt_region']:.4f} gen={pn['hit_rate_generated_region']:.4f}) "
            f"mean_q|hit oracle={pn['mean_q_given_hit_oracle']:.3f}/earliest={pn['mean_q_given_hit_earliest']:.3f} "
            f"mean_q_all oracle={pn['mean_q_all_oracle']:.4f}"
        )
    c = result["combined_best_n"]
    print(
        f"combined earliest: hit={c['earliest_pick_vllm_faithful']['hit_rate']:.4f} "
        f"mean_q_all={c['earliest_pick_vllm_faithful']['mean_q_over_all_positions']:.4f} | "
        f"oracle: hit={c['oracle_pick_upper_bound']['hit_rate']:.4f} "
        f"mean_q_all={c['oracle_pick_upper_bound']['mean_q_over_all_positions']:.4f}"
    )
    for tag, key in (("EARLIEST", "augment_simulation_earliest"), ("ORACLE", "augment_simulation_oracle")):
        s = result[key]
        print(
            f"SIM[{tag}]: E[T] mtp={s['ET_mtp_only_sim']:.3f} pld_only={s['ET_pld_only_sim']:.3f} "
            f"augment={s['ET_augment_sim']:.3f} extra/step={s['extra_tokens_per_step']:.4f} "
            f"uplift={(s['tps_uplift_augment']-1)*100:+.2f}% rescue_frac={s['rescue_step_frac']:.4f} "
            f"(from_m0={s['rescue_from_m0_step_frac']:.4f}, share_extra_m0={s['share_extra_from_m0']:.3f})"
        )
    p0 = result["mtp_conditional_acceptance_p"][0]
    print(f"CORRELATION SWEEP (oracle q, conservation-constrained; independence a_H={p0:.3f}):")
    print("  a_H   | top1-only uplift | full-span uplift")
    for s, sf in zip(
        result["augment_correlation_sweep_oracle_top1only"],
        result["augment_correlation_sweep_oracle_fullspan"],
    ):
        print(
            f"  {s['a_H']:.3f} | {s['extra_tokens_per_step']:.3f}/step {s['tps_uplift_pct']:+6.2f}% "
            f"| {sf['extra_tokens_per_step']:.3f}/step {sf['tps_uplift_pct']:+6.2f}%"
        )
    print(
        f"  -> +TPS < {result['noise_floor_pct']}% noise floor at a_H>="
        f"{result['a_H_break_below_noise_floor_top1only']} (top1-only) / "
        f"{result['a_H_break_below_noise_floor_fullspan']} (full-span)"
    )
    print(
        f"PRIMARY promptlookup_extra_accept_tokens_per_step (EARLIEST/vLLM-faithful, independence UB)="
        f"{result['primary_metric_value']:.4f} "
        f"(oracle UB={result['primary_metric_value_oracle_upper_bound']:.4f})  GATE={result['gate']}"
    )
    print(f"wrote {out}")

    if args.wandb:
        _log_wandb(result, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
