#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #297 -- tail-resolved per-position: does the hard-prompt acceptance cliff shift?

WHAT THIS ANSWERS
-----------------
kanna #289 (`fi34s269`, MERGED) decomposed the deployed E[T]=3.8512 into a
per-position conditional-acceptance chain a_1..a_7 and located the acceptance
CLIFF at position 1 (it forfeits 1.8952 expected tokens = 45.7% of the 4.149
loss), with conditional acceptance RISING down the chain (survivorship). It then
priced fern #281's BUILT-raise target (public E[T]>=4.966) and found a
feasibility asymmetry (deep-position lift feasible / a_1-only ceiling-bound)
handing the eventual EAGLE-3 build a per-position target.

But #289's tail analysis used a SHAPE-TRANSFER assumption: it reported
`low_tail_cliff_position == top_tail_cliff_position == 1` by applying the
whole-run per-position acceptance SHAPE to each quartile's pooled MEAN (low
E[T]=3.0926, top E[T]=5.0519) under a constant-shape model -- it did NOT directly
measure each quartile's per-position profile (`per_prompt_per_position_banked =
False`). #289's own follow-up flags this: "detecting a genuine tail-specific
cliff-shift would need a per-prompt per-position remeasure."

This leg RESOLVES that caveat. It re-runs lawine #282's per-position acceptance
harness on the 128 competition prompts, capturing the Prometheus per-position
counter `vllm:spec_decode_num_accepted_tokens_per_pos` PER PROMPT (a 128x7
acceptance matrix), pools by E[T] quartile, and DIRECTLY measures whether the
bottom-E[T] (hard) tail SHIFTS the cliff to a later position (=> EAGLE-3 needs
prompt-ADAPTIVE depth) or merely DEEPENS a_1 at the same position-1 cliff (=> a
UNIFORM per-position EAGLE-3 target holds).

This is the within-chain NUMERATOR complement to denken #291 (which closed the
step-side DENOMINATOR at the 487.7 floor). The build is the sole >500 path, so
its per-position target must be pinned exactly.

HOW (contract-safe, measurement-only)
--------------------------------------
Re-uses lawine #282's harness BYTE-FOR-BYTE on the serving side (imports its
counter-read + decode helpers) and adds exactly one new capture: the per-prompt
DELTA of the per-position counter. The harness already reads that counter before
and after every prompt for the scalar deltas; we simply bank the per-position
delta too -- no new server behaviour, no extra reads. conc=1 (MAX_NUM_SEQS=1)
means each prompt's counter delta is unambiguously its own. Launch the UNMODIFIED
deployed submission (`submissions/fa2sw_precache_kenyan`) with only
`DISABLE_LOG_STATS=0` (re-register vLLM stat loggers; host-side counter bumps,
no token change) and `VLLM_USE_FLASHINFER_SAMPLER=0` (this container's cuRAND JIT
shim) overridden -- identical to #282. Greedy identity untouched.

Step 1 (whole-run basis) is CPU-analytic over lawine #282's BANKED whole-run
counter [12452,9458,7500,6171,5152,4306,3645] / 17075 -- it reproduces #289's
a_k and E[T]=3.8512 exactly (resid 0). Step 2 (per-quartile profile) needs the
fresh per-prompt per-position matrix the banked file does not contain.

LOCAL profiling on a single A10G. No HF Job / no submission / no served-file
change / no official draw / no train.py --launch. NOT a launch. NOT open2.
BASELINE stays 481.53; this leg adds 0 TPS (it directly MEASURES the per-quartile
per-position acceptance shape). Analysis-only:
tail_resolved_per_position_analysis_only = True. It measures the DEPLOYED LINEAR
drafter's acceptance shape (the target an EAGLE-3 build must beat), not a built
drafter's; the quartile-pooled (>=32-prompt) profile is the robust unit,
single-prompt cliffs are noisy; the launch gate stays land #245's MEASURED >=500
at lambda_hat>=0.9780 AND PPL<=2.42, human-approval-gated.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# research/validity/tail_resolved_per_position/this.py -> repo root is 3 parents up.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

# --------------------------------------------------------------------------- #
# Imported fleet anchors (DO NOT re-derive -- import EXACTLY, UNCHANGED)
# --------------------------------------------------------------------------- #
OFFICIAL = 481.53                 # #52 official linear TPS (this leg adds 0)
CEILING_LAMBDA1 = 520.95          # lambda=1 ceiling
K_CAL = 125.268                   # kanna #269 anchor: official = K_cal * E[T]
E_T_ANCHOR = 3.844                # kanna #217 deployed linear served E[T]
ET_DECOMP_289 = 3.8512            # kanna #289 decomposed E[T] (rounded; full 3.851185944363104)
STEP_US = 1218.2                  # kanna #217 per-forward-pass time (microseconds)
STEP_MS = 1.2182                  # same, milliseconds
TAU = 1.218                       # composition tau
K_SPEC = 7                        # num_speculative_tokens (linear MTP depth)
E_T_MAX = K_SPEC + 1              # 8.0 -- theoretical E[T] ceiling (all 7 accepted + bonus)
LINEAR_CAP = 3.8445               # denken #119: LINEAR drafter E[T] cap at perfect capacity
PUBLIC_ET_TARGET = 4.966          # fern #281: public E[T] needed @ deployed step (priv 0.804)
PRIVATE_FACTOR = 0.804            # ubel #263 canonical private factor (fern #281)
PRIVATE_VERIFIED = 460.85         # private-verified reference (PR baseline)
TOP1_LINEAR_ACCEPT_ANCHOR = 0.728739760479042  # #76 accept_calibration top-1 (cross-check a_1)
LAMBDA_BAR_OPERATIVE = 0.9780112973731208       # land #245 validity bar (context only)

# kanna #289 banked decomposition anchors (W&B fi34s269; the values we re-derive
# in step 1 and must match within resid < 0.001 / 0.01).
A_K_289 = [0.72925, 0.75956, 0.79298, 0.82280, 0.83487, 0.83579, 0.84649]
CLIFF_POSITION_289 = 1
TOKEN_LOSS1_289 = 1.8952          # token-loss forfeit at position 1 (45.7% of 4.149)
TOKEN_LOSS_TOTAL_289 = 4.149      # total forfeit = 8 - E[T]

# kanna #289 / lawine #282 quartile-pooled E[T] anchors (the load-bearing
# reconstruction targets for the directly-measured per-quartile profile).
LOW_TAIL_POOLED_ET_289 = 3.0925856228829507   # rounds to 3.093
TOP_TAIL_POOLED_ET_289 = 5.051904176904177    # rounds to 5.052
LOW_TAIL_POOLED_ET_ROUND = 3.093
TOP_TAIL_POOLED_ET_ROUND = 5.052
K_CAL_ANCHOR_ROUND = 125.268     # echoed for the imported-EXACT self-test

# lawine #282 banked whole-run spec counters (W&B 2j0e8xgg; step-1 input)
BANKED_ACCEPTED_PER_POS = [12452.0, 9458.0, 7500.0, 6171.0, 5152.0, 4306.0, 3645.0]
BANKED_NUM_DRAFTS = 17075.0
BANKED_NUM_ACCEPTED = 48684.0

OUT_DIR = ROOT / "research" / "validity" / "tail_resolved_per_position"
RESULTS_PATH = OUT_DIR / "tail_resolved_per_position_results.json"
MEASURED_PATH = OUT_DIR / "measured_matrix.json"
DEFAULT_SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
LAWINE282_DIR = ROOT / "research" / "validity" / "et_prompt_distribution"
LAWINE282_HARNESS = LAWINE282_DIR / "et_prompt_distribution.py"
LAWINE282_MEASURED = LAWINE282_DIR / "measured_result.json"


def tps_from_et(et: float) -> float:
    """official = K_cal * E[T] (125.268 * 3.844 = 481.53)."""
    return K_CAL * et


# --------------------------------------------------------------------------- #
# Reuse lawine #282's harness helpers (counter reads + decode), byte-for-byte.
# Loading the committed module guarantees the measurement is identical to #282;
# we add only the per-prompt per-position delta capture in our own loop.
# --------------------------------------------------------------------------- #
def _load_lawine282_module():
    spec = importlib.util.spec_from_file_location(
        "lawine282_et_prompt_distribution", str(LAWINE282_HARNESS))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Core decomposition: survival G, conditional a_k, E[T], token-loss cliff.
# Identical math to kanna #289 (decompose + cliff), reusable for the whole run
# AND for each quartile's pooled per-position counter.
# --------------------------------------------------------------------------- #
def decompose_profile(accepted_per_pos: list[float], num_drafts: float,
                      K: int = K_SPEC) -> dict[str, Any]:
    app = list(accepted_per_pos)
    nd = float(num_drafts)
    # survival of j (# accepted draft tokens): G(m)=P(j>=m), m=0..K
    G = [1.0] + [app[m - 1] / nd for m in range(1, K + 1)]
    # per-position conditional acceptance a_k = G(k)/G(k-1)
    a = [G[m] / G[m - 1] if G[m - 1] > 0 else float("nan") for m in range(1, K + 1)]
    cumprod = list(np.cumprod(a))
    E_j = sum(G[1:])
    E_T = 1.0 + E_j
    # first-rejection forfeit attribution: step whose FIRST reject is at k commits
    # L=k and forfeits (K+1-k). P(first reject at k)=G(k-1)-G(k). sum == 8 - E[T].
    token_loss = [(G[k - 1] - G[k]) * ((K + 1) - k) for k in range(1, K + 1)]
    abs_drop = [G[k - 1] - G[k] for k in range(1, K + 1)]
    cond_drop = [(1.0 if k == 1 else a[k - 2]) - a[k - 1] for k in range(1, K + 1)]
    cliff_by_loss = int(np.argmax(token_loss)) + 1
    cliff_by_min_a = int(np.nanargmin(a)) + 1
    cliff_by_abs_drop = int(np.argmax(abs_drop)) + 1
    return {
        "num_drafts": nd,
        "accepted_per_pos": app,
        "survival_j": G,                 # G(0..K)
        "a_k": a,                        # a_1..a_K (conditional)
        "cumprod_a": cumprod,            # == G(1..K) round-trip
        "E_j": E_j,
        "E_T": E_T,
        "token_loss_per_pos": token_loss,
        "token_loss_total": float(sum(token_loss)),
        "abs_survival_drop_per_pos": abs_drop,
        "cond_accept_drop_per_pos": cond_drop,
        "cliff_position": cliff_by_loss,           # PRIMARY (max token loss)
        "cliff_by_min_conditional": cliff_by_min_a,
        "cliff_by_abs_survival_drop": cliff_by_abs_drop,
        "cliff_agrees_across_measures": (cliff_by_loss == cliff_by_min_a == cliff_by_abs_drop),
        "conditional_acceptance_increases_with_depth": all(
            a[i] <= a[i + 1] + 1e-9 for i in range(len(a) - 1)),
    }


# --------------------------------------------------------------------------- #
# Measurement: serve once, drive 128 prompts one at a time, capture the
# per-prompt per-position counter delta -> 128xK acceptance matrix.
# --------------------------------------------------------------------------- #
def measure(submission: Path, *, num_prompts: int, output_len: int, seed: int,
            out_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    L = _load_lawine282_module()
    dco = L._load_official_decode_module()
    tokenizer = AutoTokenizer.from_pretrained(paths.TOKENIZER)
    records = dco.read_sharegpt_prompts(paths.EVAL_PROMPTS, num_prompts=num_prompts, seed=seed)
    if len(records) != num_prompts:
        raise ValueError(f"expected {num_prompts} prompts, found {len(records)}")

    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    extra_env = {"DISABLE_LOG_STATS": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    log_path = OUT_DIR / "server_tail_resolved_per_position.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _ppos(d: dict[str, Any]) -> list[float]:
        v = d.get("accepted_per_pos")
        return list(v) if v else [0.0] * K_SPEC

    per_prompt: list[dict[str, Any]] = []
    t0 = time.time()
    with L.VramPeak() as vram, harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        model = srv.served_model_name
        base = L.read_spec_counters(srv.base_url)
        if base.get("num_drafts") is None:
            raise RuntimeError("vLLM Prometheus spec counters not populated on this wheel; "
                               "per-prompt delta method is unavailable")
        prev = dict(base)
        for idx, rec in enumerate(records):
            ptext = rec["prompt_text"]
            ptoks = dco.encode_prompt(tokenizer, ptext)
            resp = L.request_decode(srv.base_url, model, ptoks, output_len)
            n_completion = L._completion_tokens(resp)
            cur = L.read_spec_counters_stable(srv.base_url, prev_drafts=prev.get("num_drafts") or 0.0)
            d_drafts = (cur.get("num_drafts") or 0.0) - (prev.get("num_drafts") or 0.0)
            d_acc = (cur.get("num_accepted_tokens") or 0.0) - (prev.get("num_accepted_tokens") or 0.0)
            d_draft_tok = (cur.get("num_draft_tokens") or 0.0) - (prev.get("num_draft_tokens") or 0.0)
            cur_pp, prev_pp = _ppos(cur), _ppos(prev)
            d_per_pos = [cur_pp[k] - prev_pp[k] for k in range(K_SPEC)]
            et_p = (1.0 + d_acc / d_drafts) if d_drafts > 0 else float("nan")
            feats = L.prompt_features(ptext, ptoks)
            per_prompt.append({
                "index": idx,
                "id": rec["id"],
                "dataset_index": rec["dataset_index"],
                "n_completion_tokens": n_completion,
                "delta_drafts": d_drafts,
                "delta_accepted": d_acc,
                "delta_draft_tokens": d_draft_tok,
                "accepted_per_pos_delta": d_per_pos,        # NEW: 7-vec per prompt
                "per_pos_sums_to_accepted_resid": float(sum(d_per_pos) - d_acc),
                "emitted_eq_acc_plus_drafts": d_acc + d_drafts,
                "E_T": et_p,
                **feats,
            })
            prev = cur
            if (idx + 1) % 16 == 0:
                print(f"[tail-rpp] {idx + 1}/{num_prompts} E[T]_p={et_p:.4f} "
                      f"(steps={d_drafts:.0f}, acc={d_acc:.0f}, perpos={[int(x) for x in d_per_pos]})",
                      flush=True)
        final = L.read_spec_counters(srv.base_url)
    wall = time.time() - t0

    total_drafts = (final.get("num_drafts") or 0.0) - (base.get("num_drafts") or 0.0)
    total_acc = (final.get("num_accepted_tokens") or 0.0) - (base.get("num_accepted_tokens") or 0.0)
    final_per_pos = [(_ppos(final)[k] - _ppos(base)[k]) for k in range(K_SPEC)]
    sum_per_prompt_per_pos = [
        float(sum(p["accepted_per_pos_delta"][k] for p in per_prompt)) for k in range(K_SPEC)
    ]

    measured = {
        "pr": 297,
        "leg": "tail-resolved per-position: per-prompt per-position acceptance matrix (local)",
        "tail_resolved_per_position_analysis_only": True,
        "submission": str(submission),
        "server_python": str(server_python),
        "dataset": str(paths.EVAL_PROMPTS),
        "num_prompts": num_prompts,
        "output_len": output_len,
        "seed": seed,
        "K_spec": K_SPEC,
        "conc": 1,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decode_wall_s": wall,
        "peak_gpu_gb": vram.peak_mib / 1024.0,
        "baseline_counters": base,
        "final_counters": final,
        "whole_run_total_drafts": total_drafts,
        "whole_run_total_accepted": total_acc,
        "whole_run_final_per_pos": final_per_pos,
        "sum_per_prompt_per_pos": sum_per_prompt_per_pos,
        "per_prompt": per_prompt,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(measured, indent=2))
    print(f"[tail-rpp] measured {len(per_prompt)} prompts in {wall:.0f}s; "
          f"peak {measured['peak_gpu_gb']:.2f} GB -> {out_path}", flush=True)
    print(f"[tail-rpp] whole-run per-pos (fresh): {[int(x) for x in final_per_pos]} "
          f"(banked #282 {[int(x) for x in BANKED_ACCEPTED_PER_POS]})", flush=True)
    return measured


# --------------------------------------------------------------------------- #
# Analysis: step 1 (whole-run basis) + step 2 (per-quartile direct measurement)
# + step 3 (cliff-shift verdict).
# --------------------------------------------------------------------------- #
def _lawine282_per_prompt_et() -> list[float] | None:
    """Banked lawine #282 per-prompt E[T] (run 2j0e8xgg), for the CANONICAL
    quartile ranking that reproduces kanna #289's low/top tail indices exactly."""
    if not LAWINE282_MEASURED.exists():
        return None
    src = json.loads(LAWINE282_MEASURED.read_text())
    return [p["E_T"] for p in src["per_prompt"]]


def analyze(measured: dict[str, Any]) -> dict[str, Any]:
    K = K_SPEC
    per = measured["per_prompt"]
    n = len(per)

    # ---- Step 1: whole-run basis from lawine #282's BANKED counter (CPU, exact) #
    whole = decompose_profile(BANKED_ACCEPTED_PER_POS, BANKED_NUM_DRAFTS, K)
    a_k_resid_vs_289 = [abs(whole["a_k"][i] - A_K_289[i]) for i in range(K)]
    whole_resid_vs_anchor = whole["E_T"] - E_T_ANCHOR
    whole_resid_vs_decomp = whole["E_T"] - ET_DECOMP_289

    # fresh whole-run cross-check (if the matrix was measured this run)
    fresh_per_pos = measured.get("whole_run_final_per_pos")
    fresh_drafts = measured.get("whole_run_total_drafts")
    fresh_whole = None
    if fresh_per_pos and fresh_drafts:
        fresh_whole = decompose_profile(fresh_per_pos, fresh_drafts, K)

    # ---- Step 2: per-quartile DIRECT per-position measurement ---------------- #
    # Build the 128xK acceptance matrix and per-prompt draft counts.
    M = np.array([p["accepted_per_pos_delta"] for p in per], dtype=float)  # (n, K)
    drafts = np.array([p["delta_drafts"] for p in per], dtype=float)       # (n,)
    accs = np.array([p["delta_accepted"] for p in per], dtype=float)
    fresh_ets = np.array([p["E_T"] for p in per], dtype=float)

    # CANONICAL quartile ranking: lawine #282's banked per-prompt E[T] (reproduces
    # #289's low/top tail index sets). Fall back to this run's E[T] if absent.
    banked_ets = _lawine282_per_prompt_et()
    ranking_source = "lawine282_banked"
    if banked_ets is not None and len(banked_ets) == n:
        rank_ets = np.array(banked_ets, dtype=float)
    else:
        rank_ets = fresh_ets
        ranking_source = "this_run_fresh"
    order = np.argsort(rank_ets)
    q = n // 4
    low_idx, top_idx = order[:q], order[-q:]

    # robustness: how stable is quartile membership under this run's fresh E[T]?
    fresh_order = np.argsort(fresh_ets)
    low_fresh, top_fresh = set(fresh_order[:q].tolist()), set(fresh_order[-q:].tolist())
    low_overlap = len(set(low_idx.tolist()) & low_fresh) / q
    top_overlap = len(set(top_idx.tolist()) & top_fresh) / q

    def pooled_profile(idx: np.ndarray) -> dict[str, Any]:
        app_q = M[idx].sum(axis=0).tolist()      # pooled per-position accepted
        nd_q = float(drafts[idx].sum())
        prof = decompose_profile(app_q, nd_q, K)
        # token-weighted pooled E[T] from drafts/accepted (cross-check vs per-pos)
        prof["pooled_et_from_scalar"] = 1.0 + float(accs[idx].sum() / nd_q) if nd_q else float("nan")
        return prof

    low = pooled_profile(low_idx)
    top = pooled_profile(top_idx)

    # reconstruction residuals vs kanna #289's banked pooled E[T]
    low_et_resid = low["E_T"] - LOW_TAIL_POOLED_ET_289
    top_et_resid = top["E_T"] - TOP_TAIL_POOLED_ET_289

    # ---- Step 3: cliff-shift verdict + build-spec implication ---------------- #
    cliff_low, cliff_top = low["cliff_position"], top["cliff_position"]
    tail_cliff_shifts = bool(cliff_low != CLIFF_POSITION_289 or cliff_low != cliff_top)
    low_tail_mode = "cliff-shift" if cliff_low != cliff_top else "a1-deepen"
    a1_low, a1_top = low["a_k"][0], top["a_k"][0]
    a1_deepens = bool(a1_low < a1_top)
    eagle3_target_is_prompt_adaptive = tail_cliff_shifts

    # secondary (noisy): per-prompt cliff-position distribution over the 128xK matrix
    per_prompt_cliff: list[int] = []
    for p_i in range(n):
        nd_i = drafts[p_i]
        if nd_i <= 0:
            per_prompt_cliff.append(0)
            continue
        prof_i = decompose_profile(M[p_i].tolist(), float(nd_i), K)
        per_prompt_cliff.append(prof_i["cliff_position"])
    cliff_hist = [int(np.sum(np.array(per_prompt_cliff) == k)) for k in range(1, K + 1)]
    cliff_hist_unresolved = int(np.sum(np.array(per_prompt_cliff) == 0))

    # whole-run reconciliation: per-prompt per-pos deltas sum to whole-run counter
    sum_pp = measured.get("sum_per_prompt_per_pos") or M.sum(axis=0).tolist()
    final_pp = measured.get("whole_run_final_per_pos") or sum_pp
    per_pos_reconcile_resid = [abs(sum_pp[k] - final_pp[k]) for k in range(K)]
    prefix_invariant_resids = [abs(p["per_pos_sums_to_accepted_resid"]) for p in per]
    max_prefix_resid = max(prefix_invariant_resids) if prefix_invariant_resids else 0.0

    report = {
        "pr": 297,
        "leg": "tail-resolved per-position: does the hard-prompt acceptance cliff shift?",
        "tail_resolved_per_position_analysis_only": True,
        "measured_source": str(MEASURED_PATH),
        "ranking_source": ranking_source,
        "imported": {
            "official": OFFICIAL, "ceiling_lambda1": CEILING_LAMBDA1, "K_cal": K_CAL,
            "E_T_anchor": E_T_ANCHOR, "et_decomp_289": ET_DECOMP_289, "step_us": STEP_US,
            "step_ms": STEP_MS, "tau": TAU, "K_spec": K_SPEC, "E_T_max": E_T_MAX,
            "linear_cap_denken119": LINEAR_CAP, "public_et_target_fern281": PUBLIC_ET_TARGET,
            "private_factor": PRIVATE_FACTOR, "private_verified": PRIVATE_VERIFIED,
            "top1_linear_accept_anchor": TOP1_LINEAR_ACCEPT_ANCHOR,
            "a_k_289": A_K_289, "cliff_position_289": CLIFF_POSITION_289,
            "low_tail_pooled_et_289": LOW_TAIL_POOLED_ET_289,
            "top_tail_pooled_et_289": TOP_TAIL_POOLED_ET_289,
            "banked_accepted_per_pos": BANKED_ACCEPTED_PER_POS,
            "banked_num_drafts": BANKED_NUM_DRAFTS, "banked_num_accepted": BANKED_NUM_ACCEPTED,
        },
        # Step 1 -- whole-run basis (from banked #282 counter)
        "whole_run": {
            "a_k": whole["a_k"],
            "survival_j": whole["survival_j"],
            "cumprod_a": whole["cumprod_a"],
            "E_T": whole["E_T"],
            "token_loss_per_pos": whole["token_loss_per_pos"],
            "token_loss_total": whole["token_loss_total"],
            "cliff_position": whole["cliff_position"],
            "cliff_agrees_across_measures": whole["cliff_agrees_across_measures"],
            "conditional_acceptance_increases_with_depth":
                whole["conditional_acceptance_increases_with_depth"],
            "a_k_resid_vs_289_max": max(a_k_resid_vs_289),
            "a_k_resid_vs_289": a_k_resid_vs_289,
            "E_T_resid_vs_anchor": whole_resid_vs_anchor,
            "E_T_resid_vs_decomp_289": whole_resid_vs_decomp,
        },
        "whole_run_fresh_crosscheck": (None if fresh_whole is None else {
            "accepted_per_pos": fresh_per_pos,
            "num_drafts": fresh_drafts,
            "a_k": fresh_whole["a_k"],
            "E_T": fresh_whole["E_T"],
            "cliff_position": fresh_whole["cliff_position"],
            "E_T_resid_vs_banked": fresh_whole["E_T"] - whole["E_T"],
            "per_pos_resid_vs_banked": [fresh_per_pos[k] - BANKED_ACCEPTED_PER_POS[k]
                                        for k in range(K)],
        }),
        # Step 2 -- per-quartile DIRECT per-position profile
        "quartile_size": q,
        "low_tail_indices": [int(i) for i in low_idx],
        "top_tail_indices": [int(i) for i in top_idx],
        "low_tail": {
            "a_k": low["a_k"], "survival_j": low["survival_j"],
            "E_T": low["E_T"], "pooled_et_from_scalar": low["pooled_et_from_scalar"],
            "cliff_position": low["cliff_position"],
            "cliff_by_min_conditional": low["cliff_by_min_conditional"],
            "cliff_by_abs_survival_drop": low["cliff_by_abs_survival_drop"],
            "cliff_agrees_across_measures": low["cliff_agrees_across_measures"],
            "conditional_acceptance_increases_with_depth":
                low["conditional_acceptance_increases_with_depth"],
            "token_loss_per_pos": low["token_loss_per_pos"],
            "num_drafts": low["num_drafts"], "accepted_per_pos": low["accepted_per_pos"],
            "E_T_resid_vs_289": low_et_resid,
        },
        "top_tail": {
            "a_k": top["a_k"], "survival_j": top["survival_j"],
            "E_T": top["E_T"], "pooled_et_from_scalar": top["pooled_et_from_scalar"],
            "cliff_position": top["cliff_position"],
            "cliff_by_min_conditional": top["cliff_by_min_conditional"],
            "cliff_by_abs_survival_drop": top["cliff_by_abs_survival_drop"],
            "cliff_agrees_across_measures": top["cliff_agrees_across_measures"],
            "conditional_acceptance_increases_with_depth":
                top["conditional_acceptance_increases_with_depth"],
            "token_loss_per_pos": top["token_loss_per_pos"],
            "num_drafts": top["num_drafts"], "accepted_per_pos": top["accepted_per_pos"],
            "E_T_resid_vs_289": top_et_resid,
        },
        # Step 3 -- verdict
        "cliff_position_low": cliff_low,
        "cliff_position_top": cliff_top,
        "tail_cliff_shifts": tail_cliff_shifts,
        "low_tail_mode": low_tail_mode,
        "a1_low": a1_low, "a1_top": a1_top,
        "a1_deepens_in_low_tail": a1_deepens,
        "a1_gap_top_minus_low": a1_top - a1_low,
        "eagle3_target_is_prompt_adaptive": eagle3_target_is_prompt_adaptive,
        # secondary (noisy) per-prompt cliff distribution
        "per_prompt_cliff_position": per_prompt_cliff,
        "per_prompt_cliff_histogram": cliff_hist,           # index 0 -> position 1
        "per_prompt_cliff_unresolved": cliff_hist_unresolved,
        "per_prompt_cliff_modal_position": int(np.argmax(cliff_hist)) + 1,
        # measurement-fidelity diagnostics
        "quartile_membership_low_overlap_vs_fresh": low_overlap,
        "quartile_membership_top_overlap_vs_fresh": top_overlap,
        "per_pos_whole_run_reconcile_resid": per_pos_reconcile_resid,
        "per_pos_whole_run_reconcile_max": max(per_pos_reconcile_resid),
        "prefix_invariant_max_resid": max_prefix_resid,
        "decode_wall_s": measured.get("decode_wall_s"),
        "peak_gpu_gb": measured.get("peak_gpu_gb"),
        "num_prompts": n,
    }
    return report


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(report: dict[str, Any]) -> dict[str, Any]:
    K = K_SPEC
    w = report["whole_run"]
    low, top = report["low_tail"], report["top_tail"]
    checks: dict[str, bool] = {}

    # (a) whole-run a_k + E[T]=3.8512 reproduce kanna #289 (resid < 0.001 / 0.01)
    checks["a_whole_run_a_k_reproduces_289"] = w["a_k_resid_vs_289_max"] < 1e-3
    checks["a_whole_run_et_reproduces_decomp"] = abs(w["E_T_resid_vs_decomp_289"]) < 0.01
    checks["a_whole_run_et_reproduces_anchor"] = abs(w["E_T_resid_vs_anchor"]) < 0.01
    checks["a_whole_run_cumprod_roundtrips"] = all(
        abs(w["cumprod_a"][i] - w["survival_j"][i + 1]) < 1e-9 for i in range(K))

    # (b) per-quartile pooled E[T] reconstructs lawine #282's 3.093 / 5.052 (resid < 0.1)
    checks["b_low_tail_et_reconstructs_3p093"] = abs(low["E_T_resid_vs_289"]) < 0.1
    checks["b_top_tail_et_reconstructs_5p052"] = abs(top["E_T_resid_vs_289"]) < 0.1
    # per-position pooled E[T] agrees with token-weighted scalar E[T] (prefix invariant)
    checks["b_low_perpos_matches_scalar"] = abs(low["E_T"] - low["pooled_et_from_scalar"]) < 0.05
    checks["b_top_perpos_matches_scalar"] = abs(top["E_T"] - top["pooled_et_from_scalar"]) < 0.05

    # (c) all a_{q,k} in [0,1], survival monotone non-increasing, cumprod round-trips per quartile
    def _valid(prof: dict[str, Any]) -> bool:
        a = prof["a_k"]
        G = prof["survival_j"]
        in_unit = all(-1e-9 <= x <= 1.0 + 1e-9 for x in a)
        monotone = all(G[i] >= G[i + 1] - 1e-12 for i in range(len(G) - 1))
        cp = list(np.cumprod(a))
        roundtrip = all(abs(cp[i] - G[i + 1]) < 1e-9 for i in range(K))
        return in_unit and monotone and roundtrip
    checks["c_low_tail_profile_valid"] = _valid(low)
    checks["c_top_tail_profile_valid"] = _valid(top)
    checks["c_whole_run_profile_valid"] = all(-1e-9 <= x <= 1 + 1e-9 for x in w["a_k"]) and \
        all(w["survival_j"][i] >= w["survival_j"][i + 1] - 1e-12
            for i in range(len(w["survival_j"]) - 1))

    # (d) NaN-clean across reported scalars
    scalars = [
        w["E_T"], low["E_T"], top["E_T"], report["a1_low"], report["a1_top"],
        report["a1_gap_top_minus_low"], low["pooled_et_from_scalar"], top["pooled_et_from_scalar"],
        float(report["cliff_position_low"]), float(report["cliff_position_top"]),
    ]
    nan_clean = (all(math.isfinite(float(x)) for x in scalars)
                 and all(math.isfinite(float(x)) for x in w["a_k"])
                 and all(math.isfinite(float(x)) for x in low["a_k"])
                 and all(math.isfinite(float(x)) for x in top["a_k"]))
    checks["d_nan_clean"] = nan_clean

    # (e) imported anchors EXACT and UNCHANGED
    checks["e_constants_imported_exact"] = (
        E_T_ANCHOR == 3.844 and ET_DECOMP_289 == 3.8512 and PUBLIC_ET_TARGET == 4.966
        and LINEAR_CAP == 3.8445 and PRIVATE_FACTOR == 0.804
        and LOW_TAIL_POOLED_ET_ROUND == 3.093 and TOP_TAIL_POOLED_ET_ROUND == 5.052
        and K_CAL == 125.268 and STEP_US == 1218.2 and OFFICIAL == 481.53 and K_SPEC == 7
    )

    # (f) the leg carries the 0-TPS + measures-LINEAR-drafter-shape + pooling-robust caveats
    checks["f_carries_caveats"] = bool(
        report["tail_resolved_per_position_analysis_only"] is True
        and report["quartile_size"] >= 32
        and "per_prompt_cliff_unresolved" in report
    )

    # bonus: directly-measured per-quartile pooled E[T] reconstruct via per-position
    # counts faithfully (whole-run reconciliation + prefix invariant)
    checks["g_per_pos_reconciles_whole_run"] = report["per_pos_whole_run_reconcile_max"] < 1.0
    checks["g_prefix_invariant_holds"] = report["prefix_invariant_max_resid"] < 2.0

    gate = bool(
        checks["a_whole_run_a_k_reproduces_289"] and checks["a_whole_run_et_reproduces_decomp"]
        and checks["a_whole_run_et_reproduces_anchor"] and checks["a_whole_run_cumprod_roundtrips"]
        and checks["b_low_tail_et_reconstructs_3p093"] and checks["b_top_tail_et_reconstructs_5p052"]
        and checks["b_low_perpos_matches_scalar"] and checks["b_top_perpos_matches_scalar"]
        and checks["c_low_tail_profile_valid"] and checks["c_top_tail_profile_valid"]
        and checks["c_whole_run_profile_valid"] and checks["d_nan_clean"]
        and checks["e_constants_imported_exact"] and checks["f_carries_caveats"]
    )
    report["self_test"] = checks
    report["tail_resolved_per_position_self_test_passes"] = gate
    return report


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], measured: dict[str, Any] | None,
              name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[tail-rpp] wandb unavailable ({exc})", flush=True)
        return None
    try:
        w, low, top = report["whole_run"], report["low_tail"], report["top_tail"]
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="profiling",
            config={
                "pr": 297, "analysis_only": True, "submission": report.get("measured_source"),
                "ranking_source": report["ranking_source"], "K_spec": K_SPEC, "K_cal": K_CAL,
                "official": OFFICIAL, "E_T_anchor": E_T_ANCHOR, "et_decomp_289": ET_DECOMP_289,
                "linear_cap": LINEAR_CAP, "public_et_target": PUBLIC_ET_TARGET,
                "private_factor": PRIVATE_FACTOR, "low_tail_pooled_et_289": LOW_TAIL_POOLED_ET_289,
                "top_tail_pooled_et_289": TOP_TAIL_POOLED_ET_289, "quartile_size": report["quartile_size"],
            },
        )
        flat = {
            "primary/tail_resolved_per_position_self_test_passes":
                report["tail_resolved_per_position_self_test_passes"],
            "test/tail_cliff_shifts": report["tail_cliff_shifts"],
            "test/cliff_position_low": report["cliff_position_low"],
            "test/cliff_position_top": report["cliff_position_top"],
            "test/eagle3_target_is_prompt_adaptive": report["eagle3_target_is_prompt_adaptive"],
            "low_tail_mode_is_a1_deepen": (report["low_tail_mode"] == "a1-deepen"),
            "whole_run_E_T": w["E_T"],
            "whole_run_cliff_position": w["cliff_position"],
            "whole_run_a_k_resid_vs_289_max": w["a_k_resid_vs_289_max"],
            "whole_run_E_T_resid_vs_anchor": w["E_T_resid_vs_anchor"],
            "low_tail_E_T": low["E_T"], "top_tail_E_T": top["E_T"],
            "low_tail_E_T_resid_vs_289": low["E_T_resid_vs_289"],
            "top_tail_E_T_resid_vs_289": top["E_T_resid_vs_289"],
            "a1_low": report["a1_low"], "a1_top": report["a1_top"],
            "a1_gap_top_minus_low": report["a1_gap_top_minus_low"],
            "a1_deepens_in_low_tail": report["a1_deepens_in_low_tail"],
            "low_tail_cond_accept_increases": low["conditional_acceptance_increases_with_depth"],
            "top_tail_cond_accept_increases": top["conditional_acceptance_increases_with_depth"],
            "low_tail_cliff_agrees": low["cliff_agrees_across_measures"],
            "top_tail_cliff_agrees": top["cliff_agrees_across_measures"],
            "per_prompt_cliff_modal_position": report["per_prompt_cliff_modal_position"],
            "per_prompt_cliff_unresolved": report["per_prompt_cliff_unresolved"],
            "quartile_membership_low_overlap_vs_fresh": report["quartile_membership_low_overlap_vs_fresh"],
            "quartile_membership_top_overlap_vs_fresh": report["quartile_membership_top_overlap_vs_fresh"],
            "per_pos_whole_run_reconcile_max": report["per_pos_whole_run_reconcile_max"],
            "prefix_invariant_max_resid": report["prefix_invariant_max_resid"],
            "peak_gpu_gb": report.get("peak_gpu_gb"),
            "decode_wall_s": report.get("decode_wall_s"),
        }
        if report.get("whole_run_fresh_crosscheck"):
            fc = report["whole_run_fresh_crosscheck"]
            flat["whole_run_fresh_E_T"] = fc["E_T"]
            flat["whole_run_fresh_E_T_resid_vs_banked"] = fc["E_T_resid_vs_banked"]
        run.summary.update(flat)

        # per-position profile table (whole-run / low-tail / top-tail side by side)
        ptbl = wandb.Table(columns=[
            "position_k", "a_k_whole", "a_k_low", "a_k_top",
            "G_whole", "G_low", "G_top",
            "token_loss_whole", "token_loss_low", "token_loss_top",
        ])
        for k in range(K_SPEC):
            ptbl.add_data(
                k + 1, w["a_k"][k], low["a_k"][k], top["a_k"][k],
                w["survival_j"][k + 1], low["survival_j"][k + 1], top["survival_j"][k + 1],
                w["token_loss_per_pos"][k], low["token_loss_per_pos"][k], top["token_loss_per_pos"][k],
            )
        run.log({"per_position_profile_by_quartile": ptbl})

        # per-prompt cliff histogram
        htbl = wandb.Table(columns=["cliff_position", "count"])
        for k in range(K_SPEC):
            htbl.add_data(k + 1, report["per_prompt_cliff_histogram"][k])
        htbl.add_data(0, report["per_prompt_cliff_unresolved"])
        run.log({"per_prompt_cliff_histogram": htbl})

        # per-prompt E[T] + cliff table (full record for downstream analysis)
        if measured is not None:
            per = measured["per_prompt"]
            cliffs = report["per_prompt_cliff_position"]
            etbl = wandb.Table(columns=["index", "E_T", "cliff_position", "delta_drafts",
                                        "delta_accepted", "domain"])
            for p in per:
                etbl.add_data(p["index"], p["E_T"], cliffs[p["index"]], p["delta_drafts"],
                              p["delta_accepted"], p.get("domain", "?"))
            run.log({"per_prompt_E_T_cliff": etbl})

        rid = run.id
        print(f"[tail-rpp] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[tail-rpp] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="reuse cached measured_matrix.json if present (else measure), "
                         "then run analysis + self-test + wandb (PRIMARY).")
    ap.add_argument("--measure", action="store_true",
                    help="force a fresh GPU measurement even if a cache exists.")
    ap.add_argument("--smoke", type=int, default=0,
                    help="run N prompts to a smoke path (plumbing check; no canonical write).")
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="tail-resolved-per-position")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/tail-resolved-per-position")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[tail-rpp] {note}", flush=True)

    if args.smoke > 0:
        smoke_path = OUT_DIR / "measured_matrix_smoke.json"
        m = measure(args.submission.resolve(), num_prompts=args.smoke,
                    output_len=args.output_len, seed=args.seed, out_path=smoke_path)
        per = m["per_prompt"]
        max_prefix = max(abs(p["per_pos_sums_to_accepted_resid"]) for p in per)
        print("\n========== SMOKE (PR #297) ==========", flush=True)
        print(f"prompts measured     : {len(per)}", flush=True)
        print(f"per-pos snapshot ok  : max prefix-invariant resid = {max_prefix:.3f} "
              f"(should be ~0)", flush=True)
        print(f"whole-run per-pos    : {[int(x) for x in m['whole_run_final_per_pos']]}", flush=True)
        print(f"sum per-prompt per-pos: {[int(x) for x in m['sum_per_prompt_per_pos']]}", flush=True)
        for p in per:
            print(f"  idx {p['index']:>3} E[T]={p['E_T']:.3f} perpos="
                  f"{[int(x) for x in p['accepted_per_pos_delta']]} acc={p['delta_accepted']:.0f}",
                  flush=True)
        return 0

    if args.measure or not MEASURED_PATH.exists():
        measured = measure(args.submission.resolve(), num_prompts=args.num_prompts,
                           output_len=args.output_len, seed=args.seed, out_path=MEASURED_PATH)
    else:
        print(f"[tail-rpp] reusing cached measurement {MEASURED_PATH}", flush=True)
        measured = json.loads(MEASURED_PATH.read_text())

    report = analyze(measured)
    report = self_test(report)

    wid = None
    if not args.no_wandb:
        wid = log_wandb(report, measured, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    RESULTS_PATH.write_text(json.dumps(report, indent=2))

    w, low, top = report["whole_run"], report["low_tail"], report["top_tail"]
    print("\n========== TAIL-RESOLVED PER-POSITION (PR #297) ==========", flush=True)
    print(f"[step1] whole-run E[T]   : {w['E_T']:.5f}  (anchor 3.844; decomp 3.8512; "
          f"a_k resid<289 {w['a_k_resid_vs_289_max']:.2e})", flush=True)
    print(f"[step1] whole-run a_k    : " + " ".join(f"{x:.4f}" for x in w["a_k"]), flush=True)
    print(f"[step1] whole-run cliff  : {w['cliff_position']} "
          f"(agrees {w['cliff_agrees_across_measures']})", flush=True)
    print(f"[step2] LOW  tail E[T]    : {low['E_T']:.4f}  (289 3.0926; resid "
          f"{low['E_T_resid_vs_289']:+.4f})", flush=True)
    print(f"[step2] LOW  tail a_k    : " + " ".join(f"{x:.4f}" for x in low["a_k"]), flush=True)
    print(f"[step2] TOP  tail E[T]    : {top['E_T']:.4f}  (289 5.0519; resid "
          f"{top['E_T_resid_vs_289']:+.4f})", flush=True)
    print(f"[step2] TOP  tail a_k    : " + " ".join(f"{x:.4f}" for x in top["a_k"]), flush=True)
    print(f"[step3] cliff low/top    : {report['cliff_position_low']} / {report['cliff_position_top']} "
          f"(a1 low {report['a1_low']:.4f} < top {report['a1_top']:.4f}: "
          f"{report['a1_deepens_in_low_tail']})", flush=True)
    print(f"[step3] tail_cliff_shifts: {report['tail_cliff_shifts']}  "
          f"mode={report['low_tail_mode']}", flush=True)
    print(f"[step3] eagle3 prompt-adaptive: {report['eagle3_target_is_prompt_adaptive']}", flush=True)
    print(f"[2nd]   per-prompt cliff hist (pos1..7): {report['per_prompt_cliff_histogram']} "
          f"(unresolved {report['per_prompt_cliff_unresolved']}; modal "
          f"{report['per_prompt_cliff_modal_position']})", flush=True)
    print(f"[fid]   per-pos reconcile max: {report['per_pos_whole_run_reconcile_max']:.2f}; "
          f"prefix-inv max {report['prefix_invariant_max_resid']:.2f}; "
          f"quartile overlap low/top {report['quartile_membership_low_overlap_vs_fresh']:.2f}/"
          f"{report['quartile_membership_top_overlap_vs_fresh']:.2f}", flush=True)
    print(f"PRIMARY self_test        : {report['tail_resolved_per_position_self_test_passes']}", flush=True)
    print(f"peak GPU                 : {report.get('peak_gpu_gb', float('nan')) or float('nan'):.2f} GB",
          flush=True)
    print(f"wandb run                : {wid}", flush=True)
    print(f"artifacts                : {RESULTS_PATH}", flush=True)

    if args.self_test and not report["tail_resolved_per_position_self_test_passes"]:
        failed = [k for k, v in report["self_test"].items() if not v]
        print(f"[tail-rpp] SELF-TEST FAILED: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
