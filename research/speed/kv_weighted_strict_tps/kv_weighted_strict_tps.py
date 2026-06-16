#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""KV-weighted strict TPS: the honest official-draw number (PR #475, stark).

CPU-ONLY ANALYSIS. NO kernel re-measure, NO served-file change, NO HF Job, NO
submission. analysis_only=true, official_tps=0, no_served_file_change=true.

THE QUESTION
------------
The realized strict frontier we submit (#474) is quoted at the #472 L=640
WORST-CASE strict TPS (457.55) -- the deployed-faithful *longest* KV the M=8
verify attention sees. But the official leaderboard `summary.json:tps` is
`total_output_tokens / total_wall_clock` over the 128-prompt benchmark run at
output_len=512, single-stream -- so it reflects the REAL KV-length distribution
of the decode trajectory, not a single worst-case L. This card predicts that
official number honestly.

METHOD (instruction 1-3)
------------------------
1. Trajectory. The official TPS path (official/.../speed_benchmark) runs
   sglang.bench_serving, sharegpt dataset (eval_prompts_sharegpt.json),
   OUTPUT_LEN=512, NUM_PROMPTS=128, MAX_CONCURRENCY=1, ignore_eos=True -- so
   every prompt emits EXACTLY 512 tokens. For served prompt j of length P_j, the
   KV the full-attention M=8 verify reduction sees when emitting output token i
   is KV = P_j + i (single-stream, KV grows one position per output token). The
   P_j are the chat-templated first-human-turn lengths; we read them from the
   PPL ground-truth `context_token_ids` (same 128 prompts, pre-tokenized) by
   truncating at the `<start_of_turn>model\n` generation-prompt boundary, and we
   VALIDATE them bit-for-bit against a real server decode_outputs.jsonl capture.
2. Per-L strict tax. Reuse #472's measured whole-cycle strict tax
   whole_delta_us(L) at L in {128,384,640} (the in-graph-overlap-captured added
   us/cycle the order-preserving strict 2D reduction costs over the deployed
   permissive 3D path). Interpolate added_us(L) PIECEWISE-LINEAR through the
   three measured points (the strict tax scales ~linearly with KV, #472), with
   edge-slope extrapolation outside [128,640]. realized TPS at KV=L is the SAME
   banked mapping #472/#466 used: tps(added) = DEPLOYED*CYCLE_PERM/(CYCLE_PERM+added).
3. Aggregate. The leaderboard metric accumulates WALL TIME per token, so the
   correct aggregate is the TOKEN-WEIGHTED HARMONIC MEAN of tps(KV) over the
   realized trajectory:
       kv_weighted_strict_tps = N_tok / sum_token 1/tps(added(KV_token)).
   (Under a tax linear in KV this equals tps evaluated at the mean KV -- logged
   as a cross-check.) Band: the central estimate extrapolates the strict tax
   linearly past the measured L=640 (physical: the 2D reduction is O(KV)); the
   upper end clamps the tax at its L=640 value (tax saturates beyond the measured
   range) -- a model-uncertainty band, NOT hardware noise.

REPORTED
--------
kv_weighted_strict_tps (PRIMARY), kv_trajectory_mean_L, predicted_official_tps_band
[central, clamp@640], weighted_above_L640_worstcase (bool), l640_worstcase_tps
(457.55 headline), l128_bestcase_tps, uplift_vs_l640_worstcase, sigma_hw (between-
session 4.864), materially_above_worstcase (uplift >= sigma_hw), tps_at_mean_kv
(linear cross-check), frac_decode_tokens_kv_gt_640, served_lengths_match_real_capture,
ppl (carry 2.3772), analysis_only/official_tps/no_served_file_change.

Reproduce: cd target/ && .venv/bin/python \
  research/speed/kv_weighted_strict_tps/kv_weighted_strict_tps.py \
  --wandb_group equivalence-escalation-anchors --wandb_name stark/kv-weighted-strict-tps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", ".."))

# Inputs (all read-only; this card measures nothing on the GPU).
SWEEP_JSON = os.path.join(_root, "research/speed/strict_wholecycle_ab/strict_wholecycle_ab.json")
PPL_TOKENS = os.path.join(
    _root, "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl")
REAL_CAPTURE = os.path.join(
    _root, "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl")

# Fixed official benchmark contract (speed_benchmark/scripts/hf_bucket_single_job.py).
NUM_PROMPTS = 128
OUTPUT_LEN = 512
# Gemma chat-template generation-prompt marker: <start_of_turn>model\n
GEN_PROMPT_MARKER = (105, 4368, 107)
# Advisor-named hardware sigma (lawine #467): between-session served-TPS envelope.
SIGMA_HW_BETWEEN = 4.864
PPL_ANCHOR = 2.3772
PPL_GATE = 2.42


def find_subsequence(seq, sub):
    n, m = len(seq), len(sub)
    sub = list(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return i
    return -1


def served_prompt_lengths():
    """Chat-templated first-human-turn token length per prompt = served prompt KV at
    decode step 0. Derived from the PPL ground-truth context (same 128 prompts), cut at
    the <start_of_turn>model\\n generation boundary; validated against a real capture."""
    served = {}
    with open(PPL_TOKENS) as fh:
        for line in fh:
            rec = json.loads(line)
            ctx = rec["context_token_ids"]
            pos = find_subsequence(ctx, GEN_PROMPT_MARKER)
            if pos < 0:
                raise ValueError(f"gen-prompt marker not found in context for id={rec['id']}")
            served[str(rec["id"])] = pos + len(GEN_PROMPT_MARKER)
    return served


def real_capture_lengths():
    if not os.path.exists(REAL_CAPTURE):
        return {}
    out = {}
    with open(REAL_CAPTURE) as fh:
        for line in fh:
            r = json.loads(line)
            out[str(r["id"])] = int(r["num_prompt_tokens"])
    return out


def piecewise_linear(xs, ys):
    """Piecewise-linear interpolant through sorted (xs, ys), edge-slope extrapolation."""
    pts = sorted(zip(xs, ys))

    def f(x):
        if x <= pts[0][0]:
            (x0, y0), (x1, y1) = pts[0], pts[1]
        elif x >= pts[-1][0]:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
        else:
            for k in range(len(pts) - 1):
                if pts[k][0] <= x <= pts[k + 1][0]:
                    (x0, y0), (x1, y1) = pts[k], pts[k + 1]
                    break
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(_here, "kv_weighted_strict_tps.json"))
    ap.add_argument("--selftest-output", default=os.path.join(_here, "selftest.json"))
    ap.add_argument("--sweep-json", default=SWEEP_JSON,
                    help="strict whole-cycle A/B sweep JSON (#479: the extended 6-point tail sweep)")
    ap.add_argument("--tail-repeat-jsons", default=None,
                    help="comma-separated extra sweep JSONs (#479 L=2048 session repeats) to "
                         "give the deepest tail anchor a between-session sigma")
    ap.add_argument("--tail-anchor-L", type=int, default=2048,
                    help="the deepest tail L whose anchor is averaged over the session repeats")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="equivalence-escalation-anchors")
    ap.add_argument("--wandb_name", default="stark/kv-weighted-strict-tps")
    ap.add_argument("--job_type", default="profiling")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    # ---- (2) #472/#479 measured per-L strict tax + the banked TPS mapping (imported exact) ----
    sweep = json.load(open(args.sweep_json))
    vd = sweep["verdict"]
    DEPLOYED_TPS = float(vd["deployed_tps"])      # 481.53 (perm arm == deployed, self-tax 0)
    CYCLE_PERM_US = float(vd["cycle_perm_us"])    # 7666.83 deployed permissive cycle
    per_L = {int(L): d for L, d in sweep["per_L"].items()}
    sweep_Ls = sorted(per_L)
    added_at = {L: float(per_L[L]["whole_delta_us"]) for L in sweep_Ls}
    tps_at = {L: float(per_L[L]["whole_strict_tps"]) for L in sweep_Ls}
    # within-session sigma of each measured strict tax point (paired-diff pstdev over rounds)
    added_sigma_within = {L: float(per_L[L].get("whole_delta_sigma", 0.0)) for L in sweep_Ls}

    # ---- (#479) deepest-tail anchor: average over the L=2048 session repeats so the tail
    # carries its own BETWEEN-session sigma, not a single draw. The central tail anchor becomes
    # the 3-draw mean; the band propagates the between-session pstdev of that anchor. ----
    tail_L = args.tail_anchor_L
    tail_draws = []
    if tail_L in added_at:
        tail_draws.append(added_at[tail_L])              # the primary sweep's draw (run A)
    repeat_files = [f for f in (args.tail_repeat_jsons or "").split(",") if f.strip()]
    for rf in repeat_files:
        rj = json.load(open(rf.strip()))
        rpl = {int(L): d for L, d in rj["per_L"].items()}
        if tail_L in rpl:
            tail_draws.append(float(rpl[tail_L]["whole_delta_us"]))
    tail_anchor_added_mean = statistics.mean(tail_draws) if tail_draws else float("nan")
    tail_anchor_added_sigma_between = (statistics.pstdev(tail_draws)
                                       if len(tail_draws) > 1 else 0.0)
    tail_anchor_n_sessions = len(tail_draws)
    if tail_L in added_at and tail_draws:
        added_at[tail_L] = tail_anchor_added_mean        # honest central tail anchor = 3-draw mean
        added_sigma_within[tail_L] = max(added_sigma_within.get(tail_L, 0.0),
                                         tail_anchor_added_sigma_between)

    def tps_from_added(added):
        return DEPLOYED_TPS * CYCLE_PERM_US / (CYCLE_PERM_US + added)

    added_pw = piecewise_linear(sweep_Ls, [added_at[L] for L in sweep_Ls])
    L_lo, L_hi = sweep_Ls[0], sweep_Ls[-1]        # 128, 2048 (#479 extended; was 128,640)
    # Bracket points pinned to fixed KV (NOT sweep endpoints): L=128 best-case, L=640 the #472
    # deployed-faithful worst-case headline. With the #479 tail extension L_hi is now 2048, so we
    # must evaluate the 640 worst-case explicitly rather than at the sweep endpoint.
    l128_bestcase_tps = tps_from_added(added_pw(128))
    l640_worstcase_tps = tps_from_added(added_pw(640))

    # ---- (1) realized KV trajectory: KV(j,i) = served_P_j + i, i in [0, OUTPUT_LEN) ----
    served = served_prompt_lengths()
    P = sorted(served.values())
    assert len(P) == NUM_PROMPTS, f"expected {NUM_PROMPTS} prompts, got {len(P)}"
    real = real_capture_lengths()
    common = set(served) & set(real)
    served_match = bool(common) and all(served[i] == real[i] for i in common)
    n_validated = len(common)

    # ---- (3) token-weighted harmonic aggregate over the trajectory ----------------------
    def harmonic(added_fn):
        inv = 0.0
        n = 0
        for p in P:
            for i in range(OUTPUT_LEN):
                inv += 1.0 / tps_from_added(added_fn(p + i))
                n += 1
        return n / inv, n

    kv_weighted_central, n_tok = harmonic(added_pw)                 # measured-tail tax (#479)
    kv_weighted_clamp, _ = harmonic(lambda L: added_pw(min(L, L_hi)))  # tax saturates > L_hi (upper)

    # ---- (#479) updated band: propagate the measured per-L strict-tax sigma (within-session for
    # head points, BETWEEN-session for the L=2048 anchor from the session repeats). Perturb every
    # measured anchor by +/- its sigma (correlated worst case) and re-aggregate. Now that the tail
    # is MEASURED to 2048 (not extrapolated), this measurement band REPLACES #475's wide
    # extrapolation band -- it should be tight if the tail tax is well-determined. ----
    added_hi = piecewise_linear(sweep_Ls, [added_at[L] + added_sigma_within[L] for L in sweep_Ls])
    added_lo = piecewise_linear(sweep_Ls, [added_at[L] - added_sigma_within[L] for L in sweep_Ls])
    band_meas_lo, _ = harmonic(added_hi)     # more tax -> LOWER tps bound
    band_meas_hi, _ = harmonic(added_lo)     # less tax -> HIGHER tps bound

    # trajectory descriptive stats + tail coverage
    kv_sum = sum(p + i for p in P for i in range(OUTPUT_LEN))
    kv_trajectory_mean_L = kv_sum / n_tok
    tps_at_mean_kv = tps_from_added(added_pw(kv_trajectory_mean_L))
    over_640 = sum(1 for p in P for i in range(OUTPUT_LEN) if (p + i) > 640)   # literal 640, NOT L_hi (=2048 since #479 extended the sweep)
    over_1024 = sum(1 for p in P for i in range(OUTPUT_LEN) if (p + i) > 1024)
    frac_kv_gt_640 = over_640 / n_tok
    frac_kv_gt_1024 = over_1024 / n_tok
    kv_max = max(P) + (OUTPUT_LEN - 1)

    # ---- verdict ----
    # #475's band was the model-uncertainty between linear-extrapolated and clamped-@640 tax (wide,
    # because everything past L=640 was extrapolated). #479 MEASURES the tail to L=2048, so that
    # extrapolation band collapses; the honest remaining uncertainty is the measurement sigma of
    # the per-L tax (propagated above) plus the small >2048 saturation sliver (clamp@2048).
    updated_band_low = min(band_meas_lo, kv_weighted_central)
    updated_band_high = max(band_meas_hi, kv_weighted_clamp)
    band_lo, band_hi = updated_band_low, updated_band_high
    band_str = f"[{band_lo:.2f}, {band_hi:.2f}]"
    tail_kv_weighted_tps = kv_weighted_central          # PRIMARY: the updated honest center (#479)
    uplift_vs_worst = kv_weighted_central - l640_worstcase_tps
    weighted_above_worst = bool(kv_weighted_central > l640_worstcase_tps)
    materially_above = bool(uplift_vs_worst >= SIGMA_HW_BETWEEN)
    # how far the tail-measured center moved from #475's extrapolated 461.80
    shift_vs_475_extrapolation = kv_weighted_central - 461.80

    # ---- (#479) harness-derived tail-tax metrics (read from the extended sweep verdict) ----
    tail_tax_is_linear = bool(vd.get("tail_tax_is_linear", False))
    tail_tax_slope_ratio = float(vd.get("tail_tax_slope_ratio", float("nan")))
    tail_tax_regime = str(vd.get("tail_tax_regime", "unknown"))
    head_slope_us_per_tok = float(vd.get("head_slope_us_per_tok", float("nan")))
    tail_slope_us_per_tok = float(vd.get("tail_slope_us_per_tok", float("nan")))
    l896_tps = float(vd.get("l896_tps", float("nan")))
    l1280_tps = float(vd.get("l1280_tps", float("nan")))
    l2048_tps_runA = float(vd.get("l2048_tps", float("nan")))
    l2048_tps = tps_from_added(tail_anchor_added_mean)   # 3-session-mean tail anchor -> tps
    l2048_identity = float(vd.get("l2048_identity", float("nan")))
    if tail_L in tps_at:
        tps_at[tail_L] = l2048_tps   # keep the per-L table consistent with the averaged anchor

    # ---- self-test ----
    st = {}
    st["served_lengths_match_real_capture"] = bool(served_match and n_validated == NUM_PROMPTS)
    st["n_tokens_is_128x512"] = bool(n_tok == NUM_PROMPTS * OUTPUT_LEN)
    st["tps_zero_added_is_deployed"] = bool(abs(tps_from_added(0.0) - DEPLOYED_TPS) < 1e-6)
    st["tps_reproduces_measured_L"] = all(
        abs(tps_from_added(added_at[L]) - tps_at[L]) < 0.05 for L in sweep_Ls)
    st["weighted_between_best_and_worst"] = bool(
        l640_worstcase_tps - 0.5 <= kv_weighted_central <= l128_bestcase_tps + 0.5)
    st["mean_L_between_384_and_640"] = bool(384.0 < kv_trajectory_mean_L < 640.0)
    # harmonic == tps@meanKV iff the tax is AFFINE in KV over the trajectory; the gap is a pure
    # convexity (non-linearity) detector. A LARGE gap would itself flag a super-linear tail.
    st["harmonic_matches_mean_kv_point"] = bool(abs(kv_weighted_central - tps_at_mean_kv) < 0.75)
    st["band_brackets_central"] = bool(band_lo <= kv_weighted_central <= band_hi)
    st["updated_band_orders"] = bool(updated_band_low <= kv_weighted_central <= updated_band_high)
    st["added_monotone_increasing"] = bool(
        added_pw(128) < added_pw(384) < added_pw(640) < added_pw(1200) < added_pw(2048))
    st["tail_measured_to_2048"] = bool(max(sweep_Ls) >= 2048 and 896 in sweep_Ls and 1280 in sweep_Ls)
    # KV-coverage invariant: {KV>1024} subset of {KV>640}, and the >640 fraction must match the
    # banked 24.2% (guards the L_hi-drift bug: over_640 must threshold on literal 640, not L_hi).
    st["kv_frac_monotone_and_anchored"] = bool(
        frac_kv_gt_640 >= frac_kv_gt_1024 and abs(frac_kv_gt_640 - 0.242) < 0.03)
    st["ppl_anchor_ok"] = bool(PPL_ANCHOR <= PPL_GATE)
    finite = [kv_weighted_central, kv_weighted_clamp, kv_trajectory_mean_L, tps_at_mean_kv,
              l128_bestcase_tps, l640_worstcase_tps, uplift_vs_worst,
              updated_band_low, updated_band_high]
    st["nan_clean"] = all(math.isfinite(x) for x in finite)
    self_test_passes = all(st.values())

    verdict = {
        # ---- (#479) PRIMARY: the updated honest center with the MEASURED tail (not extrapolated) ----
        "tail_kv_weighted_tps": tail_kv_weighted_tps,              # PRIMARY (#479)
        "updated_band_low": updated_band_low,
        "updated_band_high": updated_band_high,
        "shift_vs_475_extrapolation": shift_vs_475_extrapolation,  # how far the center moved from 461.80
        "tail_tax_is_linear": tail_tax_is_linear,
        "tail_tax_slope_ratio": tail_tax_slope_ratio,
        "tail_tax_regime": tail_tax_regime,
        "head_slope_us_per_tok": head_slope_us_per_tok,
        "tail_slope_us_per_tok": tail_slope_us_per_tok,
        "l896_tps": l896_tps,
        "l1280_tps": l1280_tps,
        "l2048_tps": l2048_tps,
        "l2048_tps_runA": l2048_tps_runA,
        "l2048_identity": l2048_identity,
        "tail_anchor_added_mean_us": tail_anchor_added_mean,
        "tail_anchor_added_sigma_between_us": tail_anchor_added_sigma_between,
        "tail_anchor_n_sessions": tail_anchor_n_sessions,
        "band_meas_lo": band_meas_lo, "band_meas_hi": band_meas_hi,
        # ---- carried from #475 (now tail-informed) ----
        "kv_weighted_strict_tps": kv_weighted_central,             # == tail_kv_weighted_tps
        "kv_weighted_strict_tps_clamp2048": kv_weighted_clamp,
        "predicted_official_tps_band_lo": band_lo,
        "predicted_official_tps_band_hi": band_hi,
        "predicted_official_tps_band": band_str,
        "kv_trajectory_mean_L": kv_trajectory_mean_L,
        "tps_at_mean_kv": tps_at_mean_kv,
        "l128_bestcase_tps": l128_bestcase_tps,
        "l640_worstcase_tps": l640_worstcase_tps,
        "l640_worstcase_headline": 457.55,
        "uplift_vs_l640_worstcase": uplift_vs_worst,
        "uplift_band_lo": band_lo - l640_worstcase_tps,
        "uplift_band_hi": band_hi - l640_worstcase_tps,
        "weighted_above_L640_worstcase": weighted_above_worst,
        "materially_above_worstcase": materially_above,
        "sigma_hw_between_session": SIGMA_HW_BETWEEN,
        "frac_decode_tokens_kv_gt_640": frac_kv_gt_640,
        "frac_decode_tokens_kv_gt_1024": frac_kv_gt_1024,
        "kv_trajectory_max": kv_max,
        "served_prompt_mean": statistics.mean(P),
        "served_prompt_median": statistics.median(P),
        "served_prompt_min": min(P),
        "served_prompt_max": max(P),
        "served_lengths_validated_n": n_validated,
        "served_lengths_match_real_capture": bool(served_match),
        "num_prompts": NUM_PROMPTS,
        "output_len": OUTPUT_LEN,
        "n_decode_tokens": n_tok,
        "deployed_tps": DEPLOYED_TPS,
        "cycle_perm_us": CYCLE_PERM_US,
        "ppl": PPL_ANCHOR, "ppl_anchor": PPL_ANCHOR, "ppl_gate": PPL_GATE,
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
        "no_kernel_rebuild": True, "no_hf_job": True,
        "self_test_passes": self_test_passes,
    }

    reconcile = (
        f"(#479) Official summary.json:tps = total_output_tokens/total_wall_clock over the 128 "
        f"benchmark prompts at output_len={OUTPUT_LEN}, single-stream. The realized KV trajectory "
        f"KV=served_P+i has mean {kv_trajectory_mean_L:.1f} (served prompts mean "
        f"{statistics.mean(P):.1f}, median {int(statistics.median(P))}, max {max(P)}); "
        f"{100*frac_kv_gt_640:.1f}% of decode tokens have KV>640 (up to KV={kv_max}) -- the fat "
        f"tail #475 had to EXTRAPOLATE. #479 MEASURES the strict tax to L=2048 (grid "
        f"{sorted(sweep_Ls)}): tail-tax regime={tail_tax_regime} (slope ratio "
        f"{tail_tax_slope_ratio:.3f}, head {head_slope_us_per_tok:.4f} vs tail "
        f"{tail_slope_us_per_tok:.4f} us/tok; linear={tail_tax_is_linear}). L=2048 anchor = mean "
        f"of {tail_anchor_n_sessions} sessions ({tail_anchor_added_mean:.1f}us, between-session "
        f"sigma {tail_anchor_added_sigma_between:.2f}us) -> l2048_tps={l2048_tps:.2f}, identity "
        f"{l2048_identity:.4f}. Token-weighted HARMONIC over the MEASURED 6-point tax: "
        f"tail_kv_weighted_tps={tail_kv_weighted_tps:.2f} (= tps@meanKV {tps_at_mean_kv:.2f}; "
        f"shift vs #475's extrapolated 461.80 = {shift_vs_475_extrapolation:+.2f}); updated band "
        f"{band_str} (was #475's wider extrapolation band -- now collapsed to the measurement "
        f"sigma). vs L=640 worst-case {l640_worstcase_tps:.2f} (headline 457.55) and L=128 "
        f"best-case {l128_bestcase_tps:.2f}: uplift +{uplift_vs_worst:.2f}; "
        f"weighted_above_worstcase={weighted_above_worst}, materially_above (>= sigma_hw "
        f"{SIGMA_HW_BETWEEN})={materially_above}. Served lengths bit-match real capture "
        f"({n_validated}/{NUM_PROMPTS}). analysis_only, official_tps=0, no served change, no HF Job.")
    verdict["reconcile_line"] = reconcile

    payload = {
        "config": {
            "num_prompts": NUM_PROMPTS, "output_len": OUTPUT_LEN,
            "sweep_json": os.path.relpath(args.sweep_json, _root),
            "tail_repeat_jsons": [os.path.relpath(f.strip(), _root) for f in repeat_files],
            "ppl_tokens": os.path.relpath(PPL_TOKENS, _root),
            "real_capture": os.path.relpath(REAL_CAPTURE, _root),
            "sweep_Ls": sweep_Ls, "added_us_at_L": {str(L): added_at[L] for L in sweep_Ls},
            "tps_at_L": {str(L): tps_at[L] for L in sweep_Ls},
            "gen_prompt_marker": list(GEN_PROMPT_MARKER),
            "interp": "piecewise_linear_added_us_edge_slope_extrap",
            "sigma_hw_between_session": SIGMA_HW_BETWEEN,
            "tail_anchor_L": tail_L, "tail_anchor_n_sessions": tail_anchor_n_sessions,
            "note": "KV-weighted strict TPS, #479 tail-extended: token-weighted harmonic mean of "
                    "the strict tax MEASURED to L=2048 (#479 extends #472's 3-point sweep to 6 "
                    "points, replacing the >640 extrapolation) over the official 128-prompt "
                    "output_len=512 single-stream KV trajectory. The L=2048 anchor is the mean of "
                    "2-3 fresh-process session repeats (between-session sigma). CPU analysis only; "
                    "no kernel re-measure, no served change, no HF Job.",
        },
        "trajectory": {
            "served_prompt_lengths_sorted": P,
            "served_prompt_hist": _hist(P),
            "kv_hist": _kv_hist(P, OUTPUT_LEN),
        },
        "identity_per_L": sweep.get("identity_per_L", {}),
        "verdict": verdict,
        "self_test_conditions": st,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, (int, float)) else str(o))
    json.dump({"self_test_passes": self_test_passes, "checks": st},
              open(args.selftest_output, "w"), indent=2)

    print(f"[kvw] tail_kv_weighted_tps={tail_kv_weighted_tps:.2f} band {band_str} "
          f"(shift vs #475 461.80 = {shift_vs_475_extrapolation:+.2f}) | meanKV={kv_trajectory_mean_L:.1f} "
          f"| worst(L640)={l640_worstcase_tps:.2f} best(L128)={l128_bestcase_tps:.2f}", flush=True)
    print(f"[kvw] TAIL-TAX: regime={tail_tax_regime} slope_ratio={tail_tax_slope_ratio:.3f} "
          f"linear={tail_tax_is_linear} | l896={l896_tps:.2f} l1280={l1280_tps:.2f} "
          f"l2048={l2048_tps:.2f}(n={tail_anchor_n_sessions},sig={tail_anchor_added_sigma_between:.2f}us) "
          f"id@2048={l2048_identity:.4f}", flush=True)
    print(f"[kvw] uplift_vs_worst=+{uplift_vs_worst:.2f} (sigma_hw={SIGMA_HW_BETWEEN}) "
          f"above_worst={weighted_above_worst} materially_above={materially_above} "
          f"| kv>640: {100*frac_kv_gt_640:.1f}% of tokens (max KV {kv_max})", flush=True)
    print(f"[kvw] served lengths match real capture: {n_validated}/{NUM_PROMPTS}  "
          f"self_test={self_test_passes}", flush=True)
    print(f"[kvw] {reconcile}", flush=True)
    print(f"[kvw] wrote {args.output}", flush=True)

    if not args.no_wandb:
        _log_wandb(args, payload)
    return 0 if self_test_passes else 1


def _hist(P):
    buckets = [(0, 128), (128, 256), (256, 384), (384, 512), (512, 640),
               (640, 1024), (1024, 2048), (2048, 4096)]
    return {f"[{lo},{hi})": sum(1 for x in P if lo <= x < hi) for lo, hi in buckets}


def _kv_hist(P, out_len):
    edges = [0, 128, 256, 384, 512, 640, 768, 1024, 1536, 2048, 4096]
    h = {f"[{edges[k]},{edges[k+1]})": 0 for k in range(len(edges) - 1)}
    for p in P:
        for i in range(out_len):
            kv = p + i
            for k in range(len(edges) - 1):
                if edges[k] <= kv < edges[k + 1]:
                    h[f"[{edges[k]},{edges[k+1]})"] += 1
                    break
    return h


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type=args.job_type, config=payload.get("config", {}))
    vd = payload["verdict"]
    run.summary.update({k: v for k, v in vd.items() if isinstance(v, (int, float, bool, str))})

    cfg = payload["config"]
    idpl = payload.get("identity_per_L", {})
    t = wandb.Table(columns=["L", "added_us", "strict_tps", "strict_byte_identity", "strict_token_flips"])
    for L in cfg["sweep_Ls"]:
        idl = idpl.get(str(L), {})
        t.add_data(L, cfg["added_us_at_L"][str(L)], cfg["tps_at_L"][str(L)],
                   idl.get("strict_byte_identity_min", float("nan")),
                   idl.get("strict_token_flips", -1))
    run.log({"per_L_strict_tax": t})

    kvh = payload["trajectory"]["kv_hist"]
    kt = wandb.Table(columns=["kv_bucket", "decode_tokens"])
    for k, v in kvh.items():
        kt.add_data(k, v)
    run.log({"kv_trajectory_hist": kt})

    ph = payload["trajectory"]["served_prompt_hist"]
    pt = wandb.Table(columns=["prompt_len_bucket", "n_prompts"])
    for k, v in ph.items():
        pt.add_data(k, v)
    run.log({"served_prompt_hist": pt})
    run.finish()
    print(f"[kvw] logged W&B run {args.wandb_entity}/{args.wandb_project} "
          f"name={args.wandb_name} group={args.wandb_group}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
