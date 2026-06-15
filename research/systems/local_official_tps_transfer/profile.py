#!/usr/bin/env python
"""PR #267 — Local->official TPS transfer calibration (bank-the-analysis leg).

The launch gate is >=500 OFFICIAL TPS, but the whole fleet now measures
LOCALLY. The cleanest anchor we have is lawine #246: the EXACT deployed linear
submission (``fa2sw_precache_kenyan``, ONEGRAPH=1) that scores 481.53 OFFICIAL
reads 465.14 TPS LOCALLY (warm, CV 0.008%, PPL 2.3767, identical served config,
run ``0qc5lk4y``). So the same byte-identical submission shows a systematic
-16.39 TPS / -3.40% LOCAL-vs-OFFICIAL offset. This leg characterises WHY, and
delivers the transfer factor ``tau_lo = 481.53/465.14`` plus the rule the fleet
needs: to clear the official 500 gate a LOCAL build must read >= 500/tau_lo.

It changes NOTHING served: no submission edit, no HF Job, no submission. PPL and
greedy identity are properties of the model+config, which are unchanged
(``transfer_calibration_analysis_only=True``). 0 TPS; BASELINE stays 481.53.

Two parts:

  * ANALYTIC CORE + SELF-TEST (PRIMARY, no GPU). Imports the #246 / #52 anchors
    and the kanna #217 composition EXACTLY (do not re-derive), computes
    ``tau_lo``, the additive gap, the decomposition and the local->official map,
    and validates the round-trips. ``--self-test`` exits non-zero unless
    ``local_official_tps_transfer_self_test_passes``.

  * MEASURED DECOMPOSITION (local A10G). Re-measures the deployed path with the
    timing boundary matched to the official ``summary.json:tps`` definition. The
    KEY fact (scripts traced in this PR): the official ``tps`` is
    ``sglang.bench_serving`` ``output_throughput`` (conc=1, 128x512, seed=1, 4
    warmup requests DISCARDED, prompts pre-tokenized OUTSIDE the timer) while the
    local #246 ``wall_tps`` came from the official ``decode_outputs.py`` capture,
    whose timed loop also pays per-prompt CLIENT tokenization + per-request JSON
    hash/write/flush over ALL 128 prompts. The instrumented re-measure times
    tokenization / request / IO separately, reconstructs the official boundary on
    LOCAL hardware, and so splits the gap into HARNESS (boundary vs wall) and
    HARDWARE/CLOCK (official vs local boundary), with WARMUP and METHODOLOGY as
    sub/residual terms.

Reproduce (analytic, fast):
  CUDA_VISIBLE_DEVICES=0 python research/systems/local_official_tps_transfer/profile.py \
    --self-test --wandb_group local-official-transfer \
    --wandb_name lawine/local-official-tps-transfer
Add ``--measure`` to serve the deployed submission and bank the measured split.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
# This file is named profile.py. When it is run as a script (orchestrator or the
# --decode-worker re-invocation), Python auto-prepends its OWN directory to
# sys.path[0]. The decode worker then imports transformers, whose GenerationMixin
# pulls in torch._dynamo, whose convert_frame does `import cProfile`, whose stdlib
# source does `import profile` — which would resolve to THIS module and blow up
# with "module 'profile' has no attribute 'run'". Drop the script dir (and the
# bare cwd entry) so stdlib `profile` wins; ROOT is re-added explicitly below for
# `from scripts.local_validation import ...`.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Imported anchors — DO NOT re-derive (PR #267 instruction). Edit = self-test
# constant guard FAILS (check f).
# --------------------------------------------------------------------------- #
# lawine #246 (0qc5lk4y): deployed-config LOCAL warm anchor.
LOCAL_ANCHOR_TPS = 465.14047160458415       # wall_tps_control_median (deployed cfg, warm)
LOCAL_ANCHOR_CV_PCT = 0.007767827137263834  # warm control CV (within-config noise floor)
LOCAL_ANCHOR_PPL = 2.376682786480556
LOCAL_ANCHOR_VRAM_GB = 20.8935546875
# Official anchor — PR #52 fa2sw_precache_kenyan, 128/128.
OFFICIAL_ANCHOR_TPS = 481.53
OFFICIAL_PPL = 2.3772
# Third number: private-verified (private-official). KEEP DISTINCT from the
# local->official factor this leg owns; do not conflate.
PRIVATE_VERIFIED_TPS = 460.85
# lambda=1 ceiling (imported, unchanged).
CEILING_LAMBDA1_TPS = 520.95
# kanna #217 (vgovdrjc) composition: official ~= K_cal * E[T]; served step (ms).
K_CAL = 125.268
E_T = 3.844
SERVED_STEP_MS = 1.2182
# lawine #72 wall_tps run-to-run CV — a broader sigma_hw-like envelope than the
# #246 within-config CV.
RUN2RUN_CV_PCT = 0.035
# Official launch gate.
OFFICIAL_GATE_TPS = 500.0
# stark #256/#266 adaptive-K PROJECTED LOCAL TPS (PR #267 instruction 4 quotes
# this exact figure; imported as a PR-given number, not re-derived here). Used
# only to APPLY the local->official map to a live lever.
STARK_PROJECTED_LOCAL_TPS = 545.14

SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
OUT_ROOT = ROOT / "research" / "systems" / "local_official_tps_transfer"
PPL_CAP = 2.42
VRAM_CAP_GB = 24.0
NUM_PROMPTS = 128
OFFICIAL_OUTPUT_LEN = 512
# official sglang.bench_serving discards this many warmup requests from timing.
WARMUP_REQUESTS = 4
# Fleet E[T] range to probe tau_lo stability over (baseline 3.844 .. aggressive
# adaptive-K target). Only used for the analytic stability sweep.
ET_RANGE = (3.5, 4.5)


# ========================================================================== #
# Analytic core
# ========================================================================== #
def compute_transfer() -> dict[str, Any]:
    tau_lo = OFFICIAL_ANCHOR_TPS / LOCAL_ANCHOR_TPS
    additive_gap = OFFICIAL_ANCHOR_TPS - LOCAL_ANCHOR_TPS
    return {
        "local_anchor_tps": LOCAL_ANCHOR_TPS,
        "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
        "tau_lo": tau_lo,
        "additive_gap_tps": additive_gap,
        # The PR quotes -3.40% (official-relative); (tau_lo-1) is the
        # local-relative form. Report both, label clearly.
        "gap_pct_official_relative": 100.0 * additive_gap / OFFICIAL_ANCHOR_TPS,
        "gap_pct_local_relative": 100.0 * additive_gap / LOCAL_ANCHOR_TPS,
        "local_bar_for_official_500": OFFICIAL_GATE_TPS / tau_lo,
        "private_verified_tps": PRIVATE_VERIFIED_TPS,
        "ceiling_lambda1_tps": CEILING_LAMBDA1_TPS,
    }


def mapping_rule_application(transfer: dict[str, Any]) -> dict[str, Any]:
    """Apply the local->official map to the live lever (PR #267 instruction 4).

    The rule: official ~= tau_lo * local, so a LOCAL build must read
    >= 500/tau_lo to clear the official >=500 gate. We apply it to the one
    concrete number the PR provides — stark's adaptive-K projected LOCAL 545.14
    — and report whether it clears the local bar (and thus the official gate).
    """
    tau_lo = transfer["tau_lo"]
    local_bar = transfer["local_bar_for_official_500"]
    stark_local = STARK_PROJECTED_LOCAL_TPS
    stark_official = tau_lo * stark_local
    return {
        "rule": "official ~= tau_lo * local; local must read >= 500/tau_lo to clear official 500",
        "local_bar_for_official_500": local_bar,
        "stark266_projected_local_tps": stark_local,
        "stark266_implied_official_tps": stark_official,
        "stark266_clears_local_bar": stark_local >= local_bar,
        "stark266_clears_official_500": stark_official >= OFFICIAL_GATE_TPS,
        "stark266_margin_over_local_bar_tps": stark_local - local_bar,
        # the lambda=1 ceiling mapped the same way, for context (NOT a target).
        "ceiling_lambda1_local_equiv": CEILING_LAMBDA1_TPS / tau_lo,
    }


def build_decomposition(measured: dict[str, Any] | None) -> dict[str, Any]:
    """Attribute additive_gap = OFFICIAL - LOCAL across {harness, hardware}.

    The split telescopes through the official boundary measured on LOCAL
    hardware (``official_boundary_tps_local``):

        harness  = boundary - LOCAL      (tokenization + IO + warmup-inclusion:
                                          the local wall metric's overhead the
                                          official output_throughput excludes)
        hardware = OFFICIAL - boundary   (local A10G vs official A10G, SAME
                                          boundary; + sglang-vs-urllib residual)
        harness + hardware == OFFICIAL - LOCAL   (identity, any boundary value)

    Without a measurement we cannot place the boundary, so we report the gap as
    UNATTRIBUTED (boundary = LOCAL => all of it pending measurement) and flag
    ``measured=False`` — the self-test sum-check still holds by construction.
    """
    gap = OFFICIAL_ANCHOR_TPS - LOCAL_ANCHOR_TPS
    if measured and measured.get("official_boundary_tps_local") is not None:
        boundary = float(measured["official_boundary_tps_local"])
        wall_remeasure = float(measured.get("wall_tps_local_remeasure", LOCAL_ANCHOR_TPS))
        harness = boundary - LOCAL_ANCHOR_TPS
        hardware = OFFICIAL_ANCHOR_TPS - boundary
        # Sub-attribute the harness component by the measured overhead-time
        # fractions (tokenization vs file IO vs the warmup-inclusion drag).
        sub = measured.get("overhead_time_split_s") or {}
        sub_total = sum(v for v in sub.values() if isinstance(v, (int, float))) or None
        harness_sub = {}
        if sub_total:
            for k, v in sub.items():
                harness_sub[k] = harness * (float(v) / sub_total)
        return {
            "measured": True,
            "official_boundary_tps_local": boundary,
            "wall_tps_local_remeasure": wall_remeasure,
            "components_tps": {
                "harness_tokenization_io_warmup": harness,
                "hardware_clock_plus_methodology": hardware,
            },
            "harness_subsplit_tps": harness_sub,
            "harness_pct_of_gap": 100.0 * harness / gap if gap else None,
            "hardware_pct_of_gap": 100.0 * hardware / gap if gap else None,
            "dominant": (
                "harness" if abs(harness) >= abs(hardware) else "hardware"
            ),
            "sum_check_tps": harness + hardware,
            "gap_tps": gap,
        }
    return {
        "measured": False,
        "official_boundary_tps_local": None,
        "components_tps": {
            "harness_tokenization_io_warmup": None,
            "hardware_clock_plus_methodology": None,
        },
        "unattributed_gap_tps": gap,
        "sum_check_tps": gap,
        "gap_tps": gap,
        "note": "no local measurement available; run with --measure to attribute",
    }


def tau_lo_stability(measured: dict[str, Any] | None) -> dict[str, Any]:
    """Is tau_lo a stable scalar, or regime-dependent?

    The decomposition fixes the SHAPE of any regime-dependence:

      * a HARDWARE/clock component is a multiplicative ratio -> contributes a
        tau_lo term that is CONSTANT across E[T]/decode-length (a stable scalar);
      * a HARNESS fixed-overhead component (per-run client tokenization + IO,
        independent of accept length) makes tau_lo DRIFT with E[T]: higher E[T]
        -> fewer spec steps -> less generation time -> the fixed overhead is a
        LARGER fraction -> tau_lo rises. With generation time T_gen ~ 1/E[T],

            tau_lo(E[T]) = h_factor * (1 + f0 * E[T]/E_T0)

        where h_factor is the (E[T]-independent) hardware ratio and f0 is the
        harness-overhead fraction at the E_T0=3.844 operating point. If harness
        dominates (h_factor~1) tau_lo is mildly regime-dependent; if hardware
        dominates (f0~0) tau_lo is a stable scalar.

    We anchor f0 and h_factor from the measured split when available; otherwise
    we report the *envelope* assuming the gap is entirely harness (max drift) vs
    entirely hardware (zero drift), which BRACKETS the truth.
    """
    tau0 = OFFICIAL_ANCHOR_TPS / LOCAL_ANCHOR_TPS
    et_lo, et_hi = ET_RANGE

    def drift_model(f0: float, h_factor: float) -> dict[str, Any]:
        def tau_at(et: float) -> float:
            return h_factor * (1.0 + f0 * et / E_T)
        taus = {f"E[T]={et:.3f}": tau_at(et) for et in (et_lo, E_T, et_hi)}
        spread = tau_at(et_hi) - tau_at(et_lo)
        # local bar to clear official 500 at each E[T]
        bars = {k: OFFICIAL_GATE_TPS / v for k, v in taus.items()}
        return {
            "f0_harness_overhead_frac": f0,
            "h_factor_hardware_ratio": h_factor,
            "tau_lo_at": taus,
            "tau_lo_spread_over_ET_range": spread,
            "tau_lo_spread_pct": 100.0 * spread / tau0,
            "local_bar_for_500_at": bars,
        }

    if measured and measured.get("official_boundary_tps_local") is not None:
        boundary = float(measured["official_boundary_tps_local"])
        # boundary = LOCAL * (1 + f0) on local hw  -> f0 = boundary/LOCAL - 1
        f0 = boundary / LOCAL_ANCHOR_TPS - 1.0
        # official = boundary * h_factor          -> h_factor = OFFICIAL/boundary
        h_factor = OFFICIAL_ANCHOR_TPS / boundary
        model = drift_model(f0, h_factor)
        # tau_lo is "stable" if the drift over the fleet's E[T] range is small
        # relative to the noise floor we make decisions at (<~0.5%).
        model["tau_lo_stable"] = abs(model["tau_lo_spread_pct"]) < 0.5
        model["basis"] = "measured-split"
        return model
    # Envelope: bracket harness-only vs hardware-only.
    f0_all = tau0 - 1.0  # gap entirely harness fixed-overhead
    harness_only = drift_model(f0_all, 1.0)
    hardware_only = drift_model(0.0, tau0)
    spread = harness_only["tau_lo_spread_pct"]
    return {
        "basis": "envelope-bracket",
        "harness_only_max_drift": harness_only,
        "hardware_only_zero_drift": hardware_only,
        "tau_lo_spread_pct_max": spread,
        "tau_lo_stable": abs(spread) < 0.5,
        "note": "run --measure to collapse the bracket to the measured split",
    }


def build_self_test(transfer: dict[str, Any], decomp: dict[str, Any]) -> dict[str, Any]:
    """PRIMARY: local_official_tps_transfer_self_test_passes.

    (a) tau_lo * LOCAL == OFFICIAL                 (round-trip, resid <= 1e-3)
    (b) decomposition components sum to the gap     (telescoping, tol stated)
    (c) systematic offset >> sigma_hw variance      (cite #246 CV 0.008%)
    (d) local_bar_for_500 = 500/tau_lo round-trips  (resid <= 1e-3)
    (e) NaN-clean
    (f) constants imported EXACTLY (481.53 / 520.95 / 125.268 + step/E[T])
    """
    tau_lo = transfer["tau_lo"]
    st: dict[str, Any] = {}

    # (a) tau_lo round-trip
    resid_a = abs(tau_lo * LOCAL_ANCHOR_TPS - OFFICIAL_ANCHOR_TPS)
    st["tau_lo_roundtrip_resid"] = resid_a
    st["tau_lo_roundtrip_ok"] = resid_a <= 1e-3

    # (b) components sum to gap (telescoping). tol: exact for analytic, a small
    # measurement tol when a measured split is present.
    sum_tps = decomp.get("sum_check_tps")
    gap = transfer["additive_gap_tps"]
    tol_b = 1e-6 if not decomp.get("measured") else 1e-3
    resid_b = abs(float(sum_tps) - gap)
    st["decomp_sum_tol"] = tol_b
    st["decomp_sum_resid"] = resid_b
    st["decomp_sum_ok"] = resid_b <= tol_b

    # (c) systematic offset distinguished from sigma_hw run-to-run variance. The
    # offset is 16.39 TPS; the within-config noise is CV 0.008% (#246) and the
    # broader run-to-run envelope CV 0.035% (#72). Even the broader band is
    # ~0.16 TPS, two orders below the offset.
    noise_within = LOCAL_ANCHOR_TPS * LOCAL_ANCHOR_CV_PCT / 100.0
    noise_run2run = LOCAL_ANCHOR_TPS * RUN2RUN_CV_PCT / 100.0
    st["offset_tps"] = gap
    st["noise_within_config_tps"] = noise_within
    st["noise_run2run_tps"] = noise_run2run
    # require the offset to clear 10x the broader noise envelope.
    st["systematic_offset_margin_x"] = gap / noise_run2run if noise_run2run else math.inf
    st["systematic_offset_ok"] = gap > 10.0 * noise_run2run

    # (d) local bar round-trip
    local_bar = transfer["local_bar_for_official_500"]
    resid_d = abs(tau_lo * local_bar - OFFICIAL_GATE_TPS)
    st["local_bar_roundtrip_resid"] = resid_d
    st["local_bar_roundtrip_ok"] = resid_d <= 1e-3

    # (e) NaN-clean over every headline float
    floats = [
        tau_lo, gap, local_bar, transfer["gap_pct_official_relative"],
        transfer["gap_pct_local_relative"], resid_a, resid_b, resid_d,
        noise_within, noise_run2run,
    ]
    st["nan_clean_ok"] = all(isinstance(x, float) and math.isfinite(x) for x in floats)

    # (f) constants imported exactly (guard against accidental edits)
    st["constants_ok"] = (
        OFFICIAL_ANCHOR_TPS == 481.53
        and CEILING_LAMBDA1_TPS == 520.95
        and K_CAL == 125.268
        and E_T == 3.844
        and SERVED_STEP_MS == 1.2182
        and LOCAL_ANCHOR_TPS == 465.14047160458415
    )

    st["passes"] = bool(
        st["tau_lo_roundtrip_ok"]
        and st["decomp_sum_ok"]
        and st["systematic_offset_ok"]
        and st["local_bar_roundtrip_ok"]
        and st["nan_clean_ok"]
        and st["constants_ok"]
    )
    return st


def composition_cross_check(measured: dict[str, Any] | None) -> dict[str, Any]:
    """Hardware/clock leg via the kanna #217 composition, independent of the
    telescoping decomposition.

    E[T] is a MODEL property (greedy tokens are byte-identical local vs official
    -> identical accept length) and K_cal is a fixed calibration, so in
    ``official_tps = K_cal * E[T]`` the only term that can differ local vs
    official is the per-ACCEPT-CYCLE time (one draft-K=7 + verify + accept-E[T]
    pass) — i.e. the hardware/clock. The official accept-cycle implied by the
    composition is ``1000 * E[T] / (K_cal*E[T]) = 1000/K_cal`` ms. The
    boundary-implied LOCAL accept-cycle (``local_step_ms_implied``, request-time
    only so harness is already stripped) is compared against it: the excess is
    the hardware/clock component, and it must independently reproduce the ~3%
    hardware leg of the telescoping split.

    NOTE ``SERVED_STEP_MS`` (1.2182 ms, kanna #217) is the per-forward-pass time,
    NOT the accept cycle, so it is reported as raw context only — comparing the
    accept cycle against it would be a units mismatch (~6.7x), not a hardware
    signal.
    """
    official_cycle_ms = 1000.0 / K_CAL  # == 1000*E_T/(K_cal*E_T)
    out: dict[str, Any] = {
        "k_cal": K_CAL,
        "e_t": E_T,
        "k_cal_times_e_t": K_CAL * E_T,  # ~= 481.53 sanity
        "official_accept_cycle_ms": official_cycle_ms,
        "served_forward_step_ms": SERVED_STEP_MS,  # per-forward-pass; context only
    }
    if measured and measured.get("local_step_ms_implied") is not None:
        local_cycle = float(measured["local_step_ms_implied"])
        out["local_accept_cycle_ms"] = local_cycle
        out["cycle_excess_ms"] = local_cycle - official_cycle_ms
        out["cycle_excess_pct"] = 100.0 * (local_cycle - official_cycle_ms) / official_cycle_ms
        out["hardware_clock_implied"] = local_cycle > official_cycle_ms
    return out


# ========================================================================== #
# Measurement — instrumented local re-measure
# ========================================================================== #
def _decode_worker(args: argparse.Namespace) -> int:
    """Run UNDER the server venv (has transformers). Faithful to the official
    decode_outputs.py timed loop, but times tokenization / request / IO per
    request so the official output_throughput boundary can be reconstructed."""
    import importlib.util
    import urllib.error

    from scripts.local_validation import paths  # noqa: E402

    spec = importlib.util.spec_from_file_location("official_decode", str(paths.DECODE_SCRIPT))
    od = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(od)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = od.read_sharegpt_prompts(Path(args.dataset_path), num_prompts=args.num_prompts, seed=args.seed)
    if len(records) != args.num_prompts:
        raise ValueError(f"expected {args.num_prompts} prompts, found {len(records)}")

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    token_file = out_file.with_suffix(".tokens.jsonl")

    rows: list[dict[str, Any]] = []
    total_completion = 0
    total_prompt = 0
    wall_t0 = time.perf_counter()
    with token_file.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            prompt_text = record["prompt_text"]
            t0 = time.perf_counter()
            prompt_token_ids = od.encode_prompt(tok, prompt_text)
            t1 = time.perf_counter()
            response = od.request_decode(
                base_url=args.base_url,
                model=args.model,
                prompt_token_ids=prompt_token_ids,
                output_len=args.output_len,
                timeout_s=args.request_timeout_s,
            )
            t2 = time.perf_counter()
            choice = od.choice_from_response(response)
            completion_token_ids, source, kind = od.extract_generated_token_ids(
                response, choice, prompt_token_ids
            )
            row = {
                "id": record["id"],
                "index": index,
                "prompt_sha256": od.sha256_text(prompt_text),
                "prompt_token_sha256": od.sha256_tokens(prompt_token_ids),
                "completion_token_sha256": od.sha256_tokens(completion_token_ids),
                "num_prompt_tokens": len(prompt_token_ids),
                "num_completion_tokens": len(completion_token_ids),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            t3 = time.perf_counter()
            total_completion += len(completion_token_ids)
            total_prompt += len(prompt_token_ids)
            rows.append({
                "index": index,
                "num_prompt_tokens": len(prompt_token_ids),
                "num_completion_tokens": len(completion_token_ids),
                "t_tokenize_s": t1 - t0,
                "t_request_s": t2 - t1,
                "t_io_s": t3 - t2,
            })
    wall_s = time.perf_counter() - wall_t0

    summary = {
        "output_len": args.output_len,
        "num_records": len(records),
        "num_prompt_tokens": total_prompt,
        "num_completion_tokens": total_completion,
        "wall_s": wall_s,
        "per_request": rows,
    }
    out_file.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"decode_worker output_len={args.output_len} wall_s={wall_s:.3f} "
          f"completion_tokens={total_completion}", flush=True)
    return 0


def _aggregate_pass(summary: dict[str, Any]) -> dict[str, Any]:
    """Turn one instrumented pass into harness/hardware quantities."""
    rows = summary["per_request"]
    n = len(rows)
    L = summary["output_len"]
    t_tok = [r["t_tokenize_s"] for r in rows]
    t_req = [r["t_request_s"] for r in rows]
    t_io = [r["t_io_s"] for r in rows]
    n_comp = [r["num_completion_tokens"] for r in rows]

    T_tok, T_req, T_io = sum(t_tok), sum(t_req), sum(t_io)
    T_wall = T_tok + T_req + T_io
    N = sum(n_comp)
    # local wall_tps (reproduces the decode_outputs.py metric / #246 anchor)
    wall_tps = N / T_wall
    # official boundary on LOCAL hw: request-time only, 4 warmup discarded
    warm = slice(WARMUP_REQUESTS, n)
    T_req_warm = sum(t_req[warm])
    N_warm = sum(n_comp[warm])
    official_boundary_tps_local = N_warm / T_req_warm
    # warmup drag: per-request request time, first-4 vs warm steady-state
    mean_req_warm = statistics.fmean(t_req[warm])
    mean_req_cold4 = statistics.fmean(t_req[:WARMUP_REQUESTS])
    # implied local per-spec-step time (E[T] accepted tokens / step)
    steps_warm = N_warm / E_T
    local_step_ms_implied = 1000.0 * T_req_warm / steps_warm
    return {
        "output_len": L,
        "num_records": n,
        "T_tokenize_s": T_tok,
        "T_request_s": T_req,
        "T_io_s": T_io,
        "T_wall_s": T_wall,
        "wall_tps_local": wall_tps,
        "official_boundary_tps_local": official_boundary_tps_local,
        "T_request_warm_s": T_req_warm,
        "mean_request_warm_ms": 1000.0 * mean_req_warm,
        "mean_request_cold4_ms": 1000.0 * mean_req_cold4,
        "warmup_drag_ms_per_req": 1000.0 * (mean_req_cold4 - mean_req_warm),
        "local_step_ms_implied": local_step_ms_implied,
        # overhead-time split (the harness component's evidence): tokenization,
        # file IO, and the warmup-inclusion drag (extra time of the first 4).
        "overhead_time_split_s": {
            "client_tokenization": T_tok,
            "per_request_file_io": T_io,
            "warmup_inclusion": max(
                0.0, (mean_req_cold4 - mean_req_warm) * WARMUP_REQUESTS
            ),
        },
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    """Serve the deployed submission once; run instrumented decode passes."""
    from scripts.local_validation import harness, paths  # noqa: E402

    for note in paths.prepare_local_gpu_env():
        print(f"[measure] {note}", flush=True)
    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = OUT_ROOT / f"measure-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"

    output_lens = [int(x) for x in args.output_lens.split(",") if x.strip()]
    n_prompts = int(args.measure_num_prompts)
    # warm the server with one discarded pass at the official length first.
    warm_len = output_lens[0] if output_lens else OFFICIAL_OUTPUT_LEN
    passes: list[int] = [warm_len] + output_lens

    import os
    import threading
    import subprocess as sp

    # Pin the worker subprocess to the SERVER venv (mirrors
    # harness._participant_env): VIRTUAL_ENV/PATH to the server venv, no inherited
    # PYTHONPATH. PYTHONSAFEPATH=1 stops Python from auto-prepending the script's
    # own directory to sys.path — a second guard (besides the module-level scrub)
    # against this profile.py shadowing the stdlib `profile` module that the
    # transformers->torch._dynamo->cProfile import chain depends on.
    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH', '')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_env["PYTHONSAFEPATH"] = "1"

    def _vram_peak() -> float | None:
        try:
            out = sp.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, sp.SubprocessError):
            return None
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else None

    peak = {"mib": 0.0}
    stop = threading.Event()

    def _sample() -> None:
        while not stop.is_set():
            m = _vram_peak()
            if m:
                peak["mib"] = max(peak["mib"], m)
            stop.wait(2.0)

    measured: dict[str, Any] = {"run_dir": str(run_dir), "passes": {}}
    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=args.port,
            startup_timeout_s=1200, log_path=log_path,
        ) as srv:
            measured["model_id"] = srv.model_id
            measured["served_model_name"] = srv.served_model_name
            for i, L in enumerate(passes):
                tag = "warmup" if i == 0 else f"L{L}"
                out_file = run_dir / f"pass_{i}_{tag}.json"
                cmd = [
                    str(server_python), str(Path(__file__).resolve()),
                    "--decode-worker",
                    "--base-url", srv.base_url,
                    "--model", srv.served_model_name,
                    "--dataset-path", str(paths.EVAL_PROMPTS),
                    "--tokenizer", paths.TOKENIZER,
                    "--num-prompts", str(n_prompts),
                    "--output-len", str(L),
                    "--seed", str(paths.SEED),
                    "--out-file", str(out_file),
                ]
                print(f"[measure] pass {i} ({tag}) output_len={L}", flush=True)
                sp.run(cmd, check=True, timeout=3600, env=worker_env)
                summary = json.loads(out_file.read_text())
                agg = _aggregate_pass(summary)
                if i == 0:
                    measured["warmup_pass"] = agg  # discarded for headline
                else:
                    measured["passes"][str(L)] = agg
    finally:
        stop.set()
        sampler.join(timeout=5)

    measured["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    # headline = the official-length pass
    head = measured["passes"].get(str(OFFICIAL_OUTPUT_LEN))
    if head:
        measured["wall_tps_local_remeasure"] = head["wall_tps_local"]
        measured["official_boundary_tps_local"] = head["official_boundary_tps_local"]
        measured["local_step_ms_implied"] = head["local_step_ms_implied"]
        measured["overhead_time_split_s"] = head["overhead_time_split_s"]
    # empirical stability: tau-from-boundary-vs-wall at each decode length
    measured["length_sweep_tau"] = {
        L: p["official_boundary_tps_local"] / p["wall_tps_local"]
        for L, p in measured["passes"].items()
    }
    return measured


# ========================================================================== #
# Report + wandb
# ========================================================================== #
def build_report(measured: dict[str, Any] | None) -> dict[str, Any]:
    transfer = compute_transfer()
    decomp = build_decomposition(measured)
    stability = tau_lo_stability(measured)
    cross = composition_cross_check(measured)
    mapping = mapping_rule_application(transfer)
    self_test = build_self_test(transfer, decomp)
    return {
        "transfer_calibration_analysis_only": True,
        "baseline_official_tps": OFFICIAL_ANCHOR_TPS,
        "tps_delta": 0.0,
        "ppl_local_anchor": LOCAL_ANCHOR_PPL,
        "ppl_official": OFFICIAL_PPL,
        "transfer": transfer,
        "decomposition": decomp,
        "tau_lo_stability": stability,
        "composition_cross_check": cross,
        "mapping_rule": mapping,
        "self_test": self_test,
        "local_official_tps_transfer_self_test_passes": self_test["passes"],
        "tau_lo": transfer["tau_lo"],
        "measured": measured,
    }


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover
        print(f"[profile] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile",
        agent="lawine",
        name=args.wandb_name or "lawine/local-official-tps-transfer",
        group=args.wandb_group or "local-official-transfer",
        tags=["local-official-transfer", "tps-calibration", "linear-mtp-k7", "local-a10g"],
        notes="PR #267 local->official TPS transfer factor (bank-the-analysis)",
        config={
            "submission": str(SUBMISSION),
            "local_anchor_tps": LOCAL_ANCHOR_TPS,
            "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
            "k_cal": K_CAL, "e_t": E_T, "served_step_ms": SERVED_STEP_MS,
        },
    )
    if run is None:
        print("[profile] wandb init returned None — skipping", flush=True)
        return None
    t = report["transfer"]
    d = report["decomposition"]
    s = report["tau_lo_stability"]
    st = report["self_test"]
    mp = report["mapping_rule"]
    summary = {
        "tau_lo": t["tau_lo"],
        "additive_gap_tps": t["additive_gap_tps"],
        "gap_pct_official_relative": t["gap_pct_official_relative"],
        "local_bar_for_official_500": t["local_bar_for_official_500"],
        "stark266_implied_official_tps": mp["stark266_implied_official_tps"],
        "stark266_clears_official_500": int(bool(mp["stark266_clears_official_500"])),
        "decomp_measured": int(bool(d.get("measured"))),
        "harness_tps": (d.get("components_tps") or {}).get("harness_tokenization_io_warmup"),
        "hardware_tps": (d.get("components_tps") or {}).get("hardware_clock_plus_methodology"),
        "harness_pct_of_gap": d.get("harness_pct_of_gap"),
        "dominant_source_is_harness": int(d.get("dominant") == "harness") if d.get("measured") else None,
        "tau_lo_stable": int(bool(s.get("tau_lo_stable"))),
        "tau_lo_spread_pct": s.get("tau_lo_spread_pct", s.get("tau_lo_spread_pct_max")),
        "official_boundary_tps_local": d.get("official_boundary_tps_local"),
        "wall_tps_local_remeasure": (report.get("measured") or {}).get("wall_tps_local_remeasure"),
        "local_step_ms_implied": (report.get("measured") or {}).get("local_step_ms_implied"),
        "peak_vram_gb": (report.get("measured") or {}).get("peak_vram_gb"),
        "self_test_passes": int(bool(st["passes"])),
        "transfer_calibration_analysis_only": 1,
        "tps_delta": 0.0,
    }
    summary = {k: v for k, v in summary.items() if v is not None}
    log_summary(run, summary, step=0)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    return rid


def _print_summary(report: dict[str, Any]) -> None:
    t = report["transfer"]
    d = report["decomposition"]
    s = report["tau_lo_stability"]
    st = report["self_test"]
    line = "=" * 16 + " LOCAL->OFFICIAL TPS TRANSFER (PR #267) " + "=" * 16
    print("\n" + line, flush=True)
    print(f"  anchor pair: LOCAL {LOCAL_ANCHOR_TPS:.2f}  vs  OFFICIAL {OFFICIAL_ANCHOR_TPS:.2f}", flush=True)
    print(f"  tau_lo = {t['tau_lo']:.5f}   gap = {t['additive_gap_tps']:+.2f} TPS "
          f"({t['gap_pct_official_relative']:+.2f}% official / {t['gap_pct_local_relative']:+.2f}% local)", flush=True)
    print(f"  local bar for official 500: >= {t['local_bar_for_official_500']:.2f} local TPS", flush=True)
    mp = report["mapping_rule"]
    print(f"  apply -> stark adaptive-K local {mp['stark266_projected_local_tps']:.2f} "
          f"=> official ~{mp['stark266_implied_official_tps']:.2f} "
          f"(clears local bar={mp['stark266_clears_local_bar']}, official 500={mp['stark266_clears_official_500']})",
          flush=True)
    if d.get("measured"):
        c = d["components_tps"]
        print(f"  DECOMP (measured): harness {c['harness_tokenization_io_warmup']:+.2f} "
              f"({d['harness_pct_of_gap']:.0f}%)  hardware/method {c['hardware_clock_plus_methodology']:+.2f} "
              f"({d['hardware_pct_of_gap']:.0f}%)  dominant={d['dominant']}", flush=True)
        print(f"  official boundary on LOCAL hw = {d['official_boundary_tps_local']:.2f} TPS "
              f"(wall re-measure {d['wall_tps_local_remeasure']:.2f})", flush=True)
    else:
        print(f"  DECOMP: unattributed {d['unattributed_gap_tps']:.2f} TPS (run --measure)", flush=True)
    print(f"  tau_lo_stable={s.get('tau_lo_stable')}  spread="
          f"{s.get('tau_lo_spread_pct', s.get('tau_lo_spread_pct_max')):.3f}%  basis={s.get('basis')}", flush=True)
    print(f"\n  SELF-TEST local_official_tps_transfer_self_test_passes = {st['passes']}", flush=True)
    for k in ("tau_lo_roundtrip_ok", "decomp_sum_ok", "systematic_offset_ok",
              "local_bar_roundtrip_ok", "nan_clean_ok", "constants_ok"):
        print(f"    {k} = {st[k]}", flush=True)
    print("=" * len(line) + "\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="exit non-zero unless local_official_tps_transfer_self_test_passes")
    ap.add_argument("--measure", action="store_true",
                    help="serve the deployed submission + bank the measured harness/hardware split")
    ap.add_argument("--output-lens", default="512,256,128",
                    help="decode lengths for the stability sweep (first measured pass is 512)")
    ap.add_argument("--measure-num-prompts", type=int, default=NUM_PROMPTS,
                    help="prompts per measured pass (lower for a cheap plumbing smoke)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")

    # decode-worker (internal, runs under the server venv)
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer")
    ap.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=OFFICIAL_OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--request-timeout-s", type=int, default=180)
    ap.add_argument("--out-file")
    args = ap.parse_args(argv)

    if args.decode_worker:
        return _decode_worker(args)

    measured = None
    if args.measure:
        measured = run_measurement(args)

    report = build_report(measured)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.no_wandb:
        report["wandb_run_id"] = log_wandb(report, args)
    report["created_at"] = stamp
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    if measured:
        (Path(measured["run_dir"]) / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_summary(report)
    print(f"[profile] report: {OUT_ROOT / 'report.json'}", flush=True)

    if args.self_test:
        return 0 if report["local_official_tps_transfer_self_test_passes"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
