#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #282 -- E[T] per-prompt distribution on the 128 competition prompts.

WHAT THIS ANSWERS
-----------------
kanna #217 (`vgovdrjc`) banked the deployed linear spec-decode E[T] = 3.844 as a
single MEAN over the 128 speed-benchmark prompts AND their 512 output tokens
(token-weighted). The composition law `official = K_cal * E[T]` (K_cal=125.268,
125.268*3.844 = 481.53) makes E[T] enter LINEARLY -- every +0.1 in E[T] is
+12.5 TPS (+2.6%). This leg opens that mean up: it measures the PER-PROMPT E[T]
distribution to ask whether some prompts drag the mean down (a heavy left tail
= exploitable headroom) or whether the distribution is tight (the deployed
drafter is near its natural ceiling for this prompt mix).

This is the BETWEEN-PROMPT variance of E[T] (each prompt's mean acceptance over
its own 512 tokens). It is ORTHOGONAL to wirbel #175 (`et_second_moment`), which
is the analytic WITHIN-RUN step-to-step variance of committed length on a TREE
topology (sigma_L ~ 3.04, a per-step quantity). Here every prompt's E[T] is
itself an average over ~130-200 spec steps, so the between-prompt std is the std
of those per-prompt averages -- a different, much smaller statistic.

HOW (contract-safe, measurement-only)
--------------------------------------
Mirrors wirbel #76's `scripts/profiler/accept_calibration.py` byte-for-byte on
the serving side: launch the UNMODIFIED deployed submission
(`submissions/fa2sw_precache_kenyan`: linear MTP K=7, M=8 verify, split-KV) via
`harness.LocalServer`, overriding ONLY `DISABLE_LOG_STATS=0` (re-register vLLM's
stat loggers; <0.1 ms host-side counter bumps per step, does NOT change which
tokens are accepted/emitted -- greedy identity untouched) and
`VLLM_USE_FLASHINFER_SAMPLER=0` (this container's cuRAND JIT shim; sampler does
not touch logits). The new bit vs #76: we drive the prompts ONE AT A TIME at
conc=1 (MAX_NUM_SEQS=1 already), reading vLLM's Prometheus spec counters
`vllm:spec_decode_num_drafts` / `vllm:spec_decode_num_accepted_tokens` BEFORE and
AFTER each prompt. The per-prompt DELTA is that prompt's (steps, accepted), so
E[T]_p = 1 + accepted_p / drafts_p -- exact, no overlap to attribute because
conc=1. Prompts are the EXACT official set: the seed=1 shuffle of
`eval_prompts_sharegpt.json` (first 128), chat-templated and sent through
`/v1/completions` with `ignore_eos=True` so every prompt emits exactly 512
tokens, identical to `decode_outputs.py`.

The sum of per-prompt deltas reproduces the whole-run totals, and the
token-weighted aggregate `1 + sum(acc)/sum(drafts)` reproduces kanna's 3.844 --
the measurement-fidelity self-check.

LOCAL profiling on a single A10G. No HF Job / no submission / no served-file
change / no official draw / no train.py --launch. BASELINE stays 481.53; this
leg adds 0 TPS (it MEASURES the existing distribution). Analysis-only:
et_prompt_distribution_analysis_only = True.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

# research/validity/et_prompt_distribution/this.py -> repo root is 3 parents up.
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
STEP_MS = 1.2182                  # kanna #217 per-forward-pass time (ms)
TAU = 1.218                       # composition tau
K_SPEC = 7                        # num_speculative_tokens (manifest)
E_T_MAX = K_SPEC + 1              # 8.0 -- theoretical E[T] ceiling at lambda=1
PRIVATE_VERIFIED = 460.85         # private-verified reference (PR baseline)
LAMBDA_BAR_OPERATIVE = 0.9780112973731208  # validity bar (context only)
TOP1_LINEAR_ACCEPT_ANCHOR = 0.728739760479042  # #76 accept_calibration top-1

OUT_DIR = ROOT / "research" / "validity" / "et_prompt_distribution"
DEFAULT_SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
MEASURED_PATH = OUT_DIR / "measured_result.json"
REPORT_PATH = OUT_DIR / "report.json"

# composition helpers ------------------------------------------------------- #
def tps_from_et(et: float) -> float:
    """official = K_cal * E[T] (125.268 * 3.844 = 481.53)."""
    return K_CAL * et


# E[T] that yields exactly 500 TPS. Two readings, both reported:
#  - direct inverse of the round-tripping law official = K_cal * E[T]
ET_NEEDED_500 = 500.0 / K_CAL
#  - the PR's literal formula 500*step/(K_cal*tau); agrees to ~0.02%
ET_NEEDED_500_PR_FORMULA = 500.0 * STEP_MS / (K_CAL * TAU)


# --------------------------------------------------------------------------- #
# Prometheus spec-counter reads
# --------------------------------------------------------------------------- #
def _prom_sum(text: str, metric: str) -> float | None:
    """Sum a counter across all (non-position) label sets; None if absent."""
    pat = re.compile(
        rf"^{re.escape(metric)}(?:_total)?(?:\{{(?![^}}]*position=)[^}}]*\}})?\s+([\d.eE+-]+)$",
        re.M,
    )
    total, found = 0.0, False
    for m in pat.finditer(text):
        try:
            total += float(m.group(1))
            found = True
        except ValueError:
            pass
    return total if found else None


def _prom_per_pos(text: str, metric: str, K: int) -> list[float] | None:
    pat = re.compile(rf"^{re.escape(metric)}(?:_total)?\{{([^}}]*)\}}\s+([\d.eE+-]+)$", re.M)
    acc: dict[int, float] = {}
    for m in pat.finditer(text):
        labels, val = m.group(1), m.group(2)
        pm = re.search(r'position="(\d+)"', labels)
        if not pm:
            continue
        try:
            acc[int(pm.group(1))] = acc.get(int(pm.group(1)), 0.0) + float(val)
        except ValueError:
            pass
    if not acc:
        return None
    return [acc.get(i, 0.0) for i in range(K)]


def _get_text(url: str, timeout_s: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return r.read().decode("utf-8", "replace")


def read_spec_counters(base_url: str) -> dict[str, Any]:
    text = _get_text(f"{base_url}/metrics")
    return {
        "num_drafts": _prom_sum(text, "vllm:spec_decode_num_drafts"),
        "num_accepted_tokens": _prom_sum(text, "vllm:spec_decode_num_accepted_tokens"),
        "num_draft_tokens": _prom_sum(text, "vllm:spec_decode_num_draft_tokens"),
        "accepted_per_pos": _prom_per_pos(text, "vllm:spec_decode_num_accepted_tokens_per_pos", K_SPEC),
    }


def read_spec_counters_stable(base_url: str, prev_drafts: float, *,
                              timeout_s: float = 20.0, settle_s: float = 0.15) -> dict[str, Any]:
    """Poll /metrics until num_drafts has advanced past prev_drafts and is stable
    across two consecutive reads. conc=1 guarantees the advance belongs to the
    just-finished prompt; the stability wait absorbs vLLM's async counter flush."""
    deadline = time.time() + timeout_s
    last = read_spec_counters(base_url)
    while time.time() < deadline:
        time.sleep(settle_s)
        cur = read_spec_counters(base_url)
        d_last = last.get("num_drafts")
        d_cur = cur.get("num_drafts")
        if d_cur is None:
            return cur  # prometheus not populated -> caller handles
        if d_cur > (prev_drafts or 0.0) and d_last is not None and abs(d_cur - d_last) < 1e-9:
            return cur
        last = cur
    return last


# --------------------------------------------------------------------------- #
# prompt features
# --------------------------------------------------------------------------- #
_CODE_MARKERS = ("```", "def ", "class ", "import ", "function", "public ",
                 "private ", "#include", "return ", "const ", "var ", "let ",
                 "=>", "println", "printf", "console.log", "</", "/>", "select ",
                 "insert ", "{\n", "});", "$('", "lambda ")
_MATH_MARKERS = ("\\frac", "\\sum", "\\int", "equation", "solve for", "theorem",
                 "integral", "derivative", "probability", "matrix", "\\(", "\\)",
                 "polynomial", "compute the", "calculate the", "\\sqrt", "modulo")


def classify_domain(text: str) -> str:
    low = text.lower()
    code_hits = sum(1 for m in _CODE_MARKERS if m in low)
    sym = sum(1 for c in text if c in "{}[]();<>=/\\|&")
    sym_density = sym / max(len(text), 1)
    math_hits = sum(1 for m in _MATH_MARKERS if m in low)
    digit_density = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    if code_hits >= 2 or sym_density > 0.06:
        return "code"
    if math_hits >= 2 or (math_hits >= 1 and digit_density > 0.04):
        return "math"
    return "text"


def prompt_features(prompt_text: str, prompt_token_ids: list[int]) -> dict[str, Any]:
    n_tok = len(prompt_token_ids)
    uniq = len(set(prompt_token_ids))
    return {
        "n_prompt_tokens": n_tok,
        "n_prompt_chars": len(prompt_text),
        "vocab_diversity": (uniq / n_tok) if n_tok else 0.0,
        "domain": classify_domain(prompt_text),
    }


# --------------------------------------------------------------------------- #
# GPU memory peak sampler (the model lives in the server subprocess)
# --------------------------------------------------------------------------- #
class VramPeak:
    def __init__(self) -> None:
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

    def _sample_once(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "", 1).isdigit()]
                if vals:
                    self.peak_mib = max(self.peak_mib, max(vals))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(5.0)

    def __enter__(self) -> "VramPeak":
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2.0)
        self._sample_once()


# --------------------------------------------------------------------------- #
# decode request (identical shape to official decode_outputs.request_decode)
# --------------------------------------------------------------------------- #
def request_decode(base_url: str, model: str, prompt_token_ids: list[int],
                   output_len: int, timeout_s: int = 300) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": False,
        "add_special_tokens": False,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def _completion_tokens(resp: dict[str, Any]) -> int:
    usage = resp.get("usage") or {}
    if isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    ch = (resp.get("choices") or [{}])[0]
    tok = ch.get("token_ids")
    return len(tok) if isinstance(tok, list) else 0


# --------------------------------------------------------------------------- #
# Measurement: serve once, drive 128 prompts one at a time, read deltas
# --------------------------------------------------------------------------- #
def _load_official_decode_module():
    spec = importlib.util.spec_from_file_location("official_decode_outputs", str(paths.DECODE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(submission: Path, *, num_prompts: int, output_len: int, seed: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    dco = _load_official_decode_module()
    tokenizer = AutoTokenizer.from_pretrained(paths.TOKENIZER)
    records = dco.read_sharegpt_prompts(paths.EVAL_PROMPTS, num_prompts=num_prompts, seed=seed)
    if len(records) != num_prompts:
        raise ValueError(f"expected {num_prompts} prompts, found {len(records)}")

    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    extra_env = {"DISABLE_LOG_STATS": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    log_path = OUT_DIR / "server_et_prompt_distribution.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_prompt: list[dict[str, Any]] = []
    t0 = time.time()
    with VramPeak() as vram, harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        model = srv.served_model_name
        base = read_spec_counters(srv.base_url)
        if not base.get("num_drafts") and base.get("num_drafts") is None:
            raise RuntimeError("vLLM Prometheus spec counters not populated on this wheel; "
                               "per-prompt delta method is unavailable")
        prev = dict(base)
        for idx, rec in enumerate(records):
            ptext = rec["prompt_text"]
            ptoks = dco.encode_prompt(tokenizer, ptext)
            resp = request_decode(srv.base_url, model, ptoks, output_len)
            n_completion = _completion_tokens(resp)
            cur = read_spec_counters_stable(srv.base_url, prev_drafts=prev.get("num_drafts") or 0.0)
            d_drafts = (cur.get("num_drafts") or 0.0) - (prev.get("num_drafts") or 0.0)
            d_acc = (cur.get("num_accepted_tokens") or 0.0) - (prev.get("num_accepted_tokens") or 0.0)
            d_draft_tok = (cur.get("num_draft_tokens") or 0.0) - (prev.get("num_draft_tokens") or 0.0)
            et_p = (1.0 + d_acc / d_drafts) if d_drafts > 0 else float("nan")
            feats = prompt_features(ptext, ptoks)
            per_prompt.append({
                "index": idx,
                "id": rec["id"],
                "dataset_index": rec["dataset_index"],
                "n_completion_tokens": n_completion,
                "delta_drafts": d_drafts,
                "delta_accepted": d_acc,
                "delta_draft_tokens": d_draft_tok,
                "emitted_eq_acc_plus_drafts": d_acc + d_drafts,
                "E_T": et_p,
                **feats,
            })
            prev = cur
            if (idx + 1) % 16 == 0:
                print(f"[et-dist] {idx + 1}/{num_prompts} E[T]_p={et_p:.4f} "
                      f"(steps={d_drafts:.0f}, acc={d_acc:.0f})", flush=True)
        final = read_spec_counters(srv.base_url)
    wall = time.time() - t0

    total_drafts = (final.get("num_drafts") or 0.0) - (base.get("num_drafts") or 0.0)
    total_acc = (final.get("num_accepted_tokens") or 0.0) - (base.get("num_accepted_tokens") or 0.0)
    sum_drafts = sum(p["delta_drafts"] for p in per_prompt)
    sum_acc = sum(p["delta_accepted"] for p in per_prompt)

    measured = {
        "pr": 282,
        "leg": "E[T] per-prompt distribution on 128 competition prompts (local)",
        "et_prompt_distribution_analysis_only": True,
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
        "sum_per_prompt_drafts": sum_drafts,
        "sum_per_prompt_accepted": sum_acc,
        "per_prompt": per_prompt,
    }
    MEASURED_PATH.write_text(json.dumps(measured, indent=2))
    print(f"[et-dist] measured {len(per_prompt)} prompts in {wall:.0f}s; "
          f"peak {measured['peak_gpu_gb']:.2f} GB -> {MEASURED_PATH}", flush=True)
    return measured


# --------------------------------------------------------------------------- #
# Analysis: distribution + pricing
# --------------------------------------------------------------------------- #
def analyze(measured: dict[str, Any]) -> dict[str, Any]:
    per = measured["per_prompt"]
    ets = np.array([p["E_T"] for p in per], dtype=float)
    finite = ets[np.isfinite(ets)]
    n = int(finite.size)

    # token-weighted aggregate E[T] = 1 + sum(acc)/sum(drafts). This is kanna's
    # quantity (mean over prompts AND tokens) and the composition input.
    sum_drafts = measured["sum_per_prompt_drafts"]
    sum_acc = measured["sum_per_prompt_accepted"]
    et_aggregate = 1.0 + sum_acc / sum_drafts if sum_drafts else float("nan")

    et_arith_mean = float(np.mean(finite))
    et_median = float(np.median(finite))
    p5, p25, p75, p95 = (float(np.percentile(finite, q)) for q in (5, 25, 75, 95))
    et_std = float(np.std(finite, ddof=1))
    et_min = float(np.min(finite))
    et_max = float(np.max(finite))
    idx_sorted = sorted(per, key=lambda p: p["E_T"])
    et_min_prompt_idx = idx_sorted[0]["index"]
    et_max_prompt_idx = idx_sorted[-1]["index"]

    bins = np.arange(1.0, E_T_MAX + 0.25, 0.25)
    counts, edges = np.histogram(finite, bins=bins)
    histogram = [{"bin_left": float(edges[i]), "bin_right": float(edges[i + 1]),
                  "count": int(counts[i])} for i in range(len(counts))]

    # spread classification
    if et_std < 0.5:
        spread = "tight"
    elif et_std > 1.0:
        spread = "spread"
    else:
        spread = "moderate"

    # --- pricing (counterfactual "if all prompts had E[T]=v" -> TPS = K_cal*v) #
    et_p25_tps = tps_from_et(p25)
    et_p75_tps = tps_from_et(p75)
    et_ceiling_tps = tps_from_et(p95)        # uses p95, NOT K+1=8
    et_gain_from_p25_to_median_tps = tps_from_et(et_median) - OFFICIAL

    # precise "lift the bottom quartile up to the median" counterfactual: replace
    # every prompt below p25 with the median, recompute the equal-weight mean and
    # the token-weighted aggregate.
    lifted = np.where(finite < p25, et_median, finite)
    lifted_arith_mean = float(np.mean(lifted))
    # token-weighted: a prompt at E[T]=e over 512 tokens contributes 512/e steps.
    steps_real = 512.0 / finite
    steps_lift = 512.0 / lifted
    lifted_aggregate = (512.0 * n) / float(np.sum(steps_lift))
    real_aggregate_equalwt_check = (512.0 * n) / float(np.sum(steps_real))
    lift_bottom_quartile_to_median_tps = tps_from_et(lifted_aggregate)
    lift_bottom_quartile_gain_tps = lift_bottom_quartile_to_median_tps - tps_from_et(et_aggregate)

    # --- gap to 500 -------------------------------------------------------- #
    et_gap = ET_NEEDED_500 - et_aggregate
    et_gap_as_pct_of_range = et_gap / (E_T_MAX - et_aggregate)
    et_p75_clears_500 = p75 >= ET_NEEDED_500
    et_p95_clears_500 = p95 >= ET_NEEDED_500
    arith_minus_aggregate = et_arith_mean - et_aggregate  # harmonic<->arith gap (variance signature)

    # --- prompt-feature correlation (10 lowest vs 10 highest E[T]) --------- #
    low10 = idx_sorted[:10]
    high10 = idx_sorted[-10:]

    def _meanlen(group):
        return float(np.mean([g["n_prompt_tokens"] for g in group]))

    def _codefrac(group):
        return float(np.mean([1.0 if g["domain"] == "code" else 0.0 for g in group]))

    def _vocab(group):
        return float(np.mean([g["vocab_diversity"] for g in group]))

    low_meanlen, high_meanlen = _meanlen(low10), _meanlen(high10)
    low_codefrac, high_codefrac = _codefrac(low10), _codefrac(high10)
    low_et_prompts_are_longer = low_meanlen > high_meanlen
    low_et_prompts_are_code = low_codefrac > high_codefrac

    report = dict(measured)
    report.pop("per_prompt", None)  # keep report.json compact; full data in measured_result.json
    report.update({
        "n_finite": n,
        "n_nonfinite": int(ets.size - n),
        # imported anchors (echoed for audit)
        "imported": {
            "official": OFFICIAL, "ceiling_lambda1": CEILING_LAMBDA1, "K_cal": K_CAL,
            "E_T_anchor": E_T_ANCHOR, "step_ms": STEP_MS, "tau": TAU, "K_spec": K_SPEC,
            "E_T_max": E_T_MAX, "private_verified": PRIVATE_VERIFIED,
            "top1_linear_accept_anchor": TOP1_LINEAR_ACCEPT_ANCHOR,
        },
        # distribution
        "et_mean": et_aggregate,            # token-weighted aggregate (reproduces 3.844; composition input)
        "et_aggregate": et_aggregate,       # explicit alias
        "et_arith_mean": et_arith_mean,     # equal-weight prompt mean (distribution center)
        "et_arith_minus_aggregate": arith_minus_aggregate,
        "et_median": et_median,
        "et_p5": p5, "et_p25": p25, "et_p75": p75, "et_p95": p95,
        "et_std": et_std,
        "et_min": et_min, "et_max": et_max,
        "et_min_prompt_idx": et_min_prompt_idx,
        "et_max_prompt_idx": et_max_prompt_idx,
        "histogram_bin_size": 0.25,
        "histogram": histogram,
        "spread_classification": spread,
        # pricing
        "et_p25_tps": et_p25_tps,
        "et_p75_tps": et_p75_tps,
        "et_ceiling_tps": et_ceiling_tps,
        "et_gain_from_p25_to_median_tps": et_gain_from_p25_to_median_tps,
        "lift_bottom_quartile_to_median_tps": lift_bottom_quartile_to_median_tps,
        "lift_bottom_quartile_gain_tps": lift_bottom_quartile_gain_tps,
        "lifted_arith_mean": lifted_arith_mean,
        "real_aggregate_equalwt_check": real_aggregate_equalwt_check,
        # gap to 500
        "et_needed_for_500": ET_NEEDED_500,
        "et_needed_for_500_pr_formula": ET_NEEDED_500_PR_FORMULA,
        "et_gap": et_gap,
        "et_gap_as_pct_of_range": et_gap_as_pct_of_range,
        "et_p75_clears_500": bool(et_p75_clears_500),
        "et_p95_clears_500": bool(et_p95_clears_500),
        # feature correlation
        "feature_correlation": {
            "low10_mean_prompt_tokens": low_meanlen,
            "high10_mean_prompt_tokens": high_meanlen,
            "low10_code_fraction": low_codefrac,
            "high10_code_fraction": high_codefrac,
            "low10_vocab_diversity": _vocab(low10),
            "high10_vocab_diversity": _vocab(high10),
            "low10_domains": [g["domain"] for g in low10],
            "high10_domains": [g["domain"] for g in high10],
            "low10_indices": [g["index"] for g in low10],
            "high10_indices": [g["index"] for g in high10],
        },
        "low_et_prompts_are_longer": bool(low_et_prompts_are_longer),
        "low_et_prompts_are_code": bool(low_et_prompts_are_code),
        # composition round-trip diagnostics
        "composition_anchor_roundtrip_tps": tps_from_et(E_T_ANCHOR),
        "composition_measured_roundtrip_tps": tps_from_et(et_aggregate),
    })
    return report


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(report: dict[str, Any]) -> dict[str, Any]:
    et_agg = report["et_aggregate"]
    checks: dict[str, bool] = {}

    # (a) measured token-weighted aggregate reproduces kanna's 3.844 within 0.1
    checks["a_aggregate_reproduces_3844"] = abs(et_agg - E_T_ANCHOR) <= 0.1
    # also report the equal-weight mean's closeness (softer, informational)
    checks["a_arith_mean_within_0p2"] = abs(report["et_arith_mean"] - E_T_ANCHOR) <= 0.2

    # (b) percentiles monotone non-decreasing
    seq = [report["et_p5"], report["et_p25"], report["et_median"], report["et_p75"], report["et_p95"]]
    checks["b_percentiles_monotone"] = all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))

    # (c) composition round-trips on the IMPORTED constants (law + constant integrity)
    checks["c_composition_anchor_roundtrips_481"] = abs(tps_from_et(E_T_ANCHOR) - OFFICIAL) <= 0.5
    # measured round-trip residual reported as a diagnostic (gate uses the ±0.1
    # E[T] tolerance from (a), i.e. ±~12.5 TPS, not ±0.5 TPS)
    report["composition_measured_roundtrip_resid_tps"] = tps_from_et(et_agg) - OFFICIAL

    # (d) et_ceiling_tps uses E[T]=et_p95 (NOT E[T]=K+1=8)
    checks["d_ceiling_uses_p95_not_8"] = (
        abs(report["et_ceiling_tps"] - tps_from_et(report["et_p95"])) < 1e-6
        and report["et_p95"] < E_T_MAX
    )

    # (e) NaN-clean across reported scalars
    scalar_keys = [
        "et_mean", "et_aggregate", "et_arith_mean", "et_median", "et_p5", "et_p25",
        "et_p75", "et_p95", "et_std", "et_min", "et_max", "et_p25_tps", "et_p75_tps",
        "et_ceiling_tps", "et_gain_from_p25_to_median_tps", "et_needed_for_500",
        "et_gap", "et_gap_as_pct_of_range",
    ]
    checks["e_nan_clean"] = all(math.isfinite(float(report[k])) for k in scalar_keys) and \
        report["n_nonfinite"] == 0

    # (f) imported constants EXACT and UNCHANGED
    checks["f_constants_imported_exact"] = (
        OFFICIAL == 481.53 and CEILING_LAMBDA1 == 520.95 and K_CAL == 125.268
        and E_T_ANCHOR == 3.844 and K_SPEC == 7
    )

    # cross-check: sum of per-prompt deltas reconciles with whole-run totals
    sd = report["sum_per_prompt_drafts"]
    td = report["whole_run_total_drafts"]
    checks["g_per_prompt_reconciles_whole_run"] = (td > 0 and abs(sd - td) / td < 0.01)

    gate = bool(
        checks["a_aggregate_reproduces_3844"]
        and checks["b_percentiles_monotone"]
        and checks["c_composition_anchor_roundtrips_481"]
        and checks["d_ceiling_uses_p95_not_8"]
        and checks["e_nan_clean"]
        and checks["f_constants_imported_exact"]
        and checks["g_per_prompt_reconciles_whole_run"]
    )
    report["self_test"] = checks
    report["et_prompt_distribution_self_test_passes"] = gate
    return report


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], measured: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[et-dist] wandb unavailable ({exc})", flush=True)
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="profiling",
            config={
                "pr": 282, "submission": report["submission"],
                "num_prompts": report["num_prompts"], "output_len": report["output_len"],
                "seed": report["seed"], "K_spec": K_SPEC, "conc": 1,
                "K_cal": K_CAL, "official": OFFICIAL, "E_T_anchor": E_T_ANCHOR,
                "et_mean_definition": "token-weighted aggregate (1 + sum(acc)/sum(drafts))",
            },
        )
        flat = {
            "primary/et_prompt_distribution_self_test_passes": report["et_prompt_distribution_self_test_passes"],
            "test/et_std": report["et_std"],
            "test/et_gap": report["et_gap"],
            "et_mean_aggregate": report["et_aggregate"],
            "et_arith_mean": report["et_arith_mean"],
            "et_arith_minus_aggregate": report["et_arith_minus_aggregate"],
            "et_median": report["et_median"],
            "et_p5": report["et_p5"], "et_p25": report["et_p25"],
            "et_p75": report["et_p75"], "et_p95": report["et_p95"],
            "et_min": report["et_min"], "et_max": report["et_max"],
            "et_min_prompt_idx": report["et_min_prompt_idx"],
            "et_max_prompt_idx": report["et_max_prompt_idx"],
            "spread_classification": report["spread_classification"],
            "et_p25_tps": report["et_p25_tps"], "et_p75_tps": report["et_p75_tps"],
            "et_ceiling_tps": report["et_ceiling_tps"],
            "et_gain_from_p25_to_median_tps": report["et_gain_from_p25_to_median_tps"],
            "lift_bottom_quartile_gain_tps": report["lift_bottom_quartile_gain_tps"],
            "et_needed_for_500": report["et_needed_for_500"],
            "et_gap": report["et_gap"],
            "et_gap_as_pct_of_range": report["et_gap_as_pct_of_range"],
            "et_p75_clears_500": report["et_p75_clears_500"],
            "et_p95_clears_500": report["et_p95_clears_500"],
            "low_et_prompts_are_longer": report["low_et_prompts_are_longer"],
            "low_et_prompts_are_code": report["low_et_prompts_are_code"],
            "composition_measured_roundtrip_resid_tps": report.get("composition_measured_roundtrip_resid_tps"),
            "peak_gpu_gb": report.get("peak_gpu_gb"),
            "decode_wall_s": report.get("decode_wall_s"),
        }
        run.summary.update(flat)
        # histogram table
        htbl = wandb.Table(columns=["bin_left", "bin_right", "count"])
        for h in report["histogram"]:
            htbl.add_data(h["bin_left"], h["bin_right"], h["count"])
        run.log({"et_histogram": htbl})
        # per-prompt table
        ptbl = wandb.Table(columns=["index", "E_T", "n_prompt_tokens", "vocab_diversity",
                                    "domain", "delta_drafts", "delta_accepted"])
        for p in measured["per_prompt"]:
            ptbl.add_data(p["index"], p["E_T"], p["n_prompt_tokens"], p["vocab_diversity"],
                          p["domain"], p["delta_drafts"], p["delta_accepted"])
        run.log({"per_prompt_E_T": ptbl})
        rid = run.id
        print(f"[et-dist] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[et-dist] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="reuse cached measured_result.json if present (else measure), "
                         "then run analysis + self-test + wandb (PRIMARY).")
    ap.add_argument("--measure", action="store_true",
                    help="force a fresh GPU measurement even if a cache exists.")
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="et-prompt-distribution")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="lawine/et-prompt-distribution")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[et-dist] {note}", flush=True)

    if args.measure or not MEASURED_PATH.exists():
        measured = measure(args.submission.resolve(), num_prompts=args.num_prompts,
                           output_len=args.output_len, seed=args.seed)
    else:
        print(f"[et-dist] reusing cached measurement {MEASURED_PATH}", flush=True)
        measured = json.loads(MEASURED_PATH.read_text())

    report = analyze(measured)
    report = self_test(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    wid = None
    if not args.no_wandb:
        wid = log_wandb(report, measured, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n========== E[T] PER-PROMPT DISTRIBUTION (PR #282) ==========", flush=True)
    print(f"prompts              : {report['n_finite']} finite / {report['num_prompts']} "
          f"(nonfinite={report['n_nonfinite']})", flush=True)
    print(f"et_mean (aggregate)  : {report['et_aggregate']:.4f}  (anchor 3.844; "
          f"arith mean {report['et_arith_mean']:.4f})", flush=True)
    print(f"et_median            : {report['et_median']:.4f}", flush=True)
    print(f"p5/p25/p75/p95       : {report['et_p5']:.3f} / {report['et_p25']:.3f} / "
          f"{report['et_p75']:.3f} / {report['et_p95']:.3f}", flush=True)
    print(f"et_std (TEST)        : {report['et_std']:.4f}  -> {report['spread_classification']}", flush=True)
    print(f"et_min/max           : {report['et_min']:.3f} (idx {report['et_min_prompt_idx']}) / "
          f"{report['et_max']:.3f} (idx {report['et_max_prompt_idx']})", flush=True)
    print(f"et_needed_for_500    : {report['et_needed_for_500']:.4f}", flush=True)
    print(f"et_gap (TEST)        : {report['et_gap']:.4f}  ({report['et_gap_as_pct_of_range']*100:.1f}% "
          f"of room to {E_T_MAX:.0f})", flush=True)
    print(f"et_p75_tps           : {report['et_p75_tps']:.2f}  (clears 500: {report['et_p75_clears_500']})", flush=True)
    print(f"et_ceiling_tps (p95) : {report['et_ceiling_tps']:.2f}  (p95 clears 500: {report['et_p95_clears_500']})", flush=True)
    print(f"low_et longer/code   : {report['low_et_prompts_are_longer']} / {report['low_et_prompts_are_code']}", flush=True)
    print(f"PRIMARY self_test    : {report['et_prompt_distribution_self_test_passes']}", flush=True)
    print(f"peak GPU             : {report.get('peak_gpu_gb', float('nan')):.2f} GB", flush=True)
    print(f"wandb run            : {wid}", flush=True)
    print(f"artifacts            : {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
