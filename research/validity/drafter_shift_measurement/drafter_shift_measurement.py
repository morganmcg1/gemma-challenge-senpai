#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #495 -- MEASURE the drafter acceptance gap on a shifted private-like split.

WHAT THIS GROUNDS
-----------------
denken #492 (`cqtlhsvz`, MERGED) established ANALYTICALLY that the 3.661%
drafter-acceptance bucket of the public->private TPS gap IS a distribution-shift
term: the intrinsic a1=0.729 public cliff "cancels in the ratio"
(r_accept=0.9634), so it is NOT in the gap, and the closer is a DOMAIN-TARGETED
drafter retrain on private-like (reasoning/STEM) data (60-80% closure clears
breach<1%). That decomposition was a literature estimate (SambaNova
arXiv:2503.07807), NOT a measurement of THIS drafter on shifted data.

This leg measures it. We run the UNMODIFIED deployed submission
(`submissions/fa2sw_precache_kenyan`: linear MTP K=7 => M=8 verify, greedy,
split-KV) over TWO matched-protocol splits and read vLLM's Prometheus
spec-decode acceptance counters:
  1. PUBLIC  : the official 128 eval prompts (mmlu_pro/gpqa/aime reasoning/STEM).
  2. SHIFTED : 128 held-out private-like reasoning/STEM prompts (GSM8K + MATH +
     MMLU-STEM), built by build_shifted_split.py, zero public overlap, matched
     prompt format (only the content distribution moves).

For each split we read num_drafts, num_accepted_tokens and the per-position
accepted counts -> E_accept = 1 + acc/drafts and the conditional accept ladder
a_k, hence a1. We report:
  E_accept_public, E_accept_shifted, a1_public, a1_shifted,
  Delta_measured (abs E[T] and relative %), r_accept_shifted,
  and the two decisive bools:
    * shift_assumption_validated : measured relative drop ~ #492's 3.661%
    * a1_cancellation_holds      : the intrinsic a1 cliff cancels in the ratio
      (the shift is ~position-uniform; raising a1 by the in-repo arch lever does
      NOT close the relative gap -- #492's saturation test, now on MEASURED
      ladders).
plus a costed domain-targeted retrain recipe sized to the MEASURED shift.

Method mirrors PR #282 (`et_prompt_distribution`) on the serving side: launch
the deployed submission via harness.LocalServer, override only DISABLE_LOG_STATS=0
(re-register stat loggers; host-side counter bumps, does NOT change emitted
tokens -- greedy identity untouched) and VLLM_USE_FLASHINFER_SAMPLER=0 (cuRAND
JIT shim). conc=1 (MAX_NUM_SEQS=1) so per-prompt counter deltas are exact.

LOCAL profiling on a single A10G. analysis_only=True, official_tps=0. No HF Job,
no submission, no served-file change, no draw, no drafter training. BASELINE
481.53 unchanged; this leg adds 0 TPS (it MEASURES the existing distribution).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

# --------------------------------------------------------------------------- #
# Imported fleet anchors (import EXACTLY; do not re-derive).
# --------------------------------------------------------------------------- #
OFFICIAL = 481.53                 # #52 deployed public linear TPS
K_CAL = 125.268                   # official = K_cal * E[T]
K_SPEC = 7                        # num_speculative_tokens (manifest) => M=8 verify
E_T_MAX = K_SPEC + 1              # 8.0

# #492 (cqtlhsvz) modeled decomposition this leg grounds:
ACCEPT_BUCKET_492 = 0.0366125761625703    # the 3.661% accept shift term = 1 - r_accept
R_ACCEPT_492 = 0.9633874238374297
E_T_PUB_492 = 3.851185944363104           # public E[T] (also #282 et_aggregate)
A1_PUB_492 = 0.7292532942898975           # public a1 (also #289)
A_PUB_LADDER_492 = [0.7292532942898975, 0.759556697719242, 0.7929794882639035,
                    0.8228, 0.8348727920920435, 0.8357919254658385,
                    0.8464932652113331]   # conditional a_k

# #489 (q1ivw9tt) breach inversion (via #492 verdict):
DELTA_TARGET_BREACH5 = 0.033342302764780236   # breach<5% <=> Delta<=3.334%
DELTA_TARGET_BREACH1 = 0.02644070088473472    # breach<1% <=> Delta<=2.644%
DEPLOYED_GAP_492 = 0.04294644155088977        # full public->private TPS gap

# In-repo a1 arch lever (denken #308/#342) for the saturation counterfactual:
ARCH_LIFT_A1 = 0.04215000000000002    # in-repo native arch lift on a1
A1_ROBUST_CEIL = 0.85                 # robustness ceiling on a1

# #492 domain-targeted closure band (SambaNova arXiv:2503.07807 distillation rows):
CLOSE_DOMAIN_LO, CLOSE_DOMAIN_HI = 0.60, 0.80
CLOSE_WIDERMIX_LO, CLOSE_WIDERMIX_HI = 0.25, 0.50

# A domain-targeted retrain only has something to close if the MEASURED shifted-
# domain acceptance DEFICIT is positive (E_accept drops on the shifted split).
# When the measured shift is <=0 (drafter accepts at least as well on the shifted
# proxy) there is no deficit to close; the recipe becomes a CONTINGENT sizing
# template rather than an ACTIVE prescription, and #492's "the gap IS the shift"
# premise is refuted on that proxy.
RETRAIN_TRIGGER_PP = 0.0

# Retrain FLOP geometry (frozen 4B backbone soft-KD; denken #301/#352 methodology;
# MTP head sized from /tmp/qat-assistant config: 4 layers, hidden 256, tied vocab).
P_BACKBONE = 4.0e9
P_LMHEAD = 262144 * 2560              # tied lm_head GEMM = 6.71e8
P_MTP_HEAD_CORE = 1.5e7              # ~12-25M trainable core (tiny vs EAGLE-3 124.5M)
A10G_BF16_TFLOPS = 70.0              # nominal bf16 tensor peak
MFU = 0.35

OUT_DIR = ROOT / "research" / "validity" / "drafter_shift_measurement"
DEFAULT_SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
SHIFTED_JSON = OUT_DIR / "shifted_reasoning_stem_128.json"
MEASURED_PATH = OUT_DIR / "measured_result.json"
REPORT_PATH = OUT_DIR / "drafter_shift_measurement_results.json"


# --------------------------------------------------------------------------- #
# Prometheus spec-counter reads (copied verbatim from PR #282 for byte-identical
# parsing; credit research/validity/et_prompt_distribution/et_prompt_distribution.py)
# --------------------------------------------------------------------------- #
def _prom_sum(text: str, metric: str) -> float | None:
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
    deadline = time.time() + timeout_s
    last = read_spec_counters(base_url)
    while time.time() < deadline:
        time.sleep(settle_s)
        cur = read_spec_counters(base_url)
        d_last = last.get("num_drafts")
        d_cur = cur.get("num_drafts")
        if d_cur is None:
            return cur
        if d_cur > (prev_drafts or 0.0) and d_last is not None and abs(d_cur - d_last) < 1e-9:
            return cur
        last = cur
    return last


def request_decode(base_url: str, model: str, prompt_token_ids: list[int],
                   output_len: int, timeout_s: int = 300) -> dict[str, Any]:
    payload = {
        "model": model, "prompt": prompt_token_ids, "max_tokens": output_len,
        "temperature": 0.0, "stream": False, "add_special_tokens": False,
        "ignore_eos": True, "return_token_ids": True,
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
# Measurement
# --------------------------------------------------------------------------- #
def _source_of(rec_id: str) -> str:
    if rec_id.startswith("gsm8k"):
        return "gsm8k"
    if rec_id.startswith("math"):
        return "math"
    if rec_id.startswith("mmlu"):
        return "mmlu_stem"
    return "public"


def _drive_block(srv, dco, tokenizer, records, output_len, prev, label) -> tuple[list[dict], dict]:
    per_prompt = []
    for idx, rec in enumerate(records):
        ptext = rec["prompt_text"]
        ptoks = dco.encode_prompt(tokenizer, ptext)
        resp = request_decode(srv.base_url, srv.served_model_name, ptoks, output_len)
        n_completion = _completion_tokens(resp)
        cur = read_spec_counters_stable(srv.base_url, prev_drafts=prev.get("num_drafts") or 0.0)
        d_drafts = (cur.get("num_drafts") or 0.0) - (prev.get("num_drafts") or 0.0)
        d_acc = (cur.get("num_accepted_tokens") or 0.0) - (prev.get("num_accepted_tokens") or 0.0)
        et_p = (1.0 + d_acc / d_drafts) if d_drafts > 0 else float("nan")
        src = _source_of(rec["id"]) if label != "public" else "public"
        per_prompt.append({
            "index": idx, "id": rec["id"], "source": src,
            "n_completion_tokens": n_completion,
            "delta_drafts": d_drafts, "delta_accepted": d_acc, "E_T": et_p,
        })
        prev = cur
        if (idx + 1) % 16 == 0:
            print(f"[shift:{label}] {idx + 1}/{len(records)} E[T]_p={et_p:.4f} "
                  f"(steps={d_drafts:.0f}, acc={d_acc:.0f})", flush=True)
    block_end = read_spec_counters(srv.base_url)
    return per_prompt, block_end


def measure(submission: Path, *, num_prompts: int, output_len: int, seed: int) -> dict[str, Any]:
    import importlib.util as ilu

    from transformers import AutoTokenizer

    dco_spec = ilu.spec_from_file_location("official_decode_outputs", str(paths.DECODE_SCRIPT))
    dco = ilu.module_from_spec(dco_spec)
    dco_spec.loader.exec_module(dco)

    tokenizer = AutoTokenizer.from_pretrained(paths.TOKENIZER)
    public_records = dco.read_sharegpt_prompts(paths.EVAL_PROMPTS, num_prompts=num_prompts, seed=seed)
    shifted_records = dco.read_sharegpt_prompts(SHIFTED_JSON, num_prompts=num_prompts, seed=seed)
    if len(public_records) != num_prompts or len(shifted_records) != num_prompts:
        raise ValueError(f"expected {num_prompts}/{num_prompts}, got "
                         f"{len(public_records)}/{len(shifted_records)}")

    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    extra_env = {"DISABLE_LOG_STATS": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    log_path = OUT_DIR / "server_drafter_shift.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with VramPeak() as vram, harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env=extra_env, startup_timeout_s=1800,
    ) as srv:
        base = read_spec_counters(srv.base_url)
        if base.get("num_drafts") is None:
            raise RuntimeError("vLLM Prometheus spec counters not populated on this wheel")
        pub_pp, pub_end = _drive_block(srv, dco, tokenizer, public_records, output_len, dict(base), "public")
        sh_pp, sh_end = _drive_block(srv, dco, tokenizer, shifted_records, output_len, dict(pub_end), "shifted")
    wall = time.time() - t0

    measured = {
        "pr": 495, "issue": 481, "author": "denken",
        "leg": "measured drafter acceptance gap: public vs shifted private-like split",
        "analysis_only": True, "official_tps": 0,
        "submission": str(submission), "server_python": str(server_python),
        "public_dataset": str(paths.EVAL_PROMPTS), "shifted_dataset": str(SHIFTED_JSON),
        "num_prompts": num_prompts, "output_len": output_len, "seed": seed,
        "K_spec": K_SPEC, "M_verify": E_T_MAX, "conc": 1, "greedy": True,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decode_wall_s": wall, "peak_gpu_gb": vram.peak_mib / 1024.0,
        "counters": {"base": base, "public_end": pub_end, "shifted_end": sh_end},
        "public_per_prompt": pub_pp, "shifted_per_prompt": sh_pp,
    }
    MEASURED_PATH.write_text(json.dumps(measured, indent=2))
    print(f"[shift] measured 2x{num_prompts} prompts in {wall:.0f}s; "
          f"peak {measured['peak_gpu_gb']:.2f} GB -> {MEASURED_PATH}", flush=True)
    return measured


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _block_delta(counters_a: dict, counters_b: dict) -> dict:
    drafts = (counters_b["num_drafts"] or 0.0) - (counters_a["num_drafts"] or 0.0)
    acc = (counters_b["num_accepted_tokens"] or 0.0) - (counters_a["num_accepted_tokens"] or 0.0)
    pp_a = counters_a.get("accepted_per_pos") or [0.0] * K_SPEC
    pp_b = counters_b.get("accepted_per_pos") or [0.0] * K_SPEC
    per_pos = [pp_b[i] - pp_a[i] for i in range(K_SPEC)]
    return {"drafts": drafts, "accepted": acc, "accepted_per_pos": per_pos}


def _ladder(block: dict) -> dict:
    drafts = block["drafts"]
    per_pos = block["accepted_per_pos"]
    cum = [p / drafts for p in per_pos] if drafts > 0 else [float("nan")] * K_SPEC
    cond = []
    prev = 1.0
    for c in cum:
        cond.append(c / prev if prev > 0 else float("nan"))
        prev = c
    e_accept = 1.0 + (block["accepted"] / drafts if drafts > 0 else float("nan"))
    return {"drafts": drafts, "accepted": block["accepted"], "E_accept": e_accept,
            "a1": cum[0], "cumulative": cum, "conditional": cond}


def _e_accept_from_cond(cond: list[float]) -> float:
    """E[T] = 1 + sum_k prod_{j<=k} cond_j (chain identity)."""
    et, prod = 1.0, 1.0
    for c in cond:
        prod *= c
        et += prod
    return et


def analyze(measured: dict[str, Any]) -> dict[str, Any]:
    c = measured["counters"]
    pub_block = _block_delta(c["base"], c["public_end"])
    sh_block = _block_delta(c["public_end"], c["shifted_end"])
    pub = _ladder(pub_block)
    sh = _ladder(sh_block)

    E_pub, E_sh = pub["E_accept"], sh["E_accept"]
    a1_pub, a1_sh = pub["a1"], sh["a1"]
    r_accept = E_sh / E_pub
    delta_abs = E_pub - E_sh
    delta_rel = 1.0 - r_accept                  # comparable to #492's 3.661% (= 1 - r_accept)
    r_a1 = a1_sh / a1_pub

    # per-position ratios (cumulative + conditional)
    R_cum = [(sh["cumulative"][i] / pub["cumulative"][i]) if pub["cumulative"][i] > 0 else float("nan")
             for i in range(K_SPEC)]
    rho_cond = [(sh["conditional"][i] / pub["conditional"][i]) if pub["conditional"][i] > 0 else float("nan")
                for i in range(K_SPEC)]
    rho_finite = [x for x in rho_cond if math.isfinite(x)]
    rho_mean = float(np.mean(rho_finite))
    rho_cv = float(np.std(rho_finite) / rho_mean) if rho_mean else float("nan")
    s_geom = float(np.exp(np.mean(np.log(rho_finite))))   # geometric-mean multiplicative shift

    # ---- a1-cancellation: #492 saturation counterfactual on MEASURED ladders ----
    # Model the shift as a position-uniform multiplicative factor s on the public
    # conditional ladder (sh ~ s * pub). Then raise intrinsic a1 by the in-repo
    # arch lever and ask whether the RELATIVE gap (1 - E_sh'/E_pub') shrinks.
    s = s_geom
    cond_pub = pub["conditional"]
    cond_pub_arch = list(cond_pub)
    cond_pub_arch[0] = min(cond_pub[0] + ARCH_LIFT_A1, A1_ROBUST_CEIL)
    # baseline (no arch) modelled gap under uniform-s
    E_pub_model = _e_accept_from_cond(cond_pub)
    E_sh_model = _e_accept_from_cond([s * x for x in cond_pub])
    gap_model = 1.0 - E_sh_model / E_pub_model
    # arch-lifted gap (a1 raised, same uniform shift s)
    E_pub_arch = _e_accept_from_cond(cond_pub_arch)
    E_sh_arch = _e_accept_from_cond([min(s * x, 1.0) for x in cond_pub_arch])
    gap_arch = 1.0 - E_sh_arch / E_pub_arch
    arch_closes_gap = (gap_model - gap_arch) > 0.002    # >0.2pp reduction would mean a1 closes it
    arch_delta_gap_pp = (gap_arch - gap_model) * 100.0  # >0 => arch WORSENS (a1 not a closer)

    # a1 over-contribution test: does a1 drop disproportionately vs deeper positions?
    rho_deep = [x for x in rho_cond[1:] if math.isfinite(x)]
    rho_deep_mean = float(np.mean(rho_deep)) if rho_deep else float("nan")
    a1_overcontributes = (rho_cond[0] < rho_deep_mean - 0.05)  # a1 ratio well below deep mean
    a1_cancellation_holds = bool((not arch_closes_gap) and (not a1_overcontributes))

    # ---- shift_assumption_validated: measured relative drop ~ #492's 3.661% ----
    tol_pp = 1.0   # +/-1.0pp band around the modeled 3.661% term
    shift_assumption_validated = bool(abs(delta_rel - ACCEPT_BUCKET_492) * 100.0 <= tol_pp)
    if delta_rel > ACCEPT_BUCKET_492 + tol_pp / 100.0:
        shift_verdict = "over-shifts (proxy heavier than the mild true private shift)"
    elif delta_rel < ACCEPT_BUCKET_492 - tol_pp / 100.0:
        shift_verdict = "under-shifts (proxy milder than the modeled private shift)"
    else:
        shift_verdict = "validated (within +/-1.0pp of the 3.661% modeled term)"

    # ---- per-source E[T] (shifted) + public distribution ----
    def _src_stats(per_prompt, sources=None):
        out = {}
        groups: dict[str, list] = {}
        for p in per_prompt:
            if not math.isfinite(p["E_T"]):
                continue
            groups.setdefault(p["source"], []).append(p)
        for src, ps in groups.items():
            d = sum(x["delta_drafts"] for x in ps)
            a = sum(x["delta_accepted"] for x in ps)
            out[src] = {
                "n": len(ps),
                "E_T_aggregate": 1.0 + a / d if d > 0 else float("nan"),
                "E_T_mean": float(np.mean([x["E_T"] for x in ps])),
                "drafts": d, "accepted": a,
            }
        return out

    sh_by_source = _src_stats(measured["shifted_per_prompt"])
    pub_ets = [p["E_T"] for p in measured["public_per_prompt"] if math.isfinite(p["E_T"])]
    sh_ets = [p["E_T"] for p in measured["shifted_per_prompt"] if math.isfinite(p["E_T"])]

    # ---- TPS context (composition law; this leg adds 0 TPS) ----
    tps_pub = K_CAL * E_pub
    tps_sh = K_CAL * E_sh

    # ---- reconciliation: block per-prompt sums vs block counters ----
    sum_d_pub = sum(p["delta_drafts"] for p in measured["public_per_prompt"])
    sum_d_sh = sum(p["delta_drafts"] for p in measured["shifted_per_prompt"])

    report = {
        "pr": 495, "issue": 481, "author": "denken",
        "leg": measured["leg"], "analysis_only": True, "official_tps": 0,
        "config": {
            "submission": measured["submission"], "K_spec": K_SPEC, "M_verify": E_T_MAX,
            "num_prompts": measured["num_prompts"], "output_len": measured["output_len"],
            "seed": measured["seed"], "conc": 1, "greedy": True,
            "public_dataset": measured["public_dataset"], "shifted_dataset": measured["shifted_dataset"],
            "decode_wall_s": measured["decode_wall_s"], "peak_gpu_gb": measured["peak_gpu_gb"],
        },
        # ===== PRIMARY REPORTED QUANTITIES =====
        "E_accept_public": E_pub,
        "E_accept_shifted": E_sh,
        "a1_public": a1_pub,
        "a1_shifted": a1_sh,
        "Delta_measured_abs_ET": delta_abs,
        "Delta_measured_rel_pct": delta_rel * 100.0,
        "r_accept_shifted": r_accept,
        "r_a1": r_a1,
        "shift_assumption_validated": shift_assumption_validated,
        "a1_cancellation_holds": a1_cancellation_holds,
        # ===== decomposition detail =====
        "public_ladder": pub,
        "shifted_ladder": sh,
        "per_position_cumulative_ratio": R_cum,
        "per_position_conditional_ratio": rho_cond,
        "rho_cond_mean": rho_mean,
        "rho_cond_cv": rho_cv,
        "s_mult_geom": s_geom,
        "shift_verdict": shift_verdict,
        "shift_is_position_uniform": bool(rho_cv < 0.5),
        # a1 cancellation mechanics
        "cancellation": {
            "s_uniform": s,
            "gap_model_uniform_pct": gap_model * 100.0,
            "gap_arch_lifted_pct": gap_arch * 100.0,
            "arch_delta_gap_pp": arch_delta_gap_pp,
            "arch_closes_gap": arch_closes_gap,
            "a1_arch_lifted_to": cond_pub_arch[0],
            "rho_a1": rho_cond[0],
            "rho_deep_mean": rho_deep_mean,
            "a1_overcontributes": a1_overcontributes,
            "mechanism": ("uniform multiplicative shift s on the public conditional "
                          "ladder reproduces r_accept; raising intrinsic a1 by the "
                          "in-repo arch lever leaves the relative gap unchanged/worse "
                          "=> a1 cancels in the ratio (it is the SPEED axis, not a "
                          "gap-closer), confirming #492" if a1_cancellation_holds else
                          "a1 ratio departs from the deeper-position shift => part of "
                          "the gap is a1-intrinsic; the retrain must specifically "
                          "target first-position match, not only uniform domain lift"),
        },
        # ===== per-source / distribution =====
        "shifted_by_source": sh_by_source,
        "public_E_T_distribution": {
            "n": len(pub_ets), "mean": float(np.mean(pub_ets)), "median": float(np.median(pub_ets)),
            "std": float(np.std(pub_ets, ddof=1)), "min": float(np.min(pub_ets)), "max": float(np.max(pub_ets)),
        },
        "shifted_E_T_distribution": {
            "n": len(sh_ets), "mean": float(np.mean(sh_ets)), "median": float(np.median(sh_ets)),
            "std": float(np.std(sh_ets, ddof=1)), "min": float(np.min(sh_ets)), "max": float(np.max(sh_ets)),
        },
        # ===== TPS context (0 TPS added) =====
        "tps_public_context": tps_pub,
        "tps_shifted_context": tps_sh,
        # ===== imported #492/#489 anchors (echoed for audit) =====
        "imported": {
            "accept_bucket_492": ACCEPT_BUCKET_492, "r_accept_492": R_ACCEPT_492,
            "E_T_pub_492": E_T_PUB_492, "a1_pub_492": A1_PUB_492,
            "a_pub_ladder_492": A_PUB_LADDER_492,
            "delta_target_breach5": DELTA_TARGET_BREACH5, "delta_target_breach1": DELTA_TARGET_BREACH1,
            "deployed_gap_492": DEPLOYED_GAP_492, "K_cal": K_CAL, "official": OFFICIAL,
            "arch_lift_a1": ARCH_LIFT_A1,
        },
        "_recon": {"sum_drafts_public": sum_d_pub, "block_drafts_public": pub_block["drafts"],
                   "sum_drafts_shifted": sum_d_sh, "block_drafts_shifted": sh_block["drafts"]},
    }
    report["retrain_spec"] = retrain_spec(delta_rel, report)
    return report


# --------------------------------------------------------------------------- #
# Costed domain-targeted retrain recipe (sized to the MEASURED shift)
# --------------------------------------------------------------------------- #
def _gpu_hours_cached(n_unique_tokens: float, epochs: float) -> float:
    """Cached-teacher recipe: backbone+teacher captured ONCE per unique token;
    tiny MTP head trained over epochs. Dominant cost = the capture pass."""
    capture_flop = n_unique_tokens * (2 * P_BACKBONE + 2 * P_LMHEAD)           # 9.34 GFLOP/unique
    headtrain_flop = (n_unique_tokens * epochs) * (6 * P_MTP_HEAD_CORE + 4 * P_LMHEAD)
    total_flop = capture_flop + headtrain_flop
    eff = MFU * A10G_BF16_TFLOPS * 1e12
    return total_flop / eff / 3600.0


def retrain_spec(delta_rel: float, report: dict) -> dict:
    """Cost a domain-targeted drafter (MTP head) retrain to close 60-80% of the
    MEASURED shifted-domain acceptance DEFICIT, per #492's literature closure
    band, priced on the frozen-4B soft-KD cached recipe (MTP head geometry).

    Sign-aware: a domain retrain only has something to close when the measured
    deficit (delta_pp = 100*(1 - r_accept)) is POSITIVE. When the proxy shift is
    <=0 (drafter accepts at least as well on the shifted split) there is no
    deficit, the recipe is a CONTINGENT sizing template (priced on |delta|), and
    #492's "the 3.661% gap IS this shift" premise is REFUTED on the proxy."""
    delta_pp = delta_rel * 100.0
    retrain_warranted = bool(delta_pp > RETRAIN_TRIGGER_PP)

    if retrain_warranted:
        close_lo = CLOSE_DOMAIN_LO * delta_pp   # 60% closure
        close_hi = CLOSE_DOMAIN_HI * delta_pp   # 80% closure
        resid_most = delta_pp - close_hi        # most closure -> smallest residual
        resid_least = delta_pp - close_lo       # least closure -> largest residual
        residual_band = sorted([resid_most, resid_least])
        closure_band = sorted([close_lo, close_hi])
        recipe_status = "active"
        objective = ("ACTIVE prescription: close 60-80% of the MEASURED "
                     f"public->shifted acceptance deficit ({delta_pp:.3f}pp) via a "
                     "DOMAIN-TARGETED MTP-drafter retrain on private-like "
                     "reasoning/STEM data (shift-reduction lever; NOT arch).")
    else:
        # No positive deficit on this proxy: nothing for a domain retrain to close.
        # Residual == the measured (non-positive) shift; no closure applied.
        closure_band = [0.0, 0.0]
        residual_band = [delta_pp, delta_pp]
        recipe_status = "contingent_not_warranted"
        objective = ("NOT WARRANTED on this proxy: the MEASURED shifted-domain "
                     f"acceptance shift is non-positive ({delta_pp:.3f}pp), so a "
                     "DOMAIN-TARGETED reasoning/STEM retrain has NO acceptance "
                     "deficit to close. #492's premise -- that the 3.661% gap IS a "
                     "public->private reasoning/STEM shift -- is REFUTED on this "
                     "format-matched proxy. The recipe below is a CONTINGENT sizing "
                     "template only: it prices what closing a hypothetical POSITIVE "
                     "deficit of magnitude |delta| would cost, should a heavier "
                     "true-private split later measure a positive shift.")

    central_resid = 0.5 * (residual_band[0] + residual_band[1])
    # does residual clear the #489 breach bars (Delta<=2.644% breach<1%, <=3.334% breach<5%)?
    clears_breach1_central = central_resid <= DELTA_TARGET_BREACH1 * 100.0
    clears_breach5_central = central_resid <= DELTA_TARGET_BREACH5 * 100.0
    clears_breach1_robust = residual_band[1] <= DELTA_TARGET_BREACH1 * 100.0  # worst-case residual clears

    # token-budget band (SambaNova-style in-domain draft-head adaptation):
    budget = {
        "lo":      {"n_unique": 1.0e8, "epochs": 1},
        "central": {"n_unique": 2.0e8, "epochs": 3},
        "hi":      {"n_unique": 4.0e8, "epochs": 6},
    }
    gpu_hours = {k: _gpu_hours_cached(v["n_unique"], v["epochs"]) for k, v in budget.items()}

    return {
        "objective": objective,
        "retrain_warranted": retrain_warranted,
        "recipe_status": recipe_status,
        "measured_shift_to_close_pp": delta_pp,
        "closure_band_pp": closure_band,
        "residual_gap_band_pp": residual_band,
        "delta_target_breach1_pp": DELTA_TARGET_BREACH1 * 100.0,
        "delta_target_breach5_pp": DELTA_TARGET_BREACH5 * 100.0,
        "proxy_shift_already_within_breach1": bool(delta_pp <= DELTA_TARGET_BREACH1 * 100.0),
        "residual_clears_breach1_central": clears_breach1_central,
        "residual_clears_breach5_central": clears_breach5_central,
        "residual_clears_breach1_robust": clears_breach1_robust,
        "data_mix": {
            "sources": ["GSM8K (grade-school math)", "hendrycks MATH (competition math, all levels)",
                        "MMLU-STEM (college/HS STEM MC)",
                        "+ harder private-matched science (GPQA-style) / olympiad math to match the "
                        "graduate/competition private domain"],
            "held_out_eval": str(SHIFTED_JSON) + " (the exact split measured here -> direct closure read-out)",
            "format": "matched to the public instruction templates (MC + step-by-step math)",
            "public_overlap": 0,
        },
        "objective_fn": "soft-KD top-k distillation from the frozen 4B Gemma-4-E4B teacher "
                        "(calibrates the MTP head's ranks on the shifted distribution); the "
                        "deployed drafter is the tiny Gemma4Assistant MTP head (4 layers, "
                        "hidden 256, tied vocab, ~12-25M trainable core).",
        "token_budget_band": budget,
        "gpu_hours_A10G_band": gpu_hours,
        "gpu_hours_A10G_central": gpu_hours["central"],
        "wall_clock_8xA10G_central_h": gpu_hours["central"] / 8.0,
        "flop_terms": {
            "capture_per_unique_GFLOP": (2 * P_BACKBONE + 2 * P_LMHEAD) / 1e9,
            "headtrain_per_pass_GFLOP": (6 * P_MTP_HEAD_CORE + 4 * P_LMHEAD) / 1e9,
            "backbone_dominates": True,
            "MFU": MFU, "A10G_bf16_TFLOPS": A10G_BF16_TFLOPS,
        },
        "is_hours_not_weeks": bool(gpu_hours["hi"] < 168.0),
        "handoff": "approval-gated cluster training issue (instructions/training-request.md); "
                   "checkpoint -> repackage as submissions/<name>/ drafter; re-run THIS harness "
                   "on the held-out shifted split to read realized closure before any HF draw.",
        "note": "Cost dominated by the one-time frozen-backbone+teacher capture pass; the tiny "
                "MTP head makes head-training nearly free. Central is SAME-DAY on the 8-GPU node. "
                "Magnitude of REQUIRED closure scales with the measured shift; recipe is sized to it.",
    }


# --------------------------------------------------------------------------- #
# Self-test (PRIMARY)
# --------------------------------------------------------------------------- #
def self_test(report: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    rc = report["_recon"]
    # (a) public E[T] reproduces the #282/#492 anchor 3.851 within 0.1
    checks["a_public_ET_reproduces_anchor"] = abs(report["E_accept_public"] - E_T_PUB_492) <= 0.1
    # (b) public a1 reproduces the #289/#492 anchor 0.729 within 0.02
    checks["b_public_a1_reproduces_anchor"] = abs(report["a1_public"] - A1_PUB_492) <= 0.02
    # (c) block per-prompt drafts reconcile block counters (conc=1 exactness), both splits
    checks["c_public_recon"] = (rc["block_drafts_public"] > 0 and
                                abs(rc["sum_drafts_public"] - rc["block_drafts_public"]) / rc["block_drafts_public"] < 0.01)
    checks["c_shifted_recon"] = (rc["block_drafts_shifted"] > 0 and
                                 abs(rc["sum_drafts_shifted"] - rc["block_drafts_shifted"]) / rc["block_drafts_shifted"] < 0.01)
    # (d) r_accept in (0, 1.05]; shifted does not exceed public materially
    checks["d_r_accept_sane"] = 0.0 < report["r_accept_shifted"] <= 1.05
    # (e) Delta relative = 1 - r_accept (internal consistency)
    checks["e_delta_consistent"] = abs(report["Delta_measured_rel_pct"] / 100.0 - (1.0 - report["r_accept_shifted"])) < 1e-9
    # (f) ladders monotone non-increasing cumulative (acceptance is a prefix), both splits
    def _mono(cum):
        return all(cum[i] >= cum[i + 1] - 1e-9 for i in range(len(cum) - 1))
    checks["f_public_cum_monotone"] = _mono(report["public_ladder"]["cumulative"])
    checks["f_shifted_cum_monotone"] = _mono(report["shifted_ladder"]["cumulative"])
    # (g) chain identity: E[T] reconstructs from the conditional ladder, both splits
    checks["g_public_chain_identity"] = abs(_e_accept_from_cond(report["public_ladder"]["conditional"]) - report["E_accept_public"]) < 1e-6
    checks["g_shifted_chain_identity"] = abs(_e_accept_from_cond(report["shifted_ladder"]["conditional"]) - report["E_accept_shifted"]) < 1e-6
    # (h) retrain residual band ordered (sign-valid by construction) + delta targets ordered
    rs = report["retrain_spec"]
    checks["h_residual_band_ordered"] = rs["residual_gap_band_pp"][0] <= rs["residual_gap_band_pp"][1]
    checks["h_delta_targets_ordered"] = DELTA_TARGET_BREACH1 < DELTA_TARGET_BREACH5
    # (h2) retrain_warranted is consistent with the sign of the measured deficit
    checks["h_retrain_warranted_consistent"] = (
        bool(rs["retrain_warranted"]) == (rs["measured_shift_to_close_pp"] > RETRAIN_TRIGGER_PP))
    # (i) NaN-clean across the primary scalars
    prim = ["E_accept_public", "E_accept_shifted", "a1_public", "a1_shifted",
            "Delta_measured_abs_ET", "Delta_measured_rel_pct", "r_accept_shifted", "r_a1"]
    checks["i_nan_clean"] = all(math.isfinite(float(report[k])) for k in prim)
    # (j) retrain priced hours-not-weeks
    checks["j_retrain_hours_not_weeks"] = rs["is_hours_not_weeks"]
    # (k) imported anchors EXACT
    checks["k_anchors_exact"] = (ACCEPT_BUCKET_492 == 0.0366125761625703 and K_CAL == 125.268)

    gate = all(checks.values())
    report["self_test"] = checks
    report["drafter_shift_self_test_passes"] = bool(gate)
    return report


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def log_wandb(report: dict[str, Any], measured: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[shift] wandb unavailable ({exc})", flush=True)
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="profiling",
            config={
                "pr": 495, "issue": 481, "submission": report["config"]["submission"],
                "num_prompts": report["config"]["num_prompts"], "output_len": report["config"]["output_len"],
                "seed": report["config"]["seed"], "K_spec": K_SPEC, "M_verify": E_T_MAX, "conc": 1,
                "public_dataset": report["config"]["public_dataset"],
                "shifted_dataset": report["config"]["shifted_dataset"],
                "delta_definition": "Delta_rel = 1 - r_accept (E_accept_shifted/E_accept_public)",
            },
        )
        flat = {
            "primary/drafter_shift_self_test_passes": report["drafter_shift_self_test_passes"],
            "primary/shift_assumption_validated": report["shift_assumption_validated"],
            "primary/a1_cancellation_holds": report["a1_cancellation_holds"],
            "test/Delta_measured_rel_pct": report["Delta_measured_rel_pct"],
            "test/r_accept_shifted": report["r_accept_shifted"],
            "E_accept_public": report["E_accept_public"], "E_accept_shifted": report["E_accept_shifted"],
            "a1_public": report["a1_public"], "a1_shifted": report["a1_shifted"],
            "Delta_measured_abs_ET": report["Delta_measured_abs_ET"],
            "r_a1": report["r_a1"], "rho_cond_mean": report["rho_cond_mean"],
            "rho_cond_cv": report["rho_cond_cv"], "s_mult_geom": report["s_mult_geom"],
            "shift_is_position_uniform": report["shift_is_position_uniform"],
            "cancellation/arch_delta_gap_pp": report["cancellation"]["arch_delta_gap_pp"],
            "cancellation/arch_closes_gap": report["cancellation"]["arch_closes_gap"],
            "cancellation/rho_a1": report["cancellation"]["rho_a1"],
            "cancellation/rho_deep_mean": report["cancellation"]["rho_deep_mean"],
            "tps_public_context": report["tps_public_context"],
            "tps_shifted_context": report["tps_shifted_context"],
            "accept_bucket_492": ACCEPT_BUCKET_492 * 100.0,
            "retrain/warranted": report["retrain_spec"]["retrain_warranted"],
            "retrain/recipe_status": report["retrain_spec"]["recipe_status"],
            "retrain/measured_shift_to_close_pp": report["retrain_spec"]["measured_shift_to_close_pp"],
            "retrain/proxy_shift_already_within_breach1": report["retrain_spec"]["proxy_shift_already_within_breach1"],
            "retrain/gpu_hours_A10G_central": report["retrain_spec"]["gpu_hours_A10G_central"],
            "retrain/residual_clears_breach1_central": report["retrain_spec"]["residual_clears_breach1_central"],
            "retrain/residual_clears_breach1_robust": report["retrain_spec"]["residual_clears_breach1_robust"],
            "peak_gpu_gb": report["config"]["peak_gpu_gb"], "decode_wall_s": report["config"]["decode_wall_s"],
        }
        run.summary.update(flat)
        # per-position ladder table
        lt = wandb.Table(columns=["position", "a_pub_cond", "a_sh_cond", "rho_cond",
                                  "C_pub", "C_sh", "R_cum"])
        for i in range(K_SPEC):
            lt.add_data(i + 1, report["public_ladder"]["conditional"][i],
                        report["shifted_ladder"]["conditional"][i],
                        report["per_position_conditional_ratio"][i],
                        report["public_ladder"]["cumulative"][i],
                        report["shifted_ladder"]["cumulative"][i],
                        report["per_position_cumulative_ratio"][i])
        run.log({"accept_ladder": lt})
        # per-source table
        st = wandb.Table(columns=["source", "n", "E_T_aggregate", "E_T_mean"])
        st.add_data("public_all", report["public_E_T_distribution"]["n"],
                    report["E_accept_public"], report["public_E_T_distribution"]["mean"])
        for src, v in report["shifted_by_source"].items():
            st.add_data(src, v["n"], v["E_T_aggregate"], v["E_T_mean"])
        run.log({"per_source_E_T": st})
        rid = run.id
        print(f"[shift] W&B run: {run.url}", flush=True)
        run.finish()
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[shift] wandb log failed ({exc})", flush=True)
        return None


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="reuse cached measured_result.json if present (else measure), "
                         "then analyze + self-test + wandb (PRIMARY).")
    ap.add_argument("--measure", action="store_true", help="force fresh GPU measurement.")
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="drafter-shift-measurement")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="denken/drafter-shift-measurement")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for note in paths.prepare_local_gpu_env():
        print(f"[shift] {note}", flush=True)

    if args.measure or not MEASURED_PATH.exists():
        measured = measure(args.submission.resolve(), num_prompts=args.num_prompts,
                           output_len=args.output_len, seed=args.seed)
    else:
        print(f"[shift] reusing cached measurement {MEASURED_PATH}", flush=True)
        measured = json.loads(MEASURED_PATH.read_text())

    report = analyze(measured)
    report = self_test(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    wid = None
    if not args.no_wandb:
        wid = log_wandb(report, measured, args.wandb_name, args.wandb_group)
    report["wandb_run_id"] = wid
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n========== DRAFTER SHIFT MEASUREMENT (PR #495) ==========", flush=True)
    print(f"E_accept  public / shifted : {report['E_accept_public']:.4f} / {report['E_accept_shifted']:.4f}", flush=True)
    print(f"a1        public / shifted : {report['a1_public']:.4f} / {report['a1_shifted']:.4f}", flush=True)
    print(f"Delta_measured             : {report['Delta_measured_abs_ET']:.4f} E[T]  "
          f"({report['Delta_measured_rel_pct']:.3f}%  vs #492 modeled 3.661%)", flush=True)
    print(f"r_accept_shifted / r_a1    : {report['r_accept_shifted']:.4f} / {report['r_a1']:.4f}", flush=True)
    print(f"rho_cond (per-pos)         : "
          f"{[round(x,3) for x in report['per_position_conditional_ratio']]}", flush=True)
    print(f"shift_assumption_validated : {report['shift_assumption_validated']}  ({report['shift_verdict']})", flush=True)
    print(f"a1_cancellation_holds      : {report['a1_cancellation_holds']}  "
          f"(arch a1-lift moves gap {report['cancellation']['arch_delta_gap_pp']:+.3f}pp)", flush=True)
    print(f"retrain_warranted          : {report['retrain_spec']['retrain_warranted']}  "
          f"({report['retrain_spec']['recipe_status']})", flush=True)
    print(f"retrain central GPU-hr     : {report['retrain_spec']['gpu_hours_A10G_central']:.1f} A10G  "
          f"(contingent sizing; residual clears breach<1% central: "
          f"{report['retrain_spec']['residual_clears_breach1_central']})", flush=True)
    print(f"PRIMARY self_test          : {report['drafter_shift_self_test_passes']}", flush=True)
    print(f"peak GPU                   : {report['config'].get('peak_gpu_gb', float('nan')):.2f} GB", flush=True)
    print(f"wandb run                  : {wid}", flush=True)
    print(f"artifacts                  : {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
