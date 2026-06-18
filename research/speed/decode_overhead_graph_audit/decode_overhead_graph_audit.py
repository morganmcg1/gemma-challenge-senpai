#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #674 -- decode-overhead + CUDA-graph capture audit (orthogonal speed axis).

LOCAL-ONLY. analysis_only=true, official_tps=0. NO HF Job, NO submission, NO
served-file change. One idle pod A10G. Group ``decode-overhead-graph-audit-denken``.

Audits the SHIPPED int4_g128_lmhead body (int4 g128 body + untied int4 g128 lm_head)
at the live serve config (AR M=1, greedy, bf16 KV, max_model_len 4096, gpu_util 0.90,
max_num_batched_tokens 512, vLLM 0.22.1rc1.dev307) along a quality-IDENTICAL second
speed axis the spec-dec cards never touch: is bs=1 decode actually CUDA-graph
captured and optimal, and how much per-token wall is a reclaimable HOST/Python bubble?

Deliverables:
  1. CAPTURE AUDIT     -- resolved cudagraph_mode / capture_sizes / max_size, whether
                          bs=1 is captured, FULL vs PIECEWISE, lm_head/sampler in/out
                          of the graph, Marlin (sm_86; Machete is Hopper-only -> N/A).
  2. decode_overhead_frac = (clean_wall_per_tok - profiled_GPU_busy_per_tok)/clean_wall.
                          Measured on the DEFAULT (live) FULL_AND_PIECEWISE arm. The
                          NONE arm (compile on, graphs OFF) exposes the host-launch
                          bubble the graph already reclaims.
  3. KNOB SWEEP        -- FULL_AND_PIECEWISE (default/live) vs PIECEWISE vs
                          FULL_DECODE_ONLY, median-of-N over >=2 fresh servers each;
                          capture_sizes already [1,2] (1,2 already captured ->
                          explicit {1,2} is a no-op); VLLM_GRAPH_RESERVED_MEM does not
                          exist in 0.22.1rc1.dev307 (auto cudagraph mem profiling ->
                          nothing to sweep). break_rate guards greedy byte-identity.
  4. ANCHOR            -- the DEFAULT arm IS the re-measured g128_AR M=1 baseline
                          (~126.94 local). Any local gain -> official-equiv x 0.870.

Verdict:
  OVERHEAD_RECLAIMABLE -- a capture knob beats the default median wall-TPS (quality-
                          identical) beyond run-to-run noise -> STACKABLE speedup.
  ALREADY_OPTIMAL      -- bs=1 already captured + decode_overhead_frac negligible
                          (cleanly GPU/HBM-bound) + no knob beats default -> headroom
                          is in the GEMV / spec layers, not this axis.

Reproduce:
  cd target/ && CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
    research.speed.decode_overhead_graph_audit.decode_overhead_graph_audit \
    --server-python /tmp/senpai-venvs/5f4c623f772358a2/bin/python \
    --wandb_group decode-overhead-graph-audit-denken \
    --wandb_name denken/decode-overhead-graph-audit
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
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- #
# ANCHORS -- advisor-authored constants (PR #674 body). Cite, do NOT re-derive.
# --------------------------------------------------------------------------- #
LIVE_OFFICIAL_TPS = 126.378      # int4_g128_lmhead live submission (locked)
LIVE_PPL = 2.019
LOCAL_AR_ANCHOR = 126.94         # g128_AR M=1 local anchor (wirbel #665)
STARK_TAX = 0.870                # local -> official conversion (stark)
FIRE_BAR = LIVE_OFFICIAL_TPS     # fire iff official-equiv > 126.378
HIDDEN, VOCAB = 2560, 262144
INT4_HEAD_BYTES = VOCAB * HIDDEN * 0.5    # ~335 MB int4 head (vs 1.34 GB bf16)

# decode_overhead_frac threshold for "negligible" (cleanly GPU-bound).
NEGLIGIBLE_OVERHEAD_FRAC = 0.05
# a knob must beat the default median by more than this (relative) AND clear the
# default's own run-to-run noise band to count as a real, reclaimable gain.
MIN_REAL_GAIN_FRAC = 0.005


# --------------------------- kernel categorization ------------------------- #
def _is_attn(n: str) -> bool:
    return any(s in n for s in ("attn", "_fwd", "flash", "paged", "unified_attention",
                                "reshape_and_cache", "rotary", "rope", "fmha", "mha",
                                "reduce_segments", "merge_attn", "slot_mapping"))


def _is_matmul(n: str) -> bool:
    return any(s in n for s in ("marlin", "gptq", "gemm", "gemv", "cutlass", "wmma",
                                "splitk", "split_k", "s16816", "tensorop", "cublas",
                                "sgemm", "hgemm", "ampere_", "machete"))


def _is_marlin(n: str) -> bool:
    return "marlin" in n or "gptq" in n


def _is_machete(n: str) -> bool:
    return "machete" in n


def _is_sampling(n: str) -> bool:
    return any(s in n for s in ("log_softmax", "argmax", "topk", "top_k", "softmax",
                                "sample", "logit", "cumsum", "sort", "gather", "scatter",
                                "renorm", "multinomial", "reduce_kernel"))


def _is_norm(n: str) -> bool:
    return any(s in n for s in ("rms", "layernorm", "layer_norm", "norm_kernel"))


def _is_act(n: str) -> bool:
    return any(s in n for s in ("silu", "gelu", "swiglu", "act_and_mul", "mul_and"))


def categorize(name: str) -> str:
    n = name.lower()
    if _is_attn(n):
        return "attn"
    if _is_matmul(n):
        return "matmul"
    if _is_sampling(n):
        return "sampling"
    if _is_norm(n):
        return "norm"
    if _is_act(n):
        return "activation"
    return "other"


def analyze_kernels(rows: list[dict[str, Any]], n_tokens: int) -> dict[str, Any]:
    by_cat: dict[str, float] = {}
    total = 0.0
    any_marlin = any_machete = False
    for r in rows:
        nm, us = r["name"], float(r["self_us"])
        by_cat[categorize(nm)] = by_cat.get(categorize(nm), 0.0) + us
        total += us
        any_marlin = any_marlin or _is_marlin(nm.lower())
        any_machete = any_machete or _is_machete(nm.lower())
    per = (lambda us: us / n_tokens) if n_tokens else (lambda us: float("nan"))
    return {
        "device_us_total": total,
        "per_token_us": {k: per(v) for k, v in by_cat.items()},
        "cat_us_total": by_cat,
        "any_marlin_kernel": any_marlin, "any_machete_kernel": any_machete,
    }


# ------------------------------- worker runner ----------------------------- #
def run_arm(server_python: Path, tag: str, cg_mode: str, do_profile: bool,
            state_dir: Path, tps_tokens: int, profile_tokens: int, n_reps: int,
            model_id: str, capture_sizes: str = "") -> dict[str, Any]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "MODEL_ID": model_id, "STATE_DIR": str(state_dir),
        "CG_MODE": cg_mode, "DO_PROFILE": "1" if do_profile else "0",
        "TPS_TOKENS": str(tps_tokens), "PROFILE_TOKENS": str(profile_tokens),
        "N_TPS_REPS": str(n_reps), "BOOT_TAG": tag, "CAPTURE_SIZES": capture_sizes,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    })
    worker = HERE / "_audit_worker.py"
    log_path = state_dir / f"worker_{tag}.log"
    print(f"\n[doc] ===== ARM {tag} (cg_mode={cg_mode}, profile={do_profile}) =====", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([str(server_python), str(worker)], env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    dur = time.time() - t0
    out_json = state_dir / f"worker_{tag}.json"
    if proc.returncode != 0 or not out_json.exists():
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(f"arm '{tag}' failed (rc={proc.returncode}) in {dur:.0f}s; tail:\n{tail}")
    data = json.loads(out_json.read_text())
    data["_capture_log_lines"] = _grep_capture_lines(log_path)
    data["_boot_dur_s"] = dur
    print(f"[doc] {tag}: median TPS {data['tps_median']:.2f}  ({dur:.0f}s)", flush=True)
    return data


def _grep_capture_lines(log_path: Path) -> list[str]:
    """The authoritative 'Capturing CUDA graphs (...)' / 'Profiling CUDA graph memory'
    lines from the vLLM boot log -- the ground-truth backstop for the capture audit."""
    pats = ("Capturing CUDA graphs", "Profiling CUDA graph memory",
            "Graph capturing finished", "cudagraph_mode", "Estimated CUDA graph memory")
    out = []
    try:
        for ln in log_path.read_text(errors="ignore").splitlines():
            if any(p in ln for p in pats):
                # collapse tqdm carriage-return spam to the final state
                seg = ln.split("\r")[-1].strip()
                if seg and seg not in out:
                    out.append(seg[:400])
    except Exception:
        pass
    return out[:12]


# ------------------------------- the verdict ------------------------------- #
def _cell_median(arms: list[dict[str, Any]]) -> float:
    vals: list[float] = []
    for a in arms:
        vals.extend(a.get("tps_all", []))
    return statistics.median(vals) if vals else float("nan")


def build_verdict(arms: dict[str, list[dict[str, Any]]], model_id: str) -> dict[str, Any]:
    default = arms["FULL_AND_PIECEWISE"]
    prof = next((a for a in default if "kernel_rows" in a), default[0])

    # ---- (4) re-measured anchor: the DEFAULT (live) arm's median over its servers ---
    anchor_local = _cell_median(default)

    # ---- (1) capture audit (resolved config + log backstop) ----
    aud = prof.get("capture_audit", {})
    cap_sizes = aud.get("cudagraph_capture_sizes")
    cg_mode_resolved = aud.get("cudagraph_mode")
    bs1_captured = bool(aud.get("bs1_in_capture_sizes")) or (
        "FULL" in str(cg_mode_resolved))   # FULL decode graph covers bs=1
    sm = prof.get("sm", "")

    # ---- (2) decode_overhead_frac on the DEFAULT (graphs ON, live) arm ----
    busy_per_tok = float(prof.get("gpu_busy_per_token_us", float("nan")))
    clean_wall_per_tok = 1e6 / anchor_local if anchor_local else float("nan")
    raw_overhead_frac = (clean_wall_per_tok - busy_per_tok) / clean_wall_per_tok \
        if clean_wall_per_tok else float("nan")
    decode_overhead_frac = max(0.0, raw_overhead_frac) if math.isfinite(raw_overhead_frac) else float("nan")
    ka = analyze_kernels(prof.get("kernel_rows", []), prof.get("profile_tokens", 0) or 1)
    busy_share_profiled = float(prof.get("gpu_busy_share_of_profiled_wall_pct", float("nan")))

    # contrast: NONE arm (compile on, graphs OFF) -> host-launch bubble EXPOSED
    none_arm = arms.get("NONE", [])
    none_prof = next((a for a in none_arm if "kernel_rows" in a), None)
    overhead_frac_graphsoff = float("nan")
    if none_prof:
        none_anchor = _cell_median(none_arm)
        none_busy = float(none_prof.get("gpu_busy_per_token_us", float("nan")))
        none_wall = 1e6 / none_anchor if none_anchor else float("nan")
        if math.isfinite(none_wall) and none_wall:
            overhead_frac_graphsoff = max(0.0, (none_wall - none_busy) / none_wall)

    # ---- (3) knob sweep: per-mode median over >=2 fresh servers ----
    graph_modes = ["FULL_AND_PIECEWISE", "PIECEWISE", "FULL_DECODE_ONLY"]
    cell_tps = {m: _cell_median(arms[m]) for m in graph_modes if m in arms and arms[m]}
    cell_n_servers = {m: len(arms[m]) for m in graph_modes if m in arms and arms[m]}
    # default run-to-run noise band (max-min over its raw reps), used as the bar a
    # knob must clear to be a "real" gain.
    default_reps = [v for a in default for v in a.get("tps_all", [])]
    default_noise_band = (max(default_reps) - min(default_reps)) if len(default_reps) > 1 else 0.0
    default_noise_frac = default_noise_band / anchor_local if anchor_local else float("nan")

    best_mode = max(cell_tps, key=cell_tps.get) if cell_tps else "FULL_AND_PIECEWISE"
    best_graphtuned_walltps = cell_tps.get(best_mode, anchor_local)
    gain_local = best_graphtuned_walltps - anchor_local
    gain_local_frac = gain_local / anchor_local if anchor_local else 0.0
    # a real gain: beats default by > MIN_REAL_GAIN_FRAC AND clears the noise band.
    real_gain = bool(best_mode != "FULL_AND_PIECEWISE"
                     and gain_local_frac > MIN_REAL_GAIN_FRAC
                     and gain_local > default_noise_band)
    official_equiv_best = best_graphtuned_walltps * STARK_TAX
    official_equiv_gain = max(0.0, gain_local) * STARK_TAX if real_gain else 0.0

    # ---- graphs-off floors (context: what the graph already reclaims) ----
    none_tps = _cell_median(arms.get("NONE", [])) if arms.get("NONE") else float("nan")
    eager_tps = _cell_median(arms.get("eager", [])) if arms.get("eager") else float("nan")
    cudagraph_delta_vs_none = anchor_local - none_tps if math.isfinite(none_tps) else float("nan")
    cudagraph_delta_vs_eager = anchor_local - eager_tps if math.isfinite(eager_tps) else float("nan")

    # ---- break_rate: every arm's greedy token_ids must equal the DEFAULT reference ----
    ref_tokens = prof.get("token_ids", [])
    all_arms = [a for lst in arms.values() for a in lst]
    n_break = 0
    breakers = []
    for a in all_arms:
        tids = a.get("token_ids", [])
        ok = (len(tids) == len(ref_tokens) and tids == ref_tokens)
        # eager arm may legitimately differ if compile-off changes numerics; record it
        if not ok:
            n_break += 1
            breakers.append(a["boot_tag"])
    break_rate = n_break / len(all_arms) if all_arms else float("nan")
    # graph-only break_rate (the quality-identical claim is across CAPTURE knobs; the
    # eager/none arms drop torch.compile so are reported separately).
    graph_arms = [a for m in graph_modes for a in arms.get(m, [])]
    graph_break = sum(1 for a in graph_arms
                      if not (a.get("token_ids", []) == ref_tokens))
    graph_break_rate = graph_break / len(graph_arms) if graph_arms else float("nan")

    # ---- Marlin / Machete confirmation ----
    marlin_confirmed = bool(ka.get("any_marlin_kernel"))
    machete_present = bool(ka.get("any_machete_kernel"))
    sm86 = ("sm_86" in sm)

    # ---- verdict ----
    overhead_negligible = bool(math.isfinite(decode_overhead_frac)
                               and decode_overhead_frac < NEGLIGIBLE_OVERHEAD_FRAC)
    already_optimal = bool(bs1_captured and overhead_negligible and not real_gain)
    overhead_reclaimable = bool(real_gain and graph_break_rate == 0.0)
    verdict = "OVERHEAD_RECLAIMABLE" if overhead_reclaimable else "ALREADY_OPTIMAL"
    fires = bool(real_gain and official_equiv_best > FIRE_BAR and graph_break_rate == 0.0)

    return {
        "analysis_only": True, "official_tps": 0, "no_served_file_change": True,
        "no_hf_job": True, "pr": 674, "model_id": model_id, "sm": sm,
        # ---- anchors ----
        "live_official_tps": LIVE_OFFICIAL_TPS, "live_ppl": LIVE_PPL,
        "local_ar_anchor_ref": LOCAL_AR_ANCHOR, "stark_tax": STARK_TAX, "fire_bar": FIRE_BAR,
        # ---- (1) capture audit ----
        "cudagraph_mode_resolved": str(cg_mode_resolved),
        "cudagraph_capture_sizes": cap_sizes,
        "max_cudagraph_capture_size": aud.get("max_cudagraph_capture_size"),
        "bs1_captured": bs1_captured,
        "lmhead_inside_graph": False,   # gpu_model_runner.py:4316 compute_logits OUTSIDE forward ctx
        "sampler_inside_graph": False,  # _sample() in sample_tokens(), OUTSIDE the graph
        "marlin_confirmed": marlin_confirmed, "machete_present": machete_present,
        "is_sm86": sm86,
        "vllm_graph_reserved_mem_exists": False,   # removed in 0.22.1rc1.dev307
        # ---- (2) decode_overhead_frac ----
        "decode_overhead_frac": decode_overhead_frac,
        "decode_overhead_frac_raw": raw_overhead_frac,
        "gpu_busy_per_token_us": busy_per_tok,
        "clean_wall_per_token_us": clean_wall_per_tok,
        "gpu_busy_share_of_profiled_wall_pct": busy_share_profiled,
        "decode_overhead_frac_graphsoff_none": overhead_frac_graphsoff,
        "matmul_per_token_us": ka["per_token_us"].get("matmul", 0.0),
        "attn_per_token_us": ka["per_token_us"].get("attn", 0.0),
        "norm_per_token_us": ka["per_token_us"].get("norm", 0.0) + ka["per_token_us"].get("activation", 0.0),
        "sampling_per_token_us": ka["per_token_us"].get("sampling", 0.0),
        "other_per_token_us": ka["per_token_us"].get("other", 0.0),
        # ---- (3) knob sweep ----
        "cell_median_tps": cell_tps, "cell_n_servers": cell_n_servers,
        "default_noise_band_tps": default_noise_band, "default_noise_frac": default_noise_frac,
        "best_graphtuned_mode": best_mode, "best_graphtuned_walltps": best_graphtuned_walltps,
        "gain_local_tps": gain_local, "gain_local_frac": gain_local_frac, "real_gain": real_gain,
        "official_equiv_best_tps": official_equiv_best, "official_equiv_gain_tps": official_equiv_gain,
        "cudagraph_off_none_tps": none_tps, "cudagraph_off_eager_tps": eager_tps,
        "cudagraph_delta_vs_none_tps": cudagraph_delta_vs_none,
        "cudagraph_delta_vs_eager_tps": cudagraph_delta_vs_eager,
        # ---- (4) anchor ----
        "measured_anchor_tps": anchor_local,
        "anchor_vs_ref_frac": (anchor_local - LOCAL_AR_ANCHOR) / LOCAL_AR_ANCHOR if LOCAL_AR_ANCHOR else float("nan"),
        # ---- guards ----
        "break_rate": break_rate, "graph_break_rate": graph_break_rate,
        "break_arms": breakers,
        # ---- verdict ----
        "overhead_negligible": overhead_negligible, "bs1_captured_flag": bs1_captured,
        "already_optimal": already_optimal, "overhead_reclaimable": overhead_reclaimable,
        "verdict": verdict, "fires": fires,
        # ---- top-line ----
        "primary_metric_name": "decode_overhead_frac",
        "primary_metric_value": decode_overhead_frac,
        # ---- composition (artifact) ----
        "_default_per_token_us": ka["per_token_us"],
        "_capture_log_lines": prof.get("_capture_log_lines", []),
    }


def self_test(v: dict[str, Any], arms: dict[str, list]) -> dict[str, Any]:
    st = {}
    st["bs1_captured"] = bool(v["bs1_captured"])
    st["decode_overhead_frac_finite"] = bool(math.isfinite(v["decode_overhead_frac"]))
    st["decode_overhead_frac_in_unit"] = bool(0.0 <= v["decode_overhead_frac"] <= 1.0)
    st["marlin_confirmed"] = bool(v["marlin_confirmed"])
    st["machete_absent_on_sm86"] = bool((not v["machete_present"]) and v["is_sm86"])
    # the matmul (body + int4 head GEMV) dominates the decode step (HBM-bound)
    st["matmul_dominates_step"] = bool(v["matmul_per_token_us"] > v["attn_per_token_us"]
                                       and v["matmul_per_token_us"] > v["norm_per_token_us"])
    # capture knobs are byte-identical (the quality-identical claim)
    st["graph_capture_byte_identical"] = bool(v["graph_break_rate"] == 0.0)
    # >=2 fresh servers for the two head-to-head graph candidates
    st["two_servers_default"] = bool(v["cell_n_servers"].get("FULL_AND_PIECEWISE", 0) >= 2)
    st["two_servers_piecewise"] = bool(v["cell_n_servers"].get("PIECEWISE", 0) >= 2)
    # CUDA graph ON is faster than (or equal to) graphs OFF -- the graph helps
    st["graph_not_slower_than_none"] = bool(
        not math.isfinite(v["cudagraph_delta_vs_none_tps"]) or v["cudagraph_delta_vs_none_tps"] >= -1.0)
    # the re-measured anchor reproduces the external g128_AR M=1 anchor within 5%
    st["anchor_reproduces_ref"] = bool(abs(v["anchor_vs_ref_frac"]) < 0.05)
    # verdict consistency: exactly one of already_optimal / overhead_reclaimable
    st["verdict_consistent"] = bool(v["already_optimal"] != v["overhead_reclaimable"])
    # no-fire guard wired
    st["analysis_only_no_fire"] = bool(v["analysis_only"] and v["official_tps"] == 0)
    fin = [v["decode_overhead_frac"], v["measured_anchor_tps"], v["best_graphtuned_walltps"],
           v["gpu_busy_per_token_us"], v["clean_wall_per_token_us"]]
    st["nan_clean"] = all(math.isfinite(x) for x in fin)
    st["self_test_passes"] = all(st.values())
    return st


def _ensure_real_wandb() -> bool:
    """A local ``wandb/`` run-data dir (e.g. ``target/wandb/``, present after any
    prior served run) sits on ``sys.path[0]`` when this is launched from ``target/``
    and shadows the installed package: ``import wandb`` then resolves to a namespace
    stub with no ``init`` and ``init_wandb_run`` returns None *silently* (misreported
    as "no API key"). De-shadow by demoting cwd/repo-root path entries below
    site-packages and re-importing once; warn loudly if it is still a stub."""
    import importlib

    try:
        import wandb
        if hasattr(wandb, "init"):
            return True
    except Exception:  # noqa: BLE001
        pass
    shadow = {"", ".", os.getcwd(), str(REPO_ROOT)}
    demoted = [p for p in list(sys.path) if p in shadow]
    for p in demoted:
        while p in sys.path:
            sys.path.remove(p)
    sys.path.extend(demoted)  # keep importable for scripts.*, but after site-packages
    sys.modules.pop("wandb", None)
    try:
        wandb = importlib.import_module("wandb")
        if hasattr(wandb, "init"):
            return True
    except Exception:  # noqa: BLE001
        pass
    print("[doc] WARNING wandb import is a namespace stub (shadowed by a local "
          "wandb/ run-data dir) or wandb is not installed in the RUNNER venv -- "
          "W&B logging will be skipped. Run the parent with a wandb-capable python "
          "(repo .venv) from a dir without a wandb/ subdir.", flush=True)
    return False


def maybe_log_wandb(args, payload) -> str | None:
    _ensure_real_wandb()
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
    except Exception as exc:  # noqa: BLE001
        print(f"[doc] wandb logging unavailable: {exc}", flush=True)
        return None
    v = payload["verdict"]
    run = init_wandb_run(
        job_type="decode-overhead-graph-audit", agent="denken",
        name=args.wandb_name, group=args.wandb_group,
        notes="PR #674: CUDA-graph capture audit + decode_overhead_frac on the SHIPPED "
              "int4_g128_lmhead body at the live AR M=1 serve config. Orthogonal, quality-"
              "identical second speed axis (stacks with spec/drafter). analysis_only.",
        tags=["decode-overhead", "cuda-graph", "capture-audit", "int4-g128-lmhead",
              "graph-knob-sweep", "analysis-only", "pr-674", "local-only", "no-fire"],
        config={"analysis_only": True, "official_tps": 0, "pr": 674,
                "model_id": v["model_id"], "live_official_tps": LIVE_OFFICIAL_TPS,
                "local_ar_anchor_ref": LOCAL_AR_ANCHOR, "stark_tax": STARK_TAX,
                "fire_bar": FIRE_BAR, "vllm": "0.22.1rc1.dev307"},
    )
    if run is None:
        print("[doc] wandb: no run (no API key / disabled) -- skipping", flush=True)
        return None
    flat = {k: val for k, val in v.items()
            if isinstance(val, (int, float, bool, str)) and not k.startswith("_")}
    flat.update({f"selftest_{k}": int(b) for k, b in payload["self_test"].items()})
    log_summary(run, flat, step=0)
    log_json_artifact(run, name="decode_overhead_graph_audit",
                      artifact_type="profiling", data=payload)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[doc] wandb logged {len(flat)} keys; run id {rid}", flush=True)
    return rid


# Arm plan: (tag, cg_mode, do_profile). >=2 fresh servers for the two head-to-head
# graph candidates (FULL_AND_PIECEWISE default + PIECEWISE); FULL_DECODE_ONLY + the
# two graphs-off floors (NONE compile-on, eager compile-off) as context.
FULL_ARMS = [
    ("fap_a", "FULL_AND_PIECEWISE", True),    # default/live: audit + breakdown + anchor
    ("fap_b", "FULL_AND_PIECEWISE", False),   # 2nd fresh server
    ("pw_a", "PIECEWISE", False),
    ("pw_b", "PIECEWISE", False),             # 2nd fresh server
    ("fdo_a", "FULL_DECODE_ONLY", False),     # context
    ("none_a", "NONE", True),                 # graphs OFF (compile ON): bubble exposed
    ("eager_a", "eager", False),              # full eager floor
]
SMOKE_ARMS = [("fap_a", "FULL_AND_PIECEWISE", True)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name",
                    default="denken/decode-overhead-graph-audit")
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="decode-overhead-graph-audit-denken")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--model-id", default="/workspace/gemma_build/int4_g128_lmhead")
    ap.add_argument("--tps-tokens", type=int, default=256)
    ap.add_argument("--profile-tokens", type=int, default=256)
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="single tiny default arm (32 tok) to validate the pipeline")
    ap.add_argument("--out", type=Path, default=HERE / "decode_overhead_graph_audit.json")
    args = ap.parse_args(argv)

    if args.smoke:
        args.tps_tokens = min(args.tps_tokens, 32)
        args.profile_tokens = min(args.profile_tokens, 32)
        args.n_reps = min(args.n_reps, 2)

    try:
        from scripts.local_validation import paths
        for note in paths.prepare_local_gpu_env():
            print(f"[doc] {note}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[doc] prepare_local_gpu_env skipped: {exc}", flush=True)

    if args.server_python and Path(args.server_python).exists():
        server_python = Path(args.server_python)
    else:
        from scripts.local_validation import harness
        manifest = harness.load_manifest((REPO_ROOT / "submissions" / "int4_g128_lmhead").resolve())
        server_python = harness.ensure_server_venv(manifest["dependencies"])
    print(f"[doc] server_python={server_python}", flush=True)

    state_dir = (HERE / ("smoke" if args.smoke else "run")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    plan = SMOKE_ARMS if args.smoke else FULL_ARMS
    print(f"[doc] int4_g128_lmhead={args.model_id}  arms={[t for t,_,_ in plan]}  "
          f"tps_tokens={args.tps_tokens} profile_tokens={args.profile_tokens} reps={args.n_reps}",
          flush=True)
    print(f"[doc] anchors: live official {LIVE_OFFICIAL_TPS} TPS | local AR {LOCAL_AR_ANCHOR} | "
          f"tax {STARK_TAX} | fire bar {FIRE_BAR}", flush=True)

    t0 = time.time()
    arms: dict[str, list[dict[str, Any]]] = {}
    for tag, cg_mode, do_profile in plan:
        data = run_arm(server_python, tag, cg_mode, do_profile, state_dir,
                       args.tps_tokens, args.profile_tokens, args.n_reps, args.model_id)
        arms.setdefault(cg_mode, []).append(data)
    elapsed = time.time() - t0

    verdict = build_verdict(arms, args.model_id)
    st = self_test(verdict, arms)
    payload = {"verdict": verdict, "self_test": st, "elapsed_s": elapsed,
               "tps_tokens": args.tps_tokens, "profile_tokens": args.profile_tokens,
               "n_reps": args.n_reps,
               "arms": {m: [{k: a[k] for k in a if k != "kernel_rows"} for a in lst]
                        for m, lst in arms.items()}}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[doc] wrote {args.out}", flush=True)

    v = verdict
    print("\n============ DECODE-OVERHEAD + CUDA-GRAPH CAPTURE AUDIT (PR #674) ============", flush=True)
    print(f"(1) CAPTURE: mode={v['cudagraph_mode_resolved']} sizes={v['cudagraph_capture_sizes']} "
          f"bs1_captured={v['bs1_captured']} | lm_head_in_graph={v['lmhead_inside_graph']} "
          f"sampler_in_graph={v['sampler_inside_graph']}", flush=True)
    print(f"    Marlin={v['marlin_confirmed']} Machete={v['machete_present']} sm={v['sm']} "
          f"| VLLM_GRAPH_RESERVED_MEM_exists={v['vllm_graph_reserved_mem_exists']}", flush=True)
    for ln in v["_capture_log_lines"]:
        print(f"      log> {ln}", flush=True)
    print(f"(2) decode_overhead_frac = {v['decode_overhead_frac']*100:.2f}%  "
          f"(raw {v['decode_overhead_frac_raw']*100:+.2f}%) | busy/tok {v['gpu_busy_per_token_us']:.0f}us "
          f"vs clean wall/tok {v['clean_wall_per_token_us']:.0f}us", flush=True)
    print(f"    graphs-OFF(NONE) bubble = {v['decode_overhead_frac_graphsoff_none']*100:.1f}% "
          f"(what the graph reclaims) | matmul {v['matmul_per_token_us']:.0f}us "
          f"attn {v['attn_per_token_us']:.0f}us norm {v['norm_per_token_us']:.0f}us "
          f"samp {v['sampling_per_token_us']:.0f}us", flush=True)
    print(f"(3) KNOB SWEEP (median TPS / #servers):", flush=True)
    for m, t in v["cell_median_tps"].items():
        print(f"      {m:20s} {t:8.2f}  (n_servers={v['cell_n_servers'].get(m)})", flush=True)
    print(f"      graphs-off NONE {v['cudagraph_off_none_tps']:.2f} | eager {v['cudagraph_off_eager_tps']:.2f}",
          flush=True)
    print(f"      best={v['best_graphtuned_mode']} {v['best_graphtuned_walltps']:.2f}  "
          f"gain {v['gain_local_tps']:+.2f} ({100*v['gain_local_frac']:+.2f}%)  "
          f"real_gain={v['real_gain']} (noise band {v['default_noise_band_tps']:.2f})", flush=True)
    print(f"(4) ANCHOR: re-measured {v['measured_anchor_tps']:.2f} vs ref {LOCAL_AR_ANCHOR} "
          f"({100*v['anchor_vs_ref_frac']:+.2f}%) | official-equiv best {v['official_equiv_best_tps']:.2f} "
          f"gain {v['official_equiv_gain_tps']:+.2f}", flush=True)
    print(f"GUARDS: break_rate {v['break_rate']:.3f} (graph-only {v['graph_break_rate']:.3f})", flush=True)
    print(f"VERDICT: {v['verdict']}  fires={v['fires']}  self_test={st['self_test_passes']}", flush=True)
    print("=============================================================================", flush=True)
    for k, b in st.items():
        if not b and k != "self_test_passes":
            print(f"  [self-test FAIL] {k}", flush=True)

    rid = None
    if not args.no_wandb:
        rid = maybe_log_wandb(args, payload)

    print("\nSENPAI-RESULT: " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "self_det": st["self_test_passes"], "verdict": v["verdict"],
        "decode_overhead_frac": round(v["decode_overhead_frac"], 4),
        "best_graphtuned_walltps": round(v["best_graphtuned_walltps"], 2),
        "matched_ar_anchor": round(v["measured_anchor_tps"], 2),
        "break_rate": round(v["break_rate"], 4),
        "primary_metric": {"name": "decode_overhead_frac", "value": round(v["decode_overhead_frac"], 4)},
        "test_metric": {"name": "best_graphtuned_walltps", "value": round(v["best_graphtuned_walltps"], 2)},
    }), flush=True)
    return 0 if st["self_test_passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
