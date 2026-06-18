#!/usr/bin/env python
"""Clean-room un-rescued K-sweep for the canonical Option-B submission (PR #665).

ANALYSIS-ONLY, LOCAL single-A10G. No weights changed, NO HF job, NO submission.

Stands up the canonical submitted spec stack -- ``submissions/int4_mtp_batchinv``
(int4 W4A16 ``google/gemma-4-E4B-it-qat-w4a16-ct`` target + ``gemma4_assistant``
MTP drafter, ``VLLM_BATCH_INVARIANT=1``, ``MAX_NUM_SEQS=1``, greedy) -- on the
STOCK dev307 vLLM env, taking whatever the stock defaults are for the three
spec-path regime knobs land #664 is pinning:

  * sampler backend  (``VLLM_USE_FLASHINFER_SAMPLER``)  -- NOT preset*
  * draft cudagraph  (``ENFORCE_EAGER``)                -- NOT preset (stock = graphs on)
  * draft attn-backend (``VLLM_ATTENTION_BACKEND``)     -- NOT preset (stock auto-select)

  *Container caveat: this container's flashinfer build JITs against cuRAND headers
  that live only in the pip nvidia-cu13 package, so the stock flashinfer SAMPLER
  crashes the engine at memory-profiling (see local_validation/paths.default_native_sampler).
  At temperature=0 the sampler is pure argmax and does NOT touch logits, so it is
  REGIME-NEUTRAL for greedy decode. ``--native-sampler`` sets the documented
  container shim (VLLM_USE_FLASHINFER_SAMPLER=0). The smoke decides if it is needed.

un-rescued wall_tps = total_completion_tokens / duration_s from the OFFICIAL
``decode_outputs.py`` over 128 prompts x 512 tokens, seed 1, single-stream
(decode_outputs sends requests sequentially with ignore_eos=True so each yields
exactly 512 tokens). This is the same raw-served basis land #660 / stark #642 use.

Arms (one fresh server each; ``--arms`` selects a subset so stages fit the 90-min
per-run budget):

  g128_ar     int4_g128_lmhead plain AR (MODEL_ID=/workspace/gemma_build/int4_g128_lmhead)
              -> calibration anchor #1: local AR wall ~= 126.75 (official 126.378)
  spec_ar_m1  int4_mtp_batchinv with NUM_SPECULATIVE_TOKENS=0 (w4a16-ct body, drafter off)
              -> calibration anchor #2: spec-AR-ref ~= 77.96 (land/stark agree 0.09%)
  spec_k4/5/6 int4_mtp_batchinv with NUM_SPECULATIVE_TOKENS=4/5/6  -> BLIND un-rescued sweep
  ppl         int4_mtp_batchinv NUM_SPECULATIVE_TOKENS=0 PPL pass -> confirm ~2.0055

``--finalize`` reads all records, runs the calibration gate, the #319 byte-exact
identity gate (spec_k6 vs spec_ar_m1 completion_token_sha256 -> break_rate), the
regime classification (155-band / 170-band / intermediate), scrapes spec
acceptance / fire-rate from each server's /metrics, and logs ONE wandb run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

# --- locked anchors (PR #665) ----------------------------------------------
DEV307_VENV = Path("/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python")  # vllm 0.22.1rc1.dev307
G128_CKPT = "/workspace/gemma_build/int4_g128_lmhead"
DRAFTER_LOCAL = "/tmp/qat-assistant"  # the gemma4_assistant MTP drafter (QAT-matched)
SUB_SPEC = ROOT / "submissions" / "int4_mtp_batchinv"
SUB_G128 = ROOT / "submissions" / "int4_g128_lmhead"

ANCHOR_G128_AR = 126.75       # local AR wall (official int4_g128_lmhead 126.378)
ANCHOR_SPEC_AR = 77.96        # spec-AR-ref (land 77.96 / stark 77.89, agree 0.09%)
LAND_K6 = 170.16              # land #660 un-rescued K6 (fast regime)
STARK_K6 = 155.58             # stark #642 un-rescued K6 (slow regime)
REF_PPL = 2.0055
OFFICIAL_ANCHOR_TPS = 126.378
FIRE_BAR = 126.378
INCREMENTAL_BAR = 136.378     # +10 over the anchor
# rescue factors (un-rescued -> rescued strict-#319), from land #660 band:
RESCUE_STARK = 135.82 / 155.58   # ~0.873 conservative basis
RESCUE_LAND = 146.82 / 170.16    # ~0.863 optimistic basis

CALIB_CLEAN_PCT = 0.3   # "calibrated" if both anchors within this
CALIB_ABORT_PCT = 1.5   # HARNESS_UNCALIBRATED if either anchor off by more than this

OUT_ROOT = ROOT / "research" / "validity" / "independent_ceiling_repro" / "run"


def arm_specs() -> dict[str, dict[str, Any]]:
    """submission dir + extra_env (regime knobs intentionally absent)."""
    spec_env = lambda k: {"NUM_SPECULATIVE_TOKENS": str(k), "DRAFTER_MODEL": DRAFTER_LOCAL}
    return {
        "g128_ar":    {"sub": SUB_G128, "env": {"MODEL_ID": G128_CKPT}, "kind": "ar_anchor",
                       "anchor": ANCHOR_G128_AR},
        "spec_ar_m1": {"sub": SUB_SPEC, "env": spec_env(0), "kind": "spec_ar_anchor",
                       "anchor": ANCHOR_SPEC_AR},
        "spec_k4":    {"sub": SUB_SPEC, "env": spec_env(4), "kind": "spec", "K": 4},
        "spec_k5":    {"sub": SUB_SPEC, "env": spec_env(5), "kind": "spec", "K": 5},
        "spec_k6":    {"sub": SUB_SPEC, "env": spec_env(6), "kind": "spec", "K": 6},
        # PPL anchor: int4 body with speculation OFF (drafter irrelevant at K=0).
        # Run with --reps 0 --ppl-arm ppl: boot -> teacher-forced PPL only, no
        # throughput rep. PPL is K-independent (teacher-forced on the target's
        # full-vocab logits) so this confirms the body numerics ~= 2.0055.
        "ppl":        {"sub": SUB_SPEC, "env": spec_env(0), "kind": "ppl_anchor"},
    }


# ---------------------------------------------------------------------------
# GPU + process plumbing (mirrors scripts/profiler/serveconfig_tps_sweep.py #640)
# ---------------------------------------------------------------------------
def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def preflight(threshold_mib: int = 1500, timeout_s: int = 180) -> int | None:
    reaped = False
    for pat in ["vllm.entrypoints.openai.api_server", "VLLM::EngineCore"]:
        r = subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        reaped = reaped or (r.returncode == 0)
    if reaped:
        time.sleep(4)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = _gpu_mem_used_mib()
        if used is None or used < threshold_mib:
            return used
        time.sleep(3)
    return _gpu_mem_used_mib()


def scrape_metrics(base_url: str) -> dict[str, float]:
    """Pull spec-decode counters from the vLLM /metrics Prometheus endpoint."""
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return out
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9eE.+-]+)$", line.strip())
        if not m:
            continue
        name, val = m.group(1), m.group(3)
        if "spec_decode" in name or "accepted" in name or "draft" in name or "num_preempt" in name:
            try:
                out[name] = out.get(name, 0.0) + float(val)
            except ValueError:
                pass
    return out


def parse_server_log(log_path: Path) -> dict[str, Any]:
    """Best-effort resolved-env audit from the vLLM boot log.

    Captures the three land #664 regime knobs as the dev307 engine actually
    resolved them, so the report can state the exact stock config measured.
    """
    info: dict[str, Any] = {
        "attn_backend": None, "enforce_eager": None, "cudagraph_mode": None,
        "compilation_mode": None, "spec_num_tokens": None,
        "flashinfer_sampler_crash": False, "flashinfer_sampler_disabled": False,
        "attn_patch_failed": False, "spec_method": None, "drafter": None,
    }
    if not log_path.exists():
        return info
    txt = log_path.read_text(errors="replace")
    for pat, key in [
        # "Using AttentionBackendEnum.TRITON_ATTN backend." -> TRITON_ATTN
        (r"AttentionBackendEnum\.([A-Z0-9_]+)", "attn_backend"),
        (r"Using ([A-Z0-9_]+) backend", "attn_backend"),
        (r"enforce_eager=(\w+)", "enforce_eager"),
        # "'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE:" -> FULL_AND_PIECEWISE
        (r"CUDAGraphMode\.([A-Z_]+)", "cudagraph_mode"),
        (r"CompilationMode\.([A-Z_]+)", "compilation_mode"),
        (r"speculative_config=SpeculativeConfig\(method='([a-z0-9_]+)'", "spec_method"),
        (r"num_spec_tokens=(\d+)", "spec_num_tokens"),
        (r"SpeculativeConfig\(method='[a-z0-9_]+', model='([^']+)'", "drafter"),
    ]:
        m = re.search(pat, txt)
        if m and info.get(key) is None:
            info[key] = m.group(1)
    if "curand.h" in txt or "Ninja build failed" in txt:
        info["flashinfer_sampler_crash"] = True
    if "FlashInfer top-p/top-k sampling disabled" in txt:
        info["flashinfer_sampler_disabled"] = True
    if "failed to apply attention-group num_heads patch" in txt:
        info["attn_patch_failed"] = True
    if info["spec_method"] is None:
        m = re.search(r"gemma4_mtp|gemma4_assistant", txt)
        if m:
            info["spec_method"] = m.group(0)
    return info


# ---------------------------------------------------------------------------
# #319 fingerprint
# ---------------------------------------------------------------------------
def fingerprint(jsonl_path: Path) -> tuple[list[str], list[int]]:
    rows: list[tuple[int, str, int]] = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        rows.append((int(o["index"]), str(o["completion_token_sha256"]),
                     int(o["num_completion_tokens"])))
    rows.sort()
    return [s for _, s, _ in rows], [n for _, _, n in rows]


# ---------------------------------------------------------------------------
# measure one arm: serve -> timed reps -> kill
# ---------------------------------------------------------------------------
def measure_arm(name: str, spec: dict[str, Any], out_dir: Path, *,
                num_prompts: int, reps: int, port: int, native_sampler: bool,
                do_ppl: bool = False) -> dict[str, Any]:
    base = out_dir / name
    base.mkdir(parents=True, exist_ok=True)
    extra_env = dict(spec["env"])
    if native_sampler:
        extra_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # documented container shim

    rec: dict[str, Any] = {
        "name": name, "kind": spec["kind"], "K": spec.get("K"),
        "submission": spec["sub"].name, "extra_env": extra_env,
        "t_start_utc": datetime.now(timezone.utc).isoformat(),
        "served_ok": False, "error": None, "rep_wall_tps": [], "wall_tps": None,
        "ready_s": None, "gpu_mem_used_mib": None, "fingerprint": None,
        "completion_counts": None, "num_completion_tokens": None, "duration_s": [],
        "ppl": None, "metrics": {}, "resolved_env": {},
    }

    used_before = preflight()
    rec["gpu_mem_used_before_mib"] = used_before
    base_url = f"http://127.0.0.1:{port}"
    log_path = base / "server.log"
    server = None
    try:
        t0 = time.time()
        server = harness.LocalServer(
            spec["sub"], server_python=DEV307_VENV, port=port,
            log_path=log_path, extra_env=extra_env)
        server.__enter__()
        rec["ready_s"] = round(time.time() - t0, 1)
        rec["served_ok"] = True
        rec["model_id"] = server.model_id
        # full resolved env block the server received (manifest env + our overrides)
        rec["resolved_env"] = {k: server.env.get(k) for k in sorted(
            set(list(server.manifest.get("env", {}).keys()) + list(extra_env.keys())
                + ["VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ATTENTION_BACKEND", "ENFORCE_EAGER"]))}

        for r in range(reps):
            of = base / f"rep{r}.jsonl"
            sf = base / f"rep{r}.summary.json"
            s = harness.capture_decode(
                DEV307_VENV, base_url=base_url, model=server.served_model_name,
                out_file=of, summary_file=sf, num_prompts=num_prompts)
            n = int(s.get("num_completion_tokens", 0))
            d = float(s.get("duration_s", 0.0))
            tps = n / d if d > 0 else float("nan")
            rec["rep_wall_tps"].append(tps)
            rec["duration_s"].append(d)
            rec["num_completion_tokens"] = n
            if r == 0:
                fp, counts = fingerprint(of)
                rec["fingerprint"] = fp
                rec["completion_counts"] = counts
                rec["full_length"] = all(c == 512 for c in counts) if counts else False
            print(f"  [{name}] rep{r} wall_tps={tps:.3f} (n={n} dur={d:.1f}s)", flush=True)

        vals = [v for v in rec["rep_wall_tps"] if v == v]
        rec["wall_tps"] = statistics.median(vals) if vals else float("nan")
        rec["wall_tps_min"] = min(vals) if vals else None
        rec["wall_tps_max"] = max(vals) if vals else None
        rec["wall_tps_spread_pct"] = (100.0 * (max(vals) - min(vals)) / min(vals)
                                      if len(vals) > 1 and min(vals) else 0.0)
        rec["gpu_mem_used_mib"] = _gpu_mem_used_mib()
        rec["metrics"] = scrape_metrics(base_url)

        if do_ppl:
            try:
                ppl = harness.run_ppl(DEV307_VENV, base_url=base_url,
                                      model=server.served_model_name,
                                      out_file=base / "ppl.jsonl",
                                      summary_file=base / "ppl.summary.json")
                rec["ppl"] = ppl.get("ppl")
                print(f"  [{name}] ppl={rec['ppl']}", flush=True)
            except Exception as exc:
                rec["ppl_error"] = str(exc)
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"  [{name}] ERROR: {exc}", flush=True)
    finally:
        if server is not None:
            try:
                server.__exit__(None, None, None)
            except Exception:
                pass
    rec.update(parse_server_log(log_path))
    return rec


# ---------------------------------------------------------------------------
# finalize: calibration + identity + regime + verdict + wandb
# ---------------------------------------------------------------------------
def _delta_pct(v: float | None, ref: float) -> float | None:
    if v is None or v != v:
        return None
    return round(100.0 * (v - ref) / ref, 3)


def _read_completions(jsonl_path: Path) -> dict[int, tuple[str, list[int]]]:
    """{index: (completion_token_sha256, completion_token_ids)} from a decode jsonl."""
    out: dict[int, tuple[str, list[int]]] = {}
    if not jsonl_path.exists():
        return out
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[int(o["index"])] = (str(o["completion_token_sha256"]),
                                list(o.get("completion_token_ids") or []))
    return out


def _seq_and_hazard(a: dict[int, tuple[str, list[int]]],
                    b: dict[int, tuple[str, list[int]]]) -> dict[str, Any]:
    """Sequence-level sha256 break + per-step ROOT (first-divergence) hazard.

    The hazard CENSORS each sequence at its first divergence: once the shared
    prefix breaks, all downstream tokens are decoded from a different context
    and are no longer an independent identity test, so counting them would
    massively over-state divergence (a single early int4-grid tie-flip cascades
    to the whole tail). The root hazard is the physically meaningful per-step
    flip rate; the sha256 seq-break rate is its cascade-amplified shadow.
    """
    common = sorted(set(a) & set(b))
    if not common:
        return {"n": 0, "seq_break": None, "seq_break_rate": None,
                "root_flips": None, "at_risk": None, "hazard": None}
    seq_break = sum(1 for i in common if a[i][0] != b[i][0])
    at_risk = 0
    root = 0
    for i in common:
        x, y = a[i][1], b[i][1]
        n = min(len(x), len(y))
        for j in range(n):
            at_risk += 1
            if x[j] != y[j]:
                root += 1
                break
    return {"n": len(common), "seq_break": seq_break,
            "seq_break_rate": round(seq_break / len(common), 6),
            "root_flips": root, "at_risk": at_risk,
            "hazard": round(root / at_risk, 6) if at_risk else None}


def finalize(out_dir: Path, args) -> dict[str, Any]:
    records = []
    rp = out_dir / "records.jsonl"
    seen: dict[str, dict] = {}
    for line in rp.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["name"]] = r  # last write wins
    records = list(seen.values())
    by = seen

    # ---- calibration ----
    g128 = by.get("g128_ar", {}).get("wall_tps")
    sar = by.get("spec_ar_m1", {}).get("wall_tps")
    calib = {
        "g128_ar_wall_tps": g128, "g128_ar_delta_pct": _delta_pct(g128, ANCHOR_G128_AR),
        "spec_ar_m1_wall_tps": sar, "spec_ar_m1_delta_pct": _delta_pct(sar, ANCHOR_SPEC_AR),
    }
    worst = max([abs(d) for d in (calib["g128_ar_delta_pct"], calib["spec_ar_m1_delta_pct"])
                 if d is not None], default=None)
    calib["worst_anchor_abs_pct"] = worst
    calib["calibrated_clean"] = (worst is not None and worst <= CALIB_CLEAN_PCT)
    calib["uncalibrated"] = (worst is not None and worst > CALIB_ABORT_PCT)

    # ---- #319 identity (spec_k6 vs spec_ar_m1) ----
    fp_ref = by.get("spec_ar_m1", {}).get("fingerprint")
    fp_k6 = by.get("spec_k6", {}).get("fingerprint")
    identity: dict[str, Any] = {"break_rate": None, "n_break": None, "n_total": None,
                                "spec_fire_rate": None}
    if fp_ref and fp_k6:
        n = min(len(fp_ref), len(fp_k6))
        n_break = sum(1 for a, b in zip(fp_ref, fp_k6) if a != b)
        identity["n_total"] = n
        identity["n_break"] = n_break
        identity["break_rate"] = round(n_break / n, 6) if n else None
    # spec_fire_rate + acceptance from spec_k6 /metrics
    mk6 = by.get("spec_k6", {}).get("metrics", {}) or {}
    acc = mk6.get("vllm:spec_decode_num_accepted_tokens_total") or \
        mk6.get("vllm:spec_decode_num_accepted_tokens")
    drf = mk6.get("vllm:spec_decode_num_draft_tokens_total") or \
        mk6.get("vllm:spec_decode_num_draft_tokens")
    if acc is not None and drf:
        identity["acceptance_rate"] = round(acc / drf, 6)
    identity["spec_fire_rate"] = 1.0 if (drf and drf > 0) else (
        0.0 if by.get("spec_k6", {}).get("kind") == "spec" else None)
    identity["spec_metrics_raw"] = mk6

    # ---- identity FLOOR + token-level hazard ----
    # The strict per-prompt sha256 break_rate above is ONLY a meaningful #319
    # gate if the within-config run-to-run reproducibility floor is ~0. On stock
    # dev307 the int4 W4A16 decode is NOT byte-exact reproducible run-to-run
    # (atomic/ordering-sensitive Marlin reductions flip int4-grid ties), so we
    # measure each arm's own rep0-vs-rep1 floor and the spec-vs-AR token-level
    # ROOT hazard, and report whether the strict gate is informative at all.
    def _arm_floor(arm: str) -> dict[str, Any]:
        return _seq_and_hazard(_read_completions(out_dir / arm / "rep0.jsonl"),
                               _read_completions(out_dir / arm / "rep1.jsonl"))
    floors = {a: _arm_floor(a) for a in ("g128_ar", "spec_ar_m1", "spec_k4",
                                         "spec_k5", "spec_k6")}
    spec_vs_ar_tok = _seq_and_hazard(
        _read_completions(out_dir / "spec_ar_m1" / "rep0.jsonl"),
        _read_completions(out_dir / "spec_k6" / "rep0.jsonl"))
    ar_floor_haz = max([floors[a].get("hazard") for a in ("g128_ar", "spec_ar_m1")
                        if floors[a].get("hazard") is not None], default=None)
    identity["floors"] = floors
    identity["spec_vs_ar_token"] = spec_vs_ar_tok
    identity["ar_run_floor_hazard"] = ar_floor_haz
    # strict sha256 is only a valid identity test if the AR self-floor is ~0
    identity["strict_gate_informative"] = (
        ar_floor_haz is not None and ar_floor_haz < 1e-4)
    sk6 = floors.get("spec_k6", {})
    identity["spec_excess_over_self_floor_hazard"] = (
        round(spec_vs_ar_tok["hazard"] - sk6["hazard"], 6)
        if (spec_vs_ar_tok.get("hazard") is not None
            and sk6.get("hazard") is not None) else None)

    # ---- regime classification ----
    k6 = by.get("spec_k6", {}).get("wall_tps")
    regime: dict[str, Any] = {"blind_unrescued_k6_walltps": k6}
    if k6 is not None and k6 == k6:
        d_stark = abs(k6 - STARK_K6)
        d_land = abs(k6 - LAND_K6)
        mid = (STARK_K6 + LAND_K6) / 2
        if k6 <= STARK_K6 + (LAND_K6 - STARK_K6) * 0.33:
            band = "slow_155"
        elif k6 >= STARK_K6 + (LAND_K6 - STARK_K6) * 0.67:
            band = "fast_170"
        else:
            band = "intermediate"
        regime.update({
            "regime_band": band, "dist_to_stark_155": round(d_stark, 3),
            "dist_to_land_170": round(d_land, 3), "midpoint_155_170": mid,
            "rescued_k6_starkbasis": round(k6 * RESCUE_STARK, 3),
            "rescued_k6_landbasis": round(k6 * RESCUE_LAND, 3),
        })

    # ---- fresh-server confirmation enrichment (median-of-4 + reproducibility) ----
    # round-1 spec_k6 landed fast (172) with a counter-intuitive step_ms drop
    # K5->K6; the run_confirm/ round re-measured K4/5/6 on independent fresh
    # servers. Merge both rounds' reps into median-of-4 per K and record whether
    # the fast K6 band reproduces (so the tie-breaking vote is not single-server).
    confirm_rp = out_dir.parent / "run_confirm" / "records.jsonl"
    if confirm_rp.exists() and regime.get("regime_band"):
        cf: dict[str, dict] = {}
        for line in confirm_rp.read_text().splitlines():
            if line.strip():
                rr = json.loads(line)
                cf[rr["name"]] = rr
        combined: dict[str, float] = {}
        for arm in ("spec_k4", "spec_k5", "spec_k6"):
            reps = [v for v in (by.get(arm, {}).get("rep_wall_tps") or []) if v == v]
            reps += [v for v in (cf.get(arm, {}).get("rep_wall_tps") or []) if v == v]
            if reps:
                combined[arm] = round(statistics.median(reps), 3)
        cb_band = None
        if "spec_k6" in combined:
            kk = combined["spec_k6"]
            if kk <= STARK_K6 + (LAND_K6 - STARK_K6) * 0.33:
                cb_band = "slow_155"
            elif kk >= STARK_K6 + (LAND_K6 - STARK_K6) * 0.67:
                cb_band = "fast_170"
            else:
                cb_band = "intermediate"
        regime["confirm"] = {
            "confirm_k6_walltps": cf.get("spec_k6", {}).get("wall_tps"),
            "confirm_k5_walltps": cf.get("spec_k5", {}).get("wall_tps"),
            "confirm_k4_walltps": cf.get("spec_k4", {}).get("wall_tps"),
            "combined_k4_median4": combined.get("spec_k4"),
            "combined_k5_median4": combined.get("spec_k5"),
            "combined_k6_median4": combined.get("spec_k6"),
            "combined_k6_band": cb_band,
            "k6_reproduced_fast": bool(cb_band == regime["regime_band"]),
        }

    # ---- verdict ----
    if calib["uncalibrated"]:
        verdict = "HARNESS_UNCALIBRATED"
    elif k6 is None or k6 != k6:
        verdict = "HARNESS_UNCALIBRATED"
    else:
        band = regime["regime_band"]
        verdict = {"slow_155": "DEFAULT_IS_SLOW_REGIME",
                   "fast_170": "DEFAULT_IS_FAST_REGIME",
                   "intermediate": "DEFAULT_IS_INTERMEDIATE"}[band]

    # payoff: which rescued band, does it clear fire / +10 bars
    payoff: dict[str, Any] = {}
    if regime.get("regime_band"):
        if verdict == "DEFAULT_IS_SLOW_REGIME":
            proj = regime["rescued_k6_starkbasis"]
            chosen = "conservative_135"
        elif verdict == "DEFAULT_IS_FAST_REGIME":
            proj = regime["rescued_k6_landbasis"]
            chosen = "optimistic_147"
        else:
            proj = round((regime["rescued_k6_starkbasis"] + regime["rescued_k6_landbasis"]) / 2, 3)
            chosen = "midpoint"
        payoff = {
            "chosen_rescued_band": chosen, "projected_rescued_k6": proj,
            "clears_fire_bar_126378": proj > FIRE_BAR,
            "clears_incremental_bar_136378": proj > INCREMENTAL_BAR,
            "margin_over_fire": round(proj - FIRE_BAR, 3),
            "margin_over_incremental": round(proj - INCREMENTAL_BAR, 3),
        }

    ppl_val = next((by[a].get("ppl") for a in ("ppl", "spec_ar_m1", "spec_k6")
                    if by.get(a, {}).get("ppl") is not None), None)

    summary = {
        "verdict": verdict, "calibration": calib, "identity": identity,
        "regime": regime, "payoff": payoff, "ppl": ppl_val, "ref_ppl": REF_PPL,
        "anchors": {"g128_ar": ANCHOR_G128_AR, "spec_ar_m1": ANCHOR_SPEC_AR,
                    "land_k6": LAND_K6, "stark_k6": STARK_K6},
        "primary_metric": {"name": "blind_unrescued_k6_walltps", "value": k6},
        "test_metric": {"name": "regime_band", "value": regime.get("regime_band")},
        "arms": {r["name"]: {"wall_tps": r.get("wall_tps"), "kind": r.get("kind"),
                             "ready_s": r.get("ready_s"), "served_ok": r.get("served_ok"),
                             "resolved_env": r.get("resolved_env"),
                             "attn_backend": r.get("attn_backend"),
                             "flashinfer_sampler_crash": r.get("flashinfer_sampler_crash"),
                             "attn_patch_failed": r.get("attn_patch_failed"),
                             "rep_wall_tps": r.get("rep_wall_tps"),
                             "wall_tps_spread_pct": r.get("wall_tps_spread_pct")}
                 for r in records},
    }
    (out_dir / "summary.json").write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2, default=str))
    run_id = _log_wandb(args, records, summary) if not args.no_wandb else None
    summary["wandb_run_id"] = run_id
    (out_dir / "summary.json").write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2, default=str))
    _print_summary(summary)
    return summary


def _print_summary(s: dict[str, Any]) -> None:
    print("\n===================== VERDICT =====================", flush=True)
    print(f"  VERDICT: {s['verdict']}", flush=True)
    c = s["calibration"]
    print(f"  calib  g128_ar   = {c['g128_ar_wall_tps']} (Δ {c['g128_ar_delta_pct']}% vs {ANCHOR_G128_AR})", flush=True)
    print(f"  calib  spec_ar_m1= {c['spec_ar_m1_wall_tps']} (Δ {c['spec_ar_m1_delta_pct']}% vs {ANCHOR_SPEC_AR})", flush=True)
    print(f"  calibrated_clean={c['calibrated_clean']} uncalibrated={c['uncalibrated']} worst={c['worst_anchor_abs_pct']}%", flush=True)
    r = s["regime"]
    print(f"  BLIND un-rescued K6 = {r.get('blind_unrescued_k6_walltps')}  band={r.get('regime_band')}", flush=True)
    print(f"    dist->stark155={r.get('dist_to_stark_155')} dist->land170={r.get('dist_to_land_170')}", flush=True)
    print(f"    rescued stark-basis={r.get('rescued_k6_starkbasis')} land-basis={r.get('rescued_k6_landbasis')}", flush=True)
    i = s["identity"]
    print(f"  identity strict-sha256 break_rate={i['break_rate']} (n_break={i['n_break']}/{i['n_total']}) "
          f"spec_fire_rate={i['spec_fire_rate']} acceptance={i.get('acceptance_rate')}", flush=True)
    fl = i.get("floors", {}) or {}
    def _h(a):
        return (fl.get(a) or {}).get("hazard")
    print(f"  identity RUN-FLOOR hazard/step: g128_ar={_h('g128_ar')} spec_ar_m1={_h('spec_ar_m1')} "
          f"spec_k6={_h('spec_k6')}  (AR floor>0 => strict sha256 gate uninformative)", flush=True)
    print(f"  identity spec_vs_ar root_hazard={(i.get('spec_vs_ar_token') or {}).get('hazard')} "
          f"excess_over_self_floor={i.get('spec_excess_over_self_floor_hazard')} "
          f"strict_gate_informative={i.get('strict_gate_informative')}", flush=True)
    print(f"  PPL={s['ppl']} (ref {REF_PPL})", flush=True)
    print(f"  payoff: {s['payoff']}", flush=True)
    for name, a in s["arms"].items():
        print(f"    {name:12s} tps={a['wall_tps']} ready={a['ready_s']}s attn={a.get('attn_backend')} "
              f"reps={[round(v,2) for v in (a.get('rep_wall_tps') or [])]}", flush=True)


def _log_wandb(args, records: list[dict[str, Any]], summary: dict[str, Any]) -> str | None:
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[wandb] import failed ({exc})", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="clean-room-kceil", agent="wirbel",
        name=args.wandb_name or "wirbel/independent-ceiling-repro",
        group=args.wandb_group,
        tags=["independent-ceiling-repro", "int4_mtp_batchinv", "analysis-only",
              "dev307-stock", summary["verdict"]],
        config={
            "submission": "int4_mtp_batchinv", "drafter": DRAFTER_LOCAL,
            "engine": "vllm 0.22.1rc1.dev307+g3e8afdf78", "num_prompts": args.num_prompts,
            "reps": args.reps, "max_num_seqs": 1, "batch_invariant": 1,
            "native_sampler_shim": args.native_sampler, "analysis_only": True,
            "anchor_g128_ar": ANCHOR_G128_AR, "anchor_spec_ar_m1": ANCHOR_SPEC_AR,
            "land_k6": LAND_K6, "stark_k6": STARK_K6,
        })
    if run is None:
        print("[wandb] disabled", flush=True)
        return None
    rid = None
    try:
        rid = run.id
        for i, r in enumerate(records):
            m = {f"arm/{r['name']}/wall_tps": r.get("wall_tps"),
                 f"arm/{r['name']}/ready_s": r.get("ready_s"),
                 f"arm/{r['name']}/gpu_mem_used_mib": r.get("gpu_mem_used_mib")}
            m = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, f"arm_{r['name']}", step=i, metrics=m,
                                    data={"arm": r["name"], "kind": r.get("kind")})
        # explicit machine-checkable scalars the advisor asked for
        flat = {
            "verdict": summary["verdict"],
            "blind_unrescued_k6_walltps": summary["regime"].get("blind_unrescued_k6_walltps"),
            "regime_band": summary["regime"].get("regime_band"),
            "break_rate": summary["identity"].get("break_rate"),
            "spec_fire_rate": summary["identity"].get("spec_fire_rate"),
            "acceptance_rate": summary["identity"].get("acceptance_rate"),
            "spec_vs_ar_root_hazard": (summary["identity"].get("spec_vs_ar_token") or {}).get("hazard"),
            "ar_run_floor_hazard": summary["identity"].get("ar_run_floor_hazard"),
            "floor_g128_ar_hazard": (summary["identity"].get("floors", {}).get("g128_ar") or {}).get("hazard"),
            "floor_spec_ar_m1_hazard": (summary["identity"].get("floors", {}).get("spec_ar_m1") or {}).get("hazard"),
            "floor_spec_k6_hazard": (summary["identity"].get("floors", {}).get("spec_k6") or {}).get("hazard"),
            "floor_g128_ar_seq_break": (summary["identity"].get("floors", {}).get("g128_ar") or {}).get("seq_break"),
            "strict_gate_informative": 1 if summary["identity"].get("strict_gate_informative") else 0,
            "spec_excess_over_self_floor_hazard": summary["identity"].get("spec_excess_over_self_floor_hazard"),
            "g128_ar_wall_tps": summary["calibration"].get("g128_ar_wall_tps"),
            "g128_ar_delta_pct": summary["calibration"].get("g128_ar_delta_pct"),
            "spec_ar_m1_wall_tps": summary["calibration"].get("spec_ar_m1_wall_tps"),
            "spec_ar_m1_delta_pct": summary["calibration"].get("spec_ar_m1_delta_pct"),
            "calibrated_clean": 1 if summary["calibration"].get("calibrated_clean") else 0,
            "ppl": summary.get("ppl"),
            "rescued_k6_starkbasis": summary["regime"].get("rescued_k6_starkbasis"),
            "rescued_k6_landbasis": summary["regime"].get("rescued_k6_landbasis"),
            "projected_rescued_k6": summary["payoff"].get("projected_rescued_k6"),
            "clears_fire_bar_126378": 1 if summary["payoff"].get("clears_fire_bar_126378") else 0,
            "clears_incremental_bar_136378": 1 if summary["payoff"].get("clears_incremental_bar_136378") else 0,
            "confirm_k6_walltps": (summary["regime"].get("confirm") or {}).get("confirm_k6_walltps"),
            "combined_k4_median4": (summary["regime"].get("confirm") or {}).get("combined_k4_median4"),
            "combined_k5_median4": (summary["regime"].get("confirm") or {}).get("combined_k5_median4"),
            "combined_k6_median4": (summary["regime"].get("confirm") or {}).get("combined_k6_median4"),
            "combined_k6_band": (summary["regime"].get("confirm") or {}).get("combined_k6_band"),
            "k6_reproduced_fast": 1 if (summary["regime"].get("confirm") or {}).get("k6_reproduced_fast") else 0,
        }
        flat = {f"result/{k}": v for k, v in flat.items() if v is not None}
        wandb_logging.log_summary(run, flat, step=len(records))
        wandb_logging.log_json_artifact(run, name="clean_room_kceil",
                                        artifact_type="clean-room-kceil",
                                        data={"summary": summary, "records": records})
    except Exception as exc:
        print(f"[wandb] WARN {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    return rid


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=None, help="comma list of arms to run this stage")
    ap.add_argument("--finalize", action="store_true", help="aggregate records -> verdict + wandb")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--native-sampler", action="store_true",
                    help="set VLLM_USE_FLASHINFER_SAMPLER=0 (container shim; regime-neutral greedy)")
    ap.add_argument("--ppl-arm", default=None, help="arm name to also run a PPL pass on")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="independent-ceiling-repro-wirbel")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # GPU device normalization only (do NOT force the sampler -- that's the knob).
    note = paths.normalize_cuda_visible_devices()
    if note:
        print(f"[gpu] {note}", flush=True)
    # clean slate: the stock default must not inherit a sampler pin from the parent
    if not args.native_sampler:
        os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)

    if args.finalize:
        finalize(out_dir, args)
        return 0

    specs = arm_specs()
    plan = [a.strip() for a in (args.arms or "").split(",") if a.strip()]
    if not plan:
        print("no --arms given; nothing to do (use --finalize to aggregate)", flush=True)
        return 0
    for a in plan:
        if a not in specs:
            raise SystemExit(f"unknown arm {a!r}; choices={list(specs)}")

    print(f"[stage] arms={plan} num_prompts={args.num_prompts} reps={args.reps} "
          f"native_sampler={args.native_sampler} -> {out_dir}", flush=True)
    rp = out_dir / "records.jsonl"
    with open(rp, "a") as fh:
        for name in plan:
            print(f"\n[stage] === {name} ===", flush=True)
            rec = measure_arm(name, specs[name], out_dir,
                              num_prompts=args.num_prompts, reps=args.reps,
                              port=args.port, native_sampler=args.native_sampler,
                              do_ppl=(args.ppl_arm == name))
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"[stage] {name}: wall_tps={rec.get('wall_tps')} served_ok={rec.get('served_ok')} "
                  f"err={rec.get('error')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
