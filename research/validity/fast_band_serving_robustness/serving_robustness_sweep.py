#!/usr/bin/env python
"""Serving-config robustness sweep for the local-drafter fast ~170 band (PR #671).

ANALYSIS-ONLY, LOCAL single-A10G. No weights changed, NO HF job, NO submission,
NO served-file change to the live ``int4_g128_lmhead`` submission.

wirbel #665 measured the local ``/tmp/qat-assistant`` MTP drafter in the FAST
~170 band (un-rescued K6 wall-TPS = 172.18, median-of-4, 2 servers, reproduces to
0.04 TPS). That confirmed the drafter-swap +10 lever, but TWO serving-config
caveats stayed un-excluded as regime determinants:

  1. Gemma4 forces ``TRITON_ATTN`` — is the fast band tied to that backend?
  2. The native-sampler shim was FORCED (stock FlashInfer crashes on
     ``curand.h: No such file`` in this container), and FlashInfer is a candidate
     155<->170 regime flipper, so it could not be excluded.

This sweep HOLDS THE DRAFTER FIXED at ``/tmp/qat-assistant`` (K=6) and varies one
serving-config knob per cell, asking: is the ~170 fast band a ROBUST property of
the local drafter, or a fragile artifact of one serving-knob combination?

Axes (each K6 cell = un-rescued wall-TPS, median across reps x >=2 fresh servers):

  c0_native_fullpiece     native sampler + FULL_AND_PIECEWISE cudagraph  (the
                          #665 default; CONTROL — must reproduce ~172)
  c1_native_piecewise     native sampler + PIECEWISE-only cudagraph
  c2_native_altcapture    native sampler + FULL_AND_PIECEWISE + alternate
                          cudagraph_capture_sizes
  c3_flashinfer_fullpiece FlashInfer sampler + FULL_AND_PIECEWISE  (TIME-BOXED;
                          requires resolving the container curand.h crash)

Calibration anchors (re-logged so band labels stay official-comparable):

  cal_g128_ar     int4_g128_lmhead plain AR  -> #665 anchor 126.94
  cal_spec_ar_m1  int4_mtp_batchinv K=0      -> #665 anchor 78.50

The attn backend is NOT a sweep axis: ``--attn-probe`` boots the spec stack with
``VLLM_ATTENTION_BACKEND=FLASH_ATTN`` and reports the backend the engine actually
resolved, to document whether the override is even accepted (expected: forced
TRITON_ATTN).

``--finalize`` merges every ``r*/records.jsonl`` round under ``--out-dir``, runs
the calibration gate, assigns each cell a band label (``fast_170`` /
``slow_155`` / ``other`` using the #665 stark=155.58 / land=170.16 anchors), and
emits the single-line verdict:

  BAND_ROBUST          — every enable-able cell stays fast_170.
  BAND_FRAGILE         — some cell drops out of the fast band (knob named).
  FLASHINFER_UNTESTABLE— native axes robust but the FlashInfer confound could not
                          be excluded in this container.
  HARNESS_UNCALIBRATED — a calibration anchor drifted too far to trust the labels.

analysis_only=true and official_tps=0 are logged as explicit W&B summary scalars
(machine-checkable no-fire guard).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402

# --- locked harness anchors (reused from wirbel #665) -----------------------
DEV307_VENV = Path("/tmp/senpai-venvs/a341b8bdf5ec1fe0/bin/python")  # vllm 0.22.1rc1.dev307
G128_CKPT = "/workspace/gemma_build/int4_g128_lmhead"
DRAFTER_LOCAL = "/tmp/qat-assistant"  # the gemma4_assistant MTP drafter (QAT-matched)
SUB_SPEC = ROOT / "submissions" / "int4_mtp_batchinv"
SUB_G128 = ROOT / "submissions" / "int4_g128_lmhead"

ANCHOR_G128_AR = 126.94   # wirbel #665 local AR wall (official int4_g128_lmhead 126.378)
ANCHOR_SPEC_AR = 78.50    # wirbel #665 spec-AR-ref (drafter off)
LAND_K6 = 170.16          # land #660 un-rescued K6 (fast regime ref, baked into #665 harness)
STARK_K6 = 155.58         # stark #642 un-rescued K6 (slow regime ref, baked into #665 harness)
REF_K6_665 = 172.18       # wirbel #665 median-of-4 fast-band K6 (the value under test)
REF_PPL = 2.0055
FIRE_BAR = 126.378
INCREMENTAL_BAR = 136.378  # +10 over the locked live anchor

CALIB_CLEAN_PCT = 0.5    # anchor within this -> clean
CALIB_ABORT_PCT = 2.0    # anchor off by more than this -> HARNESS_UNCALIBRATED

# band thresholds on the stark155 -> land170 interval (same split as #665)
BAND_LO = STARK_K6 + (LAND_K6 - STARK_K6) * 0.33   # <= -> slow_155  (160.39)
BAND_HI = STARK_K6 + (LAND_K6 - STARK_K6) * 0.67   # >= -> fast_170  (165.35)

OUT_ROOT = ROOT / "research" / "validity" / "fast_band_serving_robustness" / "run"


def _flashinfer_include() -> str | None:
    """The nvidia/cu13 include dir bundled in the dev307 venv (holds curand.h).

    Stock FlashInfer JIT-compiles a cuRAND kernel at engine start; this container
    ships the cuRAND headers ONLY inside the pip nvidia-cu13 wheel, not in
    ``/usr/local/cuda/include``, so the build dies with ``curand.h: No such
    file``. Pointing the JIT host/device include search at this dir is the
    documented fix the FlashInfer arm needs.
    """
    venv = DEV307_VENV.parent.parent
    hits = glob.glob(str(venv / "lib" / "python*" / "site-packages" / "nvidia" / "cu13" / "include"))
    for h in hits:
        if Path(h, "curand.h").exists():
            return h
    return None


FLASHINFER_INCLUDE = _flashinfer_include()


def cells() -> dict[str, dict[str, Any]]:
    """cell -> serving-config spec. Drafter + K held FIXED across every K6 cell."""
    spec_env = lambda k: {"NUM_SPECULATIVE_TOKENS": str(k), "DRAFTER_MODEL": DRAFTER_LOCAL}
    return {
        # ---- calibration anchors (re-logged) ----
        "cal_g128_ar": {
            "sub": SUB_G128, "base_env": {"MODEL_ID": G128_CKPT}, "kind": "cal_ar",
            "sampler": "native", "cudagraph": "default", "anchor": ANCHOR_G128_AR},
        "cal_spec_ar_m1": {
            "sub": SUB_SPEC, "base_env": spec_env(0), "kind": "cal_spec_ar",
            "sampler": "native", "cudagraph": "default", "anchor": ANCHOR_SPEC_AR},
        # ---- serving-config K6 cells (drafter FIXED at /tmp/qat-assistant, K=6) ----
        "c0_native_fullpiece": {
            "sub": SUB_SPEC, "base_env": spec_env(6), "kind": "k6", "K": 6,
            "sampler": "native", "cudagraph": "FULL_AND_PIECEWISE(stock-default)",
            "compilation_config": None},
        "c1_native_piecewise": {
            "sub": SUB_SPEC, "base_env": spec_env(6), "kind": "k6", "K": 6,
            "sampler": "native", "cudagraph": "PIECEWISE",
            "compilation_config": '{"cudagraph_mode":"PIECEWISE"}'},
        "c2_native_altcapture": {
            "sub": SUB_SPEC, "base_env": spec_env(6), "kind": "k6", "K": 6,
            "sampler": "native", "cudagraph": "FULL_AND_PIECEWISE+altsizes[1,2,4,8,16,32]",
            "compilation_config": '{"cudagraph_capture_sizes":[1,2,4,8,16,32]}'},
        "c3_flashinfer_fullpiece": {
            "sub": SUB_SPEC, "base_env": spec_env(6), "kind": "k6", "K": 6,
            "sampler": "flashinfer", "cudagraph": "FULL_AND_PIECEWISE(stock-default)",
            "compilation_config": None},
    }


def build_extra_env(cell: dict[str, Any]) -> dict[str, str]:
    """Resolve a cell spec into the extra_env LocalServer applies over the manifest."""
    env = dict(cell["base_env"])
    if cell["sampler"] == "native":
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # documented container shim
    elif cell["sampler"] == "flashinfer":
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
        if FLASHINFER_INCLUDE:
            # make the cuRAND headers findable by the FlashInfer JIT (nvcc + host).
            # Root cause (PR #671 time-boxed FlashInfer arm): the real toolkit at
            # /usr/local/cuda (nvcc 13.2) ships NO curand.h, so the stock JIT dies
            # with "curand.h: No such file". The only curand.h on the box is the pip
            # nvidia-cu13 wheel (CUDART 13.0, matching torch cu130). Injecting it
            # resolves curand.h, but FlashInfer's bundled cccl/libcudacxx then trips
            # cuda_toolkit.h's compiler-vs-CTK guard: nvcc 13.2 != headers 13.0
            # (a MINOR-version mismatch). That guard has an official escape hatch,
            # CCCL_DISABLE_CTK_COMPATIBILITY_CHECK, intended for exactly this
            # "newer compiler than the CTK headers ship" case. ABI is stable within
            # CUDA 13.x, and the benchmark is greedy (temperature=0), so a faithful
            # build must reproduce the native cells' token fingerprints bit-for-bit;
            # we cross-check that before trusting any FlashInfer TPS cell. If the
            # build still fails, the arm reports FLASHINFER_UNTESTABLE.
            env["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            env["CPATH"] = f"{FLASHINFER_INCLUDE}:{os.environ.get('CPATH', '')}".rstrip(":")
            env["NVCC_PREPEND_FLAGS"] = (
                f"-I{FLASHINFER_INCLUDE} -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK "
                f"{os.environ.get('NVCC_PREPEND_FLAGS', '')}".strip())
    if cell.get("compilation_config"):
        env["COMPILATION_CONFIG"] = cell["compilation_config"]
    if cell.get("attn_backend_override"):
        env["VLLM_ATTENTION_BACKEND"] = cell["attn_backend_override"]
    return env


# ---------------------------------------------------------------------------
# GPU + process plumbing (mirrors wirbel #665 clean_room_kceil.py)
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
    import urllib.request
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
        if "spec_decode" in name or "accepted" in name or "draft" in name:
            try:
                out[name] = out.get(name, 0.0) + float(val)
            except ValueError:
                pass
    return out


def parse_server_log(log_path: Path) -> dict[str, Any]:
    """Resolved-env audit from the vLLM boot log: the knobs each cell actually got."""
    info: dict[str, Any] = {
        "attn_backend": None, "cudagraph_mode": None, "compilation_mode": None,
        "cudagraph_capture_sizes": None, "spec_method": None, "drafter": None,
        "spec_num_tokens": None, "flashinfer_sampler_crash": False,
        "flashinfer_sampler_disabled": False, "attn_patch_failed": False,
    }
    if not log_path.exists():
        return info
    txt = log_path.read_text(errors="replace")
    for pat, key in [
        (r"AttentionBackendEnum\.([A-Z0-9_]+)", "attn_backend"),
        (r"Using ([A-Z0-9_]+) backend", "attn_backend"),
        (r"CUDAGraphMode\.([A-Z_]+)", "cudagraph_mode"),
        (r"CompilationMode\.([A-Z_]+)", "compilation_mode"),
        (r"speculative_config=SpeculativeConfig\(method='([a-z0-9_]+)'", "spec_method"),
        (r"num_spec_tokens=(\d+)", "spec_num_tokens"),
        (r"SpeculativeConfig\(method='[a-z0-9_]+', model='([^']+)'", "drafter"),
    ]:
        m = re.search(pat, txt)
        if m and info.get(key) is None:
            info[key] = m.group(1)
    m = re.search(r"cudagraph_capture_sizes['\"]?[:=]\s*(\[[0-9,\s]*\])", txt)
    if m:
        info["cudagraph_capture_sizes"] = m.group(1)
    if "curand.h" in txt or "Ninja build failed" in txt:
        info["flashinfer_sampler_crash"] = True
    if "FlashInfer top-p/top-k sampling disabled" in txt:
        info["flashinfer_sampler_disabled"] = True
    if "failed to apply attention-group num_heads patch" in txt:
        info["attn_patch_failed"] = True
    return info


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
# measure one cell: serve -> timed reps -> kill
# ---------------------------------------------------------------------------
def measure_cell(name: str, cell: dict[str, Any], out_dir: Path, *,
                 num_prompts: int, reps: int, port: int) -> dict[str, Any]:
    base = out_dir / name
    base.mkdir(parents=True, exist_ok=True)
    extra_env = build_extra_env(cell)

    rec: dict[str, Any] = {
        "name": name, "kind": cell["kind"], "K": cell.get("K"),
        "submission": cell["sub"].name, "sampler": cell["sampler"],
        "cudagraph_requested": cell.get("cudagraph"),
        "compilation_config": cell.get("compilation_config"),
        "extra_env": extra_env, "t_start_utc": datetime.now(timezone.utc).isoformat(),
        "served_ok": False, "error": None, "rep_wall_tps": [], "wall_tps": None,
        "ready_s": None, "gpu_mem_used_mib": None, "duration_s": [],
        "full_length": None, "metrics": {}, "resolved_env": {},
    }

    rec["gpu_mem_used_before_mib"] = preflight()
    base_url = f"http://127.0.0.1:{port}"
    log_path = base / "server.log"
    server = None
    try:
        t0 = time.time()
        server = harness.LocalServer(
            cell["sub"], server_python=DEV307_VENV, port=port,
            log_path=log_path, extra_env=extra_env)
        server.__enter__()
        rec["ready_s"] = round(time.time() - t0, 1)
        rec["served_ok"] = True
        rec["model_id"] = server.model_id
        rec["resolved_env"] = {k: server.env.get(k) for k in sorted(set(
            list(extra_env.keys()) + ["VLLM_USE_FLASHINFER_SAMPLER", "VLLM_BATCH_INVARIANT",
                                      "VLLM_ATTENTION_BACKEND", "COMPILATION_CONFIG",
                                      "NUM_SPECULATIVE_TOKENS", "DRAFTER_MODEL", "MAX_NUM_SEQS"]))}

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
                _, counts = fingerprint(of)
                rec["full_length"] = all(c == 512 for c in counts) if counts else False
            print(f"  [{name}] rep{r} wall_tps={tps:.3f} (n={n} dur={d:.1f}s)", flush=True)

        vals = [v for v in rec["rep_wall_tps"] if v == v]
        rec["wall_tps"] = statistics.median(vals) if vals else float("nan")
        rec["wall_tps_min"] = min(vals) if vals else None
        rec["wall_tps_max"] = max(vals) if vals else None
        rec["gpu_mem_used_mib"] = _gpu_mem_used_mib()
        rec["metrics"] = scrape_metrics(base_url)
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


def attn_probe(out_dir: Path, port: int, override: str = "FLASH_ATTN") -> dict[str, Any]:
    """Boot the spec stack with an attn-backend override; report what it resolved.

    Answers the advisor's 'is any attn override even accepted for Gemma4?'. No
    decode reps — the resolved backend is logged during model load, before
    readiness. A boot crash is itself informative (override rejected).
    """
    cell = dict(cells()["c0_native_fullpiece"])
    cell["attn_backend_override"] = override
    print(f"[attn-probe] booting with VLLM_ATTENTION_BACKEND={override} (reps=0)", flush=True)
    rec = measure_cell(f"attn_probe_{override.lower()}", cell, out_dir,
                       num_prompts=paths.NUM_PROMPTS, reps=0, port=port)
    rec["attn_override_requested"] = override
    rec["attn_override_honored"] = (rec.get("attn_backend") == override)
    rec["attn_triton_forced"] = (rec.get("attn_backend") == "TRITON_ATTN")
    print(f"[attn-probe] requested={override} resolved={rec.get('attn_backend')} "
          f"triton_forced={rec['attn_triton_forced']} served_ok={rec.get('served_ok')}", flush=True)
    rp = out_dir / "records.jsonl"
    with open(rp, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


# ---------------------------------------------------------------------------
# finalize: calibration + per-cell band labels + verdict + wandb
# ---------------------------------------------------------------------------
def band_label(k6: float | None) -> str | None:
    if k6 is None or k6 != k6:
        return None
    if k6 <= BAND_LO:
        return "slow_155"
    if k6 >= BAND_HI:
        return "fast_170"
    return "other"


def _delta_pct(v: float | None, ref: float) -> float | None:
    if v is None or v != v:
        return None
    return round(100.0 * (v - ref) / ref, 3)


def _merge_rounds(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Collect every r*/records.jsonl round; per cell merge reps across fresh servers."""
    merged: dict[str, dict[str, Any]] = {}
    round_files = sorted(glob.glob(str(out_dir / "r*" / "records.jsonl")))
    # also accept a bare records.jsonl directly under out_dir (single-round runs)
    if (out_dir / "records.jsonl").exists():
        round_files.append(str(out_dir / "records.jsonl"))
    for rf in round_files:
        round_tag = Path(rf).parent.name
        for line in Path(rf).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            name = r["name"]
            m = merged.setdefault(name, {
                "name": name, "kind": r.get("kind"), "K": r.get("K"),
                "sampler": r.get("sampler"), "cudagraph_requested": r.get("cudagraph_requested"),
                "compilation_config": r.get("compilation_config"),
                "rounds": [], "rep_wall_tps_all": [], "per_round": {},
                "served_ok_all": [], "attn_backend": r.get("attn_backend"),
                "cudagraph_mode": r.get("cudagraph_mode"),
                "compilation_mode": r.get("compilation_mode"),
                "cudagraph_capture_sizes": r.get("cudagraph_capture_sizes"),
                "flashinfer_sampler_crash": False, "flashinfer_sampler_disabled": False,
                "attn_patch_failed": False, "resolved_env": r.get("resolved_env"),
                "full_length": r.get("full_length"), "metrics": r.get("metrics") or {},
                "error": r.get("error"), "ready_s": r.get("ready_s"),
                "gpu_mem_used_mib": r.get("gpu_mem_used_mib"),
                "attn_override_requested": r.get("attn_override_requested"),
                "attn_override_honored": r.get("attn_override_honored"),
                "attn_triton_forced": r.get("attn_triton_forced"),
            })
            reps = [v for v in (r.get("rep_wall_tps") or []) if v == v]
            m["rep_wall_tps_all"].extend(reps)
            m["per_round"][round_tag] = round(statistics.median(reps), 3) if reps else None
            m["rounds"].append(round_tag)
            m["served_ok_all"].append(bool(r.get("served_ok")))
            # backfill metrics from the first round that actually exposed them: a cell
            # whose FIRST round failed (c3 r1 crash -> empty metrics) would otherwise
            # report acceptance=None even though later successful rounds scraped it.
            if not m.get("metrics") and r.get("metrics"):
                m["metrics"] = r.get("metrics")
            for k in ("attn_backend", "cudagraph_mode", "compilation_mode",
                      "cudagraph_capture_sizes"):
                if m.get(k) is None and r.get(k) is not None:
                    m[k] = r.get(k)
            for k in ("flashinfer_sampler_crash", "flashinfer_sampler_disabled",
                      "attn_patch_failed"):
                m[k] = bool(m.get(k)) or bool(r.get(k))
            if r.get("error") and not m.get("error"):
                m["error"] = r.get("error")
    for name, m in merged.items():
        vals = m["rep_wall_tps_all"]
        m["wall_tps"] = round(statistics.median(vals), 3) if vals else None
        m["n_reps"] = len(vals)
        m["n_servers"] = len([t for t in m["per_round"].values() if t is not None])
        m["wall_tps_min"] = round(min(vals), 3) if vals else None
        m["wall_tps_max"] = round(max(vals), 3) if vals else None
        m["wall_tps_spread_pct"] = (round(100.0 * (max(vals) - min(vals)) / min(vals), 3)
                                    if len(vals) > 1 and min(vals) else 0.0)
        m["band"] = band_label(m["wall_tps"]) if m.get("kind") == "k6" else None
    return merged


def finalize(out_dir: Path, args) -> dict[str, Any]:
    merged = _merge_rounds(out_dir)

    # ---- calibration ----
    g128 = merged.get("cal_g128_ar", {}).get("wall_tps")
    sar = merged.get("cal_spec_ar_m1", {}).get("wall_tps")
    calib = {
        "cal_g128_ar_wall_tps": g128, "cal_g128_ar_delta_pct": _delta_pct(g128, ANCHOR_G128_AR),
        "cal_spec_ar_m1_wall_tps": sar, "cal_spec_ar_m1_delta_pct": _delta_pct(sar, ANCHOR_SPEC_AR),
    }
    worst = max([abs(d) for d in (calib["cal_g128_ar_delta_pct"],
                                  calib["cal_spec_ar_m1_delta_pct"]) if d is not None],
                default=None)
    calib["worst_anchor_abs_pct"] = worst
    calib["calibrated_clean"] = (worst is not None and worst <= CALIB_CLEAN_PCT)
    calib["uncalibrated"] = (worst is not None and worst > CALIB_ABORT_PCT)

    # ---- per-cell robustness table ----
    cell_order = ["c0_native_fullpiece", "c1_native_piecewise",
                  "c2_native_altcapture", "c3_flashinfer_fullpiece"]
    table: list[dict[str, Any]] = []
    for nm in cell_order:
        m = merged.get(nm)
        if not m:
            continue
        acc = (m.get("metrics") or {}).get("vllm:spec_decode_num_accepted_tokens_total") or \
            (m.get("metrics") or {}).get("vllm:spec_decode_num_accepted_tokens")
        drf = (m.get("metrics") or {}).get("vllm:spec_decode_num_draft_tokens_total") or \
            (m.get("metrics") or {}).get("vllm:spec_decode_num_draft_tokens")
        table.append({
            "cell": nm, "sampler": m.get("sampler"),
            "cudagraph_requested": m.get("cudagraph_requested"),
            "cudagraph_resolved": m.get("cudagraph_mode"),
            "compilation_resolved": m.get("compilation_mode"),
            "attn_backend": m.get("attn_backend"),
            "wall_tps": m.get("wall_tps"), "band": m.get("band"),
            "n_reps": m.get("n_reps"), "n_servers": m.get("n_servers"),
            "per_round": m.get("per_round"), "spread_pct": m.get("wall_tps_spread_pct"),
            "served_ok": all(m.get("served_ok_all") or [False]),
            "full_length": m.get("full_length"),
            "acceptance_rate": (round(acc / drf, 6) if (acc is not None and drf) else None),
            "flashinfer_sampler_crash": m.get("flashinfer_sampler_crash"),
            "error": m.get("error"),
            "delta_vs_665_k6_pct": _delta_pct(m.get("wall_tps"), REF_K6_665),
        })

    by = {t["cell"]: t for t in table}
    native_cells = ["c0_native_fullpiece", "c1_native_piecewise", "c2_native_altcapture"]
    native_present = [c for c in native_cells if c in by and by[c]["wall_tps"] is not None]
    fragile_native = [c for c in native_present if by[c]["band"] != "fast_170"]

    fi = by.get("c3_flashinfer_fullpiece")
    # "enabled" = FlashInfer served and produced a valid in-band wall-TPS on >=1 fresh
    # server. A PRE-FIX crashed boot (r1: curand.h, zero reps) that was subsequently
    # recovered (r3/r4 served cleanly under the documented include-path fix) must NOT
    # poison the verdict: the served_ok all-reduce / crash any-reduce over rounds would
    # otherwise mislabel a recovered cell FLASHINFER_UNTESTABLE. The initial crash and
    # recovery are surfaced explicitly in the summary for the writeup.
    fi_enabled = bool(fi and fi.get("wall_tps") is not None and (fi.get("n_servers") or 0) >= 1)
    fi_initial_crash = bool(fi and fi.get("flashinfer_sampler_crash"))
    fi_band = fi.get("band") if fi else None

    # ---- attn-probe (override acceptance) ----
    attn = None
    for nm, m in merged.items():
        if nm.startswith("attn_probe"):
            attn = {"requested": m.get("attn_override_requested"),
                    "resolved": m.get("attn_backend"),
                    "override_honored": m.get("attn_override_honored"),
                    "triton_forced": m.get("attn_triton_forced"),
                    "served_ok": all(m.get("served_ok_all") or [False])}
            break

    # ---- verdict ----
    fragile_knob = None
    if calib["uncalibrated"]:
        verdict = "HARNESS_UNCALIBRATED"
    elif not native_present:
        verdict = "INCOMPLETE_NO_NATIVE_CELLS"
    elif fragile_native:
        verdict = "BAND_FRAGILE"
        fragile_knob = ",".join(f"{c}:{by[c]['band']}({by[c]['wall_tps']})" for c in fragile_native)
    elif fi_enabled and fi_band != "fast_170":
        verdict = "BAND_FRAGILE"
        fragile_knob = f"flashinfer_sampler:{fi_band}({fi.get('wall_tps')})"
    elif fi_enabled and fi_band == "fast_170":
        verdict = "BAND_ROBUST"
    else:
        verdict = "FLASHINFER_UNTESTABLE"

    summary = {
        "verdict": verdict, "fragile_knob": fragile_knob,
        "analysis_only": True, "official_tps": 0,
        "calibration": calib, "robustness_table": table,
        "native_cells_all_fast": (len(fragile_native) == 0 and len(native_present) >= 1),
        "flashinfer_enabled": fi_enabled, "flashinfer_band": fi_band,
        "flashinfer_initial_crash_prefix": fi_initial_crash,
        "flashinfer_n_servers_ok": (fi.get("n_servers") if fi else 0),
        "attn_probe": attn,
        "band_thresholds": {"slow_155_at_or_below": round(BAND_LO, 3),
                            "fast_170_at_or_above": round(BAND_HI, 3),
                            "stark_k6": STARK_K6, "land_k6": LAND_K6, "ref_665_k6": REF_K6_665},
        "anchors": {"cal_g128_ar": ANCHOR_G128_AR, "cal_spec_ar_m1": ANCHOR_SPEC_AR},
        "primary_metric": {"name": "c0_native_fullpiece_k6_walltps",
                           "value": by.get("c0_native_fullpiece", {}).get("wall_tps")},
        "test_metric": {"name": "verdict", "value": verdict},
    }
    (out_dir / "summary.json").write_text(json.dumps(
        {"summary": summary, "merged": merged}, indent=2, default=str))
    run_id = _log_wandb(args, table, summary, calib, merged) if not args.no_wandb else None
    summary["wandb_run_id"] = run_id
    (out_dir / "summary.json").write_text(json.dumps(
        {"summary": summary, "merged": merged}, indent=2, default=str))
    _print_summary(summary)
    return summary


def _print_summary(s: dict[str, Any]) -> None:
    print("\n===================== ROBUSTNESS VERDICT =====================", flush=True)
    print(f"  VERDICT: {s['verdict']}   fragile_knob={s.get('fragile_knob')}", flush=True)
    c = s["calibration"]
    print(f"  calib cal_g128_ar   = {c['cal_g128_ar_wall_tps']} (Δ {c['cal_g128_ar_delta_pct']}% vs {ANCHOR_G128_AR})", flush=True)
    print(f"  calib cal_spec_ar_m1= {c['cal_spec_ar_m1_wall_tps']} (Δ {c['cal_spec_ar_m1_delta_pct']}% vs {ANCHOR_SPEC_AR})", flush=True)
    print(f"  calibrated_clean={c['calibrated_clean']} uncalibrated={c['uncalibrated']} worst={c['worst_anchor_abs_pct']}%", flush=True)
    print(f"  band thresholds: slow_155<= {s['band_thresholds']['slow_155_at_or_below']}  "
          f"fast_170>= {s['band_thresholds']['fast_170_at_or_above']}  (#665 K6={REF_K6_665})", flush=True)
    print("  --- robustness table (K6 cells) ---", flush=True)
    for t in s["robustness_table"]:
        print(f"    {t['cell']:26s} tps={t['wall_tps']} band={t['band']} "
              f"sampler={t['sampler']} cudagraph={t['cudagraph_resolved']} attn={t['attn_backend']} "
              f"n_reps={t['n_reps']} n_servers={t['n_servers']} accept={t['acceptance_rate']} "
              f"served_ok={t['served_ok']} fi_crash={t['flashinfer_sampler_crash']}", flush=True)
    print(f"  attn_probe: {s.get('attn_probe')}", flush=True)
    print(f"  analysis_only={s['analysis_only']} official_tps={s['official_tps']}", flush=True)


def _log_wandb(args, table, summary, calib, merged) -> str | None:
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[wandb] import failed ({exc})", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="serving-config-robustness", agent="wirbel",
        name=args.wandb_name or "wirbel/fast-band-serving-robustness",
        group=args.wandb_group,
        tags=["fast-band-serving-robustness", "int4_mtp_batchinv", "analysis-only",
              "dev307-stock", summary["verdict"]],
        config={
            "submission": "int4_mtp_batchinv", "drafter": DRAFTER_LOCAL,
            "engine": "vllm 0.22.1rc1.dev307+g3e8afdf78", "num_prompts": args.num_prompts,
            "reps_per_server": args.reps, "max_num_seqs": 1, "batch_invariant": 1,
            "K_fixed": 6, "analysis_only": True, "official_tps": 0,
            "anchor_g128_ar": ANCHOR_G128_AR, "anchor_spec_ar_m1": ANCHOR_SPEC_AR,
            "land_k6": LAND_K6, "stark_k6": STARK_K6, "ref_665_k6": REF_K6_665,
            "band_lo": round(BAND_LO, 3), "band_hi": round(BAND_HI, 3),
            "flashinfer_include": FLASHINFER_INCLUDE,
        })
    if run is None:
        print("[wandb] disabled", flush=True)
        return None
    rid = None
    try:
        rid = run.id
        for i, t in enumerate(table):
            m = {f"cell/{t['cell']}/wall_tps": t.get("wall_tps"),
                 f"cell/{t['cell']}/acceptance_rate": t.get("acceptance_rate"),
                 f"cell/{t['cell']}/n_reps": t.get("n_reps"),
                 f"cell/{t['cell']}/n_servers": t.get("n_servers")}
            m = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            wandb_logging.log_event(run, f"cell_{t['cell']}", step=i, metrics=m,
                                    data={"cell": t["cell"], "band": t.get("band"),
                                          "sampler": t.get("sampler")})
        flat = {
            "verdict": summary["verdict"], "fragile_knob": summary.get("fragile_knob"),
            "analysis_only": 1, "official_tps": 0,
            "native_cells_all_fast": 1 if summary["native_cells_all_fast"] else 0,
            "flashinfer_enabled": 1 if summary["flashinfer_enabled"] else 0,
            "flashinfer_band": summary.get("flashinfer_band"),
            "flashinfer_initial_crash_prefix": 1 if summary.get("flashinfer_initial_crash_prefix") else 0,
            "flashinfer_n_servers_ok": summary.get("flashinfer_n_servers_ok"),
            "cal_g128_ar_wall_tps": calib.get("cal_g128_ar_wall_tps"),
            "cal_g128_ar_delta_pct": calib.get("cal_g128_ar_delta_pct"),
            "cal_spec_ar_m1_wall_tps": calib.get("cal_spec_ar_m1_wall_tps"),
            "cal_spec_ar_m1_delta_pct": calib.get("cal_spec_ar_m1_delta_pct"),
            "calibrated_clean": 1 if calib.get("calibrated_clean") else 0,
            "worst_anchor_abs_pct": calib.get("worst_anchor_abs_pct"),
        }
        for t in table:
            flat[f"{t['cell']}_wall_tps"] = t.get("wall_tps")
            flat[f"{t['cell']}_band"] = t.get("band")
            flat[f"{t['cell']}_acceptance"] = t.get("acceptance_rate")
        if summary.get("attn_probe"):
            ap = summary["attn_probe"]
            flat["attn_override_resolved"] = ap.get("resolved")
            flat["attn_triton_forced"] = 1 if ap.get("triton_forced") else 0
            flat["attn_override_honored"] = 1 if ap.get("override_honored") else 0
        flat = {f"result/{k}": v for k, v in flat.items() if v is not None}
        wandb_logging.log_summary(run, flat, step=len(table))
        wandb_logging.log_json_artifact(run, name="serving_robustness_sweep",
                                        artifact_type="serving-config-robustness",
                                        data={"summary": summary, "merged": merged})
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
    ap.add_argument("--cells", default=None, help="comma list of cells to run this stage")
    ap.add_argument("--finalize", action="store_true", help="merge rounds -> table + verdict + wandb")
    ap.add_argument("--attn-probe", default=None,
                    help="boot the spec stack with this VLLM_ATTENTION_BACKEND override and report")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    ap.add_argument("--round", default=None, help="round tag (writes records under out-dir/<round>/)")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="fast-band-serving-robustness-wirbel")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    note = paths.normalize_cuda_visible_devices()
    if note:
        print(f"[gpu] {note}", flush=True)
    # never inherit a sampler pin from the parent: each cell sets it explicitly
    os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)

    if args.finalize:
        finalize(out_dir, args)
        return 0

    if args.attn_probe:
        probe_dir = out_dir / (args.round or "probe")
        probe_dir.mkdir(parents=True, exist_ok=True)
        attn_probe(probe_dir, args.port, override=args.attn_probe)
        return 0

    spec = cells()
    plan = [a.strip() for a in (args.cells or "").split(",") if a.strip()]
    if not plan:
        print("no --cells given; nothing to do (use --finalize to aggregate)", flush=True)
        return 0
    for a in plan:
        if a not in spec:
            raise SystemExit(f"unknown cell {a!r}; choices={list(spec)}")

    round_dir = out_dir / args.round if args.round else out_dir
    round_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage] cells={plan} round={args.round} num_prompts={args.num_prompts} "
          f"reps={args.reps} -> {round_dir}", flush=True)
    print(f"[stage] FLASHINFER_INCLUDE={FLASHINFER_INCLUDE}", flush=True)
    rp = round_dir / "records.jsonl"
    with open(rp, "a") as fh:
        for name in plan:
            print(f"\n[stage] === {name} ({spec[name].get('cudagraph')}, "
                  f"sampler={spec[name]['sampler']}) ===", flush=True)
            rec = measure_cell(name, spec[name], round_dir,
                               num_prompts=args.num_prompts, reps=args.reps, port=args.port)
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            print(f"[stage] {name}: wall_tps={rec.get('wall_tps')} band="
                  f"{band_label(rec.get('wall_tps')) if rec.get('kind') == 'k6' else 'n/a'} "
                  f"served_ok={rec.get('served_ok')} err={rec.get('error')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
