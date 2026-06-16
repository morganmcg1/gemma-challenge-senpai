"""PR #519 — EXP-1 (#516): full 128x512 served TPS+PPL recert of byte-exact split-KV.

Single-variable A/B vs the shipped surgical-357 rung. LOCAL ONLY:
``analysis_only=true``, ``official_tps=0``, NO HF job, NO ``--launch``, NO
submission, NO served-file change. The challenge is PAUSED — this is local
measurement only (``--wandb_group splitkv399-full-recert``).

Both packaged submissions are served through the IDENTICAL served-cert
measurement path (``scripts.local_validation.harness`` ``LocalServer`` + the
official ``decode_outputs.py`` / ``ppl_endpoint.py``, same 128x512 workload,
seed 1, 3 decodes, full PPL) so the ONLY changed variable is the attention
realization baked into each manifest:

  control  ``fa2sw_strict_surgical357``           ``SURGICAL_ATTN_USE_3D_OFF=1``
           -> forced 2D order-preserving sequential-KV (byte-exact, gives up
              split-KV parallelism; the shipped 357.2 rung)
  variant  ``fa2sw_strict_byteexact_splitkv399``  ``BYTEEXACT_FIXED_TPS=4`` +
           ``BYTEEXACT_NUM_SEGMENTS=64`` -> fixed-order 3D split-KV (keeps the
           split-KV parallelism AND stays byte-exact by pinning the split SIZE
           not the COUNT; lawine #496's 399.97 candidate)

Both keep ``SPECULATIVE_CONFIG`` (spec-alive) and NEITHER sets
``VLLM_BATCH_INVARIANT`` (no ~48% matmul tax), so the warm-TPS delta isolates
exactly the attention realization.

Census 1 (served-vs-served matched-config self-determinism) reuses the
surgical357 served cert's ``token_identity`` verbatim: warm round1-vs-round2
token identity on the full 65,536-token workload. (Boundary: this is the speed +
PPL leg; the full operative-identity census vs surgical-357 is land #515's leg.
We report only the self-determinism number the cert already produces.)

Usage::

    .venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --arm variant
    .venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --arm control
    .venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --arm variant --smoke
    .venv/bin/python -m research.validity.splitkv399_full_recert.recert_ab --summarize \
        --wandb-name stark/splitkv399-full-recert
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths  # noqa: E402
# Reuse the surgical-357 served cert's self-determinism math verbatim so the
# A/B uses byte-for-byte the same Census-1 computation on both arms.
from research.validity.surgical357_operative_cert.cert_served_identity import (  # noqa: E402
    token_identity,
)

OUT_ROOT = ROOT / "research" / "validity" / "splitkv399_full_recert"

# Provenance anchors.
SURGICAL357_TARGET_TPS = 357.2   # shipped strict rung, this-pod recert l0attso0 = 357.22
SPLITKV399_TARGET_TPS = 399.97   # lawine #496 42qroec1, but only on a 32x256 workload
STRICT_FLOOR_222 = 222.0
PPL_TARGET = 2.37673             # shipped surgical-357 official PPL
PPL_GATE = 2.42
SIGMA_HW = 4.864
# Byte-exact split-KV coverage arithmetic (PR #525 NUM_SEGMENTS sweep). The
# fixed-order patch makes each parallel-softmax segment cover a FIXED span of
# FIXED_TPS*TILE_SIZE keys, so total coverage = FIXED_TPS*NUM_SEGMENTS*TILE_SIZE.
# Coverage must be >= the longest decode KV length or the kernel silently drops
# the tail of the context (PPL/identity break), so the sweep's low-S end is the
# correctness bound: at FIXED_TPS=4, coverage=64*S, and S=64 -> 4096=max_model_len.
BYTEEXACT_TILE_SIZE = 16
BYTEEXACT_MAX_MODEL_LEN = 4096

ARMS = {
    "variant": {"submission": "fa2sw_strict_byteexact_splitkv399", "mode": "byteexact"},
    "control": {"submission": "fa2sw_strict_surgical357", "mode": "surgical"},
}


def assert_manifest(mode: str, env_block: dict[str, Any]) -> None:
    """Hard preconditions on the PACKAGED manifest — these pin the single variable.

    Common to BOTH arms (so the only difference is the attention realization):
    spec-alive kept, and the global batch-invariant matmul tax flag is NOT set.
    """
    assert "VLLM_BATCH_INVARIANT" not in env_block, \
        "manifest must NOT set VLLM_BATCH_INVARIANT (that is the ~48% matmul tax)"
    assert "SPECULATIVE_CONFIG" in env_block, \
        "manifest must keep SPECULATIVE_CONFIG (spec-alive -> the 357/400 rung)"
    if mode == "byteexact":
        assert int(env_block.get("BYTEEXACT_FIXED_TPS", "0") or "0") > 0, \
            "variant manifest must set BYTEEXACT_FIXED_TPS>0"
        assert int(env_block.get("BYTEEXACT_NUM_SEGMENTS", "0") or "0") > 0, \
            "variant manifest must set BYTEEXACT_NUM_SEGMENTS>0"
        assert "SURGICAL_ATTN_USE_3D_OFF" not in env_block, \
            "variant must NOT force the 2D surgical path (that is the control)"
        assert env_block.get("SPLITKV_VERIFY") == "1", \
            "variant must keep SPLITKV_VERIFY=1 (verify routes onto the fixed 3D path)"
    elif mode == "surgical":
        assert env_block.get("SURGICAL_ATTN_USE_3D_OFF") == "1", \
            "control manifest must set SURGICAL_ATTN_USE_3D_OFF=1"
        assert "BYTEEXACT_FIXED_TPS" not in env_block, \
            "control must NOT arm the byteexact split-KV lever (that is the variant)"
    else:
        raise ValueError(f"unknown mode {mode!r}")


def grep_log(log_path: Path, mode: str) -> dict[str, Any]:
    """Pull lever-mechanism signals out of the packaged stack's server log.

    For ``surgical`` (control): armed + forced + ``splitkv_redirects == 0`` (the
    2D order-preserving path was forced, so the M=8 verify did NOT redirect to
    3D). For ``byteexact`` (variant): armed + re-jitted (both @triton.jit kernels
    rebuilt with the fixed ``tiles_per_segment``) + the segment-count global set,
    with NO fail-open ``baseline kept`` line, and ``splitkv_redirects > 0`` (the
    M=8 verify DID route onto the fixed-order 3D split-KV path — the whole point).

    Both modes additionally require the global matmul tax NEVER installed
    (``init_batch_invariance`` must not run) and no fatal traceback.
    """
    out: dict[str, Any] = {
        "mode": mode,
        "splitkv_redirects": 0,
        "onegraph_captured": False,
        "init_batch_invariance_ran": False,
        "fatal_traceback": False,
        "n_tracebacks": 0,
        "benign_usage_tracebacks": 0,
        "batch_invariant_mentions": 0,
    }
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        out["log_readable"] = False
        return out
    out["log_readable"] = True
    # Per-call verify redirect log line (capped at SPLITKV_VERIFY_LOG=5 prints,
    # but only printed when a redirect actually fired): "... -> 3D split-KV (n=...)".
    out["splitkv_redirects"] = text.count("-> 3D split-KV")
    out["onegraph_captured"] = "[onegraph] captured" in text
    # The matmul tax is installed only by init_batch_invariance(); it must NEVER
    # run here (neither manifest sets VLLM_BATCH_INVARIANT).
    out["init_batch_invariance_ran"] = (
        "init_batch_invariance" in text and "Activating batch invariant" in text
    )
    n_tb = text.count("Traceback (most recent call last)")
    n_usage = text.count("_report_usage_worker")
    out["n_tracebacks"] = n_tb
    out["benign_usage_tracebacks"] = n_usage
    out["fatal_traceback"] = ("CUDA error" in text) or (n_tb > n_usage)
    low = text.lower()
    out["batch_invariant_mentions"] = low.count("batch_invariant") + low.count(
        "batch-invariant"
    )

    if mode == "surgical":
        out["surgical_armed"] = "[surgical-attn] armed" in text
        out["surgical_forced_true"] = ("[surgical-attn] forced" in text) and (
            "is_batch_invariant=True" in text
        )
        out["lever_fired"] = bool(
            out["surgical_armed"] and out["surgical_forced_true"]
            and out["splitkv_redirects"] == 0 and not out["fatal_traceback"]
        )
        # surgical FORCES 2D -> verify must NOT redirect to 3D.
        out["verify_path_as_expected"] = out["splitkv_redirects"] == 0
    else:  # byteexact
        out["byteexact_armed"] = "[byteexact] armed" in text
        out["byteexact_rejitted"] = "[byteexact] re-jitted" in text
        out["byteexact_segments_set"] = (
            "[byteexact] set triton_attn.NUM_PAR_SOFTMAX_SEGMENTS=" in text
        )
        out["byteexact_fail_open"] = (
            "[byteexact] ops patch error" in text
            or "[byteexact] backend patch error" in text
        )
        out["lever_fired"] = bool(
            out["byteexact_armed"] and out["byteexact_rejitted"]
            and out["byteexact_segments_set"] and not out["byteexact_fail_open"]
            and not out["fatal_traceback"]
        )
        # fixed-order 3D split-KV: the M=8 verify MUST route onto the 3D path the
        # patch made byte-exact (so the byte-exact identity claim is actually
        # exercised). >0 redirect log lines proves it.
        out["verify_path_as_expected"] = out["splitkv_redirects"] > 0

    out["matmul_tax_off"] = not out["init_batch_invariance_ran"]
    # Overall mechanism validity: the measured TPS/PPL came from the intended
    # path, on the fast Marlin matmuls, with the expected verify routing.
    out["mechanism_valid"] = bool(
        out["lever_fired"] and out["matmul_tax_off"]
        and out["verify_path_as_expected"]
    )
    return out


class _GpuMemSampler:
    """Poll nvidia-smi in a daemon thread and keep the peak used-MiB seen.

    vLLM pre-reserves the KV cache to GPU_MEMORY_UTILIZATION, so peak is roughly
    constant across the run; we still sample so the reported number is measured,
    not assumed.
    """

    def __init__(self, period_s: float = 3.0) -> None:
        self.period_s = period_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thr: threading.Thread | None = None

    def _sample_once(self) -> int:
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return 0
        best = 0
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                best = max(best, int(line))
        return best

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._sample_once())
            self._stop.wait(self.period_s)

    def __enter__(self) -> "_GpuMemSampler":
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=5)


def run_one_arm(arm: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = ARMS[arm]
    submission_dir = (ROOT / "submissions" / cfg["submission"]).resolve()
    if not submission_dir.exists():
        raise SystemExit(f"submission not found: {submission_dir}")
    manifest = harness.load_manifest(submission_dir)
    env_block = manifest.get("env") or {}
    assert_manifest(cfg["mode"], env_block)

    for note in paths.prepare_local_gpu_env():
        print(f"[recert] {note}", flush=True)

    server_python = harness.ensure_server_venv(manifest["dependencies"])

    # NUM_SEGMENTS sweep (PR #525): override the byte-exact segment COUNT only
    # (the occupancy knob), holding BYTEEXACT_FIXED_TPS fixed (the byte-exact
    # invariant) so the single moved variable is the parallel-softmax segment
    # count. The patch reads BYTEEXACT_NUM_SEGMENTS from the process env at
    # import time, so an extra_env override changes only the segment count;
    # everything else stays byte-identical to the packaged #519 variant manifest.
    extra_env: dict[str, str] = {}
    seg_override = getattr(args, "num_segments", None)
    if seg_override is not None:
        if cfg["mode"] != "byteexact":
            raise SystemExit("--num-segments only applies to the byteexact variant arm")
        extra_env["BYTEEXACT_NUM_SEGMENTS"] = str(seg_override)
    fixed_tps = int(env_block.get("BYTEEXACT_FIXED_TPS", 0) or 0)
    eff_segments = (seg_override if seg_override is not None
                    else int(env_block.get("BYTEEXACT_NUM_SEGMENTS", 0) or 0))
    coverage_keys = fixed_tps * eff_segments * BYTEEXACT_TILE_SIZE  # = 64*S at T=4

    tag = getattr(args, "out_tag", None) or (("smoke_" + arm) if args.smoke else arm)
    out_root = Path(getattr(args, "out_root", None) or OUT_ROOT)
    out_dir = (out_root / tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "server.log"
    if seg_override is not None:
        print(f"[recert] NUM_SEGMENTS override -> {seg_override} "
              f"(fixed_tps={fixed_tps}, coverage={coverage_keys} keys vs "
              f"max_model_len {BYTEEXACT_MAX_MODEL_LEN}) tag={tag}", flush=True)
    print(f"[recert] arm={arm} submission={submission_dir.name} mode={cfg['mode']}",
          flush=True)
    print(f"[recert] workload={args.num_prompts}x{args.output_len} seed={args.seed} "
          f"n_decodes={args.n_decodes} ppl={args.do_ppl} -> {out_dir}", flush=True)

    decodes: list[dict[str, Any]] = []
    decode_files: list[Path] = []
    ppl_summary: dict[str, Any] | None = None
    server_ready_s = None

    t0 = time.time()
    with _GpuMemSampler() as mem, harness.LocalServer(
        submission_dir, server_python=server_python, log_path=server_log,
        extra_env=extra_env or None,
    ) as server:
        server_ready_s = time.time() - t0
        print(f"[recert] server ready in {server_ready_s:.0f}s", flush=True)
        for i in range(args.n_decodes):
            decode_out = out_dir / f"decode_round{i:02d}.jsonl"
            decode_summary = out_dir / f"decode_round{i:02d}.summary.json"
            td = time.time()
            summary = harness.capture_decode(
                server_python,
                base_url=server.base_url,
                model=server.served_model_name,
                out_file=decode_out,
                summary_file=decode_summary,
                num_prompts=args.num_prompts,
                output_len=args.output_len,
                seed=args.seed,
            )
            wall_around = time.time() - td
            n_tok = int(summary.get("num_completion_tokens", 0))
            dur = float(summary.get("duration_s", wall_around))
            wall_tps = n_tok / dur if dur > 0 else float("nan")
            n_completed = int(summary.get("num_records", 0))
            rec = {
                "round": i,
                "warm": i > 0,
                "wall_tps": wall_tps,
                "num_completion_tokens": n_tok,
                "decode_duration_s": dur,
                "wall_around_decode_s": wall_around,
                "num_completed_prompts": n_completed,
                "expected_tokens": args.num_prompts * args.output_len,
            }
            decodes.append(rec)
            decode_files.append(decode_out)
            print(f"[recert] round {i} ({'warm' if i > 0 else 'cold'}): "
                  f"wall_tps={wall_tps:.2f} tok={n_tok}/{args.num_prompts * args.output_len} "
                  f"dur={dur:.1f}s completed={n_completed}", flush=True)

        if args.do_ppl:
            try:
                ppl_summary = harness.run_ppl(
                    server_python,
                    base_url=server.base_url,
                    model=server.served_model_name,
                    out_file=out_dir / "ppl.jsonl",
                    summary_file=out_dir / "ppl.summary.json",
                )
                print(f"[recert] PPL={ppl_summary.get('ppl')} "
                      f"records={ppl_summary.get('num_records')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[recert] WARN PPL failed: {exc}", flush=True)
    peak_mib = mem.peak_mib

    mech = grep_log(server_log, cfg["mode"])

    warm_files = [decode_files[i] for i in range(len(decodes)) if decodes[i]["warm"]]
    if len(warm_files) >= 2:
        self_det = token_identity(warm_files[0], warm_files[1],
                                  "warm round1 vs round2 (self-determinism)")
    else:
        self_det = {"label": "self-determinism", "available": False,
                    "note": "need >=2 warm decodes"}

    warm_tps = [d["wall_tps"] for d in decodes
                if d["warm"] and d["wall_tps"] == d["wall_tps"]]
    median_warm_tps = statistics.median(warm_tps) if warm_tps else float("nan")
    ppl_val = (ppl_summary or {}).get("ppl")
    full_completion = bool(decodes and all(
        d["num_completion_tokens"] == args.num_prompts * args.output_len
        for d in decodes
    ))
    self_det_rate = (self_det.get("token_identity_rate")
                     if self_det.get("available") else None)
    self_det_perfect = bool(self_det.get("available") and self_det_rate == 1.0)

    result = {
        "pr": 519,
        "exp": "EXP-1 (#516)",
        "arm": arm,
        "submission": cfg["submission"],
        "mode": cfg["mode"],
        "num_segments_override": seg_override,
        "byteexact_num_segments": eff_segments,
        "byteexact_fixed_tps": fixed_tps,
        "coverage_keys": coverage_keys,
        "coverage_ge_max_model_len": bool(coverage_keys >= BYTEEXACT_MAX_MODEL_LEN),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed, "n_decodes": args.n_decodes,
                     "smoke": bool(args.smoke)},
        "server_ready_s": server_ready_s,
        "peak_gpu_mem_mib": peak_mib,
        "decodes": decodes,
        "median_warm_wall_tps": median_warm_tps,
        "warm_wall_tps_values": warm_tps,
        "ppl": ppl_val,
        "ppl_passes_gate": isinstance(ppl_val, (int, float)) and ppl_val <= PPL_GATE,
        "ppl_gate": PPL_GATE,
        "ppl_target": PPL_TARGET,
        "completion_128_128": full_completion,
        "num_completed_prompts": decodes[0]["num_completed_prompts"] if decodes else None,
        "self_determinism": self_det,
        "self_determinism_rate": self_det_rate,
        "self_determinism_perfect_r1_r2": self_det_perfect,
        "mechanism": mech,
        "lever_fired": mech.get("lever_fired"),
        "mechanism_valid": mech.get("mechanism_valid"),
        "lift_vs_222": (median_warm_tps - STRICT_FLOOR_222) if warm_tps else None,
        "tps_above_222_floor": bool(
            warm_tps and median_warm_tps > STRICT_FLOOR_222 + SIGMA_HW),
        "analysis_only": True,
        "official_tps": 0,
        "no_served_file_change": True,
    }
    result_path = out_dir / "arm_result.json"
    result_path.write_text(json.dumps(result, indent=2))

    print(f"\n[recert] ===== ARM {arm} ({cfg['mode']}) =====", flush=True)
    print(f"  mechanism_valid    = {mech.get('mechanism_valid')} "
          f"(lever_fired={mech.get('lever_fired')} matmul_tax_off={mech.get('matmul_tax_off')} "
          f"verify_3d={mech.get('verify_path_as_expected')} redirects={mech.get('splitkv_redirects')})",
          flush=True)
    print(f"  median_warm_wall_tps = {median_warm_tps:.2f}  values={ [round(x,2) for x in warm_tps] }",
          flush=True)
    print(f"  PPL                = {ppl_val} (gate<={PPL_GATE}: {result['ppl_passes_gate']})",
          flush=True)
    print(f"  self_determinism   = {self_det_rate} (perfect={self_det_perfect})", flush=True)
    print(f"  completion_128_128 = {full_completion}", flush=True)
    print(f"  peak_gpu_mem_mib   = {peak_mib}", flush=True)
    print(f"[recert] artifacts -> {result_path}", flush=True)
    return result


def _load_arm(arm: str) -> dict[str, Any] | None:
    p = OUT_ROOT / arm / "arm_result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    variant = _load_arm("variant")
    control = _load_arm("control")
    if variant is None:
        raise SystemExit("missing variant arm_result.json — run --arm variant first")

    v_tps = variant["median_warm_wall_tps"]
    v_ppl = variant["ppl"]
    v_sd = variant["self_determinism_rate"]
    # Same-session control if available, else fall back to the provenance anchor.
    if control is not None:
        c_tps = control["median_warm_wall_tps"]
        control_source = "same_session_rerun"
    else:
        c_tps = SURGICAL357_TARGET_TPS
        control_source = "provenance_anchor_357.2"
    delta = (v_tps - c_tps) if isinstance(v_tps, (int, float)) else None

    beats_surgical = bool(delta is not None and delta > SIGMA_HW)
    near_399 = bool(isinstance(v_tps, (int, float))
                    and abs(v_tps - SPLITKV399_TARGET_TPS) <= SIGMA_HW)
    exceeds_399 = bool(isinstance(v_tps, (int, float))
                       and v_tps > SPLITKV399_TARGET_TPS + SIGMA_HW)
    passes_ppl = bool(variant["ppl_passes_gate"])
    sd_ok = bool(v_sd == 1.0)
    if near_399:
        vs_399 = "~399.97 confirmed"
    elif exceeds_399:
        vs_399 = f"EXCEEDS 399.97 by {v_tps - SPLITKV399_TARGET_TPS:.2f} (split-KV scales up at full-len)"
    else:
        vs_399 = "short of 399.97"

    if beats_surgical and passes_ppl and sd_ok:
        verdict = (
            f"REAL: fixed-order split-KV beats surgical-357 at full 128x512 "
            f"({v_tps:.2f} vs {c_tps:.2f} TPS, +{delta:.2f} = {delta / SIGMA_HW:.1f} sigma_hw; "
            f"{vs_399}); PPL {v_ppl} <= {PPL_GATE}, self-determinism {v_sd} "
            f"-> strictly-faster byte-exact reopen rung."
        )
    elif passes_ppl and sd_ok and not beats_surgical:
        verdict = (
            f"ARTIFACT: at full 128x512 split-KV does NOT beat surgical-357 "
            f"({v_tps:.2f} vs {c_tps:.2f} TPS, delta {delta:+.2f} <= sigma_hw "
            f"{SIGMA_HW}); the 399.97 was a 32x256 small-workload effect. "
            f"PPL/identity hold but the speed mechanism does not carry to full."
        )
    else:
        verdict = (
            f"FAIL: variant breaks a gate (passes_ppl={passes_ppl} "
            f"self_determinism={v_sd} tps={v_tps}); see arm_result for cause."
        )

    summary = {
        "pr": 519,
        "exp": "EXP-1 (#516): full 128x512 served TPS+PPL recert of byte-exact split-KV",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_only": True,
        "official_tps": 0,
        "no_hf_job": True,
        "no_served_file_change": True,
        "control_source": control_source,
        # ---- KEY OUTPUTS (PR-required) ----
        "splitkv399_full_warm_median_tps": v_tps,
        "splitkv399_full_ppl": v_ppl,
        "splitkv399_full_self_determinism": v_sd,
        "vs_surgical357_tps_delta": delta,
        "passes_ppl_gate": passes_ppl,
        # ---- supporting ----
        "surgical357_full_warm_median_tps": c_tps,
        "splitkv399_target_tps_399_97": SPLITKV399_TARGET_TPS,
        "surgical357_anchor_tps_357_2": SURGICAL357_TARGET_TPS,
        "sigma_hw": SIGMA_HW,
        "beats_surgical357_by_sigma": beats_surgical,
        "near_399_97_within_sigma": near_399,
        "exceeds_399_97_by_sigma": exceeds_399,
        "splitkv399_self_determinism_perfect": sd_ok,
        "splitkv399_completion_128_128": bool(variant["completion_128_128"]),
        "splitkv399_mechanism_valid": bool(variant["mechanism_valid"]),
        "splitkv399_peak_gpu_mem_mib": variant.get("peak_gpu_mem_mib"),
        "surgical357_mechanism_valid": bool(control["mechanism_valid"]) if control else None,
        "surgical357_self_determinism": control["self_determinism_rate"] if control else None,
        "surgical357_ppl": control["ppl"] if control else None,
        "verdict": verdict,
        "variant_arm": variant,
        "control_arm": control,
    }
    summary_path = OUT_ROOT / "ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n[recert] ================= A/B SUMMARY =================", flush=True)
    print(f"  splitkv399_full_warm_median_tps = {v_tps}", flush=True)
    print(f"  surgical357 control ({control_source}) = {c_tps}", flush=True)
    print(f"  vs_surgical357_tps_delta        = {delta}", flush=True)
    print(f"  splitkv399_full_ppl             = {v_ppl} (passes_ppl_gate={passes_ppl})", flush=True)
    print(f"  splitkv399_full_self_determinism= {v_sd}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"[recert] artifacts -> {summary_path}", flush=True)

    if not args.no_wandb:
        _log_wandb(args, summary)
    return summary


def _log_wandb(args: argparse.Namespace, summary: dict[str, Any]) -> str | None:
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        print(f"[recert] wandb import failed ({exc}); skip", flush=True)
        return None
    try:
        run = wandb_logging.init_wandb_run(
            job_type="splitkv399-full-recert",
            agent="stark",
            name=args.wandb_name,
            group=args.wandb_group,
            tags=["splitkv399-full-recert", "pr519", "exp1-516", "analysis-only",
                  "byteexact-splitkv"],
            config={
                "variant_submission": ARMS["variant"]["submission"],
                "control_submission": ARMS["control"]["submission"],
                "workload_prompts": paths.NUM_PROMPTS,
                "workload_output_len": paths.OUTPUT_LEN,
                "seed": paths.SEED,
                "analysis_only": True,
                "official_tps": 0,
                "control_source": summary["control_source"],
            },
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[recert] wandb init failed ({exc}); skip", flush=True)
        return None
    if run is None:
        print("[recert] wandb disabled (no API key); skip", flush=True)
        return None
    run_id = getattr(run, "id", None)
    try:
        flat = {
            "recert/splitkv399_full_warm_median_tps": summary["splitkv399_full_warm_median_tps"],
            "recert/splitkv399_full_ppl": summary["splitkv399_full_ppl"],
            "recert/splitkv399_full_self_determinism": summary["splitkv399_full_self_determinism"],
            "recert/vs_surgical357_tps_delta": summary["vs_surgical357_tps_delta"],
            "recert/passes_ppl_gate": int(summary["passes_ppl_gate"]),
            "recert/surgical357_full_warm_median_tps": summary["surgical357_full_warm_median_tps"],
            "recert/beats_surgical357_by_sigma": int(summary["beats_surgical357_by_sigma"]),
            "recert/near_399_97_within_sigma": int(summary["near_399_97_within_sigma"]),
            "recert/splitkv399_mechanism_valid": int(summary["splitkv399_mechanism_valid"]),
            "recert/splitkv399_completion_128_128": int(summary["splitkv399_completion_128_128"]),
            "recert/splitkv399_peak_gpu_mem_mib": summary.get("splitkv399_peak_gpu_mem_mib") or 0,
        }
        flat = {k: v for k, v in flat.items() if isinstance(v, (int, float))}
        wandb_logging.log_summary(run, flat, step=0)
        run.summary["verdict"] = summary["verdict"]
        wandb_logging.log_json_artifact(
            run, name="splitkv399_full_recert_ab",
            artifact_type="operative-cert", data=summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[recert] WARN wandb logging error: {exc}", flush=True)
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass
    print(f"[recert] wandb run_id={run_id}", flush=True)
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), default=None,
                    help="run one full served cert arm")
    ap.add_argument("--summarize", action="store_true",
                    help="combine both arm results -> ab_summary.json + W&B")
    ap.add_argument("--n-decodes", type=int, default=3)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--no-ppl", dest="do_ppl", action="store_false", default=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny serve+decode sanity (8 prompts x 16 tok, 2 decodes, no ppl)")
    ap.add_argument("--num-segments", type=int, default=None,
                    help="PR #525: override BYTEEXACT_NUM_SEGMENTS for the variant arm "
                         "(byte-exact segment-count occupancy sweep; FIXED_TPS held)")
    ap.add_argument("--out-tag", default=None,
                    help="output subdir under out-root (default: arm name); set per-S in the sweep")
    ap.add_argument("--out-root", default=None,
                    help="root dir for arm artifacts (default: this harness dir); the "
                         "#525 sweep points this at research/validity/splitkv_numseg_sweep")
    ap.add_argument("--wandb-name", default="stark/splitkv399-full-recert")
    ap.add_argument("--wandb-group", default="splitkv399-full-recert")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        args.num_prompts = min(args.num_prompts, 8)
        args.output_len = min(args.output_len, 16)
        args.n_decodes = max(2, min(args.n_decodes, 2))
        args.do_ppl = False

    if args.summarize:
        summarize(args)
        return 0
    if args.arm is None:
        ap.error("specify --arm {variant,control} or --summarize")
    run_one_arm(args.arm, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
