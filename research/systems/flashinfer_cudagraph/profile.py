#!/usr/bin/env python
"""K-1 local profiling: FlashInfer backend + CUDAGraph on the linear MTP K=7 path.

PR #246 hypothesis: FlashInfer decode backend + a CUDAGraph step replay are an
untapped, ZERO greedy-identity-risk +7-18% lever on the deployed
``fa2sw_precache_kenyan`` linear spec path (481.53 official TPS), enough to clear
500 without the stalled tree build.

The live A10G run REFUTES the premise empirically, more decisively than static
analysis predicted:

  * FlashInfer is UNREACHABLE — not merely "non-viable". vLLM's Gemma4 config
    FORCE-PINS the attention backend to TRITON_ATTN because the model has
    heterogeneous head dims (``head_dim=256`` local / ``global_head_dim=512``):
    the log says ``"Gemma4 model has heterogeneous head dimensions ... Forcing
    TRITON_ATTN backend to prevent mixed-backend numerical divergence"`` and
    additionally flags ``VLLM_ATTENTION_BACKEND`` as an *unknown* env var. So the
    request is silently dropped before FlashInfer's own window assert is ever
    reached — the flashinfer arm boots, but on the forced Triton fallback, NOT
    FlashInfer. FlashInfer is therefore not a selectable lever on this model.
  * The deployed stack ALREADY captures the drafter propose loop in a CUDA graph
    (``ONEGRAPH=1`` under vLLM PIECEWISE). FlashInfer's own graph capture needs
    ``CUDAGraphMode.FULL``, which shipped upstream after this fork's branch point.

So instead of measuring a projected speedup, this script MEASURES the corrected
picture on the live A10G:

  control    — deployed env (ONEGRAPH=1, FA2-sliding mix). The 128/128 anchor.
  eager      — ONEGRAPH off + never-capture: ablates the drafter CUDA graph so the
               captured-vs-eager delta = what the deployed graph already buys.
  flashinfer — VLLM_ATTENTION_BACKEND=FLASHINFER + FA2-sliding off: detects the
               force-override (parses the served backend); decodes are skipped
               because the request is not honored (the override IS the result).

Metrics (per arm):
  * wall_tps = num_completion_tokens / decode_duration_s — the robust,
    official-spec-aligned throughput metric (PR #72); reported as the median over
    the WARM runs (run1..; run0 is a cold-start warmup pass — see below).
  * greedy-identity (WARM runs only; run0's first-execution inductor/kernel
    autotuning perturbs the numeric path then stabilizes, so run0 is excluded):
      - warm fp-noise floor: two warm control runs. The deployed stack is
        deterministic once warm, so this is 0 divergent tokens.
      - cold-start run0 vs warm: transparency only, shows the run0 warmup
        magnitude (NOT the floor).
      - CUDAGraph lever: warm control vs warm eager — does ONEGRAPH on/off change
        output? (the real "zero-risk" test, now judged against a 0-token floor.)
      - gate-reference: warm control vs the served_spec_off reference (the deployed
        stack's pre-existing #192 spec-vs-AR status — orthogonal context, not a
        lever effect).
  * PPL on control; must be <= 2.42.
  * peak VRAM (<= 24 GB) sampled during decode.

Self-test ``flashinfer_cudagraph_self_test_passes`` PASSES iff:
  (1) PPL(control) <= 2.42,
  (2) the CUDAGraph lever adds no greedy divergence beyond the WARM FP-noise floor
      (lever divergent tokens <= warm-floor divergent tokens); with a
      deterministic warm floor of 0 this is the byte-exact test, so a non-zero
      lever divergence FAILS — honestly refuting the "zero-risk" premise, and
  (3) control wall_tps is reproducible (CV over warm runs < 1%).
FlashInfer is reported as UNREACHABLE (measured) and excluded from the pass set.

Local A10G numbers are EXPLORATORY; only HF Jobs a10g-small runs are official.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import greedy_gate, harness, paths  # noqa: E402

SUBMISSION = ROOT / "submissions" / "fa2sw_precache_kenyan"
OUT_ROOT = ROOT / "research" / "systems" / "flashinfer_cudagraph"
OFFICIAL_ANCHOR_TPS = 481.53  # PR #52 linear MTP K=7, served 128/128
PPL_CAP = 2.42
VRAM_CAP_GB = 24.0

# Arm env overrides relative to the deployed manifest env.
ARMS: dict[str, dict[str, str]] = {
    # Deployed config — no overrides. ONEGRAPH=1 + FA2-sliding mix already on.
    "control": {},
    # Remove the drafter CUDA-graph capture: ONEGRAPH off and warmup threshold
    # pushed past the run so _capture_graph never fires (pure eager drafter).
    "eager": {
        "ONEGRAPH": "0",
        "LOOPGRAPH_REQUIRE_CAPTURE": "0",
        "LOOPGRAPH_WARMUP_CALLS": "100000000",
    },
    # Force the global FlashInfer attention backend and drop the per-layer FA2
    # injection so FlashInfer is genuinely the backend under test. Expected to
    # fail at engine init on Gemma-4's heterogeneous window_left.
    "flashinfer": {
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "FA_SLIDING": "0",
        "SPLITKV_VERIFY": "0",
    },
}


def _nvidia_mem_used_mib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
    return max(vals) if vals else None


class VRAMSampler:
    """Poll nvidia-smi memory.used in a thread; expose the peak (MiB)."""

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            mib = _nvidia_mem_used_mib()
            if mib is not None:
                self.peak_mib = max(self.peak_mib, mib)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VRAMSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _wall_tps(decode_summary: dict[str, Any]) -> float:
    return float(decode_summary["num_completion_tokens"]) / float(decode_summary["duration_s"])


def _compare(file_a: Path, file_b: Path) -> dict[str, Any]:
    """Official greedy_identity comparison between two decode_outputs.jsonl files."""
    gi = paths.import_greedy_identity()
    report = gi.compare_files(str(file_a), str(file_b))
    return {
        "verdict": report.verdict,
        "num_prompts_compared": report.num_prompts_compared,
        "num_identical": report.num_identical,
        "num_divergent": report.num_divergent,
        "total_tokens_compared": report.total_tokens_compared,
        "total_divergent_tokens": report.total_divergent_tokens,
    }


def _parse_attention_backend(log_path: Path) -> dict[str, Any]:
    """Read which attention backend vLLM ACTUALLY selected from the server log.

    A 'flashinfer' arm that boots successfully is NOT proof FlashInfer ran. On
    Gemma-4 the model config force-pins TRITON_ATTN because the model has
    heterogeneous head dims (``head_dim=256`` local / ``global_head_dim=512``):
    vLLM logs ``"Gemma4 model has heterogeneous head dimensions ... Forcing
    TRITON_ATTN backend to prevent mixed-backend numerical divergence"`` and
    additionally flags ``VLLM_ATTENTION_BACKEND`` as an *unknown* env var, so the
    request is silently dropped. We parse the log for the truth rather than
    assuming ``served == backend honored`` — otherwise the flashinfer arm would be
    mislabeled viable when it is really running the forced Triton fallback.
    """
    info: dict[str, Any] = {
        "backend_used": None,
        "forced_triton": False,
        "attention_backend_env_ignored": False,
    }
    try:
        text = log_path.read_text()
    except OSError:
        return info
    used = re.findall(r"Using AttentionBackendEnum\.(\w+) backend", text)
    if used:
        info["backend_used"] = used[-1]
    info["forced_triton"] = "Forcing TRITON_ATTN backend" in text
    info["attention_backend_env_ignored"] = (
        "Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND" in text
    )
    return info


def run_arm(
    arm: str,
    *,
    server_python: Path,
    out_dir: Path,
    runs: int,
    num_prompts: int,
    output_len: int,
    port: int,
    do_ppl: bool,
    startup_timeout_s: int,
) -> dict[str, Any]:
    """Boot the submission with this arm's env overrides, run N decodes + PPL."""
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    extra_env = dict(ARMS[arm])
    result: dict[str, Any] = {
        "arm": arm,
        "extra_env": extra_env,
        "served": False,
        "decode_files": [],
        "wall_tps_runs": [],
        "ppl": None,
        "peak_vram_mib": None,
        "error": None,
    }
    log_path = arm_dir / "server.log"
    print(f"\n========== ARM: {arm}  env={extra_env} ==========", flush=True)
    try:
        with VRAMSampler() as vram, harness.LocalServer(
            SUBMISSION,
            server_python=server_python,
            port=port,
            startup_timeout_s=startup_timeout_s,
            log_path=log_path,
            extra_env=extra_env,
        ) as srv:
            result["served"] = True
            result["model_id"] = srv.model_id
            result["served_model_name"] = srv.served_model_name
            backend = _parse_attention_backend(log_path)
            result.update(backend)
            requested_fi = ARMS[arm].get("VLLM_ATTENTION_BACKEND", "").upper() == "FLASHINFER"
            result["flashinfer_requested"] = requested_fi
            result["flashinfer_selected"] = backend.get("backend_used") == "FLASHINFER"
            if requested_fi and not result["flashinfer_selected"]:
                # The override IS the measurement: FlashInfer is unreachable on
                # Gemma-4 (forced TRITON_ATTN). Decoding the fallback would
                # profile a different config, so skip the decodes/PPL.
                result["flashinfer_overridden_to"] = backend.get("backend_used")
                print(f"[{arm}] FlashInfer UNREACHABLE: request overridden -> "
                      f"{backend.get('backend_used')} "
                      f"(forced_triton={backend.get('forced_triton')}, "
                      f"env_ignored={backend.get('attention_backend_env_ignored')}); "
                      f"skipping decodes — the override is the result.", flush=True)
            else:
                for i in range(runs):
                    out_file = arm_dir / f"decode_run{i}.jsonl"
                    summary_file = arm_dir / f"decode_summary{i}.json"
                    summary = harness.capture_decode(
                        server_python,
                        base_url=srv.base_url,
                        model=srv.served_model_name,
                        out_file=out_file,
                        summary_file=summary_file,
                        num_prompts=num_prompts,
                        output_len=output_len,
                    )
                    wtps = _wall_tps(summary)
                    result["wall_tps_runs"].append(wtps)
                    result["decode_files"].append(str(out_file))
                    result.setdefault("completed", summary["num_records"])
                    print(f"[{arm}] run{i}: wall_tps={wtps:.2f} "
                          f"({summary['num_completion_tokens']} tok / {summary['duration_s']:.2f}s, "
                          f"{summary['num_records']} records)", flush=True)
                if do_ppl:
                    ppl_summary = harness.run_ppl(
                        server_python,
                        base_url=srv.base_url,
                        model=srv.served_model_name,
                        out_file=arm_dir / "ppl_results.jsonl",
                        summary_file=arm_dir / "ppl_summary.json",
                    )
                    result["ppl"] = ppl_summary["ppl"]
                    print(f"[{arm}] PPL={ppl_summary['ppl']:.4f}", flush=True)
        result["peak_vram_mib"] = vram.peak_mib or None
    except Exception as exc:  # noqa: BLE001 — a failed boot IS the measurement for flashinfer
        result["error"] = f"{type(exc).__name__}: {exc}"
        # Capture the server-log tail so the failure mode is auditable.
        try:
            tail = log_path.read_text().splitlines()[-40:]
            result["server_log_tail"] = "\n".join(tail)
        except OSError:
            result["server_log_tail"] = None
        print(f"[{arm}] FAILED: {result['error']}", flush=True)

    runs_list = result["wall_tps_runs"]
    if runs_list:
        # run0 is a cold-start warmup pass (first-execution inductor/kernel
        # autotuning + compile specialization perturb the numeric path, then it
        # stabilizes). Report TPS over the WARM runs (run1..) so the headline is
        # steady-state throughput, and so CV is not inflated by the cold pass.
        warm = runs_list[1:] if len(runs_list) >= 2 else runs_list
        result["wall_tps_all_runs"] = list(runs_list)
        result["wall_tps_warm_runs"] = warm
        result["wall_tps_median"] = statistics.median(warm)
        result["wall_tps_mean"] = statistics.fmean(warm)
        result["wall_tps_cv_pct"] = (
            100.0 * statistics.pstdev(warm) / result["wall_tps_mean"] if len(warm) > 1 else 0.0
        )
    return result


def build_report(arms: dict[str, dict[str, Any]], *, output_len: int) -> dict[str, Any]:
    control = arms.get("control", {})
    eager = arms.get("eager", {})
    flashinfer = arms.get("flashinfer", {})
    report: dict[str, Any] = {
        "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
        "ppl_cap": PPL_CAP,
        "vram_cap_gb": VRAM_CAP_GB,
        "arms": arms,
        "comparisons": {},
        "self_test": {},
    }

    # --- Greedy-identity comparisons (warm runs; run0 is cold-start) ------- #
    control_files = control.get("decode_files", [])
    eager_files = eager.get("decode_files", [])

    def _warm(files: list[str]) -> list[str]:
        return files[1:] if len(files) >= 2 else files

    c_warm = _warm(control_files)
    e_warm = _warm(eager_files)
    comparisons = report["comparisons"]
    # Warm FP-noise floor: two steady-state control runs. The deployed stack is
    # deterministic once warm, so this is expected to be 0 divergent tokens — it
    # is the honest floor the lever must be judged against (the cold run0 floor
    # is an inflated artifact, see coldstart_control_run0_vs_warm).
    if len(c_warm) >= 2:
        comparisons["fp_noise_warm_control"] = _compare(Path(c_warm[0]), Path(c_warm[1]))
    # Cold-start vs warm (transparency only — NOT the floor): the run0 warmup
    # perturbation magnitude, so a reviewer can see why run0 is excluded.
    if control_files and c_warm and control_files[0] != c_warm[0]:
        comparisons["coldstart_control_run0_vs_warm"] = _compare(
            Path(control_files[0]), Path(c_warm[0])
        )
    # CUDAGraph lever, warm-vs-warm: does toggling ONEGRAPH change the output?
    if c_warm and e_warm:
        comparisons["lever_warm_control_vs_eager"] = _compare(Path(c_warm[-1]), Path(e_warm[-1]))
    # Gate-reference (pre-existing #192 spec-vs-AR context, not a lever effect).
    gate_candidate = c_warm[-1] if c_warm else (control_files[0] if control_files else None)
    if control.get("model_id") and gate_candidate:
        ref = greedy_gate.reference_for(
            harness.reference_identity(control["model_id"], SUBMISSION)
        )
        if Path(ref).exists():
            comparisons["gate_control_vs_served_spec_off_reference"] = {
                "reference": str(ref),
                "reference_kind": greedy_gate.reference_kind(Path(ref)),
                **_compare(Path(ref), Path(gate_candidate)),
            }

    # --- captured-vs-eager TPS breakdown ---------------------------------- #
    c_tps = control.get("wall_tps_median")
    e_tps = eager.get("wall_tps_median")
    if c_tps and e_tps:
        report["captured_vs_eager_pct"] = 100.0 * (c_tps - e_tps) / e_tps
        report["wall_tps_control_median"] = c_tps
        report["wall_tps_eager_median"] = e_tps

    # --- K-1 net lever + official projection ------------------------------ #
    # FlashInfer is the only NEW attention lever, and it is UNREACHABLE here:
    # vLLM force-pins TRITON_ATTN for Gemma-4's heterogeneous head dims, so the
    # arm may still 'serve' but on the Triton fallback, NOT FlashInfer. The
    # CUDAGraph lever is already ON in the control. So the best runnable config
    # from the K-1 menu IS the control: net local gain over the deployed path = 0.
    report["flashinfer_served"] = bool(flashinfer.get("served"))
    report["flashinfer_backend_used"] = flashinfer.get("backend_used")
    report["flashinfer_forced_triton"] = bool(flashinfer.get("forced_triton"))
    report["flashinfer_env_ignored"] = bool(flashinfer.get("attention_backend_env_ignored"))
    flashinfer_reachable = bool(flashinfer.get("flashinfer_selected"))
    report["flashinfer_reachable"] = flashinfer_reachable
    report["flashinfer_viable"] = flashinfer_reachable  # back-compat alias
    report["tps_local_flashinfer_cudagraph"] = c_tps  # best runnable K-1 config = control
    report["local_tps_gain_pct"] = 0.0 if not flashinfer_reachable else None
    report["official_projection_tps"] = (
        OFFICIAL_ANCHOR_TPS * (1.0 + (report["local_tps_gain_pct"] or 0.0) / 100.0)
        if report["local_tps_gain_pct"] is not None else None
    )
    report["clears_500"] = (
        bool(report["official_projection_tps"] and report["official_projection_tps"] > 500.0)
    )

    # --- Self-test -------------------------------------------------------- #
    st = report["self_test"]
    ppl = control.get("ppl")
    st["ppl"] = ppl
    st["ppl_ok"] = bool(isinstance(ppl, (int, float)) and ppl <= PPL_CAP)

    lever = comparisons.get("lever_warm_control_vs_eager")
    fp = comparisons.get("fp_noise_warm_control")
    if fp is not None:
        st["fp_noise_warm_divergent_tokens"] = fp["total_divergent_tokens"]
        # The deployed stack is deterministic once warm iff two warm control runs
        # are byte-exact (expected: 0 divergent tokens).
        st["warm_deterministic_ok"] = fp["total_divergent_tokens"] == 0
    if lever is not None:
        st["lever_divergent_tokens"] = lever["total_divergent_tokens"]
        st["lever_divergent_prompts"] = lever["num_divergent"]
        # Is the CUDAGraph lever truly output-neutral ("zero greedy-identity
        # risk", per the K-1 premise)? Only if ONEGRAPH on vs off is byte-exact
        # warm-vs-warm.
        st["cudagraph_lever_bit_identical"] = lever["total_divergent_tokens"] == 0
    if lever is not None and fp is not None:
        # greedy-identity "holds" for the lever iff it adds no divergence beyond
        # the WARM FP-noise floor. With a deterministic warm floor (0) this is
        # exactly the bit-identical test, so a non-zero lever divergence FAILS —
        # honestly refuting the "zero-risk" premise rather than hiding it behind
        # an inflated cold-start floor.
        st["greedy_identity_ok"] = lever["total_divergent_tokens"] <= fp["total_divergent_tokens"]
    elif lever is not None:
        st["greedy_identity_ok"] = lever["verdict"] == "GREEDY_IDENTICAL"
    else:
        st["greedy_identity_ok"] = None

    cv = control.get("wall_tps_cv_pct")
    st["wall_tps_cv_pct"] = cv
    st["tps_reproducible_ok"] = bool(cv is not None and cv < 1.0)

    peak_gb = (control.get("peak_vram_mib") or 0) / 1024.0
    st["peak_vram_gb"] = peak_gb or None
    st["vram_ok"] = bool(peak_gb and peak_gb <= VRAM_CAP_GB)

    st["flashinfer_reachable"] = flashinfer_reachable
    st["passes"] = bool(
        st["ppl_ok"]
        and st.get("greedy_identity_ok")
        and st["tps_reproducible_ok"]
    )
    report["flashinfer_cudagraph_self_test_passes"] = st["passes"]
    return report


def log_wandb(report: dict[str, Any], args) -> str | None:
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_summary
    except Exception as exc:  # pragma: no cover
        print(f"[profile] wandb unavailable: {exc}", flush=True)
        return None
    run = init_wandb_run(
        job_type="systems-profile",
        agent="lawine",
        name=args.wandb_name or "lawine/flashinfer-cudagraph",
        group=args.wandb_group or "flashinfer-cudagraph-linear",
        tags=["k1-flashinfer-cudagraph", "linear-mtp-k7", "local-a10g"],
        notes="K-1 FlashInfer+CUDAGraph local profiling on the linear MTP K=7 path",
        config={
            "submission": str(SUBMISSION),
            "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
            "num_prompts": args.num_prompts,
            "output_len": args.output_len,
            "runs": args.runs,
            "arms": list(report["arms"].keys()),
        },
    )
    if run is None:
        print("[profile] wandb init returned None (no creds?) — skipping", flush=True)
        return None
    st = report["self_test"]
    summary = {
        "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
        "wall_tps_control_median": report.get("wall_tps_control_median"),
        "wall_tps_eager_median": report.get("wall_tps_eager_median"),
        "captured_vs_eager_pct": report.get("captured_vs_eager_pct"),
        "tps_local_flashinfer_cudagraph": report.get("tps_local_flashinfer_cudagraph"),
        "local_tps_gain_pct": report.get("local_tps_gain_pct"),
        "official_projection_tps": report.get("official_projection_tps"),
        "clears_500": int(bool(report.get("clears_500"))),
        "flashinfer_reachable": int(bool(report.get("flashinfer_reachable"))),
        "ppl": st.get("ppl"),
        "ppl_ok": int(bool(st.get("ppl_ok"))),
        "warm_deterministic_ok": int(bool(st.get("warm_deterministic_ok"))),
        "cudagraph_lever_bit_identical": int(bool(st.get("cudagraph_lever_bit_identical"))),
        "greedy_identity_ok": int(bool(st.get("greedy_identity_ok"))),
        "lever_divergent_tokens": st.get("lever_divergent_tokens"),
        "lever_divergent_prompts": st.get("lever_divergent_prompts"),
        "fp_noise_warm_divergent_tokens": st.get("fp_noise_warm_divergent_tokens"),
        "wall_tps_cv_pct": st.get("wall_tps_cv_pct"),
        "tps_reproducible_ok": int(bool(st.get("tps_reproducible_ok"))),
        "peak_vram_gb": st.get("peak_vram_gb"),
        "self_test_passes": int(bool(st.get("passes"))),
    }
    summary = {k: v for k, v in summary.items() if v is not None}
    log_summary(run, summary, step=0)
    run_id = getattr(run, "id", None)
    finish_wandb(run)
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="exit non-zero unless flashinfer_cudagraph_self_test_passes")
    ap.add_argument("--arms", default="control,eager,flashinfer",
                    help="comma-separated subset of: control,eager,flashinfer")
    ap.add_argument("--runs", type=int, default=3, help="decode passes per arm (median wall_tps)")
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    args = ap.parse_args(argv)

    for note in paths.prepare_local_gpu_env():
        print(f"[profile] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = args.out_dir or (OUT_ROOT / f"run-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[profile] server_python={server_python}", flush=True)
    print(f"[profile] out_dir={out_dir}", flush=True)

    requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms: dict[str, dict[str, Any]] = {}
    for arm in requested:
        if arm not in ARMS:
            print(f"[profile] unknown arm {arm!r}; skipping", flush=True)
            continue
        # FlashInfer is expected to crash at init; cap its startup wait so a hang
        # cannot eat the whole run budget. The others get the full weight-load wait.
        startup = 900 if arm == "flashinfer" else 1200
        arms[arm] = run_arm(
            arm,
            server_python=server_python,
            out_dir=out_dir,
            runs=args.runs,
            num_prompts=args.num_prompts,
            output_len=args.output_len,
            port=args.port,
            do_ppl=(arm == "control" and not args.skip_ppl),
            startup_timeout_s=startup,
        )

    report = build_report(arms, output_len=args.output_len)
    report["out_dir"] = str(out_dir)
    report["created_at"] = stamp
    report["wandb_run_id"] = log_wandb(report, args)

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    # Also write a stable top-level copy for the PR.
    (OUT_ROOT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_summary(report)
    print(f"[profile] report: {report_path}", flush=True)

    if args.self_test:
        return 0 if report["flashinfer_cudagraph_self_test_passes"] else 1
    return 0


def _f(v: Any, spec: str = ".2f") -> str:
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


def _print_summary(report: dict[str, Any]) -> None:
    line = "=" * 18 + " K-1 FLASHINFER + CUDAGRAPH (LOCAL A10G) " + "=" * 18
    print("\n" + line, flush=True)
    for arm, r in report["arms"].items():
        if r.get("served"):
            print(f"  {arm:10s} wall_tps median={_f(r.get('wall_tps_median'))} "
                  f"mean={_f(r.get('wall_tps_mean'))} cv={_f(r.get('wall_tps_cv_pct'),'.3f')}% "
                  f"runs={[round(x,2) for x in r.get('wall_tps_runs', [])]} "
                  f"ppl={_f(r.get('ppl'),'.4f')} "
                  f"peakVRAM={_f((r.get('peak_vram_mib') or 0)/1024.0,'.2f')}GB", flush=True)
        else:
            print(f"  {arm:10s} NON-VIABLE: {r.get('error')}", flush=True)
    print("", flush=True)
    cmp = report["comparisons"]
    if "fp_noise_warm_control" in cmp:
        f = cmp["fp_noise_warm_control"]
        print(f"  warm FP-noise floor (control r1 vs r2): {f['verdict']} "
              f"divergent_tokens={f['total_divergent_tokens']}/{f['total_tokens_compared']}", flush=True)
    if "coldstart_control_run0_vs_warm" in cmp:
        c = cmp["coldstart_control_run0_vs_warm"]
        print(f"  cold-start run0 vs warm (EXCLUDED):     {c['verdict']} "
              f"divergent_tokens={c['total_divergent_tokens']}/{c['total_tokens_compared']} "
              f"[run0 warmup artifact, not the floor]", flush=True)
    if "lever_warm_control_vs_eager" in cmp:
        l = cmp["lever_warm_control_vs_eager"]
        print(f"  CUDAGraph lever (control vs eager,warm):{l['verdict']} "
              f"divergent={l['num_divergent']}/{l['num_prompts_compared']} prompts "
              f"({l['total_divergent_tokens']}/{l['total_tokens_compared']} tok)", flush=True)
    if "gate_control_vs_served_spec_off_reference" in cmp:
        g = cmp["gate_control_vs_served_spec_off_reference"]
        print(f"  gate ref (control vs served_spec_off):  {g['verdict']} "
              f"({g['num_divergent']}/{g['num_prompts_compared']} prompts divergent) "
              f"[pre-existing #192 spec-vs-AR; NOT a lever effect]", flush=True)
    print("", flush=True)
    if report.get("captured_vs_eager_pct") is not None:
        print(f"  captured-vs-eager (existing CUDAGraph already buys): "
              f"{_f(report['captured_vs_eager_pct'],'+.2f')}% "
              f"(control {_f(report.get('wall_tps_control_median'))} vs eager "
              f"{_f(report.get('wall_tps_eager_median'))})", flush=True)
    print(f"  FlashInfer reachable on Gemma-4/A10G: {report.get('flashinfer_reachable')} "
          f"(arm backend_used={report.get('flashinfer_backend_used')}, "
          f"forced_triton={report.get('flashinfer_forced_triton')}, "
          f"env_ignored={report.get('flashinfer_env_ignored')})", flush=True)
    print(f"  K-1 net NEW local gain over deployed: {_f(report.get('local_tps_gain_pct'),'+.2f')}%  "
          f"-> official projection {_f(report.get('official_projection_tps'))} TPS  "
          f"clears_500={report.get('clears_500')}", flush=True)
    st = report["self_test"]
    print(f"\n  SELF-TEST flashinfer_cudagraph_self_test_passes = {st.get('passes')}", flush=True)
    print(f"    ppl_ok={st.get('ppl_ok')} (ppl={_f(st.get('ppl'),'.4f')} <= {PPL_CAP}) "
          f"tps_reproducible_ok={st.get('tps_reproducible_ok')} (cv={_f(st.get('wall_tps_cv_pct'),'.3f')}%)", flush=True)
    print(f"    warm_deterministic_ok={st.get('warm_deterministic_ok')} "
          f"(warm floor={st.get('fp_noise_warm_divergent_tokens')} tok)  "
          f"cudagraph_lever_bit_identical={st.get('cudagraph_lever_bit_identical')} "
          f"(lever={st.get('lever_divergent_tokens')} tok / "
          f"{st.get('lever_divergent_prompts')} prompts)  "
          f"-> greedy_identity_ok={st.get('greedy_identity_ok')}", flush=True)
    print("=" * len(line) + "\n", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
