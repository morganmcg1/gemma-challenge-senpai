"""Frontier decode-step profiler for the *real* served stack (PR #30).

Where ``profile_graph.py`` / ``profile_eager.py`` (PR #8) build a bare in-process
``LLM`` and so cannot see the submission's spec-decode + lmhead12k + fa2sw +
loopgraph patches, this module profiles the **actual** ``serve.py`` process — the
same OpenAI server the official harness benchmarks — and decomposes the mean
spec-decode cycle into:

  (a) drafter forward            -> STEPTIME ``kind=draft`` GPU time
  (b) verify forward            -> STEPTIME ``kind=exec`` GPU time, sub-split by
        torch-profiler kernel category (body int4-Marlin GEMM / lmhead12k GEMM /
        attention(fa2sw) / norm+elementwise / sampling)
  (c) sampling + detok          -> sampling kernels (GPU) + host detok (in host)
  (d) CUDA-graph replay / host  -> STEPTIME inter-step ``gap`` minus GPU-busy

Three measurement seams, all on the live serving path (CUDA graphs ON):

  * STEPTIME=1 (the submission's shipped, normally-inert per-step probe) gives the
    drafter/verify/host split + per-cycle wall, with zero profiler perturbation.
  * vLLM's built-in torch profiler (``--profiler-config``, triggered over HTTP
    /start_profile../stop_profile) writes a chrome trace; we categorize its GPU
    kernels for the verify sub-split. CUPTI sees kernels even under graph replay.
  * Prometheus ``/metrics`` gives the canonical E_accept (mean acceptance length
    = 1 + accepted/drafts) over the whole run.

Isolation passes (one variable each) close what a single trace cannot attribute:
  * spec on/off   -> SPECULATIVE_CONFIG=""  (pure-verify AR control; drafter cost)
  * lmhead on/off -> LM_HEAD_PRUNE=0         (full 262k vs 12k vocab head -> the
                     lm_head GEMM cost, by GEMM-time difference)

Local A10G numbers are exploratory probes, NOT the official a10g-small TPS.
"""
from __future__ import annotations

import gzip
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import harness, paths

# GPU-kernel name -> component. Mirrors profile_graph.py (PR #8) so the frontier
# breakdown is directly comparable to the int4-base breakdown.
CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("matmul_gemm", ("marlin", "gptq", "gemm", "gemv", "cutlass", "wmma", "splitk",
                     "split_k", "ampere", "s16816", "s161616", "tensorop", "dot")),
    ("attention", ("attn", "flash", "_fwd", "paged", "unified_attention",
                   "reshape_and_cache", "rotary", "rope", "fmha", "mha")),
    ("sampling", ("log_softmax", "logsoftmax", "argmax", "topk", "top_k", "softmax",
                  "sample", "logits", "cumsum", "sort", "gather", "scatter")),
    ("norm", ("rms", "layernorm", "layer_norm", "norm_kernel")),
    ("activation", ("silu", "gelu", "swiglu", "act_and_mul")),
    ("elementwise_copy", ("elementwise", "copy", "cast", "convert", "memcpy", "memset",
                          "fill", "vectorized", "index_", "_add", "_mul", "triton_poi")),
]
# GEMM wins ties: a fused "gemm+silu" is dominated by the GEMM cost at conc=1.
_GEMM_HINTS = ("marlin", "gemm", "gemv", "cutlass", "s16816", "tensorop", "wmma")


def categorize(name: str) -> str:
    n = name.lower()
    if any(h in n for h in _GEMM_HINTS):
        return "matmul_gemm"
    for cat, subs in CATEGORIES:
        if any(s in n for s in subs):
            return cat
    return "other"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only; no new deps)
# ---------------------------------------------------------------------------
def _post(url: str, timeout_s: float = 30.0) -> int:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.status


def _get_text(url: str, timeout_s: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        return r.read().decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# STEPTIME log parsing -> per-cycle drafter / verify / host split
# ---------------------------------------------------------------------------
_RAW_RE = re.compile(
    r"\[steptime\] raw i=(?P<i>\d+) kind=(?P<kind>\w+) gap=(?P<gap>[\d.]+) "
    r"cpu=(?P<cpu>[\d.]+) gpu=(?P<gpu>[\d.]+) dcpu=(?P<dcpu>[\d.]+) dgpu=(?P<dgpu>[\d.]+)"
)


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p50": s[len(s) // 2],
        "p90": s[min(len(s) - 1, int(0.9 * (len(s) - 1)))],
    }


def parse_steptime(log_text: str) -> dict[str, Any]:
    """Aggregate the per-step raw records into exec/draft component stats (ms)."""
    exec_recs: dict[str, list[float]] = {k: [] for k in ("gap", "cpu", "gpu", "dgpu")}
    draft_recs: dict[str, list[float]] = {k: [] for k in ("cpu", "gpu")}
    n_exec = n_draft = 0
    for m in _RAW_RE.finditer(log_text):
        kind = m.group("kind")
        if kind == "exec":
            n_exec += 1
            for k in ("gap", "cpu", "gpu", "dgpu"):
                exec_recs[k].append(float(m.group(k)))
        elif kind == "draft":
            n_draft += 1
            draft_recs["cpu"].append(float(m.group("cpu")))
            draft_recs["gpu"].append(float(m.group("gpu")))
    out: dict[str, Any] = {
        "raw_exec_steps": n_exec,
        "raw_draft_steps": n_draft,
        "exec": {k: _stats(v) for k, v in exec_recs.items()},
        "draft": {k: _stats(v) for k, v in draft_recs.items()},
    }
    # Steady-state decode scalars use the MEDIAN (p50), not the mean. At conc=1
    # over N sequential prompts the per-step series carries two outlier classes the
    # mean cannot reject: (a) prefill execute_model calls (huge exec gpu/cpu) and
    # (b) the ~N inter-request boundaries (huge host gap = client/prefill latency,
    # NOT decode host overhead). The smoke proved this: exec.gap mean=12.25ms but
    # p50=1.55ms / p90=1.62ms — a handful of request-switch gaps inflated the mean
    # 8x and faked a 10ms/cycle "host tax" that does not exist in steady state.
    # The median represents a typical decode cycle; means are kept as *_mean.
    def _p50(series: dict[str, dict], key: str) -> float:
        return series[key]["p50"]

    out["stat"] = "p50_steady_state"
    out["verify_gpu_ms"] = _p50(out["exec"], "gpu")
    # Drafter GPU = standalone draft.gpu if present, else the in-step dgpu (older
    # wheels call propose inside execute_model).
    out["drafter_gpu_ms"] = (_p50(out["draft"], "gpu") if n_draft
                             else _p50(out["exec"], "dgpu"))
    out["host_gap_ms"] = _p50(out["exec"], "gap")
    out["exec_cpu_ms"] = _p50(out["exec"], "cpu")
    out["verify_gpu_ms_mean"] = out["exec"]["gpu"]["mean"]
    out["drafter_gpu_ms_mean"] = (out["draft"]["gpu"]["mean"] if n_draft
                                  else out["exec"]["dgpu"]["mean"])
    out["host_gap_ms_mean"] = out["exec"]["gap"]["mean"]
    out["exec_cpu_ms_mean"] = out["exec"]["cpu"]["mean"]
    # Methodology transparency: how many exec gaps are request-switch outliers
    # (> 3x median) — these are the steps the median rejects.
    gp50 = out["host_gap_ms"]
    out["host_gap_outliers_gt3x"] = sum(1 for g in exec_recs["gap"] if g > 3.0 * gp50) if gp50 else 0
    return out


# ---------------------------------------------------------------------------
# Prometheus /metrics -> E_accept (mean acceptance length) + acceptance rate
# ---------------------------------------------------------------------------
def _prom_value(text: str, metric: str) -> float | None:
    total = 0.0
    found = False
    # Counters may be suffixed _total and split per-engine; sum all label sets.
    pat = re.compile(rf"^{re.escape(metric)}(?:_total)?(?:\{{[^}}]*\}})?\s+([\d.eE+-]+)$", re.M)
    for m in pat.finditer(text):
        try:
            total += float(m.group(1))
            found = True
        except ValueError:
            pass
    return total if found else None


def parse_spec_metrics(metrics_text: str) -> dict[str, Any]:
    drafts = _prom_value(metrics_text, "vllm:spec_decode_num_drafts")
    accepted = _prom_value(metrics_text, "vllm:spec_decode_num_accepted_tokens")
    draft_tokens = _prom_value(metrics_text, "vllm:spec_decode_num_draft_tokens")
    out: dict[str, Any] = {
        "num_drafts": drafts,
        "num_accepted_tokens": accepted,
        "num_draft_tokens": draft_tokens,
    }
    if drafts and accepted is not None:
        out["e_accept_mean_acceptance_length"] = 1.0 + accepted / drafts
    if draft_tokens and accepted is not None:
        out["draft_acceptance_rate"] = accepted / draft_tokens
    return out


# ---------------------------------------------------------------------------
# Server-log -> E_accept (vLLM's own SpecDecoding lines) + steady gen TPS
# ---------------------------------------------------------------------------
# Why parse the log and not just Prometheus: the smoke showed the /metrics
# spec_decode_* counters can come back empty (E_accept=0), but vLLM ALWAYS prints
# its own per-interval "SpecDecoding metrics: Mean acceptance length: X, ...,
# Accepted: N tokens, Drafted: M tokens" whenever spec-decode runs. Those window
# counts are non-cumulative (verified: 2997 -> 3291 -> 3007, non-monotonic), so
# summing them yields whole-run totals. With K speculative tokens/draft,
# num_drafts = drafted/K and mean acceptance length = 1 + accepted/num_drafts =
# 1 + K*accepted/drafted. (The "Accepted:/Drafted:" literals don't collide with
# the "Accepted throughput:/Drafted throughput:" lines: those have " throughput"
# between the word and the colon.)
_ACCEPT_LEN_RE = re.compile(r"Mean acceptance length:\s*([\d.]+)")
_ACCEPTED_RE = re.compile(r"Accepted:\s*(\d+)\s*tokens")
_DRAFTED_RE = re.compile(r"Drafted:\s*(\d+)\s*tokens")
_GEN_TPS_RE = re.compile(r"Avg generation throughput:\s*([\d.]+)")
_NUM_SPEC_RE = re.compile(r"num_speculative_tokens[\"']?\s*:\s*(\d+)")


def parse_spec_log(log_text: str) -> dict[str, Any]:
    """Whole-run E_accept + steady generation TPS from vLLM's own server log."""
    accepted = [int(x) for x in _ACCEPTED_RE.findall(log_text)]
    drafted = [int(x) for x in _DRAFTED_RE.findall(log_text)]
    lengths = [float(x) for x in _ACCEPT_LEN_RE.findall(log_text)]
    gen_tps = [float(x) for x in _GEN_TPS_RE.findall(log_text)]
    k_match = _NUM_SPEC_RE.search(log_text)
    num_spec = int(k_match.group(1)) if k_match else None
    total_acc = sum(accepted)
    total_draft = sum(drafted)
    out: dict[str, Any] = {
        "intervals": len(accepted),
        "num_speculative_tokens": num_spec,
        "total_accepted_tokens": total_acc,
        "total_drafted_tokens": total_draft,
        # K-free cross-check: unweighted mean of vLLM's per-interval acceptance
        # length. Intervals are ~equal-sized in wall time, so this tracks the
        # draft-weighted mean closely and needs no knowledge of K.
        "e_accept_interval_mean": statistics.fmean(lengths) if lengths else None,
        # Steady-state, whole-run decode throughput as vLLM's OWN engine meter
        # reports it (the honest local number; the probe_tps burst overstates it).
        "steady_gen_tps_mean": statistics.fmean(gen_tps) if gen_tps else None,
        "steady_gen_tps_n": len(gen_tps),
    }
    if total_draft and num_spec:
        out["e_accept_exact"] = 1.0 + num_spec * total_acc / total_draft
        out["draft_acceptance_rate"] = total_acc / total_draft
    return out


# ---------------------------------------------------------------------------
# Chrome trace -> GPU kernel category breakdown (verify sub-split)
# ---------------------------------------------------------------------------
_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def parse_trace(trace_path: Path) -> dict[str, Any]:
    """Sum self-device kernel time by component from a torch chrome trace."""
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    by_name: dict[str, float] = {}
    by_cat: dict[str, float] = {}
    total_us = 0.0
    for e in events:
        if e.get("cat") not in _GPU_CATS:
            continue
        dur = e.get("dur")
        if not dur:
            continue
        name = e.get("name", "")
        dur = float(dur)
        by_name[name] = by_name.get(name, 0.0) + dur
        by_cat[categorize(name)] = by_cat.get(categorize(name), 0.0) + dur
        total_us += dur
    top = sorted(by_name.items(), key=lambda x: -x[1])[:25]
    return {
        "trace_file": str(trace_path),
        "gpu_kernel_us_total": total_us,
        "category_ms": {k: v / 1000.0 for k, v in by_cat.items()},
        "category_pct": {k: (100.0 * v / total_us if total_us else 0.0)
                         for k, v in by_cat.items()},
        "top_kernels": [
            {"name": nm, "ms": us / 1000.0, "pct": 100.0 * us / total_us if total_us else 0.0,
             "category": categorize(nm)}
            for nm, us in top
        ],
    }


# ---------------------------------------------------------------------------
# Serving passes
# ---------------------------------------------------------------------------
def _steptime_env(expected_steps: int) -> dict[str, str]:
    """STEPTIME knobs sized so even a tiny smoke run yields stable raw records."""
    warmup = max(8, min(64, expected_steps // 5))
    raw_count = max(64, min(4000, expected_steps))
    return {
        "STEPTIME": "1",
        "STEPTIME_WARMUP_SKIP": str(warmup),
        "STEPTIME_RAW_START": str(warmup),
        "STEPTIME_RAW_COUNT": str(raw_count),
        "STEPTIME_REPORT_EVERY": "100000",  # rely on raw records, not agg
    }


def _first_prompt() -> str:
    try:
        data = json.loads(paths.EVAL_PROMPTS.read_text())
        convs = data[0].get("conversations", [])
        for turn in convs:
            if turn.get("from") == "human":
                return str(turn.get("value", ""))[:4000]
    except Exception:
        pass
    return "Explain step by step how a transformer decodes one token at a time."


def run_timing_pass(
    submission: Path, server_python: Path, out_dir: Path, label: str,
    *, num_prompts: int, output_len: int, extra_env: dict[str, str],
) -> dict[str, Any]:
    """Serve with STEPTIME=1 (no profiler) and drive the official decode workload.

    Returns the steptime split, Prometheus E_accept, and decode-throughput probes.
    """
    log_path = out_dir / f"server_{label}_timing.log"
    expected_steps = max(64, num_prompts * output_len // 2)
    # DISABLE_LOG_STATS=0 overrides the manifest's =1: the leaderboard serve path
    # ships --disable-log-stats, which de-registers vLLM's PrometheusStatLogger and
    # so hides the spec_decode_* counters (the canonical E_accept source). We need
    # them, so re-enable stats for the timing pass. V1 stat collection is a few
    # host-side counter increments per step (<0.1ms, within the host-gap noise) and
    # does not touch GPU compute, so the drafter/verify GPU split is unaffected.
    env = {
        **_steptime_env(expected_steps),
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "DISABLE_LOG_STATS": "0",
        **extra_env,
    }
    port = 8000
    res: dict[str, Any] = {"label": label, "phase": "timing"}
    with harness.LocalServer(
        submission, server_python=server_python, port=port, log_path=log_path,
        extra_env=env, startup_timeout_s=1800,
    ) as srv:
        # Confirm spec-decode + lmhead state from the launch line for the record.
        decode_summary = out_dir / f"decode_{label}.summary.json"
        decode_out = out_dir / f"decode_{label}.jsonl"
        t0 = time.time()
        summary = harness.capture_decode(
            server_python, base_url=srv.base_url, model=srv.served_model_name,
            out_file=decode_out, summary_file=decode_summary,
            num_prompts=num_prompts, output_len=output_len, timeout_s=3600,
        )
        res["decode_wall_s"] = time.time() - t0
        res["decode_summary"] = summary
        # Steady-state decode probe (prefill-isolated) — comparable to the #22
        # 867 tok/s local probe and to TPS_reconstructed.
        try:
            res["tps_probe"] = harness.probe_tps(srv.base_url, srv.served_model_name)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            res["tps_probe"] = {"error": str(exc)}
        # Prometheus E_accept (whole-run, exact) before the server is torn down.
        try:
            res["spec_metrics"] = parse_spec_metrics(_get_text(f"{srv.base_url}/metrics"))
        except (urllib.error.URLError, OSError) as exc:
            res["spec_metrics"] = {"error": str(exc)}
    log_text = log_path.read_text()
    res["steptime"] = parse_steptime(log_text)
    res["spec_log"] = parse_spec_log(log_text)
    res["server_log"] = str(log_path)
    return res


def run_kernel_pass(
    submission: Path, server_python: Path, out_dir: Path, label: str,
    *, output_len: int, kernel_window_tokens: int, extra_env: dict[str, str],
) -> dict[str, Any]:
    """Serve with vLLM's torch profiler on; capture one steady-state decode window."""
    log_path = out_dir / f"server_{label}_kernel.log"
    trace_dir = (out_dir / f"trace_{label}").resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    prof_cfg = json.dumps({
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": True,
        "ignore_frontend": True,
    })
    env = {"PROFILER_CONFIG": prof_cfg, "VLLM_USE_FLASHINFER_SAMPLER": "0", **extra_env}
    res: dict[str, Any] = {"label": label, "phase": "kernel", "trace_dir": str(trace_dir)}
    prompt = _first_prompt()
    with harness.LocalServer(
        submission, server_python=server_python, port=8000, log_path=log_path,
        extra_env=env, startup_timeout_s=1800,
    ) as srv:
        # Warmup so CUDA graphs are captured before we start recording.
        harness._completion(srv.base_url, srv.served_model_name, prompt, 16)
        _post(f"{srv.base_url}/start_profile")
        harness._completion(srv.base_url, srv.served_model_name, prompt, kernel_window_tokens)
        _post(f"{srv.base_url}/stop_profile")
        time.sleep(3)  # let the worker flush the trace handler
    traces = sorted(trace_dir.glob("*.pt.trace.json*"))
    if not traces:
        res["error"] = f"no trace written to {trace_dir}"
        res["server_log"] = str(log_path)
        return res
    res["trace"] = parse_trace(traces[-1])
    res["server_log"] = str(log_path)
    return res


# Isolation toggles. Each clears exactly one variable vs the frontier manifest.
VARIANTS: dict[str, dict[str, str]] = {
    "frontier": {},
    "spec_off": {"SPECULATIVE_CONFIG": ""},
    "lmhead_off": {"LM_HEAD_PRUNE": "0", "LM_HEAD_PRUNE_REQUIRE": "0"},
}


def profile_submission(
    submission: Path, server_python: Path, out_dir: Path,
    *, num_prompts: int, output_len: int, kernel_window_tokens: int,
    variants: list[str], do_kernel: bool,
    iso_num_prompts: int, iso_output_len: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "submission": str(submission),
        "server_python": str(server_python),
        "num_prompts": num_prompts,
        "output_len": output_len,
        "iso_num_prompts": iso_num_prompts,
        "iso_output_len": iso_output_len,
        "kernel_window_tokens": kernel_window_tokens,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": {},
    }
    for v in variants:
        extra = VARIANTS[v]
        # frontier carries the official-like 128/512 workload; the isolation
        # variants only feed verify_gpu_ms p50 + a GEMM-category trace, both stable
        # with far fewer steps, so run them light to fit the wall-clock budget.
        np_v = num_prompts if v == "frontier" else iso_num_prompts
        ol_v = output_len if v == "frontier" else iso_output_len
        print(f"\n===== variant: {v} (extra_env={extra or '{}'}) "
              f"prompts={np_v} output_len={ol_v} =====", flush=True)
        vres: dict[str, Any] = {"extra_env": extra, "num_prompts": np_v, "output_len": ol_v}
        vres["timing"] = run_timing_pass(
            submission, server_python, out_dir, v,
            num_prompts=np_v, output_len=ol_v, extra_env=extra,
        )
        if do_kernel:
            vres["kernel"] = run_kernel_pass(
                submission, server_python, out_dir, v,
                output_len=ol_v, kernel_window_tokens=kernel_window_tokens,
                extra_env=extra,
            )
        report["variants"][v] = vres
        (out_dir / "frontier_decode_profile.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------------------
# Reconstruction / analysis: turn the raw passes into the component breakdown
# ---------------------------------------------------------------------------
def _keepset_rows(model_dir: str, default: int) -> int:
    try:
        d = json.loads((Path(model_dir) / "pck04_keepset.json").read_text())
        return int(d.get("pruned_vocab_K") or len(d.get("keep_ids", [])) or default)
    except Exception:
        return default


def _lmhead_bytes(model_dir: str) -> float | None:
    """Read the packed lm_head tensor byte-size from the safetensors header."""
    st = Path(model_dir) / "model.safetensors"
    if not st.exists():
        return None
    try:
        import struct
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        total = 0
        for name, meta in header.items():
            if "lm_head" in name and isinstance(meta, dict) and "data_offsets" in meta:
                a, b = meta["data_offsets"]
                total += int(b) - int(a)
        return float(total) or None
    except Exception:
        return None


def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if v == v else 0.0  # NaN -> 0
    except (TypeError, ValueError):
        return 0.0


# A10G GDDR6 bandwidth (GB/s), for the lm_head GEMV first-principles cross-check.
A10G_BW_GBPS = 600.0
VOCAB_FULL = 262144


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    variants = report.get("variants", {})
    fr = variants.get("frontier", {})
    st = (fr.get("timing") or {}).get("steptime", {})
    tr = (fr.get("kernel") or {}).get("trace", {})

    verify_gpu = _f(st.get("verify_gpu_ms"))  # p50 steady-state (see parse_steptime)
    drafter_gpu = _f(st.get("drafter_gpu_ms"))
    host_gap = _f(st.get("host_gap_ms"))
    exec_cpu = _f(st.get("exec_cpu_ms"))
    gpu_busy = verify_gpu + drafter_gpu
    cycle_wall = exec_cpu + host_gap  # host-to-host: verify call + inter-step (incl draft)
    host_overhead = max(0.0, cycle_wall - gpu_busy)
    # Outlier-polluted means, kept only to show *why* p50 is used (the gap mean is
    # ~8x its median because a few inter-request boundaries dominate it).
    host_gap_mean = _f(st.get("host_gap_ms_mean"))
    cycle_wall_mean = _f(st.get("exec_cpu_ms_mean")) + host_gap_mean
    host_overhead_mean = max(0.0, cycle_wall_mean - gpu_busy)

    # E_accept (required test_metric). Primary source = vLLM's OWN server-log
    # SpecDecoding counters, 1 + K*accepted/drafted (whole-run, exact) — these are
    # always emitted while spec-decode runs, unlike the Prometheus /metrics
    # counters which the smoke showed can come back empty and silently report 0.
    # Fallback chain: log exact -> log per-interval mean (K-free) -> Prometheus
    # acceptance length -> Prometheus counters -> an independent TPS×cycle estimate
    # (steady_gen_tps * cycle_wall_s). The two log methods cross-validate to 4 sig
    # figs on real data (3.817 vs 3.818).
    timing = fr.get("timing") or {}
    spec_log = timing.get("spec_log") or {}
    spec = timing.get("spec_metrics") or {}
    probe = timing.get("tps_probe") or {}
    probe_decode_tps = _f(probe.get("decode_tps_single_stream"))  # warm single-burst
    steady_gen_tps = _f(spec_log.get("steady_gen_tps_mean"))      # whole-run engine meter
    # Cross-check / fallback TPS uses the steady whole-run engine meter, NOT the
    # warm probe (the probe overstates steady decode ~2x; see #22's 867 vs the
    # ~420 the engine meters over 128 prompts).
    xcheck_tps = steady_gen_tps or probe_decode_tps
    e_accept_xcheck = xcheck_tps * (cycle_wall / 1000.0) if cycle_wall else 0.0

    # E_accept source priority: vLLM's own logged counters (bulletproof) first,
    # then its logged per-interval mean, then Prometheus, then the TPS×cycle
    # fallback. The smoke's E_accept=0 was a silent Prometheus miss; the server-log
    # source removes that failure mode.
    e_accept = _f(spec_log.get("e_accept_exact"))
    e_accept_source = "server_log_counters_exact"
    if not e_accept:
        e_accept = _f(spec_log.get("e_accept_interval_mean"))
        e_accept_source = "server_log_interval_mean"
    if not e_accept:
        e_accept = _f(spec.get("e_accept_mean_acceptance_length"))
        e_accept_source = "prometheus_acceptance_length"
    if not e_accept:
        nd = _f(spec.get("num_drafts"))
        if nd:
            e_accept = 1.0 + _f(spec.get("num_accepted_tokens")) / nd
            e_accept_source = "prometheus_counters"
    if not e_accept:
        e_accept = e_accept_xcheck
        e_accept_source = "tps_cycle_cross_check_fallback"

    tps_recon = e_accept / (cycle_wall / 1000.0) if cycle_wall else float("nan")
    # Conc=1 cycle accounting has a known ambiguity: the drafter runs as its own
    # scheduler step, so whether its GPU time overlaps the verify step's host work
    # is wheel-dependent. cycle_wall (exec_cpu + host_gap) treats it as overlapped
    # (upper TPS); adding the drafter step as fully non-overlapped gives the lower
    # TPS bound. The measured steady meter should sit inside this bracket.
    cycle_wall_drafter_incl = cycle_wall + drafter_gpu
    tps_recon_drafter_incl = (e_accept / (cycle_wall_drafter_incl / 1000.0)
                              if cycle_wall_drafter_incl else float("nan"))

    cat_pct = tr.get("category_pct", {})  # % of GPU kernels (frontier window = drafter+verify)
    gemm_frac = _f(cat_pct.get("matmul_gemm")) / 100.0
    attn_frac = _f(cat_pct.get("attention")) / 100.0
    sampling_frac = _f(cat_pct.get("sampling")) / 100.0
    norm_frac = (_f(cat_pct.get("norm")) + _f(cat_pct.get("activation"))
                 + _f(cat_pct.get("elementwise_copy"))) / 100.0

    drafter_frac = (drafter_gpu / gpu_busy) if gpu_busy else 0.0

    # lm_head12k GEMM: isolate via the 16k(base)-vs-12k verify-GPU diff (bandwidth
    # model: GEMV time scales ~linearly in kept rows). Cross-check w/ first-principles.
    lm = variants.get("lmhead_off", {})
    lm_st = (lm.get("timing") or {}).get("steptime", {})
    rows_12k = _keepset_rows("/tmp/osoi5-12k-baked", 12288)
    rows_base = _keepset_rows("/tmp/osoi5-v0-baked", 16384)
    lmhead_ms_iso = None
    if lm_st and rows_base > rows_12k:
        d_verify = _f(lm_st.get("verify_gpu_ms")) - verify_gpu
        per_row = d_verify / (rows_base - rows_12k)
        lmhead_ms_iso = max(0.0, per_row * rows_12k)
    lmhead_bytes = _lmhead_bytes("/tmp/osoi5-12k-baked")
    lmhead_ms_bw = (lmhead_bytes / (A10G_BW_GBPS * 1e9) * 1000.0) if lmhead_bytes else None
    lmhead_ms = lmhead_ms_iso if lmhead_ms_iso else (lmhead_ms_bw or 0.0)
    lmhead_frac = (lmhead_ms / gpu_busy) if gpu_busy else 0.0

    # Body int4-Marlin GEMM = all GEMM minus drafter GEMM minus lm_head GEMM.
    body_gemm_frac = max(0.0, gemm_frac - drafter_frac - lmhead_frac)
    body_gemm_ms = body_gemm_frac * gpu_busy

    # spec-off control (pure verify, M=1): GEMM-flatness check vs denken #18.
    so = variants.get("spec_off", {})
    so_verify = _f(((so.get("timing") or {}).get("steptime") or {}).get("verify_gpu_ms"))
    so_tr = ((so.get("kernel") or {}).get("trace") or {}).get("category_pct", {})

    components = {  # fraction of decode GPU-busy time per cycle
        "drafter_forward": drafter_frac,
        "verify_body_int4_gemm": body_gemm_frac,
        "verify_lmhead12k_gemm": lmhead_frac,
        "verify_attention_fa2sw": attn_frac,
        "verify_norm_elementwise": norm_frac,
        "sampling": sampling_frac,
    }
    # Largest *addressable* component drives the next-lever recommendation.
    lever = max(components.items(), key=lambda kv: kv[1])

    gpu_share = (gpu_busy / cycle_wall) if cycle_wall else float("nan")
    return {
        "e_accept": e_accept,
        "e_accept_source": e_accept_source,
        "e_accept_cross_check_from_tps": e_accept_xcheck,
        "cycle": {
            "stat": "p50_steady_state",
            "verify_gpu_ms": verify_gpu, "drafter_gpu_ms": drafter_gpu,
            "gpu_busy_ms": gpu_busy, "host_overhead_ms": host_overhead,
            "cycle_wall_ms": cycle_wall, "exec_cpu_ms": exec_cpu, "host_gap_ms": host_gap,
            "cycle_wall_ms_drafter_incl": cycle_wall_drafter_incl,
            # Independent cross-check of the host-to-host cycle wall: the whole-run
            # engine meter says each cycle emits E_accept tokens, so cycle =
            # E_accept / steady_gen_tps. Agreement with cycle_wall_ms validates the
            # STEPTIME accounting.
            "cycle_wall_ms_steady_xcheck": (1000.0 * e_accept / steady_gen_tps)
            if steady_gen_tps else float("nan"),
            # >=100% means GPU-busy >= host-to-host wall, i.e. the host path is fully
            # overlapped behind async GPU work (execute_model returns before kernels
            # finish). Clamp for display; decode_bound carries the interpretation.
            "gpu_busy_share_of_wall": min(1.0, gpu_share) if gpu_share == gpu_share else gpu_share,
            "gpu_busy_share_of_wall_raw": gpu_share,
            "host_gap_ms_mean_polluted": host_gap_mean,
            "host_overhead_ms_mean_polluted": host_overhead_mean,
            "host_gap_outliers_gt3x": st.get("host_gap_outliers_gt3x"),
            "decode_bound": "gpu" if gpu_share >= 0.85 else "host",
        },
        "tps": {
            "reconstructed_decode_tps": tps_recon,
            "reconstructed_decode_tps_drafter_incl": tps_recon_drafter_incl,
            "measured_steady_gen_tps": steady_gen_tps,
            "measured_probe_decode_tps_warm": probe_decode_tps,
            "ratio_recon_over_steady": (tps_recon / steady_gen_tps)
            if steady_gen_tps else float("nan"),
        },
        "gpu_busy_composition_frac": components,
        "verify_subsplit_ms": {
            "body_int4_gemm": body_gemm_ms,
            "lmhead12k_gemm": lmhead_ms,
            "attention_fa2sw": attn_frac * gpu_busy,
            "norm_elementwise": norm_frac * gpu_busy,
            "sampling": sampling_frac * gpu_busy,
        },
        "lmhead": {
            "rows_12k": rows_12k, "rows_base": rows_base, "full_vocab": VOCAB_FULL,
            "ms_isolation": lmhead_ms_iso, "ms_bandwidth_model": lmhead_ms_bw,
            "ms_used": lmhead_ms, "frac_of_gpu_busy": lmhead_frac,
            "implied_full262k_frac": (lmhead_frac * VOCAB_FULL / rows_12k) if rows_12k else None,
        },
        "isolation": {
            "spec_off_verify_gpu_ms": so_verify,
            "frontier_verify_gpu_ms": verify_gpu,
            "verify_gemm_flat_in_M_check": so_tr.get("matmul_gemm"),
            "lmhead_off_verify_gpu_ms": _f(lm_st.get("verify_gpu_ms")) if lm_st else None,
        },
        "trace_category_pct": cat_pct,
        "primary_metric": {"name": "frontier_verify_body_gemm_frac", "value": body_gemm_frac},
        "test_metric": {"name": "frontier_E_accept_tokens_per_cycle", "value": e_accept},
        "next_lever": {"component": lever[0], "frac_of_gpu_busy": lever[1]},
    }


def render_markdown(report: dict[str, Any], a: dict[str, Any]) -> str:
    c = a["cycle"]
    comp = a["gpu_busy_composition_frac"]
    L = []
    L.append("# Frontier decode-step profile — component breakdown\n")
    L.append(f"_Submission_: `{report['submission']}`  ")
    L.append(f"_Workload_: conc=1, {report['num_prompts']} prompts, output_len "
             f"{report['output_len']}, CUDA graphs ON  ")
    L.append(f"_Captured_: {report['utc']}  ")
    L.append("_Local A10G exploratory probe — NOT the official a10g-small TPS._\n")
    L.append(f"**E_accept (mean acceptance length)** = **{a['e_accept']:.3f}** tokens/cycle "
             f"(source: `{a.get('e_accept_source')}`; TPS×cycle cross-check "
             f"{a.get('e_accept_cross_check_from_tps', 0):.2f})\n")
    L.append("## Steady-state spec-decode cycle (p50 per-step; means reject prefill + "
             "inter-request outliers)\n")
    L.append("| quantity | ms | note |")
    L.append("|---|---|---|")
    L.append(f"| drafter forward (GPU) | {c['drafter_gpu_ms']:.3f} | STEPTIME `kind=draft` |")
    L.append(f"| verify forward (GPU) | {c['verify_gpu_ms']:.3f} | STEPTIME `kind=exec` |")
    L.append(f"| GPU-busy / cycle | {c['gpu_busy_ms']:.3f} | drafter + verify |")
    L.append(f"| host overhead / cycle | {c['host_overhead_ms']:.3f} | cycle wall − GPU-busy |")
    L.append(f"| **cycle wall** | **{c['cycle_wall_ms']:.3f}** | verify call + inter-step gap |")
    L.append(f"| GPU-busy share of wall | {100*c['gpu_busy_share_of_wall']:.1f}% | "
             f"**decode is {str(c.get('decode_bound','?')).upper()}-bound** |")
    L.append(f"| _(host gap p50 vs polluted mean)_ | {c['host_gap_ms']:.3f} vs "
             f"{c.get('host_gap_ms_mean_polluted', float('nan')):.3f} | "
             f"{c.get('host_gap_outliers_gt3x')} request-switch gaps >3× median |\n")
    L.append("## Decode GPU-busy composition (share of GPU-busy/cycle)\n")
    L.append("| component | % of GPU-busy | measured/inferred |")
    L.append("|---|---|---|")
    labels = {
        "verify_body_int4_gemm": ("verify body int4-Marlin GEMM", "trace − drafter − lmhead"),
        "verify_attention_fa2sw": ("verify attention (fa2sw)", "trace (direct)"),
        "drafter_forward": ("drafter forward", "STEPTIME (direct)"),
        "verify_lmhead12k_gemm": ("verify lmhead12k GEMM", "isolation 16k↔12k"),
        "verify_norm_elementwise": ("verify norm/elementwise", "trace (direct)"),
        "sampling": ("sampling", "trace (direct)"),
    }
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        lab, src = labels.get(k, (k, ""))
        L.append(f"| {lab} | {100*v:.1f}% | {src} |")
    t = a["tps"]
    L.append("\n## TPS reconstruction (local A10G probe — not the official a10g-small TPS)\n")
    L.append(f"- TPS_reconstructed = E_accept / cycle_wall = "
             f"**{t['reconstructed_decode_tps']:.1f} tok/s** "
             f"(drafter-inclusive lower bound {_f(t.get('reconstructed_decode_tps_drafter_incl')):.1f})")
    L.append(f"- measured steady decode TPS (whole-run engine meter) = "
             f"{_f(t.get('measured_steady_gen_tps')):.1f} tok/s  ← the honest local number")
    L.append(f"- warm single-burst probe (≈ #22's ~867) = "
             f"{_f(t.get('measured_probe_decode_tps_warm')):.1f} tok/s (overstates steady; reported for #22 continuity)")
    L.append(f"- ratio recon/steady = {_f(t.get('ratio_recon_over_steady')):.3f}\n")
    lm = a["lmhead"]
    L.append("## lm_head share vs PR #8 (262k base = 26.4% of decode GPU)\n")
    L.append(f"- lmhead12k GEMM now **{100*lm['frac_of_gpu_busy']:.1f}%** of GPU-busy "
             f"({lm['rows_12k']} rows). Isolation est {_f(lm['ms_isolation']):.4f} ms, "
             f"bandwidth-model est {_f(lm['ms_bandwidth_model']):.4f} ms.")
    if lm.get("implied_full262k_frac"):
        L.append(f"- per-row scaling implies a *full* 262k head would be "
                 f"~{100*lm['implied_full262k_frac']:.1f}% — i.e. lmhead12k cut the "
                 f"head ~{VOCAB_FULL/lm['rows_12k']:.0f}×, consistent with the drop from 26.4%.")
    nl = a["next_lever"]
    L.append(f"\n## Next lever\n\n**{nl['component']}** is the largest addressable "
             f"component at **{100*nl['frac_of_gpu_busy']:.1f}%** of decode GPU-busy.\n")
    return "\n".join(L)


def _log_wandb(report: dict[str, Any], a: dict[str, Any], name: str, group: str) -> str | None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped ({exc})", flush=True)
        return None
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=name, group=group, job_type="profile",
            config={
                "submission": report["submission"], "num_prompts": report["num_prompts"],
                "output_len": report["output_len"], "variants": list(report["variants"]),
            },
        )
        flat: dict[str, Any] = {
            "e_accept": a["e_accept"],
            "e_accept_source": a.get("e_accept_source"),
            "e_accept_cross_check_from_tps": a.get("e_accept_cross_check_from_tps"),
            "next_lever_component": a["next_lever"]["component"],
            "decode_bound": a["cycle"].get("decode_bound"),
            "primary/frontier_verify_body_gemm_frac": a["primary_metric"]["value"],
            "test/frontier_E_accept_tokens_per_cycle": a["test_metric"]["value"],
            "next_lever_frac": a["next_lever"]["frac_of_gpu_busy"],
        }
        for k, v in a["cycle"].items():
            flat[f"cycle/{k}"] = v
        for k, v in a["gpu_busy_composition_frac"].items():
            flat[f"gpu_busy_frac/{k}"] = v
        for k, v in a["tps"].items():
            flat[f"tps/{k}"] = v
        run.summary.update(flat)
        tbl = wandb.Table(columns=["component", "frac_of_gpu_busy"])
        for k, v in sorted(a["gpu_busy_composition_frac"].items(), key=lambda kv: -kv[1]):
            tbl.add_data(k, v)
        run.log({"gpu_busy_composition": tbl})
        rid = run.id
        run.finish()
        print(f"[wandb] logged run {rid}", flush=True)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] log failed ({exc})", flush=True)
        return None


def run(
    submission: Path, server_python: Path, out_dir: Path,
    *, num_prompts: int, output_len: int, kernel_window_tokens: int,
    variants: list[str], do_kernel: bool, wandb_name: str | None, wandb_group: str,
    iso_num_prompts: int = 32, iso_output_len: int = 256,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = profile_submission(
        submission, server_python, out_dir,
        num_prompts=num_prompts, output_len=output_len,
        kernel_window_tokens=kernel_window_tokens, variants=variants, do_kernel=do_kernel,
        iso_num_prompts=iso_num_prompts, iso_output_len=iso_output_len,
    )
    a = analyze(report)
    report["analysis"] = a
    (out_dir / "frontier_decode_profile.json").write_text(json.dumps(report, indent=2))
    (out_dir / "breakdown.md").write_text(render_markdown(report, a))
    wid = None
    if wandb_name:
        wid = _log_wandb(report, a, wandb_name, wandb_group)
    a["wandb_run_id"] = wid
    print("\n========== FRONTIER DECODE PROFILE ==========", flush=True)
    print(f"E_accept                = {a['e_accept']:.3f} tokens/cycle", flush=True)
    print(f"cycle wall              = {a['cycle']['cycle_wall_ms']:.3f} ms "
          f"(GPU-busy {a['cycle']['gpu_busy_ms']:.3f} ms, "
          f"host {a['cycle']['host_overhead_ms']:.3f} ms)", flush=True)
    for k, v in sorted(a["gpu_busy_composition_frac"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s} {100*v:5.1f}% of GPU-busy", flush=True)
    print(f"primary frontier_verify_body_gemm_frac = "
          f"{a['primary_metric']['value']:.4f}", flush=True)
    print(f"next lever: {a['next_lever']['component']} "
          f"({100*a['next_lever']['frac_of_gpu_busy']:.1f}% of GPU-busy)", flush=True)
    print(f"artifacts -> {out_dir}", flush=True)
    return report

