"""Instruction-1 breakdown: profile the REAL served MTP K=7 drafter (frontier stack)
and attribute D's per-kernel components, isolating the drafter-SPECIFIC Triton kernels
(sparse_argmax_blocks/reduce) vs the int4 Marlin GEMM body (stark's domain) and attention.

Reuses the team-standard serve_profile harness:
  * timing pass (STEPTIME=1, no profiler)  -> drafter_gpu_ms (= D) + verify split, zero perturbation
  * kernel pass (vLLM torch profiler on)   -> per-kernel device-us chrome trace

The trace is parsed BY NAME for sparse_argmax_* (they are too small for the top-25 list).
Run under the serve venv from the repo root:
  CUDA_VISIBLE_DEVICES=0 <serve_venv>/bin/python -m research.systems.drafter_kernel_tile.breakdown_profile
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def parse_by_name(trace_path: Path):
    opener = gzip.open if str(trace_path).endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        data = json.load(f)
    by_name: dict[str, float] = {}
    by_count: dict[str, int] = {}
    by_cat: dict[str, float] = {}
    total = 0.0
    for e in data.get("traceEvents", []):
        if e.get("cat") not in _GPU_CATS:
            continue
        dur = e.get("dur")
        if not dur:
            continue
        nm = e.get("name", "")
        dur = float(dur)
        by_name[nm] = by_name.get(nm, 0.0) + dur
        by_count[nm] = by_count.get(nm, 0) + 1
        by_cat[serve_profile.categorize(nm)] = by_cat.get(serve_profile.categorize(nm), 0.0) + dur
        total += dur
    return by_name, by_cat, total, by_count


def main(onegraph: str = "1", window: int = 256) -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[breakdown] {note}", flush=True)
    submission = ROOT / "submissions" / "fa2sw_precache_kenyan"
    manifest = harness.load_manifest(submission)
    server_python = harness.ensure_server_venv(manifest["dependencies"])
    out_dir = ROOT / "research" / "systems" / "drafter_kernel_tile" / (
        f"breakdown-og{onegraph}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {} if onegraph == "1" else {"ONEGRAPH": "0"}
    report: dict = {"submission": str(submission), "onegraph": onegraph, "out_dir": str(out_dir)}

    print(f"[breakdown] timing pass (D anchor, ONEGRAPH={onegraph}) ...", flush=True)
    timing = serve_profile.run_timing_pass(
        submission, server_python, out_dir, "frontier",
        num_prompts=4, output_len=window, extra_env=extra)
    st = timing.get("steptime", {})
    report["timing"] = {
        "drafter_gpu_ms": st.get("drafter_gpu_ms"),
        "drafter_gpu_ms_mean": st.get("drafter_gpu_ms_mean"),
        "exec_gpu_ms_p50": (st.get("exec", {}).get("gpu", {}) or {}).get("p50"),
        "n_draft": (st.get("draft", {}).get("gpu", {}) or {}).get("n"),
        "spec_log": timing.get("spec_log"),
    }
    print(f"[breakdown] drafter_gpu_ms (D) = {st.get('drafter_gpu_ms')}", flush=True)

    print(f"[breakdown] kernel pass (torch profiler, ONEGRAPH={onegraph}) ...", flush=True)
    kernel = serve_profile.run_kernel_pass(
        submission, server_python, out_dir, "frontier",
        output_len=window, kernel_window_tokens=window, extra_env=extra)
    report["kernel_category_pct"] = kernel.get("trace", {}).get("category_pct")
    report["kernel_top"] = kernel.get("trace", {}).get("top_kernels")
    report["kernel_error"] = kernel.get("error")

    traces = sorted(Path(kernel.get("trace_dir", out_dir / "trace_frontier")).glob("*.pt.trace.json*"))
    if traces:
        by_name, by_cat, total_us, by_count = parse_by_name(traces[-1])
        sparse = {nm: us for nm, us in by_name.items() if "sparse_argmax" in nm.lower()}
        centroid = {nm: us for nm, us in by_name.items()
                    if any(h in nm.lower() for h in ("topk", "top_k", "centroid"))}
        report["trace_total_gpu_us"] = total_us
        report["sparse_argmax_us_total_window"] = sum(sparse.values())
        report["sparse_argmax_by_name"] = sparse
        report["sparse_argmax_calls_by_name"] = {nm: by_count.get(nm) for nm in sparse}
        report["centroid_topk_by_name"] = centroid
        report["category_us"] = by_cat
        report["category_pct"] = {k: 100.0 * v / total_us for k, v in by_cat.items()} if total_us else {}
        # per-decode-step attribution from kernel CALL COUNT (robust; E[T] spec_log can be null
        # under STEPTIME). The drafter runs K=7 width-1 iterations per decode step and calls the
        # blocks kernel exactly once per iteration -> n_decode_steps = blocks_calls / 7.
        D_ms = st.get("drafter_gpu_ms")
        blocks_calls = by_count.get("_sparse_argmax_blocks_kernel")
        n_steps = (blocks_calls / 7.0) if blocks_calls else None
        report["sparse_argmax_blocks_calls"] = blocks_calls
        report["n_decode_steps_in_window"] = n_steps
        # also surface E[T] if the spec_log captured it (informational only)
        sl = timing.get("spec_log") or {}
        for k in ("e_accept_exact", "e_accept_interval_mean", "e_accept"):
            if sl.get(k):
                report["E_T_used"] = float(sl[k]); break
        if n_steps and D_ms:
            sparse_per_step_us = sum(sparse.values()) / n_steps
            report["sparse_argmax_us_per_decode_step"] = sparse_per_step_us
            report["sparse_argmax_pct_of_D"] = 100.0 * (sparse_per_step_us / 1000.0) / D_ms
            report["sparse_argmax_us_per_call"] = sparse_per_step_us / 7.0
            report["sparse_argmax_per_step_by_name"] = {
                nm: us / n_steps for nm, us in sparse.items()}
        print(f"[breakdown] sparse_argmax kernels found: {list(sparse.keys())}", flush=True)
        print(f"[breakdown] sparse_argmax window us={sum(sparse.values()):.1f}  "
              f"per-step us={report.get('sparse_argmax_us_per_decode_step')}  "
              f"%%ofD={report.get('sparse_argmax_pct_of_D')}", flush=True)
        print(f"[breakdown] category_pct={report['category_pct']}", flush=True)
    else:
        report["trace_error"] = "no trace file found"

    out = out_dir / "breakdown.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[breakdown] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    og = sys.argv[1] if len(sys.argv) > 1 else "1"
    raise SystemExit(main(onegraph=og))
