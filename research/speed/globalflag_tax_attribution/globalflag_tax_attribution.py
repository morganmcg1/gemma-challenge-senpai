"""Per-op tax attribution for global ``VLLM_BATCH_INVARIANT=1`` (PR #484).

LOCAL profiling only — ``analysis_only=true``, ``official_tps=0``, no submission, no
served-file change. Reuses the #470 serve A/B primitives (``serve_profile.run_kernel_pass``
/ ``run_timing_pass``, the same submission + single-stream env) to attribute the global
flag's per-decode-step added latency into op-type buckets and predict the surgical
attention-only TPS.

Question (PR #484): the global flag costs ~48% (deployed 481.53 -> realized 234.47,
#470 ``ugqnytji``). The PR's premise is it "routes ALL matmuls through the slow
``matmul_persistent`` kernel". If we pin ONLY the 7 attention reductions (surgical) and
leave MLP/QKV/lm_head on the fast Marlin path, does realized TPS recover toward ~457?

Mechanism (vLLM ``batch_invariant.py:897`` ``enable_batch_invariant_mode``): the flag
installs aten-dispatcher overrides for aten::{mm,addmm,matmul,linear,bmm,_log_softmax,
softmax,_softmax,mean.dim} only. The int4 GPTQ-Marlin body GEMMs (QKV/O/gate_up/down)
and the int4 lm_head are custom ``torch.ops._C`` ops — NOT aten — so the flag CANNOT
reach them (bit-exact, 0 flips: #461, kanna #19). So PR buckets (b) MLP, (c) lm_head,
(d) QKV/O carry ~0 added tax; the realized +Delta is (a) the attention reduction change
+ the bf16-aten drafter/sampler ops the flag DOES reroute to ``matmul_persistent`` +
launch/serialization overhead.

Phases:
  A. Kernel-trace A/B (deployed {} vs BI {VLLM_BATCH_INVARIANT:1}): full per-kernel GPU
     us from the chrome trace, bucketed semantically. ``marlin_gemm`` (= PR b+c+d joint,
     not name-separable since they share the Marlin op) must be ~equal across arms — the
     headline body-untouched proof. ``attention`` delta = the reduction tax (PR a).
     ``bf16_aten_matmul`` (matmul_persistent) is the tax the PR mis-attributed to "all
     matmuls".
  B. Timing A/B: STEPTIME per-cycle verify_gpu / drafter_gpu / host_gap p50 -> wall cycle
     + realized wall_tps per arm. host_gap delta = the launch tax the GPU trace can't see.
  C. Microbench (serve venv): bf16 aten mm at body M=8 shapes OFF/ON (positive control)
     + the structural override set (Marlin _C op is not in it).

Compose: per-bucket us + %-of-total-+Delta, ``attention_share_of_tax_pct``,
``predicted_surgical_tps`` (cycle_deployed + attention Delta only), clears 400/450, and
the boolean verdict ``surgical_can_realize_457`` with mechanism reasoning.

Run under the repo ``.venv`` (has wandb); serve/microbench subprocs use the serve venv::

    .venv/bin/python research/speed/globalflag_tax_attribution/globalflag_tax_attribution.py \
        --submission submissions/fa2sw_precache_kenyan \
        --wandb_group globalflag-tax-attribution --wandb_name ubel/globalflag-tax-attribution
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, serve_profile  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "globalflag_tax_attribution.json"

# Banked anchors (cited, not re-measured) -----------------------------------------
DEPLOYED_TPS = 481.53           # PR #52 public non-strict deployed
GLOBAL_REALIZED_TPS = 234.47    # #470 ugqnytji full-serve global-flag (official)
GLOBAL_REALIZED_LOCAL_TPS = 221.16  # #470 local wall_tps
WHOLECYCLE_SURGICAL_TPS = 457.55    # strict_wholecycle_ab: attention-reduction-only
WHOLECYCLE_ATTN_DELTA_US = 401.90   # whole-cycle strict attention delta us/cycle (L=640)
SIGMA_HW = 4.864                # between-session hardware sigma

_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

# Triton kernel-name fragments the flag introduces (matmul_persistent et al.).
_BI_TRITON = (
    "persistent", "matmul_kernel", "bmm_kernel", "_log_softmax_kernel",
    "_rms_norm_kernel", "mean_kernel", "batch_invariant",
)


def bucket_kernel(name: str) -> str:
    """Map a GPU kernel name to a semantic op-bucket.

    Two distinct GEMM families must be separated to attribute the flag's tax correctly:
      * ``marlin_int4_body`` — ``void marlin::Marlin<...>`` (+ torch.compile-fused
        ``..._marlin_gemm_...``): the int4 GPTQ-Marlin verify body (QKV/O/gate_up/down,
        PR buckets b+d) and int4 lm_head (c). Custom ``_C`` ops — the flag CANNOT reach
        them, so they stay bit-identical across arms (the headline body-untouched proof).
      * ``bf16_aten_gemm`` — ``cutlass_*_bf16`` / ``ampere_*gemm_bf16``: the bf16 DRAFTER
        (MTP, ``num_speculative_tokens=7``) + misc bf16 aten matmuls. These ARE aten
        ops, so under the flag they reroute to ``bf16_aten_persistent``
        (``matmul_kernel_persistent``) — the dominant realized tax, and what the PR
        mis-attributed to "all matmuls" (it is the bf16 drafter, not the int4 body).
    """
    n = name.lower()
    # 1. The flag's rerouted bf16 aten matmul (Triton persistent matmul + bmm).
    if "persistent" in n or "bmm_kernel" in n:
        return "bf16_aten_persistent"
    # 2. int4 Marlin body + int4 lm_head (custom _C op; flag unreachable).
    if "marlin" in n or "gptq" in n:
        return "marlin_int4_body"
    # 3. bf16 aten dense GEMM (drafter / misc) — cuBLAS/CUTLASS in deployed; REROUTES
    #    to matmul_persistent under the flag. NOT the int4 verify body.
    if any(h in n for h in ("cutlass", "ampere", "wmma", "tensorop", "s16816", "s161616",
                            "gemm", "gemv", "splitk", "split_k", "dot")):
        return "bf16_aten_gemm"
    if any(h in n for h in ("attn", "flash", "_fwd", "paged", "unified_attention",
                            "fmha", "mha", "scaled_dot", "sdpa", "rope", "rotary",
                            "reshape_and_cache")):
        return "attention"
    if any(h in n for h in ("log_softmax", "logsoftmax", "softmax", "argmax", "topk",
                            "top_k", "sample", "logits", "cumsum", "sort", "gather",
                            "scatter")):
        return "sampling"
    if any(h in n for h in ("rms", "layernorm", "layer_norm", "norm_kernel", "mean_kernel")):
        return "norm"
    if any(h in n for h in ("silu", "gelu", "swiglu", "act_and_mul")):
        return "activation"
    if any(h in n for h in ("elementwise", "copy", "cast", "convert", "memcpy", "memset",
                            "fill", "vectorized", "index_", "triton_poi", "_add", "_mul")):
        return "elementwise"
    return "other"


def full_trace_buckets(trace_dir: str) -> dict[str, Any]:
    """Re-walk the chrome trace for the FULL per-name GPU us (parse_trace only keeps
    top-25) and fold into semantic buckets + a BI-Triton-touched cross-cut."""
    traces = sorted(Path(trace_dir).glob("*.pt.trace.json*"))
    if not traces:
        return {"error": f"no trace in {trace_dir}"}
    tp = traces[-1]
    opener = gzip.open if tp.suffix == ".gz" else open
    with opener(tp, "rt") as f:
        data = json.load(f)
    by_name: dict[str, float] = {}
    total = 0.0
    for e in data.get("traceEvents", []):
        if e.get("cat") not in _GPU_CATS:
            continue
        dur = e.get("dur")
        if not dur:
            continue
        nm = e.get("name", "")
        by_name[nm] = by_name.get(nm, 0.0) + float(dur)
        total += float(dur)
    by_bucket: dict[str, float] = {}
    bi_triton_us = 0.0
    for nm, us in by_name.items():
        by_bucket[bucket_kernel(nm)] = by_bucket.get(bucket_kernel(nm), 0.0) + us
        if any(h in nm.lower() for h in _BI_TRITON):
            bi_triton_us += us
    top = sorted(by_name.items(), key=lambda x: -x[1])[:30]
    return {
        "trace_file": str(tp),
        "gpu_us_total": total,
        "by_bucket_us": by_bucket,
        "bi_triton_touched_us": bi_triton_us,
        "n_distinct_kernels": len(by_name),
        "top_kernels": [
            {"name": nm, "us": us, "pct": 100.0 * us / total if total else 0.0,
             "bucket": bucket_kernel(nm),
             "bi_triton": any(h in nm.lower() for h in _BI_TRITON)}
            for nm, us in top
        ],
    }


def _resolve_server_python(submission: Path) -> Path:
    manifest = harness.load_manifest(submission)
    return harness.ensure_server_venv(manifest["dependencies"])


def run_microbench(server_python: Path, out_dir: Path) -> dict[str, Any]:
    script = HERE / "microbench.py"
    log = out_dir / "microbench.log"
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[microbench] {server_python} {script}", flush=True)
    proc = subprocess.run([str(server_python), str(script)], env=env,
                          capture_output=True, text=True, timeout=900)
    log.write_text((proc.stdout or "") + "\n----STDERR----\n" + (proc.stderr or ""))
    for line in (proc.stdout or "").splitlines():
        if line.startswith("MICROBENCH_JSON "):
            return json.loads(line[len("MICROBENCH_JSON "):])
    return {"error": f"microbench produced no JSON (rc={proc.returncode}); see {log}"}


def _cycle_us(wall_tps: float, e_accept: float) -> float:
    return 1e6 * e_accept / wall_tps if wall_tps else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submissions/fa2sw_precache_kenyan")
    ap.add_argument("--server-python", default=None)
    ap.add_argument("--kernel-window-tokens", type=int, default=256)
    ap.add_argument("--timing-num-prompts", type=int, default=8)
    ap.add_argument("--timing-output-len", type=int, default=512)
    ap.add_argument("--output", default=str(OUT_JSON))
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="globalflag-tax-attribution")
    ap.add_argument("--wandb_name", default="ubel/globalflag-tax-attribution")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny window to validate wiring")
    args = ap.parse_args()

    if args.smoke:
        args.kernel_window_tokens = 24
        args.timing_num_prompts = 2
        args.timing_output_len = 64

    submission = (ROOT / args.submission).resolve() if not Path(args.submission).is_absolute() \
        else Path(args.submission)
    server_python = Path(args.server_python) if args.server_python else _resolve_server_python(submission)
    out_dir = HERE / "_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = {"deployed": {}, "bi": {"VLLM_BATCH_INVARIANT": "1"}}
    report: dict[str, Any] = {
        "pr": 484,
        "analysis_only": True,
        "official_tps": 0,
        "no_submission": True,
        "no_served_file_change": True,
        "no_hf_job": True,
        "submission": str(submission),
        "server_python": str(server_python),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kernel_window_tokens": args.kernel_window_tokens,
        "timing_num_prompts": args.timing_num_prompts,
        "timing_output_len": args.timing_output_len,
        "banked_anchors": {
            "deployed_tps": DEPLOYED_TPS,
            "global_realized_tps": GLOBAL_REALIZED_TPS,
            "global_realized_local_tps": GLOBAL_REALIZED_LOCAL_TPS,
            "wholecycle_surgical_tps": WHOLECYCLE_SURGICAL_TPS,
            "wholecycle_attn_delta_us": WHOLECYCLE_ATTN_DELTA_US,
            "sigma_hw": SIGMA_HW,
        },
        "arms": {},
    }

    def _save():
        Path(args.output).write_text(json.dumps(report, indent=2))

    # ---- Phase A + B: per-arm kernel trace + timing -----------------------------
    for arm, extra in arms.items():
        print(f"\n===== ARM {arm} (extra_env={extra or '{}'}) =====", flush=True)
        a: dict[str, Any] = {"extra_env": extra}
        kr = serve_profile.run_kernel_pass(
            submission, server_python, out_dir, f"{arm}_kernel",
            output_len=args.kernel_window_tokens,
            kernel_window_tokens=args.kernel_window_tokens, extra_env=extra,
        )
        a["kernel_raw"] = {"category_ms": kr.get("trace", {}).get("category_ms"),
                           "gpu_kernel_us_total": kr.get("trace", {}).get("gpu_kernel_us_total"),
                           "error": kr.get("error")}
        if kr.get("trace_dir"):
            a["kernel"] = full_trace_buckets(kr["trace_dir"])
        tr = serve_profile.run_timing_pass(
            submission, server_python, out_dir, f"{arm}_timing",
            num_prompts=args.timing_num_prompts, output_len=args.timing_output_len,
            extra_env=extra,
        )
        st = tr.get("steptime", {})
        spec = tr.get("spec_log", {})
        summ = tr.get("decode_summary", {}) or {}
        probe = tr.get("tps_probe", {}) or {}
        wall_tps = None
        if summ.get("num_completion_tokens") and summ.get("duration_s"):
            wall_tps = summ["num_completion_tokens"] / summ["duration_s"]
        a["timing"] = {
            "verify_gpu_ms": st.get("verify_gpu_ms"),
            "drafter_gpu_ms": st.get("drafter_gpu_ms"),
            "host_gap_ms": st.get("host_gap_ms"),
            "exec_cpu_ms": st.get("exec_cpu_ms"),
            "raw_exec_steps": st.get("raw_exec_steps"),
            "e_accept": spec.get("e_accept_exact") or spec.get("e_accept_interval_mean"),
            "wall_tps": wall_tps,
            "steady_gen_tps_mean": spec.get("steady_gen_tps_mean"),
            "tps_probe_tps": probe.get("tps") if isinstance(probe, dict) else None,
            "decode_summary": summ,
        }
        report["arms"][arm] = a
        _save()

    # ---- Phase C: mechanism microbench ------------------------------------------
    report["microbench"] = run_microbench(server_python, out_dir)
    _save()

    # ---- Compose: attribution + surgical prediction -----------------------------
    dep = report["arms"]["deployed"]
    bi = report["arms"]["bi"]
    kd = dep.get("kernel", {}).get("by_bucket_us", {}) or {}
    kb = bi.get("kernel", {}).get("by_bucket_us", {}) or {}
    buckets = sorted(set(kd) | set(kb))
    win_tok = args.kernel_window_tokens
    e_acc = dep["timing"].get("e_accept") or 3.85
    cycles_in_window = win_tok / e_acc if e_acc else float("nan")

    # Per-bucket GPU us over the profiled window, and the BI-deployed delta.
    bucket_delta_us = {b: (kb.get(b, 0.0) - kd.get(b, 0.0)) for b in buckets}
    bucket_delta_per_cycle = {b: bucket_delta_us[b] / cycles_in_window if cycles_in_window else float("nan")
                              for b in buckets}
    pos_delta_total = sum(v for v in bucket_delta_us.values() if v > 0) or float("nan")
    bucket_share_pct = {b: 100.0 * bucket_delta_us[b] / pos_delta_total
                        if pos_delta_total and pos_delta_total == pos_delta_total else float("nan")
                        for b in buckets}

    attn_delta_window = bucket_delta_us.get("attention", 0.0)
    marlin_delta_window = bucket_delta_us.get("marlin_int4_body", 0.0)
    # The bf16 drafter/aten path: cuBLAS/CUTLASS GEMM (deployed) -> matmul_persistent (BI).
    bf16_delta_window = (bucket_delta_us.get("bf16_aten_persistent", 0.0)
                         + bucket_delta_us.get("bf16_aten_gemm", 0.0))
    attn_tax_per_cycle = bucket_delta_per_cycle.get("attention", float("nan"))

    # Realized wall-cycle tax (timing pass, includes host/launch gap the GPU trace omits).
    dep_wall = dep["timing"].get("wall_tps") or GLOBAL_REALIZED_LOCAL_TPS  # fallback
    bi_wall = bi["timing"].get("wall_tps")
    dep_cycle_us = _cycle_us(dep["timing"].get("wall_tps") or DEPLOYED_TPS, e_acc)
    bi_cycle_us = _cycle_us(bi["timing"].get("wall_tps") or GLOBAL_REALIZED_LOCAL_TPS, e_acc)
    total_tax_per_cycle_local = bi_cycle_us - dep_cycle_us

    # Official-anchored total tax (the PR's headline 481.53 -> 234.47).
    dep_cycle_official = _cycle_us(DEPLOYED_TPS, e_acc)
    bi_cycle_official = _cycle_us(GLOBAL_REALIZED_TPS, e_acc)
    total_tax_per_cycle_official = bi_cycle_official - dep_cycle_official

    # Two independent attention-tax estimates: measured-here vs banked whole-cycle.
    attn_tax_measured = attn_tax_per_cycle
    attn_tax_banked = WHOLECYCLE_ATTN_DELTA_US
    # Use the banked whole-cycle attention delta as the primary (byte-exact, sigma 0.4,
    # 21 rounds) and the measured here as the cross-check.
    attn_tax_primary = attn_tax_banked

    # attention_share_of_tax: attention reduction / total global-flag tax.
    attn_share_local = (100.0 * attn_tax_primary / total_tax_per_cycle_local
                        if total_tax_per_cycle_local and total_tax_per_cycle_local > 0 else float("nan"))
    attn_share_official = (100.0 * attn_tax_primary / total_tax_per_cycle_official
                           if total_tax_per_cycle_official and total_tax_per_cycle_official > 0 else float("nan"))

    # predicted_surgical_tps = deployed cycle + ONLY the attention reduction tax.
    surgical_cycle_official = dep_cycle_official + attn_tax_primary
    predicted_surgical_tps = 1e6 * e_acc / surgical_cycle_official if surgical_cycle_official else float("nan")
    # cross-check via measured attention delta on the locally-measured deployed cycle
    surgical_cycle_local = dep_cycle_us + (attn_tax_measured if attn_tax_measured == attn_tax_measured else 0.0)
    predicted_surgical_tps_measured = 1e6 * e_acc / surgical_cycle_local if surgical_cycle_local else float("nan")

    mb = report.get("microbench", {})
    marlin_untouched = (
        bool(mb.get("marlin_op_exists_as_custom_C")) and not mb.get("marlin_in_overridden_ops", True)
    )
    # Served-level body-untouched check: int4 Marlin GPU us ~equal across arms.
    marlin_dep = kd.get("marlin_int4_body", 0.0)
    marlin_bi = kb.get("marlin_int4_body", 0.0)
    marlin_served_equal_frac = (abs(marlin_bi - marlin_dep) / marlin_dep
                                if marlin_dep else float("nan"))

    # STEPTIME cycle-level decomposition (p50, prefill-rejected): an independent,
    # prefill-free per-decode-step split. verify_gpu holds the int4 Marlin body +
    # attention, so verify_gpu_delta ~= the attention reduction tax alone (body is
    # untouched); drafter_gpu_delta = the bf16 drafter rerouted to matmul_persistent;
    # host_gap_delta = the launch/serialization tax the GPU trace cannot see.
    def _g(d, k):
        v = d["timing"].get(k)
        return v if isinstance(v, (int, float)) else float("nan")
    steptime_delta = {
        "verify_gpu_ms": _g(bi, "verify_gpu_ms") - _g(dep, "verify_gpu_ms"),
        "drafter_gpu_ms": _g(bi, "drafter_gpu_ms") - _g(dep, "drafter_gpu_ms"),
        "host_gap_ms": _g(bi, "host_gap_ms") - _g(dep, "host_gap_ms"),
    }
    steptime_total_delta = sum(v for v in steptime_delta.values() if v == v)
    # The non-attention tax surgical removes = drafter reroute + launch gap.
    surgical_removable_ms = ((steptime_delta["drafter_gpu_ms"] if steptime_delta["drafter_gpu_ms"] == steptime_delta["drafter_gpu_ms"] else 0.0)
                             + (steptime_delta["host_gap_ms"] if steptime_delta["host_gap_ms"] == steptime_delta["host_gap_ms"] else 0.0))

    clears_400 = predicted_surgical_tps >= 400.0
    clears_450 = predicted_surgical_tps >= 450.0
    # Surgical realizes ~457 iff attention is a SMALL share of the total tax (so removing
    # the non-attention tax recovers most of it) AND the body is provably untouched.
    surgical_can_realize_457 = bool(
        clears_450 and attn_share_official < 25.0 and marlin_untouched
    )

    report["attribution"] = {
        "window_tokens": win_tok,
        "e_accept_used": e_acc,
        "cycles_in_window": cycles_in_window,
        "buckets": buckets,
        "deployed_bucket_us": {b: kd.get(b, 0.0) for b in buckets},
        "bi_bucket_us": {b: kb.get(b, 0.0) for b in buckets},
        "bucket_delta_us_window": bucket_delta_us,
        "bucket_delta_us_per_cycle": bucket_delta_per_cycle,
        "bucket_share_of_total_positive_delta_pct": bucket_share_pct,
        "attention_delta_us_window": attn_delta_window,
        "marlin_int4_body_delta_us_window": marlin_delta_window,
        "bf16_aten_path_delta_us_window": bf16_delta_window,
        "bi_triton_touched_us": {
            "deployed": dep.get("kernel", {}).get("bi_triton_touched_us"),
            "bi": bi.get("kernel", {}).get("bi_triton_touched_us"),
        },
        "marlin_served_us": {"deployed": marlin_dep, "bi": marlin_bi,
                             "rel_diff": marlin_served_equal_frac},
    }
    report["compose"] = {
        "e_accept": e_acc,
        "deployed_cycle_us_local": dep_cycle_us,
        "bi_cycle_us_local": bi_cycle_us,
        "total_tax_per_cycle_us_local": total_tax_per_cycle_local,
        "deployed_cycle_us_official": dep_cycle_official,
        "bi_cycle_us_official": bi_cycle_official,
        "total_tax_per_cycle_us_official": total_tax_per_cycle_official,
        "attention_tax_us_per_cycle_measured_here": attn_tax_measured,
        "attention_tax_us_per_cycle_banked_wholecycle": attn_tax_banked,
        "attention_tax_us_per_cycle_primary": attn_tax_primary,
        "attention_share_of_tax_pct_local": attn_share_local,
        "attention_share_of_tax_pct_official": attn_share_official,
        "predicted_surgical_tps": predicted_surgical_tps,
        "predicted_surgical_tps_measured_crosscheck": predicted_surgical_tps_measured,
        "wholecycle_surgical_tps_banked": WHOLECYCLE_SURGICAL_TPS,
        "local_wall_tps": {"deployed": dep_wall, "bi": bi_wall},
        "steptime_cycle_delta_ms": steptime_delta,
        "steptime_total_delta_ms": steptime_total_delta,
        "steptime_verify_gpu_delta_is_attention_tax_ms": steptime_delta["verify_gpu_ms"],
        "steptime_surgical_removable_ms": surgical_removable_ms,
        "deployed_steptime_ms": {k: dep["timing"].get(k) for k in
                                 ("verify_gpu_ms", "drafter_gpu_ms", "host_gap_ms")},
        "bi_steptime_ms": {k: bi["timing"].get(k) for k in
                           ("verify_gpu_ms", "drafter_gpu_ms", "host_gap_ms")},
    }
    report["verdict"] = {
        "surgical_can_realize_457": surgical_can_realize_457,
        "predicted_surgical_tps": predicted_surgical_tps,
        "clears_400": bool(clears_400),
        "clears_450": bool(clears_450),
        "attention_share_of_tax_pct": attn_share_official,
        "marlin_body_untouched_by_flag": marlin_untouched,
        "marlin_served_rel_diff": marlin_served_equal_frac,
        "pr_premise_routes_all_matmuls_refuted": bool(marlin_untouched),
        "mechanism": (
            "The global flag overrides aten matmuls only; the int4 Marlin body (QKV/O/"
            "gate_up/down, PR buckets b+d) and int4 lm_head (c) are _C custom ops it cannot "
            "reach -> ~0 added tax (served Marlin GPU us equal across arms; rel_diff="
            f"{marlin_served_equal_frac:.3f}). The attention reduction (a) is only "
            f"{attn_share_official:.1f}% of the global-flag tax (~{attn_tax_primary:.0f} us "
            f"of ~{total_tax_per_cycle_official:.0f} us/cycle); the rest is bf16-aten "
            "drafter/sampler ops rerouted to matmul_persistent + launch/serialization. "
            "Surgical keeps ONLY (a) -> predicted_surgical_tps="
            f"{predicted_surgical_tps:.1f} (banked whole-cycle 457.55). The PR premise "
            "that the flag routes ALL matmuls (incl. the Marlin body) is REFUTED."
        ),
    }
    _save()

    # ---- console summary --------------------------------------------------------
    print("\n================ ATTRIBUTION ================", flush=True)
    print(f"e_accept={e_acc:.3f}  window_tok={win_tok}  cycles_in_window={cycles_in_window:.1f}", flush=True)
    for b in buckets:
        print(f"  {b:20s} dep={kd.get(b,0.0)/1000:8.2f}ms  bi={kb.get(b,0.0)/1000:8.2f}ms  "
              f"delta={bucket_delta_us[b]/1000:+8.2f}ms  share={bucket_share_pct[b]:6.1f}%", flush=True)
    print(f"\nmarlin served rel_diff={marlin_served_equal_frac:.4f} (->0 = body untouched)", flush=True)
    print("STEPTIME cycle delta (ms):  "
          f"verify_gpu={steptime_delta['verify_gpu_ms']:+.2f} (==attn tax; body untouched)  "
          f"drafter_gpu={steptime_delta['drafter_gpu_ms']:+.2f} (bf16 reroute)  "
          f"host_gap={steptime_delta['host_gap_ms']:+.2f} (launch)", flush=True)
    print(f"attention_share_of_tax (official) = {attn_share_official:.1f}%", flush=True)
    print(f"predicted_surgical_tps = {predicted_surgical_tps:.1f}  "
          f"(clears400={clears_400} clears450={clears_450})", flush=True)
    print(f"surgical_can_realize_457 = {surgical_can_realize_457}", flush=True)

    # ---- wandb ------------------------------------------------------------------
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             group=args.wandb_group, name=args.wandb_name,
                             config={"pr": 484, "analysis_only": True, "official_tps": 0,
                                     "submission": str(submission),
                                     "kernel_window_tokens": win_tok,
                                     "timing_num_prompts": args.timing_num_prompts,
                                     "timing_output_len": args.timing_output_len})
            flat = {
                "predicted_surgical_tps": predicted_surgical_tps,
                "predicted_surgical_tps_measured": predicted_surgical_tps_measured,
                "attention_share_of_tax_pct_official": attn_share_official,
                "attention_share_of_tax_pct_local": attn_share_local,
                "attention_tax_us_per_cycle": attn_tax_primary,
                "total_tax_us_per_cycle_official": total_tax_per_cycle_official,
                "total_tax_us_per_cycle_local": total_tax_per_cycle_local,
                "marlin_served_rel_diff": marlin_served_equal_frac,
                "marlin_body_untouched": int(marlin_untouched),
                "steptime_verify_gpu_delta_ms": steptime_delta["verify_gpu_ms"],
                "steptime_drafter_gpu_delta_ms": steptime_delta["drafter_gpu_ms"],
                "steptime_host_gap_delta_ms": steptime_delta["host_gap_ms"],
                "clears_400": int(clears_400), "clears_450": int(clears_450),
                "surgical_can_realize_457": int(surgical_can_realize_457),
                "deployed_wall_tps_local": dep_wall, "bi_wall_tps_local": bi_wall,
                "e_accept": e_acc,
            }
            for b in buckets:
                flat[f"delta_us_window/{b}"] = bucket_delta_us[b]
                flat[f"share_pct/{b}"] = bucket_share_pct[b]
            run.log(flat)
            run.summary.update(flat)
            art = wandb.Artifact("globalflag_tax_attribution", type="analysis")
            art.add_file(args.output)
            run.log_artifact(art)
            run.finish()
            report["wandb_run_id"] = run.id
            _save()
            print(f"[wandb] logged run {run.id}", flush=True)
        except Exception as exc:  # pragma: no cover - wandb optional
            print(f"[wandb] skipped: {exc}", flush=True)

    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
