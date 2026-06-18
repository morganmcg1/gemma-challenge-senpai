#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #691 (land) -- long-ctx 3D-vs-2D attention-pin crossover sweep.

Measures the M=1 decode attention-step GPU-time WITH the 16-way 3D split-KV reduction
(baseline) vs WITHOUT it (pinned 2D) across a context-length ladder, to find where the
"free" strict-#319 attention pin from land #684 (-0.84 ms/step at ctx~512) flips from
net-beneficial to costly, and what the WORST-CASE pin cost is across the deployed-and-
beyond ctx range.

MECHANISM (vLLM 0.22.0, grounded in source -- triton_unified_attention.py:920-932 and
triton_attn.py:54,168):
  The Gemma4 model is forced onto the TRITON_ATTN backend (heterogeneous head dims).
  Its decode attention picks a 2D vs 3D softmax-reduction path:
      use_3d = not ( seg buffers None ... or max_seqlen_q > 1
                     or num_seqs > seq_threshold_3D or is_batch_invariant )
  where  seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv  (default 128//2 = 64).
  * M=1 AR decode (max_seqlen_q=1, num_seqs=1 <= 64) -> 3D SEGMENTED path
    (grid z-dim = NUM_PAR_SOFTMAX_SEGMENTS=16 + a reduce_segments float32 combine).
  * fixed2d pin: monkeypatch MIN_LAUNCH_GRID_SIZE_2D=0 -> seq_threshold_3D=0 ->
    num_seqs(1) > 0 -> use_3d=False -> 2D path (pure attention pin, NO aten-op swaps).
  * bi1 pin:     VLLM_BATCH_INVARIANT=1 -> is_batch_invariant -> 2D path TOO, plus the
    ctx-independent aten-op (mm/log_softmax) swaps that live OUTSIDE unified_attention.

The use_3d decision is CTX-INDEPENDENT: at M=1 the baseline always takes 3D and the pin
always takes 2D, at every ctx. What varies with ctx is the KERNEL TIME -- the 16-way 3D
split is launch/reduce overhead at short KV (so the 2D pin is FREE) but amortizes to a net
occupancy win at long KV on the A10G's SMs (so the 2D pin becomes COSTLY). 35/42 Gemma4
layers are sliding-window (KV capped at window=512, so their delta is ctx-flat past 512);
only the 7 global layers grow with ctx -> they drive the crossover.

This script measures ONE --config in a fresh process (the MIN_LAUNCH_GRID_SIZE_2D patch and
the batch-invariant snapshot are process-global, fixed in the metadata builder __init__, so
configs MUST NOT be mixed). It sweeps all --ctx points in that one process (path is fixed;
only KV length changes). Writes runs/<config>.json (KB-scale). The attention-only CUDA-event
timer brackets only the M=1 decode unified_attention launches; the config-to-config delta of
the summed-per-step time is the PURE 2D-vs-3D attention cost in ABSOLUTE ms (lm_head/body
cancel -> head-independent, transferable onto the deployed 8.14 ms AR step).

analysis_only=1, official_tps=0, no_hf_job=1, fires=0.  LOCAL A10G only.
Run with the vLLM 0.22.0 venv: /tmp/senpai-venvs/20f658587e8a6643/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # target/ repo root (research/validity/attention_pin_cost -> target)
CENSUS_DIR = ROOT / "research" / "validity" / "reduction_sensitivity_census"
sys.path.insert(0, str(CENSUS_DIR))

# attention-only CUDA-event timer state (process-global; the wrapped unified_attention
# pushes (start,end) events here only while enabled and only for M=1 decode launches).
_ATTN_TIMER: dict = {"enabled": False, "events": []}


def _setup_config(config: str) -> dict:
    """Set the process-global pin BEFORE importing vllm. Returns a provenance dict.

    Forces the in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0 -> InprocClient) so the
    monkeypatch + the path probe run in the SAME process that does the forward. Identical
    across configs -> relative timing stays apples-to-apples."""
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if config == "bi1":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    else:
        os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
    return {
        "config": config,
        "env_VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
        "env_VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
    }


def _apply_fixed2d_patch(prov: dict) -> None:
    """config==fixed2d: force seq_threshold_3D=0 (MIN_LAUNCH_GRID_SIZE_2D=0) so the M=1
    decode takes the 2D path, matching the M>1 verify -- a pure attention pin (no aten-op
    swap). Module-constant monkeypatch read once in the metadata builder __init__; no CUDA
    rebuild => config-reachable."""
    import vllm.v1.attention.backends.triton_attn as tab
    prov["MIN_LAUNCH_GRID_SIZE_2D_before"] = tab.MIN_LAUNCH_GRID_SIZE_2D
    tab.MIN_LAUNCH_GRID_SIZE_2D = 0
    prov["MIN_LAUNCH_GRID_SIZE_2D_after"] = tab.MIN_LAUNCH_GRID_SIZE_2D


def _install_path_probe(path_log: list) -> None:
    """Wrap the unified_attention entry triton_attn calls, to (a) record the GROUND-TRUTH
    2D/3D decision per launch (proves the pin took at every ctx) and (b) drive the
    attention-only CUDA-event timer for M=1 decode launches.

    Replicates the kernel's exact use_3d expression from the call kwargs:
      num_seqs = len(seqused_k);  is_batch_invariant = module snapshot.
    """
    import vllm.v1.attention.backends.triton_attn as tab
    import vllm.v1.attention.ops.triton_unified_attention as tua
    import torch

    real = tab.unified_attention
    bi_snapshot = bool(getattr(tua, "is_batch_invariant", False))
    cap = {"m1": 0, "mgt1": 0}
    CAP_N = 40

    def _wrapped(*args, **kw):
        msq = None
        try:
            msq = int(kw["max_seqlen_q"])
            num_seqs = len(kw["seqused_k"])
            thr = kw.get("seq_threshold_3D")
            seg_o = kw.get("softmax_segm_output")
            use_3d = not (
                thr is None
                or kw.get("num_par_softmax_segments") is None
                or seg_o is None
                or kw.get("softmax_segm_max") is None
                or kw.get("softmax_segm_expsum") is None
                or msq > 1
                or num_seqs > thr
                or bi_snapshot
            )
            bucket = "m1" if msq == 1 else "mgt1"
            if cap[bucket] < CAP_N:
                path_log.append({
                    "bucket": bucket, "max_seqlen_q": msq, "num_seqs": num_seqs,
                    "seq_threshold_3D": thr, "is_batch_invariant": bi_snapshot,
                    "use_3d": bool(use_3d),
                })
                cap[bucket] += 1
        except Exception:  # noqa: BLE001
            pass
        # time ONLY M=1 decode launches (prefill chunks have max_seqlen_q>1)
        if _ATTN_TIMER["enabled"] and msq == 1:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = real(*args, **kw)
            e.record()
            _ATTN_TIMER["events"].append((s, e))
            return out
        return real(*args, **kw)

    tab.unified_attention = _wrapped


def _synth_ctx_ids(n: int, vocab_lo: int = 1000, vocab_hi: int = 50000) -> list:
    """Deterministic synthetic context of EXACT length n. Token content is irrelevant to
    dense-attention timing (the kernel reads all KV); only the KV length matters. Avoids
    special-token ids by mapping into [vocab_lo, vocab_hi)."""
    span = vocab_hi - vocab_lo
    return [vocab_lo + (i * 2654435761) % span for i in range(n)]


def _measure_attn_step_ms(llm, sp, ctx_ids, n_layers, n_new, reps, warmup):
    """Per-decode-step attention GPU-time (summed over layers) via CUDA events on the
    wrapped unified_attention, at the KV length set by len(ctx_ids). Returns
    (median_ms_per_step, n_step_samples, median_layers_per_step)."""
    import torch

    prompt = [{"prompt_token_ids": list(ctx_ids)}]
    for _ in range(warmup):
        llm.generate(prompt, sp, use_tqdm=False)
    torch.cuda.synchronize()

    per_step = []
    layers_obs = []
    for _ in range(reps):
        _ATTN_TIMER["events"].clear()
        _ATTN_TIMER["enabled"] = True
        llm.generate(prompt, sp, use_tqdm=False)
        _ATTN_TIMER["enabled"] = False
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for (s, e) in _ATTN_TIMER["events"]]
        n_ev = len(times)
        if n_ev == 0:
            continue
        if n_ev % n_layers == 0:
            n_steps = n_ev // n_layers
            for i in range(n_steps):
                per_step.append(sum(times[i * n_layers:(i + 1) * n_layers]))
            layers_obs.append(n_layers)
        else:  # robust fallback: per-step = total / observed decode steps (~n_new)
            per_step.append(sum(times) / max(1, n_new))
            layers_obs.append(n_ev / max(1, n_new))
    per_step.sort()
    med = per_step[len(per_step) // 2] if per_step else float("nan")
    layers = (sum(layers_obs) / len(layers_obs)) if layers_obs else float("nan")
    return med, per_step, layers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, choices=["baseline", "fixed2d", "bi1"],
                    help="baseline=3D split (default); fixed2d=2D pin; bi1=2D + aten blanket")
    ap.add_argument("--ctx", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384],
                    help="KV-length ladder (decode context lengths)")
    ap.add_argument("--n-new", type=int, default=32, help="decode steps timed per generate")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prov = _setup_config(args.config)

    import torch  # noqa: F401
    from vllm import LLM, SamplingParams
    import reduction_sensitivity_census as rsc

    if args.config == "fixed2d":
        _apply_fixed2d_patch(prov)
    path_log: list = []
    _install_path_probe(path_log)

    model_dir = rsc.resolve_model_dir()
    full_vocab = rsc._margin_model_full_vocab(model_dir)
    try:
        n_layers = rsc.resolve_n_layers()
    except Exception:  # noqa: BLE001
        n_layers = 42

    max_ctx = max(args.ctx)
    max_model_len = max_ctx + args.n_new + 64

    print(f"[691:{args.config}] model={model_dir} full_vocab={full_vocab} "
          f"n_layers={n_layers} ctx={args.ctx} n_new={args.n_new} reps={args.reps} prov={prov}",
          flush=True)

    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        quantization="compressed-tensors",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=0,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.n_new, ignore_eos=True)

    per_ctx = []
    for ctx in args.ctx:
        ctx_ids = _synth_ctx_ids(ctx)
        med, samples, layers = _measure_attn_step_ms(
            llm, sp, ctx_ids, n_layers, args.n_new, args.reps, args.warmup)
        m1_use_3d = sorted({p["use_3d"] for p in path_log if p["bucket"] == "m1"})
        per_ctx.append({
            "ctx": ctx,
            "attn_step_ms": med,
            "n_step_samples": len(samples),
            "layers_per_step": layers,
            "samples_med": round(med, 5),
            "m1_use_3d_values": m1_use_3d,
        })
        print(f"[691:{args.config}] ctx={ctx:6d} attn_step={med:.5f} ms/step "
              f"layers/step={layers:.1f} m1_use_3d={m1_use_3d} (n_step_samples={len(samples)})",
              flush=True)

    mgt1_use_3d = sorted({p["use_3d"] for p in path_log if p["bucket"] == "mgt1"})
    peak_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    out = {
        "phase": "ctx_crossover_sweep",
        "config": args.config,
        "provenance": prov,
        "model_dir": model_dir,
        "margin_model_full_vocab": full_vocab,
        "n_layers": n_layers,
        "n_new": args.n_new, "reps": args.reps, "warmup": args.warmup,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "peak_mem_gib": peak_mem_gib,
        "per_ctx": per_ctx,
        "path_m1_use_3d_values": [p for p in {tuple(x["m1_use_3d_values"]) for x in per_ctx}],
        "path_verify_use_3d_values": mgt1_use_3d,
        "path_log_sample": path_log[:12],
    }
    out_path = Path(args.out) if args.out else (HERE / "runs" / f"{args.config}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print(f"[691:{args.config}] peak_mem={peak_mem_gib:.2f} GiB  M=1 use_3d across ctx="
          f"{[x['m1_use_3d_values'] for x in per_ctx]}  M>1 use_3d={mgt1_use_3d}", flush=True)
    print(f"[691:{args.config}] -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
