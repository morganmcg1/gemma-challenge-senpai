#!/usr/bin/env python3
"""PR #560 fern — Stage 1: MEASURE the small-M int4-nominator GEMV + the full
candidate-verify decode STEP on the real 262k head, converting #549's projected
+40 head-speed gain into a measured number (LOCAL, analysis-only, NO HF fire).

#549 measured the **bf16** head GEMV (483 GB/s @ M=1, 479 @ M=8) but only *assumed*
the **int4-nominator** runs at the same achieved BW. This is the last open quantity
in the head-ceiling stack. Here we build the integrated candidate-verify step on the
REAL Marlin-int4-quantized 262k head and time every piece:

  (a) int4 Marlin GEMV (the nominator): x[M,2560] @ W_int4[2560,262144] -> cand_logits.
      Reads the packed int4 weights + bf16 group scales = 0.346 GB (== #549 int4_g128).
      Run through the PRODUCTION forward `apply_gptq_marlin_linear` (exact served
      kernel config: atomic-add decision + fp32 reduce default).
  (b) top-K=8 fp-verify: topk(8) over cand_logits -> gather the K bf16 head rows
      (M*K*2560*2 B; 0.33 MB @ M=8,K=8) -> exact fp recompute of the K logits.
  (c) re-argmax over the K verify logits -> final token (mapped back to vocab id).

cv-step = (a)+(b)+(c). The bf16 baseline step it REPLACES = full bf16 head GEMV +
argmax over 262k. We sweep M in {1,2,4,8}: M=1 is #560's literal headline; M=8 is the
ACTUAL served spec-verify width (MTP K=7 -> verify 8; #549 m_hist={8:15933}), which is
what drives the served-TPS projection and matches #549's in-context head time 2.8382 ms.

KEY OUTPUTS (Stage 1):
  cv_int4_nominator_bw_GBs, cv_int4_nominator_pct_of_peak  (achieved int4 GEMV BW)
  cv_step_latency_ms          (full a+b+c integrated step)
  bf16_head_argmax_step_latency_ms (the full-bf16-head step candidate-verify replaces)
  cv_step_speedup             (= bf16-step / cv-step)

LOCAL only: analysis_only, official_tps=0, no HF Job, no --launch, no submission.
Run under the SERVER venv (vLLM 0.22.1rc1 Marlin kernels):
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/<hash>/bin/python \
      research/candidate_verify_realize/stage1_cv_microbench.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops  # noqa: F401  (kept for parity / debug)
from vllm.scalar_type import scalar_types
import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt

HERE = Path(__file__).resolve().parent
MODEL_DIR = (
    "/senpai-run/home/student-fern/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
DUMP_PATH = "/tmp/fullhead_hidden_fern.pt"
HIDDEN = 2560
VOCAB = 262144
HEAD_KEY = "lm_head.weight"
GROUP = 128
K_SAFE = 8
DTYPE = torch.bfloat16
A10G_HBM_GBPS = 600.0  # A10G HBM2 peak (read-bound floor context; same as #549/#550)

# --- served anchors (cited, NOT re-derived; #549/#544/#553) ---
WALL_PER_STEP_MS_SERVED = 14.871    # #549 Pass-A measured per-verify-step wall
TPS_WARM_MEDIAN_M1 = 264.8195659618872  # #549 Pass-A warm-median (the wall basis)
ANCHOR_BASE_FULLHEAD_252 = 252.31   # lawine #544 derived / wirbel #553 grounding
ANCHOR_BASE_FULLHEAD_254 = 253.78   # fern #535 fast-stack proxy
HEAD_GEMM_MS_INCONTEXT_M8 = 2.8382  # #549 in-context served head time (M=8 dominated)
# #549 projected gain band (read-bound roofline) we are replacing with a measurement:
PROJ_549_GAIN_CENTRAL = 40.10024690129677
PROJ_549_GAIN_BAND = (28.251525006837994, 43.68323839224104)  # int4_g128 [pess,opt]
PROJ_549_REALIZED_TPS_CENTRAL_254 = 292.2  # 253.78 * wall_speedup central


def _load_head(dev: str) -> torch.Tensor:
    path = Path(MODEL_DIR) / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        W_head = f.get_tensor(HEAD_KEY)  # [VOCAB, HIDDEN], logits = x @ W_head.T
    assert tuple(W_head.shape) == (VOCAB, HIDDEN), W_head.shape
    # Marlin wants w=[K=HIDDEN, N=VOCAB]; logits = x[M,K] @ w[K,N].
    return W_head.t().contiguous().to(device=dev, dtype=DTYPE)


def _time_region(fn, *, warmup: int, reps: int, iters: int) -> dict[str, float]:
    """Time `fn` (a no-arg closure). Returns ms stats over `reps` samples, each the
    mean wall of `iters` back-to-back launches (low-variance rep means -> tight CI)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(iters):
            fn()
        ev1.record()
        torch.cuda.synchronize()
        samples.append(ev0.elapsed_time(ev1) / iters)
    samples.sort()
    n = len(samples)
    mean = statistics.fmean(samples)
    std = statistics.pstdev(samples)
    return {
        "ms_mean": mean,
        "ms_std": std,
        "ms_min": samples[0],
        "ms_p10": samples[max(0, int(0.10 * n) - 1)],
        "ms_p50": samples[n // 2],
        "ms_p90": samples[min(n - 1, int(0.90 * n))],
        "reps": n,
        "iters_per_rep": iters,
        # std of the rep-mean -> 95% CI half-width on the mean
        "ci95_halfwidth_ms": 1.96 * std / (n ** 0.5),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--out-file", default=str(HERE / "stage1_cv_microbench.json"))
    args = ap.parse_args(argv)

    dev = "cuda"
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    name = torch.cuda.get_device_name(0)
    print(f"[s1] device={name} torch={torch.__version__} cap={torch.cuda.get_device_capability()}", flush=True)

    # --- build the int4-Marlin nominator from the REAL bf16 head ---
    w = _load_head(dev)  # [HIDDEN, VOCAB] bf16 (full precision)
    t0 = time.time()
    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _rand = mt.marlin_quantize(
        w, scalar_types.uint4b8, GROUP, act_order=False
    )
    torch.cuda.synchronize()
    int4_weight_bytes = (marlin_q_w.numel() * marlin_q_w.element_size()
                         + marlin_s.numel() * marlin_s.element_size())
    bf16_head_bytes = w.numel() * w.element_size()
    print(f"[s1] marlin_quantize {time.time()-t0:.1f}s | int4_read={int4_weight_bytes/1e9:.4f}GB "
          f"(q {marlin_q_w.numel()*marlin_q_w.element_size()/1e6:.1f}MB + s "
          f"{marlin_s.numel()*marlin_s.element_size()/1e6:.1f}MB) | bf16_head={bf16_head_bytes/1e9:.4f}GB", flush=True)

    zp = mu.marlin_make_empty_g_idx(dev)
    workspace = mu.marlin_make_workspace_new(torch.device(dev))

    # full-precision head rows [VOCAB, HIDDEN] for the bf16 verify gather (row v = W_head[v])
    W_head_rows = w.t().contiguous()  # [VOCAB, HIDDEN] bf16

    # realistic decode hidden states from the #549 dump (kernel time is data-independent,
    # but use real H so topk/argmax are realistic)
    if Path(DUMP_PATH).exists():
        blob = torch.load(DUMP_PATH, map_location="cpu")
        H_all = (blob["hidden"] if isinstance(blob, dict) else blob).to(torch.bfloat16)
        print(f"[s1] hidden dump rows={H_all.shape[0]} dim={H_all.shape[1]}", flush=True)
    else:
        H_all = torch.randn(4096, HIDDEN, dtype=DTYPE)
        print("[s1] WARN: no dump; using random hidden states", flush=True)

    def int4_gemv(x):
        return mu.apply_gptq_marlin_linear(
            input=x, weight=marlin_q_w, weight_scale=marlin_s, weight_zp=zp,
            g_idx=g_idx, g_idx_sort_indices=sort_indices, workspace=workspace,
            wtype=scalar_types.uint4b8, output_size_per_partition=VOCAB,
            input_size_per_partition=HIDDEN, is_k_full=True,
        )

    def bf16_gemv(x):  # full bf16 head GEMV alone (the in-context head GEMM analog)
        return x @ w

    def bf16_step(x):  # full bf16 head GEMV + argmax over 262k (the replaced step)
        logits = x @ w
        return logits.argmax(dim=1)

    def verify_tail(cand, x):  # (b)+(c): topk-K gather + fp recompute + re-argmax
        topk = cand.topk(K_SAFE, dim=1).indices  # [M, K]
        rows = W_head_rows[topk]                 # [M, K, HIDDEN] bf16 gather (0.33MB @ M8K8)
        verify = torch.einsum("mh,mkh->mk", x, rows).float()  # exact fp recompute
        local = verify.argmax(dim=1)             # [M]
        return torch.gather(topk, 1, local[:, None]).squeeze(1)  # final vocab id [M]

    def cv_step(x):  # (a) int4 GEMV (b) topk-K gather + fp recompute (c) re-argmax
        return verify_tail(int4_gemv(x), x)

    results: dict[str, Any] = {}
    for M in args.Ms:
        x = H_all[:M].to(dev).contiguous()
        if x.shape[0] < M:  # pad if dump shorter than M (won't happen)
            x = x.repeat((M + x.shape[0] - 1) // x.shape[0], 1)[:M]

        # correctness: int4-GEMV argmax-shortlist must recover the bf16-head argmax
        bf16_tok = bf16_step(x)
        cv_tok = cv_step(x)
        identity_match = bool((bf16_tok == cv_tok).all().item())

        int4 = _time_region(lambda: int4_gemv(x), warmup=args.warmup, reps=args.reps, iters=args.iters)
        cvs = _time_region(lambda: cv_step(x), warmup=args.warmup, reps=args.reps, iters=args.iters)
        bf16 = _time_region(lambda: bf16_step(x), warmup=args.warmup, reps=args.reps, iters=args.iters)
        # GEMM-to-GEMM (no argmax) + isolated verify tail, for a transparent breakdown
        bf16g = _time_region(lambda: bf16_gemv(x), warmup=args.warmup, reps=args.reps, iters=args.iters)
        cand_fixed = int4_gemv(x)  # frozen candidate logits -> time the verify tail alone
        vtail = _time_region(lambda: verify_tail(cand_fixed, x), warmup=args.warmup, reps=args.reps, iters=args.iters)

        int4_bw = int4_weight_bytes / (int4["ms_mean"] / 1e3) / 1e9
        int4_bw_lo = int4_weight_bytes / ((int4["ms_mean"] + int4["ci95_halfwidth_ms"]) / 1e3) / 1e9
        int4_bw_hi = int4_weight_bytes / ((int4["ms_mean"] - int4["ci95_halfwidth_ms"]) / 1e3) / 1e9
        bf16_bw = bf16_head_bytes / (bf16["ms_mean"] / 1e3) / 1e9  # includes argmax; ~ #549's 479-483
        speedup = bf16["ms_mean"] / cvs["ms_mean"]

        bf16_gemv_bw = bf16_head_bytes / (bf16g["ms_mean"] / 1e3) / 1e9  # GEMM-only achieved BW
        results[str(M)] = {
            "M": M,
            "int4_gemv": int4,
            "cv_step": cvs,
            "bf16_step": bf16,
            "bf16_gemv_alone": bf16g,
            "verify_tail_alone": vtail,
            "cv_int4_nominator_bw_GBs": int4_bw,
            "cv_int4_nominator_bw_GBs_ci95": [int4_bw_lo, int4_bw_hi],
            "cv_int4_nominator_pct_of_peak": int4_bw / A10G_HBM_GBPS,
            "bf16_step_achieved_bw_GBs": bf16_bw,
            "bf16_gemv_alone_bw_GBs": bf16_gemv_bw,
            "bf16_gemv_alone_pct_of_peak": bf16_gemv_bw / A10G_HBM_GBPS,
            "cv_step_latency_ms": cvs["ms_mean"],
            "bf16_head_argmax_step_latency_ms": bf16["ms_mean"],
            "bf16_gemv_alone_latency_ms": bf16g["ms_mean"],
            "int4_gemv_latency_ms": int4["ms_mean"],
            "verify_tail_latency_ms": vtail["ms_mean"],
            "cv_step_speedup": speedup,
            "gemv_only_speedup": bf16g["ms_mean"] / int4["ms_mean"],
            "identity_match_on_M_rows": identity_match,
        }
        print(f"[s1] M={M}: int4_gemv={int4['ms_mean']:.4f}ms ({int4_bw:.1f} GB/s, "
              f"{100*int4_bw/A10G_HBM_GBPS:.1f}% peak) | cv_step={cvs['ms_mean']:.4f}ms | "
              f"bf16_step={bf16['ms_mean']:.4f}ms | speedup={speedup:.3f}x | id_match={identity_match}", flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    report = {
        "pr": 560, "stage": 1, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "device": name, "hidden": HIDDEN, "vocab": VOCAB, "group_size": GROUP, "K_safe": K_SAFE,
        "int4_weight_read_bytes": int4_weight_bytes,
        "int4_weight_read_gb": int4_weight_bytes / 1e9,
        "bf16_head_bytes": bf16_head_bytes,
        "a10g_hbm_peak_gbps": A10G_HBM_GBPS,
        "Ms": args.Ms, "reps": args.reps, "iters": args.iters, "warmup": args.warmup,
        "per_M": results,
        "peak_gpu_gb": peak_gb,
        "anchors": {
            "wall_per_step_ms_served": WALL_PER_STEP_MS_SERVED,
            "tps_warm_median_m1": TPS_WARM_MEDIAN_M1,
            "anchor_base_fullhead_252": ANCHOR_BASE_FULLHEAD_252,
            "anchor_base_fullhead_254": ANCHOR_BASE_FULLHEAD_254,
            "head_gemm_ms_incontext_m8": HEAD_GEMM_MS_INCONTEXT_M8,
            "proj_549_gain_central": PROJ_549_GAIN_CENTRAL,
            "proj_549_gain_band": list(PROJ_549_GAIN_BAND),
        },
        "model_dir": MODEL_DIR,
    }
    Path(args.out_file).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"[s1] peak_gpu={peak_gb:.2f}GB report -> {args.out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
