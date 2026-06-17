#!/usr/bin/env python3
"""PR #560 fern — Stage 3 (HARD gate): re-confirm K_safe=8 candidate-verify gives
argmax_identity_rate == 1.0 on the REALIZED Marlin-int4 path over the 60k held-out
positions.

#549 proved K_safe=8 offline, but with a SIMULATED int4 (quantize->dequantize to bf16,
then a bf16 matmul). The realized path runs the ACTUAL Marlin uint4b8 GEMM kernel; its
rounding/accumulation can differ marginally from the simulation. This stage runs the
true end-to-end realized step on every held-out position:

  cv_token(n) = argmax_{v in top-8(MarlinInt4 @ H[n])} ( H[n] . W_bf16[v] )   # exact fp verify
  gold(n)     = argmax_v ( H[n] . W_bf16[v] )                                 # served greedy

  argmax_identity_rate = mean_n [ cv_token(n) == gold(n) ]

Gold is the served bf16-head greedy token (bf16-output argmax == the served token, since
the final softcap+scale are monotonic). We ALSO report identity vs the fp32-exact gold and
the conservative union, matching #549. The realized head-speed gain counts ONLY at
argmax_identity_rate == 1.0.

Analysis-only; no server, no HF fire. Run under the SERVER venv:
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/<hash>/bin/python \
      research/candidate_verify_realize/stage3_realized_identity.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

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
DTYPE = torch.bfloat16
KS = [1, 2, 4, 6, 8, 12, 16, 32]
K_SAFE = 8
MIN_ROWS = 50_000


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out-file", default=str(HERE / "stage3_realized_identity.json"))
    args = ap.parse_args(argv)

    dev = "cuda"
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    print(f"[s3] device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    # full bf16 head (verify + gold reference)
    path = Path(MODEL_DIR) / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        W_head = f.get_tensor(HEAD_KEY)  # [VOCAB, HIDDEN]
    assert tuple(W_head.shape) == (VOCAB, HIDDEN)
    w = W_head.t().contiguous().to(device=dev, dtype=DTYPE)  # [HIDDEN, VOCAB] bf16
    W_rows = W_head.to(device=dev, dtype=DTYPE)              # [VOCAB, HIDDEN] for gather
    del W_head

    # realized Marlin int4 nominator
    t0 = time.time()
    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = mt.marlin_quantize(
        w, scalar_types.uint4b8, GROUP, act_order=False
    )
    torch.cuda.synchronize()
    print(f"[s3] marlin_quantize {time.time()-t0:.1f}s", flush=True)
    zp = mu.marlin_make_empty_g_idx(dev)
    workspace = mu.marlin_make_workspace_new(torch.device(dev))

    def int4_gemv(x):
        return mu.apply_gptq_marlin_linear(
            input=x, weight=marlin_q_w, weight_scale=marlin_s, weight_zp=zp,
            g_idx=g_idx, g_idx_sort_indices=sort_indices, workspace=workspace,
            wtype=scalar_types.uint4b8, output_size_per_partition=VOCAB,
            input_size_per_partition=HIDDEN, is_k_full=True,
        )

    blob = torch.load(DUMP_PATH, map_location="cpu")
    H_all = (blob["hidden"] if isinstance(blob, dict) else blob).to(torch.bfloat16)
    if args.max_rows:
        H_all = H_all[: args.max_rows]
    n_rows = H_all.shape[0]
    print(f"[s3] hidden rows={n_rows} (>= {MIN_ROWS}: {n_rows >= MIN_ROWS})", flush=True)

    maxK = max(KS)
    miss_fp32 = {K: 0 for K in KS}   # gold(fp32) not in int4 top-K
    miss_bf16 = {K: 0 for K in KS}   # gold(bf16-served) not in int4 top-K
    cv_id_match = 0                  # realized cv_token == served bf16 gold (K_safe=8)
    cv_id_match_fp32 = 0             # realized cv_token == fp32 gold
    gold_tie = 0
    N = 0
    B = args.chunk
    for i in range(0, n_rows, B):
        Hb = H_all[i : i + B].to(dev)
        # gold (served greedy): bf16-output argmax == served token; fp32 for robustness
        Lf = Hb.float() @ w.float()             # [b, V] fp32
        gold_fp32 = Lf.argmax(dim=1)
        gold_bf16 = (Hb @ w).argmax(dim=1)       # bf16-output argmax (the served token)
        gold_tie += int((gold_fp32 != gold_bf16).sum().item())
        # realized candidate ranking from the ACTUAL Marlin int4 kernel
        cand = int4_gemv(Hb).float()             # [b, V]
        topk = cand.topk(maxK, dim=1).indices    # [b, maxK]
        eq32 = topk == gold_fp32[:, None]
        eqbf = topk == gold_bf16[:, None]
        for K in KS:
            miss_fp32[K] += int((~eq32[:, :K].any(dim=1)).sum().item())
            miss_bf16[K] += int((~eqbf[:, :K].any(dim=1)).sum().item())
        # realized end-to-end cv_token at K_safe=8 (exact bf16 verify over the shortlist)
        short = topk[:, :K_SAFE]                  # [b, K]
        rows = W_rows[short]                      # [b, K, HIDDEN]
        verify = torch.einsum("bh,bkh->bk", Hb, rows).float()  # exact fp recompute
        local = verify.argmax(dim=1)
        cv_tok = torch.gather(short, 1, local[:, None]).squeeze(1)
        cv_id_match += int((cv_tok == gold_bf16).sum().item())
        cv_id_match_fp32 += int((cv_tok == gold_fp32).sum().item())
        N += Hb.shape[0]
        del Hb, Lf, cand, topk, rows, verify
        if (i // B) % 20 == 0:
            torch.cuda.empty_cache()
            print(f"[s3] {N}/{n_rows} | miss@8(bf16)={miss_bf16[8]} miss@8(fp32)={miss_fp32[8]} "
                  f"cv_id_mismatch={N - cv_id_match}", flush=True)

    rate_fp32 = {K: miss_fp32[K] / N for K in KS}
    rate_bf16 = {K: miss_bf16[K] / N for K in KS}
    ksafe_fp32 = next((K for K in KS if miss_fp32[K] == 0), None)
    ksafe_bf16 = next((K for K in KS if miss_bf16[K] == 0), None)
    ksafe_cons = next((K for K in KS if miss_fp32[K] == 0 and miss_bf16[K] == 0), None)
    argmax_identity_rate = cv_id_match / N            # vs served bf16 gold (PRIMARY)
    argmax_identity_rate_fp32 = cv_id_match_fp32 / N

    report = {
        "pr": 560, "stage": 3, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "realized_path": "marlin_uint4b8_g128_kernel",
        "n_positions": N, "n_rows_required": MIN_ROWS, "rows_ok": N >= MIN_ROWS,
        "vocab": VOCAB, "hidden": HIDDEN, "group_size": GROUP, "K_safe_tested": K_SAFE,
        "Ks": KS,
        "miss_rate_by_K_fp32gold": rate_fp32,
        "miss_rate_by_K_bf16gold": rate_bf16,
        "K_safe_fp32": ksafe_fp32, "K_safe_bf16": ksafe_bf16, "K_safe_conservative": ksafe_cons,
        "gold_fp32_vs_bf16_disagreements": gold_tie,
        "gold_tie_frac": gold_tie / N,
        "argmax_identity_rate": argmax_identity_rate,
        "argmax_identity_rate_fp32gold": argmax_identity_rate_fp32,
        "cv_mismatch_count_bf16gold": N - cv_id_match,
        "cv_mismatch_count_fp32gold": N - cv_id_match_fp32,
        "identity_hard_gate_pass": bool(argmax_identity_rate == 1.0 and ksafe_bf16 is not None and ksafe_bf16 <= K_SAFE),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "model_dir": MODEL_DIR, "dump_path": DUMP_PATH,
    }
    Path(args.out_file).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("\n" + "=" * 12 + " PR #560 STAGE 3 — REALIZED-PATH IDENTITY (HARD GATE) " + "=" * 12, flush=True)
    print(f"  positions            = {N} (>= {MIN_ROWS}: {report['rows_ok']})", flush=True)
    print(f"  miss@8 (bf16 gold)   = {rate_bf16[8]:.3e}  miss@8 (fp32 gold) = {rate_fp32[8]:.3e}", flush=True)
    print(f"  K_safe (bf16/fp32/cons) = {ksafe_bf16}/{ksafe_fp32}/{ksafe_cons}", flush=True)
    print(f"  argmax_identity_rate = {argmax_identity_rate:.6f}  (fp32 gold {argmax_identity_rate_fp32:.6f})", flush=True)
    print(f"  >>> HARD GATE PASS   = {report['identity_hard_gate_pass']}", flush=True)
    print(f"  peak_gpu={report['peak_gpu_gb']:.2f}GB -> {args.out_file}", flush=True)
    return 0 if report["identity_hard_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
