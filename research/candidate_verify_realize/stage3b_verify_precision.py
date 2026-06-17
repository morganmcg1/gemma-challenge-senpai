#!/usr/bin/env python3
"""PR #560 Stage 3b — which verify PRECISION recovers the served bf16-head greedy
token exactly? The served token = argmax(bf16 head GEMM) (vLLM sampler upcasts bf16
logits to fp32 then argmax; softcap monotonic). The candidate shortlist is perfect
(miss@8=0), so identity is bounded only by how the verify re-ranks the K=8 shortlist.

Test 4 verify recomputes over the int4-Marlin top-8, on all 60k positions:
  A  fp32-input dot   : exact fp32 logits  -> recovers the fp32-IDEAL token
  B  bf16 bmm         : bf16 GEMM recompute (fp32 accum, bf16 round) -> targets served
  C  gather full bf16 : pick from the served (Hb@w) logits directly  -> upper bound (=1.0)
  D  fp32 round->bf16 : fp32 dot then round to bf16 -> emulates served per-col rounding

Report identity vs gold_bf16 (the served token) and vs gold_fp32 (the ideal) for each.
Analysis-only, no fire."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.scalar_type import scalar_types
import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt

HERE = Path(__file__).resolve().parent
MODEL_DIR = ("/senpai-run/home/student-fern/.cache/huggingface/hub/"
             "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
             "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0")
DUMP_PATH = "/tmp/fullhead_hidden_fern.pt"
HIDDEN, VOCAB, GROUP, K = 2560, 262144, 128, 8
DT = torch.bfloat16


def main() -> int:
    dev = "cuda"
    torch.cuda.init()
    with safe_open(str(Path(MODEL_DIR) / "model.safetensors"), framework="pt", device="cpu") as f:
        W_head = f.get_tensor("lm_head.weight")  # [V,H]
    w = W_head.t().contiguous().to(dev, DT)       # [H,V]
    W_rows = W_head.to(dev, DT)                    # [V,H]
    del W_head
    t0 = time.time()
    _wref, q, s, gidx, sidx, _ = mt.marlin_quantize(w, scalar_types.uint4b8, GROUP, act_order=False)
    torch.cuda.synchronize()
    print(f"[s3b] quantize {time.time()-t0:.1f}s", flush=True)
    zp = mu.marlin_make_empty_g_idx(dev)
    ws = mu.marlin_make_workspace_new(torch.device(dev))

    def int4(x):
        return mu.apply_gptq_marlin_linear(input=x, weight=q, weight_scale=s, weight_zp=zp,
            g_idx=gidx, g_idx_sort_indices=sidx, workspace=ws, wtype=scalar_types.uint4b8,
            output_size_per_partition=VOCAB, input_size_per_partition=HIDDEN, is_k_full=True)

    blob = torch.load(DUMP_PATH, map_location="cpu")
    H_all = (blob["hidden"] if isinstance(blob, dict) else blob).to(DT)
    n = H_all.shape[0]
    print(f"[s3b] rows={n}", flush=True)

    match = {v: {"bf16": 0, "fp32": 0} for v in "ABCD"}
    N, B = 0, 512
    for i in range(0, n, B):
        Hb = H_all[i:i+B].to(dev)
        Lbf = Hb @ w                      # served bf16 logits [b,V]
        gold_bf16 = Lbf.argmax(1)
        gold_fp32 = (Hb.float() @ w.float()).argmax(1)
        short = int4(Hb).topk(K, 1).indices          # [b,K] int4-Marlin shortlist
        rows = W_rows[short]                          # [b,K,H] bf16
        Hf = Hb.float()
        # A fp32-input dot
        tA = short.gather(1, (torch.einsum("bh,bkh->bk", Hf, rows.float())).argmax(1, keepdim=True)).squeeze(1)
        # B bf16 bmm (fp32 accum, bf16 out)
        vB = torch.bmm(Hb.unsqueeze(1), rows.transpose(1, 2)).squeeze(1)  # [b,K] bf16
        tB = short.gather(1, vB.float().argmax(1, keepdim=True)).squeeze(1)
        # C gather from the served full bf16 logits
        vC = Lbf.gather(1, short)                     # [b,K] bf16
        tC = short.gather(1, vC.float().argmax(1, keepdim=True)).squeeze(1)
        # D fp32 dot rounded to bf16
        vD = torch.einsum("bh,bkh->bk", Hf, rows.float()).to(DT)
        tD = short.gather(1, vD.float().argmax(1, keepdim=True)).squeeze(1)
        for tag, tok in (("A", tA), ("B", tB), ("C", tC), ("D", tD)):
            match[tag]["bf16"] += int((tok == gold_bf16).sum())
            match[tag]["fp32"] += int((tok == gold_fp32).sum())
        N += Hb.shape[0]
        del Hb, Lbf, rows
    rep = {"pr": 560, "stage": "3b", "n_positions": N,
           "identity_vs_served_bf16": {v: match[v]["bf16"] / N for v in "ABCD"},
           "identity_vs_fp32_ideal": {v: match[v]["fp32"] / N for v in "ABCD"},
           "mismatch_vs_served_bf16": {v: N - match[v]["bf16"] for v in "ABCD"},
           "verify_labels": {"A": "fp32_input_dot", "B": "bf16_bmm", "C": "gather_full_bf16", "D": "fp32_round_bf16"}}
    (HERE / "stage3b_verify_precision.json").write_text(json.dumps(rep, indent=2))
    print("\n=== verify precision sweep (identity over %d positions) ===" % N, flush=True)
    for v in "ABCD":
        print(f"  {v} {rep['verify_labels'][v]:18s}: vs_served_bf16={match[v]['bf16']/N:.6f} "
              f"(mismatch {N-match[v]['bf16']})  vs_fp32_ideal={match[v]['fp32']/N:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
