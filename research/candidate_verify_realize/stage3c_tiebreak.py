#!/usr/bin/env python3
"""PR #560 Stage 3c — the realized verify recovers the served token at 1.0 IFF it
breaks ties the way the server's argmax does. The server greedy token gold_bf16 =
argmax over the full 262k bf16 logits = the LOWEST vocab index among the bf16-max set
(torch.argmax returns first index). A shortlist re-rank that tie-breaks by shortlist
POSITION can pick a different (equal-logit) token -> spurious identity miss.

Fix: among the K-shortlist, pick the max verify logit, breaking ties by LOWEST VOCAB
INDEX (server convention). Since gold_bf16 is the global lowest-index max and it is in
the shortlist (miss@8=0), this recovers gold_bf16 exactly.

Verify variants x tie-break:
  posTB  = argmax (tie-break by shortlist position)   [the naive, wrong, convention]
  vocTB  = lowest vocab index among the verify-max     [server-matched convention]
recompute precision: bf16 bmm (B, cheap realized) and gather-full-bf16 (C, exact upper bound).
Report identity vs served gold_bf16 over 60k.  Analysis-only."""
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


def pick_postb(short, vlog):  # tie-break by shortlist position (naive)
    return short.gather(1, vlog.argmax(1, keepdim=True)).squeeze(1)


def pick_voctb(short, vlog):  # tie-break by lowest vocab index (server-matched)
    vmax = vlog.max(1, keepdim=True).values
    is_max = vlog >= vmax                       # [b,K] (>= guards fp equality)
    masked = torch.where(is_max, short, torch.full_like(short, VOCAB))
    return masked.min(1).values


def main() -> int:
    dev = "cuda"
    torch.cuda.init()
    with safe_open(str(Path(MODEL_DIR) / "model.safetensors"), framework="pt", device="cpu") as f:
        W_head = f.get_tensor("lm_head.weight")
    w = W_head.t().contiguous().to(dev, DT)
    W_rows = W_head.to(dev, DT)
    del W_head
    t0 = time.time()
    _wref, q, s, gidx, sidx, _ = mt.marlin_quantize(w, scalar_types.uint4b8, GROUP, act_order=False)
    torch.cuda.synchronize()
    print(f"[s3c] quantize {time.time()-t0:.1f}s", flush=True)
    zp = mu.marlin_make_empty_g_idx(dev)
    ws = mu.marlin_make_workspace_new(torch.device(dev))

    def int4(x):
        return mu.apply_gptq_marlin_linear(input=x, weight=q, weight_scale=s, weight_zp=zp,
            g_idx=gidx, g_idx_sort_indices=sidx, workspace=ws, wtype=scalar_types.uint4b8,
            output_size_per_partition=VOCAB, input_size_per_partition=HIDDEN, is_k_full=True)

    blob = torch.load(DUMP_PATH, map_location="cpu")
    H_all = (blob["hidden"] if isinstance(blob, dict) else blob).to(DT)
    n = H_all.shape[0]
    print(f"[s3c] rows={n}", flush=True)

    keys = ["B_posTB", "B_vocTB", "C_posTB", "C_vocTB"]
    m = {k: 0 for k in keys}
    contain = 0
    N, Bsz = 0, 512
    for i in range(0, n, Bsz):
        Hb = H_all[i:i+Bsz].to(dev)
        Lbf = Hb @ w
        gold = Lbf.argmax(1)                         # served token
        short = int4(Hb).topk(K, 1).indices          # int4-Marlin shortlist
        contain += int((short == gold[:, None]).any(1).sum())
        rows = W_rows[short]
        vB = torch.bmm(Hb.unsqueeze(1), rows.transpose(1, 2)).squeeze(1).float()  # cheap bf16 recompute
        vC = Lbf.gather(1, short).float()            # exact served logits
        m["B_posTB"] += int((pick_postb(short, vB) == gold).sum())
        m["B_vocTB"] += int((pick_voctb(short, vB) == gold).sum())
        m["C_posTB"] += int((pick_postb(short, vC) == gold).sum())
        m["C_vocTB"] += int((pick_voctb(short, vC) == gold).sum())
        N += Hb.shape[0]
        del Hb, Lbf, rows, vB, vC
    rep = {"pr": 560, "stage": "3c", "n_positions": N,
           "containment_rate_at_K8": contain / N,
           "identity_vs_served_bf16": {k: m[k] / N for k in keys},
           "mismatch_vs_served_bf16": {k: N - m[k] for k in keys},
           "note": "B=cheap bf16 bmm recompute; C=exact served-logit gather (upper bound). "
                   "posTB=shortlist-position tie-break; vocTB=lowest-vocab-index tie-break (server-matched)."}
    (HERE / "stage3c_tiebreak.json").write_text(json.dumps(rep, indent=2))
    print(f"\n=== tie-break fix (identity vs served bf16, {N} pos; containment={contain/N:.6f}) ===", flush=True)
    for k in keys:
        print(f"  {k:9s}: {m[k]/N:.6f}  (mismatch {N-m[k]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
