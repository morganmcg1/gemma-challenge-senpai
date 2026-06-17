#!/usr/bin/env python3
"""PR #566 fern — cheap OFFLINE validation of the EXACT scatter+native-argmax identity
path my live probe (cv_head_probe._cv_logits) relies on, run on the #560 dumped decode
hidden states. Mirrors the probe step-for-step: int4-Marlin nominator GEMV -> top-8 ->
gather bf16 rows -> bf16 verify -> scatter into [M,vocab] (rest=-inf) -> argmax. Compares
that token to the full bf16-head argmax (the served oracle). Expected identity 1.0
(reproduces #560 stage3c B_vocTB=1.0 with THIS code path). No server, ~1 min.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import torch
from safetensors import safe_open
from vllm.scalar_type import scalar_types
import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
import vllm.model_executor.layers.quantization.utils.marlin_utils_test as mt

MODEL_DIR = ("/senpai-run/home/student-fern/.cache/huggingface/hub/"
             "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0")
DUMP = "/tmp/fullhead_hidden_fern.pt"
HIDDEN, VOCAB, GROUP, KSAFE = 2560, 262144, 128, 8
dev = "cuda"

W = None
with safe_open(str(Path(MODEL_DIR) / "model.safetensors"), framework="pt", device="cpu") as f:
    W = f.get_tensor("lm_head.weight")  # [VOCAB, HIDDEN] bf16
W = W.to(device=dev, dtype=torch.bfloat16)
rows = W.contiguous()                    # [VOCAB, HIDDEN] verify gather
w_marlin = W.t().contiguous()            # [HIDDEN, VOCAB] for marlin
t0 = time.time()
_wref, q_w, s, g_idx, sort_idx, _ = mt.marlin_quantize(w_marlin, scalar_types.uint4b8, GROUP, act_order=False)
zp = mu.marlin_make_empty_g_idx(dev)
ws = mu.marlin_make_workspace_new(torch.device(dev))
torch.cuda.synchronize()
print(f"[off] quantized head in {time.time()-t0:.1f}s int4={q_w.numel()*q_w.element_size()/1e9:.4f}GB")

H = torch.load(DUMP, map_location="cpu")
H = (H["hidden"] if isinstance(H, dict) else H).to(torch.bfloat16)
N = H.shape[0]
print(f"[off] dump rows={N}")

NEG = torch.finfo(torch.float32).min
total = mism = ties = 0
bs = 256
for i in range(0, N, bs):
    x = H[i:i+bs].to(dev)
    # int4 nominator -> top-8 -> bf16 verify -> scatter -> argmax (MIRRORS the live probe)
    cand = mu.apply_gptq_marlin_linear(input=x, weight=q_w, weight_scale=s, weight_zp=zp,
        g_idx=g_idx, g_idx_sort_indices=sort_idx, workspace=ws, wtype=scalar_types.uint4b8,
        output_size_per_partition=VOCAB, input_size_per_partition=HIDDEN, is_k_full=True)
    topk = cand.topk(KSAFE, dim=1).indices
    grows = rows[topk]
    verify = torch.einsum("mh,mkh->mk", x.float(), grows.float()).to(torch.bfloat16)
    out = torch.full_like(cand, NEG, dtype=torch.float32)
    out.scatter_(1, topk, verify.float())
    cv_tok = out.argmax(dim=1)
    # oracle: full bf16 head logits, argmax (the served reference)
    ref = (x @ w_marlin)                # [M, VOCAB] bf16
    ref_tok = ref.float().argmax(dim=1)
    mism += int((cv_tok != ref_tok).sum().item())
    rmax = ref.float().max(dim=1, keepdim=True).values
    ties += int((ref.float() == rmax).sum(dim=1).gt(1).sum().item())
    total += x.shape[0]

print(json.dumps({"pr": 566, "stage": "offline_scatter_check", "n": total,
    "mismatch": mism, "identity_rate": (total - mism) / total, "bf16_top1_ties": ties,
    "k_safe": KSAFE}, indent=2))
