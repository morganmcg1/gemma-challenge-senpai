# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Train an EAGLE-3 draft head for google/gemma-4-E4B-it (PR #16, Step 3).

A faithful plain-PyTorch reimplementation of vLLM 0.22.0's `Eagle3LlamaForCausalLM`
(`vllm/model_executor/models/llama_eagle3.py`). The vLLM head is inference-only
(paged Attention, no autograd), so we reimplement the identical architecture with
trainable Linear/Attention so the checkpoint is vLLM-loadable later (deployment
gated on kanna #5). See research/eagle3_drafter/arch_notes.md.

Architecture (per arch_notes S1-S3):
  fused = cat(aux[2], aux[21], aux[39])            # [T, 7680]
  h0    = fc(input_norm(fused))                    # [T, 2560]   (norm_before_fc)
  e     = embed_tokens(input_ids)                  # [T, 2560]
  layer-0 (EAGLE twist): qkv input = cat(input_layernorm(e), hidden_norm(h0)) -> 2*H
  out   = norm( mlp(post_attention_layernorm(attn + h0)) + (attn + h0) )
  logits = lm_head(out)                            # [T, 262144]

Loss: hard cross-entropy of head logits vs next_token_ids (PR spec).
Init: draft embed_tokens + lm_head copied from the target's tied embedding table
(frozen by default); only fc + the 1 decoder layer + norms are trained.

Alignment (arch_notes S5): default --feature_shift 1 (vLLM-faithful) pairs target
feature h_{j-1} with embed(x_j) to predict x_{j+1}, matching serving.

HARD BOUNDS: this script self-enforces SENPAI_MAX_EPOCHS and SENPAI_TIMEOUT_MINUTES
(operator caps). The effective step count is auto-capped to the epoch bound and the
wall-clock bound; the binding constraint is logged loudly at startup.

Run (from target/), on the enlarged corpus (advisor option (c): 1000 steps == 2
epochs over the full ~6,734-sample MATH-train split, capped by SENPAI_MAX_EPOCHS=2):
  HF_HOME=/senpai-run/home/student-fern/.cache/huggingface \
  python scripts/drafter/train_eagle3.py \
      --corpus research/eagle3_drafter/train_data/debug_1k_corpus.pt \
      --eval_corpus research/eagle3_drafter/train_data/debug_1k_eval_corpus.pt \
      --output research/eagle3_drafter/checkpoints/debug_1k_2ep/ \
      --steps 1000 --lr 1e-4 --batch_tokens 4096 --warmup 100 --eval_every 50 \
      --wandb_group eagle3-drafter-training --wandb_name fern/eagle3-debug-1k-2ep
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100
HID = 2560
VOCAB = 262144
N_AUX = 3
HEAD_DIM = 256
N_HEADS = 8
N_KV = 2
INTER = 10240
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Model (faithful reimplementation of vLLM Eagle3LlamaForCausalLM)
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Llama-style RMSNorm (weight * x_normed; weight init ones). Matches vLLM's
    generic RMSNorm so the checkpoint is loadable."""

    def __init__(self, dim: int, eps: float = EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(dt)


def build_rope(T: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    pos = torch.arange(T, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class Attention(nn.Module):
    """GQA self-attention (Llama-style): 8 q heads, 2 kv heads, head_dim 256,
    o_proj back to hidden. First EAGLE layer feeds a 2*hidden qkv input."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(in_dim, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(in_dim, N_KV * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(in_dim, N_KV * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HID, bias=False)

    def forward(self, x, cos, sin, attn_bias):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, T, N_KV, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, T, N_KV, HEAD_DIM).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        rep = N_HEADS // N_KV
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(B, T, N_HEADS * HEAD_DIM)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HID, INTER, bias=False)
        self.up_proj = nn.Linear(HID, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HID, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """EAGLE-3 first decoder layer: concatenates normed token embeds with normed
    fused hidden along the feature dim before attention (layer_idx == 0)."""

    def __init__(self):
        super().__init__()
        self.self_attn = Attention(in_dim=2 * HID)
        self.mlp = MLP()
        self.input_layernorm = RMSNorm(HID)  # on token embeds
        self.hidden_norm = RMSNorm(HID)  # on fused hidden
        self.post_attention_layernorm = RMSNorm(HID)

    def forward(self, embeds, hidden, cos, sin, attn_bias):
        e = self.input_layernorm(embeds)
        residual = hidden  # _norm_after_residual: residual = pre-norm fused
        h = self.hidden_norm(hidden)
        x = torch.cat([e, h], dim=-1)  # [B, T, 2H]
        attn_out = self.self_attn(x, cos, sin, attn_bias)
        res1 = attn_out + residual
        y = self.post_attention_layernorm(res1)
        mlp_out = self.mlp(y)
        return mlp_out, res1


class DraftBody(nn.Module):
    """vLLM `LlamaModel` (the draft body). State-dict keys land under `model.*`."""

    def __init__(self, norm_before_fc: bool = True):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HID)
        self.norm_before_fc = norm_before_fc
        if norm_before_fc:
            self.input_norm = RMSNorm(N_AUX * HID)  # RMSNorm(7680)
        self.fc = nn.Linear(N_AUX * HID, HID, bias=False)
        self.layers = nn.ModuleList([DecoderLayer()])
        self.norm = RMSNorm(HID)

    def combine(self, fused):
        if self.norm_before_fc:
            fused = self.input_norm(fused)
        return self.fc(fused)

    def forward(self, input_ids, fused, cos, sin, attn_bias):
        h0 = self.combine(fused)
        embeds = self.embed_tokens(input_ids)
        mlp_out, res1 = self.layers[0](embeds, h0, cos, sin, attn_bias)
        return self.norm(mlp_out + res1)


class Eagle3DraftHead(nn.Module):
    def __init__(self, norm_before_fc: bool = True):
        super().__init__()
        self.model = DraftBody(norm_before_fc)
        self.lm_head = nn.Linear(HID, VOCAB, bias=False)

    def forward_hidden(self, input_ids, fused, cos, sin, attn_bias):
        return self.model(input_ids, fused, cos, sin, attn_bias)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def find_safetensors(model_id: str) -> str:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo = "models--" + model_id.replace("/", "--")
    pats = [
        os.path.join(home, "hub", repo, "snapshots", "*", "model.safetensors"),
        os.path.join(home, "hub", repo, "snapshots", "*", "*.safetensors"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no safetensors for {model_id} under {home}/hub/{repo}")


def load_target_embedding(model_id: str) -> torch.Tensor:
    from safetensors.torch import safe_open

    path = find_safetensors(model_id)
    with safe_open(path, framework="pt") as f:
        return f.get_tensor("model.language_model.embed_tokens.weight")  # [V, H]


def load_corpus(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["records"], blob.get("meta", {})


def pack_batches(records, batch_tokens, order):
    """Greedy token-budget packing over a given sample order."""
    batches, cur, cur_tok = [], [], 0
    for i in order:
        L = int(records[i]["input_ids"].shape[0])
        if cur and cur_tok + L > batch_tokens:
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += L
    if cur:
        batches.append(cur)
    return batches


def collate(records, idxs, shift, device, dtype):
    recs = [records[i] for i in idxs]
    B = len(recs)
    T = max(int(r["input_ids"].shape[0]) for r in recs)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    fused = torch.zeros(B, T, N_AUX * HID, dtype=torch.float32)
    labels = torch.full((B, T), IGNORE, dtype=torch.long)
    keep = torch.zeros(B, T, dtype=torch.bool)
    for b, r in enumerate(recs):
        L = int(r["input_ids"].shape[0])
        input_ids[b, :L] = r["input_ids"].long()
        fused[b, :L] = r["aux"].permute(1, 0, 2).reshape(L, N_AUX * HID).float()
        labels[b, :L] = r["next_token_ids"].long()
        keep[b, :L] = True
    if shift > 0:  # feat[j] = fused[j - shift]; front positions become unfed
        rolled = torch.zeros_like(fused)
        rolled[:, shift:, :] = fused[:, : T - shift, :]
        fused = rolled
        labels[:, :shift] = IGNORE
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    allow = causal[None, :, :] & keep[:, None, :]  # [B, q, k]
    bias = torch.zeros(B, 1, T, T, dtype=dtype)
    bias.masked_fill_(~allow[:, None, :, :], float("-inf"))
    return (
        input_ids.to(device),
        fused.to(device, dtype=dtype),
        labels.to(device),
        bias.to(device),
        T,
    )


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(head, records, shift, batch_tokens, device, dtype, rope_theta, chunk=1024):
    head.eval()
    order = list(range(len(records)))
    correct = total = 0
    loss_sum = 0.0
    for idxs in pack_batches(records, batch_tokens, order):
        input_ids, fused, labels, bias, T = collate(records, idxs, shift, device, dtype)
        cos, sin = build_rope(T, HEAD_DIM, rope_theta, device, dtype)
        with torch.autocast("cuda", dtype=dtype):
            hidden = head.forward_hidden(input_ids, fused, cos, sin, bias)
        flat = hidden.reshape(-1, HID)
        tgt = labels.reshape(-1)
        mask = tgt != IGNORE
        sel, tt = flat[mask], tgt[mask]
        for c0 in range(0, sel.shape[0], chunk):  # chunk the 262k-way head
            with torch.autocast("cuda", dtype=dtype):
                lc = head.lm_head(sel[c0:c0 + chunk])
            lcf = lc.float()
            ttc = tt[c0:c0 + chunk]
            loss_sum += F.cross_entropy(lcf, ttc, reduction="sum").item()
            correct += (lcf.argmax(-1) == ttc).sum().item()
        total += int(tt.numel())
    head.train()
    acc = correct / max(1, total)
    return {"tf_acceptance_rate": acc, "loss": loss_sum / max(1, total), "n": total}


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def lr_at(step, warmup, total, base):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def save_checkpoint(head, cfg, out_dir):
    """Write a self-contained checkpoint dir (config.json + model_last.pt) so
    eval_eagle3.py can score it directly. Snapshots include the frozen embed/
    lm_head tables so eval needs no target weights."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    torch.save(head.state_dict(), os.path.join(out_dir, "model_last.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--eval_corpus", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max_epochs", type=float, default=None,
                    help="epoch cap; default = SENPAI_MAX_EPOCHS env (hard bound)")
    ap.add_argument("--max_minutes", type=float, default=85.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--batch_tokens", type=int, default=4096)
    ap.add_argument("--loss_chunk", type=int, default=1024,
                    help="token chunk for the 262k-way lm_head+CE (memory guard)")
    ap.add_argument("--feature_shift", type=int, default=1)
    ap.add_argument("--rope_theta", type=float, default=1e6)
    ap.add_argument("--norm_before_fc", type=int, default=1)
    ap.add_argument("--train_lm_head", action="store_true")
    ap.add_argument("--train_embed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=0,
                    help="save an intermediate checkpoint dir step_<n>/ every N "
                         "steps (0=off); lets eval_eagle3.py score the curve")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "senpai-v1"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="eagle3-drafter-training")
    ap.add_argument("--wandb_name", default="fern/eagle3-debug-1k-2ep")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16

    records, meta = load_corpus(args.corpus)
    corpus_tokens = sum(int(r["input_ids"].shape[0]) for r in records)
    batches_per_epoch = max(1, len(pack_batches(records, args.batch_tokens,
                                                list(range(len(records))))))

    # ---- HARD BOUNDS: epoch cap + wall clock (self-enforced, never overridden) #
    env_epochs = float(os.environ.get("SENPAI_MAX_EPOCHS", "1e9"))
    env_minutes = float(os.environ.get("SENPAI_TIMEOUT_MINUTES", "1e9"))
    eff_epochs = min(args.max_epochs if args.max_epochs else env_epochs, env_epochs)
    epoch_cap_steps = int(eff_epochs * batches_per_epoch)
    total_steps = min(args.steps, epoch_cap_steps)
    max_minutes = min(args.max_minutes, env_minutes - 3.0)  # 3-min safety margin
    bound = ("epoch_cap" if total_steps == epoch_cap_steps and total_steps < args.steps
             else "requested_steps")
    warmup = min(args.warmup, max(1, total_steps // 5))

    print("=" * 72, flush=True)
    print(f"[train] corpus: {len(records)} records, {corpus_tokens} tokens, "
          f"~{batches_per_epoch} batches/epoch @ batch_tokens={args.batch_tokens}",
          flush=True)
    print(f"[train] SENPAI_MAX_EPOCHS={env_epochs}  SENPAI_TIMEOUT_MINUTES={env_minutes}",
          flush=True)
    print(f"[train] requested_steps={args.steps}  epoch_cap_steps={epoch_cap_steps} "
          f"(={eff_epochs:g} epochs)  -> total_steps={total_steps} [bound={bound}]",
          flush=True)
    if bound == "epoch_cap":
        print(f"[train] !! EPOCH-CAPPED: {args.steps} requested steps would be "
              f"~{args.steps / batches_per_epoch:.1f} epochs > hard cap "
              f"{eff_epochs:g}. Running {total_steps} steps. !!", flush=True)
    print(f"[train] wall-clock cap: {max_minutes:.1f} min; warmup={warmup}; "
          f"feature_shift={args.feature_shift}; norm_before_fc={bool(args.norm_before_fc)}",
          flush=True)
    print("=" * 72, flush=True)

    # ---- model ----
    head = Eagle3DraftHead(norm_before_fc=bool(args.norm_before_fc))
    emb = load_target_embedding(args.model)
    head.model.embed_tokens.weight.data = emb.clone().to(torch.bfloat16)
    head.lm_head.weight.data = emb.clone().to(torch.bfloat16)
    head = head.to(device)
    head.model.embed_tokens.weight.requires_grad_(bool(args.train_embed))
    head.lm_head.weight.requires_grad_(bool(args.train_lm_head))

    trainable = [p for p in head.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in head.parameters())
    print(f"[train] trainable params: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M total "
          f"(embed frozen={not args.train_embed}, lm_head frozen={not args.train_lm_head})",
          flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))

    eval_records = None
    if args.eval_corpus:
        eval_records, _ = load_corpus(args.eval_corpus)

    # ---- logging ----
    os.makedirs(args.output, exist_ok=True)
    cfg = {
        "architectures": ["Eagle3LlamaForCausalLM"], "model_type": "llama",
        "draft_vocab_size": VOCAB, "vocab_size": VOCAB, "hidden_size": HID,
        "num_hidden_layers": 1, "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV, "head_dim": HEAD_DIM, "intermediate_size": INTER,
        "rms_norm_eps": EPS, "rope_theta": args.rope_theta,
        "norm_before_fc": bool(args.norm_before_fc), "target_hidden_size": HID,
        "eagle_aux_hidden_state_layer_ids": [2, 21, 39], "num_aux_hidden_states": N_AUX,
        "feature_shift": args.feature_shift, "tie_word_embeddings": False,
        "train_meta": {"corpus_tokens": corpus_tokens, "total_steps": total_steps,
                       "lr": args.lr, "warmup": warmup, "wd": args.wd,
                       "batch_tokens": args.batch_tokens, "bound": bound,
                       "eff_epochs_cap": eff_epochs, "target_model": args.model},
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    metrics_path = os.path.join(args.output, "metrics.jsonl")
    mfile = open(metrics_path, "w")

    use_wandb = False
    run = None
    if not args.no_wandb:
        try:
            import wandb

            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             group=args.wandb_group, name=args.wandb_name,
                             config={**vars(args), **cfg, "corpus_tokens": corpus_tokens,
                                     "total_steps": total_steps, "bound": bound})
            use_wandb = True
            print(f"[train] wandb: {run.url}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[train] wandb disabled ({e!r}); logging to {metrics_path}", flush=True)

    def log(d):
        mfile.write(json.dumps(d) + "\n")
        mfile.flush()
        if use_wandb:
            run.log(d, step=d.get("step"))

    # ---- loop ----
    torch.cuda.reset_peak_memory_stats()
    head.train()
    step = 0
    tokens_seen = 0
    best_val = -1.0
    t0 = time.time()
    epoch = 0
    stop = False
    while step < total_steps and not stop:
        order = list(range(len(records)))
        random.Random(args.seed + epoch).shuffle(order)
        for idxs in pack_batches(records, args.batch_tokens, order):
            if step >= total_steps:
                break
            elapsed = (time.time() - t0) / 60.0
            if elapsed >= max_minutes:
                print(f"[train] wall-clock cap hit at step {step} ({elapsed:.1f} min)",
                      flush=True)
                stop = True
                break

            lr = lr_at(step, warmup, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            input_ids, fused, labels, bias, T = collate(
                records, idxs, args.feature_shift, device, dtype)
            cos, sin = build_rope(T, HEAD_DIM, args.rope_theta, device, dtype)
            with torch.autocast("cuda", dtype=dtype):
                hidden = head.forward_hidden(input_ids, fused, cos, sin, bias)
            flat = hidden.reshape(-1, HID)
            tgt = labels.reshape(-1)
            mask = tgt != IGNORE
            sel, tt = flat[mask], tgt[mask]
            n_tok = int(tt.numel())

            # Chunk the 262k-way lm_head + CE; per-chunk backward keeps the fp32
            # logit/grad tensors small (else a single [N, 262144] step OOMs the A10G).
            opt.zero_grad(set_to_none=True)
            loss_val, correct = 0.0, 0
            n_chunks = max(1, math.ceil(n_tok / args.loss_chunk))
            for ci in range(n_chunks):
                c0 = ci * args.loss_chunk
                ttc = tt[c0:c0 + args.loss_chunk]
                with torch.autocast("cuda", dtype=dtype):
                    lc = head.lm_head(sel[c0:c0 + args.loss_chunk])
                ce = F.cross_entropy(lc.float(), ttc, reduction="sum") / max(1, n_tok)
                ce.backward(retain_graph=(ci < n_chunks - 1))
                loss_val += float(ce.item())
                correct += int((lc.detach().argmax(-1) == ttc).sum().item())
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            loss = loss_val
            acc = correct / max(1, n_tok)
            tokens_seen += n_tok
            step += 1

            if step % args.log_every == 0 or step == 1 or step == total_steps:
                rec = {
                    "step": step, "train/loss": float(loss),
                    "train/tf_acceptance_rate": acc, "train/lr": lr,
                    "train/grad_norm": float(gnorm), "train/tokens_seen": tokens_seen,
                    "train/epoch_equiv": tokens_seen / max(1, corpus_tokens),
                    "train/minutes": (time.time() - t0) / 60.0,
                }
                log(rec)
                print(f"  step {step:4d}/{total_steps} loss={rec['train/loss']:.4f} "
                      f"acc={acc:.4f} lr={lr:.2e} gnorm={float(gnorm):.2f} "
                      f"ep={rec['train/epoch_equiv']:.2f}", flush=True)

            if args.eval_every and eval_records is not None and (
                step % args.eval_every == 0 or step == total_steps):
                ev = evaluate(head, eval_records, args.feature_shift,
                              args.batch_tokens, device, dtype, args.rope_theta)
                vrec = {"step": step, "val/tf_acceptance_rate": ev["tf_acceptance_rate"],
                        "val/loss": ev["loss"]}
                log(vrec)
                print(f"  [eval] step {step}: val tf_acc={ev['tf_acceptance_rate']:.4f} "
                      f"val loss={ev['loss']:.4f} (n={ev['n']})", flush=True)
                if ev["tf_acceptance_rate"] > best_val:
                    best_val = ev["tf_acceptance_rate"]
                    torch.save(head.state_dict(),
                               os.path.join(args.output, "model_best.pt"))

            if args.save_every and step % args.save_every == 0 and step < total_steps:
                ckpt_dir = os.path.join(args.output, f"step_{step}")
                save_checkpoint(head, cfg, ckpt_dir)
                print(f"  [ckpt] saved snapshot -> {ckpt_dir}", flush=True)
        epoch += 1

    # ---- final eval + save ----
    final = {}
    if eval_records is not None:
        ev = evaluate(head, eval_records, args.feature_shift, args.batch_tokens,
                      device, dtype, args.rope_theta)
        final = {"final_val/tf_acceptance_rate": ev["tf_acceptance_rate"],
                 "final_val/loss": ev["loss"], "final_val/n": ev["n"]}
        log({"step": step, **final})
        print(f"[train] FINAL held-out: tf_acceptance_rate={ev['tf_acceptance_rate']:.4f} "
              f"loss={ev['loss']:.4f} (n={ev['n']})", flush=True)
        if ev["tf_acceptance_rate"] > best_val:
            best_val = ev["tf_acceptance_rate"]
            torch.save(head.state_dict(), os.path.join(args.output, "model_best.pt"))

    torch.save(head.state_dict(), os.path.join(args.output, "model_last.pt"))
    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = {"step": step, "total_steps": total_steps, "bound": bound,
               "minutes": (time.time() - t0) / 60.0, "peak_gpu_gb": peak,
               "best_val_tf_acceptance": best_val if best_val >= 0 else None, **final}
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] done: {step} steps in {summary['minutes']:.1f} min, "
          f"peak GPU {peak:.2f} GB -> {args.output}", flush=True)
    if use_wandb:
        run.summary.update(summary)
        run.finish()
    mfile.close()


if __name__ == "__main__":
    main()
