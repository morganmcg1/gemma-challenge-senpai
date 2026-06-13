#!/usr/bin/env python3
"""Shared wiring for the Gemma-4 MTP (EAGLE-style) drafter: target prefill ->
hidden states + shared-KV extraction -> draft proposal loop, plus the
centroid-intersection KL used for distillation.

The Gemma4 assistant is NOT a standalone LM. Per transformers'
`SinglePositionMultiTokenCandidateGenerator`, drafting one token requires:
  * the target's last-layer hidden state at the last seen position,
  * the target's shared KV (last layer of each layer_type: full + sliding),
  * inputs_embeds = cat(target.embed(last_token), last_hidden_state)  (2*H),
  * a CONSTANT position_ids = [[prefix_len - 1]].
Each subsequent draft step feeds the draft's own argmax + its own projected
hidden state back in; shared_kv stays fixed from the single target prefill.

This module reproduces that loop exactly so we can (a) measure per-position
acceptance offline and (b) compute distillation losses, without vLLM.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_target(model_id: str, device: str = "cuda", dtype=torch.bfloat16):
    import transformers
    last = None
    for cls_name in ["Gemma4ForConditionalGeneration", "Gemma4ForCausalLM", "AutoModelForCausalLM"]:
        try:
            C = getattr(transformers, cls_name, None)
            if C is None:
                continue
            m = C.from_pretrained(model_id, dtype=dtype).to(device).eval()
            return m
        except Exception as e:  # noqa
            last = e
    raise last


def load_drafter(model_id_or_path: str, device: str = "cuda", dtype=torch.bfloat16):
    from transformers import Gemma4AssistantForCausalLM
    return Gemma4AssistantForCausalLM.from_pretrained(model_id_or_path, dtype=dtype).to(device)


def target_input_embeddings(target):
    """The target's input embedding table (used to embed the last seen token)."""
    return target.get_input_embeddings()


# ----------------------------------------------------------------------------
# Target prefill: hidden states + shared KV + logits
# ----------------------------------------------------------------------------
@torch.no_grad()
def target_prefill(target, input_ids: torch.Tensor):
    """Run the target once over `input_ids`, returning last-layer hidden states,
    the shared_kv_states dict, and the LM logits at every position.

    Returns: last_hidden [B,L,H], shared_kv {type:(K,V)}, logits [B,L,V]
    """
    out = target(
        input_ids=input_ids,
        output_hidden_states=True,
        return_shared_kv_states=True,
        use_cache=False,
    )
    last_hidden = out.hidden_states[-1]
    shared_kv = out.shared_kv_states
    logits = out.logits
    return last_hidden, shared_kv, logits


def crop_shared_kv(shared_kv: dict, length: int) -> dict:
    """Causally crop each (K,V) in the shared_kv dict to `length` kv positions."""
    return {k: (v[0][:, :, :length, :], v[1][:, :, :length, :]) for k, v in shared_kv.items()}


# ----------------------------------------------------------------------------
# Draft proposal loop (matches SinglePositionMultiTokenCandidateGenerator)
# ----------------------------------------------------------------------------
@torch.no_grad()
def propose_k(drafter, target_embed, last_token_id, last_hidden, shared_kv,
              position_index: int, K: int, attention_mask=None):
    """Draft K tokens autoregressively from a single seen position.

    last_token_id : [B,1] long   (the last seen/accepted token)
    last_hidden   : [B,1,H]       (target hidden at that position)
    shared_kv     : already cropped to the prefix length
    position_index: prefix_len - 1 (constant position id during drafting)
    Returns: drafted_ids [B,K] long, drafted_logits list of [B,V].
    """
    device = last_token_id.device
    position_ids = torch.tensor([[position_index]], dtype=torch.long, device=device)
    drafted_ids, drafted_logits = [], []
    cur_token, cur_hidden = last_token_id, last_hidden
    for _ in range(K):
        emb = target_embed(cur_token)                      # [B,1,H]
        inputs_embeds = torch.cat([emb, cur_hidden], dim=-1)  # [B,1,2H]
        out = drafter(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            shared_kv_states=shared_kv,
            use_cache=False,
        )
        logits = out.logits[:, -1, :]                      # [B,V]
        cur_token = logits.argmax(dim=-1, keepdim=True)    # [B,1]
        cur_hidden = out.last_hidden_state                 # [B,1,H]
        drafted_ids.append(cur_token)
        drafted_logits.append(logits)
    return torch.cat(drafted_ids, dim=1), drafted_logits


def draft_logits_at(drafter, target_embed, last_token_id, last_hidden, shared_kv,
                    position_index: int, attention_mask=None):
    """Single depth-1 draft forward; returns logits [B,V] and next hidden [B,1,H].
    Gradients flow (no no_grad) — used in training."""
    device = last_token_id.device
    position_ids = torch.tensor([[position_index]], dtype=torch.long, device=device)
    emb = target_embed(last_token_id)
    inputs_embeds = torch.cat([emb, last_hidden], dim=-1)
    out = drafter(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        shared_kv_states=shared_kv,
        use_cache=False,
    )
    return out.logits[:, -1, :], out.last_hidden_state


# ----------------------------------------------------------------------------
# Centroid-aware distillation loss
# ----------------------------------------------------------------------------
def surfaced_mask(draft_logits_row: torch.Tensor, gathered: torch.Tensor) -> torch.Tensor:
    """Which gathered target-ids are actually surfaced by the draft's centroid head.

    The masked embedder fills non-surfaced positions with `min(real)-1`, so a
    gathered logit equal to the row minimum is a non-surfaced token.
    """
    row_min = draft_logits_row.min(dim=-1, keepdim=True).values
    return gathered > (row_min + 0.5)


def kl_ce_loss(draft_logits, topk_ids, topk_probs, argmax_id, alpha=0.0, temperature=1.0):
    """Centroid-intersection KL (+ optional argmax CE).

    draft_logits : [B,V]  (centroid-masked; non-surfaced = fill)
    topk_ids     : [B,k]  target top-k token ids
    topk_probs   : [B,k]  target softmax probs (sum~1)
    Returns scalar loss, plus (kl, ce, frac_surfaced) diagnostics.
    """
    # Compute the KL in float32 with a finite (not -inf) mask fill: a -inf fill
    # makes the non-surfaced term 0*(logp-(-inf)) = 0*inf = NaN, which poisons
    # the sum AND its gradient. A large finite negative keeps every term finite,
    # so non-surfaced positions contribute exactly 0 (p==0 there).
    full = draft_logits.float()                              # [B,V]
    gathered = full.gather(1, topk_ids)                      # [B,k]
    row_min = full.min(dim=-1, keepdim=True).values
    mask = gathered > (row_min + 0.5)                        # [B,k] bool

    # Renormalize target probs over the surfaced intersection.
    p = topk_probs.float() * mask
    p_sum = p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    p = p / p_sum

    NEG = -1e9
    masked_logits = torch.where(mask, gathered, torch.full_like(gathered, NEG)) / temperature
    log_q = F.log_softmax(masked_logits, dim=-1)             # finite everywhere
    log_p = p.clamp(min=1e-12).log()
    kl = (p * (log_p - log_q)).sum(dim=-1)                   # [B]; 0 at non-surfaced
    # Drop rows with no surfaced overlap (undefined KL).
    valid = mask.any(dim=-1).float()
    kl = (kl * valid).sum() / valid.sum().clamp(min=1)

    if alpha > 0:
        ce = F.cross_entropy(draft_logits.float(), argmax_id)
    else:
        ce = torch.zeros((), device=draft_logits.device)

    loss = alpha * ce + (1.0 - alpha) * kl
    frac_surfaced = mask.float().mean()
    return loss, kl.detach(), ce.detach(), frac_surfaced.detach()
