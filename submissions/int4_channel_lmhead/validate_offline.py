#!/usr/bin/env python
"""Offline validation gate for the int4 g128 + untied int4 lm_head re-quant.

Runs BEFORE any benchmark/serve. Three checks:
  1. config.json parses and the quant config is well formed.
  2. every quantized tensor decompresses (unpack int32 -> int4 -> dequant bf16)
     to a finite tensor of the declared shape.  (343 body + lm_head = 344)
  3. fake-quant PPL sweep: rebuild the *text* model (Gemma4ForCausalLM) from the
     dequantized weights -- embed_tokens bf16, lm_head from the int4 head -- and
     score data/ppl_ground_truth_tokens.jsonl with the exact ppl_endpoint.py
     convention (teacher-forced NLL over the target span, ppl=exp(sum_nll/sum_tok)).

Dequant is done by hand (q * scale, symmetric, per group) so the gate does not
depend on transformers' compressed-tensors integration -- it validates the bytes
this build wrote.  GPU forward (CUDA_VISIBLE_DEVICES=0); CPU also works.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

DEFAULT_CKPT = "/workspace/gemma_build/int4_g128_lmhead"
DEFAULT_PPL = (
    "official/main_bucket/shared_resources/speed_benchmark/data/"
    "ppl_ground_truth_tokens.jsonl"
)
LANG_PREFIX = "model.language_model."


def dequant_group(packed: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """int32-packed int4 + per-group bf16 scale -> bf16 dense weight."""
    out_dim, in_dim = int(shape[0]), int(shape[1])
    q = unpack_from_int32(packed, 4, torch.Size([out_dim, in_dim]), packed_dim=1).to(torch.float32)
    assert int(q.min()) >= -8 and int(q.max()) <= 7, f"int4 range violation [{q.min()},{q.max()}]"
    n_groups = scale.shape[1]
    assert in_dim % n_groups == 0, f"in_dim {in_dim} not divisible by n_groups {n_groups}"
    gs = in_dim // n_groups
    qg = q.reshape(out_dim, n_groups, gs)
    w = (qg * scale.to(torch.float32).unsqueeze(-1)).reshape(out_dim, in_dim)
    return w.to(torch.bfloat16)


def check_config(ckpt: Path) -> None:
    cfg = json.load(open(ckpt / "config.json"))
    assert cfg["tie_word_embeddings"] is False, "top-level tie_word_embeddings must be False"
    assert cfg["text_config"]["tie_word_embeddings"] is False, "text_config tie must be False"
    q = cfg["quantization_config"]
    assert q["format"] == "pack-quantized", q["format"]
    g0 = q["config_groups"]["group_0"]["weights"]
    g1 = q["config_groups"]["group_1"]["weights"]
    assert g0["num_bits"] == 4 and g0["symmetric"] is True
    assert q["config_groups"]["group_1"]["targets"] == ["re:.*lm_head"]
    assert "lm_head" not in q["ignore"], "lm_head must be removed from ignore"
    print(f"[config] ok | body gs={g0['group_size']} | head gs={g1['group_size']} "
          f"strategy={g1['strategy']} | ignore={len(q['ignore'])} | version={q['version']}")


def rebuild_text_state_dict(ckpt: Path) -> dict[str, torch.Tensor]:
    """Reconstruct the text-only (Gemma4ForCausalLM) bf16 state dict + decompress check."""
    sd: dict[str, torch.Tensor] = {}
    n_dequant = 0
    rel_errs: list[float] = []
    bases: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(ckpt / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for name in keys:
            if name.endswith(".weight_packed") or name.endswith(".weight_scale") or name.endswith(".weight_shape"):
                base, kind = name.rsplit(".", 1)
                bases.setdefault(base, {})[kind] = f.get_tensor(name)
                continue
            # plain (bf16) tensor: keep only the text-model subset
            if name == "lm_head.weight":
                sd["lm_head.weight"] = f.get_tensor(name)
            elif name.startswith(LANG_PREFIX):
                sd["model." + name[len(LANG_PREFIX):]] = f.get_tensor(name)
            # vision_tower / audio_tower / projectors: not part of the text PPL model

        for base, parts in bases.items():
            w = dequant_group(parts["weight_packed"], parts["weight_scale"], parts["weight_shape"])
            assert torch.isfinite(w).all(), f"non-finite dequant for {base}"
            if base == "lm_head":
                key = "lm_head.weight"
            elif base.startswith(LANG_PREFIX):
                key = "model." + base[len(LANG_PREFIX):] + ".weight"
            else:
                # body modules are all language_model.*; anything else is unexpected
                raise SystemExit(f"unexpected quantized base outside text model: {base}")
            sd[key] = w
            n_dequant += 1
    print(f"[decompress] dequantized {n_dequant} quantized tensors (343 body + lm_head expected) -> all finite")
    assert n_dequant == 344, f"expected 344 quantized tensors, got {n_dequant}"
    assert "lm_head.weight" in sd and "model.embed_tokens.weight" in sd
    # untied sanity: lm_head must differ from embed (it was re-quantized)
    same = torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"])
    print(f"[untie] lm_head identical to embed_tokens? {same} (expect False -- head is int4, embed is bf16)")
    return sd


def load_text_model(ckpt: Path, sd: dict[str, torch.Tensor], device: str):
    from transformers import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg_full = json.load(open(ckpt / "config.json"))
    tc = dict(cfg_full["text_config"])
    tc["tie_word_embeddings"] = False
    cfg = Gemma4TextConfig(**tc)
    model = Gemma4ForCausalLM(cfg)
    res = model.load_state_dict(sd, strict=False, assign=True)
    unexpected = list(res.unexpected_keys)
    missing = [k for k in res.missing_keys]
    assert not unexpected, f"unexpected keys when loading text model: {unexpected[:10]}"
    # missing should only be non-persistent buffers (e.g. rotary inv_freq), never weights
    bad_missing = [k for k in missing if k.endswith(".weight") or k.endswith(".bias")]
    assert not bad_missing, f"missing weight/bias keys: {bad_missing[:10]}"
    print(f"[load] text model loaded strict-ish | unexpected=0 | missing(buffers only)={len(missing)} {missing[:4]}")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def ppl_sweep(model, ppl_path: Path, device: str) -> dict:
    records = [json.loads(l) for l in open(ppl_path) if l.strip()]
    total_nll = 0.0
    total_tok = 0
    rec_ppls = []
    for i, r in enumerate(records):
        ctx = r["context_token_ids"]
        tgt = r["target_token_ids"]
        ids = torch.tensor([ctx + tgt], dtype=torch.long, device=device)
        score_start = max(len(ctx), 1)
        score_end = ids.shape[1]
        out = model(input_ids=ids)
        logits = out.logits[0]  # [seq, vocab] (already softcapped in forward)
        # logprob of token at position idx is predicted from logits[idx-1]
        sl = logits[score_start - 1:score_end - 1].float()  # [n_score, vocab]
        lp = torch.log_softmax(sl, dim=-1)
        tgt_ids = ids[0, score_start:score_end]
        tok_lp = lp.gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
        nll = float(-tok_lp.sum().item())
        n = int(tgt_ids.numel())
        total_nll += nll
        total_tok += n
        rec_ppls.append(math.exp(nll / n))
        if (i + 1) % 32 == 0:
            print(f"  scored {i+1}/{len(records)} (running ppl={math.exp(total_nll/total_tok):.4f})", flush=True)
    summary = {
        "ppl": math.exp(total_nll / total_tok),
        "mean_record_ppl": sum(rec_ppls) / len(rec_ppls),
        "num_records": len(records),
        "num_tokens": total_tok,
        "neg_log_likelihood": total_nll,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--ppl", default=DEFAULT_PPL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="research/_probe/ppl_offline_g128_head128.json")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"[init] ckpt={ckpt} device={device} torch={torch.__version__}")

    check_config(ckpt)
    sd = rebuild_text_state_dict(ckpt)
    model = load_text_model(ckpt, sd, device)
    del sd
    summary = ppl_sweep(model, Path(args.ppl), device)
    print("[ppl] " + json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    cap = 2.42
    verdict = "PASS" if summary["ppl"] <= cap else "FAIL"
    print(f"[verdict] token-weighted PPL={summary['ppl']:.4f} (cap {cap}) -> {verdict}")


if __name__ == "__main__":
    main()
