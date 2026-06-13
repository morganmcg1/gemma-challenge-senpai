# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Generate the EAGLE-3 offline distillation corpus for google/gemma-4-E4B-it.

PR #16, Step 2 (enlarged per advisor option (c): scale the corpus so a 1000-step
run == 2 epochs at batch_tokens=4096, giving a cleaner, less-overfit held-out
signal than memorising a tiny corpus). For each MATH sample we run a single
forward pass through the target's text tower and collect, per token position:
  - input_ids
  - the 3 auxiliary residual-stream hidden states at layers (2, 21, 39),
    captured as HF output_hidden_states[2|21|39] (== vLLM's EAGLE-3 aux export;
    see research/eagle3_drafter/arch_notes.md S4), stored fp16
  - next_token_ids (input_ids shifted left by one; final position = IGNORE)

The 20-sample held-out eval is carved from a separate index range of the same
seeded shuffle (build_texts dedups, so the split is content-disjoint) and the
no-overlap is asserted explicitly at both the text and token-id level.

No HF Job. Single local A10G model load; reused across all samples.

Run (from target/):
  HF_HOME=/senpai-run/home/student-fern/.cache/huggingface \
  python scripts/drafter/gen_eagle3_corpus.py \
      --n_train 8000 --n_eval 20 --max_tokens 512 --batch 4
"""

from __future__ import annotations

import argparse
import os
import time

import torch

IGNORE = -100
DEFAULT_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
]


def find_text_model(model):
    """Return the submodule whose `.layers` has the most entries and which has an
    `embed_tokens` (the 42-layer Gemma-4 text tower)."""
    best = None
    for _, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        emb = getattr(mod, "embed_tokens", None)
        if layers is None or emb is None:
            continue
        try:
            n = len(layers)
        except TypeError:
            continue
        if best is None or n > best[0]:
            best = (n, mod)
    assert best is not None, "could not locate a text module with layers+embed_tokens"
    return best[1], best[0]


def build_texts(tokenizer, dataset_id, subjects, n_total, seed, split="train"):
    """Load + shuffle MATH samples across subjects; return up to n_total UNIQUE
    chat-formatted strings.

    Dedup-by-text means a later index-range split into train/eval is guaranteed
    content-disjoint. If the split has fewer than n_total usable samples we return
    all of them (with a warning) rather than hard-failing -- this mirror of
    Hendrycks MATH has ~6,754 train samples, below the ~8,000 target, so the
    enlarged corpus uses the full split and a 1000-step run lands at ~2 epochs."""
    from datasets import concatenate_datasets, load_dataset

    parts = [load_dataset(dataset_id, sub, split=split) for sub in subjects]
    ds = concatenate_datasets(parts).shuffle(seed=seed)

    texts, seen = [], set()
    for ex in ds:
        problem = (ex.get("problem") or "").strip()
        solution = (ex.get("solution") or "").strip()
        if not problem or not solution:
            continue
        msgs = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
        except Exception:  # noqa: BLE001 - fall back to a plain concatenation
            text = f"{problem}\n\n{solution}"
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
        if len(texts) >= n_total:
            break
    if len(texts) < n_total:
        print(f"[corpus] WARNING: only {len(texts)} unique usable samples in "
              f"{dataset_id}:{split} {subjects} < requested {n_total}; using all.",
              flush=True)
    return texts


@torch.no_grad()
def collect(model, text_model, tokenizer, texts, aux_layers, max_tokens, batch, device):
    """Run forwards and return a list of per-sample corpus dicts."""
    records = []
    n = len(texts)
    t0 = time.time()
    for start in range(0, n, batch):
        chunk = texts[start : start + batch]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = text_model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states  # tuple len num_layers+1
        for b in range(input_ids.shape[0]):
            length = int(attn[b].sum().item())
            if length < 8:
                continue
            ids = input_ids[b, :length].to(torch.int32).cpu()
            aux = torch.stack(
                [hs[li][b, :length].to(torch.float16).cpu() for li in aux_layers],
                dim=0,
            )  # [3, length, 2560]
            nxt = torch.full((length,), IGNORE, dtype=torch.int64)
            nxt[:-1] = ids[1:].to(torch.int64)
            records.append({"input_ids": ids, "aux": aux, "next_token_ids": nxt})
        done = min(start + batch, n)
        if done % 20 == 0 or done == n:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"  [{done}/{n}] {time.time()-t0:.0f}s peakmem={mem:.1f}GB "
                f"records={len(records)}",
                flush=True,
            )
    return records


def sanity_check(records, aux_layers):
    assert records, "no records collected"
    bad = 0
    for r in records:
        a = r["aux"]
        assert a.shape[0] == len(aux_layers) and a.shape[-1] == 2560, a.shape
        if torch.isnan(a.float()).any() or torch.isinf(a.float()).any():
            bad += 1
        ids, nxt = r["input_ids"], r["next_token_ids"]
        # input_ids[1:] must equal next_token_ids[:-1]
        assert torch.equal(ids[1:].to(torch.int64), nxt[:-1]), "shift mismatch"
        assert nxt[-1].item() == IGNORE
    assert bad == 0, f"{bad} records contain NaN/Inf aux"
    lens = [int(r["input_ids"].shape[0]) for r in records]
    print(
        f"  sanity OK: {len(records)} records, no NaN/Inf, shift consistent; "
        f"len min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--dataset", default="EleutherAI/hendrycks_math")
    ap.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    ap.add_argument("--n_train", type=int, default=8000,
                    help="target train samples; auto-capped to the available pool")
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux_layers", type=int, nargs=3, default=[2, 21, 39])
    ap.add_argument(
        "--out", default="research/eagle3_drafter/train_data/debug_1k_corpus.pt"
    )
    ap.add_argument(
        "--eval_out",
        default="research/eagle3_drafter/train_data/debug_1k_eval_corpus.pt",
    )
    args = ap.parse_args()

    device = "cuda"
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    print(f"[corpus] loading {args.model} (bf16) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to(device).eval()
    text_model, n_layers = find_text_model(model)
    print(f"[corpus] text tower: {n_layers} layers; aux={args.aux_layers}", flush=True)

    # Hold out the eval slice first (a separate index range of the seeded shuffle),
    # then use the remaining pool for train. build_texts dedups, so the index-range
    # split is content-disjoint; we assert it explicitly here and again at the
    # token-id level after collection (advisor option (c): "assert no id overlap").
    n_total = args.n_train + args.n_eval
    pool = build_texts(tokenizer, args.dataset, args.subjects, n_total, args.seed)
    assert len(pool) > args.n_eval, f"pool {len(pool)} <= n_eval {args.n_eval}"
    eval_texts = pool[: args.n_eval]
    train_texts = pool[args.n_eval : args.n_eval + args.n_train]
    assert set(train_texts).isdisjoint(set(eval_texts)), "train/eval text overlap!"
    print(
        f"[corpus] pool={len(pool)} -> {len(train_texts)} train + {len(eval_texts)} "
        f"eval (held-out, disjoint) from {args.dataset}:train {args.subjects}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats()
    print("[corpus] collecting TRAIN ...", flush=True)
    train = collect(
        model, text_model, tokenizer, train_texts, args.aux_layers,
        args.max_tokens, args.batch, device,
    )
    sanity_check(train, args.aux_layers)
    print("[corpus] collecting EVAL ...", flush=True)
    ev = collect(
        model, text_model, tokenizer, eval_texts, args.aux_layers,
        args.max_tokens, args.batch, device,
    )
    sanity_check(ev, args.aux_layers)

    # Token-id-level no-overlap assert (advisor: "assert no id overlap").
    eval_keys = {tuple(r["input_ids"].tolist()) for r in ev}
    overlap = sum(tuple(r["input_ids"].tolist()) in eval_keys for r in train)
    assert overlap == 0, f"{overlap} train records share token-ids with eval"
    print(f"  no-overlap OK: 0/{len(train)} train records match any of "
          f"{len(eval_keys)} eval id-sequences", flush=True)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "subjects": args.subjects,
        "aux_layers": args.aux_layers,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "hidden_size": 2560,
        "vocab_size": 262144,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"records": train, "meta": meta}, args.out)
    torch.save({"records": ev, "meta": meta}, args.eval_out)
    tr_tok = sum(int(r["input_ids"].shape[0]) for r in train)
    print(
        f"[corpus] saved {len(train)} train records ({tr_tok} tokens) -> {args.out}",
        flush=True,
    )
    print(f"[corpus] saved {len(ev)} eval records -> {args.eval_out}", flush=True)
    print(f"[corpus] peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)
    print("[corpus] done.", flush=True)


if __name__ == "__main__":
    main()
