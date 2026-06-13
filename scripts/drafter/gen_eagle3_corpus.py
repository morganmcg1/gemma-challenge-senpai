# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Generate the EAGLE-3 offline distillation corpus for google/gemma-4-E4B-it.

PR #16 (Step 2) generated a MATH-only corpus. PR #25 extends this to a mixed
MATH + ShareGPT-style corpus: the EAGLE-3 draft head is trained on the benchmark
distribution as well as STEM reasoning. For each sample we run a single forward
pass through the target's text tower and collect, per token position:
  - input_ids
  - the 3 auxiliary residual-stream hidden states at layers (2, 21, 39),
    captured as HF output_hidden_states[2|21|39] (== vLLM's EAGLE-3 aux export;
    see research/eagle3_drafter/arch_notes.md S4), stored fp16
  - next_token_ids (input_ids shifted left by one; final position = IGNORE)
  - source ("math" | "sharegpt"), so eval can break acceptance down by source.

Two sources:
  - MATH (EleutherAI/hendrycks_math): train from the `train` split; the held-out
    is drawn from the `test` split (there is no `validation` split), so it is
    split-disjoint from train.
  - ShareGPT (RyokoAI/ShareGPT52K, fallback allenai/WildChat-1M): first human turn
    -> user prompt, first gpt turn -> assistant completion, light HTML-tag strip,
    chat-templated, truncated to --max_tokens. Eval = first N unique pairs, train
    = next M, so the two are index- and content-disjoint.

Train/eval no-overlap is asserted at the token-id level across BOTH sources.

To avoid re-running the (~11 min) MATH train forward passes, --reuse_math_train
loads an existing MATH train corpus and only (re)generates ShareGPT + held-out.

No HF Job. Single local A10G model load; reused across all samples.

Run (from target/), full PR #25 corpus:
  python scripts/drafter/gen_eagle3_corpus.py \
      --n_math_train 100000 --n_sharegpt_train 1500 \
      --n_math_eval 200 --n_sharegpt_eval 50 \
      --max_tokens 512 --batch 4 \
      --out research/eagle3_drafter/train_data/full_train_corpus.pt \
      --eval_out research/eagle3_drafter/train_data/full_holdout_corpus.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


# --------------------------------------------------------------------------- #
# ShareGPT-style conversational source (PR #25)
# --------------------------------------------------------------------------- #
_HTML_TAG = re.compile(r"<[^>]+>")


def _sg_as_list(conv):
    """ShareGPT52K stores `conversations` (and sometimes each turn) as a JSON
    string rather than a struct; normalise to a list of dicts."""
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:  # noqa: BLE001
            return []
    return conv if isinstance(conv, list) else []


def _sg_as_dict(turn):
    if isinstance(turn, str):
        try:
            return json.loads(turn)
        except Exception:  # noqa: BLE001
            return {}
    return turn if isinstance(turn, dict) else {}


def _sg_first_pair(conv):
    """First human/user prompt and the following gpt/assistant reply."""
    human = gpt = None
    for turn in _sg_as_list(conv):
        t = _sg_as_dict(turn)
        role = t.get("from") or t.get("role")
        val = t.get("value") or t.get("content")
        if not val:
            continue
        if role in ("human", "user") and human is None:
            human = val
        elif role in ("gpt", "assistant", "bard", "chatgpt") and human is not None:
            gpt = val
            break
    return human, gpt


def build_sharegpt_texts(tokenizer, n_eval, n_train, repo="RyokoAI/ShareGPT52K"):
    """Stream a ShareGPT-style dataset; return (eval_texts, train_texts, repo).

    First human turn -> user prompt, first gpt turn -> assistant completion. A
    light HTML-tag strip removes web-scrape markup (e.g. the `<div class="markdown
    prose ...">` wrappers that the model would never emit). Eval = first n_eval
    unique pairs, train = next n_train, so the two are index- and content-disjoint
    (also asserted at the token-id level by the caller)."""
    from datasets import load_dataset

    need = n_eval + n_train
    repos = [repo, "allenai/WildChat-1M"]  # PR-sanctioned fallback
    texts, seen, used_repo = [], set(), None
    for r in repos:
        try:
            ds = load_dataset(r, split="train", streaming=True)
        except Exception as e:  # noqa: BLE001
            print(f"[corpus] ShareGPT repo {r} unavailable ({e!r}); next", flush=True)
            continue
        used_repo = r
        conv_key = "conversation" if "WildChat" in r else "conversations"
        for ex in ds:
            conv = ex.get(conv_key) or ex.get("conversations") or ex.get("conversation")
            human, gpt = _sg_first_pair(conv)
            if not human or not gpt:
                continue
            human = _HTML_TAG.sub("", human).strip()
            gpt = _HTML_TAG.sub("", gpt).strip()
            if len(human) < 4 or len(gpt) < 4:
                continue
            msgs = [{"role": "user", "content": human},
                    {"role": "assistant", "content": gpt}]
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False)
            except Exception:  # noqa: BLE001
                text = f"{human}\n\n{gpt}"
            if text in seen:
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) >= need:
                break
        if len(texts) >= need:
            break
    if not used_repo:
        raise RuntimeError("no ShareGPT-style dataset could be loaded")
    if len(texts) < need:
        print(f"[corpus] WARNING: only {len(texts)} ShareGPT pairs from {used_repo} "
              f"< requested {need}; using all.", flush=True)
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval : n_eval + n_train]
    assert set(eval_texts).isdisjoint(set(train_texts)), "SG train/eval overlap!"
    print(f"[corpus] ShareGPT({used_repo}): {len(train_texts)} train + "
          f"{len(eval_texts)} eval (disjoint)", flush=True)
    return eval_texts, train_texts, used_repo


@torch.no_grad()
def collect(model, text_model, tokenizer, texts, aux_layers, max_tokens, batch,
            device, source="math"):
    """Run forwards and return a list of per-sample corpus dicts tagged by source."""
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
            records.append({"input_ids": ids, "aux": aux, "next_token_ids": nxt,
                            "source": source})
        done = min(start + batch, n)
        if done % 20 == 0 or done == n:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"  [{done}/{n}] {time.time()-t0:.0f}s peakmem={mem:.1f}GB "
                f"records={len(records)}",
                flush=True,
            )
    return records


def source_stats(records):
    """Return {source: (n_records, n_tokens)} for logging/reporting."""
    out = {}
    for r in records:
        src = r.get("source", "unknown")
        n, t = out.get(src, (0, 0))
        out[src] = (n + 1, t + int(r["input_ids"].shape[0]))
    return out


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
    by_src = source_stats(records)
    src_str = ", ".join(f"{s}={n}rec/{t}tok" for s, (n, t) in sorted(by_src.items()))
    print(
        f"  sanity OK: {len(records)} records, no NaN/Inf, shift consistent; "
        f"len min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}; "
        f"by source: {src_str}",
        flush=True,
    )


def _save_per_source_eval(ev, meta, eval_out):
    """Write per-source held-out files (..._math.pt / ..._sharegpt.pt) so eval can
    report acceptance for each distribution separately."""
    base, ext = os.path.splitext(eval_out)
    for src in sorted({r.get("source", "unknown") for r in ev}):
        sub = [r for r in ev if r.get("source") == src]
        if not sub:
            continue
        path = f"{base}_{src}{ext}"
        torch.save({"records": sub, "meta": meta}, path)
        n_tok = sum(int(r["input_ids"].shape[0]) for r in sub)
        print(f"[corpus] saved {len(sub)} {src} eval records ({n_tok} tok) -> {path}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--dataset", default="EleutherAI/hendrycks_math")
    ap.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    # MATH counts: train from the `train` split, held-out from `test` (disjoint).
    ap.add_argument("--n_math_train", type=int, default=100000,
                    help="MATH train samples; auto-capped to the available pool")
    ap.add_argument("--n_math_eval", type=int, default=200,
                    help="MATH held-out samples drawn from --math_eval_split")
    ap.add_argument("--math_eval_split", default="test",
                    help="MATH split for the held-out (no `validation` split exists)")
    # ShareGPT-style conversational source.
    ap.add_argument("--n_sharegpt_train", type=int, default=1500)
    ap.add_argument("--n_sharegpt_eval", type=int, default=50)
    ap.add_argument("--sharegpt_repo", default="RyokoAI/ShareGPT52K")
    # Reuse an existing MATH train corpus instead of re-running its forward passes.
    ap.add_argument("--reuse_math_train", default=None,
                    help="path to an existing MATH train .pt; skips MATH train gen")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux_layers", type=int, nargs=3, default=[2, 21, 39])
    ap.add_argument(
        "--out", default="research/eagle3_drafter/train_data/full_train_corpus.pt")
    ap.add_argument(
        "--eval_out",
        default="research/eagle3_drafter/train_data/full_holdout_corpus.pt")
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
    torch.cuda.reset_peak_memory_stats()

    train, ev = [], []

    # ---- MATH train (train split) ----
    if args.reuse_math_train and os.path.exists(args.reuse_math_train):
        blob = torch.load(args.reuse_math_train, map_location="cpu", weights_only=False)
        math_train = blob["records"]
        for r in math_train:
            r.setdefault("source", "math")
        print(f"[corpus] reused {len(math_train)} MATH train records from "
              f"{args.reuse_math_train}", flush=True)
        train += math_train
    else:
        math_train_texts = build_texts(
            tokenizer, args.dataset, args.subjects, args.n_math_train, args.seed,
            split="train")
        print(f"[corpus] MATH train: {len(math_train_texts)} texts "
              f"({args.dataset}:train {args.subjects})", flush=True)
        print("[corpus] collecting MATH TRAIN ...", flush=True)
        math_train = collect(model, text_model, tokenizer, math_train_texts,
                             args.aux_layers, args.max_tokens, args.batch, device,
                             source="math")
        sanity_check(math_train, args.aux_layers)
        train += math_train

    # ---- MATH held-out (test split, split-disjoint from train) ----
    math_eval_texts = build_texts(
        tokenizer, args.dataset, args.subjects, args.n_math_eval, args.seed + 7,
        split=args.math_eval_split)
    print(f"[corpus] MATH eval: {len(math_eval_texts)} texts "
          f"({args.dataset}:{args.math_eval_split})", flush=True)
    print("[corpus] collecting MATH EVAL ...", flush=True)
    math_eval = collect(model, text_model, tokenizer, math_eval_texts,
                        args.aux_layers, args.max_tokens, args.batch, device,
                        source="math")
    sanity_check(math_eval, args.aux_layers)
    ev += math_eval

    # ---- ShareGPT train + held-out ----
    if args.n_sharegpt_train > 0 or args.n_sharegpt_eval > 0:
        sg_eval_texts, sg_train_texts, sg_repo = build_sharegpt_texts(
            tokenizer, args.n_sharegpt_eval, args.n_sharegpt_train,
            repo=args.sharegpt_repo)
        if sg_train_texts:
            print("[corpus] collecting SHAREGPT TRAIN ...", flush=True)
            sg_train = collect(model, text_model, tokenizer, sg_train_texts,
                              args.aux_layers, args.max_tokens, args.batch, device,
                              source="sharegpt")
            sanity_check(sg_train, args.aux_layers)
            train += sg_train
        if sg_eval_texts:
            print("[corpus] collecting SHAREGPT EVAL ...", flush=True)
            sg_eval = collect(model, text_model, tokenizer, sg_eval_texts,
                             args.aux_layers, args.max_tokens, args.batch, device,
                             source="sharegpt")
            sanity_check(sg_eval, args.aux_layers)
            ev += sg_eval
    else:
        sg_repo = None

    # ---- Token-id-level de-contamination across the full train/eval ----
    # ShareGPT contains exact-duplicate conversations, so a train record can
    # share its full token-id sequence with a held-out record. Drop the
    # contaminating train copies (eval is the stable measurement set, keep it
    # fixed) rather than aborting the whole collection.
    eval_keys = {tuple(r["input_ids"].tolist()) for r in ev}
    n_before = len(train)
    train = [r for r in train if tuple(r["input_ids"].tolist()) not in eval_keys]
    n_dropped = n_before - len(train)
    print(f"  de-contam: dropped {n_dropped}/{n_before} train records that "
          f"matched any of {len(eval_keys)} eval id-sequences; "
          f"{len(train)} train records remain", flush=True)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "subjects": args.subjects,
        "aux_layers": args.aux_layers,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "hidden_size": 2560,
        "vocab_size": 262144,
        "sources": ["math", "sharegpt"],
        "sharegpt_repo": sg_repo,
        "math_eval_split": args.math_eval_split,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"records": train, "meta": meta}, args.out)
    torch.save({"records": ev, "meta": meta}, args.eval_out)
    _save_per_source_eval(ev, meta, args.eval_out)

    tr = source_stats(train)
    evs = source_stats(ev)
    tr_tok = sum(int(r["input_ids"].shape[0]) for r in train)
    print(f"[corpus] saved {len(train)} TRAIN records ({tr_tok} tokens) -> {args.out}",
          flush=True)
    print(f"[corpus]   train by source: {tr}", flush=True)
    print(f"[corpus] saved {len(ev)} EVAL records -> {args.eval_out}", flush=True)
    print(f"[corpus]   eval by source: {evs}", flush=True)
    print(f"[corpus] peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)
    print("[corpus] done.", flush=True)


if __name__ == "__main__":
    main()
