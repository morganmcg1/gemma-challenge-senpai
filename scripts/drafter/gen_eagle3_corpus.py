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


# --------------------------------------------------------------------------- #
# Benchmark-matched reasoning source (PR #34) — greedy self-distillation
# --------------------------------------------------------------------------- #
# The official speed benchmark (eval_prompts_sharegpt.json) is 100% reasoning in
# THREE rigid templates: MMLU-Pro (A-J MCQ), GPQA (A-D MCQ), AIME (step-by-step
# math). PR #25 trained on raw MATH solutions + ShareGPT chat -> the drafter never
# saw the serve-time prompt distribution and acceptance plateaued ~0.73. Here we
# (a) render OTHER questions from the same datasets into the EXACT benchmark
# templates, (b) generate the assistant continuation by GREEDY decoding from the
# served target itself (self-distillation; EAGLE-1 2401.15077 / EAGLE-3 2503.01840
# train on target-generated, not dataset ground-truth, text), and (c) mark only
# the generated response tokens as loss targets (response-only masking is the
# canonical EAGLE loss mask; the prompt is context, never drafted at serve time).
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# GPQA proxy draws graduate-science MMLU-Pro; mmlu_pro source uses the rest so the
# two reasoning streams stay disjoint (no question rendered twice).
_SCI_CATS = {"physics", "chemistry", "biology"}


def _norm_text(s: str) -> str:
    """Whitespace/case-insensitive normalization for decontamination + dedup."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def render_mmlu_pro(question: str, options) -> str:
    """A-J MCQ template, byte-matched to eval_prompts_sharegpt.json mmlu_pro rows."""
    opts = "\n".join(f"{_LETTERS[i]}) {o}" for i, o in enumerate(options))
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: $LETTER' (without "
        "quotes) where LETTER is one of A,B,C,D,E,F,G,H,I,J. Think step by step "
        "before answering.\n\nQuestion:\n" + question + "\nOptions:\n" + opts
    )


def render_gpqa(question: str, four_opts) -> str:
    """A-D MCQ template, byte-matched to eval_prompts_sharegpt.json gpqa rows."""
    opts = "\n".join(f"{_LETTERS[i]}) {o}" for i, o in enumerate(four_opts))
    return (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: $LETTER' (without "
        "quotes) where LETTER is one of A,B,C,D. Think step by step before "
        "answering.\n\n" + question + "\n\n" + opts
    )


def render_aime(problem: str) -> str:
    """AIME step-by-step template, byte-matched to eval_prompts_sharegpt.json aime."""
    return (
        "Solve the following math problem step by step.\nThe last line of your "
        'response should be of the form "ANSWER: $ANSWER" (without quotes) where '
        "$ANSWER is the answer to the problem.\n\n" + problem + "\n\nRemember to "
        'put your answer on its own line at the end in the form "ANSWER: $ANSWER" '
        "(without quotes) where $ANSWER is the answer to the problem, and you do "
        "not need to use a \\boxed command."
    )


def _extract_eval_question(id_: str, human: str):
    """Pull the raw question/problem body out of an eval prompt so a training
    question (which we normalize raw) can be matched against it. Reliable for
    mmlu_pro (the real overlap risk: eval mmlu_pro rows come from the SAME
    TIGER-Lab/MMLU-Pro pool we sample) and aime; gpqa relies on full-text match."""
    if id_.startswith("mmlu_pro"):
        m = re.search(r"Question:\n(.*?)\nOptions:\n", human, re.S)
        return m.group(1) if m else None
    if id_.startswith("aime"):
        m = re.search(r"the problem\.\n\n(.*?)\n\nRemember", human, re.S)
        return m.group(1) if m else None
    return None


def load_eval_decontam(path: str):
    """Load the 128 official eval prompts and return (full_norms, q_norms).

    HARD requirement (PR #34): fail loud if the eval set can't be loaded — never
    silently skip dedup. full_norms = normalized full user prompts; q_norms =
    normalized raw question/problem bodies (mmlu_pro/aime)."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"[corpus] eval prompts for decontam not found: {path!r}. Refusing to "
            "build a reasoning corpus without de-contamination against the 128 "
            "official eval prompts.")
    data = json.load(open(path))
    full_norms, q_norms = set(), set()
    for x in data:
        human = None
        for t in x.get("conversations", []):
            if t.get("from") == "human":
                human = t.get("value")
                break
        if not human:
            continue
        full_norms.add(_norm_text(human))
        body = _extract_eval_question(x.get("id", ""), human)
        if body:
            q_norms.add(_norm_text(body))
    if not full_norms:
        raise RuntimeError("[corpus] eval decontam set is empty; refusing to proceed")
    print(f"[corpus] decontam: {len(full_norms)} eval prompts loaded from {path} "
          f"({len(q_norms)} question bodies extracted)", flush=True)
    return full_norms, q_norms


def _accept_prompt(rawq, rendered, full_norms, q_norms, seen_q):
    """Return the dedup key if (src question) is novel + uncontaminated, else None."""
    nq = _norm_text(rawq)
    if nq in q_norms:            # matches an eval question body (decontam)
        return None
    if nq in seen_q:             # already used by another reasoning stream
        return None
    if _norm_text(rendered) in full_norms:   # exact rendered-prompt match (decontam)
        return None
    return nq


def iter_mmlu_pro(seed, full_norms, q_norms, seen_q, sci_only: bool):
    """Yield (source, raw_question, rendered_prompt) from TIGER-Lab/MMLU-Pro.

    sci_only=True -> GPQA A-D proxy over physics/chemistry/biology (correct option
    + 3 distractors, deterministically shuffled). sci_only=False -> mmlu_pro A-J
    over the non-science categories (disjoint from the proxy stream)."""
    import random as _r

    from datasets import concatenate_datasets, load_dataset

    parts = [load_dataset("TIGER-Lab/MMLU-Pro", split=s) for s in ("test", "validation")]
    ds = concatenate_datasets(parts).shuffle(seed=seed)
    src = "gpqa" if sci_only else "mmlu_pro"
    for ex in ds:
        cat = ex.get("category")
        in_sci = cat in _SCI_CATS
        if sci_only != in_sci:
            continue
        q = (ex.get("question") or "").strip()
        opts = list(ex.get("options") or [])
        if not q or len(opts) < (4 if sci_only else 2):
            continue
        if sci_only:
            ai = ex.get("answer_index")
            if not isinstance(ai, int) or ai >= len(opts):
                continue
            correct = opts[ai]
            distract = [o for i, o in enumerate(opts) if i != ai]
            rng = _r.Random(hash(q) & 0xFFFFFFFF)
            rng.shuffle(distract)
            four = [correct] + distract[:3]
            rng.shuffle(four)
            rendered = render_gpqa(q, four)
        else:
            rendered = render_mmlu_pro(q, opts)
        key = _accept_prompt(q, rendered, full_norms, q_norms, seen_q)
        if key is None:
            continue
        seen_q.add(key)
        yield (src, q, rendered)


def iter_aime(seed, full_norms, q_norms, seen_q):
    """Yield (source, raw_problem, rendered_prompt) from AI-MO/NuminaMath-CoT,
    filtered to competition-math sources (AIME-distribution proxy)."""
    from datasets import load_dataset

    comp = {"amc_aime", "aops_forum", "olympiads", "synthetic_amc"}
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    for ex in ds:
        if ex.get("source") not in comp:
            continue
        prob = (ex.get("problem") or "").strip()
        if len(prob) < 16:
            continue
        rendered = render_aime(prob)
        key = _accept_prompt(prob, rendered, full_norms, q_norms, seen_q)
        if key is None:
            continue
        seen_q.add(key)
        yield ("aime", prob, rendered)


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, max_new_tokens, gen_batch,
                         device, stop_ids):
    """Greedy (temp=0) self-distillation: generate the target's own assistant
    continuation for each rendered user prompt. Returns a list of
    {full_ids[L], prompt_len} where full_ids = prompt tokens + generated tokens
    (trimmed at the first stop token). Left-padded batched generation."""
    results = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id
    for s in range(0, len(prompts), gen_batch):
        chunk = prompts[s:s + gen_batch]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
            for p in chunk
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        plen_full = input_ids.shape[1]
        out = model.generate(
            input_ids=input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            use_cache=True, pad_token_id=pad_id, eos_token_id=stop_ids)
        for b in range(len(chunk)):
            plen = int(attn[b].sum().item())
            prompt_ids = input_ids[b, plen_full - plen:plen_full].cpu()
            gen = out[b, plen_full:].tolist()
            trimmed = []
            for t in gen:
                if t == pad_id:
                    break
                trimmed.append(t)
                if t in stop_ids:
                    break
            if len(trimmed) < 1:
                continue
            full = torch.cat([prompt_ids,
                              torch.tensor(trimmed, dtype=prompt_ids.dtype)])
            results.append({"full_ids": full.to(torch.long), "prompt_len": int(plen)})
    tokenizer.padding_side = old_side
    return results


@torch.no_grad()
def collect_from_ids(text_model, gens, source, aux_layers, batch, device,
                     max_tokens, min_resp=3):
    """Teacher-forced forward over pre-generated full sequences to record the 3
    aux hidden states, with RESPONSE-ONLY loss masking (prompt next-token labels
    set to IGNORE; only the generated continuation is scored/trained)."""
    records = []
    n = len(gens)
    t0 = time.time()
    for s in range(0, n, batch):
        chunk = gens[s:s + batch]
        seqs = [g["full_ids"][:max_tokens] for g in chunk]
        plens = [min(g["prompt_len"], int(seq.shape[0])) for g, seq in zip(chunk, seqs)]
        T = max(int(x.shape[0]) for x in seqs)
        input_ids = torch.zeros(len(seqs), T, dtype=torch.long)
        attn = torch.zeros(len(seqs), T, dtype=torch.long)
        for b, seq in enumerate(seqs):
            L = int(seq.shape[0])
            input_ids[b, :L] = seq
            attn[b, :L] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        out = text_model(input_ids=input_ids, attention_mask=attn,
                         output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        for b in range(len(seqs)):
            L = int(seqs[b].shape[0])
            plen = plens[b]
            if L - plen < min_resp:
                continue
            ids = input_ids[b, :L].to(torch.int32).cpu()
            aux = torch.stack(
                [hs[li][b, :L].to(torch.float16).cpu() for li in aux_layers], dim=0)
            nxt = torch.full((L,), IGNORE, dtype=torch.int64)
            nxt[:-1] = ids[1:].to(torch.int64)
            if plen >= 1:
                nxt[:plen - 1] = IGNORE  # response-only: drop prompt-token labels
            records.append({"input_ids": ids, "aux": aux, "next_token_ids": nxt,
                            "source": source, "prompt_len": int(plen)})
        done = min(s + batch, n)
        if done % 40 == 0 or done == n:
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"    collect[{source}] {done}/{n} {time.time()-t0:.0f}s "
                  f"peakmem={mem:.1f}GB records={len(records)}", flush=True)
    return records


def _resp_tokens(records):
    """Total scored (non-IGNORE) response tokens — the EAGLE loss-mask token count."""
    return sum(int((r["next_token_ids"] != IGNORE).sum().item()) for r in records)


def build_reasoning_corpus(model, text_model, tokenizer, args, device):
    """PR #34 orchestrator: build a benchmark-matched, target-self-distilled
    reasoning corpus. Reserves a proportional, disjoint holdout first, then fills
    each source's TRAIN stream up to its token-volume budget."""
    full_norms, q_norms = load_eval_decontam(args.eval_prompts)
    mix = {}
    for part in args.reasoning_mix.split(","):
        k, v = part.split(":")
        mix[k.strip()] = float(v)
    tot = sum(mix.values())
    mix = {k: v / tot for k, v in mix.items()}
    print(f"[corpus] reasoning mix (token-volume): {mix}", flush=True)

    # Stop at the target's own turn-boundary tokens so the drafter learns to emit
    # them. gemma-4-E4B-it uses a <|turn>/<turn|> chat format (NOT <end_of_turn>);
    # generation_config.eos_token_id is the authoritative stop set ([1, 106, 50]).
    gc_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stop_ids = set()
    if isinstance(gc_eos, int):
        stop_ids.add(int(gc_eos))
    elif isinstance(gc_eos, (list, tuple)):
        stop_ids.update(int(x) for x in gc_eos)
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    stop_ids = sorted(stop_ids)
    print(f"[corpus] greedy stop_ids={stop_ids} max_new_tokens={args.max_new_tokens}",
          flush=True)

    seen_q = set()
    iters = {
        "gpqa": iter_mmlu_pro(args.seed + 1, full_norms, q_norms, seen_q, sci_only=True),
        "mmlu_pro": iter_mmlu_pro(args.seed, full_norms, q_norms, seen_q, sci_only=False),
        "aime": iter_aime(args.seed + 3, full_norms, q_norms, seen_q),
    }

    def gen_and_collect(rendered_prompts, src):
        gens = generate_completions(model, tokenizer, rendered_prompts,
                                    args.max_new_tokens, args.gen_batch, device, stop_ids)
        return collect_from_ids(text_model, gens, src, args.aux_layers,
                                args.collect_batch, device, args.max_tokens)

    # ---- 1) proportional, disjoint holdout (>=200 prompts) ----
    ev = []
    for src in ("mmlu_pro", "gpqa", "aime"):
        k = max(1, round(args.n_reasoning_eval * mix.get(src, 0.0)))
        prompts = []
        for _ in range(k):
            try:
                prompts.append(next(iters[src])[2])
            except StopIteration:
                break
        print(f"[corpus] HOLDOUT {src}: generating {len(prompts)} prompts ...", flush=True)
        ev += gen_and_collect(prompts, src)
    sanity_check(ev, args.aux_layers)
    print(f"[corpus] holdout: {len(ev)} records, {_resp_tokens(ev)} response tokens",
          flush=True)

    meta = {
        "model": args.model, "aux_layers": args.aux_layers,
        "max_tokens": args.max_tokens, "max_new_tokens": args.max_new_tokens,
        "seed": args.seed, "hidden_size": 2560, "vocab_size": 262144,
        "sources": ["mmlu_pro", "gpqa", "aime"], "reasoning_mix": mix,
        "self_distilled": True, "response_only_mask": True,
        "gpqa_proxy": "TIGER-Lab/MMLU-Pro physics+chemistry+biology, A-D",
        "aime_source": "AI-MO/NuminaMath-CoT competition subsets",
        "eval_prompts": args.eval_prompts,
    }
    # Save the holdout NOW: it is generated first but is the expensive, fixed
    # measurement slice. Persisting it before the long train-stream loop means a
    # mid-run interruption costs only the (re-runnable) train data, not the eval.
    os.makedirs(os.path.dirname(args.eval_out), exist_ok=True)
    torch.save({"records": ev, "meta": meta}, args.eval_out)
    _save_per_source_eval(ev, meta, args.eval_out)
    print(f"[corpus] holdout saved early -> {args.eval_out}", flush=True)

    # ---- 2) train streams to token-volume budget ----
    train = []
    for src in ("mmlu_pro", "gpqa", "aime"):
        budget = args.reasoning_tokens * mix.get(src, 0.0)
        acc, exhausted = 0, False
        print(f"[corpus] TRAIN {src}: budget {budget/1e6:.2f}M tokens ...", flush=True)
        while acc < budget and not exhausted:
            prompts = []
            for _ in range(args.gen_batch):
                try:
                    prompts.append(next(iters[src])[2])
                except StopIteration:
                    exhausted = True
                    break
            if not prompts:
                break
            recs = gen_and_collect(prompts, src)
            train += recs
            acc += sum(int(r["input_ids"].shape[0]) for r in recs)
            if (len(train) % 200) < args.gen_batch:
                print(f"    {src}: {acc/1e6:.2f}M/{budget/1e6:.2f}M tok "
                      f"({len(train)} recs total)", flush=True)
        if exhausted:
            print(f"[corpus] WARNING: {src} source exhausted at {acc/1e6:.2f}M tokens "
                  f"(< budget {budget/1e6:.2f}M)", flush=True)
    sanity_check(train, args.aux_layers)

    # ---- 3) token-id de-contamination backstop (eval is the fixed measurement) --
    eval_keys = {tuple(r["input_ids"].tolist()) for r in ev}
    n_before = len(train)
    train = [r for r in train if tuple(r["input_ids"].tolist()) not in eval_keys]
    print(f"[corpus] de-contam backstop: dropped {n_before - len(train)}/{n_before} "
          f"train records matching a holdout id-sequence", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"records": train, "meta": meta}, args.out)

    tr_by = source_stats(train)
    ev_by = source_stats(ev)
    print(f"[corpus] saved {len(train)} TRAIN records "
          f"({sum(int(r['input_ids'].shape[0]) for r in train)} tok, "
          f"{_resp_tokens(train)} response-tok) -> {args.out}", flush=True)
    print(f"[corpus]   train by source: {tr_by}", flush=True)
    print(f"[corpus] saved {len(ev)} EVAL records "
          f"({_resp_tokens(ev)} response-tok) -> {args.eval_out}", flush=True)
    print(f"[corpus]   eval by source: {ev_by}", flush=True)
    print(f"[corpus] peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
          flush=True)
    print("[corpus] done (reasoning).", flush=True)


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
        # input_ids[1:] must equal next_token_ids[:-1] at every SCORED position.
        # (response-only masking, PR #34, sets prompt next-token labels to IGNORE,
        # so only check the shift invariant where nxt != IGNORE; legacy full-seq
        # corpora have all positions scored -> unchanged.)
        m = nxt[:-1] != IGNORE
        assert torch.equal(ids[1:].to(torch.int64)[m], nxt[:-1][m]), "shift mismatch"
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
    # ---- PR #34: benchmark-matched reasoning corpus (greedy self-distillation) --
    ap.add_argument("--reasoning_mix", default=None,
                    help="enables reasoning mode; token-volume mix, e.g. "
                         "'mmlu_pro:0.445,gpqa:0.445,aime:0.11'")
    ap.add_argument("--reasoning_tokens", type=float, default=3_000_000,
                    help="total TRAIN token budget for the reasoning corpus")
    ap.add_argument("--n_reasoning_eval", type=int, default=240,
                    help="held-out benchmark-matched prompt count (proportional split)")
    ap.add_argument("--eval_prompts",
                    default="official/main_bucket/shared_resources/speed_benchmark/"
                            "data/eval_prompts_sharegpt.json",
                    help="128 official eval prompts: decontam + holdout disjointness")
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                    help="greedy generation cap for the self-distilled CoT")
    ap.add_argument("--gen_batch", type=int, default=8,
                    help="batch size for greedy generation (left-padded)")
    ap.add_argument("--collect_batch", type=int, default=4,
                    help="batch size for the teacher-forced aux-collect forward")
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

    # ---- PR #34: benchmark-matched reasoning corpus (greedy self-distillation) --
    if args.reasoning_mix:
        build_reasoning_corpus(model, text_model, tokenizer, args, device)
        return

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
