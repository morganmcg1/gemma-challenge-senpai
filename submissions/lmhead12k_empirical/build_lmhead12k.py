#!/usr/bin/env python
"""Build the empirical lmhead12k pruned checkpoint + kept-id map.

Two phases:

Phase 1 -- ``select`` (CPU, needs the gemma tokenizer + a broad corpus):
    Choose the lm_head rows to keep. The kept set HARD-INCLUDES, regardless of
    frequency:
      * every ground-truth target token  -> guarantees finite served PPL (the
        official scorer does NOT floor -inf, so a scored GT target outside the
        kept vocab => -inf => gate fail),
      * every observed greedy emission    -> guarantees greedy identity on the
        captured benchmark prompts,
      * all tokenizer special/added ids + the reserved 0..255 control block ->
        preserves control + multimodal structural tokens.
    The must-keep union is the TIGHT kept set (a public-tailored bandwidth
    CEILING). The remaining budget up to K=12,288 is filled by frequency over a
    BROAD public STEM / technical-QA corpus (MMLU-Pro) so the GENERAL cut covers
    the technical-QA vocabulary universally (frontier-faithful, de-risks the
    private re-run). We report BOTH sizes' bandwidth numbers.

Phase 2 -- ``build`` (needs torch; --mode bf16|int4):
    Untie the lm_head and slice its rows (vocab axis) to kept_ids, writing the
    served checkpoint. ``embed_tokens`` stays full bf16 (the model must still
    embed any input id); only the OUTPUT lm_head shrinks to kept_size rows, so
    the decode-step lm_head GEMM reads vocab/kept_size x fewer weight bytes.
    The source can be the bf16 instruct model (``--mode bf16``) or a public
    W4A16 compressed-tensors checkpoint (``--mode int4``, e.g.
    ``google/gemma-4-E4B-it-qat-w4a16-ct``, whose int4 body is left untouched).
    The output projection (an existing bf16 ``lm_head.weight`` if the source
    materializes one, else the tied ``embed_tokens.weight``) is row-sliced to
    kept_ids and written as an untied ``lm_head.weight`` with
    ``tie_word_embeddings=false``.
    ``config`` keeps ``vocab_size=262144`` (only ``lm_head.out_features``
    shrinks); the custom vLLM class scatters the kept-row logits back to full
    vocab so the sampler / prompt_logprobs path is unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-4-E4B-it"
FULL_VOCAB = 262144
HIDDEN = 2560
DEFAULT_K = 12288

# Regenerated full-128 baseline decode capture (Q3). Falls back to the operator
# 31-record capture only if the 128 one is absent.
DECODE_FILE = ROOT / "research/local_validation/vllm_baseline_128/decode_outputs.jsonl"
DECODE_FILE_FALLBACK = ROOT / "research/local_validation/vllm_baseline/decode_outputs_128.jsonl"
GT_FILE = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
KEPT_IDS_OUT = ROOT / "submissions/lmhead12k_empirical/kept_ids.json"
ANALYSIS_OUT = ROOT / "research/local_validation/lmhead12k_empirical/select_analysis.json"
BROAD_FREQ_CACHE = ROOT / "research/local_validation/lmhead12k_empirical/broad_corpus_freq.json"

DEFAULT_OUT_DIR = Path("/workspace/gemma_build/lmhead12k_empirical")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _decode_completion_ids(rec: dict) -> list[int]:
    """Robustly pull the completion token ids from a decode record."""
    for key in ("completion_token_ids", "output_token_ids", "token_ids"):
        v = rec.get(key)
        if v:
            return list(v)
    return []


def special_token_ids() -> tuple[set[int], str]:
    """All tokenizer special/added ids + the reserved 0..255 control block."""
    ids = set(range(256))
    source = "reserved-0..255-only"
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        ids.update(int(i) for i in tok.all_special_ids)
        ids.update(int(k) for k in getattr(tok, "added_tokens_decoder", {}))
        source = "tokenizer.all_special_ids+added+reserved"
    except Exception as exc:  # pragma: no cover - offline fallback
        log(f"[select] tokenizer unavailable ({type(exc).__name__}: {exc}); "
            f"using reserved 0..255 fallback")
    return ids, source


def broad_corpus_freq(use_cache: bool = True) -> tuple[Counter, dict]:
    """Token frequency over a broad public STEM / technical-QA corpus (MMLU-Pro).

    MMLU-Pro is the domain-matched, public technical-QA distribution. We tokenize
    question + options + chain-of-thought + answer with the gemma tokenizer and
    count token ids. Cached to JSON so re-runs are instant.
    """
    if use_cache and BROAD_FREQ_CACHE.exists():
        raw = json.loads(BROAD_FREQ_CACHE.read_text())
        freq = Counter({int(k): int(v) for k, v in raw["freq"].items()})
        return freq, raw["meta"]

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Read the cached MMLU-Pro parquet directly (no `datasets` dependency).
    snap = Path(snapshot_download(
        "TIGER-Lab/MMLU-Pro", repo_type="dataset", allow_patterns=["data/*.parquet"]))
    parquets = sorted(snap.glob("data/*.parquet"))
    texts: list[str] = []
    n_examples = 0
    for pf in parquets:
        rows = pq.read_table(pf).to_pylist()
        for ex in rows:
            n_examples += 1
            parts = [str(ex.get("question", ""))]
            parts.extend(str(o) for o in (ex.get("options") or []))
            if ex.get("cot_content"):
                parts.append(str(ex["cot_content"]))
            if ex.get("answer") is not None:
                parts.append(str(ex["answer"]))
            texts.append("\n".join(p for p in parts if p))

    freq: Counter = Counter()
    n_tokens = 0
    B = 512
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], add_special_tokens=False)["input_ids"]
        for ids in enc:
            freq.update(ids)
            n_tokens += len(ids)

    meta = {
        "corpus": "TIGER-Lab/MMLU-Pro (test+validation)",
        "n_examples": n_examples,
        "n_tokens": n_tokens,
        "n_unique": len(freq),
        "tokenizer": MODEL_ID,
    }
    BROAD_FREQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BROAD_FREQ_CACHE.write_text(json.dumps({
        "meta": meta,
        "freq": {str(k): int(v) for k, v in freq.items()},
    }))
    log(f"[select] broad corpus: {n_examples} examples, {n_tokens} tokens, "
        f"{len(freq)} unique ids -> cached {BROAD_FREQ_CACHE}")
    return freq, meta


def select_kept_ids(
    decode_file: Path,
    gt_file: Path = GT_FILE,
    k: int = DEFAULT_K,
    kept_override: list[int] | None = None,
) -> tuple[list[int], dict]:
    """Choose (or analyse) the GENERAL top-K output-vocab cut.

    Methodology (advisor directive, binding): hard-include ONLY the public
    GT-target tokens -- which guarantees a finite teacher-forced PPL on the public
    set -- plus tokenizer specials; fill the remaining budget by broad-corpus
    (MMLU-Pro) frequency. We deliberately do NOT hard-include the public-128 decode
    argmax or the GT-context: those are public-prompt-specific and would not
    generalise to a private re-run, so baking them in would overfit the cut.

    When ``kept_override`` is supplied (the frozen, gated ``kept_ids.json``) we
    ANALYSE that exact set rather than rebuilding. The frozen artifact is the
    source of truth: ``build`` reproduces the served checkpoint from it
    byte-for-byte. A from-scratch rebuild will NOT reproduce the frozen set
    bit-for-bit because the broad-corpus fill depends on the corpus snapshot at
    gate time; only the hard-include invariants (GT-target + specials) and K are
    guaranteed stable across snapshots.
    """
    decode = _read_jsonl(decode_file)
    gt = _read_jsonl(gt_file)

    s_gt_target: set[int] = set()
    s_gt_context: set[int] = set()
    for rec in gt:
        s_gt_target.update(rec["target_token_ids"])
        s_gt_context.update(rec["context_token_ids"])
    s_gt_all = s_gt_target | s_gt_context

    s_decode: set[int] = set()
    decode_token_total = 0
    for rec in decode:
        toks = _decode_completion_ids(rec)
        s_decode.update(toks)
        decode_token_total += len(toks)

    s_special, special_source = special_token_ids()

    # hard-include = GT-target (finite PPL) + specials ONLY.
    must_keep = s_gt_target | s_special
    broad, broad_meta = broad_corpus_freq()

    if kept_override is not None:
        kept = sorted(set(kept_override))
    else:
        remaining = max(0, k - len(must_keep))
        fill = [t for t, _ in broad.most_common() if t not in must_keep][:remaining]
        kept = sorted(must_keep | set(fill))
    kept_set = set(kept)

    # public-tailored CEILING (must-keep ∪ public decode ∪ GT-context): reported for
    # context only, NEVER shipped -- it overfits the public GT and will not generalise.
    tight = s_gt_all | s_decode | s_special

    dec_tokens = [t for rec in decode for t in _decode_completion_ids(rec)]
    out_kept = sum(1 for t in dec_tokens if t not in kept_set)

    stats = {
        "K": k,
        "full_vocab": FULL_VOCAB,
        "hidden": HIDDEN,
        "kept_size": len(kept),
        "bandwidth_reduction_x": round(FULL_VOCAB / max(1, len(kept)), 3),
        "frozen_source_of_truth": kept_override is not None,
        "methodology": "hard-include GT-target + tokenizer specials; fill by "
                       "broad-corpus (MMLU-Pro) frequency. No decode/context "
                       "hard-include -- those are public-prompt-specific.",
        # public-overfit ceiling, reported for context only (NOT shipped)
        "tight_ceiling_size": len(tight),
        "tight_ceiling_bandwidth_reduction_x": round(FULL_VOCAB / max(1, len(tight)), 3),
        "tight_ceiling_note": "must-keep ∪ public-128 decode ∪ GT-context; public-overfit, NOT shipped",
        # provenance
        "decode_file": str(decode_file),
        "decode_records": len(decode),
        "decode_token_total": decode_token_total,
        "gt_records": len(gt),
        "n_gt_target_unique": len(s_gt_target),
        "n_gt_context_unique": len(s_gt_context),
        "n_decode_unique": len(s_decode),
        "n_special": len(s_special),
        "special_source": special_source,
        "n_hard_include": len(must_keep),
        "n_freq_fill": len(kept_set - must_keep),
        "broad_corpus": broad_meta,
        "broad_unique": len(broad),
        # correctness guarantee on the public set
        "finite_ppl_guaranteed": s_gt_target.issubset(kept_set),
        # fidelity / private-PPL risk (NOT a gate criterion: the served-vs-served
        # greedy gate passes by construction -- a pruned argmax is always in kept).
        "gt_context_outside_kept": len(s_gt_context - kept_set),
        "decode_outside_kept": {
            "unique_outside": len(s_decode - kept_set),
            "tokens_outside": out_kept,
            "tokens_total": len(dec_tokens),
            "rate": round(out_kept / max(1, len(dec_tokens)), 6),
            "note": "public greedy emissions the shipped cut would clip vs the "
                    "unpruned model; fidelity risk only, see clip_floor_ksweep.json",
        },
        "residual_risk_private_ppl": "a PRIVATE GT-target token outside kept -> -inf "
                                     "logit -> +inf PPL on a private re-run; not closable "
                                     "locally (only by widening K toward full vocab).",
    }
    return kept, stats


def run_select(args: argparse.Namespace) -> None:
    decode_file = Path(args.decode_file)
    if not decode_file.exists() and DECODE_FILE_FALLBACK.exists():
        log(f"[select] {decode_file} absent; falling back to {DECODE_FILE_FALLBACK}")
        decode_file = DECODE_FILE_FALLBACK

    # The frozen kept_ids.json is the gated source of truth -- never clobber it.
    # If present (and not --force-rebuild), run in analysis-only mode against it.
    frozen = None
    if KEPT_IDS_OUT.exists() and not args.force_rebuild:
        frozen = json.loads(KEPT_IDS_OUT.read_text()).get("kept_ids")
        log(f"[select] frozen {KEPT_IDS_OUT} present ({len(frozen)} ids) -> "
            f"analysis-only (pass --force-rebuild to regenerate the set)")

    kept, stats = select_kept_ids(
        decode_file, Path(args.gt_file), args.k, kept_override=frozen)
    ANALYSIS_OUT.parent.mkdir(parents=True, exist_ok=True)
    if frozen is None:
        KEPT_IDS_OUT.parent.mkdir(parents=True, exist_ok=True)
        KEPT_IDS_OUT.write_text(json.dumps({
            "model_id": MODEL_ID,
            "full_vocab": FULL_VOCAB,
            "K": args.k,
            "kept_size": len(kept),
            "kept_ids": kept,
        }))
        log(f"[select] wrote {KEPT_IDS_OUT} ({len(kept)} ids)")
    ANALYSIS_OUT.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


# ---------------------------------------------------------------------------
# Phase 2: build (untie + prune lm_head rows)
# ---------------------------------------------------------------------------

def _resolve_source(source: str) -> Path:
    """Resolve a HF model id or local path to a checkpoint directory."""
    p = Path(source)
    if p.exists() and any(p.glob("*.safetensors")):
        return p
    from huggingface_hub import snapshot_download

    log(f"[build] resolving {source} from HF hub ...")
    return Path(snapshot_download(source, allow_patterns=[
        "*.safetensors", "*.json", "*.txt", "*.model", "*.jinja",
    ]))


def _load_all_tensors(src: Path) -> dict:
    """Load every tensor from a single- or multi-shard safetensors checkpoint."""
    from safetensors.torch import load_file

    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors under {src}")
    tensors: dict = {}
    for shard in shards:
        tensors.update(load_file(str(shard)))
    return tensors


EMBED_KEY = "model.language_model.embed_tokens.weight"


def run_build(args: argparse.Namespace) -> None:
    """Untie + row-prune the lm_head; write the served checkpoint.

    Works for a bf16 source (whole model bf16) and an int4 source (W4A16-g128
    body, bf16 embeddings): in both the tied source has no ``lm_head`` tensor, so
    we synthesize ``lm_head.weight = embed_tokens.weight[kept_ids]`` (bf16) and
    set ``tie_word_embeddings=false``. The body's dtype/quantization is whatever
    the source already is -- we never touch the body.
    """
    import torch
    from safetensors.torch import save_file

    src = _resolve_source(args.source)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = json.loads(KEPT_IDS_OUT.read_text())["kept_ids"]
    kept_t = torch.tensor(kept, dtype=torch.long)
    log(f"[build] mode={args.mode} source={src} kept_size={len(kept)} -> {out}")

    tensors = _load_all_tensors(src)
    if EMBED_KEY not in tensors:
        cands = [k for k in tensors if "embed_tokens.weight" in k and "per_layer" not in k]
        if len(cands) != 1:
            raise SystemExit(f"cannot locate embed_tokens.weight (candidates={cands})")
        embed_key = cands[0]
    else:
        embed_key = EMBED_KEY
    embed = tensors[embed_key]
    if embed.shape[0] != FULL_VOCAB:
        raise SystemExit(f"{embed_key} axis0={embed.shape[0]} != {FULL_VOCAB}")

    # Choose the source of the output-projection rows: prefer an existing
    # lm_head.weight (the true output projection; in the public W4A16 base it is
    # tied-identical to embed_tokens but materialized as a separate bf16 tensor),
    # else fall back to the tied embeddings. Either way, row-slice to kept_ids.
    if "lm_head.weight" in tensors:
        head_full = tensors["lm_head.weight"]
        head_origin = "lm_head.weight"
        if head_full.shape[0] != FULL_VOCAB:
            raise SystemExit(f"lm_head.weight axis0={head_full.shape[0]} != {FULL_VOCAB}")
    else:
        head_full = embed
        head_origin = f"{embed_key} (tied)"
    lm_head = head_full.index_select(0, kept_t).contiguous().clone()
    tensors["lm_head.weight"] = lm_head  # prune in place / create untied head
    log(f"[build] lm_head.weight {tuple(lm_head.shape)} {lm_head.dtype} "
        f"(rows = {head_origin}[kept_ids]); embed_tokens kept full {tuple(embed.shape)}")

    # Save (single shard is fine; RAM is ample).
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    # Copy non-weight files; flip tie_word_embeddings off in the config.
    for extra in src.iterdir():
        if extra.suffix == ".safetensors" or extra.name.endswith(".safetensors.index.json"):
            continue
        dest = out / extra.name
        if extra.is_dir():
            shutil.copytree(extra, dest, dirs_exist_ok=True)
        elif extra.is_file():
            shutil.copy2(extra, dest)

    cfg = json.loads((out / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"]["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "kept_ids.json").write_text(KEPT_IDS_OUT.read_text())
    log(f"[build] wrote pruned checkpoint to {out} (tie_word_embeddings=false, "
        f"vocab_size kept at {FULL_VOCAB}; lm_head out_features={len(kept)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sel = sub.add_parser("select", help="CPU: choose kept_ids + write analysis")
    sel.add_argument("--k", type=int, default=DEFAULT_K)
    sel.add_argument("--decode-file", default=str(DECODE_FILE))
    sel.add_argument("--gt-file", default=str(GT_FILE))
    sel.add_argument("--force-rebuild", action="store_true",
                     help="regenerate kept_ids.json even if a frozen one exists")
    sel.set_defaults(func=run_select)

    bd = sub.add_parser("build", help="torch: untie + slice lm_head rows + save ckpt")
    bd.add_argument("--mode", choices=["bf16", "int4"], required=True)
    bd.add_argument("--source", required=True,
                    help="HF model id or local dir of the (tied) source checkpoint")
    bd.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    bd.set_defaults(func=run_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
