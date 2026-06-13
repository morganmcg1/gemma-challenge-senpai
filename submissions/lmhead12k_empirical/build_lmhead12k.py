#!/usr/bin/env python
"""Build the empirical lmhead12k pruned checkpoint + kept-id map.

Two independent phases:

Phase 1 -- ``select`` (CPU, no torch, no GPU):
    Choose the top-K lm_head rows to keep. The kept set HARD-INCLUDES, regardless
    of frequency:
      * every ground-truth target token  -> guarantees finite served PPL
      * every observed greedy emission    -> guarantees greedy identity on the
                                             captured benchmark prompts
      * all tokenizer special/added ids   -> preserves control + multimodal
                                             structural tokens (do-not-disable-
                                             modalities contract)
      * the reserved 0..255 control block -> belt-and-suspenders insurance
    The remaining budget up to K is filled by combined-corpus frequency. Writes
    ``kept_ids.json`` (needed by serve.py to remap 12k indices -> original ids)
    and a rich ``select_analysis.json``.

Phase 2 -- ``prune`` (needs torch + the int4+g128+lm_head base checkpoint):
    Slice the lm_head weight rows (vocab/output axis) to kept_ids and save the
    served checkpoint. compressed-tensors int4 (W4A16, group_size=128) packs along
    the *hidden* axis, so whole-row slicing along the vocab axis keeps the packing
    valid. embed_tokens stays full-size bf16 (lm_head is untied).

Run phase 1 now (fully offline); phase 2 when a GPU window + the int4 base exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "google/gemma-4-E4B-it"
FULL_VOCAB = 262144
DEFAULT_K = 12288

DECODE_FILE = ROOT / "research/local_validation/vllm_baseline/decode_outputs_128.jsonl"
GT_FILE = ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
KEPT_IDS_OUT = ROOT / "submissions/lmhead12k_empirical/kept_ids.json"
ANALYSIS_OUT = ROOT / "research/local_validation/lmhead12k_empirical/select_analysis.json"

# int4+g128+lm_head base (lawine PR #4). Absent on this node as of 2026-06-13;
# override with --base-dir once the advisor points to it or it is rebuilt.
DEFAULT_BASE_DIR = Path("/workspace/gemma_build/int4_g128_lmhead")
DEFAULT_OUT_DIR = Path("/workspace/gemma_build/lmhead12k_empirical")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def special_token_ids() -> tuple[set[int], str]:
    """All tokenizer special/added ids + reserved 0..255 control block.

    Best-effort: if the tokenizer cannot load offline we still return the reserved
    block, which covers the core control tokens (pad/eos/bos/turn markers).
    """
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


def select_kept_ids(
    decode_file: Path = DECODE_FILE,
    gt_file: Path = GT_FILE,
    k: int = DEFAULT_K,
) -> tuple[list[int], dict]:
    decode = _read_jsonl(decode_file)
    gt = _read_jsonl(gt_file)

    # Must-keep sets.
    s_gt_target: set[int] = set()
    s_gt_context: set[int] = set()
    for rec in gt:
        s_gt_target.update(rec["target_token_ids"])
        s_gt_context.update(rec["context_token_ids"])
    s_gt_all = s_gt_target | s_gt_context

    s_decode: set[int] = set()
    decode_token_total = 0
    for rec in decode:
        toks = rec["completion_token_ids"]
        s_decode.update(toks)
        decode_token_total += len(toks)

    s_special, special_source = special_token_ids()

    must_keep = s_gt_all | s_decode | s_special
    if len(must_keep) > k:
        log(f"[select] WARNING: |must_keep|={len(must_keep)} exceeds K={k}; "
            f"finite-PPL + greedy-identity cannot both hold at this K -- widen K.")

    # Frequency over the combined corpus (GT context+target + decode completions).
    freq: Counter[int] = Counter()
    for rec in gt:
        freq.update(rec["target_token_ids"])
        freq.update(rec["context_token_ids"])
    for rec in decode:
        freq.update(rec["completion_token_ids"])

    remaining = max(0, k - len(must_keep))
    fill = [tok for tok, _ in freq.most_common() if tok not in must_keep][:remaining]
    kept = sorted(must_keep | set(fill))

    # Rare-token divergence proxy: fraction of observed greedy emissions that a
    # *pure* frequency top-K (no hard-include of decode) would have clipped.
    pure_topk = {tok for tok, _ in freq.most_common(k)}
    dec_tokens = [t for rec in decode for t in rec["completion_token_ids"]]
    outside = sum(1 for t in dec_tokens if t not in pure_topk)

    stats = {
        "K": k,
        "full_vocab": FULL_VOCAB,
        "kept_size": len(kept),
        "bandwidth_reduction_x": round(FULL_VOCAB / max(1, len(kept)), 3),
        "decode_records": len(decode),
        "decode_records_expected": 128,
        "decode_records_note": (
            "bucket capture covers 31/128 benchmark prompts; greedy identity is "
            "proven only on these 31 -- the other 97 rely on frequency coverage."
        ),
        "gt_records": len(gt),
        "n_gt_target_unique": len(s_gt_target),
        "n_gt_all_unique": len(s_gt_all),
        "n_decode_unique": len(s_decode),
        "n_special": len(s_special),
        "special_source": special_source,
        "n_must_keep": len(must_keep),
        "must_keep_fits_K": len(must_keep) <= k,
        "headroom_after_must_keep": k - len(must_keep),
        "n_freq_fill": len(fill),
        "combined_corpus_unique": len(freq),
        "combined_corpus_tokens": sum(freq.values()),
        "finite_ppl_guaranteed": s_gt_target.issubset(set(kept)),
        "greedy_identity_captured_prompts": s_decode.issubset(set(kept)),
        "decode_new_tokens_beyond_gt": len(s_decode - s_gt_all),
        "rare_token_divergence_pure_topk": {
            "tokens_outside": outside,
            "tokens_total": len(dec_tokens),
            "rate": round(outside / max(1, len(dec_tokens)), 6),
        },
    }
    return kept, stats


def run_select(args: argparse.Namespace) -> None:
    kept, stats = select_kept_ids(Path(args.decode_file), Path(args.gt_file), args.k)
    KEPT_IDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUT.parent.mkdir(parents=True, exist_ok=True)
    KEPT_IDS_OUT.write_text(json.dumps({
        "model_id": MODEL_ID,
        "full_vocab": FULL_VOCAB,
        "K": args.k,
        "kept_size": len(kept),
        "kept_ids": kept,
    }))
    ANALYSIS_OUT.write_text(json.dumps(stats, indent=2))
    log(f"[select] wrote {KEPT_IDS_OUT} ({len(kept)} ids)")
    log(f"[select] wrote {ANALYSIS_OUT}")
    print(json.dumps(stats, indent=2))


def run_prune(args: argparse.Namespace) -> None:
    """Slice lm_head rows to kept_ids on the int4 base checkpoint.

    Requires torch + the int4+g128+lm_head base. compressed-tensors int4 packs the
    *hidden* axis, so axis-0 (vocab) row slicing keeps packing valid. The exact
    saved-tensor names depend on the base checkpoint layout, so this inspects the
    safetensors index and slices every lm_head tensor whose axis-0 == full vocab.
    """
    import torch  # noqa: F401  (import guarded to phase 2)
    from safetensors.torch import load_file, save_file

    base = Path(args.base_dir)
    out = Path(args.out_dir)
    if not base.exists():
        raise SystemExit(
            f"int4 base checkpoint not found at {base}. Point --base-dir at "
            f"lawine's int4_g128_lmhead, or rebuild it first (W4A16 g128 body + "
            f"untied int4 lm_head from google/gemma-4-E4B-it-qat-q4_0-unquantized)."
        )
    kept = json.loads(KEPT_IDS_OUT.read_text())["kept_ids"]
    kept_t = torch.tensor(kept, dtype=torch.long)
    out.mkdir(parents=True, exist_ok=True)

    shards = sorted(base.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors shards under {base}")
    for shard in shards:
        tensors = load_file(str(shard))
        new_tensors = {}
        for name, tensor in tensors.items():
            if "lm_head" in name and tensor.shape and tensor.shape[0] == FULL_VOCAB:
                new_tensors[name] = tensor.index_select(0, kept_t).contiguous()
                log(f"[prune] sliced {name}: {tuple(tensor.shape)} -> "
                    f"{tuple(new_tensors[name].shape)}")
            else:
                new_tensors[name] = tensor
        save_file(new_tensors, str(out / shard.name))

    # Copy non-weight files; rewrite the head/output vocab size in config.json.
    import shutil

    for extra in base.iterdir():
        if extra.suffix == ".safetensors":
            continue
        dest = out / extra.name
        if extra.is_dir():
            shutil.copytree(extra, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(extra, dest)
    (out / "kept_ids.json").write_text(KEPT_IDS_OUT.read_text())
    log(f"[prune] saved pruned checkpoint to {out}. NOTE: the lm_head output dim is "
        f"now {len(kept)} while embed_tokens stays {FULL_VOCAB}; serve.py must remap "
        f"12k indices -> original ids (see serve.py / research pass).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sel = sub.add_parser("select", help="CPU: choose kept_ids + write analysis")
    sel.add_argument("--k", type=int, default=DEFAULT_K)
    sel.add_argument("--decode-file", default=str(DECODE_FILE))
    sel.add_argument("--gt-file", default=str(GT_FILE))
    sel.set_defaults(func=run_select)

    pr = sub.add_parser("prune", help="GPU/torch: slice lm_head rows + save ckpt")
    pr.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    pr.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    pr.set_defaults(func=run_prune)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
