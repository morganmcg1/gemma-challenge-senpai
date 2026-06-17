#!/usr/bin/env python3
"""Build the head-width keepsets for the intact-body head-prune sweep (PR #547).

Single-axis sweep over lm_head row count on the INTACT base-int4 body:
  * 12k  -- the ship's PCK04 keepset (the exact token-ID set osoi5 loads), copied
            verbatim from research/validity/keepset_coverage_gap/pck04_keepset_12k.json.
  * 32k  -- a COVERAGE-RANKED superset of the 12k, so the sweep is monotone
            (12k subset 32k subset 262144). Construction: start from the ship 12k,
            then add the highest-frequency tokens NOT already in it, by descending
            frequency on the #528 teacher-forced decode stream (the same coverage
            definition used in PR #528/#543), until |set| == 32768.
  * 262k -- no keepset (full head); handled by the serve patch as mode=off.

Answer-token guard: tokenize the MC answer letters (A..J for MMLU-Pro/GPQA) and the
GSM8K answer surface (digits 0-9, operators) and report whether each lands inside the
12k set. If the answer tokens are all in 12k, the prediction "12k holds MC quality"
is strong (the model can always emit the correct final letter/number; only the
reasoning-path vocabulary is restricted).

analysis_only -- writes JSON keepsets + a manifest. No GPU, no serve.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
KCG = HERE.parent / "keepset_coverage_gap"
SHIP_12K = KCG / "pck04_keepset_12k.json"
DECODE = KCG / "decode.jsonl"
FULL_VOCAB = 262144
MODEL_DIR = None  # resolved at runtime from the HF cache snapshot


def _load_keepset(path: Path) -> tuple[list[int], int]:
    d = json.loads(path.read_text())
    keep = [int(x) for x in d["keep_ids"]]
    fv = int(d.get("full_vocab") or d.get("vocab_size") or FULL_VOCAB)
    return keep, fv


def _freq_rank() -> list[int]:
    """Descending-frequency token ranking over all completion_token_ids in the
    #528 decode stream (728 teacher-forced base-greedy completions, mixed tasks)."""
    c: Counter = Counter()
    n_rec = 0
    n_tok = 0
    with DECODE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ids = r.get("completion_token_ids") or []
            c.update(int(t) for t in ids)
            n_rec += 1
            n_tok += len(ids)
    ranked = [tid for tid, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
    print(f"[build] decode stream: {n_rec} records, {n_tok} completion tokens, "
          f"{len(c)} distinct ids covered", flush=True)
    return ranked


def _resolve_model_dir() -> Path | None:
    import glob
    import os
    g = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/*/"))
    return Path(g[0]) if g else None


def _answer_token_report(set12k: set[int], set32k: set[int]) -> dict:
    md = _resolve_model_dir()
    rep: dict = {"model_dir": str(md) if md else None}
    if md is None:
        rep["error"] = "no local snapshot of gemma-4-E4B-it-qat-w4a16-ct"
        return rep
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(md))

    def ids_for(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)

    groups = {
        "mc_letters_bare": [chr(ord("A") + i) for i in range(10)],     # A..J
        "mc_letters_space": [" " + chr(ord("A") + i) for i in range(10)],
        "digits_bare": [str(d) for d in range(10)],
        "digits_space": [" " + str(d) for d in range(10)],
        "operators": ["+", "-", "*", "/", "=", ".", ",", "$", "%"],
    }
    out: dict = {}
    for gname, surfaces in groups.items():
        rows = []
        for s in surfaces:
            ids = ids_for(s)
            # a surface lands "in" the set iff EVERY token id of its encoding is kept
            in12 = all(t in set12k for t in ids)
            in32 = all(t in set32k for t in ids)
            rows.append({"surface": s, "ids": ids, "in_12k": in12, "in_32k": in32})
        out[gname] = {
            "all_in_12k": all(r["in_12k"] for r in rows),
            "all_in_32k": all(r["in_32k"] for r in rows),
            "rows": rows,
        }
    rep["groups"] = out
    rep["mc_answer_letters_all_in_12k"] = (
        out["mc_letters_bare"]["all_in_12k"] and out["mc_letters_space"]["all_in_12k"]
    )
    rep["gsm8k_digits_all_in_12k"] = (
        out["digits_bare"]["all_in_12k"] and out["digits_space"]["all_in_12k"]
    )
    return rep


def main() -> int:
    ship12k, fv = _load_keepset(SHIP_12K)
    assert fv == FULL_VOCAB, f"unexpected full_vocab {fv}"
    set12k = set(ship12k)
    assert len(set12k) == 12288, f"ship 12k has {len(set12k)} unique ids"

    ranked = _freq_rank()
    # 32k = ship 12k UNION next-highest-frequency tokens not already present.
    target = 32768
    sel = list(ship12k)
    present = set(set12k)
    for tid in ranked:
        if len(sel) >= target:
            break
        if tid not in present:
            sel.append(tid)
            present.add(tid)
    if len(sel) < target:
        # decode stream didn't supply enough distinct ids; pad by ascending id
        # over the remaining vocab (rare-but-safe; keeps monotone superset).
        for tid in range(FULL_VOCAB):
            if len(sel) >= target:
                break
            if tid not in present:
                sel.append(tid)
                present.add(tid)
    set32k = set(sel)
    assert len(set32k) == target, f"32k has {len(set32k)} unique ids"
    assert set12k.issubset(set32k), "12k is not a subset of 32k"
    n_from_decode = sum(1 for t in sel if t not in set12k and t in set(ranked))
    print(f"[build] 32k built: {len(set32k)} ids "
          f"(12k ship + {len(set32k) - 12288} added; "
          f"{n_from_decode} of the added came from decode-stream coverage rank)",
          flush=True)

    # write keepsets (sorted ascending, same schema the patch/pck04 expects)
    def _write(path: Path, ids: set[int], note: str) -> None:
        ids_sorted = sorted(ids)
        path.write_text(json.dumps({
            "keep_ids": ids_sorted,
            "pruned_vocab_K": len(ids_sorted),
            "full_vocab": FULL_VOCAB,
            "vocab_size": FULL_VOCAB,
            "note": note,
        }, indent=2))
        print(f"[build] wrote {path} ({len(ids_sorted)} ids)", flush=True)

    _write(HERE / "keepset_12k.json", set12k,
           "ship PCK04 12k (copied from #528 keepset_coverage_gap/pck04_keepset_12k.json)")
    _write(HERE / "keepset_32k.json", set32k,
           "coverage-ranked superset of ship 12k via #528 decode-stream frequency rank")

    ans = _answer_token_report(set12k, set32k)

    manifest = {
        "pr": 547,
        "analysis_only": True,
        "official_tps": 0,
        "full_vocab": FULL_VOCAB,
        "widths": {"12k": 12288, "32k": 32768, "262k": FULL_VOCAB},
        "monotone_12k_subset_32k": set12k.issubset(set32k),
        "n_distinct_in_decode_rank": len(ranked),
        "coverage_construction": (
            "32k = ship 12k UNION top-frequency tokens (by #528 decode-stream "
            "completion_token_ids count) not already in 12k, until 32768"),
        "answer_token_report": ans,
    }
    (HERE / "keepsets_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[build] answer-token guard:")
    print(f"  MC answer letters A..J all in 12k: {ans.get('mc_answer_letters_all_in_12k')}")
    print(f"  GSM8K digits 0-9 all in 12k:       {ans.get('gsm8k_digits_all_in_12k')}")
    print(f"[build] wrote {HERE / 'keepsets_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
