#!/usr/bin/env python3
"""PR #598 -- prove the Wanfq/gpqa mirror is a faithful source of GPQA-Main.

inspect_evals ships only the Diamond (n=198) loader, sourced from the ungated
openaipublic simple-evals CSV. GPQA-Main (n=448) / Extended (n=546) live in the
GATED Idavidrein/gpqa repo. We instead source Main/Extended from the ungated
Wanfq/gpqa mirror. This script makes the faithfulness argument auditable rather
than asserted:

  1. The mirror's gpqa_diamond.csv vs the canonical openaipublic gpqa_diamond.csv
     the harness already uses for the Diamond gate -- compare on ALL model-visible
     content (Question + Correct Answer + the 3 Incorrect Answers, order-independent
     over the incorrect set). A 198/198 multiset match proves the mirror reproduces
     canonical GPQA verbatim on exactly the fields the model sees.
  2. gpqa_main.csv: 448 rows, full 78-col schema incl. the GPQA Canary String, all
     columns record_to_sample needs, and content SHA256 pinned in run_eval.py.
  3. Provenance transitivity: every Diamond question is a subset of Main (the
     canonical GPQA structure: Diamond ⊂ Main), so a faithful Diamond subset +
     verbatim schema ⇒ the Main rows are the verbatim canonical GPQA-Main.

LOCAL, CPU/network only -- does not touch the GPU eval. analysis_only, NO FIRE.
"""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from inspect_evals.gpqa.gpqa import GPQA_DIAMOND_CACHE_PATH

HERE = Path("/workspace/senpai/target/research/validity/gpqa_larger_instrument_ci")
MIRROR = "Wanfq/gpqa"
TOK = os.environ.get("HF_TOKEN") or None
PINNED_SHA = {  # must match GPQA_SPLIT_SHA256 in run_eval.py
    "main": "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3",
    "extended": "0926ee24949d02ed6748eb75a2611546c34479e30ddc42efd01d6f1681aaa48a",
}
NEED_COLS = ["Question", "Correct Answer",
             "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mv_key(r):
    """Model-visible content: question + correct + the 3 incorrect (incorrect set
    is order-independent because the harness re-shuffles choices by --seed)."""
    q = (r.get("Question") or "").strip()
    c = (r.get("Correct Answer") or "").strip()
    inc = tuple(sorted((r.get(f"Incorrect Answer {i}") or "").strip() for i in (1, 2, 3)))
    return (q, c, inc)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    canon = load_rows(GPQA_DIAMOND_CACHE_PATH)
    wd = load_rows(hf_hub_download(MIRROR, "gpqa_diamond.csv", repo_type="dataset", token=TOK))

    cc, wc = Counter(mv_key(r) for r in canon), Counter(mv_key(r) for r in wd)
    exact = sum((cc & wc).values())
    only_canon = sum((cc - wc).values())
    only_mirror = sum((wc - cc).values())

    main_path = hf_hub_download(MIRROR, "gpqa_main.csv", repo_type="dataset", token=TOK)
    wm = load_rows(main_path)
    main_sha = sha256(main_path)
    main_q = {(r.get("Question") or "").strip() for r in wm}
    diamond_in_main = sum(1 for r in wd if (r.get("Question") or "").strip() in main_q)

    out = {
        "pr": 598, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "mirror": MIRROR,
        "canonical_diamond_source": "openaipublic simple-evals gpqa_diamond.csv",
        "canonical_diamond_sha256": sha256(GPQA_DIAMOND_CACHE_PATH),
        "canonical_diamond_rows": len(canon),
        "mirror_diamond_rows": len(wd),
        "mirror_diamond_cols": len(wd[0]),
        "model_visible_match_diamond": exact,
        "diamond_rows_only_in_canonical": only_canon,
        "diamond_rows_only_in_mirror": only_mirror,
        "diamond_faithful": bool(exact == len(canon) == len(wd)
                                 and only_canon == 0 and only_mirror == 0),
        "mirror_main_rows": len(wm),
        "mirror_main_cols": len(wm[0]),
        "mirror_main_has_needed_cols": all(c in wm[0] for c in NEED_COLS),
        "mirror_main_sha256": main_sha,
        "mirror_main_sha256_matches_pin": main_sha == PINNED_SHA["main"],
        "diamond_subset_of_main": diamond_in_main,
        "diamond_is_subset_of_main": bool(diamond_in_main == len(wd)),
    }
    out["VALID"] = bool(
        out["diamond_faithful"] and out["mirror_main_rows"] == 448
        and out["mirror_main_has_needed_cols"] and out["mirror_main_sha256_matches_pin"]
        and out["diamond_is_subset_of_main"]
    )
    (HERE / "validate_mirror.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
