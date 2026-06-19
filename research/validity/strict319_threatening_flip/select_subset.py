"""Select the strict-#319 answer-threatening subset for the PR #694 capstone.

Source of truth: kanna #689's `fork_risk_manifest.jsonl`
(W&B artifact `shift_edit_classification_detail:v0`, run tyzovau1). Each row is a
strict-#319 spec break classified on the 512-tok confirm decode.

Selection (the prompts MOST likely to flip the extracted answer end-to-end):
  * ALL `content-fork` ids  -- the perm-fork tail (the two streams genuinely
    diverge; seq_ratio/jaccard below #689's gate). GGT->CCT lives here.
  * the sharpest `content-heal` items on gpqa/aime named in the #689 hand-audit
    (the answer-THREATENING minority: AIME coordinate/series conventions, a
    physics symbol swap, a reaction-mechanism swap).

NB: every `answer_match` in the manifest is False, but that is a 512-tok
TRUNCATION artifact (#685: gpqa 0/57, aime 0/14 commit at 512 tok), NOT a measured
flip. #694 re-decodes these at the full 6144-tok natural-EOS budget so both arms
actually commit, then measures the real answer outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = Path("/tmp/wb689_artifact/fork_risk_manifest.jsonl")
DATASET = Path("official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json")

# The sharpest gpqa/aime `content-heal` items from the #689 hand-audit, with the
# short human label that #689/#694 use to refer to each one.
NAMED_HEALS = {
    "aime2026-7443669f7e": "AIME series sum S=sum 1/(10^n-1) (coordinate/series convention)",
    "aime2026-8742f9e684": "AIME segment geometry (horizontal/diagonal coordinate convention)",
    "aime2026-1eaadd490a": "AIME 'equation of the line' setup divergence",
    "gpqa_diamond-14a2ed1ddc": "physics symbol swap (n=p) vs (N=Z) [mass-number/reduced-mass family]",
    "gpqa_diamond-32bdeb2d67": "reaction mechanism: alpha-diketone vs C1/C2 carbonyl framing",
}


def main() -> int:
    rows = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    dataset = json.loads(DATASET.read_text())
    ds_ids = {r["id"] for r in dataset}

    # group manifest rows by prompt id
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)

    fork_ids = []
    for r in rows:
        if r["final_class"] == "content-fork" and r["id"] not in fork_ids:
            fork_ids.append(r["id"])

    subset = []
    for pid in fork_ids:
        rs = by_id[pid]
        fr = next((x for x in rs if x["final_class"] == "content-fork"), rs[0])
        subset.append({
            "id": pid,
            "source": fr["source"],
            "flag_reason": "content-fork",
            "label": _fork_label(pid, fr),
            "n689_cells": sorted({x["cell"] for x in rs if x["final_class"] == "content-fork"}),
            "seq_ratio": fr.get("seq_ratio"),
            "jaccard": fr.get("jaccard"),
            "n689_answer_match_512tok": fr.get("answer_match"),
        })

    for pid, label in NAMED_HEALS.items():
        rs = by_id.get(pid, [])
        src = rs[0]["source"] if rs else pid.split("-")[0]
        subset.append({
            "id": pid,
            "source": src,
            "flag_reason": "content-heal-sharp",
            "label": label,
            "n689_cells": sorted({x["cell"] for x in rs if x["final_class"] == "content-heal"}),
            "seq_ratio": None,
            "jaccard": None,
            "n689_answer_match_512tok": None,
        })

    # integrity checks
    ids = [s["id"] for s in subset]
    assert len(ids) == len(set(ids)), "duplicate ids in subset"
    missing = [i for i in ids if i not in ds_ids]
    assert not missing, f"subset ids absent from eval dataset: {missing}"

    out = {
        "n_flagged": len(subset),
        "n_content_fork": sum(1 for s in subset if s["flag_reason"] == "content-fork"),
        "n_content_heal_sharp": sum(1 for s in subset if s["flag_reason"] == "content-heal-sharp"),
        "by_source": _count(subset, "source"),
        "manifest": str(MANIFEST),
        "note": "answer_match in #689 manifest is a 512-tok truncation artifact, not a measured flip",
        "prompts": subset,
    }
    (HERE / "flagged_subset.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_flagged", "n_content_fork", "n_content_heal_sharp", "by_source")}, indent=2))
    for s in subset:
        print(f"  {s['id']:26s} {s['source']:13s} {s['flag_reason']:18s} {s['label']}")
    return 0


def _fork_label(pid: str, fr: dict) -> str:
    if pid == "gpqa_diamond-42da82ad5c":
        return "GGT->CCT mutant fork (streams analyze different mutants; seq_ratio 0.0)"
    head = str(fr.get("ar_tail_head", ""))[:48].replace("\n", " ")
    return f"content-fork (ar_tail: {head!r})"


def _count(subset, key):
    out: dict[str, int] = {}
    for s in subset:
        out[s[key]] = out.get(s[key], 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
