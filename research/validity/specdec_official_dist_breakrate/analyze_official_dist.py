#!/usr/bin/env python
"""PR #678 -- strict-#319 break-rate on the OFFICIAL SCORED distribution + rescue pricing.

ANALYSIS-ONLY. NO GPU, NO server, NO HF Job. Pure offline analysis of decode
jsonls produced by ``run_drafterfree.py`` on the leaderboard-scored prompt file
``official/.../speed_benchmark/data/eval_prompts_sharegpt.json``.

CENTRAL FINDING (verified at intake, re-asserted here from the file itself):
the file NAMED ``eval_prompts_sharegpt.json`` does NOT contain ShareGPT chat
prompts. Its 128 entries are **57 gpqa_diamond + 57 mmlu_pro + 14 aime2026** --
i.e. it IS the hard reasoning set. So the "scored distribution" and the "#673
reasoning set" are the *same 128 prompts*; #673's break-rate already WAS the
scored-distribution break-rate. This script proves that from the prompt ids,
reproduces the per-cell break-rate, and adds the per-SOURCE decomposition +
recompute-rescue pricing the PR asks for.

For each drafter-free spec cell (ngram:K / suffix:K) vs the served AR anchor it
computes, position-by-position on aligned ``completion_token_ids`` (conc=1,
ignore_eos, max_tokens=512 => equal length => positions align):

  * break_rate (per-prompt: any token differs) -- reproduces the #673 sha256 break,
  * per-SOURCE break_rate / onset / token-rate (mmlu_pro vs gpqa_diamond vs aime2026),
  * AR-vs-AR control (two AR runs) -- must be 0 or the break is engine noise,
  * rescue pricing for the stark #669 recompute-acceptor:
      rescue_positions_per_prompt ~= break_rate  (each broken prompt needs ~1
      ROOT-cause rescue; after the acceptor recomputes width-1 at the divergence
      the trajectory realigns, so the *cascade* tail -- divergent_tokens/prompt --
      is PREVENTED, not paid). Poisson refinement -ln(1-p) reported as an upper est.

Verdict (PR #678):
  * SHAREGPT_BREAK_AS_BAD  -- best-cell break_rate within/above the #673 reasoning
    band (>= 0.20) => rescue tax as heavy as feared on the scored distribution.
  * SHAREGPT_BREAK_LOW     -- best-cell break_rate <= 0.05 => cheap rescue, spec-dec
    cheaply shippable under strict #319 (feeds stark #669, flag for #481).

Logged W&B scalars include the no-fire guard: analysis_only=1, official_tps=0, fires=0.

Usage::
    PY=/tmp/senpai-venvs/<hash>/bin/python  # or any python with wandb
    $PY research/validity/specdec_official_dist_breakrate/analyze_official_dist.py \
        --screen-dir research/validity/drafterfree_specdec/_runs/screen_K56 \
        --ar-diag-dir research/validity/drafterfree_specdec/_runs/diag_ar_bi1 \
        --wandb-name kanna/specdec-official-dist-breakrate
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
OFFICIAL_PROMPTS = (ROOT / "official" / "main_bucket" / "shared_resources"
                    / "speed_benchmark" / "data" / "eval_prompts_sharegpt.json")
OUTPUT_LEN = 512  # benchmark protocol (paths.OUTPUT_LEN); every completion is this long

# Verdict thresholds (break_rate of the best = lowest-break cell).
LOW_BAR = 0.05     # <= this => SHAREGPT_BREAK_LOW (cheap rescue)
ASBAD_BAR = 0.20   # >= this => SHAREGPT_BREAK_AS_BAD (reasoning-set regime, 0.33-0.38)
# #673 reasoning-set reference band (same prompts) for the contrast print.
REF673 = {"ngram5": 0.3828125, "ngram6": 0.359375,
          "suffix5": 0.3671875, "suffix6": 0.328125}


def source_of(pid: str) -> str:
    """'aime2026-0e67..' -> 'aime2026'."""
    return pid.split("-", 1)[0]


def load_prompts_sources(prompts_file: Path) -> dict[str, str]:
    data = json.loads(prompts_file.read_text())
    return {str(x["id"]): source_of(str(x["id"])) for x in data}


def load_completions(decode_jsonl: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with decode_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["id"])] = list(row["completion_token_ids"])
    return out


def load_shas(decode_jsonl: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with decode_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["id"])] = row["completion_token_sha256"]
    return out


def _onset_ndiv(a: list[int], b: list[int]) -> tuple[int | None, int]:
    """(first differing position or None, total differing positions incl len mismatch)."""
    n = min(len(a), len(b))
    first = None
    ndiff = 0
    for i in range(n):
        if a[i] != b[i]:
            ndiff += 1
            if first is None:
                first = i
    ndiff += abs(len(a) - len(b))
    if first is None and len(a) != len(b):
        first = n
    return first, ndiff


def compare_cell(ref: dict[str, list[int]], cell: dict[str, list[int]],
                 src: dict[str, str]) -> dict[str, Any]:
    """Overall + per-source token-aligned divergence stats."""
    common = sorted(set(ref) & set(cell))
    per_prompt: list[dict[str, Any]] = []
    for pid in common:
        onset, ndiff = _onset_ndiv(ref[pid], cell[pid])
        per_prompt.append({
            "id": pid, "source": src.get(pid, "unknown"),
            "len_ref": len(ref[pid]), "len_cell": len(cell[pid]),
            "onset": onset, "n_div": ndiff, "broken": onset is not None,
        })

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        broken = [r for r in rows if r["broken"]]
        onsets = [r["onset"] for r in broken]
        tot_tokens = sum(min(r["len_ref"], r["len_cell"]) for r in rows)
        div_tokens = sum(r["n_div"] for r in rows)
        return {
            "n_prompts": n,
            "n_broken": len(broken),
            "break_rate": (len(broken) / n) if n else None,
            "token_divergence_rate": (div_tokens / tot_tokens) if tot_tokens else None,
            "divergent_tokens": div_tokens,
            "div_tokens_per_prompt": (div_tokens / n) if n else None,
            "onset_median": statistics.median(onsets) if onsets else None,
            "onset_mean": statistics.fmean(onsets) if onsets else None,
            "onset_min": min(onsets) if onsets else None,
            "onset_max": max(onsets) if onsets else None,
        }

    overall = _agg(per_prompt)
    by_source: dict[str, Any] = {}
    for s in sorted({r["source"] for r in per_prompt}):
        by_source[s] = _agg([r for r in per_prompt if r["source"] == s])

    # ---- rescue pricing (stark #669 recompute-acceptor) ----
    # Each broken prompt costs ~1 ROOT rescue (recompute width-1 at the divergence);
    # the acceptor's recompute realigns to AR so the cascade tail is PREVENTED, not
    # paid. Hence rescue_positions_per_prompt ~= break_rate (root flips are rare:
    # <=1 per prompt to first order). Two refinements are reported:
    #   * hazard*L : MLE constant per-position hazard h = n_broken / (positions at
    #     risk before first break), integrated over L -> accounts for >1 flip/prompt.
    #   * poisson  : -ln(1-break_rate), the rare-event closed form for E[flips].
    # div_tokens_per_prompt is the CASCADE over-count (what you'd pay WITHOUT a
    # realigning acceptor) -- reported but explicitly NOT the rescue cost.
    n = overall["n_prompts"] or 0
    nb = overall["n_broken"] or 0
    at_risk = sum((r["onset"] if r["broken"] else OUTPUT_LEN) for r in per_prompt)
    hazard = (nb / at_risk) if at_risk else 0.0
    p = overall["break_rate"] or 0.0
    rescue = {
        "rescue_positions_per_prompt": p,                       # primary (= break_rate)
        "rescue_per_prompt_hazardL": hazard * OUTPUT_LEN,       # MLE multi-flip refinement
        "rescue_per_prompt_poisson": (-math.log(1 - p) if 0 < p < 1 else p),
        "per_position_hazard": hazard,
        "cascade_div_tokens_per_prompt": overall["div_tokens_per_prompt"],  # NOT rescue cost
    }
    return {"overall": overall, "by_source": by_source, "rescue": rescue,
            "per_prompt": per_prompt}


def byte_identity(ref_shas: dict[str, str], cell_shas: dict[str, str]) -> dict[str, Any]:
    common = sorted(set(ref_shas) & set(cell_shas))
    mism = [pid for pid in common if ref_shas[pid] != cell_shas[pid]]
    n = len(common)
    return {"n_compared": n, "n_mismatch": len(mism),
            "break_rate": (len(mism) / n) if n else None,
            "byte_exact": (len(mism) == 0 and n > 0), "mismatch_ids": mism[:8]}


def ar_vs_ar(decode_a: Path, decode_b: Path, label: str) -> dict[str, Any] | None:
    if not (decode_a.exists() and decode_b.exists()):
        return None
    bi = byte_identity(load_shas(decode_a), load_shas(decode_b))
    bi["label"] = label
    bi["a"] = decode_a.name
    bi["b"] = decode_b.name
    return bi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen-dir", type=Path, required=True,
                    help="dir with decode_ar_r0.jsonl + decode_<cell>_r0.jsonl")
    ap.add_argument("--ref", default="ar")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--ar-diag-dir", type=Path, default=None,
                    help="dir with two AR runs (decode_ar_r0/r1.jsonl) for the within-run AR-vs-AR control")
    ap.add_argument("--ar-extra", type=Path, default=None,
                    help="a 2nd AR decode jsonl from a DIFFERENT invocation (cross-run AR-vs-AR control)")
    ap.add_argument("--prompts-file", type=Path, default=OFFICIAL_PROMPTS)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="specdec-official-dist-breakrate-kanna")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    src = load_prompts_sources(args.prompts_file)
    src_counts = dict(collections.Counter(src.values()))
    print(f"[off] scored prompt file: {args.prompts_file}")
    print(f"[off] SCORED SOURCE DISTRIBUTION (n={len(src)}): {src_counts}")
    print("[off]   -> 'eval_prompts_sharegpt.json' is the REASONING set, not chat.\n")

    screen = args.screen_dir.resolve()
    ref_jsonl = screen / f"decode_{args.ref}_r{args.repeat}.jsonl"
    if not ref_jsonl.exists():
        raise SystemExit(f"AR reference decode not found: {ref_jsonl}")
    ref = load_completions(ref_jsonl)
    ref_shas = load_shas(ref_jsonl)
    # confirm the artifacts are on the scored file
    on_scored = len(set(ref) & set(src))
    print(f"[off] AR ref {ref_jsonl.name}: {len(ref)} prompts, {on_scored} match the scored file ids")

    cell_jsonls = sorted(p for p in screen.glob(f"decode_*_r{args.repeat}.jsonl")
                         if p != ref_jsonl)
    cells: dict[str, Any] = {}
    print(f"\n[off] {'cell':9s} {'break':>7s} {'tok_rate':>8s} {'onset_med':>9s} "
          f"{'rescue/prompt':>13s}  per-source break_rate")
    for cj in cell_jsonls:
        name = cj.name[len("decode_"):-len(f"_r{args.repeat}.jsonl")]
        cmp = compare_cell(ref, load_completions(cj), src)
        cmp["byte_identity"] = byte_identity(ref_shas, load_shas(cj))
        cells[name] = cmp
        ov = cmp["overall"]
        src_str = "  ".join(
            f"{s}:{cmp['by_source'][s]['break_rate']:.3f}({cmp['by_source'][s]['n_broken']}/{cmp['by_source'][s]['n_prompts']})"
            for s in sorted(cmp["by_source"]))
        print(f"[off] {name:9s} {ov['break_rate']:>7.3f} {ov['token_divergence_rate']:>8.4f} "
              f"{str(ov['onset_median']):>9s} {cmp['rescue']['rescue_positions_per_prompt']:>13.3f}  {src_str}")

    # ---- AR-vs-AR determinism controls ----
    controls = []
    if args.ar_diag_dir:
        c = ar_vs_ar(args.ar_diag_dir / "decode_ar_r0.jsonl",
                     args.ar_diag_dir / "decode_ar_r1.jsonl", "within_diag_r0_vs_r1")
        if c:
            controls.append(c)
    if args.ar_extra:
        c = ar_vs_ar(ref_jsonl, args.ar_extra.resolve(), "cross_run_ref_vs_extra")
        if c:
            controls.append(c)
    print("\n[off] AR-vs-AR determinism controls (must be break_rate=0):")
    for c in controls:
        print(f"[off]   {c['label']:24s} break_rate={c['break_rate']} "
              f"({c['n_mismatch']}/{c['n_compared']})  byte_exact={c['byte_exact']}")
    if not controls:
        print("[off]   (none supplied)")

    # ---- best cell + verdict ----
    rated = {c: a["overall"]["break_rate"] for c, a in cells.items()
             if a["overall"]["break_rate"] is not None}
    best_cell = min(rated, key=rated.get) if rated else None
    best_break = rated[best_cell] if best_cell else None
    best_rescue = cells[best_cell]["rescue"]["rescue_positions_per_prompt"] if best_cell else None

    if best_break is None:
        verdict = "INCONCLUSIVE_NO_CELLS"
    elif best_break <= LOW_BAR:
        verdict = "SHAREGPT_BREAK_LOW"
    elif best_break >= ASBAD_BAR:
        verdict = "SHAREGPT_BREAK_AS_BAD"
    else:
        verdict = "SHAREGPT_BREAK_INTERMEDIATE"

    ar_det = all(c["byte_exact"] for c in controls) if controls else None

    result = {
        "analysis_only": True, "official_tps": 0, "fires": False,
        "scored_prompts_file": str(args.prompts_file),
        "scored_source_distribution": src_counts,
        "scored_is_reasoning_set": True,
        "scored_n_prompts": len(src),
        "ar_ref": ref_jsonl.name,
        "cells": {c: {"overall": a["overall"], "by_source": a["by_source"],
                      "rescue": a["rescue"], "byte_identity": a["byte_identity"]}
                  for c, a in cells.items()},
        "ar_vs_ar_controls": controls,
        "ar_deterministic": ar_det,
        "ref673_break_band": REF673,
        "best_cell": best_cell,
        "strict319_break_rate_sharegpt_best": best_break,
        "rescue_positions_per_prompt_sharegpt": best_rescue,
        "verdict": verdict,
        "low_bar": LOW_BAR, "asbad_bar": ASBAD_BAR,
    }
    out = args.out or (screen / "official_dist_breakrate.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[off] best_cell={best_cell} break_rate={best_break} "
          f"rescue/prompt={best_rescue}")
    print(f"[off] VERDICT: {verdict}")
    print(f"[off] wrote {out}")

    if not args.no_wandb:
        _log_wandb(result, cells, args)
    return 0


def _log_wandb(result: dict[str, Any], cells: dict[str, Any],
               args: argparse.Namespace) -> None:
    try:
        import os
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[off] wandb import failed ({exc}); skipping", flush=True)
        return
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            name=args.wandb_name or "kanna/specdec-official-dist-breakrate",
            group=args.wandb_group, job_type="analysis-only",
            tags=["specdec", "official-dist", "break-rate", "analysis-only", "pr678"],
            config={"analysis_only": True, "screen_dir": str(args.screen_dir),
                    "scored_prompts_file": str(args.prompts_file),
                    "scored_source_distribution": result["scored_source_distribution"]},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[off] wandb init failed ({exc}); skipping", flush=True)
        return
    summary: dict[str, Any] = {
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "verdict": result["verdict"],
        "best_cell": result["best_cell"],
        "strict319_break_rate_sharegpt_best": result["strict319_break_rate_sharegpt_best"],
        "rescue_positions_per_prompt_sharegpt": result["rescue_positions_per_prompt_sharegpt"],
        "scored_is_reasoning_set": 1,
        "scored_n_prompts": result["scored_n_prompts"],
        "ar_deterministic": (1 if result["ar_deterministic"] else 0)
        if result["ar_deterministic"] is not None else None,
    }
    for s, cnt in result["scored_source_distribution"].items():
        summary[f"scored_source/{s}"] = cnt
    for c, a in cells.items():
        ov = a["overall"]
        summary[f"{c}/break_rate"] = ov["break_rate"]
        summary[f"{c}/token_divergence_rate"] = ov["token_divergence_rate"]
        summary[f"{c}/onset_median"] = ov["onset_median"]
        summary[f"{c}/byte_exact"] = 1 if a["byte_identity"]["byte_exact"] else 0
        summary[f"{c}/rescue_positions_per_prompt"] = a["rescue"]["rescue_positions_per_prompt"]
        summary[f"{c}/rescue_poisson"] = a["rescue"]["rescue_per_prompt_poisson"]
        summary[f"{c}/cascade_div_tokens_per_prompt"] = a["rescue"]["cascade_div_tokens_per_prompt"]
        for s, sa in a["by_source"].items():
            summary[f"{c}/{s}/break_rate"] = sa["break_rate"]
            summary[f"{c}/{s}/onset_median"] = sa["onset_median"]
    for ctl in result["ar_vs_ar_controls"]:
        summary[f"ar_vs_ar/{ctl['label']}/break_rate"] = ctl["break_rate"]
        summary[f"ar_vs_ar/{ctl['label']}/n_mismatch"] = ctl["n_mismatch"]
    run.summary.update({k: v for k, v in summary.items() if v is not None})

    # per-cell x per-source table
    tbl = wandb.Table(columns=["cell", "source", "n_prompts", "n_broken",
                               "break_rate", "onset_median", "token_rate"])
    for c, a in cells.items():
        ov = a["overall"]
        tbl.add_data(c, "ALL", ov["n_prompts"], ov["n_broken"], ov["break_rate"],
                     ov["onset_median"], ov["token_divergence_rate"])
        for s, sa in a["by_source"].items():
            tbl.add_data(c, s, sa["n_prompts"], sa["n_broken"], sa["break_rate"],
                         sa["onset_median"], sa["token_divergence_rate"])
    run.log({"break_by_source": tbl})
    print(f"[off] wandb run: {run.url} (id={run.id})", flush=True)
    run.finish()


if __name__ == "__main__":
    sys.exit(main())
