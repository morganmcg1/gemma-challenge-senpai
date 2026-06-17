#!/usr/bin/env python3
"""PR #568 -- assemble the direct-served-quality verdict for the EXACT #519 split-KV
submission and (optionally) log it to W&B.

Consumes the three ``run_quality.py`` measurement JSONs (MMLU-Pro / GPQA-Diamond /
AIME-greedy-min8) for ``submissions/fa2sw_strict_byteexact_splitkv399`` and reconciles
them against TWO references that are banked on this branch:

  * the vanilla-base denominators (ubel #511 ``base_mmlu_pro.json`` / ``base_gpqa.json``;
    fern #514 ``base_aime.json``) -> ``pct_of_base`` + ``passes_90pct_gate``.
  * the surgical-357 SHIP collapse (ubel #511 ``ship_*`` ; fern #514 ``ship_*_aime``),
    which is the TRANSFERRED number #524 argued by byte-exactness ->
    ``matches_byteexact_transfer``.

Because #519 is byte-EXACT to surgical-357, the strongest check is not the aggregate
accuracy but PER-ITEM agreement of the extracted answer against the banked ship run:
if the greedy tokens are identical, every per-item answer is identical, so the
accuracy matches by construction. We compute and report that agreement directly.

prompt_sha integrity (MMLU/GPQA): every measured item's prompt hash must equal the
banked base run's hash for the same id, or the A/B is not apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/splitkv519_direct_quality"
DQE = ROOT / "research/validity/downstream_quality_eval"
AIME_DIR = ROOT / "research/downstream_quality_aime"

GATE = 0.90

# Banked vanilla-base denominators (the >=90% gate divisors). MMLU/AIME match the PR
# card; GPQA base is 0.4444 on this branch (PR card states 0.470 -- recorded as a
# secondary divisor so the discrepancy is explicit, not silently chosen).
BASE = {"mmlu_pro": 0.668, "gpqa": 0.4444, "aime": 0.400}
BASE_PR_STATED = {"gpqa": 0.470}
# Surgical-357 SHIP collapse == the transferred number #524 argued for #519.
TRANSFER = {"mmlu_pro": 0.274, "gpqa": 0.2323, "aime": 0.033}


def _load(path: Path) -> dict[str, Any] | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _binom_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


def _mc_gate(name: str, measured: dict[str, Any], base_acc: float, transfer_acc: float,
             ship: dict[str, Any] | None, base: dict[str, Any] | None) -> dict[str, Any]:
    """MMLU-Pro / GPQA gate + transfer + per-item byte-exactness agreement vs ship."""
    acc = measured.get("accuracy")
    n = int(measured.get("n_scored") or 0)
    nc = int(measured.get("n_correct") or 0)
    lo, hi = _wilson(nc, n)
    pct = (acc / base_acc) if (acc is not None and base_acc) else None

    # prompt_sha integrity vs banked base (apples-to-apples prompts).
    n_match = n_mismatch = 0
    if base:
        base_sha = {str(r["id"]): r.get("prompt_sha") for r in base.get("per_sample", [])}
        for r in measured.get("per_sample", []):
            bs = base_sha.get(str(r["id"]))
            if bs is None:
                continue
            if bs == r.get("prompt_sha"):
                n_match += 1
            else:
                n_mismatch += 1

    # Per-item agreement vs banked ship (byte-exactness signal).
    agree_answer = agree_correct = n_pairs = 0
    if ship:
        ship_by_id = {str(r["id"]): r for r in ship.get("per_sample", [])}
        for r in measured.get("per_sample", []):
            sr = ship_by_id.get(str(r["id"]))
            if sr is None:
                continue
            n_pairs += 1
            if r.get("answer") == sr.get("answer"):
                agree_answer += 1
            if bool(r.get("correct")) == bool(sr.get("correct")):
                agree_correct += 1

    transfer_tol = max(0.03, 2.5 * _binom_se(transfer_acc, n or 1))
    matches = (acc is not None and abs(acc - transfer_acc) <= transfer_tol)
    return {
        "axis": name,
        "accuracy": acc,
        "n_scored": n,
        "n_correct": nc,
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "n_empty": measured.get("n_empty"),
        "empty_rate": measured.get("empty_rate"),
        "base_acc": base_acc,
        "pct_of_base": round(pct, 4) if pct is not None else None,
        "passes_90pct_gate": bool(pct is not None and pct >= GATE),
        "transfer_acc": transfer_acc,
        "abs_diff_vs_transfer": round(abs(acc - transfer_acc), 4) if acc is not None else None,
        "transfer_tol": round(transfer_tol, 4),
        "matches_byteexact_transfer": bool(matches),
        "prompt_sha_match": n_match,
        "prompt_sha_mismatch": n_mismatch,
        "prompt_sha_ok": bool(n_mismatch == 0 and n_match > 0),
        "ship_pairs": n_pairs,
        "answer_agree_vs_ship": round(agree_answer / n_pairs, 4) if n_pairs else None,
        "correct_agree_vs_ship": round(agree_correct / n_pairs, 4) if n_pairs else None,
    }


def _norm_aime_id(x: Any) -> str:
    """Collapse the loader's doubled year prefix so 2024 ids match banked runs.

    load_aime prefixes ``{year}-`` to each problem id; Maxwell-Jia's ``ID`` column
    already starts with the contest year (``2024-II-4``), so the current loader yields
    ``2024-2024-II-4`` while older banked greedy runs (ship_greedy_aime) carry the bare
    ``2024-II-4``. Drop one leading ``YYYY-`` when it is immediately duplicated.
    """
    parts = str(x).split("-")
    if len(parts) >= 2 and parts[0] == parts[1] and parts[0].isdigit():
        return "-".join(parts[1:])
    return str(x)


def _aime_gate(measured: dict[str, Any], ship_greedy: dict[str, Any] | None) -> dict[str, Any]:
    acc = measured.get("maj_k_accuracy")
    pp = measured.get("per_problem", [])
    n = len(pp)
    nc = int(measured.get("n_correct_maj") or 0)
    lo, hi = _wilson(nc, n)
    pct = (acc / BASE["aime"]) if acc is not None else None

    # char-based empty rate (single greedy sample per problem).
    char_empty = sum(1 for r in pp if (r.get("sample_chars") or [1])[0] == 0)
    empty_rate = (char_empty / n) if n else None

    # 2024-only subset -> directly comparable to ship_greedy (0.0333) and the maj@8
    # transfer (0.033), both of which were measured on AIME-2024 (n=30).
    sub24 = [r for r in pp if str(r.get("year")) == "2024"]
    n24 = len(sub24)
    nc24 = sum(1 for r in sub24 if r.get("maj_correct"))
    acc24 = (nc24 / n24) if n24 else None

    # per-item agreement vs banked ship_greedy (2024 byte-exactness signal). Skip
    # items where ship_greedy emitted an empty completion (min_tokens=8 changes those).
    agree = pairs = 0
    if ship_greedy:
        sg = {_norm_aime_id(r["id"]): r for r in ship_greedy.get("per_problem", [])}
        for r in sub24:
            sr = sg.get(_norm_aime_id(r["id"]))
            if sr is None:
                continue
            if (sr.get("sample_chars") or [1])[0] == 0:
                continue  # ship_greedy empty here; min_tokens=8 makes this incomparable
            pairs += 1
            if (r.get("answers") or [None])[0] == (sr.get("answers") or [None])[0]:
                agree += 1

    transfer_tol = max(0.05, 2.5 * _binom_se(TRANSFER["aime"], 30))
    matches = (acc24 is not None and abs(acc24 - TRANSFER["aime"]) <= transfer_tol)
    return {
        "axis": "aime",
        "accuracy_full60": acc,
        "n_full60": n,
        "n_correct_full60": nc,
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "extract_fail_rate": measured.get("extract_fail_rate"),
        "empty_rate_aime": round(empty_rate, 4) if empty_rate is not None else None,
        "n_char_empty": char_empty,
        "base_acc": BASE["aime"],
        "pct_of_base": round(pct, 4) if pct is not None else None,
        "passes_90pct_gate": bool(pct is not None and pct >= GATE),
        "accuracy_2024_only": round(acc24, 4) if acc24 is not None else None,
        "n_2024": n24,
        "transfer_acc": TRANSFER["aime"],
        "abs_diff_2024_vs_transfer": round(abs(acc24 - TRANSFER["aime"]), 4) if acc24 is not None else None,
        "transfer_tol": round(transfer_tol, 4),
        "matches_byteexact_transfer": bool(matches),
        "ship_greedy_pairs_2024": pairs,
        "answer_agree_vs_ship_greedy_2024": round(agree / pairs, 4) if pairs else None,
    }


def build(tag: str) -> dict[str, Any]:
    mmlu = _load(HERE / f"mmlu_pro{tag}.json")
    gpqa = _load(HERE / f"gpqa{tag}.json")
    aime = _load(HERE / f"aime_min8{tag}.json")

    base_mmlu = _load(DQE / "base_mmlu_pro.json")
    base_gpqa = _load(DQE / "base_gpqa.json")
    ship_mmlu = _load(DQE / "ship_mmlu_pro.json")
    ship_gpqa = _load(DQE / "ship_gpqa.json")
    ship_greedy_aime = _load(AIME_DIR / "ship_greedy_aime.json")

    axes: dict[str, Any] = {}
    if mmlu:
        axes["mmlu_pro"] = _mc_gate("mmlu_pro", mmlu, BASE["mmlu_pro"], TRANSFER["mmlu_pro"], ship_mmlu, base_mmlu)
    if gpqa:
        g = _mc_gate("gpqa", gpqa, BASE["gpqa"], TRANSFER["gpqa"], ship_gpqa, base_gpqa)
        g["base_acc_pr_stated"] = BASE_PR_STATED["gpqa"]
        g["pct_of_base_pr_stated"] = round(g["accuracy"] / BASE_PR_STATED["gpqa"], 4) if g["accuracy"] is not None else None
        axes["gpqa"] = g
    if aime:
        axes["aime"] = _aime_gate(aime, ship_greedy_aime)

    # headline scalars (the PR's KEY OUTPUTS)
    out: dict[str, Any] = {
        "pr": 568,
        "analysis_only": True,
        "official_tps": 0,
        "gate_threshold": GATE,
        "submission": "submissions/fa2sw_strict_byteexact_splitkv399",
        "splitkv519_mmlu_pro": axes.get("mmlu_pro", {}).get("accuracy"),
        "splitkv519_gpqa": axes.get("gpqa", {}).get("accuracy"),
        "splitkv519_aime_min8": axes.get("aime", {}).get("accuracy_full60"),
        "empty_rate_aime": axes.get("aime", {}).get("empty_rate_aime"),
        "axes": axes,
    }
    for ax in ("mmlu_pro", "gpqa", "aime"):
        a = axes.get(ax, {})
        out[f"{ax}_pct_of_base"] = a.get("pct_of_base")
        out[f"{ax}_passes_90pct_gate"] = a.get("passes_90pct_gate")
        out[f"{ax}_matches_byteexact_transfer"] = a.get("matches_byteexact_transfer")

    present = [axes[a] for a in axes]
    out["all_axes_fail_90pct_gate"] = bool(present) and all(not a.get("passes_90pct_gate") for a in present)
    out["all_axes_match_transfer"] = bool(present) and all(a.get("matches_byteexact_transfer") for a in present)
    out["prompt_sha_all_ok"] = all(axes.get(a, {}).get("prompt_sha_ok", True) for a in ("mmlu_pro", "gpqa"))

    verdict = []
    for ax, label in (("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA-D"), ("aime", "AIME")):
        a = axes.get(ax)
        if not a:
            continue
        acc = a.get("accuracy") if ax != "aime" else a.get("accuracy_full60")
        verdict.append(f"{label} {acc:.3f} = {a.get('pct_of_base'):.0%} base "
                       f"({'PASS' if a.get('passes_90pct_gate') else 'FAIL'}90%; "
                       f"transfer {'MATCH' if a.get('matches_byteexact_transfer') else 'MISS'})")
    out["verdict"] = " | ".join(verdict)
    return out


def wandb_log(combined: dict[str, Any], tag: str, name: str, group: str) -> str | None:
    sys.path.insert(0, str(ROOT))
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[aggregate] wandb import failed (analysis unaffected): {exc}", flush=True)
        return None
    run = wandb_logging.init_wandb_run(
        job_type="downstream-quality-splitkv519",
        agent="wirbel",
        name=name,
        group=group,
        notes="Direct served MMLU-Pro/GPQA/AIME of the EXACT #519 split-KV submission (PR #568).",
        tags=["downstream-quality", "splitkv519", "analysis-only", "pr-568"],
        config={
            "analysis_only": True, "official_tps": 0, "pr": 568,
            "submission": combined["submission"], "gate_threshold": GATE,
            "base": BASE, "transfer": TRANSFER,
        },
    )
    if run is None:
        print("[aggregate] wandb disabled/unavailable; skipping", flush=True)
        return None
    summary = {k: v for k, v in combined.items() if k != "axes"}
    # flatten per-axis scalars too
    for ax, a in combined["axes"].items():
        for k, v in a.items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                summary[f"{ax}/{k}"] = v
    wandb_logging.log_summary(run, summary, step=0)
    try:
        import wandb
        cols = ["axis", "accuracy", "n", "base_acc", "pct_of_base", "passes_90pct_gate",
                "transfer_acc", "matches_transfer", "answer_agree_vs_ship"]
        t = wandb.Table(columns=cols)
        for ax in ("mmlu_pro", "gpqa", "aime"):
            a = combined["axes"].get(ax)
            if not a:
                continue
            if ax == "aime":
                t.add_data(ax, a.get("accuracy_full60"), a.get("n_full60"), a.get("base_acc"),
                           a.get("pct_of_base"), a.get("passes_90pct_gate"), a.get("transfer_acc"),
                           a.get("matches_byteexact_transfer"), a.get("answer_agree_vs_ship_greedy_2024"))
            else:
                t.add_data(ax, a.get("accuracy"), a.get("n_scored"), a.get("base_acc"),
                           a.get("pct_of_base"), a.get("passes_90pct_gate"), a.get("transfer_acc"),
                           a.get("matches_byteexact_transfer"), a.get("answer_agree_vs_ship"))
        run.log({"global_step": 0, "splitkv519_quality_table": t})
    except Exception as exc:
        print(f"[aggregate] table log skipped: {exc}", flush=True)
    wandb_logging.log_json_artifact(run, name="splitkv519_quality_combined", artifact_type="quality-eval", data=combined)
    rid = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return rid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="suffix matching run_quality outputs (e.g. _smoke)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="wirbel/splitkv519-direct-quality")
    ap.add_argument("--wandb-group", default="splitkv519-direct-quality")
    args = ap.parse_args()

    combined = build(args.tag)
    combined["created_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rid = wandb_log(combined, args.tag, args.wandb_name, args.wandb_group) if args.wandb else None
    combined["wandb_run_id"] = rid

    out_path = HERE / f"quality_combined{args.tag}.json"
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"[aggregate] wrote {out_path}", flush=True)

    senpai = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "splitkv519_mmlu_pro": combined["splitkv519_mmlu_pro"],
        "splitkv519_gpqa": combined["splitkv519_gpqa"],
        "splitkv519_aime_min8": combined["splitkv519_aime_min8"],
        "empty_rate_aime": combined["empty_rate_aime"],
        "mmlu_pro_passes_90pct_gate": combined.get("mmlu_pro_passes_90pct_gate"),
        "gpqa_passes_90pct_gate": combined.get("gpqa_passes_90pct_gate"),
        "aime_passes_90pct_gate": combined.get("aime_passes_90pct_gate"),
        "all_axes_match_transfer": combined.get("all_axes_match_transfer"),
    }
    print("SENPAI-RESULT " + json.dumps(senpai), flush=True)
    print("[aggregate] VERDICT: " + combined["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
