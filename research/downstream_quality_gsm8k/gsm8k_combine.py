"""Assemble the GSM8K named-gate verdict from per-arm eval JSONs (PR #533).

Consumes the ``gsm8k_eval.py`` outputs for up to four arms in two decode regimes
(sampled = PRIMARY, greedy = diagnostic) and computes Morgan's #515 ship-quality
gate on the missing GSM8K cell. The arm set was re-anchored by the advisor
(morganmcg1, 2026-06-16) after the premise correction that ``osoi5-v0-baked`` is
already a 16,384-row head (262k->16k baked in) and ``LM_HEAD_PRUNE`` only does
16k->12k:

  * ``base``         = vanilla gemma-4-E4B-it, native 262,144-row head (denominator)
  * ``ship12k``      = the live osoi5-12k served substrate (12,288-row head)
  * ``osoi5_16k``    = the ship substrate with ``LM_HEAD_PRUNE=0`` (16,384-row head).
                       Honestly labelled: this is NOT "full-head" and NOT
                       "quality-safe by construction" -- it just answers whether
                       16k rescues GSM8K where it recovered nothing on AIME.
  * ``truefullhead`` = the GENUINE 262,144-row full head from base-int4 x surgical
                       (fern #535). This is the cell the ship certs actually need;
                       OPTIONAL/pending until fern #535's checkpoint serves.

For each non-base arm:

  * ``<arm>_pct_of_base`` = arm acc / base acc
  * ``<arm>_meets_90pct`` = pct >= 0.90  (the #483 bar Morgan ACKed)

The headline number is the **sampled** gate (PRIMARY). Greedy is reported as a
diagnostic: under greedy the surgical-357 fused-accept kernel emits the target
argmax (spec on/off identical), so any greedy gap is purely the head-prune's
-inf mask, not the sampler.

Apples-to-apples is enforced: arms must share the same seeded item set, n_shot,
few-shot exemplars, and sampling params for a given regime, or the verdict is
marked INVALID.

``truefullhead`` (and, if a run has not been done yet, ``osoi5_16k``) are
OPTIONAL: absent JSONs are reported null / ``pending`` and the gate still emits
for whatever arms are present.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "seed", "enable_thinking")
GATE = 0.90

# Non-base arms, in report order. The genuine 262k full-head cert cell is the
# LAST one (truefullhead); osoi5_16k is the honestly-relabelled "16k point".
VARIANTS = ("ship12k", "osoi5_16k", "truefullhead")
ALL_ARMS = ("base",) + VARIANTS
# The arm whose absence keeps the leg non-terminal (the cell the certs need).
PENDING_ARM = "truefullhead"

_DISPLAY = {
    "base": "base (262k)",
    "ship12k": "ship-12k",
    "osoi5_16k": "osoi5-16k",
    "truefullhead": "true-full-head (base-int4 262k, fern #535)",
}


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _wilson(n_correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (acc CI)."""
    if n == 0:
        return (0.0, 0.0)
    phat = n_correct / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _arm_summary(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if d is None:
        return None
    n = int(d.get("n_problems", 0))
    nc = int(d.get("n_correct", 0))
    lo, hi = _wilson(nc, n)
    return {
        "label": d.get("label"),
        "regime": d.get("regime"),
        "accuracy": d.get("accuracy"),
        "n_correct": nc,
        "n_problems": n,
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "strict_rate": d.get("strict_rate"),
        "extract_fail_rate": d.get("extract_fail_rate"),
        "truncation_rate": d.get("truncation_rate"),
        "submission": d.get("submission"),
        "serve_overrides": d.get("serve_overrides"),
        "model": d.get("model"),
    }


def _parity(base: dict[str, Any], other: dict[str, Any], name: str) -> list[str]:
    issues: list[str] = []
    if list(base.get("item_ids") or []) != list(other.get("item_ids") or []):
        issues.append(f"{name}: item_ids differ from base")
    if base.get("n_shot") != other.get("n_shot"):
        issues.append(f"{name}: n_shot {other.get('n_shot')} != base {base.get('n_shot')}")
    if list(base.get("fewshot_sig") or []) != list(other.get("fewshot_sig") or []):
        issues.append(f"{name}: fewshot exemplars differ from base")
    bs, os_ = base.get("sampling") or {}, other.get("sampling") or {}
    for key in _SAMPLING_KEYS:
        if bs.get(key) != os_.get(key):
            issues.append(f"{name}: sampling.{key} {os_.get(key)} != base {bs.get(key)}")
    return issues


def _gate(variant: dict[str, Any] | None, base: dict[str, Any] | None) -> dict[str, Any]:
    if variant is None or base is None or not base.get("accuracy"):
        return {"acc": (variant or {}).get("accuracy"), "pct_of_base": None, "meets_90pct": None}
    pct = variant["accuracy"] / base["accuracy"] if base["accuracy"] else None
    return {
        "acc": variant["accuracy"],
        "pct_of_base": round(pct, 4) if pct is not None else None,
        "meets_90pct": bool(pct is not None and pct >= GATE),
    }


def combine(arms: dict[str, dict[str, dict[str, Any] | None]]) -> dict[str, Any]:
    """arms[regime][label] -> per-arm eval dict (or None). regimes: sampled, greedy."""
    out: dict[str, Any] = {"gate_threshold": GATE}
    parity_issues: list[str] = []
    n_items = n_shot = None
    sampling_by_regime: dict[str, Any] = {}

    for regime in ("sampled", "greedy"):
        regime_arms = arms.get(regime, {})
        base = regime_arms.get("base")
        suffix = "" if regime == "sampled" else "_greedy"

        out[f"gsm8k_base_acc{suffix}"] = (base or {}).get("accuracy")
        for arm in VARIANTS:
            d = regime_arms.get(arm)
            g = _gate(d, base)
            out[f"gsm8k_{arm}_acc{suffix}"] = (d or {}).get("accuracy")
            out[f"{arm}_pct_of_base{suffix}"] = g["pct_of_base"]
            out[f"{arm}_meets_90pct{suffix}"] = g["meets_90pct"]

        # Legacy/cert alias: the original PR card's "fullhead" cell == the GENUINE
        # 262k full head == truefullhead (NOT the 16k point).
        out[f"gsm8k_fullhead_acc{suffix}"] = out[f"gsm8k_truefullhead_acc{suffix}"]
        out[f"fullhead_pct_of_base{suffix}"] = out[f"truefullhead_pct_of_base{suffix}"]
        out[f"fullhead_meets_90pct{suffix}"] = out[f"truefullhead_meets_90pct{suffix}"]

        out[f"arms{suffix}"] = {arm: _arm_summary(regime_arms.get(arm)) for arm in ALL_ARMS}

        if base is not None:
            n_items = base.get("n_problems")
            n_shot = base.get("n_shot")
            sampling_by_regime[regime] = base.get("sampling")
            for arm in VARIANTS:
                d = regime_arms.get(arm)
                if d is not None:
                    parity_issues += _parity(base, d, f"{regime}/{arm}")

    out["n_items"] = n_items
    out["n_shot"] = n_shot
    out["sampling"] = sampling_by_regime
    out["parity_issues"] = parity_issues
    out["apples_to_apples"] = not parity_issues
    out["truefullhead_pending"] = arms.get("sampled", {}).get("truefullhead") is None
    out["osoi5_16k_pending"] = arms.get("sampled", {}).get("osoi5_16k") is None
    # back-compat flag consumed by older readers / the SENPAI-RESULT line
    out["fullhead_pending"] = out["truefullhead_pending"]
    out["pending_arms"] = out["truefullhead_pending"] or out["osoi5_16k_pending"]

    # ---- PRIMARY (sampled) verdict ----
    base_s = out.get("gsm8k_base_acc")
    bits: list[str] = []

    def _line(arm: str, collapse_word: str, hold_word: str) -> str | None:
        acc = out.get(f"gsm8k_{arm}_acc")
        pct = out.get(f"{arm}_pct_of_base")
        if acc is None or base_s is None or pct is None:
            return None
        collapse = pct < GATE
        return (
            f"{_DISPLAY[arm]} {collapse_word if collapse else hold_word} GSM8K: "
            f"{acc:.3f} = {pct:.1%} of base {base_s:.3f} "
            f"({'<' if collapse else '>='}90% gate)"
        )

    ship_line = _line("ship12k", "COLLAPSES", "HOLDS")
    if ship_line:
        bits.append(ship_line)
    o16_line = _line("osoi5_16k", "DOES NOT rescue", "RESCUES")
    if o16_line:
        bits.append(o16_line)
    elif out["osoi5_16k_pending"]:
        bits.append("osoi5-16k PENDING (LM_HEAD_PRUNE=0 run not yet done)")

    full_acc = out.get("gsm8k_truefullhead_acc")
    full_pct = out.get("truefullhead_pct_of_base")
    if full_acc is not None and base_s is not None and full_pct is not None:
        clears = full_pct >= GATE
        bits.append(
            f"{_DISPLAY['truefullhead']} {'CLEARS' if clears else 'FAILS'} the >=90% gate: "
            f"{full_acc:.3f} = {full_pct:.1%} of base {base_s:.3f}"
        )
    elif out["truefullhead_pending"]:
        bits.append("true-full-head PENDING (base-int4 262k x surgical from fern #535 not yet serving)")

    if parity_issues:
        bits.insert(0, "INVALID A/B (not apples-to-apples): " + "; ".join(parity_issues))
    out["verdict"] = " | ".join(bits) if bits else "no arms loaded"
    return out


def _wandb_log(combined: dict[str, Any], arms: dict[str, dict[str, dict | None]], args: argparse.Namespace) -> str | None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from scripts import wandb_logging
    except Exception as exc:
        print(f"[gsm8k-combine] wandb_logging import failed (analysis unaffected): {exc}", flush=True)
        return None

    def _meta(regime: str, label: str, key: str) -> Any:
        d = arms.get(regime, {}).get(label)
        return d.get(key) if d else None

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "experiment": "downstream-quality-gsm8k",
        "n_items": combined["n_items"],
        "n_shot": combined["n_shot"],
        "sampling": combined["sampling"],
        "gate_threshold": GATE,
        "pr": 533,
        "truefullhead_pending": combined["truefullhead_pending"],
        "osoi5_16k_pending": combined["osoi5_16k_pending"],
    }
    for arm in ALL_ARMS:
        config[f"{arm}_submission"] = _meta("sampled", arm, "submission")
        config[f"{arm}_serve_overrides"] = _meta("sampled", arm, "serve_overrides")
    run = wandb_logging.init_wandb_run(
        job_type="downstream-quality-gsm8k",
        agent="wirbel",
        name=args.wandb_name,
        group=args.wandb_group,
        notes="GSM8K 8-shot base / ship-12k / osoi5-16k / true-full-head named-gate leg (PR #533).",
        tags=["gsm8k", "downstream-quality", "analysis-only", "pr-533"],
        config=config,
    )
    if run is None:
        print("[gsm8k-combine] wandb disabled/unavailable; skipping log", flush=True)
        return None

    summary = {k: v for k, v in combined.items() if not k.startswith("arms")}
    wandb_logging.log_summary(run, summary, step=0)
    try:
        import wandb

        cols = ["regime", "arm", "accuracy", "n_correct", "n_problems", "pct_of_base",
                "meets_90pct", "strict_rate", "extract_fail_rate", "truncation_rate"]
        table = wandb.Table(columns=cols)
        for regime in ("sampled", "greedy"):
            suffix = "" if regime == "sampled" else "_greedy"
            armset = combined.get(f"arms{suffix}", {})
            for arm in ALL_ARMS:
                a = armset.get(arm)
                if not a:
                    continue
                pct = combined.get(f"{arm}_pct_of_base{suffix}") if arm != "base" else 1.0
                meets = combined.get(f"{arm}_meets_90pct{suffix}") if arm != "base" else True
                table.add_data(regime, arm, a["accuracy"], a["n_correct"], a["n_problems"],
                               pct, meets, a["strict_rate"], a["extract_fail_rate"], a["truncation_rate"])
        run.log({"global_step": 0, "gsm8k_gate_table": table})
    except Exception as exc:
        print(f"[gsm8k-combine] table log skipped: {exc}", flush=True)
    wandb_logging.log_json_artifact(run, name="gsm8k_gate_combined", artifact_type="gsm8k-eval", data=combined)
    for regime in ("sampled", "greedy"):
        for label in ALL_ARMS:
            d = arms.get(regime, {}).get(label)
            if d is not None:
                wandb_logging.log_json_artifact(
                    run, name=f"gsm8k_{label}_{regime}_raw", artifact_type="gsm8k-eval", data=d)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    return run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("research/downstream_quality_gsm8k"),
                    help="dir holding <label>_<regime>.json files")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--ship-label", default="ship12k")
    ap.add_argument("--osoi16k-label", default="osoi5_16k")
    ap.add_argument("--truefullhead-label", default="truefullhead")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="wirbel/gsm8k-base-ship-fullhead-gate")
    ap.add_argument("--wandb-group", default="gsm8k-base-ship-fullhead")
    args = ap.parse_args(argv)

    label_by_arm = {
        "base": args.base_label,
        "ship12k": args.ship_label,
        "osoi5_16k": args.osoi16k_label,
        "truefullhead": args.truefullhead_label,
    }
    arms: dict[str, dict[str, dict | None]] = {}
    for regime in ("sampled", "greedy"):
        arms[regime] = {
            arm: _load(args.dir / f"{label_by_arm[arm]}_{regime}.json") for arm in ALL_ARMS
        }

    combined = combine(arms)
    combined["created_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    run_id = _wandb_log(combined, arms, args) if args.wandb else None
    combined["wandb_run_id"] = run_id

    out_path = args.out or (args.dir / "gsm8k_gate_combined.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"[gsm8k-combine] wrote {out_path}", flush=True)

    senpai = {
        "analysis_only": True,
        "official_tps": 0,
        "terminal": not combined["pending_arms"],
        "pending_arms": combined["pending_arms"],
        "n_items": combined["n_items"],
        "n_shot": combined["n_shot"],
        "gsm8k_base_acc": combined.get("gsm8k_base_acc"),
        "gsm8k_ship12k_acc": combined.get("gsm8k_ship12k_acc"),
        "gsm8k_osoi5_16k_acc": combined.get("gsm8k_osoi5_16k_acc"),
        "gsm8k_truefullhead_acc": combined.get("gsm8k_truefullhead_acc"),
        "ship12k_pct_of_base": combined.get("ship12k_pct_of_base"),
        "osoi5_16k_pct_of_base": combined.get("osoi5_16k_pct_of_base"),
        "truefullhead_pct_of_base": combined.get("truefullhead_pct_of_base"),
        "ship12k_meets_90pct": combined.get("ship12k_meets_90pct"),
        "osoi5_16k_meets_90pct": combined.get("osoi5_16k_meets_90pct"),
        "truefullhead_meets_90pct": combined.get("truefullhead_meets_90pct"),
        "wandb_run_id": run_id,
    }
    print("SENPAI-RESULT " + json.dumps(senpai), flush=True)
    print("[gsm8k-combine] VERDICT: " + combined["verdict"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
