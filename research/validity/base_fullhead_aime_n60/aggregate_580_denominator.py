#!/usr/bin/env python3
"""ubel #580 aggregator: ground the unquantized-base AIME denominator.

The ">=90% of unquantized-base AIME" gate is only as trustworthy as its
DENOMINATOR. Morgan's posted anchor cites unquantized-base AIME = 0.400 (=> 90%
bar = 0.360). But #567 measured base_fullhead (int4 QAT + native 262k head) AIME
= 0.1167 and plain int4 base AIME = 0.0667 on the SAME greedy maj@1 / min_tokens=8
harness. If the cited 0.400 is right, the int4-quant tax (0.400 - 0.1167 ~ 0.28)
is larger than the entire 10% allowance, so the gate would be structurally
unpassable by ANY known int4 config.

This reads the bf16 unquantized-base AIME arm (run by run_bf16_base_anchor.sh,
EXACT #567 harness) plus the existing int4 arms and emits the #580 deliverables to
stdout and to W&B group ``base-quality-denominator-grounding``.
analysis_only=True, official_tps=0 -- LOCAL served measurement, NO HF Job.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
from scripts import wandb_logging  # noqa: E402

OUT = ROOT / "research" / "validity" / "base_fullhead_aime_n60"

# --- advisor anchors (PR #580 body / Morgan's posted gate) ------------------ #
CITED_DENOMINATOR = 0.400      # cited unquantized-base AIME (the gate denominator under test)
QUALITY_BAR_FRAC = 0.90        # ">=90% of base"
CITED_BAR = CITED_DENOMINATOR * QUALITY_BAR_FRAC  # 0.360 (Morgan's corrected bar)
N_EXPECTED = 60


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion (n=60 power caveat)."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def acc_of(arm: dict) -> tuple[float | None, int | None, int | None]:
    return arm.get("maj_k_accuracy"), arm.get("n_correct_maj"), arm.get("n_problems")


def main() -> int:
    bf16 = load(OUT / "aime_bf16_base_min8_n60.json")          # NEW: unquantized bf16 base
    int4_base = load(OUT / "aime_base_anchor_min8_n60.json")   # plain int4 base (0.0667)
    fullhead = load(OUT / "aime_fh_min8_n60.json")             # base_fullhead int4 (0.1167)

    if not bf16:
        print("[agg580] FATAL: missing bf16 base arm aime_bf16_base_min8_n60.json", file=sys.stderr)
        return 2

    measured, nc, n = acc_of(bf16)
    if measured is None or (isinstance(measured, float) and math.isnan(measured)):
        print("[agg580] FATAL: bf16 base accuracy is NaN/None", file=sys.stderr)
        return 2
    wlo, whi = wilson(nc, n)

    int4_base_acc = int4_base.get("maj_k_accuracy")
    fullhead_acc = fullhead.get("maj_k_accuracy")
    # Best known int4 AIME = max over the int4 configs we have measured on this harness.
    known_int4 = [a for a in (fullhead_acc, int4_base_acc) if a is not None]
    best_known_int4 = max(known_int4) if known_int4 else None

    # --- #580 KEY OUTPUTS --------------------------------------------------- #
    # cited_denominator_confirmed: is the cited 0.400 inside the measured 95% Wilson CI?
    cited_confirmed = bool(not math.isnan(wlo) and wlo <= CITED_DENOMINATOR <= whi)
    # 90% bar from the MEASURED denominator (the honest apples-to-apples bar).
    bar_measured = QUALITY_BAR_FRAC * measured
    # int4-quant tax = unquantized base - best int4 (base_fullhead 0.1167).
    int4_tax = (measured - fullhead_acc) if fullhead_acc is not None else None
    # Achievability: can the best KNOWN int4 config clear the bar?
    achievable_measured_bar = bool(best_known_int4 is not None and best_known_int4 >= bar_measured)
    achievable_cited_bar = bool(best_known_int4 is not None and best_known_int4 >= CITED_BAR)

    # Recommended denominator (free text), chosen by the measured outcome.
    delta_vs_cited = measured - CITED_DENOMINATOR
    if cited_confirmed:
        recommended = (
            f"CITE-CONFIRMED: harness-measured unquantized-base AIME={measured:.4f} "
            f"(95% CI [{wlo:.3f},{whi:.3f}]) is consistent with the cited 0.400, so the "
            f"0.360 bar stands. But the int4-quant tax ({int4_tax:.3f}) exceeds the 0.10 "
            f"allowance: best known int4 (base_fullhead {fullhead_acc:.4f}) cannot clear "
            f"0.360. AIME is therefore the WRONG axis to gate int4 submissions on an "
            f"unquantized denominator. Recommend gating int4 submissions against the "
            f"int4-base denominator (preserve-quantization gate) or dropping AIME as an "
            f"int4 gate axis."
        )
    else:
        recommended = (
            f"CITE-REFUTED: harness-measured unquantized-base AIME={measured:.4f} "
            f"(95% CI [{wlo:.3f},{whi:.3f}]) is NOT consistent with the cited 0.400 "
            f"(delta {delta_vs_cited:+.3f}); the 0.400 cite is from a different protocol "
            f"(likely maj@k/thinking-enabled), not our greedy maj@1 no-thinking min8 "
            f"harness. Replace the denominator with the harness-grounded bf16 base "
            f"{measured:.4f} -> bar {bar_measured:.4f}; best known int4 (base_fullhead "
            f"{fullhead_acc:.4f}) "
            + ("CLEARS" if achievable_measured_bar else "still misses")
            + " this honest bar."
        )

    summary = {
        # required #580 deliverables
        "unquantized_base_aime_min8": measured,
        "cited_denominator_confirmed": cited_confirmed,
        "aime_gate_bar_90pct": round(bar_measured, 6),
        "int4_quant_tax_abs": int4_tax,
        "aime_gate_achievable_by_any_int4": achievable_measured_bar,
        "recommended_denominator": recommended,
        "analysis_only": True,
        "official_tps": 0,
        # measured-denominator framing
        "unquantized_base_aime_n_correct": nc,
        "unquantized_base_aime_n_problems": n,
        "unquantized_base_aime_wilson95_lo": wlo,
        "unquantized_base_aime_wilson95_hi": whi,
        "unquantized_base_aime_wall_s": bf16.get("wall_s"),
        "unquantized_base_aime_extract_fail_rate": bf16.get("extract_fail_rate"),
        # cited-denominator framing (for the gate as currently posted)
        "cited_denominator": CITED_DENOMINATOR,
        "cited_bar_90pct": CITED_BAR,
        "delta_measured_vs_cited": delta_vs_cited,
        "aime_gate_achievable_by_any_int4_cited_bar": achievable_cited_bar,
        # int4 reference arms (same harness)
        "int4_base_aime_min8": int4_base_acc,
        "base_fullhead_aime_min8": fullhead_acc,
        "best_known_int4_aime": best_known_int4,
        "int4_quant_tax_vs_plain_int4_base": (
            (measured - int4_base_acc) if int4_base_acc is not None else None
        ),
        # provenance
        "n_expected": N_EXPECTED,
        "years": "2024,2025-I,2025-II",
        "decode": "greedy maj@1 (k=1, temp=0, top_p=1, top_k=-1), no-thinking, min_tokens=8",
        "max_tokens": (bf16.get("sampling", {}) or {}).get("max_tokens"),
        "conc": bf16.get("max_num_seqs"),
        "submission": bf16.get("submission"),
        "serve_overrides": bf16.get("serve_overrides"),
    }

    print("AGG580_SUMMARY " + json.dumps(summary, default=str))
    print(
        "SENPAI-RESULT: " + json.dumps({
            "terminal": True, "status": "complete", "pending_arms": False,
            "analysis_only": True, "official_tps": 0,
            "wandb_run_ids": ["<filled-after-init>"],
            "primary_metric": {"name": "unquantized_base_aime_min8", "value": measured},
            "test_metric": {"name": "int4_quant_tax_abs", "value": int4_tax},
        })
    )

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 580,
        "experiment": "base-aime-denominator-grounding",
        "substrate": "unquantized_bf16_base_native_262k_head",
        "submission": bf16.get("submission"),
        "serve_overrides": bf16.get("serve_overrides"),
        "eos_guard": "request min_tokens=8 (matches #567 gate arm)",
        "n_problems": n,
        "years": "2024,2025-I,2025-II",
        "cited_denominator": CITED_DENOMINATOR,
        "harness_run_ref": "ns5l6i28 (#567 base_fullhead min8)",
    }

    run = wandb_logging.init_wandb_run(
        job_type="base-aime-denominator-grounding",
        agent="ubel",
        name="ubel/base-aime-denominator",
        group="base-quality-denominator-grounding",
        notes=(
            "PR #580: ground the unquantized-base AIME denominator. Serve the "
            "UNQUANTIZED bf16 google/gemma-4-E4B-it (full native 262k head) and run "
            "AIME n=60 (2024 + 2025 I/II) greedy maj@1 min_tokens=8 -- the EXACT #567 "
            "harness (run ns5l6i28) -- so the number is directly comparable to "
            "base_fullhead int4 (0.1167) and plain int4 base (0.0667). Confirms/refutes "
            "the cited 0.400 denominator and quantifies the int4-quant tax. LOCAL served "
            "measurement; analysis_only, official_tps=0, NO FIRE."
        ),
        tags=["aime", "n60", "analysis-only", "pr-580", "bf16-base", "quality-gate",
              "denominator-grounding"],
        config=config,
    )
    if run is None:
        print("[agg580] wandb disabled/unavailable; metrics above + JSON only", flush=True)
        (OUT / "agg580_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return 0

    wandb_logging.log_summary(run, summary, step=0)
    for nm, arm in (("aime_bf16_base_min8_n60", bf16),
                    ("aime_base_anchor_min8_n60", int4_base),
                    ("aime_fh_min8_n60", fullhead)):
        if arm:
            slim = {k: v for k, v in arm.items() if k != "per_problem"}
            pp = [{k: v for k, v in p.items() if k != "texts"} for p in arm.get("per_problem", [])]
            slim["per_problem_no_texts"] = pp
            wandb_logging.log_json_artifact(run, name=nm, artifact_type="aime-n60", data=slim)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    print(f"[agg580] wandb run_id={run_id}", flush=True)
    (OUT / "agg580_summary.json").write_text(
        json.dumps({**summary, "wandb_run_id": run_id}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
