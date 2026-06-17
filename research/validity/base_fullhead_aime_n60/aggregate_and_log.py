#!/usr/bin/env python3
"""ubel #567 aggregator: close the AIME leg of Morgan's #524 quality decision.

Reads the n=60 AIME arms produced by run_fh_arms.sh (+ optional plain-base anchor
from run_base_anchor.sh) and emits the advisor's KEY OUTPUTS, both to stdout
(SENPAI-RESULT-ready) and to W&B group ``base-fullhead-aime-harden``.

Gate (advisor, PR #567): vanilla base AIME = 0.400 is the >=90% denominator, so
the pass bar is 0.36. passes_90pct_gate := base_fullhead_aime_min8 >= 0.36.
The min8 arm is the apples-to-apples gate figure (wirbel #541 EOS-guard); the
as-served arm exposes the first-token-EOS serving artifact (empty_rate_asserved).
analysis_only=True, official_tps=0 — LOCAL served measurement, NO HF Job/fire.
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

# --- advisor anchors (PR #567 body) ---------------------------------------- #
BASE_AIME_REF = 0.400        # vanilla base AIME -> the >=90% gate DENOMINATOR
BASE_AIME_GREEDY_REF = 0.267 # vanilla base greedy maj@1 reference (same source)
QUALITY_BAR_FRAC = 0.90
PASS_BAR = BASE_AIME_REF * QUALITY_BAR_FRAC  # 0.36
N_EXPECTED = 60


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (n=60 power caveat)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def empties_and_finish(arm: dict) -> tuple[int, int, dict]:
    """(n_empty, n_samples, finish_reason_counts) from per_problem.

    Empty := the model emitted a whitespace-only completion (immediate/near-EOS).
    Counts at the SAMPLE level (k=1 greedy -> one sample/problem). Prefer the raw
    `texts` (strip()==''); fall back to sample_chars==0 if texts absent.
    """
    n_empty = 0
    n_samples = 0
    fr: dict[str, int] = {}
    for pr in arm.get("per_problem", []):
        texts = pr.get("texts")
        chars = pr.get("sample_chars") or []
        reasons = pr.get("finish_reasons") or []
        m = max(len(chars), len(texts or []), len(reasons))
        for j in range(m):
            n_samples += 1
            if texts is not None and j < len(texts):
                if not str(texts[j]).strip():
                    n_empty += 1
            elif j < len(chars) and chars[j] == 0:
                n_empty += 1
            if j < len(reasons):
                fr[reasons[j]] = fr.get(reasons[j], 0) + 1
    return n_empty, n_samples, fr


def arm_summary(arm: dict, label: str) -> dict:
    acc = arm.get("maj_k_accuracy")
    n = arm.get("n_problems")
    nc = arm.get("n_correct_maj")
    n_empty, n_samples, fr = empties_and_finish(arm)
    lo, hi = (wilson(nc, n) if (nc is not None and n) else (float("nan"), float("nan")))
    return {
        f"{label}_accuracy": acc,
        f"{label}_n_problems": n,
        f"{label}_n_correct": nc,
        f"{label}_wilson95_lo": lo,
        f"{label}_wilson95_hi": hi,
        f"{label}_extract_fail_rate": arm.get("extract_fail_rate"),
        f"{label}_empty_rate": (n_empty / n_samples) if n_samples else float("nan"),
        f"{label}_n_empty": n_empty,
        f"{label}_n_samples": n_samples,
        f"{label}_finish_reasons": fr,
        f"{label}_wall_s": arm.get("wall_s"),
    }


def main() -> int:
    fh_as = load(OUT / "aime_fh_asserved_n60.json")
    fh_m8 = load(OUT / "aime_fh_min8_n60.json")
    base = load(OUT / "aime_base_anchor_min8_n60.json")

    if not fh_as or not fh_m8:
        print("[agg] FATAL: missing required arm JSON(s) "
              f"asserved={bool(fh_as)} min8={bool(fh_m8)}", file=sys.stderr)
        return 2

    as_s = arm_summary(fh_as, "base_fullhead_aime_asserved")
    m8_s = arm_summary(fh_m8, "base_fullhead_aime_min8")
    base_s = arm_summary(base, "int4_base_aime_min8") if base else {}

    acc_as = as_s["base_fullhead_aime_asserved_accuracy"]
    acc_m8 = m8_s["base_fullhead_aime_min8_accuracy"]
    empty_as = as_s["base_fullhead_aime_asserved_empty_rate"]
    n_problems = m8_s["base_fullhead_aime_min8_n_problems"]

    # KEY OUTPUTS (gate vs the advisor's 0.400 reference denominator).
    pct_of_base_min8 = (acc_m8 / BASE_AIME_REF) if acc_m8 is not None else None
    passes_90pct_gate = bool(acc_m8 is not None and acc_m8 >= PASS_BAR)

    # Supplementary apples-to-apples: same n=60 / same harness fresh plain base.
    measured_base = base_s.get("int4_base_aime_min8_accuracy")
    pct_of_measured_base_min8 = (
        (acc_m8 / measured_base) if (acc_m8 is not None and measured_base) else None
    )
    # How well the fresh greedy base reproduces the cited greedy ref 0.267.
    base_repro_delta_vs_greedy_ref = (
        (measured_base - BASE_AIME_GREEDY_REF) if measured_base is not None else None
    )

    # The EOS-guard recovery the gate hinges on (min8 - as-served).
    min8_recovery = (acc_m8 - acc_as) if (acc_m8 is not None and acc_as is not None) else None

    summary = {
        # required key outputs
        "base_fullhead_aime_asserved": acc_as,
        "base_fullhead_aime_min8": acc_m8,
        "pct_of_base_min8": pct_of_base_min8,
        "passes_90pct_gate": passes_90pct_gate,
        "empty_rate_asserved": empty_as,
        "n_problems": n_problems,
        "analysis_only": True,
        "official_tps": 0,
        # gate framing
        "base_aime_reference_denominator": BASE_AIME_REF,
        "base_aime_greedy_reference": BASE_AIME_GREEDY_REF,
        "quality_bar_frac": QUALITY_BAR_FRAC,
        "pass_bar_abs": PASS_BAR,
        "min8_eos_guard_recovery": min8_recovery,
        # supplementary measured-base apples-to-apples
        "int4_base_aime_min8_measured": measured_base,
        "pct_of_measured_base_min8": pct_of_measured_base_min8,
        "base_repro_delta_vs_greedy_ref": base_repro_delta_vs_greedy_ref,
        # provenance
        "n_expected": N_EXPECTED,
        "years": "2024,2025-I,2025-II",
        "decode": "greedy maj@1 (k=1, temp=0, top_p=1, top_k=-1)",
        "max_tokens": fh_m8.get("sampling", {}).get("max_tokens") if isinstance(fh_m8.get("sampling"), dict) else None,
        "conc": fh_m8.get("max_num_seqs"),
    }
    summary.update(as_s)
    summary.update(m8_s)
    summary.update(base_s)

    # NaN guard on the gate metric.
    if acc_m8 is None or (isinstance(acc_m8, float) and math.isnan(acc_m8)):
        print("[agg] FATAL: min8 accuracy is NaN/None — cannot adjudicate gate", file=sys.stderr)
        return 2

    print("AGG567_SUMMARY " + json.dumps(summary, default=str))

    config = {
        "analysis_only": True,
        "official_tps": 0,
        "pr": 567,
        "experiment": "base-fullhead-aime-harden",
        "substrate": "base_int4_native_262k_head",
        "submission": fh_m8.get("submission"),
        "serve_overrides": fh_m8.get("serve_overrides"),
        "eos_guard": "request min_tokens=8 (MIN_TOKENS_FLOOR disabled; request controls EOS)",
        "n_problems": n_problems,
        "years": "2024,2025-I,2025-II",
    }

    run = wandb_logging.init_wandb_run(
        job_type="base-fullhead-aime-harden",
        agent="ubel",
        name="ubel/base-fullhead-aime-n60",
        group="base-fullhead-aime-harden",
        notes=(
            "PR #567: base_fullhead AIME at full n=60 (AIME-2024 + AIME-2025 I/II), "
            "greedy maj@1, conc=32. min8 arm is the apples-to-apples #524 gate figure "
            "(wirbel #541 EOS-guard); as-served arm exposes the first-token-EOS "
            "serving artifact. Gate denominator = vanilla base 0.400 (pass bar 0.36). "
            "LOCAL served measurement; analysis_only, official_tps=0, NO FIRE."
        ),
        tags=["aime", "n60", "analysis-only", "pr-567", "base-fullhead", "quality-gate"],
        config=config,
    )
    if run is None:
        print("[agg] wandb disabled/unavailable; metrics above + JSON only", flush=True)
        (OUT / "agg567_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return 0

    wandb_logging.log_summary(run, summary, step=0)
    for nm, arm in (("aime_fh_asserved_n60", fh_as), ("aime_fh_min8_n60", fh_m8),
                    ("aime_base_anchor_min8_n60", base)):
        if arm:
            slim = {k: v for k, v in arm.items() if k != "per_problem"}
            # keep per_problem but drop the big texts to bound artifact size
            pp = [{k: v for k, v in p.items() if k != "texts"} for p in arm.get("per_problem", [])]
            slim["per_problem_no_texts"] = pp
            wandb_logging.log_json_artifact(run, name=nm, artifact_type="aime-n60", data=slim)
    run_id = getattr(run, "id", None)
    wandb_logging.finish_wandb(run)
    print(f"[agg] wandb run_id={run_id}", flush=True)
    (OUT / "agg567_summary.json").write_text(
        json.dumps({**summary, "wandb_run_id": run_id}, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
