#!/usr/bin/env python3
"""PR #564 — aggregate base_fullhead vs ple_fold strict-CI-lb at larger n. LOCAL, NO FIRE.

Stage 2 (load-bearing): does base_fullhead's Wilson CI-lb clear 0.90 x the ple_fold point
once the CIs tighten at larger n? Reports strict_cilb_{mmlu,gpqa,both}_passes_largern, the
margin (base_fullhead_cilb - 0.90*ple_fold_point) per axis, and the delta vs the #557
n=500/198 margins (computed from the banked #542/#557 files, not hardcoded).

Stage 3 (paired mechanism): on the EXACT same items, McNemar/sign-test on the discordant
pairs -> is the base_fullhead<->ple_fold point gap surgical-attention cost or sampling noise?

Usage:  aggregate.py [--no-wandb] [--mmlu-n N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# wandb OUTPUT dirs (./wandb with run-* but no __init__.py) live at the target root AND in
# this script's dir (from the heartbeat) and shadow the real `wandb` package. Drop any
# sys.path entry whose `wandb` subdir is NOT a real package, so `import wandb` resolves to
# site-packages. Keep entries whose wandb/ has __init__.py (the real package).
def _is_wandb_shadow(p: str) -> bool:
    try:
        w = Path(p or ".") / "wandb"
        return w.is_dir() and not (w / "__init__.py").exists()
    except OSError:
        return False
sys.path[:] = [p for p in sys.path if not _is_wandb_shadow(p)]
os.environ.setdefault("WANDB_DIR", "/tmp/wandb_stark564")
os.makedirs("/tmp/wandb_stark564", exist_ok=True)

ROOT = Path("/workspace/senpai/target")
HERE = ROOT / "research/validity/base_fullhead_strict_cilb_largern"
F542 = ROOT / "research/validity/base_fullhead_shortchain_quality"   # #542 banked base_fullhead n=500
F557 = ROOT / "research/validity/vanilla_base_serve_regression"      # #557 banked ple_fold n=500/198
BUILD = "vllm-0.22.1rc1.dev307+g3e8afdf78"

# Morgan #515 floors (greedy program record), for context only.
FLOOR_MMLU, FLOOR_GPQA = 0.601, 0.423


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def two_prop_z(k1, n1, k2, n2):
    if not n1 or not n2:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) / se if se else float("nan")


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p over the b+c discordant pairs (H0: p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, k) for k in range(lo + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, ValueError):
        return None


def _kn(d):
    return (d["n_correct"], d["n_scored"]) if d else (0, 0)


def _by_id(d):
    return {str(r["id"]): r for r in (d.get("per_sample") or [])}


def prompt_mismatch(a, b):
    """Count items whose prompt_sha differs across arms (byte-identical item-set check)."""
    am, bm = _by_id(a), _by_id(b)
    common = set(am) & set(bm)
    mism = [i for i in common if am[i].get("prompt_sha") != bm[i].get("prompt_sha")]
    return len(mism), len(common)


def strict_margin(fh, pf):
    """base_fullhead Wilson CI-lb - 0.90 * ple_fold point. >=0 => strict-CI-lb PASS."""
    fhk, fhn = _kn(fh)
    pfk, pfn = _kn(pf)
    _, fhlo, _ = wilson(fhk, fhn)
    pf_pt = pfk / pfn if pfn else float("nan")
    gate = 0.90 * pf_pt
    return fhlo, gate, (fhlo - gate)


def paired_mcnemar(fh, pf):
    """Discordant pairs on the byte-identical item set: b = base_fullhead-correct & ple_fold-wrong,
    c = base_fullhead-wrong & ple_fold-correct. ple_fold favored when c>b."""
    am, bm = _by_id(fh), _by_id(pf)
    common = sorted(set(am) & set(bm))
    b = c = concordant = 0
    flips_to_ple_fold = []   # items ple_fold gets right that base_fullhead gets wrong
    flips_to_fullhead = []   # items base_fullhead gets right that ple_fold gets wrong
    for i in common:
        fh_ok = bool(am[i].get("correct"))
        pf_ok = bool(bm[i].get("correct"))
        if fh_ok and not pf_ok:
            b += 1
            flips_to_fullhead.append(i)
        elif pf_ok and not fh_ok:
            c += 1
            flips_to_ple_fold.append(i)
        else:
            concordant += 1
    p = mcnemar_exact_p(b, c)
    return {
        "n_paired": len(common), "concordant": concordant,
        "b_fullhead_only_correct": b, "c_ple_fold_only_correct": c,
        "mcnemar_exact_p": p, "significant": bool(p < 0.05),
        "flips_to_ple_fold": flips_to_ple_fold, "flips_to_fullhead": flips_to_fullhead,
    }


def axis_report(name, fh, pf, fh542, pf557):
    """Full per-axis report: points+Wilson, strict gate at larger n, delta vs #557, paired McNemar."""
    fhk, fhn = _kn(fh)
    pfk, pfn = _kn(pf)
    fhp, fhlo, fhhi = wilson(fhk, fhn)
    pfp, pflo, pfhi = wilson(pfk, pfn)

    # Stage 2 — strict CI-lb at larger n
    lo_ln, gate_ln, margin_ln = strict_margin(fh, pf)
    # delta vs #557 n=500/198 (banked)
    lo_557, gate_557, margin_557 = strict_margin(fh542, pf557)

    # Stage 3 — paired mechanism at larger n
    mism, ncommon = prompt_mismatch(fh, pf)
    mc = paired_mcnemar(fh, pf)
    point_gap = (pfp - fhp)  # ple_fold - base_fullhead
    if not mc["significant"]:
        mechanism = "sampling_noise"
    elif point_gap > 0:
        mechanism = "surgical_attention_cost"
    else:
        mechanism = "surgical_attention_advantage"

    return {
        "axis": name,
        "n_used": fhn, "n_ple_fold": pfn,
        "base_fullhead_acc": fhp, "base_fullhead_k": fhk, "base_fullhead_n": fhn,
        "base_fullhead_wilson_lo": fhlo, "base_fullhead_wilson_hi": fhhi,
        "ple_fold_acc": pfp, "ple_fold_k": pfk, "ple_fold_n": pfn,
        "ple_fold_wilson_lo": pflo, "ple_fold_wilson_hi": pfhi,
        # Stage 2
        "gate_0p90x_ple_fold": gate_ln,
        "strict_cilb_passes_largern": bool(margin_ln >= 0),
        "strict_cilb_margin_largern": margin_ln,
        "base_fullhead_cilb_largern": lo_ln,
        "strict_cilb_margin_557": margin_557,
        "delta_margin_vs_557": (margin_ln - margin_557),
        "n_used_557": fh542["n_scored"] if fh542 else None,
        # Stage 3
        "point_gap": point_gap,
        "two_prop_z": two_prop_z(pfk, pfn, fhk, fhn),
        "paired": mc,
        "point_gap_is_significant": mc["significant"],
        "point_gap_mechanism": mechanism,
        "n_prompt_mismatch": mism, "n_common": ncommon,
        "floor": FLOOR_MMLU if name == "mmlu_pro" else FLOOR_GPQA,
        "base_fullhead_point_meets_floor": bool(fhp >= (FLOOR_MMLU if name == "mmlu_pro" else FLOOR_GPQA)),
    }


def _runid():
    p = HERE / "wandb_run_id.txt"
    return p.read_text().strip() if p.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--mmlu-n", type=int, default=None, help="annotate the intended MMLU n")
    a = ap.parse_args()

    # larger-n arms (this PR)
    fh_m = _load(HERE / "base_fullhead_mmlu_pro.json")
    fh_g = _load(HERE / "base_fullhead_gpqa_diamond.json")
    pf_m = _load(HERE / "ple_fold_mmlu_pro.json")
    pf_g = _load(HERE / "ple_fold_gpqa_diamond.json")
    # banked #542/#557 n=500/198 (for delta-vs-#557 margins)
    fh542_m = _load(F542 / "fullhead_mmlu_pro.json")
    fh542_g = _load(F542 / "fullhead_gpqa.json")
    pf557_m = _load(F557 / "ple_fold_mmlu_pro.json")
    pf557_g = _load(F557 / "ple_fold_gpqa_diamond.json")
    sd = _load(HERE / "selfdet.json")

    if fh_m is None or pf_m is None:
        print("[aggregate-564] MISSING MMLU arms — run not complete", file=sys.stderr)
        return 2

    mmlu = axis_report("mmlu_pro", fh_m, pf_m, fh542_m, pf557_m)
    gpqa = axis_report("gpqa_diamond", fh_g, pf_g, fh542_g, pf557_g) if (fh_g and pf_g) else None

    strict_mmlu = mmlu["strict_cilb_passes_largern"]
    strict_gpqa = gpqa["strict_cilb_passes_largern"] if gpqa else None
    strict_both = bool(strict_mmlu and (strict_gpqa if gpqa else False))
    # binding margin = the more-negative (or smaller) of the two axis margins
    margins = [mmlu["strict_cilb_margin_largern"]] + ([gpqa["strict_cilb_margin_largern"]] if gpqa else [])
    binding_margin = min(margins)

    self_det = sd.get("self_det") if sd else None

    marker = {
        "pr": 564, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "vllm_build": BUILD,
        "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct (stock int4, native 262k head)",
        "protocol": "greedy temp=0/top_p=1/top_k=0",
        # Stage 1 — points + CIs + n
        "mmlu_pro_base_fullhead_largern": mmlu["base_fullhead_acc"],
        "mmlu_pro_ple_fold_largern": mmlu["ple_fold_acc"],
        "mmlu_pro_base_fullhead_wilson_lo": mmlu["base_fullhead_wilson_lo"],
        "mmlu_pro_base_fullhead_wilson_hi": mmlu["base_fullhead_wilson_hi"],
        "mmlu_pro_ple_fold_wilson_lo": mmlu["ple_fold_wilson_lo"],
        "mmlu_pro_ple_fold_wilson_hi": mmlu["ple_fold_wilson_hi"],
        "mmlu_n_used": mmlu["n_used"],
        "gpqa_base_fullhead_largern": gpqa["base_fullhead_acc"] if gpqa else None,
        "gpqa_ple_fold_largern": gpqa["ple_fold_acc"] if gpqa else None,
        "gpqa_base_fullhead_wilson_lo": gpqa["base_fullhead_wilson_lo"] if gpqa else None,
        "gpqa_base_fullhead_wilson_hi": gpqa["base_fullhead_wilson_hi"] if gpqa else None,
        "gpqa_ple_fold_wilson_lo": gpqa["ple_fold_wilson_lo"] if gpqa else None,
        "gpqa_ple_fold_wilson_hi": gpqa["ple_fold_wilson_hi"] if gpqa else None,
        "gpqa_n_used": gpqa["n_used"] if gpqa else None,
        "gpqa_ci_untightenable_at_dataset_ceiling": True,
        # Stage 2 — strict gate + margins + delta vs #557
        "strict_cilb_mmlu_passes_largern": strict_mmlu,
        "strict_cilb_gpqa_passes_largern": strict_gpqa,
        "strict_cilb_passes_largern": strict_both,
        "strict_cilb_binding_margin_largern": binding_margin,
        "mmlu_strict_cilb_margin_largern": mmlu["strict_cilb_margin_largern"],
        "mmlu_strict_cilb_margin_557": mmlu["strict_cilb_margin_557"],
        "mmlu_delta_margin_vs_557": mmlu["delta_margin_vs_557"],
        "gpqa_strict_cilb_margin_largern": gpqa["strict_cilb_margin_largern"] if gpqa else None,
        "gpqa_strict_cilb_margin_557": gpqa["strict_cilb_margin_557"] if gpqa else None,
        "gpqa_delta_margin_vs_557": gpqa["delta_margin_vs_557"] if gpqa else None,
        # Stage 3 — paired mechanism
        "point_gap_mmlu": mmlu["point_gap"],
        "point_gap_gpqa": gpqa["point_gap"] if gpqa else None,
        "point_gap_mmlu_is_significant": mmlu["point_gap_is_significant"],
        "point_gap_gpqa_is_significant": gpqa["point_gap_is_significant"] if gpqa else None,
        "point_gap_mmlu_mechanism": mmlu["point_gap_mechanism"],
        "point_gap_gpqa_mechanism": gpqa["point_gap_mechanism"] if gpqa else None,
        "mcnemar_mmlu_p": mmlu["paired"]["mcnemar_exact_p"],
        "mcnemar_gpqa_p": gpqa["paired"]["mcnemar_exact_p"] if gpqa else None,
        # integrity
        "mmlu_n_prompt_mismatch": mmlu["n_prompt_mismatch"],
        "gpqa_n_prompt_mismatch": gpqa["n_prompt_mismatch"] if gpqa else None,
        "self_det": self_det,
    }
    report = {
        "pr": 564, "vllm_build": BUILD, "mmlu": mmlu, "gpqa": gpqa, "marker": marker,
        "analysis_only": True, "no_hf_job": True, "official_tps": 0, "selfdet": sd,
    }
    (HERE / "aggregate.json").write_text(json.dumps(report, indent=2))

    def fmt(ax):
        return (
            f"  {ax['axis']:13s} n_used={ax['n_used']} (ple_fold n={ax['n_ple_fold']})\n"
            f"    base_fullhead = {ax['base_fullhead_acc']:.4f} "
            f"(Wilson {ax['base_fullhead_wilson_lo']:.3f}-{ax['base_fullhead_wilson_hi']:.3f}, "
            f"k={ax['base_fullhead_k']}/{ax['base_fullhead_n']})\n"
            f"    ple_fold      = {ax['ple_fold_acc']:.4f} "
            f"(Wilson {ax['ple_fold_wilson_lo']:.3f}-{ax['ple_fold_wilson_hi']:.3f}, "
            f"k={ax['ple_fold_k']}/{ax['ple_fold_n']})\n"
            f"    STRICT CI-lb: fullhead_lo={ax['base_fullhead_cilb_largern']:.4f} vs "
            f"gate(0.90x ple_fold)={ax['gate_0p90x_ple_fold']:.4f} -> "
            f"PASS={ax['strict_cilb_passes_largern']} margin={ax['strict_cilb_margin_largern']:+.4f} "
            f"(#557 margin={ax['strict_cilb_margin_557']:+.4f}, delta={ax['delta_margin_vs_557']:+.4f})\n"
            f"    PAIRED: gap(ple_fold-fullhead)={ax['point_gap']:+.4f} "
            f"discordant b={ax['paired']['b_fullhead_only_correct']} c={ax['paired']['c_ple_fold_only_correct']} "
            f"McNemar p={ax['paired']['mcnemar_exact_p']:.3f} sig={ax['point_gap_is_significant']} "
            f"-> {ax['point_gap_mechanism']}  (n_prompt_mismatch={ax['n_prompt_mismatch']})"
        )

    print(f"\n==== PR #564 base_fullhead STRICT CI-lb @ larger n (build {BUILD}) ====")
    print(fmt(mmlu))
    if gpqa:
        print(fmt(gpqa))
        print(f"  NOTE: GPQA-Diamond is the FULL {gpqa['n_used']}-item set -> Wilson width fixed at "
              f"the dataset ceiling (cannot be tightened by larger n).")
    print(f"  -- VERDICT --")
    print(f"  strict_cilb_mmlu_passes_largern = {strict_mmlu}")
    print(f"  strict_cilb_gpqa_passes_largern = {strict_gpqa}")
    print(f"  strict_cilb_passes_largern(BOTH)= {strict_both}  binding_margin={binding_margin:+.4f}")
    print(f"  self_det = {self_det}")
    print("MARKER:", json.dumps(marker))

    senpai_result = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [_runid()],
        "primary_metric": {"name": "strict_cilb_passes_largern", "value": bool(strict_both)},
        "test_metric": {"name": "strict_cilb_binding_margin_largern", "value": binding_margin},
    }
    print("SENPAI-RESULT:", json.dumps(senpai_result))

    if not a.no_wandb:
        _log_wandb(report, marker)
    return 0


def _log_wandb(report, marker):
    rid = _runid()
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed: {exc!r}; JSON saved only")
        return
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
        print("[wandb] no key; JSON only")
        return
    try:
        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
            entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
            id=rid or None, resume="allow" if rid else None,
            name="stark/base-fullhead-strict-cilb-largern",
            group="base-fullhead-strict-cilb-largern", job_type="analysis",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] resume init failed: {exc!r}")
        return
    for k, v in marker.items():
        if isinstance(v, (int, float, bool, str)):
            run.summary[k] = v
    for tag, ax in (("mmlu", report["mmlu"]), ("gpqa", report["gpqa"])):
        if not ax:
            continue
        for kk in ("base_fullhead_acc", "base_fullhead_wilson_lo", "base_fullhead_wilson_hi",
                   "ple_fold_acc", "ple_fold_wilson_lo", "ple_fold_wilson_hi",
                   "gate_0p90x_ple_fold", "strict_cilb_passes_largern", "strict_cilb_margin_largern",
                   "strict_cilb_margin_557", "delta_margin_vs_557", "point_gap",
                   "point_gap_is_significant", "two_prop_z", "n_used", "n_prompt_mismatch"):
            v = ax.get(kk)
            if isinstance(v, (int, float, bool)):
                run.summary[f"{tag}/{kk}"] = v
        run.summary[f"{tag}/mcnemar_exact_p"] = ax["paired"]["mcnemar_exact_p"]
        run.summary[f"{tag}/point_gap_mechanism"] = ax["point_gap_mechanism"]
    try:
        run.finish()
    except Exception:
        pass
    print(f"[wandb] logged to run {rid}")


if __name__ == "__main__":
    raise SystemExit(main())
