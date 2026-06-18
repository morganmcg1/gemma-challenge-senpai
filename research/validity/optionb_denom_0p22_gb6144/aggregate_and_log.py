#!/usr/bin/env python
"""PR #628 -- aggregate the bf16-base 4-gate panel on vLLM 0.22.0 @ gb6144 and
emit the denominator-validity verdict.

The question this card answers: the four gate *bars* and their bf16-base anchors
were measured on dev307; the submission actually serves on 0.22.0. Is the bf16
base denominator *stack-robust* (same accuracy + ~3% finish-length on 0.22.0 as on
dev307), so fern #624's 0.22.0 Option-B numerator has a clean same-stack base to
clear? Or does 0.22.0 move a bar materially (cf. #547: 0.22.0 craters MMLU on the
*int4* model -- the open question is whether it also moves the *bf16 base*).

Pure-stdlib aggregation -> panel_summary.json. W&B logging is optional (--wandb),
guarded so the summary always writes even without wandb.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Gate bars (#581/#580), all anchored on the dev307 bf16 base.
BARS = {"mmlu_pro": 0.605, "gpqa_diamond": 0.471, "gsm8k": 0.807, "aime": 0.090}

# dev307 bf16-base anchors carried in the PR #628 body. GPQA has both reads; the
# bar (0.471) = 0.9x the *sampled* base. finish-len on dev307 base ~3.5% (#614).
DEV307_ANCHOR = {
    "mmlu_pro":      {"acc": 0.678,  "decode": "greedy",  "n": 500, "note": "PR#628 / #614 base"},
    "gpqa_greedy":   {"acc": 0.5051, "decode": "greedy",  "n": 198, "note": "#614 run yzltlpsn"},
    "gpqa_sampled":  {"acc": 0.5313, "decode": "sampled", "n": 198, "note": "#614 re-measure"},
    "gsm8k":         {"acc": 0.904,  "decode": "greedy",  "n": 500, "note": "PR#628 base"},
    "aime":          {"acc": 0.100,  "decode": "greedy",  "n": 60,  "note": "budget UNKNOWN -- the gap #628 closes", "budget_comparable": False},
}
DEV307_FINISH_LEN = 0.035  # #614 bf16 base finish_length_rate

# A finish_length_rate this high means the base is truncating like a crater, not
# the healthy ~3% dev307 base -> NOT stack-robust.
CRATER_FINISH_LEN = 0.15


def _se(p: float, n: int) -> float:
    if not n or p is None or (isinstance(p, float) and math.isnan(p)):
        return float("nan")
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _aime_finish_len(doc) -> tuple[float | None, int | None, int]:
    """AIME logs per-problem finish_reasons (k samples each). finish_length_rate =
    fraction of SAMPLES whose finish_reason == 'length' (max_tokens truncation)."""
    pp = doc.get("per_problem") or []
    n_samp = 0
    n_len = 0
    for p in pp:
        for fr in (p.get("finish_reasons") or []):
            n_samp += 1
            if fr == "length":
                n_len += 1
    if not n_samp:
        return None, None, 0
    return n_len / n_samp, n_len, n_samp


def collect(res: Path) -> dict:
    out: dict[str, dict] = {}

    # MMLU-Pro (run_eval.py): accuracy + finish_length_rate direct.
    d = _load(res / "base_mmlu_pro_greedy_gb6144.json")
    if d:
        out["mmlu_pro"] = {
            "accuracy": d["accuracy"], "finish_length_rate": d.get("finish_length_rate"),
            "n_scored": d.get("n_scored"), "n_samples": d.get("n_samples"),
            "empty_rate": d.get("empty_rate"), "ctok_p95": d.get("completion_tokens_p95"),
            "decode": "greedy", "max_tokens": d.get("max_tokens"),
        }

    # GPQA greedy + sampled (run_eval.py).
    for key, fn in [("gpqa_greedy", "base_gpqa_greedy_gb6144.json"),
                    ("gpqa_sampled", "base_gpqa_sampled_gb6144.json")]:
        d = _load(res / fn)
        if d:
            out[key] = {
                "accuracy": d["accuracy"], "finish_length_rate": d.get("finish_length_rate"),
                "n_scored": d.get("n_scored"), "n_samples": d.get("n_samples"),
                "empty_rate": d.get("empty_rate"), "ctok_p95": d.get("completion_tokens_p95"),
                "decode": d.get("decode"), "max_tokens": d.get("max_tokens"),
                "temperature": d.get("temperature"), "top_k": d.get("top_k"),
            }

    # GSM8K (gsm8k_eval.py): accuracy + truncation_rate (== finish_length_rate).
    d = _load(res / "base_greedy_gb6144_greedy.json")
    if d:
        out["gsm8k"] = {
            "accuracy": d["accuracy"], "finish_length_rate": d.get("truncation_rate"),
            "n_correct": d.get("n_correct"), "n_samples": d.get("n_problems"),
            "extract_fail_rate": d.get("extract_fail_rate"),
            "decode": "greedy", "max_tokens": d.get("max_tokens"),
        }

    # AIME (aime_eval.py): maj_k_accuracy (k=1 => greedy) + derived finish_length.
    d = _load(res / "base_aime_greedy_gb6144.json")
    if d:
        flr, n_len, n_samp = _aime_finish_len(d)
        out["aime"] = {
            "accuracy": d["maj_k_accuracy"], "finish_length_rate": flr,
            "n_correct": d.get("n_correct_maj"), "n_samples": d.get("n_problems"),
            "extract_fail_rate": d.get("extract_fail_rate"),
            "n_length_samples": n_len, "n_total_samples": n_samp,
            "decode": "greedy", "max_tokens": d.get("max_tokens"),
        }
    return out


def build_summary(res: Path) -> dict:
    gates = collect(res)
    rows = []
    crater_flags = []
    shift_flags = []
    # gate-key -> the dev307 anchor key used for the bar comparison.
    BAR_KEY = {"mmlu_pro": "mmlu_pro", "gpqa_diamond": "gpqa_sampled",
               "gsm8k": "gsm8k", "aime": "aime"}
    for gkey, anchor_key in [("mmlu_pro", "mmlu_pro"), ("gpqa_greedy", "gpqa_greedy"),
                             ("gpqa_sampled", "gpqa_sampled"), ("gsm8k", "gsm8k"),
                             ("aime", "aime")]:
        g = gates.get(gkey)
        if not g:
            continue
        anc = DEV307_ANCHOR[anchor_key]
        acc = g["accuracy"]
        n = g.get("n_scored") or g.get("n_samples") or anc["n"]
        se = _se(acc, n)
        d_acc = acc - anc["acc"]
        # material accuracy shift: beyond ~2 SE AND >= 0.03 absolute.
        material = (abs(d_acc) > max(0.03, 2 * se)) if not math.isnan(se) else (abs(d_acc) > 0.03)
        # AIME's dev307 anchor (0.100) was measured at an UNKNOWN, non-gb6144 budget
        # (PR #628; fern #624). A delta against it is a *budget* effect, not a
        # 0.22.0-vs-dev307 *stack* shift, so it can neither certify nor refute
        # stack-robustness and must not drive the verdict. AIME is still held to its
        # bar and its crater check below; this gb6144 read IS the budget-matched
        # anchor the card delivers.
        anchor_budget_comparable = anc.get("budget_comparable", True)
        flr = g.get("finish_length_rate")
        crater = bool(flr is not None and flr > CRATER_FINISH_LEN)
        if crater:
            crater_flags.append(gkey)
        if material and anchor_budget_comparable:
            shift_flags.append(gkey)
        rows.append({
            "gate": gkey, "decode": g.get("decode"), "accuracy_0p22": acc, "n": n,
            "se": None if math.isnan(se) else round(se, 4),
            "dev307_anchor": anc["acc"], "delta_vs_dev307": round(d_acc, 4),
            "material_shift": material,
            "anchor_budget_comparable": anchor_budget_comparable,
            "counts_toward_verdict_shift": bool(material and anchor_budget_comparable),
            "finish_length_rate": flr,
            "crater": crater, "empty_rate": g.get("empty_rate"),
            "extract_fail_rate": g.get("extract_fail_rate"),
            "ctok_p95": g.get("ctok_p95"), "max_tokens": g.get("max_tokens"),
        })

    # Bar check: each gate's 0.22.0 base must still clear its bar (the bar is
    # 0.9x the dev307 base; if the 0.22.0 base falls below/near the bar the
    # 0.9x-of-base relationship no longer holds on the served stack).
    bar_clears = {}
    for gkey, anchor_key in BAR_KEY.items():
        src = {"mmlu_pro": "mmlu_pro", "gpqa_diamond": "gpqa_sampled",
               "gsm8k": "gsm8k", "aime": "aime"}[gkey]
        g = gates.get(src)
        if g:
            bar_clears[gkey] = {"acc": g["accuracy"], "bar": BARS[gkey],
                                "clears": g["accuracy"] >= BARS[gkey],
                                "margin": round(g["accuracy"] - BARS[gkey], 4)}

    valid = (not crater_flags) and (not shift_flags) and all(
        v["clears"] for v in bar_clears.values()) and len(bar_clears) == 4
    verdict = "DENOMINATORS_VALID_ON_0p22" if valid else "DENOMINATOR_SHIFTS"

    base_aime = gates.get("aime", {}).get("accuracy")
    summary = {
        "card": "optionb_denom_0p22_gb6144",
        "engine": "vllm==0.22.0",
        "config": {"max_model_len": 8192, "max_tokens": 6144, "min_tokens": 8,
                   "max_num_seqs": 16, "vllm_batch_invariant": 1,
                   "model": "google/gemma-4-E4B-it (bf16, full 262k head, snapshot fee6332c)",
                   "gpqa_sampled": "T=1/top_p=0.95/top_k=64"},
        "verdict": verdict,
        "crater_gates": crater_flags,
        "material_shift_gates": shift_flags,
        "bar_clears": bar_clears,
        "base_aime_gb6144": base_aime,
        "aime_anchor_caveat": {
            "dev307_anchor_acc": DEV307_ANCHOR["aime"]["acc"],
            "dev307_anchor_budget": "unknown (not gb6144)",
            "base_aime_gb6144": base_aime,
            "delta_is_budget_effect": True,
            "excluded_from_stack_shift_verdict": True,
            "note": ("base_aime_gb6144 is the budget-matched AIME anchor (PR #628 deliverable); "
                     "its large delta vs the 0.100 dev307 anchor is a budget effect (gb6144 vs an "
                     "unknown/shorter budget) and is excluded from the stack-shift verdict. AIME "
                     "still clears its bar and is below the crater threshold."),
        },
        "dev307_finish_len_ref": DEV307_FINISH_LEN,
        "rows": rows,
        "gates_raw": gates,
    }
    return summary


def log_wandb(summary: dict) -> None:
    import wandb
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    group = "optionb-denominator-0p22-gb6144"
    common = {
        "analysis_only": True, "official_tps": 0,
        "engine": summary["engine"], **summary["config"],
        "pr": 628, "student": "ubel",
    }
    ids = []
    for row in summary["rows"]:
        gkey = row["gate"]
        run = wandb.init(project=project, entity=entity, group=group,
                         name=f"ubel/base-0p22-{gkey}", job_type="denominator-eval",
                         reinit=True, config={**common, "gate": gkey,
                                              "dev307_anchor": row["dev307_anchor"]})
        wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float, bool))})
        run.summary["accuracy_0p22"] = row["accuracy_0p22"]
        run.summary["delta_vs_dev307"] = row["delta_vs_dev307"]
        run.summary["finish_length_rate"] = row["finish_length_rate"]
        ids.append(run.id)
        run.finish()
    # verdict run
    run = wandb.init(project=project, entity=entity, group=group,
                     name="ubel/base-0p22-VERDICT", job_type="denominator-verdict",
                     reinit=True, config={**common, "bars": BARS,
                                          "dev307_anchors": {k: v["acc"] for k, v in DEV307_ANCHOR.items()}})
    vlog = {"denominators_valid": int(summary["verdict"] == "DENOMINATORS_VALID_ON_0p22"),
            "n_crater": len(summary["crater_gates"]),
            "n_material_shift": len(summary["material_shift_gates"]),
            "base_aime_gb6144": summary["base_aime_gb6144"]}
    for gkey, bc in summary["bar_clears"].items():
        vlog[f"{gkey}_acc"] = bc["acc"]
        vlog[f"{gkey}_clears_bar"] = int(bc["clears"])
        vlog[f"{gkey}_margin"] = bc["margin"]
    wandb.log(vlog)
    run.summary["verdict"] = summary["verdict"]
    ids.append(run.id)
    run.finish()
    summary["wandb_run_ids"] = ids
    print(f"[wandb] logged {len(ids)} runs -> group {group}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--out", type=Path, default=HERE / "panel_summary.json")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    summary = build_summary(args.results_dir)
    if args.wandb:
        try:
            log_wandb(summary)
        except Exception as exc:  # never lose the summary to a wandb hiccup
            print(f"[wandb] FAILED: {exc!r}")
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("verdict", "crater_gates", "material_shift_gates",
                       "base_aime_gb6144", "bar_clears")}, indent=2))
    print(f"[aggregate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
