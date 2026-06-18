#!/usr/bin/env python
"""PR #638 -- Option-B denominator LEG 3: aggregate the int4-AR (live-rung body)
5-gate panel on vLLM 0.22.0 @ gb6144 and emit the THREE-WAY denominator table +
gate verdict.

Three legs of the same gate, same engine/budget/eval code:
  bf16 base (ubel #628)        -- the denominator (literal-read base)
  int4-AR (THIS, live rung)    -- the body the submission actually ships, AR no-spec
  int4+spec Option-B (fern #629, run 2jhhk0u3) -- the numerator

Binding axis = GPQA-Diamond *sampled* 10-seed n=1980 (dseed 12345, sseeds 0..9,
T=1/top_p=0.95/top_k=64). Gate bar = 0.9x the bf16 base.

Verdict:
  int4-AR sits under 0.9x bf16 on GPQA-sampled AND AIME  -> INT4_AR_LIVE_RUNG_SHARES_DEFICIT
  int4-AR clears 0.9x bf16 but int4+spec #629 does not    -> OPTIONB_HAS_SPEC_SPECIFIC_DEFICIT

Pure-stdlib aggregation -> panel_summary_int4ar.json. W&B logging optional (--wandb).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# bf16 base denominator (ubel #628, banked). GPQA-sampled here is the SINGLE-seed
# n=198 0.5404 the PR #638 body fixes as the denominator; the bar is 0.9x it.
BF16_628 = {
    "mmlu_pro": 0.7180, "gpqa_greedy": 0.4899, "gpqa_sampled": 0.5404,
    "gsm8k": 0.9280, "aime": 0.4667,
}
# int4+spec Option-B numerator (fern #629, run 2jhhk0u3). gpqa_sampled = 10-seed
# n=1980 raw 0.46515; gpqa_greedy = lawine #627 0.4444; aime = greedy 0.36667.
INT4SPEC_629 = {
    "mmlu_pro": 0.664, "gpqa_greedy": 0.4444, "gpqa_sampled": 0.46515,
    "gsm8k": 0.926, "aime": 0.36667,
}
GATE_FRAC = 0.9  # gate bar = 0.9 x bf16 base


def _wilson(n_correct: int, n: int, z: float = 1.96):
    if not n:
        return float("nan"), float("nan")
    p = n_correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _aime_finish_len(doc):
    pp = doc.get("per_problem") or []
    n_samp = n_len = 0
    for p in pp:
        for fr in (p.get("finish_reasons") or []):
            n_samp += 1
            if fr == "length":
                n_len += 1
    return (n_len / n_samp if n_samp else None), n_len, n_samp


def aggregate_sampled(res: Path) -> dict | None:
    """Concatenate the per-seed GPQA-sampled run_eval.py JSONs into one n=1980 read."""
    files = sorted(glob.glob(str(res / "int4ar_gpqa_sampled_s*.json")))
    if not files:
        return None
    per_seed = []
    all_correct = all_scored = all_error = all_trunc = all_empty = total = 0
    for fn in files:
        d = json.loads(Path(fn).read_text())
        ps = d.get("per_sample") or []
        sc = sum(1 for r in ps if r.get("value") in ("C", "I"))
        cor = sum(1 for r in ps if r.get("correct"))
        err = sum(1 for r in ps if r.get("error"))
        trunc = sum(1 for r in ps if r.get("truncated"))
        emp = sum(1 for r in ps if r.get("empty"))
        per_seed.append({
            "sampling_seed": d.get("sampling_seed"),
            "n_scored": sc, "n_correct": cor, "n_error": err,
            "accuracy": (cor / sc) if sc else float("nan"),
            "finish_length_rate": (trunc / len(ps)) if ps else float("nan"),
        })
        all_correct += cor; all_scored += sc; all_error += err
        all_trunc += trunc; all_empty += emp; total += len(ps)
    acc = all_correct / all_scored if all_scored else float("nan")
    # deconf: drop request-errored samples (scored as incorrect under score_on_error)
    n_deconf = all_scored - all_error
    acc_deconf = all_correct / n_deconf if n_deconf else float("nan")
    lo, hi = _wilson(all_correct, all_scored)
    return {
        "decode": "sampling", "n_seeds": len(files), "n_scored": all_scored,
        "n_correct": all_correct, "n_error": all_error, "n_samples": total,
        "accuracy": acc, "accuracy_deconf": acc_deconf, "n_scored_deconf": n_deconf,
        "ci95_lo_wilson": lo, "ci95_hi_wilson": hi,
        "finish_length_rate": (all_trunc / total) if total else float("nan"),
        "empty_rate": (all_empty / total) if total else float("nan"),
        "max_tokens": 6144, "temperature": 1.0, "top_k": 64,
        "per_seed": per_seed,
    }


def collect(res: Path) -> dict:
    out: dict[str, dict] = {}

    samp = aggregate_sampled(res)
    if samp:
        out["gpqa_sampled"] = samp

    d = _load(res / "int4ar_gpqa_greedy_gb6144.json")
    if d:
        out["gpqa_greedy"] = {
            "accuracy": d["accuracy"], "finish_length_rate": d.get("finish_length_rate"),
            "n_scored": d.get("n_scored"), "n_samples": d.get("n_samples"),
            "empty_rate": d.get("empty_rate"), "ctok_p95": d.get("completion_tokens_p95"),
            "decode": "greedy", "max_tokens": d.get("max_tokens"),
        }

    d = _load(res / "int4ar_mmlu_pro_greedy_gb6144.json")
    if d:
        out["mmlu_pro"] = {
            "accuracy": d["accuracy"], "finish_length_rate": d.get("finish_length_rate"),
            "n_scored": d.get("n_scored"), "n_samples": d.get("n_samples"),
            "empty_rate": d.get("empty_rate"), "ctok_p95": d.get("completion_tokens_p95"),
            "decode": "greedy", "max_tokens": d.get("max_tokens"),
        }

    # GSM8K writes <label>_<regime>.json
    gs = sorted(glob.glob(str(res / "int4ar_greedy_gb6144*_greedy.json"))) or \
         sorted(glob.glob(str(res / "*greedy*_greedy.json")))
    d = _load(Path(gs[0])) if gs else None
    if d:
        out["gsm8k"] = {
            "accuracy": d["accuracy"], "finish_length_rate": d.get("truncation_rate"),
            "n_correct": d.get("n_correct"), "n_samples": d.get("n_problems"),
            "extract_fail_rate": d.get("extract_fail_rate"),
            "decode": "greedy", "max_tokens": d.get("max_tokens"),
        }

    d = _load(res / "int4ar_aime_greedy_gb6144.json")
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


# gate-key -> the bf16/int4spec dict key used for the bar comparison.
GATE_ROWS = [
    ("mmlu_pro", "mmlu_pro", "MMLU-Pro"),
    ("gsm8k", "gsm8k", "GSM8K"),
    ("gpqa_sampled", "gpqa_sampled", "GPQA-D sampled (BINDING)"),
    ("gpqa_greedy", "gpqa_greedy", "GPQA-D greedy"),
    ("aime", "aime", "AIME"),
]


def build_summary(res: Path) -> dict:
    gates = collect(res)
    rows = []
    for gkey, refkey, label in GATE_ROWS:
        g = gates.get(gkey)
        bf16 = BF16_628[refkey]
        spec = INT4SPEC_629[refkey]
        bar = round(GATE_FRAC * bf16, 4)
        acc = g["accuracy"] if g else None
        ratio = round(acc / bf16, 4) if acc is not None else None
        clears = bool(acc is not None and acc >= bar)
        rows.append({
            "gate": gkey, "label": label, "decode": g.get("decode") if g else None,
            "bf16_628": bf16, "int4ar": acc, "int4spec_629": spec,
            "bar_0p9xbf16": bar, "int4ar_over_bf16": ratio,
            "int4ar_clears_0p9xbf16": clears,
            "int4spec_clears_0p9xbf16": bool(spec >= bar),
            "finish_length_rate": g.get("finish_length_rate") if g else None,
            "n_scored": g.get("n_scored") if g else None,
            "n_samples": g.get("n_samples") if g else None,
            "empty_rate": g.get("empty_rate") if g else None,
        })

    samp = gates.get("gpqa_sampled", {})
    aime = gates.get("aime", {})
    samp_acc = samp.get("accuracy")
    aime_acc = aime.get("accuracy")
    samp_bar = GATE_FRAC * BF16_628["gpqa_sampled"]
    aime_bar = GATE_FRAC * BF16_628["aime"]
    samp_under = bool(samp_acc is not None and samp_acc < samp_bar)
    aime_under = bool(aime_acc is not None and aime_acc < aime_bar)
    spec_samp_under = INT4SPEC_629["gpqa_sampled"] < samp_bar

    if samp_acc is None or aime_acc is None:
        verdict = "INCOMPLETE"
    elif samp_under and aime_under:
        verdict = "INT4_AR_LIVE_RUNG_SHARES_DEFICIT"
    elif (not samp_under) and spec_samp_under:
        verdict = "OPTIONB_HAS_SPEC_SPECIFIC_DEFICIT"
    else:
        verdict = "MIXED_SEE_ROWS"

    return {
        "card": "int4ar_livrung_denominator_0p22 (PR #638)",
        "engine": "vllm==0.22.0",
        "config": {"max_model_len": 8192, "max_tokens": 6144, "min_tokens": 8,
                   "max_num_seqs": 16, "vllm_batch_invariant": 1,
                   "model": "int4_g128_lmhead (W4A16 g128 + untied int4 g128 lm_head, AR no-spec, PPL 2.0197)",
                   "gpqa_sampled": "10-seed n=1980 dseed12345 sseeds0..9 T=1/top_p=0.95/top_k=64"},
        "verdict": verdict,
        "binding": {
            "gpqa_sampled_int4ar": samp_acc, "gpqa_sampled_bar_0p9xbf16": round(samp_bar, 4),
            "gpqa_sampled_under_bar": samp_under,
            "aime_int4ar": aime_acc, "aime_bar_0p9xbf16": round(aime_bar, 4),
            "aime_under_bar": aime_under,
            "int4spec_629_gpqa_sampled_under_bar": spec_samp_under,
        },
        "rows": rows,
        "gpqa_sampled_detail": samp,
        "gates_raw": gates,
    }


def log_wandb(summary: dict) -> None:
    import wandb
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    group = "int4ar-livrung-denominator-0p22"
    common = {"analysis_only": True, "official_tps": 0, "engine": summary["engine"],
              **summary["config"], "pr": 638, "student": "ubel"}
    ids = []
    for row in summary["rows"]:
        if row["int4ar"] is None:
            continue
        run = wandb.init(project=project, entity=entity, group=group,
                         name=f"ubel/int4ar-{row['gate']}", job_type="int4ar-denominator-eval",
                         reinit=True, config={**common, "gate": row["gate"],
                                              "bf16_628": row["bf16_628"],
                                              "int4spec_629": row["int4spec_629"]})
        wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float, bool))})
        run.summary["int4ar"] = row["int4ar"]
        run.summary["int4ar_over_bf16"] = row["int4ar_over_bf16"]
        run.summary["int4ar_clears_0p9xbf16"] = int(row["int4ar_clears_0p9xbf16"])
        run.summary["finish_length_rate"] = row["finish_length_rate"]
        ids.append(run.id)
        run.finish()
    # verdict run
    run = wandb.init(project=project, entity=entity, group=group,
                     name="ubel/int4ar-VERDICT", job_type="int4ar-denominator-verdict",
                     reinit=True, config={**common, "bf16_628": BF16_628,
                                          "int4spec_629": INT4SPEC_629})
    b = summary["binding"]
    wandb.log({k: (int(v) if isinstance(v, bool) else v)
               for k, v in b.items() if isinstance(v, (int, float, bool))})
    run.summary["verdict"] = summary["verdict"]
    ids.append(run.id)
    run.finish()
    summary["wandb_run_ids"] = ids
    print(f"[wandb] logged {len(ids)} runs -> group {group}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=HERE / "results_int4ar")
    ap.add_argument("--out", type=Path, default=HERE / "panel_summary_int4ar.json")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    summary = build_summary(args.results_dir)
    if args.wandb:
        try:
            log_wandb(summary)
        except Exception as exc:
            print(f"[wandb] FAILED: {exc!r}")
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"verdict": summary["verdict"], "binding": summary["binding"],
                      "rows": [{k: r[k] for k in ("gate", "bf16_628", "int4ar",
                                "int4spec_629", "int4ar_over_bf16",
                                "int4ar_clears_0p9xbf16", "finish_length_rate")}
                               for r in summary["rows"]]}, indent=2))
    print(f"[aggregate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
