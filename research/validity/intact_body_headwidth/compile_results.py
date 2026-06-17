#!/usr/bin/env python3
"""Compile the intact-body head-width sweep into the PR #547 terminal marker (+ W&B).

Reads whatever result JSONs are present in results/ and assembles:
  quality_by_head_width, tps_by_head_width, and the verdict booleans
  (head_prune_innocent_on_intact_body, min_quality_safe_head_width,
   fast_quality_safe_ship_exists, collapse_is_100pct_body).

Robust to missing cells (prints null) so it can be run incrementally. analysis_only;
official_tps=0. Pass --wandb to also log a run to the sweep group.

File conventions (in results/):
  head{262k,12k,32k}_mmlu_pro.json   inspect run_eval output (accuracy)
  head{262k,12k,32k}_gpqa.json       inspect run_eval output (accuracy)
  gsm8k_head{262k,12k,32k}_mt0.json  gsm8k_sampled as-served (accuracy, empty_rate)
  gsm8k_head{262k,12k,32k}_mt8.json  gsm8k_sampled guarded
  tps_head{262k,12k,32k}.json        tps_probe single-stream
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

WIDTHS = ["262k", "12k", "32k"]
WIDTH_K = {"262k": 262144, "12k": 12288, "32k": 32768}

# Morgan #524 gate floors
MMLU_FLOOR = 0.601
GPQA_FLOOR = 0.400
GSM8K_REL = 0.90  # >= 90% of base (262k) GSM8K
TPS_FAST = 300.0  # "beats ~300 TPS"


def _load(name: str):
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _acc(d):
    return None if d is None else d.get("accuracy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    q: dict = {}
    tps: dict = {}
    for w in WIDTHS:
        mm = _load(f"head{w}_mmlu_pro.json")
        gp = _load(f"head{w}_gpqa.json")
        g0 = _load(f"gsm8k_head{w}_mt0.json")
        g8 = _load(f"gsm8k_head{w}_mt8.json")
        tp = _load(f"tps_head{w}.json")
        q[w] = {
            "mmlu_pro": _acc(mm),
            "gpqa_d": _acc(gp),
            "gsm8k_asserved": _acc(g0),
            "gsm8k_guarded": _acc(g8),
            "empty_rate_asserved": (g0 or {}).get("empty_rate"),
            "empty_rate_guarded": (g8 or {}).get("empty_rate"),
        }
        if tp is not None:
            tps[w] = {
                "tps_median_per_req": tp.get("tps_median_per_req"),
                "tps_end_to_end": tp.get("tps_end_to_end"),
            }
        else:
            tps[w] = {"tps_median_per_req": None, "tps_end_to_end": None}

    base_gsm = q["262k"]["gsm8k_guarded"]

    def clears_gate(w: str) -> bool | None:
        c = q[w]
        if c["mmlu_pro"] is None or c["gpqa_d"] is None:
            return None
        ok = c["mmlu_pro"] >= MMLU_FLOOR and c["gpqa_d"] >= GPQA_FLOOR
        # GSM8K relative check only if both base and cell available
        if c["gsm8k_guarded"] is not None and base_gsm:
            ok = ok and (c["gsm8k_guarded"] >= GSM8K_REL * base_gsm)
        return ok

    def is_fast(w: str) -> bool | None:
        t = tps[w]["tps_median_per_req"]
        return None if t is None else (t > TPS_FAST)

    # head_prune_innocent_on_intact_body: 12k holds MMLU>=floor AND GPQA>=floor
    c12 = q["12k"]
    innocent = (
        None if (c12["mmlu_pro"] is None or c12["gpqa_d"] is None)
        else (c12["mmlu_pro"] >= MMLU_FLOOR and c12["gpqa_d"] >= GPQA_FLOOR)
    )

    # collapse_is_100pct_body: 12k MMLU ~ 262k MMLU (head adds no damage on intact body)
    base_mm = q["262k"]["mmlu_pro"]
    collapse_100_body = (
        None if (c12["mmlu_pro"] is None or base_mm is None)
        else (abs(c12["mmlu_pro"] - base_mm) <= 0.02)  # within ~noise
    )

    # min_quality_safe_head_width: smallest pruned width clearing the MC floors
    safe_widths = []
    for w in ["12k", "32k"]:
        g = clears_gate(w)
        if g:
            safe_widths.append(w)
    # order by K ascending
    safe_sorted = sorted(safe_widths, key=lambda w: WIDTH_K[w])
    min_safe = safe_sorted[0] if safe_sorted else None
    tps_at_min_safe = tps[min_safe]["tps_median_per_req"] if min_safe else None

    # fast_quality_safe_ship_exists: ANY pruned width clears gate AND is fast
    fast_safe = False
    fast_safe_known = True
    for w in ["12k", "32k"]:
        g = clears_gate(w)
        f = is_fast(w)
        if g and f:
            fast_safe = True
        if g is None or f is None:
            fast_safe_known = False
    fast_quality_safe_ship_exists = fast_safe if (fast_safe or fast_safe_known) else None

    # head-prune marginal damage on the intact body (decomposition)
    head_prune_cost_mmlu = (
        None if (base_mm is None or c12["mmlu_pro"] is None)
        else round(base_mm - c12["mmlu_pro"], 4)
    )

    marker = {
        "pr": 547, "analysis_only": True, "official_tps": 0,
        "engine": "vllm-0.22.1rc1.dev307 (v0.22.0 craters this int4 model; see note)",
        "control_substrate": "intact base-int4 body (42L, no drop/no bake), full BF16 tied head",
        "quality_by_head_width": q,
        "tps_by_head_width": tps,
        "gate_floors": {"mmlu_pro": MMLU_FLOOR, "gpqa_d": GPQA_FLOOR,
                        "gsm8k_rel_to_base": GSM8K_REL, "fast_tps": TPS_FAST},
        "head_prune_innocent_on_intact_body": innocent,
        "head_prune_cost_mmlu_pro_12k": head_prune_cost_mmlu,
        "collapse_is_100pct_body": collapse_100_body,
        "min_quality_safe_head_width": min_safe,
        "tps_at_min_safe_width": tps_at_min_safe,
        "fast_quality_safe_ship_exists": fast_quality_safe_ship_exists,
        "tps_stack": (
            "ISOLATED head-prune on vanilla vLLM v0221 + compute_logits slice(pruned)/off(262k), "
            "single-stream MAX_NUM_SEQS=1. This is NOT the fa2sw_strict_surgical357 frontier stack "
            "(fa2sw+onegraph+16k-keyed MTP drafter), so the ~300 TPS bar (frontier-contextual: fern "
            "base_fullhead=253.78, osoi5=353.73) is not directly comparable. fast_quality_safe_ship_exists "
            "here is computed on isolated TPS and is therefore a LOWER-BOUND verdict on speed; the isolated "
            "lever is +30% full->32k (96.7->125.6), which extrapolates 253.78*1.30~=330 on the intact-body "
            "frontier (UNVERIFIED). Frontier-stack TPS at 32k is the follow-up that settles the speed half."
        ),
        "quality_stack": (
            "vanilla vLLM v0.22.1rc1.dev307 + compute_logits mask(pruned)/off(262k), greedy; reproduces "
            "ubel #538 control (0.668/0.444 -> measured 0.676/0.4697), so head-width QUALITY is "
            "apples-to-apples with the cited control."
        ),
        "answer_tokens_in_12k_keepset": True,  # all MC letters A-J and digits 0-9 retained; 12k still -0.126 MMLU
        "collapse_decomposition": {
            "control_intact_fullhead_mmlu": base_mm,
            "intact_12k_mmlu": q["12k"]["mmlu_pro"],
            "osoi5_ship_mmlu": 0.274,
            "head_share_of_total_drop": (
                None if (base_mm is None or q["12k"]["mmlu_pro"] is None)
                else round((base_mm - q["12k"]["mmlu_pro"]) / (base_mm - 0.274), 3)
            ),
            "body_share_of_total_drop": (
                None if (base_mm is None or q["12k"]["mmlu_pro"] is None)
                else round((q["12k"]["mmlu_pro"] - 0.274) / (base_mm - 0.274), 3)
            ),
        },
    }

    (RES / "headwidth_sweep_marker.json").write_text(json.dumps(marker, indent=2))
    print(json.dumps(marker, indent=2))

    if args.wandb:
        import os
        import wandb
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=os.environ.get("WANDB_PROJECT"),
            group="base-fullhead-headwidth-sweep",
            name="kanna/intact-body-headwidth-sweep",
            job_type="analysis",
            config={"pr": 547, "analysis_only": True, "official_tps": 0,
                    "engine": "vllm-0.22.1rc1.dev307",
                    "widths": WIDTH_K, "gate_floors": marker["gate_floors"]},
        )
        flat = {}
        for w in WIDTHS:
            for k, v in q[w].items():
                if v is not None:
                    flat[f"quality/{w}/{k}"] = v
            for k, v in tps[w].items():
                if v is not None:
                    flat[f"tps/{w}/{k}"] = v
        for k in ("head_prune_innocent_on_intact_body", "collapse_is_100pct_body",
                  "fast_quality_safe_ship_exists"):
            if marker[k] is not None:
                flat[f"verdict/{k}"] = int(bool(marker[k]))
        if head_prune_cost_mmlu is not None:
            flat["verdict/head_prune_cost_mmlu_pro_12k"] = head_prune_cost_mmlu
        if min_safe is not None:
            flat["verdict/min_quality_safe_head_width_K"] = WIDTH_K[min_safe]
        wandb.log(flat)
        wandb.summary.update(flat)
        print(f"[wandb] logged run {run.id} ({run.url})")
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
