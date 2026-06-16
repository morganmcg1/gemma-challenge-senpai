#!/usr/bin/env python3
"""Aggregate the keepset-width quality sweep (PR #527).

For each lm_head keepset width K, the 37L-int4 osoi5 substrate was served through
the REAL pck04 ship path (ParallelLMHead(K) + scatter-to-262144) and evaluated on
MMLU-Pro n=500 (seed 12345) + GPQA-Diamond 198, greedy, byte-identical prompts to
the 42L base (reused #511 run_eval.py). ONLY the lm_head keepset width moves; the
37L body, int4 body quant, embed_tokens, attention path and config are held fixed.

Decisive question: what is the NARROWEST keepset that RESTORES base quality
(MMLU-Pro >= GATE_MMLU AND GPQA-Diamond >= GATE_GPQA), and what TPS does it cost?
full/no-prune (K=262144) is the CEILING: if the full head misses the gate, no
narrower width can pass.

Reads:
  * base anchor JSONs (#511, 42L): base_mmlu_pro.json / base_gpqa.json
  * per-width ship JSONs: <results>/<label>/ship_mmlu_pro.json / ship_gpqa.json
  * optional speed JSON: {label: {"warm_median_tps":..,"peak_gpu_mem_mib":..}}
Emits the per-K table, the KEY OUTPUTS, a verdict, an aggregate JSON, the W&B log
(group keepset-width-quality-sweep, analysis_only, official_tps=0) and SENPAI-RESULT.
"""
import argparse
import json
import math
import os
import sys

# Gate (Morgan #483) and banked anchors (all byte-identical inspect_evals harness)
GATE_MMLU = 0.60
GATE_GPQA = 0.42
DIXIE_BASE_MMLU = 0.668          # 42L base ceiling (dixie #483 / #511)
DIXIE_SUBSTRATE_MMLU = 0.330     # dixie #483 "frontier substrate" = 37L + 16k->12k head
DIXIE_SUBSTRATE_GPQA = 0.283
# Speed anchors (local A10G, warm-median 128x512 served TPS)
ANCHOR_SURGICAL357_TPS = 357.06  # surgical-357 deployed fast stack @ 12k (W&B j7qao5e9)
ANCHOR_SPLITKV12K_TPS = 442.35   # byteexact split-KV @ 12k (W&B kwhylaeg)

# width label -> keepset K (ascending). full_pck04 == full vocab via pck04 identity scatter.
WIDTH_K = {"12k": 12288, "16k": 16384, "32k": 32768, "full_pck04": 262144}
WIDTH_ORDER = ["12k", "16k", "32k", "full_pck04"]


def _load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _ci95(p, n):
    if not n:
        return (float("nan"), float("nan"))
    h = 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def _prompt_check(base, ship):
    """prompt_identical (per-id prompt_sha) + answer_agreement of ship vs base."""
    if base is None or ship is None:
        return {"prompt_identical": None, "n_prompt_mismatch": None,
                "answer_agreement": None, "n_common": None}
    b = {r["id"]: r for r in base["per_sample"]}
    s = {r["id"]: r for r in ship["per_sample"]}
    common = sorted(set(b) & set(s))
    mism = [i for i in common if b[i].get("prompt_sha") != s[i].get("prompt_sha")]
    agree = sum(1 for i in common if b[i]["answer"] == s[i]["answer"])
    return {
        "prompt_identical": (len(mism) == 0),
        "n_prompt_mismatch": len(mism),
        "answer_agreement": (agree / len(common)) if common else float("nan"),
        "n_common": len(common),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--base-mmlu", required=True)
    ap.add_argument("--base-gpqa", required=True)
    ap.add_argument("--speed-json", default=None, help="label -> {warm_median_tps, peak_gpu_mem_mib}")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "aggregate_sweep.json"))
    ap.add_argument("--wandb_name", default="ubel/keepset-width-quality-sweep")
    ap.add_argument("--wandb_group", default="keepset-width-quality-sweep")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    bm, bg = _load(a.base_mmlu), _load(a.base_gpqa)
    base_mmlu = bm["accuracy"] if bm else None
    base_gpqa = bg["accuracy"] if bg else None
    speed = _load(a.speed_json) or {}

    points = []
    for label in WIDTH_ORDER:
        K = WIDTH_K[label]
        sm = _load(os.path.join(a.results, label, "ship_mmlu_pro.json"))
        sg = _load(os.path.join(a.results, label, "ship_gpqa.json"))
        if sm is None or sg is None:
            print(f"[warn] width {label} (K={K}) missing results -> skipped", file=sys.stderr)
            continue
        mmlu = sm["accuracy"]
        gpqa = sg["accuracy"]
        pc_m = _prompt_check(bm, sm)
        pc_g = _prompt_check(bg, sg)
        # speed_sweep.py keys full vocab as "full"; quality dir is "full_pck04".
        sp = speed.get(label) or speed.get(label.replace("_pck04", "")) or {}
        points.append({
            "label": label,
            "K": K,
            "mmlu_pro": mmlu,
            "gpqa_diamond": gpqa,
            "mmlu_pro_ci95": _ci95(mmlu, sm["n_scored"]),
            "gpqa_ci95": _ci95(gpqa, sg["n_scored"]),
            "mmlu_n": sm["n_scored"], "gpqa_n": sg["n_scored"],
            "mmlu_err": sm.get("n_error", 0), "gpqa_err": sg.get("n_error", 0),
            "gate_pass": bool(mmlu >= GATE_MMLU and gpqa >= GATE_GPQA),
            "mmlu_delta_vs_base": (mmlu - base_mmlu) if base_mmlu is not None else None,
            "gpqa_delta_vs_base": (gpqa - base_gpqa) if base_gpqa is not None else None,
            "prompt_identical_mmlu": pc_m["prompt_identical"],
            "prompt_identical_gpqa": pc_g["prompt_identical"],
            "answer_agreement_mmlu": pc_m["answer_agreement"],
            "answer_agreement_gpqa": pc_g["answer_agreement"],
            "warm_median_tps": sp.get("warm_median_tps"),
            "peak_gpu_mem_mib": sp.get("peak_gpu_mem_mib"),
        })

    # KEY OUTPUTS
    tps_by_label = {p["label"]: p["warm_median_tps"] for p in points}
    tps_12k = tps_by_label.get("12k") if tps_by_label.get("12k") is not None else ANCHOR_SURGICAL357_TPS
    safe = [p for p in points if p["gate_pass"]]
    safe.sort(key=lambda p: p["K"])
    min_safe = safe[0] if safe else None
    quality_safe_ship_exists = bool(safe)
    quality_safe_ship_tps = (min_safe.get("warm_median_tps") if min_safe else None)
    tps_cost_of_quality = (
        (tps_12k - quality_safe_ship_tps)
        if (quality_safe_ship_tps is not None and tps_12k is not None) else None
    )

    # ceiling = full head
    full_pt = next((p for p in points if p["label"] == "full_pck04"), None)
    prompt_ok = all(
        (p["prompt_identical_mmlu"] in (True, None)) and (p["prompt_identical_gpqa"] in (True, None))
        for p in points
    )

    if quality_safe_ship_exists:
        verdict = (
            f"QUALITY-SAFE SHIP EXISTS at K={min_safe['K']} ({min_safe['label']}): "
            f"MMLU-Pro={min_safe['mmlu_pro']:.4f} GPQA={min_safe['gpqa_diamond']:.4f} "
            f"(gate {GATE_MMLU}/{GATE_GPQA}); warm_median_tps={quality_safe_ship_tps}; "
            f"tps_cost_of_quality vs 12k={tps_cost_of_quality}."
        )
    else:
        cm = full_pt["mmlu_pro"] if full_pt else float("nan")
        cg = full_pt["gpqa_diamond"] if full_pt else float("nan")
        verdict = (
            f"NO QUALITY-SAFE SHIP AT ANY WIDTH. The CEILING (full/no-prune head, "
            f"K=262144) scores MMLU-Pro={cm:.4f} / GPQA={cg:.4f} -- below the gate "
            f"({GATE_MMLU}/{GATE_GPQA}) and far below base ({base_mmlu:.4f}/{base_gpqa:.4f}). "
            f"Widening the lm_head keepset 12k->full lifts MMLU-Pro only "
            f"{(full_pt['mmlu_pro']-points[0]['mmlu_pro']):+.4f} (12k={points[0]['mmlu_pro']:.4f}); "
            f"the 37L layer-reduction + int4 BODY is the dominant, non-head-recoverable "
            f"bottleneck. Keepset width is NOT the lever for quality."
        )

    report = {
        "pr": 527,
        "gate_mmlu_threshold": GATE_MMLU, "gate_gpqa_threshold": GATE_GPQA,
        "base_mmlu_pro": base_mmlu, "base_gpqa_diamond": base_gpqa,
        "dixie_substrate_mmlu": DIXIE_SUBSTRATE_MMLU, "dixie_substrate_gpqa": DIXIE_SUBSTRATE_GPQA,
        "anchor_surgical357_tps": ANCHOR_SURGICAL357_TPS,
        "anchor_splitkv12k_tps": ANCHOR_SPLITKV12K_TPS,
        "per_width": points,
        "min_quality_safe_keepset": (min_safe["K"] if min_safe else None),
        "quality_safe_ship_exists": quality_safe_ship_exists,
        "quality_safe_ship_tps": quality_safe_ship_tps,
        "tps_cost_of_quality": tps_cost_of_quality,
        "tps_12k_reference": tps_12k,
        "all_prompts_identical": prompt_ok,
        "verdict": verdict,
        "analysis_only": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps": 0,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)

    # ---- print table ----
    print("\n==== KEEPSET-WIDTH QUALITY SWEEP (PR #527) ====")
    print(f"base (42L): MMLU-Pro={base_mmlu:.4f}  GPQA-Diamond={base_gpqa:.4f}   "
          f"gate: MMLU>={GATE_MMLU} AND GPQA>={GATE_GPQA}")
    print(f"{'width':10s} {'K':>7s} {'MMLU-Pro':>9s} {'GPQA-D':>7s} {'gate':>5s} "
          f"{'dMMLU':>7s} {'dGPQA':>7s} {'TPS':>8s} {'pid':>4s}")
    for p in points:
        tps = f"{p['warm_median_tps']:.2f}" if p["warm_median_tps"] is not None else "  --  "
        pid = "Y" if (p["prompt_identical_mmlu"] and p["prompt_identical_gpqa"]) else "?"
        print(f"{p['label']:10s} {p['K']:>7d} {p['mmlu_pro']:>9.4f} {p['gpqa_diamond']:>7.4f} "
              f"{('PASS' if p['gate_pass'] else 'FAIL'):>5s} {p['mmlu_delta_vs_base']:>+7.4f} "
              f"{p['gpqa_delta_vs_base']:>+7.4f} {tps:>8s} {pid:>4s}")
    print(f"\nmin_quality_safe_keepset : {report['min_quality_safe_keepset']}")
    print(f"quality_safe_ship_exists : {quality_safe_ship_exists}")
    print(f"quality_safe_ship_tps    : {quality_safe_ship_tps}")
    print(f"tps_cost_of_quality      : {tps_cost_of_quality}")
    print(f"all_prompts_identical    : {prompt_ok}")
    print(f"VERDICT: {verdict}")

    senpai = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [],
        "primary_metric": {"name": "full_head_mmlu_pro_ceiling",
                           "value": (full_pt["mmlu_pro"] if full_pt else None)},
        "test_metric": {"name": "full_head_gpqa_diamond_ceiling",
                        "value": (full_pt["gpqa_diamond"] if full_pt else None)},
    }
    print("SENPAI-RESULT:", json.dumps(senpai))

    if not a.no_wandb:
        _log_wandb(report, a)
    return 0


def _log_wandb(report, a):
    sys.path.insert(0, "/workspace/senpai/target")
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; JSON saved, skipping wandb", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="ubel",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#527 keepset-width quality sweep: narrowest lm_head keepset that "
              "RESTORES base quality (MMLU-Pro>=0.60 AND GPQA-Diamond>=0.42) on the "
              "37L-int4 osoi5 substrate, and its TPS cost. full/no-prune is the ceiling.",
        config={
            "pr": 527, "analysis_only": True, "no_hf_job": True,
            "no_served_file_change": True, "official_tps": 0,
            "gate_mmlu_threshold": GATE_MMLU, "gate_gpqa_threshold": GATE_GPQA,
            "base_mmlu_pro": report["base_mmlu_pro"], "base_gpqa_diamond": report["base_gpqa_diamond"],
            "anchor_surgical357_tps": ANCHOR_SURGICAL357_TPS,
            "anchor_splitkv12k_tps": ANCHOR_SPLITKV12K_TPS,
            "widths": [p["K"] for p in report["per_width"]],
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    for k in ("min_quality_safe_keepset", "quality_safe_ship_exists", "quality_safe_ship_tps",
              "tps_cost_of_quality", "all_prompts_identical", "official_tps", "analysis_only",
              "base_mmlu_pro", "base_gpqa_diamond"):
        run.summary[k] = report[k]
    # per-width metrics keyed by K so the curve is queryable
    for p in report["per_width"]:
        pre = f"K{p['K']}"
        run.summary[f"{pre}/mmlu_pro"] = p["mmlu_pro"]
        run.summary[f"{pre}/gpqa_diamond"] = p["gpqa_diamond"]
        run.summary[f"{pre}/gate_pass"] = p["gate_pass"]
        run.summary[f"{pre}/mmlu_delta_vs_base"] = p["mmlu_delta_vs_base"]
        run.summary[f"{pre}/gpqa_delta_vs_base"] = p["gpqa_delta_vs_base"]
        run.summary[f"{pre}/answer_agreement_mmlu"] = p["answer_agreement_mmlu"]
        run.summary[f"{pre}/answer_agreement_gpqa"] = p["answer_agreement_gpqa"]
        if p["warm_median_tps"] is not None:
            run.summary[f"{pre}/warm_median_tps"] = p["warm_median_tps"]
        if p["peak_gpu_mem_mib"] is not None:
            run.summary[f"{pre}/peak_gpu_mem_mib"] = p["peak_gpu_mem_mib"]
    # a wandb Table for the curve
    try:
        import wandb
        cols = ["label", "K", "mmlu_pro", "gpqa_diamond", "gate_pass",
                "warm_median_tps", "answer_agreement_mmlu"]
        tbl = wandb.Table(columns=cols)
        for p in report["per_width"]:
            tbl.add_data(p["label"], p["K"], p["mmlu_pro"], p["gpqa_diamond"],
                         p["gate_pass"], p["warm_median_tps"], p["answer_agreement_mmlu"])
        run.log({"width_sweep_table": tbl})
    except Exception as exc:
        print(f"[wandb] table log skipped: {exc!r}", flush=True)
    run.summary["verdict_text"] = report["verdict"]
    print(f"[wandb] logged run id={getattr(run,'id',None)}", flush=True)
    try:
        finish_wandb(run)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
