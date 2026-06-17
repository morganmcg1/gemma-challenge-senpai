#!/usr/bin/env python3
"""PR #618: served-greedy GPQA-Diamond concurrency-determinism sweep on vLLM 0.22.0.

Tests whether served greedy GPQA-Diamond is run-to-run reproducible as a function
of client concurrency. #610 (gho8wxxs) found conc=16 greedy is NOT reproducible on
dev307 (0.4697 vs 0.4394, 64/198 answers flip). This re-measures on 0.22.0 (the
faithful determinism engine per lawine #606) and adds the conc=1 (fully serial)
arm the #610 follow-up predicted would remove the batch-composition jitter source.

6 runs total: arms {conc=1, conc=16} x 3 independent repeats. ALL greedy
(temperature=0), identical dataset seed (12345 -> byte-identical prompts, also
identical to #610) and sampling-seed (0), max-model-len 6144, max-tokens 3072,
min-tokens 8 (#541 EOS guard). Within an arm the 3 repeats are byte-identical
invocations -> any answer difference is engine run-to-run nondeterminism. Across
arms the ONLY difference is --max-connections (1 vs 16).

Resumable + detached-safe: skips completed result JSONs and resumes the same W&B
run by id (WANDB_RUN_ID/WANDB_RESUME). Inits W&B + logs config before the first
eval so the pod-liveness W&B check passes in <15 min.

analysis_only=true, official_tps=0. LOCAL single A10G. NO HF Job / NO submission.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

HERE = ROOT / "research/validity/gpqa_concurrency_determinism"
RES = HERE / "results"
RUN_EVAL = ROOT / "research/validity/downstream_quality_eval/run_eval.py"
EVAL_PY = ROOT / ".venv/bin/python"  # inspect_ai 0.3.240 client venv

PORT = int(os.environ.get("PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
DATASET_SEED = 12345          # #610 parity -> byte-identical prompts
SAMPLING_SEED = 0             # greedy: moot, held fixed across all 6 runs
MAX_TOKENS = 3072             # #610 parity
MIN_TOKENS = 8                # #541 EOS guard (#610 parity)
ARM = "int4g128"
CONCS = [16, 1]  # fast arm first: early conc=16 signal + W&B liveness; slow serial conc=1 after
REPEATS = 3
GATE_BASE = 0.5236            # vanilla-base GPQA-Diamond greedy anchor
GATE_BAR = 0.90 * GATE_BASE   # 0.47124


def out_path(conc: int, rep: int) -> Path:
    return RES / f"conc{conc}_rep{rep}.json"


def run_one(conc: int, rep: int, *, limit: int = 0) -> dict:
    out = out_path(conc, rep)
    if out.exists() and out.stat().st_size > 0:
        d = json.loads(out.read_text())
        if d.get("n_scored"):
            print(f"[sweep] SKIP existing conc={conc} rep={rep} acc={d['accuracy']:.4f}",
                  flush=True)
            return d
    cmd = [
        str(EVAL_PY), str(RUN_EVAL),
        "--task", "gpqa_diamond", "--arm", ARM, "--out", str(out),
        "--seed", str(DATASET_SEED), "--max-tokens", str(MAX_TOKENS),
        "--base-url", BASE_URL, "--model", "gemma-4-e4b-it",
        "--max-connections", str(conc), "--min-tokens", str(MIN_TOKENS),
        "--sampling-seed", str(SAMPLING_SEED),
        # temperature defaults 0.0 (greedy), top_p 1.0, top_k 0 -> #610 greedy parity
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    t0 = time.time()
    print(f"[sweep] START conc={conc} rep={rep} limit={limit or 'full'} "
          f"{time.strftime('%H:%M:%S')}", flush=True)
    subprocess.run(cmd, check=True)
    d = json.loads(out.read_text())
    print(f"[sweep] DONE  conc={conc} rep={rep} acc={d['accuracy']:.4f} "
          f"correct={d['n_correct']}/{d['n_scored']} empty={d.get('n_empty')} "
          f"err={d['n_error']} dt={time.time()-t0:.0f}s", flush=True)
    return d


def _answer_map(d: dict) -> dict[str, str | None]:
    return {r["id"]: r.get("answer") for r in d.get("per_sample", [])}


def _score_map(d: dict) -> dict[str, bool]:
    return {r["id"]: bool(r.get("correct")) for r in d.get("per_sample", [])}


def _unstable(maps: list[dict]) -> list[str]:
    """ids whose value is NOT identical across all repeats."""
    if not maps:
        return []
    common = set(maps[0])
    for m in maps[1:]:
        common &= set(m)
    return sorted(i for i in common if len({m[i] for m in maps}) > 1)


def _pairwise(maps: list[dict]) -> dict[str, int]:
    out = {}
    for a, b in itertools.combinations(range(len(maps)), 2):
        common = set(maps[a]) & set(maps[b])
        out[f"{a}v{b}"] = sum(1 for i in common if maps[a][i] != maps[b][i])
    return out


def _arm_stats(accs: list[float]) -> dict:
    n = len(accs)
    m = sum(accs) / n if n else float("nan")
    sd = math.sqrt(sum((a - m) ** 2 for a in accs) / (n - 1)) if n > 1 else 0.0
    return {"mean": m, "sd": sd, "n": n, "min": min(accs) if accs else None,
            "max": max(accs) if accs else None, "spread": (max(accs) - min(accs)) if accs else None,
            "vals": accs}


def analyze(results: dict[tuple[int, int], dict]) -> dict:
    out: dict = {"per_run": {}, "arms": {}}
    for (conc, rep), d in sorted(results.items()):
        out["per_run"][f"conc{conc}_rep{rep}"] = {
            "accuracy": d["accuracy"], "n_correct": d["n_correct"],
            "n_scored": d["n_scored"], "n_error": d["n_error"],
            "n_empty": d.get("n_empty"), "empty_rate": d.get("empty_rate"),
        }
    for conc in CONCS:
        reps = [results[(conc, r)] for r in range(REPEATS) if (conc, r) in results]
        if not reps:
            continue
        ans_maps = [_answer_map(d) for d in reps]
        sc_maps = [_score_map(d) for d in reps]
        accs = [d["accuracy"] for d in reps]
        ans_unstable = _unstable(ans_maps)
        sc_unstable = _unstable(sc_maps)
        out["arms"][f"conc{conc}"] = {
            "n_repeats": len(reps),
            "accuracy_stats": _arm_stats(accs),
            "answer_flips_pairwise": _pairwise(ans_maps),
            "answer_flips_max_pairwise": max(_pairwise(ans_maps).values()) if len(reps) > 1 else 0,
            "n_answer_unstable_union": len(ans_unstable),
            "answer_unstable_ids": ans_unstable,
            "score_flips_pairwise": _pairwise(sc_maps),
            "n_score_unstable_union": len(sc_unstable),
            "score_unstable_ids": sc_unstable,
            "deterministic": len(ans_unstable) == 0,
        }
    # (c) mean-unbiased vs systematic shift
    a1 = out["arms"].get("conc1", {}).get("accuracy_stats")
    a16 = out["arms"].get("conc16", {}).get("accuracy_stats")
    if a1 and a16:
        delta = a1["mean"] - a16["mean"]
        # spread-relative: is the arm-mean gap small vs the within-arm run-to-run spread?
        max_spread = max(a1.get("spread") or 0.0, a16.get("spread") or 0.0)
        out["mean_compare"] = {
            "conc1_mean": a1["mean"], "conc16_mean": a16["mean"],
            "delta_conc1_minus_conc16": delta,
            "conc1_spread": a1.get("spread"), "conc16_spread": a16.get("spread"),
            "max_within_arm_spread": max_spread,
            "abs_delta_le_spread": abs(delta) <= max_spread if max_spread else None,
        }
    return out


def verdict(analysis: dict) -> dict:
    c1 = analysis["arms"].get("conc1", {})
    c16 = analysis["arms"].get("conc16", {})
    mc = analysis.get("mean_compare", {})
    conc1_det = c1.get("deterministic")
    conc16_jitter = (c16.get("n_answer_unstable_union") or 0) > 0
    no_resolved_shift = mc.get("abs_delta_le_spread")
    if conc1_det and conc16_jitter:
        substrate = "conc1"
        headline = ("conc=1 is run-to-run DETERMINISTIC; conc=16 carries batch-composition "
                    "jitter -> take GPQA gate reads at --max-connections 1")
    elif conc1_det and not conc16_jitter:
        substrate = "either"
        headline = "both conc=1 and conc=16 reproducible on 0.22.0 (no jitter to remove)"
    elif not conc1_det:
        substrate = "neither-clean"
        headline = ("conc=1 is ALSO non-reproducible on 0.22.0 -> jitter source is not "
                    "batch composition alone; multi-seed CI mandatory at any concurrency")
    else:
        substrate = "unknown"
        headline = "indeterminate"
    # Honesty: the substrate recommendation (conc=1) is engine-robust -- serial decode
    # has a fixed reduction order, so determinism does not depend on 0.22.0. But the
    # mean-unbiased / 0.4414-robust questions are NOT cleanly answerable on this run:
    # 0.22.0 craters this int4 build into repetition-loop degeneration (mean 0.194,
    # below 4-way chance), which shifts the operating point off #610's dev307 regime
    # AND adds a loop-absorbing asymmetry. abs_delta_le_spread is a weak, n=3-underpowered
    # heuristic, not a robustness proof for the (sampled) 0.4414 anchor.
    return {
        "substrate_for_gate_reads": substrate,
        "headline": headline,
        "conc1_deterministic": conc1_det,
        "conc16_has_jitter": conc16_jitter,
        "no_resolved_systematic_shift": no_resolved_shift,
        "no_resolved_systematic_shift_criterion": "abs_delta_le_spread (heuristic, n=3 underpowered)",
        "sampled_mean_0p4414_robustness": "indirect_underpowered_recommend_dev307_crosscheck",
        "caveats": [
            "vLLM 0.22.0 craters this int4_g128_lmhead build (mean 0.194, below 4-way chance) via "
            "repetition-loop degeneration (~45% of items loop to the 3072-tok cap); accuracy "
            "magnitude is NOT comparable to #610's dev307 greedy 0.4697 / base 0.5236.",
            "conc=16 jitter magnitude (120/198 answer-unstable) is amplified vs #610's 64 by the "
            "loop-vs-parse degeneration boundary; the conc=1-deterministic / conc=16-jitter "
            "DIRECTION is engine-robust, the magnitude is 0.22.0-specific.",
            "mean-unbiasedness is power-limited (n=3) and regime-confounded: all 3 conc=16 reps fall "
            "below the conc=1 deterministic value and the loop-absorbing mechanism predicts a mild "
            "downward bias under jitter; one-sample t of conc16 vs the conc1 point ~ -1.68, p~0.24.",
            "greedy was measured here; the 0.4414 anchor is a 10-seed SAMPLED mean, so its "
            "robustness to concurrency is inferred from greedy determinism, not directly measured.",
        ],
    }


def peak_gpu_gb() -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, text=True, capture_output=True)
        return round(int(r.stdout.strip().splitlines()[0]) / 1024.0, 2)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="if >0, run ONE conc=1 eval limited to N items, then exit")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        d = run_one(1, 0, limit=args.smoke)
        # smoke gate: coherent => most items non-empty and answers parse
        ne = d.get("n_empty") or 0
        parsed = sum(1 for r in d.get("per_sample", []) if r.get("answer"))
        ok = (d["n_scored"] >= args.smoke) and (ne == 0) and (parsed >= max(1, args.smoke - 1))
        print(f"[smoke] scored={d['n_scored']} parsed_answers={parsed} empty={ne} "
              f"acc={d['accuracy']:.4f} -> {'COHERENT' if ok else 'DEGENERATE/SUSPECT'}",
              flush=True)
        # remove the smoke file so it doesn't pollute the full conc1_rep0 cell
        out_path(1, 0).unlink(missing_ok=True)
        return 0 if ok else 3

    # ---- W&B: resume-safe (survives pod re-invocation) ----
    run = None
    if not args.no_wandb:
        idfile = HERE / "wandb_run_id.txt"
        if idfile.exists():
            os.environ["WANDB_RUN_ID"] = idfile.read_text().strip()
            os.environ["WANDB_RESUME"] = "allow"
        from scripts import wandb_logging
        run = wandb_logging.init_wandb_run(
            job_type="gpqa-concurrency-determinism", agent="kanna",
            name="kanna/gpqa-concurrency-determinism",
            group="gpqa-concurrency-determinism",
            notes=("PR #618: served-greedy GPQA-Diamond run-to-run reproducibility vs client "
                   "concurrency on vLLM 0.22.0. {conc=1, conc=16} x 3 greedy repeats, dataset "
                   "seed 12345 (=#610), max_model_len 6144, max_tokens 3072, min_tokens 8. "
                   "Tests whether conc=1 (serial) removes the conc=16 batch-composition jitter "
                   "#610 found on dev307 (64/198 answer flips), and whether the jitter is "
                   "mean-unbiased. LOCAL A10G; analysis_only, official_tps=0, NO FIRE."),
            tags=["quality-gate", "int4_g128_lmhead", "analysis-only", "pr-618", "gpqa",
                  "gpqa-diamond", "determinism", "concurrency", "vllm-0.22.0"],
            config={"pr": 618, "analysis_only": True, "official_tps": 0,
                    "engine": "vllm-0.22.0", "checkpoint": "int4_g128_lmhead",
                    "model_len": 6144, "max_num_seqs": 16, "max_tokens": MAX_TOKENS,
                    "min_tokens_guard": MIN_TOKENS, "dataset_seed": DATASET_SEED,
                    "sampling_seed": SAMPLING_SEED, "concs": CONCS, "repeats": REPEATS,
                    "decode": "greedy", "gate_bar": GATE_BAR, "gate_base_ref": GATE_BASE,
                    "anchor_610_run": "gho8wxxs", "anchor_610_conc16_greedy": [0.4697, 0.4394],
                    "anchor_610_conc16_answer_flips": 64},
        )
        if run is not None:
            rid = getattr(run, "id", None)
            idfile.write_text(str(rid))
            print(f"[sweep] wandb run live: {rid}", flush=True)
            run.summary["status"] = "running"

    # ---- run the 6 evals (idempotent), logging each as it completes ----
    results: dict[tuple[int, int], dict] = {}
    step = 0
    for conc in CONCS:
        for rep in range(REPEATS):
            d = run_one(conc, rep)
            results[(conc, rep)] = d
            if run is not None:
                run.log({"global_step": step, "run/conc": conc, "run/rep": rep,
                         "run/accuracy": d["accuracy"], "run/n_correct": d["n_correct"],
                         "run/n_empty": d.get("n_empty") or 0, "run/n_error": d["n_error"]})
            step += 1

    # ---- analyze + verdict ----
    analysis = analyze(results)
    vd = verdict(analysis)
    detail = {
        "pr": 618, "engine": "vllm-0.22.0", "checkpoint": "/workspace/gemma_build/int4_g128_lmhead",
        "model_len": 6144, "max_num_seqs": 16, "max_tokens": MAX_TOKENS,
        "min_tokens_guard": MIN_TOKENS, "dataset_seed": DATASET_SEED, "sampling_seed": SAMPLING_SEED,
        "decode": "greedy", "n_dataset": next(iter(results.values()))["n_dataset"],
        "gate_bar": GATE_BAR, "gate_base_ref": GATE_BASE,
        "anchor_610": {"run": "gho8wxxs", "engine": "vllm-0.22.1rc1.dev307",
                       "conc16_greedy_runs": [0.4697, 0.4394], "conc16_answer_flips": 64,
                       "sampled_10seed_mean": 0.4414},
        "analysis": analysis, "verdict": vd,
        "peak_gpu_gb": peak_gpu_gb(), "analysis_only": True, "official_tps": 0,
    }
    (HERE / "determinism_summary.json").write_text(json.dumps(detail, indent=2, default=str))
    print("DETERMINISM_SUMMARY " + json.dumps(detail, default=str), flush=True)

    if run is not None:
        from scripts import wandb_logging
        ko = {
            "conc1_n_answer_unstable": analysis["arms"]["conc1"]["n_answer_unstable_union"],
            "conc16_n_answer_unstable": analysis["arms"]["conc16"]["n_answer_unstable_union"],
            "conc1_n_score_unstable": analysis["arms"]["conc1"]["n_score_unstable_union"],
            "conc16_n_score_unstable": analysis["arms"]["conc16"]["n_score_unstable_union"],
            "conc1_mean_acc": analysis["arms"]["conc1"]["accuracy_stats"]["mean"],
            "conc16_mean_acc": analysis["arms"]["conc16"]["accuracy_stats"]["mean"],
            "mean_delta_conc1_minus_conc16": analysis["mean_compare"]["delta_conc1_minus_conc16"],
            "no_resolved_systematic_shift": vd["no_resolved_systematic_shift"],
            "substrate_for_gate_reads": vd["substrate_for_gate_reads"],
            "conc1_deterministic": vd["conc1_deterministic"],
            "peak_gpu_gb": detail["peak_gpu_gb"], "analysis_only": True, "official_tps": 0,
        }
        for k, v in ko.items():
            run.summary[k] = v
        run.summary["status"] = "completed"
        wandb_logging.log_summary(run, {k: v for k, v in detail.items()
                                        if not isinstance(v, (dict, list))}, step=step)
        wandb_logging.log_json_artifact(run, name="gpqa_concurrency_determinism_detail",
                                        artifact_type="quality-eval", data=detail)
        wandb_logging.finish_wandb(run)
        print(f"[sweep] wandb finished run_id={getattr(run,'id',None)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
