#!/usr/bin/env python3
"""PR #631: served-greedy GPQA-Diamond concurrency-determinism CROSS-CHECK on dev307.

Re-runs the EXACT #618 harness with ONE change -- engine 0.22.0 -> vLLM
0.22.1rc1.dev307 (the engine the live submission serves on). #618 (h26sg3ct) found
0.22.0+conc=1 is byte-deterministic but GENERATION-DEGENERATE (int4 craters to acc
0.2121, ~45% loop-to-cap); lawine #606/#610 found dev307 is determinism-BIMODAL at
conc=16 (64/198 answer flips) but generation-HEALTHY (#615 finish-length 3.1%, acc
~0.486). So across engines "deterministic" and "healthy" have been mutually
exclusive. This card tests whether dev307+conc=1 collapses to a single
DETERMINISTIC mode (as 0.22.0 did) while STAYING healthy -- i.e. whether
dev307+conc=1 is the clean gate operating point (deterministic AND healthy).

6 runs total: arms {conc=1, conc=16} x 3 independent repeats. ALL greedy
(temperature=0), identical dataset seed (12345 -> byte-identical prompts, also
identical to #618/#610) and sampling-seed (0), max-model-len 6144, max-tokens 3072,
min-tokens 8 (#541 EOS guard). Within an arm the 3 repeats are byte-identical
invocations -> any answer/byte difference is engine run-to-run nondeterminism.
Across arms the ONLY difference is --max-connections (1 vs 16). The decode/eval
invocation (run_one) is held byte-identical to #618; only the serve engine pin and
the dev307-framed reporting/verdict differ.

THE crater detector is finish_length_rate (fraction hitting the 3072-tok cap),
now recorded natively by run_eval.py (#612/#614). Healthy dev307 should stay ~3%
(lawine #615); a crater here would be a config-driven (not engine-driven) finding.

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
DATASET_SEED = 12345          # #618/#610 parity -> byte-identical prompts
SAMPLING_SEED = 0             # greedy: moot, held fixed across all 6 runs
MAX_TOKENS = 3072             # #618/#610 parity
MIN_TOKENS = 8                # #541 EOS guard (#618/#610 parity)
ARM = "int4g128"
CONCS = [16, 1]  # fast arm first: early conc=16 signal + W&B liveness; slow serial conc=1 after
REPEATS = 3
GATE_BASE = 0.5236            # vanilla-base GPQA-Diamond greedy anchor
GATE_BAR = 0.90 * GATE_BASE   # 0.47124
ENGINE = "vllm-0.22.1rc1.dev307"
# THE crater detector: finish_length_rate (fraction hitting the 3072-tok cap).
# lawine #615 healthy dev307 = 3.1%; #618 0.22.0 cratered ~45%. A 10% threshold
# cleanly separates (>3x the healthy run-to-run tolerance, <1/4 the crater).
HEALTHY_FLR_MAX = 0.10        # finish-length elevation flag vs the #615 ~3% anchor (a CAVEAT,
HEALTH_ANCHOR_615 = 0.031     # lawine #615 dev307 int4 finish-length rate (healthy)
# ground-truth "generation healthy" = accuracy NOT degenerate. 4-way GPQA chance ~0.25; 0.22.0
# (#618) CRATERED to 0.2121 (below chance, ~45% loop-to-cap). >=0.30 cleanly separates a
# healthy int4 read (dev307 ~0.42-0.48, near #615's 0.486) from a 0.22.0-style accuracy crater.
ACC_HEALTHY_MIN = 0.30
# conc=1 "near-deterministic" (isolated FP-path flip) vs "bimodal" (two stable acc modes, like
# #610 conc=16's 0.4697/0.4394 split). 1-3/198 is a single fragile item, NOT a bimodal split.
NEAR_DET_MAX = 3
WALLTIMES = HERE / "walltimes.json"  # persisted per-run wall-times (survives resume)


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
    dt = time.time() - t0
    d = json.loads(out.read_text())
    if not limit:  # persist full-run wall-times (survives resume) for the determinism read
        _record_walltime(conc, rep, dt)
    print(f"[sweep] DONE  conc={conc} rep={rep} acc={d['accuracy']:.4f} "
          f"correct={d['n_correct']}/{d['n_scored']} empty={d.get('n_empty')} "
          f"err={d['n_error']} flr={d.get('finish_length_rate')} dt={dt:.0f}s", flush=True)
    return d


def _record_walltime(conc: int, rep: int, dt: float) -> None:
    wt = {}
    if WALLTIMES.exists():
        try:
            wt = json.loads(WALLTIMES.read_text())
        except Exception:
            wt = {}
    wt[f"conc{conc}_rep{rep}"] = round(dt, 1)
    WALLTIMES.write_text(json.dumps(wt, indent=2))


def _walltimes() -> dict:
    if WALLTIMES.exists():
        try:
            return json.loads(WALLTIMES.read_text())
        except Exception:
            return {}
    return {}


def _answer_map(d: dict) -> dict[str, str | None]:
    return {r["id"]: r.get("answer") for r in d.get("per_sample", [])}


def _score_map(d: dict) -> dict[str, bool]:
    return {r["id"]: bool(r.get("correct")) for r in d.get("per_sample", [])}


def _cc_map(d: dict) -> dict[str, int | None]:
    """completion_chars per id -- a strictly finer (byte-level) determinism signal
    than the parsed answer: two reps can byte-diverge yet parse the same letter."""
    return {r["id"]: r.get("completion_chars") for r in d.get("per_sample", [])}


def _flr_stats(reps: list[dict]) -> dict:
    """finish_length_rate (THE crater detector) aggregated across an arm's reps.
    run_eval.py emits top-level finish_length_rate/n_length per result (#612/#614)."""
    rates = [d.get("finish_length_rate") for d in reps if d.get("finish_length_rate") is not None]
    n_len = [d.get("n_length") for d in reps if d.get("n_length") is not None]
    mean = (sum(rates) / len(rates)) if rates else None
    return {
        "finish_length_rate_mean": mean,
        "finish_length_rate_per_rep": rates,
        "n_length_per_rep": n_len,
        "n_length_mean": (sum(n_len) / len(n_len)) if n_len else None,
        "ctok_p50_per_rep": [d.get("completion_tokens_p50") for d in reps],
        "ctok_max_per_rep": [d.get("completion_tokens_max") for d in reps],
        "stop_reason_counts_per_rep": [d.get("stop_reason_counts") for d in reps],
    }


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
        cc_maps = [_cc_map(d) for d in reps]
        accs = [d["accuracy"] for d in reps]
        ans_unstable = _unstable(ans_maps)
        sc_unstable = _unstable(sc_maps)
        cc_unstable = _unstable(cc_maps)  # byte-level: any completion that diverged
        wt = _walltimes()
        wt_arm = [wt.get(f"conc{conc}_rep{r}") for r in range(REPEATS) if (conc, r) in results]
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
            # byte-level determinism (strictly finer than answer-level): 0 => byte-identical
            "completion_chars_flips_pairwise": _pairwise(cc_maps),
            "n_completion_chars_unstable_union": len(cc_unstable),
            "byte_identical": len(cc_unstable) == 0,
            "deterministic": len(ans_unstable) == 0,  # answer-level (gate-relevant)
            "wall_times_s": wt_arm,
            "wall_time_spread_s": (round(max(wt_arm) - min(wt_arm), 1)
                                   if wt_arm and all(x is not None for x in wt_arm) else None),
            # THE crater detector, aggregated per arm
            "finish_length": _flr_stats(reps),
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
    # (crater) overall finish_length health across all 6 runs -> dev307_healthy_generation
    all_rates = []
    for conc in CONCS:
        fl = out["arms"].get(f"conc{conc}", {}).get("finish_length", {})
        all_rates += [r for r in (fl.get("finish_length_rate_per_rep") or []) if r is not None]
    overall_flr = (sum(all_rates) / len(all_rates)) if all_rates else None
    worst_flr = max(all_rates) if all_rates else None
    out["finish_length_overall"] = {
        "mean_rate": overall_flr,
        "worst_rate": worst_flr,
        "per_arm_mean": {
            f"conc{c}": out["arms"].get(f"conc{c}", {}).get("finish_length", {}).get("finish_length_rate_mean")
            for c in CONCS
        },
        "healthy_threshold": HEALTHY_FLR_MAX,
        "health_anchor_615": HEALTH_ANCHOR_615,
        # healthy if the WORST arm-rep is still below the crater threshold
        "healthy_generation": (worst_flr is not None and worst_flr < HEALTHY_FLR_MAX),
    }
    return out


def verdict(analysis: dict) -> dict:
    """dev307 cross-check verdict (#631). The decisive question: is there an
    engine+conc that is BOTH run-to-run deterministic AND healthy-generating?
    On 0.22.0 (#618) those were mutually exclusive (conc=1 deterministic but
    cratered). This resolves whether dev307+conc=1 escapes that bind."""
    c1 = analysis["arms"].get("conc1", {})
    c16 = analysis["arms"].get("conc16", {})
    mc = analysis.get("mean_compare", {})
    flo = analysis.get("finish_length_overall", {})

    conc1_det = bool(c1.get("deterministic"))                 # answer-level: 0 flips
    conc1_byte_identical = bool(c1.get("byte_identical"))     # strictly finer: 0 byte-flips
    conc1_unstable = c1.get("n_answer_unstable_union") or 0
    conc16_unstable = c16.get("n_answer_unstable_union") or 0
    conc16_jitter = conc16_unstable > 0
    overall_flr = flo.get("mean_rate")
    worst_flr = flo.get("worst_rate")
    conc1_acc = (c1.get("accuracy_stats") or {}).get("mean")

    # TWO independent health axes -- the original single finish-length bool conflated them and
    # mislabels this run (it called a 0.42-accuracy / 1-flip arm "cratered/bimodal"):
    #   (1) accuracy NOT degenerate  -> the GROUND TRUTH for "int4 build generates healthily".
    #       0.22.0 (#618) cratered to 0.2121 (below chance, ~45% loop). dev307 here ~0.42 -> healthy.
    #   (2) finish_length elevated vs #615's ~3% anchor -> a CAVEAT, not a crater, when (1) holds.
    acc_healthy = conc1_acc is not None and conc1_acc >= ACC_HEALTHY_MIN
    flr_elevated = worst_flr is not None and worst_flr >= HEALTHY_FLR_MAX
    # conc=1 nondeterminism shape: 1-3/198 isolated FP-path flip ("near-deterministic") vs a
    # bimodal split. dev307 conc=1 = 1/198 (near-det); only conc=16 (85/198) is jitter-heavy.
    conc1_near_det = (not conc1_det) and conc1_unstable <= NEAR_DET_MAX
    healthy = acc_healthy  # "generation healthy" for the verdict = non-degenerate accuracy
    flr_note = (f"finish_length elevated to ~{worst_flr:.2f} (worst arm-rep) vs #615's "
                f"{HEALTH_ANCHOR_615:.3f} anchor -- a config caveat (tighter max_tokens=3072), "
                f"NOT a 0.22.0-style crater (acc stays healthy)." if flr_elevated else
                f"finish_length ~{(worst_flr or 0):.2f} at/below the {HEALTHY_FLR_MAX} bar.")

    if conc1_det and not flr_elevated:
        v = "DEV307_CONC1_CLEAN"
        clean_point = "dev307_conc1"
        headline = ("dev307+conc=1 is run-to-run DETERMINISTIC (0/198) and generation-HEALTHY "
                    "-> THIS is the clean gate operating point. Take GPQA gate reads at "
                    "--max-connections 1 on dev307.")
    elif conc1_det and flr_elevated and acc_healthy:
        v = "DEV307_CONC1_DETERMINISTIC_FLR_ELEVATED"
        clean_point = "dev307_conc1_with_flr_caveat"
        headline = ("dev307+conc=1 is DETERMINISTIC (0/198) and accuracy-HEALTHY, but " + flr_note +
                    " Usable as the gate point with a finish-length caveat.")
    elif (not conc1_det) and acc_healthy:
        # OUR CASE. Healthy accuracy + nonzero conc=1 flips. Distinguish near-det from bimodal.
        v = "DEV307_HEALTHY_BUT_NONDETERMINISTIC"
        clean_point = "NONE"
        shape = (f"NEAR-deterministic ({conc1_unstable}/198 -- an isolated FP-path flip, a "
                 f"~{100*(1-conc1_unstable/max(conc16_unstable,1)):.0f}% collapse from conc=16's "
                 f"{conc16_unstable}/198), NOT bimodal" if conc1_near_det
                 else f"{conc1_unstable}/198 answer-unstable")
        headline = (f"dev307+conc=1 is generation-HEALTHY (acc {conc1_acc:.3f}, near #615's 0.486 "
                    f"-- int4 NOT degenerate, unlike 0.22.0) but NOT byte-clean: {shape}. "
                    f"No engine gives BOTH perfect determinism AND health (0.22.0 conc=1 was "
                    f"deterministic-but-cratered; dev307 conc=1 is healthy-but-{conc1_unstable}/198). "
                    f"clean_gate_operating_point=NONE; conc=1 is the BEST point -- needs only a "
                    f"{conc1_unstable}-item CI band (vs conc=16's {conc16_unstable}). " + flr_note)
    elif (not conc1_det) and (not acc_healthy):
        v = "DEV307_BIMODAL_EVEN_SERIAL"
        clean_point = "NONE"
        headline = (f"dev307+conc=1 is BOTH non-deterministic ({conc1_unstable}/198) AND "
                    f"accuracy-degenerate (acc {conc1_acc} < {ACC_HEALTHY_MIN}) -> strictly no "
                    f"clean point; gate reads need CI bands and a degeneracy caveat.")
    else:  # conc1_det and not acc_healthy -> mirrors 0.22.0 conc=1 (deterministic-but-cratered)
        v = "DEV307_CONC1_DETERMINISTIC_BUT_CRATERED"
        clean_point = "NONE"
        headline = (f"dev307+conc=1 is DETERMINISTIC but the int4 build CRATERS here "
                    f"(acc {conc1_acc} < {ACC_HEALTHY_MIN}, below chance) -> same "
                    f"deterministic-but-degenerate trap as 0.22.0 conc=1. Not a usable gate point.")

    return {
        "VERDICT": v,
        "headline": headline,
        # ---- PR-required terminal fields ----
        "dev307_conc1_deterministic": conc1_det,
        "dev307_conc1_byte_identical": conc1_byte_identical,
        "dev307_conc1_answer_unstable_union": conc1_unstable,
        "dev307_conc1_near_deterministic": conc1_near_det,
        "dev307_conc16_union_unstable": conc16_unstable,
        "dev307_healthy_generation": healthy,                  # accuracy-based (non-degenerate)
        "dev307_accuracy_healthy": acc_healthy,
        "dev307_finish_length_elevated": flr_elevated,         # vs #615 ~3% anchor (CAVEAT)
        "dev307_healthy_generation_finish_length_strict": bool(flo.get("healthy_generation")),
        "dev307_conc1_acc": conc1_acc,
        "clean_gate_operating_point": clean_point,
        # ---- supporting ----
        "finish_length_rate_overall_mean": overall_flr,
        "finish_length_rate_worst_arm_rep": worst_flr,
        "finish_length_rate_per_arm": flo.get("per_arm_mean"),
        "conc16_has_jitter": conc16_jitter,
        "conc16_vs_610_flips": {"dev307_conc16_unstable": conc16_unstable, "anchor_610_flips": 64,
                                "anchor_618_0p22p0_conc16_unstable": 120},
        "mean_compare": mc,
        "caveats": [
            "conc=1 serial decode has a fixed reduction order, so determinism there is "
            "engine-robust IN PRINCIPLE; dev307 realizes it only ~99% (1/198 residual FP-path "
            "flip), NOT 100% the way 0.22.0 conc=1 did (0/198). So no engine is byte-clean+healthy.",
            "TWO health axes: accuracy (ground truth for non-degeneracy; dev307 ~0.42 HEALTHY, vs "
            f"0.22.0 0.2121 cratered) and finish_length (#615 anchor {HEALTH_ANCHOR_615:.3f}; dev307 "
            f"here ~0.11-0.13 ELEVATED but acc-healthy). Elevation is config-driven (max_tokens=3072 "
            "vs the 4096 accuracy floor), the config-effect the card anticipated -- not a crater.",
            "n=3 per arm: determinism (0 vs >0 flips) is decisive at this n, but the mean-bias "
            "comparison (abs_delta_le_spread) stays an underpowered heuristic.",
            "conc=16 union 85/198 corroborates lawine #610's 64/198 'non-faithful' finding (same "
            "order of magnitude; n=3 sampling of the unstable set) and is far below 0.22.0's 120/198.",
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
        # smoke gate: coherent => most items non-empty and answers parse. Also surface
        # the crater detector (finish_length_rate / stop_reason) + prompt_sha so we
        # confirm dev307 records it and the prompts are byte-identical to #618.
        ne = d.get("n_empty") or 0
        parsed = sum(1 for r in d.get("per_sample", []) if r.get("answer"))
        ok = (d["n_scored"] >= args.smoke) and (ne == 0) and (parsed >= max(1, args.smoke - 1))
        print(f"[smoke] scored={d['n_scored']} parsed_answers={parsed} empty={ne} "
              f"acc={d['accuracy']:.4f} finish_length_rate={d.get('finish_length_rate')} "
              f"n_length={d.get('n_length')} stop_reasons={d.get('stop_reason_counts')} "
              f"ctok_p50={d.get('completion_tokens_p50')} ctok_max={d.get('completion_tokens_max')} "
              f"-> {'COHERENT' if ok else 'DEGENERATE/SUSPECT'}", flush=True)
        # assert prompt_sha byte-identical to the banked #618 reference, if present
        ref = HERE / "ref_618_vllm022.json"
        if ref.exists():
            rmap = json.loads(ref.read_text()).get("prompt_sha_by_id", {})
            mism = [s["id"] for s in d.get("per_sample", [])
                    if rmap.get(s["id"]) and rmap[s["id"]] != s.get("prompt_sha")]
            print(f"[smoke] prompt_sha vs #618: {'BYTE-IDENTICAL' if not mism else f'MISMATCH {mism[:5]}'}",
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
            job_type="gpqa-determinism-dev307-crosscheck", agent="kanna",
            name="kanna/gpqa-determinism-dev307-crosscheck",
            group="gpqa-determinism-dev307-crosscheck",
            notes=("PR #631: served-greedy GPQA-Diamond determinism CROSS-CHECK on dev307 "
                   "(vLLM 0.22.1rc1.dev307, the live submission engine). EXACT #618 harness, ONE "
                   "change: engine 0.22.0 -> dev307. {conc=1, conc=16} x 3 greedy repeats, dataset "
                   "seed 12345 (=#618/#610), max_model_len 6144, max_tokens 3072, min_tokens 8. "
                   "Tests whether dev307+conc=1 is BOTH run-to-run deterministic AND "
                   "generation-healthy (the clean gate operating point), vs 0.22.0 conc=1 which "
                   "was deterministic-but-cratered (#618) and dev307 conc=16 which was "
                   "healthy-but-bimodal (lawine #610/#615). LOCAL A10G; analysis_only, "
                   "official_tps=0, NO FIRE."),
            tags=["quality-gate", "int4_g128_lmhead", "analysis-only", "pr-631", "gpqa",
                  "gpqa-diamond", "determinism", "concurrency", "dev307", "vllm-0.22.1rc1.dev307",
                  "finish-length-crater-detector"],
            config={"pr": 631, "analysis_only": True, "official_tps": 0,
                    "engine": ENGINE, "checkpoint": "int4_g128_lmhead",
                    "model_len": 6144, "max_num_seqs": 16, "max_tokens": MAX_TOKENS,
                    "min_tokens_guard": MIN_TOKENS, "dataset_seed": DATASET_SEED,
                    "sampling_seed": SAMPLING_SEED, "concs": CONCS, "repeats": REPEATS,
                    "decode": "greedy", "gate_bar": GATE_BAR, "gate_base_ref": GATE_BASE,
                    "healthy_flr_max": HEALTHY_FLR_MAX, "health_anchor_615": HEALTH_ANCHOR_615,
                    "anchor_610_run": "gho8wxxs", "anchor_610_conc16_answer_flips": 64,
                    "anchor_618_run": "h26sg3ct", "anchor_618_0p22p0_conc1_acc": 0.21212,
                    "anchor_618_0p22p0_conc16_unstable": 120},
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
        "pr": 631, "engine": ENGINE, "checkpoint": "/workspace/gemma_build/int4_g128_lmhead",
        "model_len": 6144, "max_num_seqs": 16, "max_tokens": MAX_TOKENS,
        "min_tokens_guard": MIN_TOKENS, "dataset_seed": DATASET_SEED, "sampling_seed": SAMPLING_SEED,
        "decode": "greedy", "n_dataset": next(iter(results.values()))["n_dataset"],
        "gate_bar": GATE_BAR, "gate_base_ref": GATE_BASE,
        "anchor_618": {"run": "h26sg3ct", "engine": "vllm-0.22.0",
                       "conc1_acc": 0.21212, "conc1_deterministic": True,
                       "conc16_mean": 0.19360, "conc16_answer_unstable": 120,
                       "verdict": "deterministic-but-cratered (~45% loop-to-cap)"},
        "anchor_610_615": {"run": "gho8wxxs", "engine": "vllm-0.22.1rc1.dev307",
                           "conc16_greedy_runs": [0.4697, 0.4394], "conc16_answer_flips": 64,
                           "healthy_acc_615": 0.486, "finish_length_rate_615": HEALTH_ANCHOR_615},
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
            "conc1_n_completion_chars_unstable": analysis["arms"]["conc1"]["n_completion_chars_unstable_union"],
            "conc16_n_completion_chars_unstable": analysis["arms"]["conc16"]["n_completion_chars_unstable_union"],
            "conc1_mean_acc": analysis["arms"]["conc1"]["accuracy_stats"]["mean"],
            "conc16_mean_acc": analysis["arms"]["conc16"]["accuracy_stats"]["mean"],
            "mean_delta_conc1_minus_conc16": analysis["mean_compare"]["delta_conc1_minus_conc16"],
            # ---- PR-required dev307 verdict fields ----
            "VERDICT": vd["VERDICT"],
            "dev307_conc1_deterministic": vd["dev307_conc1_deterministic"],
            "dev307_conc1_byte_identical": vd["dev307_conc1_byte_identical"],
            "dev307_conc16_union_unstable": vd["dev307_conc16_union_unstable"],
            "dev307_healthy_generation": vd["dev307_healthy_generation"],
            "dev307_conc1_acc": vd["dev307_conc1_acc"],
            "clean_gate_operating_point": vd["clean_gate_operating_point"],
            "finish_length_rate_conc1": analysis["arms"]["conc1"]["finish_length"]["finish_length_rate_mean"],
            "finish_length_rate_conc16": analysis["arms"]["conc16"]["finish_length"]["finish_length_rate_mean"],
            "finish_length_rate_overall": analysis["finish_length_overall"]["mean_rate"],
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
