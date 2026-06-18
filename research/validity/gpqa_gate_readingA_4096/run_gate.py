#!/usr/bin/env python3
"""PR #643: Option-B GPQA-Diamond Reading-A verdict at the clean dev307-conc1 gate point.

Runs Option-B (int4_g128_lmhead body + Gemma4-MTP K=7 drafter, BI=1) GPQA-Diamond on
vLLM 0.22.1rc1.dev307 at the #631-validated gate operating point: --max-connections 1
(the one near-deterministic engine point; conc=16's 85/198 flips make it ungateable).

The ONE deliberate change vs #631 is max_tokens 3072 -> 4096: #631 found the ~13%
finish-length at the gate point is a max_tokens=3072 CAP ARTIFACT, not a crater. At
mt=4096 the cap should release (finish-length ~13% -> ~3%), giving a clean read. We
prove that here by (a) measuring finish_length_rate at 4096 and (b) back-deriving the
implied 3072 rate from output_tokens in the SAME run (items with output_tokens > 3072
would have truncated at 3072) -- a within-Option-B cap-release demonstration.

Both decode modes per the gate contract:
  - GREEDY  (temperature=0): the strict gate axis. Reported alongside.
  - SAMPLED (temperature=1.0, top_p=0.95, top_k=64 -- lewtun #31 gemma-4-E4B-it
    generation_config protocol): downstream evals use sampled params; THIS is the
    Reading-A axis.

Multi-seed: each --seed is a deterministic GPQA choice-shuffle (a nuisance permutation
we average over). conc=1 makes each seed near-deterministic (1/198 fragile per #631),
so between-seed spread is the real epistemic variance; mean +/- t-CI is the verdict CI.

%-of-base and the verdict (sampled axis):
  - base sampled GPQA-D = 0.5404 (ubel #628 run ilg4z6e9)
  - base greedy  GPQA-D = 0.4899 (ubel #628 run g3cig1xo)
  - bar = 0.90 * 0.5404 = 0.4864
  Verdict: READING_A_GPQA_FAILS (sampled CI upper < 0.4864) /
           READING_A_GPQA_KNIFE_EDGE (CI straddles 0.4864) /
           READING_A_GPQA_PASSES (CI lower >= 0.4864).

Resumable: skips completed result JSONs; resumes the same W&B run by id. The Option-B
server is launched SEPARATELY (serve_spec.py) and assumed live at --base-url; this
driver only runs the evals + analysis, exactly like #631's run_sweep.py.

analysis_only=true, official_tps=0. LOCAL single A10G. NO HF Job / NO submission.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

HERE = ROOT / "research/validity/gpqa_gate_readingA_4096"
RES = HERE / "results"
RUN_EVAL = ROOT / "research/validity/downstream_quality_eval/run_eval.py"
EVAL_PY = ROOT / ".venv/bin/python"  # inspect_ai 0.3.240 + wandb client venv

PORT = int(os.environ.get("PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
CONC = 1                       # the #631 clean gate operating point (near-deterministic)
MAX_TOKENS = 4096              # the cap-release change vs #631's 3072
# #631 served model_len=6144 with mt=3072 (max input that fits: 3072). At mt=4096 that
# 6144 cap collides for the 1 GPQA-D item with 2429 input tokens (2429+4096=6525>6144):
# vLLM 400s it, score_on_error scores it WRONG -> a ~0.5% headwind on Option-B right at
# the bar. model_len 6144->8192 fits 4096 for ALL 198 items (max input 2429; 6525<8192).
# At conc=1+BI=1 model_len only changes KV capacity, not per-sequence compute. PRE-SWEEP
# BYTE-CHECK FINDING: the deployed Option-B config is byte-NONdeterministic run-to-run at
# conc=1 (3-5 of 5 probe items differ) at BOTH 6144 and 8192 -- so model_len is NOT the
# cause; this is the intrinsic dev307+int4-Marlin+greedy A10G nondeterminism (note #38),
# and #631 ITSELF measured byte_identical=false at its gate point: that point is ANSWER-
# near-deterministic (1/198 fragile), NOT byte-identical. 8192 is still strictly better
# than 6144 (clears the collision; divergence is identical either way). The multi-seed
# t-CI below absorbs this within-seed decode variance conservatively (inflates between-
# seed SD), so the ACCURACY verdict is sound; the byte-nondeterminism is a reported caveat.
MODEL_LEN = 8192
MIN_TOKENS = 8                 # #541 EOS guard (parity with #631/#629)
TASK = "gpqa_diamond"
# fern #629 seed family (first 5) are the >=5 FLOOR (advisor: don't terminalize at 3). The
# +5 extension lets the resumable one-seed-per-invocation driver accumulate toward the
# ubel #638 / lawine #639 10-seed protocol (n=1980), so the Option-B sampled CI is
# comparable to those anchors rather than read against them at n<=990. Which seeds actually
# run in a given invocation is set by --seeds; this list is the full target + W&B config.
SEEDS = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 13579]
MODES = ["greedy", "sampled"]  # default run-loop modes (overridable via --modes per invocation)
ALL_MODES = ["greedy", "sampled"]  # FIXED analysis set: the summary always spans BOTH modes,
# even on a --modes sampled-only invocation (greedy is capped at a few seeds for the health
# read while sampled accumulates toward n=10), so disk-globbed analysis must not depend on MODES.
ENGINE = "vllm-0.22.1rc1.dev307"

# ---- denominators + bar (PR #643 / ubel #628 gb6144 panel) ----
BASE_SAMPLED = 0.5404          # ubel #628 run ilg4z6e9 (base sampled GPQA-D)
BASE_GREEDY = 0.4899           # ubel #628 run g3cig1xo (base greedy  GPQA-D)
BAR = 0.4864                   # 0.90 * BASE_SAMPLED (recalibrated >=90% bar)
# AR-body (int4_g128_lmhead body, NO drafter) sampled GPQA-D = 0.4990 (ubel #638, n=1980).
# Advisor #481 denominator read: does Option-B (body+drafter) sampled rise toward the AR-
# body's 0.4990 at mt=4096? pct-of-AR-body isolates the spec-vs-AR-body gap (the drafter's
# quality cost) from the AR-body-vs-base gap (quant/recipe cost). SAMPLED axis only (the
# 0.4990 anchor is a sampled number); reported alongside pct-of-base in the terminal result.
AR_BODY_SAMPLED = 0.4990

# ---- anchors for the cap-artifact health check ----
FLR_3072_ANCHOR = 0.1313      # #631 dev307 conc=1 greedy mt=3072 finish_length_rate
FLR_HEALTHY_TARGET = 0.05     # "clean read" target (~3-5%); cap released

# Student's-t 0.975 two-sided critical values by dof (n-1). df>=11 ~ z plateau.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}

WALLTIMES = HERE / "walltimes.json"


def out_path(mode: str, seed: int) -> Path:
    return RES / f"{mode}_s{seed}.json"


def _load_all_results() -> dict[tuple[str, int], dict]:
    """Glob EVERY completed result JSON on disk (mode_s<seed>.json) and key it by
    (mode, seed). The summary/verdict must reflect all seeds accumulated across resumable
    one-seed-per-invocation wakeups, NOT just the seeds run in the current invocation."""
    results: dict[tuple[str, int], dict] = {}
    for mode in ALL_MODES:
        for p in sorted(RES.glob(f"{mode}_s*.json")):
            stem = p.stem  # e.g. "sampled_s23456"
            try:
                seed = int(stem.rsplit("_s", 1)[1])
            except (IndexError, ValueError):
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            if d.get("n_scored"):
                results[(mode, seed)] = d
    return results


def _eval_cmd(mode: str, seed: int, out: Path, *, limit: int = 0) -> list[str]:
    cmd = [
        str(EVAL_PY), str(RUN_EVAL),
        "--task", TASK, "--arm", f"optionb_{mode}", "--out", str(out),
        "--seed", str(seed), "--max-tokens", str(MAX_TOKENS),
        "--base-url", BASE_URL, "--model", "gemma-4-e4b-it",
        "--max-connections", str(CONC), "--min-tokens", str(MIN_TOKENS),
    ]
    if mode == "greedy":
        cmd += ["--temperature", "0.0"]            # strict gate axis
    else:                                          # sampled: lewtun #31 protocol
        cmd += ["--temperature", "1.0", "--top-p", "0.95", "--top-k", "64"]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def run_one(mode: str, seed: int, *, limit: int = 0) -> dict:
    out = out_path(mode, seed)
    if not limit and out.exists() and out.stat().st_size > 0:
        d = json.loads(out.read_text())
        if d.get("n_scored"):
            print(f"[gate] SKIP existing {mode} s={seed} acc={d['accuracy']:.4f}", flush=True)
            return d
    cmd = _eval_cmd(mode, seed, out, limit=limit)
    t0 = time.time()
    print(f"[gate] START {mode} s={seed} limit={limit or 'full'} {time.strftime('%H:%M:%S')}",
          flush=True)
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    d = json.loads(out.read_text())
    if not limit:
        _record_walltime(mode, seed, dt)
    print(f"[gate] DONE  {mode} s={seed} acc={d['accuracy']:.4f} "
          f"correct={d['n_correct']}/{d['n_scored']} empty={d.get('n_empty')} "
          f"err={d['n_error']} flr={d.get('finish_length_rate')} "
          f"ctok_p95={d.get('completion_tokens_p95')} dt={dt:.0f}s", flush=True)
    return d


def _record_walltime(mode: str, seed: int, dt: float) -> None:
    wt = {}
    if WALLTIMES.exists():
        try:
            wt = json.loads(WALLTIMES.read_text())
        except Exception:
            wt = {}
    wt[f"{mode}_s{seed}"] = round(dt, 1)
    WALLTIMES.write_text(json.dumps(wt, indent=2))


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n if n else float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def _t_ci(xs: list[float]) -> dict:
    """Mean +/- t_{0.975,n-1} * sd/sqrt(n). n=1 -> point (no interval)."""
    n = len(xs)
    m, sd = _mean_sd(xs)
    if n <= 1:
        return {"mean": m, "sd": 0.0, "n": n, "sem": 0.0, "half_width": None,
                "lo": m, "hi": m, "t_crit": None}
    sem = sd / math.sqrt(n)
    t = T_975.get(n - 1, 1.96)
    hw = t * sem
    return {"mean": m, "sd": sd, "n": n, "sem": sem, "half_width": hw,
            "lo": m - hw, "hi": m + hw, "t_crit": t}


def _derive_flr_at(d: dict, cap: int) -> float | None:
    """Fraction of samples whose output_tokens > cap (would have truncated at `cap`).
    Lets one mt=4096 run report the implied mt=3072 finish-length within Option-B."""
    ps = [r for r in d.get("per_sample", []) if r.get("output_tokens") is not None]
    if not ps:
        return None
    return sum(1 for r in ps if r["output_tokens"] > cap) / len(ps)


def _flr_health(reps: list[dict]) -> dict:
    """Cap-artifact health: finish_length_rate at 4096 vs the back-derived 3072 rate.
    The cap-release proof = 4096 rate << 3072 rate, with the 3072 rate ~ #631's 13%."""
    flr4096 = [d.get("finish_length_rate") for d in reps if d.get("finish_length_rate") is not None]
    der3072 = [v for d in reps if (v := _derive_flr_at(d, 3072)) is not None]
    der2048 = [d.get("finish_length_rate_at_2048") for d in reps
               if d.get("finish_length_rate_at_2048") is not None]
    return {
        "finish_length_rate_4096_mean": (sum(flr4096) / len(flr4096)) if flr4096 else None,
        "finish_length_rate_4096_per_seed": flr4096,
        "implied_finish_length_rate_3072_mean": (sum(der3072) / len(der3072)) if der3072 else None,
        "implied_finish_length_rate_3072_per_seed": der3072,
        "finish_length_rate_2048_mean": (sum(der2048) / len(der2048)) if der2048 else None,
        "ctok_p50_per_seed": [d.get("completion_tokens_p50") for d in reps],
        "ctok_p95_per_seed": [d.get("completion_tokens_p95") for d in reps],
        "ctok_max_per_seed": [d.get("completion_tokens_max") for d in reps],
        "stop_reason_counts_per_seed": [d.get("stop_reason_counts") for d in reps],
        "anchor_3072_631": FLR_3072_ANCHOR,
        "healthy_target": FLR_HEALTHY_TARGET,
    }


def analyze(results: dict[tuple[str, int], dict]) -> dict:
    out: dict = {"per_run": {}, "modes": {}}
    for (mode, seed), d in sorted(results.items()):
        out["per_run"][f"{mode}_s{seed}"] = {
            "accuracy": d["accuracy"], "n_correct": d["n_correct"],
            "n_scored": d["n_scored"], "n_error": d["n_error"],
            "n_empty": d.get("n_empty"), "finish_length_rate": d.get("finish_length_rate"),
        }
    for mode in ALL_MODES:
        seeds_done = sorted(s for (m, s) in results if m == mode)
        reps = [results[(mode, s)] for s in seeds_done]
        if not reps:
            continue
        accs = [d["accuracy"] for d in reps]
        ci = _t_ci(accs)
        base = BASE_SAMPLED if mode == "sampled" else BASE_GREEDY
        pct = {"pct_of_base_mean": ci["mean"] / base,
               "pct_of_base_lo": (ci["lo"] / base) if ci["lo"] is not None else None,
               "pct_of_base_hi": (ci["hi"] / base) if ci["hi"] is not None else None,
               "base_ref": base}
        if mode == "sampled":  # spec-vs-AR-body gap on the Reading-A axis (advisor #481 read)
            pct["pct_of_ar_body_mean"] = ci["mean"] / AR_BODY_SAMPLED
            pct["pct_of_ar_body_lo"] = (ci["lo"] / AR_BODY_SAMPLED) if ci["lo"] is not None else None
            pct["pct_of_ar_body_hi"] = (ci["hi"] / AR_BODY_SAMPLED) if ci["hi"] is not None else None
            pct["ar_body_ref"] = AR_BODY_SAMPLED
        out["modes"][mode] = {
            "n_seeds": len(reps), "seeds": seeds_done,
            "accuracy_per_seed": accs,
            "accuracy_ci": ci,
            "pct_of_base": pct,
            "n_correct_per_seed": [d["n_correct"] for d in reps],
            "n_scored_per_seed": [d["n_scored"] for d in reps],
            "n_empty_per_seed": [d.get("n_empty") for d in reps],
            "n_error_per_seed": [d["n_error"] for d in reps],
            "finish_length": _flr_health(reps),
        }
    return out


def verdict(analysis: dict) -> dict:
    """Reading-A GPQA verdict on the SAMPLED axis vs the 0.4864 bar (CI-based)."""
    s = analysis["modes"].get("sampled")
    g = analysis["modes"].get("greedy")
    res: dict = {"bar": BAR, "base_sampled": BASE_SAMPLED, "base_greedy": BASE_GREEDY}

    if not s:
        res["VERDICT"] = "PENDING_SAMPLED"
        res["headline"] = "No sampled seeds complete yet."
        return res

    ci = s["accuracy_ci"]
    lo, hi, mean = ci["lo"], ci["hi"], ci["mean"]
    n = ci["n"]
    res["sampled_mean"] = mean
    res["sampled_ci_lo"] = lo
    res["sampled_ci_hi"] = hi
    res["sampled_n_seeds"] = n
    res["pct_of_base_sampled"] = s["pct_of_base"]["pct_of_base_mean"]
    res["pct_of_base_sampled_lo"] = s["pct_of_base"]["pct_of_base_lo"]
    res["pct_of_base_sampled_hi"] = s["pct_of_base"]["pct_of_base_hi"]
    res["pct_of_ar_body_sampled"] = s["pct_of_base"].get("pct_of_ar_body_mean")
    res["pct_of_ar_body_sampled_lo"] = s["pct_of_base"].get("pct_of_ar_body_lo")
    res["pct_of_ar_body_sampled_hi"] = s["pct_of_base"].get("pct_of_ar_body_hi")
    res["ar_body_sampled_ref"] = AR_BODY_SAMPLED

    if g:
        res["greedy_mean"] = g["accuracy_ci"]["mean"]
        res["greedy_ci_lo"] = g["accuracy_ci"]["lo"]
        res["greedy_ci_hi"] = g["accuracy_ci"]["hi"]
        res["greedy_n_seeds"] = g["accuracy_ci"]["n"]
        res["pct_of_base_greedy"] = g["pct_of_base"]["pct_of_base_mean"]

    # health (cap-artifact) read from the GREEDY arm (gate axis) if present, else sampled
    health_src = g or s
    fl = health_src["finish_length"]
    res["finish_length_at_4096"] = fl["finish_length_rate_4096_mean"]
    res["implied_finish_length_at_3072"] = fl["implied_finish_length_rate_3072_mean"]
    cap_released = (fl["finish_length_rate_4096_mean"] is not None
                   and fl["finish_length_rate_4096_mean"] < FLR_HEALTHY_TARGET)
    res["cap_released_healthy"] = bool(cap_released)

    if n <= 1:
        res["VERDICT"] = "PENDING_CI"
        res["headline"] = (f"Sampled point estimate {mean:.4f} ({100*mean/BASE_SAMPLED:.1f}% of base) "
                           f"on n=1 seed -- need >=3 seeds for a CI verdict.")
        return res

    if hi < BAR:
        v = "READING_A_GPQA_FAILS"
        head = (f"Option-B sampled GPQA-D {mean:.4f} (95% CI [{lo:.4f},{hi:.4f}], n={n}) -- "
                f"CI UPPER {hi:.4f} < bar {BAR:.4f}. {100*mean/BASE_SAMPLED:.1f}% of base "
                f"{BASE_SAMPLED}. Reading-A GPQA leg FAILS robustly.")
    elif lo >= BAR:
        v = "READING_A_GPQA_PASSES"
        head = (f"Option-B sampled GPQA-D {mean:.4f} (95% CI [{lo:.4f},{hi:.4f}], n={n}) -- "
                f"CI LOWER {lo:.4f} >= bar {BAR:.4f}. {100*mean/BASE_SAMPLED:.1f}% of base. "
                f"Reading-A GPQA leg PASSES.")
    else:
        v = "READING_A_GPQA_KNIFE_EDGE"
        head = (f"Option-B sampled GPQA-D {mean:.4f} (95% CI [{lo:.4f},{hi:.4f}], n={n}) "
                f"STRADDLES bar {BAR:.4f}. {100*mean/BASE_SAMPLED:.1f}% of base. Reading-A GPQA "
                f"leg is a knife-edge -- more seeds needed to resolve.")
    res["VERDICT"] = v
    res["headline"] = head
    return res


def peak_gpu_gb() -> float | None:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, text=True, capture_output=True)
        return round(int(r.stdout.strip().splitlines()[0]) / 1024.0, 2)
    except Exception:
        return None


def _server_live() -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
            return b"gemma-4-e4b-it" in r.read()
    except Exception:
        return False


def _start_wandb_keepalive(run, interval: float = 60.0) -> None:
    """Daemon heartbeat so the W&B backend doesn't false-flag the run 'crashed' during the
    ~45-min run_one() subprocess.run() block, which logs nothing to W&B for the whole eval.
    subprocess.run releases the GIL while waiting, so this thread keeps the run actively
    logging. Pure liveness signal -- never touches eval/analysis/accumulation; exits quietly
    once logging raises (e.g. after the run is finished)."""
    if run is None:
        return

    def _beat() -> None:
        t0 = time.time()
        while True:
            time.sleep(interval)
            try:
                run.log({"heartbeat/alive": 1, "heartbeat/uptime_s": round(time.time() - t0, 1)})
            except Exception:
                return

    threading.Thread(target=_beat, name="wandb-keepalive", daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, run ONE conc=1 greedy eval limited to N items, then exit")
    ap.add_argument("--seeds", default=None, help="comma list to override SEEDS (subset)")
    ap.add_argument("--modes", default=None,
                    help="comma list to override the run-loop MODES (e.g. 'sampled' for the "
                         "sampled-only toward-10 extension; analysis still spans both modes)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="no server, no evals: recompute gate_summary.json + push the W&B "
                         "summary from the result JSONs already on disk, then exit")
    ap.add_argument("--final", action="store_true",
                    help="with --analyze-only: finalize (log artifact, status=completed, "
                         "finish the W&B run)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    global SEEDS, MODES
    if args.seeds:
        SEEDS = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.modes:
        MODES = [m.strip() for m in args.modes.split(",") if m.strip()]

    if not args.analyze_only and not _server_live():
        print(f"[gate] FATAL: no live server with gemma-4-e4b-it at {BASE_URL}. "
              f"Start serve_spec.py first.", file=sys.stderr, flush=True)
        return 4

    if args.smoke:
        out = out_path("greedy", SEEDS[0])
        smoke_out = RES / "_smoke.json"
        cmd = _eval_cmd("greedy", SEEDS[0], smoke_out, limit=args.smoke)
        t0 = time.time()
        subprocess.run(cmd, check=True)
        dt = time.time() - t0
        d = json.loads(smoke_out.read_text())
        ne = d.get("n_empty") or 0
        parsed = sum(1 for r in d.get("per_sample", []) if r.get("answer"))
        ok = (d["n_scored"] >= 1) and (ne == 0) and (parsed >= max(1, args.smoke - 1))
        per_item = dt / max(1, args.smoke)
        print(f"[smoke] scored={d['n_scored']} parsed={parsed} empty={ne} acc={d['accuracy']:.4f} "
              f"flr={d.get('finish_length_rate')} ctok_p50={d.get('completion_tokens_p50')} "
              f"ctok_max={d.get('completion_tokens_max')} stop={d.get('stop_reason_counts')} "
              f"dt={dt:.0f}s per_item={per_item:.1f}s "
              f"=> est_full_seed={per_item*198/60:.0f}min "
              f"-> {'COHERENT' if ok else 'DEGENERATE/SUSPECT'}", flush=True)
        smoke_out.unlink(missing_ok=True)
        (RES / "_inspect_logs").exists() and None
        return 0 if ok else 3

    # ---- W&B: resume-safe ----
    run = None
    if not args.no_wandb:
        idfile = HERE / "wandb_run_id.txt"
        if idfile.exists():
            os.environ["WANDB_RUN_ID"] = idfile.read_text().strip()
            os.environ["WANDB_RESUME"] = "allow"
        from scripts import wandb_logging
        run = wandb_logging.init_wandb_run(
            job_type="gpqa-gate-readingA-4096", agent="kanna",
            name="kanna/gpqa-gate-readingA-4096",
            group="gpqa-gate-readingA-4096-kanna",
            notes=("PR #643: Option-B (int4_g128_lmhead + Gemma4-MTP K=7, BI=1) GPQA-Diamond "
                   "Reading-A verdict at the #631 clean gate point (dev307, --max-connections 1). "
                   "mt=4096 (cap-release vs #631's 3072). Greedy (strict gate) + sampled (lewtun "
                   "#31 temp1.0/top_p0.95/top_k64, the Reading-A axis), multi-seed mean +/- t-CI. "
                   "pct-of-base sampled vs 0.5404 (ubel #628 ilg4z6e9); bar 0.4864. LOCAL A10G; "
                   "analysis_only, official_tps=0, NO FIRE."),
            tags=["quality-gate", "int4_g128_lmhead", "mtp-k7", "option-b", "analysis-only",
                  "pr-643", "gpqa", "gpqa-diamond", "reading-a", "dev307", "conc1",
                  "vllm-0.22.1rc1.dev307", "max-tokens-4096"],
            config={"pr": 643, "analysis_only": True, "official_tps": 0,
                    "engine": ENGINE, "checkpoint": "int4_g128_lmhead",
                    "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
                    "num_speculative_tokens": 7, "batch_invariant": 1,
                    "model_len": MODEL_LEN, "max_num_seqs": 16, "max_tokens": MAX_TOKENS,
                    "min_tokens_guard": MIN_TOKENS, "max_connections": CONC,
                    "task": TASK, "seeds": SEEDS, "modes": ALL_MODES,
                    "base_sampled": BASE_SAMPLED, "base_greedy": BASE_GREEDY, "bar": BAR,
                    "ar_body_sampled": AR_BODY_SAMPLED, "ar_body_run": "ubel-638",
                    "gate_point": "dev307_conc1_631", "631_run": "zk9zffp5",
                    "base_sampled_run": "ilg4z6e9", "base_greedy_run": "g3cig1xo",
                    "flr_3072_anchor_631": FLR_3072_ANCHOR},
        )
        if run is not None:
            idfile.write_text(str(getattr(run, "id", "")))
            print(f"[gate] wandb run live: {getattr(run,'id',None)}", flush=True)
            run.summary["status"] = "running"
            _start_wandb_keepalive(run)  # keep the run heartbeating through long silent evals

    # ---- analyze-only: recompute the summary from on-disk JSONs, no server/evals ----
    if args.analyze_only:
        n_done = len(_load_all_results())
        _write_summary(run, n_done, final=args.final)
        if run is not None:
            run.summary["status"] = "completed" if args.final else "running"
            if args.final:
                from scripts import wandb_logging
                wandb_logging.finish_wandb(run)
                print(f"[gate] wandb finished run_id={getattr(run,'id',None)}", flush=True)
        print(f"[gate] analyze-only done over {n_done} on-disk cells (final={args.final})",
              flush=True)
        return 0

    # ---- run cells (idempotent): seed-outer so first cell is greedy s0 (health read) ----
    # step starts at the count of cells already on disk so global_step stays monotonic when a
    # resumed invocation appends new seeds to the same W&B run.
    step = len(_load_all_results())
    for seed in SEEDS:
        for mode in MODES:
            d = run_one(mode, seed)
            if run is not None:
                run.log({"global_step": step, "cell/seed": seed,
                         "cell/mode_sampled": int(mode == "sampled"),
                         "cell/accuracy": d["accuracy"], "cell/n_correct": d["n_correct"],
                         "cell/n_empty": d.get("n_empty") or 0, "cell/n_error": d["n_error"],
                         "cell/finish_length_rate": d.get("finish_length_rate")})
            step += 1
            # incremental analysis after each cell (so a partial sweep still yields a read);
            # _write_summary globs ALL on-disk seeds, not just this invocation's.
            _write_summary(run, step, final=False)

    _write_summary(run, step, final=True)
    if run is not None:
        from scripts import wandb_logging
        run.summary["status"] = "completed"
        wandb_logging.finish_wandb(run)
        print(f"[gate] wandb finished run_id={getattr(run,'id',None)}", flush=True)
    return 0


def _write_summary(run, step, *, final: bool) -> None:
    results = _load_all_results()  # ALL seeds accumulated on disk, not just this invocation
    analysis = analyze(results)
    vd = verdict(analysis)
    detail = {
        "pr": 643, "engine": ENGINE, "checkpoint": "/workspace/gemma_build/int4_g128_lmhead",
        "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
        "num_speculative_tokens": 7, "batch_invariant": 1,
        "model_len": MODEL_LEN, "max_num_seqs": 16, "max_tokens": MAX_TOKENS,
        "min_tokens_guard": MIN_TOKENS, "max_connections": CONC, "task": TASK,
        "base_sampled": BASE_SAMPLED, "base_greedy": BASE_GREEDY, "bar": BAR,
        "gate_point": "dev307_conc1 (#631 zk9zffp5)",
        "n_dataset": (next(iter(results.values()))["n_dataset"] if results else None),
        "analysis": analysis, "verdict": vd,
        "peak_gpu_gb": peak_gpu_gb(), "analysis_only": True, "official_tps": 0,
        "final": final,
    }
    (HERE / "gate_summary.json").write_text(json.dumps(detail, indent=2, default=str))
    print(("FINAL_" if final else "PARTIAL_") + "GATE_SUMMARY "
          + json.dumps(vd, default=str), flush=True)
    if run is not None:
        ko = {"VERDICT": vd.get("VERDICT"),
              "sampled_mean": vd.get("sampled_mean"),
              "sampled_ci_lo": vd.get("sampled_ci_lo"), "sampled_ci_hi": vd.get("sampled_ci_hi"),
              "sampled_n_seeds": vd.get("sampled_n_seeds"),
              "greedy_mean": vd.get("greedy_mean"), "greedy_n_seeds": vd.get("greedy_n_seeds"),
              "pct_of_base_sampled": vd.get("pct_of_base_sampled"),
              "pct_of_base_sampled_lo": vd.get("pct_of_base_sampled_lo"),
              "pct_of_base_sampled_hi": vd.get("pct_of_base_sampled_hi"),
              "pct_of_ar_body_sampled": vd.get("pct_of_ar_body_sampled"),
              "pct_of_ar_body_sampled_lo": vd.get("pct_of_ar_body_sampled_lo"),
              "pct_of_ar_body_sampled_hi": vd.get("pct_of_ar_body_sampled_hi"),
              "pct_of_base_greedy": vd.get("pct_of_base_greedy"),
              "finish_length_at_4096": vd.get("finish_length_at_4096"),
              "implied_finish_length_at_3072": vd.get("implied_finish_length_at_3072"),
              "cap_released_healthy": vd.get("cap_released_healthy"),
              "peak_gpu_gb": detail["peak_gpu_gb"], "analysis_only": True, "official_tps": 0}
        for k, v in ko.items():
            run.summary[k] = v
        if final:
            from scripts import wandb_logging
            wandb_logging.log_json_artifact(run, name="gpqa_gate_readingA_4096_detail",
                                            artifact_type="quality-eval", data=detail)


if __name__ == "__main__":
    raise SystemExit(main())
