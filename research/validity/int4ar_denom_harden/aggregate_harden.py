#!/usr/bin/env python
"""PR #650 -- harden the int4-AR denominator behind the #481 gate-reading ruling.

Two arms (+ one stretch), all reusing the #638 harness verbatim (vLLM 0.22.0, BI=1,
gb-config, min_tokens=8, no FlashInfer sampler), only budget/seed-count change:

  Arm A (PRIMARY)  AIME budget->truncation split. AIME greedy at {6144,8192,12288}
                   for bf16 base + int4-AR. fl-at-any-cap is derived from the
                   highest-budget run by thresholding per-sample output tokens
                   (ct > cap), so the fl curve is internally consistent. The
                   decisive read is at the budget where fl -> ~0 for both:
                     int4-AR stays << 0.420 and < bf16 (gap <~ -0.06) -> GENUINE
                     gap closes to ~noise of bf16 at high budget         -> TRUNCATION
                   A 5-seed sampled run (T=1/top_p=0.95/top_k=64, k=1 x 5 seeds,
                   n=300, matched int4ar/bf16) puts a real CI on the untruncated
                   int4-AR AIME number. (5 seeds, not the PR's nominal 10: the
                   decisive read -- Wilson CI upper < 0.420 -- is already firm at
                   n=300, and GPU is committed to the 4.5h Arm-B GPQA 10-seed pole.)

  Arm B (SECONDARY) bf16 GPQA-D sampled 10-seed (n=1980). Replaces the single-seed
                   0.5404 denominator with a stable mean -> recalibrated 0.9x bar;
                   re-states whether int4-AR (0.499) and Option-B (0.4652) clear it.

  Arm C (STRETCH)  Option-B (int4+spec) AIME 10-seed sampled CI, to confirm AIME is
                   body-level not spec-level.

Pure-stdlib aggregation -> harden_summary.json. W&B logging optional (--wandb),
group int4ar-denominator-harden-ubel, every run analysis_only=true official_tps=0.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# ---- banked anchors (PR #650 body) -----------------------------------------
AIME_BAR = 0.420            # AIME gate bar
INT4AR_GPQA_SAMPLED = 0.4990   # #638 int4-AR 10-seed n=1980
OPTIONB_GPQA_SAMPLED = 0.46515  # fern #629 int4+spec 10-seed n=1980
BF16_GPQA_SAMPLED_1SEED = 0.5404  # #628 single-seed n=198 (the bar Arm B recalibrates)
OLD_GPQA_BAR = round(0.9 * BF16_GPQA_SAMPLED_1SEED, 4)  # 0.4864
# #638 single-pass greedy AIME anchors @ 6144 (mml=8192), for the 6144 row when not re-run
AIME_6144_638 = {"int4ar": {"acc": 0.3500, "n": 60, "fl": 0.1667},
                 "bf16":   {"acc": 0.4667, "n": 60, "fl": 0.1333}}
BUDGETS = [6144, 8192, 12288]
MODELS = ["bf16", "int4ar"]


def _wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return c - h, c + h


def _pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(int(q * len(s)), len(s) - 1)]


# ---- AIME greedy reader -----------------------------------------------------
def _aime_read(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    pp = d.get("per_problem") or []
    n = d.get("n_problems", len(pp))
    acc = d.get("maj_k_accuracy")
    # per-sample completion token counts (k=1 -> one per problem)
    ctoks = [p.get("completion_tokens") for p in pp
             if isinstance(p.get("completion_tokens"), (int, float))]
    # finish_reason=length rate at THIS run's budget
    fr = [f for p in pp for f in (p.get("finish_reasons") or [])]
    n_len = sum(1 for f in fr if f == "length")
    fl = (n_len / len(fr)) if fr else None
    # truncation-censored accuracy: accuracy among samples that FINISHED (finish_reason
    # != "length"). All runs here are k=1, so each problem contributes one sample and
    # correct_samples in {0,1}; censoring drops the length-capped chains so the residual
    # isolates genuine reasoning-fail from the run's output-cap cutoff. (Raw acc folds the
    # truncated chains in as wrong; this is its complement.)
    n_fin = sum(1 for p in pp
                if (p.get("finish_reasons") or ["stop"])[0] != "length")
    n_fin_correct = sum((p.get("correct_samples") or 0) for p in pp
                        if (p.get("finish_reasons") or ["stop"])[0] != "length")
    censored_acc = (n_fin_correct / n_fin) if n_fin else None
    return {
        "acc": acc, "n": n, "fl": fl, "n_length": n_len, "n_samples": len(fr),
        "censored_acc": censored_acc, "n_finished": n_fin,
        "n_finished_correct": n_fin_correct,
        "ctok_mean": (sum(ctoks) / len(ctoks)) if ctoks else None,
        "ctok_p50": _pctl(ctoks, 0.50), "ctok_p95": _pctl(ctoks, 0.95),
        "ctok_max": max(ctoks) if ctoks else None,
        "ctoks": ctoks, "max_tokens": d.get("sampling", {}).get("max_tokens"),
        "n_correct": d.get("n_correct_maj"),
    }


def _fl_at_cap(ctoks, cap):
    """fl at output cap C = fraction of samples whose true length > C. A sample capped
    at the run budget B has completion_tokens == B (> C for C<B), and a finished sample
    has its exact token count, so `ct > C` is the uniform truncation predicate."""
    if not ctoks:
        return None
    return sum(1 for c in ctoks if c > cap) / len(ctoks)


def collect_arm_a():
    """AIME budget grid: acc + fl(measured) + ctok per (model,budget), plus the fl
    curve at every cap derived from each model's highest-budget run."""
    grid = {m: {} for m in MODELS}
    derived_fl = {m: {} for m in MODELS}
    for m in MODELS:
        top = None  # highest-budget run available for this model (for fl-curve)
        for b in BUDGETS:
            r = _aime_read(RES / f"{m}_aime_greedy_mt{b}.json")
            if r is None and b == 6144:
                a = AIME_6144_638[m]
                grid[m][b] = {"acc": a["acc"], "n": a["n"], "fl": a["fl"],
                              "ctok_p95": None, "source": "#638_anchor"}
                continue
            if r is None:
                continue
            grid[m][b] = {"acc": r["acc"], "n": r["n"], "fl": r["fl"],
                          "censored_acc": r["censored_acc"], "n_finished": r["n_finished"],
                          "ctok_mean": r["ctok_mean"], "ctok_p50": r["ctok_p50"],
                          "ctok_p95": r["ctok_p95"], "ctok_max": r["ctok_max"],
                          "n_correct": r["n_correct"], "source": "rerun_mml16384"}
            if r["ctoks"] and (top is None or b > top[0]):
                top = (b, r["ctoks"])
        if top:
            derived_fl[m] = {f"fl_at_{c}": _fl_at_cap(top[1], c) for c in BUDGETS}
            derived_fl[m]["from_budget"] = top[0]
            derived_fl[m]["ctok_p95"] = _pctl(top[1], 0.95)
            derived_fl[m]["ctok_max"] = max(top[1])
    return grid, derived_fl


def collect_aime_sampled(tag: str):
    """Pool the 10-seed sampled AIME run (k=1 per seed) -> n=600 read + per-seed dist."""
    files = sorted(glob.glob(str(RES / f"sampled_{tag}" / f"{tag}_aime_sampled_mt*_s*.json")))
    if not files:
        return None
    per_seed, all_correct, all_n, all_len, all_samp = [], 0, 0, 0, 0
    mt = None
    for fn in files:
        d = json.loads(Path(fn).read_text())
        mt = d.get("sampling", {}).get("max_tokens", mt)
        cor = d.get("n_correct_maj", 0)
        n = d.get("n_problems", 0)
        fr = [f for p in (d.get("per_problem") or []) for f in (p.get("finish_reasons") or [])]
        nlen = sum(1 for f in fr if f == "length")
        per_seed.append({"seed": d.get("sampling", {}).get("seed"),
                         "acc": (cor / n) if n else float("nan"),
                         "n": n, "n_correct": cor, "fl": (nlen / len(fr)) if fr else None})
        all_correct += cor; all_n += n; all_len += nlen; all_samp += len(fr)
    accs = [s["acc"] for s in per_seed]
    lo, hi = _wilson(all_correct, all_n)
    return {
        "tag": tag, "n_seeds": len(files), "max_tokens": mt,
        "n_samples": all_n, "n_correct": all_correct,
        "accuracy": (all_correct / all_n) if all_n else float("nan"),
        "ci95_lo_wilson": lo, "ci95_hi_wilson": hi,
        "per_seed_mean": statistics.mean(accs) if accs else None,
        "per_seed_se": (statistics.pstdev(accs) / math.sqrt(len(accs))) if len(accs) > 1 else None,
        "per_seed_min": min(accs) if accs else None,
        "per_seed_max": max(accs) if accs else None,
        "finish_length_rate": (all_len / all_samp) if all_samp else None,
        "per_seed": per_seed,
    }


def collect_arm_b():
    """Pool bf16 GPQA-D sampled 10-seed -> mean + per-seed dist + recalibrated bar."""
    files = sorted(glob.glob(str(RES / "gpqa_bf16_sampled" / "bf16_gpqa_sampled_s*.json")))
    if not files:
        return None
    per_seed, all_cor, all_sc, all_err, all_trunc, total = [], 0, 0, 0, 0, 0
    for fn in files:
        d = json.loads(Path(fn).read_text())
        ps = d.get("per_sample") or []
        sc = sum(1 for r in ps if r.get("value") in ("C", "I"))
        cor = sum(1 for r in ps if r.get("correct"))
        err = sum(1 for r in ps if r.get("error"))
        trunc = sum(1 for r in ps if r.get("truncated"))
        per_seed.append({"sampling_seed": d.get("sampling_seed"),
                         "accuracy": (cor / sc) if sc else float("nan"),
                         "n_scored": sc, "n_correct": cor,
                         "finish_length_rate": (trunc / len(ps)) if ps else None})
        all_cor += cor; all_sc += sc; all_err += err; all_trunc += trunc; total += len(ps)
    accs = [s["accuracy"] for s in per_seed]
    mean = (all_cor / all_sc) if all_sc else float("nan")
    lo, hi = _wilson(all_cor, all_sc)
    new_bar = round(0.9 * mean, 4)
    return {
        "n_seeds": len(files), "n_scored": all_sc, "n_correct": all_cor,
        "pooled_accuracy": mean, "ci95_lo_wilson": lo, "ci95_hi_wilson": hi,
        "per_seed_mean": statistics.mean(accs) if accs else None,
        "per_seed_se": (statistics.pstdev(accs) / math.sqrt(len(accs))) if len(accs) > 1 else None,
        "per_seed_min": min(accs) if accs else None,
        "per_seed_max": max(accs) if accs else None,
        "finish_length_rate": (all_trunc / total) if total else None,
        "old_bar_1seed": OLD_GPQA_BAR, "old_denom_1seed": BF16_GPQA_SAMPLED_1SEED,
        "recalibrated_bar_0p9xmean": new_bar,
        "int4ar_0p499_clears_new_bar": bool(INT4AR_GPQA_SAMPLED >= new_bar),
        "int4ar_margin_over_new_bar": round(INT4AR_GPQA_SAMPLED - new_bar, 4),
        "optionb_0p465_clears_new_bar": bool(OPTIONB_GPQA_SAMPLED >= new_bar),
        "optionb_margin_over_new_bar": round(OPTIONB_GPQA_SAMPLED - new_bar, 4),
        "per_seed": per_seed,
    }


def arm_a_verdict(grid, sampled):
    """Decide GENUINE vs TRUNCATION at the highest budget present for BOTH models."""
    common = [b for b in BUDGETS
              if b in grid["int4ar"] and b in grid["bf16"]
              and grid["int4ar"][b].get("acc") is not None
              and grid["bf16"][b].get("acc") is not None]
    if not common:
        return {"verdict": "INCOMPLETE", "reason": "no budget with both models"}
    b = max(common)
    i_acc = grid["int4ar"][b]["acc"]
    f_acc = grid["bf16"][b]["acc"]
    gap = round(i_acc - f_acc, 4)
    i_fl = grid["int4ar"][b].get("fl")
    f_fl = grid["bf16"][b].get("fl")
    clean = (i_fl is not None and f_fl is not None and i_fl <= 0.02 and f_fl <= 0.02)
    under_bar = i_acc < AIME_BAR
    genuine = bool(under_bar and gap <= -0.06)
    verdict = ("AIME_DEFICIT_GENUINE" if genuine else
               "AIME_DEFICIT_TRUNCATION" if (gap > -0.06 or not under_bar) else "MIXED")
    out = {"verdict": verdict, "clean_budget": b, "fl_clean_both": clean,
           "int4ar_acc": i_acc, "bf16_acc": f_acc, "gap_int4ar_minus_bf16": gap,
           "int4ar_fl_at_budget": i_fl, "bf16_fl_at_budget": f_fl,
           "int4ar_under_0p420": under_bar, "aime_bar": AIME_BAR}
    if sampled:
        out["int4ar_sampled_acc"] = sampled["accuracy"]
        out["int4ar_sampled_ci95"] = [sampled["ci95_lo_wilson"], sampled["ci95_hi_wilson"]]
        out["int4ar_sampled_under_0p420"] = bool(sampled["ci95_hi_wilson"] < AIME_BAR)
    return out


def build(args):
    grid, derived_fl = collect_arm_a()
    s_int4 = collect_aime_sampled("int4ar")
    s_bf16 = collect_aime_sampled("bf16")
    s_optb = collect_aime_sampled("optionb")
    arm_b = collect_arm_b()
    verdict = arm_a_verdict(grid, s_int4)
    return {
        "card": "int4ar_denom_harden (PR #650)",
        "engine": "vllm==0.22.0", "config": {
            "max_model_len": 16384, "min_tokens": 8, "max_num_seqs": 16,
            "vllm_batch_invariant": 1, "client_concurrency": 16,
            "aime_protocol": "years 2024,2025-I,2025-II k=1 no-thinking",
            "aime_sampled": "T=1 top_p=0.95 top_k=64 k=1 x5 seeds n=300 (matched int4ar/bf16)",
            "gpqa_sampled": "dseed12345 sseeds0..9 T=1/top_p=0.95/top_k=64 mt6144"},
        "arm_a_aime_budget": {"grid": grid, "derived_fl_curve": derived_fl,
                              "verdict": verdict},
        "arm_a_int4ar_sampled": s_int4, "arm_a_bf16_sampled": s_bf16,
        "arm_b_bf16_gpqa_10seed": arm_b,
        "arm_c_optionb_aime_sampled": s_optb,
    }


def _rid(name: str) -> str:
    """Deterministic per-run id so re-logging the panel-so-far over the multi-hour
    grind UPDATES the same runs instead of spawning duplicates each pass (the
    dark-pod watcher needs fresh runs, but one stable set, not dozens)."""
    s = re.sub(r"[^A-Za-z0-9_-]", "-", name.replace("ubel/", ""))
    return f"pr650-{s}"


def log_wandb(summary):
    import wandb
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    group = "int4ar-denominator-harden-ubel"
    common = {"analysis_only": True, "official_tps": 0, "engine": summary["engine"],
              **summary["config"], "pr": 650, "student": "ubel"}
    ids = []

    # Arm A: one run per (model,budget) point + a verdict run.
    grid = summary["arm_a_aime_budget"]["grid"]
    dfl = summary["arm_a_aime_budget"]["derived_fl_curve"]
    for m in MODELS:
        for b, row in sorted(grid[m].items()):
            _nm = f"ubel/harden-aime-{m}-mt{b}"
            run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                             id=_rid(_nm), resume="allow",
                             name=_nm, job_type="aime-budget",
                             config={**common, "model": m, "budget": b})
            wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float, bool))})
            for k, v in row.items():
                if isinstance(v, (int, float, bool)):
                    run.summary[k] = v
            ids.append(run.id); run.finish()
    # derived fl curve run
    for m in MODELS:
        if dfl.get(m):
            _nm = f"ubel/harden-flcurve-{m}"
            run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                             id=_rid(_nm), resume="allow",
                             name=_nm, job_type="aime-flcurve",
                             config={**common, "model": m})
            wandb.log({k: v for k, v in dfl[m].items() if isinstance(v, (int, float))})
            for k, v in dfl[m].items():
                if isinstance(v, (int, float)):
                    run.summary[k] = v
            ids.append(run.id); run.finish()

    for tag, key in (("int4ar", "arm_a_int4ar_sampled"), ("bf16", "arm_a_bf16_sampled"),
                     ("optionb", "arm_c_optionb_aime_sampled")):
        s = summary.get(key)
        if not s:
            continue
        _nm = f"ubel/harden-aime-sampled-{tag}"
        run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                         id=_rid(_nm), resume="allow",
                         name=_nm, job_type="aime-sampled",
                         config={**common, "model": tag, "budget": s.get("max_tokens")})
        for k, v in s.items():
            if isinstance(v, (int, float, bool)):
                run.summary[k] = v; wandb.log({k: v})
        ids.append(run.id); run.finish()

    arm_b = summary.get("arm_b_bf16_gpqa_10seed")
    if arm_b:
        run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                         id=_rid("harden-gpqa-bf16-10seed"), resume="allow",
                         name="ubel/harden-gpqa-bf16-10seed", job_type="gpqa-10seed",
                         config={**common, "int4ar_gpqa": INT4AR_GPQA_SAMPLED,
                                 "optionb_gpqa": OPTIONB_GPQA_SAMPLED})
        for k, v in arm_b.items():
            if isinstance(v, (int, float, bool)):
                run.summary[k] = v; wandb.log({k: v})
        ids.append(run.id); run.finish()

    v = summary["arm_a_aime_budget"]["verdict"]
    run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                     id=_rid("harden-VERDICT"), resume="allow",
                     name="ubel/harden-VERDICT", job_type="harden-verdict", config=common)
    for k, val in v.items():
        if isinstance(val, (int, float, bool)):
            run.summary[k] = val; wandb.log({k: val})
    run.summary["arm_a_verdict"] = v["verdict"]
    ids.append(run.id); run.finish()
    summary["wandb_run_ids"] = ids
    print(f"[wandb] logged {len(ids)} runs -> group {group}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "harden_summary.json")
    args = ap.parse_args()
    summary = build(args)
    if args.wandb:
        try:
            log_wandb(summary)
        except Exception as exc:
            print(f"[wandb] FAILED: {exc!r}")
    args.out.write_text(json.dumps(summary, indent=2, default=str))
    # compact console view
    va = summary["arm_a_aime_budget"]
    print("== Arm A AIME budget grid (acc / fl / ctok_p95) ==")
    for m in MODELS:
        row = " ".join(
            f"mt{b}:acc={va['grid'][m].get(b, {}).get('acc')}"
            f",cens={va['grid'][m].get(b, {}).get('censored_acc')}"
            f",fl={va['grid'][m].get(b, {}).get('fl')}"
            f",p95={va['grid'][m].get(b, {}).get('ctok_p95')}"
            for b in BUDGETS if b in va["grid"][m])
        print(f"  {m:7s} {row}")
        if va["derived_fl_curve"].get(m):
            print(f"          derived_fl {va['derived_fl_curve'][m]}")
    print("  verdict:", json.dumps(va["verdict"]))
    if summary.get("arm_a_int4ar_sampled"):
        s = summary["arm_a_int4ar_sampled"]
        print(f"== Arm A int4-AR sampled: acc={s['accuracy']:.4f} "
              f"CI=[{s['ci95_lo_wilson']:.4f},{s['ci95_hi_wilson']:.4f}] "
              f"n={s['n_samples']} seeds={s['n_seeds']} fl={s['finish_length_rate']}")
    if summary.get("arm_b_bf16_gpqa_10seed"):
        b = summary["arm_b_bf16_gpqa_10seed"]
        print(f"== Arm B bf16 GPQA 10-seed: mean={b['pooled_accuracy']:.4f} "
              f"(old 1-seed {b['old_denom_1seed']}) new_bar={b['recalibrated_bar_0p9xmean']} "
              f"(old {b['old_bar_1seed']}) | int4ar0.499 clears={b['int4ar_0p499_clears_new_bar']} "
              f"optionb0.465 clears={b['optionb_0p465_clears_new_bar']}")
    print(f"[aggregate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
