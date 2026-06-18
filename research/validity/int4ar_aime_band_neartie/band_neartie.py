"""PR #672 — int4 greedy-AIME multi-session band + near-tie argmax mechanism.

Closes out the int4-body AIME quality blocker with two co-headline deliverables,
reusing the #650/#668 greedy-AIME protocol (T=0, min_tokens=8, --no-thinking,
client-concurrency 16, seed 0, max_tokens 12288, full 60 = 2024 + 2025-I + 2025-II).

  1. Multi-session greedy-AIME band. >=4 FRESH int4-AR serve sessions (fresh
     process each; vary only the process epoch). Per-session maj@1, band
     [min,max], mean, 95% CI over sessions. Decision scalar
     ``int4_aime_band_upper95`` vs the 0.420 bar:
       BLOCKER_ROBUST    upper95 < 0.420  -> genuinely sub-bar, Option-B dead.
       BLOCKER_STRADDLES upper95 >= 0.420 -> measurement-fragile, reopen.
     bf16 >=2 sessions is the stability control (#668: 11/11 bit-exact).

  2. Near-tie argmax margin (the mechanism). Per int4 decode step, the
     argmax-vs-runner-up logprob gap (margin = chosen_lp - runnerup_lp, nats).
     Histogram + median + frac(<0.1), frac(<0.05). Tests whether per-problem
     near-tie density predicts that problem's cross-session answer instability
     (correlate per-problem near-tie density vs per-problem flip rate).

Margins are captured DURING the band sessions (one logprobs pass per session),
so the same runs feed both deliverables across the full 60 problems and all
sessions — strictly richer than the lost 11-problem #668 streams. An identity
guard (``session`` vs ``aime_eval.py`` on the same server) proves logprobs
capture does not move the greedy argmax on this stack (VLLM_BATCH_INVARIANT=1).

analysis_only: NO HF Job, NO submission, NO served-file change. live
int4_g128_lmhead @ 126.378 untouched. analysis_only=true / official_tps=0 are
logged as explicit W&B summary scalars (machine-checkable no-fire guard).

Subcommands
-----------
  selftest    no-GPU validation of the stats/aggregation math on synthetic data
  session     full-60 greedy AIME @12288 + per-token top-2 margins from a served
              endpoint -> <arm>_session<NN>.json   (needs a live server)
  aggregate   >=4 int4 + >=2 bf16 session files -> band stats + near-tie +
              density/flip correlation + first-divergence margins + verdict
  wandb       push band + near-tie + verdict to group int4ar-aime-band-neartie-ubel
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from research.downstream_quality_aime.aime_eval import (  # noqa: E402
    AIME_INSTRUCTION,
    extract_answer,
    load_aime,
)

BAR = 0.420  # = 0.9 * base AIME 0.4667 (PR #515 quality gate)
NEARTIE_THRESHOLDS = (0.20, 0.10, 0.05, 0.02, 0.01)


# --------------------------------------------------------------------------- #
# session: full-60 greedy AIME + per-token top-2 margins
# --------------------------------------------------------------------------- #
def greedy_logprobs(
    base_url: str,
    model: str,
    problem: str,
    *,
    max_tokens: int,
    min_tokens: int,
    seed: int,
    timeout_s: int,
    top_logprobs: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    """One greedy completion with logprobs. Matches the #650/#668 request EXACTLY
    (T=0, top_p=1.0, n=1, --no-thinking, min_tokens=8). logprobs only reads the
    already-computed logits under VLLM_BATCH_INVARIANT=1, so it does not move the
    argmax (proven by the identity guard)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{problem}\n\n{AIME_INSTRUCTION}"}],
        "n": 1,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "seed": seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def _margins_and_tokens(lp_content: list[dict[str, Any]]) -> tuple[list[str], list[float | None]]:
    """Parallel arrays: chosen token strings + margin (chosen_lp - runnerup_lp, nats).

    margin None means no distinct runner-up was returned (treated as fully
    confident / not near-tie downstream)."""
    toks: list[str] = []
    margins: list[float | None] = []
    for e in lp_content:
        chosen_t = e["token"]
        chosen_lp = e["logprob"]
        runner = None
        for alt in e.get("top_logprobs", []):
            if alt["token"] != chosen_t:
                runner = alt["logprob"]
                break
        toks.append(chosen_t)
        margins.append(round(chosen_lp - runner, 4) if runner is not None else None)
    return toks, margins


def cmd_session(args: argparse.Namespace) -> int:
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    problems = load_aime(years, limit=args.limit)
    print(f"[session:{args.arm}#{args.session_idx}] loaded {len(problems)} problems "
          f"years={years} conc={args.client_concurrency}", flush=True)

    def _one(idx: int, prob: dict[str, Any]) -> tuple[int, dict[str, Any], str]:
        t0 = time.time()
        resp = greedy_logprobs(
            args.base_url, args.model, prob["problem"],
            max_tokens=args.max_tokens, min_tokens=args.min_tokens, seed=args.seed,
            timeout_s=args.request_timeout_s, top_logprobs=args.top_logprobs,
            enable_thinking=args.enable_thinking,
        )
        ch = resp["choices"][0]
        text = ch.get("message", {}).get("content") or ""
        toks, margins = _margins_and_tokens(ch.get("logprobs", {}).get("content") or [])
        ans = extract_answer(text)
        gold = prob["answer"]
        correct = ans is not None and ans == gold
        rec = {
            "id": prob["id"], "year": prob["year"], "gold": gold, "answer": ans,
            "correct": correct, "finish_reason": ch.get("finish_reason"),
            "n_tokens": len(toks), "text_chars": len(text),
            "tok": toks, "mg": margins,
            "text": text if args.save_text else None,
        }
        line = (f"[session:{args.arm}#{args.session_idx}] {idx+1}/{len(problems)} id={prob['id']} "
                f"ans={ans} gold={gold} {'OK' if correct else 'x'} ntok={len(toks)} "
                f"finish={ch.get('finish_reason')} {time.time()-t0:.1f}s")
        return idx, rec, line

    t0 = time.time()
    results: list[dict[str, Any] | None] = [None] * len(problems)
    if args.client_concurrency <= 1:
        for idx, prob in enumerate(problems):
            i, rec, line = _one(idx, prob)
            results[i] = rec
            print(line, flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.client_concurrency) as ex:
            futs = {ex.submit(_one, idx, prob): idx for idx, prob in enumerate(problems)}
            done = 0
            for fut in as_completed(futs):
                i, rec, line = fut.result()
                results[i] = rec
                done += 1
                print(f"{line}  [{done}/{len(problems)} returned]", flush=True)

    per = [r for r in results if r is not None]
    n = len(per)
    n_correct = sum(int(r["correct"]) for r in per)
    extract_fail = sum(1 for r in per if r["answer"] is None)
    out = {
        "arm": args.arm, "session_idx": args.session_idx, "base_url": args.base_url,
        "model": args.model, "years": years, "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens, "seed": args.seed,
        "client_concurrency": args.client_concurrency, "top_logprobs": args.top_logprobs,
        "enable_thinking": args.enable_thinking,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "n": n, "n_correct": n_correct, "accuracy": n_correct / n if n else 0.0,
        "extract_fail": extract_fail, "wall_s": time.time() - t0,
        "per_problem": {r["id"]: r for r in per},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"[session:{args.arm}#{args.session_idx}] wrote {args.out} "
          f"acc={out['accuracy']:.4f} ({n_correct}/{n}) extract_fail={extract_fail} "
          f"wall={out['wall_s']:.0f}s", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
# Student-t 0.975 critical values by dof (small-sample two-sided 95%).
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
         20: 2.086, 30: 2.042}


def t975(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    if dof in _T975:
        return _T975[dof]
    keys = sorted(_T975)
    if dof > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k <= dof)
    hi = min(k for k in keys if k >= dof)
    if lo == hi:
        return _T975[lo]
    f = (dof - lo) / (hi - lo)
    return _T975[lo] + f * (_T975[hi] - _T975[lo])


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def bootstrap_mean_ci(vals: list[float], n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return lo, hi


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def _rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return pearson(_rank(xs), _rank(ys))


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def _load_sessions(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for p in sorted(paths):
        out.append(json.loads(p.read_text()))
    return out


def _band_stats(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    accs = [s["accuracy"] for s in sessions]
    n = len(accs)
    mean = sum(accs) / n if n else float("nan")
    sd = statistics.stdev(accs) if n >= 2 else 0.0
    se = sd / math.sqrt(n) if n else float("nan")
    t = t975(n - 1) if n >= 2 else float("nan")
    ci_lo = mean - t * se if n >= 2 else float("nan")
    ci_hi = mean + t * se if n >= 2 else float("nan")
    boot_lo, boot_hi = bootstrap_mean_ci(accs) if n >= 2 else (float("nan"), float("nan"))
    # pooled proportion over all (session, problem) cells (Wilson) — tighter but
    # treats repeated problems as independent, so it is reported as a secondary
    # lower-bound-on-uncertainty reference, not the headline band.
    tot_correct = sum(s["n_correct"] for s in sessions)
    tot_cells = sum(s["n"] for s in sessions)
    w_lo, w_hi = wilson_ci(tot_correct, tot_cells)
    return {
        "n_sessions": n,
        "per_session_acc": accs,
        "per_session_correct": [s["n_correct"] for s in sessions],
        "per_session_n": [s["n"] for s in sessions],
        "band_min": min(accs) if n else float("nan"),
        "band_max": max(accs) if n else float("nan"),
        "mean": mean, "sd": sd, "se": se,
        "t_crit": t,
        "ci95_mean_lo": ci_lo, "ci95_mean_hi": ci_hi,
        "boot_ci95_lo": boot_lo, "boot_ci95_hi": boot_hi,
        "pooled_correct": tot_correct, "pooled_cells": tot_cells,
        "pooled_acc": tot_correct / tot_cells if tot_cells else float("nan"),
        "pooled_wilson_lo": w_lo, "pooled_wilson_hi": w_hi,
    }


def _per_problem_instability(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-problem answer/correctness flips across sessions."""
    ids = sorted(sessions[0]["per_problem"].keys()) if sessions else []
    rows = {}
    for pid in ids:
        answers = [s["per_problem"].get(pid, {}).get("answer") for s in sessions]
        corrects = [bool(s["per_problem"].get(pid, {}).get("correct")) for s in sessions]
        present = [a for a in answers if a is not None or True]  # keep None as an answer value
        from collections import Counter
        ctr = Counter([str(a) for a in answers])
        modal_count = max(ctr.values()) if ctr else 0
        ns = len(answers)
        flip_fraction = 1 - modal_count / ns if ns else 0.0
        rows[pid] = {
            "answers": answers,
            "distinct_answers": len(set(str(a) for a in answers)),
            "modal_count": modal_count,
            "flip_fraction": flip_fraction,
            "n_correct": sum(corrects),
            "correctness_flips": (0 if len(set(corrects)) <= 1 else 1),
            "gold": sessions[0]["per_problem"].get(pid, {}).get("gold"),
        }
    return rows


def _neartie_density(sessions: list[dict[str, Any]], thresh: float = 0.10) -> dict[str, Any]:
    """Pooled margin histogram + per-problem near-tie density (frac steps < thresh)."""
    all_margins: list[float] = []
    per_problem_density: dict[str, list[float]] = {}
    for s in sessions:
        for pid, rec in s["per_problem"].items():
            mg = [m for m in rec.get("mg", []) if m is not None]
            all_margins.extend(mg)
            if mg:
                dens = sum(1 for m in mg if m < thresh) / len(mg)
                per_problem_density.setdefault(pid, []).append(dens)
    pp_density = {pid: sum(v) / len(v) for pid, v in per_problem_density.items() if v}
    n = len(all_margins)
    summary = {
        "n_steps": n,
        "median": statistics.median(all_margins) if all_margins else float("nan"),
        "mean": sum(all_margins) / n if n else float("nan"),
        "frac_lt": {f"{t}": (sum(1 for m in all_margins if m < t) / n if n else float("nan"))
                    for t in NEARTIE_THRESHOLDS},
    }
    # coarse histogram (nats); last bin is open-ended.
    edges = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9]
    hist = [0] * (len(edges) - 1)
    for m in all_margins:
        for b in range(len(edges) - 1):
            if edges[b] <= m < edges[b + 1]:
                hist[b] += 1
                break
    summary["hist_edges"] = edges
    summary["hist_counts"] = hist
    return {"summary": summary, "per_problem_density": pp_density}


def _first_divergence(tok_a: list[str], tok_b: list[str]) -> int | None:
    n = min(len(tok_a), len(tok_b))
    for i in range(n):
        if tok_a[i] != tok_b[i]:
            return i
    if len(tok_a) != len(tok_b):
        return n
    return None


def _divergence_margins(int4_sessions: list[dict[str, Any]],
                        bf16_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Margin at the first int4-vs-int4 (cross-session) and int4-vs-bf16 divergence.

    Cross-session uses int4 session 0 vs session 1; int4-vs-bf16 uses int4 s0 vs
    bf16 s0. Reports the int4 margin at the divergence token — the mechanism claim
    is that divergence happens at low-margin (near-tie) positions."""
    rows = []
    xs_div_margins, b_div_margins = [], []
    pp0 = int4_sessions[0]["per_problem"]
    pp1 = int4_sessions[1]["per_problem"] if len(int4_sessions) > 1 else {}
    ppb = bf16_sessions[0]["per_problem"] if bf16_sessions else {}
    for pid in sorted(pp0.keys()):
        r0 = pp0[pid]
        tok0, mg0 = r0.get("tok", []), r0.get("mg", [])
        row: dict[str, Any] = {"id": pid, "len_int4_s0": len(tok0)}
        # int4 cross-session
        if pid in pp1:
            d = _first_divergence(tok0, pp1[pid].get("tok", []))
            row["xsession_div_idx"] = d
            if d is not None and d < len(mg0) and mg0[d] is not None:
                row["xsession_div_margin"] = mg0[d]
                row["xsession_div_frac"] = d / len(tok0) if tok0 else None
                xs_div_margins.append(mg0[d])
        # int4 vs bf16
        if pid in ppb:
            d = _first_divergence(tok0, ppb[pid].get("tok", []))
            row["bf16_div_idx"] = d
            if d is not None and d < len(mg0) and mg0[d] is not None:
                row["bf16_div_margin"] = mg0[d]
                row["bf16_div_frac"] = d / len(tok0) if tok0 else None
                b_div_margins.append(mg0[d])
        rows.append(row)
    return {
        "rows": rows,
        "xsession_div_margin_median": statistics.median(xs_div_margins) if xs_div_margins else None,
        "xsession_div_margin_mean": (sum(xs_div_margins) / len(xs_div_margins)) if xs_div_margins else None,
        "xsession_n_div": len(xs_div_margins),
        "xsession_frac_div_lt_0.1": (sum(1 for m in xs_div_margins if m < 0.1) / len(xs_div_margins)) if xs_div_margins else None,
        "bf16_div_margin_median": statistics.median(b_div_margins) if b_div_margins else None,
        "bf16_div_margin_mean": (sum(b_div_margins) / len(b_div_margins)) if b_div_margins else None,
        "bf16_n_div": len(b_div_margins),
    }


def _bf16_stability(bf16_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-session answer + token-stream stability for the bf16 control."""
    if len(bf16_sessions) < 2:
        return {"n_sessions": len(bf16_sessions), "note": "need >=2 sessions"}
    ids = sorted(bf16_sessions[0]["per_problem"].keys())
    ans_flip = tok_exact = 0
    n = 0
    for pid in ids:
        recs = [s["per_problem"].get(pid) for s in bf16_sessions if pid in s["per_problem"]]
        if len(recs) < 2:
            continue
        n += 1
        answers = set(str(r["answer"]) for r in recs)
        if len(answers) > 1:
            ans_flip += 1
        toks = [tuple(r.get("tok", [])) for r in recs]
        if len(set(toks)) == 1:
            tok_exact += 1
    accs = [s["accuracy"] for s in bf16_sessions]
    return {
        "n_sessions": len(bf16_sessions),
        "n_problems": n,
        "answer_flip_problems": ans_flip,
        "token_bit_exact_problems": tok_exact,
        "per_session_acc": accs,
        "acc_min": min(accs), "acc_max": max(accs),
        "bit_exact_rate": tok_exact / n if n else None,
    }


def _official_bias_check(paths: list[Path], band: dict[str, Any]) -> dict[str, Any]:
    """Logprobs-on band vs logprobs-off (official aime_eval.py) sessions.

    The greedy argmax is read from the same logits whether or not logprobs are
    requested, so logprobs-on must measure the SAME band as the official harness;
    per-problem answers differ only by the inherent near-tie nondeterminism. This
    confirms it at n=60: the official (logprobs-off) accuracies should land inside
    the logprobs-on band."""
    sessions = [json.loads(p.read_text()) for p in sorted(paths)]
    accs = [s.get("maj_k_accuracy") for s in sessions]
    lo, hi = band["band_min"], band["band_max"]
    in_band = [bool(a is not None and lo <= a <= hi) for a in accs]
    return {
        "n_official_sessions": len(sessions),
        "official_accs": accs,
        "official_mean": (sum(accs) / len(accs)) if accs else None,
        "official_n_correct": [s.get("n_correct_maj") for s in sessions],
        "band_min": lo, "band_max": hi,
        "all_in_band": all(in_band) if in_band else None,
        "official_within_ci95": [bool(a is not None and band["ci95_mean_lo"] <= a <= band["ci95_mean_hi"]) for a in accs],
    }


def cmd_aggregate(args: argparse.Namespace) -> int:
    int4 = _load_sessions(list(args.int4))
    bf16 = _load_sessions(list(args.bf16)) if args.bf16 else []
    print(f"[agg] int4 sessions={len(int4)} bf16 sessions={len(bf16)} "
          f"official={len(args.int4_official or [])}", flush=True)
    if not int4:
        print("[agg] ERROR no int4 sessions", flush=True)
        return 1

    band = _band_stats(int4)
    upper95 = band["ci95_mean_hi"]
    verdict = "BLOCKER_ROBUST" if (upper95 == upper95 and upper95 < args.bar) else "BLOCKER_STRADDLES"
    bias_check = _official_bias_check(list(args.int4_official), band) if args.int4_official else {}

    instab = _per_problem_instability(int4)
    nt = _neartie_density(int4, thresh=args.neartie_thresh)
    pp_density = nt["per_problem_density"]

    # correlation: per-problem near-tie density vs per-problem flip fraction
    common = [pid for pid in instab if pid in pp_density]
    xs = [pp_density[pid] for pid in common]
    ys = [instab[pid]["flip_fraction"] for pid in common]
    corr_p = pearson(xs, ys)
    corr_s = spearman(xs, ys)

    div = _divergence_margins(int4, bf16) if len(int4) >= 1 else {}
    bf16_stab = _bf16_stability(bf16) if bf16 else {}

    mechanism_supported = (corr_s == corr_s and corr_s > 0.2)
    out = {
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "bar": args.bar,
        "analysis_only": True, "official_tps": 0,
        "band": band,
        "int4_aime_band_upper95": upper95,
        "verdict": verdict,
        "neartie": nt["summary"],
        "neartie_thresh": args.neartie_thresh,
        "density_flip_corr_pearson": corr_p,
        "density_flip_corr_spearman": corr_s,
        "density_flip_n": len(common),
        "mechanism_supported": mechanism_supported,
        "divergence": div,
        "bf16_stability": bf16_stab,
        "logprobs_bias_check": bias_check,
        "per_problem_instability": instab,
        "per_problem_density": pp_density,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"[agg] BAND per-session acc={['%.4f' % a for a in band['per_session_acc']]}", flush=True)
    print(f"[agg] band [{band['band_min']:.4f}, {band['band_max']:.4f}] mean={band['mean']:.4f} "
          f"sd={band['sd']:.4f} t-CI95=[{band['ci95_mean_lo']:.4f}, {band['ci95_mean_hi']:.4f}] "
          f"boot=[{band['boot_ci95_lo']:.4f}, {band['boot_ci95_hi']:.4f}]", flush=True)
    print(f"[agg] DECISION int4_aime_band_upper95={upper95:.4f} vs bar={args.bar} -> {verdict}", flush=True)
    print(f"[agg] neartie median={nt['summary']['median']:.4f} "
          f"frac<0.1={nt['summary']['frac_lt']['0.1']:.4f} frac<0.05={nt['summary']['frac_lt']['0.05']:.4f} "
          f"n_steps={nt['summary']['n_steps']}", flush=True)
    print(f"[agg] density-vs-flip corr pearson={corr_p:.3f} spearman={corr_s:.3f} "
          f"n={len(common)} -> mechanism_supported={mechanism_supported}", flush=True)
    if div:
        print(f"[agg] xsession div margin median={div.get('xsession_div_margin_median')} "
              f"n_div={div.get('xsession_n_div')} frac<0.1={div.get('xsession_frac_div_lt_0.1')}", flush=True)
    if bf16_stab:
        print(f"[agg] bf16 control: {bf16_stab}", flush=True)
    if bias_check:
        print(f"[agg] logprobs-bias check: official_accs={bias_check['official_accs']} "
              f"mean={bias_check['official_mean']} all_in_band={bias_check['all_in_band']}", flush=True)
    print(f"[agg] wrote {args.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #
def cmd_wandb(args: argparse.Namespace) -> int:
    import os
    import wandb

    agg = json.loads(args.aggregate.read_text())
    band = agg["band"]
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    group = "int4ar-aime-band-neartie-ubel"
    common = {
        "analysis_only": True, "official_tps": 0, "pr": 672, "student": "ubel",
        "budget": 12288, "engine": "vllm==0.22.0", "model_int4": "int4_g128_lmhead",
        "model_bf16": "google/gemma-4-E4B-it", "no_thinking": True, "min_tokens": 8,
        "seed": 0, "bar": agg["bar"], "protocol": "greedy T=0 conc16 #650",
    }
    ids: list[str] = []

    def _init(name: str, jt: str) -> Any:
        rid = "pr672-" + re.sub(r"[^A-Za-z0-9_-]", "-", name.replace("ubel/", ""))
        return wandb.init(project=project, entity=entity, group=group, reinit=True,
                          id=rid, resume="allow", name=name, job_type=jt, config={**common})

    def _log(run: Any, scalars: dict[str, Any]) -> None:
        for k, v in scalars.items():
            run.summary[k] = v
            if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
                wandb.log({k: v})

    # run 1: BAND
    run = _init("ubel/int4-aime-band", "band")
    band_scalars = {
        "analysis_only": True, "official_tps": 0,
        "n_sessions": band["n_sessions"], "band_min": band["band_min"], "band_max": band["band_max"],
        "band_mean": band["mean"], "band_sd": band["sd"], "band_se": band["se"],
        "ci95_mean_lo": band["ci95_mean_lo"], "ci95_mean_hi": band["ci95_mean_hi"],
        "boot_ci95_lo": band["boot_ci95_lo"], "boot_ci95_hi": band["boot_ci95_hi"],
        "pooled_acc": band["pooled_acc"], "pooled_wilson_lo": band["pooled_wilson_lo"],
        "pooled_wilson_hi": band["pooled_wilson_hi"],
        "int4_aime_band_upper95": agg["int4_aime_band_upper95"], "bar": agg["bar"],
    }
    _log(run, band_scalars)
    for i, (acc, nc) in enumerate(zip(band["per_session_acc"], band["per_session_correct"])):
        run.summary[f"session{i}_acc"] = acc
        run.summary[f"session{i}_correct"] = nc
    tbl = wandb.Table(columns=["session", "accuracy", "n_correct", "n"])
    for i, (acc, nc, nn) in enumerate(zip(band["per_session_acc"], band["per_session_correct"], band["per_session_n"])):
        tbl.add_data(i, acc, nc, nn)
    run.log({"sessions": tbl})
    ids.append(run.id); run.finish()

    # run 2: NEAR-TIE mechanism
    run = _init("ubel/int4-neartie", "neartie")
    nt = agg["neartie"]
    nt_scalars = {
        "analysis_only": True, "official_tps": 0,
        "neartie_median": nt["median"], "neartie_mean": nt["mean"], "neartie_n_steps": nt["n_steps"],
        "density_flip_corr_pearson": agg["density_flip_corr_pearson"],
        "density_flip_corr_spearman": agg["density_flip_corr_spearman"],
        "density_flip_n": agg["density_flip_n"], "mechanism_supported": agg["mechanism_supported"],
    }
    for t, v in nt["frac_lt"].items():
        nt_scalars[f"neartie_frac_lt_{t}"] = v
    d = agg.get("divergence", {})
    for k in ("xsession_div_margin_median", "xsession_div_margin_mean", "xsession_n_div",
              "xsession_frac_div_lt_0.1", "bf16_div_margin_median", "bf16_n_div"):
        if d.get(k) is not None:
            nt_scalars[f"div_{k}"] = d[k]
    _log(run, nt_scalars)
    htbl = wandb.Table(columns=["bin_lo", "bin_hi", "count"])
    edges, counts = nt["hist_edges"], nt["hist_counts"]
    for b in range(len(counts)):
        htbl.add_data(edges[b], edges[b + 1], counts[b])
    run.log({"margin_hist": htbl})
    # density-vs-flip scatter
    stbl = wandb.Table(columns=["id", "neartie_density", "flip_fraction", "distinct_answers", "n_correct"])
    inst = agg["per_problem_instability"]
    dens = agg["per_problem_density"]
    for pid in sorted(dens):
        if pid in inst:
            stbl.add_data(pid, dens[pid], inst[pid]["flip_fraction"],
                          inst[pid]["distinct_answers"], inst[pid]["n_correct"])
    run.log({"density_vs_flip": stbl})
    ids.append(run.id); run.finish()

    # run 3: VERDICT (machine-checkable guards)
    run = _init("ubel/int4-aime-VERDICT", "verdict")
    bf = agg.get("bf16_stability", {})
    v_scalars = {
        "analysis_only": True, "official_tps": 0,
        "int4_aime_band_upper95": agg["int4_aime_band_upper95"], "bar": agg["bar"],
        "band_min": band["band_min"], "band_max": band["band_max"], "band_mean": band["mean"],
        "n_sessions": band["n_sessions"],
        "neartie_frac_lt_0.1": nt["frac_lt"]["0.1"], "neartie_frac_lt_0.05": nt["frac_lt"]["0.05"],
        "density_flip_corr_spearman": agg["density_flip_corr_spearman"],
        "mechanism_supported": agg["mechanism_supported"],
        "bf16_n_sessions": bf.get("n_sessions"),
        "bf16_answer_flip_problems": bf.get("answer_flip_problems"),
        "bf16_bit_exact_rate": bf.get("bit_exact_rate"),
    }
    bc = agg.get("logprobs_bias_check", {})
    if bc:
        v_scalars["bias_official_mean"] = bc.get("official_mean")
        v_scalars["bias_n_official"] = bc.get("n_official_sessions")
        v_scalars["bias_all_in_band"] = bc.get("all_in_band")
    _log(run, v_scalars)
    run.summary["verdict"] = agg["verdict"]
    if bc.get("official_accs"):
        run.summary["bias_official_accs"] = ",".join(str(a) for a in bc["official_accs"])
    ids.append(run.id); run.finish()

    print(f"[wandb] logged {len(ids)} runs -> group {group} ids={ids}", flush=True)
    print(f"[wandb] VERDICT={agg['verdict']} upper95={agg['int4_aime_band_upper95']:.4f} bar={agg['bar']}", flush=True)
    (HERE / "wandb_runs.json").write_text(json.dumps({"group": group, "wandb_run_ids": ids,
                                                      "verdict": agg["verdict"]}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# selftest (no GPU)
# --------------------------------------------------------------------------- #
def cmd_selftest(args: argparse.Namespace) -> int:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"[selftest] {'ok' if cond else 'FAIL'}: {name}", flush=True)
        ok = ok and cond

    # t975 monotone + known anchors
    check("t975(3)=3.182", abs(t975(3) - 3.182) < 1e-6)
    check("t975(>30)->1.96", abs(t975(120) - 1.96) < 1e-6)
    # wilson sanity
    lo, hi = wilson_ci(24, 60)
    check("wilson(24/60) brackets 0.40", lo < 0.40 < hi and 0.28 < lo and hi < 0.53)
    # pearson/spearman on a perfect monotone
    check("pearson perfect=1", abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-9)
    check("spearman monotone=1", abs(spearman([1, 2, 3, 4], [10, 11, 99, 100]) - 1.0) < 1e-9)
    check("pearson neg", pearson([1, 2, 3], [3, 2, 1]) < -0.99)
    # synthetic band: 4 sessions
    fake = [{"accuracy": a, "n_correct": round(a * 60), "n": 60,
             "per_problem": {}} for a in (0.35, 0.40, 0.3667, 0.3833)]
    band = _band_stats(fake)
    check("band_min/max", abs(band["band_min"] - 0.35) < 1e-9 and abs(band["band_max"] - 0.40) < 1e-9)
    check("band mean ~0.375", abs(band["mean"] - 0.375) < 1e-3)
    check("upper95 finite & > mean", band["ci95_mean_hi"] > band["mean"])
    # instability + density correlation on synthetic per_problem
    s1 = {"accuracy": 0.5, "n_correct": 1, "n": 2, "per_problem": {
        "p1": {"answer": 1, "correct": True, "gold": 1, "tok": ["a", "b", "c"], "mg": [0.01, 0.02, 5.0]},
        "p2": {"answer": 2, "correct": False, "gold": 9, "tok": ["x", "y", "z"], "mg": [5.0, 5.0, 5.0]}}}
    s2 = {"accuracy": 0.5, "n_correct": 1, "n": 2, "per_problem": {
        "p1": {"answer": 7, "correct": False, "gold": 1, "tok": ["a", "q", "c"], "mg": [0.01, 0.03, 5.0]},
        "p2": {"answer": 2, "correct": False, "gold": 9, "tok": ["x", "y", "z"], "mg": [5.0, 5.0, 5.0]}}}
    inst = _per_problem_instability([s1, s2])
    check("p1 flips (density-driven)", inst["p1"]["flip_fraction"] == 0.5)
    check("p2 stable", inst["p2"]["flip_fraction"] == 0.0)
    nt = _neartie_density([s1, s2], thresh=0.1)
    check("p1 high near-tie density", nt["per_problem_density"]["p1"] > nt["per_problem_density"]["p2"])
    dv = _divergence_margins([s1, s2], [])
    row_p1 = next(r for r in dv["rows"] if r["id"] == "p1")
    check("p1 xsession diverges at idx1", row_p1.get("xsession_div_idx") == 1)
    print("[selftest] PASS" if ok else "[selftest] FAILURES PRESENT", flush=True)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("session", help="full-60 greedy AIME + margins from a served endpoint")
    p.add_argument("--arm", choices=["int4", "bf16"], required=True)
    p.add_argument("--session-idx", type=int, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="gemma-4-e4b-it")
    p.add_argument("--years", default="2024,2025-I,2025-II")
    p.add_argument("--max-tokens", type=int, default=12288)
    p.add_argument("--min-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--client-concurrency", type=int, default=16)
    p.add_argument("--top-logprobs", type=int, default=2)
    p.add_argument("--enable-thinking", action="store_true", help="OFF reproduces #650 --no-thinking")
    p.add_argument("--limit", type=int, default=None, help="cap problems (identity guard / smoke)")
    p.add_argument("--save-text", action="store_true")
    p.add_argument("--request-timeout-s", type=int, default=1800)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_session)

    p = sub.add_parser("aggregate", help="sessions -> band + near-tie + verdict")
    p.add_argument("--int4", type=Path, nargs="+", required=True)
    p.add_argument("--bf16", type=Path, nargs="*", default=[])
    p.add_argument("--int4-official", type=Path, nargs="*", default=[],
                   help="official aime_eval.py (logprobs-off) session JSONs for the bias check")
    p.add_argument("--bar", type=float, default=BAR)
    p.add_argument("--neartie-thresh", type=float, default=0.10)
    p.add_argument("--out", type=Path, default=HERE / "band_neartie_agg.json")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("wandb", help="log band + near-tie + verdict to W&B")
    p.add_argument("--aggregate", type=Path, default=HERE / "band_neartie_agg.json")
    p.set_defaults(func=cmd_wandb)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
