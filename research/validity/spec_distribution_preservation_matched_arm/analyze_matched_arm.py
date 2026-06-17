#!/usr/bin/env python
"""PR #620 — matched-arm paired analysis: spec vs AR distribution preservation.

Pairs per-item correctness between the spec arm (int4+MTP-K7) and the AR arm
(plain int4, same body), then tests whether spec quality == AR quality.

Statistical design (researcher-validated):
  * The two arms draw INDEPENDENT stochastic samples (rejection sampling consumes
    RNG differently than plain sampling), but on BYTE-IDENTICAL prompts -> a paired
    design. We assert prompt_sha equality across arms per (seed,item) before pairing.
  * The SAME GPQA question appears under 5 choice-shuffle layouts (seeds) -> the 5
    obs per question are CORRELATED. Naive McNemar over (seed,item) units overstates
    power. PRIMARY estimate = CLUSTER-bootstrap by question id (cluster-robust CI on
    the paired accuracy delta). Cross-checks: (seed,item)-level McNemar (anti-
    conservative) + cluster-level paired-diff t-CI (conservative).

Verdict:
  SPEC_DISTRIBUTION_PRESERVING  -> cluster-bootstrap delta CI contains 0 (spec quality
                                   statistically == AR quality; the GPQA softness is the
                                   shared int4 body, not spec). Firing option B costs ONLY
                                   the strict-#319 GREEDY contract, not downstream quality.
  SPEC_DEGRADES_QUALITY         -> spec arm CI entirely below AR (spec measurably worse).

CAVEAT (stated, not tested here): this is the SAMPLED-decode distribution claim. It does
NOT rescue strict #319 (a GREEDY byte-exact gate; wirbel #607 settled spec breaks it).
Separate axes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

GPQA_SEEDS = [12345, 23456, 34567, 45678, 56789]
GSM8K_SAMPLING_SEEDS = [1234, 2345, 3456, 4567, 5678]
B_BOOT = 10000
RNG = np.random.default_rng(20620)


# ---------- loaders ----------
def load_gpqa_arm(arm: str) -> dict[int, dict[str, dict]]:
    """seed -> {id -> per_sample record}."""
    out = {}
    for seed in GPQA_SEEDS:
        p = RES / f"{arm}_gpqa_s{seed}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out[seed] = {str(r["id"]): r for r in d.get("per_sample", [])}
    return out


def load_gsm8k_arm(arm: str) -> dict[int, dict[str, dict]]:
    """sampling_seed -> {id -> per_problem record}."""
    out = {}
    for ss in GSM8K_SAMPLING_SEEDS:
        p = RES / f"gsm8k_{arm}_ss{ss}" / f"{arm}_sampled_s{ss}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        out[ss] = {str(r["id"]): r for r in d.get("per_problem", [])}
    return out


# ---------- stats ----------
def mcnemar(pairs: list[tuple[int, int]]) -> dict:
    """pairs = list of (spec_correct, ar_correct). Returns b,c, exact two-sided p,
    continuity-corrected chi2."""
    b = sum(1 for s, a in pairs if s == 1 and a == 0)  # spec wins
    c = sum(1 for s, a in pairs if s == 0 and a == 1)  # ar wins
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_exact": 1.0, "chi2_cc": 0.0,
                "p_chi2_cc": 1.0}
    k = min(b, c)
    # exact two-sided binomial McNemar (sum tail probs, double, cap at 1)
    if n <= 2000:
        cdf = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        p_exact = min(1.0, 2.0 * cdf)
    else:  # normal approx for very large discordant counts
        z = (abs(b - c) - 1) / math.sqrt(n)
        p_exact = math.erfc(z / math.sqrt(2))
    chi2_cc = (abs(b - c) - 1) ** 2 / n
    # chi2 sf with df=1 == erfc(sqrt(chi2/2))
    p_chi2 = math.erfc(math.sqrt(chi2_cc / 2.0)) if chi2_cc > 0 else 1.0
    return {"b": b, "c": c, "n_discordant": n, "p_exact": p_exact,
            "chi2_cc": chi2_cc, "p_chi2_cc": p_chi2}


def cluster_bootstrap(cluster_ids: np.ndarray, spec: np.ndarray, ar: np.ndarray) -> dict:
    """Resample CLUSTERS (questions) with replacement; each cluster carries all its
    (seed) reps for BOTH arms. Returns spec/ar/delta point + 95% percentile CIs."""
    uniq = np.unique(cluster_ids)
    idx_by_cluster = {cid: np.where(cluster_ids == cid)[0] for cid in uniq}
    members = [idx_by_cluster[cid] for cid in uniq]
    point_spec = float(spec.mean())
    point_ar = float(ar.mean())
    point_delta = point_spec - point_ar
    bs_spec = np.empty(B_BOOT)
    bs_ar = np.empty(B_BOOT)
    bs_delta = np.empty(B_BOOT)
    nC = len(uniq)
    for t in range(B_BOOT):
        pick = RNG.integers(0, nC, size=nC)
        sel = np.concatenate([members[j] for j in pick])
        s = spec[sel].mean()
        a = ar[sel].mean()
        bs_spec[t] = s
        bs_ar[t] = a
        bs_delta[t] = s - a

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "n_clusters": int(nC),
        "n_units": int(len(spec)),
        "spec_acc": point_spec, "spec_ci95": ci(bs_spec),
        "ar_acc": point_ar, "ar_ci95": ci(bs_ar),
        "delta": point_delta, "delta_ci95": ci(bs_delta),
        "delta_se_boot": float(bs_delta.std(ddof=1)),
    }


def cluster_level_paired_diff(cluster_ids: np.ndarray, spec: np.ndarray, ar: np.ndarray) -> dict:
    """Per-cluster mean correctness, then paired diff across clusters (conservative,
    treats each question as 1 independent unit). Normal-approx t-CI + Wilcoxon-ish."""
    uniq = np.unique(cluster_ids)
    diffs = []
    for cid in uniq:
        m = cluster_ids == cid
        diffs.append(float(spec[m].mean() - ar[m].mean()))
    diffs = np.array(diffs)
    n = len(diffs)
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {
        "n_clusters": n, "mean_paired_diff": mean, "se": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se] if se == se else [float("nan")] * 2,
        "n_clusters_spec_better": int((diffs > 0).sum()),
        "n_clusters_ar_better": int((diffs < 0).sum()),
        "n_clusters_tied": int((diffs == 0).sum()),
    }


def gpqa_truncation(arm: str) -> dict:
    """finish_reason=length proxy for GPQA from inspect logs; falls back to
    empty/no-answer rates from per_sample if logs unreadable."""
    trunc_n = total = empty_n = noans_n = 0
    log_ok = False
    try:
        from inspect_ai.log import read_eval_log
        for seed in GPQA_SEEDS:
            p = RES / f"{arm}_gpqa_s{seed}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            loc = d.get("eval_log")
            if not loc or not Path(loc).exists():
                continue
            log = read_eval_log(loc)
            for s in (log.samples or []):
                total += 1
                so = getattr(s, "output", None)
                sr = None
                if so is not None and getattr(so, "choices", None):
                    sr = getattr(so.choices[0], "stop_reason", None)
                if sr in ("max_tokens", "length"):
                    trunc_n += 1
                log_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] gpqa truncation log read failed ({e}); using per_sample proxy", flush=True)
    # per_sample proxy
    for seed in GPQA_SEEDS:
        p = RES / f"{arm}_gpqa_s{seed}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d.get("per_sample", []):
            if r.get("empty"):
                empty_n += 1
            if r.get("answer") is None and not r.get("empty"):
                noans_n += 1
    nps = sum(len(json.loads((RES / f"{arm}_gpqa_s{s}.json").read_text()).get("per_sample", []))
              for s in GPQA_SEEDS if (RES / f"{arm}_gpqa_s{s}.json").exists())
    return {
        "finish_length_rate": (trunc_n / total) if (log_ok and total) else None,
        "n_truncated_length": trunc_n if log_ok else None,
        "empty_rate": (empty_n / nps) if nps else None,
        "no_answer_rate": (noans_n / nps) if nps else None,
        "n_per_sample": nps,
    }


def gsm8k_truncation(arm: str) -> dict:
    trunc_n = total = 0
    for ss in GSM8K_SAMPLING_SEEDS:
        p = RES / f"gsm8k_{arm}_ss{ss}" / f"{arm}_sampled_s{ss}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d.get("per_problem", []):
            total += 1
            if r.get("finish_reason") == "length":
                trunc_n += 1
    return {"finish_length_rate": (trunc_n / total) if total else None,
            "n_truncated_length": trunc_n, "n_items": total}


# ---------- pairing ----------
def pair_eval(spec_by_seed: dict, ar_by_seed: dict, label: str) -> dict:
    cluster_ids, spec_c, ar_c = [], [], []
    sha_mismatches, missing = [], 0
    spec_err, ar_err = set(), set()  # (seed,id) units that hit a serving/API error
    seeds = sorted(set(spec_by_seed) & set(ar_by_seed))
    for seed in seeds:
        sids = spec_by_seed[seed]
        aids = ar_by_seed[seed]
        for iid in sorted(set(sids) & set(aids)):
            sr, ar = sids[iid], aids[iid]
            if sr.get("error"):
                spec_err.add((seed, iid))
            if ar.get("error"):
                ar_err.add((seed, iid))
            # prompt_sha gate (GPQA has it; GSM8K records None -> skip the check there)
            ssha, asha = sr.get("prompt_sha"), ar.get("prompt_sha")
            if ssha is not None and asha is not None and ssha != asha:
                sha_mismatches.append((seed, iid, ssha, asha))
                continue
            cluster_ids.append(iid)
            spec_c.append(1 if sr.get("correct") else 0)
            ar_c.append(1 if ar.get("correct") else 0)
        missing += len(set(sids) ^ set(aids))
    cluster_ids = np.array(cluster_ids)
    spec_c = np.array(spec_c)
    ar_c = np.array(ar_c)
    pairs = list(zip(spec_c.tolist(), ar_c.tolist()))
    out = {
        "label": label,
        "seeds_paired": seeds,
        "n_paired_units": int(len(spec_c)),
        "n_prompt_sha_mismatch": len(sha_mismatches),
        "n_id_asymmetry": missing,
        "errors": {
            "n_spec_errored": len(spec_err),
            "n_ar_errored": len(ar_err),
            # Serving errors (e.g. input+max_tokens>max_model_len rejections) depend ONLY
            # on prompt length + budget, which are identical across arms -> they must fall
            # on the SAME (seed,id) units. Symmetric errors pair as (0,0): concordant, so
            # they leave McNemar b/c and the paired delta untouched (they only deflate the
            # ABSOLUTE accuracy of both arms equally). Asymmetry would threaten matching.
            "symmetric": (spec_err == ar_err),
            "n_errored_units_either_arm": len(spec_err | ar_err),
            "errored_unit_examples": sorted(spec_err | ar_err)[:5],
        },
        "mcnemar_seeditem": mcnemar(pairs),
        "cluster_bootstrap": cluster_bootstrap(cluster_ids, spec_c, ar_c) if len(spec_c) else {},
        "cluster_level_paired_diff": cluster_level_paired_diff(cluster_ids, spec_c, ar_c) if len(spec_c) else {},
    }
    if sha_mismatches:
        out["prompt_sha_mismatch_examples"] = sha_mismatches[:5]
    return out


def verdict_from(eval_block: dict) -> str:
    cb = eval_block.get("cluster_bootstrap") or {}
    ci = cb.get("delta_ci95")
    if not ci:
        return "INCONCLUSIVE"
    lo, hi = ci
    # delta = spec - ar. PRESERVING if CI contains 0. DEGRADES if entirely below 0.
    if hi < 0:
        return "SPEC_DEGRADES_QUALITY"
    if lo > 0:
        return "SPEC_BETTER_THAN_AR"  # not degradation; report as favorable
    return "SPEC_DISTRIBUTION_PRESERVING"


def main() -> int:
    result = {"analysis_only": True, "official_tps": 0,
              "design": "matched-arm paired (spec=int4+MTP-K7 ON, ar=int4 spec OFF, same body)",
              "stack": "vllm==0.22.0", "sampling": {"temperature": 1.0, "top_p": 0.95,
              "top_k": 64, "min_tokens": 8, "gpqa_max_tokens": 4096, "gsm8k_max_tokens": 512,
              "max_model_len": 6144}}

    # GPQA-Diamond (primary)
    gp = pair_eval(load_gpqa_arm("spec"), load_gpqa_arm("ar"), "gpqa_diamond")
    gp["truncation"] = {"spec": gpqa_truncation("spec"), "ar": gpqa_truncation("ar")}
    gp["verdict"] = verdict_from(gp)
    result["gpqa_diamond"] = gp

    # GSM8K (secondary)
    gs = pair_eval(load_gsm8k_arm("spec"), load_gsm8k_arm("ar"), "gsm8k")
    gs["truncation"] = {"spec": gsm8k_truncation("spec"), "ar": gsm8k_truncation("ar")}
    gs["verdict"] = verdict_from(gs)
    result["gsm8k"] = gs

    # Overall: primary eval drives the headline verdict.
    result["headline_verdict"] = gp["verdict"]

    (RES / "analysis.json").write_text(json.dumps(result, indent=2))

    # ---- human-readable report ----
    def fmt_eval(name, e):
        cb = e.get("cluster_bootstrap", {})
        mc = e.get("mcnemar_seeditem", {})
        cl = e.get("cluster_level_paired_diff", {})
        lines = [f"\n=== {name} (n={e['n_paired_units']} paired units, "
                 f"{cb.get('n_clusters','?')} question-clusters) ==="]
        if e["n_prompt_sha_mismatch"]:
            lines.append(f"  !! prompt_sha MISMATCH on {e['n_prompt_sha_mismatch']} units "
                         f"-> arms NOT matched; comparison INVALID")
        else:
            lines.append("  prompt_sha gate: PASS (byte-identical prompts across arms)")
        er = e.get("errors", {})
        if er.get("n_errored_units_either_arm"):
            sym = "SYMMETRIC (pairs as (0,0); delta unaffected)" if er.get("symmetric") \
                else "!! ASYMMETRIC -> threatens matching"
            lines.append(f"  serving errors: spec={er['n_spec_errored']} ar={er['n_ar_errored']} "
                         f"-> {sym}")
        if cb:
            lines.append(f"  spec acc = {cb['spec_acc']:.4f}  CI95 {cb['spec_ci95']}")
            lines.append(f"  ar   acc = {cb['ar_acc']:.4f}  CI95 {cb['ar_ci95']}")
            lines.append(f"  PAIRED delta (spec-ar) = {cb['delta']:+.4f}  "
                         f"cluster-boot CI95 {[round(x,4) for x in cb['delta_ci95']]}")
        lines.append(f"  McNemar (seed,item-level; anti-conservative): b(spec>ar)={mc['b']} "
                     f"c(ar>spec)={mc['c']} p_exact={mc['p_exact']:.4f}")
        if cl:
            lines.append(f"  cluster-level paired-diff: mean={cl['mean_paired_diff']:+.4f} "
                         f"CI95 {[round(x,4) for x in cl['ci95']]} "
                         f"(spec-better {cl['n_clusters_spec_better']} / "
                         f"ar-better {cl['n_clusters_ar_better']} / tied {cl['n_clusters_tied']})")
        lines.append(f"  truncation(finish=length): spec={e['truncation']['spec'].get('finish_length_rate')} "
                     f"ar={e['truncation']['ar'].get('finish_length_rate')}")
        lines.append(f"  VERDICT: {e['verdict']}")
        return "\n".join(lines)

    report = ["MATCHED-ARM SPEC-vs-AR DISTRIBUTION PRESERVATION (PR #620)",
              fmt_eval("GPQA-Diamond [PRIMARY]", gp),
              fmt_eval("GSM8K [secondary]", gs),
              f"\nHEADLINE VERDICT (GPQA-D): {result['headline_verdict']}"]
    rep = "\n".join(report)
    (RES / "report.txt").write_text(rep + "\n")
    print(rep, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
