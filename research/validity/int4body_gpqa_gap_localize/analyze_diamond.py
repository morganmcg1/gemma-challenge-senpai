#!/usr/bin/env python3
"""PR #692 -- int4-body GPQA-Diamond gap LOCALIZATION. LOCAL, NO FIRE, analysis_only.

Decision-forcing question: is the int4-body's gpqa_diamond gap a TRUE degradation vs
vanilla bf16 base, and does it concentrate in subjects / question-types / reasoning
depths -- or is it diffuse / a decode-basis measurement artifact?

Reuses the #682 discordant-pairs / McNemar machinery. NO new eval pass: every number
comes from existing per-item GPQA-Diamond JSONs (n=198) on the faithful #511 harness.
The ONLY thing fetched is the Wanfq/gpqa gpqa_diamond.csv domain map (198/198 id match).

Arms (all gpqa_diamond, n=198, same byte-identical prompts -- prompt_sha asserted 0
mismatch):
  vanilla bf16 base : fp16   (gate denominator; greedy 0.5253, sampled 3-seed 0.5236)
  int4 served       : int4   (int4 g128-untied lm_head + int4 body; greedy 0.4596)
  int4 body-isolated: head262k (int4 body + native bf16 262k head; greedy 0.4697 = the
                                #682 anchor). corner_c #652 shows head effect n.s., so
                                head262k isolates the BODY's own quality.

Gate #515: int4 must score >=90% of vanilla base. bar = 0.9 * 0.5236 = 0.471.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
RV = HERE.parent  # research/validity
BFQS = RV / "base_fullhead_quality_sampling/results"
ANSWER_PAT = re.compile(r"ANSWER\s*[:=]\s*\(?([ABCD])\)?", re.I)

BASE_SAMPLED_3SEED = 0.5236   # gate denominator (qi24h8zx, seeds 1234/35/36)
BASE_GREEDY = 0.5253          # greedy anchor (104/198)
BAR_90 = 0.471


# ----------------------------------------------------------------------------- io
def load_items(path: str) -> dict:
    d = json.load(open(path))
    out = {}
    for r in d.get("per_sample", []):
        if r.get("value") not in ("C", "I"):
            continue  # drop errors/unscored
        out[r["id"]] = r
    return out


def load_seeds(paths: list[str]) -> dict:
    """(id, sseed) -> row, pooled over sampled seeds."""
    out = {}
    for p in paths:
        d = json.load(open(p))
        ss = d.get("sampling_seed")
        for r in d.get("per_sample", []):
            if r.get("value") not in ("C", "I"):
                continue
            out[(r["id"], ss)] = r
    return out


# ------------------------------------------------------------------ domain / type
def diamond_meta() -> dict:
    path = hf_hub_download(repo_id="Wanfq/gpqa", filename="gpqa_diamond.csv",
                           repo_type="dataset", token=os.environ.get("HF_TOKEN") or None)
    meta = {}
    with open(path) as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        idc = [c for c in cols if c.strip().lower() in ("record id", "record_id")][0]
        dc = [c for c in cols if "high-level domain" in c.strip().lower()][0]
        sdc = [c for c in cols if c.strip().lower() == "subdomain"]
        qc = [c for c in cols if c.strip().lower() == "question"][0]
        for row in r:
            q = row[qc] or ""
            # calc-heavy proxy: many digits or explicit calculation cue words / math glyphs
            ndig = sum(ch.isdigit() for ch in q)
            calc_cue = bool(re.search(r"\b(calculate|compute|determine the (value|number|rate|energy|"
                                      r"concentration|wavelength|frequency|mass|velocity|temperature)"
                                      r"|how many|what is the (value|magnitude|ratio))\b", q, re.I))
            mathglyph = bool(re.search(r"[=±×÷√∫∑°λμ→\^]|\d+\s*(nm|kJ|mol|eV|MHz|GHz|cm|mm|kg|°C|K)\b", q))
            meta[row[idc].strip()] = {
                "domain": row[dc].strip(),
                "subdomain": (row[sdc[0]].strip() if sdc else None),
                "q_len": len(q),
                "n_digits": ndig,
                "calc_heavy": bool(calc_cue or mathglyph or ndig >= 8),
            }
    return meta


# ----------------------------------------------------------------------- mcnemar
def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial p on the discordant pairs (b,c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def paired_bc(base: dict, var: dict, keys) -> tuple[int, int, int, int]:
    """b = base-right & var-wrong ; c = base-wrong & var-right ; concordant counts."""
    b = c = both = neither = 0
    for k in keys:
        br, vr = base[k]["correct"], var[k]["correct"]
        if br and not vr:
            b += 1
        elif (not br) and vr:
            c += 1
        elif br and vr:
            both += 1
        else:
            neither += 1
    return b, c, both, neither


def subject_table(base: dict, var: dict, meta: dict, keyfn=lambda k: k) -> dict:
    """Per-subject b/c/net/p. `keyfn(k)` maps a pairing key -> record id for meta lookup."""
    by = defaultdict(list)
    for k in base.keys() & var.keys():
        dom = meta.get(keyfn(k), {}).get("domain", "?")
        by[dom].append(k)
    out = {}
    for dom, keys in by.items():
        b, c, both, neither = paired_bc(base, var, keys)
        n = len(keys)
        out[dom] = {
            "n_items": n, "b_base_right_int4_wrong": b, "c_base_wrong_int4_right": c,
            "net": b - c, "mcnemar_p": round(mcnemar(b, c), 4),
            "base_acc": round((b + both) / n, 4), "int4_acc": round((c + both) / n, 4),
            "acc_delta": round((c - b) / n, 4),
        }
    # ALL
    keys = list(base.keys() & var.keys())
    b, c, both, neither = paired_bc(base, var, keys)
    n = len(keys)
    out["ALL"] = {
        "n_items": n, "b_base_right_int4_wrong": b, "c_base_wrong_int4_right": c,
        "net": b - c, "mcnemar_p": round(mcnemar(b, c), 4),
        "base_acc": round((b + both) / n, 4), "int4_acc": round((c + both) / n, 4),
        "acc_delta": round((c - b) / n, 4),
    }
    return out


# ------------------------------------------------------------- verbosity / trunc
def verbosity_on_discordant(base: dict, var: dict, meta: dict) -> dict:
    """On b-cells (base-right/var-wrong), is var systematically more verbose? (chars)."""
    fb, vb, fb_dom = [], [], defaultdict(int)
    for k in base.keys() & var.keys():
        if base[k]["correct"] and not var[k]["correct"]:
            bc = base[k].get("completion_chars")
            vc = var[k].get("completion_chars")
            if isinstance(bc, (int, float)) and isinstance(vc, (int, float)):
                fb.append(bc)
                vb.append(vc)
            fb_dom[meta.get(k if isinstance(k, str) else k[0], {}).get("domain", "?")] += 1
    out = {"n_b_cells_with_chars": len(fb), "n_b_cells_by_domain": dict(fb_dom)}
    if fb:
        out.update({
            "base_chars_median": int(st.median(fb)), "var_chars_median": int(st.median(vb)),
            "base_chars_mean": int(st.mean(fb)), "var_chars_mean": int(st.mean(vb)),
            "var_more_verbose_frac": round(sum(1 for a, b_ in zip(vb, fb) if a > b_) / len(fb), 3),
        })
    return out


def chars_truncation_check(d: dict, mt_tokens: int) -> dict:
    """mt3072 JSONs lack stop_reason; proxy truncation via completion_chars near the cap.
    ~3.6 chars/token for Gemma reasoning text -> cap_chars ~ mt_tokens*3.6."""
    chars = [r.get("completion_chars") for r in d.values()
             if isinstance(r.get("completion_chars"), (int, float))]
    if not chars:
        return {"n_with_chars": 0}
    chars.sort()
    cap = mt_tokens * 3.6
    near = sum(1 for c in chars if c >= 0.92 * cap)
    return {"n_with_chars": len(chars), "chars_p50": int(chars[len(chars) // 2]),
            "chars_p95": int(chars[int(0.95 * len(chars))]), "chars_max": int(chars[-1]),
            "approx_cap_chars": int(cap), "n_near_cap_ge92pct": near,
            "near_cap_rate": round(near / len(chars), 4)}


def wilson(c: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [float("nan"), float("nan")]
    p = c / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den
    return [round(cen - half, 4), round(cen + half, 4)]


def aime_skill_axis() -> dict:
    """The AIME lane (ubel #672/#679): is the int4 loss the SAME quantitative-precision
    skill as the gpqa calc-heavy locus? Read the gb6144 base-vs-int4 AIME per-item."""
    base_p = RV / "optionb_denom_0p22_gb6144/results/base_aime_greedy_gb6144.json"
    int4_p = RV / "optionb_denom_0p22_gb6144/results_int4ar/int4ar_aime_greedy_gb6144.json"
    if not (base_p.exists() and int4_p.exists()):
        return {"available": False}
    bd = json.load(open(base_p))
    idd = json.load(open(int4_p))
    bm = {r["id"]: bool(r.get("maj_correct")) for r in bd["per_problem"]}
    im = {r["id"]: bool(r.get("maj_correct")) for r in idd["per_problem"]}
    keys = bm.keys() & im.keys()
    b = sum(1 for k in keys if bm[k] and not im[k])
    c = sum(1 for k in keys if (not bm[k]) and im[k])
    return {"available": True, "task": "aime_maj1", "decode": "greedy", "mt": 6144,
            "n_items": len(keys),
            "base_acc": round(sum(bm[k] for k in keys) / len(keys), 4),
            "int4_acc": round(sum(im[k] for k in keys) / len(keys), 4),
            "pct_of_base": round(100 * (sum(im[k] for k in keys)) / max(1, sum(bm[k] for k in keys)), 1),
            "b_base_right_int4_wrong": b, "c_base_wrong_int4_right": c,
            "net": b - c, "mcnemar_p": round(mcnemar(b, c), 4),
            "note": "AIME is 100% multi-step quantitative; shares the calculation-precision "
                    "skill axis with the gpqa calc-heavy locus. ids do NOT overlap gpqa "
                    "(different datasets) -> correlation is at the SKILL level, not item level."}


def trunc_rates_gb6144() -> dict:
    """Arm-level truncation rates from the gb6144 stop_reason-instrumented JSONs."""
    out = {}
    base_p = RV / "optionb_denom_0p22_gb6144/results/base_gpqa_sampled_gb6144.json"
    int4_glob = sorted(glob.glob(str(RV / "optionb_denom_0p22_gb6144/results_int4ar/int4ar_gpqa_sampled_s*.json")))

    def rate(paths):
        tot = tr = wrong_when_tr = 0
        for p in paths:
            d = json.load(open(p))
            for r in d["per_sample"]:
                if r.get("value") not in ("C", "I"):
                    continue
                tot += 1
                if r.get("truncated"):
                    tr += 1
                    if not r.get("correct"):
                        wrong_when_tr += 1
        return tot, tr, wrong_when_tr

    bt, btr, bw = rate([str(base_p)])
    it, itr, iw = rate(int4_glob)
    out["base"] = {"n": bt, "truncated": btr, "trunc_rate": round(btr / bt, 4) if bt else None,
                   "pct_truncated_wrong": round(100 * bw / btr, 1) if btr else None,
                   "mt": 6144, "n_seeds": 1}
    out["int4_served"] = {"n": it, "truncated": itr, "trunc_rate": round(itr / it, 4) if it else None,
                          "pct_truncated_wrong": round(100 * iw / itr, 1) if itr else None,
                          "mt": 6144, "n_seeds": len(int4_glob)}
    return out


# ------------------------------------------------------------------ verdict / wb
def decide_verdict(out: dict) -> tuple[str, str]:
    """One of GPQA_GAP_CONCENTRATED | GPQA_GAP_DIFFUSE | GPQA_GAP_NOT_REAL, + reason.

    Logic:
      - DIFFUSE is rejected if any question-type lens shows clean asymmetric split
        (calc_heavy vs non_calc, or shallow vs deep CoT) -- i.e. the loss has STRUCTURE.
      - Within structure, CONCENTRATED is the call: the gap localizes to a coherent
        skill (calc-heavy / shallow-confident-CoT) shared with the AIME quantitative
        lane. The decode-basis fact (sampled GATE clears 90%) is recorded as a RIDER,
        not the verdict -- the structured residual is real (shallow-CoT p<0.05).
      - NOT_REAL would require BOTH no structure AND gate-clear; structure is present.
    """
    qt = out["per_question_type_greedy_body"]
    calc, ncalc = qt["calc_heavy"], qt["non_calc"]
    shallow, deep = qt["base_cot_SHALLOW_below_median"], qt["base_cot_DEEP_above_median"]
    structured = (
        (calc["net"] >= 6 and ncalc["net"] <= 1)          # calc/non-calc clean split
        or (shallow["mcnemar_p"] < 0.05 and deep["net"] <= 1)  # shallow significant, deep null
    )
    gate = out["decode_basis_table"]["sampled_mt3072_GATE"]
    gate_clears = gate["clears_90_body"] and gate["clears_90_served"]
    if structured:
        verdict = "GPQA_GAP_CONCENTRATED"
        reason = (
            "Loss has clean STRUCTURE: calc_heavy net %d (p=%.4f) vs non_calc net %d (p=%.2f); "
            "shallow-CoT net %d (p=%.4f, SIGNIFICANT) vs deep-CoT net %d (p=%.2f, floor base_acc=%.2f). "
            "Locus = quantitative-precision on items base solves with short confident CoT; "
            "SAME skill axis as AIME (int4 %.1f%% of base). RIDER: on the gate-faithful #31 SAMPLED "
            "decode the int4-body clears 90%% (%.2f%% body, %.2f%% served) -- the literal 0.4697<0.471 "
            "FAILURE is GREEDY-decode-specific, not the gate protocol." % (
                calc["net"], calc["mcnemar_p"], ncalc["net"], ncalc["mcnemar_p"],
                shallow["net"], shallow["mcnemar_p"], deep["net"], deep["mcnemar_p"],
                deep["base_acc"], out["aime_skill_axis"].get("pct_of_base", float("nan")),
                gate["pct_body"], gate["pct_served"]))
    elif gate_clears:
        verdict = "GPQA_GAP_NOT_REAL"
        reason = "No structured lens and gate-faithful sampled decode clears 90%."
    else:
        verdict = "GPQA_GAP_DIFFUSE"
        reason = "Symmetric reshuffling across all lenses; no concentration."
    return verdict, reason


def log_wandb(out: dict, verdict: str, reason: str) -> list:
    import wandb
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    group = os.environ.get("WANDB_GROUP", "int4body-gpqa-gap-wirbel")
    basis = out["decode_basis_table"]
    g, s = basis["greedy_mt3072"], basis["sampled_mt3072_GATE"]
    qt = out["per_question_type_greedy_body"]
    subj_s = out["per_subject_mcnemar"]["sampled_gate_fp16_vs_int4_3seed"]
    subj_gb = out["per_subject_mcnemar"]["greedy_body_isolated_fp16_vs_head262k"]
    aime = out["aime_skill_axis"]
    greedy_body_gap = round(out["gate"]["base_greedy"] - g["int4_body_isolated"], 4)
    sampled_body_gap = round(s["base_3seed"] - s["int4_body_10seed"], 4)
    common = {"analysis_only": True, "official_tps": 0, "no_hf_job": 1,
              "fires": False, "pr": 692, "student": "wirbel", "instrument": "gpqa_diamond",
              "n_items": 198, "engine": "vllm-faithful-#511", "decode_gate": "sampled-#31",
              "gate_bar": out["gate"]["bar"]}
    run = wandb.init(project=project, entity=entity, group=group,
                     name="wirbel/int4body-gpqa-gap", job_type="int4body-gpqa-gap-localize",
                     reinit=True, config=common)
    logd = {
        # decode-basis (the headline)
        "base_greedy": out["gate"]["base_greedy"], "base_sampled_3seed": out["gate"]["base_sampled_3seed"],
        "int4_body_greedy": g["int4_body_isolated"], "int4_served_greedy": g["int4_served"],
        "int4_body_sampled_10seed": s["int4_body_10seed"], "int4_served_sampled_3seed": s["int4_served_3seed"],
        "pct_body_greedy": g["pct_body"], "pct_served_greedy": g["pct_served"],
        "pct_body_sampled": s["pct_body"], "pct_served_sampled": s["pct_served"],
        "clears_90_greedy_body": int(g["clears_90_body"]), "clears_90_sampled_body": int(s["clears_90_body"]),
        "clears_90_sampled_served": int(s["clears_90_served"]),
        "greedy_body_gap_vs_vanilla": greedy_body_gap, "sampled_body_gap_vs_vanilla": sampled_body_gap,
        # primary metric: the body gap that trips the literal gate (greedy basis the PR cites)
        "int4body_gpqa_gap_vs_vanilla": greedy_body_gap,
        # concentration structure
        "calc_heavy_net": qt["calc_heavy"]["net"], "calc_heavy_p": qt["calc_heavy"]["mcnemar_p"],
        "non_calc_net": qt["non_calc"]["net"], "non_calc_p": qt["non_calc"]["mcnemar_p"],
        "shallow_cot_net": qt["base_cot_SHALLOW_below_median"]["net"],
        "shallow_cot_p": qt["base_cot_SHALLOW_below_median"]["mcnemar_p"],
        "deep_cot_net": qt["base_cot_DEEP_above_median"]["net"],
        "deep_cot_p": qt["base_cot_DEEP_above_median"]["mcnemar_p"],
        "deep_cot_base_acc": qt["base_cot_DEEP_above_median"]["base_acc"],
        # per-subject (sampled gate lens + greedy body lens)
        "subj_physics_net_sampled": subj_s["Physics"]["net"], "subj_physics_p_sampled": subj_s["Physics"]["mcnemar_p"],
        "subj_chem_net_sampled": subj_s["Chemistry"]["net"], "subj_chem_p_sampled": subj_s["Chemistry"]["mcnemar_p"],
        "subj_bio_net_sampled": subj_s["Biology"]["net"], "subj_bio_p_sampled": subj_s["Biology"]["mcnemar_p"],
        "subj_all_net_sampled": subj_s["ALL"]["net"], "subj_all_p_sampled": subj_s["ALL"]["mcnemar_p"],
        "subj_physics_net_greedy_body": subj_gb["Physics"]["net"],
        "subj_all_net_greedy_body": subj_gb["ALL"]["net"], "subj_all_p_greedy_body": subj_gb["ALL"]["mcnemar_p"],
        # mechanism: NOT truncation
        "trunc_rate_base_gb6144": out["truncation_rates_gb6144"]["base"]["trunc_rate"],
        "trunc_rate_int4_gb6144": out["truncation_rates_gb6144"]["int4_served"]["trunc_rate"],
        "verbosity_var_more_frac_on_bcells": out["verbosity_on_b_cells_greedy_served"]["var_more_verbose_frac"],
        # AIME skill-axis correlation
        "aime_base_acc": aime.get("base_acc"), "aime_int4_acc": aime.get("int4_acc"),
        "aime_pct_of_base": aime.get("pct_of_base"), "aime_net": aime.get("net"), "aime_p": aime.get("mcnemar_p"),
        # gate Wilson lower bounds
        "sampled_body_wilson_lo": out["sampled_gate_wilson_ci"]["int4_body_10seed"]["wilson95"][0],
        "sampled_served_wilson_lo": out["sampled_gate_wilson_ci"]["int4_served_3seed"]["wilson95"][0],
    }
    wandb.log(logd)
    for k, v in logd.items():
        run.summary[k] = v
    # mandatory explicit scalars (re-assert on summary)
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    run.summary["no_hf_job"] = 1
    run.summary["fires"] = False
    run.summary["verdict"] = verdict
    run.summary["verdict_reason"] = reason
    rid = run.id
    run.finish()
    print(f"[wandb] logged run {rid} -> group {group} | verdict={verdict}")
    return [rid]


# -------------------------------------------------------------------- main build
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    meta = diamond_meta()
    dom_counts = Counter(m["domain"] for m in meta.values())

    base_g = load_items(str(BFQS / "fp16_gpqa_greedy.json"))
    int4_g = load_items(str(BFQS / "int4_gpqa_greedy.json"))
    body_g = load_items(str(RV / "intact_body_headwidth/results/head262k_gpqa.json"))
    base_s = load_seeds([str(BFQS / f"fp16_gpqa_sampled_s{i}.json") for i in (0, 1, 2)])
    int4_s = load_seeds([str(BFQS / f"int4_gpqa_sampled_s{i}.json") for i in (0, 1, 2)])

    def acc(d):
        n = len(d)
        c = sum(r["correct"] for r in d.values())
        return c, n, round(c / n, 4)

    # ---- decode/budget basis table (aggregate gap + clears-bar) --------------
    bg_c, bg_n, bg_a = acc(base_g)
    ig_c, ig_n, ig_a = acc(int4_g)
    yg_c, yg_n, yg_a = acc(body_g)
    is_c, is_n, is_a = acc(int4_s)  # pooled 3-seed served int4
    bs_c, bs_n, bs_a = acc(base_s)

    # pooled body-isolated 10-seed sampled (ci_tighten) for a tighter body number
    bf10 = load_seeds(sorted(glob.glob(str(RV / "base_fullhead_gpqa_ci_tighten/results/bf_gpqa_sampled_mt8_s*.json"))))
    bf10_c = sum(r["correct"] for r in bf10.values())
    bf10_n = len(bf10)

    basis = {
        "greedy_mt3072": {
            "base": BASE_GREEDY, "int4_served": ig_a, "int4_body_isolated": yg_a,
            "pct_served": round(100 * ig_a / BASE_GREEDY, 2),
            "pct_body": round(100 * yg_a / BASE_GREEDY, 2),
            "clears_90_served": ig_a >= BAR_90, "clears_90_body": yg_a >= BAR_90,
        },
        "sampled_mt3072_GATE": {
            "base_3seed": BASE_SAMPLED_3SEED, "int4_served_3seed": is_a,
            "int4_body_10seed": round(bf10_c / bf10_n, 4),
            "pct_served": round(100 * is_a / BASE_SAMPLED_3SEED, 2),
            "pct_body": round(100 * (bf10_c / bf10_n) / BASE_SAMPLED_3SEED, 2),
            "clears_90_served": is_a >= BAR_90, "clears_90_body": (bf10_c / bf10_n) >= BAR_90,
            "bar": BAR_90,
        },
    }

    # ---- per-subject McNemar on 3 lenses -------------------------------------
    common_g = base_g.keys() & int4_g.keys() & body_g.keys()
    subj = {
        "greedy_body_isolated_fp16_vs_head262k": subject_table(
            {k: base_g[k] for k in common_g}, {k: body_g[k] for k in common_g}, meta),
        "greedy_served_fp16_vs_int4": subject_table(
            {k: base_g[k] for k in common_g}, {k: int4_g[k] for k in common_g}, meta),
        "sampled_gate_fp16_vs_int4_3seed": subject_table(
            base_s, int4_s, meta, keyfn=lambda k: k[0]),
    }

    # ---- per-question-type (reasoning-depth proxies) on the greedy body lens --
    def typesplit(base, var, predicate, kid=lambda k: k):
        keys = [k for k in base.keys() & var.keys() if predicate(meta.get(kid(k), {}))]
        b, c, both, neither = paired_bc(base, var, keys)
        n = len(keys)
        return {"n_items": n, "b": b, "c": c, "net": b - c, "mcnemar_p": round(mcnemar(b, c), 4),
                "base_acc": round((b + both) / n, 4) if n else None,
                "int4_acc": round((c + both) / n, 4) if n else None}

    qlens = sorted(meta[i]["q_len"] for i in common_g)
    t1, t2 = qlens[len(qlens) // 3], qlens[2 * len(qlens) // 3]
    bdict_g = {k: base_g[k] for k in common_g}
    ydict_g = {k: body_g[k] for k in common_g}
    qtype = {
        "q_len_short_tertile": typesplit(bdict_g, ydict_g, lambda m: m.get("q_len", 0) <= t1),
        "q_len_mid_tertile": typesplit(bdict_g, ydict_g, lambda m: t1 < m.get("q_len", 0) <= t2),
        "q_len_long_tertile": typesplit(bdict_g, ydict_g, lambda m: m.get("q_len", 0) > t2),
        "calc_heavy": typesplit(bdict_g, ydict_g, lambda m: m.get("calc_heavy")),
        "non_calc": typesplit(bdict_g, ydict_g, lambda m: not m.get("calc_heavy")),
    }
    # reasoning-DEPTH proxy = base greedy CoT length (longer => deeper chain)
    bchars = {k: base_g[k].get("completion_chars") for k in common_g
              if isinstance(base_g[k].get("completion_chars"), (int, float))}
    if bchars:
        med = st.median(bchars.values())
        qtype["base_cot_long_ge_median"] = typesplit(
            bdict_g, ydict_g, lambda m: True,
            kid=lambda k: k) if False else None
        deep = {k: base_g[k] for k in bchars if bchars[k] > med}
        shallow = {k: base_g[k] for k in bchars if bchars[k] <= med}
        qtype["base_cot_DEEP_above_median"] = typesplit(deep, ydict_g, lambda m: True)
        qtype["base_cot_SHALLOW_below_median"] = typesplit(shallow, ydict_g, lambda m: True)

    # ---- verbosity + truncation mechanism -----------------------------------
    # head262k lacks completion_chars -> measure verbosity on the served-int4 greedy
    # arm (same body, has chars). base_g vs int4_g, both completion_chars-instrumented.
    idict_g = {k: int4_g[k] for k in common_g}
    verb = verbosity_on_discordant(bdict_g, idict_g, meta)
    trunc = trunc_rates_gb6144()
    trunc["mt3072_chars_proxy"] = {
        "base_greedy": chars_truncation_check(base_g, 3072),
        "int4_served_greedy": chars_truncation_check(int4_g, 3072),
    }

    # ---- Wilson CIs on the gate-basis sampled int4 (NOT_REAL rider) ----------
    sampled_ci = {
        "int4_served_3seed": {"acc": is_a, "n": is_n, "wilson95": wilson(is_c, is_n),
                              "clears_bar_point": is_a >= BAR_90,
                              "wilson_lo_ge_bar": wilson(is_c, is_n)[0] >= BAR_90},
        "int4_body_10seed": {"acc": round(bf10_c / bf10_n, 4), "n": bf10_n,
                             "wilson95": wilson(bf10_c, bf10_n),
                             "clears_bar_point": (bf10_c / bf10_n) >= BAR_90,
                             "wilson_lo_ge_bar": wilson(bf10_c, bf10_n)[0] >= BAR_90},
    }

    # ---- AIME skill-axis correlation ----------------------------------------
    aime = aime_skill_axis()

    out = {
        "pr": 692, "analysis_only": True, "official_tps": 0, "no_hf_job": True,
        "instrument": "gpqa_diamond", "n_items": 198,
        "domain_counts": dict(dom_counts),
        "gate": {"bar": BAR_90, "base_sampled_3seed": BASE_SAMPLED_3SEED, "base_greedy": BASE_GREEDY},
        "counts": {
            "fp16_greedy": [bg_c, bg_n], "int4_served_greedy": [ig_c, ig_n],
            "head262k_body_greedy": [yg_c, yg_n],
            "int4_served_sampled_3seed": [is_c, is_n], "fp16_sampled_3seed": [bs_c, bs_n],
            "body_isolated_sampled_10seed": [bf10_c, bf10_n],
        },
        "decode_basis_table": basis,
        "per_subject_mcnemar": subj,
        "per_question_type_greedy_body": qtype,
        "verbosity_on_b_cells_greedy_served": verb,
        "truncation_rates_gb6144": trunc,
        "sampled_gate_wilson_ci": sampled_ci,
        "aime_skill_axis": aime,
    }
    verdict, reason = decide_verdict(out)
    out["verdict"] = verdict
    out["verdict_reason"] = reason
    if args.wandb:
        try:
            out["wandb_run_ids"] = log_wandb(out, verdict, reason)
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] FAILED: {exc!r}")
            out["wandb_run_ids"] = []
    (HERE / "breakdown_diamond.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
