#!/usr/bin/env python
"""PR #652 -- place corner C (g128 int4 body + tied-bf16 lm_head) in the recipe
2x2 and decompose the GPQA-D AR quality.

           bf16-tied head        int4-untied head
  g32 body   A=0.5056 (lawine #639 r98w09by)   B~0.520 (in flight, provisional)
  g128 body  C=THIS                            D=0.4990 (ubel #638 u13z29hs)

Corner C is measured BYTE-FOR-BYTE with corner D's harness (run_eval.py, dseed
12345, sseeds 0..9, T=1/top_p=0.95/top_k=64, MT=6144, min_tokens=8, vLLM 0.22.0,
BI=1) so:
  * head effect @ g128 = C - D   (isolates tied-bf16 vs untied-int4 lm_head)
  * body effect @ bf16head = A - C   (isolates g32 vs g128 body)
Pooled n=1980 (10 seeds x 198), Wilson 95% CI, two-proportion z-test on each
contrast. Also reports the AR-vs-spec gap (int4-AR 0.4990 vs Option-B-spec 0.4652)
to resolve the mixed-denominator subtlety the card flags.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- fixed anchors (from the PR #652 card) ----------------------------------
BASE_GPQA_SAMPLED = 0.5404          # base GPQA-D sampled (ilg4z6e9); 90% bar below
BAR_90 = round(0.9 * BASE_GPQA_SAMPLED, 4)   # 0.4864
A_LAWINE_639 = 0.5056               # corner A: g32+bf16-tied (lawine #639 r98w09by)
D_UBEL_638 = 0.4990                 # corner D: g128+int4-untied AR (ubel #638 u13z29hs)
B_LAWINE_639 = 0.520                # corner B: g32+int4-untied (in flight; provisional)
OPTIONB_SPEC_629 = 0.4652           # Option-B WITH spec (fern #629 2jhhk0u3) -- AR-vs-spec gap only
N_SAMPLED_REF = 1980                # 10-seed n for reconstructing A/D point CIs
PPL_C = 2.0171                      # corner C PPL (wirbel #649 a09npwda)
PPL_D = 2.0197                      # corner D PPL (ubel #638)


def _wilson(n_correct: int, n: int, z: float = 1.96):
    if not n:
        return float("nan"), float("nan")
    p = n_correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return center - half, center + half


def _two_prop_z(c1: int, n1: int, c2: int, n2: int):
    """Unpaired two-proportion z-test (pooled SE). Returns (diff, z, p_two_sided)."""
    if not (n1 and n2):
        return float("nan"), float("nan"), float("nan")
    p1, p2 = c1 / n1, c2 / n2
    p = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, float("inf"), 0.0
    z = (p1 - p2) / se
    # normal two-sided p via erfc
    p_two = math.erfc(abs(z) / math.sqrt(2))
    return p1 - p2, z, p_two


def pool_sampled(files: list[str]) -> dict:
    per_seed = []
    cor = sc = err = trunc = empty = total = 0
    for fn in files:
        d = json.loads(Path(fn).read_text())
        ps = d.get("per_sample") or []
        s_sc = sum(1 for r in ps if r.get("value") in ("C", "I"))
        s_cor = sum(1 for r in ps if r.get("correct"))
        s_err = sum(1 for r in ps if r.get("error"))
        s_tr = sum(1 for r in ps if r.get("truncated"))
        s_emp = sum(1 for r in ps if r.get("empty"))
        per_seed.append({"sampling_seed": d.get("sampling_seed"), "n_scored": s_sc,
                         "n_correct": s_cor, "n_error": s_err,
                         "accuracy": (s_cor / s_sc) if s_sc else float("nan")})
        cor += s_cor; sc += s_sc; err += s_err; trunc += s_tr; empty += s_emp; total += len(ps)
    lo, hi = _wilson(cor, sc)
    return {"n_seeds": len(files), "n_scored": sc, "n_correct": cor, "n_error": err,
            "n_samples": total, "accuracy": (cor / sc) if sc else float("nan"),
            "ci95_lo": lo, "ci95_hi": hi,
            "finish_length_rate": (trunc / total) if total else float("nan"),
            "empty_rate": (empty / total) if total else float("nan"),
            "per_seed": per_seed}


def load_greedy(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    cor, sc = d.get("n_correct"), d.get("n_scored")
    lo, hi = _wilson(cor or 0, sc or 0)
    return {"accuracy": d.get("accuracy"), "n_correct": cor, "n_scored": sc,
            "ci95_lo": lo, "ci95_hi": hi,
            "finish_length_rate": d.get("finish_length_rate"),
            "empty_rate": d.get("empty_rate"),
            "ctok_p95": d.get("completion_tokens_p95"),
            "ctok_mean": d.get("completion_tokens_mean")}


def build(res_c: Path, res_d_glob: str) -> dict:
    c_files = sorted(glob.glob(str(res_c / "cornerc_gpqa_sampled_s*.json")))
    if not c_files:
        raise SystemExit(f"no corner-C sampled JSONs in {res_c}")
    C = pool_sampled(c_files)
    Cg = load_greedy(res_c / "cornerc_gpqa_greedy.json")

    # corner D pooled from ubel #638's per-seed JSONs (same tree, same harness)
    d_files = sorted(glob.glob(res_d_glob))
    D = pool_sampled(d_files) if d_files else None

    c_cor, c_n = C["n_correct"], C["n_scored"]
    # contrasts (sampled, pooled)
    contrasts = {}
    if D:
        diff, z, p = _two_prop_z(c_cor, c_n, D["n_correct"], D["n_scored"])
        contrasts["head_effect_at_g128_C_minus_D"] = {
            "C": C["accuracy"], "D": D["accuracy"], "diff": diff, "z": z, "p_two_sided": p,
            "D_source": "ubel #638 u13z29hs (pooled here)", "D_acc_pooled": D["accuracy"]}
    # A is a point estimate (lawine #639); reconstruct its count at n=1980 for the test
    a_cor = round(A_LAWINE_639 * N_SAMPLED_REF)
    diff, z, p = _two_prop_z(a_cor, N_SAMPLED_REF, c_cor, c_n)
    contrasts["body_effect_at_bf16head_A_minus_C"] = {
        "A": A_LAWINE_639, "C": C["accuracy"], "diff": diff, "z": z, "p_two_sided": p,
        "A_source": "lawine #639 r98w09by (point 0.5056, n~1980 reconstructed)"}
    # cross-checks (provisional, B in flight)
    b_cor = round(B_LAWINE_639 * N_SAMPLED_REF)
    contrasts["xcheck_head_effect_at_g32_A_minus_B_provisional"] = {
        "A": A_LAWINE_639, "B": B_LAWINE_639, "diff": round(A_LAWINE_639 - B_LAWINE_639, 4),
        "note": "B in flight @9/10 seeds; provisional"}
    if D:
        contrasts["xcheck_body_effect_at_int4_B_minus_D_provisional"] = {
            "B": B_LAWINE_639, "D": D["accuracy"], "diff": round(B_LAWINE_639 - D["accuracy"], 4),
            "note": "B provisional"}

    # AR-vs-spec gap (NOT a recipe contrast): the spec-specific GPQA deficit
    ar_vs_spec = {"int4_AR_D": D_UBEL_638, "optionb_spec_629": OPTIONB_SPEC_629,
                  "gap_AR_minus_spec": round(D_UBEL_638 - OPTIONB_SPEC_629, 4),
                  "note": "int4-AR 0.4990 vs Option-B-WITH-SPEC 0.4652; the spec effect, "
                          "NOT the body/head recipe. 0.4652 is the WRONG denominator for the 2x2."}

    # %-of-base + pass/fail
    pct_of_base = round(100.0 * C["accuracy"] / BASE_GPQA_SAMPLED, 2)
    clears_bar = bool(C["accuracy"] >= BAR_90)

    # ---- verdict ------------------------------------------------------------
    # A (0.5056) and D (0.4990) are only 0.0066 apart -- statistically one cluster
    # at n=1980 (Wilson half-width ~0.022). The verdict turns on where C lands
    # relative to that cluster and whether C is DISTINGUISHABLE from A and from D.
    c_acc = C["accuracy"]
    d_acc = D["accuracy"] if D else D_UBEL_638
    sig = 0.05
    p_CD = contrasts.get("head_effect_at_g128_C_minus_D", {}).get("p_two_sided", float("nan"))
    p_AC = contrasts["body_effect_at_bf16head_A_minus_C"]["p_two_sided"]
    dist_from_D = not (p_CD != p_CD) and p_CD < sig     # C significantly != D
    dist_from_A = not (p_AC != p_AC) and p_AC < sig     # C significantly != A
    near_A = abs(c_acc - A_LAWINE_639) <= 0.02
    near_D = abs(c_acc - d_acc) <= 0.02

    if (not dist_from_A) and (not dist_from_D):
        verdict = "RECIPE_EFFECT_IS_SMALL_AR"
        verdict_reason = (f"C={c_acc:.4f} is statistically indistinguishable from BOTH "
                          f"A={A_LAWINE_639} (p={p_AC:.3f}) and D={d_acc:.4f} (p={p_CD:.3f}); "
                          f"all three AR corners cluster ~0.50, recipe barely moves AR-GPQA.")
    elif dist_from_D and near_A and not dist_from_A:
        verdict = "QUALITY_IS_HEAD_DRIVEN"
        verdict_reason = (f"C={c_acc:.4f} recovers to A={A_LAWINE_639} (p={p_AC:.3f}, n.s.) and is "
                          f"significantly above D={d_acc:.4f} (p={p_CD:.3f}); the bf16 head drives recovery.")
    elif dist_from_A and near_D and not dist_from_D:
        verdict = "QUALITY_IS_BODY_DRIVEN"
        verdict_reason = (f"C={c_acc:.4f} stays at D={d_acc:.4f} (p={p_CD:.3f}, n.s.) and is "
                          f"significantly below A={A_LAWINE_639} (p={p_AC:.3f}); the g32 body drives recovery.")
    else:
        verdict = "QUALITY_IS_JOINT"
        verdict_reason = (f"C={c_acc:.4f} is intermediate: dist_from_A={dist_from_A} "
                          f"(p={p_AC:.3f}), dist_from_D={dist_from_D} (p={p_CD:.3f}).")

    return {
        "card": "corner_c_gpqa_headbody (PR #652)",
        "engine": "vllm==0.22.0",
        "config": {"max_model_len": 8192, "max_tokens": 6144, "min_tokens": 8,
                   "max_num_seqs": 16, "max_connections": 16, "vllm_batch_invariant": 1,
                   "flashinfer_sampler": 0,
                   "model": "g128_bf16head (W4A16 g128 body + TIED bf16 lm_head, AR no-spec, PPL 2.0171)",
                   "gpqa_sampled": "10-seed n=1980 dseed12345 sseeds0..9 T=1/top_p=0.95/top_k=64",
                   "conc_note": "seqs=16/conc=16 matches corner D's server byte-for-byte (only the checkpoint differs), so C-D is a single-variable lm_head contrast. conc16 vs conc1 is NOT bit-identical under BI=1 (token streams drift -- prefix-cache/reduction-order), but scored GPQA answers are conc-invariant (smoke: 8/8 identical answers, 4/4 correct at c1 and c16)"},
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "corner_c": {
            "ppl": PPL_C,
            "gpqa_sampled": {"accuracy": c_acc, "n_scored": c_n, "n_correct": c_cor,
                             "ci95_lo": C["ci95_lo"], "ci95_hi": C["ci95_hi"],
                             "finish_length_rate": C["finish_length_rate"],
                             "empty_rate": C["empty_rate"], "per_seed": C["per_seed"]},
            "gpqa_greedy": Cg,
            "pct_of_base": pct_of_base, "base": BASE_GPQA_SAMPLED,
            "bar_90pct": BAR_90, "clears_bar": clears_bar,
        },
        "two_by_two": {
            "A_g32_bf16tied": A_LAWINE_639, "B_g32_int4untied": B_LAWINE_639,
            "C_g128_bf16tied": c_acc, "D_g128_int4untied": d_acc,
            "ppl_C": PPL_C, "ppl_D": PPL_D,
        },
        "contrasts": contrasts,
        "ar_vs_spec_gap": ar_vs_spec,
        "corner_d_pooled": D,
    }


def log_wandb(summary: dict) -> list:
    import wandb
    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    group = os.environ.get("WANDB_GROUP", "corner-c-gpqa-headbody-wirbel")
    c = summary["corner_c"]
    common = {"analysis_only": True, "official_tps": 0, "engine": summary["engine"],
              **summary["config"], "pr": 652, "student": "wirbel"}
    run = wandb.init(project=project, entity=entity, group=group,
                     name="wirbel/cornerc-gpqa-headbody", job_type="cornerc-gpqa-eval",
                     reinit=True, config=common)
    logd = {
        "gpqa_sampled_acc": c["gpqa_sampled"]["accuracy"],
        "gpqa_sampled_ci95_lo": c["gpqa_sampled"]["ci95_lo"],
        "gpqa_sampled_ci95_hi": c["gpqa_sampled"]["ci95_hi"],
        "gpqa_sampled_n_scored": c["gpqa_sampled"]["n_scored"],
        "gpqa_sampled_finish_length_rate": c["gpqa_sampled"]["finish_length_rate"],
        "gpqa_sampled_empty_rate": c["gpqa_sampled"]["empty_rate"],
        "pct_of_base": c["pct_of_base"], "bar_90pct": c["bar_90pct"],
        "clears_bar": int(c["clears_bar"]), "ppl": c["ppl"],
        "A_g32_bf16tied": summary["two_by_two"]["A_g32_bf16tied"],
        "D_g128_int4untied": summary["two_by_two"]["D_g128_int4untied"],
        "head_effect_C_minus_D": summary["contrasts"].get("head_effect_at_g128_C_minus_D", {}).get("diff"),
        "head_effect_C_minus_D_p": summary["contrasts"].get("head_effect_at_g128_C_minus_D", {}).get("p_two_sided"),
        "body_effect_A_minus_C": summary["contrasts"]["body_effect_at_bf16head_A_minus_C"]["diff"],
        "body_effect_A_minus_C_p": summary["contrasts"]["body_effect_at_bf16head_A_minus_C"]["p_two_sided"],
        "ar_vs_spec_gap": summary["ar_vs_spec_gap"]["gap_AR_minus_spec"],
    }
    if c["gpqa_greedy"]:
        logd["gpqa_greedy_acc"] = c["gpqa_greedy"]["accuracy"]
        logd["gpqa_greedy_ci95_lo"] = c["gpqa_greedy"]["ci95_lo"]
        logd["gpqa_greedy_ci95_hi"] = c["gpqa_greedy"]["ci95_hi"]
    wandb.log(logd)
    for k, v in logd.items():
        run.summary[k] = v
    run.summary["verdict"] = summary["verdict"]
    rid = run.id
    run.finish()
    print(f"[wandb] logged run {rid} -> group {group}")
    return [rid]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--corner-d-glob", default=str(
        HERE.parent / "optionb_denom_0p22_gb6144/results_int4ar/int4ar_gpqa_sampled_s*.json"))
    ap.add_argument("--out", type=Path, default=HERE / "summary.json")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    summary = build(args.results_dir, args.corner_d_glob)
    ids = []
    if args.wandb:
        try:
            ids = log_wandb(summary)
        except Exception as exc:
            print(f"[wandb] FAILED: {exc!r}")
    summary["wandb_run_ids"] = ids
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"verdict": summary["verdict"], "verdict_reason": summary["verdict_reason"],
                      "corner_c_gpqa_sampled": summary["corner_c"]["gpqa_sampled"]["accuracy"],
                      "ci95": [summary["corner_c"]["gpqa_sampled"]["ci95_lo"],
                               summary["corner_c"]["gpqa_sampled"]["ci95_hi"]],
                      "pct_of_base": summary["corner_c"]["pct_of_base"],
                      "clears_bar": summary["corner_c"]["clears_bar"],
                      "two_by_two": summary["two_by_two"],
                      "contrasts": {k: {kk: vv for kk, vv in v.items() if kk in ("diff", "p_two_sided")}
                                    for k, v in summary["contrasts"].items()}}, indent=2))
    print(f"[aggregate] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
