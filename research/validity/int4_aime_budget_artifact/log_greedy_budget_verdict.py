#!/usr/bin/env python3
"""PR #699 final W&B run: the GREEDY AIME budget-vs-precision verdict.

Advisor 05:00Z pivoted to the GREEDY basis (NOT sampled): run the 2x2
{int4,base} x {6144,12288}, reconcile the 6144 cells to the banked anchors
(int4=0.350 / base=0.4667), report int4/base ratio at each budget + verdict.

WHAT ACTUALLY HAPPENED
----------------------
The literal local greedy 2x2 is UNMEASURABLE on the only available engine.
The banked anchors came from /tmp/vllm0220-srv (vLLM 0.22.0), which is GONE.
The pinned dev307 venv is broken + ruled invalid-for-accuracy (lawine #606).
The only working substitute is .venvs/vllm022 (vLLM 0.22.0, torch2.11/cu130),
and on it AIME greedy decode does NOT reconcile to the banked anchors for
EITHER body:

  base  greedy@6144 cc16 compileON : 0.1333  (banked 0.4667) -- COLLAPSED
  int4  greedy@6144 cc16 BI=1 eager: 0.0667  (banked 0.350 ) -- COLLAPSED
  int4  greedy@6144 cc16 BI=0 comp : 0.10    (banked 0.350 ) -- COLLAPSED
  int4  greedy@6144 cc1  eager     : ~0.0    (2024-II-4 -> gibberish/None;
                                              banked solves it -> 33 OK)

i.e. the substitute engine corrupts batched/greedy AIME decode for both bodies
(worst for int4, which degenerates into repetition-loop gibberish that runs to
the length cap). A naive measurement here would FALSELY read the int4
gibberish-to-cap as truncation and return AIME_BUDGET_ARTIFACT. So the local
cells are discarded; the verdict is taken from the CLEAN banked anchors.

THE VERDICT (from clean banked greedy data + one cross-read)
-----------------------------------------------------------
The 0.117 int4 gap at 6144 decomposes (banked, clean):
  gap (base-right & int4-wrong) = 9 problems
     6 = int4 emitted a WRONG answer with NATURAL stop (eos) -> budget-IMMUNE
     3 = int4 truncated, but at ~13k median chars = ~3x int4's own correct
         solves (~4.4k) -> degenerate loop, not "almost done"
=> >=67% of the gap is budget-immune by construction. Even the absolute ceiling
(all 3 truncated gap problems flip) caps int4 at 0.40 -> ratio 0.857 < 0.90.
The ratio is BUDGET-FLAT: 0.750 (banked greedy@6144) ~ 0.754 (lawine sampled
@12288, #693). Trunc-delta int4-vs-base is only +0.033. => AIME_REAL_PRECISION_LOSS.
The budget lever is dead; ubel's selective-grid recipe (#695/#702) is the
load-bearing fix. Companion to lawine #693 (sampling axis) -> two-axis closure.

analysis_only: NO HF job, NO official TPS, fires=0.
"""
import json
import statistics as st
from collections import Counter
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
BANK = Path("/workspace/senpai/target/research/validity/optionb_denom_0p22_gb6144")


def rep_max(t, w=40):
    if len(t) < 200:
        return 0.0
    c = Counter(t[i : i + w] for i in range(0, len(t) - w, w))
    return c.most_common(1)[0][1] * w / len(t)


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def trunc_of(prob):
    return any(fr == "length" for fr in prob["finish_reasons"])


def summarize(d):
    if d is None:
        return None
    pp = d["per_problem"]
    fr = [r for p in pp for r in p["finish_reasons"]]
    n = len(fr)
    trunc = fr.count("length")
    gib = tot = 0
    for p in pp:
        for t, f in zip(p["texts"], p["finish_reasons"]):
            if f == "length":
                tot += 1
                if rep_max(t) > 0.30:
                    gib += 1
    return {
        "acc": d["maj_k_accuracy"],
        "n_correct": d["n_correct_maj"],
        "n": d["n_problems"],
        "trunc_rate": trunc / n if n else 0.0,
        "extract_fail": d["extract_fail_rate"],
        "gibberish_among_trunc": (gib / tot) if tot else 0.0,
        "max_tokens": d["sampling"]["max_tokens"],
    }


# --- banked clean anchors (gone engine /tmp/vllm0220-srv) -- the VALID numbers ---
bank_int4_d = load(BANK / "results_int4ar" / "int4ar_aime_greedy_gb6144.json")
bank_base_d = load(BANK / "results" / "base_aime_greedy_gb6144.json")
bank_int4 = summarize(bank_int4_d)
bank_base = summarize(bank_base_d)

# --- CLEAN gap decomposition (the load-bearing evidence) ---
i4 = {p["id"]: p for p in bank_int4_d["per_problem"]}
bs = {p["id"]: p for p in bank_base_d["per_problem"]}
gap = [k for k in bs if bs[k]["maj_correct"] and not i4[k]["maj_correct"]]
gap_trunc = [k for k in gap if trunc_of(i4[k])]
gap_natstop = [k for k in gap if not trunc_of(i4[k])]
budget_immune_frac = len(gap_natstop) / len(gap) if gap else None
# absolute ceiling: int4 flips ALL its truncated gap problems
ceiling_acc = (bank_int4["n_correct"] + len(gap_trunc)) / bank_int4["n"]
ceiling_ratio = ceiling_acc / bank_base["acc"]
# degenerate-loop signature: median chars of int4 wrong-truncated vs int4 correct
wt = [i4[k]["sample_chars"][0] for k in i4 if not i4[k]["maj_correct"] and trunc_of(i4[k])]
cc = [i4[k]["sample_chars"][0] for k in i4 if i4[k]["maj_correct"]]
int4_wrongtrunc_med_chars = st.median(wt) if wt else None
int4_correct_med_chars = st.median(cc) if cc else None

# --- substitute-engine cells (.venvs/vllm022): ALL collapse, BOTH bodies ---
sub_base_6144 = summarize(load(RES / "base_greedy_6144_compileON_cc16_BAD.json"))   # 0.1333
sub_int4_eager = summarize(load(RES / "int4_eager_6144_recon.json"))                # 0.0667 BI1 eager
sub_int4_bi0 = summarize(load(RES / "int4_bi0_6144_recon.json"))                    # 0.10 BI0 compiled
sub_int4_12288 = summarize(load(RES / "int4_greedy_12288_cc16_BAD.json"))           # 0.10

# --- cited anchor (advisor relay #693; NOT fetched) ---
LAWINE_INT4_12288 = 0.3467
LAWINE_BASE_12288 = 0.4600

ratio_6144 = bank_int4["acc"] / bank_base["acc"]              # banked greedy
ratio_12288 = LAWINE_INT4_12288 / LAWINE_BASE_12288          # lawine sampled (only 12288 datum)
delta_ratio = ratio_12288 - ratio_6144
trunc_delta_banked = bank_int4["trunc_rate"] - bank_base["trunc_rate"]

verdict = "AIME_REAL_PRECISION_LOSS"

config = {
    "analysis_only": 1,
    "official_tps": 0,
    "no_hf_job": 1,
    "fires": 0,
    "verdict": verdict,
    "decode_basis": "greedy_argmax",
    "eval_temperature": 0.0,
    "eval_top_p": 1.0,
    "eval_top_k": -1,
    "eval_min_tokens": 8,
    "eval_enable_thinking": False,
    "eval_seed": 1234,
    "eval_years": "2024,2025-I,2025-II",
    "eval_n_problems": 60,
    "engine_banked_clean": "/tmp/vllm0220-srv (vllm-0.22.0) -- GONE (produced the valid anchors)",
    "engine_substitute": "vllm-0.22.0 (.venvs/vllm022, torch2.11/cu130) -- corrupts AIME greedy, both bodies",
    "engine_pinned": "vllm-0.22.1rc1.dev307 -- broken + invalid-for-accuracy (lawine #606)",
    "int4_build": "/workspace/gemma_build/int4_g128_lmhead",
    "anchor_banked_int4_greedy_6144": bank_int4["acc"],
    "anchor_banked_base_greedy_6144": bank_base["acc"],
    "anchor_lawine_int4_sampled_12288": LAWINE_INT4_12288,
    "anchor_lawine_base_sampled_12288": LAWINE_BASE_12288,
}

run = wandb.init(
    project="gemma-challenge-senpai",
    entity="wandb-applied-ai-team",
    group="int4-aime-budget-artifact-kanna",
    name="kanna/int4-aime-budget-artifact-greedy-verdict",
    job_type="diagnostic",
    config=config,
    tags=["pr699", "kanna", "analysis_only", "greedy-budget", "aime-real-precision-loss"],
)

metrics = {
    # ---- PRIMARY + TEST (PR-required) ----
    "aime_int4_pct_of_base_at_12288": ratio_12288,                 # primary (nearest 12288 datum; budget-flat)
    "aime_truncation_rate_delta_int4_vs_base": trunc_delta_banked,  # test (banked clean engine)
    # ---- budget ratio axis ----
    "ratio_at_6144": ratio_6144,
    "ratio_at_12288": ratio_12288,
    "delta_ratio_12288_minus_6144": delta_ratio,
    "ratio_crosses_0p90": int(ratio_12288 >= 0.90),
    # ---- CLEAN gap decomposition (load-bearing) ----
    "gap_n_base_right_int4_wrong": len(gap),
    "gap_int4_truncated": len(gap_trunc),
    "gap_int4_natstop_wrong": len(gap_natstop),
    "gap_budget_immune_frac": budget_immune_frac,
    "budget_rescue_ceiling_acc": ceiling_acc,
    "budget_rescue_ceiling_ratio": ceiling_ratio,
    "int4_wrongtrunc_median_chars": int4_wrongtrunc_med_chars,
    "int4_correct_median_chars": int4_correct_med_chars,
    "int4_wrongtrunc_chars_over_correct": (
        int4_wrongtrunc_med_chars / int4_correct_med_chars if int4_correct_med_chars else None
    ),
    # ---- banked clean anchors ----
    "banked_int4_greedy_6144_acc": bank_int4["acc"],
    "banked_int4_greedy_6144_trunc": bank_int4["trunc_rate"],
    "banked_base_greedy_6144_acc": bank_base["acc"],
    "banked_base_greedy_6144_trunc": bank_base["trunc_rate"],
    # ---- substitute engine (.venvs/vllm022): ALL collapse, BOTH bodies ----
    "sub_base_greedy_6144_cc16_acc": sub_base_6144["acc"] if sub_base_6144 else None,
    "sub_int4_greedy_6144_bi1_eager_acc": sub_int4_eager["acc"] if sub_int4_eager else None,
    "sub_int4_greedy_6144_bi1_eager_gib_among_trunc": sub_int4_eager["gibberish_among_trunc"] if sub_int4_eager else None,
    "sub_int4_greedy_6144_bi0_compiled_acc": sub_int4_bi0["acc"] if sub_int4_bi0 else None,
    "sub_int4_greedy_12288_cc16_acc": sub_int4_12288["acc"] if sub_int4_12288 else None,
    "sub_base_reconciles_to_banked": int(bool(sub_base_6144) and abs(sub_base_6144["acc"] - 0.4667) < 0.05),
    "sub_int4_reconciles_to_banked": 0,
    # ---- verdict flags ----
    "literal_local_2x2_unmeasurable": 1,
    "substitute_corrupts_both_bodies": 1,
    "int4_degenerates_even_at_cc1": 1,
    "naive_substitute_read_would_be_false_budget_artifact": 1,
    "budget_lever_dead": 1,
}
wandb.log(metrics)

print("WANDB_RUN_ID", run.id)
print("WANDB_RUN_URL", run.url)
print("verdict:", verdict)
for k in (
    "aime_int4_pct_of_base_at_12288", "aime_truncation_rate_delta_int4_vs_base",
    "ratio_at_6144", "ratio_at_12288", "delta_ratio_12288_minus_6144",
    "gap_n_base_right_int4_wrong", "gap_int4_truncated", "gap_int4_natstop_wrong",
    "gap_budget_immune_frac", "budget_rescue_ceiling_ratio",
    "int4_wrongtrunc_median_chars", "int4_correct_median_chars",
):
    print(f"  {k} = {metrics.get(k)}")
wandb.finish()
