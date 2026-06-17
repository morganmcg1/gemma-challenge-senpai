#!/usr/bin/env python3
"""Decompose the osoi5 body collapse into its two confounded knobs via a served
2x2 (layers x precision), ALL CORNERS AT THE FULL 262k HEAD (PR #538).

Corners (greedy, inspect_evals MMLU-Pro n=500 seed=12345 / GPQA-Diamond 198
seed=12345, byte-identical prompts via the shared run_eval.py):

           bf16                       int4
  42L   A 42L-bf16  (RUN, this PR)   B 42L-int4  (cite #511/#527 base, genuine)
  37L   C 37L-bf16  (RUN, this PR)   D 37L-int4  (cite #527 full_pck04, see caveat)

  A,C are measured fresh here (genuine full bf16 tied head; C carved from stock
  bf16 to osoi5's EXACT geometry by build_37L_bf16.py). B is the genuine stock
  int4 QAT checkpoint (42L, full tied head). D is #527's *synthesized* full-262k
  head -- present rows bit-exact, tail rows int4(embed[t]); a faithful int4-head
  UPPER-BOUND proxy (tail cos~0.990), not a from-scratch genuine int4 carve.
  That removes the head-prune confound (the whole point of "full head"), so D is
  fit for the body-knob isolation; the proxy caveat is carried into the verdict.

Required PR outputs (per task + summarized):
  int4_cost_fullbody  = A - B        (cost of int4 quant on the full 42L body)
  layerdrop_cost_bf16 = A - C        (cost of dropping 5 layers at bf16)
  interaction         = B + C - A - D
  dominant_body_knob in {int4, layer-removal, both}  (per task)
  base_int4_clears_gate (B vs gate 0.60/0.42)
  one-line verdict.

Every number is provenance-backed by a real run_eval.py JSON (the two int4
corners point at the #511/#527 artifacts), and prompt_sha is cross-checked across
all four corners so the decomposition is on byte-identical prompts.
"""
import argparse
import json
import math
import os
import sys

GATE_MMLU = 0.60
GATE_GPQA = 0.42
# comparability epsilon for "dominant knob": if the two costs are within EPS of
# each other (and both materially > EPS), call it "both".
EPS = 0.05


def _load(path):
    with open(path) as f:
        return json.load(f)


def _ci95(p, n):
    if not n:
        return (float("nan"), float("nan"))
    h = 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def _shamap(d):
    return {r["id"]: r.get("prompt_sha") for r in d["per_sample"]}


def _corner(path):
    d = _load(path)
    return {
        "path": path,
        "acc": d["accuracy"],
        "n_scored": d["n_scored"],
        "n_correct": d["n_correct"],
        "n_error": d.get("n_error", 0),
        "sha": _shamap(d),
    }


def _prompt_identity(corners):
    """All corners must share identical prompt_sha per id (byte-identical harness)."""
    ids = None
    ref = None
    mism = 0
    for c in corners:
        s = c["sha"]
        if ref is None:
            ref = s
            ids = set(s)
            continue
        common = ids & set(s)
        mism += sum(1 for i in common if ref[i] != s[i])
        ids &= set(s)
    return mism == 0, mism, (len(ids) if ids else 0)


def _dominant(int4_cost, layerdrop_cost):
    a, b = abs(int4_cost), abs(layerdrop_cost)
    if a < EPS and b < EPS:
        return "neither"  # neither knob materially costs (shouldn't happen here)
    if abs(a - b) <= EPS:
        return "both"
    return "int4" if a > b else "layer-removal"


def _task_block(name, A, B, C, D):
    a, b, c, d = A["acc"], B["acc"], C["acc"], D["acc"]
    int4_cost_fullbody = a - b
    layerdrop_cost_bf16 = a - c
    int4_cost_reducedbody = c - d
    layerdrop_cost_int4 = b - d
    interaction = b + c - a - d
    dom = _dominant(int4_cost_fullbody, layerdrop_cost_bf16)
    gate = GATE_MMLU if name == "mmlu_pro" else GATE_GPQA
    return {
        "task": name,
        "gate_threshold": gate,
        "corner_42L_bf16": a,
        "corner_42L_int4": b,
        "corner_37L_bf16": c,
        "corner_37L_int4": d,
        "ci95_42L_bf16": _ci95(a, A["n_scored"]),
        "ci95_37L_bf16": _ci95(c, C["n_scored"]),
        "ci95_42L_int4": _ci95(b, B["n_scored"]),
        "ci95_37L_int4": _ci95(d, D["n_scored"]),
        "int4_cost_fullbody": int4_cost_fullbody,
        "layerdrop_cost_bf16": layerdrop_cost_bf16,
        "int4_cost_reducedbody": int4_cost_reducedbody,
        "layerdrop_cost_int4": layerdrop_cost_int4,
        "interaction": interaction,
        "dominant_body_knob": dom,
        "gate_pass_42L_bf16": bool(a >= gate),
        "gate_pass_42L_int4": bool(b >= gate),
        "gate_pass_37L_bf16": bool(c >= gate),
        "gate_pass_37L_int4": bool(d >= gate),
        "n_42L_bf16": A["n_scored"], "n_37L_bf16": C["n_scored"],
        "n_42L_int4": B["n_scored"], "n_37L_int4": D["n_scored"],
        "err_42L_bf16": A["n_error"], "err_37L_bf16": C["n_error"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c42-bf16-mmlu", required=True)
    ap.add_argument("--c42-bf16-gpqa", required=True)
    ap.add_argument("--c37-bf16-mmlu", required=True)
    ap.add_argument("--c37-bf16-gpqa", required=True)
    ap.add_argument("--c42-int4-mmlu", required=True)
    ap.add_argument("--c42-int4-gpqa", required=True)
    ap.add_argument("--c37-int4-mmlu", required=True)
    ap.add_argument("--c37-int4-gpqa", required=True)
    ap.add_argument("--out", default="aggregate_2x2.json")
    ap.add_argument("--wandb_name", default="ubel/body-decomp-served-2x2")
    ap.add_argument("--wandb_group", default="body-decomp-served-2x2")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    A_m, A_g = _corner(a.c42_bf16_mmlu), _corner(a.c42_bf16_gpqa)
    C_m, C_g = _corner(a.c37_bf16_mmlu), _corner(a.c37_bf16_gpqa)
    B_m, B_g = _corner(a.c42_int4_mmlu), _corner(a.c42_int4_gpqa)
    D_m, D_g = _corner(a.c37_int4_mmlu), _corner(a.c37_int4_gpqa)

    mmlu_ident, mmlu_mism, mmlu_common = _prompt_identity([A_m, B_m, C_m, D_m])
    gpqa_ident, gpqa_mism, gpqa_common = _prompt_identity([A_g, B_g, C_g, D_g])

    mmlu = _task_block("mmlu_pro", A_m, B_m, C_m, D_m)
    gpqa = _task_block("gpqa_diamond", A_g, B_g, C_g, D_g)
    mmlu["prompt_identical_all4"] = mmlu_ident
    mmlu["prompt_mismatch"] = mmlu_mism
    gpqa["prompt_identical_all4"] = gpqa_ident
    gpqa["prompt_mismatch"] = gpqa_mism

    base_int4_clears_gate = bool(
        B_m["acc"] >= GATE_MMLU and B_g["acc"] >= GATE_GPQA
    )

    # one-line verdict: name the dominant knob per axis + where the collapse lives.
    def fmt(x):
        return f"{x:+.3f}"

    verdict = (
        f"FULL-HEAD body 2x2: int4-on-42L costs MMLU {fmt(mmlu['int4_cost_fullbody'])}/"
        f"GPQA {fmt(gpqa['int4_cost_fullbody'])}; layerdrop(-5L)@bf16 costs MMLU "
        f"{fmt(mmlu['layerdrop_cost_bf16'])}/GPQA {fmt(gpqa['layerdrop_cost_bf16'])}; "
        f"interaction MMLU {fmt(mmlu['interaction'])}/GPQA {fmt(gpqa['interaction'])}. "
        f"Dominant knob MMLU={mmlu['dominant_body_knob']}, GPQA={gpqa['dominant_body_knob']}. "
        f"base_int4 (42L) {'CLEARS' if base_int4_clears_gate else 'FAILS'} the gate "
        f"({B_m['acc']:.3f}/{B_g['acc']:.3f} vs {GATE_MMLU}/{GATE_GPQA})."
    )

    report = {
        "pr": 538,
        "analysis_only": True, "no_hf_job": True,
        "no_served_file_change": True, "official_tps": 0,
        "gate_mmlu_threshold": GATE_MMLU, "gate_gpqa_threshold": GATE_GPQA,
        # PR-required scalar outputs (named exactly):
        "base_int4_mmlu_pro": B_m["acc"],
        "base_int4_gpqa": B_g["acc"],
        "bf16_37L_mmlu_pro": C_m["acc"],
        "bf16_37L_gpqa": C_g["acc"],
        "bf16_42L_mmlu_pro": A_m["acc"],
        "bf16_42L_gpqa": A_g["acc"],
        "int4_37L_mmlu_pro": D_m["acc"],
        "int4_37L_gpqa": D_g["acc"],
        "int4_cost_fullbody_mmlu": mmlu["int4_cost_fullbody"],
        "int4_cost_fullbody_gpqa": gpqa["int4_cost_fullbody"],
        "layerdrop_cost_bf16_mmlu": mmlu["layerdrop_cost_bf16"],
        "layerdrop_cost_bf16_gpqa": gpqa["layerdrop_cost_bf16"],
        "interaction_mmlu": mmlu["interaction"],
        "interaction_gpqa": gpqa["interaction"],
        "dominant_body_knob_mmlu": mmlu["dominant_body_knob"],
        "dominant_body_knob_gpqa": gpqa["dominant_body_knob"],
        "base_int4_clears_gate": base_int4_clears_gate,
        "verdict": verdict,
        "prompt_identical_all4_mmlu": mmlu_ident,
        "prompt_identical_all4_gpqa": gpqa_ident,
        "mmlu_detail": mmlu,
        "gpqa_detail": gpqa,
        "corner_sources": {
            "42L_bf16": {"mmlu": a.c42_bf16_mmlu, "gpqa": a.c42_bf16_gpqa,
                         "note": "RUN this PR: stock bf16 snapshot, 42L, full tied bf16 head"},
            "42L_int4": {"mmlu": a.c42_int4_mmlu, "gpqa": a.c42_int4_gpqa,
                         "note": "CITE #511/#527 base: stock int4 QAT, 42L, full tied head (genuine)"},
            "37L_bf16": {"mmlu": a.c37_bf16_mmlu, "gpqa": a.c37_bf16_gpqa,
                         "note": "RUN this PR: build_37L_bf16.py carve of stock bf16 to osoi5 geometry, full tied bf16 head"},
            "37L_int4": {"mmlu": a.c37_int4_mmlu, "gpqa": a.c37_int4_gpqa,
                         "note": "CITE #527 full_pck04: osoi5 int4 body + SYNTHESIZED full-262k int4 head (proxy, tail cos~0.990)"},
        },
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)

    senpai = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [],
        "primary_metric": {"name": "layerdrop_cost_bf16_mmlu", "value": mmlu["layerdrop_cost_bf16"]},
        "test_metric": {"name": "bf16_37L_gpqa", "value": C_g["acc"]},
    }

    print("\n==== BODY-DECOMP SERVED 2x2 (FULL HEAD) ====")
    print(f"{'corner':<12}{'MMLU-Pro':>12}{'GPQA-D':>12}")
    print(f"{'42L-bf16':<12}{A_m['acc']:>12.4f}{A_g['acc']:>12.4f}   (RUN)")
    print(f"{'42L-int4':<12}{B_m['acc']:>12.4f}{B_g['acc']:>12.4f}   (cite #511/#527, genuine)")
    print(f"{'37L-bf16':<12}{C_m['acc']:>12.4f}{C_g['acc']:>12.4f}   (RUN)")
    print(f"{'37L-int4':<12}{D_m['acc']:>12.4f}{D_g['acc']:>12.4f}   (cite #527, synth-full proxy)")
    print(f"\nint4_cost_fullbody : MMLU {mmlu['int4_cost_fullbody']:+.4f}  GPQA {gpqa['int4_cost_fullbody']:+.4f}")
    print(f"layerdrop_cost_bf16: MMLU {mmlu['layerdrop_cost_bf16']:+.4f}  GPQA {gpqa['layerdrop_cost_bf16']:+.4f}")
    print(f"interaction        : MMLU {mmlu['interaction']:+.4f}  GPQA {gpqa['interaction']:+.4f}")
    print(f"dominant_body_knob : MMLU={mmlu['dominant_body_knob']}  GPQA={gpqa['dominant_body_knob']}")
    print(f"base_int4_clears_gate: {base_int4_clears_gate}")
    print(f"prompt_identical_all4: mmlu={mmlu_ident} (mism={mmlu_mism}) gpqa={gpqa_ident} (mism={gpqa_mism})")
    print(f"VERDICT: {verdict}")
    print("SENPAI-RESULT:", json.dumps(senpai))

    if not a.no_wandb:
        _log_wandb(report, a)
    return 0


def _log_wandb(report, a):
    sys.path.insert(0, os.path.join("/workspace/senpai/target"))
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; JSON saved, skipping wandb", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="ubel",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#538 served body-decomp 2x2 (layers x precision, FULL 262k head): "
              "isolate int4-body-quant vs 37L layer-removal in the osoi5 collapse. "
              "42L-bf16/37L-bf16 RUN here; 42L-int4 (#511/#527 base) + 37L-int4 "
              "(#527 synthesized-full proxy) cited.",
        config={
            "pr": 538, "analysis_only": True, "no_hf_job": True,
            "no_served_file_change": True, "official_tps": 0,
            "gate_mmlu_threshold": GATE_MMLU, "gate_gpqa_threshold": GATE_GPQA,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return
    scalar_keys = [
        "base_int4_mmlu_pro", "base_int4_gpqa", "bf16_37L_mmlu_pro", "bf16_37L_gpqa",
        "bf16_42L_mmlu_pro", "bf16_42L_gpqa", "int4_37L_mmlu_pro", "int4_37L_gpqa",
        "int4_cost_fullbody_mmlu", "int4_cost_fullbody_gpqa",
        "layerdrop_cost_bf16_mmlu", "layerdrop_cost_bf16_gpqa",
        "interaction_mmlu", "interaction_gpqa",
        "dominant_body_knob_mmlu", "dominant_body_knob_gpqa",
        "base_int4_clears_gate", "prompt_identical_all4_mmlu",
        "prompt_identical_all4_gpqa", "official_tps", "analysis_only",
    ]
    for k in scalar_keys:
        run.summary[k] = report[k]
    for task, d in (("mmlu", report["mmlu_detail"]), ("gpqa", report["gpqa_detail"])):
        for kk in ("corner_42L_bf16", "corner_42L_int4", "corner_37L_bf16", "corner_37L_int4",
                   "int4_cost_fullbody", "layerdrop_cost_bf16", "int4_cost_reducedbody",
                   "layerdrop_cost_int4", "interaction", "dominant_body_knob"):
            run.summary[f"{task}/{kk}"] = d[kk]
    run.summary["verdict_text"] = report["verdict"]
    print(f"[wandb] logged run id={getattr(run,'id',None)}", flush=True)
    try:
        finish_wandb(run)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
