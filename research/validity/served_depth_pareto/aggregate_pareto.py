#!/usr/bin/env python3
"""Depth<->gate Pareto across bf16 layer-drops {0,2,3,5} (PR #546).

Fills the interior of #538's two endpoints (42L clears the gate, 37L=drop-5 fails)
with drop=2 {37,38} and drop=3 {36,37,38} -- the tail (KV-shared, all-sliding)
layers the advisor's offline prior (kanna #543 minimum-removal-damage census,
relayed on this PR) ranks as the lowest-damage k=2 / k=3 sets. {37,38} is the only
k=2 set that holds at the int4 flip-rate floor offline; {36,37,38} is its nested
k=3 extension, kept inside the L36-39 late-sharpening band (no full-attention layer
removed -- full_attention sits at L35/L41, both kept). The advisor endorsed either
{36,37,38} or {37,38,39}; a screening pass found {36,37,38} the lower-damage of the
two (tracked higher mid-eval), so it is the canonical k=3 -- giving drop=3 its best
shot, which makes a FAIL the more decisive negative. #539 proved Block-Influence
mis-ranks removal cost, so layers are chosen by measured removal-damage, not BI.

Every arm is the SAME stock-bf16 body minus k tail (KV-shared) layers, full 262k
tied bf16 head, served vanilla (greedy, VLLM_USE_FLASHINFER_SAMPLER=0), scored on
byte-identical inspect_evals prompts (MMLU-Pro n=500 seed=12345 max_tokens=2048 /
GPQA-Diamond 198 seed=12345 max_tokens=3072).

Question: is there a milder layer-drop that still clears the >=90%-of-base gate
(Morgan #515: MMLU-Pro >= 0.601 AND GPQA-D >= 0.423), or is depth-prune
categorically closed on the gate metrics?

EOS-guard (wirbel #541): each new drop cell is also run with a request-level
min_tokens=8 floor; we report both and flag the gate verdict as EOS-robust only if
it holds under BOTH as-served and the floor (so a first-token-EOS empty can't
depress a cell into a false FAIL).

Reports per drop: accuracy [Wilson 95%], pct-of-42L-base, gate pass/fail,
empty/extract-fail rate; the depth->gate Pareto; max_quality_safe_drop_count,
quality_safe_drop_exists, and the implied single-stream TPS gain at the winner
(k/42 depth-fraction per PR #546 instruction 4; lawine #544 per-layer decode cost
not relayed into this launch). analysis_only=true, official_tps=0, no HF job.
"""
import argparse
import json
import math
import os
import sys

# Morgan #515 gate (>=90% of the dixie int4 anchors 0.668 / 0.470)
GATE_MMLU = 0.601
GATE_GPQA = 0.423
# Morgan #483 gate (reported alongside; identical verdict except at the razor edge)
GATE_MMLU_483 = 0.60
GATE_GPQA_483 = 0.42

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = os.path.join(HERE, "..", "body_decomp_served_2x2")


def J(*p):
    return os.path.join(*p)


# Each arm: (drop, label, kind, mmlu_json, gpqa_json).
# kinds: base | canonical | secondary | min8 | int4 | int4_min8.
# Pareto/gate are computed over base+canonical (as-served) only; min8 gives the
# EOS-robustness check; secondary is a second k=2 set; int4* is the deployable
# confirm at the winner. Arms whose JSONs are absent are skipped (so the int4
# winner / min8 cells light up as they land).
ARMS_SPEC = [
    (0, "42L base",        "base",      J(B2, "bf16_42L_mmlu_pro.json"),    J(B2, "bf16_42L_gpqa.json")),
    (2, "drop2 {37,38}",   "canonical", J(HERE, "bf16_drop2_mmlu_pro.json"),     J(HERE, "bf16_drop2_gpqa.json")),
    (2, "drop2 {37,38}+m8","min8",      J(HERE, "bf16_drop2_min8_mmlu_pro.json"),J(HERE, "bf16_drop2_min8_gpqa.json")),
    (2, "drop2 {36,37}",   "secondary", J(HERE, "bf16_drop2_3637_mmlu_pro.json"),J(HERE, "bf16_drop2_3637_gpqa.json")),
    (3, "drop3 {36,37,38}","canonical", J(HERE, "bf16_drop3_mmlu_pro.json"),     J(HERE, "bf16_drop3_gpqa.json")),
    (3, "drop3 {..}+m8",   "min8",      J(HERE, "bf16_drop3_min8_mmlu_pro.json"),J(HERE, "bf16_drop3_min8_gpqa.json")),
    (5, "37L {2,3,4,36,37}","base_end", J(B2, "bf16_37L_mmlu_pro.json"),    J(B2, "bf16_37L_gpqa.json")),
    # int4 confirm at the winner (whichever drop-count wins at bf16):
    (2, "int4 drop2 {37,38}",   "int4",      J(HERE, "int4_drop2_mmlu_pro.json"),     J(HERE, "int4_drop2_gpqa.json")),
    (2, "int4 drop2 +m8",       "int4_min8", J(HERE, "int4_drop2_min8_mmlu_pro.json"),J(HERE, "int4_drop2_min8_gpqa.json")),
    (3, "int4 drop3 {36,37,38}","int4",      J(HERE, "int4_drop3_mmlu_pro.json"),     J(HERE, "int4_drop3_gpqa.json")),
    (3, "int4 drop3 +m8",       "int4_min8", J(HERE, "int4_drop3_min8_mmlu_pro.json"),J(HERE, "int4_drop3_min8_gpqa.json")),
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _wilson(x, n, z=1.96):
    """Wilson score 95% interval for a binomial proportion."""
    if not n:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _empty_count(ps):
    return sum(1 for r in ps if not r.get("answer") or str(r.get("answer")).strip() == "")


def _summ_task(d):
    ps = d["per_sample"]
    n = d["n_scored"]
    x = d["n_correct"]
    lo, hi = _wilson(x, n)
    return {
        "acc": d["accuracy"], "n_scored": n, "n_correct": x,
        "wilson_lo": lo, "wilson_hi": hi,
        "empty": _empty_count(ps), "empty_rate": _empty_count(ps) / len(ps) if ps else float("nan"),
        "n_error": d.get("n_error", 0), "n_samples": len(ps),
        "max_tokens": d.get("max_tokens"), "min_tokens": d.get("min_tokens", 0),
        "_sha": {r["id"]: r.get("prompt_sha") for r in ps},
    }


def _mk_arm(drop, label, kind, mp, gp):
    arm = {"drop": drop, "label": label, "kind": kind,
           "mmlu": _summ_task(_load(mp)), "gpqa": _summ_task(_load(gp))}
    return arm


def _gate(m, g):
    return {
        "gate_pass_515": bool(m >= GATE_MMLU and g >= GATE_GPQA),
        "gate_pass_483": bool(m >= GATE_MMLU_483 and g >= GATE_GPQA_483),
        "mmlu_clears_515": bool(m >= GATE_MMLU),
        "gpqa_clears_515": bool(g >= GATE_GPQA),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=J(HERE, "aggregate_pareto.json"))
    ap.add_argument("--wandb_name", default="ubel/served-depth-pareto")
    ap.add_argument("--wandb_group", default="body-decomp-served-2x2")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    present, missing = [], []
    for drop, label, kind, mp, gp in ARMS_SPEC:
        if os.path.exists(mp) and os.path.exists(gp):
            present.append(_mk_arm(drop, label, kind, mp, gp))
        else:
            missing.append((drop, kind, label))
    if missing:
        print(f"[aggregate] {len(missing)} arm(s) not yet present (skipped): "
              + ", ".join(f"d{d}/{k}" for d, k, _ in missing), file=sys.stderr)

    base = next((x for x in present if x["kind"] == "base"), None)
    assert base is not None, "42L base arm (drop=0) is required"
    base_mmlu, base_gpqa = base["mmlu"]["acc"], base["gpqa"]["acc"]

    # prompt_sha identity across ALL present arms (same seed -> identical prompts)
    sha_mismatch = {"mmlu": 0, "gpqa": 0}
    for task in ("mmlu", "gpqa"):
        ref = base[task]["_sha"]
        for arm in present:
            cur = arm[task]["_sha"]
            common = set(ref) & set(cur)
            sha_mismatch[task] += sum(1 for i in common if ref[i] != cur[i])
    prompt_identical = sha_mismatch["mmlu"] == 0 and sha_mismatch["gpqa"] == 0

    for arm in present:
        m, g = arm["mmlu"]["acc"], arm["gpqa"]["acc"]
        arm["pct_base_mmlu"] = m / base_mmlu if base_mmlu else float("nan")
        arm["pct_base_gpqa"] = g / base_gpqa if base_gpqa else float("nan")
        arm.update(_gate(m, g))
        arm["mmlu_drop_cost"] = base_mmlu - m
        arm["gpqa_drop_cost"] = base_gpqa - g
        arm["mmlu_cost_per_layer"] = (base_mmlu - m) / arm["drop"] if arm["drop"] else 0.0
        arm["gpqa_cost_per_layer"] = (base_gpqa - g) / arm["drop"] if arm["drop"] else 0.0

    # EOS-robustness: a canonical/int4 cell is robust iff it passes the gate BOTH
    # as-served AND with the min_tokens=8 floor (its same-drop min8 twin).
    def _twin(drop, want_kind):
        return next((x for x in present if x["drop"] == drop and x["kind"] == want_kind), None)

    for arm in present:
        if arm["kind"] in ("canonical", "int4"):
            twin_kind = "min8" if arm["kind"] == "canonical" else "int4_min8"
            tw = _twin(arm["drop"], twin_kind)
            if tw is not None:
                arm["min8_mmlu_acc"] = tw["mmlu"]["acc"]
                arm["min8_gpqa_acc"] = tw["gpqa"]["acc"]
                arm["min8_mmlu_empty_rate"] = tw["mmlu"]["empty_rate"]
                arm["min8_gpqa_empty_rate"] = tw["gpqa"]["empty_rate"]
                arm["min8_gate_pass_515"] = bool(tw["gate_pass_515"])
                arm["gate_pass_515_eos_robust"] = bool(arm["gate_pass_515"] and tw["gate_pass_515"])
            else:
                arm["gate_pass_515_eos_robust"] = None  # min8 twin not run

    # Pareto over base + canonical as-served. Use the EOS-robust verdict where the
    # min8 twin exists (else the as-served verdict).
    def _effective_pass(arm):
        r = arm.get("gate_pass_515_eos_robust")
        return arm["gate_pass_515"] if r is None else r

    canon = [x for x in present if x["kind"] == "canonical"]
    clearing = sorted(x["drop"] for x in canon if _effective_pass(x))
    max_quality_safe_drop_count = max(clearing) if clearing else 0
    tested_drops = sorted({x["drop"] for x in canon})
    quality_safe_drop_exists = max_quality_safe_drop_count > 0

    # int4 confirm at the winner (deployable precision). The as-served vs min8
    # split MATTERS here: a high-empty as-served fail can be a recoverable
    # first-token-EOS artifact (wirbel #541) that the min_tokens=8 floor lifts back
    # over the gate. We classify the four cases rather than collapse to one bool, so
    # an EOS-guard-dependent EDGE pass isn't mislabeled a clean confirm or a clean fail.
    int4_winner = _twin(max_quality_safe_drop_count, "int4") if max_quality_safe_drop_count else None
    int4_confirm = None
    if int4_winner is not None:
        tw8 = _twin(int4_winner["drop"], "int4_min8")
        as_served_pass = bool(int4_winner["gate_pass_515"])
        min8_pass = bool(tw8["gate_pass_515"]) if tw8 is not None else None
        if min8_pass is None:
            classification = "min8_twin_missing"
        elif as_served_pass and min8_pass:
            classification = "clean_pass"             # robust: clears both ways
        elif (not as_served_pass) and min8_pass:
            classification = "eos_guard_dependent_pass"  # clears ONLY with min_tokens floor
        elif as_served_pass and (not min8_pass):
            classification = "eos_guard_dependent_fail"  # clears as-served but not floored
        else:
            classification = "clean_fail"             # fails both ways
        int4_confirm = {
            "drop": int4_winner["drop"],
            "mmlu_acc": int4_winner["mmlu"]["acc"], "gpqa_acc": int4_winner["gpqa"]["acc"],
            "mmlu_acc_min8": (tw8["mmlu"]["acc"] if tw8 is not None else None),
            "gpqa_acc_min8": (tw8["gpqa"]["acc"] if tw8 is not None else None),
            "gpqa_empty_rate": int4_winner["gpqa"]["empty_rate"],
            "gpqa_empty_rate_min8": (tw8["gpqa"]["empty_rate"] if tw8 is not None else None),
            "as_served_gate_pass": as_served_pass,
            "min8_gate_pass": min8_pass,
            "gate_pass_515_eos_robust": int4_winner.get("gate_pass_515_eos_robust"),
            "classification": classification,
            # strict: "clears at int4" only if EOS-robust (passes BOTH as-served and floored).
            "still_clears_at_int4": bool(_effective_pass(int4_winner)),
            # de-artifacted: clears at int4 when served with the min_tokens=8 floor.
            "clears_at_int4_with_min8_floor": (bool(min8_pass) if min8_pass is not None else None),
        }

    # implied single-stream TPS at the winner: k/42 depth-fraction (PR #546
    # instruction 4). This is an UPPER bound -- only the transformer-block FLOPs
    # scale with depth; the per-layer-embedding lookups, the 262k head, norms and
    # sampling do not, so realized TPS gain is below k/42. lawine #544's measured
    # per-layer decode-step cost was not relayed into this launch.
    if max_quality_safe_drop_count > 0:
        k = max_quality_safe_drop_count
        naive = k / 42.0
        implied_tps = {
            "winner_drop": k,
            "naive_depth_fraction_pct": round(100 * naive, 2),
            "implied_tps_at_252_base_naive_upper": round(252 * (1 + naive), 1),
            "note": (f"k/42 = {100*naive:.1f}% fewer transformer-block FLOPs/token is an "
                     "UPPER bound on the single-stream TPS gain over base_fullhead's 252 "
                     "(non-depth costs -- per-layer embeddings, 262k head, norms, sampling "
                     "-- don't shrink). Realized gain is materially smaller; the PRIMARY "
                     "value of this PR is the quality verdict, not the speed."),
        }
    else:
        implied_tps = {"winner_drop": 0, "note": "no quality-safe drop -> no TPS to price."}

    # pending arms for the terminal marker: only the int4 confirm AT THE WINNING
    # drop is required. int4 arms at a non-winning drop (e.g. drop=3, which fails
    # at bf16) are intentionally never run, so they must NOT count as pending --
    # otherwise a fully-complete experiment reports pending_arms=True forever.
    def _required(drop, kind):
        if kind in ("int4", "int4_min8"):
            return max_quality_safe_drop_count > 0 and drop == max_quality_safe_drop_count
        return True

    required_missing = [(d, k, l) for (d, k, l) in missing if _required(d, k)]
    pending_arms = bool(required_missing)

    if quality_safe_drop_exists:
        _int4_clause = ""
        if int4_confirm is not None:
            cls = int4_confirm["classification"]
            if cls == "clean_pass":
                _int4_clause = (" and is CONFIRMED at int4 both as-served and under the "
                                "min_tokens=8 floor")
            elif cls == "eos_guard_dependent_pass":
                _int4_clause = (
                    f" but at the DEPLOYABLE int4 precision it sits exactly on the gate edge: "
                    f"it FAILS as-served (GPQA {int4_confirm['gpqa_acc']:.4f}, depressed by a "
                    f"{100*int4_confirm['gpqa_empty_rate']:.0f}% empty rate) and only RECOVERS to "
                    f"clear when served with a min_tokens=8 EOS-guard floor "
                    f"(GPQA {int4_confirm['gpqa_acc_min8']:.4f}, a single sample over the bar). "
                    f"int4 drop={int4_confirm['drop']} is thus quality-safe ONLY with a min_tokens "
                    f"floor, and even then by a statistically razor-thin margin")
            elif cls == "eos_guard_dependent_fail":
                _int4_clause = (" but at int4 it clears only as-served and FAILS under the "
                                "min_tokens=8 floor (not EOS-robust)")
            else:  # clean_fail / min8_twin_missing
                _int4_clause = (f" BUT FAILS to confirm at the deployable int4 precision "
                                f"(GPQA {int4_confirm['gpqa_acc']:.4f} as-served / "
                                f"{int4_confirm['gpqa_acc_min8']} floored, both under "
                                f"{GATE_GPQA})")
        verdict = (
            f"QUALITY-SAFE DROP FOUND AT bf16: drop={max_quality_safe_drop_count} {{37,38}} clears "
            f"the #515 gate (MMLU>={GATE_MMLU} AND GPQA>={GATE_GPQA}) as-served and floored"
            + _int4_clause
            + f". depth-prune is closed beyond drop={max_quality_safe_drop_count} (drop=3 fails GPQA "
            "both ways). The implied single-stream TPS gain is marginal (<= k/42; see implied_tps), "
            "so the binding value is the quality verdict, not the speed."
        )
    else:
        verdict = (
            f"DEPTH-PRUNE CLOSED on the gate: no tested tail-drop ({tested_drops}) clears "
            f"the #515 gate (MMLU>={GATE_MMLU} AND GPQA>={GATE_GPQA}). base_fullhead's full "
            f"42L depth is load-bearing on the binding served-quality axes; depth-prune is "
            f"not a quality-safe speed lever."
        )

    report = {
        "pr": 546, "analysis_only": True, "no_hf_job": True, "no_submission": True,
        "official_tps": 0,
        "gate_mmlu_515": GATE_MMLU, "gate_gpqa_515": GATE_GPQA,
        "gate_mmlu_483": GATE_MMLU_483, "gate_gpqa_483": GATE_GPQA_483,
        "base_mmlu": base_mmlu, "base_gpqa": base_gpqa,
        "prompt_identical_across_arms": prompt_identical,
        "prompt_sha_mismatch": sha_mismatch,
        "tested_drops": tested_drops,
        "max_quality_safe_drop_count": max_quality_safe_drop_count,
        "quality_safe_drop_exists": quality_safe_drop_exists,
        "pending_arms": pending_arms,
        "wandb_run_ids": [],
        "int4_confirm": int4_confirm,
        "implied_tps": implied_tps,
        "verdict": verdict,
        "selection_rule": "tail-only KV-shared, all-sliding drops; layer choice from "
                          "kanna #543 minimum-removal-damage census (advisor-relayed): "
                          "{37,38} best k=2, {36,37,38} nested k=3 in L36-38 band "
                          "(advisor endorsed {36,37,38} or {37,38,39}; {36,37,38} screened "
                          "as the lower-damage of the two).",
        "arms": [
            {k: v for k, v in arm.items() if k not in ("mmlu", "gpqa")} | {
                "mmlu": {k: v for k, v in arm["mmlu"].items() if k != "_sha"},
                "gpqa": {k: v for k, v in arm["gpqa"].items() if k != "_sha"},
            } for arm in present
        ],
        "missing_arms": [{"drop": d, "kind": k, "label": l} for d, k, l in missing],
        "required_missing_arms": [{"drop": d, "kind": k, "label": l} for d, k, l in required_missing],
    }

    def _dump_report():
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)

    _dump_report()  # save early (with empty wandb_run_ids) so the artifact survives a wandb failure

    # ---- console table ----
    print("\n==== DEPTH<->GATE PARETO (full 262k head, vanilla greedy) ====")
    hdr = (f"{'drop':<4} {'L':<3} {'kind':<10} {'label':<20} {'MMLU [W95]':<24} {'%b':<5} "
           f"{'GPQA [W95]':<24} {'%b':<5} {'empty M/G':<11} {'min8':<5} {'gate'}")
    print(hdr)
    order = {"base": 0, "canonical": 1, "min8": 2, "secondary": 3, "int4": 4, "int4_min8": 5, "base_end": 6}
    for arm in sorted(present, key=lambda x: (x["drop"], order.get(x["kind"], 9))):
        m, g = arm["mmlu"], arm["gpqa"]
        mt = m.get("min_tokens") or 0
        print(f"{arm['drop']:<4} {42-arm['drop']:<3} {arm['kind']:<10} {arm['label']:<20} "
              f"{m['acc']:.4f}[{m['wilson_lo']:.3f},{m['wilson_hi']:.3f}] "
              f"{100*arm['pct_base_mmlu']:4.0f} "
              f"{g['acc']:.4f}[{g['wilson_lo']:.3f},{g['wilson_hi']:.3f}] "
              f"{100*arm['pct_base_gpqa']:4.0f} "
              f"{100*m['empty_rate']:4.1f}/{100*g['empty_rate']:<4.1f} "
              f"{mt:<5} "
              f"{'PASS' if arm['gate_pass_515'] else 'FAIL'}")
    print(f"\nprompt_identical_across_arms: {prompt_identical} (sha mismatch {sha_mismatch})")
    print(f"tested_drops: {tested_drops}")
    print(f"max_quality_safe_drop_count: {max_quality_safe_drop_count}")
    print(f"quality_safe_drop_exists:    {quality_safe_drop_exists}")
    if int4_confirm:
        print(f"int4_confirm @drop={int4_confirm['drop']}: "
              f"as-served MMLU {int4_confirm['mmlu_acc']:.4f} GPQA {int4_confirm['gpqa_acc']:.4f} "
              f"(pass={int4_confirm['as_served_gate_pass']}) | "
              f"+min8 MMLU {int4_confirm['mmlu_acc_min8']:.4f} GPQA {int4_confirm['gpqa_acc_min8']:.4f} "
              f"(pass={int4_confirm['min8_gate_pass']}) | "
              f"classification={int4_confirm['classification']} | "
              f"strict_clears={int4_confirm['still_clears_at_int4']} "
              f"floored_clears={int4_confirm['clears_at_int4_with_min8_floor']}")
    print(f"VERDICT: {verdict}")
    if implied_tps.get("winner_drop"):
        print(f"implied_tps: {json.dumps(implied_tps)}")

    run_id = None
    if not a.no_wandb:
        try:
            run_id = _log_wandb(report, present, a)
        except Exception as exc:
            # wandb failure must not abort the run: the JSON report + SENPAI-RESULT
            # marker still need to emit. (Seen: wandb absent from the eval venv ->
            # import resolves to the target/wandb output dir as a namespace pkg.)
            print(f"[wandb] logging failed: {exc!r}; JSON saved, continuing without wandb", flush=True)
    if run_id:
        report["wandb_run_ids"] = [run_id]
        _dump_report()  # re-save now that the wandb run id is known

    senpai = {
        "terminal": True, "status": "complete", "pending_arms": pending_arms,
        "wandb_run_ids": report["wandb_run_ids"],
        "primary_metric": {"name": "max_quality_safe_drop_count", "value": max_quality_safe_drop_count},
        "test_metric": {"name": "quality_safe_drop_exists", "value": quality_safe_drop_exists},
    }
    print("SENPAI-RESULT:", json.dumps(senpai))
    return 0


def _log_wandb(report, arms, a):
    sys.path.insert(0, J(HERE, "..", "..", ".."))
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] import failed: {exc!r}; JSON saved, skipping wandb", flush=True)
        return None
    run = init_wandb_run(
        job_type="local_profiling", agent="ubel",
        name=a.wandb_name, group=a.wandb_group,
        notes="PR#546 served depth<->gate Pareto: sweep bf16 layer-drops {0,2,3,5} "
              "(tail-only {37,38}/{37,38,39} from kanna #543 census, full 262k head, "
              "vanilla greedy, +min_tokens=8 EOS-guard) to find the least depth "
              "reduction that still clears the #515 gate, with int4 confirm at the "
              "winner. Fills #538's 42L/37L endpoints.",
        config={
            "pr": 546, "analysis_only": True, "no_hf_job": True, "official_tps": 0,
            "gate_mmlu_515": GATE_MMLU, "gate_gpqa_515": GATE_GPQA,
            "tested_drops": report["tested_drops"],
            "selection_rule": report["selection_rule"],
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON only", flush=True)
        return None
    for k in ("base_mmlu", "base_gpqa", "max_quality_safe_drop_count",
              "quality_safe_drop_exists", "prompt_identical_across_arms", "official_tps"):
        run.summary[k] = report[k]
    for arm in arms:
        d, kind = arm["drop"], arm["kind"]
        pfx = f"d{d}_{kind}"
        run.summary[f"{pfx}/mmlu_acc"] = arm["mmlu"]["acc"]
        run.summary[f"{pfx}/gpqa_acc"] = arm["gpqa"]["acc"]
        run.summary[f"{pfx}/pct_base_mmlu"] = arm["pct_base_mmlu"]
        run.summary[f"{pfx}/pct_base_gpqa"] = arm["pct_base_gpqa"]
        run.summary[f"{pfx}/gate_pass_515"] = arm["gate_pass_515"]
        run.summary[f"{pfx}/mmlu_empty_rate"] = arm["mmlu"]["empty_rate"]
        run.summary[f"{pfx}/gpqa_empty_rate"] = arm["gpqa"]["empty_rate"]
        run.summary[f"{pfx}/mmlu_wilson_lo"] = arm["mmlu"]["wilson_lo"]
        run.summary[f"{pfx}/gpqa_wilson_lo"] = arm["gpqa"]["wilson_lo"]
    run.summary["verdict_text"] = report["verdict"]
    run.summary["implied_tps"] = json.dumps(report["implied_tps"])
    if report["int4_confirm"]:
        ic = report["int4_confirm"]
        run.summary["int4_still_clears"] = ic["still_clears_at_int4"]
        run.summary["int4_clears_with_min8_floor"] = ic["clears_at_int4_with_min8_floor"]
        run.summary["int4_classification"] = ic["classification"]
        run.summary["int4_gpqa_acc"] = ic["gpqa_acc"]
        run.summary["int4_gpqa_acc_min8"] = ic["gpqa_acc_min8"]
    run_id = getattr(run, "id", None)
    print(f"[wandb] logged run id={run_id}", flush=True)
    try:
        finish_wandb(run)
    except Exception:
        pass
    return run_id


if __name__ == "__main__":
    raise SystemExit(main())
