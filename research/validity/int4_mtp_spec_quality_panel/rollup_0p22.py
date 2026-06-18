#!/usr/bin/env python3
"""PR #629 — roll up the Option-B quality panel on vLLM 0.22.0 (engine=manifest) and
emit the verdict vs the #624 dev307 baseline.

Single-variable A/B: same int4_g128_lmhead+MTP-K7 spec stack, BI=1, gb6144 greedy gates +
GPQA-D >=10 seeds (sampled). ONLY change vs #624 = engine dev307 -> 0.22.0.

Per gate: accuracy, finish_length_rate (the kanna #618 crater detector), bar pass.
GPQA: >=10-seed pooled Wilson CI + gpqa_ci_lo_clears_bar.

Verdict:
  OPTIONB_DEV307_ONLY   - won't serve cleanly on 0.22.0 (=> #624 dev307 panel binds)
  OPTIONB_CRATERS_ON_0p22 - serves but inherits the int4 crater (MMLU/GSM8K ~50% finish-length)
  OPTIONB_STACK_ROBUST  - healthy on 0.22.0 AND GPQA >=10-seed CI-lo clears 0.471
"""
import json
import math
from pathlib import Path

DIR = Path("research/validity/int4_mtp_spec_quality_panel")
RESG = DIR / "results-greedy-0p22"
GPQA_TAG = "0p22gb6144"

BARS = {"mmlu_pro": 0.605, "gpqa_diamond": 0.471, "gsm8k": 0.807, "aime": 0.090}
# kanna #618 crater signature on the int4 stack is ~50% finish-length on MMLU/GSM8K.
# Healthy dev307 baseline is ~0%. 0.25 is the unambiguous midpoint threshold.
CRATER_FL_THRESHOLD = 0.25

# #624 dev307 baseline (the A/B reference) for side-by-side reporting.
DEV307 = {
    "mmlu_pro": {"acc": 0.664, "fl": 0.002},
    "gsm8k": {"acc": 0.928, "fl": 0.000},
    "aime": {"acc": 0.400, "fl": 0.150},
    "gpqa_diamond": {"acc": 0.4764, "ci_lo": 0.44815, "seeds": 6},
}


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    panel = {"engine": "0.22.0", "comparison_baseline": "dev307 #624", "bars": BARS,
             "crater_fl_threshold": CRATER_FL_THRESHOLD, "gates": {}}

    # ---- GSM8K ----
    g = load(RESG / "spec_greedy_0p22_gb6144_greedy.json")
    if g:
        fl = g.get("truncation_rate")
        if fl is None:
            fl = g.get("n_trunc", 0) / g["n"] if g.get("n") else None
        panel["gates"]["gsm8k"] = {
            "accuracy": g["accuracy"], "n": g.get("n"), "n_correct": g.get("n_correct"),
            "finish_length_rate": fl, "bar": BARS["gsm8k"],
            "pass": g["accuracy"] >= BARS["gsm8k"],
            "dev307_acc": DEV307["gsm8k"]["acc"], "dev307_fl": DEV307["gsm8k"]["fl"],
            "strict_rate": g.get("strict_rate"), "extract_fail_rate": g.get("extract_fail_rate"),
        }

    # ---- MMLU-Pro (the #547 crater leg) ----
    m = load(RESG / "spec_mmlu_pro_greedy_0p22_gb6144.json")
    if m:
        fl = m.get("length_stop_rate")
        if fl is None and m.get("n_scored"):
            fl = m.get("n_length_truncated", 0) / m["n_scored"]
        panel["gates"]["mmlu_pro"] = {
            "accuracy": m["accuracy"], "n": m.get("n_scored"), "n_correct": m.get("n_correct"),
            "finish_length_rate": fl, "bar": BARS["mmlu_pro"],
            "pass": m["accuracy"] >= BARS["mmlu_pro"],
            "dev307_acc": DEV307["mmlu_pro"]["acc"], "dev307_fl": DEV307["mmlu_pro"]["fl"],
            "empty_rate": m.get("empty_rate"),
        }

    # ---- AIME ----
    a = load(RESG / "spec_aime_greedy_0p22_gb6144.json")
    if a:
        n = a.get("n_problems") or a.get("total_samples")
        nlen = sum(1 for p in a.get("per_problem", [])
                   for f in (p.get("finish_reasons") or [p.get("finish_reason")]) if f == "length")
        fl = (nlen / n) if n else None
        acc = a.get("maj_k_accuracy", a.get("accuracy"))
        panel["gates"]["aime"] = {
            "accuracy": acc, "n": n, "n_correct": a.get("n_correct_maj"),
            "finish_length_rate": fl, "bar": BARS["aime"],
            "pass": acc >= BARS["aime"],
            "dev307_acc": DEV307["aime"]["acc"], "dev307_fl": DEV307["aime"]["fl"],
            "extract_fail_rate": a.get("extract_fail_rate"),
        }

    # ---- GPQA pooled (>=10 seeds) ----
    gp = load(DIR / f"{GPQA_TAG}_pooled.json")
    if gp:
        n = gp["n_scored"]
        c = gp["n_correct"]
        acc = c / n
        lo_w, hi_w = wilson(acc, n)
        ci_lo_clears = lo_w >= BARS["gpqa_diamond"]
        # De-confounded (excl the one gb6144 context-fit overflow item per seed: a
        # prompt >=2049 tok + 6144 max_tokens > 8192 ctx -> vLLM 400, force-scored
        # wrong). #624 reported BOTH; mirror that here for an apples-to-apples A/B.
        n_dc = gp.get("n_scored_excl_request_error", n)
        acc_dc = gp.get("accuracy_excl_request_error", acc)
        lo_dc, hi_dc = wilson(acc_dc, n_dc)
        ci_lo_clears_dc = lo_dc >= BARS["gpqa_diamond"]
        panel["gates"]["gpqa_diamond"] = {
            "accuracy": acc, "n": n, "n_correct": c,
            "n_seeds": len(gp.get("seeds", [])), "seeds": gp.get("seeds"),
            "ci95_wilson": [lo_w, hi_w], "ci_lo": lo_w,
            "bar": BARS["gpqa_diamond"], "pass": acc >= BARS["gpqa_diamond"],
            "gpqa_ci_lo_clears_bar": bool(ci_lo_clears),
            "n_request_error": gp.get("n_request_error"),
            "accuracy_excl_request_error": acc_dc,
            "n_scored_excl_request_error": n_dc,
            "ci95_wilson_excl_request_error": [lo_dc, hi_dc],
            "gpqa_ci_lo_clears_bar_deconf": bool(ci_lo_clears_dc),
            "pass_excl_request_error": bool(acc_dc >= BARS["gpqa_diamond"]),
            "finish_length_rate": gp.get("pooled_length_stop_rate"),
            "per_seed_acc": [{"seed": r["seed"], "accuracy": r["accuracy"],
                              "n_correct": r["n_correct"], "n_scored": r["n_scored"]}
                             for r in gp.get("per_seed", [])],
            "dev307_acc": DEV307["gpqa_diamond"]["acc"],
            "dev307_ci_lo": DEV307["gpqa_diamond"]["ci_lo"],
            "dev307_seeds": DEV307["gpqa_diamond"]["seeds"],
        }

    # ---- crater detection + verdict ----
    gates = panel["gates"]
    crater_legs = []
    for leg in ("mmlu_pro", "gsm8k"):
        gl = gates.get(leg)
        if gl and gl.get("finish_length_rate") is not None and gl["finish_length_rate"] >= CRATER_FL_THRESHOLD:
            crater_legs.append(leg)
    # acc-crater: MMLU below bar by a wide margin is the #547 signature too
    acc_crater = bool(gates.get("mmlu_pro") and not gates["mmlu_pro"]["pass"])

    all_present = all(k in gates for k in ("mmlu_pro", "gsm8k", "aime", "gpqa_diamond"))
    panel_pass = all(gates[k]["pass"] for k in ("mmlu_pro", "gsm8k", "aime", "gpqa_diamond")
                     if k in gates)
    is_crater = bool(crater_legs) or acc_crater
    gpqa = gates.get("gpqa_diamond")
    gpqa_clears = bool(gpqa and gpqa.get("gpqa_ci_lo_clears_bar"))

    # The crater question (kanna #618) lives on the three non-GPQA gates (MMLU/GSM8K are
    # hit hardest). GPQA is the separate knife-edge leg. "Healthy band" = GPQA acc clearly
    # out of the 0.194 conc-16 crater (>=0.40), so a marginal bar-miss is a knife-edge,
    # not an inherited crater. This keeps the verdict robust to a GPQA POINT estimate that
    # straddles 0.471 from below: the binding STACK_ROBUST test is the >=10-seed CI-lo.
    non_gpqa_pass = all(gates[k]["pass"] for k in ("mmlu_pro", "gsm8k", "aime") if k in gates)
    gpqa_healthy_band = bool(gpqa and gpqa["accuracy"] >= 0.40)
    inherits_int4_crater = is_crater

    # PR #629 instruction-3 defines "stack-robust / healthy on 0.22.0" as the CRATER axis:
    # MMLU/GSM8K finish-length ~3% not ~50% (=> dev307 panel corroborated, not a stack
    # artifact). GPQA CI-lo clearing the bar is the SEPARATE second conjunct of
    # STACK_ROBUST and is reported as its own field (gpqa_ci_lo_clears_bar). So
    # optionb_healthy_on_0p22 = non-GPQA gates pass AND no int4 crater. It being True with
    # gpqa_ci_lo_clears_bar False is exactly the OPTIONB_NOT_ROBUST_GPQA_KNIFE_EDGE_FAILS case.
    optionb_healthy = bool(non_gpqa_pass and not is_crater)
    # serves_on_0p22 is set by the smoke/serve check; assume True if any gate produced output
    serves = bool(gates)

    if not serves:
        verdict = "OPTIONB_DEV307_ONLY"
    elif inherits_int4_crater:
        verdict = "OPTIONB_CRATERS_ON_0p22"
    elif non_gpqa_pass and gpqa_healthy_band and gpqa_clears:
        verdict = "OPTIONB_STACK_ROBUST"
    elif non_gpqa_pass and gpqa_healthy_band and not gpqa_clears:
        # serves + no crater + 3/4 gates pass + GPQA in the healthy band but CI-lo
        # (and/or point) misses 0.471 -> the GPQA knife-edge resolves to FAIL on a
        # >=10-seed read. NOT STACK_ROBUST, but explicitly NOT a crater either.
        verdict = "OPTIONB_NOT_ROBUST_GPQA_KNIFE_EDGE_FAILS"
    else:
        verdict = "OPTIONB_INDETERMINATE_SEE_GATES"

    panel.update({
        "crater_legs": crater_legs, "acc_crater_mmlu": acc_crater,
        "is_crater": is_crater, "inherits_int4_crater": inherits_int4_crater,
        "panel_all_gates_pass": panel_pass,
        "non_gpqa_gates_pass": non_gpqa_pass, "gpqa_healthy_band": gpqa_healthy_band,
        "all_gates_present": all_present,
        "gpqa_ci_lo_clears_bar": gpqa_clears,
        "optionb_healthy_on_0p22": optionb_healthy,
        "serves_on_0p22": serves,
        "verdict": verdict,
    })

    out = DIR / "panel_0p22.json"
    out.write_text(json.dumps(panel, indent=2))

    print("=" * 72)
    print(f"OPTION-B QUALITY PANEL @ vLLM 0.22.0  (vs #624 dev307 baseline)")
    print("=" * 72)
    for leg in ("mmlu_pro", "gsm8k", "aime", "gpqa_diamond"):
        gl = gates.get(leg)
        if not gl:
            print(f"  {leg:14s} : (missing)")
            continue
        fl = gl.get("finish_length_rate")
        fls = f"{fl:.3f}" if isinstance(fl, (int, float)) else "n/a"
        extra = ""
        if leg == "gpqa_diamond":
            dcw = gl.get("ci95_wilson_excl_request_error", [float("nan"), float("nan")])
            extra = (f" CI95w[{gl['ci95_wilson'][0]:.4f},{gl['ci95_wilson'][1]:.4f}]"
                     f" ci_lo_clears={gl['gpqa_ci_lo_clears_bar']} seeds={gl['n_seeds']}"
                     f"  | deconf(excl {gl.get('n_request_error')}err)"
                     f" acc={gl.get('accuracy_excl_request_error'):.4f}"
                     f" CI95w[{dcw[0]:.4f},{dcw[1]:.4f}]"
                     f" ci_lo_clears={gl.get('gpqa_ci_lo_clears_bar_deconf')}")
        print(f"  {leg:14s} : acc={gl['accuracy']:.4f} (bar {gl['bar']}, "
              f"{'PASS' if gl['pass'] else 'FAIL'})  finish_len={fls}  "
              f"[dev307 acc={gl['dev307_acc']} fl={gl.get('dev307_fl')}]{extra}")
    print("-" * 72)
    print(f"  crater_legs={crater_legs}  acc_crater_mmlu={acc_crater}  inherits_int4_crater={is_crater}")
    print(f"  non_gpqa_gates_pass={non_gpqa_pass}  gpqa_healthy_band={gpqa_healthy_band}")
    print(f"  panel_all_gates_pass={panel_pass}  gpqa_ci_lo_clears_bar={gpqa_clears}")
    print(f"  optionb_healthy_on_0p22={optionb_healthy}  serves_on_0p22={serves}")
    print(f"  VERDICT: {verdict}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
