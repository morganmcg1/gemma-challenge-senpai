#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #669 -- terminal synthesis for the K=5 matched-basis live identity cert.

Folds the three measured artifacts of the single decisive matched-serve arm into
one decision-grade record and logs it to W&B (group optionb-livecert-k5-stark):

  matched_identity.json   -- AR-vs-AR floor, spec-vs-AR cascade, strict verdict
  matched_speed.json      -- rate sweep (0 / pre-fork / 2x) + dummy-tau over-fire arm
  prefill_margin.json     -- pos-0 top-k logit margins for the 2 surviving divergences

It emits the advisor's load-bearing portfolio number -- the measured rescue tax in
ms/step and the rescued wall-TPS at the optimal K=5, each WITH a 95% CI -- alongside
the strict-#319 identity verdict. analysis_only, official_tps=0, NO HF Job.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
LOCKED_AR = 126.378          # LOCKED #319 official AR rung
AR_RUNG_LOCAL = 126.75       # matched local AR rung (#642 6uepftr6)
PLUS10 = LOCKED_AR + 10.0
CONV = LOCKED_AR / AR_RUNG_LOCAL  # cross-submission official-equiv (advisor-prescribed)

# pos-0 divergent prompts: AR-pick -> spec-pick (both flip to 8291 'Here')
DIVERGENCES = {
    "37227f6b": {"ar": 818, "spec": 8291, "ar_str": "The", "spec_str": "Here"},
    "74200cad": {"ar": 5471, "spec": 8291, "ar_str": "Step", "spec_str": "Here"},
}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    yhat = [a + b * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    dof = n - 2
    s2 = ss_res / dof if dof > 0 else float("nan")
    se_b = math.sqrt(s2 / sxx) if dof > 0 and sxx > 0 else float("nan")
    return {"b": b, "a": a, "se_b": se_b, "r2": r2, "n": n, "dof": dof,
            "s2": s2, "mx": mx, "sxx": sxx}


def _norm(tok: str) -> str:
    return tok.replace("▁", " ").strip()


def extract_margin(probe: dict, key8: str) -> dict:
    """For a divergent prompt, pull (AR-config, spec-config) logprobs of both the
    AR-pick and the spec-pick token, and the resulting argmax margins."""
    want = DIVERGENCES[key8]
    out = {"key": key8, **{f"want_{k}": v for k, v in want.items()}}
    by_label = {c["label"]: c for c in probe.get("configs", [])}
    for label in ("ar", "spec"):
        cfg = by_label.get(label)
        if not cfg:
            continue
        rec = next((r for r in cfg["results"] if r["key"].startswith(key8)), None)
        if not rec:
            continue
        top0 = (rec.get("parsed") or {}).get("top0") or {}
        norm = {_norm(k): v for k, v in top0.items()}
        lp_ar = norm.get(_norm(want["ar_str"]))
        lp_spec = norm.get(_norm(want["spec_str"]))
        out[f"{label}_argmax_id"] = (rec.get("parsed") or {}).get("argmax_id")
        out[f"{label}_lp_arpick"] = lp_ar
        out[f"{label}_lp_specpick"] = lp_spec
        out[f"{label}_top0"] = top0
        if lp_ar is not None and lp_spec is not None:
            # margin of THIS config's winner over the other candidate (>=0)
            out[f"{label}_margin_nats"] = (
                (lp_ar - lp_spec) if label == "ar" else (lp_spec - lp_ar))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=Path, default=HERE / "matched_k5_speed/matched_speed.json")
    ap.add_argument("--identity", type=Path, default=HERE / "matched_k5/matched_identity.json")
    ap.add_argument("--probe", type=Path, default=HERE / "prefill_margin/prefill_margin.json")
    ap.add_argument("--out", type=Path, default=HERE / "matched_synthesis.json")
    ap.add_argument("--wandb-name", default="stark/k5-matched-synthesis")
    ap.add_argument("--wandb-group", default="optionb-livecert-k5-stark")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    sp = json.loads(a.speed.read_text())
    idn = json.loads(a.identity.read_text())
    probe = json.loads(a.probe.read_text()) if a.probe.exists() else {"configs": []}

    speed = sp["speed"]
    arms = sp["arms"]
    # ---- raw per-run points for the OLS (rate sweep arms only) ----
    raw = []
    eaccs = []
    for lbl, info in arms.items():
        if lbl.startswith("tau") or not lbl.startswith("r"):
            continue
        rate = float(lbl[1:].replace("p", ".").replace("m", "-"))
        for v in (info.get("wall_tps_values") or []):
            raw.append((rate, v))
        if isinstance(info.get("e_accept_exact_mean"), (int, float)):
            eaccs.append(info["e_accept_exact_mean"])
    raw.sort()
    xs = [r for r, _ in raw]
    ys = [1.0 / t for _, t in raw]
    f = ols(xs, ys)
    C = f["b"]                      # sec / recompute fire
    tcrit = T95.get(f["dof"], 1.96)
    C_lo, C_hi = C - tcrit * f["se_b"], C + tcrit * f["se_b"]
    E_accept = sum(eaccs) / len(eaccs) if eaccs else float("nan")

    rstar = round(speed["inject_rate_per_emit"], 6)
    tps0 = speed["tps0_unrescued_ceiling_local"]
    tps_star = speed["correcting_rescue_tps_local"]
    dummy_tps = speed["dummy_tau_tps_local"]

    # ---- rescue tax in ms/step = C * rstar * E_accept (sec/step) ----
    tax_ms = C * rstar * E_accept * 1000.0
    tax_lo = C_lo * rstar * E_accept * 1000.0
    tax_hi = C_hi * rstar * E_accept * 1000.0

    # ---- rescued wall-TPS at K=5 with fit-based mean-response 95% CI ----
    y0 = f["a"] + C * rstar
    se_mean = math.sqrt(f["s2"] * (1.0 / f["n"] + (rstar - f["mx"]) ** 2 / f["sxx"]))
    inv_lo, inv_hi = y0 - tcrit * se_mean, y0 + tcrit * se_mean
    tps_fit = 1.0 / y0
    tps_fit_lo, tps_fit_hi = 1.0 / inv_hi, 1.0 / inv_lo
    oe = tps_fit * CONV
    oe_lo, oe_hi = tps_fit_lo * CONV, tps_fit_hi * CONV

    # ---- identity (matched basis) ----
    cf = idn["cascade_floor_arar"]
    cs = idn["cascade_spec"]
    pos0_floor = cf["pos0_disagree"]      # AR-vs-AR (BI=1) nondeterminism floor
    pos0_spec = cs["pos0_disagree"]       # spec-vs-AR raw token diffs
    margins = [extract_margin(probe, k) for k in DIVERGENCES]

    # The greedy gate tolerates argmax flips at numerical-noise scale (the logprob grid
    # is 0.125 nats here). A spec-vs-AR pos-0 diff is a TIE ARTIFACT (not a genuine
    # confident divergence) when the two tokens are tied to <=grid in EITHER config --
    # i.e. the "disagreement" is just which side of a 0-margin tie each config landed.
    GRID = 0.125
    artifact_flags = 0
    confident_flips = 0
    nondeterministic = 0
    for m in margins:
        am = m.get("ar_margin_nats")
        sm = m.get("spec_margin_nats")
        tied = (am is not None and am <= GRID + 1e-9) or (sm is not None and sm <= GRID + 1e-9)
        if tied:
            artifact_flags += 1
        else:
            confident_flips += 1
        # probe spec-config re-query disagreeing with the decode-run spec pick =>
        # the served pos-0 token itself is non-deterministic across invocations
        if m.get("spec_argmax_id") is not None and m["spec_argmax_id"] == m.get("want_ar"):
            nondeterministic += 1

    # ---- verdict ----
    # strict-literal lens: any spec-vs-AR token diff over floor fails (advisor's gate
    # with no artifact exemption). artifact-resolved lens (advisor's stated PASS path:
    # "prove the prefill flips are artifact on a matched reference"): only confident,
    # non-tie divergences count.
    strict_literal_holds = (pos0_spec <= pos0_floor)
    identity_artifact_resolved = (confident_flips == 0)
    verdict = ("LIVE_FLAG_REDUCIBLE_HOLDS" if identity_artifact_resolved
               else "LIVE_FLAG_IRREDUCIBLE")

    out = {
        "pr": 669, "leg": "matched_synthesis", "analysis_only": True,
        "official_tps": 0, "no_hf_job": True,
        # ---------- load-bearing portfolio numbers ----------
        "rescue_tax_ms_per_step": round(tax_ms, 4),
        "rescue_tax_ms_per_step_95ci": [round(tax_lo, 4), round(tax_hi, 4)],
        "C_ms_per_recompute": round(C * 1000.0, 4),
        "C_ms_per_recompute_95ci": [round(C_lo * 1000.0, 4), round(C_hi * 1000.0, 4)],
        "C_over_636_assumption": round(C * LOCKED_AR, 4),
        "E_accept_tokens_per_step": round(E_accept, 4),
        "rescued_walltps_local": round(tps_fit, 4),
        "rescued_walltps_local_95ci": [round(tps_fit_lo, 4), round(tps_fit_hi, 4)],
        "rescued_official_equiv": round(oe, 4),
        "rescued_official_equiv_95ci": [round(oe_lo, 4), round(oe_hi, 4)],
        "plus10_bar": PLUS10,
        "clears_plus10_point": bool(oe >= PLUS10),
        "clears_plus10_at_ci_lo": bool(oe_lo >= PLUS10),
        "clears_plus10_margin": round(oe - PLUS10, 4),
        "unrescued_ceiling_local": round(tps0, 4),
        # ---------- the over-fire (as-implemented dummy gate) floor ----------
        "dummy_tau_tps_local": round(dummy_tps, 4),
        "dummy_tau_official_equiv": round(dummy_tps * CONV, 4),
        "dummy_tau_clears_plus10": bool(dummy_tps * CONV >= PLUS10),
        "dummy_tau_realized_fire_rate_per_emit": speed["dummy_tau_realized_fire_rate"],
        "prefork_rate_per_emit": rstar,
        "overfire_ratio": round(speed["dummy_tau_realized_fire_rate"] / rstar, 4),
        "correcting_over_dummy_tau": round(tps_star / dummy_tps, 4),
        "r2_raw_fit": round(f["r2"], 8),
        # ---------- identity (matched basis) ----------
        "ar_vs_ar_pos0_floor": pos0_floor,
        "spec_vs_ar_pos0_disagree": pos0_spec,
        "pos0_logprob_grid_nats": GRID,
        "tie_artifact_flips": artifact_flags,
        "confident_genuine_flips": confident_flips,
        "nondeterministic_flips": nondeterministic,
        "draft_break_at_tau": idn["verdict"].get("draft_break_count_at_tau"),
        "prefork_draft_positions": (idn.get("cert") or {}).get("prefork_draft_positions"),
        "strict_literal_holds": strict_literal_holds,
        "identity_artifact_resolved_holds": identity_artifact_resolved,
        "verdict": verdict,
        "verdict_phase_identity": idn["verdict"]["verdict"],
        "pos0_margins": margins,
        "raw_points": raw,
    }
    a.out.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "pos0_margins"}, indent=2,
                     default=str))
    print("\n[pos0 margins]")
    for m in margins:
        print(f"  key={m['key']} want_ar={m.get('want_ar_str')!r}->spec={m.get('want_spec_str')!r} "
              f"AR-cfg margin(arpick-Here)={m.get('ar_margin_nats')} nats  "
              f"SPEC-cfg margin(Here-arpick)={m.get('spec_margin_nats')} nats  "
              f"ar_argmax={m.get('ar_argmax_id')} spec_argmax={m.get('spec_argmax_id')}")

    if not a.no_wandb:
        try:
            from scripts.wandb_logging import init_wandb_run, finish_wandb
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] import failed: {exc!r}; JSON only", flush=True)
            return 0
        run = init_wandb_run(
            job_type="local_profiling", agent="stark", name=a.wandb_name,
            group=a.wandb_group,
            notes="PR#669 K=5 matched-basis TERMINAL synthesis: rescue tax ms/step + "
                  "rescued walltps (both with 95% CI) + strict-#319 verdict. analysis_only.",
            config={"pr": 669, "harness_lineage_pr": 642, "mode": "matched_synthesis",
                    "submission": "int4_mtp_batchinv", "num_speculative_tokens": 5,
                    "tau": 0.27, "analysis_only": True, "official_tps": 0},
        )
        if run is not None:
            for k, v in out.items():
                if isinstance(v, bool):
                    run.summary[k] = int(v)
                elif isinstance(v, (int, float)):
                    run.summary[k] = v
            # CI bounds as flat scalars (W&B summary can't hold lists cleanly)
            for k in ("rescue_tax_ms_per_step_95ci", "C_ms_per_recompute_95ci",
                      "rescued_walltps_local_95ci", "rescued_official_equiv_95ci"):
                run.summary[f"{k}_lo"] = out[k][0]
                run.summary[f"{k}_hi"] = out[k][1]
            for m in margins:
                run.summary[f"margin/{m['key']}/ar_nats"] = m.get("ar_margin_nats")
                run.summary[f"margin/{m['key']}/spec_nats"] = m.get("spec_margin_nats")
            run.summary["analysis_only"] = 1
            run.summary["official_tps"] = 0
            run.summary["no_hf_job"] = 1
            run.summary["verdict"] = verdict
            finish_wandb(run)
            print(f"[wandb] logged matched-synthesis run {run.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
