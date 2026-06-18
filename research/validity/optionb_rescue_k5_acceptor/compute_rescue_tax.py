#!/usr/bin/env python3
"""PR#669 matched-basis rescue-tax + CI computation (analysis_only).

Reads matched_speed.json (rate sweep + dummy tau arm) and matched_identity.json
and reports, for the advisor's load-bearing portfolio number:

  * rescued wall-TPS at optimal K=5 (idealized pre-fork rate = the cert min-safe rate)
  * the un-rescued ceiling, the realized live-tau (global-rate cost-probe) reference
  * C (ms / recompute) with a 95% CI from an OLS fit on all RAW per-run points
  * the rescue tax in ms/step (E[accept] = acceptance length = tokens/step) with a 95% CI
  * official-equiv (x 126.378/126.75) and the +10 bar check

No fire, no HF Job, no served-file change. Pure post-hoc arithmetic over local JSON.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEED = HERE / "matched_k5_speed" / "matched_speed.json"
IDENT = HERE / "matched_k5" / "matched_identity.json"
LOCKED_AR = 126.378
AR_RUNG_LOCAL = 126.75
PLUS10 = LOCKED_AR + 10.0


def ols(xs, ys):
    """OLS y = a + b x. Returns (b, a, se_b, r2, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    yhat = [a + b * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # slope standard error
    dof = n - 2
    se_b = math.sqrt((ss_res / dof) / sxx) if dof > 0 and sxx > 0 else float("nan")
    return b, a, se_b, r2, n


# Student t 95% two-sided critical values for small dof
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def main():
    sp = json.loads(SPEED.read_text())
    speed = sp["speed"]
    arms = sp["arms"]

    # Gather RAW per-run (rate, wall_tps) points from the rate-sweep arms only.
    raw = []  # (rate, wall_tps)
    eaccs = []
    for lbl, info in arms.items():
        if not lbl.startswith("r") or lbl.startswith("tau"):
            continue
        # label like r0, r0p043015, r0p086 -> recover rate
        rate_part = lbl[1:].replace("p", ".").replace("m", "-")
        try:
            rate = float(rate_part)
        except ValueError:
            continue
        for v in info.get("wall_tps_values", []):
            raw.append((rate, v))
        ea = info.get("e_accept_exact_mean")
        if isinstance(ea, (int, float)):
            eaccs.append(ea)

    raw.sort()
    xs = [r for r, _ in raw]
    ys = [1.0 / t for _, t in raw]  # inverse-tps (sec per token)
    C, a, se_C, r2, n = ols(xs, ys)  # C in sec/recompute (per-emit-normalized)
    tcrit = T95.get(n - 2, 1.96)
    C_lo, C_hi = C - tcrit * se_C, C + tcrit * se_C

    E_accept = sum(eaccs) / len(eaccs) if eaccs else float("nan")  # tokens/step

    rates = {float(k): v for k, v in speed["rate_to_wall_tps"].items()}
    r0 = 0.0
    rstar = round(speed["inject_rate_per_emit"], 6)  # pre-fork min-safe rate
    tps0 = rates[r0]
    tps_star = rates[rstar]

    # T_step = E_accept (tokens/step) / wall_tps  ; tax = dT_step at the pre-fork rate
    Tstep0_ms = E_accept / tps0 * 1000.0
    Tstep_star_ms = E_accept / tps_star * 1000.0
    tax_ms_step_point = Tstep_star_ms - Tstep0_ms

    # CI on the tax via the fit: tax = C * rstar * E_accept * 1000 (ms/step)
    tax_ms_step_fit = C * rstar * E_accept * 1000.0
    tax_lo = C_lo * rstar * E_accept * 1000.0
    tax_hi = C_hi * rstar * E_accept * 1000.0

    C_ms = C * 1000.0
    C_ms_lo, C_ms_hi = C_lo * 1000.0, C_hi * 1000.0

    official_equiv = tps_star * LOCKED_AR / AR_RUNG_LOCAL
    clears_plus10 = official_equiv >= PLUS10

    # realized live tau (cost-probe, global rate) reference
    dummy_tau_tps = speed.get("dummy_tau_tps_local")
    dummy_fire = speed.get("dummy_tau_realized_fire_rate")
    dummy_flag = speed.get("dummy_tau_realized_flag_rate")

    out = {
        "n_raw_points": n,
        "E_accept_tokens_per_step": round(E_accept, 4),
        "unrescued_ceiling_local": round(tps0, 3),
        "rescued_prefork_local": round(tps_star, 3),
        "rescued_official_equiv": round(official_equiv, 3),
        "plus10_bar": PLUS10,
        "clears_plus10_idealized": clears_plus10,
        "clears_plus10_margin": round(official_equiv - PLUS10, 3),
        "C_ms_per_recompute": round(C_ms, 4),
        "C_ms_per_recompute_95ci": [round(C_ms_lo, 4), round(C_ms_hi, 4)],
        "C_over_636_assumption": round(C * LOCKED_AR, 4),
        "rescue_tax_ms_per_step_point": round(tax_ms_step_point, 4),
        "rescue_tax_ms_per_step_fit": round(tax_ms_step_fit, 4),
        "rescue_tax_ms_per_step_95ci": [round(tax_lo, 4), round(tax_hi, 4)],
        "Tstep_unrescued_ms": round(Tstep0_ms, 4),
        "Tstep_rescued_ms": round(Tstep_star_ms, 4),
        "prefork_rate_per_emit": rstar,
        "r2_raw_fit": round(r2, 7),
        "dummy_tau_global_tps_local": dummy_tau_tps,
        "dummy_tau_realized_fire_rate": dummy_fire,
        "dummy_tau_realized_flag_rate": dummy_flag,
        "raw_points": raw,
        "rate_to_wall_tps": speed["rate_to_wall_tps"],
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    if not SPEED.exists():
        print(f"[wait] {SPEED} not present yet", file=sys.stderr)
        sys.exit(2)
    main()
