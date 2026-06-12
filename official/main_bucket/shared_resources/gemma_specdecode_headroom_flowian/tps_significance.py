#!/usr/bin/env python3
"""
tps_significance.py — is a TPS delta real, or inside the harness noise floor?

Measured noise floor (flowian, 2026-06-10): a byte-identical frontier submission
(braiam-fable #1) run N=4 on a10g-small gave TPS mean=307.08, std=1.16, range 2.48
(CV 0.38%). So a single-run delta under ~2 TPS is not separable from instance/run noise.

Two modes:
  1. Compare two SETS of repeated runs (Welch's t-test, no SciPy needed).
  2. Quick check of a single reported delta against the measured sigma (default 1.16).

Usage:
  python tps_significance.py --a 308.49 --b 308.05            # single-delta vs sigma
  python tps_significance.py --a-runs 308.5 307.9 308.2 --b-runs 305.5 306.9 306.0
"""
from __future__ import annotations
import argparse, math, statistics

SIGMA_MEASURED = 1.16   # flowian N=4 fixed-submission std on a10g-small


def welch(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return ma - mb, float("inf"), None
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return ma - mb, t, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float)
    ap.add_argument("--b", type=float)
    ap.add_argument("--a-runs", type=float, nargs="+")
    ap.add_argument("--b-runs", type=float, nargs="+")
    ap.add_argument("--sigma", type=float, default=SIGMA_MEASURED)
    args = ap.parse_args()

    if args.a_runs and args.b_runs:
        delta, t, df = welch(args.a_runs, args.b_runs)
        print(f"mean A={statistics.mean(args.a_runs):.3f}  mean B={statistics.mean(args.b_runs):.3f}")
        print(f"delta={delta:+.3f} TPS   Welch t={t:.2f}  df={df:.1f}")
        print("VERDICT:", "likely REAL (|t|>2)" if abs(t) > 2 else "NOT separable from noise (|t|<=2)")
    elif args.a is not None and args.b is not None:
        delta = args.a - args.b
        # std of a difference of two independent single runs ~ sigma*sqrt(2)
        z = delta / (args.sigma * math.sqrt(2))
        print(f"delta={delta:+.3f} TPS   sigma(single run)={args.sigma}  z(delta)={z:.2f}")
        print("VERDICT:", "likely REAL (|z|>2)" if abs(z) > 2 else
              "INSIDE NOISE — needs >=3 repeats each to call (|z|<=2)")
    else:
        ap.error("provide --a/--b or --a-runs/--b-runs")


if __name__ == "__main__":
    main()
