#!/usr/bin/env python
"""PR #813 — plot the synthetic-acceptance ceiling curve from the sweep.

Reads the per-rate JSON written by accept_oracle.py and renders TPS vs imposed
mean-acceptance-length (the ceiling curve), annotating the r=0.56 anchor, the
r=1.00 ceiling, the realized %gain, and the advisor's 270/282 decision bands.
Text-only fallback if matplotlib is unavailable.

  uv run python research/lenient_spec_acceptance_813/plot_oracle.py [--run-dir runs/sweep]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHOR_256 = 256.74  # reconstructed int4head AR-equiv anchor (W&B 9tcygwjf)
GREENLIGHT = 282.0   # >~10% over 256.74
CLOSE = 270.0        # <~5% over 256.74


def load(run_dir: Path) -> list[dict]:
    summ = run_dir / "sweep_summary.json"
    if summ.exists():
        data = json.loads(summ.read_text())
    else:
        data = [json.loads(p.read_text()) for p in sorted(run_dir.glob("rate_*.json"))]
    return [d for d in data if "error" not in d]


def tps(d: dict) -> float | None:
    return d.get("steady_gen_tps") or d.get("decode_wall_tps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(HERE / "runs" / "sweep"))
    ap.add_argument("--wandb", action="store_true", help="log the PNG + table to W&B")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = sorted(load(run_dir), key=lambda d: d["rate"])
    if not rows:
        print(f"no completed rates in {run_dir}")
        return

    print(f"{'rate':>6} {'mean_acc_len':>12} {'steady_tps':>11} {'wall_tps':>9} {'e_accept':>9}")
    for d in rows:
        print(f"{d['rate']:>6.2f} {d['imposed_mean_acceptance_length']:>12.2f} "
              f"{(d.get('steady_gen_tps') or float('nan')):>11.2f} "
              f"{(d.get('decode_wall_tps') or float('nan')):>9.2f} "
              f"{(d.get('e_accept_exact_from_log') or d.get('e_accept_mean_acceptance_length_prom') or float('nan')):>9.3f}")

    anchor = next((tps(d) for d in rows if abs(d["rate"] - 0.56) < 1e-6), None)
    ceil = next((tps(d) for d in rows if abs(d["rate"] - 1.00) < 1e-6), None)
    verdict = "n/a"
    if anchor and ceil:
        gain = 100.0 * (ceil - anchor) / anchor
        verdict = ("GREENLIGHT custom-kernel top-k-match PR" if gain > 10.0
                   else "CLOSE acceptance axis" if gain < 5.0
                   else "MARGINAL (5-10%) — lean close")
        print(f"\nanchor(r=0.56)={anchor:.2f}  ceiling(r=1.00)={ceil:.2f}  "
              f"gain=+{gain:.1f}%  -> {verdict}")
        print(f"(advisor bands vs 256.74: GREENLIGHT>{GREENLIGHT}, CLOSE<{CLOSE})")

    png = run_dir / "ceiling_curve.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [d["imposed_mean_acceptance_length"] for d in rows]
        ys_s = [d.get("steady_gen_tps") for d in rows]
        ys_w = [d.get("decode_wall_tps") for d in rows]
        fig, ax = plt.subplots(figsize=(7, 5))
        if any(ys_s):
            ax.plot(xs, ys_s, "o-", label="steady_gen_tps (engine meter)", color="C0")
        ax.plot(xs, ys_w, "s--", label="decode_wall_tps (incl prefill)", color="C1", alpha=0.7)
        ax.axhline(ANCHOR_256, ls=":", color="gray", label=f"256.74 reconstructed anchor")
        ax.axhline(GREENLIGHT, ls=":", color="green", alpha=0.6, label=f"{GREENLIGHT} greenlight (+10%)")
        ax.axhline(CLOSE, ls=":", color="red", alpha=0.6, label=f"{CLOSE} close (+5%)")
        for d in rows:
            y = tps(d)
            if y:
                ax.annotate(f"r={d['rate']:.2f}", (d["imposed_mean_acceptance_length"], y),
                            textcoords="offset points", xytext=(5, 6), fontsize=8)
        ax.set_xlabel("imposed mean acceptance length (1 + K·rate, K=6)")
        ax.set_ylabel("local conc=1 decode TPS (tok/s)")
        ax.set_title(f"int4head synthetic-acceptance ceiling\n{verdict}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png, dpi=120)
        print(f"\nwrote {png}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[plot] matplotlib unavailable ({exc}); text table only")
        png = None

    if args.wandb:
        try:
            import wandb
            run = wandb.init(entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                             group="bi0-int4head-accept-oracle", name="stark/accept-oracle-ceiling",
                             reinit=True, config={"pr": 813, "analysis_only": True})
            tbl = wandb.Table(columns=["rate", "mean_acc_len", "steady_gen_tps", "decode_wall_tps", "e_accept"])
            for d in rows:
                tbl.add_data(d["rate"], d["imposed_mean_acceptance_length"],
                             d.get("steady_gen_tps"), d.get("decode_wall_tps"),
                             d.get("e_accept_exact_from_log") or d.get("e_accept_mean_acceptance_length_prom"))
            log = {"ceiling_table": tbl}
            if anchor and ceil:
                log["ceiling_gain_pct"] = 100.0 * (ceil - anchor) / anchor
                log["anchor_r056_tps"] = anchor
                log["ceiling_r100_tps"] = ceil
            if png:
                log["ceiling_curve"] = wandb.Image(str(png))
            run.log(log)
            print(f"[wandb] logged ceiling run {run.id}")
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] ceiling log failed ({exc})")


if __name__ == "__main__":
    main()
