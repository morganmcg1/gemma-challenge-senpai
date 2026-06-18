#!/usr/bin/env python
"""Re-log a deproject_runner result JSON to W&B.

The de-projection arms were orchestrated under the vLLM *server* venv
(``/tmp/senpai-venvs/...``), which has no ``wandb`` installed -- so the runner's
end-of-run ``_log_wandb`` raised ``AttributeError`` (a local ``wandb/`` run-data
dir shadowed the package as a namespace package) AFTER the result JSON was
written. The compute is intact in the JSON; this re-logs it to W&B from the
project ``.venv`` (run via ``uv run python``). One fresh run per JSON, same
group/name/summary schema the runner would have emitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.wandb_logging import init_wandb_run, finish_wandb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", default="optionb-rescue-deproject-stark")
    a = ap.parse_args()

    result = json.loads(a.json.read_text())
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name=a.name, group=a.group,
        notes=f"PR#642 de-project #636 recompute acceptor ({result.get('leg')}): "
              f"real served wall_tps cost [re-logged from JSON]",
        config={"pr": 642, "mode": result.get("leg"), "submission": result.get("submission"),
                "extra_env": result.get("extra_env"), "n": result.get("n"),
                **{f"workload_{k}": v for k, v in (result.get("workload") or {}).items()},
                "relogged_from_json": str(a.json)},
    )
    if run is None:
        print("[relog] init_wandb_run returned None (wandb unavailable / disabled)")
        return 1

    summary: dict = {}
    for lbl, info in (result.get("arms") or {}).items():
        if isinstance(info.get("wall_tps_median"), (int, float)):
            summary[f"arm/{lbl}/wall_tps"] = info["wall_tps_median"]
        if isinstance(info.get("e_accept_exact_mean"), (int, float)):
            summary[f"arm/{lbl}/e_accept"] = info["e_accept_exact_mean"]
    fit = result.get("additive_cost_fit") or {}
    for k_src, k_dst in [("C_sec_per_recompute", "fit/C_sec_per_recompute"),
                         ("C_over_636_assumption", "fit/C_over_636_assumption"),
                         ("tps0_fit", "fit/tps0_fit"), ("r2", "fit/r2")]:
        if isinstance(fit.get(k_src), (int, float)):
            summary[k_dst] = fit[k_src]
    ap_ = result.get("acceptor_prediction") or {}
    if ap_:
        summary["acceptor/ftr"] = ap_.get("ftr")
        summary["acceptor/wall_tps_from_slope"] = ap_.get("wall_tps_from_slope")
    rate_map = result.get("rate_to_wall_tps") or {}
    for r, t in rate_map.items():
        if isinstance(t, (int, float)):
            summary[f"rate/{r}/wall_tps"] = t
    for k, v in summary.items():
        run.summary[k] = v
    finish_wandb(run)
    print(f"[relog] logged run {run.id} ({a.name}) with {len(summary)} summary keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
