"""Default-vs-default eager-noise FLOOR control for the bm4 census (PR #442, wirbel).

The paired census (``served_bm4_census.py``) found bm4-vs-default = 53.1% token-identical
(30/64 prompts diverge) on the served M=8 verify path. The advisor's lawine #438 caveat
warns that the M=8 verify runs EAGER (vLLM v0.22.1rc1 captures only sizes [1,2]), so
cross-process FP non-determinism could inflate that divergence even with NO config change.

This control answers the question the census cannot on its own: serve the deployed DEFAULT
stack a SECOND time (fresh process, identical config, WIRBEL_BM4_AB unset, NO toggle) and
compare it token-by-token against the census's first default run. That is the eager
cross-process divergence FLOOR.

  * floor ~100% identical  -> the eager path is deterministic across processes; the census's
                              30/64 bm4 divergence is REAL (bm4 breaks greedy-identity).
  * floor ~= bm4 divergence -> the eager M=8 path is itself non-reproducible across processes;
                              the deployed stack is not token-stable and the census cannot
                              attribute the divergence to bm4 on the token axis (the wall-TPS
                              regression, being process-paired, still stands).

0 TPS, NO served-file change, NO toggle, NOT a launch/submission. Pure local measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
from scripts.local_validation import harness, paths  # noqa: E402
from research.validity.triton_attn_joint_autotune import served_bm4_census as census  # noqa: E402

SUBMISSION = "fa2sw_precache_kenyan"


def _log(msg: str) -> None:
    print(f"[bm4-floor] {msg}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--first-default-jsonl", type=Path,
                    default=HERE / "census_out" / "default" / "decode_outputs.jsonl",
                    help="the census's first default run to compare against")
    ap.add_argument("--out-root", type=Path, default=HERE / "floor_out")
    ap.add_argument("--bm4-vs-default-identical", type=float, default=0.53125,
                    help="census bm4-vs-default frac_identical, for side-by-side reporting")
    ap.add_argument("--wandb_group", default="triton-joint-autotune")
    ap.add_argument("--wandb_name", default="wirbel/served-bm4-eager-floor")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    first_default = args.first_default_jsonl.resolve()
    if not first_default.exists():
        _log(f"FATAL: first default run not found: {first_default} (run the census first)")
        return 2

    for note in paths.prepare_local_gpu_env():
        _log(note)

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = harness.load_manifest((ROOT / "submissions" / SUBMISSION).resolve())
    server_python = harness.ensure_server_venv(manifest["dependencies"])

    _log(f"eager-floor control: 2nd DEFAULT run {args.num_prompts}x{args.output_len} "
         f"seed={args.seed}; compare vs {first_default}")

    t0 = time.time()
    # plain default: WIRBEL_BM4_AB unset, no toggle applied -> injector never imported.
    arm = census._serve_and_capture(server_python, out_root, "default2", {}, args, want_ppl=False)

    cmp = census.compare_arms(first_default, Path(arm["decode_jsonl"]))

    # default arm must be clean (no bm4 anywhere) on BOTH runs by construction
    default2_clean = not arm["bm4_patched"] and not arm["bm4_hook_failed"]

    result = {
        "experiment": "served_bm4_eager_floor", "pr": 442, "student": "wirbel",
        "control": "default-vs-default (eager cross-process FP-noise floor)",
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "seed": args.seed},
        "floor_comparison": {k: v for k, v in cmp.items() if k != "per_prompt"},
        "bm4_vs_default_frac_identical": args.bm4_vs_default_identical,
        "default2_clean": default2_clean,
        "interpretation": _interpret(cmp["frac_identical"], args.bm4_vs_default_identical),
        "elapsed_s": time.time() - t0,
    }
    (out_root / "results.json").write_text(json.dumps(result, indent=2, default=str))
    (out_root / "per_prompt.json").write_text(json.dumps(cmp["per_prompt"], indent=2))

    print("\n" + "=" * 78, flush=True)
    print("DEFAULT-vs-DEFAULT EAGER-NOISE FLOOR (PR #442, wirbel)", flush=True)
    print("=" * 78, flush=True)
    print(f"  floor (default A vs default C, fresh processes, identical config):", flush=True)
    print(f"    n_prompts={cmp['n_prompts']} identical={cmp['n_identical']} "
          f"divergent={cmp['n_divergent']} frac_identical={cmp['frac_identical']}", flush=True)
    print(f"    token-prefix match = {cmp['frac_token_prefix_match']}", flush=True)
    print(f"  bm4-vs-default frac_identical (census) = {args.bm4_vs_default_identical}", flush=True)
    print(f"  >>> {result['interpretation']}", flush=True)
    print("=" * 78 + "\n", flush=True)

    if not args.no_wandb:
        _log_wandb(args, result)
    print(f"[bm4-floor] artifacts -> {out_root / 'results.json'}", flush=True)
    return 0


def _interpret(floor_identical: float | None, bm4_identical: float) -> str:
    if floor_identical is None:
        return "INCONCLUSIVE: no comparable prompts."
    if floor_identical >= 0.98:
        return (f"floor={floor_identical:.3f} ~CLEAN -> eager path is deterministic across "
                f"processes; the census bm4 divergence ({bm4_identical:.3f} identical) is REAL: "
                f"bm4 breaks greedy-identity on the served 3D verify path.")
    if floor_identical <= bm4_identical + 0.10:
        return (f"floor={floor_identical:.3f} ~= bm4 ({bm4_identical:.3f}) -> the eager M=8 path "
                f"is itself non-reproducible across processes; the token-identity census is "
                f"confounded (the wall-TPS regression still stands, being process-paired).")
    return (f"floor={floor_identical:.3f} vs bm4={bm4_identical:.3f} -> bm4 adds "
            f"{100*(floor_identical-bm4_identical):.1f}pp divergence ABOVE the eager floor: "
            f"a real bm4 identity effect on top of a non-zero eager-noise floor.")


def _log_wandb(args, result: dict[str, Any]) -> None:
    try:
        from scripts import wandb_logging
    except Exception as exc:  # noqa: BLE001
        _log(f"wandb import failed ({exc}); skipping")
        return
    run = wandb_logging.init_wandb_run(
        job_type="served-bm4-eager-floor", agent="wirbel",
        name=args.wandb_name, group=args.wandb_group,
        tags=["pr442", "eager-floor", "control", "bm4", SUBMISSION],
        config={"num_prompts": args.num_prompts, "output_len": args.output_len, "seed": args.seed},
    )
    if run is None:
        return
    try:
        c = result["floor_comparison"]
        flat = {
            "floor/n_prompts": c["n_prompts"], "floor/n_identical": c["n_identical"],
            "floor/n_divergent": c["n_divergent"],
            "floor/frac_identical": c["frac_identical"] or 0.0,
            "floor/frac_token_prefix_match": c["frac_token_prefix_match"] or 0.0,
            "bm4_vs_default_frac_identical": result["bm4_vs_default_frac_identical"],
        }
        wandb_logging.log_summary(run, flat, step=0)
        wandb_logging.log_json_artifact(run, name="served_bm4_eager_floor",
                                        artifact_type="eager-floor-control", data=result)
    except Exception as exc:  # noqa: BLE001
        _log(f"WARN wandb logging error: {exc}")
    finally:
        try:
            wandb_logging.finish_wandb(run)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
