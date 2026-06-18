#!/usr/bin/env python
"""PR #630: consolidate the prefix-cache self-determinism diagnostic + log to W&B.

Reads the captured artifacts (main 3-arm speed census, prefix-cache ON/OFF
diagnostics, the pairwise determinism matrix, the roofline) and:
  * builds one consolidated determinism report,
  * logs it as a SECOND W&B run in group ``zoomout-ar-speed-screen`` (the first
    run vxh2u99u carries the speed verdict; this one carries the determinism
    SURFACE), analysis_only / official_tps=0,
  * writes out/determinism_report.json.

Pure analysis; no GPU, no serve.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if p not in ("", str(Path(__file__).resolve().parent))]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "out"


def load(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    census = load("census_report.json")
    diag_off = load("diag_pcache_off.json")
    diag_on = load("diag_pcache_on.json")
    matrix = load("determinism_matrix.json")
    roof = load("roofline.json")

    stock_self = (census or {}).get("self_determinism_r1_vs_r2", {})
    report = {
        "kind": "zoomout-prefix-cache-determinism",
        "pr": 630,
        "analysis_only": True,
        "official_tps": 0,
        "rung": "int4_g128_lmhead",
        "engine": "vllm-0.22.0 TRITON_ATTN (forced, het head_dim 256/512)",
        "concurrency": 1,
        "num_prompts": 64,
        "output_len": 512,
        # the headline determinism contrast
        "stock_serve_py_prefix_cache_on": {
            "warm_pass_self_determinism_r1_vs_r2_byte_exact": stock_self.get("byte_exact"),
            "n_token_mismatch": stock_self.get("n_token_mismatch"),
            "n_compared": stock_self.get("n_compared"),
            "prompt_sha_parity": stock_self.get("prompt_sha_parity"),
            "min_first_divergence": stock_self.get("min_first_divergence"),
        },
        "standalone_prefix_cache_on_control": {
            "self_determinism": (diag_on or {}).get("self_determinism_r1_vs_r2"),
            "warm_median_tps_r1": (diag_on or {}).get("warm_median_tps_r1"),
            "warm_median_tps_r2": (diag_on or {}).get("warm_median_tps_r2"),
        },
        "standalone_prefix_cache_off": {
            "self_determinism": (diag_off or {}).get("self_determinism_r1_vs_r2"),
            "warm_median_tps_r1": (diag_off or {}).get("warm_median_tps_r1"),
            "warm_median_tps_r2": (diag_off or {}).get("warm_median_tps_r2"),
        },
        "determinism_matrix": matrix,
        "mechanism": (
            "enable_prefix_caching=True (vLLM V1 default): pass-2 reuses pass-1's "
            "block-cached prefix KV under a different chunk-boundary alignment than "
            "pass-1's cold full chunked prefill -> int4-Marlin grid-tie flips at "
            "greedy argmax (same int4-tie family as #616 0.43% / #607 / #621). "
            "Cold first-pass (the official sglang bench + #319 cross-start gate) is "
            "unaffected: every fresh server start re-derives the same cold tokens "
            "(corroborates lawine #606 cross-start 128/128). Disabling prefix "
            "caching is ~0 TPS at unique-prompt M=1 and restores warm-pass identity."
        ),
    }

    # derive the clean verdict booleans
    off_be = ((diag_off or {}).get("self_determinism_r1_vs_r2") or {}).get("byte_exact")
    on_be = ((diag_on or {}).get("self_determinism_r1_vs_r2") or {}).get("byte_exact")
    report["prefix_cache_is_the_driver"] = bool(off_be is True and on_be is False
                                                and stock_self.get("byte_exact") is False)
    report["prefix_cache_off_is_speed_neutral"] = None
    try:
        off_tps = (diag_off or {}).get("warm_median_tps_r1")
        stock_tps = (census or {}).get("stock_warm_median_tps_local")
        if isinstance(off_tps, (int, float)) and isinstance(stock_tps, (int, float)):
            report["prefix_cache_off_tps_delta_vs_stock"] = round(off_tps - stock_tps, 3)
            report["prefix_cache_off_is_speed_neutral"] = abs(off_tps - stock_tps) < 1.0
    except Exception:  # noqa: BLE001
        pass

    (OUT / "determinism_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("determinism_matrix", "mechanism")}, indent=2))

    # ---- W&B (second run; speed verdict already in vxh2u99u) ----
    try:
        from scripts.wandb_logging import finish_wandb, init_wandb_run, log_json_artifact, log_summary
    except Exception as exc:  # noqa: BLE001
        print(f"[finalize] wandb unavailable: {exc}")
        return 0
    run = init_wandb_run(
        job_type="systems-screen", agent="wirbel",
        name="wirbel/zoomout-prefix-cache-determinism",
        group="zoomout-ar-speed-screen",
        tags=["zoomout-ar-speed", "prefix-cache-determinism", "byte-exact-census",
              "local-a10g", "analysis-only", "pr630"],
        notes="PR #630 SURFACE: stock M=1 greedy warm-pass self-determinism = 34/64; "
              "driver is enable_prefix_caching; disabling restores 0/64 at ~0 TPS cost.",
        config={"rung": "int4_g128_lmhead", "concurrency": 1, "num_prompts": 64,
                "output_len": 512, "speed_run_id": (census or {}).get("wandb_run_id")},
    )
    if run is None:
        print("[finalize] wandb init returned None (no API key / disabled) — skipping")
        return 0
    flat = {
        "stock_on_self_determinism_mismatch": stock_self.get("n_token_mismatch"),
        "stock_on_self_determinism_byte_exact": stock_self.get("byte_exact"),
        "standalone_on_mismatch": ((diag_on or {}).get("self_determinism_r1_vs_r2") or {}).get("n_token_mismatch"),
        "standalone_off_mismatch": ((diag_off or {}).get("self_determinism_r1_vs_r2") or {}).get("n_token_mismatch"),
        "standalone_off_byte_exact": off_be,
        "prefix_cache_is_the_driver": report["prefix_cache_is_the_driver"],
        "prefix_cache_off_tps_delta_vs_stock": report.get("prefix_cache_off_tps_delta_vs_stock"),
        "prefix_cache_off_warm_tps": (diag_off or {}).get("warm_median_tps_r1"),
        "stock_warm_tps": (census or {}).get("stock_warm_median_tps_local"),
        "analysis_only": True, "official_tps": 0,
    }
    log_summary(run, {k: v for k, v in flat.items() if v is not None}, step=0)
    log_json_artifact(run, name="zoomout-prefix-cache-determinism",
                      artifact_type="determinism-census", data=report)
    if roof is not None:
        log_json_artifact(run, name="zoomout-ar-speed-roofline",
                          artifact_type="roofline", data=roof)
    rid = getattr(run, "id", None)
    finish_wandb(run)
    print(f"[finalize] wandb run id = {rid}")
    report["wandb_run_id"] = rid
    (OUT / "determinism_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
