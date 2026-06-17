#!/usr/bin/env python
"""FlashInfer M-invariance — orchestrator (PR #582, analysis-only, LOCAL A10G).

Runs probe_one.py in 3 fresh subprocesses (env must be set before vLLM import):

  default_triton  VLLM_USE_FLASHINFER_SAMPLER=0, attention force-pinned TRITON
                  -> the 252.69 no-spec base_fullhead anchor stack.
  fi_sampler_on   VLLM_USE_FLASHINFER_SAMPLER=1, attention still TRITON
                  -> the literal PR "FlashInfer ON" knob. Proven a no-op at
                     greedy (sampler.py:266 returns argmax before topk_topp).
  fi_attention    VLLM_ATTENTION_BACKEND=FLASHINFER + sampler=1
                  -> the only FlashInfer lever that touches the forward
                     (reduction) path. The real M-invariance test.

Computes the PR deliverables and logs ONE W&B run, then prints SENPAI-RESULT.
The FlashInfer config used for the headline deliverables is fi_attention if it
loaded, else fi_sampler_on.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # /workspace/senpai/target
PY = str(ROOT / ".venv" / "bin" / "python")

DEFAULT_TPS_ANCHOR = 252.69  # base_fullhead no-spec, wirbel #553 run 83jiwjr9

CONFIGS = [
    ("default_triton", {"VLLM_USE_FLASHINFER_SAMPLER": "0"}),
    ("fi_sampler_on", {"VLLM_USE_FLASHINFER_SAMPLER": "1"}),
    (
        "fi_attention",
        {"VLLM_USE_FLASHINFER_SAMPLER": "1", "VLLM_ATTENTION_BACKEND": "FLASHINFER"},
    ),
]


def run_one(tag, extra_env):
    out = HERE / f"_result_{tag}.json"
    if out.exists():
        out.unlink()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PROBE_OUT"] = str(out)
    env["PROBE_TAG"] = tag
    env.setdefault("PROBE_N", "400")
    env.setdefault("PROBE_TPS_N", "512")
    # ensure a clean attention env unless the config sets it
    env.pop("VLLM_ATTENTION_BACKEND", None)
    env.update(extra_env)
    log = HERE / f"_result_{tag}.log"
    print(f"\n=== [{tag}] launch (attn={extra_env.get('VLLM_ATTENTION_BACKEND','<force-pin>')} "
          f"fi_sampler={extra_env.get('VLLM_USE_FLASHINFER_SAMPLER')}) ===", flush=True)
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        proc = subprocess.run(
            [PY, str(HERE / "probe_one.py")], env=env, stdout=lf, stderr=subprocess.STDOUT,
        )
    dt = time.perf_counter() - t0
    data = json.loads(out.read_text()) if out.exists() else {"load_ok": False, "error": "no-json"}
    data["_subprocess_rc"] = proc.returncode
    data["_wall_s"] = round(dt, 1)
    print(f"=== [{tag}] rc={proc.returncode} {dt:.0f}s load_ok={data.get('load_ok')} "
          f"err={data.get('error')} ===", flush=True)
    return data


def main():
    results = {tag: run_one(tag, env) for tag, env in CONFIGS}
    (HERE / "all_results.json").write_text(json.dumps(results, indent=2))

    default = results["default_triton"]
    fi_samp = results["fi_sampler_on"]
    fi_attn = results["fi_attention"]

    # Pick the strongest FlashInfer config that loaded for the headline verdict.
    if fi_attn.get("load_ok") and "m_invariance" in fi_attn:
        fi = fi_attn
        fi_used = "fi_attention"
    elif fi_samp.get("load_ok") and "m_invariance" in fi_samp:
        fi = fi_samp
        fi_used = "fi_sampler_on"
    else:
        fi = None
        fi_used = "none"

    def tps_of(r):
        return (r.get("tps") or {}).get("decode_tps_warm_median")

    default_tps = tps_of(default)

    deliver = {
        "analysis_only": True,
        "official_tps": 0,
        "default_tps_anchor": DEFAULT_TPS_ANCHOR,
        "default_tps_measured": default_tps,
        "fi_config_used": fi_used,
        "fi_attention_loaded": bool(fi_attn.get("load_ok")),
        "fi_attention_resolved_backend": fi_attn.get("resolved_attention_backend"),
        "fi_attention_load_error": fi_attn.get("error"),
    }

    if fi is not None:
        mi = fi["m_invariance"]
        fi_tps = tps_of(fi)
        deliver.update(
            {
                "flashinfer_byte_exact_m_invariant": bool(mi["byte_exact_m_invariant"]),
                "flashinfer_self_det": float(mi["self_det_min"]),
                "flashinfer_tps": fi_tps,
                "flashinfer_vs_default_tps_delta": (
                    None if (fi_tps is None or default_tps is None)
                    else round(fi_tps - default_tps, 4)
                ),
                "flashinfer_free_identity_lever": bool(
                    mi["byte_exact_m_invariant"]
                    and fi_tps is not None
                    and default_tps is not None
                    and fi_tps >= default_tps
                ),
            }
        )
    else:
        deliver.update(
            {
                "flashinfer_byte_exact_m_invariant": False,
                "flashinfer_self_det": None,
                "flashinfer_tps": None,
                "flashinfer_vs_default_tps_delta": None,
                "flashinfer_free_identity_lever": False,
            }
        )

    # sampler-flag no-op control: is fi_sampler_on byte-identical to default?
    def head_ids(r):
        try:
            return r["m_invariance"]["target_ids_head"]["1"]
        except Exception:
            return None
    deliver["sampler_flag_is_noop_vs_default"] = (
        head_ids(default) is not None and head_ids(default) == head_ids(fi_samp)
    )

    print("\n==== DELIVERABLES ====")
    print(json.dumps(deliver, indent=2))
    (HERE / "deliverables.json").write_text(json.dumps(deliver, indent=2))

    _log_wandb(deliver, results)

    # SENPAI-RESULT terminal marker
    rid = deliver.get("_wandb_run_id", "")
    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "official_tps": 0,
        "wandb_run_ids": [rid] if rid else [],
        "primary_metric": {"name": "flashinfer_tps", "value": deliver.get("flashinfer_tps")},
        "test_metric": {"name": "flashinfer_self_det", "value": deliver.get("flashinfer_self_det")},
    }
    print("\nSENPAI-RESULT: " + json.dumps(marker))


def _log_wandb(deliver, results):
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] import failed: {exc!r}; JSON saved only")
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="stark",
        name="stark/flashinfer-m-invariance",
        group="flashinfer-determinism-zoomout",
        notes="PR#582 ZOOM-OUT wild-card: is FlashInfer's batch-1 GEMV/sampler reduction "
              "byte-exact M-invariant by construction (a free #319-safe identity lever)? "
              "base_fullhead no-spec stock int4 full-262k-head, greedy temp=0. Probes greedy "
              "token-id flip rate across decode batch width M={1,8,16} under default TRITON, "
              "FlashInfer sampler ON, and FlashInfer attention; warm-median decode TPS each.",
        config={"pr": 582, "analysis_only": True, "official_tps": 0,
                "checkpoint": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "stack": "base_fullhead_no_spec", "default_tps_anchor": DEFAULT_TPS_ANCHOR,
                "widths": [1, 8, 16], "decode_len": 200, "tps_len": 512},
    )
    if run is None:
        print("[wandb] disabled (no key); JSON only")
        return
    for k, v in deliver.items():
        if isinstance(v, (int, float, bool, str)):
            run.summary[k] = v
    # per-config detail
    for tag, r in results.items():
        mi = r.get("m_invariance") or {}
        cw = mi.get("cross_width") or {}
        run.summary[f"{tag}/load_ok"] = bool(r.get("load_ok"))
        run.summary[f"{tag}/resolved_backend"] = str(r.get("resolved_attention_backend"))
        if "byte_exact_m_invariant" in mi:
            run.summary[f"{tag}/byte_exact_m_invariant"] = bool(mi["byte_exact_m_invariant"])
            run.summary[f"{tag}/self_det_min"] = mi.get("self_det_min")
        for pair, d in cw.items():
            run.summary[f"{tag}/flip_{pair}"] = d.get("flip_rate")
            run.summary[f"{tag}/steadyflip_{pair}"] = d.get("steady_flip_rate")
        tps = r.get("tps") or {}
        if tps.get("decode_tps_warm_median") is not None:
            run.summary[f"{tag}/decode_tps"] = tps["decode_tps_warm_median"]
            run.summary[f"{tag}/total_tps"] = tps.get("total_tps_warm_median")
    deliver["_wandb_run_id"] = run.id
    print(f"[wandb] run id={run.id} url={run.url}")
    try:
        finish_wandb(run)
    except Exception:
        pass


if __name__ == "__main__":
    main()
