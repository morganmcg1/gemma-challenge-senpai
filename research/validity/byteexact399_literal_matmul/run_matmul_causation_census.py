"""PR #518 Task-1 — matmul-causation census on the byteexact-399 candidate.

QUESTION (advisor PR #518, instr 1): does adding the full VLLM_BATCH_INVARIANT=1
matmul tax to the byteexact-399 candidate drive its 5 e2e ULP-tie flips -> 0
(proving the flips are matmul-induced)? And what is the TPS cost of that tax?

WHY THIS IS THE MISSING CELL. The canonical #461 census (advisor branch,
``deployed_flip_attribution.py``) ran 4 arms over the SAME #461 logit-margin
locus (127 prompts x ctx 224, teacher-forced M=8 prompt_logprobs vs M=1 AR, NO
greedy cascade):
    deployed     stock adaptive 3D split-KV attn  + stock matmul  -> 4 flips
    attn_only    2D in-order attn (is_batch_invariant=True)        -> 2 flips
    lmhead_only  stock attn + aten matmul tax (enable_bi_mode)      -> 4 flips  (== deployed; tax INERT)
    all_pin      2D in-order attn + aten matmul tax (env=1)         -> 2 flips  (== attn_only)
=> #461 already shows the aten matmul tax reduces flips by EXACTLY ZERO; the only
flip-reducing lever is the ATTENTION 2D in-order pin. #501 (advisor branch,
``microbench_gemm_tax.py``) independently shows the int4 Marlin GEMM is byte-exact
M-invariant at M<=8 (0 flips, flips only at M>=32) and exposes NO Python split-K knob.

But neither ran the byteexact arm WITH the matmul tax. #500 ran byteexact WITHOUT
the tax (5 flips). This wrapper runs the missing combinations to answer instr 1
directly on byteexact, in one churn-controlled GPU session:

  byteexact         byteexact 3D fixed-order attn (is_bi=F) + stock matmul   [#500 baseline; expect ~5]
  byteexact_bi      byteexact 3D fixed-order attn (is_bi=F) + aten matmul tax [the TRUE "byteexact attn + tax";
                    enable_batch_invariant_mode() patches aten mm/addmm/matmul/linear/bmm/softmax/mean
                    WITHOUT touching the attention global -> byteexact attn KEPT. expect ~5 if tax inert
                    (REFUTES matmul causation) ]
  byteexact_fullenv byteexact patch installed + env VLLM_BATCH_INVARIANT=1      [the LITERAL advisor instr;
                    but env=1 sets is_batch_invariant=True -> attention REVERTS to 2D in-order, so the
                    byteexact 3D rewrite is moot. == all_pin. expect ~2. demonstrates the env's flip change
                    is the attention 2D-reversion, NOT the matmul. ]

KEY MECHANISM FACT (verified from vllm batch_invariant.py): on A10G sm_86
(is_device_capability_family(80)=True) enable_batch_invariant_mode() installs
aten::mm/addmm/matmul/linear/bmm/_log_softmax/softmax/mean overrides only; it does
NOT set triton_unified_attention.is_batch_invariant. The int4 production matmuls go
through torch.ops._C.marlin_gemm (a custom op, NOT aten::mm) so the tax cannot
touch them -> in an all-int4 model the tax patches only the few bf16 aten ops, none
of which move the argmax. That is the mechanistic reason the tax is flip-inert.

LOCAL ONLY. analysis_only=true, official_tps=0. No HF job, no submission.

  .venv/bin/python -m research.validity.byteexact399_literal_matmul.run_matmul_causation_census \
      --n-prompts 128 --arms byteexact,byteexact_bi,byteexact_fullenv \
      --wandb_name lawine/byteexact399-matmul-causation --wandb_group byteexact399-literal-matmul
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DFA_DIR = ROOT / "research" / "validity" / "deployed_flip_attribution"
SUBMISSION_DIR = ROOT / "submissions" / "fa2sw_strict_byteexact_splitkv399"
CERT_OUT = ROOT / "research" / "validity" / "byteexact399_literal_matmul" / "causation_census"

if str(DFA_DIR) not in sys.path:
    sys.path.insert(0, str(DFA_DIR))
import deployed_flip_attribution as dfa  # noqa: E402

NEAR_TIE = dfa.NEAR_TIE_LOGPROB_THRESH  # 0.5 nat
FIXED_TPS_DEFAULT = "4"
NUM_SEGMENTS_DEFAULT = "64"
BYTEEXACT_ARMS = ("byteexact", "byteexact_bi", "byteexact_fullenv")

_orig_apply_arm_pin = dfa.apply_arm_pin


def _install_byteexact() -> bool:
    os.environ.setdefault("BYTEEXACT_FIXED_TPS", FIXED_TPS_DEFAULT)
    os.environ.setdefault("BYTEEXACT_NUM_SEGMENTS", NUM_SEGMENTS_DEFAULT)
    if str(SUBMISSION_DIR) not in sys.path:
        sys.path.insert(0, str(SUBMISSION_DIR))
    import byteexact_splitkv_patch as bx  # noqa: E402
    return bool(bx.install())


def patched_apply_arm_pin(arm: str) -> dict:
    if arm not in BYTEEXACT_ARMS:
        return _orig_apply_arm_pin(arm)
    installed = _install_byteexact()
    import vllm.v1.attention.ops.triton_unified_attention as _ua
    import vllm.v1.attention.backends.triton_attn as _ta

    tax_installed = False
    if arm == "byteexact_bi":
        # add ONLY the aten matmul-family tax; do NOT set the attention global, so the
        # byteexact 3D fixed-order attention is kept (is_batch_invariant stays False).
        from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode
        enable_batch_invariant_mode()
        tax_installed = True
    # byteexact_fullenv: env VLLM_BATCH_INVARIANT=1 was set at process start, so
    # init_batch_invariance() already (a) set is_batch_invariant=True (attn -> 2D in-order),
    # (b) installed the aten tax, (c) TF32-off + NCCL/cuBLAS determinism. nothing to add here.

    flags = {
        "arm": arm,
        "attn_pin_requested": (arm == "byteexact_fullenv"),
        "lmhead_pin_requested": (arm in ("byteexact_bi", "byteexact_fullenv")),
        "rms_env_set": (arm == "byteexact_fullenv"),
        "byteexact_installed": installed,
        "matmul_tax_installed_inproc": tax_installed,
        "byteexact_fixed_tps_marker": getattr(_ua, "_byteexact_fixed_tps", None),
        "backend_num_par_softmax_segments": getattr(_ta, "NUM_PAR_SOFTMAX_SEGMENTS", None),
    }
    print(f"[pin:{arm}] byteexact.install()={installed} tax_inproc={tax_installed} "
          f"fixed_tps_marker={flags['byteexact_fixed_tps_marker']} "
          f"backend_segments={flags['backend_num_par_softmax_segments']} "
          f"env_VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT', '0')}", flush=True)
    return flags


def run_phase(a: argparse.Namespace) -> None:
    dfa.apply_arm_pin = patched_apply_arm_pin
    dfa.phase_arm(a.out, a.arm, a.n_prompts, a.ctx_len, a.n_verify,
                  a.gpu_mem_util, a.max_batched_tokens, a.verbose_k, a.microbench)


def _spawn_arm(a: argparse.Namespace, arm: str) -> dict:
    out_json = str(CERT_OUT / f"arm_{arm}_result.json")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["BYTEEXACT_FIXED_TPS"] = str(a.fixed_tps)
    env["BYTEEXACT_NUM_SEGMENTS"] = str(a.num_segments)
    # byteexact_fullenv pays the FULL env tax at process start (attn->2D + aten tax + TF32-off);
    # byteexact / byteexact_bi keep env=0 (attn stays byteexact 3D; byteexact_bi adds only the
    # aten tax in-process post-load).
    env["VLLM_BATCH_INVARIANT"] = "1" if arm == "byteexact_fullenv" else "0"
    micro = ["--no-microbench"]
    cmd = [sys.executable, os.path.abspath(__file__),
           "--phase", "arm", "--arm", arm, "--out", out_json,
           "--n-prompts", str(a.n_prompts), "--ctx-len", str(a.ctx_len),
           "--n-verify", str(a.n_verify), "--gpu-mem-util", str(a.gpu_mem_util),
           "--max-batched-tokens", str(a.max_batched_tokens),
           "--verbose-k", str(a.verbose_k)] + micro
    print(f"[orch] launching arm={arm} (VLLM_BATCH_INVARIANT={env['VLLM_BATCH_INVARIANT']} "
          f"BYTEEXACT_FIXED_TPS={env['BYTEEXACT_FIXED_TPS']})", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"arm subprocess failed (rc={rc}): {arm}")
    return json.load(open(out_json))


def _classify_flips(data: dict) -> dict:
    flips = []
    for pp in data["per_prompt"]:
        pid = pp.get("id", pp.get("ri"))
        for pr in pp["pos_records"]:
            if not pr["match"]:
                flips.append({"id": pid, "pos": pr["pos"], "rel_pos": pr["rel_pos"],
                              "m8_tok": pr["m8_tok"], "m1_tok": pr["m1_tok"],
                              "margin_vs_m1": pr["margin_vs_m1"], "gap_top2": pr["gap_top2"]})
    ulp = [f for f in flips if f["margin_vs_m1"] is not None and f["margin_vs_m1"] < NEAR_TIE]
    semantic = [f for f in flips if f["margin_vs_m1"] is None or f["margin_vs_m1"] >= NEAR_TIE]
    margins = [f["margin_vs_m1"] for f in flips if f["margin_vs_m1"] is not None]
    eng = data.get("engaged", {})
    return {
        "identity": data["identity"],
        "divergence": data.get("divergence"),
        "n_flips": len(flips),
        "n_ulp_near_tie": len(ulp),
        "n_semantic": len(semantic),
        "max_margin_nat": (max(margins) if margins else None),
        "min_gap_top2_nat": (min((f["gap_top2"] for f in flips if f["gap_top2"] is not None), default=None)),
        "attn_is_batch_invariant": eng.get("attn_is_batch_invariant"),
        "aten_mm_bitexact_M1_vs_M8": eng.get("aten_mm_bitexact_M1_vs_M8"),
        "determinism_M1_vs_M1": data["determinism_M1_vs_M1"],
        "determinism_M8_vs_M8": data["determinism_M8_vs_M8"],
        "within_batch_copy0_vs_copy1": data["within_batch_copy0_vs_copy1"],
        "chunk_isolated_fraction": data["chunk_isolated_fraction"],
        "median_chunk_width": data["median_chunk_width"],
        "pin_flags": data.get("pin_flags"),
        "flip_ids": sorted({str(f["id"]) for f in flips}),
        "flips": flips,
    }


def orchestrate(a: argparse.Namespace) -> None:
    CERT_OUT.mkdir(parents=True, exist_ok=True)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    print(f"[causation-census] arms={arms} n_prompts={a.n_prompts} C(ctx)={a.ctx_len} "
          f"n_verify={a.n_verify} NEAR_TIE={NEAR_TIE} nat -> {CERT_OUT}", flush=True)
    per_arm = {}
    for arm in arms:
        data = _spawn_arm(a, arm)
        per_arm[arm] = _classify_flips(data)
        c = per_arm[arm]
        print(f"[causation-census] {arm:18s} identity={c['identity']:.6f} "
              f"flips={c['n_flips']} (ulp={c['n_ulp_near_tie']} semantic={c['n_semantic']}) "
              f"attn_is_bi={c['attn_is_batch_invariant']} aten_tax={c['aten_mm_bitexact_M1_vs_M8']} "
              f"max_margin={c['max_margin_nat']} nat", flush=True)

    be = per_arm.get("byteexact")
    bi = per_arm.get("byteexact_bi")
    fe = per_arm.get("byteexact_fullenv")
    verdict = {"available": be is not None and bi is not None}
    if be is not None and bi is not None:
        # matmul-causation test: does the aten tax (attention held byteexact) zero the flips?
        tax_reduces_flips = bi["n_flips"] < be["n_flips"]
        tax_zeros_flips = bi["n_flips"] == 0
        matmul_tax_inert = (bi["n_flips"] == be["n_flips"] and set(bi["flip_ids"]) == set(be["flip_ids"]))
        verdict.update({
            "byteexact_n_flips": be["n_flips"],
            "byteexact_bi_n_flips": bi["n_flips"],
            "byteexact_bi_attn_is_batch_invariant": bi["attn_is_batch_invariant"],
            "byteexact_bi_aten_mm_bitexact": bi["aten_mm_bitexact_M1_vs_M8"],
            "matmul_tax_zeros_byteexact_flips": bool(tax_zeros_flips),
            "matmul_tax_reduces_byteexact_flips": bool(tax_reduces_flips),
            "matmul_tax_inert_on_byteexact": bool(matmul_tax_inert),
            "matmul_causation_confirmed": bool(tax_zeros_flips),
            "byteexact_semantic": be["n_semantic"],
            "byteexact_bi_semantic": bi["n_semantic"],
        })
        if fe is not None:
            verdict.update({
                "byteexact_fullenv_n_flips": fe["n_flips"],
                "byteexact_fullenv_attn_is_batch_invariant": fe["attn_is_batch_invariant"],
                "byteexact_fullenv_aten_mm_bitexact": fe["aten_mm_bitexact_M1_vs_M8"],
                "fullenv_reverts_attn_to_2d": (fe["attn_is_batch_invariant"] is True),
                "fullenv_flip_reduction_is_attn_reversion": bool(
                    fe["attn_is_batch_invariant"] is True and fe["n_flips"] < be["n_flips"]),
            })
    report = {
        "pr": 518, "task": 1, "generated_by": "lawine", "analysis_only": True, "official_tps": 0,
        "no_hf_job": True, "no_launch": True, "no_submission": True,
        "candidate_path": "submissions/fa2sw_strict_byteexact_splitkv399/",
        "methodology": "#461 logit-margin locus (teacher-forced M=8 prompt_logprobs vs M=1 AR, no greedy cascade); "
                       "enforce_eager (no cudagraph confound); same locus surgical-357 + #500 byteexact were certified on",
        "mechanism_note": "enable_batch_invariant_mode() patches aten mm/addmm/matmul/linear/bmm/softmax/mean only; "
                          "it does NOT touch triton_unified_attention.is_batch_invariant. int4 production matmuls go "
                          "through torch.ops._C.marlin_gemm (custom op, NOT aten::mm) so the tax cannot touch them.",
        "candidate": {"fixed_tps": a.fixed_tps, "num_segments": a.num_segments},
        "arms": per_arm, "verdict": verdict,
    }
    out_report = CERT_OUT / "matmul_causation_census_report.json"
    json.dump(report, open(out_report, "w"), indent=2)
    print(f"[causation-census] wrote {out_report}", flush=True)
    print("=" * 92)
    if be is not None and bi is not None:
        print(f"[causation-census] VERDICT matmul_causation_confirmed={verdict['matmul_causation_confirmed']} "
              f"| byteexact flips={be['n_flips']} -> byteexact_bi(+tax) flips={bi['n_flips']} "
              f"(tax_inert={verdict['matmul_tax_inert_on_byteexact']}, "
              f"byteexact_bi attn_is_bi={bi['attn_is_batch_invariant']} aten_tax={bi['aten_mm_bitexact_M1_vs_M8']})", flush=True)
        if fe is not None:
            print(f"[causation-census]   byteexact_fullenv(env=1) flips={fe['n_flips']} "
                  f"attn_is_bi={fe['attn_is_batch_invariant']} (env reverts attn to 2D)", flush=True)
    print("=" * 92)

    if not a.no_wandb:
        _log_wandb(a, report)


def _log_wandb(a: argparse.Namespace, report: dict) -> None:
    try:
        sys.path.insert(0, os.getcwd())
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; JSON only", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling", agent="lawine", name=a.wandb_name, group=a.wandb_group,
        notes="PR#518 Task1 matmul-causation census: does the VLLM_BATCH_INVARIANT matmul tax zero "
              "byteexact-399's e2e flips? byteexact vs byteexact_bi(+aten tax, attn kept) vs "
              "byteexact_fullenv(env=1, attn->2D). On the #461 locus.",
        config={"pr": 518, "task": 1, "n_prompts": a.n_prompts, "C": a.ctx_len, "n_verify": a.n_verify,
                "fixed_tps": a.fixed_tps, "num_segments": a.num_segments,
                "near_tie_thresh_nat": NEAR_TIE, "analysis_only": True, "official_tps": 0},
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); JSON only", flush=True)
        return
    v = report["verdict"]
    for k, val in v.items():
        if isinstance(val, (int, float, bool)) or val is None:
            run.summary[f"causation/{k}"] = val
    for arm, c in report["arms"].items():
        run.summary[f"{arm}/identity"] = c["identity"]
        run.summary[f"{arm}/n_flips"] = c["n_flips"]
        run.summary[f"{arm}/n_semantic"] = c["n_semantic"]
        run.summary[f"{arm}/attn_is_batch_invariant"] = c["attn_is_batch_invariant"]
        run.summary[f"{arm}/aten_mm_bitexact"] = c["aten_mm_bitexact_M1_vs_M8"]
        run.summary[f"{arm}/max_margin_nat"] = c["max_margin_nat"]
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["arm"], default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="byteexact,byteexact_bi,byteexact_fullenv")
    ap.add_argument("--n-prompts", dest="n_prompts", type=int, default=128)
    ap.add_argument("--ctx-len", dest="ctx_len", type=int, default=224)
    ap.add_argument("--n-verify", dest="n_verify", type=int, default=dfa.M_VERIFY)
    ap.add_argument("--gpu-mem-util", dest="gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", dest="max_batched_tokens", type=int, default=8192)
    ap.add_argument("--verbose-k", dest="verbose_k", type=int, default=3)
    ap.add_argument("--fixed-tps", dest="fixed_tps", type=int, default=4)
    ap.add_argument("--num-segments", dest="num_segments", type=int, default=64)
    ap.add_argument("--microbench", dest="microbench", action="store_true")
    ap.add_argument("--no-microbench", dest="microbench", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wandb_name", dest="wandb_name", default="lawine/byteexact399-matmul-causation")
    ap.add_argument("--wandb_group", dest="wandb_group", default="byteexact399-literal-matmul")
    ap.add_argument("--no-wandb", action="store_true")
    ap.set_defaults(microbench=False)
    a = ap.parse_args()

    if a.smoke:
        a.n_prompts = min(a.n_prompts, 6)

    if a.phase == "arm":
        run_phase(a)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
