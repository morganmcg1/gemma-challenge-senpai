#!/usr/bin/env python
"""PR #232 -- Int4 M=8 token-identity probe: the TRUE deployed greedy divergence.

The int4 counterpart of the merged #221 fp16-verify probe (6m40u2bg). #221's
Tier-2 measured M=1-vs-M=8 token identity through the int4->bf16 *decompressed*
transformers forward and found the bf16 floor: identity 0.98944 (divergence
0.01056), determinism 1.0 (RESIDUAL_BF16_BATCHVAR). This leg swaps the verify
path to the DEPLOYED int4-Marlin GEMM (served via vLLM, not bf16-decompressed)
and measures the same M=1-vs-M=8 per-position argmax identity over the served
128 prompts -- pinning the clean deployed-M=8 int4 greedy divergence.

The deliverable settles a launch-critical provenance gap: kanna #114's load-
bearing 56.08% greedy divergence (9q5yy9l1) was native-spec-vs-M1 with the
verify width UNRECORDED (reference_kind="unknown"). The clean deployed M=1-AR
vs M=8-verify int4 divergence -- the exact #114/#192 mechanism -- has never been
isolated. This probe isolates it, holding the served weights/quantization fixed
and varying ONLY the verify batch width M in {1, 8}.

Two geometries, both reported so the verdict is robust:
  (e2e PRIMARY) prompt_logprobs over M identical co-batched prefill replicas
    -- the SAME batch-width-replication geometry #221 used for the bf16 floor,
    so the int4 number is directly comparable. M=8 makes the int4-Marlin body
    GEMMs (+ bf16 tied lm_head + attention) see 8x the row count of M=1.
  (decode-width diagnostic) the isolated body GEMM row-0 bit-exactness at the
    LITERAL verify width M=8 (8 rows), reusing #221's kernel test -- localises
    whether any divergence is the int4 body or the bf16 lm_head/accumulation.

LOCAL profiling on a single A10G. No HF Job / no submission / no served-file
change / no official draw / no train.py --launch. BASELINE stays 481.53. The
served int4 path is READ, never modified. This leg adds 0 TPS (a measurement).

The GPU work runs as an isolated subprocess so vLLM gets a clean CUDA context
and releases VRAM on exit; the orchestrator stays GPU-free and owns composition,
self-test, and wandb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------------------
# Imported fleet anchors (DO NOT re-derive -- per PR #232 "import, do not re-derive")
# --------------------------------------------------------------------------------------
KANNA_114_DIV = 0.5608                 # #114 9q5yy9l1 native-spec-vs-M1 token_div_frac_max (M unrecorded)
STRICT_A_219 = 0.125                   # #219 0unwptbz strict-A per-token-theta reading
FP16_IDENTITY_221 = 0.98944091796875   # #221 6m40u2bg bf16 floor M1-vs-M8 token identity
FP16_DIVERGENCE_221 = 0.01055908203125 # #221 bf16 floor divergence (1 - identity)
FP16_SHA_EQUAL_FRAC_221 = 0.0390625    # #221 bf16 per-sequence strict (all-tokens) pass fraction
INT4_BODY_M_DEP_221 = False            # #221 isolated body GEMM row-0 bit-exact at M in {1,2,4,8,16}

OFFICIAL_BASELINE = 481.53             # #52 official TPS (this leg adds 0)
TARGET_TPS = 500.0
K_SPEC = 7                             # num_speculative_tokens (manifest)
M_VERIFY = K_SPEC + 1                  # = 8, the deployed verify batch width
CONFIRM_TOL = 0.05                     # |int4_div - 0.5608| <= tol  => confirms #114 (~0.56)

# Canonical Hub int4 checkpoint -- the SAME one #221 used. Body QKV/MLP GEMMs are
# the deployed int4 Marlin w4a16 kernel; lm_head is in the quant ignore list and
# tied to the bf16 embeddings, so the final vocab projection is bf16 (documented in
# the report's honest band -- the deployed osoi5 lm_head differs but needs serve
# patches to load under vanilla vLLM, and #221 timed/probed this same checkpoint).
MODEL_CANDIDATES = [
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    ),
    "/tmp/osoi5-v0-baked",
]
PROMPTS_JSONL = "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"

OUT_DIR = Path("research/validity/int4_tokenident_deployed_m8")


# --------------------------------------------------------------------------------------
# Small helpers (resolve_model_dir / read_text_dims / row0 test reused from #221)
# --------------------------------------------------------------------------------------
def resolve_model_dir() -> str:
    for cand in MODEL_CANDIDATES:
        p = Path(cand)
        if p.is_dir() and (p / "config.json").exists():
            return str(p)
        if p.is_dir():
            for sub in sorted(p.glob("*")):
                if (sub / "config.json").exists():
                    return str(sub)
    raise FileNotFoundError(f"no int4 model found among {MODEL_CANDIDATES}")


def read_text_dims(model_dir: str) -> dict:
    cfg = json.load(open(Path(model_dir) / "config.json"))
    tc = cfg.get("text_config", cfg)
    h = tc["hidden_size"]
    n_heads = tc["num_attention_heads"]
    n_kv = tc["num_key_value_heads"]
    hd = tc["head_dim"]
    inter = tc["intermediate_size"]
    return {
        "hidden": h, "n_heads": n_heads, "n_kv": n_kv, "head_dim": hd,
        "intermediate": inter, "num_layers": tc.get("num_hidden_layers"),
        "shapes": {
            "qkv_proj": ((n_heads + 2 * n_kv) * hd, h),
            "o_proj": (h, n_heads * hd),
            "gate_up_proj": (2 * inter, h),
            "down_proj": (h, inter),
        },
    }


# ======================================================================================
# PHASE: int4 token-identity (vLLM, deployed int4-Marlin path)
# ======================================================================================
def phase_tokenident_int4(out_path: str, n_prompts: int, max_len: int, batch_m: int,
                          gpu_mem_util: float, max_batched_tokens: int) -> None:
    import torch
    from vllm import LLM, SamplingParams

    model_dir = resolve_model_dir()
    dims = read_text_dims(model_dir)
    print(f"[int4tok] model={model_dir} layers={dims['num_layers']} hidden={dims['hidden']} "
          f"M_verify={batch_m}", flush=True)

    t0 = time.time()
    # enable_prefix_caching=False so the M identical replicas are NOT served from a
    # shared KV cache (they must all do a real forward); large max_num_batched_tokens
    # so all M replicas land in ONE prefill step (M-dim = M*seq_len, not M separate
    # seq_len forwards); enforce_eager=True so no CUDA-graph batch padding changes M.
    llm = LLM(
        model=model_dir,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=max(512, max_len + 8),
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=max(16, batch_m),
        max_num_batched_tokens=max_batched_tokens,
        enable_prefix_caching=False,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(f"[int4tok] vLLM load done in {time.time()-t0:.0f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    def argmax_seq(out) -> list[int]:
        # out.prompt_logprobs: list aligned to prompt positions; entry[i] is a dict
        # {token_id: Logprob} predicting prompt_ids[i] from prefix <i; entry[0] is None.
        pls = out.prompt_logprobs
        am: list[int] = []
        for i in range(len(pls)):
            entry = pls[i]
            if entry is None:
                continue
            # argmax = token with the highest (least negative) logprob in the dict;
            # prompt_logprobs=1 always includes the rank-1 (argmax) token.
            best_tok = max(entry.items(), key=lambda kv: getattr(kv[1], "logprob", kv[1]))[0]
            am.append(int(best_tok))
        return am

    rows = [json.loads(l) for l in open(PROMPTS_JSONL)][:n_prompts]
    per_prompt = []
    n_match = n_total = 0
    n_det_m1 = n_det_m8 = n_within = 0

    for ri, rec in enumerate(rows):
        ids = list(rec.get("context_token_ids", [])) + list(rec.get("target_token_ids", []))
        ids = ids[:max_len]
        if len(ids) < 2:
            continue
        prompt = {"prompt_token_ids": ids}

        # M=1 (pure-AR width): one replica alone -> prefill GEMM M-dim = seq_len
        am1 = argmax_seq(llm.generate([prompt], sp, use_tqdm=False)[0])
        # determinism control: M=1 again
        am1b = argmax_seq(llm.generate([prompt], sp, use_tqdm=False)[0])
        # M=8 (verify width): batch_m identical replicas co-batched in one prefill ->
        # GEMM M-dim = batch_m*seq_len. read copy 0 (and copy 1 for the within control).
        out_m8 = llm.generate([prompt] * batch_m, sp, use_tqdm=False)
        am8_0 = argmax_seq(out_m8[0])
        am8_1 = argmax_seq(out_m8[1]) if len(out_m8) > 1 else am8_0
        # determinism control: M=8 again
        am8b_0 = argmax_seq(llm.generate([prompt] * batch_m, sp, use_tqdm=False)[0])

        L = min(len(am1), len(am8_0), len(am1b), len(am8_1), len(am8b_0))
        a1, a1b, a80, a81, a8b = am1[:L], am1b[:L], am8_0[:L], am8_1[:L], am8b_0[:L]
        match = sum(1 for x, y in zip(a1, a80) if x == y)       # M1 vs M8 (the signal)
        det_m1 = sum(1 for x, y in zip(a1, a1b) if x == y)      # control: expect L
        det_m8 = sum(1 for x, y in zip(a80, a8b) if x == y)     # control: expect L
        within = sum(1 for x, y in zip(a80, a81) if x == y)     # control: expect L

        n_match += match
        n_total += L
        n_det_m1 += det_m1
        n_det_m8 += det_m8
        n_within += within
        sha1 = hashlib.sha256(bytes(str(a1), "utf8")).hexdigest()[:16]
        sha8 = hashlib.sha256(bytes(str(a80), "utf8")).hexdigest()[:16]
        per_prompt.append({
            "id": rec.get("id"), "positions": L, "argmax_match_M1_vs_M8": match,
            "argmax_sha_M1": sha1, "argmax_sha_M8": sha8, "sha_equal": sha1 == sha8,
            "det_match_M1_vs_M1": det_m1, "det_match_M8_vs_M8": det_m8,
            "within_match_copy0_vs_copy1": within,
        })
        if ri < 3 or ri == len(rows) - 1:
            print(f"[int4tok] prompt {ri} id={rec.get('id')} match={match}/{L} "
                  f"sha_eq={sha1==sha8} det_m1={det_m1}/{L} det_m8={det_m8}/{L} "
                  f"within={within}/{L}", flush=True)

    identity = (n_match / n_total) if n_total else float("nan")
    det_m1_frac = (n_det_m1 / n_total) if n_total else float("nan")
    det_m8_frac = (n_det_m8 / n_total) if n_total else float("nan")
    within_frac = (n_within / n_total) if n_total else float("nan")
    sha_equal_frac = (statistics.fmean([1.0 if p["sha_equal"] else 0.0 for p in per_prompt])
                      if per_prompt else float("nan"))

    # ---- decode-width diagnostic: isolated body GEMM row-0 bit-exactness at M in {1,8} ----
    # The LITERAL verify width is 8 rows (K+1), not 8*seq_len. #221 found the int4 body
    # GEMM bit-exact here; we re-confirm in-process so the verdict does not depend only on
    # an import. Best-effort: skipped (import #221) if the model nav path changes.
    decode_diag = {"status": "skipped_import_221", "int4_body_bitexact_M8_vs_M1": INT4_BODY_M_DEP_221 is False}
    try:
        decode_diag = isolated_decode_gemm_diag(llm, dims, torch)
    except Exception as exc:  # never block the e2e measurement on the diagnostic
        decode_diag = {"status": f"failed_import_221_fallback: {exc!r}",
                       "int4_body_bitexact_M8_vs_M1": INT4_BODY_M_DEP_221 is False}
        print(f"[int4tok] decode diagnostic unavailable -> {decode_diag['status']}", flush=True)

    nan_clean = all(math.isfinite(x) for x in (identity, det_m1_frac, det_m8_frac, within_frac))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    out = {
        "phase": "tokenident_int4",
        "model_dir": model_dir,
        "n_prompts": len(per_prompt),
        "max_len": max_len,
        "batch_m": batch_m,
        "total_positions": n_total,
        "matching_positions": n_match,
        "int4_token_identity_M1_vs_M8": identity,
        "int4_divergence_M1_vs_M8": (1.0 - identity) if math.isfinite(identity) else float("nan"),
        "determinism_M1_vs_M1": det_m1_frac,       # control: expect 1.0
        "determinism_M8_vs_M8": det_m8_frac,       # control: expect 1.0
        "within_batch_copy0_vs_copy1": within_frac,  # control: expect 1.0
        "per_sequence_strict_pass_fraction": sha_equal_frac,  # all-tokens-identical fraction
        "decode_width_diagnostic": decode_diag,
        "nan_clean": bool(nan_clean),
        "peak_gpu_gb": peak_gb,
        "per_prompt": per_prompt,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[int4tok] identity_M1_vs_M8={identity:.6f} (divergence={1.0-identity:.6f}) "
          f"strict_pass={sha_equal_frac:.4f} peak={peak_gb:.1f}GB", flush=True)
    print(f"[int4tok] controls: det_m1={det_m1_frac:.6f} det_m8={det_m8_frac:.6f} "
          f"within={within_frac:.6f}", flush=True)
    print(f"INT4TOK_DONE {out_path}", flush=True)


def isolated_decode_gemm_diag(llm, dims: dict, torch) -> dict:
    """Row-0 bit-exactness of the int4 body GEMMs at the literal verify width M=8 (8 rows).

    Reuses #221's model navigation + row-0 test. Localises whether a verify-width
    divergence is the int4 body (split-K) or the bf16 lm_head / accumulation.
    """
    import torch.nn as nn

    dev = torch.device("cuda:0")
    shapes = dims["shapes"]

    def get_model():
        paths = [
            lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
            lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        ]
        for p in paths:
            try:
                m = p()
                if m is not None:
                    return m
            except Exception:
                continue
        raise RuntimeError("could not locate model_runner.model")

    model = get_model()

    def find_layers(root):
        chains = [("model", "layers"), ("model", "language_model", "layers"),
                  ("language_model", "model", "layers"), ("language_model", "layers"),
                  ("model", "model", "layers"), ("layers",)]
        for chain in chains:
            obj = root
            ok = True
            for attr in chain:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                return obj
        for _, mod in root.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                el = mod[0]
                if hasattr(el, "self_attn") and hasattr(el.self_attn, "qkv_proj"):
                    return mod
        raise RuntimeError("could not locate decoder ModuleList")

    def module_out_in(mod):
        out = getattr(mod, "output_size_per_partition", None)
        inp = getattr(mod, "input_size_per_partition", None)
        if out is None or inp is None:
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 2:
                out, inp = int(w.shape[0]), int(w.shape[1])
        return (int(out), int(inp)) if out and inp else None

    layers = find_layers(model)
    targets = None
    for layer in layers:
        try:
            cand = {
                "qkv_proj": layer.self_attn.qkv_proj,
                "o_proj": layer.self_attn.o_proj,
                "gate_up_proj": layer.mlp.gate_up_proj,
                "down_proj": layer.mlp.down_proj,
            }
        except AttributeError:
            continue
        if all(hasattr(m, "quant_method") and module_out_in(m) == shapes[name]
               for name, m in cand.items()):
            targets = cand
            break
    if targets is None:
        raise RuntimeError("no layer matched canonical body shapes")

    torch.manual_seed(0)
    results = {}
    all_bitexact = True
    for name, (out, inp) in shapes.items():
        x = torch.randn(max(M_VERIFY, 16), inp, dtype=torch.bfloat16, device=dev)
        apply_fn = lambda t, _m=targets[name]: _m.quant_method.apply(_m, t, bias=None)
        y1 = apply_fn(x[:1].contiguous())[0].detach().float()
        y8 = apply_fn(x[:M_VERIFY].contiguous())[0].detach().float()
        torch.cuda.synchronize()
        bitexact = bool(torch.equal(y8, y1))
        results[name] = {
            "bitexact_M8_vs_M1": bitexact,
            "max_abs_diff_M8_vs_M1": float((y8 - y1).abs().max()),
        }
        all_bitexact = all_bitexact and bitexact
    return {
        "status": "ran",
        "int4_body_bitexact_M8_vs_M1": all_bitexact,
        "per_shape": results,
    }


# ======================================================================================
# Orchestrator: isolated subprocess phase, compose, self-test, wandb
# ======================================================================================
def run_phase_subprocess(args_list: list[str]) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [sys.executable, os.path.abspath(__file__)] + args_list
    print(f"[orch] launching: {' '.join(args_list)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"phase subprocess failed (rc={rc}): {args_list}")


def orchestrate(a: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_json = str(OUT_DIR / "int4_tokenident_result.json")

    run_phase_subprocess([
        "--phase", "tokenident_int4", "--out", tok_json,
        "--n-prompts", str(a.n_prompts), "--max-len", str(a.max_len),
        "--batch-m", str(a.batch_m), "--gpu-mem-util", str(a.gpu_mem_util),
        "--max-batched-tokens", str(a.max_batched_tokens),
    ])
    tok = json.load(open(tok_json))

    identity = tok["int4_token_identity_M1_vs_M8"]
    divergence = tok["int4_divergence_M1_vs_M8"]
    det_m1 = tok["determinism_M1_vs_M1"]
    det_m8 = tok["determinism_M8_vs_M8"]
    within = tok["within_batch_copy0_vs_copy1"]
    strict_pass = tok["per_sequence_strict_pass_fraction"]

    # ---- Deliverable: cross to #114 ----
    div_vs_114_delta = divergence - KANNA_114_DIV
    confirms_114 = bool(abs(div_vs_114_delta) <= CONFIRM_TOL)

    # ---- Deliverable: hand the dependent legs their number ----
    # Reading-A strict per-sequence pass fraction recomputed at the clean deployed width
    # (vs #219's strict-A 0.125): fraction of served sequences fully token-identical M1-vs-M8.
    reading_a_pass_fraction_deployed = strict_pass
    # model-free margin-gate supply cap = 1 - divergence = the per-token identity = the
    # max fraction any provable-skip scheme could ever skip.
    margin_gate_supply_cap_deployed = 1.0 - divergence

    # ---- Self-test (PRIMARY) ----
    det_m1_ok = (det_m1 == 1.0)
    det_m8_ok = (det_m8 == 1.0)
    within_ok = (within == 1.0)
    identity_in_range = (0.0 <= identity <= 1.0) and math.isfinite(identity)
    divergence_consistent = abs(divergence - (1.0 - identity)) < 1e-9
    cap_consistent = abs(margin_gate_supply_cap_deployed - (1.0 - divergence)) < 1e-9
    nan_clean = bool(tok["nan_clean"]) and math.isfinite(identity) and math.isfinite(divergence)

    self_test = {
        "det_M1_vs_M1_eq_1": det_m1_ok,                 # (a) + (e) known-identical control
        "det_M8_vs_M8_eq_1": det_m8_ok,                 # (a)
        "within_batch_copy0_vs_copy1_eq_1": within_ok,  # (a) extra: replicas identical
        "identity_in_range_finite": identity_in_range,  # (b)
        "divergence_eq_1_minus_identity": divergence_consistent,  # (c)
        "cap_eq_1_minus_divergence": cap_consistent,    # (d)
        "nan_clean": nan_clean,                         # (f)
    }
    int4_tokenident_self_test_passes = bool(
        det_m1_ok and det_m8_ok and within_ok and identity_in_range
        and divergence_consistent and cap_consistent and nan_clean
    )

    decode_diag = tok.get("decode_width_diagnostic", {})
    int4_body_bitexact_decodeM8 = bool(decode_diag.get("int4_body_bitexact_M8_vs_M1", INT4_BODY_M_DEP_221 is False))

    report = {
        "pr": 232,
        "leg": "int4 deployed-M8 token-identity probe (local)",
        "imported_anchors": {
            "kanna_114_div_native_spec_vs_M1": KANNA_114_DIV,
            "strict_A_219": STRICT_A_219,
            "fp16_identity_221": FP16_IDENTITY_221,
            "fp16_divergence_221": FP16_DIVERGENCE_221,
            "fp16_sha_equal_frac_221": FP16_SHA_EQUAL_FRAC_221,
            "int4_body_M_dependent_221": INT4_BODY_M_DEP_221,
            "official_baseline": OFFICIAL_BASELINE, "M_verify": M_VERIFY,
        },
        # TEST + PRIMARY
        "int4_token_identity_M1_vs_M8": identity,
        "int4_tokenident_self_test_passes": int4_tokenident_self_test_passes,
        # divergence + controls
        "int4_divergence_M1_vs_M8": divergence,
        "determinism_M1_vs_M1": det_m1,
        "determinism_M8_vs_M8": det_m8,
        "within_batch_copy0_vs_copy1": within,
        # cross to #114
        "int4_divergence_vs_114_delta": div_vs_114_delta,
        "deployed_m8_divergence_confirms_114": confirms_114,
        # dependent legs
        "reading_a_pass_fraction_deployed": reading_a_pass_fraction_deployed,
        "margin_gate_supply_cap_deployed": margin_gate_supply_cap_deployed,
        # bf16 contrast (#221) + decode-width localisation
        "int4_vs_fp16_identity_delta": identity - FP16_IDENTITY_221,
        "int4_vs_fp16_divergence_ratio": (divergence / FP16_DIVERGENCE_221) if FP16_DIVERGENCE_221 else float("nan"),
        "int4_body_bitexact_decode_M8": int4_body_bitexact_decodeM8,
        "decode_width_diagnostic": decode_diag,
        # bookkeeping
        "self_test": self_test,
        "n_prompts": tok["n_prompts"], "max_len": tok["max_len"], "batch_m": tok["batch_m"],
        "total_positions": tok["total_positions"], "model_dir": tok["model_dir"],
        "peak_gpu_gb": tok["peak_gpu_gb"],
    }
    report_path = OUT_DIR / "int4_tokenident_report.json"
    json.dump(report, open(report_path, "w"), indent=2)

    # ---- console summary ----
    print("\n========== INT4 DEPLOYED-M8 TOKEN-IDENTITY PROBE (PR #232) ==========", flush=True)
    print(f" int4 token identity  M1 vs M8     : {identity:.6f}", flush=True)
    print(f" int4 divergence      M1 vs M8     : {divergence:.6f}", flush=True)
    print(f"   controls: det_M1={det_m1:.6f}  det_M8={det_m8:.6f}  within={within:.6f}", flush=True)
    print(f" bf16 floor (#221) identity        : {FP16_IDENTITY_221:.6f} (div {FP16_DIVERGENCE_221:.6f})", flush=True)
    print(f" int4 vs bf16 divergence ratio     : {report['int4_vs_fp16_divergence_ratio']:.2f}x", flush=True)
    print(f" decode-width M8 int4 body bitexact: {int4_body_bitexact_decodeM8} "
          f"(diag {decode_diag.get('status')})", flush=True)
    print(f" --- cross to #114 (0.5608, M unrecorded) ---", flush=True)
    print(f" divergence vs #114 delta          : {div_vs_114_delta:+.6f}", flush=True)
    print(f" deployed-M8 confirms #114 (~0.56) : {confirms_114}", flush=True)
    print(f" --- dependent legs ---", flush=True)
    print(f" Reading-A strict pass @ deployed  : {reading_a_pass_fraction_deployed:.6f} "
          f"(vs #219 strict-A {STRICT_A_219})", flush=True)
    print(f" margin-gate supply cap @ deployed : {margin_gate_supply_cap_deployed:.6f}", flush=True)
    print(f" SELF-TEST PASSES (PRIMARY)        : {int4_tokenident_self_test_passes}  {self_test}", flush=True)
    print(f" report -> {report_path}", flush=True)
    print("=====================================================================\n", flush=True)

    if not a.no_wandb:
        log_wandb(report, tok, a)


def log_wandb(report: dict, tok: dict, a: argparse.Namespace) -> None:
    sys.path.insert(0, os.getcwd())
    try:
        from scripts.wandb_logging import init_wandb_run, finish_wandb
    except Exception as exc:
        print(f"[wandb] helper import failed: {exc!r}; skipping", flush=True)
        return
    run = init_wandb_run(
        job_type="local_profiling",
        agent="lawine",
        name=a.wandb_name,
        group=a.wandb_group,
        notes="PR#232 int4 deployed-M8 token-identity probe: clean M1-AR vs M8-verify int4 divergence",
        config={
            "pr": 232, "M_verify": report["batch_m"], "n_prompts": report["n_prompts"],
            "max_len": report["max_len"], "model_dir": report["model_dir"],
            "kanna_114_div": KANNA_114_DIV, "strict_A_219": STRICT_A_219,
            "fp16_identity_221": FP16_IDENTITY_221, "official_baseline": OFFICIAL_BASELINE,
            "confirm_tol": CONFIRM_TOL,
        },
    )
    if run is None:
        print("[wandb] disabled (no API key/mode); results saved to JSON only", flush=True)
        return

    summary = {
        "int4_tokenident_self_test_passes": report["int4_tokenident_self_test_passes"],
        "int4_token_identity_M1_vs_M8": report["int4_token_identity_M1_vs_M8"],
        "int4_divergence_M1_vs_M8": report["int4_divergence_M1_vs_M8"],
        "determinism_M1_vs_M1": report["determinism_M1_vs_M1"],
        "determinism_M8_vs_M8": report["determinism_M8_vs_M8"],
        "within_batch_copy0_vs_copy1": report["within_batch_copy0_vs_copy1"],
        "int4_divergence_vs_114_delta": report["int4_divergence_vs_114_delta"],
        "deployed_m8_divergence_confirms_114": report["deployed_m8_divergence_confirms_114"],
        "reading_a_pass_fraction_deployed": report["reading_a_pass_fraction_deployed"],
        "margin_gate_supply_cap_deployed": report["margin_gate_supply_cap_deployed"],
        "int4_vs_fp16_identity_delta": report["int4_vs_fp16_identity_delta"],
        "int4_vs_fp16_divergence_ratio": report["int4_vs_fp16_divergence_ratio"],
        "int4_body_bitexact_decode_M8": report["int4_body_bitexact_decode_M8"],
        "fp16_identity_221": FP16_IDENTITY_221,
        "kanna_114_div": KANNA_114_DIV,
        "peak_gpu_gb": report["peak_gpu_gb"],
    }
    for k, v in summary.items():
        run.summary[k] = v
    for k, v in report["self_test"].items():
        run.summary[f"selftest/{k}"] = v
    finish_wandb(run)
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["tokenident_int4"], default=None,
                    help="internal: run the GPU phase (subprocess). Omit for the orchestrator.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run (few prompts) to validate the path")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-m", type=int, default=M_VERIFY)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--wandb_group", dest="wandb_group", default="int4-tokenident-deployed-m8")
    ap.add_argument("--wandb_name", dest="wandb_name", default="lawine/int4-tokenident-deployed-m8")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()

    if a.smoke and a.phase is None:
        a.n_prompts = min(a.n_prompts, 6)

    if a.phase == "tokenident_int4":
        phase_tokenident_int4(a.out, a.n_prompts, a.max_len, a.batch_m,
                              a.gpu_mem_util, a.max_batched_tokens)
    else:
        orchestrate(a)


if __name__ == "__main__":
    main()
