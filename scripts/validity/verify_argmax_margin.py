#!/usr/bin/env python
"""Verify-GEMM argmax-margin greedy-safety gate (PR #87) — GPU capture stage.

WHY. Two in-flight TPS levers rest on an UNVERIFIED "lossless by construction /
bit-identical" claim:
  - ubel #84 (SplitK W4A16 verify-GEMM) changes the K-reduction ORDER;
  - land #71 (tree-verify M=16/32) changes the Marlin tile / M-width.
FP accumulation is non-associative, so both produce logits that are bit-CLOSE,
not bit-IDENTICAL, to the deployed Marlin-M8 verify. The official gate is
greedy-token-identity 128/128 — a SINGLE flipped argmax disqualifies. The
decisive question: at how many emitted positions is the top-2 logit margin thin
enough that a reduction-order / M-width perturbation flips the argmax?

WHAT (this stage, GPU, server venv). Reuse the DEPLOYED `fa2sw_precache_kenyan`
stack UNCHANGED (no served-file change). Load it in-process (pck04 head-prune +
PLE patches + Gemma4 softcap=30, exactly as serve.py applies them), run the
official 128-prompt greedy decode, and HOOK `Gemma4ForCausalLM.compute_logits`
to capture the hidden state h feeding the lm_head at every emitted position.
Then, from the real h:
  PHASE 1  ref = the REAL int4 W4A16 Marlin kernel at M=8 (the deployed verify
           width) -> full softcapped logits -> top-2 margin map (authoritative).
  PHASE 2a SplitK: recover the EXACT dequantized head W via apply(lm_head, I),
           then recompute logits with the K(=2560) reduction split into
           S in {2,4,8} contiguous FP32-accumulated chunks vs S=1 (single
           accumulation). This isolates ONLY the reduction order — ubel #84's
           perturbation — with the true weights. flip = argmax(S) != argmax(S=1).
  PHASE 2b M-widen: call the REAL Marlin kernel at M in {16,32} (land #71) on the
           same h grouped wider; compare the original rows' argmax to the M=8
           reference. This exercises the real wider-tile kernel-selection path.

The emulation is anchored to reality by a fidelity check: argmax(emu S=1) vs
argmax(real M=8 kernel) over all positions (expect ~100%); the recovered W is
exact because apply(lm_head, e_k) = W[:,k] with a single nonzero product (no
accumulation rounding).

Writes a compact .npz + summary.json of per-position margins, max|Δlogit|, and
argmax-flip masks. The sibling `analyze_argmax_margin.py` (CPU / repo venv) turns
that into the ULP histogram, the GREEN/RED gate, and the W&B record.

LOCAL ONLY. No HF Job, no submission, no served-file change. Single GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# stdlib-only import; safe in the server venv.
from scripts.local_validation import paths  # noqa: E402

SUBMISSION = REPO / "submissions" / "fa2sw_precache_kenyan"
BAKED_DIR = "/tmp/osoi5-12k-baked"  # deployed pck04-pruned 12k head (LM_HEAD_PRUNE_DST)
SOFTCAP = 30.0  # gemma-4 final_logit_softcapping (config.json)
HIDDEN = 2560
K_HEAD = 12288  # pruned lm_head rows
DEFAULT_SPLITS = (2, 4, 8)
DEFAULT_MWIDTHS = (16, 32)

# The deployed-stack env that affects the TARGET forward + lm_head numerics.
# (Serving-loop knobs — ONEGRAPH / LOOPGRAPH / FUSED_SPARSE_ARGMAX / SPLITKV_VERIFY
# / DIXIE / drafter — are NOT set: they optimize the drafter/serve loop, not the
# verify-GEMM math we audit. PLE + pck04 + softcap ARE replicated, because they
# shape h and the logits.)
DEPLOY_ENV = {
    "LOCAL_MODEL_DIR": BAKED_DIR,
    "PCK04_KEEPSET": f"{BAKED_DIR}/pck04_keepset.json",
    "PLE_ASSUME_VALID_TOKEN_IDS": "1",
    "PLE_FOLD_EMBED_SCALE": "1",
    "PLE_FOLD_TARGET_MODEL": BAKED_DIR,
    "PLE_SCRATCH_REUSE": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # keep engine in-process so we can hook
    "VLLM_USE_FLASHINFER_SAMPLER": "0",     # cuRAND-free (does not touch logits)
}


# --- capture state (module-level; the class hook writes here) ---------------
_CAP: dict[str, Any] = {"on": False, "h": [], "model": None, "n_calls": 0, "rows": 0}


def _install_compute_logits_hook() -> None:
    """Wrap Gemma4ForCausalLM.compute_logits at the CLASS level to capture h.

    Runs AFTER `import serve_patch_pck04` has registered its meta-path finder, so
    importing gemma4 here yields the already-pck04-patched class; we chain our
    capture on top. With multiprocessing off the engine model lives in THIS
    process, so a class-level wrap reaches the live model.
    """
    import torch
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

    orig = Gemma4ForCausalLM.compute_logits

    def capturing_compute_logits(self_model, hidden_states, *a, **kw):
        if _CAP["on"]:
            _CAP["n_calls"] += 1
            h = hidden_states.detach()
            if h.dim() == 1:
                h = h.unsqueeze(0)
            _CAP["rows"] += int(h.shape[0])
            _CAP["h"].append(h.to("cpu", copy=True))
            if _CAP["model"] is None:
                _CAP["model"] = self_model
        return orig(self_model, hidden_states, *a, **kw)

    Gemma4ForCausalLM.compute_logits = capturing_compute_logits
    print("[capture] installed compute_logits hook on Gemma4ForCausalLM", flush=True)


def _load_official_prompt_fns():
    spec = importlib.util.spec_from_file_location("decode_outputs", str(paths.DECODE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # transformers is imported only inside its main()
    return mod


def _apply_ple_patches() -> None:
    """Apply the deployed PLE/loader source patches to the installed vLLM.

    Idempotent + marker-guarded (exactly what serve.py does); replicates the
    deployed target forward so captured h matches the served distribution.
    """
    if str(SUBMISSION) not in sys.path:
        sys.path.insert(0, str(SUBMISSION))
    import serve as deployed_serve  # noqa: WPS433 (submission serve.py)

    deployed_serve.patch_ple_sources()


def _real_logits(model, h_batch, *, group_m: int):
    """Softcapped logits from the REAL Marlin kernel, forced to width `group_m`.

    Calls lm_head.quant_method.apply in fixed groups of `group_m` rows so the
    M=group_m kernel template (tile/occupancy) is the one actually exercised —
    the deployed verify runs M=8. Returns [N, K_HEAD] float32 (post-softcap),
    plus the kernel's native output dtype (detected once).
    """
    import torch

    lm_head = model.lm_head
    qm = lm_head.quant_method
    n = h_batch.shape[0]
    outs = []
    native_dtype = None
    for g in range(0, n, group_m):
        chunk = h_batch[g : g + group_m]
        pad = 0
        if chunk.shape[0] < group_m:  # pad last group so the M=group_m tile is used
            pad = group_m - chunk.shape[0]
            chunk = torch.cat([chunk, chunk[-1:].expand(pad, -1)], dim=0)
        lg = qm.apply(lm_head, chunk, bias=None)  # [group_m, K_HEAD], native dtype
        if native_dtype is None:
            native_dtype = lg.dtype
        if pad:
            lg = lg[: group_m - pad]
        outs.append(_softcap(lg).float())
    return torch.cat(outs, dim=0), native_dtype


def _softcap(lg):
    import torch

    # Exactly LogitsProcessor.forward: divide, tanh, multiply — in the logits'
    # own dtype, so the FP32->native cast boundary (where reduction-order noise
    # becomes visible) is faithful.
    lg = lg / SOFTCAP
    lg = torch.tanh(lg)
    lg = lg * SOFTCAP
    return lg


def _recover_weight(model, native_dtype):
    """Recover the EXACT dequantized head W [K_HEAD, HIDDEN] via apply(lm_head, I).

    apply(lm_head, e_k) = W[:, k] with a single nonzero product per output, so no
    accumulation rounding: the recovered weights are exact to the kernel's own
    dequant. Sidesteps Marlin's packed/permuted on-disk layout entirely.
    """
    import torch

    lm_head = model.lm_head
    qm = lm_head.quant_method
    dev = next(model.parameters()).device if hasattr(model, "parameters") else "cuda"
    eye = torch.eye(HIDDEN, dtype=native_dtype, device="cuda")
    cols = []
    for g in range(0, HIDDEN, 512):  # batch the basis to bound memory
        wt = qm.apply(lm_head, eye[g : g + 512], bias=None)  # [b, K_HEAD] = W[:, g:g+b].T
        cols.append(wt.float())
    w_t = torch.cat(cols, dim=0)  # [HIDDEN, K_HEAD] float32
    return w_t.t().contiguous()  # [K_HEAD, HIDDEN]


def _emu_logits(h_batch, w, *, splits: int, native_dtype):
    """Emulated softcapped logits with the K dim split into `splits` FP32 chunks.

    splits=1 -> single full-K FP32 accumulation (the reference order).
    splits=S -> S contiguous partial sums, each FP32-accumulated, then combined
    in ascending-chunk FP32 order (S-1 extra rounding boundaries) — a faithful
    emulation of ubel #84's reduction-order change. The FP32 result is cast to
    the kernel's native dtype BEFORE softcap, matching the real FP32->native cast
    where the perturbation becomes visible. Returns [N, K_HEAD] float32.
    """
    import torch

    h = h_batch.float()  # [N, HIDDEN]; fp16/bf16 -> fp32 is exact
    n = h.shape[0]
    acc = torch.zeros((n, w.shape[0]), dtype=torch.float32, device="cuda")
    bounds = _chunk_bounds(HIDDEN, splits)
    for lo, hi in bounds:
        # partial = h[:, lo:hi] @ W[:, lo:hi].T, FP32 accumulate within chunk
        acc += h[:, lo:hi] @ w[:, lo:hi].t()
    acc = acc.to(native_dtype)  # FP32 -> native cast (the visibility boundary)
    return _softcap(acc).float()


def _chunk_bounds(k: int, splits: int):
    step = (k + splits - 1) // splits
    return [(lo, min(lo + step, k)) for lo in range(0, k, step)]


def _reduce_logits(lg):
    """Per-row top-2 (values + index) from a [B, K_HEAD] float32 logit batch."""
    import torch

    top2 = torch.topk(lg, k=2, dim=-1)
    return (
        top2.values[:, 0].contiguous(),  # top1
        top2.values[:, 1].contiguous(),  # top2
        top2.indices[:, 0].to(torch.int32).contiguous(),  # argmax
    )


def capture(args) -> Path:
    import numpy as np
    import torch

    out_dir = Path(args.out_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] out_dir={out_dir}", flush=True)

    for k, v in DEPLOY_ENV.items():
        os.environ.setdefault(k, v)
    notes = paths.prepare_local_gpu_env()
    for n in notes:
        print(f"[gpu] {n}", flush=True)

    # Order matters: PLE source-patch + pck04 finder must be in place BEFORE vLLM
    # imports gemma4 (the LLM constructor triggers that import).
    if str(SUBMISSION) not in sys.path:
        sys.path.insert(0, str(SUBMISSION))
    _apply_ple_patches()
    import serve_patch_pck04  # noqa: F401  (registers the compute_logits meta-path finder)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    _install_compute_logits_hook()

    print("[capture] constructing in-process LLM (enforce_eager, no drafter, M=1 AR)", flush=True)
    t0 = time.time()
    llm = LLM(
        model=BAKED_DIR,
        dtype="bfloat16",
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
        max_num_seqs=1,
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")),
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    print(f"[capture] LLM ready in {time.time() - t0:.0f}s", flush=True)

    deco = _load_official_prompt_fns()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BAKED_DIR)
    records = deco.read_sharegpt_prompts(
        Path(args.dataset), num_prompts=args.num_prompts, seed=paths.SEED
    )
    prompts = [TokensPrompt(prompt_token_ids=deco.encode_prompt(tok, r["prompt_text"])) for r in records]
    print(f"[capture] {len(prompts)} prompts encoded; decoding {args.output_len} greedy tokens each", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    _CAP["on"] = True
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    _CAP["on"] = False
    gen_s = time.time() - t0
    completion_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    print(
        f"[capture] decode done in {gen_s:.0f}s: {completion_tokens} completion tokens, "
        f"{_CAP['rows']} captured logit rows ({_CAP['n_calls']} calls)",
        flush=True,
    )

    H = torch.cat(_CAP["h"], dim=0)  # [N, HIDDEN] on CPU (bf16)
    _CAP["h"].clear()
    npos = H.shape[0]
    model = _CAP["model"]
    assert model is not None, "compute_logits hook never fired — no model captured"

    # --- recover exact W + detect native kernel dtype (one M=8 probe) ---------
    with torch.inference_mode():
        probe = H[:8].to("cuda")
        _, native_dtype = _real_logits(model, probe, group_m=8)
        print(f"[capture] native kernel output dtype = {native_dtype}", flush=True)
        W = _recover_weight(model, native_dtype)  # [K_HEAD, HIDDEN] fp32
        print(f"[capture] recovered exact head W {tuple(W.shape)} via apply(lm_head, I)", flush=True)

        # --- per-position reductions, batched to bound memory -----------------
        B = int(args.batch)
        splits = [int(s) for s in args.splits]
        mwidths = [int(m) for m in args.mwidths]

        ref_top1 = np.empty(npos, np.float32)
        ref_top2 = np.empty(npos, np.float32)
        ref_argmax = np.empty(npos, np.int32)
        emu1_argmax = np.empty(npos, np.int32)
        sk_flip = {s: np.empty(npos, np.bool_) for s in splits}          # vs emu S=1
        sk_flip_ref = {s: np.empty(npos, np.bool_) for s in splits}      # vs real M=8
        sk_dmax = {s: np.empty(npos, np.float32) for s in splits}
        mw_flip = {m: np.empty(npos, np.bool_) for m in mwidths}         # vs real M=8
        mw_dmax = {m: np.empty(npos, np.float32) for m in mwidths}

        for b0 in range(0, npos, B):
            b1 = min(b0 + B, npos)
            hb = H[b0:b1].to("cuda")

            ref_lg, _ = _real_logits(model, hb, group_m=8)               # [b, K] fp32
            r1, r2, rarg = _reduce_logits(ref_lg)
            ref_top1[b0:b1] = r1.cpu().numpy()
            ref_top2[b0:b1] = r2.cpu().numpy()
            ref_argmax[b0:b1] = rarg.cpu().numpy()

            emu1 = _emu_logits(hb, W, splits=1, native_dtype=native_dtype)
            _, _, e1arg = _reduce_logits(emu1)
            emu1_argmax[b0:b1] = e1arg.cpu().numpy()

            for s in splits:
                emus = _emu_logits(hb, W, splits=s, native_dtype=native_dtype)
                _, _, esarg = _reduce_logits(emus)
                sk_flip[s][b0:b1] = (esarg != e1arg).cpu().numpy()
                sk_flip_ref[s][b0:b1] = (esarg != rarg).cpu().numpy()
                sk_dmax[s][b0:b1] = (emus - emu1).abs().amax(dim=-1).cpu().numpy()
                del emus

            for m in mwidths:
                mw_lg, _ = _real_logits(model, hb, group_m=m)
                _, _, marg = _reduce_logits(mw_lg)
                mw_flip[m][b0:b1] = (marg != rarg).cpu().numpy()
                mw_dmax[m][b0:b1] = (mw_lg - ref_lg).abs().amax(dim=-1).cpu().numpy()
                del mw_lg

            del hb, ref_lg, emu1
            if (b0 // B) % 4 == 0:
                print(f"[capture]   reduced {b1}/{npos} positions", flush=True)

    # --- persist compact artifacts -------------------------------------------
    npz_path = out_dir / "margin_perturb.npz"
    save: dict[str, Any] = {
        "ref_top1": ref_top1,
        "ref_top2": ref_top2,
        "ref_argmax": ref_argmax,
        "emu1_argmax": emu1_argmax,
    }
    for s in splits:
        save[f"sk{s}_flip_emuS1"] = sk_flip[s]
        save[f"sk{s}_flip_refM8"] = sk_flip_ref[s]
        save[f"sk{s}_dmax"] = sk_dmax[s]
    for m in mwidths:
        save[f"mw{m}_flip_refM8"] = mw_flip[m]
        save[f"mw{m}_dmax"] = mw_dmax[m]
    np.savez_compressed(npz_path, **save)

    margin = (ref_top1 - ref_top2).astype(np.float32)
    fidelity_disagree = int((ref_argmax != emu1_argmax).sum())
    summary = {
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "num_positions": int(npos),
        "completion_tokens": int(completion_tokens),
        "decode_s": round(gen_s, 1),
        "native_dtype": str(native_dtype),
        "softcap": SOFTCAP,
        "splits": splits,
        "mwidths": mwidths,
        "min_margin": float(margin.min()),
        "median_margin": float(np.median(margin)),
        "max_abs_logit": float(np.abs(np.concatenate([ref_top1, ref_top2])).max()),
        "fidelity_emuS1_vs_realM8_disagreements": fidelity_disagree,
        "splitk_flip_count_vs_emuS1": {str(s): int(sk_flip[s].sum()) for s in splits},
        "splitk_flip_count_vs_refM8": {str(s): int(sk_flip_ref[s].sum()) for s in splits},
        "splitk_max_abs_dlogit": {str(s): float(sk_dmax[s].max()) for s in splits},
        "mwiden_flip_count_vs_refM8": {str(m): int(mw_flip[m].sum()) for m in mwidths},
        "mwiden_max_abs_dlogit": {str(m): float(mw_dmax[m].max()) for m in mwidths},
        "npz": str(npz_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[capture] SUMMARY\n" + json.dumps(summary, indent=2), flush=True)
    print(f"[capture] wrote {npz_path}\n[capture] wrote {out_dir / 'summary.json'}", flush=True)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-prompts", type=int, default=paths.NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=paths.OUTPUT_LEN)
    ap.add_argument("--dataset", default=str(paths.EVAL_PROMPTS))
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--mwidths", nargs="+", default=list(DEFAULT_MWIDTHS))
    ap.add_argument("--batch", type=int, default=4096, help="positions per reduction batch")
    ap.add_argument("--out-dir", default=str(REPO / "research/validity/verify_argmax_margin"))
    ap.add_argument("--smoke", action="store_true", help="4 prompts x 32 tok plumbing check")
    args = ap.parse_args()
    if args.smoke:
        args.num_prompts = 4
        args.output_len = 32
        args.batch = 512
    capture(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
