#!/usr/bin/env python
"""Network-wide greedy-compounding gate (PR #96) — GPU capture stage.

WHY. PR #87 verified the verify-GEMM greedy-safety gate at the lm_head projection
IN ISOLATION (0/65,536 argmax flips under SplitK S∈{2,4,8} and M-widen M≤32) and
flagged ONE honest residual its single-GEMM scope could not close: *upstream
network-wide compounding*. The composed frontier — land #71's tree M-widen × ubel
#84's SplitK W4A16 kernel — changes the reduction order / tiling of EVERY GEMM in
the network, not just lm_head. Each per-layer perturbation is ≤1 bf16-ULP (the #87
regime bound), but the residual stream ACCUMULATES ~30 layers of them before the
final RMSNorm + lm_head. Does that compounding drift the final hidden state h
enough to flip the greedy argmax at any of the 65,536 emitted positions?

WHAT (this stage, GPU, server venv). Load the DEPLOYED `fa2sw_precache_kenyan`
stack UNCHANGED in-process (pck04 12k head + PLE patches + softcap=30, exactly as
#87). Self-decode the deployed model's own 128×512 greedy completion (deterministic
— #73/Step-1 confirm 0/65,536 run-to-run divergence, so this is the SAME trajectory
#87 captured and its margin map aligns position-for-position) and TEACHER-FORCE it,
so per-position argmax sensitivity is isolated from the autoregressive cascade (a
single early flip would otherwise change all later context). Then run, over the
fixed teacher-forced sequences:

  CONTROL    perturbation OFF. Capture all-position post-final-norm h (hook on
             Gemma4Model.forward). argmax_ctrl = real M=8 Marlin lm_head.
             Re-run N times -> in-process determinism baseline (0 divergent).
  PERTURBED  inject a per-op reduction-order perturbation at EVERY transformer-block
             GEMM (qkv/o/gate_up/down) via forward hooks, propagate through the REAL
             residual stream + RMSNorms, AND apply the same class to the lm_head.
    (A) realistic / in-regime — the GENUINE ≤1-ULP reduction-order envelope #87
             measured: recover each GEMM's exact dequantized W (via apply(m, I)),
             recompute its output with the K-reduction split into S∈{2,4,8} FP32
             chunks (forced FP32-reduce, atomic-add OFF; one final bf16 cast), and
             inject delta = emu_S − emu_1 onto the native output. This is the SAME
             SplitK class #87 bounded per-op (ubel #84), now applied at every GEMM
             and propagated. The TRUE compounding bound for the composed frontier.
    (B) adversarial / worst-case — delta = sign(y)·ulp_bf16(y) per element,
             sign-aligned to maximise residual-stream L2 growth (compounds
             ~linearly), plus a TARGETED lm_head shift (top1 −1 ULP, top2 +1 ULP).
             A strict upper bound unreachable by any real reduction order: if even
             this is ~0 flips, the residual is decisively closed.

The decisive output is the COMPOUNDED argmax-flip count vs the control over all
65,536 positions, plus the propagated drift (‖Δh‖, max|Δlogit|) against #87's
top-2 margin map. The sibling `analyze_compounding.py` (CPU / repo venv) turns the
.npz into the GREEN/AMBER/RED gate + the W&B record.

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

from scripts.local_validation import paths  # noqa: E402  (stdlib-only, server-venv safe)

SUBMISSION = REPO / "submissions" / "fa2sw_precache_kenyan"
BAKED_DIR = "/tmp/osoi5-12k-baked"  # deployed pck04-pruned 12k head (LM_HEAD_PRUNE_DST)
SOFTCAP = 30.0  # gemma-4 final_logit_softcapping (config.json)
HIDDEN = 2560
K_HEAD = 12288  # pruned lm_head rows
MANT_BF16 = 7   # bf16 stored-mantissa bits (ULP exponent offset)

# The four named transformer-block GEMMs the composed frontier re-tiles, per the
# PR (q/k/v/o proj, gate/up/down). qkv_proj fuses q/k/v; gate_up_proj fuses
# gate/up. lm_head is perturbed separately on the logits (it is computed by us
# from captured h, not inside the vLLM forward). PLE/router/MoE GEMMs are OUT of
# the PR's named set; --include-ple extends to the PLE projections for a
# conservative robustness check.
PERTURB_SUFFIXES = (".qkv_proj", ".o_proj", ".gate_up_proj", ".down_proj")
PLE_SUFFIXES = (".per_layer_input_gate", ".per_layer_projection",
                ".per_layer_model_projection")

# Same deployed-stack env that shapes h + logits as #87 (PLE + pck04 + softcap).
DEPLOY_ENV = {
    "LOCAL_MODEL_DIR": BAKED_DIR,
    "PCK04_KEEPSET": f"{BAKED_DIR}/pck04_keepset.json",
    "PLE_ASSUME_VALID_TOKEN_IDS": "1",
    "PLE_FOLD_EMBED_SCALE": "1",
    "PLE_FOLD_TARGET_MODEL": BAKED_DIR,
    "PLE_SCRATCH_REUSE": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",  # keep engine in-process so hooks reach the live model
    "VLLM_USE_FLASHINFER_SAMPLER": "0",     # cuRAND-free (does not touch logits)
}

# --- capture + perturbation module state (hooks write here) -----------------
_CAP: dict[str, Any] = {"on": False, "h": [], "model": None, "rows": 0, "calls": 0}
# mode: "off" | "emu" | "adversarial"; splits: K-split for genuine emu; gen: CUDA
# Generator (adversarial RNG only); max_abs_delta accumulates the per-op |Δactivation|.
_PERT: dict[str, Any] = {"mode": "off", "splits": 2, "gen": None, "scale": 1.0,
                         "max_abs_delta": 0.0, "n_perturbed": 0}
# Recovered exact dequantized GEMM weights for the genuine reduction-order emulation:
# id(module) -> W [out, in] bf16 on cuda. _PROBE: id(module) -> (in, out, dtype) shapes
# learned from a live forward (robust to vLLM's parallel-linear attribute names).
_GEMM_W: dict[int, Any] = {}
_PROBE: dict[int, Any] = {}


def _ulp_bf16(x):
    """Per-element bf16 ULP at magnitude |x| (the spacing a bf16 store rounds to)."""
    import torch

    e = torch.floor(torch.log2(x.abs().clamp_min(2.0 ** -126)))
    return torch.exp2(e - MANT_BF16)


def _chunk_bounds(k: int, splits: int):
    """K split into `splits` contiguous chunks (the SplitK reduction grouping)."""
    step = (k + splits - 1) // splits
    return [(lo, min(lo + step, k)) for lo in range(0, k, step)]


def _emu_matmul(x, W, *, splits: int):
    """FP32-MAC, FP32-reduce linear x@Wᵀ with the K(=in) dim split into `splits`
    contiguous FP32-accumulated chunks, ONE final bf16 cast — exactly the #87 SplitK
    reduction-order class, applied to a transformer-block GEMM. x:[N,in], W:[out,in]
    bf16 -> [N,out] bf16. splits=1 is the single-accumulation reference order."""
    import torch

    xf = x.float()
    n, kin = xf.shape
    acc = torch.zeros((n, W.shape[0]), dtype=torch.float32, device=xf.device)
    for lo, hi in _chunk_bounds(kin, splits):
        acc += xf[:, lo:hi] @ W[:, lo:hi].float().t()
    return acc.to(torch.bfloat16)


def _adv_tensor(t):
    """Sign-aligned full +1 bf16-ULP per element (adversarial worst case): grows
    |y|, hence the residual-stream L2 norm — the linear-compounding upper bound. The
    perturbed value is re-cast to bf16 so the injected drift is itself representable."""
    import torch

    tf = t.float()
    delta = torch.sign(tf) * _ulp_bf16(tf) * _PERT["scale"]
    md = float(delta.abs().amax().item()) if delta.numel() else 0.0
    if md > _PERT["max_abs_delta"]:
        _PERT["max_abs_delta"] = md
    _PERT["n_perturbed"] += 1
    return (tf + delta).to(t.dtype)


def _perturb_hook(module, inputs, output):
    """Inject the per-op reduction-order perturbation onto a GEMM's native output.

    emu (regime A): delta = emu_S − emu_1 over the recovered exact W — the GENUINE
        SplitK reduction-order change (≤1 ULP per element), added to the real native
        kernel output so the deployed trajectory + the pure reduction-order drift
        propagate together.
    adversarial (regime B): sign-aligned +1 ULP per element (synthetic worst case).
    """
    import torch

    mode = _PERT["mode"]
    if mode == "off":
        return None
    native = output[0] if isinstance(output, tuple) else output
    if mode == "emu":
        W = _GEMM_W.get(id(module))
        if W is None:
            return None  # un-recovered module (e.g. PLE when include_ple but skipped)
        x = inputs[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        emu1 = _emu_matmul(x, W, splits=1)
        emuS = _emu_matmul(x, W, splits=_PERT["splits"])
        delta = emuS.float() - emu1.float()
        md = float(delta.abs().amax().item()) if delta.numel() else 0.0
        if md > _PERT["max_abs_delta"]:
            _PERT["max_abs_delta"] = md
        _PERT["n_perturbed"] += 1
        pert = (native.float() + delta.reshape(native.shape)).to(native.dtype)
    elif mode == "adversarial":
        pert = _adv_tensor(native)
    else:
        return None
    if isinstance(output, tuple):
        return (pert,) + tuple(output[1:])
    return pert


def _probe_hook(module, inputs, output):
    """Record (in, out, dtype) from a live forward — robust to parallel-linear attrs."""
    mid = id(module)
    if mid not in _PROBE:
        x = inputs[0]
        y = output[0] if isinstance(output, tuple) else output
        _PROBE[mid] = (int(x.shape[-1]), int(y.shape[-1]), y.dtype)


def _recover_module_weight(module, in_dim, dtype, *, batch=512):
    """Recover the EXACT dequantized W [out, in] via apply(module, I).

    apply(m, e_k) = W[:, k] (one nonzero product per output, no accumulation rounding),
    so stacking apply over the identity gives Wᵀ exactly — sidestepping the packed
    Marlin layout. Stored bf16 (the dequant target dtype): lossless for the weights,
    upcast per-chunk to FP32 inside the emulation."""
    import torch

    qm = module.quant_method
    cols = []
    for g in range(0, in_dim, batch):
        b = min(batch, in_dim - g)
        eye = torch.zeros((b, in_dim), dtype=dtype, device="cuda")
        eye[torch.arange(b, device="cuda"), torch.arange(g, g + b, device="cuda")] = 1
        wt = qm.apply(module, eye, bias=None)  # [b, out] = W[:, g:g+b]ᵀ
        cols.append(wt.to(torch.bfloat16))
    w_t = torch.cat(cols, dim=0)  # [in, out]
    return w_t.t().contiguous()   # [out, in]


def _recover_all_weights(model, hooked_modules):
    """Fill _GEMM_W for every hooked GEMM from probed dims; report total VRAM + a
    fidelity check (emu_1 vs the native kernel) on the largest module."""
    import torch

    _GEMM_W.clear()
    total_bytes = 0
    t0 = time.time()
    check = None
    for name, mod in hooked_modules:
        mid = id(mod)
        if mid not in _PROBE:
            print(f"[emu] WARN no probe dims for {name}; skipping", flush=True)
            continue
        in_dim, out_dim, dtype = _PROBE[mid]
        W = _recover_module_weight(mod, in_dim, dtype)
        _GEMM_W[mid] = W
        total_bytes += W.numel() * W.element_size()
        if check is None and "gate_up_proj" in name:  # fidelity probe on a wide GEMM
            with torch.inference_mode():
                xp = torch.randn((8, in_dim), dtype=dtype, device="cuda")
                nat = mod.quant_method.apply(mod, xp, bias=None)
                nat = nat[0] if isinstance(nat, tuple) else nat
                e1 = _emu_matmul(xp, W, splits=1)
                check = (name, float((e1.float() - nat.float()).abs().max()))
    dt = time.time() - t0
    print(f"[emu] recovered {len(_GEMM_W)} GEMM weights "
          f"({total_bytes / 2**30:.2f} GiB bf16) in {dt:.0f}s", flush=True)
    if check is not None:
        print(f"[emu] fidelity emu_1 vs native on {check[0]}: max|Δ|={check[1]:.4g} "
              "(reduction-order envelope ~1 ULP; emu anchors to the real kernel)",
              flush=True)
    return total_bytes


def _install_perturb_hooks(model, *, include_ple: bool):
    suffixes = PERTURB_SUFFIXES + (PLE_SUFFIXES if include_ple else ())
    handles, names, mods = [], [], []
    for name, mod in model.named_modules():
        if name.endswith(suffixes):
            handles.append(mod.register_forward_hook(_perturb_hook))
            names.append(name)
            mods.append((name, mod))
    print(f"[perturb] installed {len(handles)} GEMM perturbation hooks "
          f"(include_ple={include_ple})", flush=True)
    return handles, names, mods


def _install_forward_capture_hook() -> None:
    """Capture all-position post-final-norm h + grab the lm_head-bearing module.

    The deployed multimodal arch (Gemma4ForConditionalGeneration) drives the text
    stack by calling the INNER Gemma4Model.forward directly — Gemma4ForCausalLM.forward
    is bypassed (verified empirically: only its compute_logits fires). So:
      * Gemma4Model.forward returns the POST-final-RMSNorm hidden states for EVERY
        position (the exact h feeding lm_head, since Gemma4ForCausalLM.forward just
        returns self.model(...)); wrap it to capture them.
      * Gemma4ForCausalLM.compute_logits fires once per forward with that module as
        `self`; wrap it ONLY to grab the lm_head-bearing instance for our own M=8
        reductions. Its hidden_states arg is the last position only — we ignore it.
    enforce_eager (no CUDA graph) + multiprocessing OFF keep the live modules in THIS
    process, so class-level wraps reach them. h is moved to CPU to bound VRAM.
    """
    import torch
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM, Gemma4Model

    orig_fwd = Gemma4Model.forward

    def capturing_forward(self_model, *a, **kw):
        out = orig_fwd(self_model, *a, **kw)
        if _CAP["on"]:
            h = out[0] if isinstance(out, tuple) else out
            if isinstance(h, torch.Tensor):
                h = h.detach()
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                _CAP["h"].append(h.to("cpu", copy=True))
                _CAP["rows"] += int(h.shape[0])
                _CAP["calls"] += 1
        return out

    Gemma4Model.forward = capturing_forward

    orig_cl = Gemma4ForCausalLM.compute_logits

    def capturing_compute_logits(self_model, *a, **kw):
        if _CAP["model"] is None:
            _CAP["model"] = self_model
        return orig_cl(self_model, *a, **kw)

    Gemma4ForCausalLM.compute_logits = capturing_compute_logits
    print("[capture] installed hooks: Gemma4Model.forward (all-position h) + "
          "Gemma4ForCausalLM.compute_logits (lm_head module)", flush=True)


def _softcap(lg):
    """Exactly LogitsProcessor.forward: divide, tanh, multiply, in lg's dtype."""
    import torch

    lg = lg / SOFTCAP
    lg = torch.tanh(lg)
    lg = lg * SOFTCAP
    return lg


def _real_logits(model, h_batch, *, group_m: int = 8):
    """Softcapped fp32 logits from the REAL int4 Marlin lm_head at width group_m.

    Calls lm_head.quant_method.apply in fixed groups of group_m rows so the M=8
    kernel template the deployed verify runs is the one exercised. Returns
    [N, K_HEAD] float32 (post-softcap), plus the kernel's native output dtype.
    """
    import torch

    lm_head = model.lm_head
    qm = lm_head.quant_method
    n = h_batch.shape[0]
    outs, native = [], None
    for g in range(0, n, group_m):
        chunk = h_batch[g:g + group_m]
        pad = 0
        if chunk.shape[0] < group_m:
            pad = group_m - chunk.shape[0]
            chunk = torch.cat([chunk, chunk[-1:].expand(pad, -1)], dim=0)
        lg = qm.apply(lm_head, chunk, bias=None)
        if native is None:
            native = lg.dtype
        if pad:
            lg = lg[:group_m - pad]
        outs.append(_softcap(lg).float())
    return torch.cat(outs, dim=0), native


def _emu_logits(h_batch, W, *, splits: int, native_dtype):
    """GENUINE lm_head SplitK: softcapped fp32 logits with the K(=HIDDEN) reduction
    split into `splits` FP32 chunks, one final cast to native dtype before softcap —
    exactly #87's `_emu_logits`. splits=1 is the reference order; emu_S − emu_1 is the
    pure reduction-order delta. W:[K_HEAD, HIDDEN] bf16."""
    import torch

    h = h_batch.float()
    acc = torch.zeros((h.shape[0], W.shape[0]), dtype=torch.float32, device=h.device)
    for lo, hi in _chunk_bounds(h.shape[1], splits):
        acc += h[:, lo:hi] @ W[:, lo:hi].float().t()
    acc = acc.to(native_dtype)
    return _softcap(acc).float()


def _lm_head_adv_perturb(logits, *, native_dtype):
    """Adversarial worst-case lm_head flip attempt: top1 −1 ULP, top2 +1 ULP at the
    native rung. Strictly upper-bounds any real reduction-order swap (which moves each
    logit by ≤1 native ULP) by spending the full budget toward a flip."""
    import torch

    nbits = MANT_BF16 if "bfloat16" in str(native_dtype) else 10
    ulp = torch.exp2(torch.floor(torch.log2(logits.abs().clamp_min(2.0 ** -126))) - nbits)
    top2 = torch.topk(logits, k=2, dim=-1)
    i1, i2 = top2.indices[:, 0], top2.indices[:, 1]
    rows = torch.arange(logits.shape[0], device=logits.device)
    out = logits.clone()
    out[rows, i1] = out[rows, i1] - ulp[rows, i1]
    out[rows, i2] = out[rows, i2] + ulp[rows, i2]
    return out


def _argmax_top2(lg):
    import torch

    top2 = torch.topk(lg, k=2, dim=-1)
    return (top2.values[:, 0].contiguous(), top2.values[:, 1].contiguous(),
            top2.indices[:, 0].to(torch.int32).contiguous())


def _load_decode_mod():
    spec = importlib.util.spec_from_file_location("decode_outputs", str(paths.DECODE_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply_ple_patches() -> None:
    if str(SUBMISSION) not in sys.path:
        sys.path.insert(0, str(SUBMISSION))
    import serve as deployed_serve  # noqa: WPS433

    deployed_serve.patch_ple_sources()


def _load_keep_ids():
    """keep_ids[j] = full-vocab token id of pruned lm_head row/column j.

    The pck04 scatter does out[:, keep_ids[j]] = pruned_logits[:, j], so an argmax
    over the pruned 12288-wide logits is a ROW INDEX; the emitted full-vocab token
    is keep_ids[row]. #87's ref_argmax is that pruned row index (every value < K),
    so reconstructing the fed greedy context requires this map.
    """
    import numpy as np

    keepset_path = os.environ.get("PCK04_KEEPSET", f"{BAKED_DIR}/pck04_keepset.json")
    data = json.loads(Path(keepset_path).read_text())
    return np.asarray(data["keep_ids"], dtype=np.int64)


def _run_pass(llm, seqs, prompt_lens, output_len, *, label: str):
    """Teacher-forced prefill, processing ONE sequence per generate call so the
    capture order is unambiguous (the scheduler does not guarantee output order
    matches input order even at max_num_seqs=1). For each sequence, concatenate its
    captured forward block(s) — robust to any internal chunking since chunks of one
    sequence are produced in order — assert the row count equals prompt_len+output_len,
    then slice the `output_len` completion-PREDICTING rows: block[-(output_len+1):-1]
    (row j predicts token j+1; the last fed token's prediction is unused). Returns
    [num_seqs*output_len, HIDDEN] on CPU in prompt order, plus elapsed seconds.
    """
    import torch
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    blocks = []
    t0 = time.time()
    for p, (toks, pl) in enumerate(zip(seqs, prompt_lens)):
        _CAP["h"].clear()
        _CAP["rows"] = 0
        _CAP["calls"] = 0
        _CAP["on"] = True
        llm.generate([TokensPrompt(prompt_token_ids=list(toks))], sp, use_tqdm=False)
        _CAP["on"] = False
        H = torch.cat(_CAP["h"], dim=0)
        _CAP["h"].clear()
        expect = pl + output_len
        if H.shape[0] != expect:
            raise ValueError(
                f"[pass:{label}] seq {p}: captured {H.shape[0]} rows, expected {expect} "
                f"(prompt_len={pl}+output_len={output_len}); prefix caching or chunking "
                "dropped rows")
        blocks.append(H[-(output_len + 1):-1].contiguous())  # output_len completion rows
    out = torch.cat(blocks, dim=0)  # [num_seqs*output_len, HIDDEN]
    dt = time.time() - t0
    print(f"[pass:{label}] {out.shape[0]} completion rows from {len(seqs)} seqs "
          f"in {dt:.0f}s", flush=True)
    return out, dt


def _reduce_control(model, Hc, *, batch: int):
    """argmax_ctrl + top-2 margin map from the real M=8 lm_head over control h."""
    import numpy as np
    import torch

    npos = Hc.shape[0]
    argmax = np.empty(npos, np.int32)
    top1 = np.empty(npos, np.float32)
    top2 = np.empty(npos, np.float32)
    native = None
    with torch.inference_mode():
        for b0 in range(0, npos, batch):
            b1 = min(b0 + batch, npos)
            lg, native = _real_logits(model, Hc[b0:b1].to("cuda"))
            t1, t2, am = _argmax_top2(lg)
            top1[b0:b1] = t1.cpu().numpy()
            top2[b0:b1] = t2.cpu().numpy()
            argmax[b0:b1] = am.cpu().numpy()
            del lg
    return argmax, top1, top2, native


def _reduce_perturbed(model, Hc, Hp, ctrl_argmax, *, mode, splits, lmhead_W,
                      native_dtype, batch, seed):
    """Compounded argmax under a perturbed pass; flip masks + drift diagnostics.

    For each batch, native M=8 logits over the propagated perturbed h (Hp) give the
    UPSTREAM-only argmax (lm_head unperturbed — isolates the new network-wide-h
    compounding #96 closes). The FULL composed frontier then ALSO perturbs the lm_head:
      emu (regime A): + the genuine SplitK delta emu_S − emu_1 on the final logits.
      adversarial (B): + the targeted top1/top2 worst-case ULP shift.
    flip_upstream / flip = argmax != ctrl_argmax for each. Plus ‖Δh‖ (L2, Linf) and
    max|Δlogit| (full) per position. ctrl logits = native M=8 over Hc.
    """
    import numpy as np
    import torch

    if mode == "adversarial":
        _PERT["gen"] = torch.Generator(device="cuda").manual_seed(seed)
    npos = Hc.shape[0]
    flip = np.empty(npos, np.bool_)           # full composed frontier (incl. lm_head)
    flip_up = np.empty(npos, np.bool_)         # upstream-only (lm_head unperturbed)
    pert_argmax = np.empty(npos, np.int32)
    dlogit_max = np.empty(npos, np.float32)
    dh_l2 = np.empty(npos, np.float32)
    dh_inf = np.empty(npos, np.float32)
    with torch.inference_mode():
        for b0 in range(0, npos, batch):
            b1 = min(b0 + batch, npos)
            hc = Hc[b0:b1].to("cuda")
            hp = Hp[b0:b1].to("cuda")
            lc, _ = _real_logits(model, hc)
            lp_up, _ = _real_logits(model, hp)            # native lm_head over perturbed h
            if mode == "emu":
                d = _emu_logits(hp, lmhead_W, splits=splits, native_dtype=native_dtype) \
                    - _emu_logits(hp, lmhead_W, splits=1, native_dtype=native_dtype)
                lp = lp_up + d
            else:  # adversarial
                lp = _lm_head_adv_perturb(lp_up, native_dtype=native_dtype)
            ctrl = torch.from_numpy(ctrl_argmax[b0:b1]).to("cuda")
            am_up = torch.argmax(lp_up, dim=-1).to(torch.int32)
            am = torch.argmax(lp, dim=-1).to(torch.int32)
            flip_up[b0:b1] = (am_up != ctrl).cpu().numpy()
            flip[b0:b1] = (am != ctrl).cpu().numpy()
            pert_argmax[b0:b1] = am.cpu().numpy()
            dlogit_max[b0:b1] = (lp - lc).abs().amax(dim=-1).cpu().numpy()
            dh = (hp.float() - hc.float())
            dh_l2[b0:b1] = dh.norm(dim=-1).cpu().numpy()
            dh_inf[b0:b1] = dh.abs().amax(dim=-1).cpu().numpy()
            del hc, hp, lc, lp, lp_up
    return {
        "flip": flip, "flip_upstream": flip_up, "pert_argmax": pert_argmax,
        "dlogit_max": dlogit_max, "dh_l2": dh_l2, "dh_inf": dh_inf,
    }


def _self_decode_targets(llm, args):
    """Self-decode the deployed model's own greedy completion as the teacher-forcing
    targets (deterministic — the SAME trajectory #87 captured). Returns per-prompt
    full sequences (prompt+completion), prompt lengths, and the flat completion
    token-id array (full-vocab) for the reconstruction-fidelity check."""
    import numpy as np
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    deco = _load_decode_mod()
    tok = AutoTokenizer.from_pretrained(BAKED_DIR)
    records = deco.read_sharegpt_prompts(Path(args.dataset), num_prompts=args.num_prompts,
                                         seed=paths.SEED)
    prompt_tok = [deco.encode_prompt(tok, r["prompt_text"]) for r in records]
    was_on = _CAP["on"]
    _CAP["on"] = False
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    t0 = time.time()
    print(f"[self-decode] generating {len(prompt_tok)} greedy completions "
          f"(output_len={args.output_len}, batched)...", flush=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=list(p)) for p in prompt_tok],
                        sp, use_tqdm=False)
    _CAP["on"] = was_on
    print(f"[self-decode] done in {time.time() - t0:.0f}s", flush=True)
    comps = [list(o.outputs[0].token_ids) for o in outs]
    seqs = [list(p) + c for p, c in zip(prompt_tok, comps)]
    prompt_lens = [len(p) for p in prompt_tok]
    ref_tokens = np.array([t for c in comps for t in c], dtype=np.int64)
    return seqs, prompt_lens, ref_tokens


def capture(args) -> Path:
    import numpy as np
    import torch

    out_dir = Path(args.out_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] out_dir={out_dir}", flush=True)

    for k, v in DEPLOY_ENV.items():
        os.environ.setdefault(k, v)
    for n in paths.prepare_local_gpu_env():
        print(f"[gpu] {n}", flush=True)

    if str(SUBMISSION) not in sys.path:
        sys.path.insert(0, str(SUBMISSION))
    _apply_ple_patches()
    import serve_patch_pck04  # noqa: F401  (registers compute_logits meta-path finder)

    from vllm import LLM

    _install_forward_capture_hook()

    max_len = int(os.environ.get("MAX_MODEL_LEN", "4096"))
    # genuine emulation stores ~6.4 GiB of recovered bf16 weights on-GPU alongside the
    # 8.8 GiB model; the teacher-forced single-seq KV is tiny, so cap vLLM's budget low
    # (≈0.50) to leave physical room for the recovered weights + FP32 emu scratch.
    gpu_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.50"))
    print(f"[capture] constructing in-process LLM (enforce_eager, gpu_util={gpu_util})", flush=True)
    t0 = time.time()
    # max_num_seqs only batches the capture-OFF self-decode (one generate call over
    # all prompts) for speed. The teacher-forced CAPTURE passes submit ONE prompt per
    # generate call (see _run_pass), so each capture forward still holds exactly one
    # sequence regardless of this cap — per-seq h stays contiguous and unambiguous.
    self_decode_bs = int(os.environ.get("SELF_DECODE_MAX_NUM_SEQS", "32"))
    llm = LLM(
        model=BAKED_DIR,
        dtype="bfloat16",
        max_model_len=max_len,
        max_num_seqs=self_decode_bs,
        max_num_batched_tokens=max_len,  # single-chunk prefill -> contiguous per-seq capture
        gpu_memory_utilization=gpu_util,
        enforce_eager=True,
        # all-position capture + per-op perturbation REQUIRE every prefill token to
        # be recomputed: prefix caching would reuse un-perturbed KV for shared
        # prefixes and drop those rows from the captured h. Must be OFF.
        enable_prefix_caching=False,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    print(f"[capture] LLM ready in {time.time() - t0:.0f}s", flush=True)
    fast_prefill = bool(getattr(llm.llm_engine.vllm_config.cache_config,
                                "kv_sharing_fast_prefill", False))
    if fast_prefill:
        raise RuntimeError(
            "kv_sharing_fast_prefill=True invalidates all-position h capture — the "
            "fast path only fully computes logits-index positions. Disable KV-sharing "
            "fast prefill for this gate.")

    # --- self-decode the greedy completion (teacher-forcing targets) ------------
    seqs, prompt_lens, ref_tokens = _self_decode_targets(llm, args)
    print(f"[capture] self-decoded {len(seqs)} greedy completions "
          f"(max_seq_len={max(len(s) for s in seqs)})", flush=True)

    # --- CONTROL pass 0 (perturbation OFF): h_ctrl + probe GEMM dims -------------
    _PERT["mode"] = "off"
    _PROBE.clear()
    control_runs = max(1, int(args.control_runs))
    probe_handles = [mod.register_forward_hook(_probe_hook)
                     for name, mod in llm.llm_engine.model_executor.driver_worker.model_runner
                     .model.named_modules()
                     if name.endswith(PERTURB_SUFFIXES + (PLE_SUFFIXES if args.include_ple else ()))]
    Hc, _ = _run_pass(llm, seqs, prompt_lens, args.output_len, label="control_0")
    for h in probe_handles:
        h.remove()
    npos = Hc.shape[0]
    print(f"[capture] control completion positions: {npos}; probed {len(_PROBE)} GEMM dims",
          flush=True)

    model = _CAP["model"]
    assert model is not None, "forward hook never captured a model"

    ctrl_argmax, ctrl_top1, ctrl_top2, native_dtype = _reduce_control(model, Hc, batch=args.batch)
    print(f"[capture] native kernel dtype={native_dtype}", flush=True)

    # Determinism control: re-run the control pass and confirm 0 divergent argmax.
    control_divergent = []
    for r in range(1, control_runs):
        Hc_r, _ = _run_pass(llm, seqs, prompt_lens, args.output_len, label=f"control_{r}")
        am_r, _, _, _ = _reduce_control(model, Hc_r, batch=args.batch)
        ndiv = int((am_r != ctrl_argmax).sum())
        control_divergent.append(ndiv)
        print(f"[capture] control determinism run {r}: {ndiv}/{npos} divergent", flush=True)
        del Hc_r, am_r

    # fidelity: control argmax (pruned row -> token) vs the self-decoded greedy token.
    keep_ids = _load_keep_ids()
    ctrl_tokens = keep_ids[ctrl_argmax.astype(np.int64)]
    fidelity_disagree = int((ctrl_tokens != ref_tokens).sum())
    print(f"[capture] reconstruction fidelity: prefill argmax vs self-decoded token "
          f"disagree {fidelity_disagree}/{npos} ({100.0 * fidelity_disagree / npos:.3f}%) "
          "(decode/prefill near-tie wobble)", flush=True)

    # --- recover exact GEMM weights for the genuine reduction-order emulation ----
    handles, hooked_names, hooked_mods = _install_perturb_hooks(model, include_ple=args.include_ple)
    _recover_all_weights(model, hooked_mods)
    lmhead_W = _recover_module_weight(model.lm_head, HIDDEN, native_dtype)  # [K_HEAD, HIDDEN]
    print(f"[emu] recovered lm_head W {tuple(lmhead_W.shape)}", flush=True)

    # --- PERTURBED passes -------------------------------------------------------
    regimes: list[dict[str, Any]] = []
    realistic_splits = [int(s) for s in args.realistic_splits]
    save: dict[str, Any] = {
        "ref_tokens": ref_tokens.astype(np.int64),
        "ctrl_argmax": ctrl_argmax,
        "ctrl_top1": ctrl_top1,
        "ctrl_top2": ctrl_top2,
    }

    def _do_regime(mode: str, splits: int, tag: str):
        _PERT["mode"] = mode
        _PERT["splits"] = splits
        _PERT["scale"] = float(args.scale)
        _PERT["max_abs_delta"] = 0.0
        _PERT["n_perturbed"] = 0
        _PERT["gen"] = torch.Generator(device="cuda").manual_seed(0)
        Hp, dt = _run_pass(llm, seqs, prompt_lens, args.output_len, label=tag)
        max_delta = _PERT["max_abs_delta"]
        nperturbed = _PERT["n_perturbed"]
        _PERT["mode"] = "off"
        res = _reduce_perturbed(model, Hc, Hp, ctrl_argmax, mode=mode, splits=splits,
                                lmhead_W=lmhead_W, native_dtype=native_dtype,
                                batch=args.batch, seed=0)
        del Hp
        nflip = int(res["flip"].sum())
        nflip_up = int(res["flip_upstream"].sum())
        print(f"[regime:{tag}] flips_full={nflip}/{npos} flips_upstream={nflip_up}/{npos}  "
              f"max|Δh|inf={res['dh_inf'].max():.4g}  meanΔh_l2={res['dh_l2'].mean():.4g}  "
              f"max|Δlogit|={res['dlogit_max'].max():.4g}  "
              f"hook_max|Δactivation|={max_delta:.4g}", flush=True)
        save[f"{tag}_flip"] = res["flip"]
        save[f"{tag}_flip_upstream"] = res["flip_upstream"]
        save[f"{tag}_pert_argmax"] = res["pert_argmax"]
        save[f"{tag}_dlogit_max"] = res["dlogit_max"]
        save[f"{tag}_dh_l2"] = res["dh_l2"]
        save[f"{tag}_dh_inf"] = res["dh_inf"]
        regimes.append({
            "tag": tag, "mode": mode, "splits": splits, "scale": float(args.scale),
            "flip_count": nflip, "flip_count_upstream": nflip_up, "decode_s": round(dt, 1),
            "max_abs_dh_inf": float(res["dh_inf"].max()),
            "mean_dh_l2": float(res["dh_l2"].mean()),
            "max_abs_dlogit": float(res["dlogit_max"].max()),
            "mean_abs_dlogit": float(res["dlogit_max"].mean()),
            "hook_max_abs_activation_delta": max_delta,
            "n_perturbed_ops": nperturbed,
        })

    # (A) realistic — GENUINE SplitK reduction-order emulation at S∈{2,4,8}; the
    #     worst (max-flip) split is the primary metric.
    for s in realistic_splits:
        _do_regime("emu", s, f"realistic_s{s}")
    # (B) adversarial — synthetic sign-aligned ±1-ULP worst-case upper bound.
    _do_regime("adversarial", 0, "adversarial")

    for h in handles:
        h.remove()
    _GEMM_W.clear()

    npz_path = out_dir / "compounding.npz"
    np.savez_compressed(npz_path, **save)

    margin = (ctrl_top1 - ctrl_top2).astype(np.float32)
    realistic = [r for r in regimes if r["mode"] == "emu"]
    realistic_flips = [r["flip_count"] for r in realistic]
    realistic_flips_up = [r["flip_count_upstream"] for r in realistic]
    summary = {
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "num_positions": int(npos),
        "native_dtype": str(native_dtype),
        "softcap": SOFTCAP,
        "scale": float(args.scale),
        "include_ple": bool(args.include_ple),
        "kv_sharing_fast_prefill": fast_prefill,
        "perturbation_model": "genuine_splitk_reduction_order_emu",
        "n_gemm_hooks": len(handles),
        "n_gemm_weights_recovered": len(_GEMM_W) if _GEMM_W else len(hooked_mods),
        "perturb_suffixes": list(PERTURB_SUFFIXES + (PLE_SUFFIXES if args.include_ple else ())),
        "control_runs": control_runs,
        "control_divergent_tokens": control_divergent,
        "control_divergent_max": max(control_divergent) if control_divergent else 0,
        "reconstruction_fidelity_disagreements": fidelity_disagree,
        "min_margin": float(margin.min()),
        "median_margin": float(np.median(margin)),
        "realistic_splits": realistic_splits,
        "compounded_argmax_flip_count_realistic": int(max(realistic_flips)) if realistic_flips else None,
        "compounded_argmax_flip_count_realistic_per_split": realistic_flips,
        "compounded_argmax_flip_count_realistic_upstream": int(max(realistic_flips_up)) if realistic_flips_up else None,
        "compounded_argmax_flip_count_realistic_upstream_per_split": realistic_flips_up,
        "compounded_argmax_flip_count_adversarial": next(
            (r["flip_count"] for r in regimes if r["mode"] == "adversarial"), None),
        "compounded_argmax_flip_count_adversarial_upstream": next(
            (r["flip_count_upstream"] for r in regimes if r["mode"] == "adversarial"), None),
        "regimes": regimes,
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
    ap.add_argument("--scale", type=float, default=1.0,
                    help="per-op ULP perturbation scale (adversarial regime; 1.0 = full ULP)")
    ap.add_argument("--realistic-splits", nargs="+", type=int, default=[2, 4, 8],
                    help="SplitK widths for regime (A); max flips across splits is the primary metric")
    ap.add_argument("--control-runs", type=int, default=3, help="in-process determinism re-runs")
    ap.add_argument("--include-ple", action="store_true",
                    help="also perturb PLE projections (conservative; out of PR named set)")
    ap.add_argument("--batch", type=int, default=4096, help="positions per logit-reduction batch")
    ap.add_argument("--out-dir", default=str(REPO / "research/validity/greedy_compounding"))
    ap.add_argument("--smoke", action="store_true", help="4 prompts × 32 tok plumbing check")
    args = ap.parse_args()
    if args.smoke:
        args.num_prompts = 4
        args.output_len = 32
        args.batch = 512
        args.realistic_splits = [8]
        args.control_runs = 2
        return _smoke_capture(args)
    capture(args)
    return 0


def _smoke_capture(args) -> int:
    """Smoke path: self-generate a short greedy completion as the teacher-forcing
    target, then run the full control -> weight-recovery -> perturbed plumbing at a
    tiny scale. Verifies the hooks, the exact-weight recovery + emu_1-vs-native
    fidelity, the genuine SplitK emu regime, and the adversarial regime.
    """
    import numpy as np

    out_dir = Path(args.out_dir) / ("smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] out_dir={out_dir}", flush=True)
    for k, v in DEPLOY_ENV.items():
        os.environ.setdefault(k, v)
    for n in paths.prepare_local_gpu_env():
        print(f"[gpu] {n}", flush=True)
    if str(SUBMISSION) not in sys.path:
        sys.path.insert(0, str(SUBMISSION))
    _apply_ple_patches()
    import serve_patch_pck04  # noqa: F401

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    _install_forward_capture_hook()
    deco = _load_decode_mod()
    tok = AutoTokenizer.from_pretrained(BAKED_DIR)
    records = deco.read_sharegpt_prompts(Path(args.dataset), num_prompts=args.num_prompts,
                                         seed=paths.SEED)
    prompt_tok = [deco.encode_prompt(tok, r["prompt_text"]) for r in records]
    max_len = max(len(p) for p in prompt_tok) + args.output_len + 8
    # recovered weights are tiny relative to A10G here; keep util modest for headroom.
    llm = LLM(model=BAKED_DIR, dtype="bfloat16", max_model_len=max(1024, max_len),
              max_num_seqs=1, max_num_batched_tokens=max(1024, max_len),
              gpu_memory_utilization=0.70, enforce_eager=True, trust_remote_code=True,
              enable_prefix_caching=False, disable_log_stats=True)
    # self-decode greedy completions (the teacher-forcing targets)
    _CAP["on"] = False
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len, ignore_eos=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_tok], sp, use_tqdm=False)
    comps = [list(o.outputs[0].token_ids) for o in outs]
    ref = np.array([t for c in comps for t in c], dtype=np.int64)
    seqs = [list(p) + c for p, c in zip(prompt_tok, comps)]
    prompt_lens = [len(p) for p in prompt_tok]

    # control pass + probe GEMM dims (probe hooks learn in/out shapes for recovery)
    _PERT["mode"] = "off"
    _PROBE.clear()
    runner_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    probe_handles = [mod.register_forward_hook(_probe_hook)
                     for name, mod in runner_model.named_modules()
                     if name.endswith(PERTURB_SUFFIXES + (PLE_SUFFIXES if args.include_ple else ()))]
    Hc, _ = _run_pass(llm, seqs, prompt_lens, args.output_len, label="control_0")
    for h in probe_handles:
        h.remove()
    model = _CAP["model"]
    ctrl_argmax, _, _, native = _reduce_control(model, Hc, batch=args.batch)
    # ctrl_argmax is a pruned-row index; the self-decoded ref is a full-vocab token
    # id — map the row index through keep_ids before comparing.
    keep_ids = _load_keep_ids()
    ctrl_tokens = keep_ids[ctrl_argmax.astype(np.int64)]
    fid = int((ctrl_tokens != ref).sum())
    print(f"[smoke] positions={Hc.shape[0]} fidelity_disagree={fid} native={native} "
          f"probed {len(_PROBE)} GEMM dims", flush=True)

    # recover exact dequantized weights for the genuine reduction-order emulation
    handles, _, hooked_mods = _install_perturb_hooks(model, include_ple=args.include_ple)
    _recover_all_weights(model, hooked_mods)
    lmhead_W = _recover_module_weight(model.lm_head, HIDDEN, native)

    for mode, splits, tag in [("emu", 8, "realistic_s8"), ("adversarial", 0, "adversarial")]:
        _PERT["mode"] = mode
        _PERT["splits"] = splits
        _PERT["scale"] = float(args.scale)
        _PERT["max_abs_delta"] = 0.0
        _PERT["n_perturbed"] = 0
        Hp, _ = _run_pass(llm, seqs, prompt_lens, args.output_len, label=tag)
        _PERT["mode"] = "off"
        res = _reduce_perturbed(model, Hc, Hp, ctrl_argmax, mode=mode, splits=splits,
                                lmhead_W=lmhead_W, native_dtype=native, batch=args.batch, seed=0)
        print(f"[smoke:{tag}] flips_full={int(res['flip'].sum())}/{Hc.shape[0]} "
              f"flips_up={int(res['flip_upstream'].sum())}/{Hc.shape[0]} "
              f"max|Δh|inf={res['dh_inf'].max():.4g} max|Δlogit|={res['dlogit_max'].max():.4g} "
              f"hook_max|Δact|={_PERT['max_abs_delta']:.4g}", flush=True)
    for h in handles:
        h.remove()
    _GEMM_W.clear()
    print("[smoke] OK — plumbing verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
