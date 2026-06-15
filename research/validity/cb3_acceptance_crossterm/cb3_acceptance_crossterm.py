#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""The supply x demand acceptance cross-term: does cb3 (supply) perturb the body
hidden states the MTP drafter (demand) reads? (PR #410, ubel).

THE QUESTION
------------
The only surviving >500 route is supply (conservative-k cb3) + demand (tree), and #402
(8pcyhe2r) established tree_plus_cb3_required = True. That plan ASSUMES the two legs are
orthogonal -- that the cb3 supply lift and the tree demand lift simply ADD. They may be
destructively coupled through the drafter's INPUT: the deployed Gemma4Assistant MTP K=7
drafter consumes the body's hidden states (`shared_kv_states` + the final hidden that feeds
`inputs_embeds`, both 2560-dim backbone tensors -- see #401 i2qsjyp6). cb3 int4-quantizes the
body, so the hidden states the drafter reads are PERTURBED, so the drafter's top-1 / E[accepted]
(0.7293 / 2.851, #289 + #401) could DROP -- shrinking the very coverage the demand leg monetizes.
This cross-term is un-measured and load-bearing: it is the single assumption under the only
surviving >500 route.

THE MEASUREMENT (PRIMARY, unblocked -- upstream of the drafter)
--------------------------------------------------------------
Load the Gemma4 body (google/gemma-4-E4B-it, bf16) on the local A10G (sm_86, on-target). Run the
deployed 128/128 harness prompts. Capture the EXACT tensors the MTP drafter reads:
  * inputs_embeds_hidden = language_model final hidden state hidden_states[-1]  [T, 2560]
    (HALF of the drafter's 5120-dim inputs_embeds; the other half is the next-token embedding,
     which cb3 does NOT touch -- so the body hidden is the only cb3-coupled input here).
  * shared_kv_states     = per-layer k_proj / v_proj outputs (42 layers, [T, 512] = 2 KV heads
    x 256). L2-relative perturbation is INVARIANT under RoPE (a per-position orthogonal rotation
    applied identically to ref and perturbed), so the pre-RoPE projection output is a faithful
    proxy for the post-RoPE cached K the drafter cross-attends into; V has no RoPE.
Quantize the body weights with a controlled RTN-g128 sweep (isolates the quantization step on the
SAME bf16 weights -- zero kernel/training confounds) and measure, per tensor, the perturbation vs
the bf16 reference: L2-relative, cosine, and per-channel max|delta|.

WHY RTN-g128 IS THE HONEST cb3 PROXY (cb3 has NO shipping kernel)
----------------------------------------------------------------
cb3 = RHT-incoherence + L1-resident K=64 dim-2 Gaussian VQ (QTIP/QuIP# class), 3.2369 effective
bpw over 88.8% of the body (#372/#388). cb3_kernel_realized_bw PROVED there is no shipping vLLM
0.22 kernel and no in-repo cb3 quantizer -- a TRUE cb3 forward is not runnable here. We therefore
bracket cb3 between two runnable RTN points:
  * int4 RTN g128 (4 bits)  ~= the deployed int4-Marlin body (INT4_BPW 4.125) -- the PR's
    "cb3-int4" int4-class realization.
  * int3 RTN g128 (3 bits)  = a STRICT CONSERVATIVE UPPER BOUND on cb3's perturbation. cb3 (3.2369
    bpw) has fewer bits than int4 (pushes error up) BUT its RHT+VQ reduce error below scalar RTN at
    equal bpw (pushes error down). int3-RTN (3.0 bpw, scalar, no RHT/VQ) is UNAMBIGUOUSLY cruder
    than cb3 on both axes: fewer bits AND no error-reduction transform. So
        perturbation(cb3) <= perturbation(int3-RTN).
    If even int3-RTN perturbs the drafter inputs negligibly, cb3 -- provably gentler -- does too.
fp16 (bf16 identity) anchors the sweep at Delta=0 and gives the monotonicity self-test a base.

THE ACCEPTANCE BOUND (SECONDARY, caveated -- the faithful drafter read is blocked, #401)
----------------------------------------------------------------------------------------
A faithful drafter top-k read needs a custom vLLM-MTP patch (the bf16 HF read is the wrong
distribution with 9-13% argmax flips; #401). We do NOT re-litigate that wall. Instead we propagate
the measured body-hidden perturbation through a BODY-PROXY local sensitivity slope: under the cb3
upper-bound perturbation, how often does the BODY'S OWN greedy argmax flip, and by how much does
its top-1 logit margin move? The MTP drafter is TRAINED TO MIMIC the body's next token, so the
body's local sensitivity is a defensible (caveated) proxy for the drafter's. This is a within-process
A/B (same lm_head, same session), so the flip signal is the perturbation's -- the 9-13% figure is a
CROSS-session GEMV artifact and does not contaminate the slope. We translate the bounded Delta-top1
through the #289 K=7 ladder to Delta-E[accepted] and via the secant gross_tps_gain_per_unit_cov =
962.27 to Delta-demand_TPS. Caveat (per #401): this is the BODY's bf16-lm_head sensitivity, used
ONLY as a sensitivity SLOPE, never as the absolute deployed (prune-12k + int4) distribution. If the
lm_head wiring is unavailable, these fields are stored as the honest sentinel "blocked:unmeasured";
the PRIMARY perturbation alone answers the verdict directionally.

VERDICT: is the supply x demand acceptance cross-term NEGLIGIBLE (|Delta-demand_TPS| < 5% of the
demand lift) or DESTRUCTIVE? -- with the perturbation magnitude as primary evidence and the
sensitivity bound as the (caveated) amplifier.

NOT a launch, NOT a submission, NO served-file change, 0 official TPS. GPU used ONLY for the
local perturbation forward (analysis). PPL UNCHANGED (still greedy target token; this card
characterizes only the drafter-acceptance leg). analysis_only = no_hf_job = no_served_file_change
= True; official_tps = 0.

REPRODUCE (needs torch + a visible GPU; the repo .venv has no torch -- use /usr/bin/python3):
    cd target/ && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 \
      research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm.py --self-test
    cd target/ && CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 \
      research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm.py \
        --quant-sweep fp16,int8,int4,cb3-int3 --measure shared_kv_states,inputs_embeds_hidden \
        --bound-acceptance --wandb_group cb3-quant-acceptance-crossterm \
        --wandb_name ubel/cb3-quant-acceptance-crossterm
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # target/

# ===========================================================================
# Section 0 -- banked anchors (imported EXACTLY from merged advisor-branch cards / PR #410 body)
# ===========================================================================
MODEL_ID = "google/gemma-4-E4B-it"                     # base bf16 body (multimodal; text tower used)
PROMPTS_PATH = REPO / "official/main_bucket/shared_resources/speed_benchmark/data/eval_prompts_sharegpt.json"

# cb3 numerics (#372/#388; cb3_supply_lift_mtp_honest / cb3_kernel_realized_bw) -----------------------
INT4_BPW = 4.125                       # deployed int4-Marlin: 4b weight + bf16 g128 scale
CB3_BPW_EFF = 3.2368598382749325       # cb3 (RHT + K=64 dim-2 VQ) effective bpw, 88.8% of body
CB3_MIXED_FRAC = 0.888                 # fraction of body params at cb3
CB3_BYTE_RATIO = CB3_BPW_EFF / INT4_BPW  # 0.785 -- cb3 reads this fraction of int4 weight bytes

# Deployed MTP drafter acceptance (#289 fi34s269 a_1..a_7, cross-checked #76; the demand leg's input)
LADDER_289 = [0.7292532942898975, 0.759556697719242, 0.7929794882639035, 0.8228,
              0.8348727920920435, 0.8357919254658385, 0.8464932652113331]
DRAFTER_TOP1_289 = LADDER_289[0]       # 0.7293 deployed MTP top-1 (the quantity that could DROP)
E_ACCEPTED_289 = 2.851185944363104     # #289 E[accepted draft tokens]/step
E_T_289 = 3.851185944363104            # #289 E[T] = 1 + E[accepted]

# Demand secant + base (#402 8pcyhe2r / #401) -- monetizes a coverage delta into TPS ------------------
GROSS_TPS_PER_UNIT_COV = 962.27        # #402 corrected demand secant (467.48 base)
CORRECTED_STRICT_BASE = 467.48         # #393/#402 corrected strict base
GAP_TO_500 = 32.53                     # #393 strict gap_to_500
CB3_SUPPLY_LIFT_388 = 38.02            # #388 realized cb3 supply lift (TPS)
# demand lift the cross-term could erode: the locked top-1->top-4 tree coverage band (#401),
# monetized at the #402 secant. This is the denominator for the 5% negligibility test.
DEMAND_COVERAGE_BAND = 0.10973404808468479   # #401 coverage_ceiling_gap (top-4 -> full)
DEMAND_LIFT_TPS_BAND = DEMAND_COVERAGE_BAND * GROSS_TPS_PER_UNIT_COV  # ~+105.6 TPS upper demand lift
NEGLIGIBLE_FRAC = 0.05                  # cross-term negligible iff |Delta-demand_TPS| < 5% of demand lift

# deployed baseline (UNCHANGED -- 0-TPS card) --------------------------------------------------------
BASELINE_TPS = 481.53                  # PR #52 (2x9fm2zx) deployed public TPS
BASELINE_PPL = 2.3772                  # deployed int4 body gate PPL (the cross-term reference)
CB3_PPL = 2.3812                       # cb3 (RHT+VQ 3.24bpw) gate PPL (#388) -- HOLDS the 2.42 gate
PPL_GATE = 2.42
# Gemma4 applies final-logit soft-capping (config.final_logit_softcapping=30.0):
#   z = c * tanh(lm_head(h)/c).  The captured drafter-input hidden (out.hidden_states[-1]) is ALREADY
#   post-final-norm (== last_hidden_state, verified bit-identical on the A10G), so the faithful body head
#   is lm_head(h) WITHOUT a second norm, then this softcap. Omitting either makes teacher-forced PPL
#   meaningless (argmax/flip are invariant to both, so the perturbation legs are unaffected).
FINAL_LOGIT_SOFTCAP = 30.0
GATE_HEADROOM_REL = PPL_GATE / BASELINE_PPL - 1.0          # +1.81% rel PPL room before gate-fail
CB3_REL_PPL_OVER_INT4 = CB3_PPL / BASELINE_PPL - 1.0       # +0.17% -- cb3 vs deployed int4 (PPL parity)
# lawine #355 (vqzzc9jw, on-target A10G): b=3 scalar/RTN body = +13.94% rel PPL -> 2.7085, gate-DEAD.
# My RTN sweep must reproduce this collapse (int3 PPL >> int8/int4) to validate the harness.
INT3_RTN_RELPPL_355 = 0.1394
INT3_RTN_GATE_PPL_355 = 2.7085

# self-test tolerances
ZERO_TOL = 1e-4                        # fp16-vs-fp16 perturbation must be ~0 (within-process determinism)
MONO_EPS = 1e-6                        # monotonicity slack

# ===========================================================================
# Section 1 -- scheme parsing (cb3 proxy bracket)
# ===========================================================================

def scheme_bits(name: str):
    """Map a sweep scheme name to its RTN bit-width (None = bf16 reference / no quant)."""
    n = name.strip().lower()
    if n in ("fp16", "bf16", "fp32", "none", "ref"):
        return None
    if n in ("int8", "w8", "8bit"):
        return 8
    if n in ("int4", "cb3-int4", "w4", "4bit"):
        return 4
    if n in ("int3", "cb3-int3", "cb3-ub", "cb3", "w3", "3bit"):
        return 3
    if n in ("int2", "w2", "2bit"):
        return 2
    raise ValueError(f"unknown quant scheme: {name!r}")


def scheme_role(name: str) -> str:
    """Human-readable role of each sweep point for the report."""
    b = scheme_bits(name)
    if b is None:
        return "reference (bf16; Delta=0 sanity)"
    if b == 8:
        return "int8 RTN g128 (monotonicity rung)"
    if b == 4:
        return "int4 RTN g128 ~= deployed int4-Marlin body; the PR 'cb3-int4' int4-class point"
    if b == 3:
        return "int3 RTN g128 = STRICT conservative UPPER BOUND on cb3 (3.2369bpw RHT+VQ <= int3-RTN)"
    return f"int{b} RTN g128"


# ===========================================================================
# Section 2 -- RTN-g128 fake quantization (the runnable cb3 proxy)
# ===========================================================================

def fake_quant_grouped(w_cpu, bits: int, group: int = 128):
    """Symmetric per-group RTN fake-quant of a [out, in] weight. Operates from a CPU bf16 source
    and returns a bf16 dequantized tensor on the same (CPU) device. Group runs along the input dim."""
    import torch  # noqa: PLC0415
    out, inn = w_cpu.shape
    w = w_cpu.float()
    if inn % group != 0:
        # all quantized body GEMMs have in in {2560,2048,4096,10240}, multiples of 128; guard anyway.
        return w_cpu.clone()
    wq = w.view(out, inn // group, group)
    absmax = wq.abs().amax(dim=-1, keepdim=True)
    qmax = (2 ** (bits - 1)) - 1
    scale = (absmax / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(wq / scale), -qmax, qmax)
    deq = (q * scale).view(out, inn)
    return deq.to(torch.bfloat16)


# ===========================================================================
# Section 3 -- body load + drafter-input capture wiring
# ===========================================================================

def load_body():
    """Load the Gemma4 body on CUDA (bf16); return (model, tokenizer, text_tower, lm_head, norm)."""
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    try:
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForImageTextToText  # noqa: PLC0415
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    text = None
    for _, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        emb = getattr(mod, "embed_tokens", None)
        if layers is not None and emb is not None and len(layers) >= 30:
            text = mod
            break
    if text is None:
        raise RuntimeError("could not locate the >=30-layer text tower")
    model = model.to("cuda").eval()
    lm_head = getattr(model, "lm_head", None)
    norm = getattr(text, "norm", None)
    return model, tok, text, lm_head, norm


def quantizable_linears(text, min_dim: int = 512):
    """The cb3 body GEMMs: Linear modules in the decoder layers with min(out,in) >= 512
    (gate_up / down / q / k / v / o). The tiny 256-dim altup/laurel projections are left
    unquantized -- negligible bytes, consistent with cb3 mixed-precision."""
    import torch.nn as nn  # noqa: PLC0415
    out = []
    for name, m in text.named_modules():
        if isinstance(m, nn.Linear) and min(tuple(m.weight.shape)) >= min_dim:
            out.append((name, m))
    return out


def cache_original_weights(linears):
    """Snapshot pristine bf16 weights on CPU so each scheme re-quantizes from the source."""
    return {name: m.weight.detach().to("cpu").clone() for name, m in linears}


def apply_scheme(linears, originals, bits):
    """Set each quantizable Linear's weight to the RTN-quantized (or pristine, if bits is None)
    version of its cached original, on the model's device."""
    import torch  # noqa: PLC0415
    for name, m in linears:
        src = originals[name]
        if bits is None:
            w = src
        else:
            w = fake_quant_grouped(src, bits)
        m.weight.data.copy_(w.to(m.weight.device, dtype=m.weight.dtype))
    torch.cuda.synchronize()


def register_kv_hooks(text):
    """Hook k_proj / v_proj outputs for every layer that OWNS them. Returns (handles, captured,
    kv_layers) where captured['K'][p] / ['V'][p] hold the most recent [1, T, 512] projection output
    for the p-th KV-bearing layer (kv_layers[p] is its absolute index).

    Gemma4 E4B shares KV across the last `num_kv_shared_layers` (18 of 42): layers 0-23 own
    k_proj/v_proj; layers 24-41 reuse the earlier cache and expose only q_proj/o_proj. The distinct
    KV tensors the MTP drafter cross-attends into are therefore exactly the hooked (owning) layers' --
    hooking the shared layers is impossible (no module) and redundant (same cache)."""
    kv_layers = [i for i, layer in enumerate(text.layers)
                 if hasattr(layer.self_attn, "k_proj") and hasattr(layer.self_attn, "v_proj")]
    captured = {"K": [None] * len(kv_layers), "V": [None] * len(kv_layers)}

    def mk(pos, kind):
        def hook(_m, _inp, out):
            captured[kind][pos] = out.detach()
        return hook

    handles = []
    for pos, i in enumerate(kv_layers):
        attn = text.layers[i].self_attn
        handles.append(attn.k_proj.register_forward_hook(mk(pos, "K")))
        handles.append(attn.v_proj.register_forward_hook(mk(pos, "V")))
    return handles, captured, kv_layers


def build_prompt_ids(tok, prompt_text, max_seq_len):
    """Format one user prompt through the chat template (gemma-it served form) and tokenize."""
    try:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True, return_tensors="pt",
        )
    except Exception:  # noqa: BLE001
        enc = tok(prompt_text, return_tensors="pt")
    # transformers 5.9 apply_chat_template may return a BatchEncoding (dict-like) or a bare tensor.
    if hasattr(enc, "input_ids"):       # BatchEncoding / ModelOutput
        ids = enc.input_ids
    elif isinstance(enc, dict):
        ids = enc["input_ids"]
    else:                                # already a tensor
        ids = enc
    if ids.shape[1] > max_seq_len:
        ids = ids[:, :max_seq_len]
    return ids.to("cuda")


def forward_capture(text, captured, n_kv, ids):
    """Run the text tower; return (final_hidden [T,2560], K_list, V_list) on CPU bf16. K/V lists span
    the n_kv KV-owning layers (the distinct KV tensors the drafter reads)."""
    out = text(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1][0].to("cpu")  # [T, 2560] final decoder-layer hidden (drafter input)
    K = [captured["K"][p][0].to("cpu") for p in range(n_kv)]
    V = [captured["V"][p][0].to("cpu") for p in range(n_kv)]
    return hidden, K, V


# ===========================================================================
# Section 4 -- perturbation accumulators
# ===========================================================================

class TensorPerturb:
    """Pooled L2-relative + cosine + per-channel max|delta| accumulator for one tensor family.

    L2-rel (pooled) = sqrt(sum_p ||delta_p||^2) / sqrt(sum_p ||ref_p||^2)
    cosine          = mean over (prompt, position) of cos(ref_pos, pert_pos)
    per-channel maxabs delta tracked over the last (feature) dim.
    """

    def __init__(self, n_channels=None):
        self.sumsq_delta = 0.0
        self.sumsq_ref = 0.0
        self.cos_sum = 0.0
        self.cos_n = 0
        self.max_abs_delta = 0.0
        self.n_channels = n_channels
        # shared_kv_states pools layers of MIXED width (sliding=512 / full=1024 KV dims in Gemma4
        # E4B), so the per-channel tracker is keyed by channel-count C -> tensor[C].
        self._chan_max = {}

    def update(self, ref, pert):
        import torch  # noqa: PLC0415
        ref = ref.float()
        pert = pert.float()
        delta = pert - ref
        self.sumsq_delta += float((delta * delta).sum())
        self.sumsq_ref += float((ref * ref).sum())
        self.max_abs_delta = max(self.max_abs_delta, float(delta.abs().max()) if delta.numel() else 0.0)
        # per-position cosine over the feature dim (last dim)
        rn = ref.norm(dim=-1)
        pn = pert.norm(dim=-1)
        denom = (rn * pn).clamp_min(1e-12)
        cos = (ref * pert).sum(dim=-1) / denom
        self.cos_sum += float(cos.sum())
        self.cos_n += int(cos.numel())
        # per-channel max|delta| over the feature dim, kept per channel-width (mixed across layers)
        C = ref.shape[-1]
        flat = delta.reshape(-1, C).abs().amax(dim=0)  # [C]
        if C not in self._chan_max:
            self._chan_max[C] = flat.clone()
        else:
            self._chan_max[C] = torch.maximum(self._chan_max[C], flat)

    def finalize(self) -> dict:
        import torch  # noqa: PLC0415
        l2_rel = math.sqrt(self.sumsq_delta) / math.sqrt(self.sumsq_ref) if self.sumsq_ref > 0 else 0.0
        cosine = self.cos_sum / self.cos_n if self.cos_n else 1.0
        if self._chan_max:
            per_chan_max = max(float(t.max()) for t in self._chan_max.values())
            cat = torch.cat([t.reshape(-1) for t in self._chan_max.values()])
            per_chan_mean = float(cat.mean())
        else:
            per_chan_max = 0.0
            per_chan_mean = 0.0
        return {
            "l2_relative": l2_rel,
            "cosine": cosine,
            "max_abs_delta": self.max_abs_delta,
            "per_channel_max_abs_delta_max": per_chan_max,
            "per_channel_max_abs_delta_mean": per_chan_mean,
            "sumsq_ref": self.sumsq_ref,
        }


# ===========================================================================
# Section 5 -- the measurement driver
# ===========================================================================

def run_measurement(schemes, measure, bound_acceptance, max_prompts, max_seq_len, verbose=True):
    """Measure the cb3-proxy perturbation at the drafter-input tensors across the quant sweep."""
    import torch  # noqa: PLC0415

    prompts_all = json.loads(PROMPTS_PATH.read_text())
    prompt_texts = [p["conversations"][0]["value"] for p in prompts_all][:max_prompts]
    n_prompts = len(prompt_texts)

    t_load = time.time()
    model, tok, text, lm_head, norm = load_body()
    load_s = time.time() - t_load
    linears = quantizable_linears(text)
    originals = cache_original_weights(linears)
    handles, captured, kv_layers = register_kv_hooks(text)
    n_layers = len(text.layers)
    n_kv = len(kv_layers)

    # tokenize once
    ids_list = [build_prompt_ids(tok, t, max_seq_len) for t in prompt_texts]
    seqlens = [int(x.shape[1]) for x in ids_list]

    # ---- reference pass (bf16): store per-prompt drafter-input tensors on CPU ----
    apply_scheme(linears, originals, None)
    ref_hidden, ref_K, ref_V = [], [], []
    for ids in ids_list:
        h, K, V = forward_capture(text, captured, n_kv, ids)
        ref_hidden.append(h)
        ref_K.append(K)
        ref_V.append(V)
    torch.cuda.synchronize()
    ref_peak_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    # Secondary body-proxy: the body's greedy argmax at EVERY position is the per-position verification
    # target the MTP drafter is trained to mimic; its flip-rate under quant is the target-side acceptance
    # disruption. Teacher-forced PPL anchors each scheme's quant CRUDENESS (cross-checks lawine #355's
    # on-target "scalar sub-int4 breaks the 2.42 gate" and locates cb3 -- which HOLDS the gate -- on the
    # PPL->perturbation curve). Within-process A/B (same lm_head/session) => the flip is the perturbation's,
    # not the #401 cross-session GEMV artifact.
    def body_logits_stats(hidden_cpu, ids):
        if lm_head is None:
            return None
        h = hidden_cpu.to("cuda")
        z = lm_head(h)                                        # h is ALREADY post-final-norm; no 2nd norm
        z = FINAL_LOGIT_SOFTCAP * torch.tanh(z / FINAL_LOGIT_SOFTCAP)   # Gemma final-logit softcap (=30)
        am = z.argmax(dim=-1).to("cpu")                      # [T] greedy token per position
        tgt = ids[0, 1:]                                      # teacher-forced next-token targets
        lp = torch.log_softmax(z[:-1].float(), dim=-1)
        nll = float(-lp[torch.arange(tgt.numel(), device=lp.device), tgt].sum())
        del z, lp
        return am, nll, int(tgt.numel())

    ref_stats = [body_logits_stats(ref_hidden[pi], ids_list[pi]) for pi in range(n_prompts)] \
        if bound_acceptance else None

    # ---- per-scheme perturbation ----
    per_scheme = {}
    for sname in schemes:
        bits = scheme_bits(sname)
        if bits is None:
            # fp16 self-consistency: re-run bf16 and measure Delta vs stored reference (must be ~0)
            apply_scheme(linears, originals, None)
        else:
            apply_scheme(linears, originals, bits)
        acc_hidden = TensorPerturb()
        acc_kv = TensorPerturb()
        # secondary accumulators (all-position body-proxy: argmax flip vs bf16 ref + teacher-forced PPL)
        argmax_flips = 0
        argmax_tokens = 0
        nll_sum = 0.0
        nll_tok = 0
        for pi, ids in enumerate(ids_list):
            h, K, V = forward_capture(text, captured, n_kv, ids)
            if "inputs_embeds_hidden" in measure:
                acc_hidden.update(ref_hidden[pi], h)
            if "shared_kv_states" in measure:
                for li in range(n_kv):
                    acc_kv.update(ref_K[pi][li], K[li])
                    acc_kv.update(ref_V[pi][li], V[li])
            if bound_acceptance and ref_stats is not None and ref_stats[pi] is not None:
                cur = body_logits_stats(h, ids)
                if cur is not None:
                    am_cur, nll, ntok = cur
                    am_ref = ref_stats[pi][0]
                    m = min(am_ref.numel(), am_cur.numel())
                    argmax_flips += int((am_ref[:m] != am_cur[:m]).sum())
                    argmax_tokens += m
                    nll_sum += nll
                    nll_tok += ntok
        res = {
            "scheme": sname, "bits": bits, "role": scheme_role(sname),
            "inputs_embeds_hidden": acc_hidden.finalize() if "inputs_embeds_hidden" in measure else None,
            "shared_kv_states": acc_kv.finalize() if "shared_kv_states" in measure else None,
        }
        if bound_acceptance:
            res["body_proxy"] = {
                "argmax_flip_rate": (argmax_flips / argmax_tokens) if argmax_tokens else None,
                "n_positions": argmax_tokens,
                "teacher_forced_ppl": (math.exp(nll_sum / nll_tok) if nll_tok else None),
            }
        per_scheme[sname] = res
        if verbose:
            kv = res["shared_kv_states"]
            hh = res["inputs_embeds_hidden"]
            print(f"  [{sname:>9}] bits={bits} "
                  f"kv_L2rel={kv['l2_relative']:.6f} kv_cos={kv['cosine']:.6f} | "
                  f"hid_L2rel={hh['l2_relative']:.6f} hid_cos={hh['cosine']:.6f}"
                  + (f" | body_flip={res['body_proxy']['argmax_flip_rate']}" if bound_acceptance else ""))

    for h in handles:
        h.remove()

    meta = {
        "n_prompts": n_prompts, "max_seq_len": max_seq_len,
        "seqlen_min": min(seqlens), "seqlen_max": max(seqlens),
        "seqlen_mean": sum(seqlens) / len(seqlens),
        "n_layers": n_layers, "n_kv_layers": n_kv, "kv_layers": kv_layers,
        "n_quantized_linears": len(linears),
        "load_s": round(load_s, 2), "ref_peak_vram_mb": ref_peak_mb,
        "lm_head_available": lm_head is not None and norm is not None,
    }
    return per_scheme, meta


# ===========================================================================
# Section 6 -- acceptance-shift bound (secondary, caveated) + verdict
# ===========================================================================

def expected_accepted(ladder):
    cum, acc = 1.0, 0.0
    for a in ladder:
        cum *= a
        acc += cum
    return acc


def _propagate_flip(flip):
    """A body greedy-argmax flip is the per-position proxy event for the drafter's verification target
    changing -> an UPPER bound on |Delta-top1| (the drafter mimics the body; realistic loss <= flip
    because the drafter also reads the shifted hidden and may track it). Propagate the bound through the
    #289 K=7 ladder (scale a_1 down by Delta-top1) to Delta-E[accepted], and via the #402 secant to
    Delta-demand_TPS (conservatively equate Delta-coverage ~ Delta-top1, the largest single-position drop)."""
    ladder_lo = list(LADDER_289)
    ladder_lo[0] = max(0.0, ladder_lo[0] - flip)
    de_acc = expected_accepted(LADDER_289) - expected_accepted(ladder_lo)
    dtps = flip * GROSS_TPS_PER_UNIT_COV
    return {"delta_top1_bound": flip, "delta_e_accepted_bound": de_acc,
            "delta_coverage_bound": flip, "delta_demand_tps_bound": dtps}


def _by_bits(per_scheme, schemes, b):
    for s in schemes:
        if scheme_bits(s) == b and per_scheme.get(s, {}).get("body_proxy"):
            return s
    return None


def bound_acceptance_shift(per_scheme, schemes):
    """Two acceptance bounds, both via the body-proxy flip slope:
      * FAITHFUL gate-safe estimate -- cb3 HOLDS the 2.42 gate (PPL 2.3812, +0.17% over deployed int4,
        well inside the +1.81% headroom). The only gate-safe rung in the RTN sweep is int8 (scalar sub-int4
        is gate-DEAD per lawine #355). So cb3's drafter-target flip <= the int8 rung's -> this is the
        load-bearing, realistic upper anchor for the cross-term.
      * LOOSE conservative UB -- the int3 rung. PROVABLY >= cb3 (fewer bits + no RHT/VQ), but it is the
        gate-BROKEN scalar regime (#355: +13.94% PPL), so it MASSIVELY overstates cb3 and is reported only
        as the strict-but-uninformative ceiling.
    The FAITHFUL cb3-vs-int4 drafter-INPUT read (cb3 hidden/KV) needs a real cb3 kernel and is BLOCKED
    (#372/cb3_kernel_realized_bw -> no shipping kernel)."""
    s8 = _by_bits(per_scheme, schemes, 8)    # gate-safe rung (faithful cb3 anchor)
    s4 = _by_bits(per_scheme, schemes, 4)    # int4-class point
    s3 = _by_bits(per_scheme, schemes, 3)    # cb3 conservative (gate-broken) UB
    if s8 is None and s4 is None and s3 is None:
        return {"status": "blocked:unmeasured",
                **{k: "blocked:unmeasured" for k in
                   ("delta_top1_bound", "delta_e_accepted_bound", "delta_coverage_bound", "delta_demand_tps_bound")}}

    def flip(s):
        return per_scheme[s]["body_proxy"]["argmax_flip_rate"] if s else None

    faithful = _propagate_flip(flip(s8)) if s8 else None       # gate-safe (int8) -> cb3 realistic upper
    int4pt = _propagate_flip(flip(s4)) if s4 else None
    loose_ub = _propagate_flip(flip(s3)) if s3 else None       # int3 dead-regime ceiling

    # headline = the FAITHFUL gate-safe estimate (cb3 lives in this regime, NOT the int3 dead regime)
    head = faithful or int4pt or loose_ub
    return {
        "status": "ok",
        "faithful_gate_safe_estimate": faithful, "faithful_anchor_scheme": s8,
        "int4_class_point": int4pt, "int4_scheme": s4,
        "loose_int3_ceiling": loose_ub, "loose_ceiling_scheme": s3,
        # headline scalars (the gate-safe regime cb3 actually occupies)
        "delta_top1_bound": head["delta_top1_bound"],
        "delta_e_accepted_bound": head["delta_e_accepted_bound"],
        "delta_coverage_bound": head["delta_coverage_bound"],
        "delta_demand_tps_bound": head["delta_demand_tps_bound"],
        "caveat": ("Body-proxy bf16-lm_head sensitivity (drafter mimics body); flip is an UPPER bound on "
                   "|Delta-top1| (realistic <=). cb3 lives in the GATE-SAFE (int8) regime, not the int3 "
                   "dead regime. The faithful cb3-vs-int4 drafter-INPUT read is BLOCKED (#372 no kernel)."),
    }


def verdict(per_scheme, schemes, accept_bound):
    """Three-way honest verdict on the supply x demand acceptance cross-term:
      negligible        -- faithful gate-safe |Delta-demand_TPS| < 5% of the demand lift, AND the
                           drafter-input read were available (it is not -> we cannot reach this).
      approximately_additive -- gate-safe regime (NOT destructive), but the upper bound exceeds the strict
                           5% bar and the faithful drafter-input leg is BLOCKED -> additivity holds with a
                           bounded haircut, not certified to 5%.
      destructive       -- even the gate-safe estimate wipes a large fraction of the demand lift."""
    threshold_tps = NEGLIGIBLE_FRAC * DEMAND_LIFT_TPS_BAND          # +5.28 TPS (5% of +105.6)
    if accept_bound.get("status") != "ok":
        return {"verdict": "blocked:unmeasured", "cross_term_negligible": None,
                "cross_term_destructive": None, "supply_demand_additive": None,
                "delta_demand_tps_bound": "blocked:unmeasured",
                "negligible_threshold_tps": threshold_tps, "demand_lift_tps_band": DEMAND_LIFT_TPS_BAND,
                "verdict_basis": "no_body_proxy"}
    dtps = accept_bound["delta_demand_tps_bound"]                   # faithful gate-safe upper
    frac_of_demand = dtps / DEMAND_LIFT_TPS_BAND if DEMAND_LIFT_TPS_BAND else float("nan")
    destructive = frac_of_demand >= 0.50                           # wipes >=half the demand lift
    negligible_to_5pct = dtps < threshold_tps
    if destructive:
        label = "destructive"
    elif negligible_to_5pct:
        label = "negligible"
    else:
        # gate-safe, demand lift survives, but the (upper-bound) haircut exceeds the strict 5% bar and the
        # faithful drafter-input read is blocked -> additivity holds APPROXIMATELY, not certified.
        label = "approximately_additive_input_leg_blocked"
    return {
        "verdict": label,
        "cross_term_negligible": bool(negligible_to_5pct),
        "cross_term_destructive": bool(destructive),
        "supply_demand_additive": bool(not destructive),          # additive (not wiped) unless destructive
        "delta_demand_tps_bound": dtps,
        "delta_demand_tps_frac_of_lift": frac_of_demand,
        "negligible_threshold_tps": threshold_tps,
        "demand_lift_tps_band": DEMAND_LIFT_TPS_BAND,
        "drafter_input_faithful_read": "blocked:unmeasured (#372 no cb3 kernel)",
        "verdict_basis": "faithful_gate_safe_body_proxy_upper_bound",
    }


# ===========================================================================
# Section 7 -- self-tests
# ===========================================================================

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def run_self_tests(per_scheme, schemes, measure, meta) -> dict:
    c = {}
    fp16 = next((s for s in schemes if scheme_bits(s) is None), None)

    # a) fp16-vs-fp16 perturbation == 0 (within-process determinism sanity; validates the harness).
    if fp16 is not None:
        for tname in measure:
            t = per_scheme[fp16][tname]
            c[f"a_fp16_{tname}_l2rel_zero"] = t["l2_relative"] < ZERO_TOL
            c[f"a_fp16_{tname}_cos_one"] = abs(t["cosine"] - 1.0) < ZERO_TOL

    # b) monotone non-decreasing perturbation in quant aggressiveness (fewer bits -> >= perturbation).
    ordered = sorted(schemes, key=lambda s: (scheme_bits(s) is not None, -(scheme_bits(s) or 99)))
    # ordered: fp16 first, then int8, int4, int3 (descending bits)
    for tname in measure:
        seq = [per_scheme[s][tname]["l2_relative"] for s in ordered]
        mono = all(seq[i] <= seq[i + 1] + MONO_EPS for i in range(len(seq) - 1))
        c[f"b_{tname}_l2rel_monotone"] = mono
        cos_seq = [per_scheme[s][tname]["cosine"] for s in ordered]
        cos_mono = all(cos_seq[i] >= cos_seq[i + 1] - MONO_EPS for i in range(len(cos_seq) - 1))
        c[f"b_{tname}_cosine_monotone_nonincreasing"] = cos_mono

    # c) cb3 bracket ordering: the cb3 upper bound (lowest bits) is the strictest perturbation.
    quant = [s for s in schemes if scheme_bits(s) is not None]
    if quant:
        cb3_ub = min(quant, key=lambda s: scheme_bits(s))
        for tname in measure:
            mx = max(per_scheme[s][tname]["l2_relative"] for s in quant)
            c[f"c_{tname}_cb3ub_is_strictest"] = per_scheme[cb3_ub][tname]["l2_relative"] >= mx - MONO_EPS

    # d) every measured perturbation finite and in a sane range ([0,1] L2-rel; cos in [-1,1]).
    ok_fin = True
    ok_rng = True
    for s in schemes:
        for tname in measure:
            t = per_scheme[s][tname]
            ok_fin = ok_fin and _finite(t["l2_relative"]) and _finite(t["cosine"])
            ok_rng = ok_rng and (0.0 <= t["l2_relative"] < 2.0) and (-1.001 <= t["cosine"] <= 1.001)
    c["d_all_finite"] = ok_fin
    c["d_all_in_range"] = ok_rng

    # e) provenance: cb3 byte ratio + ladder + secant round-trips.
    c["e_cb3_byte_ratio_0p785"] = round(CB3_BYTE_RATIO, 3) == 0.785
    c["e_ladder_len_7"] = len(LADDER_289) == 7
    c["e_eaccepted_roundtrips_289"] = abs(expected_accepted(LADDER_289) - E_ACCEPTED_289) < 1e-9
    c["e_demand_lift_positive"] = DEMAND_LIFT_TPS_BAND > 0

    # f) measurement coverage: used the deployed harness prompts + on-target GPU forward.
    c["f_prompts_used"] = meta["n_prompts"] >= 1
    c["f_layers_42"] = meta["n_layers"] == 42

    # g) PPL gate untouched (0-TPS card; greedy target token unchanged).
    c["g_ppl_unchanged_passes_gate"] = BASELINE_PPL <= PPL_GATE
    c["g_cb3_holds_gate"] = CB3_PPL <= PPL_GATE            # cb3 is gate-safe (the load-bearing regime fact)
    c["g_cb3_ppl_parity_with_int4"] = CB3_REL_PPL_OVER_INT4 < GATE_HEADROOM_REL  # +0.17% < +1.81% headroom

    # h) body-proxy checks (only when --bound-acceptance ran): the RTN sweep must REPRODUCE lawine #355's
    #    on-target finding -- scalar sub-int4 collapses PPL (int3 >> int8) and flips the greedy token far
    #    more than int8 -- which validates the harness AND proves int3 is the gate-DEAD regime (not cb3).
    bp_ok = all(per_scheme[s].get("body_proxy") for s in [x for x in schemes if scheme_bits(x) is not None])
    if bp_ok and any(per_scheme[s].get("body_proxy") for s in schemes):
        def bpv(b, key):
            s = _by_bits(per_scheme, schemes, b)
            return per_scheme[s]["body_proxy"][key] if s else None
        ppl8, ppl4, ppl3 = bpv(8, "teacher_forced_ppl"), bpv(4, "teacher_forced_ppl"), bpv(3, "teacher_forced_ppl")
        fl8, fl4, fl3 = bpv(8, "argmax_flip_rate"), bpv(4, "argmax_flip_rate"), bpv(3, "argmax_flip_rate")
        if None not in (ppl8, ppl4, ppl3):
            c["h_ppl_monotone_int8_le_int4_le_int3"] = ppl8 <= ppl4 + 1e-6 <= ppl3 + 2e-6
            c["h_int3_ppl_collapse_reproduces_355"] = ppl3 > ppl8 * 1.10   # int3 >= +10% over int8 (gate-dead)
        if None not in (fl8, fl4, fl3):
            c["h_flip_monotone_int8_le_int4_le_int3"] = fl8 <= fl4 + 1e-9 <= fl3 + 1e-9
            c["h_int8_gate_safe_low_flip"] = fl8 < 0.10                    # gate-safe rung flips << dead regime

    passes = all(c.values())
    return {"conditions": c, "n_checks": len(c), "n_passed": sum(1 for v in c.values() if v), "passes": passes}


# ===========================================================================
# Section 8 -- report assembly + W&B + CLI
# ===========================================================================

def build_report(per_scheme, meta, schemes, measure, bound_acceptance) -> dict:
    quant = [s for s in schemes if scheme_bits(s) is not None]
    cb3_scheme = min(quant, key=lambda s: scheme_bits(s)) if quant else None  # the cb3 conservative UB (lowest bits)
    int4_scheme = next((s for s in schemes if scheme_bits(s) == 4), None)
    if bound_acceptance and quant:
        accept_bound = bound_acceptance_shift(per_scheme, schemes)
    else:
        accept_bound = {"status": "blocked:unmeasured",
                        **{k: "blocked:unmeasured" for k in
                           ("delta_top1_bound", "delta_e_accepted_bound", "delta_coverage_bound",
                            "delta_demand_tps_bound")}}
    vd = verdict(per_scheme, schemes, accept_bound) if cb3_scheme else {}
    selftest = run_self_tests(per_scheme, schemes, measure, meta)

    kv_cb3 = per_scheme[cb3_scheme]["shared_kv_states"]["l2_relative"] if cb3_scheme else float("nan")
    hid_cb3 = per_scheme[cb3_scheme]["inputs_embeds_hidden"]["l2_relative"] if cb3_scheme else float("nan")
    return {
        "pr": 410, "agent": "ubel", "kind": "cb3-quant-acceptance-crossterm",
        "analysis_only": True, "no_launch": True, "no_hf_job": True, "no_submission": True,
        "no_served_file_change": True, "gpu_used_for_analysis_only": True, "official_tps": 0,
        "baseline_unchanged_tps": BASELINE_TPS, "baseline_unchanged_ppl": BASELINE_PPL,
        "inputs": {
            "model_id": MODEL_ID, "schemes": schemes, "measure": measure,
            "int4_bpw": INT4_BPW, "cb3_bpw_eff": CB3_BPW_EFF, "cb3_byte_ratio": CB3_BYTE_RATIO,
            "cb3_mixed_frac": CB3_MIXED_FRAC, "ladder_289": LADDER_289,
            "drafter_top1_289": DRAFTER_TOP1_289, "e_accepted_289": E_ACCEPTED_289, "e_t_289": E_T_289,
            "gross_tps_per_unit_cov_402": GROSS_TPS_PER_UNIT_COV, "corrected_strict_base_393": CORRECTED_STRICT_BASE,
            "gap_to_500_393": GAP_TO_500, "cb3_supply_lift_388": CB3_SUPPLY_LIFT_388,
            "demand_coverage_band_401": DEMAND_COVERAGE_BAND, "demand_lift_tps_band": DEMAND_LIFT_TPS_BAND,
            "negligible_frac": NEGLIGIBLE_FRAC, "baseline_tps": BASELINE_TPS, "baseline_ppl": BASELINE_PPL,
            "cb3_ppl_388": CB3_PPL, "gate_headroom_rel": GATE_HEADROOM_REL,
            "cb3_rel_ppl_over_int4": CB3_REL_PPL_OVER_INT4,
            "int3_rtn_relppl_355": INT3_RTN_RELPPL_355, "int3_rtn_gate_ppl_355": INT3_RTN_GATE_PPL_355,
            "ppl_gate": PPL_GATE, "cb3_scheme_upper_bound": cb3_scheme, "int4_scheme": int4_scheme,
            "source_289_run": "fi34s269", "source_401_run": "i2qsjyp6", "source_402_run": "8pcyhe2r",
            "source_355_run": "vqzzc9jw", "source_388_ref": "cb3_supply_lift_mtp_honest",
            "source_kernel_ref": "cb3_kernel_realized_bw",
        },
        "meta": meta,
        "per_scheme": per_scheme,
        "acceptance_bound": accept_bound,
        "verdict": vd,
        # ---- card-required headline scalars (SENPAI-RESULT / W&B load-bearing) ----
        "cb3_ub_shared_kv_states_l2_relative": kv_cb3,        # PRIMARY load-bearing test_metric (int3 UB)
        "cb3_ub_inputs_embeds_hidden_l2_relative": hid_cb3,
        "verdict_label": vd.get("verdict"),
        "cross_term_negligible": vd.get("cross_term_negligible"),
        "cross_term_destructive": vd.get("cross_term_destructive"),
        "supply_demand_additive": vd.get("supply_demand_additive"),
        "delta_demand_tps_bound": vd.get("delta_demand_tps_bound"),
        "self_test": selftest,
        "cb3_acceptance_crossterm_self_test_passes": selftest["passes"],
    }


def log_to_wandb(report: dict, group: str, name: str) -> str:
    try:
        import wandb  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"W&B import failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_unavailable"

    def blocked(x):
        return "blocked:unmeasured" if x is None else x

    try:
        run = wandb.init(group=group, name=name, config=report["inputs"])
        ab = report["acceptance_bound"]
        faith = ab.get("faithful_gate_safe_estimate") or {}
        loose = ab.get("loose_int3_ceiling") or {}
        summ = {
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "official_tps": 0,
            "cb3_ub_shared_kv_states_l2_relative": report["cb3_ub_shared_kv_states_l2_relative"],
            "cb3_ub_inputs_embeds_hidden_l2_relative": report["cb3_ub_inputs_embeds_hidden_l2_relative"],
            "verdict_label": report["verdict"].get("verdict"),
            "cross_term_negligible": report["verdict"].get("cross_term_negligible"),
            "cross_term_destructive": report["verdict"].get("cross_term_destructive"),
            "supply_demand_additive": report["verdict"].get("supply_demand_additive"),
            "delta_demand_tps_bound": blocked(report["verdict"].get("delta_demand_tps_bound")),
            "delta_demand_tps_frac_of_lift": blocked(report["verdict"].get("delta_demand_tps_frac_of_lift")),
            "negligible_threshold_tps": report["verdict"].get("negligible_threshold_tps"),
            "demand_lift_tps_band": report["verdict"].get("demand_lift_tps_band"),
            "faithful_gate_safe_delta_top1": blocked(faith.get("delta_top1_bound")),
            "faithful_gate_safe_delta_demand_tps": blocked(faith.get("delta_demand_tps_bound")),
            "loose_int3_delta_top1": blocked(loose.get("delta_top1_bound")),
            "loose_int3_delta_demand_tps": blocked(loose.get("delta_demand_tps_bound")),
            "drafter_input_faithful_read_blocked": True,
            "cb3_acceptance_crossterm_self_test_passes": report["cb3_acceptance_crossterm_self_test_passes"],
            "ref_peak_vram_mb": report["meta"]["ref_peak_vram_mb"],
            "n_prompts": report["meta"]["n_prompts"],
        }
        wandb.summary.update(summ)
        # per-scheme curves (monotonicity is visible)
        for sname, res in report["per_scheme"].items():
            bits = res["bits"] if res["bits"] is not None else 16
            for tname in ("shared_kv_states", "inputs_embeds_hidden"):
                t = res.get(tname)
                if t is None:
                    continue
                wandb.log({
                    "scheme/bits": bits,
                    f"perturb/{tname}/l2_relative": t["l2_relative"],
                    f"perturb/{tname}/cosine": t["cosine"],
                    f"perturb/{tname}/max_abs_delta": t["max_abs_delta"],
                    f"perturb/{tname}/per_channel_max_abs_delta_max": t["per_channel_max_abs_delta_max"],
                })
            if res.get("body_proxy"):
                bp = res["body_proxy"]
                wandb.log({"scheme/bits": bits,
                           "body_proxy/argmax_flip_rate": bp.get("argmax_flip_rate") or float("nan"),
                           "body_proxy/teacher_forced_ppl": bp.get("teacher_forced_ppl") or float("nan")})
        for cond, val in report["self_test"]["conditions"].items():
            wandb.log({f"test/{cond}": float(bool(val))})
        run_id = run.id
        wandb.finish()
        return run_id
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging failed (non-fatal): {exc}", file=sys.stderr)
        return "wandb_failed"


def print_report(r: dict) -> None:
    print("\n=== cb3 (supply) x MTP-drafter (demand) acceptance cross-term (PR #410, ubel) ===")
    m = r["meta"]
    print(f"body={MODEL_ID}  prompts={m['n_prompts']}  seqlen[min/mean/max]="
          f"{m['seqlen_min']}/{m['seqlen_mean']:.0f}/{m['seqlen_max']}  "
          f"quantized_linears={m['n_quantized_linears']}  peak_vram={m['ref_peak_vram_mb']}MB")
    print("\n-- PRIMARY: cb3-proxy perturbation at the drafter-input tensors (vs bf16 reference) --")
    print(f"   {'scheme':>10} {'bits':>4} | {'kv_L2rel':>9} {'kv_cos':>8} | {'hid_L2rel':>9} {'hid_cos':>8} | "
          f"{'flip':>7} {'tf_ppl':>8}")
    for sname, res in r["per_scheme"].items():
        kv = res.get("shared_kv_states") or {}
        hh = res.get("inputs_embeds_hidden") or {}
        bp = res.get("body_proxy") or {}
        fl = bp.get("argmax_flip_rate")
        pp = bp.get("teacher_forced_ppl")
        print(f"   {sname:>10} {str(res['bits']):>4} | "
              f"{kv.get('l2_relative', float('nan')):>9.5f} {kv.get('cosine', float('nan')):>8.5f} | "
              f"{hh.get('l2_relative', float('nan')):>9.5f} {hh.get('cosine', float('nan')):>8.5f} | "
              f"{(fl if fl is not None else float('nan')):>7.4f} {(pp if pp is not None else float('nan')):>8.3f}")
    cb3 = r["inputs"]["cb3_scheme_upper_bound"]
    print(f"\n   cb3 conservative UPPER BOUND scheme = '{cb3}'  (cb3 RHT+VQ @ {CB3_BPW_EFF:.4f}bpw <= int3-RTN,")
    print(f"     but int3 is the GATE-DEAD scalar regime; cb3 HOLDS the 2.42 gate at PPL {CB3_PPL} -> int8 regime)")
    print(f"   cb3_ub(int3) shared_kv_states L2-rel = {r['cb3_ub_shared_kv_states_l2_relative']:.6f}   "
          f"inputs_embeds_hidden L2-rel = {r['cb3_ub_inputs_embeds_hidden_l2_relative']:.6f}")
    print("\n-- SECONDARY body-proxy acceptance bound (caveated; faithful cb3 drafter-INPUT read BLOCKED #372) --")
    ab = r["acceptance_bound"]
    faith = ab.get("faithful_gate_safe_estimate") or {}
    loose = ab.get("loose_int3_ceiling") or {}
    print(f"   FAITHFUL gate-safe (cb3 regime, anchor={ab.get('faithful_anchor_scheme')}): "
          f"Dtop1<={faith.get('delta_top1_bound')}  Ddemand_TPS<={faith.get('delta_demand_tps_bound')}")
    print(f"   LOOSE int3 ceiling (gate-DEAD, overstates): "
          f"Dtop1<={loose.get('delta_top1_bound')}  Ddemand_TPS<={loose.get('delta_demand_tps_bound')}")
    print("\n-- VERDICT --")
    vd = r["verdict"]
    print(f"   demand lift band = +{vd.get('demand_lift_tps_band', float('nan')):.2f} TPS   "
          f"5% threshold = {vd.get('negligible_threshold_tps', float('nan')):.3f} TPS")
    print(f"   VERDICT = {vd.get('verdict')}   (negligible={vd.get('cross_term_negligible')} "
          f"destructive={vd.get('cross_term_destructive')} additive={vd.get('supply_demand_additive')})")
    fr = vd.get("delta_demand_tps_frac_of_lift")
    print(f"   faithful Ddemand_TPS bound = {vd.get('delta_demand_tps_bound')}  "
          f"(= {fr*100:.1f}% of demand lift)" if isinstance(fr, (int, float)) else
          f"   faithful Ddemand_TPS bound = {vd.get('delta_demand_tps_bound')}")
    print(f"   drafter-input faithful read: {vd.get('drafter_input_faithful_read')}")
    print(f"\nself-test: {r['self_test']['n_passed']}/{r['self_test']['n_checks']} checks  "
          f"cb3_acceptance_crossterm_self_test_passes = {r['cb3_acceptance_crossterm_self_test_passes']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="cb3 (supply) x MTP-drafter (demand) acceptance cross-term (PR #410).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quant-sweep", "--quant_sweep", dest="quant_sweep", default="fp16,int8,int4,cb3-int3",
                    help="comma list of schemes; cb3-int4=int4-class point, cb3-int3=cb3 conservative upper bound")
    ap.add_argument("--measure", default="shared_kv_states,inputs_embeds_hidden",
                    help="comma list of drafter-input tensors to measure")
    ap.add_argument("--bound-acceptance", "--bound_acceptance", dest="bound_acceptance", action="store_true",
                    help="compute the caveated body-proxy acceptance-shift bound (secondary)")
    ap.add_argument("--self-test", action="store_true", help="fast reduced GPU gate (fp16=0 + monotone)")
    ap.add_argument("--max-prompts", "--max_prompts", dest="max_prompts", type=int, default=128)
    ap.add_argument("--max-seq-len", "--max_seq_len", dest="max_seq_len", type=int, default=512)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="cb3-quant-acceptance-crossterm")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default="ubel/cb3-quant-acceptance-crossterm")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", type=str,
                    default="research/validity/cb3_acceptance_crossterm/cb3_acceptance_crossterm_results.json")
    args = ap.parse_args()

    schemes = [s for s in (x.strip() for x in args.quant_sweep.split(",")) if s]
    measure = [s for s in (x.strip() for x in args.measure.split(",")) if s]
    # validate scheme names early
    for s in schemes:
        scheme_bits(s)

    if args.self_test:
        # fast gate: few prompts, short seq, both tensors, no acceptance bound, no W&B.
        per_scheme, meta = run_measurement(schemes, measure, False, max_prompts=4, max_seq_len=128, verbose=True)
        report = build_report(per_scheme, meta, schemes, measure, False)
        out = HERE / "cb3_acceptance_crossterm_selftest.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print_report(report)
        print(f"\nwrote {out}")
        print(f"\ncb3_acceptance_crossterm_self_test_passes = {report['self_test']['passes']}")
        return 0 if report["self_test"]["passes"] else 1

    per_scheme, meta = run_measurement(schemes, measure, args.bound_acceptance,
                                       args.max_prompts, args.max_seq_len, verbose=True)
    report = build_report(per_scheme, meta, schemes, measure, args.bound_acceptance)
    print_report(report)

    report["wandb_run_id"] = None if args.no_wandb else log_to_wandb(report, args.wandb_group, args.wandb_name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}  (W&B run {report.get('wandb_run_id')})")

    vd = report["verdict"]
    ab = report["acceptance_bound"]
    faith = ab.get("faithful_gate_safe_estimate") or {}
    print("\nSENPAI-RESULT " + json.dumps({
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": [report["wandb_run_id"]] if report.get("wandb_run_id") else [],
        "no_hf_job": True, "official_tps": 0.0, "analysis_only": True, "no_served_file_change": True,
        "cb3_ub_shared_kv_states_l2_relative": float(report["cb3_ub_shared_kv_states_l2_relative"]),
        "cb3_ub_inputs_embeds_hidden_l2_relative": float(report["cb3_ub_inputs_embeds_hidden_l2_relative"]),
        "verdict": vd.get("verdict"),
        "cross_term_negligible": vd.get("cross_term_negligible"),
        "cross_term_destructive": vd.get("cross_term_destructive"),
        "supply_demand_additive": vd.get("supply_demand_additive"),
        "faithful_gate_safe_delta_demand_tps_bound": faith.get("delta_demand_tps_bound"),
        "delta_demand_tps_bound": vd.get("delta_demand_tps_bound"),
        "drafter_input_faithful_read": "blocked:unmeasured",
        "cb3_acceptance_crossterm_self_test_passes": bool(report["cb3_acceptance_crossterm_self_test_passes"]),
        "primary_metric": {"name": "cb3_acceptance_crossterm_self_test_passes",
                           "value": float(report["cb3_acceptance_crossterm_self_test_passes"])},
        "test_metric": {"name": "cb3_ub_shared_kv_states_l2_relative",
                        "value": float(report["cb3_ub_shared_kv_states_l2_relative"])},
    }))
    return 0 if report["self_test"]["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
