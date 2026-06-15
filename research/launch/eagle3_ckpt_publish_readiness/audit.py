# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Static publish + vLLM-load readiness audit for the {2,21,39} EAGLE-3 head (PR #328).

0-GPU, 0-network, pure-stdlib analytic audit. It does NOT load any checkpoint, does
NOT touch a GPU, does NOT publish or launch anything. It answers the single #319
Option-A gating precondition (ubel #322 §0): *is the in-repo `gua9x68j` / `56ksyxgw`
EAGLE-3 head (fern #34) publish + vLLM-load ready, or does it need an adapter?*

Method — cross-check two committed, authoritative sources (no other-branch artifacts):
  1. SOURCE  = the exact `head.state_dict()` produced by
     `scripts/drafter/train_eagle3.py` (`Eagle3DraftHead`), the script that trained
     `56ksyxgw`. Tensor names/shapes/dtypes are derived analytically from that code.
  2. TARGET  = vLLM 0.22.1rc1's `Eagle3LlamaForCausalLM.load_weights` /
     `LlamaModel.load_weights` name+shape contract
     (`vllm/model_executor/models/llama_eagle3.py`), the load path the served
     `submissions/fa2sw_precache_kenyan` wheel pin actually runs.

We faithfully PORT vLLM's weight-name transformation (the `model.`-prepend, the
`midlayer.`->`layers.0.` remap, and the q/k/v->qkv_proj + gate/up->gate_up_proj
stacked fusion) and run it over (a) the raw saved keys and (b) a proposed published
key set, verifying every tensor lands on a real vLLM parameter/shard with a matching
shape.

Run (analytic, 0-GPU):
    cd target/ && python research/launch/eagle3_ckpt_publish_readiness/audit.py

Outputs `_results.json` next to this file and prints a readiness verdict +
`ckpt_publish_readiness_blocking_issues`. Optional W&B logging
(`--wandb_group eagle3-ckpt-publish-readiness`, `--no_wandb` to skip).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Analytic facts — Gemma-4-E4B EAGLE-3 draft head dims.
# Source: scripts/drafter/train_eagle3.py module constants (HID, VOCAB, ...).
# --------------------------------------------------------------------------- #
HID = 2560        # draft hidden size
VOCAB = 262144    # full Gemma vocab (identity draft<->target, no d2t/t2d)
N_AUX = 3         # aux layers {2, 21, 39}
HEAD_DIM = 256
N_HEADS = 8
N_KV = 2
INTER = 10240

Q = N_HEADS * HEAD_DIM   # 2048  q-proj rows
KV = N_KV * HEAD_DIM     # 512   k/v-proj rows
QKV = (N_HEADS + 2 * N_KV) * HEAD_DIM  # 3072 fused qkv rows
GU = 2 * INTER           # 20480 fused gate_up rows
TWO_H = 2 * HID          # 5120  layer-0 qkv input (embeds ++ hidden)
FUSED_IN = N_AUX * HID   # 7680  fc input / input_norm dim

BF16, FP32 = "bfloat16", "float32"

# --------------------------------------------------------------------------- #
# SOURCE: exactly what `Eagle3DraftHead.state_dict()` writes (train_eagle3.py
# save_checkpoint / torch.save(head.state_dict(), ...)). 15 tensors.
#   dtype: embed_tokens + lm_head are explicitly `.to(bfloat16)` (lines 586-587,
#   597-598); EVERY other tensor is an nn.Parameter left at torch default float32
#   (`.to(device)` does not change dtype; autocast does not change stored dtype).
#   -> the saved file is MIXED dtype: 2 bf16 tables + 13 fp32 body tensors.
# (name, [shape], dtype)
# --------------------------------------------------------------------------- #
SOURCE_STATE_DICT = [
    ("model.embed_tokens.weight", [VOCAB, HID], BF16),
    ("model.input_norm.weight", [FUSED_IN], FP32),
    ("model.fc.weight", [HID, FUSED_IN], FP32),
    ("model.norm.weight", [HID], FP32),
    ("model.layers.0.self_attn.q_proj.weight", [Q, TWO_H], FP32),
    ("model.layers.0.self_attn.k_proj.weight", [KV, TWO_H], FP32),
    ("model.layers.0.self_attn.v_proj.weight", [KV, TWO_H], FP32),
    ("model.layers.0.self_attn.o_proj.weight", [HID, Q], FP32),
    ("model.layers.0.mlp.gate_proj.weight", [INTER, HID], FP32),
    ("model.layers.0.mlp.up_proj.weight", [INTER, HID], FP32),
    ("model.layers.0.mlp.down_proj.weight", [HID, INTER], FP32),
    ("model.layers.0.input_layernorm.weight", [HID], FP32),
    ("model.layers.0.hidden_norm.weight", [HID], FP32),
    ("model.layers.0.post_attention_layernorm.weight", [HID], FP32),
    ("lm_head.weight", [VOCAB, HID], BF16),
]

# --------------------------------------------------------------------------- #
# TARGET: vLLM `Eagle3LlamaForCausalLM` loadable parameters (named_parameters),
# AFTER the q/k/v->qkv_proj and gate/up->gate_up_proj fusion. Shapes are the vLLM
# Linear/Embedding/RMSNorm weights. (llama_eagle3.py LlamaModel.__init__ +
# Eagle3LlamaForCausalLM.__init__.) `draft_id_to_target_id` is a buffer that
# DEFAULTS to identity zeros and need not be present in the checkpoint.
# --------------------------------------------------------------------------- #
VLLM_PARAMS = {
    "model.embed_tokens.weight": [VOCAB, HID],
    "model.input_norm.weight": [FUSED_IN],
    "model.fc.weight": [HID, FUSED_IN],
    "model.norm.weight": [HID],
    "model.layers.0.self_attn.qkv_proj.weight": [QKV, TWO_H],
    "model.layers.0.self_attn.o_proj.weight": [HID, Q],
    "model.layers.0.mlp.gate_up_proj.weight": [GU, HID],
    "model.layers.0.mlp.down_proj.weight": [HID, INTER],
    "model.layers.0.input_layernorm.weight": [HID],
    "model.layers.0.hidden_norm.weight": [HID],
    "model.layers.0.post_attention_layernorm.weight": [HID],
    "lm_head.weight": [VOCAB, HID],
}
# Shard decomposition of the fused params (shard_id -> expected shard shape).
VLLM_SHARDS = {
    "model.layers.0.self_attn.qkv_proj.weight": {
        "q": [Q, TWO_H], "k": [KV, TWO_H], "v": [KV, TWO_H],
    },
    "model.layers.0.mlp.gate_up_proj.weight": {0: [INTER, HID], 1: [INTER, HID]},
}

# vLLM stacked_params_mapping (llama_eagle3.py LlamaModel.load_weights:257-264).
STACKED = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]


def vllm_remap(name: str):
    """Faithful port of vLLM 0.22.1rc1 weight-name resolution for a draft ckpt key.

    Replicates `Eagle3LlamaForCausalLM.load_weights` (llama_eagle3.py:400-451) then
    the `model.*` subtree handling in `LlamaModel.load_weights` (:256-288). Returns
    (internal_param_name, shard_id) where shard_id is None for a direct (non-fused)
    param, or the sentinel ("<skip>", reason) for a dropped key.
    """
    # ---- Eagle3LlamaForCausalLM.load_weights name stage ----
    if "t2d" in name:
        return ("<skip>", "t2d dropped")
    if "d2t" in name:
        name = name.replace("d2t", "draft_id_to_target_id")
    elif "mask_hidden" in name:
        return ("<skip>", "mask_hidden (parallel-draft only)")
    elif "lm_head" not in name:
        name = "model." + name  # <-- unconditional model. prepend

    # ---- AutoWeightsLoader dispatch + LlamaModel.load_weights for model.* ----
    if name.startswith("model.") and name != "model.draft_id_to_target_id":
        sub = name[len("model."):]
        if "midlayer." in sub:
            sub = sub.replace("midlayer.", "layers.0.")
        for param_name, weight_name, shard_id in STACKED:
            if weight_name in sub:
                sub = sub.replace(weight_name, param_name)
                return ("model." + sub, shard_id)
        return ("model." + sub, None)
    return (name, None)


def published_name(src: str) -> str:
    """The rename a vLLM-loadable converter must apply to a saved key.
    Strip the leading `model.` from body keys; rename the single decoder layer
    `layers.0.` -> the canonical EAGLE-3 `midlayer.`; keep `lm_head.*` as-is.
    """
    if src.startswith("lm_head."):
        return src
    body = src[len("model."):] if src.startswith("model.") else src
    if body.startswith("layers.0."):
        body = "midlayer." + body[len("layers.0."):]
    return body


def check_set(tensors):
    """Run each (name, shape, dtype) through vLLM's resolver; record where it lands
    and whether the shape matches the target param/shard. Returns per-tensor rows +
    the set of (param, shard) load-targets that got filled."""
    rows = []
    filled = set()
    for name, shape, dtype in tensors:
        internal, shard = vllm_remap(name)
        if internal == "<skip>":
            rows.append({"src": name, "internal": None, "shard": None,
                         "status": "skipped", "detail": shard})
            continue
        if shard is not None and internal in VLLM_SHARDS:
            want = VLLM_SHARDS[internal].get(shard)
            ok = want == shape
            rows.append({"src": name, "internal": internal, "shard": shard,
                         "src_shape": shape, "want_shape": want,
                         "status": "ok" if ok else "shape_mismatch"})
            if ok:
                filled.add((internal, shard))
        elif internal in VLLM_PARAMS:
            want = VLLM_PARAMS[internal]
            ok = want == shape
            rows.append({"src": name, "internal": internal, "shard": None,
                         "src_shape": shape, "want_shape": want,
                         "status": "ok" if ok else "shape_mismatch"})
            if ok:
                filled.add((internal, None))
        else:
            rows.append({"src": name, "internal": internal, "shard": shard,
                         "src_shape": shape, "want_shape": None,
                         "status": "unexpected"})
    return rows, filled


def required_targets():
    """Every (param, shard) load-target a complete EAGLE-3 head must fill."""
    req = set()
    for p in VLLM_PARAMS:
        if p in VLLM_SHARDS:
            for s in VLLM_SHARDS[p]:
                req.add((p, s))
        else:
            req.add((p, None))
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="path for _results.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "senpai-v1"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="eagle3-ckpt-publish-readiness")
    ap.add_argument("--wandb_name", default="ubel/eagle3-ckpt-publish-readiness")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(here, "_results.json")

    req = required_targets()
    raw_rows, raw_filled = check_set(SOURCE_STATE_DICT)
    pub_tensors = [(published_name(n), s, d) for n, s, d in SOURCE_STATE_DICT]
    pub_rows, pub_filled = check_set(pub_tensors)

    raw_ok = sum(1 for r in raw_rows if r["status"] == "ok")
    raw_bad = [r for r in raw_rows if r["status"] not in ("ok", "skipped")]
    pub_ok = sum(1 for r in pub_rows if r["status"] == "ok")
    pub_bad = [r for r in pub_rows if r["status"] not in ("ok", "skipped")]
    pub_missing = sorted(f"{p}::{s}" for (p, s) in (req - pub_filled))
    pub_complete = (pub_filled == req) and not pub_bad

    # ---- the blocking issues (must-fix before the one #319 launch can load it) ---
    blocking = [
        {
            "id": "no_published_path",
            "severity": "blocking",
            "kind": "publish",
            "summary": "Artifact is a LOCAL-ONLY pickled .pt in a gitignored workdir "
                       "(research/eagle3_drafter/checkpoints/, .gitignore); the HF "
                       "a10g runner cannot pull a /senpai-run|/workspace local path.",
            "fix": "Export to a single model.safetensors + config.json and publish to "
                   "DRAFTER_BUCKET (scratch bucket) or a private Hub model repo; set "
                   "DRAFTER_SHA256. serve.py:720 ensure_drafter() syncs that bucket.",
            "deterministic": True,
        },
        {
            "id": "weight_key_namespace_mismatch",
            "severity": "blocking",
            "kind": "load",
            "summary": "head.state_dict() emits model.-prefixed body keys + "
                       "model.layers.0.* ; vLLM's loader UNCONDITIONALLY prepends "
                       "model. (llama_eagle3.py:423-424) and remaps midlayer.->"
                       "layers.0. , so the raw keys double-prefix (model.model.*) and "
                       "fail. AutoWeightsLoader RAISES on unexpected/missing tensors.",
            "fix": "Deterministic, lossless key rename: strip leading 'model.' from "
                   "body keys (fc/embed_tokens/norm/input_norm), rename 'layers.0.'"
                   "->'midlayer.', keep 'lm_head.*'. q/k/v + gate/up stay SEPARATE "
                   "(vLLM fuses them via stacked_params_mapping).",
            "deterministic": True,
        },
        {
            "id": "container_format",
            "severity": "blocking",
            "kind": "publish",
            "summary": "Checkpoint is a torch-pickled state_dict (model_best.pt). "
                       "serve.py:745 + vLLM require a file literally named "
                       "model.safetensors (+ config.json) in the drafter dir; a .pt "
                       "is neither HF-discoverable nor sha256-checked by ensure_drafter.",
            "fix": "Write the renamed tensors to model.safetensors (recommend uniform "
                   "bf16 = serving dtype); co-locate the vLLM EAGLE-3 config.json.",
            "deterministic": True,
        },
    ]
    blocking_count = len(blocking)

    # ---- verify-at-smoke caveats (NOT counted as blocking; all shapes already match) #
    caveats = [
        "HF AutoConfig(model_type=llama) must retain the custom fields "
        "(norm_before_fc, target_hidden_size, num_aux_hidden_states, "
        "eagle_aux_hidden_state_layer_ids). Recommend ALSO nesting them under an "
        "`eagle_config` dict (vLLM reads eagle_config first). Confirm 0 missing/"
        "unexpected at smoke.",
        "Body tensors are fp32 in the .pt; vLLM serves bf16 (params_dtype=model "
        "dtype) and default_weight_loader casts on copy_. A uniform-bf16 export is "
        "cleaner; greedy token-identity must be confirmed at smoke regardless.",
        "draft_id_to_target_id/d2t is ABSENT -> vLLM defaults it to identity zeros; "
        "compute_logits then scatters full-vocab logits 1:1 (correct). Not a blocker.",
        "vLLM version: load path read from 0.22.1rc1.dev307 (the exact wheel pinned "
        "in submissions/fa2sw_precache_kenyan/manifest.json dependencies). The "
        "vllm_baseline manifest's vllm==0.22.0 pin is a DIFFERENT submission.",
    ]

    # All shapes/dtypes/semantics align; the only gaps are mechanical packaging /
    # naming, each deterministically fixable with no retrain -> YELLOW (loadable with
    # a documented shim), not RED (irreducible mismatch), not GREEN (loadable as-is).
    loadable_as_is = (raw_ok == len(SOURCE_STATE_DICT)) and not pub_missing
    loadable_with_shim = pub_complete
    all_blocking_deterministic = all(b["deterministic"] for b in blocking)
    if loadable_as_is:
        verdict = "GREEN"
    elif loadable_with_shim and all_blocking_deterministic:
        verdict = "YELLOW"
    else:
        verdict = "RED"

    # ---- self-tests: the audit's own internal consistency ----
    self_tests = {
        "source_has_15_tensors": len(SOURCE_STATE_DICT) == 15,
        "raw_only_lm_head_maps": raw_ok == 1
        and all(r["src"] == "lm_head.weight" for r in raw_rows if r["status"] == "ok"),
        "raw_has_14_unexpected": len(raw_bad) == 14
        and all(r["status"] == "unexpected" for r in raw_bad),
        "published_maps_all_15": pub_ok == 15,
        "published_no_unexpected": len(pub_bad) == 0,
        "published_covers_all_targets": pub_complete and not pub_missing,
        "published_no_shape_mismatch":
            all(r["status"] != "shape_mismatch" for r in pub_rows),
        "verdict_is_yellow": verdict == "YELLOW",
        "blocking_all_deterministic": all_blocking_deterministic,
    }
    self_test_passes = 1 if all(self_tests.values()) else 0

    results = {
        "card": "eagle3_ckpt_publish_readiness",
        "pr": 328,
        "author": "ubel",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "head": {
            "name": "gua9x68j (eval) / 56ksyxgw (train)",
            "provenance": "fern #34 benchmark-matched reasoning head; warm-started "
                          "from #25 full_20k/model_best.pt",
            "aux_layers": [2, 21, 39],
            "source_save_code": "scripts/drafter/train_eagle3.py (Eagle3DraftHead)",
            "local_only_path_pattern":
                "research/eagle3_drafter/checkpoints/<run>/model_best.pt (gitignored)",
            "format": "torch-pickled state_dict (.pt), mixed dtype "
                      "(embed/lm_head bf16, 13 body tensors fp32)",
            "n_tensors": len(SOURCE_STATE_DICT),
        },
        "vllm": {
            "version": "0.22.1rc1.dev307+g3e8afdf78",
            "load_path": "vllm/model_executor/models/llama_eagle3.py "
                         "Eagle3LlamaForCausalLM.load_weights + LlamaModel.load_weights",
            "n_loadable_params": len(VLLM_PARAMS),
        },
        "verdict": verdict,
        "ckpt_publish_readiness_blocking_issues": blocking_count,
        "ckpt_publish_readiness_self_test_passes": self_test_passes,
        "loadable_as_is": loadable_as_is,
        "loadable_with_documented_shim": loadable_with_shim,
        "raw_state_dict_check": {
            "tensors_mapped_ok": raw_ok, "tensors_failed": len(raw_bad),
            "rows": raw_rows,
        },
        "published_check": {
            "tensors_mapped_ok": pub_ok, "tensors_failed": len(pub_bad),
            "covers_all_load_targets": pub_complete, "missing_targets": pub_missing,
            "rename_rule": "strip leading 'model.' ; 'layers.0.'->'midlayer.' ; "
                           "'lm_head.*' unchanged ; q/k/v + gate/up kept separate",
            "rows": pub_rows,
        },
        "blocking_issues": blocking,
        "verify_at_smoke_caveats": caveats,
        "self_tests": self_tests,
    }

    wandb_run_ids: list[str] = []

    # ---- human-readable summary ----
    bar = "=" * 72
    print(bar)
    print("EAGLE-3 {2,21,39} head — publish + vLLM-load readiness audit (PR #328)")
    print(bar)
    print(f"head: {results['head']['name']}  ({results['head']['n_tensors']} tensors, "
          f"{results['head']['format'].split(',')[0]})")
    print(f"vLLM: {results['vllm']['version']}  ({len(VLLM_PARAMS)} loadable params)")
    print("-" * 72)
    print(f"RAW state_dict() load:   {raw_ok:2d}/15 map  -> NOT loadable as-is "
          f"({len(raw_bad)} double-prefixed / unexpected)")
    print(f"PUBLISHED (renamed) load: {pub_ok:2d}/15 map, all "
          f"{len(req)} vLLM load-targets covered  -> loadable via shim")
    print("-" * 72)
    print(f"VERDICT: {verdict}   blocking_issues={blocking_count}   "
          f"self_test_passes={self_test_passes}")
    for b in blocking:
        print(f"  [{b['kind']:>7}] {b['id']}: {b['summary'].splitlines()[0]}")
    print(bar)

    if not args.no_wandb:
        try:
            import wandb

            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             group=args.wandb_group, name=args.wandb_name,
                             config={"pr": 328, "verdict": verdict,
                                     "vllm_version": results["vllm"]["version"]})
            wandb.log({
                "ckpt_publish_readiness_self_test_passes": self_test_passes,
                "ckpt_publish_readiness_blocking_issues": blocking_count,
                "raw_tensors_mapped_ok": raw_ok,
                "published_tensors_mapped_ok": pub_ok,
                "verdict_code": {"GREEN": 2, "YELLOW": 1, "RED": 0}[verdict],
            })
            run.summary.update({
                "ckpt_publish_readiness_self_test_passes": self_test_passes,
                "ckpt_publish_readiness_blocking_issues": blocking_count,
                "verdict": verdict,
            })
            wandb_run_ids.append(run.id)
            print(f"[audit] wandb run: {run.url}  (id={run.id})")
            run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[audit] wandb disabled ({e!r})")

    results["wandb_run_ids"] = wandb_run_ids
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[audit] wrote {out_path}")

    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "wandb_run_ids": wandb_run_ids, "primary_metric": {
            "name": "ckpt_publish_readiness_self_test_passes",
            "value": self_test_passes},
        "test_metric": {"name": "ckpt_publish_readiness_blocking_issues",
                        "value": blocking_count},
    }
    print("SENPAI-RESULT: " + json.dumps(marker))
    return 0 if self_test_passes == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
