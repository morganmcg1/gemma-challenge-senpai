#!/usr/bin/env python
"""PR #713 — g32-on-locus AIME necessity driver (one cell per invocation).

Mirrors fern's #659 mixed-precision harness VERBATIM in protocol (banked
prompt_token_ids, evalsets.score_item, idempotent per-item resume, conc1 BI=1
gb6144 greedy AR M=1, gen_config sampled for Phase-2) but swaps the *body source*:
instead of an on-disk built mixed checkpoint, every cell serves the SAME read-only
bf16 qat-unquantized master and bakes the int4 rounding IN-MEMORY at serve time
(serve_fakequant.py). A "cell" is therefore defined purely by --g32-layers:

    ""        -> N=0 anchor: all body modules g128 (== the operative int4 body)
    "14-27"   -> g32 on the proven recovery locus (mid-third 14 layers)
    "14,15,.."-> a sub-locus (ubel #700 top-energy layers)

So the bf16 GEMM on these fake-quantized weights carries the identical per-group
int4 rounding error of a real serve -> AIME *quality* is a faithful proxy (speed is
priced separately from ubel #700's byte-law; the bf16 substrate is NOT a speed proxy).

ANALYSIS-ONLY. Local A10G. NO HF Job, NO submission, NO served-file change.
analysis_only=True, official_tps=0, no_hf_job=1, fires=0. W&B group
aime-g32-locus-necessity-fern, pr=713.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.local_validation import harness, paths  # noqa: E402

MAT = ROOT / "research" / "validity" / "optionb_319_answer_materiality"
sys.path.insert(0, str(MAT))
import evalsets  # noqa: E402

_DO = evalsets._DO

RES = HERE / "results"
BANK = MAT / "results"

# bf16 qat-unquantized master (read-only, peer cache — public Google weights; the
# int4-body source). The local qat_unq copy + #659 disk headroom are gone, so we read
# this in place and fake-quant in memory (no download, no write).
MASTER_SRC = Path(
    "/senpai-run/home/student-lawine/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-q4_0-unquantized/snapshots/"
    "dfc5b925ddb1d41aaf1fe9679abdcfb0805e1aa6"
)
# The peer snapshot pulled only text-gen files and is missing processor_config.json,
# so vLLM (Gemma4ForConditionalGeneration is multimodal) crashes building the HF
# processor (auto-builds a feature extractor from a non-existent preprocessor_config).
# The int4 body + #659 mixed builds serve fine because they ship a self-contained
# processor_config.json (inline feature/image/video processors). We reconstruct a
# servable bf16 "view": SYMLINK the master weights + configs (no 15.88GB copy, no peer
# write) and add that one processor_config.json. quant_config stays None -> plain bf16.
MASTER_VIEW = Path("/workspace/gemma_build/fqg32_master_view")
INT4_PROCESSOR_CFG = Path("/workspace/gemma_build/int4_g128_lmhead/processor_config.json")
MASTER = MASTER_VIEW  # the served path (built by materialize_master_view)


# vLLM's Gemma4 attention creates an (unused) k_norm for EVERY layer (gemma4.py:429),
# but for the top num_kv_shared_layers it never applies it in forward (gemma4.py:522,
# `if not self.is_kv_shared_layer`). The official qat-unquantized checkpoint therefore
# (correctly) OMITS self_attn.k_norm.weight for those KV-shared layers, but vLLM's strict
# track_weights_loading then raises "weights were not initialized from checkpoint". We
# supply those k_norm as RMSNorm identity (ones[head_dim], bf16) via a tiny supplement
# shard + a weight index, so the strict loader is satisfied WITHOUT copying the 15.88GB
# blob. The tensors are UNUSED in forward (output-neutral) and 1-D (never fake-quanted).
KNORM_SUPP_FILE = "model-knorm-shared.safetensors"
KNORM_KEY = "model.language_model.layers.{L}.self_attn.k_norm.weight"
QNORM_KEY = "model.language_model.layers.{L}.self_attn.q_norm.weight"


def _read_st_header(path: Path) -> dict:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n).decode())


def _write_bf16_ones_safetensors(path: Path, items: list[tuple[str, int]]) -> None:
    """Minimal safetensors writer (stdlib only): each (name, length) -> bf16 ones [length]."""
    one = b"\x80\x3f"  # bf16 1.0, little-endian (fp32 0x3F800000 -> top16 0x3F80)
    header: dict = {}
    off = 0
    for nm, length in items:
        nbytes = length * 2
        header[nm] = {"dtype": "BF16", "shape": [length], "data_offsets": [off, off + nbytes]}
        off += nbytes
    hdr_bytes = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr_bytes)))
        fh.write(hdr_bytes)
        for nm, length in items:
            fh.write(one * length)


def materialize_master_view() -> None:
    """Reconstruct the servable bf16 view: symlink the peer master's weights+configs, add
    the self-contained processor_config.json the snapshot lacks, and supply the KV-shared
    k_norm vLLM demands (see KNORM note) via a supplement shard + weight index. Idempotent."""
    MASTER_VIEW.mkdir(parents=True, exist_ok=True)
    for f in ("model.safetensors", "config.json", "generation_config.json",
              "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
        dst = MASTER_VIEW / f
        src = (MASTER_SRC / f).resolve()  # follow the hub blob symlink
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    shutil.copy2(INT4_PROCESSOR_CFG, MASTER_VIEW / "processor_config.json")

    cfg = json.loads((MASTER_VIEW / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    n_layers = int(tc["num_hidden_layers"])
    n_shared = int(tc.get("num_kv_shared_layers", 0))
    shared = list(range(n_layers - n_shared, n_layers)) if n_shared > 0 else []

    # k_norm dim is per-layer-type: gemma4.py creates q_norm/k_norm as RMSNorm(head_dim)
    # where head_dim varies (full_attention layers -> 512, sliding -> 256). The shared
    # layers' k_norm is unused in forward but its SHAPE must match the param, so mirror
    # each shared layer's q_norm length (q_norm ships for all layers in the base blob).
    base_hdr = _read_st_header(MASTER_VIEW / "model.safetensors")
    items = [(KNORM_KEY.format(L=L), int(base_hdr[QNORM_KEY.format(L=L)]["shape"][0]))
             for L in shared]
    _write_bf16_ones_safetensors(MASTER_VIEW / KNORM_SUPP_FILE, items)

    weight_map: dict[str, str] = {}
    total = 0
    for k, v in base_hdr.items():
        if k == "__metadata__":
            continue
        weight_map[k] = "model.safetensors"
        s, e = v["data_offsets"]
        total += e - s
    for nm, length in items:
        weight_map[nm] = KNORM_SUPP_FILE
        total += length * 2
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (MASTER_VIEW / "model.safetensors.index.json").write_text(json.dumps(index))
    dims = sorted({length for _, length in items})
    print(f"[master-view] supplemented {len(items)} kv-shared k_norm "
          f"(layers {shared[:2]}..{shared[-2:] if shared else []}, dims {dims}) + index "
          f"({len(weight_map)} tensors)", flush=True)

# Serve from a stable copy OUTSIDE the tracked git tree (the entrypoint git-checkouts
# the advisor branch every ~10 min, unlinking a tracked dir held as a live server CWD
# -> os.getcwd() crash mid-run). Materialized from the research dir at startup.
SUBMISSION = Path("/workspace/gemma_build/sub_int4_fakequant_g32")
SERVER_PY = Path("/tmp/senpai-venvs/20f658587e8a6643/bin/python")

# Phase-2 sampled params = generation_config.json (lewtun #31): same sampler the bf16
# AIME endpoint was measured under, so a sampled cell is apples-to-apples.
SAMPLED_PARAMS = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

PORT = 8000
MAX_MODEL_LEN = 8192   # gb6144 = (--max-model-len 8192, max_tokens 6144)
MAX_TOKENS = 6144
MIN_TOKENS = 8         # #541 first-token-EOS guard
CONTEXT_MARGIN = 8
REQUEST_TIMEOUT_S = 1200
SOFT_CAP_MIN_DEFAULT = 82.0

# AIME ladder endpoints (greedy maj@1). int4=denken #637 banked AR conc1 (== my conc);
# bf16=ubel #628. 0.420 = 0.9*bf16 gate the g32-on-locus cell must clear.
ENDPOINTS = {
    "aime": {"bf16": 0.4667, "bf16_run": "zoszxnb0", "int4": 0.4000, "bar90": 0.4200, "n": 60},
    "gpqa": {"bf16": 0.4899, "bf16_run": "g3cig1xo", "int4": 0.4798, "bar90": 0.4409, "n": 198},
}


def materialize_submission() -> None:
    """Copy the canonical serve script + manifest + fakequant module + the
    spawn-safe sitecustomize injector into the out-of-tree serve dir so
    LocalServer's CWD survives the entrypoint git churn. sitecustomize.py is what
    the spawned engine-core child auto-imports (via PYTHONPATH) to fake-quant."""
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "serve_fakequant.py", SUBMISSION / "serve.py")
    shutil.copy2(HERE / "manifest.json", SUBMISSION / "manifest.json")
    shutil.copy2(HERE / "fakequant.py", SUBMISSION / "fakequant.py")
    shutil.copy2(HERE / "sitecustomize.py", SUBMISSION / "sitecustomize.py")


def _env(g32_layers: str, g32: int, g128: int, quant_head: bool,
         enforce_eager: bool = True, max_num_seqs: int = 1,
         base: bool = False) -> dict[str, str]:
    """extra_env: bf16 master + fake-quant spec + the #659 protocol (BI=1,
    gb6144, PyTorch-native lowest-index argmax tie-break). max_num_seqs is the
    serve concurrency; with VLLM_BATCH_INVARIANT=1 on this *bf16* substrate the
    per-sequence token ids are concurrency-INVARIANT (proven by a conc1-vs-concN
    sha256 parity gate), so conc>1 is a faithful throughput multiplier — NOT a
    quality change. It is pinned identically across every cell so the paired/
    apples-to-apples comparisons are unaffected (see aime concurrency confound)."""
    return {
        "MODEL_ID": str(MASTER),
        "SERVED_MODEL_NAME": "gemma-4-e4b-it",
        "SERVE_DTYPE": "bfloat16",
        "MAX_MODEL_LEN": str(MAX_MODEL_LEN),
        "MAX_NUM_SEQS": str(max_num_seqs),
        "VLLM_BATCH_INVARIANT": "1",
        "GPU_MEMORY_UTILIZATION": "0.90",
        "MAX_NUM_BATCHED_TOKENS": "2048",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_SEED": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "FQ_G32_LAYERS": g32_layers,
        "FQ_G32_GROUP": str(g32),
        "FQ_G128_GROUP": str(g128),
        "FQ_QUANT_HEAD": "1" if quant_head else "0",
        "FQ_ENFORCE_EAGER": "1" if enforce_eager else "0",
        # FQ_BASE=1 -> serve the unmodified bf16 master (NO fake-quant): the bf16
        # base engine-health control (kanna #699). No-op otherwise.
        "FQ_BASE": "1" if base else "0",
    }


# --------------------------------------------------------------------------- items
def load_bank_items(kind: str, limit: int = 0) -> list[dict[str, Any]]:
    path = BANK / f"ar_{kind}.jsonl"
    items: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        it: dict[str, Any] = {
            "id": r["id"], "kind": kind,
            "prompt_token_ids": r["prompt_token_ids"],
            "prompt_sha256": r["prompt_sha256"],
        }
        if kind == "gpqa":
            it["target"] = r["target"]
            it["n_choices"] = r["n_choices"]
        else:
            it["gold"] = r.get("gold")
            it["year"] = r.get("year")
        items.append(it)
    if limit and limit > 0:
        items = items[:limit]
    return items


# --------------------------------------------------------------------------- request
def request_greedy(base_url: str, model: str, prompt_ids: list[int], max_tokens: int,
                   sample: dict[str, Any] | None = None, seed: int = 0) -> dict[str, Any]:
    if sample is None:
        sp = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}
    else:
        sp = {"temperature": sample["temperature"], "top_p": sample["top_p"],
              "top_k": sample["top_k"]}
    payload = {
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
        "min_tokens": MIN_TOKENS, **sp,
        "seed": seed, "stream": False, "add_special_tokens": False,
        "ignore_eos": False, "return_token_ids": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:300]}") from exc


# --------------------------------------------------------------------------- VRAM
def _gpu_used_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
        return max(vals) if vals else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def _sample_vram(stop: threading.Event, peak: dict[str, float]) -> None:
    while not stop.is_set():
        peak["mib"] = max(peak["mib"], _gpu_used_mib())
        stop.wait(2.0)


# --------------------------------------------------------------------------- gen
def _arm_path(cell: str, kind: str) -> Path:
    return RES / f"{cell}_{kind}.jsonl"


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                done.add(str(json.loads(line)["id"]))
            except (ValueError, KeyError):
                continue
    return done


def _gen_one(cell: str, kind: str, it: dict, srv, sample, seed: int) -> dict[str, Any]:
    """Generate + score ONE item. Pure w.r.t. shared state (urllib request +
    stateless scoring), so it is safe to run in a worker thread."""
    eff_max = max(MIN_TOKENS,
                  min(MAX_TOKENS, MAX_MODEL_LEN - len(it["prompt_token_ids"]) - CONTEXT_MARGIN))
    rec: dict[str, Any] = {
        "id": it["id"], "kind": kind,
        "prompt_sha256": it["prompt_sha256"], "max_tokens_eff": eff_max,
    }
    try:
        _treq = time.time()
        resp = request_greedy(srv.base_url, srv.served_model_name,
                              it["prompt_token_ids"], eff_max, sample=sample, seed=seed)
        rec["t_req_s"] = round(time.time() - _treq, 3)
        choice = _DO.choice_from_response(resp)
        comp_ids, _src, src_kind = _DO.extract_generated_token_ids(
            resp, choice, it["prompt_token_ids"])
        text = _DO.generated_text_from_choice(choice)
        finish = choice.get("finish_reason")
        scored = evalsets.score_item(it, text)
        rec.update({
            "completion_token_ids": comp_ids,
            "completion_token_sha256": evalsets.sha256_tokens(comp_ids),
            "completion_text": text,
            "num_completion_tokens": len(comp_ids),
            "finish_reason": finish,
            "token_id_source_kind": src_kind,
            "error": None,
            **scored,
        })
        if kind == "gpqa":
            rec["target"] = it["target"]; rec["n_choices"] = it["n_choices"]
        else:
            rec["gold"] = it.get("gold"); rec["year"] = it.get("year")
    except Exception as exc:  # noqa: BLE001
        rec.update({
            "completion_token_ids": [], "completion_token_sha256": None,
            "completion_text": "", "num_completion_tokens": 0,
            "finish_reason": "error", "answer": None, "correct": False,
            "extract_mode": "error", "error": repr(exc)[:300],
        })
        print(f"[gen] {cell}/{kind} id={it['id']} ERROR: {repr(exc)[:160]}", flush=True)
    return rec


def gen_cell(cell: str, kind: str, items: list[dict], srv, soft_deadline: float,
             sample: dict[str, Any] | None = None, seed: int = 0,
             max_num_seqs: int = 1) -> bool:
    """Drive generation with up to `max_num_seqs` requests in flight, matching the
    serve concurrency so vLLM continuous-batches them. On this bf16 + BI=1 substrate
    the per-sequence token ids are concurrency-invariant (sha256 parity-gated), so
    conc>1 only multiplies throughput, never changes a result. Bounded-inflight:
    never more than `max_num_seqs` submitted, deadline stops *refilling* (in-flight
    requests drain and are written — no wasted work), idempotent per-item resume.
    At conc=1 this is exactly the prior serial behavior."""
    out_path = _arm_path(cell, kind)
    done = _load_done_ids(out_path)
    todo = [it for it in items if it["id"] not in done]
    print(f"[gen] {cell}/{kind}: {len(done)} done, {len(todo)} to generate "
          f"(conc={max_num_seqs})", flush=True)
    if not todo:
        return True
    t0 = time.time()
    n_done = 0
    soft_capped = False
    it_iter = iter(todo)
    inflight: set = set()
    with open(out_path, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=max(1, max_num_seqs)) as ex:
        def _fill() -> None:
            while len(inflight) < max(1, max_num_seqs):
                nxt = next(it_iter, None)
                if nxt is None:
                    break
                inflight.add(ex.submit(_gen_one, cell, kind, nxt, srv, sample, seed))
        if time.time() < soft_deadline:
            _fill()
        while inflight:
            done_set, _ = futures_wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_set:
                inflight.discard(fut)
                rec = fut.result()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if n_done % 8 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    print(f"[gen] {cell}/{kind} {n_done}/{len(todo)} "
                          f"({el:.0f}s, {el/max(n_done,1):.1f}s/item)", flush=True)
            if time.time() < soft_deadline:
                _fill()
            elif not soft_capped:
                soft_capped = True
                print(f"[gen] {cell}/{kind} SOFT-CAP hit after {n_done} items — "
                      f"draining {len(inflight)} in-flight, no refill (resume)", flush=True)
    return not soft_capped


# --------------------------------------------------------------------------- stats
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_cell(cell: str, kind: str, max_num_seqs: int = 1) -> dict[str, Any]:
    path = _arm_path(cell, kind)
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(recs)
    err = sum(1 for r in recs if r.get("error"))
    n_eff = n - err
    correct = sum(1 for r in recs if r.get("correct"))
    trunc = sum(1 for r in recs if r.get("finish_reason") == "length")
    extract_fail = sum(1 for r in recs if r.get("answer") is None and not r.get("error"))
    toks = [r.get("num_completion_tokens", 0) for r in recs if not r.get("error")]
    acc = correct / n_eff if n_eff else 0.0
    lo, hi = wilson_ci(correct, n_eff)
    timed = [(r.get("t_req_s"), r.get("num_completion_tokens", 0))
             for r in recs if not r.get("error") and r.get("t_req_s") is not None]
    tot_wall = sum(t for t, _ in timed)
    tot_tok = sum(k for _, k in timed)
    mean_s_per_item = (tot_wall / len(timed)) if timed else None
    tokens_per_s_proxy = (tot_tok / tot_wall) if tot_wall > 0 else None
    ep = ENDPOINTS.get(kind, {})
    bf16 = ep.get("bf16")
    summ = {
        "cell": cell, "kind": kind, "n": n, "n_eff": n_eff, "errors": err,
        "correct": correct, "acc": acc, "ci_lo": lo, "ci_hi": hi,
        "truncation_rate": trunc / n if n else 0.0, "n_truncated": trunc,
        "extract_fail": extract_fail,
        "mean_completion_tokens": (sum(toks) / len(toks)) if toks else 0.0,
        "max_completion_tokens": max(toks) if toks else 0,
        "mean_s_per_item": mean_s_per_item, "tokens_per_s_proxy": tokens_per_s_proxy,
        "n_timed": len(timed), "total_decode_wall_s": tot_wall,
        "bf16_endpoint": bf16, "bf16_run": ep.get("bf16_run"),
        "int4_endpoint": ep.get("int4"), "bar90": ep.get("bar90"),
        "clears_90pct_bar": (acc >= ep["bar90"]) if ep.get("bar90") else None,
        "max_model_len": MAX_MODEL_LEN, "max_tokens": MAX_TOKENS, "min_tokens": MIN_TOKENS,
        "max_num_seqs": max_num_seqs, "batch_invariant": 1,
        # tokens_per_s_proxy sums per-request wall times; at conc>1 those OVERLAP, so
        # the proxy under-reads (≈per-stream, not aggregate) and is NOT a speed metric.
        # Speed is priced analytically from the byte-law (ubel #700). Quality (acc/CI)
        # is concurrency-invariant on this bf16+BI=1 substrate (sha256 parity-gated).
        "tps_proxy_is_speed_metric": False,
        "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
    }
    return summ


def log_wandb(summ: dict[str, Any], peak_vram_gb: float, group: str,
              meta: dict[str, Any]) -> str | None:
    try:
        import wandb
    except ImportError:
        print("[wandb] not available — skipping", flush=True)
        return None
    cell, kind = summ["cell"], summ["kind"]
    decode = meta.get("decode", "greedy_t0")
    run = wandb.init(
        project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
        group=group, name=f"fern/{cell}-{kind}-{decode}",
        config={
            "pr": 713, "cell": cell, "eval": kind,
            "decode": decode, "seed": meta.get("seed", 0),
            "g32_layers": meta.get("g32_layers", ""),
            "n_g32_layers": meta.get("n_g32_layers", 0),
            "g32_group": meta.get("g32_group", 32),
            "g128_group": meta.get("g128_group", 128),
            "quant_head": meta.get("quant_head", True),
            "base": meta.get("base", False),
            "substrate": ("bf16_master_no_fakequant" if meta.get("base")
                          else "bf16_fakequant_inmemory"),
            "skeleton": ("bf16 master (no quant)" if meta.get("base")
                         else "int4_g128 body (operative)"),
            "lm_head": ("bf16 (no quant)" if meta.get("base") else "int4_g128 (locked)"),
            "source_base": "qat-unquantized bf16 master (peer cache, read-only)",
            "max_model_len": MAX_MODEL_LEN, "max_tokens": MAX_TOKENS, "min_tokens": MIN_TOKENS,
            "max_num_seqs": meta.get("max_num_seqs", 1), "batch_invariant": 1,
            "vllm": "0.22.0", "spec": "off_AR_M1",
            "analysis_only": True, "official_tps": 0, "no_hf_job": 1, "fires": 0,
            "wandb_group": group,
        },
        reinit=True,
    )
    log = {f"g32/{k}": v for k, v in summ.items() if isinstance(v, (int, float, bool)) or v is None}
    log["g32/peak_vram_gb"] = peak_vram_gb
    log["g32/n_g32_layers"] = meta.get("n_g32_layers", 0)
    # Headline primary metric: g32-locus greedy AIME accuracy. The bf16 base control
    # logs under its OWN name so it never shadows the locus primary metric.
    if kind == "aime" and decode == "greedy_t0":
        metric = "bf16_base_aime_greedy" if meta.get("base") else "g32_locus_aime_greedy"
        log[metric] = summ["acc"]
        run.summary[metric] = summ["acc"]
    wandb.log(log)
    for k, v in summ.items():
        if isinstance(v, (int, float, bool)):
            run.summary[k] = v
    run.summary["peak_vram_gb"] = peak_vram_gb
    run.summary["n_g32_layers"] = meta.get("n_g32_layers", 0)
    rid = run.id
    wandb.finish()
    print(f"[wandb] logged {cell}/{kind} -> run {rid}", flush=True)
    return rid


# --------------------------------------------------------------------------- main
def _count_layers(spec: str) -> int:
    s = (spec or "").strip().lower()
    if s in ("", "none"):
        return 0
    seen: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-")
            seen.update(range(int(a), int(b) + 1))
        else:
            seen.add(int(tok))
    return len(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-name", required=True,
                    help="identity for jsonl/summary (e.g. fqg32_N0, fqg32_L14-27, fqg32_L14-27_s12345)")
    ap.add_argument("--g32-layers", default="",
                    help="decoder layers on the finer g32 grid (e.g. '14-27'; '' = N=0 all-g128)")
    ap.add_argument("--g32-group", type=int, default=32)
    ap.add_argument("--g128-group", type=int, default=128)
    ap.add_argument("--no-quant-head", action="store_true",
                    help="skip lm_head/embed fake-quant (held fixed across cells; only the abs anchor needs it)")
    ap.add_argument("--evals", default="aime")
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--soft-cap-min", type=float, default=SOFT_CAP_MIN_DEFAULT)
    ap.add_argument("--decode", default="greedy", choices=["greedy", "sampled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-group", default="aime-g32-locus-necessity-fern")
    ap.add_argument("--max-num-seqs", type=int, default=1,
                    help="serve+drive concurrency. On this bf16+BI=1 substrate token ids "
                         "are concurrency-invariant (sha256 parity-gated), so conc>1 is a "
                         "faithful throughput multiplier; pin it identically across cells")
    ap.add_argument("--cudagraph", action="store_true",
                    help="enable CUDA graphs (drop --enforce-eager) for ~2x decode speedup; "
                         "ONLY use after proving byte-identical greedy token ids vs enforce-eager")
    ap.add_argument("--base", action="store_true",
                    help="bf16 base control (kanna #699 engine-health gate): serve the "
                         "UNMODIFIED bf16 master with NO fake-quant; reconcile this engine's "
                         "greedy AIME against the bf16 endpoint (~0.4667, coherent not collapsed)")
    args = ap.parse_args()
    if args.base:
        args.g32_layers = ""  # fake-quant is skipped entirely; no layers are upgraded
    RES.mkdir(parents=True, exist_ok=True)
    if not MASTER_SRC.exists():
        print(f"[fatal] bf16 master source missing: {MASTER_SRC}", flush=True)
        return 2
    materialize_submission()
    materialize_master_view()
    if not (MASTER_VIEW / "model.safetensors").exists():
        print(f"[fatal] master view not built: {MASTER_VIEW}", flush=True)
        return 2
    evals = [e.strip() for e in args.evals.split(",") if e.strip()]
    limit = args.limit or (4 if args.mode == "smoke" else 0)
    cell = args.cell_name
    quant_head = not args.no_quant_head
    n_g32 = _count_layers(args.g32_layers)
    sample = SAMPLED_PARAMS if args.decode == "sampled" else None
    decode_tag = "greedy_t0" if args.decode == "greedy" else f"sampled_s{args.seed}"
    meta = {
        "decode": decode_tag, "seed": args.seed,
        "g32_layers": args.g32_layers or "none", "n_g32_layers": n_g32,
        "g32_group": args.g32_group, "g128_group": args.g128_group,
        "quant_head": quant_head, "max_num_seqs": args.max_num_seqs,
        "base": args.base,
    }
    print(f"[cfg] cell={cell} g32_layers={args.g32_layers or 'none'} (n={n_g32}) "
          f"decode={decode_tag} quant_head={quant_head}", flush=True)

    for note in paths.prepare_local_gpu_env():
        print(f"[gpu] {note}", flush=True)

    eval_items = {k: load_bank_items(k, limit=limit) for k in evals}
    for k, its in eval_items.items():
        print(f"[items] {k}: {len(its)} banked items", flush=True)

    soft_deadline = time.time() + args.soft_cap_min * 60.0
    peak = {"mib": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram, args=(stop, peak), daemon=True)
    sampler.start()
    log_path = RES / f"_serve_{cell}.log"
    complete: dict[str, bool] = {}
    extra_env = _env(args.g32_layers, args.g32_group, args.g128_group, quant_head,
                     enforce_eager=not args.cudagraph, max_num_seqs=args.max_num_seqs,
                     base=args.base)
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=SERVER_PY, port=PORT,
            log_path=log_path, extra_env=extra_env, startup_timeout_s=1800,
        ) as srv:
            print(f"[serve] {cell} ready at {srv.base_url} model={srv.served_model_name}", flush=True)
            for kind in evals:
                complete[kind] = gen_cell(cell, kind, eval_items[kind], srv, soft_deadline,
                                          sample=sample, seed=args.seed,
                                          max_num_seqs=args.max_num_seqs)
    finally:
        stop.set()
        sampler.join(timeout=5)
    peak_gb = (peak["mib"] or 0.0) / 1024.0
    print(f"[serve] {cell} peak {peak_gb:.1f} GB", flush=True)

    for kind in evals:
        summ_path = RES / f"summary_{cell}_{kind}.json"
        prior_rid = None
        if summ_path.exists():
            try:
                prior_rid = json.loads(summ_path.read_text()).get("wandb_run_id")
            except (ValueError, OSError):
                prior_rid = None
        summ = summarize_cell(cell, kind, max_num_seqs=args.max_num_seqs)
        summ.update({"decode": decode_tag, "seed": args.seed,
                     "g32_layers": meta["g32_layers"], "n_g32_layers": n_g32,
                     "g32_group": args.g32_group, "g128_group": args.g128_group,
                     "quant_head": quant_head, "peak_vram_gb": peak_gb})
        if prior_rid:
            summ["wandb_run_id"] = prior_rid
        sps = summ.get("mean_s_per_item")
        tps = summ.get("tokens_per_s_proxy")
        print(f"[summary] {cell}/{kind}: acc={summ['acc']:.4f} "
              f"CI[{summ['ci_lo']:.4f},{summ['ci_hi']:.4f}] n_eff={summ['n_eff']} "
              f"trunc={summ['truncation_rate']:.1%} extract_fail={summ['extract_fail']} "
              f"s/item={sps if sps is None else round(sps,1)} "
              f"tok/s={tps if tps is None else round(tps,1)} "
              f"clears0.42={summ['clears_90pct_bar']} complete={complete.get(kind)}", flush=True)
        summ_path.write_text(json.dumps(summ, indent=2))
        if not args.no_wandb and complete.get(kind) and not prior_rid:
            try:
                rid = log_wandb(summ, peak_gb, args.wandb_group, meta)
                if rid:
                    summ["wandb_run_id"] = rid
                    summ_path.write_text(json.dumps(summ, indent=2))
            except Exception as exc:  # noqa: BLE001
                print(f"[wandb] log failed: {repr(exc)[:200]}", flush=True)

    all_done = all(complete.get(k) for k in evals)
    print(f"[done] {cell} evals={evals} all_complete={all_done} {time.strftime('%H:%M:%S')}", flush=True)
    return 0 if all_done else 3


if __name__ == "__main__":
    raise SystemExit(main())
