#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""What does TRUE full-vocab lm_head equivalence cost in TPS? (PR #414).

RE-SCOPE FROM THE HUMAN (Issue #407, 2026-06-15)
------------------------------------------------
"we need token equivalence for any speculator we build ... find the fastest implementation that also
respects this equivalence ... Forget 500+ for now." This card prices the TWO strict-equivalence notions the
deployed lm_head truncation sits between, so the humans know WHICH gate "respect equivalence" should mean.

THE TWO EQUIVALENCES (the deployed 16384-row lm_head sits between them)
-----------------------------------------------------------------------
* SELF-REFERENTIAL (vs the submission's OWN truncated-head greedy AR -- the official scorer's operative gate):
  deployed PASSES. A pruned-head argmax is in the kept set BY CONSTRUCTION, so the served checkpoint matches
  its own plain greedy AR with 0 out-of-keepset emissions (#406 `8hpn489x`: deployed_submission_self_referential
  = True; deployed_submission_emissions_outside_keepset = 0).
* ABSOLUTE (vs TRUE full-vocab google/gemma-4-E4B greedy): deployed FAILS. The 16384-row cut is a LOSSY
  truncation of the 262144-row head; full-vocab greedy emits hundreds of distinct ids the truncated head cannot
  represent (#406: 548 distinct public ids clipped, 1.019% bf16 / 0.568% int4 of positions; 15.16% on a small
  OOD probe). 245,592 of the 245,760 pruned rows are separation-CERTIFIED reachable; provably_unreachable_rows
  = 0 -> essentially the WHOLE vocab is reachable, so absolute identity needs essentially the WHOLE head.

WHAT THIS CARD QUANTIFIES (the true-equivalence COST of the lm_head leg)
-----------------------------------------------------------------------
1. ABSOLUTE non-equivalence rate on a LARGE held-out corpus (>=50k positions, natural-language + code +
   multilingual): held_out_clip_rate = fraction of positions whose TRUE full-vocab greedy argmax falls OUTSIDE
   the deployed 16384 keepset (the rate at which the deployed config silently diverges from true gemma greedy
   on held-out data -- the +inf-PPL exposure surface on a private eval). full-vocab plain-greedy AR via vLLM on
   the on-target A10G, bf16 google/gemma-4-E4B-it (full 262144-row head). held_out_distinct_ids_clipped.
2. min_identity_safe_keepset_size = |union( true full-vocab greedy argmax ids over the held-out set,
   the #406 separation-certified reachable rows, the official-128 reachable support, the deployed kept rows )|.
   #406 already proved this is >= 16384 (the reachable support OVERFLOWS the keepset). Because 0 rows are
   provably unreachable and 245,592 are provably reachable, the minimal ABSOLUTE-identity-safe head is
   essentially the FULL 262144-row vocab. Distribution caveat: a larger/different corpus only GROWS the
   observed support; the provable component already forces near-full-vocab.
3. truevocab_lmhead_tps_cost = deployed_equiv_TPS - TPS_at_widened_head, modelling the lm_head read as
   PROPORTIONAL to kept rows on the #398/#283 head-read roofline (21.0 MB/step at 16384, BODY int4 +
   lm_head int4, A10G 600 GB/s). DIAGNOSTIC read-model delta, NOT an official measurement. Prices ONLY the
   TRUNCATION (row-count) dimension at the deployed int4 head format; the head's own int4-vs-bf16 quant
   identity is a SEPARATE additive leg priced elsewhere (#371 fusion/quant identity).

VERDICT (this card)
-------------------
* deployed_passes_self_referential = True (#406). deployed_passes_absolute = False (#406 + this card's
  held-out probe). The deployed baseline respects the SELF-REFERENTIAL gate at 0 cost; ABSOLUTE (true
  full-vocab) equivalence costs ~truevocab_lmhead_tps_cost TPS and a near-full-vocab head.
* For the humans: "the fastest implementation that respects equivalence" is correctly measured against the
  SELF-REFERENTIAL gate (which the deployed config and any in-keepset speculator satisfy for free). Demanding
  ABSOLUTE equivalence re-prices the head ~16x wider and forfeits a measurable chunk of head-read TPS for
  every token, with no benefit to the official self-referential scorer.

PRIMARY metric  truevocab_lmhead_equivalence_self_test_passes  (>=20 pure-logic checks of the read math,
clip-rate / distinct-clip logic, the min-keepset union assembly, the TPS-cost model, and the verdict gating;
env-independent, runs under the numpy-only .venv).
SCOPE: analysis / microbench. NO HF Job, NO submission, NO served-file change, 0 official TPS. The provability
separation certificate (loads lm_head.weight + final norm from a LOCAL safetensors extract) and the >=50k
held-out greedy probe (vLLM, full bf16 head) are GPU/torch enrichments that DEGRADE GRACEFULLY; the verdict
stands on the 0-GPU official-128 reachability + the provability reduction + #406 anchors.

Run:
  # PRIMARY self-test only (numpy-only, no torch/GPU):
  .venv/bin/python -m research.validity.truevocab_lmhead_equivalence_cost.truevocab_lmhead_equivalence_cost --self-test
  # held-out probe ONLY (isolates the 16GB vLLM load into its own process; writes the sidecar):
  CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python -m research.validity.truevocab_lmhead_equivalence_cost.truevocab_lmhead_equivalence_cost --heldout-probe-only
  # full card (torch+vLLM env, e.g. /tmp/server-venv on the A10G box):
  CUDA_VISIBLE_DEVICES=0 /tmp/server-venv/bin/python -m research.validity.truevocab_lmhead_equivalence_cost.truevocab_lmhead_equivalence_cost \
    --wandb_group truevocab-lmhead-equivalence-cost --wandb_name land/truevocab-lmhead-equivalence-cost
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Pre-import the REAL wandb BEFORE putting REPO_ROOT (= target/, has a ./wandb run-output dir that shadows
# the package as a PEP-420 namespace) on sys.path[0]. Mirrors the #385/#398/#406 house pattern.
try:
    import wandb as _wandb_preimport  # noqa: F401
except Exception:  # noqa: BLE001
    _wandb_preimport = None

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------- #
# Geometry + bandwidth constants (provenance documented; consistent with #385/#390/#398/#406/#344).
# ---------------------------------------------------------------------------- #
HIDDEN = 2560                       # gemma-4-E4B hidden_size
FULL_VOCAB = 262144                 # gemma-4-E4B text vocab (== embed_tokens rows; tied lm_head)
A10G_BW_GBPS = 600.0                # A10G HBM peak bandwidth (the roofline figure, #385)
BODY_INT4_GB = 1.6973824           # int4 body bytes/step (#371/#344/#278/#385)

DEPLOYED_LMHEAD_ROWS = 16384        # #390 osoi5 baked, channel-wise int4-Marlin (the DEPLOYED truncation)
LMHEAD12K_ROWS = 12288             # #385 further-prune lever (more aggressive; same lossy family)
INT4_BITS = 4
SCALE_DTYPE_BYTES = 2               # bf16 channel scales
FINAL_LOGIT_SOFTCAP = 30.0         # config.json final_logit_softcapping (MONOTONIC -> argmax-preserving)
RMS_NORM_EPS = 1e-6                # config.json rms_norm_eps

# #398/#283/#406 deployed anchors (CITE; not re-derived).
DEPLOYED_LMHEAD_READ_MB_398 = 21.0         # #398: deployed 16384-row int4-Marlin channel read MB/step
DEPLOYED_ROOFLINE_TPS_283 = 349.16471606185996  # #283/#398/#406: deployed head-read roofline
OFFICIAL_DEPLOYED_TPS = 481.53             # PR #52 deployed (non-strict)
PPL_GATE = 2.42
PPL_BASELINE = 2.3772
MILESTONE = 500.0
OFFICIAL_FRONTIER_TPS = 481.53             # PR #52 `2x9fm2zx` official frontier (UNCHANGED by this card)

# #406 anchors (`8hpn489x`) -- the provability backbone this card extends (used as graceful fallbacks).
N406_SEPARATION_REACHABLE_PRUNED = 245592  # separation-certified reachable pruned rows (#406)
N406_PROVABLY_UNREACHABLE = 0              # provably-unreachable rows (#406): only exact-zero-norm; none exist
N406_OFFICIAL_OOK_IDS = 548                # distinct official-128 ids clipped (bf16, #406)
N406_OFFICIAL_CLIP_RATE_BF16 = 0.01019287109375
N406_OOD_SMALL_CLIP_RATE = 0.1515768056968464  # #406 small (24-prompt) OOD probe clip rate

# ---- local artifact paths (degrade gracefully if absent) ----
KEEPSET_16K_CANDIDATES = [
    "/tmp/osoi5-v0-baked/pck04_keepset.json",
]
# full-vocab plain-greedy-AR reference decode of the OFFICIAL 128 (the in-distribution reachable support).
OFFICIAL_BF16_DECODE = REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl"
GT_TOKENS = REPO_ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
# Full-vocab lm_head.weight BF16 [262144,2560] + final RMSNorm (land-owned extract from the shared base model
# google/gemma-4-E4B-it-qat-w4a16-ct, reused from #406). Needed for the separation certificate.
LMHEAD_W_CANDIDATES = [
    "/tmp/land-lmhead-prov/lmhead_norm.safetensors",
]
# FULL bf16 model dir for the held-out greedy AR probe (vLLM). MUST be a full-vocab head (262144 rows) so the
# emitted token IS the TRUE full-vocab greedy argmax. land-owned google/gemma-4-E4B-it snapshot is used first.
_LAND_HUB = "/senpai-run/home/student-land/.cache/huggingface/hub"
HELDOUT_MODEL_DIR_CANDIDATES = [
    f"{_LAND_HUB}/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187",
    "google/gemma-4-E4B-it",
]
HELDOUT_SIDECAR = HERE / "heldout_reachable_support.json"

TOL = 1e-9
NEARZERO_NORM = 1e-3               # a row with ||w_r|| <= this is treated as a provably-dominated zero row
HELDOUT_TARGET_POSITIONS = 50000   # PR #414 aim: >=50k held-out positions


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


# ---------------------------------------------------------------------------- #
# (1) Read-byte model + roofline (reused from #398/#406, the diagnostic TPS basis).
# ---------------------------------------------------------------------------- #
def lmhead_read_bytes(rows: int, bits: int = INT4_BITS, channel: bool = True) -> float:
    """Per-decode-step truncated lm_head weight read = packed int4 weights + bf16 channel scales."""
    weight_bytes = rows * HIDDEN * bits / 8.0
    scale_bytes = rows * SCALE_DTYPE_BYTES if channel else 0.0
    return weight_bytes + scale_bytes


def roofline_tps(lmhead_read_gb: float) -> float:
    """Single-token decode roofline TPS (body int4 + lm_head), A10G BW. Diagnostic ceiling, NOT official."""
    return A10G_BW_GBPS / (BODY_INT4_GB + lmhead_read_gb)


def softcap(x: np.ndarray, cap: float = FINAL_LOGIT_SOFTCAP) -> np.ndarray:
    return cap * np.tanh(x / cap)


# ---------------------------------------------------------------------------- #
# (2) Keepset load.
# ---------------------------------------------------------------------------- #
def _load_keepset(paths: list[str]) -> tuple[list[int] | None, str | None]:
    p = _first_existing(paths)
    if p is None:
        return None, None
    meta = json.loads(Path(p).read_text())
    ids = meta.get("keep_ids") or meta.get("kept_ids")
    return (sorted(int(i) for i in ids) if ids else None), p


def _emission_ids(path: Path) -> list[int]:
    ids: list[int] = []
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ids.extend(int(t) for t in rec.get("completion_token_ids", []))
    return ids


# ---------------------------------------------------------------------------- #
# (3) Held-out corpus (OOD): natural-language + code + multilingual + math/reasoning. Programmatic so the
#     diversity (and the reachable support) is materially larger than #406's 24-prompt probe.
# ---------------------------------------------------------------------------- #
def build_heldout_corpus() -> list[str]:
    # curated core (continuity with the #406 24-prompt OOD probe + extensions).
    core = [
        "Translate to French and explain: The quick brown fox jumps over the lazy dog near the riverbank.",
        "Écris un court poème en français sur la mer en hiver, puis explique tes choix de mots.",
        "Schreibe einen ausführlichen Absatz auf Deutsch über die Geschichte der künstlichen Intelligenz.",
        "用中文写一段关于量子计算及其潜在应用的详细介绍，并举三个例子。",
        "日本語で、四季それぞれについての俳句を書き、その後に解説を付けてください。",
        "اكتب فقرة مفصلة باللغة العربية عن أهمية الماء في الحياة اليومية مع أمثلة.",
        "Напиши развёрнутый рассказ на русском языке о путешествии к далёкой звезде.",
        "Escribe una receta detallada de paella valenciana paso a paso, con cantidades.",
        "한국어로 인공지능 윤리에 대한 짧은 에세이를 쓰고, 핵심 논점을 정리해 주세요.",
        "Scrivi una breve storia in italiano su un orologiaio che ripara il tempo.",
        "Escreva um conto em português sobre uma cidade que flutua sobre as nuvens.",
        "Schrijf een korte tekst in het Nederlands over de toekomst van duurzame energie.",
        "Napisz krótkie opowiadanie po polsku o bibliotekarzu, który kolekcjonuje zapomniane słowa.",
        "हिंदी में कृत्रिम बुद्धिमत्ता के लाभ और जोखिमों पर एक संक्षिप्त निबंध लिखिए।",
        "Yaz bir kısa hikaye Türkçe olarak: zamanı durdurabilen bir saatçi hakkında.",
        "Write a Python function that computes the SHA-256 of a file in 8 KB chunks, with type hints and a docstring.",
        "Explain the difference between a B-tree and an LSM-tree, with a small code sketch for each.",
        "def fibonacci(n):\n    # complete this generator, add type hints, and explain the time complexity\n",
        "Give the full JSON schema for an OpenAI-compatible /v1/completions request and annotate each field.",
        "Write idiomatic Rust that reads lines from stdin, counts word frequencies, and prints the top 10.",
        "Implement a thread-safe LRU cache in Go with generics and unit tests.",
        "Write a SQL query to find the second-highest salary per department using a window function, then explain it.",
        "Summarize the plot of a noir detective novel set in 1940s Lisbon, then list its themes.",
        "List five rare chemical elements, one industrial use of each, and a safety note.",
        "Prove that the square root of 2 is irrational, step by step, then generalize to sqrt(p).",
        "Derive the closed form of the sum 1 + 2 + ... + n and verify it for n = 100.",
        "Compose a haiku about debugging at 3am, then a limerick about the same.",
        "Describe the Krebs cycle for a first-year biochemistry student, with the key intermediates.",
        "Generate a regex matching valid IPv6 addresses and explain each component in detail.",
        "Write a dialogue between a Stoic philosopher and a startup founder about failure.",
        "Explain gradient checkpointing, when it trades compute for memory, and show a PyTorch snippet.",
        "Translate '안녕하세요, 만나서 반갑습니다' into five languages with romanizations.",
        "Outline a threat model for a self-hosted password manager, listing assets and adversaries.",
        "Explain how a transformer attention head works, with the scaled dot-product formula.",
        "Write a short fairy tale in the style of the Brothers Grimm about a clockwork nightingale.",
        "Describe the process of nuclear fusion in the sun and why it is hard to replicate on Earth.",
        "Write a bash script that backs up a directory, rotates 7 daily snapshots, and logs to syslog.",
        "Explain the CAP theorem with a concrete example for each of the three trade-offs.",
        "Write an essay on the economics of open-source software and its sustainability problem.",
        "Give a step-by-step proof that there are infinitely many prime numbers.",
    ]
    # programmatic multilingual x task expansion for breadth (greedy reaches far up the vocab on these).
    langs = ["English", "French", "German", "Spanish", "Mandarin Chinese", "Japanese", "Arabic", "Russian",
             "Hindi", "Portuguese", "Korean", "Italian", "Turkish", "Vietnamese", "Greek", "Hebrew",
             "Thai", "Swahili", "Polish", "Indonesian", "Bengali", "Tamil", "Ukrainian", "Czech",
             "Romanian", "Hungarian", "Finnish", "Swedish", "Persian", "Urdu", "Malay", "Dutch"]
    tasks = [
        "Write a detailed ~400-word essay in {lang} about the history and future of space exploration.",
        "Explain at length in {lang}, for a curious teenager, how vaccines train the immune system.",
        "Write a multi-paragraph short story in {lang} about a lighthouse keeper who befriends a storm.",
        "In {lang}, describe in detail a traditional dish from a culture that speaks it, with full recipe steps.",
        "Write in {lang} a thorough persuasive argument for protecting urban green spaces, with examples.",
        "Write a detailed travel guide in {lang} for a week in a famous city, day by day.",
        "In {lang}, explain the water cycle to a school class with a clear step-by-step description.",
    ]
    expanded = [t.format(lang=lang) for lang in langs for t in tasks]
    # extra code / structured-output prompts.
    code_extra = [
        "Write a complete FastAPI app with one POST endpoint that validates a JSON body with Pydantic.",
        "Implement Dijkstra's shortest path in Python with a binary heap and explain the complexity.",
        "Write a TypeScript React component that fetches and paginates a list with error and loading states.",
        "Implement quicksort and mergesort in C, then discuss when each is preferable.",
        "Write a Dockerfile and docker-compose.yml for a Flask app with Postgres and Redis.",
        "Write a recursive-descent parser for arithmetic expressions in Python with operator precedence.",
        "Implement a small key-value store with a write-ahead log in Go.",
        "Write a CUDA kernel that adds two vectors and the host code that launches it.",
        "Write a GraphQL schema and resolvers for a blog with posts, authors, and comments.",
        "Implement a Kalman filter in NumPy for 1-D position-velocity tracking, with a demo.",
    ]
    corpus = core + expanded + code_extra
    # de-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in corpus:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _ensure_vllm_env() -> None:
    """Make vLLM's flashinfer JIT (sampling kernel) able to build under /tmp/server-venv: put the bundled
    nvidia cu* headers (curand.h etc.) on CPATH, and disable the flashinfer sampler (greedy = argmax needs
    no curand). Without this, EngineCore init fails on `fatal error: curand.h: No such file or directory`."""
    import glob
    import os
    cands = glob.glob("/tmp/server-venv/lib/python*/site-packages/nvidia/cu*/include")
    cands += glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/nvidia/cu*/include"))
    for inc in cands:
        if Path(inc, "curand.h").exists():
            cur = os.environ.get("CPATH", "")
            if inc not in cur.split(os.pathsep):
                os.environ["CPATH"] = inc + (os.pathsep + cur if cur else "")
            break
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def heldout_probe(max_tokens: int = 512, force: bool = False) -> dict[str, Any]:
    """Greedy-decode a LARGE diverse OOD corpus with the FULL-vocab bf16 head (vLLM) and measure how many
    positions / distinct argmax rows fall OUTSIDE the deployed 16384 keepset (the held-out absolute
    non-equivalence surface). Emits the distinct argmax id set so the caller can union it into the minimal
    identity-safe keepset. Caches to a sidecar; degrades gracefully if vLLM / the model dir is unavailable."""
    if HELDOUT_SIDECAR.exists() and not force:
        try:
            cached = json.loads(HELDOUT_SIDECAR.read_text())
            if cached.get("available"):
                cached["from_cache"] = True
                return cached
        except Exception:  # noqa: BLE001
            pass
    res: dict[str, Any] = {"available": False}
    k16, _ = _load_keepset(KEEPSET_16K_CANDIDATES)
    model_dir = _first_existing([str(Path(p) / "config.json") for p in HELDOUT_MODEL_DIR_CANDIDATES])
    if model_dir is not None:
        model_dir = str(Path(model_dir).parent)
    elif HELDOUT_MODEL_DIR_CANDIDATES:
        model_dir = HELDOUT_MODEL_DIR_CANDIDATES[-1]  # last resort: a Hub id (may download)
    if k16 is None:
        res["note"] = "deployed keepset not found locally (held-out probe skipped)"
        return res
    _ensure_vllm_env()
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"vLLM unavailable ({exc!r}); held-out probe skipped (verdict stands on #406 anchors)"
        return res
    try:
        s16 = set(k16)
        prompts = build_heldout_corpus()
        llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.90,
                  max_model_len=2048, enforce_eager=True, trust_remote_code=True)
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = llm.generate(prompts, sp)
        all_ids: list[int] = []
        per_prompt_tokens: list[int] = []
        for o in outs:
            toks = [int(t) for t in o.outputs[0].token_ids]
            all_ids.extend(toks)
            per_prompt_tokens.append(len(toks))
        distinct = sorted(set(all_ids))
        ook = [v for v in distinct if v not in s16]
        ook_pos = sum(1 for t in all_ids if t not in s16)
        prompts_affected = 0
        idx = 0
        for n in per_prompt_tokens:
            seg = all_ids[idx:idx + n]
            idx += n
            if any(t not in s16 for t in seg):
                prompts_affected += 1
        res = {
            "available": True,
            "model_dir": model_dir,
            "heldout_prompts": len(prompts),
            "heldout_total_positions": len(all_ids),
            "max_tokens_per_prompt": max_tokens,
            "mean_tokens_per_prompt": (len(all_ids) / len(prompts)) if prompts else 0.0,
            "distinct_reachable_rows_heldout": len(distinct),
            "held_out_distinct_ids_clipped": len(ook),
            "held_out_clipped_positions": ook_pos,
            "held_out_clip_rate": (ook_pos / len(all_ids)) if all_ids else 0.0,
            "heldout_prompts_with_clip": prompts_affected,
            "max_reachable_emitted_id_heldout": distinct[-1] if distinct else None,
            "reached_50k_target": bool(len(all_ids) >= HELDOUT_TARGET_POSITIONS),
            # the full distinct argmax id support, for the min-keepset union (kept compact as a sorted list).
            "distinct_argmax_ids": distinct,
            "distinct_argmax_ids_outside_keepset": ook,
        }
        HELDOUT_SIDECAR.write_text(json.dumps(res, indent=2))
        # free vLLM before the caller's torch separation certificate (best-effort).
        try:
            import contextlib
            import gc
            del llm
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
        except Exception:  # noqa: BLE001
            pass
        return res
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"held-out probe failed: {exc!r}"
        return res


# ---------------------------------------------------------------------------- #
# (4) Official-128 reachability (0-GPU; from the existing full-vocab greedy reference capture).
# ---------------------------------------------------------------------------- #
def analyze_official_reachability() -> dict[str, Any]:
    k16, _ = _load_keepset(KEEPSET_16K_CANDIDATES)
    res: dict[str, Any] = {"available": bool(k16 is not None and OFFICIAL_BF16_DECODE.exists())}
    if not res["available"]:
        res["note"] = "official full-vocab greedy reference or keepset not found locally"
        return res
    s16 = set(k16)
    bf16 = _emission_ids(OFFICIAL_BF16_DECODE)
    bf16_distinct = sorted(set(bf16))
    ook16 = [v for v in bf16_distinct if v not in s16]
    ook16_pos = sum(1 for t in bf16 if t not in s16)
    res.update({
        "official_prompts": 128,
        "official_total_positions": len(bf16),
        "distinct_reachable_rows_official": len(bf16_distinct),
        "official_distinct_ids_clipped": len(ook16),
        "official_clip_rate_bf16": (ook16_pos / len(bf16)) if bf16 else 0.0,
        "max_reachable_emitted_id_official": bf16_distinct[-1] if bf16_distinct else None,
        "official_distinct_argmax_ids": bf16_distinct,
        "official_distinct_argmax_ids_outside_keepset": ook16,
    })
    if GT_TOKENS.exists():
        tgt: set[int] = set()
        for line in GT_TOKENS.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tgt.update(int(t) for t in r.get("target_token_ids", []))
        res["gt_target_distinct"] = len(tgt)
        res["gt_target_outside_keepset16384"] = len(tgt - s16)
    return res


# ---------------------------------------------------------------------------- #
# (5) Provability separation certificate (reused from #406): how many pruned rows are CERTIFIED reachable,
#     and (NEW) the boolean must-keep mask over the full vocab so the caller can size the minimal keepset.
# ---------------------------------------------------------------------------- #
def separation_reachable(W_kept: np.ndarray, w_r: np.ndarray) -> bool:
    """RIGOROUS sufficient certificate that row r is REACHABLE over the (centered, full-dim) attainable set.

    If max_{s in kept} (w_s . hat w_r) < ||w_r|| then d = hat w_r separates w_r from conv(kept): row r
    strictly beats every kept row in direction d, so some attainable h makes r the greedy argmax. (Sound for
    reachability; a FAILED test does NOT certify unreachability.)"""
    nr = float(np.linalg.norm(w_r))
    if nr <= NEARZERO_NORM:
        return False
    sup = float(np.max(W_kept @ (w_r / nr)))
    return sup < nr - 1e-9


def separation_certificate() -> dict[str, Any]:
    """REAL separation certificate on the deployed lm_head: load lm_head.weight + final norm from the local
    safetensors extract (lazy torch import; GPU/tiled). Returns the certified-reachable count AND a boolean
    must-keep mask over the full vocab (kept rows + separation-certified pruned rows). Degrades gracefully."""
    res: dict[str, Any] = {"available": False}
    st_path = _first_existing(LMHEAD_W_CANDIDATES)
    k16, _ = _load_keepset(KEEPSET_16K_CANDIDATES)
    if st_path is None or k16 is None:
        res["note"] = "lm_head safetensors or keepset not found locally (separation certificate degraded)"
        return res
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"torch/safetensors unavailable ({exc!r}); run under a torch env e.g. /tmp/server-venv"
        return res
    try:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            W = f.get_tensor("lm_head.weight").float()
            g = f.get_tensor("model.language_model.norm.weight").float()
        V, n = W.shape
        norms = W.norm(dim=1)
        kept_t = torch.tensor(sorted(k16), dtype=torch.long)
        kept_mask = torch.zeros(V, dtype=torch.bool)
        kept_mask[kept_t] = True
        pruned_mask = ~kept_mask
        kn = norms[kept_mask]
        pn = norms[pruned_mask]
        Rkmax = float(kn.max())
        c = (1.0 + g)
        # must-keep mask: kept rows + separation-certified-reachable pruned rows (the minimal-keepset support).
        must_keep = kept_mask.clone()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Wk = W[kept_mask].to(dev)                       # [16384, n]
        pruned_idx = torch.where(pruned_mask)[0]
        normball_reachable = int((pn > Rkmax).sum())
        zero_rows = int((pn <= NEARZERO_NORM).sum())
        sep = 0
        CH = 8192
        with torch.no_grad():
            for i in range(0, pruned_idx.numel(), CH):
                idx = pruned_idx[i:i + CH]
                Wp = W[idx].to(dev)
                npr = Wp.norm(dim=1)
                dirs = Wp / npr.clamp_min(1e-12).unsqueeze(1)
                sup = (Wk @ dirs.t()).max(dim=0).values     # support of conv(kept) in each dir
                cert = (sup < (npr - 1e-6)) & (npr > NEARZERO_NORM)
                sep += int(cert.sum())
                must_keep[idx[cert.cpu()]] = True
        res.update({
            "available": True,
            "lmhead_weight_path": st_path,
            "vocab": int(V), "hidden": int(n),
            "kept_norm_min": float(kn.min()), "kept_norm_max": Rkmax, "kept_norm_mean": float(kn.mean()),
            "pruned_norm_min": float(pn.min()), "pruned_norm_max": float(pn.max()),
            "pruned_norm_mean": float(pn.mean()),
            "norm_bands_overlap": bool(float(pn.max()) >= float(kn.min()) and float(pn.min()) <= Rkmax),
            "c_max": float(c.max()), "c_min": float(c.min()),
            "provably_reachable_pruned_normball": normball_reachable,
            "provably_reachable_pruned_separation": sep,
            "provably_unreachable_rows": zero_rows,
            "deployed_truncation_provably_safe": bool(sep == 0 and normball_reachable == 0),
            "separation_certified_must_keep_size": int(must_keep.sum()),
            "_must_keep_mask": must_keep.cpu().numpy(),   # consumed in-process by the min-keepset assembly
        })
        del Wk, W, g
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res
    except Exception as exc:  # noqa: BLE001
        res["note"] = f"separation certificate failed: {exc!r}"
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return res


# ---------------------------------------------------------------------------- #
# (6) Minimal absolute-identity-safe keepset = union( kept, separation-certified-reachable pruned,
#     held-out greedy argmax ids, official-128 greedy argmax ids ). Exact via a full-vocab boolean mask.
# ---------------------------------------------------------------------------- #
def compute_min_identity_safe_keepset(keep_ids: list[int] | None, sep: dict, heldout: dict,
                                      official: dict, vocab_size: int = FULL_VOCAB) -> dict[str, Any]:
    out: dict[str, Any] = {}
    mask = np.zeros(vocab_size, dtype=bool)
    components: dict[str, int] = {}
    used_provable = False

    if keep_ids is not None:
        ki = np.asarray([i for i in keep_ids if 0 <= i < vocab_size], dtype=np.int64)
        mask[ki] = True
        components["deployed_kept"] = int(ki.size)

    # provable component (the dominant one): separation-certified must-keep mask.
    mk = sep.get("_must_keep_mask") if sep.get("available") else None
    if mk is not None:
        m = np.asarray(mk, dtype=bool)
        if m.shape[0] == vocab_size:
            before = int(mask.sum())
            mask |= m
            components["separation_certified_added"] = int(mask.sum()) - before
            used_provable = True
    elif sep.get("available"):
        # mask absent but count present: fold the certified count in analytically (still >= the union).
        components["separation_certified_count_only"] = int(sep.get("provably_reachable_pruned_separation", 0))

    def _fold(ids, key):
        if not ids:
            components[key] = 0
            return
        arr = np.asarray([i for i in ids if 0 <= i < vocab_size], dtype=np.int64)
        before = int(mask.sum())
        mask[arr] = True
        components[key] = int(mask.sum()) - before

    _fold(heldout.get("distinct_argmax_ids") if heldout.get("available") else None, "heldout_added")
    _fold(official.get("official_distinct_argmax_ids") if official.get("available") else None, "official_added")

    union_size = int(mask.sum())
    # if the provable mask was unavailable, fold its count in analytically as a floor (PR explicitly unions the
    # #406 separation-certified rows, which are all OUTSIDE the deployed keepset and disjoint from observed).
    floor_with_anchor = union_size
    if not used_provable:
        anchor = max(int(sep.get("provably_reachable_pruned_separation", 0)) if sep.get("available") else 0,
                     N406_SEPARATION_REACHABLE_PRUNED)
        # observed out-of-keepset ids are a subset of reachable, so union >= kept + anchor.
        floor_with_anchor = max(union_size, components.get("deployed_kept", DEPLOYED_LMHEAD_ROWS) + anchor)

    min_keepset = floor_with_anchor
    # reference = the actual number of deployed kept rows (== DEPLOYED_LMHEAD_ROWS for the real keepset).
    deployed_ref = components.get("deployed_kept", DEPLOYED_LMHEAD_ROWS)
    out.update({
        "min_identity_safe_keepset_size": int(min_keepset),
        "deployed_kept_rows": int(deployed_ref),
        "min_keepset_exceeds_deployed": bool(min_keepset > deployed_ref),
        "min_keepset_overflow_rows": int(min_keepset - deployed_ref),
        "min_keepset_used_provable_mask": bool(used_provable),
        "min_keepset_components": components,
        "min_keepset_fraction_of_full_vocab": float(min_keepset / FULL_VOCAB),
    })
    return out


# ---------------------------------------------------------------------------- #
# (7) TPS cost of widening the head from deployed 16384 -> min_identity_safe_keepset_size.
# ---------------------------------------------------------------------------- #
def price_truevocab_tps_cost(min_keepset: int) -> dict[str, Any]:
    deployed_read_gb = lmhead_read_bytes(DEPLOYED_LMHEAD_ROWS) / 1e9
    widened_rows = max(DEPLOYED_LMHEAD_ROWS, int(min_keepset))
    widened_read_gb = lmhead_read_bytes(widened_rows) / 1e9
    deployed_tps = roofline_tps(deployed_read_gb)
    widened_tps = roofline_tps(widened_read_gb)
    cost = deployed_tps - widened_tps
    full_read_gb = lmhead_read_bytes(FULL_VOCAB) / 1e9
    full_tps = roofline_tps(full_read_gb)
    return {
        "deployed_equiv_TPS": deployed_tps,
        "widened_rows": widened_rows,
        "TPS_at_widened_head": widened_tps,
        "truevocab_lmhead_tps_cost": cost,
        "truevocab_lmhead_tps_cost_frac": (cost / deployed_tps) if deployed_tps else 0.0,
        "deployed_lmhead_read_mb": deployed_read_gb * 1e3,
        "widened_lmhead_read_mb": widened_read_gb * 1e3,
        "full_vocab_lmhead_read_mb": full_read_gb * 1e3,
        "full_vocab_roofline_tps": full_tps,
    }


# ---------------------------------------------------------------------------- #
# (8) Verdict assembly.
# ---------------------------------------------------------------------------- #
def assemble_verdict(keep_ids, sep, heldout, official, minks, tps) -> dict[str, Any]:
    # held-out absolute non-equivalence (prefer the >=50k probe; fall back to #406 small-OOD anchor).
    if heldout.get("available"):
        held_out_clip_rate = float(heldout["held_out_clip_rate"])
        held_out_distinct_ids_clipped = int(heldout["held_out_distinct_ids_clipped"])
        held_out_positions = int(heldout["heldout_total_positions"])
        heldout_source = "vllm_full_bf16_>=50k" if heldout.get("reached_50k_target") else "vllm_full_bf16_partial"
    else:
        held_out_clip_rate = N406_OOD_SMALL_CLIP_RATE
        held_out_distinct_ids_clipped = 33  # #406 small-OOD probe
        held_out_positions = 983
        heldout_source = "fallback_#406_small_ood"

    provably_unreachable_rows = int(sep["provably_unreachable_rows"]) if sep.get("available") else N406_PROVABLY_UNREACHABLE
    sep_reachable = int(sep["provably_reachable_pruned_separation"]) if sep.get("available") else N406_SEPARATION_REACHABLE_PRUNED

    # self-referential: deployed pruned head matches its OWN greedy AR by construction (#406, 0 out-of-keepset).
    deployed_passes_self_referential = True
    # absolute: deployed FAILS iff the full-vocab greedy support overflows the keepset on held-out/official.
    absolute_clip_evidence = held_out_clip_rate > 0.0 or (official.get("available")
                                                          and official.get("official_distinct_ids_clipped", 0) > 0)
    deployed_passes_absolute = not bool(absolute_clip_evidence)

    return {
        # ---- PR #414 headline deliverables ----
        "truevocab_lmhead_equivalence_self_test_passes": None,  # filled by synthesize()
        "held_out_clip_rate": held_out_clip_rate,
        "held_out_distinct_ids_clipped": held_out_distinct_ids_clipped,
        "held_out_positions": held_out_positions,
        "heldout_source": heldout_source,
        "min_identity_safe_keepset_size": int(minks["min_identity_safe_keepset_size"]),
        "min_keepset_exceeds_deployed": bool(minks["min_keepset_exceeds_deployed"]),
        "min_keepset_overflow_rows": int(minks["min_keepset_overflow_rows"]),
        "min_keepset_fraction_of_full_vocab": float(minks["min_keepset_fraction_of_full_vocab"]),
        "truevocab_lmhead_tps_cost": float(tps["truevocab_lmhead_tps_cost"]),
        "truevocab_lmhead_tps_cost_frac": float(tps["truevocab_lmhead_tps_cost_frac"]),
        "deployed_passes_self_referential": deployed_passes_self_referential,
        "deployed_passes_absolute": deployed_passes_absolute,
        # ---- supporting ----
        "deployed_rows_kept": DEPLOYED_LMHEAD_ROWS,
        "provably_unreachable_rows": provably_unreachable_rows,
        "provably_reachable_pruned_separation": sep_reachable,
        "deployed_equiv_TPS": float(tps["deployed_equiv_TPS"]),
        "TPS_at_widened_head": float(tps["TPS_at_widened_head"]),
        "widened_rows": int(tps["widened_rows"]),
        "official_distinct_ids_clipped": int(official.get("official_distinct_ids_clipped", 0)) if official.get("available") else N406_OFFICIAL_OOK_IDS,
        "official_clip_rate_bf16": float(official.get("official_clip_rate_bf16", N406_OFFICIAL_CLIP_RATE_BF16)) if official.get("available") else N406_OFFICIAL_CLIP_RATE_BF16,
        "final_logit_softcap_monotonic_preserves_argmax": True,
        "gate_for_respect_equivalence": "self_referential",
    }


# ---------------------------------------------------------------------------- #
# (9) PRIMARY self-test -- >=20 pure-logic checks (numpy + stdlib; env-independent).
# ---------------------------------------------------------------------------- #
def self_test() -> dict[str, Any]:
    c: dict[str, bool] = {}

    # ---- read-byte math + roofline (the diagnostic TPS basis) ----
    r16 = lmhead_read_bytes(DEPLOYED_LMHEAD_ROWS)
    rfull = lmhead_read_bytes(FULL_VOCAB)
    c["t01_deployed_read_mb_matches_398"] = abs(r16 / 1e6 - DEPLOYED_LMHEAD_READ_MB_398) < 0.2
    c["t02_read_is_int4_plus_scales"] = abs(r16 - (DEPLOYED_LMHEAD_ROWS * HIDDEN * 0.5
                                                   + DEPLOYED_LMHEAD_ROWS * 2)) < 1.0
    c["t03_read_proportional_to_rows"] = abs(lmhead_read_bytes(2 * DEPLOYED_LMHEAD_ROWS) - 2 * r16) < 1.0
    c["t04_full_reads_more_than_deployed"] = rfull > r16
    c["t05_deployed_roofline_matches_283"] = abs(roofline_tps(r16 / 1e9) - DEPLOYED_ROOFLINE_TPS_283) < 0.05
    c["t06_roofline_monotonic_decreasing"] = roofline_tps(rfull / 1e9) < roofline_tps(r16 / 1e9)

    # ---- softcap monotonic (argmax-preserving) ----
    xs = np.array([-100.0, -5.0, -0.3, 0.0, 0.7, 5.0, 250.0])
    c["t07_softcap_monotonic"] = bool(np.all(np.diff(softcap(xs)) > 0))
    perm = np.array([2, 0, 1, 6, 3, 5, 4])
    c["t08_softcap_preserves_argmax"] = int(np.argmax(xs[perm])) == int(np.argmax(softcap(xs[perm])))

    # ---- separation certificate (the reachability backbone) on a 2D toy ----
    W = np.array([[1.0, 0.0], [-1.0, 0.7], [-1.0, -0.7], [-0.3, 0.0], [3.0, 0.0], [0.2, 0.95], [0.0, 0.0]])
    Wk = W[np.array([0, 1, 2])]
    c["t09_inside_hull_not_reachable"] = (separation_reachable(Wk, W[3]) is False)
    c["t10_outside_hull_reachable"] = (separation_reachable(Wk, W[4]) is True)
    c["t11_edge_outside_reachable"] = (separation_reachable(Wk, W[5]) is True)
    c["t12_zero_row_not_reachable"] = (separation_reachable(Wk, W[6]) is False)

    # ---- held-out clip-rate / distinct-clip logic (synthetic emissions + keepset) ----
    keep = list(range(10))
    s = set(keep)
    emissions = [0, 1, 2, 11, 12, 11, 5, 200]          # 4 of 8 positions outside {0..9}; distinct ook {11,12,200}
    clip_pos = sum(1 for t in emissions if t not in s)
    distinct_ook = sorted(set(t for t in emissions if t not in s))
    c["t13_clip_rate_fraction_correct"] = abs(clip_pos / len(emissions) - 4 / 8) < TOL
    c["t14_distinct_clipped_correct"] = distinct_ook == [11, 12, 200]

    # ---- min-keepset union assembly (kept + certified + held-out + official), exact via mask ----
    keep_ids = list(range(8))                           # deployed kept = 8 (toy vocab_size=64)
    mk = np.zeros(64, dtype=bool); mk[keep_ids] = True
    mk[[20, 21, 22, 23, 24]] = True                     # 5 separation-certified pruned rows
    sep = {"available": True, "_must_keep_mask": mk,
           "provably_reachable_pruned_separation": 5, "provably_unreachable_rows": 0}
    heldout = {"available": True, "held_out_clip_rate": 0.25, "held_out_distinct_ids_clipped": 2,
               "heldout_total_positions": 100, "distinct_argmax_ids": [0, 1, 20, 30],  # 30 is NEW
               "reached_50k_target": False}
    official = {"available": True, "distinct_reachable_rows_official": 4, "official_distinct_ids_clipped": 1,
                "official_clip_rate_bf16": 0.01, "official_distinct_argmax_ids": [0, 2, 25]}  # 25 is NEW
    minks = compute_min_identity_safe_keepset(keep_ids, sep, heldout, official, vocab_size=64)
    # union = {0..7} u {20,21,22,23,24} u {30} u {25} = 8 + 5 + 1 + 1 = 15
    c["t15_min_keepset_union_exact"] = minks["min_identity_safe_keepset_size"] == 15
    c["t16_min_keepset_exceeds_deployed"] = minks["min_keepset_exceeds_deployed"] is True
    c["t17_min_keepset_used_provable_mask"] = minks["min_keepset_used_provable_mask"] is True
    c["t18_overflow_rows_positive"] = minks["min_keepset_overflow_rows"] == 15 - 8
    # degrade: no provable mask -> fold the #406 anchor as a floor (>= kept + anchor)
    sep_nomask = {"available": False}
    minks2 = compute_min_identity_safe_keepset(keep_ids, sep_nomask, heldout, official, vocab_size=64)
    c["t19_anchor_floor_when_no_mask"] = minks2["min_identity_safe_keepset_size"] >= 8 + N406_SEPARATION_REACHABLE_PRUNED

    # ---- TPS-cost model (widening 16384 -> min_keepset costs head-read TPS) ----
    tps_full = price_truevocab_tps_cost(FULL_VOCAB)
    c["t20_cost_equals_deployed_minus_widened"] = abs(
        tps_full["truevocab_lmhead_tps_cost"]
        - (tps_full["deployed_equiv_TPS"] - tps_full["TPS_at_widened_head"])) < 1e-9
    c["t21_widening_costs_positive_tps"] = tps_full["truevocab_lmhead_tps_cost"] > 0.0
    c["t22_deployed_equiv_matches_roofline"] = abs(tps_full["deployed_equiv_TPS"] - DEPLOYED_ROOFLINE_TPS_283) < 0.05
    c["t23_widened_tps_below_deployed"] = tps_full["TPS_at_widened_head"] < tps_full["deployed_equiv_TPS"]
    # widening to a SMALLER target (<=16384) costs 0 (floored at deployed rows).
    tps_noop = price_truevocab_tps_cost(1000)
    c["t24_no_cost_when_not_widening"] = abs(tps_noop["truevocab_lmhead_tps_cost"]) < 1e-9

    # ---- verdict gating: self-referential PASS, absolute FAIL when clip>0; counterfactual ----
    keep_ids = list(range(DEPLOYED_LMHEAD_ROWS))
    sep_v = {"available": True, "provably_unreachable_rows": 0, "provably_reachable_pruned_separation": 245592}
    held_fail = {"available": True, "held_out_clip_rate": 0.1, "held_out_distinct_ids_clipped": 100,
                 "heldout_total_positions": 60000, "reached_50k_target": True}
    minks_v = {"min_identity_safe_keepset_size": 261976, "min_keepset_exceeds_deployed": True,
               "min_keepset_overflow_rows": 245592, "min_keepset_fraction_of_full_vocab": 261976 / FULL_VOCAB}
    tps_v = price_truevocab_tps_cost(261976)
    off_v = {"available": True, "official_distinct_ids_clipped": 548, "official_clip_rate_bf16": 0.0102}
    v = assemble_verdict(keep_ids, sep_v, held_fail, off_v, minks_v, tps_v)
    c["t25_self_referential_passes"] = v["deployed_passes_self_referential"] is True
    c["t26_absolute_fails_when_clip"] = v["deployed_passes_absolute"] is False
    c["t27_gate_is_self_referential"] = v["gate_for_respect_equivalence"] == "self_referential"
    # counterfactual: zero held-out clip AND no official clip -> absolute would PASS
    held_ok = {"available": True, "held_out_clip_rate": 0.0, "held_out_distinct_ids_clipped": 0,
               "heldout_total_positions": 60000, "reached_50k_target": True}
    off_ok = {"available": True, "official_distinct_ids_clipped": 0, "official_clip_rate_bf16": 0.0}
    v2 = assemble_verdict(keep_ids, sep_v, held_ok, off_ok, minks_v, tps_v)
    c["t28_absolute_passes_when_no_clip"] = v2["deployed_passes_absolute"] is True

    # ---- constants ----
    c["t29_constants_exact"] = bool(HIDDEN == 2560 and FULL_VOCAB == 262144
                                    and DEPLOYED_LMHEAD_ROWS == 16384
                                    and abs(FINAL_LOGIT_SOFTCAP - 30.0) < TOL)

    passes = bool(all(c.values()))
    return {"conditions": c, "n_checks": len(c), "truevocab_lmhead_equivalence_self_test_passes": passes}


# ---------------------------------------------------------------------------- #
# Synthesis / report / W&B (house pattern).
# ---------------------------------------------------------------------------- #
def synthesize(run_measurements: bool) -> dict[str, Any]:
    st = self_test()
    k16, keepset_path = _load_keepset(KEEPSET_16K_CANDIDATES)
    official = analyze_official_reachability()
    sep = separation_certificate() if run_measurements else {"available": False, "note": "skipped (self-test only)"}
    heldout = heldout_probe() if run_measurements else {"available": False, "note": "skipped"}
    minks = compute_min_identity_safe_keepset(k16, sep, heldout, official)
    tps = price_truevocab_tps_cost(minks["min_identity_safe_keepset_size"])
    verdict = assemble_verdict(k16, sep, heldout, official, minks, tps)
    verdict["truevocab_lmhead_equivalence_self_test_passes"] = st["truevocab_lmhead_equivalence_self_test_passes"]

    # strip the in-process-only mask before serialization.
    sep_ser = {k: v for k, v in sep.items() if k != "_must_keep_mask"}
    # keep the big id lists out of the top-level payload (compact the heldout/official records).
    heldout_ser = {k: v for k, v in heldout.items()
                   if k not in ("distinct_argmax_ids", "distinct_argmax_ids_outside_keepset")}
    if heldout.get("available"):
        heldout_ser["distinct_argmax_ids_count"] = len(heldout.get("distinct_argmax_ids", []))
    official_ser = {k: v for k, v in official.items()
                    if k not in ("official_distinct_argmax_ids", "official_distinct_argmax_ids_outside_keepset")}
    return {
        "self_test": st,
        "truevocab_lmhead_equivalence_self_test_passes": st["truevocab_lmhead_equivalence_self_test_passes"],
        "n_self_test_checks": st["n_checks"],
        "keepset_path": keepset_path,
        "official_reachability": official_ser,
        "separation_certificate": sep_ser,
        "heldout_probe": heldout_ser,
        "min_keepset": minks,
        "tps_cost": tps,
        "verdict_fields": verdict,
        "verdict": _build_verdict(verdict, sep, heldout, official, minks, tps),
        "analysis_only": True,
        "no_hf_job": True,
        "no_served_file_change": True,
        "official_tps": 0,
    }


def _build_verdict(v, sep, heldout, official, minks, tps) -> str:
    parts = [
        "Q (PR #414): what does TRUE full-vocab lm_head equivalence cost in TPS? VERDICT: "
        "deployed_passes_self_referential={}, deployed_passes_absolute={}, held_out_clip_rate={:.4f}, "
        "min_identity_safe_keepset_size={} ({:.1%} of the 262144 vocab), truevocab_lmhead_tps_cost={:.2f} TPS "
        "({:.1%} of the deployed head-read roofline).".format(
            v["deployed_passes_self_referential"], v["deployed_passes_absolute"], v["held_out_clip_rate"],
            v["min_identity_safe_keepset_size"], v["min_keepset_fraction_of_full_vocab"],
            v["truevocab_lmhead_tps_cost"], v["truevocab_lmhead_tps_cost_frac"]),
        "SELF-REFERENTIAL: the deployed pruned head matches its OWN plain greedy AR with 0 out-of-keepset "
        "emissions BY CONSTRUCTION (#406) -> the official scorer's operative gate PASSES at 0 cost.",
    ]
    if heldout.get("available"):
        parts.append(
            "ABSOLUTE (held-out, {} positions across {} NL+code+multilingual prompts via full bf16 greedy AR): "
            "{:.3f}% of positions ({} distinct ids) have a TRUE full-vocab greedy argmax OUTSIDE the 16384 "
            "keepset -> the deployed config silently diverges from true gemma greedy at this rate on held-out "
            "data (the +inf-PPL exposure surface on a private eval).".format(
                heldout.get("heldout_total_positions"), heldout.get("heldout_prompts"),
                100.0 * heldout.get("held_out_clip_rate", 0.0), heldout.get("held_out_distinct_ids_clipped")))
    else:
        parts.append(
            "ABSOLUTE (held-out probe unavailable; #406 anchors): small-OOD clip {:.2f}% and official-128 clip "
            "{:.3f}% ({} distinct ids) already show full-vocab greedy OVERFLOWS the keepset.".format(
                100.0 * N406_OOD_SMALL_CLIP_RATE, 100.0 * N406_OFFICIAL_CLIP_RATE_BF16, N406_OFFICIAL_OOK_IDS))
    parts.append(
        "MIN KEEPSET: union(deployed kept, separation-certified reachable pruned, held-out + official greedy "
        "argmax ids) = {} rows. Because provably_unreachable_rows={} and {} pruned rows are separation-certified "
        "reachable, ABSOLUTE identity needs essentially the WHOLE 262144-row head (a larger corpus only grows "
        "the observed support; the provable component already forces near-full-vocab).".format(
            v["min_identity_safe_keepset_size"], v["provably_unreachable_rows"],
            v["provably_reachable_pruned_separation"]))
    parts.append(
        "TPS COST: widening 16384 -> {} rows on the #398/#283 head-read roofline (read proportional to rows, "
        "int4 head format held fixed) drops the head-read roofline {:.2f} -> {:.2f} TPS = {:.2f} TPS lost per "
        "token. DIAGNOSTIC read-model delta; prices ONLY the truncation dimension (the head's int4-vs-bf16 "
        "quant identity is a separate additive leg, #371).".format(
            tps["widened_rows"], tps["deployed_equiv_TPS"], tps["TPS_at_widened_head"],
            tps["truevocab_lmhead_tps_cost"]))
    parts.append(
        "FOR THE HUMANS: 'the fastest implementation that respects equivalence' is correctly measured against "
        "the SELF-REFERENTIAL gate (deployed + any in-keepset speculator satisfy it for free); demanding "
        "ABSOLUTE (true full-vocab) equivalence re-prices the head ~16x wider and forfeits ~{:.0f} TPS/token "
        "for no benefit to the official self-referential scorer. LOCAL/analysis-only; 0 official TPS; NO HF "
        "Job / submission / served-file change.".format(tps["truevocab_lmhead_tps_cost"]))
    return " ".join(parts)


def _assert_nan_clean(payload: dict, path: str = "payload") -> list[str]:
    bad: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            for k, val in node.items():
                walk(val, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, val in enumerate(node):
                walk(val, f"{p}[{i}]")
        elif isinstance(node, float) and not math.isfinite(node):
            bad.append(p)

    walk(payload, path)
    return bad


def _print_report(syn: dict) -> None:
    st = syn["self_test"]
    v = syn["verdict_fields"]
    print("\n" + "=" * 100, flush=True)
    print("TRUE full-vocab lm_head EQUIVALENCE COST (PR #414)", flush=True)
    print("=" * 100, flush=True)
    print(f"  (PRIMARY) truevocab_lmhead_equivalence_self_test_passes = "
          f"{st['truevocab_lmhead_equivalence_self_test_passes']} ({st['n_checks']} checks)", flush=True)
    fail = [k for k, val in st["conditions"].items() if not val]
    if fail:
        print(f"          FAILED: {fail}", flush=True)
    print("-" * 100, flush=True)
    print("  --- PR #414 HEADLINE FIELDS ---", flush=True)
    for k in ("deployed_passes_self_referential", "deployed_passes_absolute", "held_out_clip_rate",
              "held_out_distinct_ids_clipped", "held_out_positions", "heldout_source",
              "min_identity_safe_keepset_size", "min_keepset_exceeds_deployed", "min_keepset_overflow_rows",
              "min_keepset_fraction_of_full_vocab", "truevocab_lmhead_tps_cost", "truevocab_lmhead_tps_cost_frac",
              "deployed_equiv_TPS", "TPS_at_widened_head", "widened_rows", "provably_unreachable_rows",
              "provably_reachable_pruned_separation", "official_distinct_ids_clipped"):
        print(f"  {k:<42} = {v.get(k)}", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {syn['verdict']}", flush=True)
    print("=" * 100, flush=True)


def _maybe_log_wandb(args, payload: dict) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[truevocab-lmhead] wandb logging unavailable: {exc}", flush=True)
        return

    syn = payload["synthesis"]
    st = syn["self_test"]
    v = syn["verdict_fields"]
    sep = syn["separation_certificate"]
    official = syn["official_reachability"]
    heldout = syn["heldout_probe"]

    def _num(x):
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None

    summary: dict[str, Any] = {
        "truevocab_lmhead_equivalence_self_test_passes": int(bool(st["truevocab_lmhead_equivalence_self_test_passes"])),
        "n_self_test_checks": st["n_checks"],
        **{f"selftest_{k}": int(bool(val)) for k, val in st["conditions"].items()},
        "nan_clean": int(bool(payload["nan_clean"])),
        "peak_mem_mib": payload["peak_mem_mib"],
        # ---- PR #414 headline keys ----
        "held_out_clip_rate": v["held_out_clip_rate"],
        "held_out_distinct_ids_clipped": v["held_out_distinct_ids_clipped"],
        "held_out_positions": v["held_out_positions"],
        "min_identity_safe_keepset_size": v["min_identity_safe_keepset_size"],
        "min_keepset_exceeds_deployed": int(bool(v["min_keepset_exceeds_deployed"])),
        "min_keepset_overflow_rows": v["min_keepset_overflow_rows"],
        "min_keepset_fraction_of_full_vocab": v["min_keepset_fraction_of_full_vocab"],
        "truevocab_lmhead_tps_cost": v["truevocab_lmhead_tps_cost"],
        "truevocab_lmhead_tps_cost_frac": v["truevocab_lmhead_tps_cost_frac"],
        "deployed_passes_self_referential": int(bool(v["deployed_passes_self_referential"])),
        "deployed_passes_absolute": int(bool(v["deployed_passes_absolute"])),
        "deployed_rows_kept": v["deployed_rows_kept"],
        "deployed_equiv_TPS": v["deployed_equiv_TPS"],
        "TPS_at_widened_head": v["TPS_at_widened_head"],
        "widened_rows": v["widened_rows"],
        "provably_unreachable_rows": v["provably_unreachable_rows"],
        "provably_reachable_pruned_separation": v["provably_reachable_pruned_separation"],
        "official_distinct_ids_clipped": v["official_distinct_ids_clipped"],
        "official_clip_rate_bf16": v["official_clip_rate_bf16"],
        "separation_certificate_available": int(bool(sep.get("available"))),
        "heldout_probe_available": int(bool(heldout.get("available"))),
        "heldout_reached_50k": int(bool(heldout.get("reached_50k_target"))) if heldout.get("available") else 0,
        "official_reachability_available": int(bool(official.get("available"))),
        "analysis_only": int(True), "no_hf_job": int(True), "no_served_file_change": int(True),
        "official_tps": 0,
    }
    for src in (official, sep, heldout, syn["min_keepset"], syn["tps_cost"]):
        if isinstance(src, dict):
            for k, val in src.items():
                n = _num(val)
                if n is not None and k not in summary:
                    summary[f"m_{k}"] = n
    summary = {k: val for k, val in summary.items()
               if val is not None and not (isinstance(val, float) and not math.isfinite(val))}

    run = init_wandb_run(
        job_type="validity-gate", agent="land", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "lm_head-truncation", "truevocab-equivalence", "self-referential-vs-absolute",
              "greedy-identity", "held-out-clip-rate", "min-identity-safe-keepset", "tps-cost", "analysis-only",
              "bank-the-analysis", "pr-414"],
        config={
            "hidden": HIDDEN, "full_vocab": FULL_VOCAB, "deployed_lmhead_rows": DEPLOYED_LMHEAD_ROWS,
            "a10g_bw_gbps": A10G_BW_GBPS, "body_int4_gb": BODY_INT4_GB,
            "final_logit_softcap": FINAL_LOGIT_SOFTCAP, "rms_norm_eps": RMS_NORM_EPS,
            "deployed_roofline_tps_283": DEPLOYED_ROOFLINE_TPS_283, "official_frontier_tps": OFFICIAL_FRONTIER_TPS,
            "ppl_gate": PPL_GATE, "ppl_baseline": PPL_BASELINE, "milestone_tps": MILESTONE,
            "heldout_target_positions": HELDOUT_TARGET_POSITIONS, "wandb_group": args.wandb_group,
        },
    )
    if run is None:
        print("[truevocab-lmhead] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="truevocab_lmhead_equivalence_cost_result",
                      artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[truevocab-lmhead] wandb logged: self_ref={v['deployed_passes_self_referential']} "
          f"absolute={v['deployed_passes_absolute']} clip={v['held_out_clip_rate']:.4f} "
          f"min_keepset={v['min_identity_safe_keepset_size']} tps_cost={v['truevocab_lmhead_tps_cost']:.2f}",
          flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the PRIMARY pure-logic self-validation only (numpy-only; no torch/GPU)")
    ap.add_argument("--no-measurements", action="store_true",
                    help="skip the torch/vLLM separation + held-out enrichments (self-test + 0-GPU official only)")
    ap.add_argument("--heldout-probe-only", action="store_true",
                    help="ONLY run the vLLM held-out greedy probe and write the sidecar, then exit "
                         "(isolates the 16GB model load from the torch separation certificate)")
    ap.add_argument("--heldout-max-tokens", type=int, default=512)
    ap.add_argument("--force-heldout", action="store_true", help="ignore the cached held-out sidecar")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="truevocab-lmhead-equivalence-cost")
    args = ap.parse_args(argv)

    if args.heldout_probe_only:
        res = heldout_probe(max_tokens=args.heldout_max_tokens, force=args.force_heldout)
        print(json.dumps({k: val for k, val in res.items()
                          if k not in ("distinct_argmax_ids", "distinct_argmax_ids_outside_keepset")},
                         indent=2, default=float), flush=True)
        ok = bool(res.get("available"))
        print(f"[truevocab-lmhead] held-out probe {'OK' if ok else 'FAILED'}: "
              f"positions={res.get('heldout_total_positions')} clip_rate={res.get('held_out_clip_rate')} "
              f"reached_50k={res.get('reached_50k_target')}", flush=True)
        return 0 if ok else 1

    run_measurements = not (args.self_test or args.no_measurements)
    syn = synthesize(run_measurements=run_measurements)

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = {
        "created_at": created_at, "pr": 414, "agent": "land",
        "kind": "truevocab-lmhead-equivalence-cost", "synthesis": syn,
        "peak_mem_mib": round(peak_kib / 1024.0, 3),
    }
    nan_paths = _assert_nan_clean(payload)
    payload["nan_clean"] = not nan_paths
    if nan_paths:
        print(f"[truevocab-lmhead] WARNING non-finite values at: {nan_paths}", flush=True)

    _print_report(syn)

    out_dir = args.out_dir or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "truevocab_lmhead_equivalence_cost_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[truevocab-lmhead] wrote {out_path}", flush=True)

    _maybe_log_wandb(args, payload)

    passes = bool(syn["truevocab_lmhead_equivalence_self_test_passes"]) and payload["nan_clean"]
    v = syn["verdict_fields"]
    print(f"  PRIMARY truevocab_lmhead_equivalence_self_test_passes = {passes} "
          f"({syn['n_self_test_checks']} checks)", flush=True)
    print(f"  deployed_passes_self_referential={v['deployed_passes_self_referential']} "
          f"deployed_passes_absolute={v['deployed_passes_absolute']} "
          f"truevocab_lmhead_tps_cost={v['truevocab_lmhead_tps_cost']:.2f}", flush=True)
    print(f"  peak_mem_mib = {payload['peak_mem_mib']}", flush=True)

    if args.self_test:
        print(f"[truevocab-lmhead] self-test {'PASS' if passes else 'FAIL'}", flush=True)
        return 0 if passes else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
