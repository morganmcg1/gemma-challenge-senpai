#!/usr/bin/env python
"""PR #573 fern — BOUND the spec-dec axis on base_fullhead: prompt-lookup ngram
served TPS + the acceptance-length -> TPS model + the analytical ship-clearing
acceptance threshold A_ship. LOCAL A10G, analysis-only, NO HF fire.

Complement to lawine #572 (which measures the ship's own MTP K=7 drafter, the
strongest single design point). This card supplies the FRAMEWORK that bounds ALL
drafters: the always-available prompt-lookup ngram realization + the
acceptance->TPS curve + the analytical A_ship that base_fullhead would need to
clear the 375.857 ship. Both spec-dec points then land on the same curve.

Substrate (PR instruction 1) = the quality-safe base_fullhead config: stock
base-int4 body + FULL native 262k bf16 lm_head (prune OFF). Identical head
overrides as wirbel #553 / lawine #544 / fern #566:
  LM_HEAD_PRUNE=0, LM_HEAD_PRUNE_REQUIRE=0, PCK04_KEEPSET="",
  LOCAL_MODEL_DIR/PLE_FOLD_TARGET_MODEL = own stock qat-w4a16-ct snapshot,
  PLE_FOLD_EMBED_SCALE=1.

Passes (one fresh server each; the drafter is fixed at engine init, so the only
changed variable across passes is the drafter -- the verify-side stack is
byte-identical):
  * REF       SPECULATIVE_CONFIG=""  -> plain M=1 AR. Gives t_1 (the no-spec
              per-token cost) AND the greedy completion-token oracle (identity).
  * NGRAM_kN  SPECULATIVE_CONFIG=ngram (prompt_lookup_max=2, num_spec=K). One per
              K in --ngram-ks (default 7,3). Gives realized served TPS + the
              served acceptance counters (Prometheus + vLLM server log) + the
              greedy completion-token ids (for the identity check).

All passes set DISABLE_LOG_STATS=0 (so vLLM prints SpecDecoding lines AND exposes
the /metrics spec counters) and VLLM_USE_FLASHINFER_SAMPLER=0 (torch-native
argmax, lowest-vocab-index tie-break -- the #319 / #566 convention the spec
acceptance path must inherit).

Acceptance -> TPS model (PR instruction 3), energy conservation for a
partial-coverage drafter over the warm request slice:
  W = d * t_v + u * t_1
where d = draft (verify) steps, u = non-draft M=1 steps, t_v = per-verify-step
cost (M=K+1 positions), t_1 = 1/TPS_base. Solve t_v, then for an ALWAYS-drafting
drafter TPS(A) = A / t_v. The ship-clearing acceptance MUST be a frame-invariant
RATIO -- never absolute-slow-pod-t_v x fast-harness-bar (that frame-mixes a ~3x
pod-slowness into A and inflates it ~3x). With c := t_v/t_1 (the same-pod verify
overhead, hardware-invariant) and the OFFICIAL anchor (base_fullhead no-spec):
  A_ship = c * (375.857 / anchor_official) ,  A_500 = c * (500 / anchor_official) ,
and break-even (TPS=TPS_base) needs A = c. If A_ship > K+1 (the max achievable
acceptance at K), NO drafter at that K clears the ship -> the axis is bounded for
all drafters, not just ngram. (Sanity: A_ship MUST be <= K+1 at the ship's own
K=7, since the ship is itself an MTP K=7 spec-dec point hitting 375.857.)

An exact-verify OFFLINE ngram simulator runs on the REF greedy completions (the
acceptance an exact-verify ngram drafter achieves is fully determined by the
greedy target sequence). It (a) cross-checks the served Prometheus acceptance,
(b) supplies the per-prompt-TYPE breakdown for free, (c) sweeps the #503 grid
without extra server passes.

LOCAL only: analysis_only=true, official_tps=0, NO HF Job, NO /v1/jobs:run, NO
--launch, NO submission, NO served-file change. A clear-the-ship result is an
ESCALATION (approval issue), never an auto-fire.

Run (smoke first):
  CUDA_VISIBLE_DEVICES=0 python research/base_fullhead_specdec/ngram_acceptance_model.py --smoke --no-wandb
Full:
  CUDA_VISIBLE_DEVICES=0 python research/base_fullhead_specdec/ngram_acceptance_model.py \
    --num-prompts 128 --ngram-ks 7,3 \
    --wandb_name fern/base-fullhead-ngram-specdec --wandb_group base-fullhead-specdec-ceiling
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _SCRIPT_DIR)]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submissions" / "fa2sw_strict_surgical357"
MEASURE_FLOOR = ROOT / "research" / "base_int4_floor_tps" / "measure_floor.py"
OUT_ROOT = HERE

MODEL_DIR = (
    "/senpai-run/home/student-fern/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)

HIDDEN = 2560
VOCAB = 262144
SEED = 1
NGRAM_LOOKUP_MAX = 2          # prompt_lookup_max for the SERVED ngram passes (n2: most coverage)
NGRAM_LOOKUP_MIN = 2          # prompt_lookup_min

# Cited anchors (NOT re-derived here).
ANCHOR_BASE_FULLHEAD_NOSPEC = 252.69   # wirbel #553 (83jiwjr9) base_fullhead no-spec LOCAL warm-median
SHIP_TPS = 375.857                     # official ship (the verdict-flip bar)
CAPSTONE_FLOOR = 311.25                # lawine #554 magically-free floor
GATE_500 = 500.0                       # leaderboard gate
OFFICIAL_1 = 481.53                    # public #1 (untouched baseline)

# local<->official map (deployed-ship anchor #267 / measure_floor.py): OFFICIAL = TAU_LO * LOCAL.
# Critical: the base_fullhead anchor 252.69 is a LOCAL warm-median, but the ship/floor/gate bars
# are OFFICIAL numbers. Our served TPS here is also a LOCAL warm-median, so to compare it against
# the OFFICIAL bars we either (a) lift LOCAL->OFFICIAL via TAU_LO, or (b) push the bars OFFICIAL->
# LOCAL via /TAU_LO. We do BOTH (report anchored-official headline; model A_ship in LOCAL space so
# it pairs with the LOCAL-measured per-step cost t_v). Mixing the two spaces (raw LOCAL TPS vs the
# OFFICIAL 375.857) is the ~3.5% anchoring error this map removes.
TAU_LO = 481.53 / 465.14047160458415   # = 1.035236 (#267)
SHIP_LOCAL = SHIP_TPS / TAU_LO              # official ship in LOCAL warm-median space (= 363.08)
GATE_500_LOCAL = GATE_500 / TAU_LO
CAPSTONE_FLOOR_LOCAL = CAPSTONE_FLOOR / TAU_LO
ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL = ANCHOR_BASE_FULLHEAD_NOSPEC * TAU_LO  # 261.6 OFFICIAL

# Offline-sweep grid (ubel #503 public-screen set: n in {2,3,4} x K in {3,5,7}).
SWEEP_GRID = [(n, k) for n in (2, 3, 4) for k in (3, 5, 7)]


# ========================================================================== #
# serve env
# ========================================================================== #
def build_env(*, spec_config: str, model_dir: str, relax_loopgraph: bool,
              gpu_mem_util: str, log_stats: bool = False) -> dict[str, str]:
    """base_fullhead serve recipe (full 262k head, prune OFF) + ngram/AR drafter.

    spec_config == "" -> plain M=1 AR (serve.append_env_arg no-ops on empty).

    log_stats: DISABLE_LOG_STATS=0 emits the SpecDecoding log lines + /metrics spec
    counters, BUT on this onegraph fast stack turning stats ON forces a per-decode-step
    GPU->CPU sync (to read token/accept counts) that breaks the captured-graph pipeline
    and runs ~3x slower (full-head REF 252->84 LOCAL; cross-checked vs #553 stats-OFF=252
    and fern #566 stats-OFF=299, both this-pod, full head). So TIMED passes keep stats OFF
    (comparable TPS); served acceptance is taken from a separate stats-ON pass or the
    offline exact-greedy-verify sim. Default OFF.
    """
    env: dict[str, str] = {
        # --- base_fullhead head overrides (quality-safe substrate) ---
        "LM_HEAD_PRUNE": "0",
        "LM_HEAD_PRUNE_REQUIRE": "0",
        "PCK04_KEEPSET": "",
        "LOCAL_MODEL_DIR": model_dir,
        "PLE_FOLD_TARGET_MODEL": model_dir,
        "PLE_FOLD_EMBED_SCALE": "1",
        "LM_HEAD_FULL_REQUIRE": "1",
        "GPU_MEMORY_UTILIZATION": gpu_mem_util,
        # --- drafter (the ONLY changed variable across passes) ---
        "SPECULATIVE_CONFIG": spec_config,
        # --- measurement plumbing ---
        "DISABLE_LOG_STATS": "0" if log_stats else "1",  # OFF for timed passes (see docstring)
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # torch argmax, lowest-index tie-break (#319/#566)
    }
    if relax_loopgraph:
        # Static-capture / fast-path REQUIRE guards are tuned for the deployed
        # MTP K=7 (M=8). ngram's variable M (1..K+1) or the M=1 REF can trip a
        # hard REQUIRE. Relaxing only drops the *assertion* that a fast path was
        # taken (not correctness); applied to BOTH REF and NGRAM so the M=1 step
        # cost stays identical and the energy identity holds apples-to-apples.
        env["LOOPGRAPH_REQUIRE_CAPTURE"] = "0"
        env["FUSED_SPARSE_ARGMAX_REQUIRE"] = "0"
        env["DIXIE_FUSED_ACCEPT_PREP_REQUIRE"] = "0"
        env["PRECACHE_REQUIRE"] = "0"
    return env


def ngram_spec_config(k: int, lookup_max: int = NGRAM_LOOKUP_MAX,
                      lookup_min: int = NGRAM_LOOKUP_MIN) -> str:
    return json.dumps({
        "method": "ngram", "num_speculative_tokens": k,
        "prompt_lookup_max": lookup_max, "prompt_lookup_min": lookup_min,
    })


# ========================================================================== #
# prompt typing (cheap per-type breakdown, PR instruction 2)
# ========================================================================== #
def classify_prompt(text: str) -> str:
    t = text or ""
    code_markers = ("```", "def ", "import ", "class ", "function ", "#include",
                    "public static", "</", "/>")
    brace = t.count("{") + t.count("}") + t.count(";")
    if any(m in t for m in code_markers) or brace >= 8:
        return "code"
    if len(t) >= 2000:
        return "long_prose"
    return "short_prose"


# ========================================================================== #
# token-capturing decode worker (runs UNDER the server venv)
# ========================================================================== #
def _decode_worker(args: argparse.Namespace) -> int:
    from scripts.local_validation import paths  # noqa: E402

    spec = importlib.util.spec_from_file_location("official_decode", str(paths.DECODE_SCRIPT))
    od = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(od)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = od.read_sharegpt_prompts(Path(args.dataset_path), num_prompts=args.num_prompts, seed=args.seed)
    if len(records) != args.num_prompts:
        raise ValueError(f"expected {args.num_prompts} prompts, found {len(records)}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        t0 = time.perf_counter()
        prompt_token_ids = od.encode_prompt(tok, record["prompt_text"])
        t1 = time.perf_counter()
        response = od.request_decode(
            base_url=args.base_url, model=args.model,
            prompt_token_ids=prompt_token_ids, output_len=args.output_len,
            timeout_s=args.request_timeout_s,
        )
        t2 = time.perf_counter()
        choice = od.choice_from_response(response)
        completion_token_ids, _, _ = od.extract_generated_token_ids(response, choice, prompt_token_ids)
        rows.append({
            "index": index,
            "t_tokenize_s": t1 - t0,
            "t_request_s": t2 - t1,
            "num_prompt_tokens": len(prompt_token_ids),
            "num_completion_tokens": len(completion_token_ids),
            "completion_token_ids": list(completion_token_ids),
            "prompt_token_ids": list(prompt_token_ids),
            "prompt_type": classify_prompt(record.get("prompt_text", "")),
        })
        print(f"[ng-worker] {index + 1}/{len(records)} req_ms={1000.0 * (t2 - t1):.1f} "
              f"comp={len(completion_token_ids)} prompt={len(prompt_token_ids)}", flush=True)

    out = {"output_len": args.output_len, "num_records": len(records), "per_request": rows}
    Path(args.out_file).write_text(json.dumps(out))
    return 0


# ========================================================================== #
# exact-verify OFFLINE ngram simulator (greedy target = REF completions)
# ========================================================================== #
def _ngram_propose(context: list[int], n_max: int, n_min: int, k: int) -> list[int]:
    """Most-recent earlier-occurrence prompt-lookup proposal (vLLM NgramProposer
    semantics): longest pattern first; propose the k tokens following the latest
    earlier match. [] if no match (no draft this step)."""
    L = len(context)
    for n in range(n_max, n_min - 1, -1):
        if L < n + 1:
            continue
        pattern = context[L - n:]
        for s in range(L - n - 1, -1, -1):
            if context[s:s + n] == pattern:
                draft = context[s + n:s + n + k]
                if draft:
                    return draft
        # else: shorter n
    return []


def simulate_ngram_sequence(prompt_ids: list[int], completion_ids: list[int],
                            n_max: int, n_min: int, k: int) -> dict[str, Any]:
    """Walk the greedy completion; at each emit position propose via prompt-lookup
    over (prompt + generated-so-far) and accept the longest common prefix with the
    actual greedy continuation (exact greedy verify). Returns draft/accept counts."""
    context = list(prompt_ids)
    G = completion_ids
    i = 0
    draft_steps = 0
    no_draft_steps = 0
    accepted = 0
    n = len(G)
    while i < n:
        draft = _ngram_propose(context, n_max, n_min, k)
        if not draft:
            no_draft_steps += 1
            context.append(G[i])
            i += 1
            continue
        # exact greedy verify: longest common prefix of draft and G[i:i+k]
        a = 0
        cap = min(len(draft), n - i)
        while a < cap and draft[a] == G[i + a]:
            a += 1
        draft_steps += 1
        accepted += a
        emit = a + 1                       # a accepted + 1 bonus (corrected/next greedy token)
        emit = min(emit, n - i)            # clamp at sequence end
        context.extend(G[i:i + emit])
        i += emit
    tokens = n
    e_accept = (1.0 + accepted / draft_steps) if draft_steps else float("nan")
    coverage = draft_steps / (draft_steps + no_draft_steps) if (draft_steps + no_draft_steps) else float("nan")
    return {
        "tokens": tokens, "draft_steps": draft_steps, "no_draft_steps": no_draft_steps,
        "accepted": accepted, "e_accept": e_accept, "coverage": coverage,
    }


def offline_sim_pass(ref_rows: list[dict], n_max: int, n_min: int, k: int,
                     warm: int) -> dict[str, Any]:
    """Aggregate the exact-verify ngram sim over the warm REF records; also bucket
    e_accept/coverage per prompt-type."""
    warm_rows = ref_rows[warm:]
    agg = {"tokens": 0, "draft_steps": 0, "no_draft_steps": 0, "accepted": 0}
    by_type: dict[str, dict[str, int]] = {}
    per_seq: list[dict[str, Any]] = []
    for r in warm_rows:
        s = simulate_ngram_sequence(r["prompt_token_ids"], r["completion_token_ids"], n_max, n_min, k)
        per_seq.append({"index": r["index"], "prompt_type": r.get("prompt_type"),
                        "e_accept": s["e_accept"], "coverage": s["coverage"],
                        "draft_steps": s["draft_steps"], "accepted": s["accepted"]})
        for key in agg:
            agg[key] += s[key]
        t = r.get("prompt_type", "unknown")
        bt = by_type.setdefault(t, {"tokens": 0, "draft_steps": 0, "no_draft_steps": 0,
                                    "accepted": 0, "n_seq": 0})
        for key in ("tokens", "draft_steps", "no_draft_steps", "accepted"):
            bt[key] += s[key]
        bt["n_seq"] += 1
    d = agg["draft_steps"]
    u = agg["no_draft_steps"]
    type_breakdown = {}
    for t, bt in by_type.items():
        dd = bt["draft_steps"]
        type_breakdown[t] = {
            "n_seq": bt["n_seq"], "tokens": bt["tokens"],
            "e_accept": (1.0 + bt["accepted"] / dd) if dd else float("nan"),
            "coverage": dd / (dd + bt["no_draft_steps"]) if (dd + bt["no_draft_steps"]) else float("nan"),
            "draft_steps": dd, "accepted": bt["accepted"],
        }
    return {
        "n_max": n_max, "n_min": n_min, "k": k,
        "tokens": agg["tokens"], "draft_steps": d, "no_draft_steps": u, "accepted": agg["accepted"],
        "e_accept": (1.0 + agg["accepted"] / d) if d else float("nan"),
        "coverage": d / (d + u) if (d + u) else float("nan"),
        "type_breakdown": type_breakdown,
        "per_seq": per_seq,
    }


# ========================================================================== #
# server pass
# ========================================================================== #
def _fetch_metrics(base_url: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return f"__metrics_error__ {exc!r}"


def run_pass(mf: Any, harness: Any, paths: Any, serve_profile: Any, *, server_python: Path,
             label: str, extra_env: dict[str, str], num_prompts: int, output_len: int,
             port: int, request_timeout_s: int, n_decodes: int = 1) -> dict[str, Any]:
    """Boot one server with extra_env, run n_decodes identical decode passes on it
    (n_decodes=2 gives a same-boot self-determinism control), capture per-pass
    warm TPS + token ids, and the post-run spec counters (Prometheus + server log)."""
    log_path = OUT_ROOT / f"server_{label}.log"

    worker_env = os.environ.copy()
    worker_env.pop("PYTHONPATH", None)
    worker_env["VIRTUAL_ENV"] = str(server_python.parent.parent)
    worker_env["PATH"] = f"{server_python.parent}{os.pathsep}{worker_env.get('PATH', '')}"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    worker_env["PYTHONSAFEPATH"] = "1"

    peak = {"mib": 0.0}
    stop = threading.Event()

    def _sample_vram() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                )
                vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "").isdigit()]
                if vals:
                    peak["mib"] = max(peak["mib"], max(vals))
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(2.0)

    sampler = threading.Thread(target=_sample_vram, daemon=True)
    sampler.start()
    measured: dict[str, Any] = {"label": label, "num_prompts": num_prompts, "output_len": output_len}
    metrics_text = ""
    try:
        with harness.LocalServer(
            SUBMISSION, server_python=server_python, port=port,
            startup_timeout_s=1800, log_path=log_path, extra_env=extra_env,
        ) as srv:
            measured["model_id"] = srv.model_id
            measured["served_model_name"] = srv.served_model_name
            print(f"[ng] [{label}] warming server ({srv.base_url})", flush=True)
            mf._warm_server(srv.base_url, srv.served_model_name, n=mf.WARMUP_REQUESTS)
            decodes: list[dict[str, Any]] = []
            for di in range(n_decodes):
                pass_file = OUT_ROOT / f"{label}_pass{di}.json"
                cmd = [
                    str(server_python), str(Path(__file__).resolve()), "--decode-worker",
                    "--base-url", srv.base_url, "--model", srv.served_model_name,
                    "--dataset-path", str(paths.EVAL_PROMPTS), "--tokenizer", paths.TOKENIZER,
                    "--num-prompts", str(num_prompts), "--output-len", str(output_len),
                    "--seed", str(SEED), "--out-file", str(pass_file),
                    "--request-timeout-s", str(request_timeout_s),
                ]
                print(f"[ng] [{label}] decode {di + 1}/{n_decodes} {num_prompts}x{output_len} "
                      f"conc=1 -> {pass_file}", flush=True)
                subprocess.run(cmd, check=True, timeout=7200, env=worker_env)
                summary = json.loads(pass_file.read_text())
                try:
                    agg = mf._aggregate(summary)
                except Exception as exc:  # noqa: BLE001
                    agg = {"warm_median_tps": float("nan"), "aggregate_error": repr(exc)}
                decodes.append({"per_request": summary["per_request"], "tps": agg})
            metrics_text = _fetch_metrics(srv.base_url)
            measured["decodes"] = decodes
            measured["tps"] = decodes[0]["tps"]
            measured["per_request"] = decodes[0]["per_request"]
    finally:
        stop.set()
        sampler.join(timeout=5)

    log_text = ""
    try:
        log_text = Path(log_path).read_text(errors="ignore")
    except OSError:
        pass
    measured["prom_spec"] = serve_profile.parse_spec_metrics(metrics_text) if metrics_text else {}
    measured["spec_log"] = serve_profile.parse_spec_log(log_text) if log_text else {}
    measured["acceptance"] = _resolve_acceptance(measured["prom_spec"], measured["spec_log"])
    measured["peak_vram_gb"] = (peak["mib"] or 0.0) / 1024.0
    measured["log_path"] = str(log_path)
    return measured


def _resolve_acceptance(prom: dict[str, Any], slog: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    acc = prom.get("num_accepted_tokens")
    drf = prom.get("num_draft_tokens")
    nd = prom.get("num_drafts")
    if acc is not None and drf:
        out["acceptance_rate"] = acc / drf
        out["e_accept"] = prom.get("e_accept_mean_acceptance_length")
        out["num_drafts"] = nd
        out["accepted_tokens"] = acc
        out["draft_tokens"] = drf
        out["source"] = "prometheus"
    elif slog.get("draft_acceptance_rate") is not None:
        out["acceptance_rate"] = slog.get("draft_acceptance_rate")
        out["e_accept"] = slog.get("e_accept_exact") or slog.get("e_accept_interval_mean")
        out["accepted_tokens"] = slog.get("total_accepted_tokens")
        out["draft_tokens"] = slog.get("total_drafted_tokens")
        out["num_speculative_tokens"] = slog.get("num_speculative_tokens")
        out["source"] = "server_log"
    else:
        out["acceptance_rate"] = None
        out["e_accept"] = slog.get("e_accept_interval_mean")
        out["source"] = "none"
    out["steady_gen_tps"] = slog.get("steady_gen_tps_mean")
    return out


def grep_log(log_path: str, needles: list[str]) -> dict[str, bool]:
    try:
        text = Path(log_path).read_text(errors="ignore")
    except OSError:
        return {n: False for n in needles}
    return {n: (n in text) for n in needles}


# ========================================================================== #
# identity (REF vs NGRAM completion token ids)  [copied from #566 standard]
# ========================================================================== #
def compare_identity(ref_rows: list[dict], var_rows: list[dict]) -> dict[str, Any]:
    by_ref = {r["index"]: r for r in ref_rows}
    by_var = {r["index"]: r for r in var_rows}
    common = sorted(set(by_ref) & set(by_var))
    seq_exact = 0
    total_tokens = 0
    matched_tokens = 0
    matched_prefix_tokens = 0
    first_divergences: list[dict] = []
    for idx in common:
        a = by_ref[idx]["completion_token_ids"]
        b = by_var[idx]["completion_token_ids"]
        n = min(len(a), len(b))
        total_tokens += max(len(a), len(b))
        matched_tokens += sum(1 for i in range(n) if a[i] == b[i])
        div = next((i for i in range(n) if a[i] != b[i]), None)
        if div is None and len(a) == len(b):
            seq_exact += 1
            matched_prefix_tokens += n
        else:
            dpos = div if div is not None else n
            matched_prefix_tokens += dpos
            if len(first_divergences) < 16:
                first_divergences.append({
                    "index": idx, "first_divergence_token": dpos,
                    "len_ref": len(a), "len_var": len(b),
                    "ref_tok": a[dpos] if dpos < len(a) else None,
                    "var_tok": b[dpos] if dpos < len(b) else None,
                })
    return {
        "n_sequences": len(common),
        "n_sequences_byte_exact": seq_exact,
        "sequence_exact_rate": seq_exact / len(common) if common else None,
        "total_tokens": total_tokens,
        "matched_tokens": matched_tokens,
        "token_identity_rate": matched_tokens / total_tokens if total_tokens else None,
        "matched_prefix_tokens": matched_prefix_tokens,
        "prefix_identity_rate": matched_prefix_tokens / total_tokens if total_tokens else None,
        "first_divergences": first_divergences,
    }


# ========================================================================== #
# acceptance -> TPS model (energy identity)  [PR instruction 3]
# ========================================================================== #
def _warm_sums(rows: list[dict], warm: int) -> tuple[float, int]:
    w = rows[warm:]
    return sum(r["t_request_s"] for r in w), sum(r["num_completion_tokens"] for r in w)


def acceptance_model(ref: dict, ng: dict, sim_warm: dict, *, k: int, warm: int) -> dict[str, Any]:
    """Energy identity W = d*t_v + u*t_1 -> t_v, c, A_ship, A_500. Computed two
    ways (offline-sim counts; served Prometheus counts) and cross-checked."""
    ref_agg = ref["tps"]
    ng_agg = ng["tps"]
    ref_tps = ref_agg.get("warm_aggregate_tps")
    t1 = (1.0 / ref_tps) if (ref_tps and math.isfinite(ref_tps) and ref_tps > 0) else float("nan")

    # warm wall + warm token totals for the ngram pass
    W_w, N_w = _warm_sums(ng["per_request"], warm)

    out: dict[str, Any] = {
        "k": k, "t_1_s_per_tok": t1, "ref_warm_aggregate_tps": ref_tps,
        "ngram_warm_aggregate_tps": ng_agg.get("warm_aggregate_tps"),
        "ngram_warm_median_tps": ng_agg.get("warm_median_tps"),
        "warm_wall_s": W_w, "warm_tokens": N_w, "max_acceptance_at_k": k + 1,
    }

    def _invert(t_v: float, tag: str) -> None:
        # t_v is a per-step verify cost on THIS pod (s/step); t1 = no-spec per-token cost on
        # THIS pod. Their ratio c = t_v/t1 is hardware-INVARIANT (both scale with pod speed),
        # so it is the only portable quantity. An always-drafting drafter with mean acceptance
        # A runs at speedup A/c over no-spec; clearing an OFFICIAL bar B needs speedup
        # B/anchor_official, hence the acceptance threshold is A_B = c * (B / anchor_official).
        # DO NOT use absolute slow-pod t_v against the OFFICIAL bar (A_B = B * t_v): this pod
        # serves base_fullhead ~3x slower than the 252.69 anchor (#553), so B*t_v inflates A_B
        # ~3x (frame-mixing). See the PR #573 writeup.
        if t_v is not None and math.isfinite(t_v) and t_v > 0 and t1 and math.isfinite(t1):
            c = t_v / t1
            out[f"t_v_s_per_step_{tag}"] = t_v
            out[f"c_overhead_{tag}"] = c
            out[f"A_ship_{tag}"] = c * (SHIP_TPS / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL)
            out[f"A_500_{tag}"] = c * (GATE_500 / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL)
            out[f"A_capstone_floor_{tag}"] = c * (CAPSTONE_FLOOR / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL)
        else:
            for key in ("t_v_s_per_step", "c_overhead", "A_ship", "A_500", "A_capstone_floor"):
                out[f"{key}_{tag}"] = float("nan")

    # --- way A: offline-sim counts over the same warm slice (exact for greedy) ---
    d_sim = sim_warm["draft_steps"]
    u_sim = sim_warm["no_draft_steps"]
    t_v_sim = ((W_w - u_sim * t1) / d_sim) if (d_sim and math.isfinite(t1)) else float("nan")
    out["sim_draft_steps"] = d_sim
    out["sim_no_draft_steps"] = u_sim
    out["sim_accepted"] = sim_warm["accepted"]
    out["sim_e_accept"] = sim_warm["e_accept"]
    out["sim_coverage"] = sim_warm["coverage"]
    _invert(t_v_sim, "sim")

    # --- way B: served Prometheus counts (whole run) + whole-run wall ---
    prom = ng.get("prom_spec") or {}
    d_prom = prom.get("num_drafts")
    a_prom = prom.get("num_accepted_tokens")
    W_all_ref = sum(r["t_request_s"] for r in ref["per_request"])
    N_all_ref = sum(r["num_completion_tokens"] for r in ref["per_request"])
    t1_all = (W_all_ref / N_all_ref) if N_all_ref else float("nan")
    W_all_ng = sum(r["t_request_s"] for r in ng["per_request"])
    N_all_ng = sum(r["num_completion_tokens"] for r in ng["per_request"])
    t_v_prom = float("nan")
    if d_prom and a_prom is not None and math.isfinite(t1_all):
        u_prom = N_all_ng - d_prom - a_prom
        out["prom_draft_steps"] = d_prom
        out["prom_accepted"] = a_prom
        out["prom_no_draft_steps"] = u_prom
        out["prom_e_accept"] = prom.get("e_accept_mean_acceptance_length")
        out["prom_coverage"] = d_prom / (d_prom + u_prom) if (d_prom + u_prom) else float("nan")
        if u_prom >= 0:
            t_v_prom = (W_all_ng - u_prom * t1_all) / d_prom
    out["t_1_all_s_per_tok"] = t1_all
    _invert(t_v_prom, "prom")

    # primary = prometheus (actual server behavior) when available, else sim
    primary = "prom" if (out.get(f"t_v_s_per_step_prom") is not None
                         and math.isfinite(out.get("t_v_s_per_step_prom", float("nan")))) else "sim"
    out["primary_source"] = primary
    for key in ("t_v_s_per_step", "c_overhead", "A_ship", "A_500", "A_capstone_floor"):
        out[key] = out.get(f"{key}_{primary}", float("nan"))
    return out


# ========================================================================== #
# synthesis
# ========================================================================== #
def _f(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def synthesize(ref: dict, ngrams: list[dict], models: list[dict], idents: dict,
               sweep: list[dict], self_det: dict) -> dict[str, Any]:
    # headline = best served ngram warm-median TPS
    best_i = max(range(len(ngrams)),
                 key=lambda i: _f(ngrams[i]["tps"].get("warm_median_tps")) if not math.isnan(
                     _f(ngrams[i]["tps"].get("warm_median_tps"))) else -1.0)
    best_ng = ngrams[best_i]
    best_model = models[best_i]
    best_k = best_ng["k"]
    headline_tps = _f(best_ng["tps"].get("warm_median_tps"))
    ref_tps = _f(ref["tps"].get("warm_median_tps"))

    served_e_accept = _f((best_ng.get("acceptance") or {}).get("e_accept"))
    sim_e_accept = _f(best_model.get("sim_e_accept"))
    # served (Prometheus/log) acceptance needs DISABLE_LOG_STATS=0, which on this onegraph
    # stack forces a per-step GPU sync (~3x slower, non-comparable TPS). The timed passes
    # therefore run stats OFF, so served_e_accept is typically nan here. The offline
    # exact-greedy-verify sim is the principled acceptance source for an exact-verify ngram
    # drafter (acceptance on a greedy target is fully determined by the target sequence);
    # served acceptance is cross-checked from a separate stats-ON run. Prefer served if present.
    mean_accept = served_e_accept if not math.isnan(served_e_accept) else sim_e_accept
    mean_accept_source = "served_prometheus" if not math.isnan(served_e_accept) else "offline_sim_exact_verify"
    # frame-correct acceptance thresholds, recomputed from the portable c (NOT the pre-baked
    # model A_ship which used absolute slow-pod t_v): A_B = c * (B / anchor_official).
    c_best = _f(best_model.get("c_overhead"))
    A_ship = c_best * (SHIP_TPS / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL) if not math.isnan(c_best) else float("nan")
    A_500 = c_best * (GATE_500 / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL) if not math.isnan(c_best) else float("nan")
    A_floor = c_best * (CAPSTONE_FLOOR / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL) if not math.isnan(c_best) else float("nan")
    max_A = best_k + 1
    ident = idents.get(best_k, {})

    # This pod serves base_fullhead ~3x slower than the 252.69 anchor, so the raw this-pod
    # warm-median (headline_tps) is NOT comparable to the OFFICIAL ship/floor/gate bars. The
    # spec-dec speedup (ngram/ref, both on THIS pod) IS portable (set by the hardware-invariant
    # c), so project to the anchor frame: anchor_official * speedup.
    speedup = (headline_tps / ref_tps) if (not math.isnan(headline_tps) and not math.isnan(ref_tps)
                                           and ref_tps > 0) else float("nan")
    anchor_projected_tps = (ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL * speedup
                            if not math.isnan(speedup) else float("nan"))
    exceeds_ship = anchor_projected_tps >= SHIP_TPS if not math.isnan(anchor_projected_tps) else None
    gap_to_ship = SHIP_TPS - anchor_projected_tps if not math.isnan(anchor_projected_tps) else float("nan")
    beats_capstone = anchor_projected_tps > CAPSTONE_FLOOR if not math.isnan(anchor_projected_tps) else None
    ngram_clears = (mean_accept >= A_ship) if (not math.isnan(mean_accept)
                                               and not math.isnan(A_ship)) else None
    # Can ANY always-drafting drafter at this K clear the ship? only if A_ship <= max achievable
    # (K+1). The ship is itself an MTP K=7 spec-dec point at 375.857, so this MUST be True at K=7.
    any_drafter_clears_at_k = (A_ship <= max_A) if not math.isnan(A_ship) else None

    # --- identity vs base_fullhead, framed against the base self-determinism floor ---
    # base_fullhead greedy is only ~26% self-deterministic run-to-run (#535: bf16 /
    # non-batch-invariant CUDA reductions), so STRICT byte-exact identity is
    # unattainable even base-vs-base. The meaningful test (the #566 standard): does
    # exact-verify spec-dec reproduce base AT LEAST as faithfully as base reproduces
    # itself? If ngram-vs-ref >= base self-det, the residual divergence is base
    # nondeterminism, NOT a spec acceptance/tie-break bug.
    base_sd_seq = _f(self_det.get("sequence_exact_rate"))
    base_sd_tok = _f(self_det.get("token_identity_rate"))
    ng_seq = _f(ident.get("sequence_exact_rate"))
    ng_tok = _f(ident.get("token_identity_rate"))
    greedy_identity_strict = (ng_seq == 1.0 and ng_tok == 1.0)
    identity_within_base = bool(
        not math.isnan(ng_tok) and not math.isnan(base_sd_tok) and ng_tok >= base_sd_tok - 0.02
    ) if not math.isnan(base_sd_tok) else None
    greedy_identity = identity_within_base

    return {
        "base_fullhead_ngram_spec_tps": headline_tps,                      # this-pod LOCAL warm-median (~3x slow pod)
        "base_fullhead_ngram_spec_tps_official": anchor_projected_tps,     # anchor_official * on-pod speedup
        "base_fullhead_ngram_spec_tps_anchor_projected": anchor_projected_tps,
        "base_fullhead_ngram_best_config": {"k": best_k, "prompt_lookup_max": NGRAM_LOOKUP_MAX,
                                            "prompt_lookup_min": NGRAM_LOOKUP_MIN},
        "ref_no_spec_tps": ref_tps,                                        # this-pod LOCAL warm-median
        "ref_no_spec_tps_official": ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL,  # base_fullhead no-spec official = the anchor
        "ngram_speedup_over_ref": speedup,                                 # hardware-portable spec-dec speedup
        "ship_speedup_required": SHIP_TPS / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL,  # 1.437x over base_fullhead
        "anchor_base_fullhead_nospec_official": ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL,
        "tau_lo": TAU_LO,
        "mean_acceptance_length": mean_accept,
        "mean_acceptance_length_source": mean_accept_source,
        "mean_acceptance_length_served": served_e_accept,
        "mean_acceptance_length_offline_sim": sim_e_accept,
        "ngram_coverage": _f(best_model.get("prom_coverage") if best_model.get("primary_source") == "prom"
                             else best_model.get("sim_coverage")),
        "acceptance_length_to_clear_ship": A_ship,        # A_ship = c*(ship/anchor_official)
        "acceptance_length_to_clear_500": A_500,          # A_500
        "acceptance_length_to_clear_floor": A_floor,      # A_capstone_floor
        "max_acceptance_at_k": max_A,
        "t_v_s_per_step": _f(best_model.get("t_v_s_per_step")),
        "c_overhead_factor": _f(best_model.get("c_overhead")),
        "break_even_acceptance": _f(best_model.get("c_overhead")),  # A_break == c
        "acceptance_model_primary_source": best_model.get("primary_source"),
        "ngram_acceptance_clears_ship": ngram_clears,
        "any_drafter_at_k_clears_ship": any_drafter_clears_at_k,
        "greedy_identity_vs_base_fullhead": greedy_identity,            # ngram-vs-ref >= base self-det floor?
        "greedy_identity_strict": greedy_identity_strict,               # byte-exact 1.0 (expected FALSE: base nondet)
        "greedy_identity_seq_exact": ident.get("n_sequences_byte_exact"),
        "greedy_identity_n_seq": ident.get("n_sequences"),
        "greedy_identity_token_rate": ng_tok,
        "ngram_vs_ref_seq_exact_rate": ng_seq,
        "ngram_vs_ref_token_rate": ng_tok,
        "base_self_determinism_seq_exact": base_sd_seq,
        "base_self_determinism_token_rate": base_sd_tok,
        "exceeds_ship": exceeds_ship,
        "gap_to_ship": gap_to_ship,
        "beats_capstone_floor": beats_capstone,
        "quality_gate_passes_by_construction": True,
        "ship_tps": SHIP_TPS, "capstone_floor": CAPSTONE_FLOOR, "gate_500": GATE_500,
        "anchor_base_fullhead_nospec": ANCHOR_BASE_FULLHEAD_NOSPEC,
        "self_det": base_sd_seq,  # base_fullhead run-to-run self-determinism (same-boot REF x2), the identity floor
        "peak_vram_gb": max(_f(ref.get("peak_vram_gb")),
                            *[_f(n.get("peak_vram_gb")) for n in ngrams]),
        "analysis_only": True,
        "official_tps": 0,
    }


def log_wandb(report: dict[str, Any], args: argparse.Namespace) -> str | None:
    # The report is already on disk before this runs, so W&B must NEVER crash the driver
    # (a ~1h measurement should not be lost to a logging hiccup). Guard the whole path,
    # and explicitly detect the namespace-package shadow: `sys.path` carries ROOT, and
    # ROOT holds a `wandb/` run-output dir, so an interpreter WITHOUT real wandb installed
    # imports that dir as a namespace pkg (no `.init`) and would AttributeError. Run
    # --resynthesize / measurement under a venv that has wandb (e.g. target/.venv).
    try:
        import wandb
        if not hasattr(wandb, "init"):
            print(f"[ng] wandb is a namespace shadow ({getattr(wandb, '__path__', '?')}) — real "
                  "wandb not installed in this interpreter; skipping W&B (report saved). "
                  "Re-run under a venv with wandb (e.g. target/.venv/bin/python).", flush=True)
            return None
    except Exception as exc:  # noqa: BLE001
        print(f"[ng] wandb import failed: {exc}; skipping W&B (report saved).", flush=True)
        return None
    try:
        from scripts.wandb_logging import (finish_wandb, init_wandb_run,
                                           log_json_artifact, log_summary)
        run = init_wandb_run(
            job_type="systems-profile", agent="fern",
            name=args.wandb_name or "fern/base-fullhead-ngram-specdec",
            group=args.wandb_group or "base-fullhead-specdec-ceiling",
            tags=["spec-dec", "ngram", "prompt-lookup", "base-fullhead", "acceptance-model",
                  "head-ceiling", "local-a10g", "analysis-only", "pr573"],
            notes="PR #573: bound the spec-dec axis on base_fullhead — ngram served TPS + "
                  "acceptance->TPS model + analytical A_ship (complement to lawine #572 MTP K=7)",
            config={
                "submission": str(SUBMISSION), "model_dir": MODEL_DIR,
                "num_prompts": args.num_prompts, "output_len": args.output_len,
                "concurrency": 1, "seed": SEED, "hidden": HIDDEN, "vocab": VOCAB,
                "ngram_ks": args.ngram_ks, "prompt_lookup_max": NGRAM_LOOKUP_MAX,
                "prompt_lookup_min": NGRAM_LOOKUP_MIN,
            },
        )
        if run is None:
            return None
        s = report["synthesis"]
        summary = {k: v for k, v in s.items()
                   if isinstance(v, (int, float, bool)) and (not isinstance(v, float) or math.isfinite(v))}
        summary["primary_metric"] = s["base_fullhead_ngram_spec_tps"]
        log_summary(run, summary, step=0)
        log_json_artifact(run, name="base-fullhead-specdec-report", artifact_type="specdec-report", data=report)
        rid = getattr(run, "id", None)
        finish_wandb(run)
        return rid
    except Exception as exc:  # noqa: BLE001
        print(f"[ng] wandb logging failed: {exc}; skipping W&B (report saved).", flush=True)
        return None


def _print_summary(s: dict[str, Any]) -> None:
    line = "=" * 10 + " PR #573 — BASE_FULLHEAD NGRAM SPEC-DEC CEILING " + "=" * 10
    print("\n" + line, flush=True)
    print(f"  REF no-spec TPS  LOCAL={s['ref_no_spec_tps']:.2f}  "
          f"OFFICIAL~{s['ref_no_spec_tps_official']:.2f}  (anchor 252.69/261.6; #553)", flush=True)
    print(f"  ngram spec TPS   LOCAL={s['base_fullhead_ngram_spec_tps']:.2f}  "
          f"OFFICIAL~{s['base_fullhead_ngram_spec_tps_official']:.2f}  "
          f"(K={s['base_fullhead_ngram_best_config']['k']}, speedup {s['ngram_speedup_over_ref']:.3f}x; "
          f"ship needs {s['ship_speedup_required']:.3f}x)", flush=True)
    print(f"  mean acceptance length ({s['mean_acceptance_length_source']}) = "
          f"{s['mean_acceptance_length']:.3f}  "
          f"(served {s['mean_acceptance_length_served']}, offline-sim "
          f"{s['mean_acceptance_length_offline_sim']:.3f}, coverage {s['ngram_coverage']:.3f})", flush=True)
    print(f"  per-step verify cost t_v          = {1000.0 * s['t_v_s_per_step']:.3f} ms  "
          f"(c={s['c_overhead_factor']:.3f}x, break-even A={s['break_even_acceptance']:.3f})", flush=True)
    print(f"  A_ship (clear 375.857)            = {s['acceptance_length_to_clear_ship']:.3f}  "
          f"(max achievable at K = {s['max_acceptance_at_k']})", flush=True)
    print(f"  A_500  (clear 500)                = {s['acceptance_length_to_clear_500']:.3f}", flush=True)
    print(f"  ngram acceptance clears ship?     = {s['ngram_acceptance_clears_ship']}", flush=True)
    print(f"  ANY drafter at K clears ship?     = {s['any_drafter_at_k_clears_ship']}", flush=True)
    print(f"  base self-determinism (REF x2)    = seq {s['base_self_determinism_seq_exact']:.4f}  "
          f"tok {s['base_self_determinism_token_rate']:.4f}  (the identity FLOOR)", flush=True)
    print(f"  ngram vs REF (same oracle)        = seq {s['ngram_vs_ref_seq_exact_rate']:.4f}  "
          f"tok {s['ngram_vs_ref_token_rate']:.4f}", flush=True)
    print(f"  >>> identity within base nondet?  = {s['greedy_identity_vs_base_fullhead']}  "
          f"(strict byte-exact: {s['greedy_identity_strict']})", flush=True)
    print(f"  exceeds ship (>=375.857)?         = {s['exceeds_ship']}  (gap {s['gap_to_ship']:+.2f})", flush=True)
    print(f"  beats capstone floor (>311.25)?   = {s['beats_capstone_floor']}", flush=True)
    print(f"  peak VRAM                         = {s['peak_vram_gb']:.2f} GB", flush=True)
    print("=" * len(line) + "\n", flush=True)


# ========================================================================== #
# resynthesize (recompute synthesis from a saved report — no re-serve)
# ========================================================================== #
def resynthesize_report(report_path: Path, args: argparse.Namespace) -> int:
    """Reload a saved specdec_report.json and recompute ONLY the synthesis with the
    current (frame-correct) code — the measured passes (TPS, acceptance, identity,
    self-det, per-K models) are unchanged. Used to correct the frame-mixed A_ship in
    reports produced before the c*(B/anchor) fix, without a costly ~1h re-serve."""
    report = json.loads(report_path.read_text())
    ref = report["ref"]                                   # ['tps'], ['peak_vram_gb']
    ngrams = list(report["ngrams"])                       # ['tps'], ['acceptance'], ['k'], ['peak_vram_gb']
    models = list(report["acceptance_models"])            # ['c_overhead'], ['t_v_s_per_step'], ['sim_e_accept'], ...
    # A report saved by pre-frame-fix code carries per-K A_ship/A_500/A_floor computed as
    # B*t_v (slow-pod, ~3x inflated). c_overhead is portable, so re-derive every per-K
    # threshold from it: A_B = c * (B / anchor_official). synthesize() already recomputes
    # the headline from c; this keeps the raw per-K entries self-consistent too.
    _bars = {"A_ship": SHIP_TPS, "A_500": GATE_500, "A_capstone_floor": CAPSTONE_FLOOR}
    for m in models:
        for csuf in ("", "_sim", "_prom"):
            c = _f(m.get(f"c_overhead{csuf}"))
            if math.isnan(c):
                continue
            for akey, bar in _bars.items():
                m[f"{akey}{csuf}"] = c * (bar / ANCHOR_BASE_FULLHEAD_NOSPEC_OFFICIAL)
    report["acceptance_models"] = models
    idents = {int(k): v for k, v in report.get("identities", {}).items()}
    self_det = report["base_self_determinism"]
    synthesis = synthesize(ref, ngrams, models, idents, [], self_det)
    report["synthesis"] = synthesis
    report["resynthesized_from"] = str(report_path)
    report["resynthesized_at"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_file = report_path.with_name(report_path.stem + "_corrected.json")
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(synthesis)
    print(f"[ng] resynthesized {report_path.name} -> {out_file.name}", flush=True)
    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id_corrected"] = rid
            out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            print(f"[ng] wandb (corrected) run id={rid}", flush=True)
    return 0


# ========================================================================== #
# main
# ========================================================================== #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny REF + one NGRAM (4x48) plumbing check")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--ngram-ks", default="7,3", help="comma list of num_speculative_tokens to serve")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server-python", type=Path, default=None)
    ap.add_argument("--request-timeout-s", type=int, default=900)
    ap.add_argument("--gpu-mem-util", default="0.90")
    ap.add_argument("--relax-loopgraph", action="store_true",
                    help="drop REQUIRE-capture/fast-path guards on BOTH passes (if ngram trips them)")
    ap.add_argument("--warm", type=int, default=None, help="warmup requests discarded (default mf.WARMUP_REQUESTS)")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--resynthesize", type=Path, default=None,
                    help="recompute synthesis from a saved specdec_report.json (no re-serve)")
    # internal worker mode
    ap.add_argument("--decode-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--dataset-path")
    ap.add_argument("--tokenizer"); ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-file")
    args = ap.parse_args(argv)

    if args.decode_worker:
        return _decode_worker(args)

    if args.resynthesize is not None:
        return resynthesize_report(args.resynthesize, args)

    mf_spec = importlib.util.spec_from_file_location("measure_floor", str(MEASURE_FLOOR))
    mf = importlib.util.module_from_spec(mf_spec)
    assert mf_spec and mf_spec.loader
    mf_spec.loader.exec_module(mf)
    from scripts.local_validation import harness, paths, serve_profile

    for note in paths.prepare_local_gpu_env():
        print(f"[ng] {note}", flush=True)

    manifest = harness.load_manifest(SUBMISSION)
    server_python = args.server_python or harness.ensure_server_venv(manifest["dependencies"])
    warm = args.warm if args.warm is not None else mf.WARMUP_REQUESTS

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ks = [int(x) for x in str(args.ngram_ks).split(",") if x.strip()]

    if args.smoke:
        k = ks[0]
        ref = run_pass(mf, harness, paths, serve_profile, server_python=server_python, label="ref_smoke",
                       extra_env=build_env(spec_config="", model_dir=args.model_dir,
                                           relax_loopgraph=args.relax_loopgraph, gpu_mem_util=args.gpu_mem_util),
                       num_prompts=4, output_len=48, port=args.port, request_timeout_s=args.request_timeout_s)
        ng = run_pass(mf, harness, paths, serve_profile, server_python=server_python, label=f"ngram_smoke_k{k}",
                      extra_env=build_env(spec_config=ngram_spec_config(k), model_dir=args.model_dir,
                                          relax_loopgraph=args.relax_loopgraph, gpu_mem_util=args.gpu_mem_util),
                      num_prompts=4, output_len=48, port=args.port, request_timeout_s=args.request_timeout_s)
        sim = offline_sim_pass(ref["per_request"], NGRAM_LOOKUP_MAX, NGRAM_LOOKUP_MIN, k, warm=0)
        ident = compare_identity(ref["per_request"], ng["per_request"])
        print(f"\n[ng] SMOKE ref_tps={_f(ref['tps'].get('warm_median_tps')):.2f} "
              f"ngram_tps={_f(ng['tps'].get('warm_median_tps')):.2f}", flush=True)
        print(f"[ng] SMOKE served_accept={ng.get('acceptance')}", flush=True)
        print(f"[ng] SMOKE offline_sim e_accept={sim['e_accept']:.3f} coverage={sim['coverage']:.3f} "
              f"draft_steps={sim['draft_steps']} no_draft={sim['no_draft_steps']}", flush=True)
        print(f"[ng] SMOKE identity seq_exact={ident['n_sequences_byte_exact']}/{ident['n_sequences']} "
              f"tok_rate={ident['token_identity_rate']}", flush=True)
        boot_ok = all(grep_log(ng["log_path"], ["Avg generation throughput"]).values()) or \
            _f(ng["tps"].get("warm_median_tps")) > 0
        print(f"[ng] SMOKE {'PASS' if boot_ok else 'CHECK'} ({time.time()-t_start:.0f}s)", flush=True)
        return 0 if boot_ok else 1

    # --- REF (no-spec), decoded TWICE on the same boot: pass0 is the greedy
    # oracle for the acceptance sim + identity; (pass0 vs pass1) is the base
    # self-determinism floor the ngram identity is judged against (base_fullhead
    # is only ~26% self-det run-to-run, #535 — strict byte-identity is unattainable). ---
    ref = run_pass(mf, harness, paths, serve_profile, server_python=server_python, label="ref",
                   extra_env=build_env(spec_config="", model_dir=args.model_dir,
                                       relax_loopgraph=args.relax_loopgraph, gpu_mem_util=args.gpu_mem_util),
                   num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                   request_timeout_s=args.request_timeout_s, n_decodes=2)
    ref_rows = ref["decodes"][0]["per_request"]
    self_det = compare_identity(ref["decodes"][0]["per_request"], ref["decodes"][1]["per_request"])
    print(f"[ng] REF warm_median_tps={_f(ref['tps'].get('warm_median_tps')):.2f} "
          f"peak={_f(ref.get('peak_vram_gb')):.2f}GB self_det seq={self_det['sequence_exact_rate']} "
          f"tok={self_det['token_identity_rate']} ({time.time()-t_start:.0f}s)", flush=True)

    # --- offline ngram sweep on REF greedy completions (free; full #503 grid) ---
    sweep = [offline_sim_pass(ref_rows, n, NGRAM_LOOKUP_MIN, k, warm=warm)
             for (n, k) in SWEEP_GRID]
    sweep_brief = [{"n_max": s["n_max"], "k": s["k"], "e_accept": s["e_accept"],
                    "coverage": s["coverage"]} for s in sweep]
    print(f"[ng] offline sweep: {sweep_brief}", flush=True)

    # --- NGRAM passes (one per K), each with its matched offline sim + model ---
    ngrams: list[dict] = []
    models: list[dict] = []
    idents: dict[int, dict] = {}
    for k in ks:
        ng = run_pass(mf, harness, paths, serve_profile, server_python=server_python, label=f"ngram_k{k}",
                      extra_env=build_env(spec_config=ngram_spec_config(k), model_dir=args.model_dir,
                                          relax_loopgraph=args.relax_loopgraph, gpu_mem_util=args.gpu_mem_util),
                      num_prompts=args.num_prompts, output_len=args.output_len, port=args.port,
                      request_timeout_s=args.request_timeout_s)
        ng["k"] = k
        sim_k = offline_sim_pass(ref_rows, NGRAM_LOOKUP_MAX, NGRAM_LOOKUP_MIN, k, warm=warm)
        model_k = acceptance_model(ref, ng, sim_k, k=k, warm=warm)
        ident_k = compare_identity(ref_rows, ng["per_request"])
        ng["offline_sim_matched"] = sim_k
        ngrams.append(ng)
        models.append(model_k)
        idents[k] = ident_k
        print(f"[ng] NGRAM K={k} warm_median_tps={_f(ng['tps'].get('warm_median_tps')):.2f} "
              f"served_e_accept={_f((ng.get('acceptance') or {}).get('e_accept'))} "
              f"sim_e_accept={_f(sim_k['e_accept']):.3f} cov={_f(sim_k['coverage']):.3f} "
              f"t_v_ms={1000.0*_f(model_k.get('t_v_s_per_step')):.3f} "
              f"A_ship={_f(model_k.get('A_ship')):.3f} "
              f"identity={ident_k['n_sequences_byte_exact']}/{ident_k['n_sequences']} "
              f"({time.time()-t_start:.0f}s)", flush=True)

    synthesis = synthesize(ref, ngrams, models, idents, sweep, self_det)

    def _strip(d: dict) -> dict:
        # drop bulky per-request token arrays (both the top-level and the per-decode copies)
        out = {kk: vv for kk, vv in d.items() if kk != "per_request"}
        if "decodes" in out:
            out["decodes"] = [{"tps": dd.get("tps")} for dd in out["decodes"]]
        return out

    report = {
        "pr": 573, "analysis_only": True, "official_tps": 0,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "submission": str(SUBMISSION), "model_dir": args.model_dir,
        "num_prompts": args.num_prompts, "output_len": args.output_len,
        "ngram_ks": ks, "prompt_lookup_max": NGRAM_LOOKUP_MAX, "prompt_lookup_min": NGRAM_LOOKUP_MIN,
        "warm_discarded": warm, "relax_loopgraph": args.relax_loopgraph,
        "ref": _strip(ref),
        "base_self_determinism": self_det,
        "ngrams": [_strip(n) for n in ngrams],
        "acceptance_models": models,
        "identities": {str(k): v for k, v in idents.items()},
        "offline_sweep": sweep_brief,
        "offline_sweep_full": [{k: v for k, v in s.items() if k != "per_seq"} for s in sweep],
        "synthesis": synthesis,
        "elapsed_s": time.time() - t_start,
    }
    out_file = OUT_ROOT / "specdec_report.json"
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _print_summary(synthesis)
    print(f"[ng] report -> {out_file} (elapsed {report['elapsed_s']:.0f}s)", flush=True)

    rid = None
    if not args.no_wandb:
        rid = log_wandb(report, args)
        if rid:
            report["wandb_run_id"] = rid
            out_file.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
            print(f"[ng] wandb run id={rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
