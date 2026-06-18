STUDENT kanna: process bug found + fixed (NOT terminal — sweep continuing). One discarded seed, one launcher hardening.

## What happened (conc=1 contract violation from a double-launch)

While accumulating sampled seeds I found **two eval clients had been launched for the same cell** (seed 12345 sampled), and they ran **concurrently 06:54Z–07:21Z**:

- a stray `run_eval.py` (PID 336138, started 06:35Z) launched **directly**, bypassing `run_gate.py` and the `_gate.pid` guard, and
- `run_gate.py` (PID 341838, started 06:54Z) which independently re-launched the same seed because the stray hadn't written its JSON yet, so the resume-skip didn't see it.

Two clients on the `--max-connections 1` server means the engine batched across them for ~27 min — i.e. that seed was **not produced under the conc=1 gate-point contract**. With the verdict sitting right at the 0.4864 bar, I won't put a concurrency-contaminated seed in the pool.

**Action (07:21–07:27Z):** killed the in-flight duplicate (341838 + child), **discarded** the contaminated `sampled_s12345.json` (it had reported acc 0.4798), verified the server is healthy + idle (GPU 0% util, only `gemma-4-e4b-it` served), and **re-launched seed 12345 sampled clean** — confirmed exactly **one** eval client now, W&B run `cr3c4y3q` resumed. `greedy_s12345` is **unaffected** (it ran 05:57–06:35Z, before the concurrency window) — clean, acc **0.5202**, finish_length **0.0152** @ mt4096.

## Bug fix (separate, small — flagging for review)

Hardened `research/validity/gpqa_gate_readingA_4096/_launch_seed.sh`: the pre-launch guard now refuses if **any** `run_gate.py` **or** `run_eval.py` is alive (via `pgrep -f`), not just the one recorded in `_gate.pid`. The original guard only checked `_gate.pid`, so a stray `run_eval.py` launched outside the driver was invisible to it — exactly the race above. This makes the conc=1 single-client invariant enforceable regardless of how a cell was started.

## Status

- Server unchanged: int4_g128_lmhead body + Gemma4-MTP K=7 + BI=1, dev307, conc=1, model_len 8192, mt=4096. `analysis_only=true`, `official_tps=0`, LOCAL A10G, no HF Job / no submission.
- Re-running seed 12345 sampled now; then accumulating toward sampled **n≥5 (floor)** and onward to **n=10** per your steer, one clean cell per detached invocation (resumable, each invocation <90 min). Greedy capped at ~3 seeds for the health read.
- Keeping `status:wip`; no blocking question. Terminal `SENPAI-RESULT` (with pct-of-base 0.5404, pct-of-AR-body 0.4990, finish_length 4096 + implied-3072, CIs, n_seeds) to follow once seeds land.
