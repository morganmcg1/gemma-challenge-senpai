STUDENT kanna: heartbeat — the run is **alive, not crashed**; NO relaunch needed (NOT terminal — panel continuing)

Thanks for the fast crash flag. I checked the pod before touching anything, and the situation on the ground differs from the W&B state you saw: **the process never died.** No relaunch — I'm resuming, which is the path you preferred.

## What's actually true at 10:53Z

- **Driver PID 396465 (`run_gate.py --seeds 34567`) is alive and progressing** — currently step ~76/198 on `sampled_s34567`, `openai-api: 1/1`, **HTTP retries 0**. Server PID 320465 is up and healthy (`/v1/models` serves `gemma-4-e4b-it`). **Exactly one** eval client (verified full cmdline; the conc=1 contract holds).
- **No crash traceback exists.** I grepped the full driver log for `traceback|out of memory|cuda error|killed process|oom-kill|exception` → empty. GPU is **20.1/23.0 GB** (peak this PR 19.62 GB) — **no OOM**.
- So **OOM / CUDA / preemption are all ruled out.**

## Diagnosis of the W&B "crashed" state

`cr3c4y3q` is a **resume-by-id** run: each `run_gate.py` invocation does `wandb.init(id=cr3c4y3q, resume)`, and there are 6 local sessions stacked on that id. The "crashed / last-heartbeat 10:18Z" is a **W&B heartbeat lapse at the `greedy_s34567`→`sampled_s34567` cell transition** (10:18Z is exactly when greedy finished and the sampled eval subprocess spawned) — the wandb service for that session went quiet (no per-step logging during the 40-min sampled eval; the service got squeezed at the subprocess fork) while the **main driver kept running**. The summary's stale `status: running` and W&B's `crashed` are **both** misleading; **the on-disk result JSONs + `gate_summary.json` are the source of truth, and they're complete and uncontaminated.** The resume design self-heals — the next invocation re-attaches and revives the run to `running`.

## Real panel state

- **greedy n=3 DONE:** mean **0.5034**, seeds [0.5202, 0.5202, 0.4697] — clears the 0.4864 bar. finish_length@4096 **2.5%** (implied-3072 **11.8%**) → cap-artifact confirmed.
- **sampled n=2 done** (0.4823, CI [0.386, 0.579]) **+ seed 34567 ~38% in flight** → **n=3 imminent**.

## Plan (your steers folded in)

1. **Not killing the healthy conc=1-clean seed** — that would burn ~35 min of clean compute and contradict "resume from seed 3." Letting `sampled_s34567` finish (ETA ~**11:50Z**; note it's running ~2× slower than earlier seeds — consistent with mild fragmentation on the ~5h-old server, more below).
2. **Fresh-engine restart for the new batch** (your suggestion): after this seed, I restart `serve_spec.py` clean (same config — int4_g128_lmhead + K=7 + BI=1 + dev307 + model_len 8192, `expandable_segments` already on) to clear the slowdown. Numerically equivalent at conc=1+BI=1; byte-variance already absorbed by the between-seed CI.
3. **Accumulate sampled toward n=10 (≥5 floor)** — seeds 45678→13579 (the remaining 7 of the canonical 10-seed `SEEDS`), sampled-only (greedy capped at n=3 per your steer), one resumable detached invocation per wakeup (<90 min each). Final summary re-logged cleanly to `cr3c4y3q` so the Option-B number is n-comparable to ubel #638 (0.4990) / lawine #639 (0.5056).

Holding `status:wip`. Next heartbeat when `sampled_s34567` lands + the fresh server is up. No blocking question.
