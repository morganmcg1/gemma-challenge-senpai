STUDENT kanna: heartbeat — **run healthy (hard proof-of-life), sampled n=7 in flight**; plus a small W&B keepalive hardening that should end the "crashed" flapping. NOT terminal.

## Hard proof-of-life @ 14:08Z (the W&B `crashed` reads are false)
- **eval client PID 472391 ALIVE**, server PID 320465 up the whole time, **GPU 100% util / 20.1 GB** (peak this PR 19.62 — no OOM). **Exactly one** eval client → conc=1 contract holds; **HTTP retries 0**.
- seed **78901** sampled is grinding on a **long single item near the 4096-token cap** (the client is correctly blocked on the HTTP response while the server generates at 100% GPU — not a deadlock; `max_tokens=4096` bounds it). On-disk result JSONs + `gate_summary.json` for n=6 are complete and uncontaminated — these are the source of truth, not the W&B run-state field.

## Panel state (unchanged from 13:42Z; seed 78901 will make it n=7)
- **sampled n=6:** mean **0.4655**, 95% t-CI **[0.4424, 0.4886]** → **STRADDLES** bar 0.4864 → `READING_A_GPQA_KNIFE_EDGE`. 86.1% of base (0.5404), 93.3% of AR-body (0.4990).
- **greedy n=3 (capped):** mean **0.5034**, clears the bar. **finish_length@4096 = 2.5% → implied-3072 = 11.8%** (≈ #631's 13%) → **cap-artifact confirmed, gate point genuinely healthy.**

## Root-cause + fix for the recurring W&B "crashed" flapping (small, flagging for review)
I root-caused the false `crashed` reads that have spooked the heartbeat checks: each resumable one-seed invocation `wandb.init(resume)`s, then **blocks ~45 min in `subprocess.run()` (the eval) logging *nothing* to W&B**, so the backend heartbeat-times-out **mid-eval** and flips the run to `crashed` — even though the process never dies and the on-disk JSONs are always intact (exactly what we've seen 3×).

**Fix (`run_gate.py`):** a tiny **daemon keepalive thread** that logs `heartbeat/*` every 60s for the life of the driver. `subprocess.run()` releases the GIL while waiting, so the thread keeps the run actively logging → no heartbeat timeout → no false `crashed`. Pure liveness signal; it never touches the eval, analysis, resume, or accumulation. Validated (import/argparse + a mock-run smoke: beats fire, uptime monotonic, exits quietly after `finish`).

**Caveat:** seed **78901 is running the OLD code** (no keepalive), so it *may still* flicker to `crashed` once more before it lands (~14:35Z) — that's still the benign heartbeat lapse, not a death. From seed **89012** onward the keepalive is active and the run should stay `running`.

## Plan
Let 78901 land (→ n=7), then 89012 → 90123 → 13579 to reach the canonical **n=10** (CI-comparable to ubel #638 0.4990 / lawine #639 0.5056), one clean resumable cell per detached invocation (<90 min each), keepalive active. Between-seed SD ≈ 0.022 with seed 67890 a low outlier ⇒ n=10 CI-upper is tracking just under the bar → on course to resolve to a robust `READING_A_GPQA_FAILS`, but I'm not terminalizing on a grazing CI. Terminal `SENPAI-RESULT` at n=10 (or earlier clean CI resolution).

Guardrails held: `analysis_only=true`, `official_tps=0`, LOCAL A10G, **no HF Job / no submission**. Holding `status:wip`; no blocking question.
