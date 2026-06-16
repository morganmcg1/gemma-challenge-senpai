"""Hardware-ceiling probe for async-pipelined drafting (PR #444).

Question: on THIS A10G (sm_86, 72 SMs), at the measured drafter/verify sizes,
can a small "drafter" kernel (~1.43 ms) actually overlap a large compute-bound
"verify" GEMM (~6.44 ms) when launched on a separate CUDA stream?  If wall
~= max(D,V) there is real two-stream headroom; if wall ~= D+V the verify kernels
saturate the SMs and a concurrent drafter just queues behind them (no overlap).

This is a representative scheduling probe (fp16 GEMMs), NOT the exact int4-Marlin
verify forward.  It brackets the compute-bound case (the hardest case for a
concurrent small kernel to find idle SMs).  The real bs=1 verify is more
memory-bandwidth-bound; per CUTLASS/PyTorch Ampere evidence it also issues warps
across all SMs and leaves no free overlap.  Either way the conclusion converges.
"""
import time, json, sys
import torch

assert torch.cuda.is_available(), "no CUDA"
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

D_TARGET_MS = 1.433   # measured drafter (propose) gpu p50
V_TARGET_MS = 6.445   # measured verify (execute_model) gpu p50, decode-only

def time_stream(fn, iters=50, warmup=10):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(s):
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters

def make_gemm(m, k, n, reps):
    a = torch.randn(m, k, device=dev, dtype=torch.float16)
    b = torch.randn(k, n, device=dev, dtype=torch.float16)
    def fn():
        x = a
        for _ in range(reps):
            x = (x @ b) * 0.0009765625  # scale to avoid inf blowup
            x = x[:, :k] if x.shape[1] >= k else torch.nn.functional.pad(x, (0, k - x.shape[1]))
    return fn

# Calibrate a big compute-bound GEMM (~V) and a small one (~D).
def calibrate(target_ms, m, k, n):
    reps = 1
    while True:
        fn = make_gemm(m, k, n, reps)
        ms = time_stream(fn, iters=30, warmup=8)
        if ms >= target_ms or reps > 4096:
            return fn, ms, reps
        reps = max(reps + 1, int(reps * max(1.3, target_ms / max(ms, 1e-3))))

verify_fn, v_ms, v_reps = calibrate(V_TARGET_MS, 4096, 4096, 4096)
draft_fn,  d_ms, d_reps = calibrate(D_TARGET_MS, 2048, 2048, 2048)
print(f"[probe] calibrated verify-like: {v_ms:.3f} ms (reps={v_reps}), draft-like: {d_ms:.3f} ms (reps={d_reps})", flush=True)

# Serial: both on the same stream, back to back (== current sync cycle).
def serial():
    draft_fn(); verify_fn()
serial_ms = time_stream(serial, iters=40, warmup=10)

# Concurrent: verify on stream A, draft on stream B, forked from the default
# stream and JOINED back each iter so the timing captures real concurrent GPU
# execution.  This is the OPTIMISTIC contaminated overlap (no data dep); the
# byte-exact variant would add sB.wait_event(verify_done) which re-serializes
# draft AFTER verify -> identical to serial.
sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
def concurrent_once():
    fork = torch.cuda.Event(); fork.record()          # default stream point
    sA.wait_event(fork); sB.wait_event(fork)          # both start together
    with torch.cuda.stream(sA):
        verify_fn(); evA = torch.cuda.Event(); evA.record()
    with torch.cuda.stream(sB):
        draft_fn();  evB = torch.cuda.Event(); evB.record()
    torch.cuda.current_stream().wait_event(evA)        # join
    torch.cuda.current_stream().wait_event(evB)
for _ in range(10):
    concurrent_once()
torch.cuda.synchronize()
e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
e0.record()
for _ in range(40):
    concurrent_once()
e1.record()
torch.cuda.synchronize()
concurrent_ms = e0.elapsed_time(e1) / 40

ideal_overlap_ms = max(d_ms, v_ms)            # if drafter fully hidden
no_overlap_ms = d_ms + v_ms                   # if fully serial
# overlap efficiency: 0 == no overlap (==serial), 1 == fully hidden draft
overlap_eff = (no_overlap_ms - concurrent_ms) / max(no_overlap_ms - ideal_overlap_ms, 1e-6)
overlap_eff = max(0.0, min(1.0, overlap_eff))

out = {
    "draft_like_ms": round(d_ms, 4),
    "verify_like_ms": round(v_ms, 4),
    "serial_ms": round(serial_ms, 4),
    "concurrent_two_stream_ms": round(concurrent_ms, 4),
    "ideal_overlap_ms_maxDV": round(ideal_overlap_ms, 4),
    "no_overlap_ms_sumDV": round(no_overlap_ms, 4),
    "overlap_efficiency_0to1": round(overlap_eff, 4),
    "concurrent_speedup_vs_serial_pct": round((serial_ms - concurrent_ms) / serial_ms * 100, 3),
    "note": "fp16 GEMM scheduling probe; contaminated/no-wait overlap (best case). byte-exact wait_event re-serializes -> serial.",
}
print("[probe] RESULT " + json.dumps(out), flush=True)
json.dump(out, open("research/local_validation/async_steptime_20260616/twostream_probe.json", "w"), indent=2)
