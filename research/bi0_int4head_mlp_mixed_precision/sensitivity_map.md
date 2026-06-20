# PR #810 — Body-MLP per-layer precision sensitivity map (stark)

LOCAL A10G offline analysis. Base = int4head body (`gemma-4-E4B-it-qat-w4a16-ct`, W4A16 g32 sym, 42 layers). Reference for recon = dequantized-W4 (`W4deq`, what the base serves).

- Offline baseline PPL (all-W4, transformers bf16-dequant, official 128-record harness): **2.0067** (official served int4head base = 2.0029; Δ +0.0038 = harness numerics, validates the offline replica).
- Body-MLP weight read @ W4 = **1.86 GB/token** (per layer 44.24 MB; W3 34.41 MB, W2 24.58 MB — bf16 g32 scales 4.92 MB/layer are a FIXED overhead, so W3 saves 22% / W2 44% of layer bytes, below the naive 25%/50%).

## SERVE VIABILITY (Step-2 kill-gate): **FAIL** — sub-4-bit does not serve on sm_86 under vLLM 0.22.0.
- `compressed_tensors_wNa16.py:66-70` raises `ValueError: Unsupported num_bits = {3,2}. Supported = [4, 8]` at scheme `__init__` (load-time, pre-kernel). Hard load failure, not a dense fallback.
- `marlin_utils.py:75-83`: Ampere Marlin int types = `{uint4,uint4b8,uint8b128}` only (4/8-bit). No uint3/uint2. Machete (sub-4-bit) is sm_90 + g∈{-1,64,128} only. ⇒ no W3/W2 kernel on A10G. Step 3 (served TPS/quality) dead.

## Per-layer sensitivity (signal a: W4deq recon error; signal b: single-layer ΔPPL on official harness)

| Layer | recon W3 rel | W3 SQNR dB | recon W2 rel | W2 SQNR dB | ΔPPL@W3 | ΔPPL@W2 |
|------:|-------------:|-----------:|-------------:|-----------:|--------:|--------:|
| 0 | 0.2085 | 13.6 | 0.4498 | 6.9 | -0.0002 | -0.0045 |
| 1 | 0.2117 | 13.5 | 0.4550 | 6.8 | +0.0007 | -0.0021 |
| 2 | 0.2137 | 13.4 | 0.4576 | 6.8 | +0.0003 | +0.0039 |
| 3 | 0.2095 | 13.6 | 0.4516 | 6.9 | -0.0012 | +0.0001 |
| 4 | 0.2119 | 13.5 | 0.4549 | 6.8 | +0.0000 | -0.0013 |
| 5 | 0.2139 | 13.4 | 0.4577 | 6.8 | -0.0006 | +0.0001 |
| 6 | 0.2141 | 13.4 | 0.4587 | 6.8 | +0.0004 | +0.0111 |
| 7 | 0.2096 | 13.6 | 0.4514 | 6.9 | +0.0000 | +0.0066 |
| 8 | 0.2156 | 13.3 | 0.4595 | 6.8 | -0.0047 | +0.0061 |
| 9 | 0.2194 | 13.2 | 0.4637 | 6.7 | +0.0020 | +0.0085 |
| 10 | 0.2219 | 13.1 | 0.4662 | 6.6 | +0.0027 | +0.0032 |
| 11 | 0.2141 | 13.4 | 0.4576 | 6.8 | +0.0052 | +0.0319 |
| 12 | 0.2100 | 13.6 | 0.4523 | 6.9 | +0.0125 | +0.0189 |
| 13 | 0.2061 | 13.7 | 0.4465 | 7.0 | +0.0099 | +0.0306 |
| 14 | 0.2084 | 13.6 | 0.4493 | 6.9 | -0.0116 | +0.0199 |
| 15 | 0.2167 | 13.3 | 0.4597 | 6.8 | -0.0067 | +0.0074 |
| 16 | 0.2156 | 13.3 | 0.4572 | 6.8 | +0.0000 | -0.0041 |
| 17 | 0.2180 | 13.2 | 0.4611 | 6.7 | -0.0036 | +0.0007 |
| 18 | 0.2132 | 13.4 | 0.4555 | 6.8 | -0.0062 | +0.0033 |
| 19 | 0.2166 | 13.3 | 0.4595 | 6.8 | -0.0010 | -0.0297 |
| 20 | 0.2222 | 13.1 | 0.4661 | 6.6 | +0.0020 | +0.0104 |
| 21 | 0.2321 | 12.7 | 0.4782 | 6.4 | +0.0194 | -0.0221 |
| 22 | 0.2254 | 12.9 | 0.4705 | 6.5 | -0.0031 | -0.0302 |
| 23 | 0.2100 | 13.6 | 0.4526 | 6.9 | +0.0022 | -0.0055 |
| 24 | 0.2095 | 13.6 | 0.4524 | 6.9 | -0.0039 | +0.0077 |
| 25 | 0.2034 | 13.8 | 0.4428 | 7.1 | -0.0056 | +0.0072 |
| 26 | 0.2027 | 13.9 | 0.4412 | 7.1 | -0.0026 | +0.0126 |
| 27 | 0.2060 | 13.7 | 0.4461 | 7.0 | +0.0048 | +0.0126 |
| 28 | 0.2089 | 13.6 | 0.4501 | 6.9 | +0.0016 | +0.0210 |
| 29 | 0.2112 | 13.5 | 0.4536 | 6.9 | -0.0009 | +0.0238 |
| 30 | 0.2092 | 13.6 | 0.4513 | 6.9 | -0.0078 | +0.0098 |
| 31 | 0.2048 | 13.8 | 0.4439 | 7.1 | +0.0037 | +0.0229 |
| 32 | 0.2041 | 13.8 | 0.4431 | 7.1 | +0.0032 | +0.0351 |
| 33 | 0.2040 | 13.8 | 0.4432 | 7.1 | +0.0042 | +0.0160 |
| 34 | 0.2039 | 13.8 | 0.4432 | 7.1 | +0.0122 | +0.0299 |
| 35 | 0.2055 | 13.7 | 0.4454 | 7.0 | +0.0011 | +0.0199 |
| 36 | 0.2038 | 13.8 | 0.4431 | 7.1 | +0.0031 | +0.0107 |
| 37 | 0.2030 | 13.9 | 0.4416 | 7.1 | +0.0047 | +0.0053 |
| 38 | 0.2033 | 13.8 | 0.4419 | 7.1 | +0.0019 | +0.0048 |
| 39 | 0.2036 | 13.8 | 0.4426 | 7.1 | +0.0048 | +0.0130 |
| 40 | 0.2055 | 13.7 | 0.4448 | 7.0 | +0.0053 | +0.0241 |
| 41 | 0.2094 | 13.6 | 0.4502 | 6.9 | -0.0045 | -0.0045 |

- **Most robust (lowest ΔPPL@W3):** L14(-0.0116), L30(-0.0078), L15(-0.0067), L18(-0.0062), L25(-0.0056), L8(-0.0047)
- **Least robust (highest ΔPPL@W3):** L11(+0.0052), L40(+0.0053), L13(+0.0099), L34(+0.0122), L12(+0.0125), L21(+0.0194)
- ΔPPL@W3 across 42 layers: min -0.0116, max +0.0194, mean +0.0011. recon W3 rel_err spread 0.203–0.232 (NEARLY UNIFORM — no large robust subset to exploit).

## Bytes-saved vs PPL Pareto front (cumulative mixed configs, full 128)

| Config | layers down | weight bytes saved/token | full PPL | ΔPPL | PPL ≤ 2.42? |
|--------|------------:|-------------------------:|---------:|-----:|:----------:|
| all_W4 (base) | 0 | 0 MB | 2.0067 | +0.0000 | ✅ |
| robust10_W2 | 10 | 197 MB | 2.2613 | +0.2545 | ✅ |
| robust10_W3 | 10 | 98 MB | 1.9774 | -0.0293 | ✅ |
| robust21_W3 | 21 | 206 MB | 1.9811 | -0.0256 | ✅ |
| all_W3 | 42 | 413 MB | 2.1432 | +0.1365 | ✅ |

- Body-MLP @ all-W3 would save ~0.41 GB/token (22% of MLP weight bytes) IF it could serve.
- **Quality budget is wide open** (huge PPL headroom 2.01→2.42); the binding constraint is the SERVE kill-gate, not quality.
- **Reusable guidance for W4-legal levers (wirbel #807 W4A8, fern #808 2:4):** sensitivity is near-uniform, so apply uniformly; if forced to spare layers, spare the least-robust L13, L34, L12, L21.
