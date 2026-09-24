# fp32_gemm_rms (P1-2)

Deterministic FP32 controller projection plus controller RMS scale for the
DSv4 mHC block (issue #2, task P1-2), forward and backward, in CUDA and
Triton. Both backends are byte-equal to the pinned FP32 oracle
(`rl_engine/mhc/oracle.py`, numeric profile `oracle-fp32-mhc-v1`).

## Semantics

Inputs `X_flat: FP32 [T, K]` (the flattened four-stream residual, production
`K = 4 x 4096 = 16384`) and `W: FP32 [24, K]`:

- `P[t, n] = sum_k X[t, k] * W[n, k]` — one FP32 accumulator per output,
  k ascending, mul and add rounded separately (no FMA), single unsplit K pass.
- `s_t = sum_k X[t, k]^2` (same fold); `norm = sqrt(s)`;
  `q = norm / sqrt(K)`; `r = 1 / (q + eps)`, `eps = 1e-6`.

`r` is the **controller** RMS `1/(sqrt(mean(X^2)) + eps)` — deliberately not
the `rsqrt(mean + eps)` used by `rmsnorm_residual`. The two differ at the
byte level and the test suite asserts the distinction.

Backward, with upstream `dP` and `g_r`:

- `dX_gemm = dP @ W` (n ascending), `dW = dP^T @ X` (t ascending),
- `dX_rms = g_r * ((-(r^2) * X) / (K * q))`,
- `dX = dX_gemm + dX_rms` in that fixed order.

Determinism is by construction: every reduced output element is produced by a
single accumulator walking its axis in ascending order, and parallelism comes
only from independent output elements. There is no Split-K / Stream-K /
atomic partial accumulation, and bytes cannot depend on batch size, token
count, SM count or launch geometry. `P` and `r` stay FP32.

## Entry points

- `rl_engine.mhc.fp32_gemm_rms.fp32_gemm_rms(x, w, eps, backend=...)` —
  differentiable autograd entry (`backend` in `reference|cuda|triton`).
- `rl_engine.mhc.fp32_gemm_rms.CudaGemmRMSProvider` /
  `TritonGemmRMSProvider` — P1 providers for `scripts/check_p1.py`; they also
  implement the P1-D6 `fixed_k_gemm_fwd/bwd` core.
- `fixed_k_gemm_reference(x, w)` — the golden fixed-K GEMM (any device).
- `fixed_k_gemm_bitequal_harness(candidate, x, w)` — raw-byte comparison
  harness; downstream fast paths (DeepGEMM / TE) may swap in only when
  `bitwise_equal` and faster.

The CUDA kernels live in the standalone extension `rl_engine._C_mhc`
(`csrc/cuda/mhc/fp32_gemm_rms.cu`), always compiled without fast-math and
with FMA contraction disabled. The Triton kernels
(`rl_engine/mhc/fp32_gemm_rms_triton.py`) use plain `+`/`*` (round-to-nearest,
non-FTZ) with `enable_fp_fusion=False`, and inline PTX `div.rn.f32` /
`sqrt.rn.f32` for division and square root; `libdevice.*_rn` is deliberately
avoided because Triton links libdevice with FTZ enabled and flushes subnormal
results.

## Build

```bash
pip install --no-build-isolation -e .
# or: python setup.py build_ext --inplace
```

## Validation

```bash
python -m pytest tests/test_p1_fp32_gemm_rms.py -q -rs
python scripts/check_p1.py --provider rl_engine.mhc.fp32_gemm_rms:CudaGemmRMSProvider --device cuda
python scripts/check_p1.py --provider rl_engine.mhc.fp32_gemm_rms:TritonGemmRMSProvider --device cuda
python scripts/check_operator.py --op fp32_gemm_rms --candidate cuda --device cuda \
    --dtype fp32 --batch 1 --seq 257 --k-dim 16384 --check-grad
python scripts/check_operator.py --op fp32_gemm_rms --candidate triton --device cuda \
    --dtype fp32 --batch 1 --seq 257 --k-dim 16384 --check-grad
```

The gtest op class is `mhc_controller` with `atol = rtol = 0` (bitwise) for
FP32 in all four judgments; other dtypes are out of scope because the
controller path never leaves FP32.

## Benchmark

```bash
python benchmarks/benchmark_fp32_gemm_rms.py --tokens 1,16,128,512 --k 16384 \
    --warmup 5 --iterations 20 --json benchmark.json
```

torch-native (cuBLAS matmul + fused torch RMS) is an unconstrained,
different-contract reference: it may reassociate reductions freely, which
the strict contract forbids. On the H200 the strict CUDA backend runs the
full forward+backward step 1.16-2.3x faster than torch-native at every
benchmarked shape (T = 1 to 512, K = 16384), and forward alone wins or ties
for T <= 128. Forward-only trails torch-native at larger T: the frozen
ascending fold gives every output a ~33 us dependent-add chain at K = 16384
that no launch geometry can shorten, while cuBLAS is free to tree-reduce.
