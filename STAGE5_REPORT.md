# Stage 5 — SM80 `linear_logp` productionization

## 1. Production architecture

The production backend keeps the validated Stage 4 architecture unchanged:

- SM80 WMMA BF16 to FP32
- two-stage `cp.async` pipeline
- BM=32, BN=128, BK=32, 8 warps/CTA
- split-V online softmax with `O(N * split_v)` FP32 workspace
- `split_v=32` for `N<=512`, otherwise `split_v=16`
- no `[N,V]` logits materialization

The native CUDA entry point is deliberately restricted to A100/SM80, BF16,
`D=4096`, `V=128256`, `0<N<=1024`, bias-free, forward-only, single-GPU calls.
The Python op delegates every unsupported configuration to the existing Triton
backend (or PyTorch where Triton is unavailable).

## 2. Files changed

- `csrc/cuda/fused_linear_logp_sm80.cu`: production validation, internal
  primary/combine launchers, current-stream launches, benchmark-backed split
  policy.
- `csrc/ops.cpp`: one public binding, `_C.fused_linear_logp_sm80`; experimental
  split/primary/combine profiling bindings removed.
- `setup.py`: normal NVIDIA CUDA build includes the SM80 source; ROCm does not.
- `rl_engine/kernels/ops/cuda/loss/linear_logp_sm80.py`: production
  `FusedLinearLogpSM80Op`, validation, provenance, flatten/restore, and fallback.
- `rl_engine/kernels/registry.py`: SM80 backend registration without changing
  SM90, ROCm, CPU, or other device priorities.
- `tests/test_linear_logp_sm80.py`: correctness, ownership boundaries,
  dispatch/fallback, SM90 isolation, unsupported inputs, and custom stream.
- `benchmarks/benchmark_linear_logp_sm80.py`: reproducible 5-warmup/20-iteration
  A100 benchmark.
- `benchmarks/sanitize_linear_logp_sm80.py`: memcheck target for partial token
  blocks, both production split policies, vocab edges, and split boundaries.
- Removed `linear_logp_sm80_poc.py` and historical CSV/JSON/log artifacts.

## 3. Registry policy

| Condition | Selected execution |
|---|---|
| SM90 with existing SM90 symbol | Existing `FusedLinearLogpSM90Op` |
| SM80 and compiled SM80 symbol | `FusedLinearLogpSM80Op` wrapper |
| Wrapper: BF16, D=4096, V=128256, no bias/grad/TP, N<=1024 | Native SM80 |
| Wrapper: N>1024 or any unsupported configuration | Existing Triton path |
| Non-CUDA / Triton unavailable | Existing PyTorch fallback |

The registry performs architecture selection; the wrapper performs runtime
shape and token-count selection. This avoids a runtime-N hack in the static
registry abstraction.

## 4. Formal tests

Final result: **32 passed** (`tests/test_linear_logp_sm80.py` plus the existing
`tests/test_kernel_registry.py`).

Coverage includes:

- native correctness at N=1,16,31,32,33,128,256,512,1024;
- FP32-upcast `F.linear + log_softmax + gather` oracle;
- split-32 and split-16 target ownership at target 0, V-1, split start-1,
  split start, split end-1, and next split start;
- native selection at N=128/512/1024 and Triton fallback at N=2048;
- unsupported dtype, shape, autograd, device/fallback behavior;
- mocked SM80 and SM90 registry capability, proving SM80 does not preempt SM90;
- custom CUDA stream correctness and workspace lifetime.

Across the final benchmark, native max absolute error was `1.77e-5` to
`2.48e-5`, mean absolute error was `5.52e-6` to `6.00e-6`, and all outputs were
finite.

## 5. Stream and sanitizer

- Launchers use `at::cuda::getCurrentCUDAStream()`; no `cudaDeviceSynchronize`
  or default-stream dependency is present.
- Custom-stream test passed.
- `compute-sanitizer --tool memcheck` covered N=1,31,32,33,128 (split-32) and
  N=513 (split-16), including vocab/split boundary targets.
- Result: **`ERROR SUMMARY: 0 errors`**.

## 6. Final benchmark

A100 80GB PCIe, GPU 1, BF16, D=4096, V=128256, 5 warmups, 20 iterations.

| N | cuBLAS materialized ms | Triton ms | production ms | selected backend | Triton / production |
|---:|---:|---:|---:|---|---:|
| 128 | 0.977 | 61.512 | 7.927 | native SM80 | 7.76x |
| 256 | 1.642 | 61.306 | 11.461 | native SM80 | 5.35x |
| 512 | 3.128 | 61.317 | 18.824 | native SM80 | 3.26x |
| 1024 | 6.452 | 61.530 | 37.271 | native SM80 | 1.65x |
| 2048 | 12.752 | 62.565 | 62.581 | Triton fallback | 1.00x |
| 4096 | 25.562 | 82.944 | 82.930 | Triton fallback | 1.00x |

| N | production tokens/s | peak activation MiB | workspace policy | max abs error | mean abs error |
|---:|---:|---:|---|---:|---:|
| 128 | 16,147 | 0.048 | split-32 | 1.77e-5 | 5.76e-6 |
| 256 | 22,336 | 0.096 | split-32 | 2.19e-5 | 5.52e-6 |
| 512 | 27,200 | 0.191 | split-32 | 2.19e-5 | 6.00e-6 |
| 1024 | 27,475 | 0.195 | split-16 | 2.48e-5 | 5.83e-6 |
| 2048 | 32,726 | 0.023 | Triton, no split workspace | 2.19e-5 | 5.82e-6 |
| 4096 | 49,391 | 0.047 | Triton, no split workspace | 2.77e-5 | 5.84e-6 |

The policy captures the small/medium-N gain and prevents the measured Stage 4
native regressions at N=2048 and N=4096.

## 7. Code cleanup

Removed PoC naming, explicit experimental split binding, isolated primary and
combine profiling APIs, and historical CSV/JSON/log outputs. No hand-written
`mma.sync` experiment, debug kernel, absolute machine path, build artifact,
cache, or shared object remains in the final tree.

## 8. Git diff summary

The final change consists of three logical areas:

1. production CUDA backend and one stable C++ binding;
2. runtime wrapper plus isolated SM80 registry integration;
3. formal tests, sanitizer target, reproducible benchmark, report, and artifact
   cleanup.

## 9. PR readiness

**A. Ready for PR?** Yes.

**B. Correctness or safety blocker?** None found. Numerical error remains at the
validated `~1e-5` level, custom-stream behavior passes, and memcheck is clean.

**C. Existing backend regression?** None observed. SM90 remains higher priority
on SM90; ROCm/CPU mappings are unchanged; unsupported SM80 inputs explicitly
delegate to existing implementations; large N now avoids the native regression.

**D. Suggested PR title**

`cuda: productionize fused SM80 linear_logp for A100`

**E. Draft PR description**

> Adds a production A100/SM80 BF16 fused `linear_logp` backend using the
> benchmark-validated WMMA + two-stage cp.async + split-V architecture. The
> registry selects an SM80 wrapper only on cc 8.0; the wrapper uses native CUDA
> for D=4096/V=128256 forward-only inputs with N<=1024 and transparently falls
> back to Triton otherwise. This yields 7.76x/5.35x/3.26x/1.65x speedups over
> Triton at N=128/256/512/1024 while retaining Triton performance at N>=2048,
> without materializing `[N,V]`. Includes correctness, split-boundary,
> dispatch, SM90-isolation, custom-stream, and compute-sanitizer coverage.
