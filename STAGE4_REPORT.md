# A100/SM80 native `linear_logp` — Stage 4 report

- Date: 2026-09-22
- Branch: `sm80-linear-logp`
- GPU: NVIDIA A100 80GB PCIe / SM80, GPU 1
- Workload: BF16, D=4096, V=128256, forward-only, single GPU
- Benchmark: CUDA events; final results use 5 warmups and 20 iterations

## 1. Stage 3 recap

Stage 3 showed that the hand-written `mma.sync.m16n8k16` experiment was negative: registers increased from 72 to 80 and N=4096 regressed from 246 ms to 337 ms. That route was not continued.

Split-V was positive. Fixed split=16 reached 10.5 ms at N=128 and 143.7 ms at N=4096, but the latter remained 1.7x slower than Triton and showed a large-N plateau. Stage 4 therefore kept the Stage 2 WMMA/cp.async primary kernel and optimized split selection and resource residency.

## 2. Full split sweep

All entries below are measured total latency in milliseconds. Only split count changes; BM/BN/BK=32/128/32, 8 warps, WMMA and two-stage cp.async remain fixed.

| N | split1 | split2 | split4 | split8 | split16 | split32 | measured best |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 166.08 | 83.09 | 41.60 | 20.90 | 10.48 | **7.85** | 32 |
| 256 | 166.43 | 83.03 | 41.61 | 20.94 | 15.53 | **11.18** | 32 |
| 512 | 166.35 | 83.13 | 41.73 | 31.02 | 22.14 | **18.58** | 32 |
| 1024 | 166.53 | 83.32 | 61.78 | 44.15 | **36.80** | 37.59 | 16 |
| 2048 | 166.71 | 123.26 | 88.02 | 73.32 | 75.02 | **72.17** | 32 |
| 4096 | 246.43 | 176.19 | 146.57 | 149.08 | **145.22** | 145.93 | 16 |

Primary CTA count is `ceil(N/32)*split`; workspace is exactly `12*N*split` bytes. Examples:

| N | split | primary CTA | workspace |
|---:|---:|---:|---:|
| 128 | 32 | 128 | 48 KiB |
| 1024 | 16 | 512 | 192 KiB |
| 2048 | 16 | 1024 | 384 KiB |
| 4096 | 16 | 2048 | 768 KiB |

All split configurations produced finite outputs. Differences from split=1 were at most 1.91e-6, caused by the expected different order of the final FP32 partial reduction.

## 3. Shape-aware policy

The final deterministic policy is:

```cpp
split_v = (N <= 512) ? 32 : 16;
```

Rationale:

- N=128/256/512: split=32 is clearly fastest and creates enough CTAs to fill the A100.
- N=1024: split=16 is the measured optimum.
- N=2048: split=32 was only about 4% faster than split=16 in the sweep, while doubling workspace and CTA count. The simpler split=16 policy was retained.
- N=4096: split=16 was the measured optimum, although split=4/16/32 were within about 1%.

This deliberately avoids fitting a special N=2048 branch to a small noisy difference.

## 4. Primary versus combine

Measured independently with CUDA events using each shape's selected split:

| N | split | primary ms | combine ms | total ms | combine / total |
|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 7.8356 | 0.0071 | 7.8453 | 0.091% |
| 1024 | 16 | 36.7928 | 0.0070 | 36.7842 | 0.019% |
| 4096 | 16 | 143.3666 | 0.0117 | 143.3712 | 0.008% |

Combine is negligible, so no combine optimization was performed. Nearly all remaining time is in the primary vocabulary scan.

## 5. Large-N analysis

Measured facts:

- N=4096 changes little across split=4/8/16/32: 146.6/149.1/145.2/145.9 ms.
- The chosen N=4096 split=16 grid contains 2048 primary CTAs.
- N=8192 with split=16 reaches 288.96 ms versus Triton 125.24 ms.
- Workspace/combine cost remains tiny; combine is only 0.008% of N=4096 total.

Estimated, not hardware-counter measurements:

- With 56 registers/thread, 31,744 B shared/CTA and 256 threads, at most four primary CTAs can reside per SM. On 108 SMs, N=4096 split=16 represents about 4.7 CTA waves.
- Total weight work does not grow with split because splits partition V, but finer splits increase CTA scheduling, partial writes, hidden-tile duplication boundaries and loss of long per-CTA locality.
- The Stage 2 traffic model gives roughly 134.5 GB of weight reads for N=4096. At 143.6 ms this is about 0.94 TB/s effective bandwidth, versus about 0.55 TB/s for the 246 ms non-split kernel. These are workload-model estimates, not Nsight measurements.
- Effective GEMM work is roughly 30 TFLOP/s at N=4096, still far below A100 peak and Triton's effective rate. The plateau is therefore a mixture of primary-kernel instruction/latency cost, occupancy/resource limits, memory-system pressure and excess CTA fragmentation—not combine.

Nsight Compute counters remain unavailable due system profiling permissions.

## 6. Resource usage

A limited resource-pressure experiment added `__launch_bounds__(256,4)` without changing the algorithm.

| Resource | Before | Final |
|---|---:|---:|
| registers/thread | 72 | **56** |
| shared memory/CTA | 31,744 B | 31,744 B |
| local memory | 0 | 0 |
| threads/CTA | 256 | 256 |
| theoretical CTA residency | 3/SM | **4/SM** |
| theoretical warp occupancy | 37.5% | **50%** |

The final N=4096 smoke result improved slightly; N=1024 changed by about 1%. No spilling was reported by cubin resource usage, so the launch-bound version was retained.

## 7. Final benchmark

Latency in milliseconds:

| N | cuBLAS materialized | Triton | Stage 2 non-split | Stage 3 fixed split16 | Stage 4 shape-aware |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.980 | 61.585 | 167.373 | 10.491 | **7.869** |
| 256 | 1.645 | 61.384 | 167.455 | 15.472 | **11.406** |
| 512 | 3.205 | 61.399 | 167.489 | 21.929 | **18.770** |
| 1024 | 6.489 | 61.515 | 167.512 | **36.712** | 37.216 |
| 2048 | 12.908 | 62.593 | 167.608 | 74.716 | **73.691** |
| 4096 | 25.846 | 83.903 | 246.116 | 143.696 | **143.589** |

Stage 4 throughput is 16.3k, 22.4k, 27.3k, 27.5k, 27.8k and 28.5k tokens/s respectively. It beats Triton through N=1024, is 18% slower at N=2048 and 71% slower at N=4096.

Optional N=8192 result: cuBLAS 51.83 ms, Triton 125.24 ms, Stage 4 288.96 ms; activation/workspace is 1.53 MB versus 10,020 MB for materialized cuBLAS.

## 8. Correctness and safety

- All six final shapes: max absolute error 1.72e-5 to 2.77e-5; mean error 5.61e-6 to 5.91e-6.
- N=8192 max/mean error: 3.15e-5 / 5.86e-6.
- No NaN or Inf in any sweep or final benchmark.
- Boundary ownership tests covered every split start, preceding split end, vocab 0 and vocab V-1 for split=16 and split=32.
- Boundary max/mean errors: split16 1.62e-5 / 5.25e-6; split32 2.00e-5 / 5.69e-6.
- Final compute-sanitizer memcheck: `ERROR SUMMARY: 0 errors`.
- No `[N,V]` logits are materialized.
- Final peak activation/workspace: 0.05/0.09/0.19/0.19/0.38/0.77 MB for N=128..4096.

## 9. PR readiness assessment

### A. Production candidate?

Yes, as an SM80 native production candidate with explicit shape-aware split selection. The architecture is stable, memory-safe, numerically accurate, has bounded O(N*split) workspace, and substantially outperforms Triton for N<=1024.

### B. Remaining blockers?

There is no known correctness or memory-safety blocker. There is a performance limitation for N>=2048: the native kernel is slower than Triton, increasingly so at N=4096/8192. Therefore it should not unconditionally replace Triton for every shape in registry dispatch.

Before integration, formal tests must cover supported dtype/device/shape validation, random and boundary targets, all heuristic branches, stream behavior, and fallback behavior. The experimental primary/combine profiling bindings should be kept private or removed during cleanup.

### C. Recommended Stage 5

Proceed to registry integration, formal tests, cleanup and PR preparation, with a benchmark-backed fallback policy: prefer this native path for the proven small/medium range (currently N<=1024) and retain Triton for larger N unless subsequent optimization closes the gap. Further kernel optimization is optional for broadening the native range, not required to integrate the already-profitable range.

Stage 5 was not started.
