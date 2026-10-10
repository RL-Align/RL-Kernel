# Qwen FFN H100 kernel profile

These charts are aggregated from the Nsight Systems `cuda_gpu_kern_sum` report
for the same `tokens=1,8`, `hidden=4096`, `intermediate=12288` profiling run.
The benchmark correctness checks passed in both worktrees.

| Profile | Kernel | Total time | Launches | Average |
|---|---|---:|---:|---:|
| Before | `det_gemm_sm90_kernel` | 392.262584 ms | 643 | 610.051 us |
| Before | `det_gemm_naive<bf16,true>` | 157.657338 ms | 90 | 1,751.748 us |
| Before | `det_gemm_naive<bf16,false>` | 116.315558 ms | 84 | 1,384.709 us |
| After | `det_gemm_sm90_kernel` | 391.452104 ms | 643 | 608.790 us |
| After | `det_gemm_naive<bf16,false>` | 116.338858 ms | 84 | 1,384.986 us |
| After | `det_gemm_db_small_k<bf16>` | 12.006408 ms | 90 | 133.405 us |

The optimized kernel replaces the 90 `det_gemm_naive<bf16,true>` launches:

- 157.657338 ms → 12.006408 ms;
- 13.13× kernel-time speedup;
- 92.38% reduction for the replaced kernel;
- 145.651 ms saved in the selected deterministic GEMM kernel aggregate.

The unchanged SM90 and `det_gemm_naive<bf16,false>` rows provide a useful
control: the observed reduction is localized to the intended short-token dW
path rather than a general profiling artifact.

Raw `.nsys-rep` files remain on the H100 profiling node because they are large;
the SVGs and this reproducible kernel summary are the review artifacts.
