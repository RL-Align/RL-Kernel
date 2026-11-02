## det_gemm overhead — NVIDIA H100 80GB HBM3 (SM90)

| shape | M | K | N | cuBLAS tf32 | cuBLAS fp32 | det CUDA | det Triton | overhead |
|---|---|---|---|---|---|---|---|---|
| qkv | 4096 | 4096 | 12288 | 0.538 | 0.538 | 3.280 | 1.421 | 6.1x |
| o_proj | 4096 | 4096 | 4096 | 0.190 | 0.190 | 1.164 | 0.478 | 6.1x |
| mlp_up | 4096 | 4096 | 14336 | 0.656 | 0.704 | 3.800 | 1.688 | 5.4x |
| mlp_dn | 4096 | 14336 | 4096 | 0.629 | 0.685 | 3.779 | 1.787 | 5.5x |
| lm_head | 4096 | 4096 | 32000 | 1.513 | 1.528 | 8.269 | 3.897 | 5.4x |

_Overhead = det CUDA vs cuBLAS (TF32 disabled). The det CUDA path uses SM90 TMA + mma.sync tensor cores with a fixed single-CTA-per-tile schedule (no split-K) for bitwise batch-invariance; both det paths trade speed for invariance. Throughput tuning is deferred per #146._
