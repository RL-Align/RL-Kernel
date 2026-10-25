# F2 Nsight timing analysis on A100

Nsight Systems 2025.3.1 captured the current implementation successfully. Nsight
Compute 2025.3.1 failed with **ERR_NVGPUCTRPERM**, because this host restricts GPU
performance counters (`RmProfilingAdminOnly: 1`). Container root did not grant
access. No driver/security settings were changed. Occupancy, bandwidth utilization,
cache-hit rates and scheduler-stall counters were not obtained.

Environment: A100 80GB PCIe, CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0.
Current code includes CUDA bounds-check deduplication, SM80 Triton 1 warp, and SM80
CUDA fixed-tree shared/shuffle reduction. This profiling task changed no operator.

## Workload and artifacts

`workload.py`: five warmups before capture, NVTX labels for 16 cases spanning
T=1/2048, Q32/K2/D128, local/global, CUDA/Triton and forward/saved-state backward.
Each range contains ten calls plus an explicit final device synchronization.
Capture starts/stops with cudaProfilerStart/Stop, excluding compilation and table
construction. Input seed is 501; table is the reference BF16 CUDA-built table.
Backward excludes forward/autograd overhead. Shared inputs/tables are retained
by each closure throughout capture.

`nsight-results/timeline.nsys-rep` opens in the Nsight Systems GUI. The corresponding
SQLite, CSV summaries, nsys log and ncu failure log are preserved beside it.
`analyze.py` groups kernel/API activities wholly within each completed NVTX range
and asserts ten fused-kernel events per range. `summary.json` holds derived metrics.

## Measured bottlenecks

Per-call times under Nsight tracing, microseconds:

| Case | Range wall / 10 | Fused kernel mean | Stream synchronizations / call |
| --- | ---: | ---: | ---: |
| T1 local CUDA forward | 128.5 | 2.9 | 2 |
| T1 local Triton forward | 226.8 | 2.7 | 1 |
| T2048 global CUDA forward | 82.6 | 75.8 | 0 |
| T2048 local CUDA forward | 191.8 | 108.7 | 2 |
| T2048 local CUDA backward | 195.6 | 124.6 | 2 |
| T2048 local Triton forward | 230.4 | 56.2 | 1 |
| T2048 local Triton backward | 69.8 | 58.2 | 0 |

Observed:

- Every CUDA local forward/backward call launches two additional bounds-reduction
  kernels and performs two scalar-read stream synchronizations, consistent with
  `min().item()` and `max().item()` in the native checker.
- Local Triton forward launches three additional elementwise validation kernels
  for T=1, and four validation kernels for T=2048 (including the reduction), with
  one scalar-read stream synchronization. At T=1 the one-element reduction does
  not need another kernel. Triton internal backward has no corresponding check.
- Small inputs have only approximately 2–3 us of fused GPU work per call, while
  the host-inclusive range is much longer. Launch/validation/host scheduling are
  more promising targets than reduction arithmetic for these shapes.
- At T2048, global CUDA fused execution occupies most of the range; local Triton
  forward has a shorter fused kernel but a much larger total range. Fast kernels
  alone do not remove local-path validation overhead.

Inference and limits:

The large gap between wall and fused-kernel time includes other kernels, transfers,
host work, launch gaps and explicit synchronization. It is **not** a measurement
of Python CPU time alone. API calls and GPU work overlap; summing their times is
invalid. Nsight instrumentation perturbs short operations and these wall values
must not replace the preceding unprofiled performance baselines. A single range
per case and unpinned clocks do not support precise speedup claims.

Systems provides launch/resource metadata but not the missing Compute counters.
It cannot establish that these kernels are memory-bandwidth-bound, compute-bound,
occupancy-limited or primarily stalled on barriers. Those claims remain unproven.

## Recommended optimization order

1. Combine CUDA bounds validation into one device-side check and one scalar read,
   preserving fail-closed behavior in native and Python entry points.
2. Fuse Triton position bounds checking into one validation kernel rather than
   multiple eager elementwise/reduction launches, then measure whether host sync
   still dominates. Do not silently cache validation across mutable tensors.
3. Design a separately validated graph/capture-compatible interface if required
   by runtime integration; the current public API does not support graph capture.
4. For large-shape kernel changes, collect Nsight Compute on a host that grants
   performance-counter access before selecting occupancy/bandwidth/stall-specific
   optimizations. Preserve the fixed addition tree and rounding contract.

## Commands used

From `/workspace/rlk-f2-integration-e3gNYR1a`, with Python pointing to
`/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python`:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=nsight-results/timeline python nsight_workload.py
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  --format csv --output nsight-results/stats nsight-results/timeline.nsys-rep
ncu --set basic --profile-from-start off \
  --kernel-name 'regex:.*f2_forward.*' --launch-count 1 \
  --export nsight-results/cuda-local-forward \
  python nsight_workload.py --backend cuda --phase forward --tokens 2048 --local 1 --repeats 1
```

The last command fails with the recorded counter-permission error. The user-visible
source filename `workload.py` was uploaded as `nsight_workload.py`. The analysis can
be reproduced locally using `python reports/f2-nsight-2026-10-24/analyze.py`.
