# F2 position-validation fusion

Implemented one GPU predicate kernel and one host scalar read for each required
position-range validation. **94 targeted tests passed**, fixed-data outputs remain
bitwise identical, and Nsight Systems confirms fewer validation kernels/syncs.

## Changes

- CUDA native forward/backward: replace `min().item()` and `max().item()` with
  `f2_check_positions`. A 256-thread block scans int64 positions in strided chunks,
  combines invalid flags using `__syncthreads_or`, and writes one int32 scalar.
- Triton forward: replace eager comparisons, OR and `any()` with a single
  `_check_positions_kernel`. One program scans 256-element chunks, reduces integer
  invalid flags and writes one int32 scalar for `.item()`.
- Existing shape/dtype/device/layout checks, error types and native entry-point
  protection remain. Empty positions skip the kernel. No validity result is cached.
- This is an integer predicate reduction, separate from the fixed FP32 norm/dot
  reduction. No floating-point arithmetic, casts, rsqrt or RoPE math changed.
- A host read remains: public graph capture is still unsupported. A single-block
  scan prioritizes low launch overhead; very large token counts may need a separate
  scalable design. No long-input performance claim is made beyond measured shapes.

## Validation

Environment: A100 80GB PCIe SM80, CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0.
Standard native extension rebuilt successfully. Remote directory is
`/workspace/rlk-f2-integration-e3gNYR1a`; Python is
`/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python`.

`fused-bounds-evidence/pytest.log`: **94 passed, no skips**, in 7.35 seconds.
Added tests cover indices 255,256,1024, int64 minimum/maximum, a 131073-element
Triton scan with invalid tail, empty inputs and native backward revalidation.
Existing tests retain invalid/empty-table rejection, mutation rechecking, FP64
smoke comparisons, forward/backward parity and invariance. Local CPU references:
14 passed. Formatting and diff checks passed.

Compute Sanitizer Memcheck and Synccheck each ran the 14 new boundary tests:
**0 errors** for both tools. Logs are archived. ROCm and other NVIDIA architectures
were not tested for this newly added validation implementation.

Fixed-data input and output SHA-256 values match the previous baseline. Output hash:
`262a18bde3fc01848225b22b0945074b07e7fe163243498fde8f5b4e405be88a`.

## Performance: unprofiled paired calls

Same-instance median synchronized wall times, microseconds:

| Tokens | Backend | Local forward before → after | Local saved-state backward before → after |
| ---: | --- | ---: | ---: |
| 1 | CUDA | 78.1 → 50.1 | 52.9 → 29.1 |
| 128 | CUDA | 80.8 → 51.7 | 54.4 → 30.7 |
| 2048 | CUDA | 156.7 → 134.8 | 163.0 → 146.2 |
| 1 | Triton | 165.0 → 135.4 | 45.8 → 45.2 |
| 128 | Triton | 163.7 → 137.2 | 45.7 → 45.4 |
| 2048 | Triton | 172.0 → 143.3 | 63.6 → 63.6 |

Global controls are approximately unchanged; e.g. T2048 CUDA forward 74.8 → 74.6 us.
These are call-level timings including validation, allocation and launches, using
five warmups and seven samples of ten calls. Clocks are not pinned; these results
are not training-throughput or pure-kernel speedup claims. Full samples are in
`fused-bounds-before/results.json` and `fused-bounds-after/results.json`.

## Nsight confirmation

The same 16-case NVTX workload from `../f2-nsight-2026-10-24/workload.py` was rerun
with Nsight Systems. `summary.json` and `analyze.py` derive per-range statistics:

| Path | Validation kernels per call, before → after | Stream syncs per call, before → after |
| --- | ---: | ---: |
| CUDA local forward/backward | 2 → 1 | 2 → 1 |
| Triton local forward, T1 | 3 → 1 | 1 → 1 |
| Triton local forward, T2048 | 4 → 1 | 1 → 1 |

Global and Triton internal backward have no added range check. Fused compute-kernel
execution time remains approximately unchanged. The wall-time improvement comes
from the surrounding validation path. Instrumented times are not mixed with the
unprofiled performance table above. Nsight Compute counter access is still blocked
by the host, as documented in the earlier Nsight report.

`fused-bounds-evidence/timeline.nsys-rep` opens in Nsight Systems; its SQLite export,
CSV statistics and profiling logs are saved alongside it.

## Reproduce

Use the earlier `probe.py` and archived fixed tensors, plus `nsight_workload.py`:

```bash
RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 python setup.py build_ext --inplace
RL_KERNEL_F2_TEST_JIT=0 python -m pytest tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py tests/build/test_build_platform_collectives.py -q
python probe.py --inputs artifacts-jit/inputs.pt --compare artifacts-jit/outputs.pt --out fused-bounds-after
RL_KERNEL_F2_TEST_JIT=0 compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -m pytest tests/ops/rope/test_qknorm_scaled_rope_forward.py -q \
  -k 'fused_bounds_scan or bounds_scan_large or native_backward_rechecks'
```

Repeat the sanitizer command with `--tool synccheck` for synchronization checks.
Nsight capture options are unchanged from the preceding Nsight report. All source
fingerprints for this change are in `source-sha256.json`.
