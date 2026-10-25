# F2 Triton SM80 warp tuning

Selected **1 warp per head on SM80**, retaining 4 warps for other architectures.
The 1/2/4-warp sweep passed bitwise CUDA comparisons for forward outputs, saved
FP32 rstd and input gradients across all 30 tested configurations. With the selected
configuration enabled, **80 targeted regression tests passed without skips**.
All 24 archived fixed-data tensors and their serialized output SHA-256 remain
identical to the earlier baseline. Arithmetic, reduction tree, rsqrt and BF16
rounding points were not changed; fast math remains disabled.

Environment: A100 80GB PCIe (SM80), CUDA 13.0, PyTorch 2.14.1+cu130, Triton 3.8.0.
Remote workspace: `/workspace/rlk-f2-integration-e3gNYR1a`.
Interpreter: `/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python`.

## Sweep method

`sweep.py` imports `bits_equal` from the preceding integration experiment's
`probe.py`; place both scripts together with the repository on PYTHONPATH.

```bash
PYTHONPATH=. python sweep.py --inputs artifacts-jit/inputs.pt --out warp-results.json
```

T=1,128,257,2048,8192, Q32/K2/D128, local/global, warps=4,1,2. T=257 uses the
archived stress inputs, gradients and exact BF16 tables; other shapes use seeded
random Q/K and gradients and the same tables. Every configuration compares six
result tensors against the checked native CUDA implementation.

Timing directly launches Triton kernels with preallocated outputs and valid inputs.
After five warmups, a CUDA graph captures 50 raw kernel calls; seven replays are
measured with CUDA events, divided by 50. These are device-side graph-replay times
per launch, not full API timings. Capturing this internal benchmark does **not**
make the public API graph-compatible: its bounds checks still synchronize.

Local-layer median device times in microseconds:

| Tokens | Warps | Forward | Backward |
| ---: | ---: | ---: | ---: |
| 1 | 4 | 2.46 | 2.40 |
| 1 | 1 | 2.27 | 1.99 |
| 128 | 4 | 7.86 | 7.91 |
| 128 | 1 | 6.00 | 5.55 |
| 2048 | 4 | 108.20 | 115.81 |
| 2048 | 2 | 95.64 | 90.13 |
| 2048 | 1 | 55.34 | 57.43 |
| 8192 | 4 | 424.20 | 452.28 |
| 8192 | 1 | 210.45 | 218.89 |

One warp was fastest in each measured forward/backward case, including global.
At T=2048 local, the measured device-time reductions are approximately 49% forward
and 50% backward. These are hardware/software/shape-specific measurements, not
model-level throughput claims. Clocks were not pinned; fixed scan order and repeated
buffers can affect timings. Full samples are preserved in `warp-results.json`.

## Integrated validation

`warps-pytest.log`: 80 passed in 7.20 seconds via the standard native extension and
the newly selected Triton geometry. Includes empty/sharded shapes, partition and
permutation checks, fixed-tree/VJP parity, FP64 smoke comparisons, malformed inputs,
autograd behavior and dynamic-range tests.

```bash
RL_KERNEL_F2_TEST_JIT=0 python -m pytest \
  tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py \
  tests/build/test_build_platform_collectives.py -q
python probe.py --inputs artifacts-jit/inputs.pt \
  --compare artifacts-jit/outputs.pt --out warps-after
```

`warps-after/results.json` records complete call timings after integration. T=2048
local Triton measured 172.3 us forward and 63.6 us saved-state backward, including
Python/allocation/validation overhead. These differ from raw kernel measurements;
small-shape calls remain dominated by host overhead. Input/output artifact hashes
match the baseline, so identical large tensor files were not downloaded again.

Only SM80 selects the new geometry. SM89/SM90/ROCm keep their existing configuration;
this experiment does not establish cross-GPU parity or validate those platforms.
`source-sha256.json` identifies the implementation and sweep script.
