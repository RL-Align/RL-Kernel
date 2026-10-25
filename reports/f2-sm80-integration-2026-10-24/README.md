# F2 A100 integration, reproducible fixtures and initial timings

The standard `rl_engine._C` extension built successfully on A100 after removing
its forced host C++17 flag. PyTorch now supplies the required C++ standard (C++20
for this installation). **73 targeted tests passed without skips** through the
native extension: 68 F2 tests plus 5 build-configuration tests. This is not the
repository's entire test suite or a wheel-install test.

Environment: A100 80GB PCIe SM80, CUDA 13.0.88, PyTorch 2.14.1+cu130, Triton 3.8.0,
Python 3.12.15. See the preceding `../f2-sm80-2026-10-24/environment.txt`.
Remote directory: `/workspace/rlk-f2-integration-e3gNYR1a`.
Interpreter: `/workspace/rlk-f2-sm80-DpNQapig/test-venv/bin/python`.

## Build and validation

`build-original.log` records PyTorch's explicit C++20 requirement error under the
previous forced C++17 flag. `build-fixed.log` records the successful standard build:

```bash
RL_KERNEL_REQUIRE_EXT=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
  python setup.py build_ext --inplace
RL_KERNEL_F2_TEST_JIT=0 python -m pytest \
  tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py \
  tests/build/test_build_platform_collectives.py -q
```

The standard build does not enable fast math. F2's CUDA arithmetic uses explicit
rounding intrinsics; Triton disables fusion and FTZ. Standard build flags are
preserved in `build/temp.linux-x86_64-cpython-312/build.ninja`. No operator arithmetic
was changed. The build-standard change also affects ROCm, which was not tested.

## Fixed cross-device artifacts

`artifacts-jit/inputs.pt` contains CPU BF16 Q/K, upstream gradients, exact BF16
CUDA-generated cos/sin tables, and int64 positions. It includes 257 tokens,
32 Q heads, 2 K heads, dimensions 128, scales 2^-60 to 2^60 and a zero row.
`artifacts-jit/outputs.pt` stores results for local/global and CUDA/Triton, each in
order: Q output, K output, Q rstd, K rstd, Q gradient, K gradient.

CUDA and Triton matched bitwise for these data. The standard-build rerun also
matched all 24 result tensors from the isolated JIT build. Serialized input and
output SHA-256 values were identical between runs and verified after download.
These are same-device comparisons, not evidence of A100/4090/H100 bitwise parity.

Run from a repository root with a built extension, supplying paths to the archived
inputs and baseline outputs (do not regenerate the table on the target device):

```bash
PYTHONPATH=. python /path/to/probe.py \
  --inputs /path/to/artifacts-jit/inputs.pt \
  --compare /path/to/artifacts-jit/outputs.pt \
  --out /path/to/new-device-results
```

Input/output files total about 54 MiB and are local experiment artifacts; they need
not be included in a source PR. `source-sha256.json` identifies key source files.

## Timing method and limits

`artifacts-native/results.json` contains the final measurements after standard
compilation completed. Each case warms up five calls, then records seven samples
of ten calls with synchronization at each sample boundary. Values below are median
microseconds per call, using Q32/K2/D128 and T=2048:

| Layer | Backend | Forward | Backward with saved state |
| --- | --- | ---: | ---: |
| Global | CUDA | 84.2 | 96.0 |
| Global | Triton | 87.7 | 101.9 |
| Local | CUDA | 205.7 | 173.0 |
| Local | Triton | 206.1 | 121.4 |

Also measured T=1 and 128; full samples are in the JSON. This measures synchronized
wall time including Python dispatch, allocation, validation and GPU work, not
kernel-only latency. Backward calls the dispatcher with saved rstd, without forward
or autograd-engine overhead. CUDA local backward validates positions; the internal
Triton backward relies on previously validated inputs, so these timings include
different host overheads. No compiled/eager baseline, speedup or saturation claim
is made. GPU clocks were not pinned. Further optimization requires profiling and
separate kernel-level measurements rather than extrapolating from this baseline.

## Remaining work

Cross-GPU comparisons, H100/ROCm validation, production table identity enforcement,
runtime integration and agreed final error thresholds remain open. The archived
fixtures make future cross-GPU comparison possible without retaining this instance.
