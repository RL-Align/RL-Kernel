# A100–H100 F2 bitwise validation

**121 tests passed without skips in 152.71 seconds** on H100. The cross-device
probe passed 24 cases / 144 tensor comparisons with **zero bit mismatches**.

Compared BF16 Q/K outputs, FP32 Q/K rstd, and BF16 dQ/dK through public first-order
autograd against the archived A100 VJP v2 baseline. Cases cover local/global,
CUDA/Triton, full 257-token input, token partitions [0:1], [1:8], [8:136],
[136:257], and the first 4 Q/1 K heads. Input spans powers-of-two scales from
2^-60 through 2^60, plus zero/subnormal cases from the original probe.

Input, expected-output and shared-table hashes are verified before running.
The exact A100-built BF16 table is loaded using the trusted P2 identity; it is
not independently rebuilt for the cross-device comparison. The regular 120-test
suite also runs on H100, including independent FP64 tests, analytic large-input
counterexamples, local/global parity, chunking, bounds and table lifecycle tests.

## Environments and scope

- A100 80GB PCIe SM80 baseline: PyTorch 2.14.1+cu130, CUDA 13.0, Triton 3.8.0.
- H100 80GB HBM3 SM90: Python 3.12.15, PyTorch 2.14.1+cu132, nvcc 13.2.86,
  Triton 3.8.0, driver 610.57.04.
- H100 built the isolated F2 extension with `TORCH_CUDA_ARCH_LIST=9.0`,
  `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`; no fast math.
  This does not qualify the entire repository's normal SM90 extension build.
- CUDA uses the original shared-memory reduction on SM90 and the validated
  shared/shuffle mapping on SM80. Triton uses four warps on SM90, one on SM80.
  Despite those mappings and the different CUDA versions, tested bits agree.

This is empirical equality on these two actual environments and these inputs,
not proof for every input, compiler version or GPU. No ROCm/SM89 claim is added.
No sanitizer or performance run on H100 is claimed. Historical reports remain
records of their earlier validation scope.

## Reproduce and artifacts

From the repository root on H100 (source and script uploaded under `/workspace`):

```bash
PYTHONPATH=. RL_KERNEL_F2_TEST_JIT=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2 \
  python -m pytest /workspace/cross_gpu.py \
  tests/ops/rope/test_rope_table_identity.py \
  tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py \
  tests/build/test_build_platform_collectives.py -q
```

`cross_gpu.py` loads `/workspace/inputs.pt` and `/workspace/outputs.pt` (override
with `F2_BASELINE_DIR`), verifies hashes, writes counts even if equality fails,
and saves full H100 outputs. Baseline files are the local integration inputs and
backward-v2 outputs; expected hashes are recorded in `cross-gpu-results.json`.

Evidence: `rlk-f2-h100-tests.log`, `cross-gpu-results.json`, `source-sha256.json`.
The downloaded `h100-outputs.pt` is retained locally, excluded from the proposed PR.
