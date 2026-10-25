# F2 SM80 validation

Result: **68 passed, no skips**, in 59.86 seconds including isolated compilation.
This is test-suite duration, not operator latency. See `pytest.log`.

## Environment

- NVIDIA A100 80GB PCIe, compute capability 8.0, driver 590.48.01.
- CUDA toolkit 13.0.88; PyTorch 2.14.1+cu130; Triton 3.8.0; Python 3.12.15.
- Remote workspace: `/workspace/rlk-f2-sm80-DpNQapig`.
- Isolated `test-venv` inherits `/venv/main` packages; pytest 9.1.1 and ninja
  1.13.2 were installed only in the test environment.
- Native CUDA target `-gencode=arch=compute_80,code=sm_80` confirmed in `build.ninja`.
- CUDA flags: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.
  Triton FP fusion and NVIDIA libdevice FTZ disabled. No fast math enabled.
- `environment.txt`, `setup.log`, and `source-sha256.json` preserve provenance.

## Reproduce

From the remote workspace:

```bash
RL_KERNEL_F2_TEST_JIT=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 \
TORCH_EXTENSIONS_DIR=/workspace/rlk-f2-sm80-DpNQapig/build \
  test-venv/bin/python -m pytest \
  tests/ops/rope/test_qknorm_scaled_rope.py \
  tests/ops/rope/test_qknorm_scaled_rope_forward.py -q
```

## Coverage and interpretation

The unchanged 68-test suite previously run on RTX 4090 passes on A100:

- Local/global forward and first-order backward, CUDA and Triton.
- Bitwise parity against declared fixed-tree forward and explicit VJP references;
  direct CUDA/Triton parity on this device, including saved FP32 rstd.
- Independent FP64 forward/gradient smoke bounds, not final acceptance thresholds.
- Token partition/permutation invariance and gradient repeatability.
- Empty inputs, noncontiguous upstream gradients, unused outputs, invalid inputs,
  saved-input mutation and rejection of higher derivatives.
- Numerical stress over scales 2^-60 through 2^60, sparse/alternating-sign/near-zero
  rows, and repeated maximum valid RoPE positions.

This establishes execution and correctness for the tested cases on SM80. It does
not establish direct A100/4090 bitwise parity: identical serialized inputs/tables
and outputs were not exchanged between devices. Their toolkits also differ
(CUDA 13.0 versus 13.2). No H100/ROCm or performance tests were performed here.

The extension was built through the isolated JIT fixture with PyTorch's default
C++20 standard. The repository-wide extension build and its C++17 compatibility
issue remain unvalidated. Production table identity enforcement, runtime integration,
final numerical acceptance thresholds and broader nonfinite behavior remain open.
