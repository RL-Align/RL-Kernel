# F2 fused forward validation

Date: 2026-10-24 (America/Los_Angeles).

Result: **37 passed, no skips**, in 75.56 seconds including the isolated build.
This is test-suite duration, not an operator benchmark. See `pytest.log`.

## Environment

- RTX 4090 (SM89), CUDA Toolkit 13.2.
- PyTorch 2.14.1+cu132, Triton 3.8.0, Python 3.12.15.
- Remote directory: `/workspace/rlk-f2-forward-pTXhO0GT`.
- Isolated test venv: `test-venv`, reusing the existing PyTorch environment with
  pytest/ninja installed for testing. Existing project files and SSH keys were not modified.

## Implementation and coverage

The CUDA and Triton kernels fuse head-local weightless RMSNorm, Q-only scaling,
and optional rotate-half RoPE. They use the proposed seven-stage FP32 half-split
tree, the rsqrt candidates from the preceding experiment, and explicit BF16 casts.
Both consume the same table, built on CUDA by the reference helper for these tests.

The test suite covers:

- BF16 Q/K and FP32 rstd bit-pattern equality with the fixed-tree reference.
- Direct CUDA versus Triton equality for local and global variants.
- Empty, single-token, non-power-of-two and representative head-count inputs.
- Token partitioning and permutation, preserving absolute positions.
- Positions 0, 1, 2047, 2048, 2049, 65535 and 131071.
- Independent FP64 forward error smoke bounds, and the prior reference tests.
- Input immutability, out-of-range rejection, direct CUDA binding validation,
  missing CPU fallback, and explicit rejection of unsupported training calls.

The forward returns FP32 rstd for a future custom backward. Returning this state
does not implement autograd: calls requiring Q/K gradients are rejected unless
gradient recording is disabled explicitly.

## Reproduction

From the repository root on this GPU, with PyTorch, Triton, pytest and ninja available:

```bash
RL_KERNEL_F2_TEST_JIT=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pytest tests/ops/rope/test_qknorm_scaled_rope.py \
    tests/ops/rope/test_qknorm_scaled_rope_forward.py -q
```

The test fixture builds only the F2 source using `torch.utils.cpp_extension.load_inline`.
CUDA flags explicitly include `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`;
fast math is not enabled. Triton disables FP fusion and NVIDIA libdevice FTZ.

The first CUDA build failed because the fixture forced C++17 and this PyTorch version
requires C++20. The corrected fixture lets PyTorch choose its default standard.
The full repository extension build still specifies C++17 and was not validated here;
the isolated build must not be reported as full build qualification.

## Remaining work

Custom backward/autograd integration, benchmark/optimization, runtime trace binding,
table hash/provenance enforcement, CUDA graph-compatible validation, and A100/H100/ROCm
qualification remain outstanding. The provisional FP64 bounds are not final acceptance
tolerances. These results do not establish end-to-end model train/rollout consistency.
