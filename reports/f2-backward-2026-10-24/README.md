# F2 first order backward validation

Date: 2026-10-24 (America/Los_Angeles).

Result: **64 passed, no skips**, in 79.04 seconds including the isolated build.
See `pytest.log`. Suite duration is not an operator latency measurement.

Follow-up numerical stress regression: **68 passed, no skips**, in 4.12 seconds
with the compiled extension cached; see `stress-pytest.log`. The same environment
and compilation options below were used. Added CUDA and Triton checks for both
local and global layers over 257 tokens, input scales from 2^-60 to 2^60, zero,
near-zero, sparse and alternating-sign rows, and repeated maximum valid positions.
Forward outputs, saved rstd and first-order gradients match the declared arithmetic
reference bitwise; outputs and gradients are finite. This checks arithmetic parity,
not FP64 error bounds across that entire range. FP32 overflow/nonfinite inputs and
cross-architecture consistency remain outside this experiment.

## Environment and reproduction

- RTX 4090 (SM89), CUDA Toolkit 13.2.
- PyTorch 2.14.1+cu132, Triton 3.8.0, Python 3.12.15.
- Remote directory: `/workspace/rlk-f2-backward-kNuzuyRC`.
- Test interpreter reused from `/workspace/rlk-f2-forward-pTXhO0GT/test-venv/bin/python`.
- CUDA compile options: `--fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.
- Triton: FP fusion disabled, NVIDIA libdevice FTZ disabled. Fast math was not enabled.

```bash
RL_KERNEL_F2_TEST_JIT=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 \
  python -m pytest tests/ops/rope/test_qknorm_scaled_rope.py \
    tests/ops/rope/test_qknorm_scaled_rope_forward.py -q
```

The second file now covers forward and backward. The fixture compiles both bindings
from the F2 CUDA source using the installed PyTorch's default C++ standard. The
repository-wide extension build was not run; its C++17 compatibility issue with
PyTorch 2.14 remains separate from this isolated C++20 build.

## Implemented backward arithmetic

Local RoPE uses the transpose with separately BF16-rounded gradient products and
sum. Q gradients are scaled by FP32 3.87 and rounded to BF16. Norm backward uses
saved FP32 rstd, a seven-stage fixed-tree dot product, and explicit FP32 grouping:

```text
dot = fixed_sum_128(g * x)
r2 = r * r
r3 = r2 * r
coefficient = (dot * (1/128)) * r3
dx = BF16((r * g) - (x * coefficient))
```

This is the proposed strict first-order VJP, not a claim of bitwise equivalence to
PyTorch eager-autograd scheduling. The independent FP64 derivative and the explicit
VJP reference serve different purposes: mathematical correctness and arithmetic parity.

## Checks passed

- Existing forward/reference regression tests.
- CUDA/Triton gradients bitwise equal to the explicit VJP reference.
- Direct CUDA versus Triton gradient parity, both local and global.
- Gradient repeatability, token partitioning and permutation invariance.
- FP64 gradient comparisons per head, including zero and near-zero input heads;
  the provisional normalized L2 bound of 0.03 was unchanged.
- Empty inputs, single-token/full-model heads and a shard-shaped input.
- Noncontiguous upstream gradients; table gradients intentionally absent.
- Either output unused; the corresponding input gradient remains absent.
- Saved-input mutation rejected by autograd version checking.
- Higher derivatives rejected; malformed saved rstd rejected by raw CUDA binding.

## API and limits

Use `qknorm_scaled_rope(..., backend="cuda" or "triton", is_local=...)` for training.
It returns only Q/K and supplies custom autograd. The lower-level
`qknorm_scaled_rope_forward` still rejects gradient-enabled training calls to avoid
silent detach; it returns Q/K plus FP32 saved rstd for explicit forward inspection.

Only first-order derivatives are supported. Production runtime trace/registry binding,
table identity enforcement, CUDA graph capture, performance benchmarks, final error
thresholds and A100/H100/ROCm validation remain outstanding. These tests do not establish
end-to-end model train/rollout consistency or complete RFC acceptance.
