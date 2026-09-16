# RMSNorm with a Residual Fork

## Summary

`rmsnorm_residual` implements the rank-local P1-5 task in
[the DSv4 P1 starter kit](https://github.com/RL-Align/RL-Kernel/pull/383).
It normalizes a BF16 activation and returns an independent copy of the original
activation for the residual branch. It does **not** add another residual input
before normalization.

```text
r[t]           = rsqrt(sum_d(FP32(x[t,d])²) / D + eps)
y[t,d]        = BF16((FP32(x[t,d]) * r[t]) * gamma[d])
residual[t,d] = x[t,d]
```

## Entry Point

Use the registry for ordinary autograd-enabled calls:

```python
import torch
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("rmsnorm_residual", device="cuda")
x = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
gamma = torch.ones(4096, device="cuda", dtype=torch.float32, requires_grad=True)

y, residual = op(x, gamma, eps=1e-6)
torch.autograd.backward(
    (y, residual), (torch.ones_like(y), torch.ones_like(residual))
)
assert x.grad.dtype == torch.bfloat16
assert gamma.grad.dtype == torch.float32
```

For explicit forward/backward calls, select a backend directly:

```python
from rl_engine.kernels.ops.cuda.norm.rmsnorm_residual import (
    cuda_rmsnorm_residual_fwd,
    cuda_rmsnorm_residual_bwd,
)

# Reuse x and gamma from the example above, without an autograd graph.
x0, gamma0 = x.detach(), gamma.detach()
y, residual, saved = cuda_rmsnorm_residual_fwd(x0, gamma0, eps=1e-6)
dx, dgamma = cuda_rmsnorm_residual_bwd(
    torch.ones_like(y), torch.ones_like(residual), x0, gamma0, saved
)
assert dx.dtype == dgamma.dtype == torch.float32
```

The Triton module exposes the corresponding `triton_rmsnorm_residual_fwd` and
`triton_rmsnorm_residual_bwd` functions with the same argument order. Use the
input, gain, and saved state from the matching forward call without mutation.

## Tensor Contract

| Argument/result | Shape | Dtype | Requirements |
|---|---|---|---|
| `x` | `[T,D]` | BF16 | Input activation |
| `gamma` | `[D]` | FP32 | Gain retained in FP32, not first rounded to BF16 |
| `dy`, `d_residual` | `[T,D]` | BF16 | Upstream gradients of the two outputs |
| `y` | `[T,D]` | BF16 | Final normalized output |
| `residual` | `[T,D]` | BF16 | Byte-preserving copy; does not alias `x` |
| `saved["r"]` | `[T]` | FP32 | Inverse RMS from forward |
| `saved["d"]` | Scalar | Python integer | Hidden dimension `D` |
| Explicit `dx` | `[T,D]` | FP32 | Sum of normalization and residual gradients |
| Explicit `dgamma` | `[D]` | FP32 | Gradient reduced across tokens |

CUDA and Triton require tensors on the same NVIDIA CUDA device,
`1 <= T <= 2**31 - 1`, `D` in `{128, 4096}`, and `eps == 1e-6`.
`D=4096` is the production geometry; `D=128` supports reduced P1 fixtures.
Their Python wrappers make contiguous copies of strided inputs; low-level CUDA
extension calls require contiguous tensors.

Autograd exposes only `(y, residual)`, supplies zeros for an unused output's
gradient, and returns gradients in their input dtypes: BF16 `x.grad` and FP32
`gamma.grad`. This differs intentionally from the FP32 explicit backward output.
Double backward is not supported.

## Backends and Dispatch

| Backend | Wrapper | Implementation |
|---|---|---|
| NVIDIA CUDA | `RMSNormResidualCudaOp` | `_C.mhc_rmsnorm_residual_forward` and `_C.mhc_rmsnorm_residual_backward` |
| NVIDIA Triton | `RMSNormResidualTritonOp` | `rl_engine.kernels.ops.triton.rmsnorm_residual_triton` |
| PyTorch reference | `NativeRMSNormResidualOp` | Autograd wrapper around `rl_engine.mhc.oracle` |

The CUDA registry order is CUDA → Triton → PyTorch reference. CUDA construction
checks NVIDIA CUDA availability and both extension symbols; Triton construction
checks NVIDIA CUDA and Triton availability. The registry can try the next backend
when one is unavailable. This is availability fallback, not automatic recovery
from an invalid tensor or a runtime kernel error. CPU uses the PyTorch reference.
There is no dedicated ROCm implementation for this operator.

Explicit backend functions, gtest candidates, and P1 providers do not silently
switch backend. `CudaMHCProvider` and `TritonMHCProvider` implement only P1-5;
the remaining P1 operators stay on the oracle and provenance reports
`+oracle-rest`.

## Accuracy and Numerical Behavior

The reference is `rl_engine/mhc/oracle.py`. Candidate acceptance requires raw-byte
equality for `y`, `residual`, saved `r`, `dx`, and `dgamma`, not just a tolerance
comparison.

- Sum-of-squares and the backward inner product use ascending-index FP32
  left folds over `D`. Weight gradients use an ascending-token FP32 left fold.
- Products and additions round separately; output scaling remains
  `(x * r) * gamma`. No atomic partial accumulation, Split-K, Stream-K, or
  runtime-dependent reduction tree is used.
- Use `rsqrt(mean + eps)`, not `1 / sqrt(mean + eps)` and not the controller's
  `1 / (sqrt(mean) + eps)` formula.
- Keep intermediate values and gamma in FP32. Cast normalized output to BF16
  only at the final store; preserve residual bytes without a conversion.
- The residual copy shares the forward kernel without changing its reduction
  layout. Batching, padding, and input stride normalization do not regroup the
  arithmetic of existing rows.

The FP32-gamma contract is reflected in the seeded fixtures and frozen CPU-oracle
manifest. Regenerating a manifest is a reviewed contract migration, not a way to
make a failing candidate pass. The oracle saves its FP32 input and accepts
`(dy, d_residual, gamma, saved)` in backward; provider/backend interfaces keep
explicit `x`, with an adapter for the reference provider.

## Build and Tests

With the project's PyTorch/CUDA development environment already installed,
rebuild after CUDA/C++ changes:

```bash
env MAX_JOBS=4 RL_KERNEL_REQUIRE_EXT=1 KERNEL_ALIGN_USE_FAST_MATH=0 \
  KERNEL_ALIGN_FORCE_SM90=0 \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps \
  --reinstall --no-cache -e .
```

`uv run --no-sync` does not rebuild the extension. Both native symbols listed
above must exist before CUDA validation.

```bash
uv run --no-sync python -m pytest \
  tests/test_rmsnorm_residual.py \
  tests/test_rmsnorm_residual_triton.py \
  tests/test_rmsnorm_residual_gamma_precision.py -q -rs

uv run --no-sync python -m pytest \
  tests/test_p1_oracle.py tests/test_p1_provider.py \
  tests/test_operator_inputs.py tests/test_kernel_registry.py -q

uv run --no-sync python scripts/check_operator.py \
  --op rmsnorm_residual --candidate cuda --device cuda \
  --dtype bf16 --batch 1 --seq 16 --normalized-dim 4096 --check-grad

uv run --no-sync python scripts/check_operator.py \
  --op rmsnorm_residual --candidate triton --device cuda \
  --dtype bf16 --batch 1 --seq 16 --normalized-dim 4096 --check-grad

uv run --no-sync python scripts/check_p1.py \
  --provider rl_engine.mhc.cuda_provider:CudaMHCProvider --device cuda

uv run --no-sync python scripts/check_p1.py \
  --provider rl_engine.mhc.triton_provider:TritonMHCProvider --device cuda
```

Coverage includes `T=1,7,16`, `D=128,4096`, edge values, repeated execution,
strided inputs, single-row versus batched execution, padding, both autograd
branches, invalid inputs, missing symbols, and FP32 gain values that BF16 cannot
represent. Batch-invariance checks compare row-local outputs and `dx`;
`dgamma` is checked against the oracle for each batch, not for equality across
different token sets.

H100 PCIe validation with PyTorch 2.8.0+cu128 and Triton 3.4.0 recorded 28 backend
tests plus 4 gain-precision tests passing, and 77 adjacent regression tests
passing. Both shared operator checks and both P1 providers passed. Skipped GPU
tests do not establish backend correctness. The shared gtest uses tolerances;
the separate byte-level tests enforce strict alignment.

## Performance Notes

```bash
uv run --no-sync python benchmarks/benchmark_rmsnorm_residual.py \
  --tokens 32768 131072 262144 --warmup 50 --iterations 1000 \
  --output benchmark-rmsnorm-residual.json
```

The script compares torch-native, Triton, and CUDA on identical inputs, reporting
forward, backward, and combined latency plus peak extra allocated memory.
Candidate byte-level validation and warmup occur before timing. Measurements use
CUDA-event means around eager calls without CUDA Graphs; memory is the allocator
peak above live inputs/saved state, not total GPU memory or reserved memory.

The torch-native baseline uses vectorized reductions. It is a performance
comparator, not the fixed-order oracle; its numerical differences are reported
separately. The PyTorch registry fallback instead uses the strict oracle.

For the reported H100 PCIe FP32-gamma run at the three shapes above, CUDA combined
forward/backward speedup over torch-native was 1.29×, 1.31×, and 1.27× respectively,
with approximately 71.4% lower combined peak extra allocated memory. Both CUDA and
Triton passed pre-timing byte-level checks. These results are workload-specific,
not a performance guarantee for other shapes or devices.

## Known Limitations

- CUDA/Triton support only the shapes, dtypes, and epsilon specified above.
- Fixed serial reduction order prioritizes oracle alignment; do not substitute
  a faster reduction tree without an approved numerical-contract change.
- Providers support rank-local replicated placement. TP/SP semantics and
  cross-rank `dgamma` belong to P1-8.
- P1 acceptance supplies sublayer outputs and upstream gradients as data. It
  is not a complete Attention/FFN training, prefill, or decode validation.
- No independent Transformer Engine bitwise equivalence is claimed.
