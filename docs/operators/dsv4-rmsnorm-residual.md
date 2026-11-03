# DSV4 RMSNorm with Residual Fork (ROCm, task 5)

## Summary

`rmsnorm_residual` normalizes one BF16 row of the DSV4 decoder block and returns
the *unnormalized* row for the residual branch, byte for byte and in separate
storage. It does not add a residual before normalizing. The operator is task 5 of
the DSV4 ROCm plan ([RFC #509](https://github.com/RL-Align/RL-Kernel/issues/509),
work item 5) and sits between the mHC pre-aggregation (task 3) and the sublayer.

```text
r[t]           = rsqrt(sum_d FP32(x[t,d])^2 / D + eps)
y[t,d]         = BF16((FP32(x[t,d]) * r[t]) * gamma[d])
residual[t,d]  = x[t,d]
```

Training and rollout must produce the same bytes for the same row, so every
reduction order and rounding point is frozen (see **Accuracy**). The controller RMS
of `fp32_gemm_rms` (task 1) uses `1 / (sqrt(mean) + eps)`; the two forms are never
interchangeable.

## Entry Point

```python
import torch
from rl_engine.ops.norm.rmsnorm_residual import rmsnorm_residual

x = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
gamma = torch.ones(4096, device="cuda", dtype=torch.float32, requires_grad=True)

y, residual = rmsnorm_residual(x, gamma, eps=1e-6, backend="rocm")
torch.autograd.backward((y, residual), (torch.ones_like(y), torch.ones_like(residual)))
assert x.grad.dtype == torch.bfloat16 and gamma.grad.dtype == torch.float32
```

`backend` is mandatory: `"reference"` runs the serial FP32 reference on any device,
`"rocm"` runs the MI300X/MI355X Triton candidate. Explicit forward/backward calls
without autograd use the backend classes directly:

```python
from rl_engine.backends.rocm.norm.rmsnorm_residual import RocmRMSNormResidual

y, residual, saved = RocmRMSNormResidual.forward(x.detach(), gamma.detach(), 1e-6)
dx, dgamma = RocmRMSNormResidual.backward(dy, d_residual, x.detach(), gamma.detach(), saved)
```

`rmsnorm_residual(..., recompute=True)` saves only the inputs and recomputes the
inverse RMS in backward with the same kernel; the bytes equal the saved path.

## Backends

| Backend | Wrapper | Implementation | Status |
| --- | --- | --- | --- |
| ROCm (gfx942, gfx950) | `RocmRMSNormResidual` | `rl_engine/backends/rocm/norm/rmsnorm_residual_kernels.py` (Triton) | Written against the contract; MI300X/MI355X byte acceptance pending |
| Reference | `RMSNormResidualReference` | `rl_engine/reference/norm/rmsnorm_residual.py` | CPU tests pass |
| AITER | — | — | Not used; see **AITER integration boundary** |

## Tensor Contract

| Argument / result | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[T, D]` | BF16 | Any strides; `0 <= T`, `1 <= D <= 16384` |
| `gamma` | `[D]` | FP32 | Kept in FP32, never rounded to BF16 |
| `eps` | scalar | Python float | Exactly `1e-6` |
| `y` | `[T, D]` | BF16 | Contiguous; the operator's only downcast |
| `residual` | `[T, D]` | BF16 | Contiguous byte copy of `x`; never aliases `x` |
| `saved.r` | `[T]` | FP32 | Inverse RMS per row |
| `dy`, `d_residual` | `[T, D]` | BF16 or FP32 | Upstream gradients of the two outputs |
| `dx` | `[T, D]` | FP32 | `dx_norm + d_residual` |
| `dgamma` | `[D]` | FP32 | Ascending token fold |

Autograd returns `x.grad` in BF16 and `gamma.grad` in FP32. The ROCm candidate
reads strided `[T, D]` inputs in place and writes contiguous outputs. `T = 0` runs
and returns empty outputs and an all-zero `dgamma`.

## Dispatch Behavior

There is no registry entry and no fallback. `select_backend("rocm")` on a CPU-only
or non-ROCm build raises `RuntimeError`; a contract violation raises
`RMSNormResidualContractError` before any kernel launch. A backend that is
unavailable is reported, not replaced.

## Accuracy

The reference is `rl_engine/reference/norm/rmsnorm_residual.py`. Candidate acceptance
is raw-byte equality for `y`, `residual`, `saved.r`, `dx` and `dgamma` against the
reference executed on the same device.

- Sum-of-squares and the backward inner product `q = sum_d (dy*gamma)*x` are serial
  ascending FP32 left folds over `D` with one accumulator; `dgamma` is a serial
  ascending FP32 fold over tokens. No Split-K, Stream-K, atomics, `tl.sum` or any
  tree that depends on `T`, `D` or the device.
- Multiplies and adds round separately (`enable_fp_fusion=False`); scaling stays
  `(x * r) * gamma`; divisions use `tl.div_rn`.
- `rsqrt(mean + eps)`, never `1 / sqrt(mean + eps)`.
- `gamma` and all intermediates stay FP32; `y` is rounded to BF16 once at the store;
  `residual` is copied without conversion.
- Rows are independent: the same row gives the same bytes at any position in any
  batch, with any padding or input strides.

`rsqrt` is the device's rsqrt in both the kernel and the reference (`torch.rsqrt`).
Byte equality against a CPU reference is therefore not claimed; the CPU reference is
checked against FP64 math instead.

## AITER integration boundary

At the pinned AITER revision
[326cffa](https://github.com/ROCm/aiter/tree/326cffa38c9eda16bef19375ba55334e4d3acd59),
`aiter/ops/rmsnorm.py` provides `rms_norm`, the in-place `fused_add_rms_norm_opus`
and `rmsnorm2d_fwd_with_add`, and `aiter/ops/triton/normalization/rmsnorm.py`
provides a Triton RMSNorm with `dx`/`dgamma` backward. None of these keeps the
unnormalized row for the residual branch, returns the per-row FP32 inverse RMS to the
caller, or pins the sum-of-squares and `dgamma` reduction order; the in-place variants
overwrite the input and residual. This PR therefore carries an explicitly selected RLK
Triton implementation and does not modify AITER. Routing these entry points through
the AITER kernels is possible once AITER exposes a fixed-order variant with the saved
`r`; it is not a drop-in today.

## Performance Notes

Not measured. The serial folds trade throughput for a pinned reduction order; the
kernel launches one program per row for the forward and `dx`, and one program per
128 columns for `dgamma`. Performance work is out of scope for task 5 acceptance.

## Tests

```bash
# CPU: contract, reference math vs FP64, batch/padding/stride invariance, autograd.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/ops/norm/test_rmsnorm_residual.py

# MI300X / MI355X: byte equality against the on-device reference.
ROCR_VISIBLE_DEVICES=0 RLK_REQUIRE_ROCM=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/backends/rocm/test_dsv4_rmsnorm_residual.py -ra
```

The GPU suite covers `T = 0, 1, 7, 16, 17, 64, 65`, `D = 1, 7, 128, 4096`, BF16 and
FP32 upstream gradients, edge magnitudes, same-row bytes across batch position,
padding, strides and repeats, FP32 gain values BF16 cannot represent, both autograd
branches with saved and recomputed state, and contract rejections. Skipped GPU tests
do not establish correctness.

## Known Limitations

- `eps` is frozen at `1e-6` and `D <= 16384`.
- Only gfx942 and gfx950 are accepted by the ROCm candidate.
- Double backward is not supported.
- TP/SP semantics and the cross-rank `dgamma` reduction belong to the WS2 item
  (KLR #21); this operator is rank-local.
- Graph capture and performance are not validated.
