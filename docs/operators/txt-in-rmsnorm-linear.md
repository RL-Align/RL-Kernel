# txt_in RMSNorm → Linear (`txt_in_rmsnorm_linear`)

Deterministic text-patch input block for the Qwen-Image MMDiT:
`y = single_cast(Linear(RMSNorm(x)) + b)` with `RMSNorm(3584)` (weight, no
bias, no mean subtraction) followed by `Linear(3584 -> 3072)` with bias.

WS1 #386 kernel-table row (forward + backward). The numeric contract
version is `txt-in-rmsnorm-linear-v1`, frozen as follows:

- **RMSNorm→Linear seam is FP32**: no intermediate cast anywhere inside the
  block; a single RNE cast happens at the output boundary (issue rule: cast
  FP32→BF16 only at the declared kernel output). Every gradient is likewise
  computed in fp32 and cast once at return.
- **H/N-dim reduction trees**: the reduction length splits into 32-wide
  leaves (short tail allowed); each leaf is an ascending-k fp32 chain from
  `+0.0`; leaves merge through a mid-split tree `T(l, r) = T(l, m) + T(m, r)`.
  The tree depends only on the reduction length, so row outputs are
  batch-invariant by construction. The forward GEMM reduces over H (112
  leaves), the backward `dz` over N (96 leaves).
- The CUDA kernels use `__fmaf_rn` and Triton uses explicit FP32 FMA.
  The PyTorch reference reuses the single-rounding FMA from #459 for both
  the projection and the normalization reductions.
- **Three-step rstd** (whole-fp64 `1/sqrt` is forbidden): `var = sumsq/3584`
  (true division), `t = var + eps32` (pinned bit pattern `0x358637BD`),
  `sq32 = correctly-rounded sqrt(t)`, `rstd = 1/sq32`. Only the sqrt step may
  relay through fp64 (`sqrt.rn(x)` == `fp32(sqrt_fp64(x))` for fp32 `x` by
  the double-rounding theorem for sqrt); the whole-fp64 form skips the sq32
  rounding and differs in ~29% of samples.
- **True division everywhere**: torch CUDA `tensor / python_float` silently
  multiplies by the single-rounded reciprocal (CPU divides), so the two
  quotients run through `true_div_rn` (0-dim tensor divisor, tensor/tensor
  kernel, correctly rounded on every device).
- **Backward chain** (every step one isolated correctly-rounded op):
  `dz = TreeN(g, W.T)`; `du := dz` (identity seam -- no cast, no STE);
  `dxhat = du*γ`; `dot = TreeH FMA(dxhat, xhat)`; `t1 = dot/3584`;
  `t2 = xhat*t1`; `t3 = dxhat-t2` (contraction into FMS forbidden);
  `dx = rstd*t3`. `dgamma`/`dW` are ascending-row left FMA folds, `db` a
  pure-add fold.
- **RNE everywhere**; no atomics, no split-K, no tensor-core mma, no
  fast-math, no TF32.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

op = kernel_registry.get_op("txt_in_rmsnorm_linear")       # dispatch
out = op(x, norm_weight, weight, bias=bias)                # [*lead, 3072]
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton (in-kernel projection tree, chunked row reductions) | `TritonTxtInRMSNormLinearOp` | — | Supported (default on CUDA) |
| CUDA (per-thread row trees) | `CudaTxtInRMSNormLinearOp` | `_C.txt_in_norm_stats_cuda`, `_C.txt_in_dx_cuda`, `_C.txt_in_dgamma_fold_cuda`, `_C.txt_in_row_tree_reduce_cuda` | Supported |
| PyTorch same-tree reference | `NativeTxtInRMSNormLinearOp` | — | Gold + CPU fallback |

The device GEMM/fold entry points reuse the attn-out kernels
(`attn_out_bias_gemm_cuda_forward`, `attn_out_tree_gemm_cuda`,
`attn_out_dw_left_fold_cuda`). The CUDA extension exposes the three shared operations and four
normalization operations together in strict-math builds. This PR depends on
[#459](https://github.com/RL-Align/RL-Kernel/pull/459).

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[*lead, 3584]` | bf16 / fp32 | all four inputs share one dtype (fail-closed) |
| `norm_weight` | `[3584]` | same as `x` | RMSNorm weight γ |
| `weight` | `[3072, 3584]` | same as `x` | HF `[out, in]` convention |
| `bias` | `[3072]` or omitted | same as `x` | added once, in FP32, after the tree when present |
| return | `[*lead, 3072]` | same as `x` | single RNE cast at the output |

## Dispatch Behavior

- NVIDIA CUDA tries Triton, then the strict native extension, then PyTorch.
- Explicit CPU dispatch selects the PyTorch reference.

## Accuracy

The dedicated tests compare each GPU backend directly with the CPU reference
for forward at 1, 2, 7, 300 and 1024 rows, and for all four gradients at
5 rows, using BF16 and FP32. A separate 9-row forward test compares CUDA
with Triton. The checks compare the logical elements at the declared dtype
by reinterpreting their bits.

An independent scalar FP32 implementation checks one row of forward and
backward. It compares `dx`, `dgamma`, and `db` in full, and checks selected
rows and columns of `dW`. Additional cases cover empty inputs, omitted bias,
and row invariance.

The generic checker runs its PyTorch gold on the selected device. Strict
acceptance requires finite outputs and identical logical-element bits. With
`--check-grad` it differentiates the gold operator using the same upstream
gradient as the candidate and records the backend that computed the gradients.

## Performance Notes

```bash
python benchmarks/benchmark_txt_in_rmsnorm_linear.py --dtype bf16
python benchmarks/benchmark_txt_in_rmsnorm_linear.py --rows 512,1024,4096 --backward
```

The earlier implementation at `3d7b6e93` measured BF16 Triton forward
at 2.9, 5.0 and 28.3 ms on an RTX 5090 for 512, 1024 and 4096 rows.
That version used chunked projection leaves. Rerun the benchmark above
to measure the in-kernel projection tree inherited from #459.

## Tests

```bash
python -m pytest tests/ops/norm/test_txt_in_rmsnorm_linear.py -q
python scripts/check_operator.py --op txt_in_rmsnorm_linear --candidate pytorch --device cpu --dtype fp32
python scripts/check_operator.py --op txt_in_rmsnorm_linear --candidate triton --device cuda --dtype bf16 --check-grad
python scripts/check_operator.py --op txt_in_rmsnorm_linear --candidate cuda --device cuda --dtype bf16 --check-grad
```

The test file also carries the bit-equality harness (four-step comparison:
shape, dtype, contiguous materialisation, dtype-bitcast of the logical
elements) required by the issue acceptance list, plus the pinned primitive
tests (three-step rstd vs `sqrtf`, forbidden whole-fp64 form, FMA-dot not
mul-then-sum, true-division discipline).

## Implementation Files

- `rl_engine/reference/norm/txt_in_rmsnorm_linear.py` — FP32 same-tree reference (gold)
- `rl_engine/backends/shared/triton/norm/txt_in_rmsnorm_linear.py` — Triton projection and row-reduction backend
- `rl_engine/backends/cuda/norm/txt_in_rmsnorm_linear.py` — CUDA backend wrapper
- `csrc/cuda/norm/txt_in_rmsnorm_linear.cu` — CUDA kernels
- `rl_engine/runtime/registry.py`
- `rl_engine/validation/operators/operator_specs.py`
- `tests/ops/norm/test_txt_in_rmsnorm_linear.py`
- `benchmarks/benchmark_txt_in_rmsnorm_linear.py`

## Known Limitations

- Triton combines normalization leaves in chunks of `[chunk, rows]` FP32
  elements. Projection tiles evaluate the whole GEMM tree inside the kernel.
- Tensor-core (mma) paths are excluded on purpose: PTX does not specify the
  internal accumulation tree of `mma.sync`, so the reduction tree cannot be
  frozen. Any future mma path must re-pass the bit-equality harness and
  revise the contract first.
- Shapes are contract-frozen (H=3584, O=3072): the op validates and rejects
  any other geometry (fail-closed), matching the WS1 kernel-table row.
