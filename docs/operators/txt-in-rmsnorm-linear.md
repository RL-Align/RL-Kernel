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
- **Multiply-add discipline**: one correctly-rounded FP32 FMA on BOTH sides
  of every multiply-into-add site -- device backends use explicit intrinsics
  (`libdevice.fma_rn` / `__fmaf_rn`) and the FP32 reference uses
  `torch.addcmul` (a correctly-rounded single-rounding FP32 FMA, verified
  against the `libm fmaf` oracle). One discipline, one set of bits.
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
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("txt_in_rmsnorm_linear")       # dispatch
out = op(x, norm_weight, weight, bias=bias)                # [*lead, 3072]
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton (leaf chunks + shared host tree combine) | `TritonTxtInRMSNormLinearOp` | — | Supported (default on CUDA) |
| CUDA (per-thread row trees) | `CudaTxtInRMSNormLinearOp` | `_C.txt_in_norm_stats_cuda`, `_C.txt_in_dx_cuda`, `_C.txt_in_dgamma_fold_cuda`, `_C.txt_in_row_tree_reduce_cuda` | Supported |
| PyTorch same-tree reference | `NativeTxtInRMSNormLinearOp` | — | Gold + CPU fallback |

The device GEMM/fold entry points reuse the attn-out kernels
(`attn_out_bias_gemm_cuda_forward`, `attn_out_tree_gemm_cuda`,
`attn_out_dw_left_fold_cuda`): the tree is the same object. ROCm dispatches
Triton then PyTorch (the CUDA source is not ROCm-validated).

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` | `[*lead, 3584]` | bf16 / fp32 | all four inputs share one dtype (fail-closed) |
| `norm_weight` | `[3584]` | same as `x` | RMSNorm weight γ |
| `weight` | `[3072, 3584]` | same as `x` | HF `[out, in]` convention |
| `bias` | `[3072]` | same as `x` | added once, in fp32, after the tree |
| return | `[*lead, 3072]` | same as `x` | single RNE cast at the output |

## Dispatch Behavior

- CUDA: Triton -> CUDA -> PyTorch (Triton leaf chunks are the product path;
  the CUDA per-thread kernel is the memory-light correctness anchor)
- ROCm: Triton -> PyTorch
- CPU: PyTorch

## Accuracy

The acceptance structure is star-shaped with a single gold: the FP32 CPU
same-tree reference (explicit elementwise ops, no matmul, no implicit
reductions, bit-identical on every device).

- **bf16 and fp32 inputs**: every backend matches the reference byte for
  byte on `y`, `dx`, `dgamma`, `dW`, `db` (single FMA discipline on both
  sides; verified across the acceptance tiers S ∈ {1, 2, 5, 7, 300, 1024}).
- **batch invariance**: bitwise (`torch.equal` plus a dtype-bitcast
  assertion on the logical elements) across batch composition, leading
  shapes, padding, and repeat runs, for outputs and `dx` alike; parameter
  gradients are deterministic (fixed row order + fixed upstream gradient).
- An independent scalar spec (own `libm fmaf` chains, own indexing) pins the
  forward and the full backward chain element-by-element.
- **Acceptance is bitwise and only bitwise**: forward and backward, both
  dtypes, logical-element bit patterns (the harness in
  `tests/test_txt_in_rmsnorm_linear.py`: shape, dtype, dtype-bitcast; zero
  tolerance). A result that differs in any bit fails acceptance; error
  statistics are diagnostics only. The generic gtest matrix
  (`scripts/check_operator.py`, `tolerance_contract.json`) is a diagnostic
  overlay for this operator -- its bf16 forward leg compares the fp32 gold
  against the bf16 output and its gradient leg compares
  autograd-through-gold against the frozen backward, both tolerance-based
  by construction -- and cannot weaken the bitwise bar.

## Performance Notes

```bash
python benchmarks/benchmark_txt_in_rmsnorm_linear.py --dtype bf16
python benchmarks/benchmark_txt_in_rmsnorm_linear.py --rows 512,1024,4096 --backward
```

Measured on RTX 5090 (sm_120), bf16, torch 2.13.0+cu130: Triton leaf-chunk
backend 3.4 ms at S=512 / 5.1 ms at S=1024 / 21 ms at S=4096 forward
(26.4x / 28.2x / 21.5x vs the reference); forward+backward 15.4 ms / 21.6 ms
/ 86 ms (13.8x / 15.9x / 13.3x). The CUDA per-thread row-tree kernel trades
speed (~2-5x slower than Triton) for O(S*H) memory (no partial planes).

## Tests

```bash
python -m pytest tests/test_txt_in_rmsnorm_linear.py -q
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

- `rl_engine/kernels/ops/pytorch/norm/txt_in_rmsnorm_linear.py` — FP32 same-tree reference (gold)
- `rl_engine/kernels/ops/triton/norm/txt_in_rmsnorm_linear.py` — Triton leaf-chunk backend
- `rl_engine/kernels/ops/cuda/norm/txt_in_rmsnorm_linear.py` — CUDA backend wrapper
- `csrc/cuda/norm/txt_in_rmsnorm_linear.cu` — CUDA kernels
- `rl_engine/kernels/registry.py`
- `rl_engine/kernels/gtest/operator_specs.py`
- `tests/test_txt_in_rmsnorm_linear.py`
- `benchmarks/benchmark_txt_in_rmsnorm_linear.py`

## Known Limitations

- The Triton forward materialises leaf partial planes per chunk; the chunked
  depth-first combination bounds them (row reductions are `[chunk, S]`
  vectors; GEMM partials as in the attn-out page).
- Tensor-core (mma) paths are excluded on purpose: PTX does not specify the
  internal accumulation tree of `mma.sync`, so the reduction tree cannot be
  frozen. Any future mma path must re-pass the bit-equality harness and
  revise the contract first.
- Shapes are contract-frozen (H=3584, O=3072): the op validates and rejects
  any other geometry (fail-closed), matching the WS1 kernel-table row.
