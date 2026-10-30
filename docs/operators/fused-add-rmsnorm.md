# Fused Add RMSNorm

Residual addition followed by RMS normalization, for the block norms and final
`norm_f` work item in [Nemotron roadmap #434](https://github.com/RL-Align/RL-Kernel/issues/434).
The target model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, whose roadmap
specifies hidden width **D = 2688**, FP32 norm computation/weights, and eps = 1e-5.

This operator exposes an **FP32-output** contract. The roadmap also records
`residual_in_fp32=false`; model integration must align residual/output cast
points with the maintainer before treating this as a drop-in strict model
implementation. Generic registration does not establish full-model parity.

## Definition and tensor contract

Flatten the leading dimensions into M rows. For each row i and feature j:

$$
u_{ij}=x_{ij}+r_{ij},\qquad
q_i=\left(\frac{1}{D}\sum_{k=0}^{D-1}u_{ik}^2+\epsilon\right)^{-1/2},\qquad
y_{ij}=(u_{ij}q_i)w_j.
$$

Forward returns `(y, updated_residual)`, where `updated_residual = u`.

| Direction | Tensor | Shape | Dtype / requirements |
| --- | --- | --- | --- |
| Input | `x` | `[..., D]` | FP16, BF16, or FP32; D > 0 |
| Input | `residual` | same as x | FP16, BF16, or FP32, independently of x |
| Input | `weight` | `[D]` | FP16, BF16, or FP32; model workload uses FP32 |
| Input | `eps` | scalar | finite positive Python value; default 1e-5; no gradient |
| Output | `y`, `updated_residual` | same as x | FP32 |

All tensors share a device. Noncontiguous inputs are accepted and copied to
contiguous storage in the Triton wrapper. Empty leading dimensions (M = 0) are
supported. Residual addition, row statistics, normalization, and weight multiply
use FP32 without an intermediate downcast. Inputs must stay within a range where
FP32 residual addition and squared sums are finite; validation checks metadata
and eps, without a device-wide scan of tensor values.

For backward, let g and h be the incoming gradients of y and the residual output.
Define z = u q and a = g w, elementwise:

$$
c_i=\frac{1}{D}\sum_{j=0}^{D-1}a_{ij}z_{ij},\qquad
\frac{\partial L}{\partial x_{ij}}=
\frac{\partial L}{\partial r_{ij}}=q_i(a_{ij}-z_{ij}c_i)+h_{ij},\qquad
\frac{\partial L}{\partial w_j}=\sum_{i=0}^{M-1}g_{ij}z_{ij}.
$$

| Direction | Tensor | Shape | Dtype / requirements |
| --- | --- | --- | --- |
| Upstream | `grad_y`, `grad_updated_residual_output` | same as x | FP32 from the public outputs; an unused branch contributes zero |
| Saved | `updated_residual`, `inverse_rms`, `weight` | `[..., D]`, `[M]`, `[D]` | FP32 sum/statistic; original weight dtype |
| Returned | `grad_x`, `grad_residual` | same as x | cast once to each corresponding input dtype |
| Returned | `grad_weight` | `[D]` | cast once to the weight dtype |

Backward arithmetic and reduction workspace are FP32. Triton supports first-order
autograd; higher-order derivatives are not supported. The PyTorch reference uses
ordinary autograd. Do not modify the returned residual in place before backward:
it is also a saved value protected by autograd's version check.

## Usage and backends

```python
import torch
from rl_engine.runtime.registry import KernelRegistry

x = torch.randn(2, 16, 2688, device="cuda", dtype=torch.bfloat16, requires_grad=True)
residual = torch.randn_like(x, requires_grad=True)
weight = torch.ones(2688, device=x.device, dtype=torch.float32, requires_grad=True)
op = KernelRegistry().get_op("fused_add_rmsnorm", device=x.device)
y, updated_residual = op(x, residual, weight, eps=1e-5)
(y.square().mean() + updated_residual.square().mean()).backward()
```

| Backend | Implementation | Status |
| --- | --- | --- |
| CUDA | `TritonFusedAddRMSNormOp` | preferred; GPU qualification of the revised ordered reductions pending |
| ROCm | same Triton implementation via `torch.cuda` | preferred; validation pending |
| PyTorch | `NativeFusedAddRMSNormOp` | CPU reference and fallback when Triton cannot load |

CPU/MUSA/NPU registry entries use PyTorch; accelerator validation for MUSA/NPU is
not claimed. A loaded Triton Op does not silently retry through PyTorch on a
validation or launch failure. Each implementation owns its validation and dtype
constants independently.

Direct Triton calls accept `cuda`, `hip`, `xpu`, and `musa`, following existing
operators. Execution requires a compatible Triton backend; XPU/MUSA validation
is pending. Standard ROCm PyTorch uses the `cuda` device type. CUDA/ROCm launches
use the input GPU's device context and current stream.

## Kernel design and automatic selection

Forward uses one program per row, four warps and `next_power_of_2(D)` lanes,
masking padding to zero. D = 2688 uses a 4096-element tile. It saves the FP32
inverse RMS and reuses the existing FP32 residual output for backward. Both
forward and per-row backward use `enable_fp_fusion=False` and fixed row
arithmetic independently of batch size.

Both feature-axis reductions use the shared embedded Triton `sum_tile` helper:
forward reduces `updated_residual * updated_residual`, while input backward
reduces `grad_normalized * normalized`. Each entire padded row is one tile;
the shared whole-vector default of 1024 does not restrict this tile's width.
For D=2688, the 4096 lanes form 512 sequential eight-value groups, followed by
four sections of 128 group totals and the final fixed section tree. Padding
contributes zero, and both means still divide by the actual D=2688.

Changing M, gradient mode, or the weight-gradient strategy does not change D,
the padded width, the group size (8), or the section count (4). These determine
the same feature tree in every path. Changing D or the arithmetic configuration
can change that tree and requires numerical requalification. Fixed order alone
does not promise exact real arithmetic or cross-device/compiler bitwise equality.
The old `tl.sum` tree may have rounded differently, so old output bits and
performance figures are not evidence for this revision. `tl.rsqrt` is unchanged;
its accuracy is checked separately from reduction conformance.

The weight gradient uses a single declared FP32 row fold, starting from positive
zero: `acc = ((0 + contribution[0]) + contribution[1]) + ...`. Only the completed
sum is converted to the weight dtype. The two surviving strategies implement
this exact order, including the same rounding after every addition:

| Strategy | Backward organization |
| --- | --- |
| SEQUENTIAL | Write FP32 [M, D] contributions; load and add one row at a time per column tile. |
| TILED | Write the same contributions; load a row tile, gather its rows one at a time, and add them in ascending order. |

TILED's `block_rows` now controls loads only. It no longer accumulates independent
row lanes or reduces their partial sums. Neither strategy uses floating-point
atomics, locks, reassociation, or partial-result downcasts. Both use the same
per-row backward kernel; dependent launches share the input device's stream.

The previous PARALLEL and FUSED strategies were removed because their partial
sums change addition grouping. A fixed group size makes repeated calls
reproducible, but does not make gradients equal to another reduction strategy.
The former TILED lane accumulator and its tuning results are also superseded.

AUTO conservatively selects SEQUENTIAL (128 columns, four warps) at every shape
and dtype. The new TILED implementation remains explicit until GPU correctness
and fresh performance measurements justify automatic rules. There is no
model-width-specific or large-row fallback to a grouped reduction. Both paths
currently allocate `4 * M * D` bytes for weight contributions: at `[65536, 2688]`
this is 672 MiB. The previous 10.5 MiB FUSED scratch claim no longer applies.

The resolved plan is saved in each autograd context. Changing the reusable Op
cannot change an existing graph. The public constructor accepts only a strategy;
its configurations remain internal:

```python
from rl_engine.backends.shared.triton.norm.fused_add_rmsnorm import (
    RMSNormWeightGradStrategy,
    TritonFusedAddRMSNormOp,
)

op = TritonFusedAddRMSNormOp(weight_grad_strategy=RMSNormWeightGradStrategy.TILED)
```

One `_WEIGHT_GRAD_PLANS` registry groups immutable plans by strategy. Each plan
contains matching conditions, its strategy-specific config and its launcher.
A `key=None` entry is the explicit default. Other entries are automatic rules,
ordered by device, dtype, then width specificity. Device identity uses
`RMSNormDefaultDevice()` or `RMSNormDevice(backend, gpu_name)`, and lookups are
cached by exact metadata. Future tuning may change load tile sizes or warp
counts only when all outputs and gradients remain bitwise identical. The
previous H100 policy and benchmark numbers do not qualify this implementation.

### Logical-token aggregation across microbatches

Changing a single call's strategy preserves all three gradients. Splitting a
logical batch across calls is a separate concern: adding already-reduced
microbatch `weight.grad` values introduces new rounding points, even if every
call uses SEQUENTIAL. Permuting rows also changes a plain floating-point fold.

`TritonFusedAddRMSNormOp.parameter_vjp_contributions_fp32` exposes the existing
project's parameter-contribution protocol for integration. It returns a
`{"weight": rows}` mapping with unreduced FP32 `[M, D]` contributions. It
recomputes statistics with this operator's production forward kernel and uses
the exact contribution expression from its backward. `grad_output` is the
upstream of `y`; the residual output has no weight dependency, and an unused
`y` can be represented by `grad_output=None`.

The caller must retain each row's logical token identity, collect all chunks,
restore their common logical order, and call the project's `reduce_rows_fp32`
with FP32 rows. Convert to `weight.dtype` only after the entire logical batch.
The upstreams and loss-normalization denominator must be identical across
layouts. This route is tested against the actual whole-batch `grad_weight`,
not merely against another recomputed reference. The default PyTorch gradient
accumulation loop does not automatically use this protocol; model integration
must explicitly consume these contributions. This PR does not claim ordinary
microbatch `.backward()` followed by summing `.grad` is bitwise invariant.

## Correctness and training/inference consistency

Tests cover both retained strategies and automatic selection. Coverage includes:

- Independent FP64 output/gradient references; random upstream gradients for
  both outputs; unused output branches; mixed input dtypes and partial gradient
  requirements; noncontiguous tensors and empty rows.
- Byte equality of grad-enabled, `no_grad` and `inference_mode` outputs. Row
  permutation/subset tests compare corresponding outputs and input gradients;
  all three gradients, including grad_weight, are compared across strategies.
- Repeated backward, saved-state integrity, per-call plan retention, overwritten
  scratch buffers, load-tile tails, and CUDA Graph replay. Public autograd captures
  use fresh leaves to isolate stream metadata from existing eager graphs.
- Device restoration and noncurrent-GPU launches when two GPUs are available.
  CPU tests verify policy precedence, dtype/device keys, cache behavior, width
  fallbacks and extrapolation beyond measured rows.

Same-input cross-strategy comparisons include every output and gradient, with
no tolerance. Direct reduction tests also compare different load-tile/warp
configurations to an independent ordered FP32 fold. Cancellation inputs that
failed the old row-lane summation are retained. Chunk/permutation tests preserve
the same logical row set and upstreams, export FP32 contributions, and compare
the canonical aggregate bitwise to whole-batch backward. Independent FP64 and
native PyTorch accuracy tests remain separate and keep their existing limits.
GPU execution of the revised feature trees, TILED kernel, and new tests remains pending.

The reference module also provides `ordered_forward`, `ordered_backward`, and
named intermediate results for fixed-path diagnostics. `ordered_row_sum` pads
to the same power of two and calls the PyTorch translation of `sum_tile`.
Backward is hand-written: it uses the feature tree and a physical-row FP32
left fold for weight gradients, without delegating reductions to autograd.
Tensor division by a device FP32 scalar mirrors `tl.div_rn`; `torch.rsqrt`
remains a separate elementary-math implementation.

Tests separate the two purposes:

- Conformance: expose the production row helpers, compare square/dot sum bits
  with the ordered reference, check launch layouts, and compare all backward
  outputs with the explicit VJP using the *same saved statistics*. Cross-strategy,
  repeated-call, and training/inference tests use exact byte comparisons.
- Accuracy: retain `NativeFusedAddRMSNormOp` with ordinary PyTorch reductions
  and autograd, plus independent FP64 equations/autograd. Neither oracle calls
  the ordered helpers. The ordered reference is not the benchmark baseline.

The shared reduction package and its tests are reused unchanged from the
softcapped-logprob work; this branch carries them as a shared dependency rather
than an RMSNorm-specific copy of the algorithm.

```bash
uv run --no-sync python -m pytest \
  tests/ops/norm/test_fused_add_rmsnorm*.py \
  tests/backends/shared/triton/norm/test_fused_add_rmsnorm*.py \
  tests/reference/norm/test_fused_add_rmsnorm_ordered_reference.py \
  tests/reference/reduction tests/backends/shared/triton/reduction \
  tests/runtime/test_fused_add_rmsnorm_registry.py \
  tests/validation/operators/test_fused_add_rmsnorm.py \
  tests/benchmarks/test_fused_add_rmsnorm_benchmark.py -q -rs -x
```

The shared operator harness also covers both outputs and all three input
gradients through `fused_add_rmsnorm` registration.

## Benchmark

The standalone entry point compares **eager PyTorch with the public automatic
Triton Op** for forward, backward and forward+backward. Default inputs are
M = 1, 32, 8192, 16384, 65536 at D = 2688 in FP16/BF16/FP32, with FP32 weights
and both random upstream gradients: 15 cases / 45 comparison records. Each
record contains both providers' measurements and the selected strategy/config.

```bash
uv run --no-sync python benchmarks/operators/norm/benchmark_fused_add_rmsnorm.py
```

For a small check or custom shape selection:

```bash
uv run --no-sync python benchmarks/operators/norm/benchmark_fused_add_rmsnorm.py \
  --dtypes bf16 --shapes 3x129 --warmup 1 --repeat 2 \
  --output-dir /tmp/fused-rmsnorm-smoke
```

`--modes` selects any of `forward`, `backward`, `forward_backward`; `--dry-run`
prints the case plan without a GPU.

The shared `PerformanceProfiler` event timer measures 10 warmups / 50 repetitions
by default. It records median latency, sample standard deviation and extra peak
allocation. Allocation and public dispatch are included; input preparation,
correctness checks and JIT compilation are excluded. The selector cache is warm.
Unused allocator cache is released before timing, without explicit GPU
data-cache eviction. Small cases can include substantial host dispatch gaps.
There is no `torch.compile` or CUDA Graph timing.

Forward uses `no_grad`; backward reuses a prebuilt graph; combined calls build a
fresh graph each time. Extra peak allocation excludes inputs and prebuilt graphs.
Speedup is PyTorch latency divided by Triton latency. Every case must pass output
and gradient accuracy, forward byte equality, and cross-strategy byte equality
of all five results before timing. Weight
gradient tolerance is `rtol=2e-5, atol=2e-5*sqrt(M)` for FP32 weights; JSON records
actual errors and tolerances for all five values.

Reports default to `artifacts/benchmarks/fused-add-rmsnorm/{report.md,results.json}` and are
checkpointed after each comparison. `complete` becomes true only after the
entire plan finishes. Attach results to the PR instead of committing them.

Prior H100 reports describe the superseded grouped reductions and cannot be
used as performance evidence for this ordered implementation. Rerun GPU
correctness first, then the public benchmark. Its JSON records the validated
strategy names in `strategy_bitwise`; timings still compare only public AUTO
against eager PyTorch. Single-GPU runs skip the noncurrent-device tests. ROCm
validation remains pending.
