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
from rl_engine.kernels.registry import KernelRegistry

x = torch.randn(2, 16, 2688, device="cuda", dtype=torch.bfloat16, requires_grad=True)
residual = torch.randn_like(x, requires_grad=True)
weight = torch.ones(2688, device=x.device, dtype=torch.float32, requires_grad=True)
op = KernelRegistry().get_op("fused_add_rmsnorm", device=x.device)
y, updated_residual = op(x, residual, weight, eps=1e-5)
(y.square().mean() + updated_residual.square().mean()).backward()
```

| Backend | Implementation | Status |
| --- | --- | --- |
| CUDA | `TritonFusedAddRMSNormOp` | preferred; automatic dispatch and operator regression validated on H100 |
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

The weight gradient sums contributions across rows. Four strategies are available:

| Strategy | Backward organization |
| --- | --- |
| SEQUENTIAL | Compute all per-row contributions, then fold rows in ascending order per feature tile. |
| TILED | Compute all per-row contributions, accumulate several row lanes per feature tile, then reduce the lanes. |
| PARALLEL | Compute all per-row contributions, reduce independent row blocks into partials, then merge partials. |
| FUSED | Compute input gradients while accumulating fixed row-group weight partials, then merge partials. Avoid the full per-row contribution buffer. |

FUSED follows the grouped-accumulation idea in
[Mamba's backward](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layer_norm.py),
but uses fixed row counts rather than the GPU's SM count and retains this
operator's FP32 outputs, saved statistics, and cast points. Its input-gradient
kernel keeps four warps and the original per-row expressions. Only its merge
uses the configured `block_cols` and `num_warps`.

No strategy uses floating-point atomics or locks. Dependent kernels launch on
the same stream without an added host synchronization. Fixed grouping makes
reduction order independent of runtime scheduling.

The default policy is based on H100 80 GB measurements:

| Width D | Flattened rows M | Strategy | Rows / columns / warps |
| --- | --- | --- | --- |
| Any | 0–8 | SEQUENTIAL | 1 / 128 / 4 |
| Any | 9–32 | TILED | 32 / 64 / 4 |
| 2688 | 33–16383 | TILED | 64 / 64 / 8 |
| 2688 | >=16384 | FUSED | 64 / 64 / 4 |
| Other widths | >=33 | TILED | 64 / 64 / 8 |

These rules are shared defaults for devices without their own entries, not
proven optima for every GPU. D is matched exactly; other widths do not inherit
the model's FUSED cutoff. Rows above 65536 explicitly extrapolate the large-row
rule. Dtypes currently share configurations, and eager/Graph execution uses the
same policy. The metadata lookup has a bounded 1024-entry cache.

The resolved plan is saved in each autograd context. Changing the Op's settings
after forward cannot change that call's backward. Explicit strategies and
configurations remain available for experiments:

```python
from rl_engine.kernels.ops.triton.norm import (
    RMSNormWeightGradConfig,
    RMSNormWeightGradStrategy,
    TritonFusedAddRMSNormOp,
)

op = TritonFusedAddRMSNormOp(
    weight_grad_strategy=RMSNormWeightGradStrategy.FUSED,
    weight_grad_config=RMSNormWeightGradConfig(block_rows=64, block_cols=64, num_warps=4),
)
```

### Policy rationale

H100 boundary and confirmation runs support FUSED 64 for large model-width
inputs. Larger groups offered only marginal additional gains, while smaller-row
and other-width results did not justify broader rules. Explicit configurations
remain available for other hardware; detailed strategy comparisons belong in
PR attachments rather than the routine benchmark.

At `[65536, 2688]`, FUSED 64 replaces the 672 MiB FP32 per-row contribution buffer
with 10.5 MiB of grouped partials. These are calculated scratch sizes, not total
peak allocation.

## Correctness and training/inference consistency

Tests retain all four strategies and automatic selection. Coverage includes:

- Independent FP64 output/gradient references; random upstream gradients for
  both outputs; unused output branches; mixed input dtypes and partial gradient
  requirements; noncontiguous tensors and empty rows.
- Byte equality of grad-enabled, `no_grad` and `inference_mode` outputs. Row
  permutation/subset tests compare corresponding outputs and input gradients;
  automatic-policy boundary tests also cross grouping strategies.
- Repeated backward, saved-state integrity, per-call plan retention, overwritten
  scratch buffers, group tails, and CUDA Graph replay. Public autograd captures
  use fresh leaves to isolate stream metadata from existing eager graphs.
- Device restoration and noncurrent-GPU launches when two GPUs are available.
  CPU tests verify policy precedence, dtype/device keys, cache behavior, width
  fallbacks and extrapolation beyond measured rows.

Repeated calls with an identical configuration must reproduce gradients bitwise.
Different weight-reduction groupings, or separately reduced microbatches, can
change FP32 addition order and are checked numerically rather than promised to
have identical weight-gradient bytes. FP32 compute alone does not guarantee
full-model training/inference parity.

```bash
uv run --no-sync python -m pytest tests/nemotron/test_fused_add_rmsnorm*.py -q -rs -x
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
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py
```

For a small check or custom shape selection:

```bash
uv run --no-sync python benchmarks/benchmark_fused_add_rmsnorm.py \
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
and gradient accuracy plus forward byte-equality checks before timing. Weight
gradient tolerance is `rtol=2e-5, atol=2e-5*sqrt(M)` for FP32 weights; JSON records
actual errors and tolerances for all five values.

Reports default to `reports/fused-add-rmsnorm/{report.md,results.json}` and are
checkpointed after each comparison. `complete` becomes true only after the
entire plan finishes. Attach results to the PR instead of committing them.

The H100 regression passed forward/backward accuracy, training/inference byte
equality, automatic-selection boundaries, and Graph replay. The final public
benchmark completed 30 input cases / 90 comparisons, including the FUSED cutoff
and other-width fallbacks. Single-GPU runs skip the noncurrent-device tests.
Performance reports retain timing variation alongside medians; small advantages
should not be treated as stable wins. ROCm validation, model cast-point alignment,
and full-model / distributed validation remain separate work.
