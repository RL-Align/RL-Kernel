# Final Logit Softcap

## Summary

Final logit softcapping bounds logits smoothly before a vocabulary softmax. This
operator implements the fixed cap of 30 requested for Gemma in
[WS1 issue #415](https://github.com/RL-Align/RL-Kernel/issues/415).
It acts independently on each element; it does not perform softmax or a reduction.

For input $x_i$, output $y_i$, and upstream gradient $g_{y_i}$:

$$
y_i = 30\tanh\left(\frac{x_i}{30}\right)
$$

$$
\frac{\partial y_i}{\partial x_i}
= 1-\tanh^2\left(\frac{x_i}{30}\right),
\qquad
g_{x_i} = g_{y_i}\left[1-\tanh^2\left(\frac{x_i}{30}\right)\right].
$$

## Entry Point

```python
import torch
from rl_engine.kernels.registry import kernel_registry

x = torch.randn(2, 3, 262144, device="cuda", dtype=torch.bfloat16, requires_grad=True)
op = kernel_registry.get_op("final_logit_softcap", device=x.device)
y = op(x)  # Same shape and device as x; FP32 output.
grad_y = torch.randn_like(y)
(grad_x,) = torch.autograd.grad(y, x, grad_outputs=grad_y)
assert grad_x.dtype == x.dtype
```

The public wrappers are `NativeFinalLogitSoftcapOp` in
`rl_engine.kernels.ops.pytorch.activation` and `TritonFinalLogitSoftcapOp` in
`rl_engine.kernels.ops.triton.activation`. There is no separate `forward_fp32`
method: `forward` already returns FP32.

## Backends and Dispatch

| Platform | Registry priority | Validation status |
| --- | --- | --- |
| NVIDIA CUDA | Triton → PyTorch native | CUDA correctness validated; evidence linked below |
| ROCm | Triton → PyTorch native | Hardware execution pending |
| CPU | PyTorch native | Covered by the reference and integration tests |
| MUSA / NPU | PyTorch native | Registry selection tested; hardware execution not validated |

The Triton implementation has forward and backward kernels; there is no separate
CUDA C++ implementation. The registry falls back when a backend cannot be loaded
or instantiated. It does not retry a failed kernel launch using another backend.
The native implementation computes on the input's device.

## Tensor and Precision Contract

| Value | Shape | Dtype | Behavior |
| --- | --- | --- | --- |
| Input `x` | Any shape, including scalar and empty | FP16 / BF16 / FP32 | Finite floating-point logits |
| Intermediate values | Elementwise | FP32 | Input converted before division and tanh |
| Output `y` | Same as `x` | FP32 | Same device as `x` |
| Upstream gradient `grad_y` | Same as `y` | FP32 | Supplied by autograd |
| Input gradient `grad_x` | Same as `x` | Input dtype | Computed in FP32, then cast once |

The native reference explicitly converts with `x.to(torch.float32)` and uses
`30.0 * torch.tanh(x_f / 30.0)`. The Triton kernels use `tl.div_rn` and
`libdevice.tanh`. FP32 arithmetic specifies the working dtype; it does not promise
identical transcendental-function rounding across hardware backends.

The wrappers accept noncontiguous inputs and upstream gradients. Triton makes
contiguous copies where necessary, preserving logical shape and values. This may
add allocation and copy costs. Inputs and upstream gradients are not modified.
Backward saves the contiguous input in its original dtype and recomputes tanh.
Empty inputs return empty buffers without launching a kernel.

Both Triton kernels convert the program ID to `int64` before multiplying by
the block size, keeping element-offset arithmetic in 64 bits. This removes the
previous $2^{31}$-element indexing restriction without changing floating-point
arithmetic or output dtypes. Allocation and launch limits still apply.

## Accuracy and Invariance

Tests obtain thresholds from the shared WS1 `tolerance_contract.json` through
`resolve_tolerance`, using the `elementwise` operator class. The current accuracy
thresholds for both forward and gradient comparisons are:

| Compared tensor dtype | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| FP32 | 1e-5 | 1e-5 |
| FP16 | 1e-3 | 1e-3 |
| BF16 | 2e-2 | 1.6e-2 |

Forward results always use the FP32 row, even for BF16 or FP16 inputs. Gradient
comparisons use the input dtype. Native calculations are also checked against an
FP64 reference whose derivative uses $1/\cosh^2(x/30)$ and a random upstream
gradient.

The GPU suite checks bitwise equality for repeated calls, slicing, padding and
reshaping, for both outputs and input gradients. It also compares training-mode
forward output with `torch.no_grad()` output. These are operator-level checks on
one device; they do not establish full-model train/rollout parity or cross-device
bitwise equality.

## Tests

```bash
uv sync --extra dev
source .venv/bin/activate
python -m pytest tests/gemma/test_final_logit_softcap.py -v -rs
python -m pytest tests/gemma/test_final_logit_softcap_benchmark.py -v
python scripts/check_operator.py --op final_logit_softcap --candidate triton \
  --device cuda --dtype bf16 --batch 2 --seq 3 --vocab 262144 --check-grad
```

The focused operator suite covers all three input dtypes, block boundaries,
scalar/empty tensors, multidimensional shapes, saturation, strided/expanded
views, random upstream gradients, registry fallback and the accuracy harness.
An index regression exercises the production forward/backward kernels on blocks
straddling $2^{31}$, rebasing pointers into small buffers and checking the masked
tail. Earlier blocks perform no memory access; the test does not allocate a
multi-billion-element tensor.
CPU-only runs skip GPU cases. On a GPU host, a missing Triton backend fails the
GPU tests instead of silently testing the native fallback.

CUDA validation logs and hardware details are recorded in
[PR #444](https://github.com/RL-Align/RL-Kernel/pull/444#2-validation-and-performance).

## Benchmark

Run on a CUDA (NVIDIA) or ROCm (AMD) host with the development dependencies
installed:

```bash
# Quick execution check before the complete run.
python benchmarks/benchmark_final_logit_softcap.py \
  --shapes 1x1x1025 --dtypes fp32 --warmup 3 --repeat 5 \
  --output-dir ../softcap-results/smoke

# FP16, BF16 and FP32; small tails through 64 vocabulary rows.
python benchmarks/benchmark_final_logit_softcap.py \
  --output-dir ../softcap-results/run-1
```

This standalone operator benchmark reuses the existing `PerformanceProfiler`
accelerator-event timer (CUDA events on NVIDIA, HIP events on ROCm). It directly
constructs the native and Triton wrappers on the same GPU, checks output and
gradient accuracy before timing, and writes `results.json` and `report.md`.
Failure to import or run Triton is an error.

Default shapes are `1x1x1025`, `1x1x262144`, `1x16x262144` and `1x64x262144`.
Inputs are contiguous, with random FP32 upstream gradients. Noncontiguous layouts
are covered by correctness tests, but their copy overhead is not measured by this
benchmark. Use `--shapes`, `--dtypes`, `--warmup`, `--repeat` and `--device` to
select another workload. Run on an otherwise idle GPU.

- **Forward:** inference under `torch.no_grad()`.
- **Backward:** `torch.autograd.grad` on a prebuilt, retained graph; excludes forward.
- **Forward + backward:** a fresh forward graph and gradient computation per call.

The report contains median latency, sample standard deviation, incremental peak
allocated memory, and `native_ms / triton_ms` speedup. Extra peak allocation excludes
existing inputs and any prebuilt backward graph, so it is not total training memory.
Input generation, correctness checks and compilation are outside the timing window.
Public-wrapper allocation and autograd dispatch are included. For small tensors,
host dispatch gaps can dominate accelerator-event measurements. This compares eager
PyTorch with Triton, not `torch.compile` or graph capture.

The report records the backend and its runtime version, the GPU and architecture,
compute capability (CUDA only), software versions, commit, tracked changes, input
shapes/dtypes, warmup/repetitions, seed, block size and tolerance contract
fingerprint.

### Reports and validation evidence

See [PR #444](https://github.com/RL-Align/RL-Kernel/pull/444#2-validation-and-performance)
for measured results, hardware/software fingerprints, timing variability and
regressions. Markdown reports, JSON measurements and validation logs are PR
attachments rather than tracked repository files. Keep generated reports outside
the checkout, as in the commands above.

## Known Limitations

- The cap is fixed at 30; configurable or trainable caps are not implemented.
- The Triton autograd wrapper supports first-order gradients only.
- The current Triton configuration uses a fixed block size of 1024, without autotuning.
- FP64, integer and complex inputs are rejected. NaN/Inf semantics have not been
  separately validated.
- ROCm and other accelerator hardware validation remain pending.
