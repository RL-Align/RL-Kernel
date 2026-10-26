# Final Logit Softcap

## Summary

Final logit softcapping bounds logits smoothly before a vocabulary softmax. The
operator implements `c * tanh(x / c)` with the cap `c` supplied at runtime, so
one shared kernel serves every model that needs a different cap.

| Model | Cap | Note |
| --- | ---: | --- |
| Gemma-4-31B-it | 30 | Default value of the public interface. |
| Muse-Glimmer-30B | 20 | The model applies `output_multiplier` before calling this operator; that multiplier stays on the model side. |

The operator acts independently on each element. It does not perform a softmax,
a reduction, or the `output_multiplier` scaling.

For input $x_i$, output $y_i$, upstream gradient $g_{y_i}$, and cap $c$:

$$
y_i = c\tanh\left(\frac{x_i}{c}\right)
$$

$$
\frac{\partial y_i}{\partial x_i}
= 1-\tanh^2\left(\frac{x_i}{c}\right),
\qquad
g_{x_i} = g_{y_i}\left[1-\tanh^2\left(\frac{x_i}{c}\right)\right].
$$

The cap cancels in the derivative: the outer factor $c$ and the $1/c$ from the
chain rule multiply to one.

## Entry Point

```python
import torch
from rl_engine.runtime.registry import kernel_registry

x = torch.randn(2, 3, 262144, device="cuda", dtype=torch.bfloat16, requires_grad=True)

# Registry selection uses the default cap of 30.
op = kernel_registry.get_op("final_logit_softcap", device=x.device)
y = op(x)

# A model that needs another cap constructs the wrapper directly.
from rl_engine.backends.shared.triton.activation import TritonFinalLogitSoftcapOp

op20 = TritonFinalLogitSoftcapOp(softcap=20.0)
y20 = op20(x)

grad_y = torch.randn_like(y)
(grad_x,) = torch.autograd.grad(y, x, grad_outputs=grad_y)
assert grad_x.dtype == x.dtype
```

The public wrappers are `NativeFinalLogitSoftcapOp` in
`rl_engine.reference.activation` and `TritonFinalLogitSoftcapOp` in
`rl_engine.backends.shared.triton.activation`. The legacy import paths
`rl_engine.kernels.ops.pytorch.activation` and
`rl_engine.kernels.ops.triton.activation` remain available as compatibility
shims.

Both wrappers take `softcap` as the first constructor argument and default it to
`30.0`. The value must be finite and positive; anything else raises
`ValueError` at construction rather than producing `NaN` inside a kernel.

## Backends

| Platform | Registry priority | Validation status |
| --- | --- | --- |
| NVIDIA CUDA | Triton → PyTorch native | CUDA correctness and performance validated on H100; SM89 development on RTX 4090 |
| ROCm | Triton → PyTorch native | Kernel uses `libdevice.tanh`, which maps to HIP; hardware execution pending |
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
| Output `y` (Triton) | Same as `x` | FP32 | Same device as `x` |
| Output `y` (`forward`) | Same as `x` | Input dtype | FP32 math, cast once at the end |
| Output `y` (`forward_fp32`) | Same as `x` | FP32 | Ground-truth path |
| Upstream gradient `grad_y` | Same as `y` | FP32 | Supplied by autograd |
| Input gradient `grad_x` | Same as `x` | Input dtype | Computed in FP32, then cast once |

The native reference converts with `x.to(torch.float32)` and uses
`c * torch.tanh(x_f / c)`. The Triton kernels use `tl.div_rn` and
`libdevice.tanh`. FP32 arithmetic specifies the working dtype; it does not
promise identical transcendental-function rounding across hardware backends.

`softcap` is a runtime kernel argument, not a `tl.constexpr`. A constant would
recompile the kernel once per distinct cap; passing it at launch keeps a single
compiled variant for 20, 30, and any later value. The trade-off is a small loss
of compile-time optimization that does not matter for an elementwise kernel.

The wrappers accept noncontiguous inputs and upstream gradients. Triton makes
contiguous copies where necessary, preserving logical shape and values. This may
add allocation and copy costs. Inputs and upstream gradients are not modified.
Backward saves the contiguous input in its original dtype and recomputes tanh.
Empty inputs return empty buffers without launching a kernel.

Both Triton kernels convert the program ID to `int64` before multiplying by the
block size, keeping element-offset arithmetic in 64 bits. This removes the
previous $2^{31}$-element indexing restriction without changing floating-point
arithmetic or output dtypes. Allocation and launch limits still apply.

## Dispatch Behavior

`kernel_registry.get_op("final_logit_softcap", device=...)` resolves through the
`OpBackend` priority map:

| Platform | Priority |
| --- | --- |
| `cuda` | Triton → PyTorch native |
| `rocm` | Triton → PyTorch native |
| `musa` | PyTorch native |
| `cpu` | PyTorch native |
| `npu` | PyTorch native |

The registry constructs the backend with no arguments, so the public wrappers
must keep a default for `softcap`. A model that needs a different cap constructs
its wrapper directly or passes a configured instance through its own provider.

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
FP64 reference whose derivative uses $1/\cosh^2(x/c)$ and a random upstream
gradient.

Every numerical test runs for `softcap=20` and `softcap=30`. Input scales are
rescaled with the cap, so both caps exercise the linear region, the transition
region, and saturation of `tanh`.

The GPU suite checks bitwise equality for repeated calls, slicing, padding and
reshaping, for both outputs and input gradients. It also compares training-mode
forward output with `torch.no_grad()` output. These are operator-level checks on
one device; they do not establish full-model train/rollout parity or cross-device
bitwise equality.

## Tests

```bash
python -m pytest tests/ops/activation/test_final_logit_softcap.py -v -rs
python tools/validation/operators/check_operator.py --op final_logit_softcap \
  --candidate triton --device cuda --dtype bf16 --batch 2 --seq 3 --vocab 262144 --check-grad
```

The focused operator suite covers both caps, all three input dtypes, block
boundaries, scalar/empty tensors, multidimensional shapes, saturation,
strided/expanded views, random upstream gradients, registry fallback, the
accuracy harness, and a regression that compares `softcap=30` against the
original hard-coded formula bitwise.

CPU-only runs skip GPU cases. On a GPU host, a missing Triton backend fails the
GPU tests instead of silently testing the native fallback.

## Benchmark

Run on a CUDA (NVIDIA) or ROCm (AMD) host with the development dependencies
installed:

```bash
# Quick execution check before the complete run.
python benchmarks/operators/activation/benchmark_final_logit_softcap.py \
  --shapes 1x1x1025 --dtypes fp32 --softcaps 30 --warmup 3 --repeat 5 \
  --output-dir artifacts/benchmarks/softcap-smoke

# FP16, BF16 and FP32 at both caps; small tails through a full vocabulary row.
python benchmarks/operators/activation/benchmark_final_logit_softcap.py \
  --softcaps 20 30 --output-dir artifacts/benchmarks/softcap-run-1
```

This standalone operator benchmark reuses the existing `PerformanceProfiler`
accelerator-event timer (CUDA events on NVIDIA, HIP events on ROCm). It directly
constructs the native and Triton wrappers on the same GPU with the same cap,
checks output and gradient accuracy before timing, and writes `results.json` and
`report.md`. Failure to import or run Triton is an error.

Default shapes are `1x1x1025`, `1x1x262144`, `1x16x262144` and `1x64x262144`.
Default caps are `20` and `30`. Inputs are contiguous, with random FP32 upstream
gradients. Noncontiguous layouts are covered by correctness tests, but their copy
overhead is not measured by this benchmark. Use `--shapes`, `--dtypes`,
`--softcaps`, `--warmup`, `--repeat` and `--device` to select another workload.
Run on an otherwise idle GPU.

- **Forward:** inference under `torch.no_grad()`.
- **Backward:** `torch.autograd.grad` on a prebuilt, retained graph; excludes forward.
- **Forward + backward:** a fresh forward graph and gradient computation per call.

The report contains median latency, sample standard deviation, incremental peak
allocated memory, and `native_ms / triton_ms` speedup. Extra peak allocation excludes
existing inputs and any prebuilt backward graph, so it is not total training memory.
Input generation, correctness checks and compilation are outside the timing window.
Public-wrapper allocation and autograd dispatch are included. For small tensors,
host dispatch gaps can dominate accelerator-event measurements.

The report records the backend and its runtime version, the GPU and architecture,
compute capability (CUDA only), software versions, commit, tracked changes, input
shapes/dtypes/caps, warmup/repetitions, seed, block size and tolerance contract
fingerprint.

### Reports and validation evidence

Write local outputs to `artifacts/benchmarks/`, which is git-ignored. Reviewed
measurements belong in `reports/experiments`; keep generated reports out of the
checkout otherwise.

## Known Limitations

- The cap must be finite and positive; trainable or learned caps are not implemented.
- The Triton autograd wrapper supports first-order gradients only.
- The current Triton configuration uses a fixed block size of 1024, without autotuning.
- FP64, integer and complex inputs are rejected. NaN/Inf input semantics have not been
  separately validated.
- ROCm and MUSA hardware execution remain pending.

## Implementation Files

- `rl_engine/reference/activation/final_logit_softcap.py` — PyTorch reference (`forward`, `forward_fp32`)
- `rl_engine/backends/shared/triton/activation/final_logit_softcap.py` — Triton kernels and autograd
- `rl_engine/kernels/ops/pytorch/activation/final_logit_softcap.py` — compatibility shim
- `rl_engine/kernels/ops/triton/activation/final_logit_softcap.py` — compatibility shim
- `rl_engine/runtime/registry.py` — dispatch registration
- `rl_engine/runtime/provenance/module_names.json` — stable operator paths
- `rl_engine/validation/operators/operator_specs.py` — accuracy harness entry
- `tests/ops/activation/test_final_logit_softcap.py` — correctness and invariance tests
- `benchmarks/operators/activation/benchmark_final_logit_softcap.py` — performance comparison
