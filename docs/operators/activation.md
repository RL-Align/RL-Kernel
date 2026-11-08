# SiLU / SwiGLU Activation

The activation operators are the element-wise core of the Qwen3/Llama gated MLP. They
implement the WS1 dual-path contract (issue #108): pure-PyTorch fp32 ground truth, plus
CUDA, Triton and Ascend C candidates that validate against it.

- **SiLU** (`NativeSiLUOp` / `SiLUCudaOp` / `TritonSiLUOp`): `silu(x) = x * sigmoid(x)` —
  the `hidden_act="silu"` gate.
- **SwiGLU** (`NativeSwiGLUOp` / `SwiGLUCudaOp` / `TritonSwiGLUOp`):
  `swiglu(gate, up) = silu(gate) * up` — the gated MLP middle stage. `gate` / `up` are the
  `gate_proj` / `up_proj` outputs (already at the intermediate width); the following
  `down_proj` is a plain Matmul and is **not** part of this operator.

```text
hidden --gate_proj--> gate --\
                              swiglu --> down_proj --> hidden
hidden --up_proj----> up ----/
```

Packed projections compute both operands with one linear layer. For example,
MiniMax-H3 uses `[up | gate]` in its last dimension. The activation is still
`SiLU(gate) * up`; only the input layout changes:

```text
hidden --packed_proj--> [ up | gate ] --SwiGLU--> output --down_proj--> hidden
                         [..., 2*I]              [..., I]
```

## Entry Point
```python
from rl_engine.runtime.registry import kernel_registry

silu = kernel_registry.get_op("silu")
swiglu = kernel_registry.get_op("swiglu")

# SiLU: single element-wise activation
y = silu(x)                       # [..., N]  ->  [..., N]

# SwiGLU: gated activation (gate and up must share shape)
h = swiglu(gate, up)              # [..., I], [..., I]  ->  [..., I]
```

All backends expose the WS1 dual-path contract:

- `forward(...)` — computes in fp32, casts back to the input dtype (Axis-B accuracy
  candidate / dtype-behavior path).
- `forward_fp32(...)` — computes and returns fp32 (the ground-truth golden path).

### Packed input (PyTorch and Triton)

The common PyTorch and Triton classes also accept one tensor. Its last dimension
must contain two equal halves; an explicit order prevents silently swapping the
operands. Both `forward` and `forward_fp32` support this form:

```python
from rl_engine.contracts.operators.activation import SwiGLUPackedOrder
from rl_engine.backends.shared.triton.activation.swiglu import TritonSwiGLUOp
from rl_engine.reference.activation.swiglu import NativeSwiGLUOp

swiglu = TritonSwiGLUOp()
y = swiglu(gate, up)  # Existing positional calls remain valid.
y = swiglu(input=gate, up=up)
y = swiglu(projected, packed_order=SwiGLUPackedOrder.GATE_UP)
y = swiglu(input=projected, packed_order=SwiGLUPackedOrder.UP_GATE)  # H3 order
y_fp32 = swiglu.forward_fp32(projected, packed_order=SwiGLUPackedOrder.UP_GATE)
reference = NativeSwiGLUOp()(projected, packed_order=SwiGLUPackedOrder.UP_GATE)
```

In these two classes, the first parameter is named `input`: it is the gate when
`up` is supplied, or the complete packed projection when `up` is omitted. Calls
that previously used `gate=...` must now use `input=...`, including `forward_fp32`.

The packed form supports FP16/BF16/FP32 input `[..., 2 * I]` and produces `[..., I]`,
with positive `I`. Leading dimensions may be empty. A contiguous packed input is
read directly; a strided input is made contiguous once. Backward writes the two
gradient halves into one packed tensor in the original order and dtype.

Splitting a packed input with `chunk` only creates views. For multiple rows, those
halves are generally not contiguous: calling the two-input Triton path copies each
half, and autograd reconstructs the packed gradient. The new packed path avoids
these copies and reconstruction. It retains the same FP32 expressions and the
two-stage backward's FP32 SiLU-derivative workspace; no extra low-precision
intermediate rounding is introduced.

This overload is currently implemented by `NativeSwiGLUOp` and `TritonSwiGLUOp`.
The standalone CUDA and Ascend wrappers still use two inputs. Instantiate the
desired class explicitly for packed input: automatic registry selection does not
inspect the argument layout and may select a two-input-only backend.

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeSiLUOp` / `NativeSwiGLUOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA | `SiLUCudaOp` / `SwiGLUCudaOp` | `_C.silu_*` / `_C.swiglu_*` | General CUDA (fp16/bf16/fp32); math in fp32. |
| Triton | `TritonSiLUOp` / `TritonSwiGLUOp` | Triton JIT | Portable GPU baseline; same fp32 math contract. |
| Ascend C | `SiLUAscendOp` / `SwiGLUAscendOp` | `_C_npu.silu_*` / `_C_npu.swiglu_*` | NPU forward and backward; fp16/bf16/fp32 inputs, FP32 math. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` (SiLU) | `[..., N]` | float (fp16/bf16/fp32) | Any shape; last dim arbitrary (Qwen3-8B `I=12288`). |
| `gate` (SwiGLU) | `[..., I]` | float | `gate_proj` output. |
| `up` (SwiGLU) | `[..., I]` | float | `up_proj` output; **must share `gate`'s shape, dtype, and device**. |
| `input` (packed SwiGLU) | `[..., 2 * I]` | float | Positive, even last dimension; explicit `packed_order`. Native and Triton only. |
| output | `[..., N]` for SiLU; `[..., I]` for SwiGLU | `forward`: input dtype · `forward_fp32`: float32 | Packed SwiGLU halves the last dimension. |

Element-wise and shape-agnostic: the Qwen3-8B intermediate dim `I=12288` is just one valid
last-dim size, not a hard requirement. Pure functions — no randomness, no in-place
mutation, device/dtype follow the inputs.

## Dispatch Behavior

`kernel_registry.get_op("silu" | "swiglu")` resolves through the `OpBackend` priority map:

| Platform | Priority |
| --- | --- |
| `cuda` | CUDA → Triton → PyTorch native |
| `rocm` | Triton → PyTorch native |
| `cpu` | PyTorch native |
| `npu` | Ascend C → PyTorch native |

If the CUDA extension is not built (or symbols are missing), the registry falls back to
Triton, then to the native gold.

On NPU, a missing Ascend extension or missing SiLU/SwiGLU symbols causes the registry
to select PyTorch native. Construct `SiLUAscendOp` / `SwiGLUAscendOp` directly when the
Ascend C kernel is required; the constructors raise if a native symbol is missing.

Both NPU kernels share one tile geometry (`TILE_LENGTH = 2048`, `MAX_BLOCKS = 32`) and
the same FP32 `1 / (1 + exp(-x))` sequence, so `silu(x)` is bitwise equal to
`swiglu(x, ones)` on this hardware. Each element is evaluated by a fixed expression
independent of tensor size and of the launched block count, which is what makes the
op batch-invariant for the WS1 `ascend_bf16` profile (`silu` is a required C2 chain
node, so the profile needs its own kernel rather than a SwiGLU with a unit operand).

## Ascend C Build and Validation

On a Linux Ascend host with matching PyTorch, `torch_npu` and CANN installed, source
the CANN environment and build the existing NPU extension:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
KERNEL_ALIGN_FORCE_ASCEND=1 KERNEL_ALIGN_ASCEND_ARCH=dav-2201 \
  python -m pip install --no-build-isolation -e .
python -m pytest tests/ops/activation/test_swiglu.py -v
python tools/validation/operators/check_operator.py --op swiglu --candidate ascend --dtype bf16 --device npu --check-grad
```

The kernel uses the A2/A3 vector programming model (`dav-2201`). Other architectures
require separate build and device validation. `setup.py` automatically includes all
`csrc/ascend/*.asc` files; `bindings.asc` defines the shared `_C_npu` module entry.

```python
import torch
import torch_npu
from rl_engine.runtime.registry import kernel_registry

swiglu = kernel_registry.get_op("swiglu", device="npu")
gate = torch.randn(2, 12288, device="npu", dtype=torch.bfloat16, requires_grad=True)
up = torch.randn_like(gate, requires_grad=True)
out = swiglu(gate, up)
out.float().sum().backward()
```

The wrapper accepts scalars, empty tensors and strided views. It makes inputs and
upstream gradients contiguous before invoking the kernels. Both kernels use fixed
2048-element tiles, bounded UB storage, FP32 intermediates and a single output cast.
FP32-to-FP16/BF16 uses ties-to-even rounding (`CAST_RINT`), as specified by the
[Ascend C Cast API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/ascendcopapi/atlasascendc_api_07_0073.html).
`forward_fp32` returns FP32 while retaining gradients to the original inputs.
Only first-order autograd is supported by this backend.

The tests cover dtype accuracy, both input gradients, tails, multiple tiles,
noncontiguous views, input validation, batch-position invariance, stream ordering
and multiple devices. CPU runs check Python integration and skip hardware tests;
on an NPU host a missing extension fails the hardware tests. Hardware compilation,
numerical acceptance and performance must be validated on the target NPU.

## Accuracy

Reference semantics (`forward_fp32`, fp32 accumulation):

```python
# SiLU
out = x.float() * torch.sigmoid(x.float())

# SwiGLU
gate_f = gate.float()
out = gate_f * torch.sigmoid(gate_f) * up.float()
```

- **Ground truth**: `forward_fp32` always accumulates in and returns fp32.
- **Dtype path**: `forward` runs the same fp32 math, then casts back to the input dtype.
- **Axis A — batch invariance**: element-wise and row-independent, so a row's output is
  bitwise-identical regardless of batch size or padding (`torch.equal`, `atol=0`).
- **Axis B — tolerance**: as `elementwise` ops, low-precision tolerance follows the
  `elementwise` row of the WS1 numerical contract (`tolerance_contract.json`).

## Ground-truth harness

CUDA and Triton candidates are registered in `OP_SPECS` and can be checked with the
shared issue-#108 CLI:

```bash
python tools/validation/operators/check_operator.py --op silu --candidate cuda --dtype bf16 --device cuda
python tools/validation/operators/check_operator.py --op swiglu --candidate triton --dtype bf16 --device cuda --check-grad
python tools/validation/operators/check_operator.py --op swiglu --candidate triton --packed-order up_gate --dtype bf16 --device cuda --check-grad
python tools/validation/operators/check_operator.py --op silu --candidate pytorch --dtype fp32 --device cpu --check-grad
```

Gold path: `NativeSiLUOp.forward_fp32` / `NativeSwiGLUOp.forward_fp32`.

For SwiGLU, omitting `--packed-order` checks separate `gate` and `up` inputs.
Use `--packed-order gate_up` or `--packed-order up_gate` to check one packed input
with the PyTorch or Triton candidate. Both forms use the same seeded operand
values. `--check-grad` compares `gate`/`up` gradients for separate inputs and the
whole `input` gradient for packed inputs; the channel-order enum is not a gradient
input. Packed report names include the full input shape and channel order.

## Performance Notes

The separate-input Triton path uses a 1-D grid. Packed Triton uses
`(M, ceil(I / 1024))`, where `M` is the product of leading dimensions; row tails are
masked. Rows occupy axis 0 to support more than 65535 rows on CUDA. Both separate
and packed SwiGLU launches use the input backend's device context and restore the
caller's current device. This includes the separate-input device-guard fix from
[PR #665](https://github.com/RL-Align/RL-Kernel/pull/665), tracked in issue #664.

These remain standalone WS1 activation paths. Fused bias+SiLU MLP kernels are a
separate future work item and should continue to validate against this reference.

The benchmark compares eager PyTorch, a `chunk` + two-input Triton adapter, and
direct packed Triton. `--input-layout separate` instead measures the existing
two-input Triton path against PyTorch, using two independent contiguous tensors.
Both layouts share the timing and memory measurement code. Width and packed order
are configurable; the default width 14336 is H3's motivating workload, not an
operator constraint.

```bash
python benchmarks/operators/activation/benchmark_swiglu_packed.py --dry-run
python benchmarks/operators/activation/benchmark_swiglu_packed.py \
  --input-layout separate --width 14336 --output-dir reports/swiglu-separate
python benchmarks/operators/activation/benchmark_swiglu_packed.py \
  --packed-order up_gate --width 14336 --output-dir reports/swiglu-packed
```

Each case checks packed/separate Triton output and gradient bits, then independent
FP32 PyTorch accuracy, before timing. Separate-input cases check training/inference
output bits and independent FP32 accuracy for the output and both input gradients.
CUDA events measure forward, backward and forward+backward, both eagerly and
through CUDA Graph replay. Forward uses
`no_grad`; backward reuses a graph; forward+backward builds a graph per call. Inputs,
validation, JIT and capture are outside timing. Provider order is randomized across
rounds; there is no `torch.compile` or explicit cache eviction. JSON records sample
times, environment and source hashes; reports mark incomplete runs explicitly.

Plots show latency and extra peak allocated memory separately, with distinct
marker sizes for overlapping curves. Latency bands show the observed round range.
Memory is measured with eager calls above the starting allocation, excluding
inputs and prebuilt graphs; it is not CUDA Graph pool size or total model VRAM.
Generated measurements and figures belong in PR attachments, not source control.

## Tests

```bash
python -m pytest tests/ops/activation/test_swiglu.py \
  tests/ops/activation/test_swiglu_packed.py tests/reference/test_swiglu_packed.py \
  tests/benchmarks/test_swiglu_packed_benchmark.py -v
```

Covers: correctness vs an independent fp32 formula, dtype paths, Axis-A batch invariance
(slice + padding), input purity, gradient flow, the SwiGLU shape guard, CUDA/Triton vs
native forward+backward, registry dispatch, and the issue-#108 `OP_SPECS` harness.
Packed tests cover both orders, bitwise agreement with the separate-input Triton
path, training/no-grad/inference equality, batch/padding invariance, independent
FP32 forward/gradient accuracy, strided tensors and gradients, empty rows, tile
tails, CUDA Graph replay, and noncurrent-device restoration. Hardware checks skip
without a GPU; device restoration requires two GPUs. The `OP_SPECS` `swiglu` entry
checks both input forms through `--packed-order`, including forward and backward
accuracy against the FP32 gold path.

## Implementation Files

- `rl_engine/reference/activation/swiglu.py` — gold
- `rl_engine/contracts/operators/activation.py` — explicit packed channel order
- `rl_engine/backends/cuda/activation/swiglu.py` — CUDA wrappers
- `rl_engine/backends/shared/triton/activation/swiglu.py` — Triton kernels
- `rl_engine/backends/ascend/activation/swiglu.py` — Ascend SwiGLU autograd wrapper
- `rl_engine/backends/ascend/activation/silu.py` — Ascend SiLU autograd wrapper
- `csrc/ascend/activation.asc` — Ascend C SiLU + SwiGLU forward/backward kernels
- `csrc/ascend/bindings.asc` — shared NPU extension bindings
- `csrc/cuda/activation/activation.cu` — CUDA kernels
- `rl_engine/runtime/registry.py`
- `rl_engine/validation/operators/operator_specs.py`
- `tests/ops/activation/test_swiglu.py`
- `tests/backends/ascend/test_silu_ascend.py`

## Known Limitations

- SwiGLU requires `gate` and `up` to share shape, dtype, and device; no broadcasting.
- No fused `bias + SiLU` variant is provided.
- Packed input support here covers the common PyTorch and Triton classes; it does
  not change the CUDA FFN's separate packed implementation or model bindings.
