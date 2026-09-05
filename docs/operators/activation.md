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

## Entry Point
```python
from rl_engine.kernels.registry import kernel_registry

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

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| PyTorch fallback | `NativeSiLUOp` / `NativeSwiGLUOp` | None | fp32 ground-truth reference; CPU and any GPU. |
| CUDA | `SiLUCudaOp` / `SwiGLUCudaOp` | `_C.silu_*` / `_C.swiglu_*` | General CUDA (fp16/bf16/fp32); math in fp32. |
| Triton | `TritonSiLUOp` / `TritonSwiGLUOp` | Triton JIT | Portable GPU baseline; same fp32 math contract. |
| Ascend C | `SwiGLUAscendOp` | `_C_npu.swiglu_forward` / `swiglu_backward` | NPU SwiGLU forward and backward; fp16/bf16/fp32 inputs, FP32 math. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `x` (SiLU) | `[..., N]` | float (fp16/bf16/fp32) | Any shape; last dim arbitrary (Qwen3-8B `I=12288`). |
| `gate` (SwiGLU) | `[..., I]` | float | `gate_proj` output. |
| `up` (SwiGLU) | `[..., I]` | float | `up_proj` output; **must share `gate`'s shape, dtype, and device**. |
| output | same as input | `forward`: input dtype · `forward_fp32`: float32 | Same shape as input. |

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
| `npu` | SwiGLU: Ascend C → PyTorch native; SiLU: PyTorch native |

If the CUDA extension is not built (or symbols are missing), the registry falls back to
Triton, then to the native gold.

On NPU, a missing Ascend extension or missing SwiGLU symbols causes the registry to
select PyTorch native. Construct `SwiGLUAscendOp` directly when the Ascend C kernel
is required; its constructor raises an error if either native symbol is missing.

## Ascend C Build and Validation

On a Linux Ascend host with matching PyTorch, `torch_npu` and CANN installed, source
the CANN environment and build the existing NPU extension:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
KERNEL_ALIGN_FORCE_ASCEND=1 KERNEL_ALIGN_ASCEND_ARCH=dav-2201 \
  python -m pip install --no-build-isolation -e .
python -m pytest tests/test_swiglu.py -v
python scripts/check_operator.py --op swiglu --candidate ascend --dtype bf16 --device npu --check-grad
```

The kernel uses the A2/A3 vector programming model (`dav-2201`). Other architectures
require separate build and device validation. `setup.py` automatically includes all
`csrc/ascend/*.asc` files; `bindings.asc` defines the shared `_C_npu` module entry.

```python
import torch
import torch_npu
from rl_engine.kernels.registry import kernel_registry

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
python scripts/check_operator.py --op silu --candidate cuda --dtype bf16 --device cuda
python scripts/check_operator.py --op swiglu --candidate triton --dtype bf16 --device cuda --check-grad
python scripts/check_operator.py --op silu --candidate pytorch --dtype fp32 --device cpu --check-grad
```

Gold path: `NativeSiLUOp.forward_fp32` / `NativeSwiGLUOp.forward_fp32`.

## Performance Notes

Element-wise kernels with a fixed 1-D grid (CUDA) / `BLOCK=1024` (Triton). Suitable as the
standalone WS1 activation path; fused bias+SiLU MLP kernels remain a separate future work
item and should continue to validate against this reference.

## Tests

```bash
python -m pytest tests/test_swiglu.py -v
```

Covers: correctness vs an independent fp32 formula, dtype paths, Axis-A batch invariance
(slice + padding), input purity, gradient flow, the SwiGLU shape guard, CUDA/Triton vs
native forward+backward, registry dispatch, and the issue-#108 `OP_SPECS` harness.

## Implementation Files

- `rl_engine/kernels/ops/pytorch/activation/swiglu.py` — gold
- `rl_engine/kernels/ops/cuda/activation/swiglu.py` — CUDA wrappers
- `rl_engine/kernels/ops/triton/activation/swiglu.py` — Triton kernels
- `rl_engine/kernels/ops/ascend/activation/swiglu.py` — Ascend autograd wrapper
- `csrc/ascend/activation.asc` — Ascend C forward/backward kernels
- `csrc/ascend/bindings.asc` — shared NPU extension bindings
- `csrc/cuda/activation.cu` — CUDA kernels
- `rl_engine/kernels/registry.py`
- `rl_engine/kernels/gtest/operator_specs.py`
- `tests/test_swiglu.py`

## Known Limitations

- SwiGLU requires `gate` and `up` to share shape, dtype, and device; no broadcasting.
- No fused `bias + SiLU` or `chunk(y,2) + silu_and_mul` variant yet (vLLM-style
  `SiluAndMul` on a packed gate/up tensor). Callers that hold a packed tensor should
  split first, then call `swiglu`.
