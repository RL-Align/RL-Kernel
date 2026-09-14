# Fused LogP

Fused LogP computes selected token log probabilities from model logits. It targets RL
post-training workloads where repeated `log_softmax + gather` operations create memory
pressure at large group sizes.

## Entry Point

```python
from rl_engine.kernels.registry import kernel_registry

logp_op = kernel_registry.get_op("logp")
output = logp_op(logits, token_ids)
```

The PyTorch native reference also exposes the Issue #108 interface:

```python
from rl_engine.kernels.ops.pytorch.loss.logp import NativeLogpOp

logp_ref = NativeLogpOp()
output = logp_ref.forward(logits, token_ids)
reference = logp_ref.forward_fp32(logits, token_ids)
```

`apply(...)` and `apply_fp32(...)` remain available as backward-compatible aliases.

## Backends

| Backend | Wrapper | Extension entry point | Notes |
| --- | --- | --- | --- |
| CUDA SM90 | `FusedLogpSM90Op` | `_C.fused_logp_sm90`, `_C.fused_logp_sm90_with_lse` | Experimental TMA path for eligible bf16 logits on Hopper-class GPUs. |
| CUDA generic | `FusedLogpGenericOp` | `_C.fused_logp` | Generic compiled extension fallback. |
| PyTorch native | `NativeLogpOp` | — | PyTorch baseline/reference path. |

## Tensor Contract

Let `N` be the product of the leading dimensions of `logits`.

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `logits` | `[..., V]` | Floating point | The generic and native wrappers flatten leading dimensions. The SM90 fast path requires contiguous 2D bf16 `[N, V]`. |
| `token_ids` | `[...]` | Integer | Shape must match the leading dimensions of `logits`, with every value in `[0, V)`. Wrappers move IDs to the logits device and use int64 for native/generic kernels or int32 for the SM90 forward. |
| `row_indices` | `[K]` | Integer | Optional flattened row indices used by indexed variants. Values must be in `[0, N)` and are converted to int64. |
| Output | `[...]` | See below | One selected log probability per input row. |

`forward` / `apply` return the logits dtype on the native and generic paths, while
`forward_fp32` / `apply_fp32` and the other allocating `*_fp32` variants return
float32. Caller-provided `*_out` variants use the output buffer dtype. An eligible
SM90 TMA call always returns float32.

The experimental SM90 path is selected only when all of the following hold:

- the extension was built with `KERNEL_ALIGN_FORCE_SM90=1` for a supported GPU;
- `RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP=1` is set at runtime;
- `logits` is a contiguous 2D bf16 tensor; and
- `V` is divisible by 8, so each bf16 row has the 16-byte-aligned stride required
  by TMA.

If the build, environment, or GPU requirements are not met, the registry does not
select the SM90 backend. If an input tensor is ineligible, the SM90 wrapper delegates
to the CUDA generic backend.

## Reference Semantics

```python
ref = torch.log_softmax(logits.float(), dim=-1)
ref = torch.gather(ref, dim=-1, index=token_ids.unsqueeze(-1).long()).squeeze(-1)
```

## Backward / Autograd

The CUDA generic backend is differentiable with respect to `logits`. When
`logits.requires_grad` is set under grad mode, the allocating variants —
`apply` / `apply_fp32` / `indexed_fp32` / `online_fp32` / `online_indexed_fp32` —
route through a `torch.autograd.Function` using the same forward reduction path as
the corresponding no-grad call. It additionally saves separate float32 `row_max`
and `log_sum` statistics, avoiding the precision loss that can occur when a large
constant logit offset is folded into one float32 log-sum-exp value.

Backward rebuilds probabilities as `exp((logit - row_max) - log_sum)` and computes

```
grad_logits[v] = grad_out * (1[v == token_id] - softmax(logits)[v])
```

in a dedicated kernel without materializing another logits-sized intermediate.
Indexed variants only touch selected rows; all other rows receive exactly-zero
gradient.

The generic CUDA `*_out` variants remain non-differentiable, matching PyTorch's
`out=` convention, and raise `RuntimeError` for grad-requiring `logits`.
`DeterministicLogpCUDAOp` is also forward-only.

Eligible SM90 calls are differentiable too. Grad mode uses
`fused_logp_sm90_with_lse`, which runs the same TMA reduction as the no-grad entry
point while also returning `row_max` and `log_sum`. It then reuses the generic
elementwise CUDA backward kernel; no SM90-specific backward kernel is needed.
Grad mode changes neither the float32 output contract nor the forward values.

## Tests

```bash
python -m pytest tests/test_logp.py -q -rs
python -m pytest tests/test_op_accuracy.py -q -rs
python -m pytest tests/test_fused_logp_backward.py -q -rs
```

`tests/test_logp.py` covers the PyTorch reference contract, dtype behavior,
backward-compatible aliases, batch invariance, and registry dispatch. The operator
accuracy tests validate native/CUDA fused API compatibility.
`tests/test_fused_logp_backward.py` covers gradients across dtypes and variants,
train/inference bitwise consistency, forward-only guards, and SM90 multi-tile,
partial-tile, and fallback behavior.

CUDA cases require the compiled extension. The SM90 cases additionally require a
Hopper GPU and an SM90-enabled build; otherwise pytest reports them as skipped. A
matching H100 build can be installed with:

```bash
FORCE_CUDA=1 KERNEL_ALIGN_FORCE_SM90=1 TORCH_CUDA_ARCH_LIST="9.0+PTX" \
  python -m pip install --no-build-isolation --no-deps -e .
```

## Implementation Files

- `rl_engine/kernels/registry.py`
- `rl_engine/_C.pyi`
- `rl_engine/kernels/ops/pytorch/loss/logp.py`
- `rl_engine/kernels/ops/cuda/loss/logp.py`
- `csrc/ops.cpp`
- `csrc/fused_logp_kernel.cu`
- `csrc/cuda/fused_logp_sm90.cu`
- `tests/test_logp.py`
- `tests/test_op_accuracy.py`
- `tests/test_fused_logp_backward.py`
