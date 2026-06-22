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

| Backend | Wrapper | Native symbol | Notes |
| --- | --- | --- | --- |
| CUDA SM90 | `FusedLogpSM90Op` | `_C.fused_logp_sm90` | TMA-oriented path for Hopper-class GPUs. |
| CUDA generic | `FusedLogpGenericOp` | `_C.fused_logp` | Generic compiled extension fallback. |
| PyTorch native | `NativeLogpOp` | None | PyTorch baseline/reference path. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `logits` | `[..., V]` | Floating point | Contiguous for fused CUDA paths; arbitrary leading dimensions. |
| `token_ids` / `labels` | `[...]` | Integer | Must match `logits.shape[:-1]`. |
| Output | `[...]` | See below | One selected log probability per row. |

For `NativeLogpOp`, `forward(...)` returns the input dtype and `forward_fp32(...)`
returns `torch.float32`.

## Reference Semantics

```python
ref = torch.log_softmax(logits.float(), dim=-1)
ref = torch.gather(ref, dim=-1, index=token_ids.unsqueeze(-1).long()).squeeze(-1)
```

## Tests

```bash
python -m pytest tests/test_logp.py -q
python -m pytest tests/test_op_accuracy.py -q
```

`tests/test_logp.py` covers the PyTorch reference contract, dtype behavior,
backward-compatible aliases, batch invariance, and registry dispatch. The existing
operator accuracy tests continue to validate native/CUDA fused API compatibility.

## Implementation Files

- `rl_engine/kernels/registry.py`
- `rl_engine/kernels/ops/pytorch/loss/logp.py`
- `rl_engine/kernels/ops/cuda/loss/logp.py`
- `csrc/ops.cpp`
- `csrc/fused_logp_kernel.cu`
- `csrc/cuda/fused_logp_sm90.cu`
- `tests/test_logp.py`
