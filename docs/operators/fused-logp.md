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
| CUDA SM90 | `FusedLogpSM90Op` | `_C.fused_logp_sm90` | Experimental TMA-oriented path for 2D contiguous bf16 logits on Hopper-class GPUs. It is disabled by default and requires `RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP=1`; otherwise the wrapper delegates to the CUDA generic fallback. |
| CUDA generic | `FusedLogpGenericOp` | `_C.fused_logp` | Generic compiled extension fallback. |
| Ascend NPU | `FusedLogpAscendOp` | `_C_npu.fused_logp_ascend` | Batch-invariant Ascend C forward: two-pass (row max, then sum-exp) fp32 reduction with a fixed tile order, mirroring the CUDA deterministic kernel. Output is fp32, matching `DeterministicLogpCUDAOp`'s contract; out-of-range targets yield 0.0. |
| PyTorch native | `NativeLogpOp` | None | PyTorch baseline/reference path. |

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `logits` | `[N, V]` | `bfloat16` for the experimental SM90 fast path; fp16/fp32 use generic fallback | Contiguous, on the target device for the experimental SM90 fast path. |
| `token_ids` / `labels` | `[N]` | Converted to `int32` | Same logical device as `logits`. |
| Output | `[N]` | Backend-defined tensor dtype | One selected log probability per row. |

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
- `rl_engine/kernels/ops/pytorch/loss/logp.py` — PyTorch native reference
- `rl_engine/kernels/ops/cuda/loss/logp.py` — CUDA fused LogP (SM90 + generic)
- `rl_engine/kernels/ops/ascend/loss/logp.py` — Ascend deterministic op
- `csrc/ops.cpp`
- `csrc/fused_logp_kernel.cu`
- `csrc/cuda/fused_logp_sm90.cu`
- `csrc/deterministic_logp_kernel.cu` — CUDA deterministic kernel (reference reduction)
- `csrc/ascend/fused_logp_ascend.asc` — Ascend C forward kernel
- `csrc/ascend/npu_module.cpp` — shared pybind entry for `rl_engine._C_npu`
- `tests/test_logp.py`
- `tests/test_logp_ascend.py` — Ascend correctness + batch-invariance tests
