# Fused LogP

Fused LogP computes selected token log probabilities from model logits. It targets RL
post-training workloads where repeated `log_softmax + gather` operations create memory
pressure at large group sizes.

## Entry Point

```python
from rl_engine.runtime.registry import kernel_registry

logp_op = kernel_registry.get_op("logp")
output = logp_op(logits, token_ids)
```

The PyTorch native reference also exposes the Issue #108 interface:

```python
from rl_engine.reference.logprob.logp import NativeLogpOp

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
python -m pytest tests/ops/logprob/test_logp.py -q
python -m pytest tests/ops/test_op_accuracy.py -q
```

`tests/ops/logprob/test_logp.py` covers the PyTorch reference contract, dtype behavior,
backward-compatible aliases, batch invariance, and registry dispatch. The existing
operator accuracy tests continue to validate native/CUDA fused API compatibility.

## Shared Python ownership (RFC #642 Task 19)

`rl_engine/ops/logprob/fused_logp.py` owns common preparation, native variant
delegation, autograd forward/save mechanics, and the existing FP32 PyTorch VJP.
CUDA and MUSA each own their concrete wrapper, extension binding, and backward
selection. MUSA no longer imports CUDA. CUDA's canonical and legacy leaf imports
still share one module object; `_FusedLogpAutograd` remains in the CUDA module.
Native symbols, arithmetic, and registry priorities are unchanged.

The CPU ownership tests use recording backends and a frozen source oracle from
`main@49941801f942428fd52ac5b66a1b0c070418724e`. The snapshot is test-only and lets
the before/after checks run in shallow checkouts. Native MUSA execution requires
separate hardware validation.

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest -q \
  tests/ops/logprob/test_fused_logp_ownership.py \
  tests/ops/logprob/test_logp.py tests/runtime/test_layout_compatibility.py \
  tests/runtime/test_kernel_registry.py
```

On a provisioned single H100 with this checkout's CUDA extension built, run:

```bash
unset RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP
PYTHONPATH=. python -m pytest -q tests/ops/test_op_accuracy.py -k 'logp or logprob'
PYTHONPATH=. python -m pytest -q tests/ops/logprob/test_deterministic_logp.py
PYTHONPATH=. python -m pytest -q \
  tests/e2e/test_rl_kernel_loss_step.py::test_minimal_rl_loss_step_fused_logp_candidate_cuda
PYTHONPATH=. python -m tools.validation.operators.compare_fused_logp_ownership \
  --baseline 49941801f942428fd52ac5b66a1b0c070418724e \
  --output /tmp/task19-h100.json
```

The comparison command requires the baseline Git object, asserts real CUDA
extension use, and compares 18 dtype/shape/layout cases byte-for-byte for forward,
backward, and eight variants. It independently checks reference accuracy and
records GPU, software, extension and tensor hashes. The generic CUDA backward
remains PyTorch; there is no new CUDA native backward in Task 19. Missing native
execution or skipped accelerator tests do not establish hardware acceptance.

Shared source is included in the gradient audit inventory. Runtime fingerprints
also include the defining module of cross-module runtime entrypoints, so moving
the VJP cannot hide later shared-helper source changes. Same-module fingerprint
payloads and stable public operator identifiers are preserved.

## Implementation Files

- `rl_engine/runtime/registry.py`
- `rl_engine/reference/logprob/logp.py` — PyTorch native reference
- `rl_engine/backends/cuda/logprob/logp.py` — CUDA fused LogP (SM90 + generic)
- `rl_engine/backends/ascend/logprob/logp.py` — Ascend deterministic op
- `csrc/bindings/ops.cpp`
- `csrc/cuda/logprob/fused_logp_kernel.cu`
- `csrc/cuda/logprob/fused_logp_sm90.cu`
- `csrc/cuda/logprob/deterministic_logp_kernel.cu` — CUDA deterministic kernel (reference reduction)
- `csrc/ascend/fused_logp_ascend.asc` — Ascend C forward kernel
- `csrc/ascend/npu_module.cpp` — shared pybind entry for `rl_engine._C_npu`
- `tests/ops/logprob/test_logp.py`
- `tests/backends/ascend/test_logp_ascend.py` — Ascend correctness + batch-invariance tests
