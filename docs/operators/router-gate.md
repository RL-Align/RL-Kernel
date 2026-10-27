# FP32 Router Gate Projection

Task 28 of the DSV4 ROCm plan projects `hidden[M,K] @ weight[E,K].T`.
Both operands are unquantized BF16; the output is FP32 `[M,E]`. The model
adapter uses the same entry for training, inference, and recomputation.

## API and dispatch

```python
from rl_engine.ops.moe.router_gate import router_gate_projection

result = router_gate_projection(hidden, weight, model_version="synchronized-version-v1")
logits = result.logits          # FP32; no intermediate BF16 logits
execution = result.execution   # JSON-compatible actual backend and configuration
```

`DeepSeekV4Gate` in `rl_engine.models.deepseek_v4.gate` accepts the already
loaded `torch.nn.Parameter` and returns logits. It preserves parameter identity
and exposes `last_execution`. The caller owns the synchronization barrier and
updates `model_version` after loading new weights, LoRA, or scales. There is no
derived weight cache. Concurrent callers should use the per-call `result.execution`
record instead of the module's last-call diagnostic.

The semantic catalog registers `router_gate_projection` with backend ID
`rlkernel.aiter_fixed_bf16_fp32.v1`, for training/rollout, ROCm, BF16, TP1.
Unsupported devices, dtypes, dimensions, zero-stride expanded inputs, and Graph
capture fail explicitly. Positive strides, row padding, transposes, and empty
M are supported; K and E must be positive. Eligible architectures are gfx942
and gfx950; only gfx942 has measurements in this change.

## Arithmetic contract and task-9 dependency

The backend reuses the task-table AITER entry
`aiter.ops.triton.gemm.basic.gemm_a16w16.gemm_a16w16`, fixing its configuration:

| Setting | Value |
| --- | --- |
| Backend | Triton, nonpersistent |
| M/N/K tiles | 16 / 64 / 32 |
| Warps / stages | 4 / 2 |
| MFMA non-K dimension / kpack | 16 / 1 |
| Group size M / split count | 1 / 1 |
| K span | `ceil(K/32)*32`, independent of M |
| Accumulation / output | FP32 / FP32 |
| Bias / activation | absent / absent |
| Partial merge / atomic writes | none / none |
| Temporary workspace | zero; caller receives a newly allocated output |

Every row visits the K tiles in ascending order, using the same dot product
and FP32 accumulation. M affects the grid and masked loads/stores, never tile
sizes or the K reduction. No tuning database is consulted. We do not use the
existing RLK BF16 tree output followed by `.float()`, which would discard close
gate scores. This wrapper establishes the BF16/FP32 subset needed by task 28;
it does not certify all other task-9 GEMM modes, bias semantics, or low precision.

Both the older Triton-only AITER API and the RFC-pinned multi-backend API are
recognized. The latter receives `backend="triton", persistent=False` explicitly.
Per-call records include the profile, wrapper/kernel SHA256, AITER/Triton/PyTorch/
ROCm versions, GPU, shapes, strides, model version, and whether a kernel launched.
Import, compilation, or execution failures propagate. There is no fallback.

First-order projection autograd computes `dX=dlogits@W`, `dW=dlogits.T@X`
through RLK's existing fixed FP32-dot Triton kernel (64/64/32 tiles, no TF32).
Upstream FP32 gradients are preserved until the final cast to each BF16 operand's
gradient dtype. Mathematical accuracy and repeatability are tested separately.
Different microbatch accumulation orders and P6 branch merging are not certified
by these local projection derivatives. Higher-order derivatives are unsupported.

## References and frozen replay

The design follows the BI branch of
[vLLM GateLinear at d262044](https://github.com/inaniloquentee/vllm/blob/d262044021bda30273cb0833cd8bf05dd27deab1/vllm/model_executor/layers/fused_moe/router/gate_linear.py)
and retains both the small-M CuTe and `torch.mm` entries from
[mLite at 327a238](https://github.com/ISEEKYAN/Megatron-LM/blob/327a238239448de6771beea185dc6902a8de3eaa/experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/moe/module.py).
`rl_engine.reference.moe.router_gate.cuda_gate_reference` exposes them separately;
CuTe absence raises an error. CUDA comparison is separate from AMD acceptance.
The task-table AITER reference is
[326cffa](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/gemm/basic/gemm_a16w16.py).

Replay fixes DeepSeek-V4-Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. The checkpoint header confirms
`layers.0.ffn.gate.weight` is BF16 `[256,4096]`; its raw tensor SHA256 is
`e110e3c4e310e7c9981f287038c5a0c8ad73da1b36b36feb3b909e04477a6a12`.
Only this approximately 2 MB tensor is downloaded. Activations are synthetic,
CPU-generator seed 28; no tokenizer, prompts, Preview weights, expert FP4
conversion, or LoRA update is part of this operator fixture.

```bash
python -m tools.weights.prepare_router_gate_weights \
  --output artifacts/router-gate/0731-layer0.pt
python -m tools.validation.operators.check_router_gate \
  --weights artifacts/router-gate/0731-layer0.pt \
  --output artifacts/router-gate/replay.json
python -m pytest tests/ops/moe/test_router_gate.py tests/runtime/test_router_gate_dispatch.py -q
```

Tests compare raw FP32 bits at M=1/16/17 and larger batches, changing row
positions, neighboring rows, padding, strides, and gradient mode. Cases include
K/N tails, empty M, repeated gradients, live weight updates, explicit rejection,
and a close-score negative control: `1` versus `1+2^-17` must remain distinct.
Accuracy uses FP64 reference GEMM with `atol=2e-4, rtol=3e-5`; tolerance never
replaces the zero-bit-mismatch criterion.

The checked-in report is
`reports/experiments/router-gate/task28-mi300x.json`. Timings use 10 warmups
and 30 GPU-event samples per shape, reporting median adapter latency, including
validation/provenance overhead. The report identifies the installed AITER
`0.1.12.post2.dev214+gb5e03ed19` and its exact source hashes. That installation
has the older Triton-only API; the newer RFC-pinned API was source-reviewed,
not executed here. Changing dependencies requires rerunning this acceptance.

## Support scope

MI300X local gate projection is tested. MI355X and the CUDA mm/CuTe comparisons
remain untested on hardware here. This component is not a complete DSV4 model
or a vLLM/Megatron/verl/vime hook. Those framework integrations and the global
two-update, two-node logprob acceptance remain separate work packages.
Graph is explicitly disabled, as in the first-round plan. Score transforms,
Hash routing, Top-6 tie rules, selection bias, capacity, and auxiliary losses
belong to tasks 29–34; no selection bias is added by this projection.
