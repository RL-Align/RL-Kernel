# Router Sqrt-Softplus

DSV4 task 29 transforms raw gate logits into independently observable scores:
`score = sqrt(softplus(logit))`. It does not select experts, add selection bias,
normalize selected weights, apply routing scales, enforce capacity, or compute
auxiliary losses. Those operations belong to tasks 30–34.

## API and input contract

```python
from rl_engine.ops.moe.router_sqrt_softplus import router_sqrt_softplus

result = router_sqrt_softplus(logits, model_version="synchronized-model-v1")
scores = result.scores       # FP32 [M, E], with first-order autograd
execution = result.execution
```

Inputs and outputs are FP32 `[M,E]`, with `E > 0`; empty M is supported.
All finite FP32 logits and finite upstream gradients are admitted. Positive
strides, transposes, column slicing and row padding are supported. Expanded
zero-stride input views, other dtypes/devices, NaN/Inf, Graph capture and
higher-order derivatives are unsupported. Upstream autograd gradients may be
expanded/noncontiguous; the backend makes them contiguous before its VJP.

The ROCm implementation admits gfx942/gfx950; only MI300X/gfx942 has hardware
evidence here. The semantic catalog registers the exact backend
`rlkernel.router_sqrt_softplus.fp32.stable_tail.v1` for training/rollout, FP32,
TP1. It never selects a PyTorch or unpatched AITER fallback.

## Frozen operation and derivative order

The table references AITER
[`_apply_score_mode` at 326cffa](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py)
and mLite's
[fixed-route derivative at 327a238](https://github.com/ISEEKYAN/Megatron-LM/blob/327a238239448de6771beea185dc6902a8de3eaa/experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/moe/module.py).
Source review found two precision gaps that this companion patch explicitly
changes before using the forward for training and inference:

- AITER's `log(1 + exp(-abs(x)))` loses negative scores when `1+exp(x)` rounds
  to one. The patch uses OCML/libdevice `log1p` and `exp`, retaining FP32 values.
- Forming softplus before sqrt can underflow even when the final score is
  representable; mLite's FP32 `sqrt(F.softplus(x))` also exhibits this at -120.
  A fixed negative-tail branch avoids this intermediate underflow.

Let `s` be the output and `g` the upstream FP32 gradient. The versioned contract
uses these rules, with FP32 intermediates and no final downcast:

| Region | Forward | VJP |
| --- | --- | --- |
| `x >= -16` | `sqrt(max(x,0) + log1p(exp(-abs(x))))` | Form stable sigmoid, then `slope = sigmoid(x)/(2*s)`, then `g*slope` |
| `-160 <= x < -16` | `exp(x/2)` | Form `slope = 0.5*s`, then `g*slope` |
| `x < -160` | `exp(x/2)` | `copysign(exp((log(abs(g)) + x/2) - log(2)), g)` |

Sigmoid uses `e=exp(-abs(x))`, then `1/(1+e)` for nonnegative x and `e/(1+e)`
otherwise. The deep-tail VJP preserves a representable gradient even if the
FP32 score itself has rounded to zero; zero upstream gradients yield signed
zeros. No epsilon or clipping is inserted. Before FP32 rounding, the negative-tail
approximation differs from the mathematical score by less than `exp(-16)/4`
in relative value, and from its slope by less than `3*exp(-16)/4`. Subnormal rounding and
eventual underflow remain platform-specific FP32 behavior, tested explicitly.

The AITER standalone forward and VJP use **256 elements/program, 4 warps,
1 stage, `enable_fp_fusion=False`**. M only changes grid size and masks.
The shared helper is also used by the existing AITER Top-K, Hash and Grouped
sqrt-softplus branches; their sqrt-softplus launches disable FP fusion to
match the standalone operation's rounding. Other score modes retain their
existing fusion setting. Selected score values are checked byte for byte;
this is not signoff of expert ordering, tie-breaking or complete routing.

## AITER companion patch and reproduction

GPU arithmetic lives in the AITER companion patch under
`rl_engine/backends/rocm/moe/patches/router-sqrt-softplus.patch`. RLK adds input
checks, first-order autograd, contract registration and execution records.
The patch is based on AITER commit
`326cffa38c9eda16bef19375ba55334e4d3acd59`; it is not an upstream AITER release.

Use an isolated checkout rather than changing an existing runtime installation:

```bash
git clone https://github.com/ROCm/aiter.git artifacts/task29/aiter
git -C artifacts/task29/aiter checkout 326cffa38c9eda16bef19375ba55334e4d3acd59
git -C artifacts/task29/aiter apply --check \
  "$PWD/rl_engine/backends/rocm/moe/patches/router-sqrt-softplus.patch"
git -C artifacts/task29/aiter apply \
  "$PWD/rl_engine/backends/rocm/moe/patches/router-sqrt-softplus.patch"

AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task29/aiter:$PWD" \
  python -m pytest tests/ops/moe/test_router_sqrt_softplus.py \
  tests/runtime/test_router_sqrt_softplus_dispatch.py tests/runtime/test_kernel_registry.py -q

AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task29/aiter:$PWD" \
  python -m tools.validation.operators.check_router_sqrt_softplus \
  --output artifacts/task29/replay.json
```

The binding verifies the companion API ID, launch configuration, patch hash,
and hashes of all three affected AITER files. A missing patch or different
source fails explicitly. Execution records include those fingerprints, the
base AITER revision, GPU, ROCm/PyTorch/Triton versions, shape/strides, model
version and actual forward/VJP kernels. The caller owns model-version barriers.
There are no derived weights or route-state caches.

## Validation and performance

The independent CPU FP64 **mathematical oracle is not a production-arithmetic
emulator**. It deliberately uses a more conservative approximation cutoff of
**-36**, compared with the production contract's **-16**. In [-36,-16), it
retains `sqrt(logaddexp(x,0))` and the mathematical derivative, allowing tests
to measure the production approximation's error independently. Below -36,
the score/slope approximation bounds (`exp(-36)/4`, `3*exp(-36)/4`) are below
FP64 machine epsilon and avoid intermediate underflow.

Separate tests validate the production -16 branch formula and its FP32 rounding
near the boundary; raw-bit BI checks cover repeated execution and batch/layout
changes. Mathematical accuracy checks alone cannot certify branch/order
equivalence when different formulas fall within tolerance. Finite differences
validate the mathematical oracle separately. Score tolerance is `rtol=5e-7`;
VJP tolerance is `rtol=2e-5`, including deep-tail rescaling with large upstream
gradients. Both use an absolute allowance of eight minimum FP32 subnormal
quanta (`2^-146`). Tolerances never replace raw-bit BI checks.

Tests cover 256 experts, tails at E=1/6/255/257, M=0/1/16/17/32/65/257,
changed neighboring rows and row placement, ten random seeds, padded/transposed
storage, branch-neighbor values, the full finite FP32 endpoints, repeated VJPs,
input mutation and explicit rejection. mLite comparisons cover [-80,80]; its
deep-tail underflow is retained as a negative control. NaN/Inf are rejected
before arithmetic. Graph is explicitly rejected as in the first-round plan.

The replay tool records deterministic synthetic FP32 logits/upstream gradients
and their hashes. It does not load a checkpoint, tokenizer or Preview data.
See `reports/experiments/router-sqrt-softplus/` for MI300X results. Timing
separates the full RLK adapter (including finite checks/synchronization and
provenance) from raw AITER forward/VJP calls, using 10 warmups and 30 GPU-event
samples around each Python call. These are not isolated instruction timings
or full-model throughput measurements.

MI355X, NVIDIA, Graph, distributed execution and full-model/framework
integration are unvalidated. Task 28's Gate PR is an independent dependency
for eventual composition; this PR consumes FP32 logits directly. Tasks 30–34
own complete routing and its training semantics.
