# Qwen-Image TP Replicated Modules (CUDA)

## Summary

`tp_replicated_modules` qualifies the replicated boundaries in RFC
[#386](https://github.com/RL-Align/RL-Kernel/issues/386): image/text input,
timestep MLP, separate stream modulation, parameter-free block LayerNorm,
output AdaLN and image output. Q/K RMSNorm has head-sharded activations and
replicated gain, so its parameter gradient also needs communication.

The adapters reuse the existing CUDA Linear, RMSNorm, FP32 row fold,
canonical backward session and deterministic AllGather. They add no CUDA
kernel and do not change the existing WS1 entries.

## Entry Point

```python
from rl_engine.backends.cuda.model_specific.qwenimage.replicated_modules import (
    broadcast_condition,
    replicated_linear,
    replicated_qk_rmsnorm,
    replicated_silu_projection,
)
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

with canonical_backward_session() as session:
    y = replicated_linear(
        x, weight, bias, logical_keys=keys,
        parameter_id="block.0.image.projection",
    )
    y.backward(upstream)
session.validate_complete()
```

For `replicated_qk_rmsnorm`, pass `[M,24/P,128]`, a complete `[128]` gain,
stable row keys and a `DeterministicCollective` for TP2/4/8. TP1 uses
`collective=None`. Set the current device and initialize the process group
before constructing the collective. Each rank must execute matching calls
and own the corresponding contiguous global heads.

`broadcast_condition(condition, keys, parameter_id=...)` maps complete
request conditions to token rows. The first key indexes the condition batch.
When the same condition has multiple consumers, share its parameter ID and
use `[request, consumer, token]` keys with stable consumer identities.

Complete all forward registrations before backward in the same session.
Use separate parameter IDs for layers, streams and distinct gain parameters.
Call `validate_complete()` after backward. Q/K norm adds an autograd dependency
between successive uses of the same collective, following the existing TP
MLP adapter, to fix their reverse communication order.

## Backends

| Backend | Entry or primitive | Role |
| --- | --- | --- |
| CUDA | `replicated_linear` | Existing #537 forward/dX/dW/dbias with canonical rows |
| CUDA | `replicated_silu_projection` | #537 forward and FP32 analytic SiLU VJP with canonical rows |
| CUDA | `canonical_cuda_rmsnorm` | Existing text-entry RMSNorm with canonical gain gradient |
| CUDA | `replicated_qk_rmsnorm` | Existing RMSNorm plus global head-ordered gain contributions |
| CUDA / CPU | `broadcast_condition` | Canonical request/consumer/token fold |
| ATen | SiLU, sinusoid, FP32 LayerNorm, scale/shift/gate | Declared pointwise and parameter-free boundaries |
| CPU | Fixture `module_forward(..., backend="pytorch")` | Independent FP32 autograd comparison |

## Tensor Contract

`P` is 1, 2, 4 or 8. Linear operands use BF16 or FP32 with matching dtype
and device. The SiLU projection preserves #537's FP32 dActivation and analytic
SiLU derivative before its single input-gradient cast. Time values are fixed FP32 scheduler inputs; the timestep MLP's
weights and biases train. Logical keys are int64 `[M,2]` or `[M,3]` on the
input device. A negative first key marks padding, whose upstream gradient
must be zero. Empty forward calls are omitted by the fixture.

| Boundary | Input | Parameters / output | Ownership |
| --- | --- | --- | --- |
| Image input | `[M,64]` | Linear `64→3072` | Full replicas |
| Text input | `[M,3584]` | RMSNorm, Linear `3584→3072` | Full replicas |
| Timestep | `[R,1]` | Cos-first sinusoid 256, Linear `256→3072→3072`, SiLU | Full replicas |
| Image / text modulation | `[R,3072]` | Separate SiLU + Linear `3072→18432` | Full replicas |
| Block norm | `[M,3072]` | Parameter-free LayerNorm, epsilon `1e-6` | Full replicas |
| Modulation / residual probe | `[M,3072]`, `[R,9216]` | Shift, scale, gate and token broadcast | Full replicas |
| Output AdaLN | `[M,3072]`, condition `[R,3072]` | SiLU + Linear `3072→6144`, scale/shift, LayerNorm | Full replicas |
| Image output | `[M,3072]` | Linear `3072→64` | Full replicas |
| Q/K RMSNorm | `[M,24/P,128]` | Shared gain `[128]`, epsilon `1e-6` | Local heads, full gain |

Linear reuses the native vector-alignment restrictions. All listed model widths
satisfy them. Adapters materialize noncontiguous operands when required and
support first-order gradients. Q/K norm and trainable Linear parameters require
a canonical session. Inference uses the same native forward arithmetic.

## Dispatch Behavior

These are model-specific qualification adapters, imported explicitly rather
than registered as a new standalone operator. The existing WS1 dispatch and
three TP projection registry entries remain unchanged. CUDA entry checks fail
when the required strict extension is unavailable. The PyTorch fixture is an
explicit control, not an automatic backend fallback.

## Accuracy

Independent complete small-tensor comparisons use CPU FP32 autograd. FP32
uses `atol=rtol=1e-4`. BF16 uses the preceding TP contributions' tolerances:
forward `atol=0.05`, gradients `atol=0.1`, and `rtol=0.02`. Time values are fixed
scheduler metadata; precision checks cover the encoding output and all learned
MLP parameters, not a derivative with respect to the scheduler value.

BI is a separate raw-byte comparison of complete active forward values, input
and condition gradients, and every learned parameter gradient. It covers
repeat, row permutation, microbatch, per-request, padding and sequential CFG
layouts after restoring logical order.

For complete replicated inputs and complete upstream gradients, each rank
already has the full parameter gradient. Summing those replicas would multiply
it by `P`. For head-sharded Q/K, the adapter gathers per-row/head FP32 gain
contributions, restores `[request,token,global_head]` order, then uses the
existing ascending FP32 row fold. It does not round a rank-local partial gain
gradient before communication.

The source fixtures reuse the preceding QKV layouts: 1024×1024, 1664×928 and
1328×1328 images; text lengths 17/129/512/1024; mixed-length batch 2; two images
per prompt; and batched/sequential classifier-free guidance (CFG). Six request
workloads are tested through seven call-layout transformations. Values are
created locally from seed 386, without model weights or a text encoder.

## Performance Notes

Run from the repository root after following the target branch's CUDA CI
installation steps. This matches PyTorch `2.4.1+cu124` and Triton `3.0.0`.

```bash
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3.11 benchmarks/distributed/tp/benchmark_qwenimage_replicated_modules.py \
  --warmup 3 --samples 10 --output /tmp/replicated-tp1.json
for p in 2 4 8; do
  torchrun --standalone --nproc-per-node="$p" \
    benchmarks/distributed/tp/benchmark_qwenimage_replicated_modules.py \
    --warmup 3 --samples 10 --output "/tmp/replicated-tp${p}.json"
done
```

The default two workloads are a single 1024×1024 image with 17 text tokens and
the batch-2 1328×1328 CFG workload with text up to 1024. Use `--workloads` to
select other qualified cases. JSON records each module, active rows, seed,
dtype, source hashes, raw samples and per-rank allocated-memory statistics.
Timing uses eager synchronized wall time, including Python, canonical sorting,
gradient collection and communication. TP timings use the slowest rank.
The PyTorch control executes the full TP1 module boundary.

Replicated modules perform the full computation on every rank; increasing
TP does not divide their GEMM work. Q/K normalization additionally gathers
unreduced gain contributions to preserve its global FP32 summation order.

## Tests

```bash
python3.11 -m pytest tests/distributed/tp/test_qwenimage_tp_replicated_modules.py -q
```

An eight-GPU host exercises TP2/4/8 rather than skipping those cases. Tests
include frozen Linear/QK leaves, noncontiguous inputs and upstream gradients,
noncurrent-device Q/K execution, and a shared-condition cancellation witness.
A composed BF16 VJP probe connects replicated input/norm/modulation to the
existing TP QKV, attention-output and MLP adapters, followed by replicated
image output; it checks all parameter gradients and local shard ownership.

## Known Limitations

Qualification covers the RFC's Phase A pre-encoded inputs and first-order
module boundaries. The composed VJP probe uses a row-local linear combination
of Q, K and V; it does not implement attention, RoPE, VAE or a complete model
training step. All ranks must use matching logical calls. Q/K norm materializes
global per-head gradient contributions before the canonical fold.
