# Learned router weights and capacity

Task 32 implements the [P3-B RFC](https://github.com/RL-Align/RL-Kernel/issues/516)
contract: **bias affects selection only; weights use the original scores; capacity
masking precedes normalization in the retained Top-6 slot order.** This is the
Flash-0731 learned-router path for absolute layers 3–42, with 256 experts.

## Arithmetic and capacity contract

For each row, the pinned AITER task-29 sqrt-softplus produces FP32 scores.
Selection uses `FP32(score + bias[global_expert_id])`. The task-31 AITER streaming
Top-6 orders these values descending, breaking ties by smaller global expert ID.
Its six slots remain in that order through admission, weighting and dispatch.

Weights **directly gather pre-bias scores**. Subtracting bias from a rounded
selection score loses small scores when the bias is large, so that expression is
not used. Bias has no gradient through discrete selection or through weights.

The RFC leaves capacity competition priority unspecified. The agreed, versioned
policy is `complete_domain.global_logical_token_id_ascending.v1`:

1. Declare all unique, nonnegative global logical token IDs in one routing domain
   for one absolute layer and model version. These are token-instance identities,
   not vocabulary IDs or local batch offsets.
2. For each expert, retain selected tokens in ascending global logical token ID
   order up to its declared capacity. Accepted dispatch positions start at zero;
   rejected slots have position `-1`.
3. Mask overflowing slots without changing the six selected IDs, reordering slots,
   replacing dropped experts, or compacting survivors.
4. Sum surviving original FP32 scores in slots 0, 1, …, 5. Set
   `d = max(sum, FP32(1e-20))`, then compute each FP32 weight as
   `(surviving_score / d) * routed_scaling_factor`.

The scale defaults to 1.5 and is rounded to FP32 once when preparing the plan.
All-dropped rows and empty domains return zero weights, zero dlogits and no
invented uniform distribution. Zero scale is supported. Negative, nonfinite,
overflowing, or positive-but-underflowing scales are rejected.

Capacity may be an integer, an I32/I64 `[256]` vector of nonnegative counts, or
`None` for unbounded capacity. Planning snapshots these values into immutable
tuples. Planning must see the complete declared domain; independently assigning
the full capacity to each microbatch or rank is incorrect.

## API and state ownership

```python
from rl_engine.ops.moe.learned_route import (
    prepare_capacity_domain, prepare_learned_route_plan, learned_route,
)

domain = prepare_capacity_domain(
    all_logical_ids, 128, absolute_layer=3,
    model_version="weights.v1", domain_version="domain.v1",
)
plan = prepare_learned_route_plan(
    all_logits, all_logical_ids, selection_bias, domain,
    bias_version="bias.v1", plan_version="plan.v1",
)
result = learned_route(
    local_logits, local_logical_ids, selection_bias, plan,
    absolute_layer=3, model_version="weights.v1", bias_version="bias.v1",
    domain_version="domain.v1", plan_version="plan.v1",
)
result.weights.backward(upstream_weights)
```

`all_logits` and `local_logits` are finite FP32 `[M,256]` tensors on the same ROCm
GPU as the finite FP32 canonical-global-order bias `[256]`. Positive tensor
strides are supported by explicit contiguous materialization. Logical IDs are
I32/I64 `[M]` metadata. An optional I32/I64 global expert map `[256]` must cover
0–255 exactly once; it describes the input columns and is saved in the plan.

Outputs are FP32 `weights [M,6]`, I64 `expert_ids [M,6]`, I64 input
`column_indices [M,6]`, boolean `surviving_mask [M,6]`, I64
`dispatch_positions [M,6]`, FP32 `denominators [M]`, and an execution record.
Expert IDs and capacity decisions are discrete and nondifferentiable.

The immutable plan records domain/model/bias/plan versions, absolute layer,
canonical logit-row hashes, bias hash, selected global IDs, masks and dispatch
positions. Row permutations and expert-column permutations produce the same
semantic plan identity. Planning uses a deterministic CPU admission pass after
GPU selection. It does not maintain mutable runtime capacity counters.

Each forward **recomputes normal selection** using current logits and bias and
checks it against the prepared plan. It never injects saved IDs as fixed-route
selection. Raw content hashes also reject stale logits or bias changed through
`.data`. Layer/version mismatches and modified plan state fail before routing.
Recomputation and any microbatch subset of the same complete-domain plan produce
the same per-token bytes. Changing the actual competing domain or capacity can
legitimately change admission and requires a new plan.

## Training and numerical checks

The first-order local VJP differentiates both the surviving numerator and all
surviving denominator terms. Below the denominator clamp it differentiates only
the numerator. Dropped and unselected logits get zero; selection bias gets no
gradient. Six distinct columns are written without atomics or cross-row sums.
The negative tail combines normalization and sqrt-softplus derivatives in log
space to retain representable final gradients even when intermediate scores
underflow. Higher-order differentiation is unsupported. Nonfinite upstreams and
FP32-overflowing outputs or gradients raise errors.

The independent CPU oracle selects normal Top-6 and groups tokens **by expert**
to decide admission; production instead traverses tokens with expert counters.
Discrete IDs, masks and positions are compared exactly. The mathematical FP64
weight/VJP oracle uses a conservative `-36` tail cutoff; the production task-29
FP32 score transform uses `-16`. This is an explicit mathematical oracle, not a
bitwise emulator of production rounding. Regular-domain checks use weight
`rtol=2e-6, atol=1e-8` and VJP `rtol=3e-4, atol=2e-7`; extreme-tail tests account
for FP32 subnormal-score error amplified by the fixed denominator clamp.
Determinism and microbatch checks compare actual FP32 payload bits independently
of those mathematical tolerances.

Negative controls cover bias incorrectly added to weights, cancellation from
subtracting rounded bias, and detached overflow denominators. Tests also cover
partial/all overflow, zero/unbounded/vector capacities, equal scores, row and
column permutations, noncontiguous layouts, empty inputs, stale state and Graph
rejection. Shared-plan tests check two ROCm streams (without claiming overlap)
and unchanged live GPU allocation across 50 calls. Two-MI300X RCCL checks assemble the complete domain in different rank
orders and verify matching plan identities, normal selection, weights and local
VJPs, including an empty rank and an empty global domain.

## Backend and reproduction

The implementation uses AITER base
`326cffa38c9eda16bef19375ba55334e4d3acd59` plus three companion patches, in order:
`router-sqrt-softplus.patch`, `stable-topk6.patch`, `learned-route.patch` under
`rl_engine/backends/rocm/moe/patches`. The first two are identical to the task-29
and task-31 profiles. Task 32 adds its own module, reusing both APIs without
changing their source identities. Its weighting/VJP fixes normalization order
and directly gathers raw scores, replacing the existing Top-K bias-subtraction
weight path. The manifest validates all three patch hashes and all six final
source files; launch settings are four warps, one stage, FP fusion disabled.

Use an isolated checkout of that AITER revision and apply the patches with
`git apply` in the stated order. The installed AITER package is not modified.
With the patched source at `artifacts/task32/aiter`:

```bash
export AITER_TRITON_ONLY=1
export PYTHONPATH=artifacts/task32/aiter
python -m tools.weights.prepare_learned_route_bias \
  --output artifacts/task32/0731-layer3-bias.pt
python -m pytest -q tests/ops/moe/test_learned_route.py \
  tests/runtime/test_learned_route_dispatch.py \
  tests/validation/operators/test_learned_route_fixture.py \
  tests/runtime/test_kernel_registry.py
python -m tools.validation.operators.check_learned_route \
  --bias-fixture artifacts/task32/0731-layer3-bias.pt \
  --raw-output artifacts/task32/raw-a.pt \
  --output reports/experiments/learned-route/task32-mi300x.json
python -m tools.validation.distributed.check_learned_route_capacity \
  --output reports/experiments/learned-route/two-mi300x-capacity.json
```

The bounded HTTP-range downloader fetches only the safetensors header and the
1-KiB `layers.3.ffn.gate.bias` tensor from Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Both downloader and replay loader
check its frozen raw SHA256
`34d941394db670f6de58ccb9bb781158d12407cc27613f8ae90756e2623fb236`.
Replay uses this real bias with **synthetic gate logits**, not full-model
activations. Weights and downloaded AITER sources remain ignored artifacts.

## Supported scope and performance

The exact semantic registration is
`rlkernel.learned_route.fp32.bias_selection.capacity_mask.v1`, with training and
rollout targets, TP1 and Graph disabled. Source/API/launch mismatch, missing
patches or unsupported inputs fail without falling back to another operator.
MI300X gfx942 is measured; gfx950 is accepted but unmeasured here. No
cross-architecture bitwise guarantee is made.

Committed reports under `reports/experiments/learned-route` contain replay
hashes, execution identities, mathematical errors, fresh-process comparison and
timing. Reported AITER timings separate normal selection plus weighting from
VJP; plan preparation and the complete RLK adapter are timed separately. The
adapter deliberately includes host admission, metadata validation, content hashes
and copies. These costs are material; kernel-only measurements are not
end-to-end throughput claims. GPU events can include host launch gaps.

This task supplies local learned-weight autograd and a capacity-plan interface.
Framework wiring, task-33 route-replay integration, task-34 auxiliary losses,
distributed autograd, P4 expert communication, full-model logprob validation and
MI355X acceptance remain separate work.
