# Router auxiliary loss

Task 34 implements the [P3-B RFC](https://github.com/RL-Align/RL-Kernel/issues/516)
requirements for optional routing auxiliary loss: fixed denominators, atomic-free
local reduction, P4 cross-card integration, and overflow-denominator gradients.
The [Core #7441 reference](https://github.com/NVIDIA/Megatron-LM/pull/7441) rejects
the TE fused auxiliary-loss path containing float atomics under deterministic
mode. This implementation uses an explicit AITER companion profile instead.

## Frozen formula

The RFC does not specify a concrete loss formula or coefficient. The following
profile was explicitly confirmed for this task and is versioned as
`rlkernel.router_aux_loss.preload.postcapacity.v1`:

\[
L=\alpha\,256\sum_{e=0}^{255} f_e p_e,\qquad
f_e=\frac{\text{pre-capacity Top-6 selections of }e}{6N},\qquad
p_e=\frac{1}{N}\sum_{t=0}^{N-1}q_{t,e}.
\]

`N` is the number of **valid logical tokens in the complete routing domain**.
For each token, `q` directly gathers original pre-bias sqrt-softplus scores,
masks capacity overflow, and divides surviving scores by their ordered sum
clamped to `1e-20`. These probabilities are **not multiplied by route scale**.
They are never recovered by dividing scaled route weights, including when the
main-route scale is zero. The selection bias participates only in actual normal
Top-6 selection, not in `q` or its derivatives.

This is the agreed post-capacity profile, not a claim that the standard Core or
mLite auxiliary loss has this exact formula. In particular, it does not use an
all-256-expert softmax mean. The pre-capacity frequency `f` is discrete and has no
derivative. `p` retains the numerator and all surviving denominator derivatives.
Unselected/dropped logits receive zero; bias and expert IDs receive no gradient.

The task-32 capacity policy remains
`complete_domain.global_logical_token_id_ascending.v1`. Admission uses the complete
domain and ascending global logical token IDs; overflow is masked without
reordering or replacing the six selected slots. The auxiliary denominator stays
`N`, including tokens with all slots dropped. All-drop domains give loss and
dlogits zero. Empty domains define loss zero and empty gradients as a boundary
case, and are explicitly not acceptance samples.

## API and optional training integration

```python
from rl_engine.ops.moe.router_aux_loss import (
    AuxLossConfig, prepare_router_aux_context, router_aux_loss,
)

config = AuxLossConfig(enabled=True, alpha=0.03, version="aux.v1")
context = prepare_router_aux_context(
    plan, config=config, version="loss-domain.v1",
)
result = router_aux_loss(
    logits, logical_ids, selection_bias, plan,
    config=config, context=context, microbatch_size=17,
    absolute_layer=3, model_version="model.v1", bias_version="bias.v1",
    domain_version="domain.v1", plan_version="plan.v1",
)
(main_loss + result.loss).backward()
```

The task-32 `plan` must cover all valid domain tokens. Inputs are finite FP32
logits `[M,256]`, FP32 canonical-global-order bias `[256]` on the same ROCm GPU,
and I32/I64 logical token-instance IDs `[M]`. Logical IDs are not vocabulary IDs.
Optional `valid_mask` is boolean `[M]`: excluded rows do not enter normal routing,
`N`, frequencies or loss, and receive zero gradients. Padding may carry invalid
logit values and sentinel IDs because it is removed before routing. Layouts are
explicitly materialized after valid-row selection. Expert-column permutations
are interpreted through the prepared plan's global expert map.

`AuxLossConfig()` defaults to **disabled**. Disabled calls return `loss=None`,
empty routes and an execution record containing the disabled configuration;
they perform no router/loss GPU calculation or communication and add no autograd
branch. Enabling requires an explicit nonnegative finite FP32 `alpha`; positive
values underflowing FP32 are rejected. Alpha is rounded once, with requested and
actual values recorded. Zero alpha is supported as an explicitly enabled profile.

`RouterAuxLossResult` exposes scalar FP32 `loss`, local unscaled FP32
`probabilities [N_local,6]`, I64 selected global expert IDs, global I64
`counts [256]`, FP32 `masses [256]`, the normal route results for each compute
microbatch, and execution provenance. Counts and masses are diagnostic outputs;
the scalar's custom first-order VJP supplies the actual selected-weight gradient.
Normal route weights remain attached, so a caller may compose main/auxiliary
objectives and differentiate their sum. Tests exercise two optimizer updates of
a trainable-logit proxy; full framework parameter placement remains separate.

Each invocation must contain **all tokens owned by that rank in this loss
domain**. `microbatch_size` only chunks normal routing; all chunks are concatenated
and the global auxiliary objective is formed once. Calling this API separately
on fragments with independent denominators is rejected by ownership checks.
This makes compute M=1/16/17/32/65/257 a scheduling choice, not a redefinition of
the loss. Upstream activations may already be concatenated with an autograd graph.

## Fixed reduction and P4 boundary

The arithmetic order is
`global_logical_token_id_ascending.then_global_expert_id_ascending.v1`:

1. Produce actual normal learned selection, validating it against the immutable
   complete-domain plan. Recompute normal scale-one probabilities with the same
   AITER score/weight/VJP implementation; do not inject diagnostic saved IDs.
2. Use P4 `all_gather`/`all_gather_many` only for bit-preserving transport of
   token IDs, selected experts and probabilities. No floating sum occurs in
   RCCL or in a rank-dependent transport tree.
3. Restore ascending global logical token ID order. One AITER program per global
   expert traverses token blocks of 32, adding row 0 through 31 in order. It
   accumulates pre-capacity counts in I64 and post-capacity mass in FP32.
4. Divide by the declared `6N` and `N`, multiply `f_e*p_e`, sum experts 0 through
   255 in order, multiply by 256 and then alpha. Warps/stages/fusion are fixed;
   there are no floating atomics or M-dependent autotuning.

For multiple ranks, supply an existing
`rl_engine.distributed.transports.rccl.RCCLDeterministicCollective` and complete
disjoint logical ownership metadata:

```python
context = prepare_router_aux_context(
    plan, config=config, ownership=(rank0_logical_ids, rank1_logical_ids),
    collective=p4_collective, version="loss-domain.v1",
)
```

Ownership entries are Python integer lists/tuples, not vocabulary indices.
They must cover every valid domain token exactly once. Different local lengths
and empty ranks use padded transport, then exclude padding before reduction.
Config/domain/plan/ownership identity is checked by a tensor handshake on all
participating ranks. The caller owns collective lifetime, process-group domain,
and matching participation/call order; every rank, including empty ones, enters
forward and backward. Enabled/disabled configuration must be synchronized before
entering the optional branch.

The global loss is replicated, but backward returns only each rank's local token
gradient. It does **not** sum identical scalar-loss replicas or multiply by world
size. All ranks must supply identical scalar upstream scaling; a backward
handshake rejects different scaling. Framework loss scaling, shared-parameter
gradient averaging/accumulation and additional reductions remain caller-owned.
The local selected-weight VJP differentiates the capacity denominator before
scatter. Higher-order gradients are unsupported. Nonfinite upstreams and
overflowing FP32 losses/coefficients/gradients fail explicitly.

## Reproduction and evidence

Use AITER base `326cffa38c9eda16bef19375ba55334e4d3acd59` and apply companion
patches under `rl_engine/backends/rocm/moe/patches`, in order:
`router-sqrt-softplus.patch`, `stable-topk6.patch`, `learned-route.patch`,
`router-aux-loss.patch`. The three prerequisite patches/manifests are unchanged.
The task-34 manifest verifies four patch hashes and seven resulting AITER sources.
Disabled mode requires no patched AITER; enabled mode fails on missing/mismatched
source or launch identity without silently selecting a different backend.

```bash
export AITER_TRITON_ONLY=1
export PYTHONPATH=artifacts/task34/aiter
python -m tools.weights.prepare_learned_route_bias \
  --output artifacts/task34/0731-layer3-bias.pt
python -m pytest -q tests/ops/moe/test_router_aux_loss.py \
  tests/runtime/test_router_aux_loss_dispatch.py tests/runtime/test_kernel_registry.py
python -m tools.validation.operators.check_router_aux_loss \
  --bias-fixture artifacts/task34/0731-layer3-bias.pt \
  --raw-output artifacts/task34/raw-a.pt \
  --output reports/experiments/router-aux-loss/task34-mi300x.json
python -m tools.validation.distributed.check_router_aux_loss --transport rccl \
  --output reports/experiments/router-aux-loss/two-mi300x-rccl.json
python -m tools.validation.distributed.check_router_aux_loss --transport p4-auto \
  --output reports/experiments/router-aux-loss/two-mi300x-p4.json
```

The single-device replay uses real fixed Flash-0731 layer-3 bias with synthetic
FP32 logits. The bias revision is
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`; the loader independently verifies raw
SHA256 `34d941394db670f6de58ccb9bb781158d12407cc27613f8ae90756e2623fb236`.
No full checkpoint shard or full-model activation trace is used.

The independent CPU oracle performs normal selection and expert-centric capacity
admission, then builds a FP64 autograd graph for the agreed formula. Its
conservative -36 tail is a mathematical reference, distinct from production's
task-29 -16 cutoff. Checks compare discrete decisions first, then loss
`rtol=3e-6, atol=2e-8` and dlogits `rtol=5e-4, atol=2e-8`. Separate finite
differences recompute normal selection/admission at each perturbation. Negative
controls detect post-capacity frequency and detached overflow denominators.
Exact payload checks separately cover repetition, logical/column permutations,
compute microbatches, inference/training and fresh processes with different JIT
caches, including signed zero.

The two-card reports distinguish automatic P4 HIP IPC from an explicit P4 RCCL
tensor transport validation profile. Both compare the replicated global scalar
and local gradients byte-for-byte to the same single-device complete domain.
The harness's transport-only subclass disables IPC to exercise existing P4 RCCL
all-gather; it adds no communication or reduction kernel. Cases include uneven
lengths, empty rank, overflow/all drop, empty domain, and no capacity bound.

## Costs and support limits

Exact semantic registration is `rlkernel.router_aux_loss.preload.postcapacity.v1`,
training only, FP32 logit input, ROCm TP1, Graph disabled. The enabled profile is
for learned absolute layers 3–42; Hash-layer auxiliary objectives and other loss
formulas are unsupported. MI300X gfx942 is measured; gfx950 is admitted by the
underlying route but unmeasured. No cross-architecture byte guarantee is made.

This correctness profile gathers O(N×6) probabilities/IDs on every rank, maintains
the complete routing plan and sorts token metadata on the host. It validates
content hashes, synchronizes with the host and currently computes main-scale and
scale-one normal routes separately. These costs are material. Two-stream tests
check ownership and workspace isolation, not overlapping GPU execution. Fifty
repeated calls show no live-allocation growth. Detailed kernel/config/source and
loss-normalizer metadata are recorded, including disabled configuration.

Reports under `reports/experiments/router-aux-loss` separate arithmetic evidence
from timing and transport scope. This task adds the optional local training API
and P4 loss-gradient boundary, not complete Megatron/vLLM/verl/vime wiring, full
distributed model autograd, MI355X/multi-node acceptance or full-model raw-logprob
signoff. Downloaded sources, fixtures, raw outputs and logs remain ignored.
