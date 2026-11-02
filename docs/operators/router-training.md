# Router backward and result records

Task 33 follows [P3-B RFC #516](https://github.com/RL-Align/RL-Kernel/issues/516):
**discrete expert IDs have no derivative; only selected weight branches are
differentiated. Compare normal routing independently, comparing discrete choices
before numbers. Fixed-route replay is diagnostic only.**

## Scope and reuse

Normal Hash routing reuses task 30, including absolute-layer table selection,
table checksums, original slot order, optional normalization and local AITER VJP.
Normal learned routing reuses task 32: task-29 sqrt-softplus, task-31 stable
Top-6, selection-only bias, capacity mask, ordered post-mask normalization and
its complete denominator derivative. No auxiliary loss is introduced here.

The existing AITER weight derivatives scatter into six distinct logit columns
without atomics. Dropped and unselected columns receive zero. Selection bias,
logical/vocabulary IDs, capacity masks and dispatch positions receive no gradient.
The task-32 capacity policy remains
`complete_domain.global_logical_token_id_ascending.v1`: a complete logical domain
admits tokens by global logical token ID, masks overflow, then normalizes in
the original six-slot order. Recompute uses the same immutable plan, while a
changed model or true competing domain requires a new versioned plan.

Task 33 adds normal-route result/VJP recording, exact comparisons, diagnostic
replay and a local training bridge through a frozen BF16 Gate. The replay
companion adds **no new GPU numerical kernel**. It reuses the same AITER forward
weight and backward implementations as normal routing. Hash replay materializes
the saved six IDs as one table row per token; learned replay computes scores and
weights at saved columns using the saved mask. It does not run normal selection.

## Public route and record APIs

```python
from rl_engine.ops.moe.router_training import (
    normal_learned_route, normal_hash_route, record_route_vjp,
    save_route_record, load_route_record, save_vjp_record, load_vjp_record,
    compare_route_records, compare_vjp_records, diagnostic_fixed_route,
)

run = normal_learned_route(
    logits, logical_ids, bias, plan, record_version="request.v1",
    absolute_layer=3, model_version="model.v1", bias_version="bias.v1",
    domain_version="domain.v1", plan_version="plan.v1", rank=0, world_size=1,
)
save_route_record(run.record, "artifacts/route.json")

# Ordinary training follows the existing autograd graph into upstream parameters.
run.weights.backward(weight_gradient)
```

`logits` is finite FP32 `[M,256]` on ROCm; `bias` is finite canonical-global-order
FP32 `[256]` on the same device. Positive strides are accepted through explicit
materialization. Logical IDs are unique nonnegative I32/I64 token-instance IDs
`[M]`, not vocabulary IDs. The learned plan supplies its complete global expert
column map and capacity/domain/model/bias/plan versions. Learned layers are
absolute layers 3–42.

`normal_hash_route(logits, logical_ids, vocabulary_ids, table, ...)` also takes
I32/I64 vocabulary IDs `[M]` on the logit device. Vocabulary IDs may repeat;
logical IDs remain unique. The prepared task-30 table supplies the absolute
layer 0–2 and model/table version; the call requires `absolute_layer`,
`model_version`, `table_version`, and `record_version`. Optional
`renormalize=True` and `routed_scaling_factor=1.5` retain the task-30 contract.
Hash routes do not acquire learned-route capacity handling.

The `RecordedRoute` result contains attached `logits`, differentiable FP32
`weights [M,6]`, nondifferentiable I64 `expert_ids [M,6]`, owned immutable
`RouteRecord`, and the input row's logical-ID tuple. The record contains:

- Absolute layer, route kind/mode, record/model/route versions, rank/world size.
- Canonical logical-token rows, global expert IDs, input columns, slot order,
  surviving masks and dispatch positions.
- Per-row canonical logit hashes and original weight FP32 payload bits.
- Scale, normalization mode, actual backend/kernel/configuration/source hashes,
  and the normal source record identity for diagnostic replays.

Records own canonical JSON strings, not mutable caller tensors or dictionaries.
Reading `.payload()` returns a fresh checked copy. JSON loading verifies checksum,
schema, types, shapes, expert-column correspondence, distinct IDs, mask/position
consistency, finite payloads and versions before replay can launch a kernel.
Checksums detect corruption; they do not authenticate an untrusted producer.
Forward/backward guards also check actual saved tensor bytes, catching `.data`
mutation of logits, bias, selected IDs, masks, weights and protected Gate inputs.

To explicitly observe the local selected-weight VJP:

```python
dlogits, vjp_record = record_route_vjp(
    run, weight_gradient, loss_version="global-token-denominator.v1",
    retain_graph=True,
)
save_vjp_record(vjp_record, "artifacts/vjp.json")
```

This helper uses `torch.autograd.grad(weights, logits)` and records canonical
upstream/dlogits FP32 bits plus its forward-record identity. It does not itself
accumulate gradients into earlier parameters; use ordinary `weights.backward()`
for training. Choose `retain_graph=True` if both observations are needed.
`weight_gradient` must be finite FP32 `[M,6]` on the route device. Callers own
loss normalization and its nonempty version label. Higher-order gradients are
unsupported. FP32 output/gradient overflow fails explicitly.

## Comparison and replay boundaries

`compare_route_records(actual, expected)` first requires compatible versions,
layer, rank, backend/source/configuration identities, loss-independent scale/mode
and logical domain. It then compares
global expert IDs, surviving mask and dispatch positions before logit identity
and exact weight bits. The first difference includes logical token, slot and
field. Expert-column permutations are interpreted through the global map.

`compare_vjp_records` first performs that forward comparison and checks the loss
version, then compares upstream and full dlogits payload bits in canonical global
expert order. Gradient differences identify the logical token and slot or global
expert. Signed zero and one-ULP changes are observable. Empty inputs are boundary
tests and never earn normal acceptance.

```python
saved = load_route_record("artifacts/route.json")
data = saved.payload()
debug = diagnostic_fixed_route(
    replay_logits, replay_logical_ids, saved,
    absolute_layer=data["absolute_layer"], versions=data["versions"],
)
diagnosis = compare_route_records(debug.record, saved, require_normal=False)
assert not diagnosis["acceptance"]
```

Replay defaults to the recorded logits' exact canonical identities, supports
logical-row reordering/subsets and preserves the recorded expert-column map.
Changed logits require explicit `allow_changed_logits=True` and remain
diagnostic. Model/layer/route version mismatches fail. Replaying a diagnostic
record as another source is rejected. Records marked `diagnostic_fixed` are
rejected by normal acceptance comparison even if every output bit matches.
Normal calls always execute actual Hash lookup or actual biased stable Top-6;
saved IDs never establish normal-routing correctness.

## Frozen Gate training bridge

`rl_engine.models.deepseek_v4.router_training.DeepSeekV4RouterTraining` accepts a
loaded **frozen** BF16 Gate `Parameter [256,K]`. Learned layers additionally bind
frozen FP32 selection bias and its version; Hash layers bind their prepared
table. Its `prepare_plan(hidden, logical_ids, domain, plan_version=...)` prepares
the complete learned domain. `forward(hidden, logical_ids, record_version=...,
plan=...)` returns `RecordedRoute`. Hash calls supply `vocabulary_ids` instead.
Hidden states are BF16 `[M,K]`; Gate logits remain FP32. Gate provenance is
embedded in the route record.

The bridge uses task-28 fixed AITER BF16→FP32 forward GEMM and its existing RLK
fixed FP32 backward GEMM with BF16 input-gradient conversion. It rejects trainable
base Gate or selection bias. Gradients pass through the frozen Gate to externally
owned upstream LoRA parameters. Tests explicitly use a **synthetic upstream
residual LoRA**, BF16 rank 4, scale 0.5, K=32 and K=4096, and two SGD updates.
That verifies this boundary without adding a Gate LoRA attachment or claiming a
full framework LoRA topology. Rebind synchronized model/bias/table versions and
rebuild capacity plans after relevant updates. Frozen parameter content changes
and old plans are rejected.

Per-token routing VJPs are invariant across compute microbatches when the complete
domain and upstream values remain fixed. Shared-parameter accumulation order and
loss denominator remain framework-owned; no bitwise guarantee is claimed for
arbitrary native LoRA GEMM or framework accumulation orders.

## Backend setup and validation

Use AITER `326cffa38c9eda16bef19375ba55334e4d3acd59` and apply companion patches
from `rl_engine/backends/rocm/moe/patches` in this order:

1. `router-sqrt-softplus.patch` (task 29).
2. `hash-route.patch` (task 30).
3. `stable-topk6.patch` (task 31).
4. `learned-route.patch` (task 32).
5. `router-training.patch` (task 33).

The four prerequisite patches/manifests are unchanged. The task-33 manifest
verifies five patch hashes and eleven AITER sources, including both Gate GEMM
sources. Source archives without AITER's generated `_version` file report
`unpackaged_source; see source_sha256`; no installed wheel's unrelated version
is substituted. API/source mismatch fails without backend fallback.

With that checkout in `artifacts/task33/aiter`:

```bash
export AITER_TRITON_ONLY=1
export PYTHONPATH=artifacts/task33/aiter
python -m tools.weights.prepare_learned_route_bias \
  --output artifacts/task33/0731-layer3-bias.pt
python -m tools.weights.prepare_hash_route_tables --output-dir artifacts/task33/tables
python -m pytest -q tests/ops/moe/test_router_training.py \
  tests/ops/moe/test_router_records.py tests/runtime/test_router_training_dispatch.py \
  tests/runtime/test_kernel_registry.py
python -m tools.validation.operators.check_router_training \
  --bias-fixture artifacts/task33/0731-layer3-bias.pt \
  --hash-fixture-dir artifacts/task33/tables --record-dir artifacts/task33/records-a \
  --raw-output artifacts/task33/raw-a.pt \
  --output reports/experiments/router-training/task33-mi300x.json
python -m tools.validation.distributed.check_router_training \
  --output reports/experiments/router-training/two-mi300x-records.json
```

The fixtures pin Flash-0731 revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
and raw tensor checksums. Download uses bounded HTTP ranges and never a whole
shard. Replay logits are synthetic FP32; this is not a full-model activation
trace. The CPU mathematical oracle uses FP64 with a conservative -36 tail,
distinct from production's task-29 -16 cutoff. Discrete choices are exact;
mathematical weights use `rtol=2e-6, atol=1e-8` and dlogits use
`rtol=3e-4, atol=2e-7`. Independent normal-route finite differences additionally
check overflow denominator derivatives. BF16 LoRA checks separately allow BF16
gradient rounding (`rtol=0.03, atol=2e-5`). Byte checks do not use these tolerances.

## Limits and costs

Exact semantic registration is
`rlkernel.router_training.selected_weights.records.v1` for FP32 logit routes,
training/rollout, ROCm TP1, Graph disabled. MI300X gfx942 is measured; gfx950 is
admitted by the underlying paths but unmeasured. No cross-architecture byte
guarantee or higher-order autograd support is claimed.

Records synchronize with the host for owned CPU snapshots, content hashes and
JSON validation. Full VJP records store O(M×256) FP32 payload bits; callers must
choose appropriate diagnostic boundaries and file lifecycle. File I/O is
explicit. Two-stream tests verify ownership/isolation, not GPU overlap. Timing
includes these costs, without a speedup or full-model throughput claim. Downloaded
sources, fixtures, raw payloads, detailed records and logs stay in ignored
`artifacts/task33`; committed reports are under `reports/experiments/router-training`.

RCCL evidence covers complete-domain transport and per-rank local records/VJPs,
including empty ranks. It does not validate distributed autograd, expert dispatch,
Megatron/vLLM/verl/vime framework wiring, auxiliary loss, full-model logprobs or
the RFC's final multi-node/MI355X acceptance.
