# Native runtime capture extensions

These APIs extend the portable evidence path to native checkpoint construction,
TP/CP ownership, paged KV state, graph replay, MoE routing, kernel debug outputs,
and training updates. All bundles use the existing `rlk debug /path/to/bundle`
offline comparator. They do **not** enable an RL-Kernel replacement.

## Supported contracts, not universal automatic inference

| Requested capability | Implemented path | Required native information / limit |
| --- | --- | --- |
| Construct and load an unfamiliar Megatron model | `load_megatron_model` calls the application's provider, then the native distributed-checkpoint loader with access-integrity and strict-key checks | A trusted provider, native config/layer specs, initialized process groups and a native distributed **model** checkpoint. No model-name adapter; no automatic HF-to-Megatron architecture/weight conversion. |
| TP/CP > 1 mapping | `megatron_weight_shards` derives global parameter ownership from native checkpoint metadata; `parallel_tensor` generates contiguous/strided TP and contiguous/zigzag CP activation ownership | Actual group ranks/sizes and declared activation axes/layout. This is not automatic semantic matching of arbitrary unfamiliar activations, nor an extension of the TP1 `rlk discover` launcher. |
| Paged decode / KV cache | `paged_kv` canonicalizes live logical slots and head shards; `DecodeCapture` observes arguments, query rows and cache before/after one original decode call | Actual block tables, lengths, stable sample IDs and lossless `[page, slot, head, dim]` views. Packed/quantized layouts need a native adapter. Empty-cache prefill is not this decode contract. |
| Graph capture | `GraphCapture` stages preallocated device buffers and execution/order counters in a CUDA/HIP graph; serialization happens after each replay | Native code calls `stage` inside the captured graph. Each declared slot executes once, in declared order. Merely installing Python hooks during warmup does not qualify. |
| MoE dispatch | `moe_dispatch` captures real router weights, global expert IDs, drop masks and permutation; EP shards merge by logical route | Full logical router domain and local source-token/top-k-slot maps. Padding must be explicitly removed by the runtime's validity mask. This does not guess all-to-all ownership or inspect an expert's hidden operations. |
| Fused kernel internals | `FusedCapture` verifies every declared intermediate exported by one instrumented native execution | The kernel must actually export debug buffers/probes. Arbitrary opaque kernels remain unobservable. A decomposed reference implementation is **not** evidence of native intermediates. |
| Backward / optimizer update | `TrainingCapture` observes native autograd, accumulation, parameter gradients, optimizer inputs/outputs, weights, buffers and exposed optimizer state | A PyTorch optimizer with step hooks, or explicit state/gradient readers plus an optimizer adapter. Hidden backward kernels, scaler/scheduler state and arbitrary distributed optimizer internals are not automatically inspected. |

These are opt-in **diagnostic** paths. Copies, device counters and synchronization
may change memory use, graph scheduling and performance. They must not be used as
unqualified production-performance measurements or as proof about an unobserved
uninstrumented kernel.

## Native Megatron provider and checkpoint

The native application initializes its distributed groups and supplies its usual
provider. No model class is selected from a checkpoint name:

```python
from rl_engine.alignment.debug import load_megatron_model, megatron_weight_shards

model = load_megatron_model(
    native_model_provider,
    "/checkpoints/model/iter_0001000",
    provider_kwargs={"pre_process": True, "post_process": True},
    model_key="model",  # omit for a bare sharded model state dict
)
weights = lambda: megatron_weight_shards(model)
```

Loading is strict: extra optimizer/training envelope keys, missing weights,
incompatible shapes and a provider without `sharded_state_dict()` are not ignored.
Native `allow_shape_mismatch` loading is disabled on copied descriptors, including
factory-built shards: strict key checks alone would still permit padding or
truncation. Provider metadata and native factory merge functions are preserved.
Use the matching native model checkpoint layout. Provider code is supplied by
the application, never imported/executed from untrusted checkpoint metadata.
Virtual pipeline model lists need an application provider/adapter; this helper
returns one native module and does not silently unwrap distributed wrappers.

Weight inventory reads `ShardedTensor` global keys, offsets and prepended layer
axes. Native `ShardedTensorFactory` transformations are expanded using their own
build contract. Inventory compares the complete declared global storage shape,
including any padded vocabulary rows; a native `allow_shape_mismatch` loading
flag alone is not evidence of actual padding. Logical unpadding, flattened
optimizer/DP storage and opaque extra state require an explicit mapping.
Explicit native `ShardedObject(data=None)` and TE's empty one-dimensional
`uint8` no-quantizer-state buffer are allowed. Nonempty opaque extra state such
as FP8 histories is not silently discarded or unpickled.

## TP/CP evidence and automatic ownership reconstruction

`TensorShard(value, global_shape, indices)` specifies the global indices for
each local axis. `ShardSet` groups several fragments of one global parameter.
The comparator reconstructs global tensors, checks overlapping replicas bitwise
and rejects any coverage gap, shape mismatch or conflicting replica. Global
tensors may be compared with differently partitioned shards. Reconstruction is
offline: complete weight/optimizer inventories can require multiple model-sized
CPU copies and substantial disk space. Scope diagnostic runs accordingly; these
interfaces are not a streaming large-checkpoint comparison service.

For a declared `[token, hidden]` native activation:

```python
from rl_engine.alignment.debug import parallel_tensor

def logical_activation(value):
    return parallel_tensor(
        value,
        tp_rank=tp_group_rank, tp_size=tp_size, tp_axis=1, tp_stride=1,
        cp_rank=cp_group_rank, cp_size=cp_size, cp_axis=0, cp_layout="zigzag",
    )
```

Use this function in `ModuleBoundary` selectors. Provide the real subgroup rank,
not `global_rank % size`. A replicated axis uses size one in this view; arbitrary
sequence-parallel/uneven layouts can instead use explicit `TensorShard` indices.
Shape alone is never enough to choose TP/CP ownership.

Use `prepare_auto(..., sharded_weights=True, ranks=expected_workers)` and
`AutoCapture(..., sharded_weights=weights, rank=worker_rank)` for native sharded
weight inventories. The capture checks local weights for mutation and records
both initial/final shards; the comparator verifies global cross-runtime equality.
Do not pass `weights=` simultaneously. Each side must expose matching canonical
weight keys and semantics; storage ownership is not a parameter rename proof.

## Paged decode

Declare `DecodeCapture.boundaries()` using `EventCapture.prepare`, then wrap the
actual decode callable:

```python
from rl_engine.alignment.debug import DecodeCapture, paged_kv

def cache_view():
    return paged_kv(
        key_pages, value_pages, actual_block_table, actual_sequence_lengths,
        sample_ids=stable_sample_ids,
        head_offset=first_local_kv_head, total_heads=global_kv_heads,
    )

observer = DecodeCapture(event_capture, cache_view)
result = observer.run(
    native_decode,
    input_tokens,
    query_rows=actual_sample_position_pairs,
)
```

Only active slots enter the evidence. Different physical page allocation and
uninitialized unused pages are allowed; different logical cache values, sequence
lengths, sample ownership or head dimensions are not. Both sides must execute the
same declared decode schedule. This API does not equate a full prefill with an
arbitrarily reordered chunked/decode schedule.

Default input/output recording preserves all tensor/scalar leaves. An opaque
framework object requires an explicit `inputs(args, kwargs)` or `outputs(result)`
selector. Such a selector narrows the evidence and must include relevant masks,
positions, scheduler state and cache ownership rather than silently dropping them.

## Native CUDA/HIP graph replay

Declare slots outside capture; stage the real values inside capture:

```python
from rl_engine.alignment.debug import GraphCapture

observer = GraphCapture(event_capture)
observer.declare("attention.input", static_input)
observer.declare("attention.output", output_template, inputs=("attention.input",))

# Perform normal framework warmup on the required stream before graph capture.
with torch.cuda.graph(native_graph):
    observer.stage("attention.input", static_input)
    static_output = native_attention(static_input)
    observer.stage("attention.output", static_output)

# Update static_input using the normal native input-copy path.
observer.replay(native_graph, frame="decode:0")
```

The graph contains only device-to-device probe copies and device counters, not
`torch.save`, fingerprints, CPU tensor transfers or `.item()` calls. Counters are
reset before each diagnostic replay and checked afterward, including stage order.
Missing, repeated and stale probes invalidate the session even if the caller
catches the exception. Each replay is serialized before the next replay overwrites
its buffers. Dynamic scalar state must be stored in tensors; changing a slot's
shape, dtype, device or static metadata requires a new capture.

Graph and eager events can share a bundle when their declared logical order and
frame IDs match. Do **not** wrap the graph-capture context in eager `AutoCapture`.
To avoid editing native module forwards, use
`with observer.observe(model, bindings):` around warmup and graph construction.
It accepts the existing `ModuleBoundary` mappings and stages the predeclared
`role.input` / `role.output` slots through temporary module hooks. Selectors must
be graph-safe. Removing the Python hooks after construction does not remove
their captured device-copy operations from the graph.
Native multistream dependencies remain the graph owner's responsibility; a single
observer is single-device and not a concurrent-replay service.

## MoE dispatch

`moe_dispatch(...)` is suitable for a native dispatch output selector. Pass the
actual dispatched tensor, `expert_indices`, `routing_weights`, `dropped`,
`source_tokens`, `source_slots`, stable `token_rows` and `num_experts`.
It does not recalculate top-k, permutation, weights or expert outputs.

Rows are compared by `(sample, token, top-k slot, global expert ID)`, not all-to-all
arrival order. EP workers may own disjoint subsets, including no local routes.
Every non-dropped global route must be observed; overlapping replicas must agree.
All-dropped routing is represented explicitly. Expert-local feature sharding and
MoE combine/reduction internals require additional native boundary mappings.

## Fused debug probes

Prepare `FusedCapture.boundaries("fused", ["normalized", "projected"])`, then:

```python
from rl_engine.alignment.debug import FusedCapture

observer = FusedCapture(
    event_capture, "fused", ["normalized", "projected"],
    kernel_id=actual_native_kernel_id,
    probe_contract="rmsnorm-gemm-debug-v1",
)

def execute(emit):
    # This entry point must be implemented by the real kernel/runtime.
    output, debug_norm, debug_gemm = native_fused_with_debug_outputs(x, weight)
    emit("normalized", debug_norm)
    emit("projected", debug_gemm)
    return output

output = observer.run(execute, inputs={"x": x, "weight": weight})
```

The declared stage order is part of the supplied native computation contract.
Missing, duplicate, unknown or out-of-order probes fail closed. Different native
kernel IDs are retained as routes; matching shared probe semantics are required.
This is a verified instrumentation protocol, **not** a universal fused-kernel
decompiler. For in-graph debug outputs, stage preallocated debug buffers with
`GraphCapture` instead of calling this eager serializer.

## Backward and optimizer updates

Two independent diagnostic training arms use the existing side names `training`
and `rollout`; the latter label does not imply a rollout engine performs backward.

```python
from rl_engine.alignment.debug import TrainingCapture

TrainingCapture.prepare(bundle, scope="native optimizer diagnostic replay")
with TrainingCapture(bundle, side, identity, model, optimizer) as observer:
    for microbatch in batches:
        loss = native_forward_loss(microbatch)
        observer.backward(loss)
    # Keep the application's original clipping/unscale/step order.
    optimizer.step()
```

Call `prepare` once, not on every rank/side. Keep actual input, weight and runtime
identities independent on each side. `backward` delegates to native autograd once;
it neither clears gradients nor applies an optimizer update. Normal optimizer
pre/post hooks capture updates, including overridden steps calling `super().step()`.
Hooks are removed on all exits. Errors, missing updates and unfinished
accumulation invalidate comparison rather than becoming an equality result.
Backward output input-evidence remains **unknown**: equal loss values and
incoming gradients do not prove equality of all autograd saved forward tensors.

Default state capture includes model state, canonical parameter-group membership,
optimizer options and per-parameter optimizer state. Custom master-parameter or
Megatron optimizers require native state/gradient readers and appropriate step
hooks; local shards can be returned as `TensorShard` values. GradScaler-skipped
steps, closure-only backwards not routed through `observer.backward`, scheduler
state, concurrent updates and pipeline execution need explicit event adapters.

## Validation

`tests/test_debug_native_runtime.py` covers CPU protocols, TP2/CP2 reconstruction,
strided/zigzag layout, sharded weight verification, paged cache mutation, EP
routing, stale/repeated/missing graph probes, incomplete fused probes, gradient
accumulation, clipping, SGD/AdamW updates and fault localization. Mocked Megatron
tests verify provider/loading contracts only; they are not live framework tests.

The same file contains a **real** CUDA/HIP graph capture/replay test, gated on
device availability. It must run on the target device before claiming GPU graph
acceptance. CPU fake-replay tests do not substitute for it.

These additions do not retroactively expand any previously recorded Qwen3
Megatron/vLLM device acceptance. Distributed collectives, production paged
schedulers, MoE kernels and other instrumented fused kernels still need their own
target-stack acceptance. The bounded native device acceptance below is separate.

### MI300X native acceptance — 2026-10-26

Validated on one isolated **AMD Instinct MI300X VF** (`gfx942`), using PyTorch
`2.12.0+rocm7.14.0a20260608`, HIP `7.14.60850`, Triton `3.7.0` and native Megatron
checkout `1dcf0dafa884ad52ffb243625717a3471643e087`. The native Megatron checkout
was on `PYTHONPATH`; no framework installation or existing model workspace was
replaced.

`tests/test_debug_device_runtime.py` adds seven **real-device** cases:

| Cases | Observed result |
| --- | --- |
| Native Megatron GPTModel with GELU and SwiGLU, each 2 layers / hidden 16 / FFN 32 / 4 heads / vocab 32 / FP32 | Original native forward, distributed checkpoint save/load, actual weight inventory and loaded logits agree bitwise. Native SwiGLU checkpoint factories are exercised. Loading that checkpoint into vocab 64 is rejected instead of padding/truncating. |
| Native module hooks captured into a HIP graph | Hooks are removed after graph construction; two replays still execute device probes, capture fresh values and preserve native outputs. |
| Instrumented Triton affine/ReLU fused kernel, normal and controlled-fault cases | Intermediates come from one actual fused kernel launch, not a reference forward. The normal case agrees; the fault localizes to the native affine probe with equal observed inputs. |
| Native GPU SGD and AdamW | Backward, accumulation, gradient clipping and optimizer updates preserve the unobserved reference run's parameter bits; paired evidence agrees. |

The native checkpoint tests exposed and fixed two compatibility issues:

1. `allow_shape_mismatch` is a **loader policy**, not proof that an actual tensor
   contains unknown padding. Inventory now observes the complete declared shape,
   while loading disables resizing on copied descriptors and factory outputs.
2. Native `None` and TE's zero-length `uint8` extra-state markers mean no extra
   payload. They are accepted; nonempty opaque payloads remain unsupported.

Regression command:

```bash
python -m pytest \
  tests/test_debug_native_runtime.py tests/test_debug_events.py \
  tests/test_debug_discovery.py tests/test_debug_discovery_weights.py \
  tests/test_alignment_diagnostics.py tests/test_consistency_debugger.py \
  tests/test_module_debug_matrix.py tests/test_repro_launcher.py \
  tests/test_repro_platform_alignment.py tests/test_rocm_repro_profile.py \
  tests/test_rocm_e2e_ablation.py tests/test_debug_device_runtime.py -q -rs
```

Result on MI300X: **261 passed, no skips** (51 upstream dependency warnings).
The existing real-device graph/event smoke tests also ran rather than skipping.
On Windows/PyTorch 2.10 CPU the same suite produced **251 passed, 10 skipped**:
nine device/compiler-gated cases and the existing Windows symlink-permission test.
Ruff and `git diff --check` pass.

This is correctness acceptance on small native workloads, **not** a production
performance result, an arbitrary-architecture inference claim, or multi-GPU
TP/CP/EP acceptance. It does not certify vLLM's real paged scheduler, arbitrary
MoE dispatch kernels, opaque fused internals, FP8 metadata, Megatron distributed
optimizer internals or every graph-capture configuration.
