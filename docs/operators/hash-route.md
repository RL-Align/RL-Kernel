# Hash Routing

DSV4 task 30 selects experts by **absolute layer number and a versioned tid2eid
table**. The output preserves tid2eid column order, independently of logit rank
or expert number. Hash selection is discrete; only the selected weight branch
has a first-order local VJP.

## API, geometry and version barriers

```python
from rl_engine.ops.moe.hash_route import hash_route, prepare_hash_route_table

table = prepare_hash_route_table(
    tid2eid, absolute_layer=0, model_version="model-v1", table_version="tables-v1",
    expected_sha256=verified_source_sha256,
)
result = hash_route(
    logits, token_ids, table, absolute_layer=0,
    model_version="model-v1", table_version="tables-v1",
    renormalize=True, routed_scaling_factor=1.5,
)
weights, expert_ids, execution = result.weights, result.expert_ids, result.execution
```

`logits` are finite FP32 `[M,256]`; token IDs are I32/I64 `[M]`.
Weights are FP32 `[M,6]`; global expert IDs are nondifferentiable I64 `[M,6]`.
All inputs and the prepared table share one ROCm GPU. Empty M and positive
noncontiguous strides are supported. The binding explicitly materializes
contiguous logits/token IDs for AITER; expanded zero-stride views are rejected.
Token IDs are logical vocabulary IDs, not token positions; negative or
out-of-vocabulary IDs fail before narrowing to I32. IDs are never clamped.

Source tables are I32/I64 `[V,6]`, with positive V below 2^31, and six distinct
global experts in `[0,256)` per row. Range checks precede I64-to-I32 narrowing.
The source checksum covers raw bytes in column order; source dtype is recorded
separately.
Preparation owns a detached I32 GPU snapshot, so later source changes cannot
mutate the active version. **Every route verifies the entire prepared-content
SHA256**, including a device-to-host copy; it does not rely on `Tensor._version`
or pointers. Direct `.data.copy_` changes are detected. A reload needs a new
prepared table and explicit matching model/table version barriers.

The default Hash layers are absolute **0,1,2**. An explicitly supplied tuple
`hash_layers=(...)` admits other agreed nonnegative global layer numbers; it is
not inferred from a PP stage-local index. `DeepSeekV4HashRouter` in
`rl_engine.models.deepseek_v4.hash_router` selects among prepared tables by
absolute layer, and supports a subset owned by a PP stage. Missing layers,
misbound tables and version mismatches fail. This is a local adapter, not PP
communication or a framework hook.

## Frozen 0731 checkpoint tables

Revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` of
`deepseek-ai/DeepSeek-V4-Flash-0731` stores all three tables as **I64 [129280,6]**,
each 6,205,440 bytes. There are no duplicate expert IDs within a row.

| Absolute layer | Tensor | Raw I64 SHA256 |
| --- | --- | --- |
| 0 | `layers.0.ffn.gate.tid2eid` | `b310d7aa41cf4967e3db21bb0b00b12357f4423b4347e71760a29d340031b897` |
| 1 | `layers.1.ffn.gate.tid2eid` | `0e20c4b0af4a7c2c06a73ad62f575b31379200cf8247f99c4536273cc6d3c271` |
| 2 | `layers.2.ffn.gate.tid2eid` | `cfd0cc9496950c0fc26ee82d1a919f85a3ba54c357ec7844b65654cc0d7c326d` |

The downloader fetches only these three tensor ranges, not full shards. It
validates the frozen shard, I64 geometry, exact nonnegative byte spans and
fixed SHA256 before constructing/saving tensors. The replay loader independently
checks model/revision/absolute layer/tensor/shard/layout and actual bytes before
selecting a GPU. A replacement table with a self-computed checksum is rejected.
Synthetic tables remain available through the generic preparation API with an
explicit expected checksum; checkpoint acceptance always uses the frozen ones.

## Arithmetic and local first-order derivative

The implementation adapts the table's
[AITER `_hash_routing` at 326cffa](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py)
and its
[Python wrapper](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/moe/moe_routing/topk.py).
Lookup, score/gather and bitmatrix construction retain that implementation;
the task-30 AITER profile fixes normalization and adds the selected-weight VJP.

1. Lookup `ids = tid2eid[token_id, 0:6]` without sorting.
2. Compute the task-29 FP32 sqrt-softplus score for each selected expert.
3. Sum scores in ascending **slot 0 through 5**, with FP32 rounding after
   each addition. If renormalization is enabled, use `d=max(sum,1e-20)`;
   otherwise d=1.
4. Produce `(score/d)*routed_scaling_factor`, with no intermediate BF16 cast.

The pinned AITER original uses `sum+1e-20`; this profile explicitly changes it
to `clamp_min`, matching the
[mLite fixed-route normalization](https://github.com/ISEEKYAN/Megatron-LM/blob/327a238239448de6771beea185dc6902a8de3eaa/experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/moe/module.py).
Tests around tiny score sums distinguish these choices. No epsilon is added
to an already larger sum. Scale is applied exactly once, after normalization.

For upstream g, form `h=g*scale`. With renormalization, let p=score/d and
`dot=sum(h*p)` in ascending slot order. If sum>=1e-20, the score adjoint is
`(h-dot)/d`; below the clamp it is `h/d`. Without renormalization it is h.
Multiply by the task-29 score derivative; below -16 combine this product and
division in log space to preserve representable gradients after score underflow.
The clamp equality branch follows mLite/PyTorch's differentiable `clamp_min`.
Only six selected positions are written to a zero-initialized dlogits row.
Tables/IDs have no gradients; distinct table IDs permit writes without atomics.

All VJP arithmetic is FP32 and FP fusion is disabled. Finite input/scale/output
is required; overflowing weights/VJPs are rejected explicitly. Higher-order
derivatives, Graph capture, selection bias, Top-K, capacity masks/overflow,
auxiliary losses and distributed gradient/branch merging are outside this task.
The local VJP does not certify all of task 33 or P6 training semantics.

## Companion patch and reproduction

Task 28/29 PRs are separate and need not be merged to test this branch.
The exact task-29 AITER patch/manifest are included as a prerequisite;
`hash-route.patch` then **adds two AITER files** adapted from the pinned Hash
kernel/wrapper. It does not change any of task 29's source/API fingerprints,
so task 29's independently validated binding remains compatible.
These are companion patches, not APIs already released in upstream AITER.

```bash
git clone https://github.com/ROCm/aiter.git artifacts/task30/aiter
git -C artifacts/task30/aiter checkout 326cffa38c9eda16bef19375ba55334e4d3acd59
git -C artifacts/task30/aiter apply \
  "$PWD/rl_engine/backends/rocm/moe/patches/router-sqrt-softplus.patch"
git -C artifacts/task30/aiter apply \
  "$PWD/rl_engine/backends/rocm/moe/patches/hash-route.patch"
python -m tools.weights.prepare_hash_route_tables --output-dir artifacts/task30

AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task30/aiter:$PWD" \
  python -m pytest tests/ops/moe/test_hash_route.py tests/runtime/test_hash_route_dispatch.py \
  tests/validation/operators/test_hash_route_fixture.py tests/runtime/test_kernel_registry.py -q

AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task30/aiter:$PWD" \
  python -m tools.validation.operators.check_hash_route --tables-dir artifacts/task30 \
  --output artifacts/task30/replay.json
```

The RLK binding verifies both patch hashes, the exact API ID and all five
affected AITER source hashes. Missing/modified dependencies fail without a
fallback. Execution records include actual forward/VJP kernels, AITER base,
patches, ROCm/PyTorch/Triton versions, GPU/architecture, absolute layer,
table source/prepared hashes, versions, dtype conversions, layouts and options.
Forward uses fixed 32x128 tiles and 8 warps; the VJP uses one 8-slot program per
row and one warp. The reused forward allocates its AITER bitmatrix/scratchpads;
they are not exposed as an RLK dispatch plan.

## Validation, error bounds and performance disclosure

Independent CPU FP64 references use table gather, mathematical sqrt-softplus
and its analytic VJP; finite differences test that VJP independently. Like
task 29, the mathematical reference's conservative -36 tail is distinct from
the production -16 approximation. It does not emulate FP32 rounding.

Normal-value weight tolerance is `rtol=2e-6`. A subnormal FP32 score is rounded
before normalization; division by the clamped 1e-20 can amplify that error.
The absolute allowance is eight minimum FP32 subnormal quanta times scale/d,
bounded by `(2^-146)*scale/1e-20`. A separate -200 case explicitly checks the
FP32-rounded score normalization in ULPs. VJP comparison uses `rtol=3e-4,
atol=2e-7` for cancellation near zero, with tighter tail-specific tests.
Discrete IDs and BI always require zero differences, independently of tolerance.

Tests cover all three real checkpoint tables and vocabulary endpoints, repeated
tokens, unsorted/tied scores, ten random seeds, M=0/1/16/17/32/65/257, changed
neighboring rows and layouts, local gradients, table/source mutation, absolute
layer/PP-stage ownership, explicit custom Hash layer maps and stale barriers.
Detailed results and performance live in `reports/experiments/hash-route/`.

Latency separates the full adapter, including whole-table checksum transfers
and synchronization, from raw AITER calls. Raw timings include Python allocation
and the reused bitmatrix/scratchpad costs. No full-model speedup is claimed.
Only MI300X/gfx942 is measured. MI355X/gfx950, NVIDIA, full-model/framework
integration, LoRA synchronization and EP/PP transport remain unvalidated.
