# Task 21 — request-owned compressor state and single cache commit

This change is stacked on task 19 / PR #618 at `8793052`; its target remains
`dsv4-rocm`. It adds lifecycle/control code, not numerical GPU kernels.

## Reused providers and explicit arithmetic

The backend directly calls aiter `flydsl_fused_compress_attn` to read previous
state and compress/scatter complete rows, then vLLM `save_partial_states` to
update FP32 `[kv, score+ape]` state. The order is fixed: **compress before
updating history**. The installed aiter ABI needs contiguous split readers;
preallocated copies adapt vLLM's packed state without changing its values.

The admitted main-KV profiles are:

- `aiter.main_c4.online_fp32.wave64.vllm_partial_f32.v1`
- `aiter.main_c128.online_fp32.wave64.vllm_partial_f32.v1`

Both use a fixed single-wave online-FP32 recipe. **C128 is not task 18's
eight-wave HCA profile.** The required `expected_arithmetic` constructor
argument rejects a different recipe rather than silently substituting it.
Callers must select this same profile for the paths they compare; shape
compatibility alone is not a numerical compatibility claim. Training/rollout
integration with the eight-wave HCA path is not covered by this profile.

## API and ownership

Construct `CompressorStateCache` with a `CompressorStateContract`, FP32 APE/RMS
weights, BF16 position tables, a nonempty model/position version and a writer
rank. The cache owns snapshots of the parameters and a fixed-capacity workspace.
No model downloads or provider replacement occurs during construction.

1. `create_request(request_id, version=..., writer_rank=...)` returns a handle
   bound to this arena, a generation, version and private page slot.
2. `prepare(handle, values, start_position=...)` validates detached BF16 input
   and owns its staging copy. Only contiguous new logical tokens are accepted.
3. `commit(lease)` consumes that **exact lease once**, performs ordered native
   writes and waits for device completion before publishing the token cursor.
4. `append(...)` combines prepare and commit under the same host lock.
5. `cancel(lease)` aborts a prepared, not-yet-executed transaction.
6. `snapshot(handle)` returns caller-owned complete rows and the FP32 ring,
   plus token/group/version metadata. It does not expose mutable arena storage.
7. `release(handle)` clears its state/pages before slot reuse. The old handle
   cannot access a later generation with the same logical request ID.
8. `replace_model(...)` requires all requests and leases to be released, a
   different version, and fully validated replacement parameters.

There is one staging lease per arena. Thread races, stale/foreign handles,
wrong writers, duplicate commits, noncontiguous tokens and capacity violations
are rejected. Errors after mutable GPU work poison the arena; callers must
discard it instead of retrying possibly partial writes.

Only the final ring-capacity suffix of an oversized input chunk is eligible
to write partial state. Earlier writers receive vLLM's invalid slot sentinel,
so one native launch never has competing writers to the same ring slot.

The request's complete-row interval is determined by the logical cursor:
`[old_tokens // ratio, new_tokens // ratio)`. Incomplete groups are never
committed. The ring also retains the predecessor needed for C4 overlap.

## Graph and memory

Graph mode captures the same upstream operations with fixed input, metadata,
state-reader and cache addresses. Warmup/capture uses only invalid plans and
write slots and is checked not to mutate state/cache. Per-request lifecycle,
version validation, metadata preparation and exactly-once checks stay **outside**
capture. Direct external capture of lifecycle methods is rejected.

Commits synchronously finish their device event. This intentionally conservative
policy prevents early staging/page reuse between threads or streams; asynchronous
communication/compute overlap is a later task. Input producers must order their
writes before `prepare`; private arena storage must not be modified externally.

The byte budget bounds explicitly owned resident tensors, not PyTorch/FlyDSL
compiler caches, Graph memory pools, admission temporaries or caller snapshots.
Provider-addressing bounds are checked before allocation.

## Integration and limits

The vLLM and Megatron opt-in factories share this exact lifecycle implementation.
This is not complete framework model registration. Cache writes are deliberately
non-differentiable; detached inputs are required. Native CP transport must occur
before ingestion; a writer-rank check is not a distributed lock or collective.

No multi-GPU/RCCL, multi-node, full model or eight-wave C128 training/rollout
equivalence is claimed. Indexer MXFP4 state and cache belong to task 22.

Run the targeted suite and benchmark in the member container, keeping results
in the mounted output directory:

```bash
python3 -m pytest tests/contracts/operators tests/models/deepseek_v4 -q \
  --basetemp=/outputs/state-tests --junitxml=/outputs/state-tests.xml
python3 benchmarks/benchmark_compressor_state.py \
  --repeats 30 --output /outputs/state-benchmark.json
```

The existing task-19 regression tests require the pinned mLite source through
`RLK_MLITE_DSV4_BACKWARD_SOURCE`; the state backend itself has no mLite dependency.
