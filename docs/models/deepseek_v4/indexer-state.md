# Task 22 — independent Indexer partial state and MXFP4 cache

This task is stacked on task 20 / PR #624 and task 21 / PR #625; the PR target
is `dsv4-rocm`. It reuses both prerequisite implementations.

## Boundary and reused code

`IndexerStateContract` admits only Indexer C4: head_dim=128, four-token groups,
eight-token overlap, input `[tokens,512]` BF16, APE `[4,256]` FP32 and RMS
weight `[128]` FP32. Partial history remains FP32 `[kv,score+ape]`.

`IndexerStateCache` inherits task 21's request creation, staging, commit,
cancellation, release, model replacement and failure handling. It does not
implement a second lifecycle state machine. It specializes only allocation,
output metadata, clearing and snapshotting. Main-KV and Indexer contracts and
handles cannot be mixed.

Numerical work is composed from existing operators:

1. Task 21's aiter state-read/compression/RMSNorm/RoPE into a private BF16
   temporary cache, using the fixed single-wave profile.
2. Task 20's native aiter Hadamard, including its eight-physical-row tail guard.
3. Task 20's aiter MXFP4 group-32 quantizer, with unshuffled scales.
4. Native Torch `masked_scatter_` for unique, sorted complete-row payload and
   scale writes.
5. Task 21's existing vLLM partial-state update **after** compression and commit.

No numerical GPU kernel is newly written. The generic control path is shared
with main-KV state, but the backing tensors, handles, fingerprints and parameters
remain separate.

## Cache layout and invalid rows

- Packed cache: uint8 `[pages,64,64]`; each logical row is 128 E2M1 values,
  low nibble first.
- Scale cache: uint8 `[pages,64,4]`; each byte is an E8M0 scale for 32 values.
- Layout ID: `e2m1.low_nibble_first.row_major.e8m0_group32`.
- Unused/released rows are initialized to payload byte 0 and scale byte 127.
  This pattern is **not a validity flag**; the committed logical cursor decides
  which rows are valid. A real zero row may have different provider scale bytes.

The private BF16 scratch buffer may be cleared/reused freely. Persistent packed
data/scales are written only where the prepared complete-row mask is true.
The dense quantized source prefix and mask traversal have the same increasing
logical order. Padded source rows and incomplete groups never write persistent
cache. Payload and scales are snapshotted under the same lock/cursor.

The returned layout is intentionally unshuffled and directly compatible with
the declared raw layout boundary; it is **not** a drop-in replacement for an
engine's swizzled/TMA-packed cache ABI without an explicit adapter.

## Lifecycle, failure and Graph guarantees

All task-21 rules apply: request generation/model version and writer checks,
one outstanding staging lease, exactly-once commit, synchronous device completion
before cursor publication, owned inputs/snapshots, release-before-reuse, and
poisoning after partial mutable work fails.

In particular, a failure between payload and scale writes does not publish a
new cursor and does not allow retry or reading partial state.

Graph captures the same fixed-capacity pipeline. Gather indices and complete-row
masks are prepared outside capture and reused at fixed addresses. Warmup uses
invalid metadata and verifies both payload and scales remain at their initial
values. The actual native gather/quantize/scatter operations run during replay;
this is not a CPU emulation of Graph state.

## Verification and limitations

Tests compare packed data and scale bytes to task 20's standalone native operator,
whose numerical/FP64/gradient tests are also included in the combined suite.
They cover prefill/decode/chunks, multiple requests/seeds, page reuse, partial
state bytes, invalid rows, model replacement, original-input aliases, threads,
streams, Graph, native call ordering, allocation stability and injected
payload-before-scale failure.

The cache API is non-differentiable. Quantized training VJP is task 26. This does
not claim actual distributed CP/RCCL, multi-node operation, complete framework
registration, a swizzled engine cache ABI or full-model acceptance.

The shared task-20 postprocessor currently resolves its pinned mLite source
during construction, even though this state API does not execute backward.
Set `RLK_MLITE_DSV4_BACKWARD_SOURCE` to the verified source used by task 20.

```bash
python3 -m pytest tests/contracts/operators tests/models/deepseek_v4 -q \
  --basetemp=/outputs/indexer-state-tests \
  --junitxml=/outputs/indexer-state-tests.xml
python3 benchmarks/benchmark_indexer_state.py \
  --repeats 30 --output /outputs/indexer-state-benchmark.json
```
