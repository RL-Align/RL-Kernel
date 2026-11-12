# WS2 CP-Aware Attention Contract

Status: PR1 contract and dispatch metadata

Tracking and shared contracts:

- [#235: CP-aware deterministic Attention](https://github.com/RL-Align/RL-Kernel/issues/235)
- [#83: WS2 roadmap](https://github.com/RL-Align/RL-Kernel/issues/83)
- [#108: WS1 numerical contract](https://github.com/RL-Align/RL-Kernel/issues/108)
- [#111: WS2 cross-config alignment](https://github.com/RL-Align/RL-Kernel/issues/111)
- [#207: cross-config logprob drift contract](https://github.com/RL-Align/RL-Kernel/issues/207)

## Scope

This contract describes the logical inputs and deterministic reduction semantics for standard
softmax Attention under tensor parallelism (TP) and context parallelism (CP). It lets runtime
dispatch reject a backend whose numerical semantics do not match the requested layout.

This PR1 layer does not shard tensors, launch a collective, merge CP partial states, or implement
a fused kernel. The deterministic CP reference implementation and its distributed numerical tests
belong to later work in #235.

## Contract Objects

`rl_engine.contracts.operators.attention` defines:

- `AttentionContract`: role, mode, dtype, causal metadata, sharding, reduction, and optional cache
  identity;
- `ShardingSpec`: TP-local head ownership and CP block-to-token ownership;
- `ReductionSpec`: fixed `(out, lse)` merge semantics;
- `KVCacheSpec`: decode replay cache identity;
- `RoPESpec`: Qwen3 RoPE state, position identity, and fused/unfused boundary metadata;
- `AttentionBackendCapability`: the layouts and semantics a backend explicitly supports.

Construction performs validation immediately. A structurally valid contract means that the
request is complete and internally consistent; it does not mean that an installed backend can
materialize it.

`AttentionContract.batch_size` is the logical sequence count. For packed varlen input it must
equal `len(packed_sequence_offsets) - 1`; it is not the physical leading dimension of a flattened
token tensor.

For full `prefill`, `query_sequence_length` equals the local sequence length described by
`ShardingSpec`. Chunked prefill and decode may use shorter query lengths than their available KV
context.

## Qwen3-8B TP=2 CP=2 Example

```python
from rl_engine.contracts.operators.attention import (
    AttentionContract,
    ReductionSpec,
    RoPESpec,
    ShardingSpec,
)

sharding = ShardingSpec(
    tp_rank=0,
    tp_world_size=2,
    cp_rank=0,
    cp_world_size=2,
    global_q_heads=32,
    global_kv_heads=8,
    local_q_head_start=0,
    local_q_heads=16,
    local_kv_head_start=0,
    local_kv_heads=4,
    global_sequence_length=4096,
    local_sequence_length=2048,
    global_block_indices=(0,),
    global_block_token_starts=(0,),
    local_block_offsets=(0, 2048),
)

contract = AttentionContract(
    role="infer",
    mode="prefill",
    dtype="bf16",
    batch_size=1,
    query_sequence_length=2048,
    head_dim=128,
    causal=True,
    causal_offsets=(0,),
    sharding=sharding,
    reduction=ReductionSpec(),
    rope=RoPESpec(
        q_state="post_rope",
        k_state="post_rope",
        k_cache_state="post_rope",
        theta=1.0e6,
        rotary_dim=128,
        query_position_offsets=(0,),
        key_position_offsets=(0,),
        cast_at="after_rope",
        output_dtype="bf16",
        fusion_boundary="unfused_rope_attention",
    ),
)
```

The TP fields preserve the global Qwen3 GQA mapping: each rank owns 16 of 32 query heads and 4 of
8 KV heads. The CP fields map local tensor slices to stable logical global block ids. A rank that
owns non-contiguous blocks uses one global token start per block and one extra local boundary:

```python
global_block_indices=(0, 3)
global_block_token_starts=(0, 3072)
local_block_offsets=(0, 1024, 2048)
```

This metadata is sufficient for a later implementation to restore logical global order without
using ring arrival order.

## RoPE / Position Semantics

RoPE is part of the attention contract because rollout can materialize
`RoPE+Attention` as a fused or cache-aware path while training may materialize
`RoPE -> Attention` as separate operators. PR1 does not execute the RoPE kernel,
but it records the metadata required to prove both materializations use the same
model semantics.

`RoPESpec` records:

- whether Q, K, and cached K are `pre_rope` or `post_rope`;
- `theta`, optional `rope_scaling`, and `rotary_dim`;
- dense `position_ids` or per-sequence `query_position_offsets` /
  `key_position_offsets`;
- the RoPE cast point and output dtype;
- `fusion_boundary`, either `unfused_rope_attention` or `fused_rope_attention`.

When RoPE metadata is present, construction validates that rotary dimensions fit
the attention head dimension and that offset metadata matches the logical batch
shape. Backends must declare RoPE support through `AttentionBackendCapability`;
a backend that cannot consume RoPE/position metadata or cannot support a fused
RoPE+Attention boundary is rejected before dispatch.

## Reduction Semantics

The only PR1 reduction contract is:

```text
partial state: (out, attention-domain lse)
merge: online_softmax_lse
acc_dtype: fp32
order: global_block_index
downcast_at: final_write
engine: in_op_reference
```

CP output is not a plain sum. A backend that cannot export attention-domain LSE or cannot merge
partial states in fixed logical order is incompatible with this contract.

The acceptable output and selected-logprob drift thresholds remain owned by #108. This contract
does not introduce another tolerance table. When connected to the rollout/training chain, the
selected-token metric remains the #207 convention:

```text
dlogp = training-side recomputed logp - rollout-side old logp
```

## Mode-Specific Metadata

All causal calls provide `causal_offsets`. Packed varlen calls provide one causal offset per
packed sequence and validated `packed_sequence_offsets`.

Decode additionally requires `KVCacheSpec` with:

- one cache position and KV sequence length per logical sequence;
- a block/page table;
- the physical page size;
- global token positions for every logical cached token;
- a prefix-cache key and explicit shared-prefix page count when prefix caching is enabled.

Within each logical sequence, global token positions must be strictly increasing. Block-table
padding must be trailing, the active page count must match `ceil(kv_seq_len / page_size)`, and a
sequence cannot repeat one physical page id. Different sequences may share physical pages for an
equivalent prefix only when those pages are declared by `shared_prefix_page_count`, use the same
leading page ids and logical positions, and are fully populated. Declared shared prefix pages are
read-only; all suffix pages are exclusive to one sequence, providing the contract boundary needed
for copy-on-write before divergent decode. When prefix caching is disabled, no active page may be
shared across sequences. Missing or inconsistent decode cache identity is an error at contract
construction time.

Each `cache_positions` entry is the terminal logical position already present in that sequence's
KV cache, so it must equal the final corresponding `global_token_positions` entry. It is not the
next position to be written.

## Contract-Aware Dispatch

Legacy callers continue to use `KernelRegistry.get_op()`. WS2 callers use:

```python
result = kernel_registry.get_attention_op(contract)
op = result.op
provenance = result.provenance
```

Dispatch considers only backends with an `AttentionBackendCapability`. It checks role, attention
mode, dtype, TP/CP degree, LSE export, deterministic CP merge, packed varlen, and KV-cache support.
When RoPE metadata is present, dispatch also checks whether the backend explicitly supports
RoPE/position metadata and fused RoPE+Attention boundaries.
An undeclared or incompatible backend is skipped with an explicit rejection reason.

The current WS1 PyTorch Attention implementations support local reference math but do not export
attention-domain LSE or materialize deterministic CP merge. Strict WS2 requests therefore fail
clearly today. A later deterministic backend becomes selectable by registering a capability that
truthfully declares those features; no grid-planner branch or silent fallback is required.

Successful dispatch provenance records:

- requested and actual backend ids;
- platform and fallback status;
- prior candidate rejection reasons;
- the complete requested contract;
- the selected backend capability descriptor.

## Validation

Contract and dispatch behavior are covered by:

```bash
python -m pytest tests/contracts/operators/test_attention_contract.py -q
```

The tests include Qwen3 TP=2/CP=2 construction, GQA ownership errors, non-contiguous CP blocks,
packed varlen metadata, decode cache identity, undeclared backend rejection, no incompatible
fallback, RoPE metadata validation, and JSON-compatible provenance.
