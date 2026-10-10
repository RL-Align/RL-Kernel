# WS2 attention decode replay

PR6 of issue #235 extends the single-GPU attention attribution harness with a
correctness-first decode-stage KV-cache replay. The reference compares each
decode query with the same query evaluated over a fully materialized logical KV
sequence. Both paths export attention output and attention-domain LSE.

## Cache identity

`DecodeKVCacheMetadata` separates logical sequence identity from physical cache
layout:

- `cache_position` and `query_position_ids` identify each decode query;
- `kv_seq_lens` records the active cached sequence length;
- `block_table` maps logical blocks to physical pages;
- `global_token_positions` identifies every populated physical cache slot;
- `key_position_ids` binds cached K to its RoPE positions;
- `q_rope_state` and `k_cache_rope_state` declare whether tensors are before or
  after RoPE;
- `prefix_cache_key`, `prefix_length`, and `prefix_cache_fingerprint` identify a
  reused logical prefix and bind it to the cached K/V content;
- `cp_block_owners` records logical CP ownership without changing merge order.

Metadata is validated before attention runs. Missing pages, duplicated active
pages, non-canonical `-1` page/owner tails, out-of-range positions, mismatched
RoPE positions, or inconsistent prefix-cache identity fail with an explicit
error. Logical positions must be non-negative and strictly increasing, but may
start at a nonzero global offset (for example after a sliding-window eviction).
Prefix identity is verified by recomputing a physical-layout-invariant
SHA-256 fingerprint over logical prefix positions, cached K/V content, storage
dtypes, and the cache-side RoPE materialization configuration (`theta`, rotary
dimension, cast boundary, and output dtype). The prefix key and enabled state are
part of that fingerprint. Every replay report also includes a physical-layout-
invariant full cache execution fingerprint over logical positions, page size, CP
ownership, prefix/RoPE identity, dtypes, and active K/V content. The current reference requires
`rope_cast_at="after_rope"` and `rope_rotary_dim == head_dim`; partial rotary is
not supported yet. Q and cached K retain separate RoPE output dtype contracts so
mixed rollout query/KV storage dtypes are represented faithfully.

## Deterministic replay

The replay restores logical KV order from the block table, computes one FP32
partial `(out, lse)` state per logical block, and merges partial states in
`global_block_index` order. CP ownership and physical page order never determine
the numerical reduction order. Both the full logical-KV reference and paged
replay accumulate and export LSE in FP32; output downcast occurs only at final
write.

For each decode query at logical position `t`, only cached positions less than
or equal to `t` participate. This supports `Sq=1` and few-query replay.

The reference path removes physical page boundaries after reconstructing the
same logical KV sequence. Therefore the primary check is:

```text
decode_paged_kv(query_t) == full_logical_kv(query_t)
```

The report reuses the PR2 drift format and includes maximum, mean, p95, and p99
absolute drift for output and LSE, plus decode/cache/RoPE provenance.

## Transformer Engine

Transformer Engine remains optional. When requested, the replay passes the same
sorted partial states to the capability-probed TE context-parallel correction
helpers already used by the PR2 harness. TE validates merge arithmetic only; it
does not interpret cache metadata, choose logical order, or replace RL-Kernel's
reference semantics. The comparison always reports the RL-Kernel result; when
TE is requested but unavailable or incompatible, it records the reason in the
report instead of failing the core decode harness.

## Current boundary

The harness models CP block ownership on one device so cache construction,
logical ordering, RoPE identity, and deterministic merging can be attributed
without communication. It does not implement P2P NCCL transport, production
custom CUDA all-gather/reduce-scatter, FlashInfer runtime execution, or training
backward. Reports explicitly record `single_device_logical_reference`,
`runtime_verified=false`, `supports_backward=false`, and `communication=none`;
they are not valid Megatron/vLLM runtime readbacks. A distributed caller can gather the same partial states using the PR3
transport layer; numerical merging must still happen in the fixed logical order
described above. This PR targets the shared `test` integration branch and has a
logical dependency on PR2/#253.
