# DSV4 item 19: C4 KV compressor

This item provides the main-KV C4 compressor with the fixed eight-token overlap
window. It is separate from:

- item 18 C128 main-KV compression;
- item 20 the independent 128-dimensional Indexer C4 compressor;
- item 21 state/cache lifecycle and exactly-once commit;
- item 27 prefill/decode/recompute/Graph runtime integration.

## Contract

```text
input:  BF16 [tokens, 2048]
        [content1024, score1024] where each 1024-wide stream has two 512 halves
        int64 logical token IDs [request_id, global_token_position]
output: BF16 [complete_groups, 512]
        logical IDs [request_id, global_group_id]
```

The compressor consumes four current tokens per output row. For group `g`, the
softmax window is:

```text
previous group content/gate first half (g-1), then
current group content/gate second half (g)
```

The first group has four masked predecessor rows. Incomplete groups are never
padded into a submitted output. Global position is `group_id * 4`; local rank
order is not substituted for it. The optional `emit_group_ids` binding argument
allows a microbatch to include one completed predecessor group as context while
emitting only its requested groups.

CP callers must provide complete current groups and their overlap predecessor
before binding. The rank owning the current group’s fourth token emits that row.
No collective is performed by this item.

## Reuse and paged output

The ROCm backend directly reuses aiter’s
`flydsl_fused_compress_attn` with `overlap=True`, `ratio=4`, `head_dim=512`,
BF16 output, and the fixed one-wave (`k_split_num_waves=1`) arithmetic path.
The private output uses the provider’s real compressed-cache layout:
`k_per_block=64`, a per-request block table, and cache rows selected by
`group_id % 64`. This is important for correctness; treating every row as a
one-row block is not an accepted shortcut.

The overlap input is materialized as eight direct input rows per plan row. This
reuses aiter’s optimized online-softmax/norm/RoPE kernel while leaving live
request state mutation and cache commit to item 21. The first group’s four
predecessor scores use a finite BF16 floor below the admitted score range,
avoiding the input-phase `-inf - (-inf)` case; the independent oracle retains
the exact masked semantics.

Backward reuses the pinned mLite/Megatron `compressed_compact_graph` ratio-4
functional graph. One output’s VJP is evaluated against its previous/current
context. Overlapping row contributions are applied in a fixed Python logical
order, not with unordered floating-point atomics. APE and RMS rows are
registered with RLK’s existing `CanonicalBackwardSession`.

Provider source hashes are checked before admission; missing or different
aiter/mLite sources fail closed. No PyTorch numerical fallback is exposed.

## Verification

The MI300X tests cover:

- complete and partial groups, first-group masking, overlap previous/current
  halves, global positions, request permutation and padding;
- CP final-token ownership, context predecessor admission and block-table
  mapping;
- softmax/content scale extremes, fixed overlap plan, `k_per_block=64`,
  output/cache isolation, streams and fresh-process/JIT fingerprints;
- independent FP64 forward arithmetic;
- full/microbatch/reversed-order backward, overlap gradient accumulation and
  independent autograd/finite-difference checks;
- malformed ABI/metadata, budget, mutation, version and Graph rejection.

This item does not claim RCCL/multi-GPU CP, live cache commit, Graph replay,
prefill/decode framework integration or full-model acceptance.
