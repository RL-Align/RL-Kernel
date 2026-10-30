# DSV4 item 18: C128 KV compressor

This item provides an explicit operator boundary for the **main KV** C128
compressor. It is intentionally separate from:

- item 16 raw Indexer ICV;
- item 20 the independent 128-dimensional Indexer C4 compressor;
- item 21 request state, cache ownership and single-write commit;
- item 27 runtime/prefill/decode/recompute/Graph integration.

## Contract

```text
input:  BF16 [tokens, 1024] = [content512, score512]
        int64 logical token IDs [request_id, global_token_position]
output: BF16 [complete_groups, 512]
        logical group IDs [request_id, global_group_id]
```

Only a complete group of **128 consecutive tokens** is emitted. A partial
leading or trailing group is reported as incomplete and is never padded or
submitted to the provider. CP callers must transport the complete group before
binding. Exactly one rank emits a complete group: the rank that owns its final
token. This item does not perform a collective or mutate a live cache.

The admitted model profile is `head_dim=512`, `ratio=128`, `rope_dim=64`,
BF16 output, and finite FP32 APE/RMS parameters. Position is global:
`group_id * 128`; local rank order is not used as a substitute.

## Reuse and arithmetic

The ROCm backend does not add a numerical GPU kernel. It pins and calls:

1. aiter's `flydsl_hca_compress_forward` for FP32 online softmax pooling;
2. aiter's compiled `compile_hca_norm_rope_scatter` for FP32 RMSNorm and
   BF16 GPT-J RoPE output;
3. the mLite/Megatron `compressed_compact_graph` for the functional VJP;
4. RL Kernel's existing `CanonicalBackwardSession` to register logical groups
   and reduce parameter rows in a fixed order.

The forward recipe is fixed to aiter C128's eight K-split waves and 64-element
head-dimension slices, followed by the one-wave norm/RoPE kernel. Batch size
does not choose a different arithmetic plan. The output is a private compact
buffer; live state and cache commit are deliberately deferred to item 21.

The aiter and mLite source revisions are checked by SHA-256 before the backend
is admitted. Missing or different provider sources fail closed. No fallback to
PyTorch is exposed by the production backend.

## Backward and lifetime

The VJP uses the pinned mLite functional graph unchanged, one complete group at
a time. Group-local input gradients are copied back through unique selectors
(no atomic scatter). A training call requires a canonical backward session and
a stable layer parameter ID. Duplicate logical groups, mismatched parameter
identity/version and incomplete sessions are rejected. Parameter rows are
combined in the existing canonical session with a prescribed FP32 left fold.
Higher-order gradients are not admitted.

Bound inputs and provider workspaces are guarded by storage/version stamps.
The caller must not mutate bound values, metadata, parameter or RoPE storage
until the result is consumed and backward completes. This item does not claim
Graph capture, full CP/RCCL transport, cache reuse, or full-model integration.

## Verification

The MI300X acceptance tests cover:

- 67 CPU-safe contract/adversarial/autograd tests;
- 30 MI300X tests using the actual pinned aiter kernels and mLite VJP;
- complete/partial groups, global positions, padding/permutation, duplicate
  metadata, CP ownership, finite spectra, softmax extremes and budgets;
- fixed plan assertions, stream reuse, fresh-process/JIT isolation and an
  uneven two-rank Gloo transport fixture;
- forward numerical comparison to an independent FP64 formula;
- full/microbatch/reversed-order backward and finite differences for APE/RMS.

The Gloo fixture validates transport and logical ownership only. It is not a
claim of RCCL or multi-GPU acceptance. Benchmark output reports Python launch
and synchronization time, not full model throughput.

## 2026-10-29 packaged ROCm-image compatibility

The adapter also accepts the explicitly pinned, self-contained HCA source
shipped in the current aiter 0.1.16 ROCm image:
`e7aaf267eb78dd16c24f2967c6f5f825781a8dbac81d837dad74f8218a2aad5a`.
It reuses `flydsl_hca_compress_attn` with explicit eight-wave/slice-64 pooling
and that source's fixed one-wave norm/RoPE kernel. It does not replace or
upgrade the installed library, and unknown source revisions still fail closed.

The newer split-forward ABI pinned by
`183136a19a0c76fe19fd99b084b9dca9631b5e1fcee0ff2a5c440c271eea07d8`
remains supported unchanged. The provider fingerprint records the selected
ABI and all relevant source hashes. Train/rollout must agree on this exact
fingerprint; byte equality across different library revisions is not claimed.

This remains the **eight-wave HCA** profile, not item 21's single-wave C128
state profile. This compatibility update does not waive that arithmetic
handshake or claim cross-profile equivalence.

See `reports/kv-compressor-c128-new-host-20261029.md`.
