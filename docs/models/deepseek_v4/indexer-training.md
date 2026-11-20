# Indexer and compressor training VJPs (item 26)

This opt-in training boundary composes the existing RLK item-13 Indexer
projection, item-16 ICV, item-20 Indexer C4, item-18/19 main compressors,
and item-25 joint attention with native aiter, vLLM and mLite operators.
It does not copy a numerical GPU kernel into RLK. The branch starts from
`main` `8588158650aff58c9c1f527674de94bad06c3d10` and stacks those
still-open prerequisites; review the item-26 commit separately.

`IndexerAuxContract` fixes global query normalization, C4 logical group
identities, explicit CP ownership, teacher geometry, MXFP4 or FP8 encoding,
the dequantized-operand identity STE and the mLite sparse-KL objective.
`IndexerAuxMetadata.bind` snapshots query/key IDs, owners and selected groups,
sorts selected logical groups canonically and rejects absent/future keys,
duplicate queries and stale versions. `IndexerAuxiliary` reuses the existing
quantized codecs, aiter fixed FP32 GEMM, vLLM BI batched GEMM and aiter MHA
row reduction; its VJPs feed RLK's canonical backward session. The main and
Indexer C4 compression branches remain separate. A query's compressed K
gradient is reduced in canonical logical query order across microbatches.

The process must start with `TRITON_F32_DEFAULT=ieee`; TF32 is explicitly
rejected. All external numerical source files are pinned by SHA256 through
`scripts/prepare_dsv4_indexer_training_sources.py`, and unsupported precision,
layout, workspace, Graph capture or provider revisions fail closed. The
training API does not silently select another backend. The mLite reference
checks are a mathematical/semantic comparison, not proof that full Megatron
or vLLM model execution already uses this bridge.

## MI300X validation

In the `rl-kernel-ljj` ROCm container (gfx942, one visible GPU), the four
item-26 test files passed **101/101**. The operator-contract and DeepSeek-V4
suite, including all stacked prerequisites, passed **963/963** with 19
warnings. Ruff check and format-check passed for the ten new Python files.
Tests cover metadata and version guards, FP32 IEEE native primitive accuracy,
MXFP4/FP8 quantized Q/K inputs and surrogate VJPs, independent reference
gradients, microbatch repartition/permutation, fresh-process byte digests,
and a composed projection/Indexer C4/main C4/C128/joint-attention training
chain. This is single-card operator evidence, not a complete model claim.

Run with the explicitly provisioned sources in
`/workspace/upstream/indexer-training-pins`:

```bash
export TRITON_F32_DEFAULT=ieee
export RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE=/workspace/upstream/indexer-training-pins/sparse_kernels.py
export RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE=/workspace/upstream/indexer-training-pins/sparse_wrapper.py
export RLK_AITER_INDEXER_REFERENCE=/workspace/upstream/indexer-training-pins/indexer_reference.py
export RLK_MLITE_DSV4_BACKWARD_SOURCE=/workspace/upstream/indexer-training-pins/mlite_backward.py
export RLK_MEGATRON_DENSE_SOURCE=/workspace/upstream/indexer-training-pins/mlite_dense.py
export RLK_MLITE_DSA_SOURCE=/workspace/upstream/indexer-training-pins/mlite_dsa.py
PYTHONPATH=. pytest -q tests/contracts/operators tests/models/deepseek_v4
```

## Exclusions

No actual two-rank CP gradient exchange, PP stage mapping, whole-engine
Graph, weight/version synchronization, or 0731 four-path end-to-end logprob
comparison has been signed off here. The current `ljj` container exposes one
GPU and has no 0731 checkpoint. Item 27 and communication tasks 39-42 must
be integrated and tested separately before claiming the issue-81 target.
Benchmark results are in `reports/indexer-training-mi300x-20261029.md`;
the full JSON is kept in `/outputs/item26-20261029/benchmark.json`.
