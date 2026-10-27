# Task 30: MI300X Hash route

This implements task 30 on `dsv4-rocm` baseline
`5ba769f681b7fa765ea6b29f9939454a858390ba`, independently of the unmerged
task-28/29 PRs. It covers absolute-layer Hash lookup, preserved table slots,
selected weights and their local first-order VJP. Complete routing, transport
and framework/full-model acceptance remain separate work.

## Frozen inputs and arithmetic

The checkpoint is DeepSeek-V4-Flash-0731 revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Real `tid2eid` tables from absolute
layers 0,1,2 are I64 `[129280,6]`. Each table is 6,205,440 bytes; all IDs are in
`[0,256)`, with six distinct experts per row. Their frozen source SHA256 values
are in the operator contract and `task30-mi300x.json`. Download/replay tools
independently validate byte spans, model/tensor identities and actual bytes.

Synthetic FP32 logits/upstream gradients use CPU generator seed 30. Replay
includes repeated IDs and vocabulary endpoints, with the same logical token
placed at different rows/batches. Checkpoint gate weights, tokenizer processing
and complete model activations are not used.

The AITER profile reuses the pinned Hash kernel/wrapper implementation and the
exact task-29 score helper. It preserves table column order, sums selected
scores in ascending slot order, clamps the denominator at 1e-20, and scales
once after normalization. The original AITER sum-plus-epsilon denominator is
explicitly changed to match mLite clamp semantics. First-order VJP is local to
the selected weights; tables/expert IDs are discrete and receive no gradients.

Both companion patches apply cleanly to pinned AITER commit
`326cffa38c9eda16bef19375ba55334e4d3acd59`. Task 30 adds two AITER files without
altering task 29's validated API/source hashes. The installed AITER package
was not replaced. Fingerprints in the report identify the executed patched
source; this is not an already released upstream API.

## Tests

```bash
AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task30/aiter:$PWD" \
  python -m pytest tests/ops/moe/test_hash_route.py tests/runtime/test_hash_route_dispatch.py \
  tests/validation/operators/test_hash_route_fixture.py tests/runtime/test_kernel_registry.py -q
```

| Suite | Result |
| --- | ---: |
| MI300X table/layer/slot, weights, VJP, BI and runtime guards | 68 passed |
| CPU-safe contracts, dispatch and bounded/frozen checkpoint validation | 41 passed |
| Existing kernel registry regression | 12 passed |
| **Total** | **121 passed; no failures or skips** |

Ruff check/format, patch application/fingerprints, whitespace checks and strict
MkDocs build passed. The empty-graph warning is expected from explicit capture
rejection.

Tests cover all three actual tables, I32/I64 source profiles, unsorted slots and
tied scores, vocabulary bounds before narrowing, ten random seeds, both
renormalization modes, scale 0/1/1.5/2, M=0/1/16/17/32/65/257 and varied row
placement/strides/neighbors. They include table ownership and `.data` mutation,
layout changes preserving bytes, absolute-layer/PP-stage ownership, explicit
custom Hash layers, stale model/table barriers, nonfinite values, Graph,
higher-order differentiation and missing/changed AITER sources.

Mathematical correctness uses an independent CPU FP64 gather/score/analytic-VJP
reference, separately validated by finite differences. The mathematical score
oracle uses a conservative -36 cutoff instead of production -16, intentionally
retaining the mathematical expression over the intervening range. Normal
weight tolerance is 2e-6 relative. Subnormal-score rounding before normalization
admits the explicit eight-quanta-times-scale/denominator absolute bound; the
-200 case independently checks normalization of rounded FP32 scores in ULPs.
VJP tests allow cancellation near zero and use tighter negative-tail cases.
These tolerances never replace discrete-ID equality or zero-bit BI.

The 18 checked-in replay cases have zero expert-ID mismatches, zero fixed-row
weight/VJP bit mismatches and zero training/inference forward mismatches.
Maximum observed absolute errors against FP64 are **2.64e-7** for weights and
**6.53e-8** for VJPs.

Two independent processes with distinct Triton JIT caches reproduce identical
table/input/token/upstream/expert-ID/weight/VJP hashes for all 18 cases.
`fresh-process-hashes.json` records those comparisons.

## Performance and limits

Replay uses 10 warmups and 30 GPU-event samples per case. Median full-adapter
latency is **2.42–2.76 ms**, including complete prepared-table SHA256 validation,
device-to-host transfers, synchronization, input checks, autograd and provenance.
Raw AITER forward is **106–176 µs** and local VJP **34–107 µs**, including Python
allocation/launch costs. Forward retains the original bitmatrix/scratchpad
allocation, although that state is not exposed as an RLK dispatch plan.

Whole-table integrity checks deliberately have measurable overhead. The report
does not claim serving speedup, isolated instruction timings or end-to-end
throughput improvement. Any future lifecycle-managed checksum optimization
needs its own table update/barrier validation.

Only MI300X/gfx942 has hardware evidence. MI355X/gfx950, NVIDIA, Graph,
EP/PP communication, complete vLLM/Megatron/verl/vime hooks, two-update LoRA
synchronization and full-model raw-logprob acceptance remain unvalidated.
Tasks 31–34 own additional selection, capacity, auxiliary-loss and routing
training integration; no full task-33 or P6 signoff is inferred here.

See [the operator contract](../../../docs/operators/hash-route.md) for exact
operation/VJP order, table checksums and reproduction commands.
