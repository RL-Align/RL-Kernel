# Stable Top-6 Expert Selection

DSV4 task 31 has one ordering contract: **score descending; on equal scores,
smaller global expert ID first; preserve that order as the six output slots**.
This page follows the task table's AITER implementation references and isolates
selection from the score transform, bias, weighting and capacity tasks.

## Contract-to-implementation mapping

| Issue #81, item 31 requirement | Implementation and acceptance |
| --- | --- |
| Top-6 by score descending | Final merged candidates sorted by the full score key before retaining six |
| Equal scores prefer smaller global expert IDs | Global ID is the secondary key; local column/rank cannot replace it |
| Preserve slot sequence | Return this score order directly; no final expert-ID sort |
| Same/very close scores and 256→6 | Exact CPU-oracle and raw-bit tests, including the sixth-place boundary and adjacent FP32 values |
| Consistent global numbering across cards | Require complete global coverage; two MI300X/RCCL tests use different shard concatenation orders |
| Modify/reuse AITER `topk/streaming_topk` | Companion AITER profile adapts the pinned streaming merge and reuses `fpval_to_key`; RLK binds it explicitly |

## API and complete global coverage

```python
from rl_engine.ops.moe.stable_topk6 import stable_topk6

result = stable_topk6(
    selection_scores, model_version="synchronized-model-v1",
    global_expert_ids=global_ids, global_expert_count=256,
)
scores = result.selected_scores  # FP32 [M,6], original score payload bits
experts = result.expert_ids      # nondifferentiable I64 [M,6], GLOBAL IDs
columns = result.column_indices # nondifferentiable I32 [M,6], caller input columns
execution = result.execution
```

The input is finite FP32 `[M,E]`, E=256 for the target geometry. Explicit test/
alternate geometries admit `6 <= E <= 32767`. `global_expert_count` must equal E;
the optional I32/I64 global-ID vector is a permutation of `[0,E)`, on the same
ROCm GPU. If omitted, columns have canonical global IDs 0..E-1. IDs are validated
before any narrowing/packing. Invalid, duplicate, negative, missing or oversized
IDs fail instead of silently treating columns as global experts.

An EP/TP-local shard is **not** a complete global Top-6. The caller materializes
all global scores and their ID map before this operator. It owns transport,
row/token alignment and model-version barriers. A partial `[M,128]` shard with
global_expert_count=256 is rejected. Local column order may be arbitrary; the
same global score/ID pairs must produce the same ordered global IDs on every
rank. Column indices remain local to the caller's materialization and are
returned for downstream gathering, not as global expert identifiers.

`selected_scores` are the actual **selection scores**, not normalized routing
weights. A task-32 caller may form biased selection scores and use the returned
columns to gather separately preserved unbiased scores. This task does not add
bias, transform gate logits, normalize/scale weights, mask capacity, compute
auxiliary loss or implement complete Learned Router semantics.

Positive strides, row padding, transposes and empty M are supported. RLK makes
contiguous score storage and a private validated global-ID snapshot for AITER.
Expanded zero-stride inputs, BF16/FP16, NaN/Inf, Graph capture and unsupported
architectures fail explicitly. Eligible architectures are gfx942/gfx950;
only MI300X/gfx942 has hardware evidence here.

## Upstream reuse and exact comparison

The task-table reference is AITER
[`streaming_topk` at 326cffa](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py)
and its
[`topk` wrapper](https://github.com/ROCm/aiter/blob/326cffa38c9eda16bef19375ba55334e4d3acd59/aiter/ops/triton/moe/moe_routing/topk.py).
The pinned source encodes larger column IDs as larger tie keys, then re-sorts
selected candidates into expert-ID order. Both behaviors differ from item 31.
Tests retain the original entry as a negative control for ties and slot order.

The companion profile keeps AITER's peeled last block, 128-column streaming
blocks, padded eight-candidate Top-K and bitonic merge. It changes the key and
final ordering and accepts explicit global IDs. The packed uint64 key is:

```text
[32-bit monotone FP32 score key | 16-bit (65535-global_ID) | 16-bit input column]
```

The score key dominates, then smaller global ID wins. Columns only carry the
original location; valid IDs are unique, so a column cannot break a real tie.
Invalid padded candidates have the minimum uint64 key, below every admitted
finite score. After streaming, all eight candidates are sorted descending by
the complete key before taking slots 0..5. No post-selection ID sort occurs.

`+0` and `-0` compare equal for selection. Their ranking keys are canonicalized
using integer bits; raw selected payloads are reloaded unchanged, preserving
signed zeros and subnormal values. Other finite FP32 scores compare exactly,
with no epsilon tie window, quantization, score transformation or FP64 upcast.

An optional **local selected-value VJP** writes each upstream gradient into its
selected input column in a zero-initialized output. Selected columns are
distinct; there are no atomics or gradient reductions. IDs/columns have no
gradients. This is not differentiation of expert selection, complete routing
backward, task-33 replay semantics or distributed autograd. Saved tensor mutation
and nonfinite upstream gradients are rejected; higher-order gradients are
unsupported.

The fixed launch is one row/program, 128 columns/block, padded K=8, four warps,
one stage and FP fusion disabled. M affects the grid, not selection arithmetic.
Forward uses no temporary global workspace beyond the three returned tensors;
the VJP allocates/zeros `[M,E]`.

## Companion setup and reproducibility

Numerical kernels live in the AITER patch
`rl_engine/backends/rocm/moe/patches/stable-topk6.patch`. It adds two files and
does not modify existing AITER Top-K, task-29 or task-30 entry points. It is not
an already released upstream AITER API. RLK adds validation, local autograd,
semantic registration and execution provenance.

```bash
git clone https://github.com/ROCm/aiter.git artifacts/task31/aiter
git -C artifacts/task31/aiter checkout 326cffa38c9eda16bef19375ba55334e4d3acd59
git -C artifacts/task31/aiter apply --check \
  "$PWD/rl_engine/backends/rocm/moe/patches/stable-topk6.patch"
git -C artifacts/task31/aiter apply \
  "$PWD/rl_engine/backends/rocm/moe/patches/stable-topk6.patch"

AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task31/aiter:$PWD" \
  python -m pytest tests/ops/moe/test_stable_topk6.py \
  tests/runtime/test_stable_topk6_dispatch.py tests/runtime/test_kernel_registry.py -q

AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task31/aiter:$PWD" \
  python -m tools.validation.operators.check_stable_topk6 \
  --output artifacts/task31/replay.json

AITER_TRITON_ONLY=1 PYTHONPATH="$PWD/artifacts/task31/aiter:$PWD" \
  python -m tools.validation.distributed.check_stable_topk6_global_ids \
  --devices 0 1 --output artifacts/task31/two-gpu-global-ids.json
```

The exact API/launch IDs, patch hash and two new source hashes are verified.
The shared upstream key helper is admitted only from two explicitly pinned
source profiles: unmodified AITER 326cffa, or that base plus the exact task-29
score patch used by task 30. These are explicit validated profiles, not fallback
selection. Missing or different sources fail. Execution records retain profile,
source hashes, GPU/software, layouts, model version and global-ID-map hash.

## Acceptance and performance limits

The independent CPU oracle uses Python lexicographic sorting by `(-score,ID)`
and raw FP32 gathering. Every FP32 value is represented exactly as a Python
float; signed zeros tie. Tests require exact IDs, columns, payload bits and
local VJP bits. No numerical tolerance substitutes for the item-31 contract.

Coverage includes all ties, sixth-place ties across streaming blocks, adjacent
FP32 values, positive/negative scores, signed zeros/subnormals, padded K=8→6,
E=6/7/31/127/128/129/255/256/257/513/1025/32767, ten random seeds, M=0/1/16/17/
32/65/257, changed neighbors/row positions, layouts and global-ID permutations.
The original AITER behavior must fail the tie/order negative controls.

The two-GPU harness uses **real RCCL all_gather** to assemble complete scores/
IDs from disjoint expert shards, then runs this complete-input operator on
both GPUs with different column materialization orders. It checks exact global
IDs, score bits and local value VJPs after canonicalizing gradient columns.
This establishes one-node cross-card numbering/selection consistency, not a
new transport backend, multi-node support, framework EP hooks or distributed
autograd.

Checked-in evidence is under `reports/experiments/stable-topk6/`. Replay uses
synthetic selection scores, not checkpoint weights or tokenizer preprocessing.
Timing separates the full RLK adapter (finite/coverage checks, ID snapshot/hash,
synchronization, provenance and autograd) from raw AITER calls. Each uses ten
warmups and 30 GPU-event samples around Python calls; no isolated-instruction,
serving speedup or full-model throughput claim is made. MI355X, NVIDIA, Graph,
framework integration, LoRA synchronization and full-model logprob acceptance
remain unvalidated.
