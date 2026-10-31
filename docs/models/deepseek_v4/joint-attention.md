# Item 24: joint sparse attention with sink

The MI300X profile uses **one unmodified aiter training-forward kernel** for
training-forward records, recomputation, prefill and decode. It does not try
to reconcile outputs from two different softmax implementations afterward.

Compressed KV rows and the complete recent window participate in one
softmax denominator with a per-head sink logit. **The sink has no V.**
K and V are the same BF16 vector. The upstream kernel uses FP32 softmax
statistics, BF16 P at the value dot boundary, BF16 O, and natural-log FP32
LSE. This approximate arithmetic is explicit; the FP64 mathematical oracle
is a tolerance check, not a bytewise oracle.

## Reused providers

* aiter commit `326cffa38c9eda16bef19375ba55334e4d3acd59`,
  `aiter/ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py`.
* vLLM `dequantize_and_gather_k_cache_triton` for the actual DSv4 mixed
  FP8/BF16 paged-cache ABI, with explicit OCP or FNUZ decoding.
* Torch layout/gather/masking/cloning and CUDA Graph/event primitives.

No numerical Triton/HIP kernel is copied into RLK or newly authored.
The adapter directly calls the pinned aiter JIT function with fixed
H32/K32, four-warps/two-stages launch metadata. It bypasses the upstream
environment-dependent tuning lookup, not the upstream arithmetic.
One query/head-group CTA visits the canonical KV list; there is **one KV
partition**, with fixed 32-entry online-softmax tiles and no split-K merge.

The installed image may not contain this older training source. Provision
it explicitly, preserving its upstream MIT notice:

```bash
python3 scripts/prepare_dsv4_sparse_provider.py \
  --output /workspace/upstream/aiter-sparse-326c/kernels.py
export RLK_AITER_DSV4_SPARSE_KERNEL_SOURCE=/workspace/upstream/aiter-sparse-326c/kernels.py
```

The helper downloads only that pinned file, verifies SHA256
`56db3b46c58f8fb30912d4564189b214e2c9a37fa2138a155b45a4ce51b5c844`,
and refuses to overwrite different existing content. The runtime never
downloads code or falls back to an unverified provider.

## Logical boundary

`JointMetadata.bind` receives:

* query keys, int64 `[M,2]`: `(request_id, global_position)`;
* KV keys, int64 `[N,3]`: `(request_id, kind, global_end_position)`,
  with kind 0 for compressed rows and kind 1 for the recent window;
* compressed Top-K indices into the **original input KV row order**;
* an explicit model/cache version.

It owns its metadata, sorts KV identities canonically and remaps Top-K.
The selected order is compressed rows in increasing global position, then
the complete recent window in increasing position, then invalid padding.
It checks request ownership, causality, C4/C128 complete-group boundaries,
duplicates, capacities and window completeness. A query cannot silently
lose a window entry. Padding is exactly `(-1,-1)` with no selected KV.
Query ordering is preserved in O/LSE.

Host validation/readback occurs at admission, outside the captured graph.
Native execution cannot follow an unchecked out-of-bounds index.

## Forward record, not a hidden backward stub

```python
from rl_engine.contracts.operators.joint_attention import JointAttentionContract
from rl_engine.models.deepseek_v4.joint_attention import JointAttentionPlan, JointMetadata

c = JointAttentionContract(
    num_heads=64, head_dim=512, max_queries=32, max_kv=1024,
    compressed_topk=512, window_size=128,
)
plan = JointAttentionPlan(c)
metadata = JointMetadata.bind(
    c, query_keys, kv_keys, compressed_indices, version="step-10",
)
plan.capture()
saved = plan.forward_record(
    q, kv, sink, metadata=metadata, version="step-10",
    mode="decode", use_graph=True,
)
```

`forward_record` deliberately returns **detached, owned forward state**,
including Q, canonically ordered KV, sink, indices, O and LSE. It is **not
an autograd Function**. Item 25 supplies backward/autograd separately.
The name `mode="train"` selects the same numerical forward, not an implicit
claim that `.backward()` is available on this record's output.

The identical plan/metadata boundary is explicitly re-exported through
the vLLM and Megatron integration namespaces. This is opt-in plumbing,
not automatic model registration or an end-to-end engine signoff.

## Graph and ownership

The plan allocates fixed-capacity Q/KV/sink/index/mask/O/LSE buffers.
Capture warms the real native kernel and records that kernel plus padding
masks. Replay uses those same addresses. Results and backward records are
cloned, never returned as aliases of reusable graph output.

A per-plan lock serializes host submissions. An event after all returned
clones orders the next stream's reuse of workspace. Independent plans
have independent buffers/graphs. Failed mutable launches poison the plan;
construct a new plan instead of retrying unknown workspace state.
Metadata/saved-state mutations are rejected by version stamps.

The workspace budget covers resident plan tensors, not compiler memory,
graph-pool storage, input validation temporaries or returned record clones.
The checked API performs finite-input admission checks and copies/snapshots;
its latency must not be reported as isolated attention-kernel latency.

## Packed cache boundary

`gather_packed_joint_kv` accepts contiguous uint8
`[physical_pages, block_bytes]`, block size 64 or 128, and explicit int64
physical slots in the original logical KV input order. Block storage holds
576 bytes/token of data and eight bytes/token of scales, with optional
aligned trailing block padding. vLLM decodes only referenced pages;
Torch gathers the requested rows from those decoded pages.

The reader requires the encoder's actual `encoding="ocp"` or `"fnuz"`.
It does not infer cache encoding from the GPU. The caller owns synchronization
with external cache writers and must keep slots/data valid until the snapshot
has been read. The helper's dynamic unique-page admission is **outside**
the attention graph; it is not a captured whole-engine cache pipeline.

Train/decode byte equality is tested for the **same dequantized BF16 values**.
Unquantized training KV and lossy packed decode KV are not declared equal.

## Validated envelope and exclusions

MI300X/gfx942; D128/D512, up to 128 heads, C4/C128 metadata, at most 640
selected entries, one KV partition, O/LSE, sink-only/empty/padded cases,
native OCP/FNUZ cache reads, real Graph replay, streams and host threads.
Larger selected lists are rejected pending a separately validated profile.

Not signed off here: backward/autograd (item 25), CP/DCP exchange, multi-rank
transport, full model/engine integration, cross-profile equality, split-K
long-sequence merge, or graph-captured external cache lifetime management.
See `reports/joint-attention-mi300x-20261029.md`.
