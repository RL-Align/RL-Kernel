# DSV4 recent-128 cache: reuse-first ROCm boundary

This implements item 12 of the DSV4 ROCm cache work, on RL-Align/RL-Kernel main,
not the historical KLR fork. Initial base: `5ba769f681b7fa765ea6b29f9939454a858390ba`.

## Scope and existing kernels

There is **no new quantization, gather, indexing, attention, or backward kernel**.

| Boundary | Reused implementation |
| --- | --- |
| Post-RoPE cache write | vLLM `cache_utils.quantize_and_insert_k_cache` |
| Read/convert paged KV | vLLM `cache_utils.dequantize_and_gather_k_cache_triton` |
| Chronological recent-window indices | vLLM `cache_utils.combine_topk_swa_indices`, `topk=0` |
| Training visible-K and its VJP | Megatron mLite `attention.runtime.quantized_main_k_visible` |
| Physical allocation and eviction | Caller/framework page allocator; RLK allocates no physical pages |

RLK adds a typed storage contract, immutable metadata binding, version/incarnation
checks, and explicit adapters. Existing attention dispatch is not globally patched.
The DSV4 model package is not a complete model implementation.

The aiter `concat_and_cache_mla(_seg)` functions are not substituted for the
mixed-dtype DS-MLA format: their uniform dtype/static-scale ABI is not equivalent.
The chosen vLLM numerical path is also the path used by the pinned Megatron
visible-K helper. Neither the FP8 conversion nor its surrogate/native VJP is
rewritten in RLK.

## Storage and numerical contract

`Recent128Contract` freezes:

- A 128-token causal window **including the current token**.
- Post-RoPE BF16 input rows of width 512; no additional RoPE inside this boundary.
- 448 FP8 elements, 64 BF16 elements, 7 real UE8M0 group-64 scales and one
  zero scale pad per row.
- 64- or 128-token physical pages, byte stride
  `round_up(block_size * 584, 576)`.
- Explicit OCP `e4m3fn` or FNUZ `e4m3fnuz` encoding. The encoder and decoder
  receive the same `use_fnuz`; architecture does not silently choose the format.
- CP1 and eager operation. Stateful Graph replay and CP ownership are rejected.
- A fixed `max_query_tokens`; gathered workspace width is
  `127 + max_query_tokens`, not a maximum derived from neighboring requests.

The Megatron native helper uses OCP/64 and is admitted only under that contract.
An incompatible contract fails **before** calling it. The helper owns its
visible-value expression and backward, including signed-zero behavior. Raw
cache equality, visible-value equality and full attention/logprob equality are
different checks; passing one is not reported as passing all of them.

The backend explicitly invokes the Triton gather instead of a dispatcher that
can select NVIDIA CuTe. Required ABI symbols and explicit-encoding parameters are
checked at construction. Missing dependencies fail without a reference fallback.
Actual module path, source SHA256, Torch and HIP versions are available through
`VllmRecent128Backend.provenance`; an installed function name alone is not BI proof.

## Minimal rollout integration

```python
from rl_engine.contracts.operators.recent128 import (
    CacheRequest, CacheVersion, Recent128Contract,
)
from rl_engine.integrations.engines.rollout.vllm.recent128 import (
    bind_recent128_metadata, recent128_cache_update,
)
from rl_engine.models.deepseek_v4.recent128 import (
    Recent128Cache, Recent128Workspace,
)

contract = Recent128Contract(max_query_tokens=256, block_size=64, encoding="e4m3fn")
version = CacheVersion(model_revision, lora_revision, scale_revision)
requests = tuple(CacheRequest(request_id, generation) for request_id, generation in handles)

# Called once per scheduler step, not once per layer.
batch = bind_recent128_metadata(
    swa_metadata, requests=requests, version=version, contract=contract,
)

# One cache object per layer, retained between steps; pages come from vLLM.
layer_cache = Recent128Cache(
    swa_cache.view(swa_cache.shape[0], -1), contract, layer_id="layer-0",
)
for request in requests:
    layer_cache.register(request, version)  # Once per request incarnation.

workspace = Recent128Workspace.allocate(batch)  # May be reused for compatible shapes.
view = recent128_cache_update(layer_cache, post_rope_kv, batch, workspace=workspace)
# view.kv: [requests, fixed_workspace_width, 512], BF16
# view.indices: [packed_rows, 128], int32, chronological flattened-workspace indices
# view.lengths: [packed_rows], int32; padded rows have length 0 and indices -1.
```

The loop above illustrates initial registration, not registration on every
decode step. At subsequent steps, rebind metadata and reuse the per-layer cache.
At cancellation/completion call `release`; reusing a request ID requires a larger
generation. A weight, LoRA or scale change requires explicit invalidation and
rebuilding, not changing the version label of existing KV.

`register(..., cached_length=N)` explicitly adopts already-populated framework
pages with an asserted version. `replay(batch)` only gathers and builds indices:
it neither writes KV nor advances length. It can be used on externally populated
pages, but is not a graph replay or a substitute for training autograd.

Do **not** invoke `update` after a native fused Q-norm/RoPE/cache insertion and
thereby write/rotate the same input twice. For that path retain the native fused
writer, bind its actual storage encoding and adopt/read its output. The
post-RoPE update adapter is the unfused boundary for item 11's output; it does not
replace the entire existing fused attention preparation pipeline.

## Scheduler and lifetime responsibilities

The existing vLLM metadata fields are consumed directly:
`block_table`, `slot_mapping`, `seq_lens`, `query_start_loc`, `block_size`.
Binding takes private tensor snapshots and validates them on the host once.
This supports vLLM inference tensors and prevents later scheduler mutations from
changing an already-bound step. Snapshot mutation is rejected. Bind outside the
layer loop and share across layers. This validation has a measured scheduler
cost; it is not advertised as a zero-cost metadata path.

Every real token's write slot must agree with its request-global logical position.
Only trailing `-1` padding is supported; holes inside a real query are rejected.
Out-of-range pages, missing live prefix pages, live physical-page aliases,
incorrect versions, duplicate updates, workspace aliasing and unsupported
configurations fail before the writer. An exception after numerical dispatch
poisons the affected request states until release/rebuild.

For a prefix of length `s` and chunk length `C`, all rows in
`[max(0, s-127), s+C)` must remain allocated until the gather finishes. In
particular, **a 128-slot ring cannot be filled with a multi-token chunk before
all queries consume it**. The fixture with old positions 0..127 and new 128/129
guards this bug. Each query still exposes at most 128 rows through its indices.

The framework must retain and own pages, perform any needed cross-stream events,
and reclaim them only after the queued gather. RLK does not pretend to detect
allocator reuse by requests absent from the current batch. Shared writable or
prefix pages are rejected without an explicit framework copy-on-write adapter.
Old pages outside the live union may be reused. No modulo-128 address manager,
copy-on-write allocator or secondary model scheduler is introduced.

The hot update/replay path performs no `.cpu()`, `.item()`, `.tolist()`, or device
synchronization. Caller-owned workspaces avoid repeated output allocations.
Views alias their workspace and expire when it is reused.

## Megatron/recompute boundary

```python
from rl_engine.integrations.engines.train.megatron.recent128 import recent128_visible_k

visible_k = recent128_visible_k(functional_post_rope_k, contract=contract)
```

This imports the existing mLite DSV4 helper and preserves its forward and VJP.
There is no mutable rollout-cache access and no hand-written autograd function.
Full Megatron mLite is an optional dependency; a plain MCore installation is not
silently treated as that runtime.

The GPU helper-parity tests can load the three **unmodified** native helper
bodies from the pinned runtime source without importing NVIDIA-only CSA modules.
That validates these helpers and this adapter, **not** a full Megatron training
run or the complete attention backward. The distinction must remain in reports.

## Verification and reproduction

```bash
python -m pytest tests/models/deepseek_v4/test_recent128.py -q
RLK_MEGATRON_DSV4_RUNTIME_SOURCE=/path/to/pinned/attention/runtime.py \
  python -m pytest tests/models/deepseek_v4/test_recent128_rocm.py -q
python benchmarks/benchmark_recent128_cache.py \
  --iterations 50 --output .codex-work/benchmark.json
```

ROCm tests fail, rather than skip, when the real GPU is present but the vLLM ABI
is missing. Native Megatron source parity is a named skip if its source path was
not supplied. GPU tests check bytes for data, scales and untouched padding;
window values and index order across full/chunked prefill and decode; mixed
batch/reordering/page permutations; invalid suffix slots; page reuse; and
native visible-K gradients. CPU tests cover metadata and lifetime failures.

The benchmark reports synchronized wall time including Python launch overhead,
separates metadata binding from the layer hot path, and measures the real upstream
writer and gather/index kernels against the contract wrapper. It is not model
throughput, a speedup claim, or an attention-kernel comparison.

Remaining integration/signoff boundaries: complete Flash-0731 model, C4/C128
compression and Top-K composition, O/LSE and raw logprob, CP>1, stateful Graph,
shared-prefix COW and multi-node tests. None are implied by item-12 cache tests.

## Source anchors

- [Current work-item grouping](https://github.com/RL-Align/RL-Kernel/issues/512)
- [vLLM decode-index reference, #60444](https://github.com/vllm-project/vllm/pull/60444)
- [vLLM sparse-decode reference, #60454](https://github.com/vllm-project/vllm/pull/60454)
- [Pinned cache utility reference](https://github.com/inaniloquentee/vllm/blob/ff4d1371ad329039c2b20bbdb80191559f8a7fa8/vllm/models/deepseek_v4/common/ops/cache_utils.py)
- [Pinned Megatron visible-K/runtime source](https://github.com/ISEEKYAN/Megatron-LM/blob/327a238239448de6771beea185dc6902a8de3eaa/experimental/lite/megatron/lite/model/deepseek_v4/vllm/primitive/attention/runtime.py)
