# WS2 PR7 Fused Attention Backend Alignment

Status: PR7 candidate plus deterministic communication integration for #235

## Goal

PR7 is the production-backend alignment layer after PR2/PR3/PR6 have defined
the reference semantics.  It evaluates fused prefill/decode candidates while
keeping RL-Kernel's reference contract as the source of truth.

The full PR7 plan has two backend lanes:

| Lane | Role | Status in this scaffold |
| --- | --- | --- |
| Training full-prefill candidate | Evaluate TE public `DotProductAttention` / fused attention for training-style full prefill | Planned; not implemented here |
| Rollout paged prefill/decode candidate | Evaluate FlashInfer paged attention with Qwen3 RoPE fusion, split-KV policy, LSE export, and batch-invariant sweep | Implemented as opt-in scaffold |

This file therefore documents both the original PR7 requirements and the new
FlashInfer/RoPE/split-KV/batch-invariant additions.  It should not be read as a
claim that FlashInfer or TE replaces the deterministic reference.

## Source Of Truth

The correctness source remains:

```text
AttentionContract
PR2 single-GPU full/chunked/paged attention reference
PR3 deterministic CP reference and global_block_index merge
PR6 decode paged-KV replay vs full logical KV reference
attention-domain lse
dlogp drift report
```

PR7 backends are candidates.  A candidate can be promoted only if it exports or
reconstructs the states required by the contract and passes the drift gates.

## Original PR7 Requirements

| Requirement | PR7 rule |
| --- | --- |
| Backend selection | Record `requested_backend`, `actual_backend`, `fallback`, and `fallback_reason`; no silent fallback. |
| Training path | Training-side forward is full-prefill / teacher-forcing.  TE may be evaluated through its public `DotProductAttention` path, not by redefining RL-Kernel semantics. |
| Rollout path | Rollout-side forward is paged KV prefill/decode.  FlashInfer is a rollout candidate, not a training backward backend. |
| LSE | Candidate must return attention-domain `lse` shaped as `[B, Hq, Sq]`. |
| CP/TP communication | PR7 preserves the P2P NCCL reference and maps the production `cuda_ag_rs` path to the deterministic CUDA AllGather/ReduceScatter operators from PR311/PR312. Missing compiled operators fail closed. |
| CP order | CP correctness remains PR3's `global_block_index` merge.  A black-box backend that only returns final output must be validated against PR3/PR6; it cannot define CP merge order. |
| Paged KV | Page table, `cache_position`, `query_position_ids`, `key_position_ids`, and logical token order must be validated before execution. |
| Precision | Record accumulation/downcast policy.  BF16 output is acceptable only after FP32/reference drift is reported. |
| Backward | Training backward belongs to PR8.  PR7 forward results must not claim `dq/dk/dv` alignment. |

## Current Implementation

Implemented files:

```text
rl_engine/backends/cuda/attention/cp_comm.py
rl_engine/backends/cuda/attention/flashinfer_paged_attention.py
tests/backends/cuda/test_flashinfer_pr7_attention.py
tools/validation/distributed/ws2_pr7_flashinfer_attention_check.py
```

The FlashInfer adapter is opt-in and lazy-imported.  Importing RL-Kernel does not
require FlashInfer.  Real FlashInfer execution requires CUDA tensors; CPU tests
use a fake wrapper only to validate parameter binding and provenance.

Current code path:

```text
FlashInferQwen3PagedAttentionOp.forward(...)
  validate q / k_cache / v_cache
  validate Qwen3 RoPE fusion config
  validate split-KV policy
  validate CP=2 / TP=2 AG/RS communication interface contract
  if strict_mode and require_cp_comm:
    validate owner-local Q/K/V and real position IDs
    deterministic AG(Q/K/V/position IDs)
    apply WS1 RoPESM90Op row-by-row
    run the shared WS1 no-Split-K deterministic Attention core
    deterministic RS(Out, LSE) -> local query shard
    return without constructing a FlashInfer arithmetic wrapper
  build_flashinfer_paged_kv_plan(...)
    validate page bounds
    validate block_table/global_token_positions logical order
    validate key_position_ids
    reject duplicate pages and non-canonical inactive metadata
  bind metadata q/k RoPE state to the fused-RoPE config
  validate cache_position == query_position_ids and trailing query positions
  validate prefix-cache fingerprint against K/V content and RoPE identity
  materialize_flashinfer_paged_kv_cache(...)
  if strict_mode:
    materialize logical paged KV and run the shared WS1 deterministic core
  else:
    flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper
    or flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
  wrapper.plan(..., pos_encoding_mode="ROPE_LLAMA", rope_theta=1e6, rope_scale=1.0, ...)
  wrapper.run_return_lse(...)
  restore out -> [B, Hq, Sq, D]
  restore lse -> [B, Hq, Sq]
```

## CP/TP Communication Interface

Issue #235 targets `Qwen3-8B, TP=2, CP=2, BF16`. PR7 surfaces the distributed
attention boundary and reuses the self-owned deterministic CUDA collectives.

The exposed interface is:

```text
AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
AttentionCPCommunicationPlan(backend="cuda_ag_rs", status="implemented")
AttentionCPPartialState(out, lse, AttentionCPBlockMetadata(...))
CUDAAGRSAttentionCPCommunication.all_gather_query(...)
CUDAAGRSAttentionCPCommunication.all_gather_kv(...)
CUDAAGRSAttentionCPCommunication.all_gather_position_ids(...)
CUDAAGRSAttentionCPCommunication.all_gather_partial_states(...)
CUDAAGRSAttentionCPCommunication.reduce_scatter_merged_state(...)
CUDAAGRSAttentionCPCommunication.reduce_scatter_strict_result(...)
sort_attention_cp_partial_states(..., plan=...)
```

The CP execution order is explicit and keeps compute and communication
decoupled:

```text
strict production:
  local Q/K/V + position IDs
  -> custom CUDA AG(Q/K/V/position IDs)
  -> shared full-logical-QKV WS1 deterministic Attention core
  -> custom CUDA RS(Out, LSE)
  -> local query shard

reference compatibility path:
  local Q shard -> custom AG(Q)
  -> owner-local partial AttentionCPState(out, lse, global_block_index)
  -> custom AG(partial states) -> PR3 FP32 ordered merge
  -> custom RS(merged state)
```

`CUDAAGRSAttentionCPCommunication` uses the self-owned deterministic CUDA
collectives from PR311/PR312 for strict Q/K/V/position AG and the final Out/LSE
RS. `P2PNCCLAttentionCPCommunication` implements the same strict boundary as
an independent NCCL reference. The old partial-state methods remain available
for PR3/reference validation, but strict mode never labels a full-KV result as
an owner-local partial or enters native FlashInfer Attention arithmetic.

Ordering is part of the interface:

```text
merge_key = AttentionCPBlockMetadata.global_block_index
accum_dtype = fp32
required_state = (out, lse)
communication_pattern = ag_rs
compute_communication = decoupled
duplicate global_block_index -> error
```

Strict production disables Split-KV in the shared CUDA core. This gives CP=1
and CP=2 the same full-QKV kernel grid and reduction graph. Fixed/auto
Split-KV remains available only on the diagnostic/reference FlashInfer lane;
it cannot claim strict bitwise identity.

## FlashInfer RoPE Fusion

The implemented fused boundary is:

```text
pre-RoPE Q, pre-RoPE K cache, V
  -> FlashInfer paged attention with pos_encoding_mode="ROPE_LLAMA"
  -> out, attention-domain lse
```

The accepted Qwen3 settings are locked to:

```text
pos_encoding_mode = "ROPE_LLAMA"
rope_theta        = 1_000_000.0
rope_scale        = 1.0
rotary_dim        = head_dim
layout            = Qwen3 rotate-half / non-interleaved
```

The adapter rejects post-RoPE Q/K in both the config and the actual runtime
metadata. It also requires `cache_position == query_position_ids` and trailing
contiguous query positions, because this adapter currently relies on the
wrapper's implicit RoPE positions. That avoids silent double rotation and
position drift. If a later rollout path stores post-RoPE K or supports arbitrary
query positions, it must be represented as a separate capability with explicit
position tensors.

Prefix-cache identity binds logical K/V content, storage dtypes, cached-K RoPE
state, theta, rotary dim, cast boundary, and output dtype. Equivalent physical
page placement produces the same logical identity; stale content or a changed
RoPE configuration fails before execution.

## Split-KV Policy

PR7 exposes split-KV as a contract/provenance knob instead of inheriting
backend defaults:

| Policy | FlashInfer plan kwargs | Batch-invariant status |
| --- | --- | --- |
| `disabled` | `disable_split_kv=True` | strict candidate |
| `fixed:<N>` | `fixed_split_size=N`, `disable_split_kv=False` | candidate; must pass drift sweep |
| `auto` | `disable_split_kv=False` | rejected when batch invariance is required |

RL-Kernel's fixed size is measured in logical KV tokens. FlashInfer 0.6 names
`fixed_split_size` in physical pages, so PR7 accepts a fixed token size only
when it is divisible by `page_size`, passes `fixed_split_size / page_size` to
the backend, and converts page boundaries back to logical token boundaries in
the report. Runtime callbacks must declare `split_size_unit="pages"` and
`boundary_unit="pages"`; otherwise strict provenance fails closed.

The shared WS2 contract now treats Split-KV as a first-class semantic field. PR7 still
must export the backend's actual token boundaries; a requested FlashInfer knob alone is
not sufficient evidence of train/rollout equivalence.
Once PR1/PR4 add a first-class field, the PR7 provenance can be wired into that
field without changing the backend adapter.

## Batch-Invariant Validation

PR7 distinguishes "configured for batch invariance" from "proven batch
invariant":

```text
configured:
  split-KV disabled or fixed
  no auto backend scheduling in the contract
  page/order metadata validated

proven:
  same sample alone vs same sample inside a batch has zero or tolerated drift
  out and lse both reported
  real CUDA/H-card run, not fake wrapper
```

The validation script reports:

```text
batch_invariant_sweep.method = single_row_vs_same_row_inside_batch
batch_invariant_sweep.out_max_abs
batch_invariant_sweep.lse_max_abs
page_layout_invariant_sweep.out.max_abs
page_layout_invariant_sweep.lse.max_abs
drift.out.{max_abs,mean_abs,p95_abs,p99_abs}
drift.lse.{max_abs,mean_abs,p95_abs,p99_abs}
drift.dlogp.{max_abs,mean_abs,p95_abs,p99_abs}
```

FlashInfer is not declared batch-invariant by default.  It becomes an accepted
candidate only if the real-hardware sweep passes under the selected split-KV
policy.

## TE Relationship

The TE lane is still part of the full PR7 plan, but it is not implemented in
this scaffold.

| TE use | PR7 position |
| --- | --- |
| TE CP correction helpers | Already used by PR2/PR3/PR6 as optional merge oracle. |
| TE full fused attention | Future PR7 training full-prefill candidate through public `DotProductAttention`. |
| TE internal CP black box | Not a source of RL-Kernel `global_block_index` order unless it exposes compatible partial states or passes reference drift gates. |
| TE backward | PR8 only, and only if compatible saved forward/backward state is available. |

FlashInfer and TE can coexist in PR7:

```text
training candidate:
  TE DotProductAttention full prefill

rollout candidate:
  FlashInfer ROPE_LLAMA paged prefill/decode

shared gate:
  compare both against RL-Kernel reference states and selected-logprob drift
```

## Relation To Existing PRs

| PR | Relationship |
| --- | --- |
| PR1 | Carries the strict Split-KV policy and capability requirements; runtime adapters still export actual boundaries and fallback provenance. |
| PR2 | Provides full/chunked/paged single-GPU references and fused-like RoPE attribution. PR7 uses the same semantic boundary, but calls a real backend candidate. |
| PR3 | Owns CP=1/2 deterministic semantics and `global_block_index` merge. PR7 does not replace this. |
| PR4 / #263 | Records actual backend and split-KV provenance. PR7 matches that policy and can later wire into PR4 fields. |
| PR5 | Should run the cross matrix once PR7 backends exist: full/chunked/paged, split-KV disabled/fixed, batch shape, TP/CP, dtype. |
| PR6 / #260 | Provides decode paged-KV replay and full logical KV reference. PR7 consumes PR6-style metadata and validates the same page/position identity before FlashInfer execution. |
| PR8 | Owns training backward alignment. PR7 forward-only results are not backward claims. |

## Conflict Check

| Topic | Possible conflict | PR7 resolution |
| --- | --- | --- |
| RoPE fusion vs PR2/PR6 post-RoPE references | A fused backend might hide post-RoPE Q/K state. | PR7 records `rope_fusion_boundary`, rejects post-RoPE inputs for this path, and compares final `out/lse` against references. |
| FlashInfer split-KV vs PR3 CP merge order | FlashInfer internal split-KV order is not PR3 `global_block_index`. | Treat split-KV as backend-local reduction.  CP merge remains PR3; FlashInfer must pass drift gates and cannot define CP order. |
| CP=2/TP=2 target vs communication ordering | The issue requires actual distributed semantics, not an interface claim. | `CUDAAGRSAttentionCPCommunication` executes PR311/PR312, keeps FP32 partial states, sorts the authoritative manifest by `global_block_index`, and fails closed if the compiled collective is absent. |
| Batch-invariant claim vs backend heuristics | Auto split-KV may depend on batch composition/runtime scheduling. | Reject `auto` when `require_batch_invariant=True`; report disabled/fixed policy explicitly. |
| TE training lane vs FlashInfer rollout lane | Two libraries may have different materialization boundaries. | Both bind to the same RL-Kernel semantic contract and are judged by shared `out/lse/dlogp` reports. |
| Requested policy vs actual plan | A requested fixed/max-splits knob may not equal the runtime token boundaries. | Require actual runtime plan callbacks in strict fixed mode; fail closed when unavailable. |
| Decode vs training full prefill | Training does not own a persistent paged KV cache. | PR7 compares rollout paged KV decode/prefill to training-style full logical KV reference, not to a nonexistent training decode cache. |

No direct conflict was found between the three new points and the original PR7
plan.  The important restriction is that FlashInfer split-KV/RoPE fusion remains
a candidate path until real CUDA/H-card drift reports prove it.

## Validation

CPU-safe structural tests:

```bash
pytest tests/backends/cuda/test_flashinfer_pr7_attention.py -q
```

Dry-run plan/provenance checks without CUDA or FlashInfer:

```bash
python tools/validation/distributed/ws2_pr7_flashinfer_attention_check.py --dry-run --json
python tools/validation/distributed/ws2_pr7_flashinfer_attention_check.py --dry-run --mode prefill --query-len 16 --json
```

CUDA/H-card validation once hardware is available:

```bash
python tools/validation/distributed/ws2_pr7_flashinfer_attention_check.py \
  --no-dry-run \
  --device cuda \
  --mode decode \
  --split-kv-policy disabled \
  --output artifacts/pr7-decode-disabled.json \
  --json

python tools/validation/distributed/ws2_pr7_flashinfer_attention_check.py \
  --no-dry-run \
  --device cuda \
  --mode prefill \
  --query-len 16 \
  --split-kv-policy fixed \
  --fixed-split-size 4 \
  --output artifacts/pr7-prefill-fixed.json \
  --json
```

The CUDA report must include:

```text
passed / errors
drift.out / drift.lse / drift.dlogp
batch_invariant_sweep
page_layout_invariant_sweep
rope_fusion_boundary
split_kv_policy
actual_split_kv_plans
actual_split_kv_plan_set
arithmetic_semantics_verified
actual_backend
fallback / fallback_reason
```

## Non-Claims

This candidate does not enable FlashInfer by default, does not implement the TE
training lane, does not replace PR3 CP merge, does not prove real H-card
batch-invariance locally, and does not implement training backward.
# P2P NCCL reference and strict arithmetic provenance

The existing `CUDAAGRSAttentionCPCommunication` interface is preserved and now
uses the self-owned deterministic CUDA AG/RS implementation. It remains
fail-closed when PR311/PR312 or their compiled symbols are absent. For GPU
reference validation, `P2PNCCLAttentionCPCommunication` implements the same partial-state
and Q-AG protocol with `torch.distributed.batch_isend_irecv` on an NCCL process
group.
It requires an authoritative manifest containing every logical KV block,
token range, CP owner, TP owner, and query scatter range.  Missing blocks,
gaps, overlaps, wrong owners, incomplete gathers, non-NCCL groups, and CPU
tensors are rejected before merge.

Strict FlashInfer validation also requires runtime callbacks for both:

- the actual Split-KV token boundaries for every request; and
- arithmetic provenance declaring FP32 accumulation, FP32 LSE, and downcast
  only at the final output write.

Requested knobs, maximum split counts, or labels such as
`flashinfer_internal` are not accepted as proof.

The repository pins the PR7 lane to `flashinfer-python>=0.6.0,<0.7`; older
versions do not expose both `fixed_split_size` and `disable_split_kv` plan
knobs. Upstream 0.6 wrappers still do not expose RL-Kernel's three strict
provenance callbacks, so an unpatched stock wrapper is expected to fail closed.
Passing strict acceptance requires a small wrapper/adapter that reads the actual
plan information returned by FlashInfer and reports it through the documented
callbacks; the requested plan arguments alone are deliberately insufficient.

Run the two-GPU transport check with:

```bash
torchrun --standalone --nproc-per-node=2 \
  tools/validation/distributed/ws2_p2p_nccl_attention_reference_check.py
```

The validation CLI materializes the TP-local Qwen3-8B shard. For the default
`TP=2` target this is `Hq=16`, `Hkv=4`, `D=128`; `32/8` are global model head
counts and are rejected as a mislabeled local execution. Any non-finite drift
statistic also fails strict acceptance.
