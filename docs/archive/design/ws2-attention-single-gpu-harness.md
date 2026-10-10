# WS2 Attention Single-GPU Comparison Harness

Status: PR2 harness for [#235](https://github.com/RL-Align/RL-Kernel/issues/235)

## Scope

This harness compares attention materializations on one device before CP
communication is introduced. It is diagnostic infrastructure: it does not launch
collectives and does not replace the deterministic CP reference planned in PR3.

Implemented paths:

- `full_prefill`: training-style full-sequence softmax attention;
- `chunked_prefill`: rollout-style query chunk replay over full KV;
- `rl_kernel_paged_kv`: rollout-style KV page replay with fp32 attention-domain
  LSE merge by logical KV block order;
- `transformer_engine_paged_kv`: optional oracle that reuses NVIDIA Transformer
  Engine's context-parallel PyTorch correction helpers when TE is installed.

RoPE scope:

- `unfused_rope_attention`: canonical `RoPE -> Attention` path;
- `fused_like_rope_attention`: semantic `RoPE+Attention` path that applies the
  same canonical RoPE rules before attention, then records the fused boundary in
  provenance.

The RoPE path is still single-GPU attribution. It proves that both sides agree
on post-RoPE Q/K, `out`, attention-domain `lse`, and optional active-token
`dlogp` before CP communication or production fused kernels are introduced.

## Report

`rl_engine.validation.common.attention_comparison.compare_single_gpu_attention` emits a
structured report with:

- `out` max / mean / p95 / p99 absolute drift;
- attention-domain `lse` max / mean / p95 / p99 absolute drift;
- optional active-token-only `dlogp` drift when `lm_head_weight`, `target_ids`,
  and an active token mask are provided;
- per-path provenance including chunk/page sizes, KV page bounds, merge backend,
  merge order, and LSE domain;
- optional-backend unavailability reasons.

`compare_single_gpu_rope_attention` emits the same drift schema and additionally
reports post-RoPE Q/K drift. Its provenance records:

- Q/K state as `post_rope`;
- `position_ids` shape and range;
- `rope_theta`, `rotary_dim`, `rope_cast_at`, and `rope_output_dtype`;
- `fusion_boundary` as either `unfused_rope_attention` or
  `fused_rope_attention`.

The selected-logprob convention follows #207:

```text
dlogp = candidate selected logp - full_prefill selected logp
```

## Transformer Engine Reuse

The harness does not make Transformer Engine a runtime dependency. When
available, it lazily imports:

```text
transformer_engine.pytorch.attention.dot_product_attention.context_parallel
```

and calls:

```text
flash_attn_fwd_softmax_lse_correction
flash_attn_fwd_out_correction_init
flash_attn_fwd_out_correction
```

Those helpers provide an industrial implementation oracle for the same fp32
`(out, lse)` online-softmax merge policy that later CP/fused paths must match.
When TE is not installed, the TE path is reported as unavailable and the local
RL-Kernel paths still run.

## CLI Registration

The existing generic operator harness now registers `attention`, so a local
candidate smoke can run with:

```bash
python tools/validation/operators/check_operator.py --op attention --candidate pytorch --dtype fp32
```

The attention-specific WS2 comparison entry point is Python-first for now:

```python
from rl_engine.validation.common.attention_comparison import (
    AttentionComparisonInputs,
    compare_single_gpu_rope_attention,
    compare_single_gpu_attention,
)

report = compare_single_gpu_attention(
    AttentionComparisonInputs(q=q, k=k, v=v, target_ids=target_ids, lm_head_weight=w),
    query_chunk_size=512,
    kv_page_size=512,
    include_transformer_engine=True,
)
print(report.to_dict())

rope_report = compare_single_gpu_rope_attention(
    AttentionComparisonInputs(
        q=q,
        k=k,
        v=v,
        rope_positions=torch.arange(q.size(2), device=q.device),
        target_ids=target_ids,
        lm_head_weight=w,
    )
)
print(rope_report.to_dict())
```

## Validation

```bash
python -m pytest tests/validation/common/test_attention_comparison.py -q
```

The tests cover full vs chunked/paged equivalence, active-token `dlogp` drift,
optional TE correction-helper reuse through a fake TE module, JSON-compatible
reports, RoPE+Attention post-RoPE Q/K attribution, and `attention` registration
in the generic operator comparison specs.
