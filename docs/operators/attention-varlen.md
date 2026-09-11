# FlashAttention: LSE Export + Variable-Length Packing (Triton)

## Summary

Extends the existing pure-Triton FlashAttention fallback
(`rl_engine/kernels/ops/triton/triton_attn.py`) with two capabilities needed for
long-context RL rollout/training workloads:

- **Attention-domain LSE export** — the per-query-row log-sum-exp of the scaled,
  masked `QKᵀ` logits, in the same fixed online-softmax reduction order the
  kernel already used internally for the backward pass. Useful for backward
  recomputation, diagnostics, and rollout/training attention alignment checks.
  **This is not the vocab-domain LSE produced by the `fused_logp` /
  `linear_logp` kernels** — same name, different tensor domain (per key-length
  reduction vs. per-vocab reduction). Do not conflate the two.
- **Variable-length (packed) attention** — operates directly on `cu_seqlens`-packed
  `[total_tokens, H, D]` tensors instead of a padded `[B, H, S, D]` batch, so RL
  batches with wildly different response lengths (or empty/fully-masked
  responses) don't pay for padding in either compute or memory.

This module is the **cross-platform semantic baseline**: the planned SM90
WGMMA+TMA, SM80 `mma.sync`, and ROCm MFMA fused-attention kernels are checked
against it (and, transitively, against `NativeAttentionOp`) rather than against
each other. See `docs/operators/attention.md` for the pure-PyTorch WS1
ground-truth reference this whole family is validated against.

## Entry Point

```python
from rl_engine.kernels.ops.triton.triton_attn import (
    triton_flash_attention,
    triton_flash_attention_varlen,
)

# Dense — unchanged default behavior, opt into LSE:
out = triton_flash_attention(q, k, v, causal=True)                       # [B, H, S, D]
out, lse = triton_flash_attention(q, k, v, causal=True, return_lse=True)  # lse: [B, H, S] fp32

# Packed variable-length:
out = triton_flash_attention_varlen(
    q, k, v,                      # [total_q, H, D], [total_k, H, D], [total_k, H, D]
    cu_seqlens_q, cu_seqlens_k,   # int32 [batch + 1], cu_seqlens[0] == 0
    max_seqlen_q, max_seqlen_k,   # host ints, size the launch grid
    causal=True,
)
out, lse = triton_flash_attention_varlen(..., return_lse=True)           # lse: [total_q, H] fp32
```

## Backends

| Backend | Wrapper | Native symbol | Status |
| --- | --- | --- | --- |
| Triton (CUDA) | `triton_flash_attention` / `triton_flash_attention_varlen` | `_fwd_kernel[_varlen]`, `_bwd_kernel[_varlen]` | Cross-platform semantic baseline. |
| CUDA SM90 (WGMMA+TMA) | — | — | Planned; validates against this Triton path. |
| CUDA SM80 (`mma.sync`) | — | — | Planned; validates against this Triton path. |
| ROCm (MFMA) | — | — | Planned; validates against this Triton path. |

Not yet wired into `KernelRegistry` — this is the standalone kernel-development
stage (mirrors how `prefix_shared_attention` and the SM90 logp kernels were
built before registry integration). Registry dispatch is future work once a
production op_type contract for causal/varlen attention is settled.

## Tensor Contract

| Argument | Shape | Dtype | Requirements |
| --- | --- | --- | --- |
| `q` (dense) | `[B, H, Sq, D]` | fp16/bf16/fp32 | `D ∈ {16,32,64,128,256}`. |
| `k`, `v` (dense) | `[B, H, Skv, D]` | same as `q` | **No GQA**: `k`/`v` head count must equal `q`'s (matches the pre-existing dense kernel's limitation, not new). |
| `q` (varlen) | `[total_q, H, D]` | fp16/bf16/fp32 | Packed, no padding. |
| `k`, `v` (varlen) | `[total_k, H, D]` | same as `q` | Same no-GQA constraint. |
| `cu_seqlens_q`, `cu_seqlens_k` | `[batch + 1]` | int32 | Cumulative offsets, `cu_seqlens[0] == 0`. Same convention as `flash_attn_varlen_func` and this repo's `pack` op (#182). |
| `max_seqlen_q`, `max_seqlen_k` | — | Python `int` | Host-side; sizes the launch grid. |
| `causal` | — | bool | Per-sequence anchor `Skv - Sq` (see Accuracy) — identical formula for prefill (`Sq == Skv`) and decode (`Sq < Skv`). |
| `return_lse` | — | bool | If `True`, also returns the attention-domain LSE, float32, **non-differentiable** (`ctx.mark_non_differentiable`) — diagnostics/backward-recompute only, matching `flash_attn`'s `softmax_lse` external contract. |
| output | dense: `[B, H, Sq, D]`; varlen: `[total_q, H, D]` | input dtype | — |

## Dispatch Behavior

Direct function calls only (see Backends). No registry op_type yet.

## Accuracy

Both paths are checked against an **independent** fp32 masked-softmax + LSE
closed form (not against each other, so a shared online-softmax bug can't
hide):

```python
scores = einsum("hqd,hkd->hqk", q.float(), k.float()) * scale
if causal:
    mask = torch.triu(torch.ones(Sq, Skv, bool), diagonal=Skv - Sq + 1)
    scores = scores.masked_fill(mask, -inf)
probs = softmax(scores, dim=-1)
out = einsum("hqk,hkd->hqd", probs, v.float())
lse = torch.logsumexp(scores, dim=-1)
```

Measured on an H100 SXM5 (fp16 inputs, fp32 accumulation in-kernel):

- Dense: `out` max-abs-diff ≈ `1.4e-3`, `lse` max-abs-diff ≈ `1e-6`.
- Varlen: `out`/`dq`/`dk`/`dv` max-abs-diff in the `5e-4`–`1.1e-2` range across
  uneven, non-block-aligned sequence lengths; `lse` ≈ `1e-6`.

Varlen boundary handling: since packed sequences sit back-to-back in memory (no
padding), reading past a sequence's own `cu_seqlens` bound is a genuine
correctness bug (you'd read the *next* sequence's tokens), not just wasted
compute. All loads/stores in the varlen kernels are therefore explicitly masked
against `seqlen_q`/`seqlen_k` (not `boundary_check` on dynamically-shaped block
pointers) — this is deliberate and was chosen specifically to make the
padding-vs-next-sequence distinction unambiguous.

## Performance Notes

No standalone benchmark script yet (unlike `fused-logp.md` / `linear-logp.md`).
Follow-up: add `benchmarks/benchmark_attention_varlen.py` measuring
packed-vs-padded latency/VRAM at representative RL group sizes, alongside the
existing `benchmarks/benchmark_attention.py`.

## Tests

```bash
python -m pytest tests/test_triton_attention_varlen.py -v
```

Covers: dense LSE vs. independent reference, LSE non-differentiability with
backward still populating `q`/`k`/`v` grads, default-call backward
compatibility (bare tensor when `return_lse=False`), varlen forward+backward
across uneven non-block-aligned seqlens, non-causal, decode-style (`Sq < Skv`
varying per sequence), `head_dim=128`, and a zero-length sequence in the batch
(a real occurrence for fully-masked/empty RL responses).

## Known Limitations

- No GQA support on either path (`Hk` must equal `Hq`) — same constraint the
  pre-existing dense kernel already had; not introduced by this change.
- Not wired into `KernelRegistry`; no automatic backend selection yet.
- SM90 WGMMA+TMA, SM80 `mma.sync`, and ROCm MFMA kernels are not implemented —
  this page covers the Triton baseline only.
- No standalone performance benchmark yet (see Performance Notes).
