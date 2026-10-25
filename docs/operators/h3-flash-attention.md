# h3-flash-attention-v1 (optional fast backend for `h3_full_attention`)

Row `h3_full_attention` was delivered by #553 (query-chunk recompute, #240 arithmetic). That path is bitwise stable but slow on
long sequences. `h3-flash-attention-v1` is an additional, optional strict backend with the same contract and a fused Triton
implementation. It does not replace or modify #553 and does not claim the RFC row.

Scope equals #553: BF16 contiguous BSHD `[B,S,56,128]`, non-causal MHA, first derivatives, Q/K already normalised/rotated.
Anything else fails closed (`ValueError` for invalid tensors, `RuntimeError` when Triton is missing; no fallback). CUDA only.

## Usage
Public entry point: `rl_engine.backends.shared.triton.attention.h3_flash.H3FlashAttentionOp` (drop-in for `H3FullAttentionOp`).

- Inputs: `q`, `k`, `v`, BF16, contiguous, shape `[B, S, 56, 128]`, on one NVIDIA CUDA device, fewer than 2**31 elements each.
- Output: BF16 `[B, S, 56, 128]`; gradients to `q`, `k`, `v` are BF16 with the same shape.
- Invalid tensors (shape, dtype, layout, device, size) raise `ValueError`; a missing Triton raises `RuntimeError`. No fallback.

```python
op = H3FlashAttentionOp()
out = op(q, k, v)        # q, k, v: BF16 [B, S, 56, 128] on CUDA, requires_grad as needed
out.backward(grad_out)
```

## Contract
- One program per (batch, head, row tile) walks key tiles in ascending `BLOCK_N=64` order; fixed tiles (fwd 128x64, bwd 64), no
  autotune, no split-KV, no atomics. A row's output depends only on its own query row and K/V, so it is batch-invariant and
  head-independent (TP / Ulysses CP shards after all-to-all give the same bytes).
- dK/dV walk query tiles ascending; dQ walks key tiles ascending; separate kernels, FP32 register accumulators, one rounding.
- `P = P_hi + P_lo` (two BF16 terms) feeds the second MMA, so probabilities are not rounded to BF16. libdevice `expf`, no TF32, no
  fast-math.
- Bitwise per architecture family (Hopper wgmma / Ampere mma.sync), same statement as `h3-projection-gemm-v1`.
- The summation order differs from #553, so the two backends are each deterministic but not bitwise equal to each other.

## Evidence (H100, `tools/validation/models/h3_flash_attention_bench.py`)
Relative error vs FP64 (seq 925, out/dq/dk/dv): flash 1.66e-3 to 1.67e-3, #553 1.66e-3, provider SDPA 2.2e-3 to 2.4e-3.

| tokens | flash fwd / fwd+bwd (ms) | #553 fwd / fwd+bwd (ms) | SDPA provider fwd / fwd+bwd (ms) |
|---|---|---|---|
| 12 | 0.08 / 0.48 | 0.16 / 0.67 | 0.04 / 0.38 |
| 925 | 0.22 / 0.77 | 5.67 / 18.6 | 0.10 / 0.49 |
| 4096 | 2.01 / 7.20 | 104.8 / 338.6 | 0.76 / 3.12 |
| 8349 | 7.96 / 29.7 | 436.7 / 1427 | 3.22 / 13.4 |

Flash is 24x (925) to 48x (8349) faster than #553 in forward+backward, and 1.3x to 2.3x slower than SDPA in
forward+backward. The SDPA provider selects among available implementations; this measurement did not establish whether the
selected one is deterministic, so no determinism claim is made for it. Timings are medians on one H100 and vary by about 10 to 30 percent
between runs at small sizes.

## Tests and tools
- `tests/backends/cuda/test_h3_flash_attention.py`: FP64 accuracy, bitwise repeat, batch 3 vs 1, head-shard invariance, fail-closed, FP32-output delta regression, int32 size guard, auxiliary-tensor validation. Skipped without NVIDIA CUDA and Triton.
- `tools/validation/models/h3_flash_attention_bench.py`: accuracy and timing against #553, SDPA and FP64.
