# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""
Prefill / decode parity for the real CUDA attention kernel.

The CPU suite (``test_kv_cache_consistency.py``) validates the fixed reduction-order
*contract* with a reference implementation. This suite exercises the production
FlashAttention kernel itself: it checks that the value the kernel produces for query
position ``t`` is the same (within dtype tolerance) whether ``t`` is computed as part
of a full causal prefill or as a single-query decode step against the cached prefix.

The repository exposes only a full-sequence attention kernel; a decode step is driven
by calling that kernel with ``seqlen_q == 1`` against the keys/values seen so far. Flash
kernels reduce in a hardware-dependent order, so parity here is tolerance-based rather
than bitwise. The whole module skips cleanly when no CUDA FlashAttention kernel is
available.
"""

from __future__ import annotations

import pytest
import torch

AVAILABILITY_ERRORS = (ImportError, ModuleNotFoundError, OSError, RuntimeError)

DTYPE_CASES = [
    pytest.param(torch.float16, 1e-3, 1e-3, id="fp16"),
    pytest.param(torch.bfloat16, 2e-2, 2e-2, id="bf16"),
]

# (batch, seqlen, nheads, nheads_k, head_dim) -- includes a GQA case (nheads_k < nheads).
SHAPE_CASES = [
    pytest.param(1, 96, 4, 4, 64, id="b1-s96-mha-d64"),
    pytest.param(2, 64, 8, 2, 32, id="b2-s64-gqa-d32"),
]


def cuda_flash_attention_availability():
    if not torch.cuda.is_available():
        return False, "CUDA device is not available"
    if torch.version.hip is not None:
        return False, "current torch build is not the CUDA platform"
    try:
        from rl_engine.kernels.ops.cuda.attention.flash_attn import FlashAttentionOp

        FlashAttentionOp()
    except AVAILABILITY_ERRORS as exc:
        return False, f"CUDA FlashAttentionOp is unavailable: {exc}"
    return True, ""


def _make_qkv(batch, seqlen, nheads, nheads_k, head_dim, dtype):
    gen = torch.Generator(device="cuda").manual_seed(0)
    shape_q = (batch, seqlen, nheads, head_dim)
    shape_kv = (batch, seqlen, nheads_k, head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=dtype, generator=gen)
    k = torch.randn(shape_kv, device="cuda", dtype=dtype, generator=gen)
    v = torch.randn(shape_kv, device="cuda", dtype=dtype, generator=gen)
    return q, k, v


def _decode_replay(op, q, k, v, softmax_scale):
    """Drive a decode path with the real kernel: one query vs the cached prefix."""
    seqlen = q.shape[1]
    steps = []
    for t in range(seqlen):
        out_t = op(
            q[:, t : t + 1].contiguous(),
            k[:, : t + 1].contiguous(),
            v[:, : t + 1].contiguous(),
            softmax_scale=softmax_scale,
            causal=False,  # a single query legitimately attends to all cached keys 0..t
        )
        steps.append(out_t[:, 0])
    return torch.stack(steps, dim=1)


@pytest.mark.parametrize(("dtype", "atol", "rtol"), DTYPE_CASES)
@pytest.mark.parametrize(("batch", "seqlen", "nheads", "nheads_k", "head_dim"), SHAPE_CASES)
def test_decode_matches_prefill_cuda(dtype, atol, rtol, batch, seqlen, nheads, nheads_k, head_dim):
    available, reason = cuda_flash_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.kernels.ops.cuda.attention.flash_attn import FlashAttentionOp

    op = FlashAttentionOp()
    q, k, v = _make_qkv(batch, seqlen, nheads, nheads_k, head_dim, dtype)
    softmax_scale = 1.0 / head_dim**0.5

    prefill = op(q, k, v, softmax_scale=softmax_scale, causal=True)
    decode = _decode_replay(op, q, k, v, softmax_scale)

    torch.testing.assert_close(decode.float(), prefill.float(), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# Triton FlashAttention: prefill/decode consistency at block-aligned positions.
#
# Unlike FlashAttentionOp, the Triton kernel runs without the compiled _C
# extension, so this executes on any CUDA box. The kernel requires seqlen_q ==
# seqlen_k and a block-aligned length, so true per-token decode is not
# expressible; instead we assert the equivalent invariance: the output at a
# block-aligned position must not depend on tokens that come after it, i.e. a
# length-L causal prefill and a longer length-S prefill agree at position L-1.
# This is bitwise on real hardware (verified fp16/bf16, Blackwell).
# --------------------------------------------------------------------------- #


def triton_attention_availability():
    if not torch.cuda.is_available():
        return False, "CUDA device is not available"
    try:
        from rl_engine.kernels.ops.triton.triton_attn import triton_flash_attention  # noqa: F401
    except AVAILABILITY_ERRORS as exc:
        return False, f"Triton FlashAttention is unavailable: {exc}"
    return True, ""


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_prefill_position_invariant_to_future_tokens(dtype):
    available, reason = triton_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.kernels.ops.triton.triton_attn import triton_flash_attention

    batch, heads, head_dim, block = 2, 4, 64, 64
    total = 4 * block
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(seq):
        return torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dtype, generator=gen)

    q, k, v = randn(total), randn(total), randn(total)
    full = triton_flash_attention(q, k, v, causal=True)

    for length in range(block, total + 1, block):  # block-aligned prefixes
        prefix = triton_flash_attention(
            q[:, :, :length].contiguous(),
            k[:, :, :length].contiguous(),
            v[:, :, :length].contiguous(),
            causal=True,
        )
        boundary = length - 1
        assert torch.equal(prefix[:, :, boundary], full[:, :, boundary]), (
            f"position {boundary} changed when future tokens were added (dtype={dtype})"
        )
