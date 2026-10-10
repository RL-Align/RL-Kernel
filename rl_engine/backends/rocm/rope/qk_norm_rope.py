# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fuse the pinned eager Q/K RMSNorm and FP32-table RoPE during decode.

The intermediate normalized values are rounded to BF16 before rotation,
exactly as in the separate operators. Unsupported inputs retain those operators.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rl_engine.backends.rocm.norm import rmsnorm
from rl_engine.backends.rocm.rope.rope import RocmDeterministicRoPEOp


@triton.jit
def _qk_norm_rope(
    Q,
    K,
    WQ,
    WK,
    P,
    C,
    S,
    OQ,
    OK,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    EQ: tl.constexpr,
    EK: tl.constexpr,
):
    row = tl.program_id(0)
    token, head = row // (HQ + HK), row % (HQ + HK)
    is_query = head < HQ
    X = Q if is_query else K
    W = WQ if is_query else WK
    output = OQ if is_query else OK
    head = head if is_query else head - HQ
    stride = SQ if is_query else SK
    heads = HQ if is_query else HK
    eps = EQ if is_query else EK
    base = token * stride + head * 128
    # Same four-element accumulators and warp tree as eager PyTorch 2.12.
    thread = tl.arange(0, 256)
    acc = tl.full((256,), 0.0, tl.float32)
    for element in tl.static_range(4):
        offset = thread * 4 + element
        value = tl.load(X + base + offset, offset < 128, 0).to(tl.float32)
        acc = acc + value * value
    for shift in tl.static_range(5, -1, -1):
        peer = (thread // 64) * 64 + ((thread % 64 + (1 << shift)) % 64)
        acc = acc + tl.gather(acc, peer, 0)
    warps = tl.gather(acc, tl.arange(0, 4) * 64, 0)
    variance = tl.sum(tl.sum(tl.reshape(warps, (2, 2)), 0), 0) / 128
    rstd = libdevice.rsqrt(variance + eps)
    offset = tl.arange(0, 64)
    low = tl.load(X + base + offset).to(tl.float32)
    high = tl.load(X + base + offset + 64).to(tl.float32)
    low = (tl.load(W + offset).to(tl.float32) * (rstd * low)).to(tl.bfloat16).to(tl.float32)
    high = (tl.load(W + offset + 64).to(tl.float32) * (rstd * high)).to(tl.bfloat16).to(tl.float32)
    position = tl.load(P + token)
    cosine = tl.load(C + position * 64 + offset)
    sine = tl.load(S + position * 64 + offset)
    output_base = (token * heads + head) * 128
    tl.store(output + output_base + offset, low * cosine - high * sine)
    tl.store(output + output_base + offset + 64, high * cosine + low * sine)


@torch.library.custom_op("rl_kernel::strict_qk_norm_rope_rocm", mutates_args=())
def strict_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    query_eps: float,
    key_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fast = (
        query.ndim == key.ndim == 3
        and query.shape[-1] == key.shape[-1] == 128
        and query.shape[0] == key.shape[0]
        and query.stride(1) == key.stride(1) == 128
        and rmsnorm.supports(query, query_weight, None)
        and rmsnorm.supports(key, key_weight, None)
    )
    positions = positions.reshape(-1).to(dtype=torch.int64).contiguous()
    if not fast:
        op = RocmDeterministicRoPEOp()
        outputs = []
        for value, weight, eps in ((query, query_weight, query_eps), (key, key_weight, key_eps)):
            normalized = torch.nn.functional.rms_norm(value, (value.shape[-1],), weight, eps)
            rotated = op.forward_token_major(
                normalized.reshape(value.shape[0], -1),
                positions,
                cosine,
                sine,
                head_dim=value.shape[-1],
            )
            outputs.append(rotated.reshape(value.shape))
        return outputs[0], outputs[1]
    output_q = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    output_k = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    _qk_norm_rope[(query.shape[0] * (query.shape[1] + key.shape[1]),)](
        query,
        key,
        query_weight,
        key_weight,
        positions,
        cosine,
        sine,
        output_q,
        output_k,
        query.shape[1],
        key.shape[1],
        query.stride(0),
        key.stride(0),
        query_eps,
        key_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output_q, output_k


@strict_qk_norm_rope.register_fake
def _fake(query, key, query_weight, key_weight, positions, cosine, sine, query_eps, key_eps):
    return (
        torch.empty(query.shape, dtype=query.dtype, device=query.device),
        torch.empty(key.shape, dtype=key.dtype, device=key.device),
    )
