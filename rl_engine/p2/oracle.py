# SPDX-License-Identifier: Apache-2.0
"""Slow functional FP32 oracles, not alternative production GEMM/norm/RoPE kernels.

Inputs and intermediates are intentionally observable. These small CPU references
are independent of T02--T07 and never enter the shared production registry.
"""

from __future__ import annotations

import base64
import hashlib
import math

import torch
from torch import Tensor

from .contract import Status, require


def finite(*xs: Tensor) -> None:
    require(all(bool(torch.isfinite(x).all()) for x in xs), Status.NON_FINITE, "oracle input")


def tensor_record(x: Tensor) -> dict:
    x = x.detach().cpu().contiguous()
    raw = x.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(x.dtype).removeprefix("torch."),
        "shape": list(x.shape),
        "endian": "little",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def fixed_sum(x: Tensor, dim: int = -1) -> Tensor:
    """Adjacent pairs, right-zero padding, exactly FP32 at every level."""
    x = x.float().movedim(dim, -1)
    n = x.shape[-1]
    if n == 0:
        return x.sum(-1)  # Empty sum has no arithmetic order.
    width = 1 << (n - 1).bit_length()
    x = torch.cat((x, x.new_zeros(*x.shape[:-1], width - n)), -1)
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def scale(u: Tensor, *, backward: bool = False) -> Tensor:
    finite(u)
    # Global, not local, head count. Reverse association in backward.
    a, b = (128**-0.5, 64**-0.5) if backward else (64**-0.5, 128**-0.5)
    return (u.float() * a) * b


def rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
    *,
    rope_dim: int = 64,
    inverse: bool = False,
    variant: str = "gptj",
) -> Tensor:
    require(variant == "gptj", Status.INVALID_ROPE_VARIANT, variant)
    require(
        x.ndim >= 2 and x.shape[-1] in (128, 512) and rope_dim == 64,
        Status.UNSUPPORTED_CAPABILITY,
        "P2 rotary ranges",
    )
    require(
        cos.dtype == sin.dtype == torch.float32
        and cos.shape == sin.shape
        and cos.ndim == 2
        and cos.shape[1] == 32,
        Status.SCHEMA_MISMATCH,
        "FP32 cos/sin table",
    )
    require(
        positions.dtype == torch.int64
        and positions.shape == (x.shape[0],)
        and bool(((positions >= 0) & (positions < cos.shape[0])).all()),
        Status.INVALID_GLOBAL_POSITION,
        "RoPE position",
    )
    finite(x, cos, sin)
    table_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (32,)
    c, s = cos[positions].reshape(table_shape), sin[positions].reshape(table_shape)
    if inverse:
        s = -s
    part = x[..., -64:].float()
    even, odd = part[..., ::2], part[..., 1::2]
    rotated = torch.stack((even * c - odd * s, odd * c + even * s), -1).flatten(-2)
    return torch.cat((x[..., :-64], rotated.to(x.dtype)), -1)


def hadamard(x: Tensor) -> Tensor:
    require(x.shape[-1] == 128, Status.UNSUPPORTED_CAPABILITY, "H128")
    finite(x)
    y = x.float()
    stride = 1
    while stride < 128:
        chunks = y.reshape(*y.shape[:-1], -1, 2, stride)
        a, b = chunks[..., 0, :], chunks[..., 1, :]
        y = torch.cat((a + b, a - b), -1).reshape_as(y)
        stride *= 2
    return y * (128**-0.5)


def pack_mxfp4(x: Tensor) -> tuple[Tensor, Tensor]:
    """Contract E2M1 nearest-even; low nibble first; per-row blocks of 32."""
    finite(x)
    require(
        x.ndim >= 1 and x.shape[-1] % 32 == 0 and x.shape[-1] > 0,
        Status.SCHEMA_MISMATCH,
        "MXFP4 block alignment",
    )
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    # FP64 only for *selecting* the exponent, to avoid FP32 underflow at 2^-127.
    amax = blocks.double().abs().amax(-1)
    exponent = torch.ceil(torch.log2(amax.clamp_min(6 * 2.0**-126) / 6)).clamp(-127, 127)
    normalized = blocks.double() / torch.pow(2.0, exponent).unsqueeze(-1)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64, device=x.device)
    distance = (normalized.abs().unsqueeze(-1) - levels).abs()
    nearest = distance.amin(-1, keepdim=True)
    codes = torch.arange(8, device=x.device)
    # Even mantissa/code wins exact half-way ties, then ascending code.
    priority = codes + (codes % 2) * 8
    chosen = torch.where(distance == nearest, priority, 99).argmin(-1).to(torch.uint8)
    chosen = chosen | (torch.signbit(blocks).to(torch.uint8) * 8)
    flat = chosen.reshape_as(x)
    return flat[..., ::2] | (flat[..., 1::2] << 4), (exponent + 127).to(torch.uint8)


def unpack_mxfp4(packed: Tensor, scales: Tensor) -> Tensor:
    require(
        packed.dtype == scales.dtype == torch.uint8
        and packed.ndim >= 1
        and packed.shape[-1] > 0
        and packed.shape[-1] % 16 == 0
        and scales.shape == (*packed.shape[:-1], packed.shape[-1] // 16),
        Status.SCHEMA_MISMATCH,
        "MXFP4 payload shapes",
    )
    require(bool((scales != 255).all()), Status.NON_FINITE, "UE8M0 NaN")
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=packed.device)
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    blocks = values.reshape(*values.shape[:-1], -1, 32)
    result = (blocks * torch.pow(2.0, scales.float() - 127).unsqueeze(-1)).reshape_as(values)
    # A finite E2M1/UE8M0 pair can exceed FP32 range after dequantization.
    # Do not silently saturate or allow an infinity into the reference path.
    finite(result)
    return result


def pool(k: Tensor, logits: Tensor) -> tuple[Tensor, Tensor]:
    """FP32 per-channel pool, token axis -2; -inf allowed only as an invalid slot."""
    require(
        k.shape == logits.shape and k.ndim >= 2 and k.shape[-2] > 0,
        Status.SCHEMA_MISMATCH,
        "pool shapes",
    )
    finite(k)
    require(
        bool((torch.isfinite(logits) | torch.isneginf(logits)).all())
        and bool(torch.isfinite(logits).any(-2).all()),
        Status.NON_FINITE,
        "pool needs one finite slot per channel",
    )
    e = torch.exp(logits.float() - logits.float().amax(-2, keepdim=True))
    alpha = e / fixed_sum(e, -2).unsqueeze(-2)
    return fixed_sum(alpha * k.float(), -2), alpha


def c4_pool(k: Tensor, scores: Tensor, ape: Tensor) -> tuple[Tensor, Tensor]:
    """[groups,4,2,D]: previous FIRST half, current SECOND half; APE first."""
    require(
        k.shape == scores.shape
        and k.ndim == 4
        and k.shape[1:3] == (4, 2)
        and ape.shape == k.shape[1:],
        Status.SCHEMA_MISMATCH,
        "C4 [G,4,2,D]",
    )
    finite(k, scores, ape)
    a = scores.float() + ape.float()
    previous_k = torch.cat((torch.zeros_like(k[:1, :, 0]), k[:-1, :, 0]), 0)
    previous_a = torch.cat((torch.full_like(a[:1, :, 0], -math.inf), a[:-1, :, 0]), 0)
    return pool(torch.cat((previous_k, k[:, :, 1]), 1), torch.cat((previous_a, a[:, :, 1]), 1))


def c128_pool(k: Tensor, scores: Tensor, ape: Tensor) -> tuple[Tensor, Tensor]:
    require(
        k.shape == scores.shape and k.ndim == 3 and k.shape[1] == 128 and ape.shape == k.shape[1:],
        Status.INVALID_COMPRESSION_PLAN,
        "C128 full group",
    )
    finite(k, scores, ape)
    return pool(k, scores.float() + ape.float())


def icv(q: Tensor, k: Tensor) -> Tensor:
    require(
        q.ndim == 3 and q.shape[1:] == (64, 128) and k.ndim == 2 and k.shape[1] == 128,
        Status.SCHEMA_MISMATCH,
        "ICV shapes",
    )
    finite(q, k)
    # Contractually separate inner 32-element and outer four-block trees.
    product = q.float().unsqueeze(-2) * k.float()
    return fixed_sum(fixed_sum(product.reshape(*product.shape[:-1], 4, 32), -1), -1)


def relu_score(a: Tensor, w: Tensor) -> Tensor:
    require(
        a.ndim == 3 and a.shape[1] == 64 and w.shape == a.shape[:2],
        Status.SCHEMA_MISMATCH,
        "score shapes",
    )
    finite(a, w)
    return fixed_sum(torch.relu(a.float()) * w.float().unsqueeze(-1), 1)


def topk512(scores: Tensor, valid: Tensor, global_ids: Tensor) -> tuple[Tensor, Tensor]:
    require(
        scores.ndim == 2
        and valid.shape == scores.shape
        and valid.dtype == torch.bool
        and global_ids.shape == (scores.shape[1],)
        and global_ids.dtype == torch.int64,
        Status.SCHEMA_MISMATCH,
        "Top-K shapes/dtypes",
    )
    finite(scores)
    ids = global_ids.cpu().tolist()
    require(
        len(ids) == len(set(ids)) and all(i >= 0 for i in ids),
        Status.AMBIGUOUS_LOGICAL_INDEX,
        "global ids must be unique/nonnegative",
    )
    output = torch.full((scores.shape[0], 512), -1, dtype=torch.int64, device=scores.device)
    counts = torch.zeros(scores.shape[0], dtype=torch.int64, device=scores.device)
    for row, (ss, vv) in enumerate(
        zip(scores.detach().cpu().tolist(), valid.cpu().tolist(), strict=False)
    ):
        selected = sorted(((-s, i) for s, v, i in zip(ss, vv, ids, strict=False) if v))[:512]
        counts[row] = len(selected)
        if selected:
            output[row, : len(selected)] = torch.tensor(
                [i for _, i in selected], device=scores.device
            )
    return output, counts


def joint_attention(q: Tensor, kv: Tensor, sink: Tensor) -> tuple[Tensor, dict]:
    require(
        q.ndim == 2
        and q.shape == (64, 512)
        and kv.ndim == 2
        and kv.shape[1] == 512
        and sink.shape == (64,),
        Status.SCHEMA_MISMATCH,
        "MQA shapes",
    )
    finite(q, kv, sink)
    logits = fixed_sum(q.float().unsqueeze(1) * kv.float(), -1) * (512**-0.5)
    maximum = torch.cat((logits, sink.float().unsqueeze(-1)), -1).amax(-1, keepdim=True)
    e = torch.exp(logits - maximum)
    es = torch.exp(sink.float().unsqueeze(-1) - maximum)
    z = es + fixed_sum(e, -1).unsqueeze(-1)
    p = e / z
    output = fixed_sum(p.unsqueeze(-1) * kv.float(), 1)
    return output, {"logits": logits, "m": maximum, "Z": z, "p": p, "p_sink": es / z}


def joint_attention_backward(q: Tensor, kv: Tensor, saved: dict, grad: Tensor) -> dict:
    p, ps = saved["p"], saved["p_sink"]
    dp = fixed_sum(grad.float().unsqueeze(1) * kv.float(), -1)
    mu = fixed_sum(p * dp, -1)
    dl = p * (dp - mu.unsqueeze(-1))
    dq = fixed_sum(dl.unsqueeze(-1) * kv.float(), 1) * (512**-0.5)
    dk = fixed_sum(dl.unsqueeze(-1) * q.float().unsqueeze(1), 0) * (512**-0.5)
    dv = fixed_sum(p.unsqueeze(-1) * grad.float().unsqueeze(1), 0)
    return {"dQ": dq, "dKV": dk + dv, "dsink": -ps.squeeze(-1) * mu}
