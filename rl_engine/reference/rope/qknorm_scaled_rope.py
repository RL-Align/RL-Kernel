# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer F2 model and independent FP64 mathematical references.

These are eager references, not strict fixed-tree kernels. CPU table construction
is useful for tests; production strict execution requires a shared CUDA-built table.
"""

from __future__ import annotations

import torch
from torch import Tensor

HEAD_DIM = 128
EPS = 1e-5
Q_SCALE = 3.87
THETA = 500000.0


def fixed_sum_128(values: Tensor) -> Tensor:
    """FP32 half-split tree v1; output [..., 1], preserving leading dimensions.

    Stages pair i with i+64, then i+32, ..., i+1. Run eagerly: compiler
    reassociation is outside this reference contract. No cross-row reduction.
    """
    if values.dtype != torch.float32 or values.ndim < 1 or values.shape[-1] != 128:
        raise ValueError("fixed_sum_128 requires FP32 [..., 128]")
    acc = values
    for half in (64, 32, 16, 8, 4, 2, 1):
        acc = acc[..., :half] + acc[..., half:]
    return acc


def fixed_dot_128(a: Tensor, b: Tensor) -> Tensor:
    """Separate FP32 products followed by the same tree, used by the declared VJP."""
    if a.shape != b.shape or a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("fixed_dot_128 requires matching FP32 tensors")
    return fixed_sum_128(a * b)


def _validate(q: Tensor, k: Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("Q/K must have shape [T, H, 128]")
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
        raise ValueError("head dimension must be 128")
    if q.shape[0] != k.shape[0] or q.shape[1] < 1 or k.shape[1] < 1:
        raise ValueError("Q/K require equal token counts and positive head counts")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Q/K must be BF16")
    if q.device != k.device:
        raise ValueError("Q/K must be on the same device")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q/K must be contiguous")


def _validate_positions(positions: Tensor | None, q: Tensor, limit: int) -> Tensor:
    if positions is None:
        raise ValueError("local layers require positions")
    if positions.dtype != torch.int64 or positions.shape != (q.shape[0],):
        raise ValueError("positions must be int64 [T]")
    if positions.device != q.device:
        raise ValueError("positions must be on the Q/K device")
    if positions.numel() and bool(((positions < 0) | (positions >= limit)).any()):
        raise ValueError("positions outside table or model context")
    return positions


@torch.no_grad()
def build_rope_table(length: int, *, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    """Build FP32 cos/sin and round to BF16, storing only 64 unique channels.

    This reference helper does not establish a production table hash/algorithm contract.
    """
    if not 0 <= length <= 131072:
        raise ValueError("table length must be in [0, 131072]")
    exponent = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device) / HEAD_DIM
    inv_freq = 1.0 / (THETA**exponent)
    angles = torch.arange(length, dtype=torch.float32, device=device)[:, None] * inv_freq
    return angles.cos().bfloat16(), angles.sin().bfloat16()


def _norm(x: Tensor, *, fixed_tree: bool = False) -> Tensor:
    x32 = x.float()
    if fixed_tree:
        mean = fixed_sum_128(x32 * x32) * (1.0 / HEAD_DIM)
    else:
        mean = x32.pow(2).mean(dim=-1, keepdim=True)
    variance = mean + EPS
    return (x32 * torch.pow(variance, -0.5)).bfloat16()


def _rotate_bf16(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rotated = torch.cat((-x[..., 64:], x[..., :64]), dim=-1)
    a = (x.float() * cos.float()).bfloat16()
    b = (rotated.float() * sin.float()).bfloat16()
    return (a.float() + b.float()).bfloat16()


def qknorm_scaled_rope_reference(
    q: Tensor,
    k: Tensor,
    positions: Tensor | None = None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    *,
    is_local: bool,
) -> tuple[Tensor, Tensor]:
    """BF16 eager model semantics with ordinary autograd and explicit cast boundaries.

    Global layers ignore positions/tables. Tables are constants, even if supplied
    with requires_grad=True. Input validation may synchronize a GPU; this is not
    a performance implementation.
    """
    return _reference(q, k, positions, cos, sin, is_local=is_local, fixed_tree=False)


def qknorm_scaled_rope_fixed_tree_reference(
    q: Tensor,
    k: Tensor,
    positions: Tensor | None = None,
    cos: Tensor | None = None,
    sin: Tensor | None = None,
    *,
    is_local: bool,
) -> tuple[Tensor, Tensor]:
    """Forward fixed-tree candidate, NOT a fully frozen strict reference.

    Uses torch.pow and ordinary autograd; its automatic gradient schedule is not
    the explicit candidate VJP. Use qknorm_scaled_rope_backward_reference to check
    the custom kernel's declared first-order arithmetic.
    """
    return _reference(q, k, positions, cos, sin, is_local=is_local, fixed_tree=True)


def _reference(q, k, positions, cos, sin, *, is_local, fixed_tree):
    _validate(q, k)
    if is_local:
        if cos is None or sin is None:
            raise ValueError("local layers require cosine and sine tables")
        if cos.ndim != 2 or cos.shape[1] != 64 or sin.shape != cos.shape:
            raise ValueError("tables must have matching shape [P, 64]")
        if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
            raise ValueError("tables must be BF16")
        if cos.device != q.device or sin.device != q.device:
            raise ValueError("tables must be on the Q/K device")
        positions = _validate_positions(positions, q, min(cos.shape[0], 131072))
    q_out = (_norm(q, fixed_tree=fixed_tree).float() * Q_SCALE).bfloat16()
    k_out = _norm(k, fixed_tree=fixed_tree)
    if is_local:
        c = cos.detach()[positions].repeat(1, 2).unsqueeze(1)
        s = sin.detach()[positions].repeat(1, 2).unsqueeze(1)
        q_out = _rotate_bf16(q_out, c, s)
        k_out = _rotate_bf16(k_out, c, s)
    return q_out, k_out


def qknorm_scaled_rope_fp64_reference(
    q: Tensor, k: Tensor, positions: Tensor | None = None, *, is_local: bool
) -> tuple[Tensor, Tensor]:
    """Independent real-formula reference with FP64-generated tables and no BF16 casts.

    For FP64 input gradients, differentiate with respect to FP64 leaf copies using
    ``fp64_math``; gradients accumulated into the original BF16 leaves are BF16.
    """
    _validate(q, k)
    if is_local:
        positions = _validate_positions(positions, q, 131072)
    return fp64_math(q.double(), k.double(), positions, is_local=is_local)


def fp64_math(
    q: Tensor, k: Tensor, positions: Tensor | None, *, is_local: bool
) -> tuple[Tensor, Tensor]:
    """FP64 formula entry point for independent gradient checks on FP64 leaves."""
    if q.dtype != torch.float64 or k.dtype != torch.float64:
        raise ValueError("fp64_math requires FP64 inputs")
    outputs = []
    for x, scale in ((q, Q_SCALE), (k, 1.0)):
        y = x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) / HEAD_DIM + EPS)
        y = y * scale
        if is_local:
            if positions is None:
                raise ValueError("local layers require positions")
            index = torch.arange(64, device=x.device, dtype=torch.float64)
            frequency = torch.exp(-torch.log(x.new_tensor(THETA)) * index / 64)
            angle = positions.double()[:, None] * frequency[None, :]
            c, s = angle.cos()[:, None, :], angle.sin()[:, None, :]
            low, high = y[..., :64], y[..., 64:]
            y = torch.cat((low * c - high * s, high * c + low * s), dim=-1)
        outputs.append(y)
    return outputs[0], outputs[1]


def qknorm_scaled_rope_backward_reference(
    q,
    k,
    grad_q,
    grad_k,
    query_rstd,
    key_rstd,
    positions=None,
    cos=None,
    sin=None,
    *,
    is_local,
):
    """Declared first-order VJP candidate, independent of eager autograd scheduling.

    BF16 rotation products, rotation sum and Q-scale gradients are rounded.
    Norm uses saved FP32 rstd and the fixed dot tree. The exact grouping below
    is part of this candidate, not a claim of bitwise eager-autograd parity.
    """
    outputs = []
    for x, grad, saved_r, scale in ((q, grad_q, query_rstd, Q_SCALE), (k, grad_k, key_rstd, None)):
        g = grad.float()
        if is_local:
            c = cos.detach()[positions].repeat(1, 2)[:, None].float()
            s = sin.detach()[positions].repeat(1, 2)[:, None].float()
            partner = torch.cat((g[..., 64:], g[..., :64]), dim=-1)
            a = (g * c).bfloat16().float()
            b = (partner * s).bfloat16().float()
            signed_b = torch.cat((b[..., :64], -b[..., 64:]), dim=-1)
            g = (a + signed_b).bfloat16().float()
        if scale is not None:
            g = (g * scale).bfloat16().float()
        x32 = x.float()
        r = saved_r[..., None]
        # VJP v2: FP32 normalized values avoid forming an underflowing r^3.
        u = x32 * r
        dot = fixed_dot_128(g, u)
        mean_dot = dot * (1.0 / HEAD_DIM)
        correction = u * mean_dot
        outputs.append((r * (g - correction)).bfloat16())
    return outputs[0], outputs[1]
