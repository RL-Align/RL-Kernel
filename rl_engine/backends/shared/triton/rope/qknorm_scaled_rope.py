# SPDX-License-Identifier: Apache-2.0
"""F2 fused forward using the FP32 half-split tree v1 and explicit BF16 casts."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _check_positions_kernel(P, INVALID, N, COVERAGE, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    invalid = tl.full((BLOCK,), False, tl.int1)
    for chunk in range(tl.cdiv(N, BLOCK)):
        offsets = chunk * BLOCK + lanes
        position = tl.load(P + offsets, mask=offsets < N, other=0)
        invalid = invalid | ((offsets < N) & ((position < 0) | (position >= COVERAGE)))
    # Integer predicate reduction: unrelated to the fixed FP32 arithmetic tree.
    tl.store(INVALID, tl.max(invalid.to(tl.int32), axis=0))


def _positions_outside_table(positions, coverage):
    """Caller validates metadata; no cached validation and exactly one host read."""
    if not positions.numel():
        return False
    invalid = torch.empty((), device=positions.device, dtype=torch.int32)
    with torch.cuda.device(positions.device):
        _check_positions_kernel[(1,)](
            positions,
            invalid,
            positions.numel(),
            coverage,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return bool(invalid.item())


def _num_warps(device):
    # SM80: measured 1/2/4 warps with bitwise forward/rstd/backward checks.
    # Keep the previously tested geometry on other architectures until validated.
    if torch.version.hip is None and torch.cuda.get_device_capability(device) == (8, 0):
        return 1
    return 4


@triton.jit
def _bf16_round(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _kernel(Q, K, P, C, S, OQ, OK, RQ, RK, HQ: tl.constexpr, HK: tl.constexpr, LOCAL: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    token = row // (HQ + HK)
    combined = row % (HQ + HK)
    is_q = combined < HQ
    head = tl.where(is_q, combined, combined - HQ)
    heads = tl.where(is_q, HQ, HK)
    src = Q if is_q else K
    dst = OQ if is_q else OK
    rdst = RQ if is_q else RK
    head_row = token * heads + head
    d = tl.arange(0, 128)
    x = tl.load(src + head_row * 128 + d).to(tl.float32)
    acc = x * x
    # Explicit logical tree; never substitute tl.sum without proving its order.
    for stride in tl.static_range(6, -1, -1):
        half = 1 << stride
        peer = tl.gather(acc, (d + half) % 128, axis=0)
        acc = tl.where(d < half, acc + peer, acc)
    total = tl.gather(acc, tl.full((128,), 0, tl.int32), axis=0)
    variance = total * (1.0 / 128) + 1.0e-5
    r = libdevice.rsqrt(variance)
    tl.store(rdst + head_row + tl.zeros((128,), tl.int32), r, mask=d == 0)
    value = _bf16_round(x * r)
    if is_q:
        value = _bf16_round(value * 3.87)
    if LOCAL:
        position = tl.load(P + token)
        c = tl.load(C + position * 64 + d % 64).to(tl.float32)
        s = tl.load(S + position * 64 + d % 64).to(tl.float32)
        other = tl.gather(value, (d + 64) % 128, axis=0)
        rotated = tl.where(d < 64, -other, other)
        a = _bf16_round(value * c)
        b = _bf16_round(rotated * s)
        value = a + b
    tl.store(dst + head_row * 128 + d, value.to(tl.bfloat16, fp_downcast_rounding="rtne"))


def _forward(q, k, positions, cos, sin, *, is_local):
    """Internal entry point; public dispatcher validates geometry and autograd use."""
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    rq = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    rk = torch.empty(k.shape[:2], device=k.device, dtype=torch.float32)
    if q.shape[0]:
        # Global specialization never dereferences these dummy pointers.
        p = positions if is_local else q
        c, s = (cos, sin) if is_local else (q, q)
        options = {"num_warps": _num_warps(q.device), "enable_fp_fusion": False}
        if torch.version.hip is None:
            options["enable_reflect_ftz"] = False
        with torch.cuda.device(q.device):
            _kernel[(q.shape[0] * (q.shape[1] + k.shape[1]),)](
                q, k, p, c, s, oq, ok, rq, rk, q.shape[1], k.shape[1], is_local, **options
            )
    return oq, ok, rq, rk


@triton.jit
def _backward_kernel(
    Q, K, GQ, GK, RQ, RK, P, C, S, DQ, DK, HQ: tl.constexpr, HK: tl.constexpr, LOCAL: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    token = row // (HQ + HK)
    combined = row % (HQ + HK)
    is_q = combined < HQ
    head = tl.where(is_q, combined, combined - HQ)
    heads = tl.where(is_q, HQ, HK)
    src = Q if is_q else K
    grad = GQ if is_q else GK
    saved_r = RQ if is_q else RK
    dst = DQ if is_q else DK
    head_row = token * heads + head
    d = tl.arange(0, 128)
    x = tl.load(src + head_row * 128 + d).to(tl.float32)
    g = tl.load(grad + head_row * 128 + d).to(tl.float32)
    if LOCAL:
        position = tl.load(P + token)
        c = tl.load(C + position * 64 + d % 64).to(tl.float32)
        s = tl.load(S + position * 64 + d % 64).to(tl.float32)
        partner = tl.gather(g, (d + 64) % 128, axis=0)
        a = _bf16_round(g * c)
        b = _bf16_round(partner * s)
        g = _bf16_round(a + tl.where(d < 64, b, -b))
    if is_q:
        g = _bf16_round(g * 3.87)
    # VJP v2: keep u in FP32, without reusing the BF16 forward norm output.
    r = tl.load(saved_r + head_row)
    u = x * r
    acc = g * u
    for stride in tl.static_range(6, -1, -1):
        half = 1 << stride
        peer = tl.gather(acc, (d + half) % 128, axis=0)
        acc = tl.where(d < half, acc + peer, acc)
    dot = tl.gather(acc, tl.full((128,), 0, tl.int32), axis=0)
    mean_dot = dot * (1.0 / 128)
    correction = u * mean_dot
    dx = r * (g - correction)
    tl.store(dst + head_row * 128 + d, dx.to(tl.bfloat16, fp_downcast_rounding="rtne"))


def _backward(q, k, gq, gk, rq, rk, positions, cos, sin, *, is_local):
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    if q.shape[0]:
        p = positions if is_local else q
        c, s = (cos, sin) if is_local else (q, q)
        options = {"num_warps": _num_warps(q.device), "enable_fp_fusion": False}
        if torch.version.hip is None:
            options["enable_reflect_ftz"] = False
        with torch.cuda.device(q.device):
            _backward_kernel[(q.shape[0] * (q.shape[1] + k.shape[1]),)](
                q, k, gq, gk, rq, rk, p, c, s, dq, dk, q.shape[1], k.shape[1], is_local, **options
            )
    return dq, dk
