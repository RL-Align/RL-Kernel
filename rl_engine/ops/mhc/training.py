# SPDX-License-Identifier: Apache-2.0
"""Rank-local DSV4 mHC: single-row arithmetic and ordered parameter gradients.

The reference is an explicit CPU diagnostic. ROCm uses the same owned arithmetic
on-device. ``rocm_triton`` runs RL-Kernel's batch-invariant Triton mHC ops
(rl_engine/backends/rocm/mhc/triton_mhc.py) instead.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.norm.rmsnorm_residual import supported_device
from rl_engine.ops.norm.rmsnorm_residual import rmsnorm_residual

PROFILE = "dsv4-p1-row-tree-v1"
EPS = 1e-6
N = 4


def _sum(x, dim=-1):
    """Adjacent FP32 pairs, carrying an odd last value without arithmetic."""
    v = x.movedim(dim, -1)
    while v.shape[-1] > 1:
        pairs = v.shape[-1] // 2 * 2
        s = v[..., :pairs:2] + v[..., 1:pairs:2]
        v = torch.cat((s, v[..., -1:]), -1) if pairs < v.shape[-1] else s
    return v.squeeze(-1)


def _row_state(residual, fn, scale, base, head, rounds):
    x = residual.float().reshape(-1)
    # FP32 fn, as DeepSeek-V4 inference/model.py hc_pre and hc_head.
    r = torch.rsqrt(_sum(x * x) / x.numel() + EPS)
    proj = _sum(fn * x) * r
    return mix_state(r, proj, scale, base, head, rounds)


def mix_state(r, proj, scale, base, head, rounds):
    """Complete a single controller row from its normalized FP32 projection."""
    if head:
        s = torch.sigmoid(proj * scale[0] + base)
        return (r, proj, s)
    a, b, c = proj.split((4, 4, 16))
    pre = torch.sigmoid(a * scale[0] + base[:4])
    post = torch.sigmoid(b * scale[1] + base[4:8])
    z = (c * scale[2] + base[8:]).reshape(4, 4)
    tape = []
    if rounds:
        exp = torch.exp(z - z.amax(-1, keepdim=True))
        soft = exp / _sum(exp).unsqueeze(-1)
        tape.append(soft)
        z = soft + EPS
        for step in range(2 * rounds - 1):
            axis = 0 if step % 2 == 0 else 1
            denom = _sum(z, axis).unsqueeze(axis) + EPS
            z = z / denom
            tape.extend((z, denom))
    return (r, proj, pre, post, z, *tape)


def _outputs(row, state, head):
    mix = state[2] + EPS
    h = _sum(mix[:, None] * row.float(), 0).bfloat16()
    return (h,) if head else (2.0 * state[3], state[4], h)


def _row_vjp(row, fn, scale, state, grads, head, rounds, need_fn):
    r, proj, sig = state[:3]
    x = row.float()
    gh = grads[-1].float()
    dpre = (_sum(x * gh) * sig) * (1.0 - sig)
    direct = (sig + EPS)[:, None] * gh
    if head:
        db = dpre
        ds = _sum(db * proj).reshape(1)
        dm = db * scale[0]
    else:
        post, z, *tape = state[3:]
        dp = ((2.0 * grads[0]) * post) * (1.0 - post)
        dz = grads[1]
        if rounds:
            soft = tape[0]
            for step in reversed(range(2 * rounds - 1)):
                normalized, denom = tape[1 + 2 * step : 3 + 2 * step]
                axis = 0 if step % 2 == 0 else 1
                dz = (dz - _sum(dz * normalized, axis).unsqueeze(axis)) / denom
            dz = soft * (dz - _sum(dz * soft).unsqueeze(-1))
        dz = dz.flatten()
        db = torch.cat((dpre, dp, dz))
        ds = torch.stack((_sum(dpre * proj[:4]), _sum(dp * proj[4:8]), _sum(dz * proj[8:])))
        dm = torch.cat((dpre * scale[0], dp * scale[1], dz * scale[2]))
    flat = x.flatten()
    dl = dm * r
    dx = _sum(dl[:, None] * fn, 0)
    # proj already includes r; d(r)/dx = -x*r^3/K.
    dx = dx - flat * ((_sum(dm * proj) * (r * r)) / flat.numel())
    dw = dl[:, None] * flat if need_fn else None
    return ((direct + dx.reshape_as(x)).bfloat16(), dw, ds, db)


def _check(row, backend):
    if row.ndim != 3 or row.shape[1] != N or row.shape[2] == 0 or row.dtype != torch.bfloat16:
        raise ValueError("expected BF16 residual [tokens,4,hidden] with nonzero hidden size")
    if backend == "reference" and row.device.type == "cpu":
        return
    if backend in ("rocm", "rocm_triton") and supported_device(row.device):
        return
    raise ValueError(f"unsupported mHC backend, shape or device for {PROFILE}")


@lru_cache(maxsize=3)
def _triton_op(kind):
    from rl_engine.backends.rocm.mhc.triton_mhc import TritonMHCPreOp

    return {"pre": TritonMHCPreOp}[kind]()


def _triton_mix(kind, rows, fn, scale, base, recompute, **kwargs):
    # The Triton ops save their forward values; fn, scale and base gradients go through
    # the canonical backward session (logical_keys, parameter_id).
    _check(rows, "rocm_triton")
    if recompute:
        raise ValueError("rocm_triton saves the forward values; recompute is not supported")
    return _triton_op(kind)(rows, fn, scale, base, **kwargs)


class _Mix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, fn, scale, base, head, rounds, recompute, backend):
        _check(rows, backend)
        nrows = 4 if head else 24
        for val, shape in (
            (fn, (nrows, 4 * rows.size(-1))),
            (scale, (1 if head else 3,)),
            (base, (nrows,)),
        ):
            if val.shape != shape or val.dtype != torch.float32 or val.device != rows.device:
                raise ValueError(f"expected FP32 parameter {shape} on residual device")
        if type(rounds) is not int or not 0 <= rounds <= 20:
            raise ValueError("sinkhorn_repeat must be an integer in [0,20]")
        ctx.head, ctx.rounds, ctx.recompute = head, rounds, recompute
        t, _, c = rows.shape
        h = rows.new_empty((t, c))
        outs = (h,) if head else (base.new_empty((t, 4)), base.new_empty((t, 4, 4)), h)
        saved = []
        for i in range(t):
            state = _row_state(rows[i], fn, scale, base, head, rounds)
            if not recompute and any(ctx.needs_input_grad[:4]):
                saved.extend(state)
            values = _outputs(rows[i], state, head)
            for dst, val in zip(outs, values, strict=True):
                dst[i].copy_(val)
        ctx.save_for_backward(rows, fn, scale, base, *saved)
        return outs[0] if head else outs

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        rows, fn, scale, base, *saved = ctx.saved_tensors
        t = rows.size(0)
        dx = torch.empty_like(rows) if ctx.needs_input_grad[0] else None
        params = [
            torch.zeros_like(p) if need else None
            for p, need in zip((fn, scale, base), ctx.needs_input_grad[1:4], strict=True)
        ]
        step = len(saved) // t if t else 0
        for i in range(t):
            state = (
                _row_state(rows[i], fn, scale, base, ctx.head, ctx.rounds)
                if ctx.recompute
                else saved[i * step : (i + 1) * step]
            )
            dr, *partials = _row_vjp(
                rows[i],
                fn,
                scale,
                state,
                [g[i] for g in grads],
                ctx.head,
                ctx.rounds,
                params[0] is not None,
            )
            if dx is not None:
                dx[i].copy_(dr)
            for total, value in zip(params, partials, strict=True):
                if total is not None:
                    total.add_(value)
        return (dx, *params, None, None, None, None)


def mhc_pre(
    rows,
    fn,
    scale,
    base,
    *,
    backend,
    recompute=False,
    sinkhorn_repeat=20,
    logical_keys=None,
    parameter_id=None,
):
    if backend == "rocm_triton":
        return _triton_mix(
            "pre",
            rows,
            fn,
            scale,
            base,
            recompute,
            sinkhorn_repeat=sinkhorn_repeat,
            logical_keys=logical_keys,
            parameter_id=parameter_id,
        )
    return _Mix.apply(rows, fn, scale, base, False, sinkhorn_repeat, recompute, backend)


def mhc_head(rows, fn, scale, base, *, backend, recompute=False):
    return _Mix.apply(rows, fn, scale, base, True, 0, recompute, backend)


def _post_row(y, x, post, comb):
    routed = _sum(comb[:, :, None] * x.float()[:, None, :], 0)
    return (routed + post[:, None] * y.float()).bfloat16()


class _Post(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, post, comb, backend):
        out = torch.empty_like(x)
        ctx.save_for_backward(y, x, post, comb)
        for i in range(x.size(0)):
            out[i].copy_(_post_row(y[i], x[i], post[i], comb[i]))
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        y, x, post, comb = ctx.saved_tensors
        g = grad.float()
        dy = _sum(post[..., None] * g, 1)
        dx = torch.stack([_sum(comb[:, j, :, None] * g, 1) for j in range(4)], 1)
        dp = _sum(g * y.float()[:, None, :])
        dc = torch.stack([_sum(g * x[:, j, None, :].float()) for j in range(4)], 1)
        return dy.bfloat16(), dx.bfloat16(), dp, dc, None


def mhc_post(y, rows, post, comb, *, backend):
    _check(rows, backend)
    t, _, c = rows.shape
    for val, shape, dtype in (
        (y, (t, c), torch.bfloat16),
        (post, (t, 4), torch.float32),
        (comb, (t, 4, 4), torch.float32),
    ):
        if val.shape != shape or val.dtype != dtype or val.device != rows.device:
            raise ValueError(f"invalid post input; expected {shape} and {dtype} on residual device")
    return _Post.apply(y, rows, post, comb, backend)


def mhc_pre_norm(
    rows,
    fn,
    scale,
    base,
    gamma,
    *,
    backend,
    recompute=False,
    sinkhorn_repeat=20,
    logical_keys=None,
    parameter_id=None,
):
    """Return mixes, normalized sublayer input, and the separate raw task-5 branch."""
    post, comb, raw = mhc_pre(
        rows,
        fn,
        scale,
        base,
        backend=backend,
        recompute=recompute,
        sinkhorn_repeat=sinkhorn_repeat,
        logical_keys=logical_keys,
        parameter_id=parameter_id,
    )
    y, residual = rmsnorm_residual(
        raw, gamma, backend="reference" if backend == "reference" else "rocm", recompute=recompute
    )
    return post, comb, y, residual
