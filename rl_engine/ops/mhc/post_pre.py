# SPDX-License-Identifier: Apache-2.0
"""Adjacent post/pre with an identical BF16 rounding point in either schedule."""
import torch
from torch.autograd.function import once_differentiable

from rl_engine.ops.mhc.training import EPS, _check, _outputs, mhc_post, mhc_pre, mix_state


def _split(y, x, p, b, w, s, base, backend, rounds):
    res = mhc_post(y, x, p, b, backend=backend)
    return (res, *mhc_pre(res, w, s, base, backend=backend, sinkhorn_repeat=rounds, recompute=True))


class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, p, b, w, s, base, rounds):
        from rl_engine.backends.rocm.mhc.post_project import post_project

        res, lin, ss = post_project(y, x, p, b, w)
        t, _, c = x.shape
        outs = (base.new_empty((t, 4)), base.new_empty((t, 4, 4)), x.new_empty((t, c)))
        for i in range(t):
            r = torch.rsqrt(ss[i] / (4 * c) + EPS)
            state = mix_state(r, lin[i] * r, s, base, False, rounds)
            for dst, val in zip(outs, _outputs(res[i], state, False), strict=True):
                dst[i].copy_(val)
        ctx.save_for_backward(y, x, p, b, w, s, base)
        ctx.rounds = rounds
        return (res, *outs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad):
        original = ctx.saved_tensors
        with torch.enable_grad():
            args = [v.detach().requires_grad_(True) for v in original]
            out = _split(*args, "rocm", ctx.rounds)
            ds = torch.autograd.grad(out, args, grad)
        return (*ds, None)


def post_pre(y, x, p, b, w, s, base, *, backend, mode, sinkhorn_repeat=20):
    if mode == "split":
        return _split(y, x, p, b, w, s, base, backend, sinkhorn_repeat)
    if mode != "fused" or backend != "rocm":
        raise ValueError("fused post/pre requires the owned ROCm profile")
    _check(x, backend)
    t, _, c = x.shape
    for v, shape, dt in (
        (y, (t, c), x.dtype),
        (p, (t, 4), torch.float32),
        (b, (t, 4, 4), torch.float32),
        (w, (24, 4 * c), torch.float32),
        (s, (3,), torch.float32),
        (base, (24,), torch.float32),
    ):
        if v.device != x.device or v.shape != shape or v.dtype != dt:
            raise ValueError(f"expected {shape} {dt} on {x.device}")
    if type(sinkhorn_repeat) is not int or not 0 <= sinkhorn_repeat <= 20:
        raise ValueError("invalid sinkhorn_repeat")
    return _Fused.apply(y, x, p, b, w, s, base, sinkhorn_repeat)
