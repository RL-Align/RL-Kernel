# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Transformable, recursively differentiable row-local CUDA primitives.

Only probability evaluation and row reduction need custom kernels. All other
formulas use shape-matched CUDA pointwise operations and the linear primitives
below, never implicit differentiable broadcasting or a library softmax.
"""

from functools import wraps

import torch


def require(backend, *names):
    if any(not hasattr(backend, name) for name in names):
        raise RuntimeError("Rebuild rl_engine._C to enable deterministic LogP autograd/FP64")


def transformed():
    return torch._C._are_functorch_transforms_active()


def needs_ad(x):
    return (
        (torch.is_grad_enabled() and x.requires_grad)
        or transformed()
        or (torch.autograd.forward_ad.unpack_dual(x).tangent is not None)
    )


def storage_tensor(x):
    # Inspect storage only; computations must retain their transform wrappers.
    while torch._C._functorch.is_functorch_wrapped_tensor(x):
        x = torch._C._functorch.get_unwrapped(x)
    return x


class _JVPContext:
    """Expose primals at the current dual level, retaining outer AD wrappers."""

    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        if name == "saved_tensors":
            return tuple(
                torch.autograd.forward_ad.unpack_dual(t).primal for t in self.context.saved_tensors
            )
        return getattr(self.context, name)


def _nested_jvp(rule):
    @wraps(rule)
    def wrapped(ctx, *tangents):
        # The Function engine disables forward AD while evaluating a JVP.
        # Re-enable it for outer JVP levels, but strip the current level from
        # saved inputs to avoid assigning a tangent that is dual at that level.
        with torch.autograd.forward_ad._set_fwd_grad_enabled(True):
            return rule(_JVPContext(ctx), *tangents)

    return wrapped


class _Mapped(torch.autograd.Function):
    @classmethod
    def vmap(cls, info, in_dims, *args):
        def lane(i):
            return cls.apply(*(a if d is None else a.select(d, i) for a, d in zip(args, in_dims)))

        if info.batch_size:
            values = [lane(i) for i in range(info.batch_size)]
            if isinstance(values[0], tuple):
                return tuple(torch.stack(v) for v in zip(*values)), (0,) * len(values[0])
            return torch.stack(values), 0
        # Sum an empty mapped axis to construct shape-correct zero prototypes,
        # keeping an autograd connection to empty inputs without indexing lane 0.
        prototype = cls.apply(
            *(a if d is None else a.sum(d).to(a.dtype) for a, d in zip(args, in_dims))
        )
        if isinstance(prototype, tuple):
            return tuple(v.unsqueeze(0)[:0] for v in prototype), (0,) * len(prototype)
        return prototype.unsqueeze(0)[:0], 0


class _Sum(_Mapped):
    @staticmethod
    def forward(x, backend):
        require(backend, "deterministic_logp_row_sum")
        return backend.deterministic_logp_row_sum(x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1])

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ctx.backend = inputs
        ctx.width = x.shape[-1]

    @staticmethod
    def backward(ctx, g):
        return _Expand.apply(g, ctx.width, ctx.backend), None

    @staticmethod
    @_nested_jvp
    def jvp(ctx, t, _):
        return _Sum.apply(t, ctx.backend)


class _Expand(_Mapped):
    @staticmethod
    def forward(x, width, backend):
        return x.unsqueeze(-1).expand(*x.shape, width).clone()

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.width, ctx.backend = inputs

    @staticmethod
    def backward(ctx, g):
        return _Sum.apply(g, ctx.backend), None, None

    @staticmethod
    @_nested_jvp
    def jvp(ctx, t, _width, _backend):
        return _Expand.apply(t, ctx.width, ctx.backend)


class _Gather(_Mapped):
    @staticmethod
    def forward(x, ids, active, backend):
        values = x.gather(-1, ids.clamp(0, x.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
        return torch.where(active, values, 0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ids, active, ctx.backend = inputs
        ctx.width = x.shape[-1]
        ctx.save_for_backward(ids, active)
        ctx.save_for_forward(ids, active)

    @staticmethod
    def backward(ctx, g):
        ids, active = ctx.saved_tensors
        return _Scatter.apply(g, ids, active, ctx.width, ctx.backend), None, None, None

    @staticmethod
    @_nested_jvp
    def jvp(ctx, t, _ids, _active, _backend):
        ids, active = ctx.saved_tensors
        return _Gather.apply(t, ids, active, ctx.backend)


class _Scatter(_Mapped):
    @staticmethod
    def forward(x, ids, active, width, backend):
        output = x.new_zeros((*x.shape, width))
        return output.scatter(
            -1, ids.clamp(0, width - 1).unsqueeze(-1), torch.where(active, x, 0).unsqueeze(-1)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ids, active, ctx.width, ctx.backend = inputs
        ctx.save_for_backward(ids, active)
        ctx.save_for_forward(ids, active)

    @staticmethod
    def backward(ctx, g):
        ids, active = ctx.saved_tensors
        return _Gather.apply(g, ids, active, ctx.backend), None, None, None, None

    @staticmethod
    @_nested_jvp
    def jvp(ctx, t, _ids, _active, _width, _backend):
        ids, active = ctx.saved_tensors
        return _Scatter.apply(t, ids, active, ctx.width, ctx.backend)


class _Probability(_Mapped):
    @staticmethod
    def forward(x, active, backend):
        require(backend, "deterministic_logp_probability")
        matrix = x.reshape(-1, x.shape[-1])
        ids = torch.zeros(matrix.shape[0], device=x.device, dtype=torch.long)
        with torch.cuda.device(x.device):
            if x.dtype == torch.float64:
                _, lse = backend.deterministic_logp_forward_fp64(matrix, ids, None, True)
            else:
                _, lse = backend.deterministic_logp_forward_with_lse(matrix, ids, None)
        return backend.deterministic_logp_probability(matrix, lse, active.reshape(-1)).reshape(
            x.shape
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, active, ctx.backend = inputs
        ctx.input_dtype = x.dtype
        # Save the OUTPUT, whose grad_fn reconnects every higher derivative.
        ctx.save_for_backward(output, active)
        ctx.save_for_forward(output, active)

    @staticmethod
    def backward(ctx, g):
        p, active = ctx.saved_tensors
        return probability_vjp(p, g, active, ctx.backend).to(ctx.input_dtype), None, None

    @staticmethod
    @_nested_jvp
    def jvp(ctx, t, _active, _backend):
        p, active = ctx.saved_tensors
        return probability_vjp(p, t.to(p.dtype), active, ctx.backend)


def probability_vjp(p, v, active, backend):
    v = torch.where(active.unsqueeze(-1), v, 0)
    return p * (v - _Expand.apply(_Sum.apply(p * v, backend), p.shape[-1], backend))


class _Selection(_Mapped):
    @staticmethod
    def forward(indices, rows, backend):
        require(backend, "deterministic_logp_selection")
        return backend.deterministic_logp_selection(indices, rows)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.mark_non_differentiable(output)

    @staticmethod
    def backward(ctx, g):
        return None, None, None

    @staticmethod
    @_nested_jvp
    def jvp(ctx, *tangents):
        return None


def selection(indices, rows, backend):
    return _Selection.apply(indices, rows, backend)


class _Logp(_Mapped):
    @staticmethod
    def forward(x, ids, indices, backend, fp64):
        if fp64:
            if x.dtype != torch.float64:
                raise TypeError("FP64 LogP requires float64 logits")
            require(backend, "deterministic_logp_forward_fp64", "deterministic_logp_backward_fp64")
        else:
            if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise TypeError(
                    "FP32 LogP autograd supports float32, float16, bfloat16; "
                    "use apply_fp64 or indexed_fp64"
                )
            require(backend, "deterministic_logp_forward_with_lse", "deterministic_logp_backward")
        matrix = x.reshape(-1, x.shape[-1])
        labels = ids.reshape(-1).to(device=x.device, dtype=torch.long).contiguous()
        with torch.cuda.device(x.device):
            if fp64:
                y, lse = backend.deterministic_logp_forward_fp64(matrix, labels, indices, True)
            else:
                y, lse = backend.deterministic_logp_forward_with_lse(matrix, labels, indices)
        return y.reshape(x.shape[:-1]), lse.reshape(x.shape[:-1])

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ids, _, ctx.backend, _ = inputs
        _, lse = output
        labels = ids.to(device=x.device, dtype=torch.long).reshape(x.shape[:-1])
        active = torch.isfinite(lse) & (labels >= 0) & (labels < x.shape[-1])
        ctx.save_for_backward(x, labels, lse, active)
        ctx.save_for_forward(x, labels, active)
        ctx.mark_non_differentiable(lse)

    @staticmethod
    def backward(ctx, g, _lse_grad):
        x, ids, lse, active = ctx.saved_tensors
        backend = ctx.backend
        if torch.is_grad_enabled() or transformed():
            p = _Probability.apply(x, active, backend)
            weights = torch.where(active, g.to(p.dtype), 0)
            grad = (
                _Scatter.apply(weights, ids, active, x.shape[-1], backend)
                - _Expand.apply(weights, x.shape[-1], backend) * p
            )
            return grad.to(x.dtype), None, None, None, None
        # Ordinary first backward remains the original fused CUDA VJP.
        coef = torch.where(active, g, 0).reshape(-1).contiguous()
        targets = ids.masked_fill(~active, -1).reshape(-1).contiguous()
        matrix = x.reshape(-1, x.shape[-1])
        with torch.cuda.device(x.device):
            if x.dtype == torch.float64:
                grad = backend.deterministic_logp_backward_fp64(
                    matrix, lse.reshape(-1).contiguous(), coef.double(), targets
                )
            else:
                grad = backend.deterministic_logp_backward(
                    matrix,
                    lse.reshape(-1).contiguous(),
                    coef.float(),
                    torch.zeros_like(coef, dtype=torch.float32),
                    targets,
                    0,
                    x.shape[-1],
                    False,
                )
        return grad.reshape(x.shape), None, None, None, None

    @staticmethod
    @_nested_jvp
    def jvp(ctx, t, _ids, _indices, _backend, _fp64):
        x, ids, active = ctx.saved_tensors
        p = _Probability.apply(x, active, ctx.backend)
        vector = torch.where(active.unsqueeze(-1), t.to(p.dtype), 0)
        return (
            _Gather.apply(vector, ids, active, ctx.backend) - _Sum.apply(p * vector, ctx.backend),
            None,
        )


def logp(x, ids, indices, backend, fp64):
    return _Logp.apply(x, ids, indices, backend, fp64)[0]
