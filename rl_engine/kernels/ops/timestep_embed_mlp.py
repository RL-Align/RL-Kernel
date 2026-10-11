# SPDX-License-Identifier: Apache-2.0
"""Deterministic timestep MLP, explicit backend and observable opt-in fallback."""

import math
from dataclasses import dataclass, field

import torch
from torch.autograd.function import once_differentiable

from .pytorch.timestep_embed_mlp import NativeTimestepEmbedMLPOp, validate


@dataclass
class TimestepTrace:
    """Per-call launch record; backward updates the same caller-owned record."""

    requested_backend: str
    actual_backend: str
    fallback_reason: str | None = None
    accumulation_dtype: str = "float32"
    kernels: list[str] = field(default_factory=list)


class _TimestepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, w1, b1, w2, b2, primitives, trace, chunk_size, f):
        ctx.primitives, ctx.trace = primitives, trace
        ctx.dtypes = (t.dtype, w1.dtype, b1.dtype, w2.dtype, b2.dtype)
        t, w1, b1, w2, b2 = (x.float().contiguous() for x in (t, w1, b1, w2, b2))
        p = primitives
        e = p.embedding(t, f, trace)
        # Chunk only row-local forward computation. Save the complete logical
        # tensors; backward never merges independently rounded chunk gradients.
        n, h = t.numel(), w1.shape[0]
        if chunk_size is None or chunk_size >= n:
            z = p.mm(e, w1.T, trace) + b1
            a = p.unary(z, False, trace)
            y = p.mm(a, w2.T, trace) + b2
        else:
            z, a, y = (torch.empty((n, h), device=t.device, dtype=torch.float32) for _ in range(3))
            for start in range(0, n, chunk_size):
                stop = min(n, start + chunk_size)
                z[start:stop] = p.mm(e[start:stop], w1.T, trace) + b1
                a[start:stop] = p.unary(z[start:stop], False, trace)
                y[start:stop] = p.mm(a[start:stop], w2.T, trace) + b2
        ctx.save_for_backward(w1, w2, e, z, a, f)
        return y.to(ctx.dtypes[1])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        p, trace = ctx.primitives, ctx.trace
        w1, w2, e, z, a, f = ctx.saved_tensors
        g = grad_output.float().contiguous()
        need = ctx.needs_input_grad
        dt = dw1 = db1 = dw2 = db2 = None
        ones = torch.ones((1, g.shape[0]), device=g.device, dtype=torch.float32)
        if need[3]:
            dw2 = p.mm(g.T, a, trace)
        if need[4]:
            db2 = p.mm(ones, g, trace).flatten()
        if any(need[:3]):
            dz = p.mm(g, w2, trace) * p.unary(z, True, trace)
            if need[1]:
                dw1 = p.mm(dz.T, e, trace)
            if need[2]:
                db1 = p.mm(ones, dz, trace).flatten()
            if need[0]:
                de = p.mm(dz, w1, trace)
                dt = p.timestep_grad(de, e, f, trace)
        grads = (dt, dw1, db1, dw2, db2)
        return (
            *[x.to(d) if x is not None else None for x, d in zip(grads, ctx.dtypes, strict=True)],
            None,
            None,
            None,
            None,
        )


class TimestepEmbedMLPOp:
    """Standard 256→H→H MLP (production H=3072).

    sample_ids identifies canonical logical order for a permuted/padded batch.
    active_mask excludes padding. chunk_size changes forward execution only.
    Parameter gradients reduce the full active set in ascending sample-id order.
    External microbatch .grad accumulation is outside this guarantee.
    """

    def __init__(self, backend="cuda", *, allow_fallback=False):
        if backend not in ("pytorch", "cuda", "triton"):
            raise ValueError(f"unknown backend: {backend}")
        self.backend = backend
        self.allow_fallback = allow_fallback
        self.last_trace = None
        self._frequencies = {}

    def __call__(
        self,
        timestep,
        weight1,
        bias1,
        weight2,
        bias2,
        *,
        chunk_size=None,
        sample_ids=None,
        active_mask=None,
        return_trace=False,
    ):
        validate(timestep, weight1, bias1, weight2, bias2)
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
            raise ValueError("chunk_size must be a positive integer")
        n = timestep.numel()
        transformed = sample_ids is not None or active_mask is not None
        indices = torch.arange(n, device=timestep.device) if transformed else None
        if active_mask is not None:
            if (
                active_mask.shape != (n,)
                or active_mask.dtype != torch.bool
                or active_mask.device != timestep.device
            ):
                raise ValueError("active_mask must be a device-local bool [B]")
            indices = indices[active_mask]
        if sample_ids is not None:
            if (
                sample_ids.shape != (n,)
                or sample_ids.dtype != torch.int64
                or sample_ids.device != timestep.device
            ):
                raise ValueError("sample_ids must be a device-local int64 [B]")
            ids = sample_ids[indices]
            if ids.unique().numel() != ids.numel():
                raise ValueError("active sample_ids must be unique")
            indices = indices[torch.argsort(ids, stable=True)]
        t = timestep[indices] if transformed else timestep
        trace = TimestepTrace(self.backend, self.backend)
        primitives = None
        if self.backend != "pytorch":
            reason = None
            if timestep.device.type != "cuda" or torch.version.hip is not None:
                reason = "requested backend requires an NVIDIA CUDA device"
            else:
                try:
                    if self.backend == "cuda":
                        from .cuda.timestep_embed_mlp import CudaPrimitives

                        primitives = CudaPrimitives()
                    else:
                        from .triton.timestep_embed_mlp import TritonPrimitives

                        primitives = TritonPrimitives()
                except (ImportError, RuntimeError, OSError) as exc:
                    reason = f"backend unavailable: {type(exc).__name__}: {exc}"
            if reason:
                if not self.allow_fallback:
                    raise RuntimeError(reason)
                trace.actual_backend, trace.fallback_reason = "pytorch", reason
        if trace.actual_backend == "pytorch":
            y = NativeTimestepEmbedMLPOp()(t, weight1, bias1, weight2, bias2)
            trace.kernels.append("pytorch.product_sum_gold")
        else:
            if t.device not in self._frequencies:
                exponent = -math.log(10000.0) * torch.arange(128, dtype=torch.float32)
                self._frequencies[t.device] = torch.exp(exponent / 128).to(t.device)
            frequency = self._frequencies[t.device]
            # Constants are immutable, but an op can be destroyed while another
            # stream still reads them. Protect allocator lifetime in that stream.
            frequency.record_stream(torch.cuda.current_stream(t.device))
            y = _TimestepFunction.apply(
                t, weight1, bias1, weight2, bias2, primitives, trace, chunk_size, frequency
            )
        if transformed:
            y = y.new_zeros((n, weight1.shape[0])).index_copy(0, indices, y)
        self.last_trace = trace
        return (y, trace) if return_trace else y
