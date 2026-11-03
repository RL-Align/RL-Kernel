# SPDX-License-Identifier: Apache-2.0
"""One P5 forward for training, recomputation and rollout; explicit staged VJP."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from rl_engine.contracts.operators.expert_training import (
    ExpertActivationState,
    ExpertGradients,
    ExpertSavedState,
    ExpertTrainingBackend,
    ExpertTrainingConfig,
    PackedExpertWeight,
    require_tensor,
    validate_request,
)


def _checked(value, shape, dtype, x, name):
    require_tensor(value, shape, dtype, x.device, name)
    return value


@torch.no_grad()
def expert_forward(
    config: ExpertTrainingConfig,
    backend: ExpertTrainingBackend,
    x: Tensor,
    p: Tensor,
    offsets: Tensor,
    w1: PackedExpertWeight | None,
    w2: PackedExpertWeight | None,
    adapters: tuple,
    *,
    need_route_grad: bool = False,
) -> tuple[Tensor, ExpertSavedState]:
    validate_request(config, x, p, offsets, w1, w2, adapters)
    backend.validate(config, x.device, need_route_grad)
    m = x.shape[0]
    with torch.autocast(device_type=x.device.type, enabled=False):

        def projection(inp, weight, a, b, width):
            base = delta = None
            u = torch.empty((m, 0), device=x.device, dtype=torch.bfloat16)
            if config.use_base:
                q = backend.quantize(inp)
                q.validate(inp)
                base = _checked(
                    backend.base_forward(q, weight, offsets),
                    (m, width),
                    torch.float32,
                    x,
                    "base output",
                )
            if config.use_lora:
                delta, u = backend.lora_forward(inp, a, b, config.alpha)
                _checked(delta, (m, width), torch.float32, x, "LoRA output")
                _checked(u, (m, a.shape[0]), torch.bfloat16, x, "LoRA U")
            if base is None:
                return delta, u
            return base if delta is None else base + delta, u

        a1, b1, a2, b2 = adapters
        z, u1 = projection(x, w1, a1, b1, 2 * config.ffn)
        gate, up = z.chunk(2, dim=1)
        h, activation = backend.activation_forward(gate, up, p)
        _checked(h, (m, config.ffn), torch.bfloat16, x, "weighted activation")
        for t in activation:
            _checked(t, (m, config.ffn), torch.float32, x, "activation saved value")
        y32, u2 = projection(h, w2, a2, b2, config.hidden)
        return y32.bfloat16(), ExpertSavedState(gate, up, h, *activation, u1, u2)


@torch.no_grad()
def expert_backward(
    config: ExpertTrainingConfig,
    backend: ExpertTrainingBackend,
    dy: Tensor,
    x: Tensor,
    p: Tensor,
    offsets: Tensor,
    w1: PackedExpertWeight | None,
    w2: PackedExpertWeight | None,
    adapters: tuple,
    saved: ExpertSavedState,
    *,
    need_route_grad: bool,
) -> ExpertGradients:
    validate_request(config, x, p, offsets, w1, w2, adapters)
    backend.validate(config, x.device, need_route_grad)
    m = x.shape[0]
    if dy.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("expert dY must be BF16 or FP32")
    _checked(dy, x.shape, dy.dtype, x, "dY")
    # Low-level callers own matching saved state. Autograd additionally protects
    # all saved tensors (including base codes/scales and offsets) from mutation.
    for name in ("gate", "up", "sigmoid", "silu", "unweighted"):
        _checked(getattr(saved, name), (m, config.ffn), torch.float32, x, name)
    _checked(saved.h, (m, config.ffn), torch.bfloat16, x, "saved activation")
    with torch.autocast(device_type=x.device.type, enabled=False):

        def projection_vjp(grad, inp, weight, a, b, u):
            base_dx = lora_dx = da = db = None
            if config.use_base:
                base_dx = _checked(
                    backend.base_backward(grad, weight, offsets),
                    inp.shape,
                    torch.float32,
                    x,
                    "base dX",
                )
            if config.use_lora:
                _checked(u, (m, a.shape[0]), torch.bfloat16, x, "saved LoRA U")
                lora_dx, da, db = backend.lora_backward(grad, inp, a, b, config.alpha, u)
                _checked(lora_dx, inp.shape, torch.float32, x, "LoRA dX")
                _checked(da, a.shape, torch.float32, x, "dA")
                _checked(db, b.shape, torch.float32, x, "dB")
            if base_dx is None:
                return lora_dx, da, db
            return base_dx if lora_dx is None else base_dx + lora_dx, da, db

        a1, b1, a2, b2 = adapters
        # Merge both FP32 branches BEFORE the one BF16 cast at each edge.
        dh32, da2, db2 = projection_vjp(dy.bfloat16(), saved.h, w2, a2, b2, saved.u2)
        dh = dh32.bfloat16()  # activation-quantization backward is the identity STE
        dg, du, dp = backend.activation_backward(
            dh,
            saved.gate,
            saved.up,
            p,
            ExpertActivationState(saved.sigmoid, saved.silu, saved.unweighted),
            need_route_grad,
        )
        for name, value in (("dgate", dg), ("dup", du)):
            _checked(value, (m, config.ffn), torch.float32, x, name)
        if need_route_grad:
            if dp is None:
                raise RuntimeError("task 49 backend dropped the required route-weight gradient")
            _checked(dp, (m,), torch.float32, x, "dp_s")
        dz = torch.cat((dg, du), dim=1).bfloat16()
        dx, da1, db1 = projection_vjp(dz, x, w1, a1, b1, saved.u1)
        return ExpertGradients(dx, dp if need_route_grad else None, da1, db1, da2, db2)


class _ExpertTraining(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, offsets, c1, s1, c2, s2, a1, b1, a2, b2, config, backend, recompute):
        weights = _weights(config, c1, s1, c2, s2)
        adapters = (a1, b1, a2, b2)
        ctx.config, ctx.backend, ctx.recompute = config, backend, recompute
        ctx.profile = backend.profile
        y, saved = expert_forward(
            config,
            backend,
            x,
            p,
            offsets,
            *weights,
            adapters,
            need_route_grad=ctx.needs_input_grad[1],
        )
        roots = (x, p, offsets, c1, s1, c2, s2, a1, b1, a2, b2)
        ctx.save_for_backward(*roots, *(() if recompute else saved))
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        if ctx.backend.profile != ctx.profile:
            raise RuntimeError("expert backend profile changed between forward and backward")
        x, p, offsets, c1, s1, c2, s2, a1, b1, a2, b2, *intermediates = ctx.saved_tensors
        weights = _weights(ctx.config, c1, s1, c2, s2)
        adapters = (a1, b1, a2, b2)
        if ctx.recompute:
            _, saved = expert_forward(
                ctx.config,
                ctx.backend,
                x,
                p,
                offsets,
                *weights,
                adapters,
                need_route_grad=ctx.needs_input_grad[1],
            )
        else:
            saved = ExpertSavedState(*intermediates)
        grads = expert_backward(
            ctx.config,
            ctx.backend,
            dy,
            x,
            p,
            offsets,
            *weights,
            adapters,
            saved,
            need_route_grad=ctx.needs_input_grad[1],
        )
        params = tuple(
            g.bfloat16() if g is not None and ctx.needs_input_grad[i] else None
            for i, g in zip(range(7, 11), grads[2:])
        )
        return (
            grads.dx.bfloat16() if ctx.needs_input_grad[0] else None,
            grads.dp_s,
            None,
            None,
            None,
            None,
            None,
            *params,
            None,
            None,
            None,
        )


def _weights(config, c1, s1, c2, s2):
    if not config.use_base:
        return None, None
    return (
        PackedExpertWeight(c1, s1, (config.experts, 2 * config.ffn, config.hidden)),
        PackedExpertWeight(c2, s2, (config.experts, config.hidden, config.ffn)),
    )


def routed_expert_training(config, backend, x, p, offsets, w1, w2, adapters, *, recompute=False):
    """Tensor entry for task 72/73 adapters; no engine imports or global dispatch."""
    validate_request(config, x, p, offsets, w1, w2, adapters)
    backend.validate(config, x.device, torch.is_grad_enabled() and p.requires_grad)
    if not torch.is_grad_enabled():
        # Rollout and framework-level checkpoint forward use the SAME math.
        return expert_forward(config, backend, x, p, offsets, w1, w2, adapters)[0]
    return _ExpertTraining.apply(
        x,
        p,
        offsets,
        None if w1 is None else w1.codes,
        None if w1 is None else w1.scales,
        None if w2 is None else w2.codes,
        None if w2 is None else w2.scales,
        *adapters,
        config,
        backend,
        recompute,
    )
