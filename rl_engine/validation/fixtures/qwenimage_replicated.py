# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded real-width boundaries for Qwen-Image replicated modules."""

import math

import torch
import torch.nn.functional as F

from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.validation.fixtures.qwenimage_qkv import QKV_WORKLOADS, qkv_workload_rows

MODULES = (
    "image_input",
    "text_input",
    "timestep",
    "image_modulation",
    "text_modulation",
    "block_norm",
    "modulation_apply",
    "output_norm",
    "image_output",
    "qk_norm",
)
_WIDTHS = {
    "image_input": (64, 3072),
    "text_input": (3584, 3072),
    "image_output": (3072, 64),
    "image_modulation": (3072, 18432),
    "text_modulation": (3072, 18432),
}


def make_replicated_case(kind, rows, dtype, device, seed=386, requests=1):
    """Make independent parameter sets, complete gradients and logical rows."""
    if kind not in MODULES or rows < 1 or requests < 1:
        raise ValueError("unknown module or empty layout")
    generator = torch.Generator().manual_seed(seed)

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)

    width, out = _WIDTHS.get(kind, (3072, 3072))
    if kind == "timestep":
        width = 1
    shape = (rows, 24, 128) if kind == "qk_norm" else (rows, width)
    x = rand(shape, 0.3)
    if kind == "timestep":
        x = torch.rand(shape, generator=generator, dtype=torch.float32).to(device)
    parameters = []
    if kind in _WIDTHS:
        if kind == "text_input":
            parameters.append(rand((3584,), 0.1) + 1)
        parameters.extend((rand((out, width), width**-0.5), rand((out,), 0.02)))
    elif kind == "timestep":
        parameters.extend(
            (
                rand((3072, 256), 256**-0.5),
                rand((3072,), 0.02),
                rand((3072, 3072), 3072**-0.5),
                rand((3072,), 0.02),
            )
        )
    elif kind == "output_norm":
        parameters.extend(
            (rand((requests, 3072), 0.3), rand((6144, 3072), 3072**-0.5), rand((6144,), 0.02))
        )
    elif kind == "modulation_apply":
        parameters.append(rand((requests, 9216), 0.1))
    elif kind == "qk_norm":
        parameters.append(rand((128,), 0.1) + 1)
    grad = rand((rows, 24, 128) if kind == "qk_norm" else (rows, out), out**-0.5)
    keys = torch.stack(
        (
            torch.arange(rows, device=device) % requests,
            torch.arange(rows, device=device) // requests,
        ),
        1,
    )
    return x, tuple(parameters), grad, keys


def module_forward(kind, x, parameters, keys, *, backend="cuda", collective=None):
    """Apply the real-width boundary using declared native and ATen primitives."""
    from rl_engine.backends.cuda.model_specific.qwenimage.replicated_modules import (
        broadcast_condition,
        replicated_linear,
        replicated_qk_rmsnorm,
        replicated_silu_projection,
    )
    from rl_engine.ops.norm.canonical_rmsnorm import canonical_cuda_rmsnorm

    native = backend == "cuda"
    name = "replicated." + kind

    def linear(a, w, b, suffix="", row_keys=None, silu=False):
        if native:
            operation = replicated_silu_projection if silu else replicated_linear
            return operation(
                a,
                w,
                b,
                logical_keys=keys if row_keys is None else row_keys,
                parameter_id=name + suffix,
            )
        return F.linear(F.silu(a) if silu else a, w, b)

    def broadcast(c, suffix):
        if native:
            return broadcast_condition(c, keys, parameter_id=name + suffix)
        return c.index_select(0, keys[:, 0].clamp_min(0))

    if kind == "qk_norm":
        if native:
            return replicated_qk_rmsnorm(
                x,
                parameters[0],
                logical_keys=keys,
                parameter_id=name + ".gain",
                collective=collective,
            )
        value = x.float()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * parameters[0].float()
        ).to(x.dtype)
    if kind in _WIDTHS:
        if kind == "text_input":
            gain, w, b = parameters
            if native:
                x = canonical_cuda_rmsnorm(
                    x, gain, eps=1e-6, logical_keys=keys, parameter_id=name + ".gain"
                )
            else:
                value = x.float()
                x = (
                    value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * gain.float()
                ).to(x.dtype)
        else:
            w, b = parameters
        return linear(x, w, b, silu=kind.endswith("modulation"))
    if kind == "timestep":
        frequencies = torch.exp(-math.log(10000) * torch.arange(128, device=x.device).float() / 128)
        phase = (x.float() * frequencies[None, :]) * 1000
        embed = torch.cat((phase.cos(), phase.sin()), -1).to(parameters[0].dtype)
        w0, b0, w1, b1 = parameters
        return linear(F.silu(linear(embed, w0, b0, ".in")), w1, b1, ".out")
    normalized = F.layer_norm(x.float(), (3072,), eps=1e-6).to(x.dtype)
    if kind == "block_norm":
        return normalized
    if kind == "modulation_apply":
        shift, scale, gate = broadcast(parameters[0], ".broadcast").chunk(3, -1)
        return x + gate * (normalized * (1 + scale) + shift)
    condition, w, b = parameters
    condition_keys = torch.stack(
        (
            torch.arange(condition.shape[0], device=x.device),
            torch.zeros(condition.shape[0], dtype=torch.int64, device=x.device),
        ),
        1,
    )
    modulation = linear(condition, w, b, ".projection", condition_keys, silu=True)
    scale, shift = broadcast(modulation, ".broadcast").chunk(2, -1)
    return normalized * (1 + scale) + shift


def replicated_layout(x, grad, keys, layout):
    """Reuse the established ordering, padding and call-cut transformations."""
    # The row helper treats leading axes as branch, request and sequence. It
    # handles a one-request flattened boundary here; the logical keys retain
    # all original request identities for canonical ordering.
    shape = (1, 1, x.shape[0], -1)
    a, g, k, cuts = qkv_workload_rows(
        {"x": x.reshape(shape), "grad": grad.reshape(shape), "keys": keys[None, None]}, layout
    )
    if x.ndim == 3:
        a, g = a.reshape(-1, 24, 128), g.reshape(-1, 24, 128)
    return a, g, k, cuts


def make_replicated_workload(kind, name, dtype, device, seed=386, stream="image"):
    """Use the existing model requests at this module's actual tensor boundary."""
    spec = QKV_WORKLOADS[name]
    branches = 2 if spec.negative_lengths else 1
    requests = branches * spec.batch
    request_module = kind in ("timestep", "image_modulation", "text_modulation")
    if request_module:
        width = 1
        lengths = torch.ones((branches, spec.batch), device=device, dtype=torch.int64)
    elif stream == "image":
        width = spec.image_tokens
        lengths = torch.full((branches, spec.batch), width, device=device, dtype=torch.int64)
    else:
        sets = [spec.positive_lengths]
        if branches == 2:
            sets.append(spec.negative_lengths)
        lengths = torch.tensor(sets, device=device).repeat_interleave(spec.images_per_prompt, 1)
        width = int(lengths.max().item())
    count = branches * spec.batch * width
    x, parameters, grad, _ = make_replicated_case(kind, count, dtype, device, seed, requests)
    valid = torch.arange(width, device=device)[None, None, :] < lengths[:, :, None]
    sample = torch.arange(spec.batch, device=device)[None, :, None].expand_as(valid)
    branch = torch.arange(branches, device=device)[:, None, None].expand_as(valid)
    position = torch.arange(width, device=device)[None, None, :].expand_as(valid)
    keys = torch.stack((sample * branches + branch, position), -1)
    keys[..., 0] = torch.where(valid, keys[..., 0], -1)
    x = x.reshape(branches, spec.batch, width, -1) * valid[..., None]
    grad = grad.reshape(branches, spec.batch, width, -1) * valid[..., None]
    return {
        "x": x,
        "grad": grad,
        "keys": keys,
        "parameters": parameters,
        "metadata": {
            "workload": name,
            "module": kind,
            "stream": stream,
            "height": spec.height,
            "width": spec.width,
            "prompt_batch": len(spec.positive_lengths),
            "images_per_prompt": spec.images_per_prompt,
            "cfg_branches": branches,
            "active_rows": int(valid.sum().item()),
        },
    }


def replicated_calls(kind, x, parameters, grad, keys, cuts, *, backend="cuda", collective=None):
    """Build normal forward/autograd calls after preparing the row layout."""
    # Sigma is fixed scheduler metadata; only the timestep MLP parameters train.
    x = x.detach().requires_grad_(kind != "timestep")
    parameters = tuple(value.detach().requires_grad_() for value in parameters)

    def outputs():
        return tuple(
            module_forward(
                kind,
                x[start:stop],
                parameters,
                keys[start:stop],
                backend=backend,
                collective=collective,
            )
            for start, stop in cuts
            if stop > start
        )

    def forward():
        with torch.no_grad(), canonical_backward_session():
            return outputs()

    def backward():
        with canonical_backward_session() as session:
            ys = outputs()
            leaves = parameters if kind == "timestep" else (x, *parameters)
            gradients = torch.autograd.grad(ys, leaves, [grad[s:e] for s, e in cuts if e > s])
        session.validate_complete()
        return ys, *gradients

    return forward, backward


def evaluate_replicated(kind, x, parameters, grad, keys, cuts, *, backend="cuda", collective=None):
    """Return complete active tensors in canonical order for byte comparison."""
    _, backward = replicated_calls(
        kind, x, parameters, grad, keys, cuts, backend=backend, collective=collective
    )
    ys, *gradients = backward()
    order = torch.arange(keys.shape[0], device=keys.device)[keys[:, 0] >= 0]
    for column in range(keys.shape[1] - 1, -1, -1):
        order = order[torch.argsort(keys[order, column], stable=True)]
    if kind != "timestep":
        gradients[0] = gradients[0][order]
    return (torch.cat(ys)[order].detach(), *(value.detach() for value in gradients))
