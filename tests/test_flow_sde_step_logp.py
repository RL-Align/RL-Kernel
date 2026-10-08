# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Independent accuracy and raw-byte invariance gates for Qwen-Image SDE.

CUDA requires the compiled native forward AND backward; never substitutes CPU.
"""

import math

import pytest
import torch

from rl_engine.kernels.ops.pytorch.diffusion.flow_sde_step_logp import (
    NativeFlowSDEStepLogpOp,
    fixed_mean,
)

REFERENCE_TOKENS = (4096, 6889, 6032)  # 1024^2, 1328^2, 1664x928 at VAE/pack /16.


def inputs(shape=(2, 5, 64), dtype=torch.float32):
    generator = torch.Generator().manual_seed(386)
    return {
        "sample": torch.randn(shape, generator=generator).to(dtype),
        "model_output": torch.randn(shape, generator=generator).to(dtype),
        "noise": torch.randn(shape, generator=generator),
        "sigma": 0.75,
        "sigma_next": 0.5,
        "sigma_max": 0.98,
        "noise_level": 0.7,
    }


def to_device(data, device):
    return {
        key: (
            value.to(device) if key in ("sample", "model_output", "noise", "prev_sample") else value
        )
        for key, value in data.items()
    }


def bits_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    # A size-one view may be "contiguous" while its stride is still greater
    # than one. Allocate dense storage before changing the element size.
    left = torch.empty(a.shape, dtype=a.dtype, device=a.device).copy_(a)
    right = torch.empty(b.shape, dtype=b.dtype, device=b.device).copy_(b)
    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def independent_double(data):
    """Mathematical accuracy oracle, separate from the FP32 arithmetic contract."""
    x, v = data["sample"].double(), data["model_output"].double()
    shape = (-1,) + (1,) * (x.ndim - 1)

    def parameter(name):
        # Match the pinned FP32 inputs, then evaluate the formula in FP64.
        value = torch.as_tensor(data[name], dtype=torch.float32).reshape(-1)
        return value.expand(x.shape[0]).double().reshape(shape)

    s, sn, sm, level = (
        parameter(name) for name in ("sigma", "sigma_next", "sigma_max", "noise_level")
    )
    dt = sn - s
    std = torch.sqrt(s / (1 - torch.where(s == 1, sm, s))) * level
    a = 1 + std.square() / (2 * s) * dt
    b = (1 + std.square() * (1 - s) / (2 * s)) * dt
    tau = std * torch.sqrt(-dt)
    mean = x * a + v * b
    target = data.get("prev_sample")
    if target is None:
        target = mean + tau * data["noise"].double()
    else:
        target = target.double()
    density = -(target.detach() - mean).square() / (2 * tau.square())
    density = density - tau.log() - 0.5 * math.log(2 * math.pi)
    return target, density.flatten(1).mean(1), mean, std.flatten()


@pytest.fixture(params=("cpu", "cuda"))
def backend(request):
    if request.param == "cpu":
        return NativeFlowSDEStepLogpOp(), "cpu"
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA not available")
    # A visible GPU with missing symbols is a failure, not a skip/fallback.
    from rl_engine.kernels.ops.cuda.diffusion.flow_sde_step_logp import CUDAFlowSDEStepLogpOp

    return CUDAFlowSDEStepLogpOp(), "cuda"


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("shape", ((2, 3, 64), (1, 257), (2, 1, 1)))
def test_independent_accuracy(backend, dtype, shape):
    op, device = backend
    data = inputs(shape, dtype)
    actual = op(**to_device(data, device))
    expected = independent_double(data)
    for lhs, rhs in zip(actual, expected):
        assert lhs.dtype == torch.float32
        torch.testing.assert_close(lhs.cpu().double(), rhs, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("sigma,sigma_next", ((1.0, 0.98), (0.02, 0.0)))
def test_first_and_terminal_step(backend, sigma, sigma_next):
    op, device = backend
    data = inputs()
    data.update(sigma=sigma, sigma_next=sigma_next)
    for lhs, rhs in zip(op(**to_device(data, device)), independent_double(data)):
        torch.testing.assert_close(lhs.cpu().double(), rhs, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tokens", (*REFERENCE_TOKENS, 7))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_reference_shapes_replay_and_batch_position(backend, tokens, dtype):
    op, device = backend
    single = to_device(inputs((1, tokens, 64), dtype), device)
    reference = op(**single)
    for batch, position in ((2, 0), (4, 3)):
        data = to_device(inputs((batch, tokens, 64), dtype), device)
        for name in ("sample", "model_output", "noise"):
            data[name][position].copy_(single[name][0])
        actual = op(**data)
        for lhs, rhs in zip(actual, reference):
            bits_equal(lhs[position : position + 1], rhs)
        repeated = op(**data)
        for lhs, rhs in zip(actual, repeated):
            bits_equal(lhs, rhs)
        replay = dict(data)
        replay.pop("noise")
        replay["prev_sample"] = actual.prev_sample.detach()
        recomputed = op(**replay)
        for lhs, rhs in zip(actual, recomputed):
            bits_equal(lhs, rhs)


def test_independent_cpu_fp32_reference(backend):
    op, device = backend
    data = inputs((3, 17, 64))
    expected = NativeFlowSDEStepLogpOp()(**data)
    actual = op(**to_device(data, device))
    for lhs, rhs in zip(actual, expected):
        torch.testing.assert_close(lhs.cpu(), rhs, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("replay", (False, True))
def test_backward_independent_autograd(backend, replay):
    op, device = backend
    data = inputs()
    if replay:
        data["prev_sample"] = NativeFlowSDEStepLogpOp()(**data).prev_sample.detach()
        data.pop("noise")
    cpu = dict(data)
    cpu["sample"] = data["sample"].clone().requires_grad_(True)
    cpu["model_output"] = data["model_output"].clone().requires_grad_(True)
    candidate = to_device(data, device)
    candidate["sample"] = candidate["sample"].clone().requires_grad_(True)
    candidate["model_output"] = candidate["model_output"].clone().requires_grad_(True)
    result = op(**candidate)
    expected = NativeFlowSDEStepLogpOp()(**cpu)
    generator = torch.Generator().manual_seed(204)
    upstream_logp = torch.randn(result.logp.shape, generator=generator)
    upstream_mean = torch.randn(result.mean.shape, generator=generator)
    upstream_target = torch.randn(result.prev_sample.shape, generator=generator)

    def loss(output, dev):
        total = (output.logp * upstream_logp.to(dev)).sum()
        total = total + (output.mean * upstream_mean.to(dev)).sum()
        if not replay:
            total = total + (output.prev_sample * upstream_target.to(dev)).sum()
        return total

    got = torch.autograd.grad(
        loss(result, device), (candidate["sample"], candidate["model_output"])
    )
    want = torch.autograd.grad(loss(expected, "cpu"), (cpu["sample"], cpu["model_output"]))
    for lhs, rhs in zip(got, want):
        torch.testing.assert_close(lhs.cpu(), rhs, atol=3e-6, rtol=3e-6)


def test_logp_gradient_has_no_path_through_sampled_target(backend):
    op, device = backend
    data = to_device(inputs((1, 2, 64)), device)
    data["model_output"].requires_grad_(True)
    sampled = op(**data)
    gradient = torch.autograd.grad(sampled.logp.sum(), data["model_output"])[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    replay = dict(data)
    replay.pop("noise")
    replay["prev_sample"] = sampled.prev_sample.detach()
    rescored = op(**replay)
    replay_gradient = torch.autograd.grad(rescored.logp.sum(), data["model_output"])[0]
    bits_equal(gradient, replay_gradient)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_backward_batch_invariance(backend, dtype):
    op, device = backend
    single = to_device(inputs((1, 17, 64), dtype), device)
    gradients = []
    for batch in (1, 4):
        data = dict(single)
        for name in ("sample", "model_output", "noise"):
            data[name] = single[name].expand(batch, -1, -1).clone()
        data["sample"].requires_grad_(True)
        data["model_output"].requires_grad_(True)
        result = op(**data)
        gradient = torch.autograd.grad(result.logp.sum(), (data["sample"], data["model_output"]))
        gradients.append(gradient)
    for lhs, rhs in zip(gradients[0], gradients[1]):
        bits_equal(lhs, rhs[3:4])


def test_strides_and_sample_specific_schedule(backend):
    op, device = backend
    data = to_device(inputs((3, 9, 64)), device)
    data.update(sigma=torch.tensor([1.0, 0.75, 0.02]), sigma_next=torch.tensor([0.98, 0.5, 0.0]))
    expected = op(**data)
    strided = dict(data)
    for name in ("sample", "model_output", "noise"):
        value = data[name]
        storage = torch.empty((*value.shape[:-1], value.shape[-1] * 2), device=device)
        storage[..., ::2].copy_(value)
        strided[name] = storage[..., ::2]
    for lhs, rhs in zip(op(**strided), expected):
        bits_equal(lhs, rhs)
    order = torch.tensor([2, 0, 1])
    permuted = {
        key: (
            value[order.to(value.device)]
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 3
            else value
        )
        for key, value in data.items()
    }
    for lhs, rhs in zip(op(**permuted), expected):
        bits_equal(lhs, rhs[order.to(rhs.device)])


def test_full_synthetic_trajectory_replay(backend):
    op, device = backend
    data = to_device(inputs((2, 7, 64)), device)
    schedule = (1.0, 0.98, 0.75, 0.4, 0.02, 0.0)
    stored = []
    for index, (sigma, next_sigma) in enumerate(zip(schedule[:-1], schedule[1:])):
        data.update(sigma=sigma, sigma_next=next_sigma)
        # Fixed synthetic velocities/noise are trajectory fixtures, not a model.
        data["noise"] = torch.full_like(data["noise"], (index + 1) * 0.125)
        result = op(**data)
        stored.append((dict(data), result))
        data["sample"] = result.prev_sample.detach()
    for call, expected in stored:
        replay = dict(call)
        replay.pop("noise")
        replay["prev_sample"] = expected.prev_sample.detach()
        for lhs, rhs in zip(op(**replay), expected):
            bits_equal(lhs, rhs)


@pytest.mark.parametrize(
    "changes",
    (
        {"noise_level": 0},
        {"sigma": 0},
        {"sigma": 1.1},
        {"sigma_next": 0.9},
        {"sigma_next": -0.1},
        {"sigma_max": 1},
        {"noise_level": float("nan")},
        {"noise_level": 1e-40},
        {"sigma": torch.tensor([0.5], dtype=torch.float64)},
    ),
)
def test_reject_invalid_metadata(changes):
    data = inputs()
    data.update(changes)
    with pytest.raises(ValueError):
        NativeFlowSDEStepLogpOp()(**data)


def test_reject_ambiguous_noise_and_differentiable_replay():
    op = NativeFlowSDEStepLogpOp()
    data = inputs()
    with pytest.raises(ValueError):
        op(**data, prev_sample=data["sample"])
    data.pop("noise")
    with pytest.raises(ValueError):
        op(**data)
    data["prev_sample"] = data["sample"].clone().requires_grad_(True)
    with pytest.raises(ValueError):
        op(**data)


def test_reference_fixed_tree_cancellation():
    values = torch.zeros((1, 258))
    values[0, :4] = torch.tensor([1e8, 1, -1e8, 1])
    values[0, 256:] = torch.tensor([2.0, 3.0])
    # Adjacent pairs in first tile cancel to zero; final tile contributes five.
    bits_equal(fixed_mean(values), torch.tensor([5.0 / 258]))


def test_registry_and_trace(backend):
    op, device = backend
    from rl_engine.kernels.registry import kernel_registry

    selected = kernel_registry.get_op("flow_sde_step_logp", device=device)
    assert type(selected) is type(op)
    trace = selected.execution_trace()
    assert trace["actual_backend"] == ("pytorch" if device == "cpu" else "cuda")
    assert trace["fallback"] is False
    selected(**to_device(inputs(), device))
    trace = selected.execution_trace()
    assert trace["execution_recorded"] is True and trace["mode"] == "sampling"
    assert trace["shape"] == [2, 5, 64]


def test_operator_consumes_no_rng(backend):
    op, device = backend
    data = to_device(inputs(), device)
    cpu_state = torch.get_rng_state().clone()
    gpu_state = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    op(**data)
    bits_equal(torch.get_rng_state(), cpu_state)
    if gpu_state is not None:
        bits_equal(torch.cuda.get_rng_state(), gpu_state)


def test_replay_gradient_against_independent_finite_difference(backend):
    op, device = backend
    data = inputs((1, 2, 64))
    data["prev_sample"] = independent_double(data)[0].float()
    data.pop("noise")
    candidate = to_device(data, device)
    candidate["model_output"] = candidate["model_output"].clone().requires_grad_(True)
    result = op(**candidate)
    gradient = torch.autograd.grad(result.logp.sum(), candidate["model_output"])[0].cpu()
    delta = 1e-3
    for index in (0, 37, 127):
        plus, minus = dict(data), dict(data)
        plus["model_output"] = data["model_output"].double().clone()
        minus["model_output"] = data["model_output"].double().clone()
        plus["model_output"].flatten()[index] += delta
        minus["model_output"].flatten()[index] -= delta
        expected = (independent_double(plus)[1] - independent_double(minus)[1]) / (2 * delta)
        torch.testing.assert_close(
            gradient.flatten()[index].double(), expected.squeeze(), atol=2e-7, rtol=1e-4
        )


def test_cuda_fails_closed_on_cpu():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA not available")
    from rl_engine.kernels.ops.cuda.diffusion.flow_sde_step_logp import CUDAFlowSDEStepLogpOp

    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        CUDAFlowSDEStepLogpOp()(**inputs())
