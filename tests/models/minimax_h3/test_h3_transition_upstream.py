# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_sample_previous_step`` against verl-omni's ``sample_h3_transition`` (``sde_type="cps"``).

The reference below is the arithmetic of verl-omni ``FlowMatchSDEDiscreteScheduler.
sample_previous_step`` (cps branch, ``verl_omni/pipelines/schedulers/flow_match_sde.py`` at
``a634f4d8``) and of ``sample_h3_transition`` (sign flip of the H3 velocity), written out so that
the test does not need verl-omni installed.
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_engine.models.minimax_h3 import transition as tr

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _upstream_cps(sample, h3_velocity, sigmas, index, noise_level, generator=None, prev=None):
    sample = sample.float()
    model_output = -h3_velocity.float()
    view = (-1,) + (1,) * (sample.ndim - 1)
    idx = torch.full((sample.shape[0],), index)
    sigma = sigmas[idx].view(view).to(sample.device)
    sigma_prev = sigmas[idx + 1].view(view).to(sample.device)
    dt = sigma_prev - sigma
    sin_noise = math.sin(noise_level * math.pi / 2)
    std_dev_t = sigma_prev * sin_noise
    pred_original_sample = sample - sigma * model_output
    noise_estimate = sample + model_output * (1 - sigma)
    mean = pred_original_sample * (1 - sigma_prev) + noise_estimate * torch.sqrt(
        sigma_prev**2 - std_dev_t**2
    )
    if prev is None:
        rand_device = generator.device if generator is not None else sample.device
        noise = torch.randn(
            model_output.shape, generator=generator, device=rand_device, dtype=torch.float32
        ).to(sample.device)
        prev = mean + std_dev_t * noise
    per_elem = -((prev.detach() - mean) ** 2)
    log_prob = per_elem.mean(dim=tuple(range(1, per_elem.ndim)))
    sqrt_dt = torch.sqrt(-1 * dt).reshape(sample.shape[0])
    return prev, log_prob, mean, std_dev_t, sqrt_dt, per_elem


def _grid():
    # A shifted H3-style grid with a terminal zero (values only need to be a valid grid).
    base = torch.linspace(1.0, 0.0, 11, dtype=torch.float32)
    shift = 12.0
    return shift * base / (1 + (shift - 1) * base)


def _inputs(device, rows=37, width=96, batch=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    sample = torch.randn(batch, rows, width, generator=gen).to(device)
    velocity = torch.randn(batch, rows, width, generator=gen).to(torch.bfloat16).to(device)
    return sample, velocity


@CUDA
@pytest.mark.parametrize("gen_device", ["cuda", "cpu"])
@pytest.mark.parametrize("index", [0, 4, 9])
def test_sampling_matches_upstream_bitwise(gen_device, index):
    sample, velocity = _inputs("cuda")
    sigmas = _grid()
    ours = tr.h3_sample_previous_step(
        sample,
        velocity,
        sigmas,
        index,
        noise_level=0.8,
        generator=torch.Generator(device=gen_device).manual_seed(43),
    )
    ref = _upstream_cps(
        sample,
        velocity,
        sigmas,
        index,
        0.8,
        generator=torch.Generator(device=gen_device).manual_seed(43),
    )
    for name, a, b in zip(("prev_sample", "log_prob", "mean", "std_dev_t", "sqrt_dt"), ours, ref):
        assert a.shape == b.shape, name
    for name, i in (("prev_sample", 0), ("prev_sample_mean", 2), ("std_dev_t", 3), ("sqrt_dt", 4)):
        assert torch.equal(ours[i], ref[i]), name
    # The log-prob mean is the fixed-order one: equal to it bit for bit, close to Tensor.mean.
    assert torch.equal(ours[1], tr.fixed_order_mean(ref[5]))
    torch.testing.assert_close(ours[1], ref[1], rtol=1e-5, atol=1e-7)


@CUDA
def test_scoring_a_stored_sample_matches_upstream():
    sample, velocity = _inputs("cuda", seed=1)
    sigmas = _grid()
    stored = torch.randn(sample.shape, generator=torch.Generator().manual_seed(2)).cuda()
    ours = tr.h3_sample_previous_step(sample, velocity, sigmas, 3, prev_sample=stored)
    ref = _upstream_cps(sample, velocity, sigmas, 3, 0.8, prev=stored)
    assert torch.equal(ours[0], stored)
    assert torch.equal(ours[2], ref[2])
    assert torch.equal(ours[1], tr.fixed_order_mean(ref[5]))


@CUDA
def test_log_prob_of_a_sample_does_not_depend_on_its_batch():
    sample, velocity = _inputs("cuda", batch=4, seed=3)
    sigmas = _grid()
    stored = torch.randn(sample.shape, generator=torch.Generator().manual_seed(4)).cuda()
    full = tr.h3_sample_previous_step(sample, velocity, sigmas, 2, prev_sample=stored)[1]
    for i in range(4):
        one = tr.h3_sample_previous_step(
            sample[i : i + 1], velocity[i : i + 1], sigmas, 2, prev_sample=stored[i : i + 1]
        )[1]
        assert torch.equal(one, full[i : i + 1])


@CUDA
def test_sampling_log_prob_of_a_sample_does_not_depend_on_its_batch():
    # The generator stream is consumed in batch order (as the scheduler's), so the drawn x_prev
    # depends on the batch position; its log-prob is the score of that x_prev alone.
    sample, velocity = _inputs("cuda", batch=4, seed=5)
    sigmas = _grid()
    gen = torch.Generator(device="cuda").manual_seed(6)
    prev, log_prob, *_ = tr.h3_sample_previous_step(sample, velocity, sigmas, 2, generator=gen)
    for i in range(4):
        one = tr.h3_sample_previous_step(
            sample[i : i + 1], velocity[i : i + 1], sigmas, 2, prev_sample=prev[i : i + 1]
        )[1]
        assert torch.equal(one, log_prob[i : i + 1])


@CUDA
def test_per_sample_noise_makes_the_draw_independent_of_the_batch():
    sample, velocity = _inputs("cuda", batch=4, seed=7)
    sigmas = _grid()
    eps = [tr.step_noise(11, 2, i, "video", tuple(sample.shape[1:])) for i in range(4)]
    full = tr.h3_sample_previous_step(sample, velocity, sigmas, 2, noise=torch.stack(eps))
    for i in (3, 0):
        one = tr.h3_sample_previous_step(
            sample[i : i + 1], velocity[i : i + 1], sigmas, 2, noise=eps[i][None]
        )
        for a, b in zip(one, full):
            assert torch.equal(a, b[i : i + 1])
    ref = _upstream_cps(sample, velocity, sigmas, 2, 0.8, prev=full[0])
    assert torch.equal(full[2], ref[2])


def test_noise_excludes_generator_and_prev_sample_and_needs_the_sample_shape():
    sample, velocity = _inputs("cpu")
    eps = torch.zeros(sample.shape)
    with pytest.raises(ValueError, match="excludes"):
        tr.h3_sample_previous_step(
            sample, velocity, _grid(), 0, noise=eps, generator=torch.Generator()
        )
    with pytest.raises(ValueError, match="excludes"):
        tr.h3_sample_previous_step(sample, velocity, _grid(), 0, noise=eps, prev_sample=sample)
    with pytest.raises(ValueError, match="shape"):
        tr.h3_sample_previous_step(sample, velocity, _grid(), 0, noise=eps[:1])


def test_refuses_other_sde_types_and_a_tensor_noise_level():
    sample, velocity = _inputs("cpu")
    with pytest.raises(ValueError, match="cps"):
        tr.h3_sample_previous_step(sample, velocity, _grid(), 0, sde_type="sde")
    with pytest.raises(TypeError, match="noise_level"):
        tr.h3_sample_previous_step(sample, velocity, _grid(), 0, noise_level=torch.tensor(0.8))


@pytest.mark.parametrize(
    "grid, index, error",
    [
        (torch.tensor([1.0, 0.5, 0.1]), 0, ValueError),  # no terminal zero
        (torch.tensor([[1.0, 0.0]]), 0, ValueError),  # not one-dimensional
        (torch.tensor([1.0, 0.5, 0.0]), 2, IndexError),  # last index has no successor
        (torch.tensor([1.0, 0.5, 0.0]), -1, IndexError),
    ],
)
def test_refuses_a_bad_grid_or_index(grid, index, error):
    sample, velocity = _inputs("cpu")
    with pytest.raises(error):
        tr.h3_sample_previous_step(sample, velocity, grid, index)
