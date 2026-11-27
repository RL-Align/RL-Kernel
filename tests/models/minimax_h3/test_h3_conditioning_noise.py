# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The conditioning noise mix on its row: ``bind`` against the provider's ``scale_noise``.

Every call is bitwise the provider's, whichever path ran; the row runs where the installed
provider's arithmetic is the row's (``KNOWN_SCALE_NOISE``).
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.reference.conditioning.conditioning_noise_mix import NativeConditioningNoiseMixOp
from rl_engine.validation.models import h3_conditioning_noise as hcn

scheduling = pytest.importorskip(
    "diffusers.schedulers.scheduling_minimax_h3",
    reason="the Diffusers MiniMax-H3 scheduler is the reference",
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _scheduler():
    return scheduling.MiniMaxH3Scheduler()


def _case(dtype, noise_dtype=torch.float32, shape=(1, 24, 3, 8, 12), device="cpu"):
    gen = torch.Generator().manual_seed(11)
    sample = torch.randn(shape, generator=gen).to(dtype)
    noise = torch.randn(shape, generator=gen).to(noise_dtype)
    return sample.to(device), noise.to(device)


def _expected_row_calls(variant, timestep):
    if variant == "sample_dtype":
        return 1
    if variant == "python_float_double":
        return int(isinstance(timestep, torch.Tensor))
    return 0


def _check(op, t, device="cpu", shape=(1, 24, 3, 8, 12)):
    scheduler = _scheduler()
    sample, noise = _case(torch.float32, shape=shape, device=device)
    timestep = torch.tensor(float(t), device=device) if isinstance(t, str) else t
    want = scheduler.scale_noise(sample, timestep, noise)
    with hcn.bind(scheduler, op) as calls:
        got = scheduler.scale_noise(sample, timestep, noise)
    variant = hcn.scale_noise_variant(scheduler)
    row = _expected_row_calls(variant, timestep)
    assert calls == {"row": row, "provider": 1 - row, "scale_noise": variant}
    assert got.dtype == want.dtype and torch.equal(got.view(-1), want.view(-1))
    assert "scale_noise" not in scheduler.__dict__
    return calls


def test_the_installed_scale_noise_is_known():
    assert hcn.scale_noise_variant(_scheduler()) in hcn.KNOWN_SCALE_NOISE.values()


@pytest.mark.parametrize("t", [0.999, 0.5, 1.0, "0.999", "0.5"])  # str: a 0-d tensor timestep
def test_same_dtype_is_bitwise_the_provider_on_either_path(t):
    _check(NativeConditioningNoiseMixOp(), t)


def test_mixed_dtypes_stay_on_the_provider():
    scheduler = _scheduler()
    sample, noise = _case(torch.bfloat16)
    want = scheduler.scale_noise(sample, 0.999, noise)
    with hcn.bind(scheduler, NativeConditioningNoiseMixOp()) as calls:
        got = scheduler.scale_noise(sample, 0.999, noise)
    assert (calls["row"], calls["provider"]) == (0, 1)
    assert torch.equal(got, want)


def test_an_unknown_scale_noise_stays_on_the_provider():
    class Other(scheduling.MiniMaxH3Scheduler):
        def scale_noise(self, sample, timestep, noise):
            """Not a provider scale_noise."""
            return timestep * sample + (1.0 - timestep) * noise

    scheduler = Other()
    sample, noise = _case(torch.float32)
    with hcn.bind(scheduler, NativeConditioningNoiseMixOp()) as calls:
        scheduler.scale_noise(sample, torch.tensor(0.5), noise)
    assert calls == {"row": 0, "provider": 1, "scale_noise": "unknown"}


def test_a_wrapped_scale_noise_is_refused():
    scheduler = _scheduler()
    scheduler.scale_noise = lambda *a: None
    with pytest.raises(RuntimeError, match="already wrapped"):
        with hcn.bind(scheduler, NativeConditioningNoiseMixOp()):
            pass


@needs_cuda
@pytest.mark.parametrize("t", [0.999, 0.5, 1.0, "0.999", "0.5"])
def test_the_cuda_row_is_bitwise_the_provider(t):
    """The registry's CUDA ``conditioning_noise_mix`` (what the generation tool binds), on a GPU."""

    from rl_engine.backends.cuda.conditioning.conditioning_noise_mix import (
        ConditioningNoiseMixCudaOp,
    )
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("conditioning_noise_mix", device="cuda")
    assert isinstance(op, ConditioningNoiseMixCudaOp)
    _check(op, t, device="cuda", shape=(1, 48, 31, 30, 52))


def test_an_unreadable_scale_noise_source_is_unknown():
    scheduler = _scheduler()
    namespace: dict = {}
    exec("def scale_noise(self, sample, timestep, noise):\n    return sample\n", namespace)
    scheduler.__class__ = type(
        "_Generated", (type(scheduler),), {"scale_noise": namespace["scale_noise"]}
    )
    assert hcn.scale_noise_variant(scheduler) == "unknown"


@pytest.mark.parametrize(
    "timestep",
    [
        1,  # an int: the provider takes it, the row's validator refuses it
        torch.tensor(0.999, dtype=torch.float64),
        torch.tensor(0.999, requires_grad=True),
    ],
    ids=["int", "float64", "requires_grad"],
)
def test_a_timestep_the_row_refuses_stays_on_the_provider(timestep):
    scheduler = _scheduler()
    sample, noise = _case(torch.float32)
    t = timestep.detach() if isinstance(timestep, torch.Tensor) else timestep
    want = scheduler.scale_noise(sample, t, noise)
    with hcn.bind(scheduler, NativeConditioningNoiseMixOp()) as calls:
        got = scheduler.scale_noise(sample, timestep, noise)
    assert calls["row"] == 0 and calls["provider"] == 1
    assert torch.equal(got.detach().view(-1), want.view(-1))


@needs_cuda
def test_a_cpu_timestep_with_a_cuda_sample_stays_on_the_provider():
    scheduler = _scheduler()
    sample, noise = _case(torch.float32, device="cuda")
    t = torch.tensor(0.5)
    want = scheduler.scale_noise(sample, t, noise)
    with hcn.bind(scheduler, NativeConditioningNoiseMixOp()) as calls:
        got = scheduler.scale_noise(sample, t, noise)
    assert calls == {"row": 0, "provider": 1, "scale_noise": hcn.scale_noise_variant(scheduler)}
    assert torch.equal(got, want)
