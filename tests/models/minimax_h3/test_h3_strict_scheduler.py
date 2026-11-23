# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The scheduler binding of ``h3_strict_scheduler``: rows where bitwise, the provider elsewhere."""

from __future__ import annotations

import pytest
import torch

from rl_engine.reference.flow.ode_step import NativeH3OdeStepOp
from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp
from rl_engine.validation.models import h3_strict_scheduler as hss

scheduling = pytest.importorskip("diffusers.schedulers.scheduling_minimax_h3")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(torch.uint8)


def _run(scheduler, steps, *, seed=0, shape=(37, 48)):
    gen = torch.Generator().manual_seed(seed)
    scheduler.set_timesteps(steps, device="cpu")
    x = torch.randn(shape, generator=gen)
    out = [x]
    for t in scheduler.timesteps:
        v = torch.randn(shape, generator=gen)
        x = scheduler.step(v, t, x, return_dict=False)[0]
        out.append(x)
    return scheduler.sigmas.clone(), scheduler.timesteps.clone(), out


def _bound(scheduler, **kw):
    return hss.bind(
        scheduler,
        sigma_op=NativeH3SigmaScheduleOp(),
        step_op=NativeH3OdeStepOp(),
        grid_device="cpu",
        **kw,
    )


@pytest.mark.parametrize("shift", [12.0, 3.0])
@pytest.mark.parametrize("steps", [10, 50])
def test_the_bound_scheduler_is_bitwise_the_provider(shift, steps):
    sigmas, timesteps, want = _run(scheduling.MiniMaxH3Scheduler(shift=shift), steps)
    scheduler = scheduling.MiniMaxH3Scheduler(shift=shift)
    with _bound(scheduler) as calls:
        got_sigmas, got_timesteps, got = _run(scheduler, steps)
    assert torch.equal(_bits(got_sigmas), _bits(sigmas))
    assert torch.equal(_bits(got_timesteps), _bits(timesteps))
    assert len(got) == len(want)
    assert all(torch.equal(_bits(g), _bits(w)) for g, w in zip(got, want, strict=True))
    assert calls["grids"] == 1
    assert calls["row"] + calls["provider"] == len(timesteps)
    assert calls["row"] > 0
    # the row runs exactly where the timestep's sigma is the grid's sigma
    exact = [
        i
        for i, t in enumerate(timesteps)
        if torch.equal(_bits(1 - t.reshape(1)), _bits(sigmas[i].reshape(1)))
    ]
    assert calls["row"] == len(exact)
    assert calls["provider_steps"] == [i for i in range(len(timesteps)) if i not in exact]
    assert "step" not in scheduler.__dict__ and "set_timesteps" not in scheduler.__dict__


def test_return_dict_gives_the_scheduler_output():
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    with _bound(scheduler):
        scheduler.set_timesteps(10, device="cpu")
        x = torch.zeros(4, 8)
        out = scheduler.step(torch.ones(4, 8), scheduler.timesteps[0], x)
    assert isinstance(out, scheduling.MiniMaxH3SchedulerOutput)
    assert scheduler.step_index == 1


def test_half_precision_samples_stay_on_the_provider():
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    reference = scheduling.MiniMaxH3Scheduler(shift=12.0)
    reference.set_timesteps(10, device="cpu")
    x = torch.randn(4, 8).bfloat16()
    v = torch.randn(4, 8)
    want = reference.step(v, reference.timesteps[0], x, return_dict=False)[0]
    with _bound(scheduler) as calls:
        scheduler.set_timesteps(10, device="cpu")
        got = scheduler.step(v, scheduler.timesteps[0], x, return_dict=False)[0]
    assert torch.equal(_bits(got), _bits(want))
    assert calls == {"grids": 1, "row": 0, "provider": 1, "provider_steps": [0]}


def test_an_explicit_schedule_is_used_as_given():
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    with _bound(scheduler) as calls:
        scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0], device="cpu")
    assert calls["grids"] == 0
    assert scheduler.sigmas.tolist() == [1.0, 0.5, 0.0]


def test_binding_twice_is_refused():
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    with _bound(scheduler):
        with pytest.raises(RuntimeError, match="already wrapped"):
            with _bound(scheduler):
                pass
