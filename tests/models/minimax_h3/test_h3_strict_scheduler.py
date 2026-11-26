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
    with _bound(scheduler) as calls:
        scheduler.set_timesteps(10, device="cpu")
        x = torch.zeros(4, 8)
        out = scheduler.step(torch.ones(4, 8), scheduler.timesteps[0], x)
    assert isinstance(out, scheduling.MiniMaxH3SchedulerOutput)
    assert scheduler.step_index == 1
    assert calls["row"] == 1 and calls["provider"] == 0  # the row's branch built that output


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


class _ShiftedGrid:
    """A sigma op whose grid is off by one bit: the binding must refuse it."""

    def __call__(self, num_inference_steps, *, shift, device=None):
        grid = NativeH3SigmaScheduleOp()(num_inference_steps, shift=shift, device=device)
        sigmas = grid.sigmas.clone()
        sigmas[1] = torch.nextafter(sigmas[1], torch.tensor(2.0))
        return type(grid)(**{**grid.__dict__, "sigmas": sigmas})


def test_a_sigma_grid_that_differs_from_the_providers_is_refused():
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    with hss.bind(
        scheduler, sigma_op=_ShiftedGrid(), step_op=NativeH3OdeStepOp(), grid_device="cpu"
    ) as calls:
        with pytest.raises(RuntimeError, match="differs from the provider's grid"):
            scheduler.set_timesteps(10, device="cpu")
    assert calls["grids"] == 0


def test_extra_arguments_are_the_providers_and_are_counted():
    scheduler = scheduling.MiniMaxH3Scheduler(shift=12.0)
    x = torch.zeros(4, 8)
    with _bound(scheduler) as calls:
        scheduler.set_timesteps(10, device="cpu")
        out = scheduler.step(torch.ones(4, 8), scheduler.timesteps[0], x, True)  # 4th positional
        scheduler._step_index = None
        out2 = scheduler.step(torch.ones(4, 8), scheduler.timesteps[0], x, return_dict=False)
    assert isinstance(out, scheduling.MiniMaxH3SchedulerOutput)
    assert isinstance(out2, tuple)
    assert calls["row"] == 1 and calls["provider"] == 1 and calls["provider_steps"] == [None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_production_ops_bind_on_cuda_bitwise():
    """The candidate arm's ops, ``H3SigmaScheduleCudaOp`` and ``H3OdeStepCUDAOp``, on CUDA."""

    from rl_engine.backends.cuda.flow.ode_step import H3OdeStepCUDAOp
    from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
        H3SigmaScheduleCudaOp,
    )

    try:
        sigma_op, step_op = H3SigmaScheduleCudaOp(), H3OdeStepCUDAOp()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"the CUDA ops are not available here: {exc}")
    gen = torch.Generator("cuda").manual_seed(0)
    steps, shift = 10, 12.0
    want = scheduling.MiniMaxH3Scheduler(shift=shift)
    want.set_timesteps(steps, device="cuda")
    got_sched = scheduling.MiniMaxH3Scheduler(shift=shift)
    x = torch.randn(37, 48, device="cuda", generator=gen)
    w = x.clone()
    with hss.bind(got_sched, sigma_op=sigma_op, step_op=step_op, grid_device="cuda") as calls:
        got_sched.set_timesteps(steps, device="cuda")
        assert torch.equal(_bits(got_sched.sigmas.cpu()), _bits(want.sigmas.cpu()))
        for t in want.timesteps:
            v = torch.randn(37, 48, device="cuda", generator=gen)
            w = want.step(v, t, w, return_dict=False)[0]
            x = got_sched.step(v, t, x, return_dict=False)[0]
            assert torch.equal(_bits(x), _bits(w))
    assert calls["grids"] == 1 and calls["row"] > 0
