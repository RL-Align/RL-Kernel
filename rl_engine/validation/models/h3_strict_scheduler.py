# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bind a ``MiniMaxH3Scheduler`` to the RFC #420 scheduler rows, where they are bitwise.

Two rows replace the provider scheduler's arithmetic:

* ``h3_sigma_schedule`` builds the sigma grid. It is byte-frozen against the provider's
  ``set_timesteps`` (see ``rl_engine.reference.minimax_h3.sigma_schedule``), so the bound
  ``set_timesteps`` passes the op's grid to the provider as an explicit ``sigmas`` schedule,
  which the provider uses verbatim. Before that it builds the provider's own grid on a copy of
  the scheduler and requires the op's grid to equal it bitwise; a difference raises.
  ``timesteps = 1 - sigmas[:-1]`` stays the provider's.
* ``h3_ode_step`` is the Euler step ``x0 = x_t + sigma * v``,
  ``x_next = r * x_t + (1 - r) * x0`` with ``r = sigma_next / sigma``, in FP32.

The provider step computes ``x0`` with the sigma recovered from the timestep,
``1 - (1 - sigma)``, and the ratio with the grid sigma. The two agree exactly for
``sigma >= 0.5`` and for some smaller values, not for all of them; the op takes a single sigma.
A step whose timestep sigma equals its grid sigma bitwise therefore runs the op, which is then
the same FP32 program as the provider step. Any other step runs the provider step, and is
counted. ``bind`` yields those counts, so a report states how many steps ran each path.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator

import torch

ROWS = {"sigma_schedule": "h3_sigma_schedule", "ode_step": "h3_ode_step"}


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


@contextmanager
def bind(
    scheduler: Any, *, sigma_op: Any, step_op: Any, grid_device: str = "cuda"
) -> Iterator[dict[str, Any]]:
    """Route ``scheduler.set_timesteps`` / ``scheduler.step`` to the two rows; restore on exit.

    ``sigma_op(n, shift=..., device=...)`` returns an ``H3SigmaSchedule`` (``.sigmas``);
    ``step_op(xt, v, sigma, sigma_next)`` returns ``(x_next, x0)`` in ``xt``'s dtype. The grid
    is built on ``grid_device`` (the CUDA op's), then handed to the provider on the CPU.
    Yields ``{"grids": n, "row": n, "provider": n, "provider_steps": [...]}``.
    """

    for name in ("set_timesteps", "step"):
        if name in scheduler.__dict__:
            raise RuntimeError(f"scheduler.{name} is already wrapped on the instance")
    cls = type(scheduler)
    calls: dict[str, Any] = {"grids": 0, "row": 0, "provider": 0, "provider_steps": []}

    def provider_grid(num_inference_steps: int) -> torch.Tensor:
        probe = copy.copy(scheduler)
        for name in ("set_timesteps", "step"):
            probe.__dict__.pop(name, None)
        cls.set_timesteps(probe, num_inference_steps, device="cpu")
        return probe.sigmas

    def set_timesteps(num_inference_steps=None, device=None, sigmas=None, **kwargs):
        if sigmas is not None or num_inference_steps is None:
            # an explicit schedule is the caller's, not the row's
            return cls.set_timesteps(
                scheduler, num_inference_steps, device=device, sigmas=sigmas, **kwargs
            )
        grid = sigma_op(int(num_inference_steps), shift=float(scheduler.shift), device=grid_device)
        if not _same_bits(provider_grid(int(num_inference_steps)), grid.sigmas.cpu()):
            raise RuntimeError("the sigma-grid row differs from the provider's grid")
        calls["grids"] += 1
        return cls.set_timesteps(
            scheduler, None, device=device, sigmas=grid.sigmas.detach().cpu(), **kwargs
        )

    def step(model_output, timestep, sample, *args, return_dict=True, **kwargs):
        # anything the row does not represent (extra arguments, an integer timestep) is the
        # provider's, forwarded as given and counted
        if (
            args
            or kwargs
            or not isinstance(timestep, torch.Tensor)
            or not timestep.is_floating_point()
        ):
            calls["provider"] += 1
            calls["provider_steps"].append(None)
            if args:  # the provider's own 4th positional is ``return_dict``: pass it as given
                return cls.step(scheduler, model_output, timestep, sample, *args, **kwargs)
            return cls.step(
                scheduler, model_output, timestep, sample, return_dict=return_dict, **kwargs
            )
        index = scheduler.step_index
        if index is None:
            index = (
                scheduler.index_for_timestep(timestep)
                if scheduler.begin_index is None
                else scheduler.begin_index
            )
        sigma = scheduler.sigmas[index].to(device=sample.device, dtype=torch.float32)
        sigma_next = scheduler.sigmas[index + 1].to(device=sample.device, dtype=torch.float32)
        # the provider's x0 sigma, computed as it does (in the sample's dtype)
        from_timestep = 1 - timestep.to(device=sample.device, dtype=sample.dtype)
        if (
            sample.dtype != torch.float32
            or model_output.dtype != torch.float32
            or not _same_bits(from_timestep, sigma)
        ):
            calls["provider"] += 1
            calls["provider_steps"].append(int(index))
            return cls.step(scheduler, model_output, timestep, sample, return_dict=return_dict)
        x_next, _ = step_op(sample.contiguous(), model_output.contiguous(), sigma, sigma_next)
        scheduler._step_index = index + 1
        calls["row"] += 1
        if not return_dict:
            return (x_next,)
        from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput

        return MiniMaxH3SchedulerOutput(prev_sample=x_next)

    scheduler.set_timesteps = set_timesteps
    scheduler.step = step
    try:
        yield calls
    finally:
        scheduler.__dict__.pop("set_timesteps", None)
        scheduler.__dict__.pop("step", None)
