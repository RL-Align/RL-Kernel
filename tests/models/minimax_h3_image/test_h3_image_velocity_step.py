# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``image_velocity_step`` (RFC #738): generated-row-only updates on the sealed PDD grid.

Pins:

* six sampler steps equal the pinned ``MiniMaxH3ImageLoopSchedulerStep`` loop bit for bit,
  for T2I and for an edit (reference prefix untouched), on CPU and CUDA, with the reference
  op and with the Triton kernel (Diffusers-gated);
* steps run in order, head ``i`` at step ``i``, six at most;
* an op that writes a reference row is refused (``GeneratedRowsOnly``); writing every image
  row (probe I14) changes the reference prefix at step 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_engine.models.minimax_h3_image import pdd_schedule as ps
from rl_engine.models.minimax_h3_image import velocity_step as vs

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rows(n=40, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 96, generator=g).to(device)


def _velocities(n=40, device="cpu"):
    g = torch.Generator().manual_seed(99)
    return [torch.randn(1, n, 96, generator=g).to(device) for _ in range(6)]


def _bits(t):
    return t.contiguous().view(torch.int32)


def _run_ours(rows, velocities, r, op=None):
    sampler = vs.ImageVelocitySampler(op=op, device=rows.device)
    for i, v in enumerate(velocities):
        sampler.step(i, v, rows, r, pdd_head=i)
    return rows, sampler


def _run_pinned(rows, velocities, r):
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    from diffusers.modular_pipelines.minimax_h3_image.denoise import MiniMaxH3ImageLoopSchedulerStep
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    scheduler = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    scheduler.set_timesteps(sigmas=ps.released_schedule().sigmas, device=rows.device)
    components = SimpleNamespace(scheduler=scheduler)
    state = SimpleNamespace(latents=rows, noise_pred=None, num_reference_rows=r)
    for i, t in enumerate(scheduler.timesteps):
        state.noise_pred = velocities[i]
        MiniMaxH3ImageLoopSchedulerStep()(components, state, i, t)
    return state.latents


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("r", [0, 13], ids=["t2i", "edit"])
def test_six_steps_equal_the_pinned_loop(device, r) -> None:
    rows = _rows(device=device)
    vel = _velocities(device=device)
    initial = rows.clone()
    ours, sampler = _run_ours(rows.clone(), vel, r)
    theirs = _run_pinned(rows.clone(), vel, r)
    assert torch.equal(_bits(ours), _bits(theirs))
    assert torch.equal(_bits(ours[:r]), _bits(initial[:r]))
    assert not torch.equal(_bits(ours[r:]), _bits(initial[r:]))
    assert [rec["pdd_head"] for rec in sampler.records] == list(range(6))
    assert sampler.provenance()["semantic_op"] == "h3_image_velocity_step"


@CUDA
def test_triton_sampler_equals_the_reference_sampler() -> None:
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.flow.velocity_step import H3ImageVelocityStepTritonOp

    rows = _rows(n=4099, device="cuda")
    vel = _velocities(n=4099, device="cuda")
    ref, _ = _run_ours(rows.clone(), vel, 1024)
    tri, sampler = _run_ours(rows.clone(), vel, 1024, op=H3ImageVelocityStepTritonOp())
    assert torch.equal(_bits(ref), _bits(tri))
    assert sampler.backend_id == "H3ImageVelocityStepTritonOp"


def test_steps_run_in_order_six_at_most() -> None:
    rows, vel = _rows(), _velocities()
    sampler = vs.ImageVelocitySampler()
    with pytest.raises(ps.OffGridTimestep, match="expected step 0"):
        sampler.step(1, vel[1], rows, 0)
    sampler.step(0, vel[0], rows, 0)
    with pytest.raises(ps.OffGridTimestep, match="expected step 1"):
        sampler.step(0, vel[0], rows, 0)
    with pytest.raises(ps.OffGridTimestep, match="PDD head 1"):
        sampler.step(1, vel[1], rows, 0, pdd_head=2)
    for i in range(1, 6):
        sampler.step(i, vel[i], rows, 0)
    with pytest.raises(ps.OffGridTimestep):
        sampler.step(6, vel[0], rows, 0)


@pytest.mark.parametrize(
    "velocity, rows, r",
    [
        (torch.zeros(2, 40, 96), torch.zeros(40, 96), 0),  # batch of two
        (torch.zeros(40, 95), torch.zeros(40, 95), 0),  # wrong width
        (torch.zeros(39, 96), torch.zeros(40, 96), 0),  # row count mismatch
        (torch.zeros(40, 96), torch.zeros(40, 96), 40),  # no generated row left
        (torch.zeros(40, 96), torch.zeros(40, 96), -1),
    ],
)
def test_bad_shapes_fail_closed(velocity, rows, r) -> None:
    with pytest.raises(ValueError):
        vs.ImageVelocitySampler().step(0, velocity, rows, r)


def test_an_op_that_writes_a_reference_row_is_refused() -> None:
    def leaky(xt, v, s, sn):
        xt._base[0, 0] += 1.0  # writes through the view into the reference prefix
        return xt

    with pytest.raises(vs.GeneratedRowsOnly, match="step 0"):
        vs.ImageVelocitySampler(op=leaky).step(0, _velocities()[0], _rows(), 13)


def test_writing_every_row_moves_the_references_at_step_zero() -> None:  # I14
    rows, vel = _rows(), _velocities()
    good = vs.ImageVelocitySampler().step(0, vel[0], rows.clone(), 13)
    bad = vs.ImageVelocitySampler().step(0, vel[0], rows.clone(), 0)
    assert torch.equal(_bits(good[:13]), _bits(rows[:13]))
    assert not torch.equal(_bits(bad[:13]), _bits(rows[:13]))
    assert torch.equal(_bits(good[13:]), _bits(bad[13:]))
