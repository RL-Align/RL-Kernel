# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``image_sigma_pdd_schedule`` (RFC #738): the sealed six-step PDD grid.

Pins:

* the grid is the pinned ``pdd_config.json`` grid, through the FP32 round trip, six calls;
* sigmas, timesteps and the per-row timestep table equal the pinned
  ``MiniMaxH3ImageSetTimestepsStep`` and ``MiniMaxH3Scheduler`` for T2I and edit layouts
  (Diffusers-gated);
* head ``i`` belongs to step ``i`` only; nearest-head selection and off-grid timesteps fail;
* probe arms: the H3 shifted ``linspace`` construction (#420 ``h3_sigma_schedule``), the
  Diffusers ``shift_provided_sigmas=True`` path and a grid where the round trip moves a sigma
  each give a different schedule.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import pdd_schedule as ps

RELEASED = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724]
RELEASED += [0.23200887441635132, 0.0]


def _layout(refs=((4, 6), (2, 12)), target=(6, 8)) -> pl.PackedLayout:
    tags = torch.ones(9, dtype=torch.long)
    if refs:
        tags[1:4] = 0
    return pl.build_layout(tags, list(refs), target)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous().view(torch.int32)


# --- the grid ----------------------------------------------------------------------------------


def test_released_grid_is_the_sealed_pdd_grid() -> None:
    assert ps.config_sigmas() == RELEASED
    schedule = ps.released_schedule()
    grid = fp.sigma_grid()
    assert schedule.num_model_evals == fp.NUM_PDD_HEADS == 6
    assert schedule.sigmas.dtype == schedule.timesteps.dtype == torch.float32
    assert [float(s) for s in schedule.sigmas] == grid["step_sigmas"]
    assert [float(t) for t in schedule.timesteps] == grid["timesteps"]
    assert float(schedule.sigmas[-1]) == 0.0 and float(schedule.timesteps[0]) == 0.0
    assert ps.round_trip_moved(RELEASED) == []


def test_round_trip_moves_a_sigma_below_one_half() -> None:
    moved = []
    for k in range(1, 500):
        s = float(torch.tensor(k / 1000, dtype=torch.float32))
        if ps.round_trip_moved([1.0, s, 0.0]) == [1]:
            moved.append(s)
    assert moved, "some FP32 sigma below 0.5 must move through 1 - (1 - s)"
    schedule = ps.build_schedule([1.0, moved[0], 0.0])
    assert float(schedule.sigmas[1]) != moved[0]


@pytest.mark.parametrize(
    "bad", [[1.0], [1.0, 0.5], [1.0, 0.5, 0.5, 0.0], [0.5, 1.0, 0.0], [1.0, 0.5, 0.1]]
)
def test_malformed_grids_fail_closed(bad) -> None:
    with pytest.raises(ValueError):
        ps.build_schedule(bad)


def test_artifact_is_stable_and_complete() -> None:
    art = ps.artifact()
    assert art == ps.artifact()
    assert art["head_for_step"] == list(range(6))
    assert art["num_model_evals"] == 6 and art["round_trip_moved"] == []
    assert art["reference_timestep"] == float(torch.tensor(0.999, dtype=torch.float32)).hex()
    assert len(art["step_sigmas"]) == 7 and len(art["timesteps"]) == 6


# --- heads -------------------------------------------------------------------------------------


def test_each_grid_timestep_maps_to_its_own_head() -> None:
    schedule = ps.released_schedule()
    for i, t in enumerate(schedule.timesteps):
        assert ps.head_for_timestep(schedule, t) == ps.head_for_step(i) == i
        assert ps.head_for_timestep(schedule, float(t)) == i


@pytest.mark.parametrize("step", [-1, 6, 7, True, 2.0, "1"])
def test_steps_outside_the_six_are_refused(step) -> None:
    with pytest.raises(ps.OffGridTimestep):
        ps.head_for_step(step)


def test_nearest_head_selection_is_refused() -> None:
    schedule = ps.released_schedule()
    t = schedule.timesteps[3]
    for off_grid in (
        torch.nextafter(t, torch.tensor(1.0)),  # one ulp later
        torch.nextafter(t, torch.tensor(0.0)),  # one ulp earlier
        (schedule.timesteps[3] + schedule.timesteps[4]) / 2,  # between two steps
        torch.tensor(0.999),  # the reference timestep is not a model step
        torch.tensor(1.0),  # terminal: no model call
    ):
        with pytest.raises(ps.OffGridTimestep):
            ps.head_for_timestep(schedule, off_grid)
    with pytest.raises(ps.OffGridTimestep, match="not an FP32 value"):
        ps.head_for_timestep(schedule, math.pi)


# --- row tables --------------------------------------------------------------------------------


def test_t2i_table_has_one_value_and_edit_two() -> None:
    for i, (values, index) in enumerate(ps.row_timestep_plan(_layout(refs=()))):
        assert values.numel() == 1 and bool((index == 0).all())
    layout = _layout()
    for values, index in ps.row_timestep_plan(layout):
        assert values.numel() == 2 and float(values[1]) == float(
            torch.tensor(0.999, dtype=torch.float32)
        )
        refs = layout.image_indices[: layout.num_reference_rows]
        assert bool((index[refs] == 1).all())
        assert int((index == 1).sum()) == layout.num_reference_rows


def test_adaln_indices_address_timestep_then_modality() -> None:
    layout = _layout()
    _, index = ps.row_timestep_table(layout, ps.released_schedule().timesteps[2])
    rows = ps.adaln_indices(index, layout.token_tags)
    assert bool((rows == index * 3 + layout.token_tags).all())
    assert int(rows.max()) < 2 * ps.MODALITY_NUM
    with pytest.raises(ValueError):
        ps.adaln_indices(index[:-1], layout.token_tags)


# --- pinned Diffusers --------------------------------------------------------------------------


def _pipeline():
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    from diffusers.modular_pipelines.minimax_h3_image.modular_blocks_minimax_h3_image import (
        MiniMaxH3ImageBlocks,
    )
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    pipe = MiniMaxH3ImageBlocks().init_pipeline()
    pipe.update_components(
        scheduler=MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form="velocity")
    )
    return pipe


@pytest.mark.parametrize("refs", [(), ((4, 6),), ((4, 6), (2, 12), (30, 98))], ids=len)
def test_schedule_and_tables_match_the_pinned_step(refs) -> None:
    pipe = _pipeline()
    from diffusers.modular_pipelines.minimax_h3_image import before_denoise as bd
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    layout = _layout(refs=refs)
    state = PipelineState()
    for key in ("image_indices", "text_indices", "num_reference_rows"):
        state.set(key, getattr(layout, key))
    _, state = bd.MiniMaxH3ImageSetTimestepsStep()(pipe, state)
    assert list(pipe.config.sample_sigmas) == RELEASED
    ours = ps.released_schedule()
    assert torch.equal(_bits(ours.sigmas), _bits(pipe.scheduler.sigmas))
    assert torch.equal(_bits(ours.timesteps), _bits(state.get("timesteps")))
    plan = state.get("row_timestep_plan")
    assert len(plan) == ours.num_model_evals
    for (values, index), (d_values, d_index) in zip(ps.row_timestep_plan(layout, ours), plan):
        assert torch.equal(_bits(values), _bits(d_values)) and torch.equal(index, d_index.cpu())


def test_pinned_adaln_formula_and_reference_level() -> None:
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    from diffusers.models.transformers import transformer_minimax_h3_image as tm
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
        MiniMaxH3ImageModularPipeline as P,
    )

    assert tm.MINIMAX_H3_MODALITY_NUM == ps.MODALITY_NUM
    assert "timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags" in inspect.getsource(tm)
    assert P.reference_noise_aug.fget(None) == ps.REFERENCE_TIMESTEP


# --- probe arms --------------------------------------------------------------------------------


def test_h3_shifted_construction_is_a_different_schedule() -> None:
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    h3 = NativeH3SigmaScheduleOp()(7, shift=12.0)
    ours = ps.released_schedule()
    assert h3.num_model_evals == ours.num_model_evals == 6
    assert not torch.equal(_bits(h3.sigmas), _bits(ours.sigmas))
    assert not torch.equal(_bits(h3.timesteps), _bits(ours.timesteps))


def test_shift_provided_sigmas_is_a_different_schedule() -> None:
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    shifted = MiniMaxH3Scheduler(shift_provided_sigmas=True, euler_update_form="velocity")
    shifted.set_timesteps(sigmas=ps.released_schedule().sigmas)
    assert not torch.equal(_bits(shifted.sigmas), _bits(ps.released_schedule().sigmas))
