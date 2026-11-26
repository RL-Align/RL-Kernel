# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``ws1_full_h3_image_t2i`` / ``ws1_full_h3_image_edit`` (RFC #738): six strict steps.

The chain is ``full_chain.run_chain`` with the strict transformer: full-width synthetic weights cut
to two blocks (as in ``test_h3_image_step``), the tiny image VAE and fixed text features. The
released-weight run is ``tools/validation/models/h3_image_trajectory.py``. Pins:

* six model calls, heads 0..5 in order, and reference rows equal to their conditioned value after
  every update (I-C6);
* the batched loop reproduces the chain's step records exactly: alone (I-C2), at either batch
  position next to another request (I-C1, I-C5) and in a physically permuted layout (I-C3);
* a changed start row or a tampered record is located at its step and field;
* strict and Diffusers arms agree on inputs, heads, sigmas and timestep tables at every step and
  differ only in values, which ``trajectory_distance`` measures.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_engine.models.minimax_h3_image import full_chain as fc
from rl_engine.models.minimax_h3_image import pdd_head as ph
from rl_engine.models.minimax_h3_image import trajectory as tj
from rl_engine.models.minimax_h3_image import velocity_step as vs
from rl_engine.validation.models import h3_block
from rl_engine.validation.models import h3_image_step as hs
from rl_engine.validation.models import h3_image_trajectory as ht

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image")
sys.path.insert(0, str(Path(__file__).parent))

import test_h3_image_step as step_tests  # noqa: E402
import test_h3_image_vae as vae_tests  # noqa: E402

BLOCKS = step_tests.BLOCKS


def _image(height: int, width: int, seed: int):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


REQUESTS = {
    "t2i": fc.ImageRequest("a cat", 64, 96, seed=5, min_generation_area=1024),
    # two references of different geometry (I10)
    "edit": fc.ImageRequest(
        "<Picture 1> and <Picture 2>",
        64,
        96,
        seed=9,
        min_generation_area=1024,
        references=(_image(1030, 1100, 1), _image(300, 1500, 2)),
    ),
}


def _text(request: fc.ImageRequest, vision_rows: int = 3) -> fc.TextConditions:
    rows = 7 + vision_rows * len(request.references)
    tags = torch.ones(rows, dtype=torch.long)
    start = 1
    for _ in request.references:
        tags[start : start + vision_rows] = 0
        start += vision_rows + 1
    g = torch.Generator().manual_seed(3)
    embeds = torch.randn(1, rows, 5120, generator=g).cuda()
    return fc.TextConditions("fixed", request.prompt, list(range(100, 100 + rows)), embeds, tags)


@pytest.fixture(scope="module")
def weights():
    out = step_tests._synthetic_weights()
    yield out
    out.clear()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def strict(weights):
    heads = ph.from_diffusers(weights, sealed=False)
    return hs.StrictImageTransformer(weights, heads, num_layers=BLOCKS, ops=h3_block.CandidateOps())


@pytest.fixture(scope="module")
def vae():
    from rl_engine.models.minimax_h3_image import image_vae as iv

    return iv.ImageVAEContract(vae_tests._tiny_vae("cuda"), require_release=False)


def _sampler():
    return vs.ImageVelocitySampler(device="cuda")


def _chain(request, denoiser, vae):
    return fc.run_chain(
        request,
        _text(request),
        vae_contract=vae,
        denoiser=denoiser,
        device="cuda",
        sampler=_sampler(),
        keep_tensors=True,
    )


_RUNS: dict = {}


def _strict_run(task, strict, vae, seed_offset=0):
    key = (task, seed_offset)
    if key not in _RUNS:
        request = REQUESTS[task]
        request = dataclasses.replace(request, seed=request.seed + seed_offset)
        denoiser = ht.CountingDenoiser(hs.StrictImageDenoiser(strict))
        run = _chain(request, denoiser, vae)
        _RUNS[key] = (run, denoiser, ht.chain_start(run, _text(request)))
    return _RUNS[key]


# --- the strict chain --------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_strict_chain_runs_six_heads_and_keeps_references(strict, vae, task):
    run, denoiser, start = _strict_run(task, strict, vae)
    fc.verify(run.artifact)
    assert denoiser.readback() == {"calls": list(range(6)), "six_in_order": True}
    records = run.artifact["trajectory"]["step_records"]
    assert [s["pdd_head"] for s in records] == list(range(6))
    fixed = ht.fixed_reference_rows(run.artifact)
    if task == "edit":
        refs = run.artifact["trajectory"]["inputs"]["references"]
        assert len({tuple(r["latent"]) for r in refs}) == 2  # heterogeneous geometry
        assert fixed["fixed"] is True and fixed["num_reference_rows"] > 0
        assert all(fixed["steps_equal_to_conditioned"])
    else:
        assert fixed == {"num_reference_rows": 0, "fixed": None}
    assert len({s["latent_sha256"] for s in records}) == 6


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_loop_reproduces_the_chain(strict, vae, task):
    run, _, start = _strict_run(task, strict, vae)
    loop = ht.run_loop(strict, start, sampler=_sampler, keep_inputs=True)
    assert loop.calls == list(range(6))
    assert ht.first_record_drift(loop.records[0], ht.chain_records(run.artifact)) is None
    assert torch.equal(loop.step_inputs[0], start.image_rows)
    assert [u["pdd_head"] for u in loop.updates[0]] == list(range(6))


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_batch_position_and_permutation_are_bitwise_at_every_step(strict, vae, task):
    run, _, start = _strict_run(task, strict, vae)
    other_run, _, other = _strict_run(task, strict, vae, seed_offset=1)
    ours, theirs = ht.chain_records(run.artifact), ht.chain_records(other_run.artifact)
    assert ht.first_record_drift(ours, theirs) is not None  # really another request
    for position, pair in ((0, [start, other]), (1, [other, start])):
        loop = ht.run_loop(strict, ht.stack(pair), sampler=_sampler)
        assert ht.first_record_drift(loop.records[position], ours) is None
        assert ht.first_record_drift(loop.records[1 - position], theirs) is None

    g = torch.Generator().manual_seed(3)
    order = torch.randperm(start.layout.sequence_length, generator=g)
    moved = ht.run_loop(strict, start, sampler=_sampler, order=order)
    assert ht.first_record_drift(moved.records[0], ours) is None


@pytest.mark.parametrize(
    ("change", "field", "rows"),
    [
        ("ulp", "latent_sha256", "generated"),  # rounded away inside the model, kept by the update
        ("large", "velocity_sha256", "all_image"),
    ],
)
def test_a_changed_start_row_drifts_at_step_zero(strict, vae, change, field, rows):
    run, _, start = _strict_run("edit", strict, vae)
    x = start.image_rows.clone()
    r = start.layout.num_reference_rows
    value = x[0, r + 1, 5]
    x[0, r + 1, 5] = (
        torch.nextafter(value, torch.tensor(math.inf)) if change == "ulp" else value + 1
    )
    loop = ht.run_loop(strict, dataclasses.replace(start, image_rows=x), sampler=_sampler)
    drift = ht.first_record_drift(loop.records[0], ht.chain_records(run.artifact))
    assert drift == {"step": 0, "pdd_head": 0, "field": field, "rows": rows}


@pytest.mark.parametrize(
    ("step", "key", "rows"),
    [
        (4, "latent_sha256", "generated"),
        (2, "reference_rows_sha256", "reference"),
        (3, "timestep_table", "schedule"),
    ],
)
def test_a_tampered_record_is_located(strict, vae, step, key, rows):
    run, _, _ = _strict_run("edit", strict, vae)
    records = ht.chain_records(run.artifact)
    bad = copy.deepcopy(records)
    bad[step][key] = "0" * 64
    assert ht.first_record_drift(records, bad) == {
        "step": step,
        "pdd_head": step,
        "field": key,
        "rows": rows,
    }
    assert ht.first_record_drift(records, records[:5])["field"] == "steps"


def test_a_moved_reference_row_is_reported(strict, vae):
    run, _, _ = _strict_run("edit", strict, vae)
    bad = copy.deepcopy(run.artifact)
    bad["trajectory"]["step_records"][3]["reference_rows_sha256"] = "0" * 64
    fixed = ht.fixed_reference_rows(bad)
    assert fixed["fixed"] is False
    assert fixed["steps_equal_to_conditioned"] == [True, True, True, False, True, True]


def test_requests_of_different_layouts_are_not_batched(strict, vae):
    _, _, t2i = _strict_run("t2i", strict, vae)
    _, _, edit = _strict_run("edit", strict, vae)
    with pytest.raises(ht.TrajectoryError, match="one packed layout"):
        ht.stack([t2i, edit])


def test_the_layout_must_be_the_chains(strict, vae):
    run, _, _ = _strict_run("edit", strict, vae)
    tags = _text(REQUESTS["edit"]).text_token_tags.clone()
    tags[1] = 1
    with pytest.raises(ht.TrajectoryError, match="not the chain's"):
        ht.chain_layout(run.artifact, tags)
    with pytest.raises(ht.TrajectoryError, match="keep_tensors"):
        ht.chain_start(dataclasses.replace(run, tensors={}), _text(REQUESTS["edit"]))


# --- against the Diffusers transformer ---------------------------------------------------------


@pytest.fixture(scope="module")
def provider(weights):
    pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3_image")
    from diffusers import MiniMaxH3ImageTransformer2DModel

    with torch.device("meta"):
        model = MiniMaxH3ImageTransformer2DModel(num_layers=BLOCKS)
    missing, unexpected = model.load_state_dict(weights, strict=True, assign=True)
    assert not missing and not unexpected
    model.rope = type(model.rope)(rope_freq_dim=16, rope_theta=10000.0).cuda()
    return model.eval()


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_arms_agree_on_everything_but_values(strict, vae, provider, task):
    run, _, _ = _strict_run(task, strict, vae)
    counted = ht.CountingDenoiser(fc.DiffusersTransformerDenoiser(provider))
    theirs = _chain(REQUESTS[task], counted, vae)
    assert counted.readback()["six_in_order"]
    ours_traj, theirs_traj = run.artifact["trajectory"], theirs.artifact["trajectory"]
    assert tj.input_mismatches(ours_traj, theirs_traj) == []
    assert ht.schedule_records(run.artifact) == ht.schedule_records(theirs.artifact)
    drift = tj.first_drift(ours_traj, theirs_traj)
    assert drift is not None and drift["step"] == 0 and drift["kind"] == "velocity"
    r = ours_traj["inputs"]["layout"]["num_reference_rows"]
    rows = ht.trajectory_distance(run.tensors, theirs.tensors, r)
    assert [row["step"] for row in rows] == list(range(6))
    for row in rows:
        assert 0 < row["velocity_target"] < 0.1 and 0 < row["latent"] < 0.1
        assert ("velocity_reference" in row) == (task == "edit")
    assert math.isfinite(ht.psnr(run.output, theirs.output))


def test_distance_of_a_run_to_itself_is_zero(strict, vae):
    run, _, _ = _strict_run("edit", strict, vae)
    r = run.artifact["trajectory"]["inputs"]["layout"]["num_reference_rows"]
    for row in ht.trajectory_distance(run.tensors, run.tensors, r):
        assert row["velocity_bitwise"] and row["latent_bitwise"]
        assert row["velocity_target"] == row["velocity_reference"] == row["latent"] == 0
    assert ht.psnr(run.output, run.output) == math.inf
    with pytest.raises(ValueError, match="uint8"):
        ht.psnr(run.output.float(), run.output.float())
