# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``h3_image_evidence_schema`` (RFC #738): the sealed six-step H3-Image trajectory.

A synthetic loop drives the recorder exactly the way the released sampler is laid out:
packed positions from ``image_position_grid``, a per-row timestep table with references
held at ``t = 0.999``, head ``i`` at step ``i``, and the FP32 velocity-form update applied to
the generated rows only. The model is a fixed function of the latents, so every check is
about the record, not about a network. One Diffusers-gated test confirms that the loop's
timestep table and update equal the pinned pipeline and scheduler bitwise.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3_image import fingerprint as fp
from rl_engine.models.minimax_h3_image import position_grid as pg
from rl_engine.models.minimax_h3_image import trajectory as tr

WIDTH = 96
REF_GRIDS = [(4, 4), (2, 6)]
TARGET = (4, 6)


def _f32(hex_value: str) -> float:
    return tr.f32_from_hex(hex_value)


def _inputs(task: str, seed: int, layout: pg.PackedPositions, noise, refs, conditioned) -> dict:
    canvas = [TARGET[0] * 16, TARGET[1] * 16]
    return {
        "task": task,
        "seed": seed,
        "canvas": {
            "requested": canvas,
            "canvas": canvas,
            "latent": list(TARGET),
            "min_generation_area": 0,
            "resize": "none",
        },
        "presentation": {
            "prompt_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "token_tags_sha256": "c" * 64,
            "num_text_tokens": layout.num_text_tokens,
        },
        "references": [
            {
                "source_sha256": f"{index:064x}",
                "vae_input_shape": [1, 3, 1, h * 16, w * 16],
                "latent": [h, w],
                "posterior_fp16_sha256": tr.tensor_digest(ref.half()),
                "conditioned_rows_sha256": tr.tensor_digest(ref),
            }
            for index, ((h, w), ref) in enumerate(zip(REF_GRIDS, refs))
        ],
        "vae": {
            "latent_stats_sha256": fp.vae_facts()["latent_stats_sha256"],
            "tile_size": 256,
            "tile_overlap_min": 64,
            "encode_tiled": True,
            "decode_tiled": True,
        },
        "layout": {
            "position_sha256_fp64": layout.artifact()["sha256_fp64"],
            "position_sha256_fp32": layout.artifact()["sha256_fp32"],
            "sequence_length": layout.sequence_length,
            "image_indices_sha256": tr.tensor_digest(
                torch.arange(layout.num_text_tokens, layout.sequence_length)
            ),
            "text_indices_sha256": tr.tensor_digest(torch.arange(layout.num_text_tokens)),
            "num_reference_rows": layout.num_reference_rows,
            "conditioned_reference_rows_sha256": conditioned,
        },
        "noise": {
            "target_sha256": tr.tensor_digest(noise),
            "references": [tr.tensor_digest(r) for r in refs],
        },
        "schedule": tr.released_schedule(),
    }


def timestep_plan(schedule: dict, step: int, num_text: int, num_ref_rows: int, seq: int):
    """``MiniMaxH3ImageSetTimestepsStep``: refs held at 0.999, unique sorted table + inverse."""

    rows = torch.full((seq,), _f32(schedule["timesteps"][step]), dtype=torch.float32)
    rows[num_text : num_text + num_ref_rows] = tr.f32_from_hex(schedule["reference_timestep"])
    return torch.unique(rows, sorted=True, return_inverse=True)


def velocity_update(latents, velocity, sigma: float, sigma_next: float, num_ref_rows: int):
    """FP32 velocity-form Euler update of the generated rows only."""

    out = latents.clone()
    gen = latents[:, num_ref_rows:].to(torch.float32)
    step = torch.tensor(sigma, dtype=torch.float32) - torch.tensor(sigma_next, dtype=torch.float32)
    out[:, num_ref_rows:] = (gen + step * velocity[:, num_ref_rows:].to(torch.float32)).to(
        latents.dtype
    )
    return out


def run(
    task: str = "edit",
    *,
    seed: int = 0,
    perturb_step: int | None = None,
    perturb_rows: str = "generated",
    pdd_offset: int = 0,
    move_references: bool = False,
    keep_tensors: bool = False,
    decoded_bias: int = 0,
) -> tuple[dict, tr.ImageTrajectoryRecorder]:
    grids = REF_GRIDS if task == "edit" else []
    layout = pg.packed_positions(9, grids, TARGET)
    r = layout.num_reference_rows
    generator = torch.Generator().manual_seed(seed)
    target_rows = (TARGET[0] // 2) * (TARGET[1] // 2)
    noise = torch.randn(1, target_rows, WIDTH, generator=generator)
    refs = [torch.randn(1, (h // 2) * (w // 2), WIDTH, generator=generator) for h, w in grids]
    latents = torch.cat(refs + [noise], dim=1)
    conditioned = tr.tensor_digest(latents[:, :r]) if r else ""
    inputs = _inputs(task, seed, layout, noise, refs, conditioned)
    recorder = tr.ImageTrajectoryRecorder(
        label=f"{task}-{seed}", inputs=inputs, keep_tensors=keep_tensors
    )
    schedule = inputs["schedule"]
    for step in range(fp.NUM_PDD_HEADS):
        values, indices = timestep_plan(schedule, step, 9, r, layout.sequence_length)
        velocity = torch.sin(latents * (step + 1)) * 0.5
        if step == perturb_step:
            rows = slice(0, r) if perturb_rows == "reference" else slice(r, None)
            velocity[:, rows] = velocity[:, rows] + 1e-3
        latents = velocity_update(
            latents,
            velocity,
            _f32(schedule["step_sigmas"][step]),
            _f32(schedule["step_sigmas"][step + 1]),
            r,
        )
        if move_references and step == 2:
            latents[:, :r] += 1.0
        recorder.record_step(
            step,
            pdd_head=step + pdd_offset,
            timestep_values=values,
            timestep_indices=indices,
            velocity=velocity,
            image_latents=latents,
        )
    pixels = ((latents[:, r:].clamp(-1, 1) + 1) * 127).to(torch.uint8) + decoded_bias
    recorder.record_decoded(pixels, output_size=[TARGET[0] * 16, TARGET[1] * 16])
    return recorder.seal(), recorder


@pytest.mark.parametrize("task", tr.TASKS)
def test_seal_verify_round_trip(task: str) -> None:
    artifact, _ = run(task)
    assert tr.loads(tr.dumps(artifact)) == artifact
    assert artifact["steps"] == 6
    assert [r["pdd_head"] for r in artifact["step_records"]] == list(range(6))


def test_repeated_runs_are_bitwise_equal() -> None:  # I-C0 / I-C2
    assert tr.first_drift(run()[0], run()[0]) is None


def test_edit_reference_rows_hold_their_conditioned_value_every_step() -> None:  # I-C6
    artifact, _ = run("edit")
    digests = {r["reference_rows_sha256"] for r in artifact["step_records"]}
    assert digests == {artifact["inputs"]["layout"]["conditioned_reference_rows_sha256"]}


def test_t2i_has_no_reference_rows() -> None:
    artifact, _ = run("t2i")
    assert {r["reference_rows_sha256"] for r in artifact["step_records"]} == {""}
    assert all(len(r["timestep_table"]["values"]) == 1 for r in artifact["step_records"])


def test_edit_timestep_table_holds_the_schedule_and_reference_timestep() -> None:
    artifact, _ = run("edit")
    ref_t = artifact["inputs"]["schedule"]["reference_timestep"]
    for index, record in enumerate(artifact["step_records"]):
        expected = {artifact["inputs"]["schedule"]["timesteps"][index], ref_t}
        assert set(record["timestep_table"]["values"]) == expected


def test_updating_reference_rows_fails_while_recording() -> None:  # I14
    with pytest.raises(ValueError, match="reference rows differ"):
        run("edit", move_references=True)


def test_wrong_pdd_head_fails_while_recording() -> None:  # I1
    with pytest.raises(ValueError, match="must run PDD head 0"):
        run(pdd_offset=1)


def test_incomplete_trajectory_cannot_be_sealed() -> None:
    _, recorder = run()
    recorder.records.pop()
    with pytest.raises(ValueError, match="recorded 5 of 6"):
        recorder.seal()


# --- drift localisation -------------------------------------------------------------------


def test_generated_velocity_drift_is_located() -> None:
    drift = tr.first_drift(run()[0], run(perturb_step=3)[0])
    assert drift == {
        "stage": "step",
        "step": 3,
        "pdd_head": 3,
        "kind": "velocity",
        "rows": "all_image",
    }


def test_reference_only_velocity_drift_does_not_reach_the_latent() -> None:
    lhs, rhs = run()[0], run(perturb_step=2, perturb_rows="reference")[0]
    drift = tr.first_drift(lhs, rhs)
    assert (drift["step"], drift["kind"]) == (2, "velocity")
    # The update ignores reference-row velocity: generated rows and latents still match.
    for a, b in zip(lhs["step_records"], rhs["step_records"]):
        assert a["generated_velocity_sha256"] == b["generated_velocity_sha256"]
        assert a["latent_sha256"] == b["latent_sha256"]


def test_tensor_files_add_mismatch_counts(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    lhs, rec_l = run(keep_tensors=True)
    rhs, rec_r = run(perturb_step=4, keep_tensors=True)
    rec_l.save_tensors(tmp_path / "l.safetensors")
    rec_r.save_tensors(tmp_path / "r.safetensors")
    drift = tr.first_drift(
        lhs, rhs, tensors_lhs=tmp_path / "l.safetensors", tensors_rhs=tmp_path / "r.safetensors"
    )
    assert drift["step"] == 4 and drift["mismatch_count"] > 0
    assert drift["max_abs_diff"] == pytest.approx(1e-3, rel=1e-3)


def test_differing_inputs_are_reported_before_steps() -> None:
    drift = tr.first_drift(run(seed=0)[0], run(seed=1)[0])
    assert drift["stage"] == "inputs"
    assert drift["field"] == "seed"
    assert "noise.references" in drift["fields"] or "noise.target_sha256" in drift["fields"]


def test_reference_order_is_an_input_drift() -> None:  # I9
    lhs = run()[0]
    rhs = copy.deepcopy(lhs)
    refs = rhs["inputs"]["references"]
    refs[0], refs[1] = refs[1], refs[0]
    rhs["inputs_sha256"] = {k: tr._sha(rhs["inputs"][k]) for k in tr.INPUT_FIELDS}
    rhs["artifact_sha256"] = tr._sha({k: v for k, v in rhs.items() if k != "artifact_sha256"})
    assert tr.first_drift(lhs, rhs)["field"] == "references[0]"


def test_decoded_bytes_are_secondary_evidence() -> None:
    assert tr.first_drift(run()[0], run(decoded_bias=1)[0]) == {
        "stage": "decoded",
        "field": "sha256",
    }


# --- fail closed --------------------------------------------------------------------------


def _reseal(artifact: dict) -> dict:
    artifact["inputs_sha256"] = {k: tr._sha(artifact["inputs"][k]) for k in tr.INPUT_FIELDS}
    artifact["artifact_sha256"] = tr._sha(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    )
    return artifact


def test_edited_artifact_fails_to_load() -> None:
    artifact, _ = run()
    artifact["step_records"][1]["latent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact_sha256"):
        tr.verify(artifact)
    with pytest.raises(ValueError, match="hash chain"):
        tr.verify(_reseal(artifact))


@pytest.mark.parametrize(
    ("edit", "needle"),
    [
        (lambda a: a["step_records"][2].update(pdd_head=3), "ran PDD head"),
        (lambda a: a["step_records"][1].update(sigma=tr.f32_hex(0.9)), "sigma disagrees"),
        (
            lambda a: a["step_records"][0]["timestep_table"].update(
                values=[a["inputs"]["schedule"]["timesteps"][0]]
            ),
            "timestep table holds",
        ),  # I21: references not on their own timestep
        (lambda a: a["inputs"]["schedule"].update(euler_update_form="blend"), "released PDD"),
        (
            lambda a: a["inputs"]["schedule"]["step_sigmas"].__setitem__(5, tr.f32_hex(0.25)),
            "released PDD",
        ),  # I3
        (lambda a: a["inputs"].update(task="t2i"), "no references"),
        (lambda a: a["inputs"]["canvas"].update(canvas=[80, 96]), "multiple of 32"),  # I5
        (lambda a: a["inputs"]["canvas"].update(latent=[5, 6]), "canvas / 16"),
        (lambda a: a["inputs"]["noise"]["references"].pop(), "reference noise"),
        (lambda a: a["inputs"]["layout"].pop("num_reference_rows"), "missing"),
        (lambda a: a.update(steps=7), "exactly 6"),
    ],
)
def test_inconsistent_artifacts_fail_closed(edit, needle: str) -> None:
    artifact, _ = run()
    edit(artifact)
    with pytest.raises(ValueError, match=needle):
        tr.verify(_reseal(artifact))


def test_recorder_rejects_bad_inputs() -> None:
    artifact, _ = run("t2i")
    inputs = copy.deepcopy(artifact["inputs"])
    inputs["task"] = "edit"
    with pytest.raises(ValueError, match="at least one reference"):
        tr.ImageTrajectoryRecorder(label="x", inputs=inputs)
    inputs = copy.deepcopy(artifact["inputs"])
    inputs["extra"] = 1
    with pytest.raises(ValueError, match="exactly"):
        tr.ImageTrajectoryRecorder(label="x", inputs=inputs)


def test_artifact_is_json_and_bound_to_the_fingerprint(monkeypatch) -> None:
    artifact, _ = run()
    assert json.loads(tr.dumps(artifact)) == artifact
    original = fp.build_fingerprint()
    monkeypatch.setattr(fp, "build_fingerprint", lambda: {**original, "fingerprint_sha256": "0"})
    with pytest.raises(ValueError, match="different H3-Image fingerprint"):
        tr.verify(artifact)
    tr.verify(artifact, check_fingerprint=False)


# --- pinned pipeline ----------------------------------------------------------------------


def test_loop_matches_pinned_scheduler_and_timestep_plan() -> None:
    # An older Diffusers (0.41) has a MiniMaxH3Scheduler without shift_provided_sigmas /
    # euler_update_form; only the pinned commit ships the H3-Image pipeline package.
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    schedulers = pytest.importorskip("diffusers.schedulers")
    scheduler = schedulers.MiniMaxH3Scheduler(
        shift_provided_sigmas=False, euler_update_form="velocity"
    )
    config_sigmas = fp.pinned_checkpoint()["configs"]["pdd_config"]["sigmas"]
    timesteps = torch.tensor([1.0 - s for s in config_sigmas[:-1]], dtype=torch.float32)
    scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]), device="cpu")
    schedule = tr.released_schedule()
    assert [tr.f32_hex(s) for s in scheduler.sigmas.tolist()] == schedule["step_sigmas"]
    assert [tr.f32_hex(t) for t in scheduler.timesteps.tolist()] == schedule["timesteps"]

    layout = pg.packed_positions(9, REF_GRIDS, TARGET)
    r, seq = layout.num_reference_rows, layout.sequence_length
    generator = torch.Generator().manual_seed(3)
    latents = torch.randn(1, seq - 9, WIDTH, generator=generator)
    image_indices = torch.arange(9, seq)
    for step, timestep in enumerate(scheduler.timesteps):
        # MiniMaxH3ImageSetTimestepsStep, verbatim.
        rows = torch.full((seq,), float(timestep), dtype=torch.float32)
        rows[image_indices[:r]] = 0.999
        expected = torch.unique(rows, sorted=True, return_inverse=True)
        values, indices = timestep_plan(schedule, step, 9, r, seq)
        assert torch.equal(values, expected[0]) and torch.equal(indices, expected[1])

        velocity = torch.randn(latents.shape, generator=generator)
        ours = velocity_update(
            latents,
            velocity,
            _f32(schedule["step_sigmas"][step]),
            _f32(schedule["step_sigmas"][step + 1]),
            r,
        )
        theirs = scheduler.step(velocity[:, r:], timestep, latents[:, r:], return_dict=False)[0]
        assert torch.equal(ours[:, r:], theirs)
        assert torch.equal(ours[:, :r], latents[:, :r])
        latents = ours
