# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 Phase B ``phase_b_full_chain``: sealed chain, first-drift localization, latent gate.

Everything runs on CPU with the stand-ins of ``_toy_chain`` (seconds). Diffusers-backed
checks (the scheduler step against ``MiniMaxH3Scheduler``, a tiny real
``MiniMaxH3Transformer3DModel``) skip when Diffusers with MiniMax-H3 is not importable.

The two sealed fixtures under ``fixtures/`` are checked for their seal and structure on every
host. Their bytes are host specific (Pillow's LANCZOS resize and the oneDNN convolutions of the
audio VAE differ between hosts with the same torch and CPU capability), so byte determinism is
checked by sealing and replaying in two processes on the same host, at 4 intra-op threads. The
bitwise replay of the committed bytes runs with ``RL_KERNEL_H3_REPLAY_CHAIN_FIXTURES=1`` on the
host that sealed them; ``RL_KERNEL_H3_REGEN_CHAIN_FIXTURES=1`` reseals them.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3 import full_chain as fc
from rl_engine.models.minimax_h3 import packed_layout as pl
from tests.models.minimax_h3 import _toy_chain as toy

FIXTURES = Path(__file__).with_name("fixtures")
REGEN_ENV = "RL_KERNEL_H3_REGEN_CHAIN_FIXTURES"
REPLAY_ENV = "RL_KERNEL_H3_REPLAY_CHAIN_FIXTURES"
PROFILE_KEYS = ("torch", "cpu_capability", "platform_machine", "num_threads")
# CPU convolutions (the audio VAE) change bytes with the intra-op thread count, so the
# fixtures are sealed and replayed at a fixed one.
FIXTURE_THREADS = 4


def _variant(workflow: str) -> str:
    return "FL2VA" if workflow == "fl2va" else "Ref2VA"


def _request(workflow: str):
    return toy.fl2va_request() if workflow == "fl2va" else toy.ref2va_request()


_RUNS: dict[tuple[str, bool], fc.ChainRun] = {}


def clean_run(workflow: str, decode: bool = False) -> fc.ChainRun:
    """One clean run per workflow and process (kept tensors), shared by the tests.

    Decoding (the tiny BigVGAN on CPU) dominates the run time, so only the tests about
    decoded media decode.
    """

    if (workflow, decode) not in _RUNS:
        comps = toy.components(_variant(workflow))
        _RUNS[workflow, decode] = fc.run_chain(
            _request(workflow),
            comps,
            label=f"{workflow}-clean",
            text_source=toy.TEXT_SOURCE,
            keep_tensors=True,
            decode=decode,
        )
    return _RUNS[workflow, decode]


def injected(workflow: str, inject: fc.DriftInjection, decode: bool = False) -> fc.ChainRun:
    comps = toy.components(_variant(workflow))
    return fc.run_chain(
        _request(workflow),
        comps,
        label="injected",
        text_source=toy.TEXT_SOURCE,
        keep_tensors=True,
        inject=inject,
        decode=decode,
    )


# --- sealing and repeatability ------------------------------------------------------------


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_repeat_run_is_bitwise_equal_at_every_stage(workflow: str) -> None:
    first = clean_run(workflow, decode=True)
    again = fc.run_chain(
        _request(workflow),
        toy.components(_variant(workflow)),
        label="again",
        keep_tensors=True,
        text_source=toy.TEXT_SOURCE,
    )
    report = fc.compare(
        first.artifact, again.artifact, tensors_lhs=first.tensors, tensors_rhs=again.tensors
    )
    assert report["first_drift"] is None
    assert all(stage["equal"] for stage in report["stages"])
    assert report["gate"]["hard_gate"] == "pass"
    assert report["gate"]["decoded_media_equal"] == {"video_decode": True, "audio_decode": True}
    assert report["trajectory"]["bitwise_equal"]
    secondary = report["secondary_metrics"]  # undefined: null with the reason, not Infinity
    assert secondary["video_psnr_db"] is None and secondary["video_psnr_db_note"] == "identical"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_artifact_records_every_stage_with_provenance(workflow: str) -> None:
    art = clean_run(workflow, decode=True).artifact
    assert [s["stage"] for s in art["stages"]] == list(fc.STAGES)
    stages = {s["stage"]: s for s in art["stages"]}
    assert stages["text_conditions"]["provenance"]["source"] == "synthetic (tests)"
    assert stages["denoise"]["provenance"]["execution"]["case_id"] == "P/P"
    assert stages["denoise"]["provenance"]["transformer"]["variant"] == _variant(workflow)
    assert stages["schedule"]["provenance"]["shifts"] == {"video": 12.0, "audio": 3.0}
    assert stages["visual_vae_encode"]["provenance"]["contract_id"] == "h3-visual-vae-f16t4d24-v1"
    assert stages["video_decode"]["components"][0][0] == "video"
    assert art["layout"]["pipeline_identity"]["variant"] == _variant(workflow)
    assert art["gate_policy"]["secondary"] == ["video_decode", "audio_decode"]
    steps = art["transformer_probe"]["steps"]
    assert len(steps) == art["trajectory"]["steps"]
    names = [name for name, _ in steps[0]]
    assert names[:4] == ["proj_in", "audio_proj_in", "context_embedder", "token_refiner"]
    assert names[-3:] == ["norm_out", "proj_out", "audio_proj_out"]
    assert [n for n in names if n.startswith("transformer_blocks.")] == [
        f"transformer_blocks.{i}" for i in range(3)
    ]
    if workflow == "ref2va":
        assert stages["audio_vae_encode"]["provenance"]["contract"] == "h3-audio-vae-v1"
        assert len(stages["audio_vae_encode"]["components"]) == 2  # soundtrack + audio reference
    else:
        assert stages["audio_vae_encode"]["provenance"] == {
            "skipped": "FL2VA has no audio conditions"
        }
    json.loads(fc.dumps(art))  # round-trips


def test_edited_or_reordered_artifact_is_refused() -> None:
    art = clean_run("fl2va").artifact
    edited = copy.deepcopy(art)
    edited["stages"][6]["components"][0][1] = "0" * 64
    with pytest.raises(ValueError, match="artifact_sha256"):
        fc.verify(edited)
    resealed = copy.deepcopy(edited)
    resealed["artifact_sha256"] = fc._sha(
        {k: v for k, v in edited.items() if k != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="hash chain"):
        fc.verify(resealed)
    swapped = copy.deepcopy(art)
    swapped["stages"][3], swapped["stages"][4] = swapped["stages"][4], swapped["stages"][3]
    swapped["artifact_sha256"] = fc._sha(
        {k: v for k, v in swapped.items() if k != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="out of order"):
        fc.verify(swapped)
    probe = copy.deepcopy(art)
    probe["transformer_probe"]["steps"][0][0][1] = "0" * 64
    probe["artifact_sha256"] = fc._sha({k: v for k, v in probe.items() if k != "artifact_sha256"})
    with pytest.raises(ValueError, match="transformer probe"):
        fc.verify(probe)


# --- injected drift is localized ----------------------------------------------------------


STAGE_INJECTIONS = [
    ("fl2va", "media_preprocess", "keyframe_pixels.1", 3),
    ("fl2va", "text_conditions", "prompt_embeds", 4),
    ("fl2va", "visual_vae_encode", "condition_latents.0", 0),
    ("fl2va", "initial_rows", "video_rows", 9),
    ("fl2va", "unpack", "video_latents", 5),
    ("ref2va", "media_preprocess", "reference.1.waveform", 0),
    ("ref2va", "visual_vae_encode", "condition_latents.1", 2),
    ("ref2va", "audio_vae_encode", "audio_rows.1", 7),
    ("ref2va", "initial_rows", "audio_rows", 61),
]


@pytest.mark.parametrize("workflow,stage,component,row", STAGE_INJECTIONS)
def test_stage_injection_is_the_first_drift(workflow, stage, component, row) -> None:
    clean = clean_run(workflow)
    run = injected(workflow, fc.DriftInjection(stage, component=component, row=row))
    report = fc.compare(
        clean.artifact, run.artifact, tensors_lhs=clean.tensors, tensors_rhs=run.tensors
    )
    drift = report["first_drift"]
    assert (drift["stage"], drift["component"]) == (stage, component)
    before = fc.STAGES.index(stage)
    assert all(s["equal"] for s in report["stages"][:before])
    assert report["gate"]["hard_gate"] == "fail"
    if (stage, component) in fc._COMPONENT_ROWS:
        assert drift["first_row"] == row
        assert drift["row"]["local_row"] == row
    else:  # no row space: the unraveled index of the first differing element
        row_dim = {"media_preprocess": 3 if "pixels" in component else 1}.get(stage)
        row_dim = row_dim or {"visual_vae_encode": 1, "audio_vae_encode": 0, "unpack": 2}[stage]
        assert drift["first_index"][row_dim] == row


def test_initial_row_injection_names_the_packed_identity() -> None:
    clean = clean_run("fl2va")
    layout = clean.artifact["layout"]
    ncv = layout["num_condition_video_rows"]  # 2 keyframes x 4 patches
    run = injected("fl2va", fc.DriftInjection("initial_rows", component="video_rows", row=ncv + 5))
    drift = fc.first_drift(
        clean.artifact, run.artifact, tensors_lhs=clean.tensors, tensors_rhs=run.tensors
    )
    row = drift["row"]
    assert row["role"] == "target" and row["modality"] == "video"
    assert (row["latent_frame"], row["patch_row"], row["patch_col"]) == (1, 0, 1)
    assert row["packed_row"] == int(clean.tensors["layout/video_indices"][ncv + 5])
    keyframe = injected("fl2va", fc.DriftInjection("initial_rows", component="video_rows", row=6))
    row = fc.first_drift(
        clean.artifact, keyframe.artifact, tensors_lhs=clean.tensors, tensors_rhs=keyframe.tensors
    )["row"]
    assert row["role"] == "condition" and row["condition_row"] == 6


@pytest.mark.parametrize(
    "workflow,step,module,row",
    [
        ("fl2va", 0, "transformer_blocks.0", 20),
        ("fl2va", 2, "transformer_blocks.2", 300),
        ("fl2va", 1, "token_refiner", 3),
        ("fl2va", 1, "proj_in", 40),
        ("ref2va", 1, "transformer_blocks.1", 30),
        ("ref2va", 2, "audio_proj_in", 70),
        ("ref2va", 0, "norm_out", 500),
    ],
)
def test_transformer_injection_names_step_module_and_row(workflow, step, module, row) -> None:
    clean = clean_run(workflow)
    run = injected(workflow, fc.DriftInjection("denoise", step=step, module=module, row=row))
    report = fc.compare(
        clean.artifact, run.artifact, tensors_lhs=clean.tensors, tensors_rhs=run.tensors
    )
    drift = report["first_drift"]
    assert drift["stage"] == "denoise" and drift["step"] == step
    if drift["trajectory"] is not None:  # a 1-ULP change may be rounded away before the heads
        assert drift["trajectory"]["step"] >= step
    else:
        assert report["gate"]["hard_gate"] == "pass"
        assert not report["gate"]["transformer_intermediates_equal"]
    found = drift["transformer"]
    assert found["module"] == module
    assert found["first_row"] == row and found["differing_rows"] == 1
    space, _ = fc.module_row_space(module)
    identity = found["row"]
    assert identity["space"] == space and identity["local_row"] == row
    if space == "packed":
        assert identity["packed_row"] == row
        tag = int(clean.tensors["layout/token_tags"][row])
        assert identity["modality"] == {0: "video", 1: "text", 2: "audio"}[tag]
    else:
        assert identity["packed_row"] == int(clean.tensors[f"layout/{space}_indices"][row])
    # every earlier step is byte-equal, every stage before denoise too
    assert all(s["equal"] for s in report["stages"][: fc.STAGES.index("denoise")])
    for earlier in range(step):
        assert clean.artifact["transformer_probe"]["steps"][earlier] == (
            run.artifact["transformer_probe"]["steps"][earlier]
        )


def test_velocity_injection_after_the_heads_has_no_module() -> None:
    clean = clean_run("ref2va")
    run = injected(
        "ref2va", fc.DriftInjection("denoise", step=1, component="velocity.audio", row=5)
    )
    drift = fc.first_drift(
        clean.artifact, run.artifact, tensors_lhs=clean.tensors, tensors_rhs=run.tensors
    )
    assert (drift["step"], drift["kind"], drift["modality"]) == (1, "velocity", "audio")
    assert drift["transformer"]["module"] is None
    row = drift["trajectory"]["row"]
    assert row["local_row"] == 5 and row["role"] == "reference"


def test_decode_drift_is_secondary_and_does_not_fail_the_gate() -> None:
    clean = clean_run("fl2va", decode=True)
    run = injected("fl2va", fc.DriftInjection("video_decode", component="video", row=0), True)
    gate = fc.gate(clean.artifact, run.artifact)
    assert gate["hard_gate"] == "pass" and gate["latent_trajectory_equal"]
    assert gate["decoded_media_equal"] == {"video_decode": False, "audio_decode": True}
    assert gate["first_drift"]["stage"] == "video_decode"
    report = fc.compare(
        clean.artifact, run.artifact, tensors_lhs=clean.tensors, tensors_rhs=run.tensors
    )
    assert 100.0 < report["secondary_metrics"]["video_psnr_db"] < float("inf")


def test_artifacts_without_tensors_still_localize_stage_step_and_module() -> None:
    clean = clean_run("fl2va")
    run = injected("fl2va", fc.DriftInjection("denoise", step=1, module="transformer_blocks.1"))
    drift = fc.first_drift(json.loads(fc.dumps(clean.artifact)), json.loads(fc.dumps(run.artifact)))
    assert (drift["stage"], drift["step"]) == ("denoise", 1)
    assert drift["transformer"]["module"] == "transformer_blocks.1"
    assert "row" not in drift["transformer"]


def test_capture_keeps_only_the_selected_probe_tensors() -> None:
    run = fc.run_chain(
        toy.fl2va_request(),
        toy.components("FL2VA"),
        label="capture",
        text_source=toy.TEXT_SOURCE,
        capture=lambda step, path: step == 1 and path == "transformer_blocks.2",
        decode=False,
    )
    assert sorted(run.tensors) == ["probe/1/transformer_blocks.2"]


# --- identity, text conditions, schedule --------------------------------------------------


def test_variant_mismatch_is_refused_before_any_forward() -> None:
    comps = toy.components("FL2VA", denoiser_variant="Ref2VA")
    with pytest.raises(pl.VariantMismatch):
        fc.run_chain(toy.fl2va_request(), comps, label="mixed", text_source=toy.TEXT_SOURCE)
    assert comps.denoiser.model.calls == 0


@pytest.mark.parametrize("mode", ["strict", "audit"])
def test_bound_modes_fail_closed_on_a_model_that_is_not_the_full_h3(mode: str) -> None:
    """The strict graph binds the 50 released blocks only: a toy or layer-truncated model is
    refused before any forward, never silently run on provider arithmetic."""

    comps = toy.components("FL2VA")
    model = comps.denoiser.model
    comps.denoiser = fc.TransformerDenoiser(model, variant="FL2VA", mode=mode)
    refused = "full H3 requires|50 distinct|strict H3 needs CUDA parameters"
    with pytest.raises((ValueError, RuntimeError), match=refused):
        fc.run_chain(
            toy.fl2va_request(), comps, label=mode, decode=False, text_source=toy.TEXT_SOURCE
        )
    assert model.calls == 0


def test_declared_text_source_must_match_the_conditioner() -> None:
    with pytest.raises(fc.TextSourceMismatch, match="declared text source"):
        fc.run_chain(
            toy.fl2va_request(),
            toy.components("FL2VA"),
            label="lie",
            text_source=fc.TEXT_SOURCE_QWEN3VL,  # the synthetic conditioner is not the contract
            decode=False,
        )


def test_pinned_text_features_replay_and_refuse_other_inputs(tmp_path: Path) -> None:
    clean = clean_run("fl2va")
    text = {s["stage"]: s for s in clean.artifact["stages"]}["text_conditions"]
    path = tmp_path / "text.safetensors"
    file_sha = fc.save_pinned_text_features(
        path,
        clean.tensors["text_conditions/prompt_embeds"],
        clean.tensors["text_conditions/text_token_tags"],
        input_sha256=text["provenance"]["conditioner_input_sha256"],
        producer={"name": "SyntheticTextConditioner", "note": "tests"},
    )
    pinned = fc.PinnedTextFeatures(path, expected_sha256=file_sha)
    run = fc.run_chain(
        toy.fl2va_request(),
        toy.components("FL2VA", text=pinned),
        label="pinned",
        text_source=fc.TEXT_SOURCE_PINNED,
        keep_tensors=True,
        decode=False,
    )
    with pytest.raises(fc.TextSourceMismatch):
        fc.compare(clean.artifact, run.artifact)  # a replay with another source is refused
    report = fc.compare(clean.artifact, run.artifact, allow_text_source_change=True)
    assert report["first_drift"] is None and report["gate"]["hard_gate"] == "pass"
    assert report["text_sources"] == ["synthetic (tests)", fc.TEXT_SOURCE_PINNED]
    stage = {s["stage"]: s for s in run.artifact["stages"]}["text_conditions"]
    assert stage["provenance"]["source"] == fc.TEXT_SOURCE_PINNED
    assert stage["provenance"]["file_sha256"] == file_sha
    other = fc.FL2VARequest(**{**toy.fl2va_request().__dict__, "prompt": "another prompt"})
    with pytest.raises(fc.PinnedFeaturesMismatch, match="another prompt|other media"):
        fc.run_chain(
            other,
            toy.components("FL2VA", text=pinned),
            label="x",
            text_source=fc.TEXT_SOURCE_PINNED,
        )
    with pytest.raises(fc.PinnedFeaturesMismatch, match="expected"):
        fc.PinnedTextFeatures(path, expected_sha256="0" * 64).encode(
            fc.TextRequest("fl2va", "p", (), (), "0" * 64)
        )


def test_dual_schedule_is_the_released_grid() -> None:
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    sched = fc.dual_schedule(50)
    video = NativeH3SigmaScheduleOp()(50, shift=12.0)
    audio = NativeH3SigmaScheduleOp()(50, shift=3.0)
    assert torch.equal(sched.video_sigmas, video.sigmas)
    assert torch.equal(sched.audio_sigmas, audio.sigmas)
    assert sched.steps == 49 and float(sched.video_sigmas[-1]) == 0.0


def test_bump_one_ulp_moves_exactly_one_element() -> None:
    for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
        x = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(0)).to(dtype)
        y = fc.bump_one_ulp(x, row=2, row_dim=2)
        assert int((x != y).sum()) == 1 and bool(x[0, 0, 2] != y[0, 0, 2])


def test_injection_arguments_are_checked() -> None:
    with pytest.raises(ValueError):
        fc.DriftInjection("denoise", step=1)
    with pytest.raises(ValueError):
        fc.DriftInjection("initial_rows")
    with pytest.raises(ValueError):
        fc.DriftInjection("nope", component="x")


# --- negative controls that would prove nothing, and reports that would mislead --------------


@pytest.mark.parametrize(
    "inject,kwargs",
    [
        (dict(stage="denoise", step=500, component="velocity.video"), {}),
        (dict(stage="denoise", step=0, component="velocity.nothing"), {}),
        (dict(stage="denoise", step=0, module="transformer_blocks.99"), {}),
        (dict(stage="denoise", step=0, module="transformer_blocks.0"), {"probe": False}),
        (dict(stage="initial_rows", component="nothing"), {}),
        (dict(stage="audio_decode", component="waveform"), {}),  # decode is off
    ],
)
def test_an_injection_that_did_not_run_is_refused(inject, kwargs) -> None:
    with pytest.raises(fc.InjectionNotRun, match="matched nothing"):
        fc.run_chain(
            toy.fl2va_request(),
            toy.components("FL2VA"),
            label="x",
            text_source=toy.TEXT_SOURCE,
            inject=fc.DriftInjection(**inject),
            decode=False,
            **kwargs,
        )


def _resealed(artifact: dict) -> dict:
    previous = fc.GENESIS
    for record in artifact["stages"]:
        inner = {k: v for k, v in record.items() if k != "chain_sha256"}
        record["chain_sha256"] = fc._sha({"previous": previous, "stage": inner})
        previous = record["chain_sha256"]
    body = {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    return {**body, "artifact_sha256": fc._sha(body)}


def test_duplicate_component_names_are_refused_when_verifying() -> None:
    artifact = copy.deepcopy(clean_run("fl2va").artifact)
    fc.verify(_resealed(artifact))  # the helper reseals correctly
    stage = next(s for s in artifact["stages"] if s["stage"] == "unpack")
    stage["components"].insert(0, ["video_latents", "0" * 64])
    with pytest.raises(ValueError, match="duplicate component names"):
        fc.verify(_resealed(artifact))


def test_the_audio_decoder_reads_the_recorded_latents(monkeypatch) -> None:
    rows = torch.randn(2 * 5, 7)
    assert torch.equal(fc._audio_rows(fc._unpack_audio(rows, 5)), rows)
    comps = toy.components("FL2VA")
    seen = []
    decode = comps.audio_vae.decode_rows

    def recorded(target_rows, num_latents):
        seen.append(target_rows.clone())
        return decode(target_rows, num_latents)

    monkeypatch.setattr(comps.audio_vae, "decode_rows", recorded)
    run = fc.run_chain(
        toy.fl2va_request(),
        comps,
        label="x",
        text_source=toy.TEXT_SOURCE,
        inject=fc.DriftInjection("unpack", component="audio_latents", row=2),
    )
    assert torch.equal(seen[0], fc._audio_rows(run.audio_latents).float())
    clean = clean_run("fl2va").audio_latents
    assert int((clean != run.audio_latents).sum()) == 1


def test_a_sidecar_that_is_not_the_recorded_run_gives_no_row_evidence() -> None:
    clean = clean_run("fl2va")
    for inject, where in (
        (fc.DriftInjection("initial_rows", component="video_rows", row=13), None),
        (fc.DriftInjection("denoise", step=1, module="transformer_blocks.1", row=3), "transformer"),
        (fc.DriftInjection("denoise", step=1, component="velocity.video", row=3), "trajectory"),
    ):
        run = injected("fl2va", inject)
        drift = fc.first_drift(
            clean.artifact, run.artifact, tensors_lhs=clean.tensors, tensors_rhs=clean.tensors
        )
        found = drift if where is None else drift[where]
        assert found["row_evidence"] == "unavailable: the rhs tensor is not the recorded one"
        assert "first_row" not in found and "row" not in found


@pytest.mark.parametrize("name", ["position_ids", "token_tags", "video_indices"])
def test_foreign_layout_tensors_give_no_packed_identity(name) -> None:
    from tools.validation.models import h3_full_chain as tool

    clean = clean_run("fl2va")
    run = injected("fl2va", fc.DriftInjection("initial_rows", component="video_rows", row=13))
    sides = dict(tensors_lhs=clean.tensors, tensors_rhs=run.tensors)
    assert "row" in fc.first_drift(clean.artifact, run.artifact, **sides)
    foreign = {**clean.tensors, f"layout/{name}": clean.tensors[f"layout/{name}"] + 1}
    missing = {k: v for k, v in clean.tensors.items() if k != f"layout/{name}"}
    for kept in (foreign, missing):
        if "layout/token_tags" not in kept:
            continue  # no layout kept at all: no identity is attempted
        report = fc.compare(clean.artifact, run.artifact, **{**sides, "tensors_lhs": kept})
        drift = report["first_drift"]
        assert drift["first_row"] == 13 and drift["differing_rows"] == 1  # still verified
        assert "row" not in drift and name in drift["row_identity"]
        (cells,) = tool._first_drift_table(report)["rows"]
        assert cells[3] == "n/a" and cells[4].startswith("row 13 of the tensor (unavailable")


def test_runs_without_a_probe_are_not_called_equal() -> None:
    def run(inject=None):
        return fc.run_chain(
            toy.fl2va_request(),
            toy.components("FL2VA"),
            label="x",
            text_source=toy.TEXT_SOURCE,
            probe=False,
            inject=inject,
            decode=False,
        ).artifact

    report = fc.compare(
        run(), run(fc.DriftInjection("denoise", step=1, component="velocity.video", row=3))
    )
    assert report["gate"]["hard_gate"] == "fail"
    assert report["gate"]["transformer_intermediates_equal"] is None
    assert "not compared" in report["first_drift"]["transformer"]["note"]


def test_probes_that_observed_nothing_are_not_called_equal(monkeypatch) -> None:
    def run(inject=None):
        return fc.run_chain(
            toy.fl2va_request(),
            toy.components("FL2VA"),
            label="x",
            text_source=toy.TEXT_SOURCE,
            inject=inject,
            decode=False,
        ).artifact

    monkeypatch.setattr(fc.TransformerProbe, "modules", lambda self: [])  # hooks on nothing
    clean = run()
    assert clean["transformer_probe"] == {
        "steps": [[] for _ in range(clean["trajectory"]["steps"])]
    }
    report = fc.compare(clean, run())
    assert report["gate"]["hard_gate"] == "pass"
    assert report["gate"]["transformer_intermediates_equal"] is None
    drifted = run(fc.DriftInjection("denoise", step=1, component="velocity.video", row=3))
    report = fc.compare(clean, drifted)
    assert report["gate"]["hard_gate"] == "fail"
    assert report["gate"]["transformer_intermediates_equal"] is None
    assert "not compared" in report["first_drift"]["transformer"]["note"]


def test_a_probe_call_only_one_run_made_is_a_drift() -> None:
    def artifact(*calls, steps=1):
        step = [[name, "0" * 64] for name in calls]
        return {"transformer_probe": {"steps": [step] * steps}, "layout": {}}

    short, long = artifact("proj_in", "proj_out"), artifact("proj_in", "proj_out", "proj_out#1")
    drift = fc._probe_drift(short, long, None, None)
    assert (drift["step"], drift["module"], drift["execution_index"]) == (0, "proj_out#1", 2)
    assert drift["note"] == "only the rhs run made this call"
    assert fc._probe_drift(long, short, None, None)["note"] == "only the lhs run made this call"
    drift = fc._probe_drift(short, artifact("proj_in", "proj_out", steps=2), None, None)
    assert drift["step"] == 1 and "only one run recorded" in drift["note"]
    assert fc._probe_drift(short, short, None, None) is None
    assert fc._intermediates_equal(short, short) is True
    assert fc._intermediates_equal(short, long) is False
    empty = {"transformer_probe": {"steps": [[]]}, "layout": {}}
    assert fc._intermediates_equal(empty, empty) is None  # nothing was compared
    assert fc._intermediates_equal(empty, short) is False  # a call only one run made
    assert fc._intermediates_equal({"transformer_probe": None}, short) is None


def test_the_trajectory_fields_belong_to_the_reported_step(monkeypatch) -> None:
    later = {"step": 2, "kind": "velocity", "modality": "video"}
    monkeypatch.setattr(fc.tj, "first_drift", lambda *a, **kw: dict(later))
    probe = {"step": 1, "module": "transformer_blocks.0", "execution_index": 0}
    monkeypatch.setattr(fc, "_probe_drift", lambda *a: dict(probe))
    artifact = {"transformer_probe": {"steps": []}, "trajectory": {}, "layout": {}}
    out = fc._denoise_drift(artifact, artifact, None, None)
    assert out["step"] == 1 and "kind" not in out and "modality" not in out
    assert out["trajectory"]["step"] == 2
    monkeypatch.setattr(fc, "_probe_drift", lambda *a: {**probe, "step": 2})
    out = fc._denoise_drift(artifact, artifact, None, None)
    assert (out["step"], out["kind"], out["modality"]) == (2, "velocity", "video")


def test_undefined_media_metrics_are_null_with_a_reason() -> None:
    wave = torch.randn(2, 64, generator=torch.Generator().manual_seed(0))
    key, name = "audio_decode/waveform", "audio_snr_db"

    def closeness(a, b):
        return fc._media_closeness({key: a}, {key: b})

    assert closeness(wave, wave) == {name: None, f"{name}_note": "identical"}
    assert closeness(torch.zeros(2, 64), wave)[f"{name}_note"] == "silent reference"
    bad = wave.clone()
    bad[0, 0] = float("nan")
    assert closeness(wave, bad) == {name: None, f"{name}_note": "non-finite samples"}
    assert closeness(wave, wave + 1)[name] < 10
    metrics = fc.audio_metrics(bad)
    assert metrics["peak"] is None and metrics["rms_lr"][0] is None
    json.dumps([metrics, fc.finite({"a": [float("inf"), (1.0, float("nan"))]})], allow_nan=False)


def test_pinned_features_hash_and_load_one_snapshot(tmp_path: Path, monkeypatch) -> None:
    clean = clean_run("fl2va")
    text = {s["stage"]: s for s in clean.artifact["stages"]}["text_conditions"]
    path = tmp_path / "text.safetensors"
    embeds = clean.tensors["text_conditions/prompt_embeds"]
    tags = clean.tensors["text_conditions/text_token_tags"]
    meta = dict(
        input_sha256=text["provenance"]["conditioner_input_sha256"], producer={"name": "tests"}
    )
    sha = fc.save_pinned_text_features(path, embeds, tags, **meta)
    read = Path.read_bytes

    def read_then_replace(self):
        data = read(self)
        if self == path:
            monkeypatch.setattr(Path, "read_bytes", read)
            fc.save_pinned_text_features(path, embeds + 1, tags, **meta)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    request = fc.TextRequest("fl2va", "p", (), (), meta["input_sha256"])
    got = fc.PinnedTextFeatures(path, expected_sha256=sha).encode(request)
    assert torch.equal(got.prompt_embeds, embeds)
    with pytest.raises(fc.PinnedFeaturesMismatch, match="expected"):  # now the replaced file
        fc.PinnedTextFeatures(path, expected_sha256=sha).encode(request)


def test_the_first_drift_table_outside_denoise() -> None:
    from tools.validation.models import h3_full_chain as tool

    clean = clean_run("fl2va")
    ncv = clean.artifact["layout"]["num_condition_video_rows"]
    run = injected("fl2va", fc.DriftInjection("initial_rows", component="video_rows", row=ncv + 5))
    report = fc.compare(
        clean.artifact, run.artifact, tensors_lhs=clean.tensors, tensors_rhs=run.tensors
    )
    (cells,) = tool._first_drift_table(report)["rows"]
    assert cells[0] == "initial_rows" and cells[2] == "n/a"
    assert cells[3] == str(report["first_drift"]["row"]["packed_row"])
    assert cells[4].startswith("target video, latent frame 1")
    assert "equal" not in cells[6]  # the trajectories do differ, or are not comparable
    report = fc.compare(clean.artifact, run.artifact)  # no tensors kept
    (cells,) = tool._first_drift_table(report)["rows"]
    assert cells[3] == "n/a" and cells[4] == "n/a (no tensors kept)"
    same = {"stage": "media_preprocess", "component": "x"}
    for whole, text in (
        ({"bitwise_equal": True}, "none (velocities and latents equal)"),
        ({"comparable": False, "reason": "seed differs"}, "not comparable: seed differs"),
        ({}, "n/a"),
    ):
        table = tool._first_drift_table({"first_drift": same, "trajectory": whole})
        assert table["rows"][0][6] == text


def _plot_args(tmp_path: Path, lhs: dict, rhs: dict):
    import argparse

    for name, artifact in (("a", lhs), ("b", rhs)):
        (tmp_path / f"{name}.json").write_text(fc.dumps(artifact))
    return argparse.Namespace(
        lhs=str(tmp_path / "a.json"),
        rhs=str(tmp_path / "b.json"),
        png=str(tmp_path / "p.png"),
        compare=None,
        title="t",
        pair_label="A vs B",
    )


def test_the_drift_chart_counts_a_trajectory_only_difference(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from tools.validation.models import h3_full_chain as tool

    clean = clean_run("fl2va")
    run = injected("fl2va", fc.DriftInjection("denoise", step=0, component="velocity.video"))
    out = tool.plot(_plot_args(tmp_path, clean.artifact, run.artifact))
    assert out["first_drift_cell"] == {"step": 0, "module": "velocity video"}
    out = tool.plot(_plot_args(tmp_path, clean.artifact, clean.artifact))
    assert out["first_drift_cell"] is None


def test_cuda_analysis_is_not_labelled_cpu(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import argparse

    from tools.validation.models import h3_full_chain as tool

    check = dict(bitwise_equal=False, relative_l2=0.02, shape=[1, 4, 8])
    for wf in ("fl2va", "ref2va"):
        for dev in ("cpu", "cuda"):
            (tmp_path / f"{wf}-{dev}.json").write_text(json.dumps(check))
    per_token = dict(per_token_relative_l2_text=0.25, per_token_relative_l2_vision=0.5)
    (tmp_path / "analysis.json").write_text(json.dumps({"fl2va": per_token, "ref2va": per_token}))
    out = tool.plot_text(
        argparse.Namespace(
            dir=str(tmp_path),
            analysis=str(tmp_path / "analysis.json"),
            png=str(tmp_path / "p.png"),
            title="t",
        )
    )
    rows = out["markdown"].splitlines()
    assert all("0.25" in r and "0.5" in r for r in rows if "| CUDA |" in r)
    assert all("0.25" not in r and "n/a | n/a" in r for r in rows if "| CPU |" in r)
    assert sum("| CPU |" in r for r in rows) == 2


# --- Diffusers-backed ---------------------------------------------------------------------


def _diffusers_h3():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "MiniMaxH3Scheduler"):
        pytest.skip("this diffusers has no MiniMax-H3 (needs f53d552 or later)")
    return diffusers


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_euler_step_is_bitwise_the_released_scheduler(device: str, shift: float) -> None:
    diffusers = _diffusers_h3()
    scheduler = diffusers.MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(30, device=device)
    sched = fc.dual_schedule(30)
    sigmas = (sched.video_sigmas if shift == 12.0 else sched.audio_sigmas).to(device)
    timesteps = (sched.video_timesteps if shift == 12.0 else sched.audio_timesteps).to(device)
    assert torch.equal(scheduler.sigmas.cpu(), sigmas.cpu())
    g = torch.Generator().manual_seed(int(shift))
    ours = theirs = torch.randn(64, 96, generator=g).to(device)
    for i, t in enumerate(scheduler.timesteps):
        velocity = torch.randn(64, 96, generator=g).to(device)
        theirs = scheduler.step(velocity, t, theirs, return_dict=False)[0]
        ours = fc.euler_step(velocity, timesteps[i], ours, sigmas[i], sigmas[i + 1])
        assert torch.equal(ours, theirs), f"step {i}"


def _tiny_diffusers_transformer(diffusers, seed: int = 0):
    torch.manual_seed(seed)
    model = diffusers.MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=32,
        hidden_size=48,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        text_dim=toy.TEXT_DIM,
        freq_dim=32,
        time_embed_hidden_dim=48,
        time_embed_dim=24,
        rope_freq_dim=4,
    )
    with torch.no_grad():
        gen = torch.Generator().manual_seed(seed + 1)
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
    return model.eval()


def test_tiny_diffusers_transformer_chain_localizes_a_block_drift() -> None:
    diffusers = _diffusers_h3()
    model = _tiny_diffusers_transformer(diffusers)

    def comps() -> fc.ChainComponents:
        c = toy.components("Ref2VA")
        c.denoiser = fc.TransformerDenoiser(model, variant="Ref2VA")
        return c

    req = toy.ref2va_request(steps=3)
    a = fc.run_chain(req, comps(), label="a", keep_tensors=True, text_source=toy.TEXT_SOURCE)
    b = fc.run_chain(req, comps(), label="b", keep_tensors=True, text_source=toy.TEXT_SOURCE)
    assert fc.compare(a.artifact, b.artifact)["first_drift"] is None
    names = [name for name, _ in a.artifact["transformer_probe"]["steps"][0]]
    assert names[:5] == [
        "proj_in",
        "audio_proj_in",
        "context_embedder",
        "token_refiner",
        "time_embedder",
    ]
    assert names[5:7] == ["transformer_blocks.0", "transformer_blocks.1"]
    inject = fc.DriftInjection("denoise", step=1, module="transformer_blocks.1", row=40)
    c = fc.run_chain(
        req, comps(), label="c", keep_tensors=True, inject=inject, text_source=toy.TEXT_SOURCE
    )
    drift = fc.first_drift(a.artifact, c.artifact, tensors_lhs=a.tensors, tensors_rhs=c.tensors)
    assert drift["step"] == 1 and drift["transformer"]["module"] == "transformer_blocks.1"
    assert drift["transformer"]["row"]["packed_row"] == 40


# --- qwen3vl_layer50_contract as the text source ------------------------------------------


QWEN3VL_DIR_ENV = "RL_KERNEL_H3_QWEN3VL_DIR"


@pytest.fixture(scope="module")
def tiny_qwen3vl_conditioner():
    """The contract on a tiny random Qwen3-VL (53 layers) with the released tokenizer/processor."""

    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers without Qwen3-VL")
    root = os.environ.get(QWEN3VL_DIR_ENV)
    if not root:
        pytest.skip(f"{QWEN3VL_DIR_ENV} not set (tokenizer/ and processor/ of the release)")
    from rl_engine.models.minimax_h3 import qwen3vl_text as qt
    from tests.models.minimax_h3.test_h3_qwen3vl_layer50 import _tiny_config

    torch.manual_seed(0)
    model = transformers.Qwen3VLForConditionalGeneration(_tiny_config()).eval()
    return fc.Qwen3VLLayer50Conditioner(
        qt.Qwen3VLLayer50Encoder(model, check_release=False),
        transformers.AutoTokenizer.from_pretrained(str(Path(root) / "tokenizer")),
        transformers.AutoProcessor.from_pretrained(str(Path(root) / "processor")),
    )


def _diffusers_text_step(conditioner, workflow: str, prepared: fc.PreparedMedia):
    """Diffusers' own text-encoder block on the same model, tokenizer and processor."""

    from types import SimpleNamespace

    encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.encoders")
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    components = SimpleNamespace(
        text_encoder=conditioner.encoder.text_encoder,
        processor=conditioner.processor,
        tokenizer=conditioner.tokenizer,
        text_tag=pl.TEXT_TAG,
        video_tag=pl.VIDEO_TAG,
        text_encoder_layer=50,
        fps=pl.FPS,
        _execution_device=torch.device("cpu"),
    )
    state = PipelineState()
    state.set("prompt", prepared.text_request.prompt)
    if workflow == "fl2va":
        block = encoders.MiniMaxH3FL2VATextEncoderStep()
        state.set("keyframes", list(prepared.text_request.media))
    else:
        block = encoders.MiniMaxH3Ref2VATextEncoderStep()
        state.set("normalized_references", list(prepared.normalized))
    with torch.no_grad():
        block(components, state)
    return state.get("prompt_embeds"), state.get("text_token_tags")


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_qwen3vl_source_matches_the_diffusers_text_step(tiny_qwen3vl_conditioner, workflow):
    prepared = fc.prepare(_request(workflow))
    ours = tiny_qwen3vl_conditioner.encode(prepared.text_request)
    assert ours.provenance["source"] == fc.TEXT_SOURCE_QWEN3VL
    assert ours.provenance["contract_id"] == "h3-qwen3vl-layer50-v1"
    assert ours.provenance["feature_sha256"] == fc.tensor_digest(ours.prompt_embeds)
    ids, tags, _ = tiny_qwen3vl_conditioner.presentation(prepared.text_request)
    assert ours.text_token_tags.tolist() == tags and len(ids) == len(tags)
    assert (ours.text_token_tags == pl.VIDEO_TAG).sum() > 0  # vision blocks are tagged video
    embeds, ref_tags = _diffusers_text_step(tiny_qwen3vl_conditioner, workflow, prepared)
    assert torch.equal(ours.text_token_tags, ref_tags.long())
    assert torch.equal(ours.prompt_embeds, embeds), "contract features differ from Diffusers"


def test_chain_with_the_qwen3vl_source_records_the_contract(tiny_qwen3vl_conditioner) -> None:
    comps = toy.components("FL2VA", text=tiny_qwen3vl_conditioner, text_dim=64)
    run = fc.run_chain(
        toy.fl2va_request(),
        comps,
        label="qwen3vl",
        text_source=fc.TEXT_SOURCE_QWEN3VL,
        keep_tensors=True,
        decode=False,
    )
    stage = {s["stage"]: s for s in run.artifact["stages"]}["text_conditions"]
    assert stage["provenance"]["source"] == fc.TEXT_SOURCE_QWEN3VL
    assert stage["provenance"]["contract_id"] == "h3-qwen3vl-layer50-v1"
    assert dict(stage["components"])["prompt_embeds"] == stage["provenance"]["feature_sha256"]
    assert fc.text_source(run.artifact) == fc.TEXT_SOURCE_QWEN3VL
    with pytest.raises(fc.TextSourceMismatch, match="declared text source"):
        fc.run_chain(
            toy.fl2va_request(),
            comps,
            label="pinned-claim",
            text_source=fc.TEXT_SOURCE_PINNED,
            decode=False,
        )


# --- sealed fixtures ----------------------------------------------------------------------
#
# The toy fixtures' bytes are host specific, even with the same torch, ATen CPU capability,
# machine and thread count. On another host, two stages gave other bytes:
# - media_preprocess: Pillow's LANCZOS resize of FL2VA's keyframe (80x48 -> 64x64); that
#   host's Pillow version or build is not recorded;
# - audio_vae_encode: the oneDNN convolutions of the audio VAE. oneDNN picks its kernels from
#   the CPU's ISA, which torch's ``cpu_capability`` does not show (ONEDNN_MAX_CPU_ISA=AVX2
#   reproduces this drift on an AVX512 host).
# So the committed fixtures are checked for their seal and structure on every host, byte
# determinism is checked by sealing and replaying in two processes on this host, and the
# bitwise replay of the committed bytes is opt-in, for the host that sealed them.


def _profile(runtime: dict) -> dict:
    return {key: runtime.get(key) for key in PROFILE_KEYS}


def _fixture_run(workflow: str) -> fc.ChainRun:
    threads = torch.get_num_threads()
    torch.set_num_threads(FIXTURE_THREADS)
    try:
        return fc.run_chain(
            _request(workflow),
            toy.components(_variant(workflow)),
            label=f"{workflow}-fixture",
            text_source=toy.TEXT_SOURCE,
        )
    finally:
        torch.set_num_threads(threads)


def _structure(artifact: dict) -> dict:
    """What a fixture fixes independently of the host: stages, components, provenance keys."""

    return {
        "workflow": artifact["workflow"],
        "variant": artifact["variant"],
        "contract": artifact["contract"],
        "text_source": fc.text_source(artifact),
        "stages": [
            [s["stage"], [c[0] for c in s["components"]], sorted(s["provenance"])]
            for s in artifact["stages"]
        ],
        "trajectory_steps": [
            artifact["trajectory"]["steps"],
            len(artifact["trajectory"]["step_records"]),
        ],
        "modalities": artifact["trajectory"]["modalities"],
        "probe_modules": [
            [name for name, _ in step] for step in artifact["transformer_probe"]["steps"]
        ],
    }


_FRESH_PROCESS_RUN = """
import sys
from pathlib import Path
from rl_engine.models.minimax_h3 import full_chain as fc
from tests.models.minimax_h3 import test_h3_full_chain as t
Path(sys.argv[2]).write_text(fc.dumps(t._fixture_run(sys.argv[1]).artifact))
"""


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_sealed_fixture_seal_and_structure(workflow: str) -> None:
    path = FIXTURES / f"h3_full_chain_{workflow}_toy.json"
    run = _fixture_run(workflow)
    if os.environ.get(REGEN_ENV):
        path.parent.mkdir(exist_ok=True)
        path.write_text(fc.dumps(run.artifact))
    sealed = fc.loads(path.read_text())
    assert sealed["variant"] == _variant(workflow)
    assert _structure(sealed) == _structure(run.artifact)


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_fixture_bytes_replay_in_a_fresh_process(workflow: str, tmp_path: Path) -> None:
    """Seal in this process, replay in a new one on the same host: every stage byte-equal."""

    lhs = tmp_path / "this-process.json"
    rhs = tmp_path / "fresh-process.json"
    lhs.write_text(fc.dumps(_fixture_run(workflow).artifact))
    root = Path(__file__).resolve().parents[3]
    path = os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, PYTHONPATH=path)
    subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_RUN, workflow, str(rhs)],
        check=True,
        cwd=root,
        env=env,
        timeout=600,
    )
    report = fc.compare(fc.loads(lhs.read_text()), fc.loads(rhs.read_text()))
    assert report["first_drift"] is None, report["first_drift"]
    assert report["gate"]["hard_gate"] == "pass"
    assert all(s["equal"] for s in report["stages"])


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_sealed_fixture_replays_bitwise_on_its_host(workflow: str) -> None:
    if not os.environ.get(REPLAY_ENV):
        pytest.skip(f"set {REPLAY_ENV}=1 on the host that sealed the fixtures")
    sealed = fc.loads((FIXTURES / f"h3_full_chain_{workflow}_toy.json").read_text())
    run = _fixture_run(workflow)
    if _profile(sealed["runtime"]) != _profile(run.artifact["runtime"]):
        pytest.skip(
            f"fixture sealed on {_profile(sealed['runtime'])}, this is "
            f"{_profile(run.artifact['runtime'])}"
        )
    report = fc.compare(sealed, run.artifact)
    assert report["first_drift"] is None, report["first_drift"]
    assert report["gate"]["hard_gate"] == "pass"
