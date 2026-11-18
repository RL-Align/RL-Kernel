# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""RFC #420 Phase B ``phase_b_full_chain``: sealed chain, first-drift localization, latent gate.

Everything runs on CPU with the stand-ins of ``_toy_chain`` (seconds). Diffusers-backed
checks (the scheduler step against ``MiniMaxH3Scheduler``, a tiny real
``MiniMaxH3Transformer3DModel``) skip when Diffusers with MiniMax-H3 is not importable.

The two sealed fixtures under ``fixtures/`` are replayed when the runtime profile (torch
version, CPU capability, machine) matches the one they were sealed on; set
``RL_KERNEL_H3_REGEN_CHAIN_FIXTURES=1`` to reseal them.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3 import full_chain as fc
from rl_engine.models.minimax_h3 import packed_layout as pl
from tests.models.minimax_h3 import _toy_chain as toy

FIXTURES = Path(__file__).with_name("fixtures")
REGEN_ENV = "RL_KERNEL_H3_REGEN_CHAIN_FIXTURES"
PROFILE_KEYS = ("torch", "cpu_capability", "platform_machine")


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
        keep_tensors=True,
        inject=inject,
        decode=decode,
    )


# --- sealing and repeatability ------------------------------------------------------------


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_repeat_run_is_bitwise_equal_at_every_stage(workflow: str) -> None:
    first = clean_run(workflow, decode=True)
    again = fc.run_chain(
        _request(workflow), toy.components(_variant(workflow)), label="again", keep_tensors=True
    )
    report = fc.compare(
        first.artifact, again.artifact, tensors_lhs=first.tensors, tensors_rhs=again.tensors
    )
    assert report["first_drift"] is None
    assert all(stage["equal"] for stage in report["stages"])
    assert report["gate"]["hard_gate"] == "pass"
    assert report["gate"]["decoded_media_equal"] == {"video_decode": True, "audio_decode": True}
    assert report["trajectory"]["bitwise_equal"]
    assert report["secondary_metrics"]["video_psnr_db"] == float("inf")


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
        capture=lambda step, path: step == 1 and path == "transformer_blocks.2",
        decode=False,
    )
    assert sorted(run.tensors) == ["probe/1/transformer_blocks.2"]


# --- identity, text conditions, schedule --------------------------------------------------


def test_variant_mismatch_is_refused_before_any_forward() -> None:
    comps = toy.components("FL2VA", denoiser_variant="Ref2VA")
    with pytest.raises(pl.VariantMismatch):
        fc.run_chain(toy.fl2va_request(), comps, label="mixed")
    assert comps.denoiser.model.calls == 0


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
        keep_tensors=True,
        decode=False,
    )
    report = fc.compare(clean.artifact, run.artifact)
    assert report["first_drift"] is None and report["gate"]["hard_gate"] == "pass"
    stage = {s["stage"]: s for s in run.artifact["stages"]}["text_conditions"]
    assert stage["provenance"]["source"] == fc.TEXT_SOURCE_PINNED
    assert stage["provenance"]["file_sha256"] == file_sha
    other = fc.FL2VARequest(**{**toy.fl2va_request().__dict__, "prompt": "another prompt"})
    with pytest.raises(fc.PinnedFeaturesMismatch, match="another prompt|other media"):
        fc.run_chain(other, toy.components("FL2VA", text=pinned), label="x")
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
    a = fc.run_chain(req, comps(), label="a", keep_tensors=True)
    b = fc.run_chain(req, comps(), label="b", keep_tensors=True)
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
    c = fc.run_chain(req, comps(), label="c", keep_tensors=True, inject=inject)
    drift = fc.first_drift(a.artifact, c.artifact, tensors_lhs=a.tensors, tensors_rhs=c.tensors)
    assert drift["step"] == 1 and drift["transformer"]["module"] == "transformer_blocks.1"
    assert drift["transformer"]["row"]["packed_row"] == 40


# --- sealed fixtures ----------------------------------------------------------------------


def _profile(runtime: dict) -> dict:
    return {key: runtime.get(key) for key in PROFILE_KEYS}


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_sealed_fixture_replays(workflow: str) -> None:
    path = FIXTURES / f"h3_full_chain_{workflow}_toy.json"
    run = clean_run(workflow, decode=True)
    if os.environ.get(REGEN_ENV):
        path.parent.mkdir(exist_ok=True)
        path.write_text(fc.dumps(run.artifact))
    sealed = fc.loads(path.read_text())
    assert sealed["variant"] == _variant(workflow)
    if _profile(sealed["runtime"]) != _profile(run.artifact["runtime"]):
        pytest.skip(
            f"fixture sealed on {_profile(sealed['runtime'])}, this is "
            f"{_profile(run.artifact['runtime'])}: bytes are only claimed within one profile"
        )
    report = fc.compare(sealed, run.artifact)
    assert report["first_drift"] is None, report["first_drift"]
    assert report["gate"]["hard_gate"] == "pass"
