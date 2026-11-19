# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared core of the FL2VA / Ref2VA pipeline contracts (RFC #420 Phase B, H16, H-C7)."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
import torch

from rl_engine.models.minimax_h3 import fingerprint as fp
from rl_engine.models.minimax_h3 import packed_layout as pl


def _toy_layout(workflow: str = "t2va", *, origin_shift: float = 0.0, cond: int = 0):
    """Hand-built layout: 3 text rows, ``cond`` condition rows, 2 audio rows, 4 video rows."""

    tags = torch.tensor([1, 0, 1])
    s = 3 + cond + 2 + 4
    pos = torch.zeros(s, 3, dtype=torch.float64)
    pos[:3, 0] = torch.arange(3, dtype=torch.float64)
    pos[3 + cond + 2 :, 0] = 3.0 + origin_shift
    video = torch.cat([torch.arange(3, 3 + cond), torch.arange(3 + cond + 2, s)])
    audio = torch.arange(3 + cond, 3 + cond + 2)
    return pl.build_layout(
        workflow=workflow,
        text_token_tags=tags,
        position_ids=pos,
        video_indices=video,
        audio_indices=audio,
        num_condition_video_rows=cond,
        num_condition_audio_rows=0,
        geometry={"toy": True},
    )


# --- geometry -----------------------------------------------------------------------------


def test_canvas_and_frame_counts_match_released_defaults():
    assert pl.resolve_canvas_size(16, 9) == (768, 1344)
    assert pl.resolve_canvas_size(9, 16) == (1344, 768)
    assert pl.resolve_canvas_size(1, 1) == (768, 768)
    with pytest.raises(ValueError, match="aspect ratio"):
        pl.resolve_canvas_size(5, 1)
    assert pl.align_num_frames(121) == 124 and pl.align_num_frames(124) == 124
    assert pl.video_latent_num_frames(124) == 37
    assert pl.audio_latent_num_frames(124) == 207
    assert pl.resolve_num_frames(120) == 124
    with pytest.raises(ValueError, match="5-15 s"):
        pl.resolve_num_frames(346)  # aligns to 362 = 15.08 s
    with pytest.raises(ValueError, match="multiples of 32"):
        pl.check_canvas(768, 1340)


def _terms(n):
    return [5.0 / 3.0 * (1, 4, 4, 4, 4)[i % 5] for i in range(n)]


def test_last_anchor_span_is_numpy_pairwise_written_out():
    import numpy as np

    for n in range(1, 1200):
        spans = np.ones(n, dtype=np.float64) * (5.0 / 3.0)
        for offset, k in enumerate((1, 4, 4, 4, 4)):
            spans[offset::5] *= k
        assert pl.last_anchor_span_pairwise(n) == float(spans.sum()), n


def test_video_span_summations_are_explicit_and_interpreter_independent():
    import sys

    neumaier = [pl.video_span(n, "neumaier") for n in range(1, 400)]
    sequential = [pl.video_span(n, "sequential") for n in range(1, 400)]
    differ = [n for n in range(1, 400) if neumaier[n - 1] != sequential[n - 1]]
    assert differ[:4] == [6, 10, 11, 15]  # first latent-frame counts where the ulp moves
    # This interpreter's own ``sum`` is one of the two, depending on its version.
    builtin = [sum(_terms(n)) for n in range(1, 400)]
    expected = neumaier if sys.version_info >= (3, 12) else sequential
    assert builtin == expected
    # Pairwise (FL2VA last anchor) and plain sequential agree below 16 frames, the property
    # the pinned reference documents.
    assert all(
        pl.last_anchor_span_pairwise(n) == pl.video_span(n, "sequential") for n in range(1, 16)
    )
    with pytest.raises(ValueError, match="span summation"):
        pl.video_span(3, "kahan")


def test_spatial_grid_is_numpy_linspace_without_right_endpoint():
    import numpy as np

    grid = pl.spatial_position_grid(48, 2, float(np.sqrt(48 * 84)))
    assert grid.dtype == torch.float64 and grid.numel() == 24
    ratio = 48 / np.sqrt(48 * 84)
    left = (1 - ratio) / 2
    expected = np.linspace(left, left + ratio, 24, endpoint=False) * 32
    assert torch.equal(grid, torch.from_numpy(expected))


# --- layout invariants --------------------------------------------------------------------


def test_layout_validation_rejects_broken_partitions():
    layout = _toy_layout()
    assert layout.sequence_length == 9 and layout.variant == "FL2VA"
    with pytest.raises(ValueError, match="partition"):
        pl.build_layout(
            workflow="t2va",
            text_token_tags=torch.tensor([1, 1, 1]),
            position_ids=torch.zeros(9, 3, dtype=torch.float64),
            video_indices=torch.arange(5, 9),
            audio_indices=torch.arange(4, 6),  # overlaps video row 5, leaves row 3
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
            geometry={},
        )
    with pytest.raises(ValueError, match="float64"):
        pl.build_layout(
            workflow="t2va",
            text_token_tags=torch.tensor([1, 1, 1]),
            position_ids=torch.zeros(9, 3, dtype=torch.float32),
            video_indices=torch.arange(5, 9),
            audio_indices=torch.arange(3, 5),
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
            geometry={},
        )
    with pytest.raises(ValueError, match="text tag"):
        pl.check_text_token_tags(torch.tensor([1, 2]))


def test_row_timesteps_pin_conditions_and_text_inherits_video():
    layout = _toy_layout("fl2va", cond=2)
    timesteps, index = pl.row_timesteps(
        layout, video_timestep=0.25, audio_timestep=0.5, condition_video_timestep=0.999
    )
    assert timesteps.dtype == torch.float32
    assert timesteps.tolist() == pytest.approx([0.25, 0.5, 0.999])
    per_row = timesteps[index]
    assert per_row[:3].tolist() == pytest.approx([0.25] * 3)  # text
    assert per_row[3:5].tolist() == pytest.approx([0.999] * 2)  # keyframe rows
    assert per_row[5:7].tolist() == pytest.approx([0.5] * 2)  # audio
    plan = pl.row_timestep_plan(layout, [0.0, 1.0], [0.0, 1.0])
    assert plan[1][0].tolist() == [1.0]  # max(1.0, 0.999) collapses with the video level


# --- conditioning rows --------------------------------------------------------------------


def test_patchify_is_frame_major_then_row_major():
    x = torch.arange(1 * 2 * 1 * 2 * 4, dtype=torch.float32).reshape(1, 2, 1, 2, 4)
    rows = pl.patchify_video_latents(x)
    assert rows.shape == (2, 8)
    # first patch: channel 0 then 1, each (h, w) = (0,0),(0,1),(1,0),(1,1)
    assert rows[0].tolist() == [0, 1, 4, 5, 8, 9, 12, 13]


def test_noise_level_is_mixed_in_fp32_not_fp64():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4096, generator=g)
    noise = torch.randn(4096, generator=g)
    ours = pl.noise_condition(x0, noise)
    t = torch.tensor(0.999, dtype=torch.float32)
    assert torch.equal(ours, t * x0 + (1.0 - t) * noise)
    python_float = 0.999 * x0 + (1.0 - 0.999) * noise  # the H15 probe arm
    assert not torch.equal(ours, python_float)


def test_condition_noise_is_drawn_one_condition_at_a_time():
    conds = [torch.zeros(1, 4, 1, 4, 6), torch.ones(1, 4, 1, 4, 6)]
    rows = pl.pack_condition_rows(conds, generator=torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)
    first = torch.randn(conds[0].shape, generator=g)
    second = torch.randn(conds[1].shape, generator=g)
    t = torch.tensor(0.999)
    expected = torch.cat(
        [
            pl.patchify_video_latents(t * conds[0] + (1 - t) * first),
            pl.patchify_video_latents(t * conds[1] + (1 - t) * second),
        ]
    )
    assert torch.equal(rows, expected)


# --- identity (H16) -----------------------------------------------------------------------


def test_variants_share_components_except_the_transformer():
    fl2va = pl.checkpoint_component_hashes("FL2VA")
    ref2va = pl.checkpoint_component_hashes("Ref2VA")
    assert set(fl2va) == set(ref2va)
    assert fl2va["transformer"] != ref2va["transformer"]
    for name in set(fl2va) - {"transformer"}:
        assert fl2va[name] == ref2va[name], name
    assert pl.PIPELINES["FL2VA"].recipe_sha256() != pl.PIPELINES["Ref2VA"].recipe_sha256()
    # Same config and tensor table, so nothing but the shards can tell them apart.
    assert fp.transformer_config("FL2VA") == fp.transformer_config("Ref2VA")
    shards = {v: pl.transformer_shard_hashes(v) for v in ("FL2VA", "Ref2VA")}
    assert shards["FL2VA"].keys() == shards["Ref2VA"].keys()
    assert all(shards["FL2VA"][k] != shards["Ref2VA"][k] for k in shards["FL2VA"])


def _fake_download(root, variant, *, swap=()):
    """A ``hf_hub_download(local_dir=root)`` tree: empty shards plus the Hub metadata files."""

    directory = root / "transformer"
    meta = root / ".cache" / "huggingface" / "download" / "transformer"
    directory.mkdir(parents=True)
    meta.mkdir(parents=True)
    other = "Ref2VA" if variant == "FL2VA" else "FL2VA"
    for name, sha in pl.transformer_shard_hashes(variant).items():
        if name in swap:
            sha = pl.transformer_shard_hashes(other)[name]
        (directory / name).write_bytes(b"")
        (meta / f"{name}.metadata").write_text(f"{fp.REVISION}\n{sha}\n0\n")
    return directory


@pytest.mark.parametrize("variant", ["FL2VA", "Ref2VA"])
def test_identify_transformer_dir_from_hub_metadata(tmp_path, variant):
    assert pl.identify_transformer_dir(_fake_download(tmp_path, variant)) == variant


def test_identify_transformer_dir_refuses_mixed_shards(tmp_path):
    name = sorted(pl.transformer_shard_hashes("FL2VA"))[3]
    directory = _fake_download(tmp_path, "FL2VA", swap=(name,))
    with pytest.raises(pl.VariantMismatch, match="neither variant"):
        pl.identify_transformer_dir(directory)


def test_identify_transformer_dir_needs_a_hash_source(tmp_path):
    directory = tmp_path / "transformer"
    directory.mkdir()
    for name in pl.transformer_shard_hashes("FL2VA"):
        (directory / name).write_bytes(b"")
    with pytest.raises(ValueError, match="full_hash=True"):
        pl.identify_transformer_dir(directory)
    with pytest.raises(pl.VariantMismatch):  # empty files hash to neither variant
        pl.identify_transformer_dir(directory, full_hash=True)


def _content_download(root, monkeypatch, *, metadata, contents):
    """A ``--local-dir`` tree whose shard bytes and Hub metadata name the given variants.

    The pinned shard hashes are replaced by hashes of small per-variant contents, so the
    files can be hashed for real.
    """

    names = sorted(pl.transformer_shard_hashes("FL2VA"))
    payload = {v: {n: f"{v}:{n}".encode() for n in names} for v in ("FL2VA", "Ref2VA")}
    table = {v: {n: hashlib.sha256(b).hexdigest() for n, b in payload[v].items()} for v in payload}
    monkeypatch.setattr(pl, "transformer_shard_hashes", lambda variant: dict(table[variant]))
    directory = root / "transformer"
    meta = root / ".cache" / "huggingface" / "download" / "transformer"
    directory.mkdir(parents=True)
    meta.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(payload[contents][name])
        (meta / f"{name}.metadata").write_text(f"{fp.REVISION}\n{table[metadata][name]}\n0\n")
    return directory


@pytest.mark.parametrize("variant", ["FL2VA", "Ref2VA"])
def test_identify_transformer_dir_full_hash_reads_shards_despite_metadata(
    tmp_path, monkeypatch, variant
):
    directory = _content_download(tmp_path, monkeypatch, metadata=variant, contents=variant)
    reads = []
    original = pl._file_sha256
    monkeypatch.setattr(pl, "_file_sha256", lambda path: reads.append(path) or original(path))
    assert pl.identify_transformer_dir(directory) == variant
    assert reads == []  # metadata only
    assert pl.identify_transformer_dir(directory, full_hash=True) == variant
    assert len(reads) == len(pl.transformer_shard_hashes(variant))  # every shard read


def test_identify_transformer_dir_full_hash_catches_metadata_disagreeing_with_shards(
    tmp_path, monkeypatch
):
    # Metadata says FL2VA, the bytes are Ref2VA's: only a real hash can tell.
    directory = _content_download(tmp_path, monkeypatch, metadata="FL2VA", contents="Ref2VA")
    assert pl.identify_transformer_dir(directory) == "FL2VA"
    with pytest.raises(pl.VariantMismatch, match="download metadata records"):
        pl.identify_transformer_dir(directory, full_hash=True)


def test_structure_must_match_declared_workflow():
    assert pl.structural_workflow(_toy_layout("t2va")) == "t2va"
    keyframed = _toy_layout("fl2va", cond=2)
    assert pl.structural_workflow(keyframed) == "fl2va"
    pl.check_pipeline_identity(keyframed, transformer_variant="FL2VA")
    with pytest.raises(pl.VariantMismatch, match="paired with the Ref2VA"):
        pl.check_pipeline_identity(keyframed, transformer_variant="Ref2VA")
    # Conditioning rows that push the target clock forward are a Ref2VA layout, whatever the
    # layout claims to be.
    shifted = _toy_layout("fl2va", cond=2, origin_shift=1.0)
    assert pl.structural_workflow(shifted) == "ref2va"
    with pytest.raises(pl.VariantMismatch, match="structure"):
        pl.check_pipeline_identity(shifted)
    with pytest.raises(pl.VariantMismatch, match="structure"):
        pl.check_pipeline_identity(_toy_layout("t2va", cond=2))


def test_structure_of_a_layout_without_target_video_rows_is_refused():
    # validate() lets every video row be a condition row; the structural check must still
    # answer with an error, not an IndexError.
    no_target = dataclasses.replace(_toy_layout("ref2va", cond=2), num_condition_video_rows=6)
    no_target.validate()
    for check in (pl.structural_workflow, pl.check_pipeline_identity, pl.layout_artifact):
        with pytest.raises(ValueError, match="no target video rows"):
            check(no_target)


# --- artifact (H-C7) ----------------------------------------------------------------------


def test_artifact_is_sealed_and_names_the_first_difference(tmp_path):
    layout = _toy_layout("fl2va", cond=2)
    artifact = pl.layout_artifact(layout, inputs={"condition_rows": "ab" * 32})
    assert artifact == pl.layout_artifact(layout, inputs={"condition_rows": "ab" * 32})
    path = tmp_path / "layout.json"
    pl.save_artifact(artifact, path)
    assert pl.load_artifact(path) == artifact
    tampered = json.loads(path.read_text())
    tampered["num_condition_video_rows"] = 0
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="seal"):
        pl.load_artifact(path)

    other = _toy_layout("fl2va", cond=2, origin_shift=0.0)
    other.position_ids[3, 1] = 0.5  # one coordinate of one keyframe row
    second = pl.layout_artifact(other)
    assert pl.first_difference(artifact, second) == "input_sha256.condition_rows"
    assert (
        pl.first_difference({**artifact, "input_sha256": {}}, second)
        == "layout_component_sha256.position_ids"
    )

    pl.check_pipeline_identity(layout, artifact=artifact)
    with pytest.raises(pl.VariantMismatch, match="do not describe"):
        pl.check_pipeline_identity(other, artifact=artifact)


def test_denoiser_inputs_move_without_a_byte_change():
    layout = _toy_layout("fl2va", cond=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = layout.denoiser_inputs(device)
    for name, tensor in moved.items():
        assert tensor.dtype == getattr(layout, name).dtype
        assert torch.equal(tensor.cpu(), getattr(layout, name)), name


def test_written_out_sums_match_the_interpreter_on_random_series():
    import random
    import sys

    rng = random.Random(0)
    for _ in range(500):
        values = [
            rng.uniform(-1e3, 1e3) * 10 ** rng.randint(-8, 8) for _ in range(rng.randint(0, 64))
        ]
        builtin = sum(values)
        if sys.version_info >= (3, 12):
            assert pl.neumaier_sum(values) == builtin
        else:
            assert pl.sequential_sum(values) == builtin
