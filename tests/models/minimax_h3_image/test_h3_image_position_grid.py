# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``image_position_grid`` (RFC #738): packed float64 ``(t, h, w)`` coordinates.

CPU only. Pins:

* bitwise equality with the pinned Diffusers ``build_image_packed_sequence`` (skipped when
  that Diffusers is not installed) and with committed digests (always run);
* the image clock: text at its row index, one integer slot per reference, target last;
* each reference keeps its own geometry (I10) and reordering references moves rows (I9);
* the NumPy grid equals the H3 video grid construction of RFC #420 for the same frame;
* the Torch ``linspace`` arm (I8) is a different profile, and ``first_drift`` locates it;
* unsupported geometry fails before any coordinate is built.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.minimax_h3_image import position_grid as pg

# (num_text_tokens, reference latent grids, target latent grid) -> (sha256 fp64, sha256 fp32).
# Generated from Diffusers 875373ac build_image_packed_sequence and checked against it in
# test_matches_pinned_diffusers below.
GOLDEN: dict[tuple, tuple[str, str]] = {
    (7, (), (64, 64)): (
        "a2f7b22c2276eb462847c4f3d57c79aa0b0b3fd29f00abe7acd78924936e9573",
        "e373a46634953f92df9c22c154d5c778bf0e93bef75efcffb9f71a9bd8ef58f7",
    ),
    (19, (), (96, 64)): (
        "f098a79a58abc50968cc7d975e2bd42a6b9e8d555236b76817a2654deadb3a40",
        "5ce1401fdffea3899e4d97103c8157318744eadd896569b4dbaab520008733df",
    ),
    (5, (), (2, 250)): (
        "a7f468ca6cacd58f979d6cd372b669e84d3a70cd6ae439a7167cbf688dda6d9a",
        "5ee064a3263916ea9b11f97056036786e98dcb4b7c4a96af21ab44e57103b170",
    ),
    (333, ((64, 64),), (96, 64)): (
        "b32536a4ed661bc79d58ef77efd960a75c815cfa101d365b747fa7b9468d930b",
        "0eebb26826f8e2edaf77f35675765abddf91c8347a03a7e1e9aed6495ebdadf1",
    ),
    (333, ((64, 64), (48, 80), (126, 2)), (96, 64)): (
        "44e0b8da26a6fd5dda31349a44936e71696dc859c7395481a3a0d07ff6787f84",
        "1b01c1bb841497e2443fbf7070570de50795f4d1a50008b992100b99ce1d9780",
    ),
    (0, ((30, 30),), (62, 98)): (
        "1b116bfbd20006c364324baea84209edf41bcf2d9db62b6eece19d298a8791c3",
        "ad538690982ef1b4a20129c142eeb09055405f38e92d15fdf5019e5c378a64be",
    ),
}


def _layouts() -> list[tuple[int, tuple[tuple[int, int], ...], tuple[int, int]]]:
    return [
        (7, (), (64, 64)),  # square 1024 px T2I
        (19, (), (96, 64)),  # non-square
        (5, (), (2, 250)),  # extreme aspect
        (333, ((64, 64),), (96, 64)),  # one reference
        (333, ((64, 64), (48, 80), (126, 2)), (96, 64)),  # heterogeneous references
        (0, ((30, 30),), (62, 98)),  # non-power-of-two, no text
    ]


def test_text_rows_sit_at_their_index() -> None:
    p = pg.packed_positions(11, [], (8, 8))
    assert torch.equal(p.position_ids[:11, 0], torch.arange(11, dtype=torch.float64))
    assert torch.count_nonzero(p.position_ids[:11, 1:]) == 0


def test_image_clock_gives_each_reference_one_slot_and_target_the_next() -> None:
    p = pg.packed_positions(10, [(4, 4), (8, 2)], (6, 6))
    kinds = [(s.kind, s.index, s.rotary_time) for s in p.sources]
    assert kinds == [("reference", 0, 10.0), ("reference", 1, 11.0), ("target", -1, 12.0)]
    for s in p.sources:
        assert torch.all(p.position_ids[s.start : s.stop, 0] == s.rotary_time)
    assert p.sources[-1].stop == p.sequence_length
    assert p.num_reference_rows == 4 + 4


def test_spatial_axis_is_centred_and_scaled() -> None:
    p = pg.packed_positions(0, [], (8, 8))
    h = p.position_ids[:, 1].unique()
    assert h.tolist() == [0.0, 8.0, 16.0, 24.0]  # square spans [0, 32)


def test_references_keep_their_own_geometry() -> None:
    p = pg.packed_positions(3, [(4, 12)], (12, 4))
    ref, target = p.sources
    ref_grid = p.position_ids[ref.start : ref.stop, 1:]
    assert torch.equal(ref_grid, pg.frame_grid(4, 12))
    # I10 probe: forcing the target geometry on the reference is a different grid.
    assert not torch.equal(ref_grid, pg.frame_grid(12, 4))


def test_reference_order_moves_rows() -> None:
    a = pg.packed_positions(3, [(4, 4), (4, 8)], (4, 4))
    b = pg.packed_positions(3, [(4, 8), (4, 4)], (4, 4))
    assert a.sequence_length == b.sequence_length
    assert pg.first_drift(a.position_ids, b.position_ids) is not None


def test_target_grid_does_not_depend_on_text_or_references() -> None:
    base = pg.packed_positions(0, [], (12, 20)).position_ids[:, 1:]
    for text, refs in ((17, []), (4, [(6, 6)]), (900, [(2, 2), (40, 8)])):
        p = pg.packed_positions(text, refs, (12, 20))
        target = p.sources[-1]
        assert torch.equal(p.position_ids[target.start :, 1:], base)


def test_matches_the_h3_video_frame_grid() -> None:
    h3_cases = pytest.importorskip("rl_engine.validation.models.h3_cases")
    import numpy as np

    for height, width in ((64, 64), (96, 64), (30, 62)):
        sqrt_area = float(np.sqrt(height * width))
        assert torch.equal(
            h3_cases._spatial_grid(height, 2, sqrt_area),
            torch.from_numpy(pg.spatial_axis(height, 2, sqrt_area)),
        )


def test_artifact_records_both_digests_and_the_row_map() -> None:
    p = pg.packed_positions(5, [(4, 4)], (8, 4))
    art = p.artifact()
    assert art["construction"] == pg.CONSTRUCTION
    assert art["shape"] == [5 + 4 + 8, 3]
    assert art["num_reference_rows"] == 4
    assert [s["kind"] for s in art["sources"]] == ["reference", "target"]
    assert art["sha256_fp64"] != art["sha256_fp32"]
    assert art == pg.packed_positions(5, [(4, 4)], (8, 4)).artifact()


@pytest.mark.parametrize("layout", _layouts(), ids=lambda lay: f"t{lay[0]}-r{len(lay[1])}-{lay[2]}")
def test_committed_digests(layout) -> None:
    text, refs, target = layout
    art = pg.packed_positions(text, list(refs), target).artifact()
    assert (art["sha256_fp64"], art["sha256_fp32"]) == GOLDEN[layout]


@pytest.mark.parametrize("layout", _layouts(), ids=lambda lay: f"t{lay[0]}-r{len(lay[1])}-{lay[2]}")
def test_matches_pinned_diffusers(layout) -> None:
    module = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
    text, refs, target = layout
    expected = module.build_image_packed_sequence(
        torch.ones(text, dtype=torch.long), list(refs), target[0], target[1], (1, 2, 2), 0
    )
    p = pg.packed_positions(text, list(refs), target)
    assert expected[0].dtype == torch.float64
    assert torch.equal(p.position_ids, expected[0])
    assert p.num_reference_rows == expected[4]
    assert torch.equal(expected[2], torch.arange(text, p.sequence_length))


def test_torch_linspace_probe_is_a_different_profile() -> None:
    drifted = [
        (h, w)
        for h in range(2, 130, 2)
        for w in (2, 62, 64, 98)
        if not torch.equal(pg.frame_grid(h, w), pg.frame_grid(h, w, axis="torch"))
    ]
    assert drifted, "the Torch arm should differ from NumPy on some canvas"
    h, w = drifted[0]
    report = pg.first_drift(pg.frame_grid(h, w), pg.frame_grid(h, w, axis="torch"))
    assert report is not None and report["reason"] == "value"
    assert report["axis"] in ("h", "w")
    assert report["max_abs"] < 1e-12
    probe = pg.packed_positions(0, [], (h, w), axis="torch").artifact()
    assert probe["construction"] == "probe:torch"


def test_torch_probe_can_survive_the_fp32_cast() -> None:
    # On an 18 x 2 latent the NumPy height axis holds an exact 0.0 where torch.linspace gives
    # 3.6e-15. The rotary module casts to float32, and the value stays non-zero there.
    numpy_grid, torch_grid = pg.frame_grid(18, 2), pg.frame_grid(18, 2, axis="torch")
    report = pg.first_drift(numpy_grid.float().double(), torch_grid.float().double())
    assert report is not None
    assert (report["row"], report["axis"], report["a"]) == (3, "h", 0.0)
    assert report["b"] != 0.0


def test_first_drift_rejects_unknown_layouts() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        pg.first_drift(torch.zeros(4, 5), torch.ones(4, 5))


def test_first_drift_on_equal_and_reshaped_tensors() -> None:
    a = pg.frame_grid(4, 4)
    assert pg.first_drift(a, a.clone()) is None
    assert pg.first_drift(a, a[:-1])["reason"] == "shape"


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"target_grid": (63, 64)}, "not divisible"),
        ({"target_grid": (0, 64)}, "positive"),
        ({"reference_grids": [(64, 64), (5, 4)]}, "reference 1"),
        ({"reference_grids": [(2, 2)] * 17}, "exceed"),
        ({"num_text_tokens": -1}, ">= 0"),
        ({"patch_size": (2, 2, 2)}, "temporal patch"),
        ({"patch_size": (1, 2)}, "three positive"),
    ],
)
def test_unsupported_geometry_fails_closed(kwargs, needle) -> None:
    args = {"num_text_tokens": 4, "reference_grids": [], "target_grid": (64, 64)}
    args.update(kwargs)
    with pytest.raises(ValueError, match=needle):
        pg.packed_positions(**args)


def test_unknown_axis_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="axis"):
        pg.frame_grid(4, 4, axis="jax")
