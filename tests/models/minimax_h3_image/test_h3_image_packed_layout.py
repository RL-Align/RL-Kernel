# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``image_packed_layout_schema`` (RFC #738): the packed ``[text | refs | target]`` layout.

CPU only. Pins:

* ``build_layout`` equals the pinned Diffusers ``build_image_packed_sequence`` on every
  output (Diffusers-gated), and its source metadata describes each row;
* a fully consistent permutation keeps the row-identity digest, changes the layout digest,
  and leaves a tiny pinned-Diffusers transformer's output unchanged (RFC #738 I-C3);
* a partial permutation (one per-row tensor left behind) changes that output;
* every misalignment ``validate`` guards against fails closed.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.minimax_h3_image import packed_layout as pl

REFS = [(4, 4), (2, 6)]
TARGET = (4, 6)


def _tags(num_text: int = 9, vision: slice = slice(3, 5)) -> torch.Tensor:
    tags = torch.full((num_text,), pl.TEXT_TAG, dtype=torch.long)
    tags[vision] = pl.IMAGE_TAG  # a reference's vision-block rows are image-tagged
    return tags


def _layout(refs=REFS, target=TARGET, num_text: int = 9) -> pl.PackedLayout:
    return pl.build_layout(_tags(num_text), refs, target)


def _shuffle(seq: int, seed: int = 0) -> torch.Tensor:
    return torch.randperm(seq, generator=torch.Generator().manual_seed(seed))


def test_canonical_layout_order_and_counts() -> None:
    layout = _layout()
    assert layout.sequence_length == 9 + 4 + 3 + 6
    assert layout.num_reference_rows == 4 + 3
    assert torch.equal(layout.text_indices, torch.arange(9))
    assert torch.equal(layout.image_indices, torch.arange(9, 22))
    kinds = layout.source_kind[layout.image_indices].tolist()
    assert kinds == [pl.SOURCE_REFERENCE] * 7 + [pl.SOURCE_TARGET] * 6


def test_tags_follow_the_presentation_and_latents_are_image() -> None:
    layout = _layout()
    assert torch.equal(layout.token_tags[:9], _tags())
    assert bool((layout.token_tags[9:] == pl.IMAGE_TAG).all())


def test_source_metadata_describes_each_row() -> None:
    layout = _layout()
    assert layout.reference_index[9:13].tolist() == [0] * 4
    assert layout.reference_index[13:16].tolist() == [1] * 3
    assert bool((layout.reference_index[16:] == -1).all())
    # reference 1 is a 2 x 6 latent: one patch row of three patches.
    assert layout.patch_coord[13:16].tolist() == [[0, 0], [0, 1], [0, 2]]
    # the target is 4 x 6: two patch rows of three, row-major.
    assert layout.patch_coord[16:].tolist() == [[r, c] for r in range(2) for c in range(3)]
    assert bool((layout.patch_coord[:9] == -1).all())


def test_t2i_layout_has_no_reference_rows() -> None:
    layout = pl.build_layout(_tags(5, slice(0, 0)), [], (4, 4))
    assert layout.num_reference_rows == 0
    assert bool((layout.source_kind[layout.image_indices] == pl.SOURCE_TARGET).all())


def test_artifact_feeds_the_trajectory_layout_block() -> None:
    art = _layout().artifact()
    for key in (
        "position_sha256_fp64",
        "position_sha256_fp32",
        "sequence_length",
        "image_indices_sha256",
        "text_indices_sha256",
        "num_reference_rows",
    ):
        assert key in art
    assert art == _layout().artifact()


# --- permutations --------------------------------------------------------------------------


def test_consistent_permutation_keeps_row_identity() -> None:
    layout = _layout()
    permuted = pl.permute(layout, _shuffle(layout.sequence_length))
    assert pl.row_identity_digest(permuted) == pl.row_identity_digest(layout)
    assert pl.layout_digest(permuted) != pl.layout_digest(layout)
    # Each selection still points at the same rows, in the same order.
    for name in ("position_ids", "token_tags", "source_kind", "reference_index", "patch_coord"):
        for selection in ("image_indices", "text_indices"):
            a = getattr(layout, name)[getattr(layout, selection)]
            b = getattr(permuted, name)[getattr(permuted, selection)]
            assert torch.equal(a, b), (name, selection)
    assert permuted.num_reference_rows == layout.num_reference_rows


def test_permutation_round_trip_restores_the_layout() -> None:
    layout = _layout()
    order = _shuffle(layout.sequence_length, 3)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel())
    back = pl.permute(pl.permute(layout, order), inverse)
    assert pl.layout_digest(back) == pl.layout_digest(layout)


def test_reordering_references_is_not_a_consistent_permutation() -> None:  # I9
    a = pl.build_layout(_tags(), [(4, 4), (2, 6)], TARGET)
    b = pl.build_layout(_tags(), [(2, 6), (4, 4)], TARGET)
    assert pl.row_identity_digest(a) != pl.row_identity_digest(b)


def test_bad_permutations_fail_closed() -> None:
    layout = _layout()
    seq = layout.sequence_length
    with pytest.raises(ValueError, match="permutation"):
        pl.permute(layout, torch.zeros(seq, dtype=torch.long))
    with pytest.raises(ValueError, match="permutation"):
        pl.permute(layout, torch.arange(seq - 1))
    with pytest.raises(ValueError, match="int64"):
        pl.permute(layout, torch.arange(seq, dtype=torch.int32))


# --- validation ---------------------------------------------------------------------------


def _edit(layout: pl.PackedLayout, **changes) -> pl.PackedLayout:
    from dataclasses import replace

    return replace(layout, **changes)


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        (lambda lay: {"image_indices": lay.image_indices.to(torch.int32)}, "int64"),
        (lambda lay: {"image_indices": lay.image_indices[None]}, "rank 1"),
        (lambda lay: {"image_indices": lay.image_indices + 1}, "out-of-range"),
        (
            lambda lay: {
                "image_indices": torch.cat([lay.image_indices[:-1], lay.image_indices[:1]])
            },
            "duplicate",
        ),
        (lambda lay: {"text_indices": lay.text_indices[:-1]}, "partition"),
        (lambda lay: {"num_reference_rows": 6}, "count the reference rows"),
        (
            lambda lay: {
                "image_indices": torch.cat([lay.image_indices[7:], lay.image_indices[:7]])
            },
            "prefix",
        ),
        (lambda lay: {"token_tags": torch.ones_like(lay.token_tags)}, "image tag"),
        (lambda lay: {"token_tags": lay.token_tags.clone().fill_(2)}, "image tag"),
        (lambda lay: {"source_kind": torch.zeros_like(lay.source_kind)}, "source_kind"),
        (lambda lay: {"reference_index": torch.zeros_like(lay.reference_index)}, "reference_index"),
        (lambda lay: {"position_ids": lay.position_ids.float()}, "float64"),
        (lambda lay: {"patch_coord": lay.patch_coord[:-1]}, "patch_coord"),
        (lambda lay: {"reference_grids": ((4, 4),)}, "reference grids"),
    ],
)
def test_misaligned_layouts_fail_closed(change, needle: str) -> None:
    layout = _layout()
    with pytest.raises(ValueError, match=needle):
        pl.validate(_edit(layout, **change(layout)))


def test_text_tags_must_be_image_or_text() -> None:
    with pytest.raises(ValueError, match="text_token_tags"):
        pl.build_layout(torch.full((4,), 2, dtype=torch.long), [], TARGET)
    with pytest.raises(ValueError, match="int64"):
        pl.build_layout(torch.ones(4, dtype=torch.int32), [], TARGET)


def test_reference_only_selection_fails_closed() -> None:
    layout = _layout()
    with pytest.raises(ValueError, match="generated rows"):
        pl.validate(
            _edit(
                layout,
                image_indices=layout.image_indices[:7],
                text_indices=torch.cat([layout.text_indices, layout.image_indices[7:]]),
                source_kind=torch.where(
                    layout.source_kind == pl.SOURCE_TARGET, pl.SOURCE_TEXT, layout.source_kind
                ),
            )
        )


# --- pinned Diffusers ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_text", "refs", "target"),
    [(9, REFS, TARGET), (5, [], (4, 4)), (40, [(30, 30), (6, 62), (2, 2)], (62, 98))],
)
def test_matches_pinned_build_image_packed_sequence(num_text, refs, target) -> None:
    module = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
    tags = _tags(num_text, slice(1, min(4, num_text)) if refs else slice(0, 0))
    ours = pl.build_layout(tags, refs, target).diffusers_tuple()
    theirs = module.build_image_packed_sequence(tags, refs, target[0], target[1], (1, 2, 2), 0)
    for a, b in zip(ours[:4], theirs[:4]):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert ours[4] == theirs[4]


TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,
    "patch_size": (1, 2, 2),
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
    "num_pdd_heads": 6,
}


def _tiny_forward(model, layout: pl.PackedLayout, image_rows, text_rows, timestep_indices):
    return model(
        hidden_states=image_rows,
        encoder_hidden_states=text_rows,
        timestep=torch.tensor([0.25, 0.999], dtype=torch.float64),
        timestep_indices=timestep_indices,
        token_tags=layout.token_tags,
        position_ids=layout.position_ids,
        image_indices=layout.image_indices,
        text_indices=layout.text_indices,
        pdd_step_index=3,
        return_dict=False,
    )[0]


def _tiny_case():
    # Gate on the pinned pipeline package, not on diffusers.models.transformers: an older
    # Diffusers (0.41 has MiniMax-H3 but not H3-Image) imports fine and then has no class.
    pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image")
    transformers = pytest.importorskip("diffusers.models.transformers")
    torch.manual_seed(0)
    model = transformers.MiniMaxH3ImageTransformer2DModel(**TINY).to(torch.float64).eval()
    layout = _layout()
    generator = torch.Generator().manual_seed(1)
    image_rows = torch.randn(1, layout.image_indices.numel(), 16, generator=generator).double()
    text_rows = torch.randn(1, layout.num_text_tokens, 8, generator=generator).double()
    # references at t = 0.999 (table row 1), everything else at the schedule t (row 0)
    timestep_indices = (layout.source_kind == pl.SOURCE_REFERENCE).long()
    return model, layout, image_rows, text_rows, timestep_indices


def test_consistent_permutation_preserves_the_transformer_output() -> None:  # I-C3
    model, layout, image_rows, text_rows, steps = _tiny_case()
    order = _shuffle(layout.sequence_length, 7)
    permuted = pl.permute(layout, order)
    with torch.no_grad():
        base = _tiny_forward(model, layout, image_rows, text_rows, steps)
        moved = _tiny_forward(model, permuted, image_rows, text_rows, steps[order])
    # Full attention sums keys in a different order after the permutation, so this is the
    # float64 reassociation bound, not bitwise equality.
    torch.testing.assert_close(moved, base, rtol=0, atol=1e-12)


def test_partial_permutation_changes_the_transformer_output() -> None:
    model, layout, image_rows, text_rows, steps = _tiny_case()
    order = _shuffle(layout.sequence_length, 7)
    permuted = pl.permute(layout, order)
    with torch.no_grad():
        base = _tiny_forward(model, layout, image_rows, text_rows, steps)
        # timestep_indices left in the old order: rows get each other's AdaLN modulation.
        stale = _tiny_forward(model, permuted, image_rows, text_rows, steps)
    assert (stale - base).abs().max() > 1e-6
