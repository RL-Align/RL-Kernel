# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``image_patch_pack_unpack`` (RFC #738): image latents <-> width-96 transformer rows.

CPU only. Pins:

* pack and unpack are exact inverses, bit for bit, for every supported dtype and for
  non-square and non-power-of-two latents;
* the row order is the one ``packed_layout`` records in ``patch_coord``;
* packing is batch-invariant (a request's rows do not depend on its batch neighbours);
* the adapter equals the pinned Diffusers ``patchify_video_latents`` on targets and
  references and the pinned ``MiniMaxH3ImageAfterDenoiseStep`` inverse on the generated
  suffix (Diffusers-gated);
* unsupported geometry raises before any data moves.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.minimax_h3_image import packed_layout as pl
from rl_engine.models.minimax_h3_image import patch_pack as pp

SHAPES = [(2, 2), (4, 6), (62, 98), (128, 128), (6, 250)]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _latents(height: int, width: int, *, batch: int = 1, dtype=torch.float32, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 24, height, width, generator=generator).to(dtype)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("hw", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_pack_unpack_is_an_exact_inverse(hw, dtype) -> None:
    packer = pp.ImagePatchPacker()
    latents = _latents(*hw, batch=2, dtype=dtype)
    rows = packer.pack_image(latents)
    assert tuple(rows.shape) == (2, (hw[0] // 2) * (hw[1] // 2), 96)
    assert rows.dtype == dtype
    back = packer.unpack_image(rows, hw)
    assert torch.equal(_bits(back), _bits(latents))


def test_nan_payloads_survive_the_round_trip() -> None:
    packer = pp.ImagePatchPacker()
    latents = _latents(4, 4)
    raw = latents.view(torch.int32)
    raw[0, 3, 1, 2] = 0x7FC0_1234  # a NaN with a payload
    back = packer.unpack_image(packer.pack_image(latents), (4, 4))
    assert torch.equal(back.view(torch.int32), raw)


def test_five_d_single_frame_input_is_accepted() -> None:
    packer = pp.ImagePatchPacker()
    latents = _latents(4, 6)
    assert torch.equal(packer.pack_image(latents), packer.pack_image(latents[:, :, None]))


def test_row_order_matches_the_layout_patch_coord() -> None:
    packer = pp.ImagePatchPacker()
    refs, target = [(4, 4), (2, 6)], (4, 6)
    layout = pl.build_layout(torch.ones(3, dtype=torch.long), refs, target)
    latents = [_latents(*g, seed=i) for i, g in enumerate(refs + [target])]
    rows = torch.cat([packer.pack_image(x) for x in latents], dim=1)[0]
    for k, seq_row in enumerate(layout.image_indices.tolist()):
        kind = int(layout.source_kind[seq_row])
        source = int(layout.reference_index[seq_row]) if kind == pl.SOURCE_REFERENCE else -1
        r, c = layout.patch_coord[seq_row].tolist()
        image = latents[source]
        block = image[0, :, 2 * r : 2 * r + 2, 2 * c : 2 * c + 2]  # (24, 2, 2): channel-major
        assert torch.equal(rows[k], block.reshape(-1)), k
    for row in range(6):
        assert pp.patch_of_row(row, target) == tuple(
            layout.patch_coord[layout.image_indices[7 + row]].tolist()
        )


def test_pack_is_batch_invariant() -> None:
    packer = pp.ImagePatchPacker()
    batch = _latents(6, 10, batch=3)
    rows = packer.pack_image(batch)
    for b in range(3):
        assert torch.equal(rows[b : b + 1], packer.pack_image(batch[b : b + 1].contiguous()))


def test_references_concatenate_in_packed_order() -> None:
    packer = pp.ImagePatchPacker()
    a, b = _latents(4, 4, seed=1), _latents(2, 6, seed=2)
    rows = packer.pack_references([a, b])
    assert tuple(rows.shape) == (1, 4 + 3, 96)
    assert torch.equal(rows[:, :4], packer.pack_image(a))
    assert torch.equal(rows[:, 4:], packer.pack_image(b))
    assert not torch.equal(packer.pack_references([b, a]), rows)  # I9: order matters


def test_unpack_target_reads_only_the_generated_suffix() -> None:
    packer = pp.ImagePatchPacker()
    refs = packer.pack_references([_latents(4, 4, seed=1)])
    target = _latents(4, 6, seed=2)
    image_rows = torch.cat([refs, packer.pack_image(target)], dim=1)
    assert torch.equal(packer.unpack_target(image_rows, 4, (4, 6)), target)
    poisoned = image_rows.clone()
    poisoned[:, :4] = float("nan")  # reference rows are never read
    assert torch.equal(packer.unpack_target(poisoned, 4, (4, 6)), target)


def test_provenance_names_the_h3_backend() -> None:
    prov = pp.ImagePatchPacker().provenance()
    assert prov["semantic_op"] == "video_patch_pack_unpack"
    assert prov["backend_id"] == "pytorch-video-patch-pack-unpack-v1"
    assert prov["width"] == 96


def test_registered_backend_is_the_default_op() -> None:
    registry = pytest.importorskip("rl_engine.runtime.registry")
    descriptors = [
        d
        for d in registry._default_semantic_descriptors()
        if d.semantic_op == "video_patch_pack_unpack"
    ]
    assert [d.backend_id for d in descriptors] == [pp.ImagePatchPacker().backend_id]
    assert descriptors[0].determinism_or_alignment_properties["bitwise"] is True


# --- fail closed --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("latents", "needle"),
    [
        (torch.zeros(1, 24, 4), "(B, 24, H, W)"),
        (torch.zeros(1, 24, 2, 4, 4), "T = 1"),
        (torch.zeros(1, 16, 4, 4), "24 channels"),
        (torch.zeros(1, 24, 5, 4), "not divisible"),
        (torch.zeros(1, 24, 4, 3), "not divisible"),
        (torch.zeros(1, 24, 4, 4, dtype=torch.float64), "dtype"),
        (torch.zeros(1, 24, 4, 8)[..., ::2], "contiguous"),
    ],
)
def test_unsupported_latents_fail_closed(latents: torch.Tensor, needle: str) -> None:
    with pytest.raises((ValueError, TypeError), match=needle):
        pp.ImagePatchPacker().pack_image(latents)


def test_unsupported_rows_fail_closed() -> None:
    packer = pp.ImagePatchPacker()
    rows = packer.pack_image(_latents(4, 6))
    with pytest.raises(ValueError, match="shape"):
        packer.unpack_image(rows, (4, 4))  # row count disagrees with the grid
    with pytest.raises(ValueError, match="not divisible"):
        packer.unpack_image(rows, (3, 8))
    with pytest.raises(ValueError, match="reference"):
        packer.unpack_target(rows, 2, (4, 6))
    with pytest.raises(ValueError, match="non-negative"):
        packer.unpack_target(rows, -1, (4, 6))
    with pytest.raises(ValueError, match="at least one"):
        packer.pack_references([])
    with pytest.raises(ValueError, match="dtype"):
        packer.pack_references([_latents(4, 4), _latents(4, 4, dtype=torch.bfloat16)])
    with pytest.raises(ValueError, match="outside"):
        pp.patch_of_row(6, (4, 6))


# --- pinned Diffusers ---------------------------------------------------------------------


@pytest.mark.parametrize("hw", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_pack_matches_pinned_patchify(hw) -> None:
    module = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
    latents = _latents(*hw)
    # MiniMaxH3ImagePrepareLatentsStep: (1, 24, 1, H, W) -> (rows, 96), batch flattened.
    expected = module.patchify_video_latents(latents[:, :, None], (1, 2, 2))
    assert torch.equal(pp.ImagePatchPacker().pack_image(latents)[0], expected)


def test_references_match_the_pinned_condition_rows() -> None:
    module = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.before_denoise")
    refs = [_latents(4, 4, seed=1), _latents(2, 6, seed=2), _latents(30, 30, seed=3)]
    expected = torch.cat([module.patchify_video_latents(r[:, :, None], (1, 2, 2)) for r in refs])
    assert torch.equal(pp.ImagePatchPacker().pack_references(refs)[0], expected)


@pytest.mark.parametrize("hw", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_unpack_matches_the_pinned_unpack_step(hw) -> None:
    decoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3_image.decoders")
    import inspect

    source = inspect.getsource(decoders.MiniMaxH3ImageAfterDenoiseStep.__call__)
    for line in (
        "rows = block_state.latents[block_state.num_reference_rows :]",
        "rows = rows.permute(0, 3, 1, 4, 2, 5)",
        "-1, channels, block_state.latent_height, block_state.latent_width",
    ):
        assert line in source, line  # the copy below is still the pinned inverse
    packer = pp.ImagePatchPacker()
    refs = packer.pack_references([_latents(4, 4, seed=5)])[0]
    target = _latents(*hw, seed=6)
    latents = torch.cat([refs, packer.pack_image(target)[0]])
    # MiniMaxH3ImageAfterDenoiseStep.__call__, verbatim on the pipeline's 2-D rows.
    rows = latents[4:]
    rows = rows.reshape(-1, hw[0] // 2, hw[1] // 2, 24, 2, 2)
    rows = rows.permute(0, 3, 1, 4, 2, 5)
    expected = rows.reshape(-1, 24, hw[0], hw[1]).contiguous()
    assert torch.equal(packer.unpack_target(latents[None], 4, hw), expected)
    assert torch.equal(expected, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_cuda_rows_equal_cpu_rows(dtype) -> None:
    packer = pp.ImagePatchPacker()
    for hw in SHAPES:
        latents = _latents(*hw, batch=2, dtype=dtype)
        cpu = packer.pack_image(latents)
        gpu = packer.pack_image(latents.cuda())
        assert torch.equal(_bits(gpu.cpu()), _bits(cpu))
        back = packer.unpack_image(gpu, hw)
        assert back.is_cuda and torch.equal(_bits(back.cpu()), _bits(latents))
