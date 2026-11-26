# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Packed layout of MiniMax-H3-Image requests (RFC #738 ``image_packed_layout_schema``).

One request is one packed sequence. Besides the rotary coordinates (``image_position_grid``)
every row carries semantic metadata the transformer and the scheduler depend on:

* ``token_tags``: modality tag (image 0, text 1). Text rows take the tags of the prompt
  presentation, where a reference's vision-block rows are image-tagged; every latent row
  is image-tagged;
* ``image_indices`` / ``text_indices``: sequence positions of the image rows (references
  first, then the target, in selection order) and of the text rows. The transformer
  scatters its projected rows to these positions and selects its output through
  ``image_indices``; the scheduler updates only the selection suffix after
  ``num_reference_rows``;
* source metadata: ``source_kind`` (text / reference / target), ``reference_index`` and the
  patch coordinate ``(patch_row, patch_col)`` inside the source image.

The released pipeline builds the canonical layout in order (``build_image_packed_sequence``
in Diffusers ``modular_pipelines/minimax_h3_image/before_denoise.py`` at commit
875373ac7f5856071e345a3e394c74efccc6b9e6). ``build_layout`` reproduces it exactly and adds
the source metadata.

A *fully consistent* permutation moves every per-row tensor together and rewrites the
index tensors so each selection keeps its order (``permute``). The selected image rows
are then the same rows in the same selection order, so the transformer output and the
scheduler slices are unchanged. ``row_identity_digest`` is order-free and equal across
such permutations; ``layout_digest`` is order-sensitive and changes. ``validate`` fails
closed on anything that would silently misalign rows: wrong dtype or rank, duplicate or
out-of-range indices, selections that do not partition the sequence, references that are
not a prefix of the image selection, or tags that disagree with the source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch

from rl_engine.models.minimax_h3.trajectory import tensor_digest
from rl_engine.models.minimax_h3_image import position_grid as pg
from rl_engine.models.minimax_h3_image.fingerprint import MODALITY_TAGS

SCHEMA_VERSION = "rlkernel.h3_image.packed_layout.v1"
IMAGE_TAG = MODALITY_TAGS["image"]
TEXT_TAG = MODALITY_TAGS["text"]
SOURCE_TEXT, SOURCE_REFERENCE, SOURCE_TARGET = 0, 1, 2
SOURCE_KINDS = {"text": SOURCE_TEXT, "reference": SOURCE_REFERENCE, "target": SOURCE_TARGET}
PER_ROW = (
    "position_ids",
    "token_tags",
    "source_kind",
    "reference_index",
    "patch_coord",
)


@dataclass(frozen=True)
class PackedLayout:
    """Every per-row tensor of one packed request, plus the two selections."""

    position_ids: torch.Tensor  # float64 [S, 3]
    token_tags: torch.Tensor  # int64 [S]
    image_indices: torch.Tensor  # int64 [N_image], references first
    text_indices: torch.Tensor  # int64 [N_text]
    source_kind: torch.Tensor  # int64 [S]: 0 text, 1 reference, 2 target
    reference_index: torch.Tensor  # int64 [S]: reference number, -1 otherwise
    patch_coord: torch.Tensor  # int64 [S, 2]: (row, col) of the patch in its image, -1 for text
    num_reference_rows: int
    reference_grids: tuple[tuple[int, int], ...]
    target_grid: tuple[int, int]
    patch_size: tuple[int, int, int] = pg.DEFAULT_PATCH

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def num_text_tokens(self) -> int:
        return int(self.text_indices.numel())

    def diffusers_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """``(position_ids, token_tags, image_indices, text_indices, num_reference_rows)``."""

        return (
            self.position_ids,
            self.token_tags,
            self.image_indices,
            self.text_indices,
            self.num_reference_rows,
        )

    def artifact(self) -> dict[str, Any]:
        """The ``inputs.layout`` fields of an H3-Image trajectory plus the layout digests."""

        return {
            "schema_version": SCHEMA_VERSION,
            "position_sha256_fp64": tensor_digest(self.position_ids),
            "position_sha256_fp32": tensor_digest(self.position_ids.to(torch.float32)),
            "sequence_length": self.sequence_length,
            "image_indices_sha256": tensor_digest(self.image_indices),
            "text_indices_sha256": tensor_digest(self.text_indices),
            "token_tags_sha256": tensor_digest(self.token_tags),
            "num_reference_rows": self.num_reference_rows,
            "reference_grids": [list(g) for g in self.reference_grids],
            "target_grid": list(self.target_grid),
            "layout_sha256": layout_digest(self),
            "row_identity_sha256": row_identity_digest(self),
        }


def _long(name: str, value: Any, rank: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.dtype != torch.long or value.ndim != rank:
        raise ValueError(
            f"{name} must be int64 of rank {rank}, got {value.dtype} rank {value.ndim}"
        )
    return value


def build_layout(
    text_token_tags: torch.Tensor,
    reference_grids: Sequence[tuple[int, int]],
    target_grid: tuple[int, int],
    patch_size: Sequence[int] = pg.DEFAULT_PATCH,
) -> PackedLayout:
    """Canonical ``[text | references | target]`` layout, as the released pipeline builds it."""

    tags = _long("text_token_tags", text_token_tags, 1).cpu()
    if bool(((tags != IMAGE_TAG) & (tags != TEXT_TAG)).any()):
        raise ValueError(f"text_token_tags may only hold {IMAGE_TAG} or {TEXT_TAG}")
    num_text = int(tags.numel())
    positions = pg.packed_positions(num_text, list(reference_grids), target_grid, patch_size)
    seq = positions.sequence_length
    token_tags = torch.full((seq,), IMAGE_TAG, dtype=torch.long)
    token_tags[:num_text] = tags
    source_kind = torch.full((seq,), SOURCE_TEXT, dtype=torch.long)
    reference_index = torch.full((seq,), -1, dtype=torch.long)
    patch_coord = torch.full((seq, 2), -1, dtype=torch.long)
    _, patch_h, patch_w = (int(p) for p in patch_size)
    for source in positions.sources:
        rows = slice(source.start, source.stop)
        source_kind[rows] = SOURCE_KINDS[source.kind]
        reference_index[rows] = source.index
        cols = source.latent_width // patch_w
        flat = torch.arange(source.stop - source.start, dtype=torch.long)
        patch_coord[rows, 0] = flat // cols
        patch_coord[rows, 1] = flat % cols
    layout = PackedLayout(
        position_ids=positions.position_ids,
        token_tags=token_tags,
        image_indices=torch.arange(num_text, seq, dtype=torch.long),
        text_indices=torch.arange(num_text, dtype=torch.long),
        source_kind=source_kind,
        reference_index=reference_index,
        patch_coord=patch_coord,
        num_reference_rows=positions.num_reference_rows,
        reference_grids=tuple((int(h), int(w)) for h, w in reference_grids),
        target_grid=(int(target_grid[0]), int(target_grid[1])),
        patch_size=(int(patch_size[0]), patch_h, patch_w),
    )
    validate(layout)
    return layout


def validate(layout: PackedLayout) -> None:
    """Fail closed unless every per-row tensor and both selections are mutually consistent."""

    seq = layout.sequence_length
    if layout.position_ids.dtype != torch.float64 or tuple(layout.position_ids.shape[1:]) != (3,):
        raise ValueError("position_ids must be float64 [S, 3]")
    image = _long("image_indices", layout.image_indices, 1)
    text = _long("text_indices", layout.text_indices, 1)
    for name in ("token_tags", "source_kind", "reference_index"):
        tensor = _long(name, getattr(layout, name), 1)
        if tensor.shape[0] != seq:
            raise ValueError(f"{name} has {tensor.shape[0]} rows, sequence has {seq}")
    coord = _long("patch_coord", layout.patch_coord, 2)
    if tuple(coord.shape) != (seq, 2):
        raise ValueError(f"patch_coord must be [{seq}, 2]")
    for name, index in (("image_indices", image), ("text_indices", text)):
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= seq):
            raise ValueError(f"{name} holds an out-of-range position")
        if torch.unique(index).numel() != index.numel():
            raise ValueError(f"{name} holds duplicate positions")
    covered = torch.zeros(seq, dtype=torch.long)
    covered[image] += 1
    covered[text] += 1
    if not bool((covered == 1).all()):
        raise ValueError("image_indices and text_indices must partition the sequence")
    kind = layout.source_kind
    if bool((kind[text] != SOURCE_TEXT).any()) or bool((kind[image] == SOURCE_TEXT).any()):
        raise ValueError("source_kind disagrees with the selections")
    if bool((layout.token_tags[image] != IMAGE_TAG).any()):
        raise ValueError("every latent row must carry the image tag")
    text_tags = layout.token_tags[text]
    if bool(((text_tags != IMAGE_TAG) & (text_tags != TEXT_TAG)).any()):
        raise ValueError("text rows may only carry the image or text tag")
    selected = kind[image]
    references = layout.num_reference_rows
    if int((selected == SOURCE_REFERENCE).sum()) != references:
        raise ValueError("num_reference_rows does not count the reference rows")
    if bool((selected[:references] != SOURCE_REFERENCE).any()):
        raise ValueError("reference rows must be a prefix of the image selection")
    if references == int(image.numel()):
        raise ValueError("the image selection needs generated rows after the references")
    if bool((layout.reference_index[kind != SOURCE_REFERENCE] != -1).any()):
        raise ValueError("reference_index is set on a non-reference row")
    _, patch_h, patch_w = layout.patch_size
    expected_refs = sum((h // patch_h) * (w // patch_w) for h, w in layout.reference_grids)
    if expected_refs != references:
        raise ValueError("num_reference_rows does not match the reference grids")


def permute(layout: PackedLayout, order: torch.Tensor) -> PackedLayout:
    """Apply a fully consistent row permutation: new row ``i`` is old row ``order[i]``.

    Every per-row tensor moves with its row, and each selection is rewritten to point at
    the new positions of the same rows in the same selection order.
    """

    order = _long("order", order, 1)
    seq = layout.sequence_length
    if order.numel() != seq or not torch.equal(torch.sort(order).values, torch.arange(seq)):
        raise ValueError("order must be a permutation of the sequence rows")
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(seq, dtype=torch.long)
    moved = {name: getattr(layout, name)[order] for name in PER_ROW}
    permuted = replace(
        layout,
        **moved,
        image_indices=inverse[layout.image_indices],
        text_indices=inverse[layout.text_indices],
    )
    validate(permuted)
    return permuted


def _rows(layout: PackedLayout) -> list[bytes]:
    """One byte string per row: its full semantic identity, position included."""

    rows = []
    for row in range(layout.sequence_length):
        rows.append(
            b"|".join(
                (
                    layout.position_ids[row].numpy().tobytes(),
                    layout.token_tags[row : row + 1].numpy().tobytes(),
                    layout.source_kind[row : row + 1].numpy().tobytes(),
                    layout.reference_index[row : row + 1].numpy().tobytes(),
                    layout.patch_coord[row].numpy().tobytes(),
                )
            )
        )
    return rows


def row_identity_digest(layout: PackedLayout) -> str:
    """Order-free digest of the multiset of row identities plus the selection contents."""

    rows = _rows(layout)
    image = sorted(rows[int(i)] for i in layout.image_indices)
    text = [rows[int(i)] for i in layout.text_indices]  # text order is the presentation order
    selection = [rows[int(i)] for i in layout.image_indices]
    payload = b"\n".join(
        [b"rows"]
        + sorted(rows)
        + [b"image-set"]
        + image
        + [b"image-order"]
        + selection
        + [b"text-order"]
        + text
    )
    return hashlib.sha256(payload).hexdigest()


def layout_digest(layout: PackedLayout) -> str:
    """Order-sensitive digest of the packed sequence as laid out."""

    digest = hashlib.sha256()
    for name in PER_ROW + ("image_indices", "text_indices"):
        digest.update(name.encode() + tensor_digest(getattr(layout, name)).encode())
    digest.update(str(layout.num_reference_rows).encode())
    return digest.hexdigest()
